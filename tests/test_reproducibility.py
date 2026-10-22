import json
import unittest
from datetime import datetime, timedelta, timezone

from modelbench.config import ExperimentConfig
from modelbench.datasets import Table, make_version
from modelbench.errors import ReproError, StoreError
from modelbench.reproducibility import (
    DATASET_CHANGED,
    NOT_REPRODUCED,
    REPRODUCED,
    RERUN_FAILED,
    check_reproducibility,
    recover_stale_runs,
)
from modelbench.runner import run_experiment
from modelbench.store import Store

ENV = {"python": "3.12.1", "lib:numpy": None}


def blobs():
    rows = [(f"{i * 0.1:.1f}", "0", "a") for i in range(20)]
    rows += [(f"{10 + i * 0.1:.1f}", "5", "b") for i in range(20)]
    return Table(("x1", "x2", "label"), tuple(rows))


def make_config(**overrides):
    fields = {
        "experiment_id": "exp-a",
        "model": "knn",
        "seed": 7,
        "dataset_id": "blobs",
        "metrics": ("accuracy", "f1"),
        "params": {"k": 3},
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


class CheckTestCase(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)
        self.table = blobs()

    def make_run(self, config=None, **kwargs):
        kwargs.setdefault("environment", ENV)
        return run_experiment(
            self.store, config or make_config(), self.table, "label", **kwargs
        ).run_id

    def check(self, run_id, **kwargs):
        kwargs.setdefault("environment", ENV)
        return check_reproducibility(self.store, run_id, self.table, "label", **kwargs)

    def set_accuracy(self, run_id, value):
        self.tamper(
            "UPDATE metrics SET value = ? WHERE run_id = ? AND name = 'accuracy'", (value, run_id)
        )

    def tamper(self, sql, args=()):
        with self.store._conn:
            self.store._conn.execute(sql, args)


class ReproducedTests(CheckTestCase):
    def test_untouched_run_reproduces(self):
        run_id = self.make_run()
        report = self.check(run_id)
        self.assertEqual(report.verdict, REPRODUCED)
        self.assertTrue(report.reproduced)
        self.assertTrue(report.predictions_match)
        self.assertEqual([c.name for c in report.metrics], ["accuracy", "f1"])
        self.assertTrue(all(c.difference == 0.0 and c.within_tolerance for c in report.metrics))
        self.assertEqual(report.environment_differences, ())
        self.assertFalse(report.explained_by_environment)

    def test_check_leaves_history_unchanged(self):
        run_id = self.make_run()
        before = self.store.list_runs()
        self.check(run_id)
        self.assertEqual(self.store.list_runs(), before)
        self.assertEqual(self.store.get_run(run_id).status, "completed")

    def test_seeded_stochastic_model_reproduces(self):
        run_id = self.make_run(make_config(model="stratified", params={}))
        self.assertTrue(self.check(run_id).reproduced)

    def test_feature_columns_are_passed_through(self):
        run_id = self.make_run(feature_columns=["x1"])
        self.assertTrue(self.check(run_id, feature_columns=["x1"]).reproduced)


class MismatchTests(CheckTestCase):
    def test_changed_predictions_hash_is_not_reproduced_even_if_metrics_match(self):
        run_id = self.make_run()
        self.tamper(
            "UPDATE tags SET value = ? WHERE run_id = ? AND key = 'predictions_sha256'",
            ("0" * 64, run_id),
        )
        report = self.check(run_id)
        self.assertEqual(report.verdict, NOT_REPRODUCED)
        self.assertFalse(report.predictions_match)
        self.assertTrue(all(c.within_tolerance for c in report.metrics))
        self.assertIn("predictions differ", report.detail)

    def test_metric_outside_tolerance(self):
        run_id = self.make_run()
        self.set_accuracy(run_id, 0.75)
        report = self.check(run_id)
        self.assertEqual(report.verdict, NOT_REPRODUCED)
        self.assertTrue(report.predictions_match)
        accuracy = report.metrics[0]
        self.assertEqual(
            (accuracy.original, accuracy.rerun, accuracy.difference), (0.75, 1.0, 0.25)
        )
        self.assertFalse(accuracy.within_tolerance)
        self.assertIn("outside the tolerance", report.detail)

    def test_difference_is_absolute_when_original_is_higher(self):
        run_id = self.make_run()
        self.set_accuracy(run_id, 1.5)  # rerun gives 1.0, so rerun - original is negative
        report = self.check(run_id, tolerance=0.25)
        self.assertEqual(report.metrics[0].difference, 0.5)
        self.assertFalse(report.metrics[0].within_tolerance)
        self.assertTrue(self.check(run_id, tolerance=0.5).reproduced)

    def test_tolerance_boundary_is_inclusive(self):
        run_id = self.make_run()
        self.set_accuracy(run_id, 0.75)
        self.assertTrue(self.check(run_id, tolerance=0.25).reproduced)
        self.assertFalse(self.check(run_id, tolerance=0.2499).reproduced)

    def test_metric_the_rerun_does_not_produce(self):
        run_id = self.make_run()
        self.tamper(
            "INSERT INTO metrics (run_id, name, step, value) VALUES (?, 'extra', 0, 1.0)", (run_id,)
        )
        report = self.check(run_id, tolerance=1.0)
        extra = [c for c in report.metrics if c.name == "extra"][0]
        self.assertIsNone(extra.rerun)
        self.assertIsNone(extra.difference)
        self.assertFalse(extra.within_tolerance)
        self.assertEqual(report.verdict, NOT_REPRODUCED)

    def test_rerun_failure_is_reported_not_raised(self):
        run_id = self.make_run()
        report = self.check(run_id, feature_columns=["missing"])
        self.assertEqual(report.verdict, RERUN_FAILED)
        self.assertIsNone(report.predictions_match)
        self.assertIn("RunnerError", report.detail)
        self.assertIn("missing", report.detail)
        self.assertFalse(report.reproduced)


class DatasetAndEnvironmentTests(CheckTestCase):
    def test_changed_data_is_reported_without_rerunning(self):
        run_id = self.make_run()
        rows = list(self.table.rows)
        rows[0] = ("99.0", "0", "a")
        changed = Table(self.table.columns, tuple(rows))
        report = check_reproducibility(self.store, run_id, changed, "label", environment=ENV)
        self.assertEqual(report.verdict, DATASET_CHANGED)
        self.assertIsNone(report.predictions_match)
        self.assertIn(make_version("blobs", changed, "label").version, report.detail)
        self.assertIn(self.store.get_run(run_id).dataset_version, report.detail)
        self.assertFalse(report.explained_by_environment)

    def test_mismatch_with_environment_drift_is_explained(self):
        run_id = self.make_run()
        self.tamper(
            "UPDATE tags SET value = ? WHERE run_id = ? AND key = 'predictions_sha256'",
            ("0" * 64, run_id),
        )
        report = self.check(run_id, environment={"python": "3.13.0", "lib:numpy": None})
        self.assertTrue(report.explained_by_environment)
        self.assertEqual([d.key for d in report.environment_differences], ["python"])
        self.assertEqual(report.environment_differences[0].old, "3.12.1")
        self.assertEqual(report.environment_differences[0].new, "3.13.0")

    def test_drift_without_a_mismatch_is_listed_but_not_blamed(self):
        run_id = self.make_run()
        report = self.check(run_id, environment={"python": "3.13.0", "lib:numpy": None})
        self.assertTrue(report.reproduced)
        self.assertEqual(len(report.environment_differences), 1)
        self.assertFalse(report.explained_by_environment)

    def test_default_environment_is_the_current_one(self):
        run_id = self.make_run(environment=None)
        report = check_reproducibility(self.store, run_id, self.table, "label")
        self.assertTrue(report.reproduced)
        self.assertEqual(report.environment_differences, ())


class ArgumentTests(CheckTestCase):
    def test_only_completed_runs_can_be_checked(self):
        dataset = make_version("blobs", self.table, "label")
        running = self.store.create_run(make_config(), dataset, ENV)
        with self.assertRaisesRegex(ReproError, "running"):
            self.check(running)
        failed = self.store.create_run(make_config(), dataset, ENV)
        self.store.fail_run(failed, "boom")
        with self.assertRaisesRegex(ReproError, "failed"):
            self.check(failed)

    def test_run_without_predictions_hash(self):
        dataset = make_version("blobs", self.table, "label")
        run_id = self.store.create_run(make_config(), dataset, ENV)
        self.store.finish_run(run_id)
        with self.assertRaisesRegex(ReproError, "predictions hash"):
            self.check(run_id)

    def test_unknown_run(self):
        with self.assertRaises(StoreError):
            self.check("run-000099")

    def test_bad_tolerance(self):
        run_id = self.make_run()
        for bad in (-0.1, float("nan"), float("inf"), True, "0.1", None):
            with self.subTest(bad=bad), self.assertRaises(ReproError):
                self.check(run_id, tolerance=bad)

    def test_report_is_json_serialisable(self):
        run_id = self.make_run()
        data = self.check(run_id, environment={"python": "3.13.0", "lib:numpy": None}).to_dict()
        text = json.dumps(data, sort_keys=True)
        loaded = json.loads(text)
        self.assertEqual(loaded["verdict"], REPRODUCED)
        self.assertEqual(loaded["run_id"], run_id)
        self.assertEqual(loaded["metrics"][0]["name"], "accuracy")
        self.assertEqual(loaded["environment_differences"][0]["key"], "python")


NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def stamp(seconds_ago):
    return (NOW - timedelta(seconds=seconds_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.clock_value = stamp(0)
        self.store = Store(":memory:", clock=lambda: self.clock_value)
        self.addCleanup(self.store.close)
        self.dataset = make_version("blobs", blobs(), "label")

    def start(self, seconds_ago):
        self.clock_value = stamp(seconds_ago)
        return self.store.create_run(make_config(), self.dataset, ENV)

    def test_only_old_running_runs_are_failed(self):
        old = self.start(7200)
        young = self.start(60)
        done = self.start(9000)
        self.store.finish_run(done)
        recovered = recover_stale_runs(self.store, 3600, now=NOW)
        self.assertEqual(recovered, [old])
        self.assertEqual(self.store.get_run(old).status, "failed")
        self.assertIn("abandoned", self.store.get_run(old).error)
        self.assertIn("3600", self.store.get_run(old).error)
        self.assertEqual(self.store.get_run(young).status, "running")
        self.assertEqual(self.store.get_run(done).status, "completed")

    def test_cutoff_is_exclusive_and_second_precise(self):
        exactly = self.start(3600)
        just_older = self.start(3601)
        self.assertEqual(recover_stale_runs(self.store, 3600, now=NOW), [just_older])
        self.assertEqual(self.store.get_run(exactly).status, "running")

    def test_returns_oldest_first_and_is_idempotent(self):
        first = self.start(5000)
        second = self.start(4000)
        self.assertEqual(recover_stale_runs(self.store, 100, now=NOW), [first, second])
        self.assertEqual(recover_stale_runs(self.store, 100, now=NOW), [])

    def test_nothing_to_recover(self):
        self.assertEqual(recover_stale_runs(self.store, 100, now=NOW), [])

    def test_default_now_is_the_current_time(self):
        self.clock_value = "2000-01-01T00:00:00Z"
        run_id = self.store.create_run(make_config(), self.dataset, ENV)
        self.assertEqual(recover_stale_runs(self.store, 3600), [run_id])

    def test_bad_arguments(self):
        for bad in (0, -5, float("nan"), float("inf"), True, "60", None):
            with self.subTest(bad=bad), self.assertRaises(ReproError):
                recover_stale_runs(self.store, bad, now=NOW)
        with self.assertRaises(ReproError):
            recover_stale_runs(self.store, 60, now=datetime(2026, 1, 1))
        with self.assertRaises(ReproError):
            recover_stale_runs(
                self.store, 60, now=datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=2)))
            )

    def test_store_rejects_malformed_cutoff(self):
        for bad in ("2026-01-01", "yesterday", "2026-01-01 12:00:00", None, 5):
            with self.subTest(bad=bad), self.assertRaises(StoreError):
                self.store.stale_running(bad)


if __name__ == "__main__":
    unittest.main()
