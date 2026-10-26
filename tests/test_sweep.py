import json
import tempfile
import unittest
from pathlib import Path

from modelbench.config import ExperimentConfig
from modelbench.datasets import Table
from modelbench.errors import ConfigError, SweepError
from modelbench.runner import run_experiment
from modelbench.store import Store
from modelbench.sweep import load_sweep, parse_sweep, run_sweep

ENV = {"python": "3.12.1", "lib:numpy": None}


def blobs(n=20):
    rows = [(f"{i * 0.1:.1f}", "0", "a") for i in range(n)]
    rows += [(f"{10 + i * 0.1:.1f}", "5", "b") for i in range(n)]
    return Table(("x1", "x2", "label"), tuple(rows))


def line_table(n=40):
    return Table(("x", "y"), tuple((str(i), repr(2.0 * i + 1.0)) for i in range(n)))


def knn_config(**overrides):
    fields = {
        "experiment_id": "sw",
        "model": "knn",
        "seed": 3,
        "dataset_id": "blobs",
        "metrics": ("accuracy", "f1"),
        "params": {},
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


def ridge_config(**overrides):
    fields = {
        "experiment_id": "ridge",
        "model": "linear_regression",
        "seed": 3,
        "dataset_id": "line",
        "metrics": ("mae", "r2"),
        "params": {},
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


class SweepTestCase(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)

    def sweep(self, config, grid, table=None, label="label", **kwargs):
        kwargs.setdefault("environment", ENV)
        return run_sweep(self.store, config, grid, table or blobs(), label, **kwargs)


class RunSweepTests(SweepTestCase):
    def test_one_run_per_combination_in_grid_order(self):
        result = self.sweep(knn_config(), {"k": [5, 1, 3]})
        self.assertEqual([e.params for e in result.entries], [{"k": 5}, {"k": 1}, {"k": 3}])
        self.assertEqual(len(self.store.list_runs()), 3)
        self.assertTrue(all(e.error is None and not e.reused for e in result.entries))
        for entry in result.entries:
            self.assertEqual(self.store.get_config(entry.run_id).params, entry.params)

    def test_the_spread_metric_needs_cross_validation(self):
        with self.assertRaisesRegex(SweepError, "cannot rank by 'accuracy_std'"):
            self.sweep(knn_config(), {"k": [1]}, rank_by="accuracy_std")
        with self.assertRaisesRegex(SweepError, "cannot rank by 'recall_std'"):
            self.sweep(knn_config(cv_folds=3), {"k": [1]}, rank_by="recall_std")

    def test_two_parameters_make_the_full_product(self):
        result = self.sweep(knn_config(), {"k": [1, 3], "weights": ["uniform", "distance"]})
        self.assertEqual(
            [e.params for e in result.entries],
            [
                {"k": 1, "weights": "uniform"},
                {"k": 1, "weights": "distance"},
                {"k": 3, "weights": "uniform"},
                {"k": 3, "weights": "distance"},
            ],
        )

    def test_ties_share_a_rank_and_the_best_is_the_earliest_run(self):
        result = self.sweep(knn_config(), {"k": [1, 3]})  # both classify the clusters perfectly
        self.assertEqual([r.rank for r in result.ranking.ranked], [1, 1])
        best = result.best()
        self.assertEqual(best.params, {"k": 1})
        self.assertEqual(best.run_id, result.entries[0].run_id)

    def test_higher_is_better_metric_ranks_the_largest_first(self):
        result = self.sweep(ridge_config(), {"l2": [1e9, 0.0]}, line_table(), "y", rank_by="r2")
        self.assertEqual(result.rank_metric, "r2")
        self.assertEqual(result.best().params, {"l2": 0.0})
        self.assertEqual([r.rank for r in result.ranking.ranked], [1, 2])
        self.assertAlmostEqual(result.best().metrics["r2"], 1.0, places=8)

    def test_lower_is_better_metric_ranks_the_smallest_first(self):
        result = self.sweep(ridge_config(), {"l2": [0.0, 1e9]}, line_table(), "y", rank_by="mae")
        self.assertEqual(result.best().params, {"l2": 0.0})
        values = [r.value for r in result.ranking.ranked]
        self.assertLess(values[0], values[1])

    def test_default_ranking_metric_is_the_first_configured(self):
        result = self.sweep(ridge_config(metrics=("r2", "mae")), {"l2": [0.0]}, line_table(), "y")
        self.assertEqual(result.rank_metric, "r2")

    def test_a_bad_parameter_value_is_reported_and_the_sweep_goes_on(self):
        result = self.sweep(knn_config(), {"k": [0, 1]})
        bad, good = result.entries
        self.assertIsNone(bad.run_id)  # rejected before any run existed
        self.assertIn("k must be a positive integer", bad.error)
        self.assertIsNone(good.error)
        self.assertEqual(result.failed, (bad,))
        self.assertEqual(result.best(), good)
        self.assertEqual(len(self.store.list_runs()), 1)

    def test_a_rejection_after_a_successful_run_does_not_borrow_its_id(self):
        result = self.sweep(knn_config(), {"k": [1, 0]})
        good, bad = result.entries
        self.assertIsNotNone(good.run_id)
        self.assertIsNone(bad.run_id)

    def test_a_failure_during_fitting_keeps_the_failed_run_id(self):
        result = self.sweep(knn_config(), {"k": [100, 1]})  # 100 > 32 training rows
        bad, good = result.entries
        self.assertIn("larger than", bad.error)
        self.assertEqual(self.store.get_run(bad.run_id).status, "failed")
        self.assertIsNone(good.error)
        self.assertEqual([e for _, e in result.ranked_entries()], [good, bad])
        self.assertEqual([r for r, _ in result.ranked_entries()], [1, 0])

    def test_stop_on_error_stops_at_the_first_failure(self):
        result = self.sweep(knn_config(), {"k": [0, 1, 3]}, stop_on_error=True)
        self.assertEqual(len(result.entries), 1)
        self.assertIsNone(result.best())
        self.assertEqual(self.store.list_runs(), [])

    def test_when_every_combination_fails_there_is_no_best(self):
        result = self.sweep(knn_config(), {"k": [0, -1]})
        self.assertEqual(len(result.failed), 2)
        self.assertIsNone(result.best())
        self.assertEqual(result.ranking.ranked, ())

    def test_running_the_sweep_again_reuses_every_run(self):
        first = self.sweep(knn_config(), {"k": [1, 3]})
        second = self.sweep(knn_config(), {"k": [1, 3]})
        self.assertEqual(len(self.store.list_runs()), 2)
        self.assertTrue(all(e.reused for e in second.entries))
        self.assertEqual([e.run_id for e in first.entries], [e.run_id for e in second.entries])
        self.assertEqual(first.entries[0].metrics, second.entries[0].metrics)

    def test_extending_a_sweep_only_runs_the_new_combinations(self):
        self.sweep(knn_config(), {"k": [1, 3]})
        extended = self.sweep(knn_config(), {"k": [1, 3, 5]})
        self.assertEqual([e.reused for e in extended.entries], [True, True, False])
        self.assertEqual(len(self.store.list_runs()), 3)

    def test_reuse_can_be_switched_off(self):
        self.sweep(knn_config(), {"k": [1]})
        again = self.sweep(knn_config(), {"k": [1]}, reuse_existing=False)
        self.assertFalse(again.entries[0].reused)
        self.assertEqual(len(self.store.list_runs()), 2)

    def test_a_grid_key_that_is_also_a_base_parameter_is_rejected(self):
        with self.assertRaises(ConfigError):
            self.sweep(knn_config(params={"k": 3}), {"k": [1, 5]})
        self.assertEqual(self.store.list_runs(), [])

    def test_ranking_by_an_unconfigured_metric_is_rejected_before_any_run(self):
        with self.assertRaisesRegex(SweepError, "cannot rank by 'recall'"):
            self.sweep(knn_config(), {"k": [1]}, rank_by="recall")
        self.assertEqual(self.store.list_runs(), [])

    def test_cross_validated_sweep_ranks_by_the_fold_mean(self):
        result = self.sweep(knn_config(cv_folds=4, stratify=True), {"k": [1, 3]})
        for entry in result.entries:
            self.assertIn("accuracy_std", entry.metrics)
            self.assertEqual(self.store.get_config(entry.run_id).cv_folds, 4)
        self.assertEqual(result.best().metrics["accuracy"], 1.0)

    def test_the_spread_across_folds_can_be_the_ranking_metric(self):
        result = self.sweep(
            knn_config(cv_folds=4, stratify=True), {"k": [1, 3]}, rank_by="accuracy_std"
        )
        self.assertEqual(result.rank_metric, "accuracy_std")
        self.assertEqual(result.ranking.ranked[0].value, 0.0)

    def test_sweep_runs_match_individual_runs(self):
        result = self.sweep(knn_config(), {"k": [3]})
        other = Store(":memory:")
        self.addCleanup(other.close)
        single = run_experiment(
            other, knn_config(params={"k": 3}), blobs(), "label", environment=ENV
        )
        self.assertEqual(result.entries[0].metrics, single.metrics)


class ParseSweepTests(unittest.TestCase):
    def good(self, **overrides):
        document = {
            "config": {
                "experiment_id": "sw",
                "model": "knn",
                "seed": 1,
                "dataset_id": "d",
                "metrics": ["accuracy"],
                "params": {},
            },
            "grid": {"k": [1, 3]},
        }
        document.update(overrides)
        return document

    def test_a_valid_document(self):
        config, grid = parse_sweep(self.good())
        self.assertEqual(config.model, "knn")
        self.assertEqual(grid, {"k": [1, 3]})

    def test_rejects_malformed_documents(self):
        cases = {
            "must be a JSON object": [1, 2],
            "unknown key": {**self.good(), "extra": 1},
            "needs both": {"grid": {"k": [1]}},
            "at least one parameter": self.good(grid={}),
            "non-empty list": self.good(grid={"k": []}),
        }
        for message, document in cases.items():
            with self.subTest(message), self.assertRaisesRegex(SweepError, message):
                parse_sweep(document)
        with self.assertRaisesRegex(SweepError, "non-empty list"):
            parse_sweep(self.good(grid={"k": 3}))
        with self.assertRaises(ConfigError):  # the config part is validated like any config
            parse_sweep(self.good(config={"model": "knn"}))


class LoadSweepTests(unittest.TestCase):
    def write(self, folder, content, name="sweep.json", mode="w"):
        path = Path(folder) / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content if isinstance(content, str) else json.dumps(content))
        return path

    def test_reads_a_file_with_a_byte_order_mark(self):
        document = ParseSweepTests().good()
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, b"\xef\xbb\xbf" + json.dumps(document).encode())
            config, grid = load_sweep(path)
        self.assertEqual((config.experiment_id, grid), ("sw", {"k": [1, 3]}))

    def test_reports_unusable_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = {
                "cannot read": Path(tmp) / "missing.json",
                "not valid JSON": self.write(tmp, "{nope", "bad.json"),
                "not valid UTF-8": self.write(tmp, b"\xff\xfe\x00", "utf.json"),
            }
            for message, path in cases.items():
                with self.subTest(message), self.assertRaisesRegex(SweepError, message):
                    load_sweep(path)

    def test_rejects_an_oversized_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, b" " * 1_000_001, "big.json")
            with self.assertRaisesRegex(SweepError, "larger than"):
                load_sweep(path)


if __name__ == "__main__":
    unittest.main()
