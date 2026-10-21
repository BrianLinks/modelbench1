import json
import unittest

from modelbench.config import ExperimentConfig
from modelbench.datasets import Table, make_version
from modelbench.errors import CompareError, IdentifierError, StoreError
from modelbench.run_comparison import compare_runs
from modelbench.store import Store

DATASET = make_version("iris", Table(("x", "y"), (("1", "a"), ("2", "b"))))
OTHER_DATASET = make_version("iris", Table(("x", "y"), (("1", "a"), ("2", "b"), ("3", "a"))))
ENV = {"python": "3.12.1", "lib:numpy": None}


def make_config(**overrides):
    fields = {
        "experiment_id": "exp-a",
        "model": "knn",
        "seed": 1,
        "dataset_id": "iris",
        "metrics": ("accuracy",),
        "params": {"k": 3},
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


class ComparisonTestCase(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)

    def finished(self, metrics, dataset=DATASET, env=ENV, **config_overrides):
        run_id = self.store.create_run(make_config(**config_overrides), dataset, env)
        self.store.log_metrics(run_id, metrics)
        self.store.finish_run(run_id)
        return run_id


class BasicComparisonTests(ComparisonTestCase):
    def test_deltas_params_and_rankings_against_the_baseline(self):
        base = self.finished({"accuracy": 0.8, "log_loss": 0.5})
        other = self.finished({"accuracy": 0.9, "log_loss": 0.4}, params={"k": 5})
        result = compare_runs(self.store, [base, other])

        self.assertEqual((result.baseline, result.run_ids), (base, (base, other)))
        (diff,) = result.diffs
        self.assertEqual(diff.run_id, other)
        self.assertEqual([(p.key, p.old, p.new) for p in diff.params], [("k", 3, 5)])
        by_name = {m.name: m for m in diff.metrics.deltas}
        self.assertAlmostEqual(by_name["accuracy"].delta, 0.1)
        self.assertTrue(by_name["accuracy"].improved)
        self.assertTrue(by_name["log_loss"].improved)  # lower loss is better
        self.assertFalse(diff.same_config)
        self.assertTrue(diff.same_dataset)
        self.assertTrue(diff.same_seed)
        self.assertEqual(diff.environment, ())
        self.assertEqual(result.warnings, ())

        ranks = {r.metric: [(x.rank, x.run_id) for x in r.ranked] for r in result.rankings}
        self.assertEqual(ranks["accuracy"], [(1, other), (2, base)])
        self.assertEqual(ranks["log_loss"], [(1, other), (2, base)])
        self.assertEqual([r.metric for r in result.rankings], ["accuracy", "log_loss"])

    def test_first_id_is_the_baseline_and_diffs_follow_the_given_order(self):
        a = self.finished({"accuracy": 0.7})
        b = self.finished({"accuracy": 0.9})
        c = self.finished({"accuracy": 0.8})
        result = compare_runs(self.store, [c, a, b])
        self.assertEqual(result.baseline, c)
        self.assertEqual([d.run_id for d in result.diffs], [a, b])
        self.assertEqual([x.run_id for x in result.rankings[0].ranked], [b, c, a])

    def test_identical_runs_have_the_same_config_and_no_improvement(self):
        a = self.finished({"accuracy": 0.8})
        b = self.finished({"accuracy": 0.8}, experiment_id="exp-b")  # label differs only
        (diff,) = compare_runs(self.store, [a, b]).diffs
        self.assertTrue(diff.same_config)
        self.assertEqual(diff.params, ())
        self.assertEqual(diff.metrics.deltas[0].delta, 0.0)
        self.assertIsNone(diff.metrics.deltas[0].improved)

    def test_tied_runs_share_a_rank(self):
        a = self.finished({"accuracy": 0.9})
        b = self.finished({"accuracy": 0.8})
        c = self.finished({"accuracy": 0.8})
        ranking = compare_runs(self.store, [a, b, c]).rankings[0]
        self.assertEqual([x.rank for x in ranking.ranked], [1, 2, 2])

    def test_final_values_use_each_metrics_last_step(self):
        a = self.store.create_run(make_config(), DATASET, ENV)
        for step, value in enumerate((0.5, 0.7, 0.9)):
            self.store.log_metric(a, "accuracy", value, step=step)
        self.store.finish_run(a)
        b = self.finished({"accuracy": 0.8})
        result = compare_runs(self.store, [a, b])
        self.assertEqual(result.values[a], {"accuracy": 0.9})


class DifferenceReportingTests(ComparisonTestCase):
    def test_different_dataset_versions_warn_and_are_flagged(self):
        a = self.finished({"accuracy": 0.8})
        b = self.finished({"accuracy": 0.9}, dataset=OTHER_DATASET)
        result = compare_runs(self.store, [a, b])
        self.assertFalse(result.diffs[0].same_dataset)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn(DATASET.version, result.warnings[0])
        self.assertIn(OTHER_DATASET.version, result.warnings[0])

    def test_different_seeds_are_flagged(self):
        a = self.finished({"accuracy": 0.8}, seed=1)
        b = self.finished({"accuracy": 0.8}, seed=2)
        (diff,) = compare_runs(self.store, [a, b]).diffs
        self.assertFalse(diff.same_seed)
        self.assertFalse(diff.same_config)

    def test_environment_differences_are_explained(self):
        a = self.finished({"accuracy": 0.8})
        b = self.finished({"accuracy": 0.8}, env={"python": "3.13.0", "lib:numpy": None})
        (diff,) = compare_runs(self.store, [a, b]).diffs
        self.assertEqual([e.key for e in diff.environment], ["python"])
        self.assertIn("Python", diff.environment[0].impact)

    def test_metrics_missing_from_some_runs_are_reported_not_hidden(self):
        a = self.finished({"accuracy": 0.8, "f1": 0.7})
        b = self.finished({"accuracy": 0.9})
        result = compare_runs(self.store, [a, b])
        (diff,) = result.diffs
        self.assertEqual(diff.metrics.only_in_old, ("f1",))
        self.assertEqual([m.name for m in diff.metrics.deltas], ["accuracy"])
        self.assertTrue(any("'f1'" in w and b in w for w in result.warnings))
        f1_ranking = next(r for r in result.rankings if r.metric == "f1")
        self.assertEqual(f1_ranking.missing, (b,))
        self.assertEqual([x.run_id for x in f1_ranking.ranked], [a])


class MetricSelectionTests(ComparisonTestCase):
    def test_metric_filter_limits_everything_reported(self):
        a = self.finished({"accuracy": 0.8, "f1": 0.7})
        b = self.finished({"accuracy": 0.9, "f1": 0.6})
        result = compare_runs(self.store, [a, b], metrics=["f1"])
        self.assertEqual(result.values[a], {"f1": 0.7})
        self.assertEqual([m.name for m in result.diffs[0].metrics.deltas], ["f1"])
        self.assertEqual([r.metric for r in result.rankings], ["f1"])
        self.assertFalse(result.diffs[0].metrics.deltas[0].improved)

    def test_requesting_a_metric_nobody_reported_is_an_error(self):
        a = self.finished({"accuracy": 0.8})
        b = self.finished({"accuracy": 0.9})
        with self.assertRaises(CompareError):
            compare_runs(self.store, [a, b], metrics=["f1"])
        with self.assertRaises(CompareError):
            compare_runs(self.store, [a, b], metrics="accuracy")

    def test_unknown_direction_is_compared_but_not_ranked_until_given(self):
        a = self.finished({"score": 1.0})
        b = self.finished({"score": 2.0})
        plain = compare_runs(self.store, [a, b])
        self.assertEqual(plain.rankings, ())
        self.assertIsNone(plain.diffs[0].metrics.deltas[0].improved)
        self.assertTrue(any("'score'" in w and "direction" in w for w in plain.warnings))

        lower = compare_runs(self.store, [a, b], directions={"score": False})
        self.assertEqual([x.run_id for x in lower.rankings[0].ranked], [a, b])
        self.assertFalse(lower.diffs[0].metrics.deltas[0].improved)
        self.assertEqual(lower.warnings, ())

    def test_directions_override_the_builtin_table(self):
        a = self.finished({"accuracy": 0.8})
        b = self.finished({"accuracy": 0.9})
        flipped = compare_runs(self.store, [a, b], directions={"accuracy": False})
        self.assertEqual([x.run_id for x in flipped.rankings[0].ranked], [a, b])
        self.assertFalse(flipped.diffs[0].metrics.deltas[0].improved)


class ValidationTests(ComparisonTestCase):
    def test_only_completed_runs_can_be_compared(self):
        done = self.finished({"accuracy": 0.8})
        running = self.store.create_run(make_config(), DATASET, ENV)
        failed = self.store.create_run(make_config(), DATASET, ENV)
        self.store.fail_run(failed, "boom")
        with self.assertRaises(CompareError) as ctx:
            compare_runs(self.store, [done, running, failed])
        message = str(ctx.exception)
        self.assertIn(f"{running} is running", message)
        self.assertIn(f"{failed} is failed", message)

    def test_needs_two_distinct_run_ids(self):
        a = self.finished({"accuracy": 0.8})
        for bad in ([], [a], [a, a], a, None):
            with self.assertRaises(CompareError, msg=repr(bad)):
                compare_runs(self.store, bad)

    def test_unknown_and_malformed_ids(self):
        a = self.finished({"accuracy": 0.8})
        with self.assertRaises(StoreError):
            compare_runs(self.store, [a, "run-999999"])
        with self.assertRaises(IdentifierError):
            compare_runs(self.store, [a, "../etc/passwd"])

    def test_comparing_does_not_change_the_store(self):
        a = self.finished({"accuracy": 0.8})
        b = self.finished({"accuracy": 0.9})
        before = (self.store.list_runs(), self.store.get_metrics(a), self.store.get_metrics(b))
        compare_runs(self.store, [a, b])
        after = (self.store.list_runs(), self.store.get_metrics(a), self.store.get_metrics(b))
        self.assertEqual(before, after)


class SerialisationTests(ComparisonTestCase):
    def build(self):
        a = self.finished({"accuracy": 0.8, "score": 1.0}, env={"python": "3.12.1"})
        b = self.finished(
            {"accuracy": 0.9, "score": 2.0}, params={"k": 5}, env={"python": "3.13.0"}, seed=2
        )
        return compare_runs(self.store, [a, b])

    def test_to_dict_is_json_serialisable_and_repeatable(self):
        result = self.build()
        text = json.dumps(result.to_dict(), sort_keys=True)
        self.assertEqual(json.dumps(result.to_dict(), sort_keys=True), text)
        data = json.loads(text)
        self.assertEqual(data["baseline"], result.baseline)
        changed_k = {"key": "k", "kind": "changed", "old": 3, "new": 5}
        self.assertEqual(data["diffs"][0]["params"], [changed_k])
        self.assertEqual(data["diffs"][0]["same_seed"], False)
        self.assertEqual(data["diffs"][0]["environment"][0]["key"], "python")
        self.assertEqual(data["rankings"][0]["metric"], "accuracy")
        self.assertEqual(len(data["warnings"]), 1)  # 'score' has no known direction

    def test_to_dict_orders_values_by_metric_name(self):
        data = self.build().to_dict()
        for metrics in data["values"].values():
            self.assertEqual(list(metrics), sorted(metrics))


if __name__ == "__main__":
    unittest.main()
