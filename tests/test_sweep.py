import json
import math
import tempfile
import unittest
from pathlib import Path

from modelbench.config import ExperimentConfig
from modelbench.datasets import Table
from modelbench.errors import ConfigError, SweepError
from modelbench.runner import run_experiment
from modelbench.seeding import rng_for
from modelbench.store import Store
from modelbench.sweep import (
    MAX_TRIALS,
    check_space,
    load_search,
    load_sweep,
    parse_sweep,
    run_search,
    run_sweep,
    sample_params,
)

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


class SpaceTests(unittest.TestCase):
    def test_a_valid_space_is_returned_in_name_order(self):
        space = check_space(
            {
                "w": {"choice": ["a", 2]},
                "k": {"int": [1, 5]},
                "a": {"uniform": [0, 1]},
                "l": {"log_uniform": [0.001, 10]},
            }
        )
        self.assertEqual(list(space), ["a", "k", "l", "w"])
        self.assertEqual(space["k"], ("int", (1, 5)))
        self.assertEqual(space["a"], ("uniform", (0.0, 1.0)))

    def test_invalid_spaces(self):
        cases = {
            "at least one": {},
            "one of": {"k": {"int": [1, 2], "uniform": [0, 1]}},
            "unknown kind": {"k": {"normal": [0, 1]}},
            "needs \\[low, high\\]": {"k": {"int": [1]}},
            "finite number": {"k": {"uniform": [0, float("inf")]}},
            "got True": {"k": {"uniform": [True, 3]}},
            "whole numbers": {"k": {"int": [1.5, 4]}},
            "low <= high": {"k": {"int": [5, 1]}},
            "low < high": {"k": {"uniform": [1, 1]}},
            "low > 0": {"k": {"log_uniform": [0, 1]}},
            "non-empty list": {"k": {"choice": []}},
            "must be an object": {"k": [1, 2]},
        }
        for message, space in cases.items():
            with self.subTest(message), self.assertRaisesRegex(SweepError, message):
                check_space(space)
        self.assertEqual(check_space({"k": {"int": [3, 3]}})["k"], ("int", (3, 3)))


class SampleParamsTests(unittest.TestCase):
    SPACE = check_space(
        {
            "k": {"int": [1, 4]},
            "lr": {"log_uniform": [0.001, 10]},
            "mix": {"uniform": [-1, 1]},
            "w": {"choice": ["uniform", "distance"]},
        }
    )

    def test_values_come_from_the_trials_own_generator_in_name_order(self):
        for trial in range(5):
            rng = rng_for(9, "search", trial)
            expected = {"k": rng.randint(1, 4)}
            expected["lr"] = float(f"{math.exp(rng.uniform(math.log(0.001), math.log(10))):.6g}")
            expected["mix"] = float(f"{rng.uniform(-1, 1):.6g}")
            expected["w"] = rng.choice(["uniform", "distance"])
            self.assertEqual(sample_params(self.SPACE, 9, trial), expected, trial)

    def test_a_trial_depends_only_on_the_seed_and_its_number(self):
        self.assertEqual(sample_params(self.SPACE, 9, 3), sample_params(self.SPACE, 9, 3))
        self.assertNotEqual(sample_params(self.SPACE, 9, 3), sample_params(self.SPACE, 10, 3))
        self.assertNotEqual(sample_params(self.SPACE, 9, 3), sample_params(self.SPACE, 9, 4))

    def test_draws_stay_in_range_and_reach_both_ends_of_an_int_range(self):
        ks, lrs = set(), []
        for trial in range(300):
            drawn = sample_params(self.SPACE, 1, trial)
            ks.add(drawn["k"])
            lrs.append(drawn["lr"])
            self.assertTrue(-1 <= drawn["mix"] <= 1)
            self.assertIn(drawn["w"], ("uniform", "distance"))
            self.assertIsInstance(drawn["k"], int)
        self.assertEqual(ks, {1, 2, 3, 4})
        self.assertTrue(all(0.001 <= v <= 10 for v in lrs))
        below = sum(v < 0.1 for v in lrs)  # log-uniform: half of the log range is below 0.1
        self.assertTrue(100 < below < 200, below)

    def test_choices_keep_their_json_type(self):
        space = check_space({"c": {"choice": [None, True, 3, "x", [1, 2]]}})
        seen = {repr(sample_params(space, 2, t)["c"]) for t in range(200)}
        self.assertEqual(seen, {"None", "True", "3", "'x'", "[1, 2]"})


class RunSearchTests(SweepTestCase):
    SPACE = {"k": {"int": [1, 9]}, "weights": {"choice": ["uniform", "distance"]}}

    def test_each_trial_is_a_run_with_the_drawn_parameters(self):
        result = self.sweep_search(knn_config(seed=4), 6)
        checked = check_space(self.SPACE)
        expected = []
        for trial in range(6):
            drawn = sample_params(checked, 4, trial)
            if drawn not in expected:
                expected.append(drawn)
        self.assertEqual([e.params for e in result.entries], expected)
        self.assertEqual(len(self.store.list_runs()), len(expected))
        self.assertTrue(all(e.error is None for e in result.entries))

    def sweep_search(self, config, trials, space=None, **kwargs):
        kwargs.setdefault("environment", ENV)
        return run_search(
            self.store, config, space or self.SPACE, trials, blobs(), "label", **kwargs
        )

    def test_repeated_draws_are_skipped(self):
        result = self.sweep_search(knn_config(), 30, {"k": {"int": [1, 2]}})
        self.assertEqual(sorted(e.params["k"] for e in result.entries), [1, 2])

    def test_more_trials_extend_the_earlier_ones(self):
        few = self.sweep_search(knn_config(seed=8), 4)
        more = self.sweep_search(knn_config(seed=8), 8)
        self.assertEqual([e.params for e in more.entries[: len(few.entries)]],
                         [e.params for e in few.entries])  # fmt: skip
        self.assertTrue(all(e.reused for e in more.entries[: len(few.entries)]))

    def test_base_parameters_are_kept_and_clashes_rejected(self):
        result = self.sweep_search(
            knn_config(params={"metric": "manhattan"}), 3, {"k": {"int": [1, 5]}}
        )
        self.assertTrue(all(e.params["metric"] == "manhattan" for e in result.entries))
        with self.assertRaisesRegex(SweepError, "both the space and the config"):
            self.sweep_search(knn_config(params={"k": 3}), 3)

    def test_a_bad_draw_fails_alone(self):
        result = self.sweep_search(knn_config(), 5, {"k": {"int": [100, 101]}})
        self.assertTrue(all(e.error is not None for e in result.entries))
        self.assertIsNone(result.best())

    def test_trial_counts_are_checked(self):
        for bad in (0, -1, MAX_TRIALS + 1, True, 2.5):
            with self.subTest(trials=bad), self.assertRaisesRegex(SweepError, "trials"):
                self.sweep_search(knn_config(), bad)
        self.assertEqual(self.store.list_runs(), [])

    def test_ranking_matches_the_runs(self):
        result = self.sweep_search(knn_config(), 5, rank_by="f1")
        self.assertEqual(result.rank_metric, "f1")
        values = [r.value for r in result.ranking.ranked]
        self.assertEqual(values, sorted(values, reverse=True))


class LoadSearchTests(unittest.TestCase):
    def doc(self, **overrides):
        document = {
            "config": ParseSweepTests().good()["config"],
            "space": {"k": {"int": [1, 5]}},
            "trials": 5,
        }
        document.update(overrides)
        return document

    def load(self, document):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            path.write_text(json.dumps(document))
            return load_search(path)

    def test_a_valid_file(self):
        config, space, trials = self.load(self.doc())
        self.assertEqual((config.model, trials), ("knn", 5))
        self.assertEqual(space, {"k": {"int": [1, 5]}})

    def test_malformed_files(self):
        cases = {
            "must be a JSON object": [1],
            "unknown key": self.doc(extra=1),
            "needs": {"config": self.doc()["config"]},
            "trials": self.doc(trials=0),
            "unknown kind": self.doc(space={"k": {"bogus": [1]}}),
        }
        for message, document in cases.items():
            with self.subTest(message), self.assertRaisesRegex(SweepError, message):
                self.load(document)
        with self.assertRaisesRegex(SweepError, "cannot read search file"):
            load_search("/nonexistent/search.json")


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
