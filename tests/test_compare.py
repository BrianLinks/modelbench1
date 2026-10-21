import math
import random
import unittest

from modelbench.compare import (
    MAX_RESAMPLES,
    MIN_RESAMPLES,
    ParamDiff,
    compare_metrics,
    diff_params,
    flatten_params,
    higher_is_better,
    paired_bootstrap,
    percentile,
    rank_runs,
)
from modelbench.errors import CompareError
from modelbench.seeding import derive_seed


class FlattenTests(unittest.TestCase):
    def test_nested_objects_get_dotted_keys_and_lists_stay_whole(self):
        flat = flatten_params({"a": {"b": 1, "c": {"d": 2}}, "e": [1, 2], "f": {}, "g": None})
        self.assertEqual(flat, {"a.b": 1, "a.c.d": 2, "e": [1, 2], "f": {}, "g": None})

    def test_ambiguous_names_are_rejected(self):
        with self.assertRaises(CompareError):
            flatten_params({"a.b": 1, "a": {"b": 2}})

    def test_non_string_names_are_rejected(self):
        with self.assertRaises(CompareError):
            flatten_params({1: "x"})


class DiffParamsTests(unittest.TestCase):
    def test_added_removed_changed_sorted_by_key(self):
        old = {"k": 3, "weights": "uniform", "gone": True}
        new = {"k": 5, "weights": "uniform", "extra": [1]}
        self.assertEqual(
            diff_params(old, new),
            [
                ParamDiff("extra", "added", None, [1]),
                ParamDiff("gone", "removed", True, None),
                ParamDiff("k", "changed", 3, 5),
            ],
        )

    def test_equal_parameters_have_no_diff_regardless_of_key_order(self):
        self.assertEqual(diff_params({"a": 1, "b": {"c": 2}}, {"b": {"c": 2}, "a": 1}), [])
        self.assertEqual(diff_params({}, {}), [])

    def test_nested_changes_use_dotted_keys(self):
        diffs = diff_params({"opt": {"lr": 0.1}}, {"opt": {"lr": 0.2}})
        self.assertEqual(diffs, [ParamDiff("opt.lr", "changed", 0.1, 0.2)])

    def test_values_are_compared_like_the_config_hash_does(self):
        for old, new in ((1, 1.0), (1, True), (0, False), ("1", 1)):
            self.assertEqual(len(diff_params({"x": old}, {"x": new})), 1, (old, new))
        self.assertEqual(diff_params({"x": [1, 2]}, {"x": [1, 2]}), [])
        self.assertEqual(len(diff_params({"x": [1, 2]}, {"x": [2, 1]})), 1)

    def test_object_replaced_by_a_scalar_is_a_removal_plus_an_addition(self):
        diffs = diff_params({"a": {"b": 1}}, {"a": 5})
        self.assertEqual([(d.key, d.kind) for d in diffs], [("a", "added"), ("a.b", "removed")])

    def test_non_json_values_raise_a_compare_error(self):
        with self.assertRaises(CompareError):
            diff_params({"x": {1, 2}}, {"x": {1, 3}})


class DirectionTests(unittest.TestCase):
    def test_known_metrics(self):
        for name in ("accuracy", "precision", "recall", "f1", "roc_auc", "r2"):
            self.assertTrue(higher_is_better(name), name)
        for name in ("log_loss", "mae", "mse", "rmse"):
            self.assertFalse(higher_is_better(name), name)

    def test_prefixes_are_ignored_and_unknown_names_are_errors(self):
        self.assertTrue(higher_is_better("test/accuracy"))
        self.assertFalse(higher_is_better("val/fold/rmse"))
        with self.assertRaises(CompareError):
            higher_is_better("custom_score")


class CompareMetricsTests(unittest.TestCase):
    def test_deltas_relative_change_and_improvement(self):
        old = {"accuracy": 0.8, "log_loss": 0.5, "custom": 1.0, "only_old": 1.0}
        new = {"accuracy": 0.9, "log_loss": 0.4, "custom": 2.0, "only_new": 3.0}
        result = compare_metrics(old, new)
        self.assertEqual([d.name for d in result.deltas], ["accuracy", "custom", "log_loss"])
        accuracy, custom, log_loss = result.deltas
        self.assertAlmostEqual(accuracy.delta, 0.1)
        self.assertAlmostEqual(accuracy.relative, 0.125)
        self.assertTrue(accuracy.improved)
        self.assertAlmostEqual(log_loss.delta, -0.1)
        self.assertTrue(log_loss.improved)  # lower loss is better
        self.assertEqual((custom.delta, custom.relative, custom.improved), (1.0, 1.0, None))
        self.assertEqual(result.only_in_old, ("only_old",))
        self.assertEqual(result.only_in_new, ("only_new",))

    def test_worse_unchanged_and_zero_baseline(self):
        old = {"accuracy": 0.9, "f1": 0.5, "mae": 0.0}
        new = {"accuracy": 0.8, "f1": 0.5, "mae": 1.0}
        result = compare_metrics(old, new)
        by_name = {d.name: d for d in result.deltas}
        self.assertFalse(by_name["accuracy"].improved)
        self.assertIsNone(by_name["f1"].improved)  # no change is neither better nor worse
        self.assertFalse(by_name["mae"].improved)  # error went up
        self.assertIsNone(by_name["mae"].relative)  # no relative change from zero

    def test_invalid_values_are_rejected(self):
        for bad in (float("nan"), float("inf"), True, "0.5", None):
            with self.assertRaises(CompareError, msg=repr(bad)):
                compare_metrics({"accuracy": bad}, {"accuracy": 0.5})
            with self.assertRaises(CompareError, msg=repr(bad)):
                compare_metrics({"accuracy": 0.5}, {"accuracy": bad})

    def test_no_overlap_gives_no_deltas(self):
        result = compare_metrics({"a": 1.0}, {"b": 2.0})
        self.assertEqual(result.deltas, ())
        self.assertEqual((result.only_in_old, result.only_in_new), (("a",), ("b",)))


class RankTests(unittest.TestCase):
    RESULTS = {
        "r1": {"accuracy": 0.9},
        "r2": {"accuracy": 0.8},
        "r3": {"accuracy": 0.8},
        "r4": {"accuracy": 0.7},
        "r5": {},
    }

    def test_ties_share_a_rank_and_the_next_rank_is_skipped(self):
        ranking = rank_runs(self.RESULTS, "accuracy")
        self.assertEqual(
            [(r.rank, r.run_id) for r in ranking.ranked],
            [(1, "r1"), (2, "r2"), (2, "r3"), (4, "r4")],
        )
        self.assertEqual(ranking.missing, ("r5",))

    def test_lower_is_better_metrics_rank_ascending(self):
        results = {"a": {"rmse": 3.0}, "b": {"rmse": 1.0}, "c": {"rmse": 2.0}}
        self.assertEqual([r.run_id for r in rank_runs(results, "rmse").ranked], ["b", "c", "a"])

    def test_input_order_never_changes_the_output(self):
        forward = rank_runs(self.RESULTS, "accuracy")
        backward = rank_runs(dict(reversed(list(self.RESULTS.items()))), "accuracy")
        self.assertEqual(forward, backward)

    def test_unknown_metrics_need_an_explicit_direction(self):
        results = {"a": {"score": 1.0}, "b": {"score": 2.0}}
        with self.assertRaises(CompareError):
            rank_runs(results, "score")
        best_high = rank_runs(results, "score", higher_is_better_override=True)
        best_low = rank_runs(results, "score", higher_is_better_override=False)
        self.assertEqual([r.run_id for r in best_high.ranked], ["b", "a"])
        self.assertEqual([r.run_id for r in best_low.ranked], ["a", "b"])

    def test_invalid_values_are_rejected_and_empty_input_is_fine(self):
        with self.assertRaises(CompareError):
            rank_runs({"a": {"accuracy": float("nan")}}, "accuracy")
        ranking = rank_runs({}, "accuracy")
        self.assertEqual((ranking.ranked, ranking.missing), ((), ()))

    def test_a_metric_nobody_reported_ranks_nobody(self):
        ranking = rank_runs({"a": {"f1": 1.0}, "b": {"f1": 0.5}}, "accuracy")
        self.assertEqual((ranking.ranked, ranking.missing), ((), ("a", "b")))


class PercentileTests(unittest.TestCase):
    def test_known_values_by_linear_interpolation(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(percentile(data, 0), 1.0)
        self.assertEqual(percentile(data, 0.25), 2.0)
        self.assertEqual(percentile(data, 0.5), 3.0)
        self.assertEqual(percentile(data, 1), 5.0)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.1), 1.3)
        self.assertEqual(percentile([7.0], 0.9), 7.0)

    def test_invalid_arguments(self):
        with self.assertRaises(CompareError):
            percentile([], 0.5)
        for q in (-0.1, 1.1, True, "0.5", None):
            with self.assertRaises(CompareError, msg=repr(q)):
                percentile([1.0, 2.0], q)


class BootstrapTests(unittest.TestCase):
    def test_identical_scores_have_a_zero_width_interval_at_zero(self):
        scores = [1, 0, 1, 1, 0, 1, 0, 0]
        result = paired_bootstrap(scores, scores, seed=1)
        self.assertEqual((result.mean_diff, result.ci_low, result.ci_high), (0.0, 0.0, 0.0))
        self.assertEqual(result.prob_a_better, 0.0)

    def test_a_uniformly_better_model_has_a_degenerate_interval_at_the_gap(self):
        result = paired_bootstrap([1.0] * 10, [0.0] * 10, seed=1)
        self.assertEqual((result.mean_diff, result.ci_low, result.ci_high), (1.0, 1.0, 1.0))
        self.assertEqual(result.prob_a_better, 1.0)

    def test_mean_difference_known_answer(self):
        # per-example differences are 1, 0, 0, 1, so the mean is 0.5
        result = paired_bootstrap([1, 1, 0, 1], [0, 1, 0, 0], seed=1)
        self.assertEqual(result.mean_diff, 0.5)
        self.assertEqual((result.n_samples, result.n_resamples), (4, 2000))

    def test_interval_matches_an_independent_hand_computation(self):
        a = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
        b = [0, 1, 0, 0, 0, 1, 0, 0, 1, 0]
        diffs = [x - y for x, y in zip(a, b)]
        draws = random.Random(derive_seed(9, "bootstrap"))
        means = sorted(
            sum(diffs[int(draws.random() * 10)] for _ in range(10)) / 10 for _ in range(1000)
        )
        # 95% interval from 1000 sorted means: q = 0.025 -> position 24.975, q = 0.975 -> 974.025
        low = means[24] + 0.975 * (means[25] - means[24])
        high = means[974] + 0.025 * (means[975] - means[974])
        result = paired_bootstrap(a, b, n_resamples=1000, seed=9)
        self.assertAlmostEqual(result.ci_low, low)
        self.assertAlmostEqual(result.ci_high, high)
        self.assertAlmostEqual(result.prob_a_better, sum(1 for m in means if m > 0) / 1000)

    def test_a_clear_gap_gives_an_interval_that_excludes_zero(self):
        a = [1] * 90 + [0] * 10  # 90% correct
        b = [1] * 70 + [0] * 30  # 70% correct, on the first 70 examples
        result = paired_bootstrap(a, b, seed=3)
        self.assertAlmostEqual(result.mean_diff, 0.2)
        self.assertGreater(result.ci_low, 0.0)
        self.assertLess(result.ci_high, 0.4)
        self.assertGreater(result.prob_a_better, 0.999)

    def test_no_real_difference_gives_an_interval_around_zero(self):
        a = [1, 0] * 50
        b = [0, 1] * 50  # equally accurate overall, but wrong on different examples
        result = paired_bootstrap(a, b, seed=3)
        self.assertEqual(result.mean_diff, 0.0)
        self.assertLess(result.ci_low, 0.0)
        self.assertGreater(result.ci_high, 0.0)
        self.assertTrue(0.3 < result.prob_a_better < 0.6)

    def test_higher_confidence_widens_the_interval(self):
        rng = random.Random(5)
        a = [rng.random() for _ in range(60)]
        b = [rng.random() for _ in range(60)]
        narrow = paired_bootstrap(a, b, confidence=0.8, seed=2)
        wide = paired_bootstrap(a, b, confidence=0.99, seed=2)
        self.assertLess(wide.ci_low, narrow.ci_low)
        self.assertGreater(wide.ci_high, narrow.ci_high)
        self.assertEqual((narrow.confidence, wide.confidence), (0.8, 0.99))

    def test_same_seed_repeats_exactly_and_different_seeds_differ(self):
        rng = random.Random(8)
        a = [rng.random() for _ in range(40)]
        b = [rng.random() for _ in range(40)]
        first = paired_bootstrap(a, b, seed=4)
        self.assertEqual(paired_bootstrap(a, b, seed=4), first)
        self.assertNotEqual(paired_bootstrap(a, b, seed=5), first)

    def test_global_random_state_and_inputs_are_untouched(self):
        a, b = [1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0]
        a_copy, b_copy = list(a), list(b)
        random.seed(77)
        expected = random.random()
        random.seed(77)
        paired_bootstrap(a, b, seed=1)
        self.assertEqual(random.random(), expected)
        self.assertEqual((a, b), (a_copy, b_copy))

    def test_interval_always_brackets_correctly_ordered_values(self):
        rng = random.Random(21)
        for _ in range(15):
            n = rng.randint(2, 40)
            a = [rng.choice((0, 1)) for _ in range(n)]
            b = [rng.choice((0, 1)) for _ in range(n)]
            result = paired_bootstrap(a, b, n_resamples=200, seed=rng.randint(0, 999))
            self.assertLessEqual(result.ci_low, result.ci_high)
            self.assertTrue(-1.0 <= result.ci_low and result.ci_high <= 1.0)
            self.assertTrue(0.0 <= result.prob_a_better <= 1.0)

    def test_validation(self):
        good = ([1.0, 0.0], [0.0, 1.0])
        with self.assertRaises(CompareError):
            paired_bootstrap([1.0, 0.0], [1.0])
        with self.assertRaises(CompareError):
            paired_bootstrap([1.0], [1.0])
        for n in (True, 100.0, "100", MIN_RESAMPLES - 1, MAX_RESAMPLES + 1):
            with self.assertRaises(CompareError, msg=repr(n)):
                paired_bootstrap(*good, n_resamples=n)
        for c in (0, 1, -0.5, 1.5, True, "0.95", None):
            with self.assertRaises(CompareError, msg=repr(c)):
                paired_bootstrap(*good, confidence=c)
        with self.assertRaises(ValueError):
            paired_bootstrap(*good, seed=-1)
        for bad in (float("nan"), float("inf"), True, "1"):
            with self.assertRaises(CompareError, msg=repr(bad)):
                paired_bootstrap([1.0, bad], [0.0, 1.0])
            with self.assertRaises(CompareError, msg=repr(bad)):
                paired_bootstrap([1.0, 0.0], [0.0, bad])

    def test_result_values_are_finite(self):
        result = paired_bootstrap([0.1, 0.2, 0.3], [0.3, 0.2, 0.1], seed=0)
        for value in (result.mean_diff, result.ci_low, result.ci_high, result.prob_a_better):
            self.assertTrue(math.isfinite(value))


if __name__ == "__main__":
    unittest.main()
