import math
import unittest

from modelbench import metrics as m
from modelbench.compare import higher_is_better
from modelbench.config import ExperimentConfig
from modelbench.datasets import Table
from modelbench.errors import ConfigError, MetricError, RunnerError
from modelbench.runner import run_experiment
from modelbench.store import Store

Y = [0, 0, 1, 1]
S = [0.1, 0.4, 0.35, 0.8]  # the classic four-point example: ROC AUC 0.75, AP 5/6


def trapezoid(points):
    return math.fsum(
        (b.x - a.x) * (a.y + b.y) / 2 for a, b in zip(points, points[1:], strict=False)
    )


class RocCurveTests(unittest.TestCase):
    def test_hand_computed_points(self):
        points = m.roc_curve(Y, S)
        got = [(p.x, p.y, p.threshold) for p in points]
        self.assertEqual(
            got,
            [
                (0.0, 0.0, math.inf),
                (0.0, 0.5, 0.8),
                (0.5, 0.5, 0.4),
                (0.5, 1.0, 0.35),
                (1.0, 1.0, 0.1),
            ],
        )

    def test_area_equals_roc_auc(self):
        self.assertAlmostEqual(trapezoid(m.roc_curve(Y, S)), 0.75)
        self.assertAlmostEqual(trapezoid(m.roc_curve(Y, S)), m.roc_auc(Y, S))

    def test_tied_scores_move_together(self):
        points = m.roc_curve([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
        self.assertEqual([(p.x, p.y) for p in points], [(0.0, 0.0), (1.0, 1.0)])
        self.assertAlmostEqual(trapezoid(points), 0.5)

    def test_positive_label_selects_the_other_class(self):
        flipped = m.roc_curve(["n", "n", "p", "p"], S, positive_label="p")
        self.assertEqual(flipped, m.roc_curve(Y, S))

    def test_needs_both_classes_and_at_most_two(self):
        with self.assertRaises(MetricError):
            m.roc_curve([1, 1], [0.2, 0.9])
        with self.assertRaises(MetricError):
            m.roc_curve([0, 1, 2], [0.1, 0.2, 0.3])
        with self.assertRaises(MetricError):
            m.roc_curve([0, 1], [0.1])


class PrecisionRecallCurveTests(unittest.TestCase):
    def test_hand_computed_points(self):
        points = m.precision_recall_curve(Y, S)
        got = [(p.x, p.y, p.threshold) for p in points]
        self.assertEqual(
            got,
            [(0.5, 1.0, 0.8), (0.5, 0.5, 0.4), (1.0, 2 / 3, 0.35), (1.0, 0.5, 0.1)],
        )

    def test_step_sum_equals_average_precision(self):
        points = m.precision_recall_curve(Y, S)
        previous, total = 0.0, 0.0
        for p in points:
            total += (p.x - previous) * p.y
            previous = p.x
        self.assertAlmostEqual(total, 0.5 + 0.5 * (2 / 3))
        self.assertAlmostEqual(total, m.average_precision(Y, S))

    def test_ties_are_taken_together(self):
        points = m.precision_recall_curve([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])
        self.assertEqual([(p.x, p.y) for p in points], [(1.0, 0.5)])

    def test_needs_a_positive(self):
        with self.assertRaises(MetricError):
            m.precision_recall_curve([0, 0], [0.1, 0.2])


class BrierTests(unittest.TestCase):
    def test_hand_computed(self):
        # errors 0.1, 0.2, 0.4, 0.4 -> squares 0.01, 0.04, 0.16, 0.16 -> 0.37 / 4
        self.assertAlmostEqual(m.brier_score([1, 0, 1, 0], [0.9, 0.2, 0.6, 0.4]), 0.0925)

    def test_extremes(self):
        self.assertEqual(m.brier_score([1, 0], [1.0, 0.0]), 0.0)
        self.assertEqual(m.brier_score([1, 0], [0.0, 1.0]), 1.0)
        self.assertEqual(m.brier_score([1, 0, 1, 0], [0.5] * 4), 0.25)

    def test_positive_label_and_validation(self):
        self.assertAlmostEqual(m.brier_score(["a", "b"], [0.8, 0.3], positive_label="a"), 0.065)
        for probs in ([1.2, 0.1], [-0.1, 0.1], [float("nan"), 0.1]):
            with self.subTest(probs), self.assertRaises(MetricError):
                m.brier_score([1, 0], probs)
        with self.assertRaises(MetricError):
            m.brier_score([0, 1, 2], [0.1, 0.2, 0.3])


class CalibrationTests(unittest.TestCase):
    Y = [0, 0, 1, 1, 0]
    P = [0.1, 0.2, 0.8, 0.9, 0.7]

    def test_hand_computed_bins(self):
        bins = m.calibration_bins(self.Y, self.P, n_bins=2)
        self.assertEqual([(b.low, b.high, b.count) for b in bins], [(0.0, 0.5, 2), (0.5, 1.0, 3)])
        self.assertAlmostEqual(bins[0].mean_predicted, 0.15)
        self.assertEqual(bins[0].observed_rate, 0.0)
        self.assertAlmostEqual(bins[1].mean_predicted, 0.8)
        self.assertAlmostEqual(bins[1].observed_rate, 2 / 3)

    def test_expected_calibration_error(self):
        # 2/5 * |0 - 0.15| + 3/5 * |2/3 - 0.8| = 0.06 + 0.08
        self.assertAlmostEqual(m.expected_calibration_error(self.Y, self.P, n_bins=2), 0.14)

    def test_a_perfectly_calibrated_model(self):
        # ten rows at 0.2 with two positives, ten at 0.8 with eight positives
        y = [1] * 2 + [0] * 8 + [1] * 8 + [0] * 2
        p = [0.2] * 10 + [0.8] * 10
        self.assertAlmostEqual(m.expected_calibration_error(y, p, n_bins=5), 0.0)

    def test_probability_one_falls_in_the_last_bin_and_empty_bins_are_skipped(self):
        bins = m.calibration_bins([1, 0], [1.0, 0.0], n_bins=10)
        self.assertEqual([(b.low, b.count) for b in bins], [(0.0, 1), (0.9, 1)])
        self.assertEqual(bins[1].high, 1.0)

    def test_every_row_is_counted_once(self):
        probs = [i / 37 for i in range(38)]
        y = [i % 2 for i in range(38)]
        for n_bins in (1, 3, 10, 100):
            self.assertEqual(sum(b.count for b in m.calibration_bins(y, probs, n_bins=n_bins)), 38)

    def test_bad_bin_counts(self):
        for bad in (0, -1, 101, True, 2.5, "3"):
            with self.subTest(bad), self.assertRaises(MetricError):
                m.calibration_bins([0, 1], [0.1, 0.9], n_bins=bad)
        m.calibration_bins([0, 1], [0.1, 0.9], n_bins=100)


def two_class_table(n=40):
    rows = [(f"{i * 0.1:.1f}", "0", "a") for i in range(n // 2)]
    rows += [(f"{10 + i * 0.1:.1f}", "5", "b") for i in range(n // 2)]
    return Table(("x1", "x2", "label"), tuple(rows))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)

    def run_it(self, table=None, **overrides):
        fields = {
            "experiment_id": "cal", "model": "knn", "seed": 3, "dataset_id": "d",
            "metrics": ("brier_score", "ece"), "params": {"k": 3}, "stratify": True,
        }  # fmt: skip
        fields.update(overrides)
        return run_experiment(
            self.store, ExperimentConfig(**fields), table or two_class_table(), "label",
            environment={"python": "3.12.1", "lib:numpy": None},
        )  # fmt: skip

    def test_a_clean_separation_is_perfectly_scored(self):
        result = self.run_it()
        self.assertEqual(result.metrics, {"brier_score": 0.0, "ece": 0.0})

    def test_cross_validation_reports_a_mean_and_spread(self):
        result = self.run_it(cv_folds=4)
        self.assertEqual(set(result.metrics), {"brier_score", "brier_score_std", "ece", "ece_std"})
        self.assertEqual(result.metrics["brier_score"], 0.0)

    def test_three_classes_are_rejected_before_a_run_exists(self):
        rows = [(str(i), c) for i in range(5) for c in "abc"]
        table = Table(("x", "label"), tuple(rows))
        with self.assertRaisesRegex(RunnerError, "brier_score needs exactly two classes"):
            self.run_it(table, metrics=("brier_score",))
        self.assertEqual(self.store.list_runs(), [])

    def test_regressors_reject_them(self):
        line = Table(("x", "y"), tuple((str(i), str(2 * i)) for i in range(20)))
        with self.assertRaisesRegex(RunnerError, "do not apply"):
            run_experiment(
                self.store,
                ExperimentConfig("r", "linear_regression", 1, "l", ("brier_score",), {}),
                line, "y",
            )  # fmt: skip

    def test_directions_and_config_validation(self):
        self.assertFalse(higher_is_better("brier_score"))
        self.assertFalse(higher_is_better("ece"))
        self.assertFalse(higher_is_better("fold/ece"))
        with self.assertRaises(ConfigError):
            ExperimentConfig("e", "knn", 1, "d", ("brier",), {})


if __name__ == "__main__":
    unittest.main()
