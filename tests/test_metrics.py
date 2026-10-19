"""Metric tests. Expected values are worked out by hand (see comments), not produced by the code."""

import math
import random
import unittest

from modelbench.errors import MetricError
from modelbench.metrics import (
    accuracy,
    confusion_matrix,
    log_loss,
    mae,
    mse,
    per_class_metrics,
    precision_recall_f1,
    r2,
    rmse,
    roc_auc,
)

# Three classes, two samples each.
#   true: 0 1 2 0 1 2
#   pred: 0 2 1 0 0 1
# Matrix rows=true, cols=pred:  [[2,0,0],[1,0,1],[0,2,0]]
THREE_TRUE = [0, 1, 2, 0, 1, 2]
THREE_PRED = [0, 2, 1, 0, 0, 1]

# Binary: tp=3, fn=1, fp=1, tn=5  ->  precision = recall = f1 = 0.75, accuracy = 0.8
BIN_TRUE = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
BIN_PRED = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]


class ConfusionMatrixTests(unittest.TestCase):
    def test_multiclass_matrix(self):
        cm = confusion_matrix(THREE_TRUE, THREE_PRED)
        self.assertEqual(cm.labels, (0, 1, 2))
        self.assertEqual(cm.matrix, ((2, 0, 0), (1, 0, 1), (0, 2, 0)))
        self.assertEqual(cm.total, 6)

    def test_binary_matrix(self):
        cm = confusion_matrix(BIN_TRUE, BIN_PRED)
        self.assertEqual(cm.labels, (0, 1))
        self.assertEqual(cm.matrix, ((5, 1), (1, 3)))

    def test_string_labels_sorted_and_explicit_order_respected(self):
        true, pred = ["cat", "dog", "dog"], ["cat", "cat", "dog"]
        self.assertEqual(confusion_matrix(true, pred).labels, ("cat", "dog"))
        cm = confusion_matrix(true, pred, labels=["dog", "cat"])
        self.assertEqual(cm.matrix, ((1, 1), (0, 1)))

    def test_labels_absent_from_data_get_zero_rows(self):
        cm = confusion_matrix([0, 1], [0, 1], labels=[0, 1, 2])
        self.assertEqual(cm.matrix, ((1, 0, 0), (0, 1, 0), (0, 0, 0)))

    def test_unorderable_labels_keep_first_seen_order(self):
        cm = confusion_matrix([1, "a"], [1, "a"])
        self.assertEqual(cm.labels, (1, "a"))

    def test_value_outside_given_labels_is_rejected(self):
        with self.assertRaises(MetricError):
            confusion_matrix([0, 1], [0, 2], labels=[0, 1])
        with self.assertRaises(MetricError):
            confusion_matrix([0, 1], [0, 1], labels=[0, 0, 1])


class AccuracyTests(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(accuracy(THREE_TRUE, THREE_PRED), 2 / 6)
        self.assertAlmostEqual(accuracy(BIN_TRUE, BIN_PRED), 0.8)
        self.assertEqual(accuracy([1, 2], [1, 2]), 1.0)
        self.assertEqual(accuracy([1, 2], [2, 1]), 0.0)


class PrecisionRecallF1Tests(unittest.TestCase):
    def test_per_class_known_values(self):
        # class 0: predicted 3 times, 2 correct -> P=2/3; 2 true, 2 found -> R=1; F1=0.8
        # classes 1 and 2: never correct -> all zeros
        result = per_class_metrics(THREE_TRUE, THREE_PRED)
        self.assertAlmostEqual(result[0].precision, 2 / 3)
        self.assertAlmostEqual(result[0].recall, 1.0)
        self.assertAlmostEqual(result[0].f1, 0.8)
        self.assertEqual(result[0].support, 2)
        for label in (1, 2):
            self.assertEqual((result[label].precision, result[label].recall), (0.0, 0.0))
            self.assertEqual(result[label].f1, 0.0)

    def test_macro_known_values(self):
        # macro P = (2/3 + 0 + 0)/3 = 2/9, macro R = 1/3, macro F1 = 0.8/3
        prf = precision_recall_f1(THREE_TRUE, THREE_PRED, average="macro")
        self.assertAlmostEqual(prf.precision, 2 / 9)
        self.assertAlmostEqual(prf.recall, 1 / 3)
        self.assertAlmostEqual(prf.f1, 0.8 / 3)
        self.assertEqual(prf.support, 6)

    def test_micro_equals_accuracy_for_single_label_data(self):
        prf = precision_recall_f1(THREE_TRUE, THREE_PRED, average="micro")
        self.assertAlmostEqual(prf.precision, 1 / 3)
        self.assertAlmostEqual(prf.recall, 1 / 3)
        self.assertAlmostEqual(prf.f1, 1 / 3)

    def test_weighted_uses_support(self):
        # supports are 3 and 1, so class "a" counts three times as much as class "b".
        #   true: a a a b   pred: a a b b
        #   a: predicted 2 (both right) -> P=1; 3 true, 2 found -> R=2/3; F1=0.8
        #   b: predicted 2 (1 right) -> P=0.5; 1 true, 1 found -> R=1; F1=2/3
        true, pred = ["a", "a", "a", "b"], ["a", "a", "b", "b"]
        prf = precision_recall_f1(true, pred, average="weighted")
        self.assertAlmostEqual(prf.precision, (3 * 1.0 + 1 * 0.5) / 4)
        self.assertAlmostEqual(prf.recall, (3 * (2 / 3) + 1 * 1.0) / 4)
        self.assertAlmostEqual(prf.f1, (3 * 0.8 + 1 * (2 / 3)) / 4)

    def test_binary_average(self):
        prf = precision_recall_f1(BIN_TRUE, BIN_PRED, average="binary", positive_label=1)
        self.assertAlmostEqual(prf.precision, 0.75)
        self.assertAlmostEqual(prf.recall, 0.75)
        self.assertAlmostEqual(prf.f1, 0.75)
        self.assertEqual(prf.support, 4)
        negative = precision_recall_f1(BIN_TRUE, BIN_PRED, average="binary", positive_label=0)
        self.assertAlmostEqual(negative.precision, 5 / 6)
        self.assertAlmostEqual(negative.recall, 5 / 6)

    def test_never_predicted_class_scores_zero_not_error(self):
        prf = precision_recall_f1([0, 1], [0, 0], average="binary", positive_label=1)
        self.assertEqual((prf.precision, prf.recall, prf.f1), (0.0, 0.0, 0.0))

    def test_invalid_arguments(self):
        with self.assertRaises(MetricError):
            precision_recall_f1([0, 1], [0, 1], average="median")
        with self.assertRaises(MetricError):
            precision_recall_f1([0, 1], [0, 1], average="binary", positive_label=7)


class RocAucTests(unittest.TestCase):
    def test_known_value(self):
        # positives {0.35, 0.8}, negatives {0.1, 0.4}: pairs won = 3 of 4 -> 0.75
        self.assertAlmostEqual(roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)

    def test_perfect_and_inverted_separation(self):
        self.assertEqual(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)

    def test_ties_count_half(self):
        self.assertAlmostEqual(roc_auc([0, 1], [0.5, 0.5]), 0.5)
        # one positive tied with one of two negatives: (1 + 0.5) / 2 = 0.75
        self.assertAlmostEqual(roc_auc([0, 0, 1], [0.2, 0.7, 0.7]), 0.75)

    def test_positive_label_selects_the_positive_class(self):
        self.assertAlmostEqual(roc_auc(["n", "n", "p", "p"], [0.1, 0.4, 0.35, 0.8], "p"), 0.75)

    def test_invalid_inputs(self):
        with self.assertRaises(MetricError):
            roc_auc([1, 1], [0.1, 0.2])  # no negatives
        with self.assertRaises(MetricError):
            roc_auc([0, 1, 2], [0.1, 0.2, 0.3])  # three classes
        with self.assertRaises(MetricError):
            roc_auc([0, 1], [0.1, float("nan")])


class LogLossTests(unittest.TestCase):
    def test_binary_single_number_probabilities(self):
        # true [1, 0], P(1) [0.8, 0.1]: -(ln 0.8 + ln 0.9) / 2 = 0.1642520335
        self.assertAlmostEqual(log_loss([1, 0], [0.8, 0.1]), 0.1642520335, places=9)

    def test_probability_rows_match_single_number_form(self):
        rows = [[0.2, 0.8], [0.9, 0.1]]
        self.assertAlmostEqual(log_loss([1, 0], rows), log_loss([1, 0], [0.8, 0.1]))

    def test_multiclass_rows(self):
        # -(ln 0.7 + ln 0.5 + ln 0.6) / 3, with ln 0.7 = -0.3566749439,
        # ln 0.5 = -0.6931471806, ln 0.6 = -0.5108256238  ->  about 0.5202
        rows = [[0.7, 0.2, 0.1], [0.25, 0.5, 0.25], [0.1, 0.3, 0.6]]
        expected = (0.3566749439 + 0.6931471806 + 0.5108256238) / 3
        self.assertAlmostEqual(log_loss([0, 1, 2], rows), expected, places=9)

    def test_confident_wrong_answer_is_clipped_not_infinite(self):
        value = log_loss([1], [0.0], labels=[0, 1])
        self.assertTrue(math.isfinite(value))
        self.assertAlmostEqual(value, -math.log(1e-15))

    def test_single_class_truth_needs_explicit_labels(self):
        with self.assertRaises(MetricError):
            log_loss([1], [0.5])  # cannot infer a second class from the data

    def test_invalid_inputs(self):
        with self.assertRaises(MetricError):
            log_loss([0, 1], [[0.5, 0.6], [0.5, 0.5]])  # row does not sum to 1
        with self.assertRaises(MetricError):
            log_loss([0, 1], [[1.0], [1.0]])  # wrong width
        with self.assertRaises(MetricError):
            log_loss([0, 1, 2], [0.1, 0.2, 0.3])  # single numbers need two labels
        with self.assertRaises(MetricError):
            log_loss([5], [0.5], labels=[0, 1])


class RegressionTests(unittest.TestCase):
    TRUE = [3, -0.5, 2, 7]
    PRED = [2.5, 0, 2, 8]

    def test_known_values(self):
        # abs errors .5 .5 0 1 -> MAE .5; squared .25 .25 0 1 -> MSE .375
        # mean(true)=2.875, SS_tot=29.1875, SS_res=1.5 -> R2 = 1 - 1.5/29.1875
        self.assertAlmostEqual(mae(self.TRUE, self.PRED), 0.5)
        self.assertAlmostEqual(mse(self.TRUE, self.PRED), 0.375)
        self.assertAlmostEqual(rmse(self.TRUE, self.PRED), 0.6123724357, places=9)
        self.assertAlmostEqual(r2(self.TRUE, self.PRED), 0.9486081370, places=9)

    def test_perfect_prediction(self):
        self.assertEqual((mae(self.TRUE, self.TRUE), rmse(self.TRUE, self.TRUE)), (0.0, 0.0))
        self.assertEqual(r2(self.TRUE, self.TRUE), 1.0)

    def test_predicting_the_mean_gives_zero_r2(self):
        mean = sum(self.TRUE) / len(self.TRUE)
        self.assertAlmostEqual(r2(self.TRUE, [mean] * 4), 0.0)

    def test_constant_target(self):
        self.assertEqual(r2([2, 2, 2], [2, 2, 2]), 1.0)
        self.assertEqual(r2([2, 2, 2], [2, 2, 3]), 0.0)

    def test_rejects_bad_numbers(self):
        for bad in ([1, "x"], [1, None], [1, True], [1, float("inf")]):
            with self.assertRaises(MetricError, msg=repr(bad)):
                mae([1, 2], bad)


class InputValidationTests(unittest.TestCase):
    def test_empty_and_mismatched_inputs(self):
        for func in (accuracy, mae, mse, r2, confusion_matrix):
            with self.assertRaises(MetricError, msg=func.__name__):
                func([], [])
            with self.assertRaises(MetricError, msg=func.__name__):
                func([1, 2], [1])

    def test_metric_error_is_a_value_error(self):
        with self.assertRaises(ValueError):
            accuracy([], [])


class PropertyTests(unittest.TestCase):
    """Seeded randomised checks of invariants that must hold for any input."""

    def setUp(self):
        self.rng = random.Random(20261019)

    def random_labels(self, n, k):
        return [self.rng.randrange(k) for _ in range(n)]

    def test_confusion_matrix_total_and_trace(self):
        for _ in range(50):
            n, k = self.rng.randint(1, 60), self.rng.randint(1, 5)
            true, pred = self.random_labels(n, k), self.random_labels(n, k)
            cm = confusion_matrix(true, pred)
            self.assertEqual(cm.total, n)
            trace = sum(cm.matrix[i][i] for i in range(len(cm.labels)))
            self.assertAlmostEqual(trace / n, accuracy(true, pred))

    def test_results_do_not_depend_on_row_order(self):
        for _ in range(30):
            n = self.rng.randint(2, 40)
            true, pred = self.random_labels(n, 3), self.random_labels(n, 3)
            order = list(range(n))
            self.rng.shuffle(order)
            true2, pred2 = [true[i] for i in order], [pred[i] for i in order]
            self.assertEqual(confusion_matrix(true, pred), confusion_matrix(true2, pred2))
            for average in ("macro", "micro", "weighted"):
                a = precision_recall_f1(true, pred, average=average)
                b = precision_recall_f1(true2, pred2, average=average)
                self.assertAlmostEqual(a.f1, b.f1)
                self.assertAlmostEqual(a.precision, b.precision)

    def test_scores_stay_in_unit_interval_and_micro_equals_accuracy(self):
        for _ in range(50):
            n, k = self.rng.randint(1, 50), self.rng.randint(2, 5)
            true, pred = self.random_labels(n, k), self.random_labels(n, k)
            for average in ("macro", "micro", "weighted"):
                prf = precision_recall_f1(true, pred, average=average)
                for value in (prf.precision, prf.recall, prf.f1):
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)
            micro = precision_recall_f1(true, pred, average="micro")
            self.assertAlmostEqual(micro.f1, accuracy(true, pred))

    def test_auc_invariant_to_monotonic_transform_and_complements_under_negation(self):
        for _ in range(40):
            n = self.rng.randint(4, 40)
            true = self.random_labels(n, 2)
            if len(set(true)) < 2:
                true[0], true[1] = 0, 1
            scores = [round(self.rng.random(), 2) for _ in range(n)]  # rounding creates ties
            auc = roc_auc(true, scores)
            self.assertGreaterEqual(auc, 0.0)
            self.assertLessEqual(auc, 1.0)
            self.assertAlmostEqual(roc_auc(true, [3 * s + 7 for s in scores]), auc)
            self.assertAlmostEqual(roc_auc(true, [-s for s in scores]), 1.0 - auc)

    def test_regression_relationships(self):
        for _ in range(50):
            n = self.rng.randint(1, 30)
            true = [self.rng.uniform(-10, 10) for _ in range(n)]
            pred = [t + self.rng.uniform(-3, 3) for t in true]
            self.assertGreaterEqual(rmse(true, pred) + 1e-12, mae(true, pred))
            self.assertAlmostEqual(rmse(true, pred) ** 2, mse(true, pred))
            self.assertLessEqual(r2(true, pred), 1.0)

    def test_log_loss_is_nonnegative_and_zero_only_when_certain(self):
        for _ in range(30):
            n = self.rng.randint(1, 20)
            true = self.random_labels(n, 2)
            probs = [self.rng.random() for _ in range(n)]
            self.assertGreaterEqual(log_loss(true, probs, labels=[0, 1]), 0.0)
        self.assertAlmostEqual(log_loss([1, 0], [1.0, 0.0]), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
