import math
import random
import unittest

from modelbench.errors import ModelError
from modelbench.models import create_model
from modelbench.models.naive_bayes import GaussianNB


def fitted(X, y, **params):
    model = GaussianNB(**params)
    model.fit(X, y)
    return model


def logistic(z):
    return 1.0 / (1.0 + math.exp(-z))


# class "a": 1, 3 (mean 2, variance 1); class "b": 7, 9 (mean 8, variance 1)
TWO_BLOBS_X = [[1.0], [3.0], [7.0], [9.0]]
TWO_BLOBS_Y = ["a", "a", "b", "b"]


class KnownAnswerTests(unittest.TestCase):
    def test_equal_variances_give_a_logistic_posterior(self):
        # query 3: squared distances to the means are 1 and 25, so the log-odds are
        # (25 - 1) / (2 * 1) = 12 and P(a) = 1 / (1 + e^-12)
        (row,) = fitted(TWO_BLOBS_X, TWO_BLOBS_Y).predict_proba([[3.0]])
        self.assertAlmostEqual(row[0], logistic(12), places=7)
        self.assertAlmostEqual(row[1], 1 - logistic(12), places=7)

    def test_midpoint_with_equal_priors_is_exactly_even_and_ties_go_to_the_first_class(self):
        model = fitted(TWO_BLOBS_X, TWO_BLOBS_Y)
        self.assertEqual(model.predict_proba([[5.0]]), [[0.5, 0.5]])
        self.assertEqual(model.predict([[5.0]]), ["a"])

    def test_class_priors_shift_the_posterior(self):
        # 4 rows of "a" (1, 3, 1, 3: mean 2, variance 1) against 2 of "b": priors 2/3 and 1/3.
        # At the midpoint the likelihoods cancel, leaving the priors.
        X = [[1.0], [3.0], [1.0], [3.0], [7.0], [9.0]]
        y = ["a", "a", "a", "a", "b", "b"]
        (row,) = fitted(X, y).predict_proba([[5.0]])
        self.assertAlmostEqual(row[0], 2 / 3, places=7)
        self.assertAlmostEqual(row[1], 1 / 3, places=7)

    def test_features_combine_independently(self):
        # two identical features double the log-odds: 24, not 12
        X = [[1.0, 1.0], [3.0, 3.0], [7.0, 7.0], [9.0, 9.0]]
        (row,) = fitted(X, TWO_BLOBS_Y).predict_proba([[3.0, 3.0]])
        self.assertAlmostEqual(row[0], logistic(24), places=7)

    def test_a_wider_class_is_less_certain_at_its_centre(self):
        # "a" has variance 1 and "b" variance 9, both centred on 0. At 0 the densities are in
        # the ratio sqrt(9) : 1 = 3 : 1, so P(a) = 3 / 4.
        X = [[-1.0], [1.0], [-3.0], [3.0]]
        (row,) = fitted(X, ["a", "a", "b", "b"]).predict_proba([[0.0]])
        self.assertAlmostEqual(row[0], 0.75, places=6)

    def test_single_class_is_certain(self):
        model = fitted([[0.0], [1.0]], ["only", "only"])
        self.assertEqual(model.predict_proba([[5.0]]), [[1.0]])


class NumericalStabilityTests(unittest.TestCase):
    def check_valid(self, rows):
        for row in rows:
            self.assertTrue(all(math.isfinite(p) for p in row), row)
            self.assertAlmostEqual(sum(row), 1.0)

    def test_constant_features_are_smoothed_not_divided_by_zero(self):
        model = fitted([[2.0], [2.0], [5.0], [5.0]], ["a", "a", "b", "b"])
        near_a, middle, far = model.predict_proba([[2.0], [3.5], [1000.0]])
        self.check_valid([near_a, middle, far])
        self.assertAlmostEqual(near_a[0], 1.0)
        self.assertAlmostEqual(middle[0], 0.5)
        self.assertAlmostEqual(far[1], 1.0)

    def test_an_all_constant_dataset_still_works(self):
        model = fitted([[1.0], [1.0], [1.0], [1.0]], ["a", "a", "b", "b"])
        self.check_valid(model.predict_proba([[1.0], [2.0]]))

    def test_extreme_queries_do_not_underflow_or_overflow(self):
        model = fitted(TWO_BLOBS_X, TWO_BLOBS_Y)
        rows = model.predict_proba([[1e6], [-1e6], [1e150]])
        self.check_valid(rows)
        self.assertEqual(model.predict([[1e6], [-1e6]]), ["b", "a"])

    def test_more_smoothing_softens_the_posterior(self):
        sharp = fitted([[2.0], [2.0], [5.0], [5.0]], ["a", "a", "b", "b"], var_smoothing=1e-9)
        soft = fitted([[2.0], [2.0], [5.0], [5.0]], ["a", "a", "b", "b"], var_smoothing=1.0)
        self.assertGreater(sharp.predict_proba([[2.5]])[0][0], soft.predict_proba([[2.5]])[0][0])


class ParameterTests(unittest.TestCase):
    def test_var_smoothing_validation(self):
        for bad in (0, -1e-9, float("nan"), float("inf"), True, "1e-9", None):
            with self.assertRaises(ModelError, msg=repr(bad)):
                GaussianNB(var_smoothing=bad)
        self.assertEqual(GaussianNB(var_smoothing=1).get_params(), {"var_smoothing": 1.0})

    def test_registry_construction(self):
        self.assertEqual(create_model("gaussian_nb", var_smoothing=1e-6).var_smoothing, 1e-6)
        with self.assertRaises(ModelError):
            create_model("gaussian_nb", alpha=1.0)


class PropertyTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(20261020)

    def blobs(self, per_class, centres, spread=0.5):
        X, y = [], []
        for label, centre in centres.items():
            for _ in range(per_class):
                X.append([self.rng.gauss(c, spread) for c in centre])
                y.append(label)
        return X, y

    def test_separated_blobs_are_classified_perfectly(self):
        centres = {"a": (0.0, 0.0), "b": (10.0, 0.0), "c": (0.0, 10.0)}
        X, y = self.blobs(30, centres)
        model = fitted(X, y)
        self.assertEqual(model.predict(X), y)
        test_X, test_y = self.blobs(20, centres)
        self.assertEqual(model.predict(test_X), test_y)

    def test_training_row_order_does_not_change_the_model(self):
        X, y = self.blobs(25, {"a": (0.0, 1.0), "b": (2.0, 0.0)}, spread=1.5)
        queries, _ = self.blobs(10, {"a": (1.0, 0.5)}, spread=2.0)
        baseline = fitted(X, y).predict_proba(queries)
        for _ in range(5):
            order = list(range(len(X)))
            self.rng.shuffle(order)
            shuffled = fitted([X[i] for i in order], [y[i] for i in order])
            self.assertEqual(shuffled.predict_proba(queries), baseline)

    def test_probabilities_are_always_valid(self):
        for _ in range(30):
            n_features = self.rng.randint(1, 4)
            centres = {c: tuple(self.rng.uniform(-5, 5) for _ in range(n_features)) for c in "abc"}
            X, y = self.blobs(self.rng.randint(2, 15), centres, spread=self.rng.uniform(0.1, 3))
            queries = [[self.rng.uniform(-20, 20) for _ in range(n_features)] for _ in range(8)]
            for row in fitted(X, y).predict_proba(queries):
                self.assertEqual(len(row), 3)
                self.assertAlmostEqual(sum(row), 1.0)
                self.assertTrue(all(0.0 <= p <= 1.0 for p in row))


if __name__ == "__main__":
    unittest.main()
