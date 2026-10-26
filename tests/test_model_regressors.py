import math
import random
import unittest

from modelbench.errors import ModelError
from modelbench.models import (
    REGISTRY,
    REGRESSORS,
    all_models,
    available_models,
    available_regressors,
    create_regressor,
    model_kind,
)
from modelbench.models.linear_regression import LinearRegression, cholesky_solve
from modelbench.models.mean import MeanRegressor


class MeanRegressorTests(unittest.TestCase):
    def test_predicts_the_training_mean_for_every_row(self):
        model = MeanRegressor()
        model.fit([[0.0], [1.0], [2.0]], [1, 2, 6])
        self.assertEqual(model.predict([[9.0], [-4.0]]), [3.0, 3.0])
        self.assertEqual(model.get_params(), {})

    def test_refit_replaces_the_mean(self):
        model = MeanRegressor()
        model.fit([[0.0]], [10])
        model.fit([[0.0], [1.0]], [2, 4])
        self.assertEqual(model.predict([[0.0]]), [3.0])


class CholeskyTests(unittest.TestCase):
    def test_solves_a_known_system(self):
        # [[4, 2], [2, 3]] x = [10, 8]  ->  x = [1.75, 1.5]
        x = cholesky_solve([[4.0, 2.0], [2.0, 3.0]], [10.0, 8.0])
        self.assertAlmostEqual(x[0], 1.75)
        self.assertAlmostEqual(x[1], 1.5)

    def test_singular_matrix_is_an_error(self):
        with self.assertRaises(ModelError):
            cholesky_solve([[1.0, 1.0], [1.0, 1.0]], [1.0, 1.0])


class LinearRegressionTests(unittest.TestCase):
    def test_recovers_an_exact_line(self):
        model = LinearRegression()
        model.fit([[0.0], [1.0], [2.0], [3.0]], [1.0, 3.0, 5.0, 7.0])
        self.assertAlmostEqual(model.coef_[0], 2.0)
        self.assertAlmostEqual(model.intercept_, 1.0)
        for got, want in zip(model.predict([[10.0], [-1.0]]), [21.0, -1.0], strict=True):
            self.assertAlmostEqual(got, want)

    def test_recovers_an_exact_plane_with_different_feature_scales(self):
        rng = random.Random(3)
        X = [[rng.uniform(-1, 1), rng.uniform(-1000, 1000)] for _ in range(30)]
        y = [3 * a - 0.002 * b + 5 for a, b in X]
        model = LinearRegression()
        model.fit(X, y)
        self.assertAlmostEqual(model.coef_[0], 3.0, places=6)
        self.assertAlmostEqual(model.coef_[1], -0.002, places=9)
        self.assertAlmostEqual(model.intercept_, 5.0, places=6)

    def test_ridge_shrinks_by_the_documented_amount(self):
        # z = x = [-1, 1], centred y = [-1, 1]: gram = 1 + l2, rhs = 1, so w = 1 / (1 + l2)
        model = LinearRegression(l2=1.0)
        model.fit([[-1.0], [1.0]], [-1.0, 1.0])
        self.assertAlmostEqual(model.coef_[0], 0.5)
        self.assertAlmostEqual(model.intercept_, 0.0)
        self.assertAlmostEqual(model.predict([[2.0]])[0], 1.0)

    def test_noisy_fit_beats_the_mean_baseline(self):
        rng = random.Random(1)
        X = [[float(i)] for i in range(50)]
        y = [0.5 * i + rng.gauss(0, 1) for i in range(50)]
        line, base = LinearRegression(), MeanRegressor()
        line.fit(X, y)
        base.fit(X, y)

        def sse(model):
            return sum((a - b) ** 2 for a, b in zip(model.predict(X), y, strict=True))

        self.assertLess(sse(line), sse(base) / 10)

    def test_constant_column_gets_zero_weight(self):
        model = LinearRegression()
        model.fit([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]], [2.0, 4.0, 6.0])
        self.assertAlmostEqual(model.coef_[0], 2.0)
        self.assertEqual(model.coef_[1], 0.0)
        self.assertAlmostEqual(model.predict([[4.0, 99.0]])[0], 8.0)  # the constant is ignored

    def test_all_constant_features_predict_the_mean(self):
        model = LinearRegression()
        model.fit([[1.0], [1.0], [1.0]], [1.0, 2.0, 6.0])
        self.assertEqual(model.coef_, [0.0])
        self.assertAlmostEqual(model.predict([[7.0]])[0], 3.0)

    def test_linearly_dependent_features_need_a_penalty(self):
        X = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]
        y = [1.0, 2.0, 3.0, 4.0]
        with self.assertRaisesRegex(ModelError, "l2"):
            LinearRegression().fit(X, y)
        model = LinearRegression(l2=0.1)
        model.fit(X, y)
        self.assertTrue(all(math.isfinite(v) for v in model.predict(X)))

    def test_ridge_penalises_every_weight_equally(self):
        # two uncorrelated standardised features, y = z1 + z2: each weight is 1 / (1 + l2)
        X = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]
        model = LinearRegression(l2=1.0)
        model.fit(X, [-2.0, 0.0, 0.0, 2.0])
        self.assertAlmostEqual(model.coef_[0], 0.5)
        self.assertAlmostEqual(model.coef_[1], 0.5)

    def test_nearly_dependent_features_are_refused_without_a_penalty(self):
        a = [float(i) for i in range(10)]
        X = [[v, v + 1e-7 * (-1) ** i] for i, v in enumerate(a)]
        with self.assertRaises(ModelError):
            LinearRegression().fit(X, a)

    def test_strongly_correlated_features_are_still_solved(self):
        rng = random.Random(0)
        a = [float(i) for i in range(10)]
        X = [[v, v + rng.gauss(0, 0.09)] for v in a]  # correlation about 0.9997
        y = [2 * r[0] - r[1] + 3 for r in X]
        model = LinearRegression()
        model.fit(X, y)
        self.assertAlmostEqual(model.coef_[0], 2.0, places=6)
        self.assertAlmostEqual(model.coef_[1], -1.0, places=6)

    def test_column_constant_up_to_rounding_is_ignored(self):
        # the mean of six 0.1s is not exactly 0.1, so the column's sd is about 1e-17, not 0
        X = [[float(i), 0.1] for i in range(6)]
        model = LinearRegression()
        model.fit(X, [2.0 * i + 1 for i in range(6)])
        self.assertEqual(model.coef_[1], 0.0)
        self.assertAlmostEqual(model.coef_[0], 2.0)
        self.assertAlmostEqual(model.predict([[10.0, 1e300]])[0], 21.0)

    def test_fit_is_deterministic(self):
        X = [[math.sin(i), i % 5] for i in range(20)]
        y = [i * 0.3 for i in range(20)]
        a, b = LinearRegression(l2=0.2), LinearRegression(l2=0.2)
        a.fit(X, y)
        b.fit(X, y)
        self.assertEqual(a.predict(X), b.predict(X))

    def test_parameter_validation(self):
        for bad in (-0.1, float("nan"), float("inf"), "1", True, None):
            with self.assertRaises(ModelError, msg=repr(bad)):
                LinearRegression(l2=bad)
        self.assertEqual(LinearRegression(l2=2).get_params(), {"l2": 2.0})

    def test_unfitted_use_is_an_error(self):
        model = LinearRegression()
        for call in (lambda: model.predict([[1.0]]), lambda: model.coef_, lambda: model.intercept_):
            with self.assertRaises(ModelError):
                call()

    def test_wrong_feature_count_and_bad_input(self):
        model = LinearRegression()
        model.fit([[0.0, 1.0], [1.0, 3.0], [2.0, 2.0]], [1.0, 2.0, 3.0])
        with self.assertRaises(ModelError):
            model.predict([[1.0]])
        with self.assertRaises(ModelError):
            model.predict([[1.0, float("nan")]])

    def test_target_validation(self):
        X = [[0.0], [1.0], [2.0]]
        for bad in ([1, 2], "abc", [1, 2, None], [1, 2, True], [1, 2, "3"],
                    [1, 2, float("nan")], [1, 2, float("inf")], 5):  # fmt: skip
            with self.assertRaises(ModelError, msg=repr(bad)):
                MeanRegressor().fit(X, bad)

    def test_failed_refit_leaves_the_model_unfitted(self):
        model = LinearRegression()
        model.fit([[0.0], [1.0]], [0.0, 1.0])
        with self.assertRaises(ModelError):  # fails inside the solver, after training started
            model.fit([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], [1.0, 2.0, 3.0])
        with self.assertRaises(ModelError):
            model.predict([[0.0]])

    def test_overflow_is_a_model_error(self):
        model = LinearRegression()
        with self.assertRaises(ModelError):
            model.fit([[1e300], [-1e300], [0.0]], [1.0, 2.0, 3.0])
        model.fit([[0.0], [1e-3]], [0.0, 1.0])
        with self.assertRaises(ModelError):  # standardising 1e308 by a tiny sd gives inf
            model.predict([[1e308]])
        self.assertTrue(all(math.isfinite(v) for v in model.predict([[0.5e-3]])))


class RegistryTests(unittest.TestCase):
    def test_kinds_and_listings(self):
        self.assertEqual(
            available_regressors(),
            ("decision_tree_regressor", "knn_regressor", "linear_regression", "mean"),
        )
        self.assertEqual(set(all_models()), set(REGISTRY) | set(REGRESSORS))
        self.assertEqual(list(all_models()), sorted(all_models()))
        self.assertNotIn("mean", available_models())
        self.assertEqual(model_kind("knn"), "classification")
        self.assertEqual(model_kind("linear_regression"), "regression")
        for bad in ("nope", None, ["knn"]):
            with self.assertRaises(ModelError):
                model_kind(bad)

    def test_create_regressor(self):
        self.assertIsInstance(create_regressor("mean"), MeanRegressor)
        self.assertEqual(create_regressor("linear_regression", l2=0.5).get_params(), {"l2": 0.5})
        for call in (
            lambda: create_regressor("knn"),
            lambda: create_regressor(["mean"]),
            lambda: create_regressor("mean", l2=1.0),
        ):
            with self.assertRaises(ModelError):
                call()

    def test_every_regressor_follows_the_interface_contract(self):
        X = [[float(i), float(i % 3)] for i in range(10)]
        y = [float(i) for i in range(10)]
        for name in available_regressors():
            model = create_regressor(name)
            with self.assertRaises(ModelError, msg=name):
                model.predict(X)
            model.fit(X, y)
            predictions = model.predict(X)
            self.assertEqual(len(predictions), 10, name)
            self.assertTrue(all(type(p) is float for p in predictions), name)
            with self.assertRaises(ModelError, msg=name):
                model.predict([[1.0]])
            rebuilt = create_regressor(name, **model.get_params())
            self.assertEqual(rebuilt.get_params(), model.get_params(), name)


if __name__ == "__main__":
    unittest.main()
