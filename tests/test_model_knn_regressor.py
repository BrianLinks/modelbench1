import unittest

from modelbench.errors import ModelError
from modelbench.models import create_regressor
from modelbench.models.knn_regressor import KNNRegressor

X = [[0.0], [1.0], [2.0], [10.0]]
Y = [0.0, 10.0, 20.0, 100.0]


def fitted(**params):
    model = KNNRegressor(**params)
    model.fit(X, Y)
    return model


class PredictionTests(unittest.TestCase):
    def test_averages_the_k_nearest_targets(self):
        model = fitted(k=2)
        # 0.4 is nearest rows 0 and 1 (0.4, 0.6): (0 + 10) / 2. 1.6 is nearest rows 2 and 1.
        self.assertEqual(model.predict([[0.4], [1.6]]), [5.0, 15.0])

    def test_k_one_copies_the_nearest_target_and_k_all_gives_the_mean(self):
        self.assertEqual(fitted(k=1).predict([[9.0], [-5.0]]), [100.0, 0.0])
        self.assertEqual(fitted(k=4).predict([[0.0]]), [32.5])

    def test_equal_distances_go_to_the_earlier_training_row(self):
        model = KNNRegressor(k=1)
        model.fit([[0.0], [2.0]], [1.0, 3.0])
        self.assertEqual(model.predict([[1.0]]), [1.0])
        model = KNNRegressor(k=1)
        model.fit([[2.0], [0.0]], [3.0, 1.0])  # the same rows in the other order
        self.assertEqual(model.predict([[1.0]]), [3.0])

    def test_distance_weights_favour_closer_rows(self):
        model = KNNRegressor(k=2, weights="distance")
        model.fit([[0.0], [3.0]], [0.0, 30.0])
        # distances 1 and 2 weigh 1 and 0.5: (0 * 1 + 30 * 0.5) / 1.5
        self.assertAlmostEqual(model.predict([[1.0]])[0], 10.0)

    def test_an_exact_match_overrides_the_other_neighbours(self):
        rows, targets = [[0.0], [0.0], [5.0]], [2.0, 4.0, 100.0]
        weighted = KNNRegressor(k=3, weights="distance")
        weighted.fit(rows, targets)
        self.assertEqual(weighted.predict([[0.0]]), [3.0])
        uniform = KNNRegressor(k=3)
        uniform.fit(rows, targets)
        self.assertAlmostEqual(uniform.predict([[0.0]])[0], 106.0 / 3)

    def test_the_distance_metric_changes_who_is_nearest(self):
        rows, targets = [[3.0, 0.0], [2.0, 2.0]], [10.0, 20.0]
        euclid = KNNRegressor(k=1, metric="euclidean")
        euclid.fit(rows, targets)
        manhattan = KNNRegressor(k=1, metric="manhattan")
        manhattan.fit(rows, targets)
        # from the origin: euclidean 3.0 vs 2.83, manhattan 3 vs 4
        self.assertEqual(euclid.predict([[0.0, 0.0]]), [20.0])
        self.assertEqual(manhattan.predict([[0.0, 0.0]]), [10.0])

    def test_changing_the_input_after_fitting_does_not_change_the_model(self):
        rows = [[0.0], [1.0]]
        model = KNNRegressor(k=1)
        model.fit(rows, [5.0, 6.0])
        rows[0][0] = 100.0
        self.assertEqual(model.predict([[0.0]]), [5.0])

    def test_refit_replaces_the_training_data(self):
        model = fitted(k=1)
        model.fit([[0.0]], [7.0])
        self.assertEqual(model.predict([[3.0]]), [7.0])

    def test_integer_targets_give_float_predictions(self):
        model = KNNRegressor(k=1)
        model.fit([[0], [1]], [3, 4])
        (prediction,) = model.predict([[0]])
        self.assertIs(type(prediction), float)


class ValidationTests(unittest.TestCase):
    def test_bad_parameters(self):
        for params in (
            {"k": 0},
            {"k": -1},
            {"k": 2.0},
            {"k": True},
            {"k": "3"},
            {"metric": "cosine"},
            {"weights": "gaussian"},
        ):
            with self.subTest(params), self.assertRaises(ModelError):
                KNNRegressor(**params)

    def test_k_larger_than_the_training_set_is_an_error_at_fit(self):
        with self.assertRaisesRegex(ModelError, "larger than the 4 training rows"):
            KNNRegressor(k=5).fit(X, Y)

    def test_not_fitted_and_wrong_width(self):
        with self.assertRaisesRegex(ModelError, "not been fitted"):
            KNNRegressor().predict([[1.0]])
        with self.assertRaisesRegex(ModelError, "1 features"):
            fitted(k=1).predict([[1.0, 2.0]])

    def test_registry_and_params_round_trip(self):
        model = create_regressor("knn_regressor", k=3, weights="distance")
        self.assertEqual(model.get_params(), {"k": 3, "metric": "euclidean", "weights": "distance"})
        with self.assertRaises(ModelError):
            create_regressor("knn_regressor", depth=3)


if __name__ == "__main__":
    unittest.main()
