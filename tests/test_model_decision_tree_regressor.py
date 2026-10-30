import random
import unittest

from modelbench.errors import ModelError
from modelbench.models import create_regressor
from modelbench.models.decision_tree_regressor import DecisionTreeRegressor


def fitted(X, y, **params):
    model = DecisionTreeRegressor(**params)
    model.fit(X, y)
    return model


def column(values):
    return [[float(v)] for v in values]


class SplitTests(unittest.TestCase):
    def test_a_step_function_splits_at_the_midpoint_and_predicts_the_leaf_means(self):
        model = fitted(column(range(1, 7)), [0, 0, 0, 10, 10, 10])
        self.assertEqual(model.describe().splitlines()[0], "if x0 <= 3.5:")
        self.assertEqual(model.predict(column([2, 5, 100, -100])), [0.0, 10.0, 10.0, 0.0])
        self.assertEqual((model.depth_, model.n_leaves_), (1, 2))

    def test_chooses_the_split_with_the_smallest_squared_error_by_hand(self):
        # y = 1 2 6 7. Cutting after the second row leaves squared errors 0.5 + 0.5 = 1; after
        # the first row 0 + 14 (2, 6, 7 around 5), after the third 14 + 0.
        model = fitted(column([1, 2, 3, 4]), [1, 2, 6, 7], max_depth=1)
        self.assertEqual(model.describe().splitlines()[0], "if x0 <= 2.5:")
        self.assertEqual(model.predict(column([1.2, 3.9])), [1.5, 6.5])

    def test_picks_the_feature_that_explains_the_target(self):
        X = [[5.0, 1.0], [1.0, 2.0], [4.0, 3.0], [2.0, 10.0], [6.0, 11.0], [3.0, 12.0]]
        model = fitted(X, [0, 0, 0, 9, 9, 9])
        self.assertEqual(model.describe().splitlines()[0], "if x1 <= 6.5:")

    def test_equal_scores_go_to_the_lowest_feature_index(self):
        X = [[float(i), float(i)] for i in range(6)]
        model = fitted(X, [0, 0, 0, 5, 5, 5])
        self.assertEqual(model.describe().splitlines()[0], "if x0 <= 2.5:")

    def test_rounding_noise_does_not_override_the_lowest_feature(self):
        # Both features cut the rows into {0, 1, 2} and {3, 4, 5}, so the splits are equally good.
        # Feature 1 lists the rows in a different order inside each side, so its sums round
        # differently, and without a tolerance it would win by one rounding error.
        X = [[0.0, 2.0], [1.0, 1.0], [2.0, 0.0], [10.0, 10.0], [11.0, 12.0], [12.0, 11.0]]
        y = [0.032, 0.282, 0.459, 0.664, 0.878, 0.962]
        model = fitted(X, y, max_depth=1)
        self.assertEqual(model.describe().splitlines()[0], "if x0 <= 6:")

    def test_a_value_equal_to_the_threshold_goes_left(self):
        model = fitted(column([0, 10]), [0, 1])
        self.assertEqual(model.predict(column([5.0, 5.0001])), [0.0, 1.0])

    def test_constant_targets_and_constant_features_make_a_single_leaf(self):
        flat = fitted(column(range(5)), [3, 3, 3, 3, 3])
        self.assertEqual((flat.depth_, flat.n_leaves_), (0, 1))
        self.assertEqual(flat.predict(column([100])), [3.0])
        same = fitted([[1.0], [1.0], [1.0]], [1, 2, 6])
        self.assertEqual(same.predict([[1.0]]), [3.0])
        self.assertEqual(same.describe(), "predict 3 (n=3)")

    def test_a_dataset_with_a_single_row(self):
        self.assertEqual(fitted([[1.0]], [4]).predict([[9.0]]), [4.0])

    def test_first_split_matches_a_brute_force_search(self):
        for seed in range(20):
            rng = random.Random(seed)
            X = [[rng.uniform(-5, 5), rng.uniform(-5, 5)] for _ in range(14)]
            y = [rng.uniform(-3, 3) + (2 if row[1] > 1 else 0) for row in X]
            best = None
            for f in range(2):
                values = sorted({row[f] for row in X})
                for lo, hi in zip(values, values[1:], strict=False):
                    t = lo + (hi - lo) / 2
                    sse = 0.0
                    for side in ([v for r, v in zip(X, y, strict=True) if r[f] <= t],
                                 [v for r, v in zip(X, y, strict=True) if r[f] > t]):  # fmt: skip
                        mean = sum(side) / len(side)
                        sse += sum((v - mean) ** 2 for v in side)
                    if best is None or sse < best[0] - 1e-9:
                        best = (sse, f, t)
            expected = f"if x{best[1]} <= {best[2]:.6g}:"
            model = fitted(X, y, max_depth=1)
            self.assertEqual(model.describe().splitlines()[0], expected, msg=f"seed {seed}")


class ShapeTests(unittest.TestCase):
    def test_max_depth_limits_the_tree(self):
        X, y = column(range(8)), [0, 1, 2, 3, 4, 5, 6, 7]
        self.assertEqual(fitted(X, y, max_depth=1).depth_, 1)
        self.assertEqual(fitted(X, y, max_depth=2).n_leaves_, 4)
        stump = fitted(X, y, max_depth=1)
        self.assertEqual(stump.predict(column([0, 7])), [1.5, 5.5])

    def test_unlimited_depth_memorises_distinct_rows(self):
        rng = random.Random(4)
        X = [[rng.uniform(0, 1), rng.uniform(0, 1)] for _ in range(40)]
        y = [rng.uniform(-10, 10) for _ in X]
        model = fitted(X, y, max_depth=None)
        for got, want in zip(model.predict(X), y, strict=True):
            self.assertAlmostEqual(got, want)

    def test_min_samples_split_stops_small_nodes(self):
        X, y = column(range(1, 7)), [0, 0, 0, 5, 5, 5]
        self.assertEqual(fitted(X, y, min_samples_split=7).n_leaves_, 1)
        self.assertEqual(fitted(X, y, min_samples_split=6).n_leaves_, 2)

    def test_min_samples_leaf_changes_the_best_split(self):
        X, y = column(range(1, 7)), [0, 0, 0, 0, 0, 100]
        # without the limit the best split cuts off the last row; leaves of two rows or more
        # make the cut after the fourth row the cheapest (squared error 5000 against 6667)
        self.assertEqual(fitted(X, y, max_depth=1).describe().splitlines()[0], "if x0 <= 5.5:")
        limited = fitted(X, y, max_depth=1, min_samples_leaf=2)
        self.assertEqual(limited.describe().splitlines()[0], "if x0 <= 4.5:")
        self.assertEqual(limited.predict(column([6])), [50.0])

    def test_every_leaf_holds_at_least_min_samples_leaf_rows(self):
        rng = random.Random(8)
        X = [[rng.uniform(0, 1)] for _ in range(60)]
        y = [rng.uniform(0, 1) for _ in X]
        model = fitted(X, y, max_depth=None, min_samples_leaf=7)
        sizes = [int(line.rsplit("n=", 1)[1].rstrip(")")) for line in model.describe().splitlines()
                 if "predict" in line]  # fmt: skip
        self.assertEqual(sum(sizes), 60)
        self.assertGreaterEqual(min(sizes), 7)

    def test_row_order_does_not_change_the_tree(self):
        rng = random.Random(2)
        X = [[rng.randint(0, 5), rng.randint(0, 5)] for _ in range(50)]
        X = [[float(a), float(b)] for a, b in X]
        y = [rng.uniform(-1, 1) for _ in X]
        reference = fitted(X, y, max_depth=4)
        for _ in range(5):
            order = list(range(50))
            rng.shuffle(order)
            shuffled = fitted([X[i] for i in order], [y[i] for i in order], max_depth=4)
            self.assertEqual(shuffled.describe(), reference.describe())

    def test_refit_replaces_the_old_tree(self):
        model = fitted(column(range(6)), [0, 0, 0, 9, 9, 9])
        model.fit(column(range(4)), [1, 1, 1, 1])
        self.assertEqual(model.predict(column([100])), [1.0])
        self.assertEqual(model.n_leaves_, 1)

    def test_a_tree_fitted_to_a_line_is_a_staircase_within_the_data_range(self):
        X = column(range(32))
        y = [2.0 * v[0] for v in X]
        model = fitted(X, y, max_depth=3)
        errors = [abs(a - b) for a, b in zip(model.predict(X), y, strict=True)]
        self.assertLessEqual(max(errors), 4.0)  # 8 leaves of 4 rows: at most 3 * 2 away
        self.assertEqual(model.predict(column([-50]))[0], model.predict(column([0]))[0])
        self.assertEqual(model.predict(column([500]))[0], model.predict(column([31]))[0])


class RobustnessTests(unittest.TestCase):
    def test_extreme_and_adjacent_values(self):
        big = fitted(column([-1e300, 0.0, 1e300]), [1, 2, 3], max_depth=None)
        self.assertEqual(big.predict(column([-1e300, 1e300])), [1.0, 3.0])
        a = 1.0
        b = 1.0 + 2.220446049250313e-16
        close = fitted(column([a, b]), [0, 1])
        self.assertEqual(close.predict(column([a, b])), [0.0, 1.0])

    def test_adjacent_negative_floats_keep_the_threshold_on_the_left_value(self):
        a = -1.0000000000000002
        b = -1.0
        # the midpoint rounds to b, which would send the left row to the right
        model = fitted(column([a, b]), [0, 1])
        self.assertEqual(model.predict(column([a, b])), [0.0, 1.0])

    def test_huge_targets_are_a_clear_error(self):
        with self.assertRaises(ModelError):
            fitted(column([0, 1, 2]), [1e200, -1e200, 1e200])

    def test_negative_targets_and_mixed_scale_features(self):
        model = fitted([[-1000.0, 0.001], [-900.0, 0.002], [5.0, 0.003]], [-5, -5, 7])
        self.assertEqual(model.predict([[-950.0, 0.0015]]), [-5.0])


class InterfaceTests(unittest.TestCase):
    def test_defaults_params_and_registry(self):
        model = create_regressor("decision_tree_regressor")
        self.assertEqual(
            model.get_params(), {"max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1}
        )
        self.assertEqual(
            create_regressor("decision_tree_regressor", max_depth=None).get_params()["max_depth"],
            None,
        )
        with self.assertRaises(ModelError):
            create_regressor("decision_tree_regressor", k=3)

    def test_invalid_parameters(self):
        for params in (
            {"max_depth": 0},
            {"max_depth": 1.5},
            {"max_depth": True},
            {"min_samples_split": 1},
            {"min_samples_leaf": 0},
            {"min_samples_leaf": "2"},
        ):
            with self.subTest(params), self.assertRaises(ModelError):
                DecisionTreeRegressor(**params)

    def test_unfitted_use(self):
        with self.assertRaisesRegex(ModelError, "not been fitted"):
            DecisionTreeRegressor().predict([[1.0]])
        self.assertEqual(DecisionTreeRegressor().describe(), "")

    def test_a_refit_that_fails_while_building_leaves_no_usable_model(self):
        model = fitted(column(range(4)), [1, 2, 3, 4])
        with self.assertRaises(ModelError):
            model.fit(column(range(3)), [1e200, -1e200, 1e200])
        with self.assertRaisesRegex(ModelError, "not been fitted"):
            model.predict([[1.0]])

    def test_input_rejected_by_validation_leaves_the_fitted_model_alone(self):
        model = fitted(column(range(4)), [1, 2, 3, 4])
        with self.assertRaises(ModelError):
            model.fit([[1.0], [2.0]], [1.0])
        self.assertEqual(model.predict(column([0])), [1.0])


if __name__ == "__main__":
    unittest.main()


class LeafAccessTests(unittest.TestCase):
    def stump(self):
        tree = DecisionTreeRegressor(max_depth=1)
        tree.fit([[0.0], [1.0], [10.0], [11.0]], [0.0, 0.0, 5.0, 5.0])
        return tree

    def test_apply_numbers_the_leaves_left_first(self):
        tree = self.stump()
        self.assertEqual(tree.n_leaves_, 2)
        self.assertEqual(tree.apply([[0.0], [11.0], [1.0], [5.0], [6.0]]), [0, 1, 0, 0, 1])

    def test_a_single_leaf_tree_has_one_leaf(self):
        tree = DecisionTreeRegressor()
        tree.fit([[1.0], [2.0]], [3.0, 3.0])
        self.assertEqual(tree.apply([[9.0], [-9.0]]), [0, 0])

    def test_leaf_ids_cover_every_leaf_of_a_deep_tree(self):
        rng = random.Random(1)
        X = [[rng.random(), rng.random()] for _ in range(60)]
        tree = DecisionTreeRegressor(max_depth=4)
        tree.fit(X, [a * 3 + b for a, b in X])
        ids = tree.apply(X)
        self.assertEqual(set(ids), set(range(tree.n_leaves_)))

    def test_set_leaf_values_changes_predictions_leaf_by_leaf(self):
        tree = self.stump()
        tree.set_leaf_values([100.0, -7.5])
        self.assertEqual(tree.predict([[0.0], [1.0], [10.0], [50.0]]), [100.0, 100.0, -7.5, -7.5])
        self.assertIn("predict -7.5", tree.describe())

    def test_set_leaf_values_checks_its_input(self):
        tree = self.stump()
        for values in ([1.0], [1.0, 2.0, 3.0], [1.0, float("nan")], [float("inf"), 0.0]):
            with self.subTest(values=values), self.assertRaises(ModelError):
                tree.set_leaf_values(values)
        self.assertEqual(tree.predict([[0.0], [11.0]]), [0.0, 5.0])  # nothing changed
        with self.assertRaises(ModelError):
            DecisionTreeRegressor().set_leaf_values([1.0])

    def test_apply_checks_fit_and_width(self):
        with self.assertRaises(ModelError):
            DecisionTreeRegressor().apply([[1.0]])
        with self.assertRaises(ModelError):
            self.stump().apply([[1.0, 2.0]])
