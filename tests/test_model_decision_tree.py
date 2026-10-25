import random
import unittest

from modelbench.errors import ModelError
from modelbench.models import create_model
from modelbench.models.decision_tree import DecisionTreeClassifier


def fitted(X, y, **params):
    model = DecisionTreeClassifier(**params)
    model.fit(X, y)
    return model


def gini_scores(X, y, rows):
    """Brute force: (weighted impurity * n, feature, threshold) for every midpoint split."""
    out = []
    for feature in range(len(X[0])):
        values = sorted({X[i][feature] for i in rows})
        for low, high in zip(values, values[1:], strict=False):
            threshold = low + (high - low) / 2
            left = [y[i] for i in rows if X[i][feature] <= threshold]
            right = [y[i] for i in rows if X[i][feature] > threshold]
            total = 0.0
            for side in (left, right):
                counts = {label: side.count(label) for label in set(side)}
                total += len(side) * (1 - sum((c / len(side)) ** 2 for c in counts.values()))
            out.append((total, feature, threshold))
    return out


class KnownAnswerTests(unittest.TestCase):
    def test_separable_data_splits_at_the_midpoint_between_neighbours(self):
        model = fitted([[1], [2], [3], [10], [11], [12]], ["a", "a", "a", "b", "b", "b"])
        self.assertEqual((model.depth_, model.n_leaves_), (1, 2))
        self.assertEqual(model.describe().splitlines()[0], "if x0 <= 6.5:")
        self.assertEqual(model.predict([[6.5], [6.6], [-100], [100]]), ["a", "b", "a", "b"])
        self.assertEqual(model.predict_proba([[0]]), [[1.0, 0.0]])

    def test_picks_the_feature_that_separates_the_classes(self):
        X = [[5, 1], [1, 2], [4, 3], [2, 10], [3, 11], [6, 12]]
        y = ["a", "a", "a", "b", "b", "b"]
        model = fitted(X, y)
        self.assertEqual(model.describe().splitlines()[0], "if x1 <= 6.5:")

    def test_equal_scores_go_to_the_lowest_feature_then_the_lowest_threshold(self):
        model = fitted([[1, 1], [2, 2], [3, 3], [4, 4]], ["a", "a", "b", "b"])
        self.assertEqual(model.describe().splitlines()[0], "if x0 <= 2.5:")
        # a split between 1|2 and 3|4 would each leave one mixed side; only 2|3 is pure
        zigzag = fitted([[1], [2], [3], [4]], ["a", "b", "a", "b"], max_depth=1)
        self.assertEqual(zigzag.describe().splitlines()[0], "if x0 <= 1.5:")

    def test_xor_needs_a_split_that_does_not_lower_impurity(self):
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [0, 1, 1, 0]
        model = fitted(X, y, max_depth=None)
        self.assertEqual(model.predict(X), y)
        self.assertEqual((model.depth_, model.n_leaves_), (2, 4))

    def test_leaf_probabilities_are_class_fractions(self):
        model = fitted([[1], [2], [3], [10], [11]], ["a", "b", "a", "b", "b"], max_depth=1)
        # impurity x rows: split 6.5 gives 4/3 (left a, b, a; right pure), 2.5 gives 7/3, 1.5
        # gives 3/2, 10.5 gives 2, so 6.5 wins and its left leaf holds two a and one b
        self.assertEqual(model.describe().splitlines()[0], "if x0 <= 6.5:")
        (left,) = model.predict_proba([[0]])
        self.assertAlmostEqual(left[0], 2 / 3)
        self.assertAlmostEqual(left[1], 1 / 3)
        self.assertEqual(model.predict_proba([[20]]), [[0.0, 1.0]])

    def test_equal_class_counts_in_a_leaf_go_to_the_first_class(self):
        model = fitted([[1], [1], [1], [1]], ["b", "a", "b", "a"])
        self.assertEqual((model.depth_, model.n_leaves_), (0, 1))
        self.assertEqual(model.predict_proba([[1]]), [[0.5, 0.5]])
        self.assertEqual(model.predict([[1]]), ["a"])
        self.assertIn("predict 'a'", model.describe())

    def test_a_pure_node_is_not_split(self):
        model = fitted([[1], [2], [3]], ["a", "a", "a"])
        self.assertEqual((model.depth_, model.n_leaves_), (0, 1))
        self.assertEqual(model.predict([[99]]), ["a"])

    def test_constant_features_cannot_split(self):
        model = fitted([[2.0], [2.0], [2.0]], ["a", "b", "b"])
        self.assertEqual(model.n_leaves_, 1)
        for (pa, pb), expected in zip(model.predict_proba([[2.0]]), [(1 / 3, 2 / 3)], strict=True):
            self.assertAlmostEqual(pa, expected[0])
            self.assertAlmostEqual(pb, expected[1])

    def test_a_value_equal_to_the_threshold_goes_left(self):
        model = fitted([[0.0], [1.0]], ["a", "b"])
        self.assertEqual(model.predict([[0.5]]), ["a"])


class StoppingRuleTests(unittest.TestCase):
    DATA_X = [[i] for i in range(16)]
    DATA_Y = ["a", "b"] * 8

    def test_max_depth_limits_the_tree(self):
        for depth in (1, 2, 3, 4):
            model = fitted(self.DATA_X, self.DATA_Y, max_depth=depth)
            self.assertEqual(model.depth_, depth)  # alternating labels stay impure at every level
            self.assertLessEqual(model.n_leaves_, 2**depth)
        unlimited = fitted(self.DATA_X, self.DATA_Y, max_depth=None)
        self.assertGreater(unlimited.depth_, 4)
        self.assertEqual(unlimited.predict(self.DATA_X), self.DATA_Y)

    def test_block_labels_with_a_tied_root_split_use_the_lowest_threshold(self):
        X = [[i] for i in range(16)]
        y = ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4
        # root: threshold 3.5 gives impurity 0 + 8 and 7.5 gives 4 + 4, an exact tie, so 3.5 wins
        two = fitted(X, y, max_depth=2)
        self.assertEqual(two.describe().splitlines()[0], "if x0 <= 3.5:")
        self.assertEqual((two.depth_, two.n_leaves_), (2, 3))
        self.assertEqual(two.predict([[2], [5], [10]]), ["a", "b", "c"])  # c absorbs c and d
        three = fitted(X, y, max_depth=3)
        self.assertEqual((three.depth_, three.n_leaves_), (3, 4))
        self.assertEqual(three.predict(X), y)
        self.assertEqual(fitted(X, y, max_depth=9).depth_, 3)  # pure leaves stop the growth

    def test_min_samples_split(self):
        model = fitted([[1], [2], [3], [10]], ["a", "b", "a", "b"], min_samples_split=5)
        self.assertEqual(model.n_leaves_, 1)
        model = fitted(
            [[1], [2], [3], [10]], ["a", "b", "a", "b"], min_samples_split=4, max_depth=1
        )
        self.assertEqual(model.n_leaves_, 2)

    def test_min_samples_leaf_forces_larger_leaves(self):
        X = [[1], [2], [3], [4], [5]]
        y = ["a", "a", "a", "a", "b"]
        self.assertEqual(fitted(X, y, max_depth=1).describe().splitlines()[0], "if x0 <= 4.5:")
        constrained = fitted(X, y, max_depth=1, min_samples_leaf=2)
        self.assertEqual(constrained.describe().splitlines()[0], "if x0 <= 3.5:")
        self.assertEqual(fitted(X, y, min_samples_leaf=3).n_leaves_, 1)  # 5 rows cannot make 2x3

    def test_every_leaf_respects_min_samples_leaf(self):
        rng = random.Random(3)
        X = [[rng.random(), rng.random()] for _ in range(60)]
        y = [rng.choice("abc") for _ in range(60)]
        model = fitted(X, y, max_depth=None, min_samples_leaf=4)
        sizes = [leaf.n for leaf in _leaves(model)]
        self.assertEqual(sum(sizes), 60)
        self.assertGreaterEqual(min(sizes), 4)


def _leaves(model):
    stack, out = [model._root], []
    while stack:
        node = stack.pop()
        if node.left is None:
            out.append(node)
        else:
            stack += [node.left, node.right]
    return out


class IndependentCheckTests(unittest.TestCase):
    def test_first_split_matches_a_brute_force_search(self):
        for seed in range(25):
            rng = random.Random(seed)
            n = rng.randint(4, 14)
            X = [[rng.randint(0, 5), rng.randint(0, 5)] for _ in range(n)]
            y = [rng.choice("abc") for _ in range(n)]
            if len(set(y)) < 2 or not gini_scores(X, y, list(range(n))):
                continue
            best_score, best_feature, best_threshold = min(
                gini_scores(X, y, list(range(n))), key=lambda t: (round(t[0], 9), t[1], t[2])
            )
            model = fitted(X, y, max_depth=1)
            expected = f"if x{best_feature} <= {best_threshold:.6g}:"
            self.assertEqual(model.describe().splitlines()[0], expected, msg=f"seed {seed}")

    def test_unlimited_tree_memorises_data_without_conflicting_duplicates(self):
        rng = random.Random(11)
        X = [[rng.random(), rng.random(), rng.random()] for _ in range(80)]
        y = [rng.choice("abcd") for _ in range(80)]
        model = fitted(X, y, max_depth=None)
        self.assertEqual(model.predict(X), y)

    def test_training_row_order_does_not_change_the_tree(self):
        rng = random.Random(5)
        X = [[rng.randint(0, 6), rng.randint(0, 6)] for _ in range(40)]
        y = [rng.choice("abc") for _ in range(40)]
        reference = fitted(X, y, max_depth=None)
        order = list(range(40))
        for seed in range(6):
            random.Random(seed).shuffle(order)
            shuffled = fitted([X[i] for i in order], [y[i] for i in order], max_depth=None)
            self.assertEqual(shuffled.describe(), reference.describe())
            self.assertEqual(shuffled.predict_proba(X), reference.predict_proba(X))

    def test_refit_is_identical(self):
        rng = random.Random(8)
        X = [[rng.random(), rng.random()] for _ in range(30)]
        y = [rng.choice("ab") for _ in range(30)]
        first, second = fitted(X, y), fitted(X, y)
        self.assertEqual(first.predict_proba(X), second.predict_proba(X))


class NumericEdgeTests(unittest.TestCase):
    def test_extreme_values_split_without_overflow(self):
        model = fitted([[-1e308], [1e308]], ["a", "b"])
        self.assertEqual(model.predict([[-1e308], [1e308], [0.0]]), ["a", "b", "b"])

    def test_adjacent_floats_are_separable(self):
        import math

        low = 1.0
        high = math.nextafter(1.0, 2.0)
        model = fitted([[low], [high]], ["a", "b"])
        self.assertEqual(model.predict([[low], [high]]), ["a", "b"])

    def test_negative_and_mixed_scale_features(self):
        model = fitted([[-5.0, 1e6], [-4.0, 2e6], [4.0, 1e6], [5.0, 2e6]], ["a", "a", "b", "b"])
        self.assertEqual(model.predict([[-6.0, 1e6], [6.0, 2e6]]), ["a", "b"])


class ParameterTests(unittest.TestCase):
    def test_defaults_and_get_params(self):
        self.assertEqual(
            DecisionTreeClassifier().get_params(),
            {"max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1},
        )
        self.assertIsNone(DecisionTreeClassifier(max_depth=None).max_depth)

    def test_invalid_parameters(self):
        bad = [
            {"max_depth": 0}, {"max_depth": True}, {"max_depth": 2.5}, {"max_depth": "3"},
            {"min_samples_split": 1}, {"min_samples_split": False},
            {"min_samples_leaf": 0}, {"min_samples_leaf": -1}, {"min_samples_leaf": 1.0},
        ]  # fmt: skip
        for params in bad:
            with self.subTest(params=params), self.assertRaises(ModelError):
                DecisionTreeClassifier(**params)

    def test_registry_name_and_unknown_parameter(self):
        self.assertIsInstance(create_model("decision_tree", max_depth=2), DecisionTreeClassifier)
        with self.assertRaises(ModelError):
            create_model("decision_tree", depth=2)

    def test_unfitted_use_is_an_error(self):
        model = DecisionTreeClassifier()
        with self.assertRaises(ModelError):
            model.describe()
        with self.assertRaises(ModelError):
            model.predict([[1.0]])

    def test_refit_replaces_the_old_tree(self):
        model = fitted([[1], [2]], ["a", "b"])
        model.fit([[1], [2], [3]], ["x", "x", "y"])
        self.assertEqual(model.classes_, ("x", "y"))
        self.assertEqual(model.predict([[1]]), ["x"])

    def test_integer_and_string_labels(self):
        self.assertEqual(fitted([[1], [9]], [0, 1]).predict([[2], [8]]), [0, 1])
        self.assertEqual(fitted([[1], [9]], ["no", "yes"]).predict([[2], [8]]), ["no", "yes"])


if __name__ == "__main__":
    unittest.main()
