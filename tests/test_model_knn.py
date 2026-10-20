import random
import unittest

from modelbench.errors import ModelError
from modelbench.models import create_model
from modelbench.models.knn import KNNClassifier

# Five points on a line: three "a" on the left, two "b" on the right.
LINE_X = [[0.0], [1.0], [2.0], [10.0], [11.0]]
LINE_Y = ["a", "a", "a", "b", "b"]


def fitted(X=LINE_X, y=LINE_Y, **params):
    model = KNNClassifier(**params)
    model.fit(X, y)
    return model


class VoteTests(unittest.TestCase):
    def test_unanimous_neighbourhood(self):
        # query 1.5: nearest are 1 (0.5), 2 (0.5), 0 (1.5), all "a"
        self.assertEqual(fitted(k=3).predict_proba([[1.5]]), [[1.0, 0.0]])
        self.assertEqual(fitted(k=3).predict([[1.5]]), ["a"])

    def test_split_neighbourhood_gives_vote_shares(self):
        # query 9: nearest are 10 (1, b), 11 (2, b), 2 (7, a)
        model = fitted(k=3)
        (row,) = model.predict_proba([[9.0]])
        self.assertAlmostEqual(row[0], 1 / 3)
        self.assertAlmostEqual(row[1], 2 / 3)
        self.assertEqual(model.predict([[9.0]]), ["b"])

    def test_distance_weighting_known_answer(self):
        # weights 1/1 (b), 1/2 (b), 1/7 (a): b = 1.5, a = 1/7, so P(b) = 1.5 / (1.5 + 1/7) = 21/23
        (row,) = fitted(k=3, weights="distance").predict_proba([[9.0]])
        self.assertAlmostEqual(row[1], 21 / 23)
        self.assertAlmostEqual(row[0], 2 / 23)

    def test_exact_match_dominates_under_distance_weighting(self):
        # query 10 matches the "b" training point exactly; uniform voting over all five rows
        # says "a" (3 of 5), distance weighting trusts the exact match
        self.assertEqual(fitted(k=5).predict([[10.0]]), ["a"])
        model = fitted(k=5, weights="distance")
        self.assertEqual(model.predict([[10.0]]), ["b"])
        self.assertEqual(model.predict_proba([[10.0]]), [[0.0, 1.0]])

    def test_several_exact_matches_share_the_vote(self):
        model = fitted([[0.0], [0.0], [5.0]], ["a", "b", "b"], k=3, weights="distance")
        self.assertEqual(model.predict_proba([[0.0]]), [[0.5, 0.5]])

    def test_k_equal_to_training_size_gives_the_class_priors(self):
        self.assertEqual(fitted(k=5).predict_proba([[100.0]]), [[0.6, 0.4]])


class TieBreakingTests(unittest.TestCase):
    def test_equal_votes_go_to_the_first_class(self):
        # query 6: nearest are 2 (4, a) and 10 (4, b): one vote each
        model = fitted(k=2)
        self.assertEqual(model.predict_proba([[6.0]]), [[0.5, 0.5]])
        self.assertEqual(model.predict([[6.0]]), ["a"])

    def test_equal_votes_go_to_the_first_class_even_when_it_is_not_alphabetical_input_order(self):
        model = fitted([[0.0], [2.0]], ["z", "a"], k=2)
        self.assertEqual(model.classes_, ("a", "z"))
        self.assertEqual(model.predict([[1.0]]), ["a"])

    def test_equal_distances_prefer_the_earlier_training_row(self):
        # query 5.5 with k=2: row 2 (3.5) is closest, then rows 1 and 3 tie at 4.5.
        # The earlier row (1, "a") must win the second slot, giving votes a,a.
        self.assertEqual(fitted(k=2).predict_proba([[5.5]]), [[1.0, 0.0]])
        # reversing the training order flips which row is earlier, so the result flips too
        flipped = fitted(LINE_X[::-1], LINE_Y[::-1], k=2)
        self.assertEqual(flipped.predict_proba([[5.5]]), [[0.5, 0.5]])


class MetricTests(unittest.TestCase):
    # a = (3, 0) is 3 away from the origin either way; b = (2, 2) is 2.83 away by Euclid but
    # 4 away by Manhattan
    X = [[3.0, 0.0], [2.0, 2.0]]
    Y = ["a", "b"]

    def test_euclidean_and_manhattan_can_disagree(self):
        origin = [[0.0, 0.0]]
        euclid = fitted(self.X, self.Y, k=1, metric="euclidean")
        manhattan = fitted(self.X, self.Y, k=1, metric="manhattan")
        self.assertEqual(euclid.predict(origin), ["b"])
        self.assertEqual(manhattan.predict(origin), ["a"])

    def test_metrics_give_different_distance_weights(self):
        # "a" at (3, 4) and "b" at (0, 7), query at the origin.
        # Euclid distances are 5 and 7, so P(a) = (1/5) / (1/5 + 1/7) = 7/12.
        # Manhattan distances are 7 and 7, so P(a) = 1/2.
        X, y = [[3.0, 4.0], [0.0, 7.0]], ["a", "b"]
        euclid = fitted(X, y, k=2, weights="distance", metric="euclidean")
        manhattan = fitted(X, y, k=2, weights="distance", metric="manhattan")
        self.assertAlmostEqual(euclid.predict_proba([[0.0, 0.0]])[0][0], 7 / 12)
        self.assertAlmostEqual(manhattan.predict_proba([[0.0, 0.0]])[0][0], 0.5)


class ParameterTests(unittest.TestCase):
    def test_validation(self):
        for k in (0, -1, 1.5, "3", None, True):
            with self.assertRaises(ModelError, msg=repr(k)):
                KNNClassifier(k=k)
        with self.assertRaises(ModelError):
            KNNClassifier(metric="cosine")
        with self.assertRaises(ModelError):
            KNNClassifier(weights="triangular")

    def test_k_larger_than_the_training_set_is_rejected_at_fit(self):
        with self.assertRaises(ModelError):
            fitted(k=6)
        with self.assertRaises(ModelError):
            create_model("knn").fit([[0.0], [1.0]], ["a", "b"])  # default k=5

    def test_params_round_trip_through_the_registry(self):
        model = create_model("knn", k=7, metric="manhattan", weights="distance")
        self.assertEqual(model.get_params(), {"k": 7, "metric": "manhattan", "weights": "distance"})

    def test_training_data_is_copied(self):
        X = [[0.0], [1.0], [10.0]]
        model = fitted(X, ["a", "a", "b"], k=1)
        X[0][0] = 1000.0
        X[2][0] = -1000.0
        self.assertEqual(model.predict([[9.0]]), ["b"])

    def test_multiclass_with_mixed_label_types_sorted_when_possible(self):
        model = fitted([[0.0], [5.0], [10.0]], [3, 1, 2], k=1)
        self.assertEqual(model.classes_, (1, 2, 3))
        self.assertEqual(model.predict([[4.0], [9.0], [-1.0]]), [1, 2, 3])


class PropertyTests(unittest.TestCase):
    """Seeded random checks of invariants that must hold for any data."""

    def setUp(self):
        self.rng = random.Random(20261020)

    def random_data(self, n, features, classes):
        X = [[round(self.rng.uniform(-5, 5), 2) for _ in range(features)] for _ in range(n)]
        y = [self.rng.choice(classes) for _ in range(n)]
        return X, y

    def test_one_nearest_neighbour_recalls_distinct_training_points(self):
        for _ in range(20):
            n = self.rng.randint(2, 30)
            X = [[float(i), round(self.rng.uniform(-5, 5), 2)] for i in range(n)]  # distinct rows
            y = [self.rng.choice("abc") for _ in range(n)]
            for metric in ("euclidean", "manhattan"):
                model = fitted(X, y, k=1, metric=metric)
                self.assertEqual(model.predict(X), y)

    def test_probabilities_are_valid_for_any_input(self):
        for _ in range(30):
            n = self.rng.randint(3, 25)
            X, y = self.random_data(n, self.rng.randint(1, 4), "abc")
            queries, _ = self.random_data(10, len(X[0]), "a")
            k = self.rng.randint(1, n)
            for weights in ("uniform", "distance"):
                model = fitted(X, y, k=k, weights=weights)
                for row in model.predict_proba(queries):
                    self.assertAlmostEqual(sum(row), 1.0)
                    self.assertTrue(all(0.0 <= p <= 1.0 for p in row))
                    self.assertEqual(len(row), len(model.classes_))

    def test_power_of_two_scaling_never_changes_predictions(self):
        for _ in range(25):
            X, y = self.random_data(self.rng.randint(5, 25), 3, "ab")
            queries, _ = self.random_data(8, 3, "a")
            for metric in ("euclidean", "manhattan"):
                base = fitted(X, y, k=3, metric=metric).predict(queries)
                for factor in (0.5, 2.0, 8.0):
                    scaled_X = [[v * factor for v in row] for row in X]
                    scaled_q = [[v * factor for v in row] for row in queries]
                    self.assertEqual(
                        fitted(scaled_X, y, k=3, metric=metric).predict(scaled_q), base
                    )

    def test_repeated_fit_and_predict_is_deterministic(self):
        X, y = self.random_data(40, 2, "abc")
        queries, _ = self.random_data(15, 2, "a")
        first = fitted(X, y, k=5).predict_proba(queries)
        self.assertEqual(fitted(X, y, k=5).predict_proba(queries), first)


if __name__ == "__main__":
    unittest.main()
