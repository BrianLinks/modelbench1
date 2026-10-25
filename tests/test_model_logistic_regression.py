import math
import random
import unittest

from modelbench.errors import ModelError
from modelbench.models import create_model
from modelbench.models.logistic_regression import MAX_ITER_LIMIT, LogisticRegression


def fitted(X, y, **params):
    params.setdefault("max_iter", 5000)
    params.setdefault("tol", 1e-9)
    model = LogisticRegression(**params)
    model.fit(X, y)
    return model


def blobs(seed=0, n=30, classes=("a", "b", "c"), spread=1.2):
    """Overlapping gaussian clusters on a line in 2-D, so the classes are not separable."""
    rng = random.Random(seed)
    X, y = [], []
    for k, label in enumerate(classes):
        for _ in range(n):
            X.append([k * 2.0 + rng.gauss(0, spread), rng.gauss(0, spread) + k])
            y.append(label)
    return X, y


def _sd(X, j):
    mean = math.fsum(r[j] for r in X) / len(X)
    variance = math.fsum((r[j] - mean) ** 2 for r in X) / len(X)
    return math.sqrt(variance) if variance > 0 else 1.0


def gradient_max(model, X, y, l2):
    """Largest absolute entry of the objective's gradient on the standardised scale, computed
    from the public coefficients (mapped back to the standardised parametrisation)."""
    n, d = len(X), len(X[0])
    means = [math.fsum(r[j] for r in X) / n for j in range(d)]
    sds = [_sd(X, j) for j in range(d)]
    index = {c: i for i, c in enumerate(model.classes_)}
    k = len(index)
    w = [[model.coef_[c][j] * sds[j] for j in range(d)] for c in range(k)]
    b = [
        model.intercept_[c] + sum(model.coef_[c][j] * means[j] for j in range(d)) for c in range(k)
    ]
    g_w = [[l2 * w[c][j] for j in range(d)] for c in range(k)]
    g_b = [0.0] * k
    for row, label in zip(X, y, strict=True):
        z = [(row[j] - means[j]) / sds[j] for j in range(d)]
        scores = [b[c] + sum(w[c][j] * z[j] for j in range(d)) for c in range(k)]
        top = max(scores)
        exps = [math.exp(v - top) for v in scores]
        for c in range(k):
            err = (exps[c] / sum(exps) - (1.0 if index[label] == c else 0.0)) / n
            g_b[c] += err
            for j in range(d):
                g_w[c][j] += err * z[j]
    return max(max(abs(v) for r in g_w for v in r), max(abs(v) for v in g_b))


def objective(model, X, y, l2):
    """Mean cross-entropy plus (l2 / 2) * squared weights on the standardised scale, written
    from the definition using only the public coef_ and intercept_."""
    n = len(X)
    sd = [_sd(X, j) for j in range(len(X[0]))]
    index = {c: i for i, c in enumerate(model.classes_)}
    loss = 0.0
    for row, label in zip(X, y, strict=True):
        scores = [
            b + sum(w * v for w, v in zip(ws, row, strict=True))
            for ws, b in zip(model.coef_, model.intercept_, strict=True)
        ]
        top = max(scores)
        log_norm = top + math.log(sum(math.exp(s - top) for s in scores))
        loss += log_norm - scores[index[label]]
    penalty = sum((w * s) ** 2 for ws in model.coef_ for w, s in zip(ws, sd, strict=True))
    return loss / n + 0.5 * l2 * penalty


def shifted(model, class_index, feature, delta):
    """A copy of the fitted parameters with one coefficient (feature >= 0) or intercept
    (feature == -1) moved by `delta`, wrapped so `objective` can evaluate it."""

    class Moved:
        classes_ = model.classes_
        coef_ = [list(r) for r in model.coef_]
        intercept_ = list(model.intercept_)

    if feature < 0:
        Moved.intercept_[class_index] += delta
    else:
        Moved.coef_[class_index][feature] += delta
    return Moved


class OptimalityTests(unittest.TestCase):
    def assert_stationary(self, X, y, l2, bound=1e-5):
        model = fitted(X, y, l2=l2)
        self.assertTrue(model.converged_, msg=f"n_iter_={model.n_iter_}")
        spreads = [_sd(X, j) for j in range(len(X[0]))]
        worst = 0.0
        for c in range(len(model.classes_)):
            for feature in range(-1, len(X[0])):
                # step in units of the feature's spread, so every step moves the scores equally
                h = 1e-5 / (spreads[feature] if feature >= 0 else 1.0)
                up = objective(shifted(model, c, feature, h), X, y, l2)
                down = objective(shifted(model, c, feature, -h), X, y, l2)
                worst = max(worst, abs((up - down) / (2 * h)) * (h / 1e-5))
        self.assertLess(worst, bound)
        return model

    def test_gradient_of_an_independent_objective_vanishes_binary(self):
        X, y = blobs(seed=1, classes=("a", "b"))
        self.assert_stationary(X, y, 0.01)

    def test_gradient_vanishes_multiclass_and_for_other_penalties(self):
        X, y = blobs(seed=2)
        for l2 in (0.001, 0.01, 0.5):
            with self.subTest(l2=l2):
                self.assert_stationary(X, y, l2)

    def test_gradient_vanishes_with_unequal_feature_scales_and_a_constant_column(self):
        X, y = blobs(seed=3, classes=("a", "b"))
        X = [[row[0] * 1000.0, row[1] * 0.001, 7.0] for row in X]
        model = self.assert_stationary(X, y, 0.05)
        self.assertEqual([r[2] for r in model.coef_], [0.0, 0.0])  # constant: no weight

    def test_converged_means_the_gradient_is_below_tol_at_the_returned_parameters(self):
        for seed in range(4):
            X, y = blobs(seed=seed)
            for tol in (1e-2, 1e-3, 1e-5):
                with self.subTest(seed=seed, tol=tol):
                    model = fitted(X, y, tol=tol)
                    self.assertTrue(model.converged_)
                    self.assertLessEqual(gradient_max(model, X, y, 0.01), tol * (1 + 1e-9))

    def test_the_fitted_point_beats_random_nearby_points(self):
        X, y = blobs(seed=4)
        model = fitted(X, y)
        best = objective(model, X, y, 0.01)
        rng = random.Random(0)
        for _ in range(40):
            nearby = shifted(model, 0, 0, 0.0)
            nearby.coef_ = [[w + rng.uniform(-0.05, 0.05) for w in r] for r in model.coef_]
            nearby.intercept_ = [b + rng.uniform(-0.05, 0.05) for b in model.intercept_]
            self.assertLessEqual(best, objective(nearby, X, y, 0.01) + 1e-12)

    def test_regularisation_shrinks_the_weights(self):
        X, y = blobs(seed=5, classes=("a", "b"))
        sizes = [sum(abs(w) for w in fitted(X, y, l2=l2).coef_[1]) for l2 in (0.001, 0.1, 10.0)]
        self.assertGreater(sizes[0], sizes[1])
        self.assertGreater(sizes[1], sizes[2])


class KnownAnswerTests(unittest.TestCase):
    def test_symmetric_data_gives_antisymmetric_weights_and_an_even_midpoint(self):
        model = fitted([[-1.0], [1.0]], ["a", "b"])
        self.assertAlmostEqual(model.coef_[0][0], -model.coef_[1][0], places=7)
        self.assertAlmostEqual(model.intercept_[0], model.intercept_[1], places=7)
        low, high = model.predict_proba([[-1.0]])[0], model.predict_proba([[1.0]])[0]
        self.assertAlmostEqual(low[0], high[1], places=9)
        self.assertGreater(low[0], 0.5)
        mid = model.predict_proba([[0.0]])[0]
        self.assertAlmostEqual(mid[0], 0.5, places=7)

    def test_without_usable_features_probabilities_equal_the_class_frequencies(self):
        y = ["a"] * 6 + ["b"] * 3 + ["c"] * 1
        model = fitted([[5.0]] * 10, y)
        (row,) = model.predict_proba([[5.0]])
        for got, want in zip(row, (0.6, 0.3, 0.1), strict=True):
            self.assertAlmostEqual(got, want, places=6)
        self.assertEqual(model.predict([[5.0]]), ["a"])

    def test_one_class_always_has_probability_one(self):
        model = fitted([[1.0], [2.0]], ["only", "only"])
        self.assertEqual(model.predict_proba([[3.0]]), [[1.0]])
        self.assertTrue(model.converged_)
        self.assertEqual(model.n_iter_, 1)

    def test_separable_data_is_classified_and_probabilities_grow_with_distance(self):
        X = [[float(i)] for i in range(-5, 0)] + [[float(i)] for i in range(1, 6)]
        y = ["neg"] * 5 + ["pos"] * 5
        model = fitted(X, y)
        self.assertEqual(model.predict(X), y)
        chances = [model.predict_proba([[x]])[0][1] for x in (-4.0, -1.0, 1.0, 4.0)]
        self.assertEqual(chances, sorted(chances))
        self.assertTrue(all(0.0 < c < 1.0 for c in chances))

    def test_three_clusters(self):
        X, y = blobs(seed=6, spread=0.3)
        # with the default l2 = 0.01 the penalised optimum misclassifies one point; weaker
        # regularisation separates all 90 (the optimum itself is verified in OptimalityTests)
        self.assertLess(sum(a != b for a, b in zip(fitted(X, y).predict(X), y, strict=True)), 3)
        model = fitted(X, y, l2=1e-4)
        self.assertEqual(model.predict(X), y)
        for row in model.predict_proba(X):
            self.assertAlmostEqual(sum(row), 1.0, places=12)

    def test_swapping_label_names_mirrors_the_probabilities(self):
        X, y = blobs(seed=7, classes=("a", "b"))
        flipped = ["b" if label == "a" else "a" for label in y]
        original, mirror = fitted(X, y), fitted(X, flipped)
        for p, q in zip(original.predict_proba(X), mirror.predict_proba(X), strict=True):
            self.assertAlmostEqual(p[0], q[1], places=9)
            self.assertAlmostEqual(p[1], q[0], places=9)


class InvarianceTests(unittest.TestCase):
    def test_outputs_do_not_depend_on_feature_units(self):
        X, y = blobs(seed=8)
        rescaled = [[row[0] * 1000.0 + 5.0, row[1] * 0.01 - 3.0] for row in X]
        a, b = fitted(X, y), fitted(rescaled, y)
        for p, q in zip(a.predict_proba(X), b.predict_proba(rescaled), strict=True):
            for u, v in zip(p, q, strict=True):
                self.assertAlmostEqual(u, v, places=7)

    def test_back_transformed_coefficients_reproduce_the_probabilities(self):
        X, y = blobs(seed=9)
        model = fitted(X, y)
        for row, probs in zip(X[:10], model.predict_proba(X[:10]), strict=True):
            scores = [
                b + sum(w * v for w, v in zip(ws, row, strict=True))
                for ws, b in zip(model.coef_, model.intercept_, strict=True)
            ]
            top = max(scores)
            exps = [math.exp(s - top) for s in scores]
            for e, p in zip(exps, probs, strict=True):
                self.assertAlmostEqual(e / sum(exps), p, places=9)

    def test_refitting_the_same_data_is_bit_identical(self):
        X, y = blobs(seed=10)
        self.assertEqual(fitted(X, y).predict_proba(X), fitted(X, y).predict_proba(X))

    def test_row_order_changes_probabilities_only_at_rounding_level(self):
        X, y = blobs(seed=11)
        order = list(range(len(X)))
        random.Random(1).shuffle(order)
        a = fitted(X, y)
        b = fitted([X[i] for i in order], [y[i] for i in order])
        for p, q in zip(a.predict_proba(X), b.predict_proba(X), strict=True):
            for u, v in zip(p, q, strict=True):
                self.assertAlmostEqual(u, v, places=8)


class RobustnessTests(unittest.TestCase):
    def check_sane(self, X, y, **params):
        model = LogisticRegression(**params)
        model.fit(X, y)
        for row in model.predict_proba(X):
            self.assertAlmostEqual(math.fsum(row), 1.0, places=9)
            self.assertTrue(all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in row))
        self.assertTrue(all(math.isfinite(v) for r in model.coef_ for v in r))
        self.assertTrue(all(math.isfinite(v) for v in model.intercept_))
        return model

    def test_hard_inputs_neither_diverge_nor_produce_nan(self):
        rng = random.Random(12)
        cases = {
            "huge scale": ([[rng.gauss(0, 1) * 1e6, rng.gauss(0, 1) * 1e-6] for _ in range(40)],
                           [rng.choice("ab") for _ in range(40)]),
            "duplicate columns": ([[v, v, v] for v in (rng.gauss(0, 1) for _ in range(40))],
                                  [rng.choice("ab") for _ in range(40)]),
            "more features than rows": ([[rng.gauss(0, 1) for _ in range(30)] for _ in range(12)],
                                        [rng.choice("abc") for _ in range(12)]),
            "one rare class": ([[float(i)] for i in range(100)], ["a"] * 99 + ["b"]),
            "five classes": ([[rng.gauss(k, 1)] for k in range(5) for _ in range(8)],
                             [c for c in "abcde" for _ in range(8)]),
            "perfectly separable": ([[float(i)] for i in range(40)], ["a"] * 20 + ["b"] * 20),
        }  # fmt: skip
        for name, (X, y) in cases.items():
            with self.subTest(case=name):
                self.check_sane(X, y, max_iter=300)

    def test_training_never_leaves_the_objective_worse_than_the_start(self):
        for seed in range(8):
            X, y = blobs(seed=seed, n=12)
            model = fitted(X, y, max_iter=200, tol=1e-12)
            start = math.log(len(model.classes_))  # cross-entropy of the all-zero model
            self.assertLessEqual(objective(model, X, y, 0.01), start + 1e-9)

    def test_iteration_limit_and_convergence_flags(self):
        X, y = blobs(seed=13)
        short = LogisticRegression(max_iter=3, tol=1e-12)
        short.fit(X, y)
        self.assertFalse(short.converged_)
        self.assertEqual(short.n_iter_, 3)
        loose = LogisticRegression(tol=10.0)  # the gradient at zero is already below 10
        loose.fit(X, y)
        self.assertTrue(loose.converged_)
        self.assertEqual(loose.n_iter_, 1)
        for row in loose.predict_proba(X[:3]):
            for p in row:
                self.assertAlmostEqual(p, 1 / 3, places=12)

    def test_acceleration_keeps_the_iteration_count_low(self):
        for seed in range(3):
            X, y = blobs(seed=seed)
            model = LogisticRegression(max_iter=100000)  # default l2 and tol
            model.fit(X, y)
            self.assertTrue(model.converged_)
            self.assertLess(model.n_iter_, 250)  # plain gradient descent needs about 500

    def test_a_constant_training_column_is_ignored_whatever_value_arrives_later(self):
        X, y = blobs(seed=3, classes=("a", "b"))
        model = fitted([[r[0], r[1], 0.1] for r in X], y)
        usual = model.predict_proba([[r[0], r[1], 0.1] for r in X[:5]])
        for odd in (-3.0, 1e300):
            self.assertEqual(model.predict_proba([[r[0], r[1], odd] for r in X[:5]]), usual)

    def test_probabilities_stay_finite_for_inputs_far_outside_the_training_range(self):
        X = [[float(i)] for i in range(-5, 0)] + [[float(i)] for i in range(1, 6)]
        model = fitted(X, ["n"] * 5 + ["p"] * 5, l2=1e-8, max_iter=3000)
        self.assertEqual(model.predict_proba([[1e9], [-1e9]]), [[0.0, 1.0], [1.0, 0.0]])

    def test_extreme_values_raise_model_error_not_overflow(self):
        with self.assertRaisesRegex(ModelError, "overflow"):
            fitted([[1e300], [-1e300], [1e300], [-1e300]], ["a", "b", "a", "b"])


class ParameterAndStateTests(unittest.TestCase):
    def test_defaults_and_get_params(self):
        self.assertEqual(
            LogisticRegression().get_params(), {"l2": 0.01, "max_iter": 500, "tol": 1e-6}
        )

    def test_invalid_parameters(self):
        bad = [
            {"l2": 0}, {"l2": 1e-9}, {"l2": -1}, {"l2": float("nan")}, {"l2": float("inf")},
            {"l2": True}, {"l2": "0.1"}, {"l2": None},
            {"max_iter": 0}, {"max_iter": MAX_ITER_LIMIT + 1}, {"max_iter": True},
            {"max_iter": 10.0}, {"max_iter": "10"},
            {"tol": 0}, {"tol": -1e-3}, {"tol": float("nan")}, {"tol": float("inf")},
            {"tol": True},
        ]  # fmt: skip
        for params in bad:
            with self.subTest(params=params), self.assertRaises(ModelError):
                LogisticRegression(**params)
        LogisticRegression(l2=1e-8, max_iter=MAX_ITER_LIMIT, tol=1e-300)

    def test_registry(self):
        self.assertIsInstance(create_model("logistic_regression", l2=0.1), LogisticRegression)
        with self.assertRaises(ModelError):
            create_model("logistic_regression", learning_rate=0.1)

    def test_attributes_need_a_fitted_model(self):
        model = LogisticRegression()
        for attribute in ("coef_", "intercept_"):
            with self.subTest(attribute=attribute), self.assertRaises(ModelError):
                getattr(model, attribute)
        with self.assertRaises(ModelError):
            model.predict([[1.0]])

    def test_coefficient_shapes_follow_classes_and_features(self):
        X, y = blobs(seed=14)
        model = fitted(X, y)
        self.assertEqual(model.classes_, ("a", "b", "c"))
        self.assertEqual((len(model.coef_), len(model.coef_[0])), (3, 2))
        self.assertEqual(len(model.intercept_), 3)

    def test_integer_labels_and_refit(self):
        model = fitted([[1.0], [9.0], [1.5], [8.5]], [0, 1, 0, 1])
        self.assertEqual(model.predict([[1.0], [9.0]]), [0, 1])
        model.fit([[1.0], [9.0]], ["x", "y"])
        self.assertEqual(model.classes_, ("x", "y"))


if __name__ == "__main__":
    unittest.main()
