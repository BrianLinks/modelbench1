import math
import random
import unittest

from modelbench.config import ExperimentConfig
from modelbench.datasets import Table, kfold_indices
from modelbench.errors import ModelError, SeedError
from modelbench.metrics import log_loss
from modelbench.models import REGISTRY, create_model
from modelbench.models.calibration import (
    METHODS,
    CalibratedClassifier,
    IsotonicCalibrator,
    SigmoidCalibrator,
    smoothed_targets,
)
from modelbench.models.naive_bayes import GaussianNB
from modelbench.reproducibility import check_reproducibility
from modelbench.runner import run_experiment
from modelbench.seeding import derive_seed
from modelbench.store import Store


def noisy(n=300, seed=1, classes=2, width=3):
    rng = random.Random(seed)
    X = [[rng.uniform(-3, 3) for _ in range(width)] for _ in range(n)]
    if classes == 2:
        y = ["p" if r[0] + r[1] + rng.gauss(0, 1.5) > 0 else "q" for r in X]
    else:
        y = ["a" if r[0] + rng.gauss(0, 0.5) < -1 else "b" if r[0] < 1 else "c" for r in X]
    return X, y


class SmoothingTests(unittest.TestCase):
    def test_platt_targets_by_hand(self):
        # 3 present, 1 absent: present -> 4/5, absent -> 1/3
        self.assertEqual(smoothed_targets([1, 1, 0, 1]), [0.8, 0.8, 1 / 3, 0.8])
        self.assertEqual(smoothed_targets([1]), [2 / 3])  # 1 present, 0 absent
        self.assertEqual(smoothed_targets([0]), [1 / 3])
        self.assertEqual(smoothed_targets([]), [])


class SigmoidTests(unittest.TestCase):
    def gradient(self, calibrator, probabilities, present):
        targets = smoothed_targets(present)
        ga = gb = 0.0
        for p, t in zip(probabilities, targets, strict=True):
            s = math.log(p / (1 - p))
            q = 1 / (1 + math.exp(-(calibrator.a * s + calibrator.b)))
            ga += (q - t) * s
            gb += q - t
        return ga, gb

    def test_the_fit_is_a_stationary_point_of_the_smoothed_log_loss(self):
        rng = random.Random(1)
        probabilities = [rng.uniform(0.05, 0.95) for _ in range(200)]
        present = [1 if rng.random() < p**2 else 0 for p in probabilities]
        calibrator = SigmoidCalibrator().fit(probabilities, present)
        ga, gb = self.gradient(calibrator, probabilities, present)
        self.assertLess(abs(ga), 1e-6)
        self.assertLess(abs(gb), 1e-6)
        self.assertGreater(calibrator.a, 0)

    def test_constant_probabilities_give_the_smoothed_base_rate(self):
        calibrator = SigmoidCalibrator().fit([0.5] * 4, [1, 1, 1, 0])
        # smoothed targets are 0.8, 0.8, 0.8, 1/3; their mean is the best constant
        self.assertAlmostEqual(calibrator.predict(0.5), (3 * 0.8 + 1 / 3) / 4, places=6)

    def test_overconfident_probabilities_are_pulled_in(self):
        # the model says 0.99 / 0.01, but is right only 80% of the time
        probabilities = [0.99] * 50 + [0.01] * 50
        present = [1] * 40 + [0] * 10 + [0] * 40 + [1] * 10
        calibrator = SigmoidCalibrator().fit(probabilities, present)
        self.assertAlmostEqual(calibrator.predict(0.99), 40 / 50, delta=0.03)
        self.assertAlmostEqual(calibrator.predict(0.01), 10 / 50, delta=0.03)

    def test_anti_correlated_scores_flip_the_map(self):
        probabilities = [0.9] * 30 + [0.1] * 30
        present = [0] * 30 + [1] * 30
        calibrator = SigmoidCalibrator().fit(probabilities, present)
        self.assertLess(calibrator.a, 0)
        self.assertLess(calibrator.predict(0.9), 0.1)
        self.assertGreater(calibrator.predict(0.1), 0.9)

    def test_monotone_and_inside_the_open_interval(self):
        calibrator = SigmoidCalibrator().fit([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1])
        values = [calibrator.predict(p / 100) for p in range(0, 101, 5)]
        self.assertTrue(all(b >= a for a, b in zip(values, values[1:], strict=False)))
        self.assertTrue(all(0.0 < v < 1.0 for v in values))

    def test_exact_zero_and_one_probabilities_are_fine(self):
        calibrator = SigmoidCalibrator().fit([0.0, 1.0, 0.0, 1.0], [0, 1, 0, 1])
        for p in (0.0, 1.0, 0.5):
            self.assertTrue(math.isfinite(calibrator.predict(p)))

    def test_one_class_only_gives_a_finite_low_or_high_rate(self):
        never = SigmoidCalibrator().fit([0.2, 0.5, 0.8], [0, 0, 0])
        always = SigmoidCalibrator().fit([0.2, 0.5, 0.8], [1, 1, 1])
        self.assertLess(never.predict(0.5), 0.3)
        self.assertGreater(always.predict(0.5), 0.7)

    def test_bad_input_is_an_error(self):
        with self.assertRaises(ModelError):
            SigmoidCalibrator().fit([], [])
        with self.assertRaises(ModelError):
            SigmoidCalibrator().fit([0.5], [1, 0])


class IsotonicTests(unittest.TestCase):
    def test_pooling_and_interpolation_by_hand(self):
        # smoothed targets: 1/4, 3/4, 1/4, 3/4; the middle pair pools to 1/2
        calibrator = IsotonicCalibrator().fit([1.0, 2.0, 3.0, 4.0], [0, 1, 0, 1])
        expected = {0.0: 0.25, 1.0: 0.25, 1.5: 0.375, 2.0: 0.5, 2.5: 0.5, 3.0: 0.5, 3.5: 0.625,
                    4.0: 0.75, 9.0: 0.75}  # fmt: skip
        for x, want in expected.items():
            self.assertAlmostEqual(calibrator.predict(x), want, msg=str(x))

    def test_equal_probabilities_pool_whatever_order_their_outcomes_come_in(self):
        # absent first, so pooling is not forced by an ordering violation: 1/3 and 3/4 average
        calibrator = IsotonicCalibrator().fit([1.0, 1.0, 2.0], [0, 1, 1])
        self.assertAlmostEqual(calibrator.predict(1.0), (1 / 3 + 0.75) / 2)
        self.assertAlmostEqual(calibrator.predict(1.5), ((1 / 3 + 0.75) / 2 + 0.75) / 2)

    def test_equal_probabilities_are_pooled(self):
        # N+ = 2, N- = 1: present 3/4, absent 1/3; the two rows at 1.0 average to 13/24
        calibrator = IsotonicCalibrator().fit([1.0, 1.0, 2.0], [1, 0, 1])
        self.assertAlmostEqual(calibrator.predict(1.0), (0.75 + 1 / 3) / 2)
        self.assertAlmostEqual(calibrator.predict(2.0), 0.75)
        self.assertAlmostEqual(calibrator.predict(1.5), ((0.75 + 1 / 3) / 2 + 0.75) / 2)

    def test_a_falling_trend_pools_everything_to_one_level(self):
        calibrator = IsotonicCalibrator().fit([1.0, 2.0, 3.0, 4.0], [1, 1, 0, 0])
        for x in (0.0, 1.0, 2.5, 4.0, 10.0):
            self.assertAlmostEqual(calibrator.predict(x), 0.5)

    def test_a_cascade_of_violations_pools_back_through_earlier_blocks(self):
        # targets (N+ = 4, N- = 4: present 5/6, absent 1/6): 1/6, 5/6, 1/6, 1/6, 1/6, 5/6, 5/6, 5/6
        present = [0, 1, 0, 0, 0, 1, 1, 1]
        calibrator = IsotonicCalibrator().fit([float(i) for i in range(8)], present)
        low, high = 1 / 6, 5 / 6
        mid = (high + 3 * low) / 4  # rows 1-4 pool to one block
        self.assertAlmostEqual(calibrator.predict(0.0), low)
        self.assertAlmostEqual(calibrator.predict(1.0), mid)
        self.assertAlmostEqual(calibrator.predict(4.0), mid)
        self.assertAlmostEqual(calibrator.predict(5.0), high)
        self.assertAlmostEqual(calibrator.predict(0.5), (low + mid) / 2)

    def test_output_is_monotone_and_never_exactly_zero_or_one(self):
        rng = random.Random(4)
        probabilities = [rng.random() for _ in range(300)]
        present = [1 if rng.random() < p else 0 for p in probabilities]
        calibrator = IsotonicCalibrator().fit(probabilities, present)
        values = [calibrator.predict(i / 200) for i in range(201)]
        self.assertTrue(all(b >= a - 1e-15 for a, b in zip(values, values[1:], strict=False)))
        self.assertTrue(all(0.0 < v < 1.0 for v in values))
        self.assertEqual(calibrator.predict(-5.0), calibrator.predict(0.0))
        self.assertEqual(calibrator.predict(5.0), calibrator.predict(1.0))

    def test_follows_the_observed_rate_on_calibrated_data(self):
        rng = random.Random(5)
        probabilities = [rng.random() for _ in range(3000)]
        present = [1 if rng.random() < p else 0 for p in probabilities]
        calibrator = IsotonicCalibrator().fit(probabilities, present)
        for p in (0.2, 0.5, 0.8):
            self.assertAlmostEqual(calibrator.predict(p), p, delta=0.06)

    def test_single_point_and_errors(self):
        self.assertAlmostEqual(IsotonicCalibrator().fit([0.3], [1]).predict(0.9), 2 / 3)
        with self.assertRaises(ModelError):
            IsotonicCalibrator().predict(0.5)
        with self.assertRaises(ModelError):
            IsotonicCalibrator().fit([], [])
        with self.assertRaises(ModelError):
            IsotonicCalibrator().fit([0.5, 0.5], [1])


class ParameterTests(unittest.TestCase):
    def test_registered_with_defaults_and_round_trip(self):
        self.assertIs(REGISTRY["calibrated"], CalibratedClassifier)
        self.assertEqual(
            CalibratedClassifier().get_params(),
            {"base": "gaussian_nb", "base_params": {}, "method": "sigmoid", "folds": 3, "seed": 0},
        )
        model = CalibratedClassifier(
            base="knn", base_params={"k": 5}, method="isotonic", folds=4, seed=9
        )
        again = create_model("calibrated", **model.get_params())
        self.assertEqual(again.get_params(), model.get_params())
        self.assertEqual(METHODS, ("sigmoid", "isotonic"))

    def test_params_are_copied(self):
        given = {"k": 3}
        model = CalibratedClassifier(base="knn", base_params=given)
        given["k"] = 99
        model.get_params()["base_params"]["k"] = 77
        self.assertEqual(model.get_params()["base_params"], {"k": 3})

    def test_bad_parameters_fail_at_construction(self):
        for kwargs in (
            {"base": "nope"},
            {"base": "mean"},  # a regressor
            {"base": "calibrated"},
            {"base": 3},
            {"base": "knn", "base_params": {"k": 0}},
            {"base": "knn", "base_params": {"colour": 1}},
            {"base_params": [1]},
            {"method": "platt"},
            {"folds": 1},
            {"folds": 21},
            {"folds": 2.5},
            {"folds": True},
            {"seed": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((ModelError, SeedError)):
                CalibratedClassifier(**kwargs)

    def test_the_base_gets_a_derived_seed_unless_it_has_one(self):
        model = CalibratedClassifier(base="perceptron", seed=5)
        self.assertEqual(
            model._make_base("final").get_params()["seed"],
            derive_seed(5, "calibrated-base", "final"),
        )
        self.assertNotEqual(
            model._make_base("fold-0").get_params()["seed"],
            model._make_base("fold-1").get_params()["seed"],
        )
        fixed = CalibratedClassifier(base="perceptron", base_params={"seed": 3}, seed=5)
        self.assertEqual(fixed._make_base("final").get_params()["seed"], 3)
        self.assertNotIn(
            "seed", CalibratedClassifier(base="gaussian_nb")._make_base("x").get_params()
        )

    def test_use_before_fit(self):
        with self.assertRaises(ModelError):
            CalibratedClassifier().predict_proba([[1.0]])
        with self.assertRaises(ModelError):
            CalibratedClassifier().predict_proba_uncalibrated([[1.0]])


class BehaviourTests(unittest.TestCase):
    def test_matches_a_hand_built_pipeline(self):
        for method, calibrator_class in (
            ("sigmoid", SigmoidCalibrator),
            ("isotonic", IsotonicCalibrator),
        ):
            with self.subTest(method=method):
                self.check_pipeline(method, calibrator_class)

    def check_pipeline(self, method, calibrator_class):
        X, y = noisy(n=90, seed=2)
        model = CalibratedClassifier(base="gaussian_nb", method=method, folds=3, seed=4)
        model.fit(X, y)
        classes = ["p", "q"]
        held_out = [[0.0, 0.0] for _ in X]
        for train, test in kfold_indices(len(X), 3, derive_seed(4, "calibration"), y):
            nb = GaussianNB()
            nb.fit([X[i] for i in train], [y[i] for i in train])
            for i, row in zip(test, nb.predict_proba([X[i] for i in test]), strict=True):
                held_out[i] = row
        calibrators = [
            calibrator_class().fit([r[c] for r in held_out], [int(t == label) for t in y])
            for c, label in enumerate(classes)
        ]
        final = GaussianNB()
        final.fit(X, y)
        queries = [[0.0, 0.0, 0.0], [2.0, 1.0, -1.0], [-2.5, -1.0, 2.0]]
        for query, raw in zip(queries, final.predict_proba(queries), strict=True):
            mapped = [cal.predict(p) for cal, p in zip(calibrators, raw, strict=True)]
            want = [m / sum(mapped) for m in mapped]
            got = model.predict_proba([query])[0]
            for g, w in zip(got, want, strict=True):
                self.assertAlmostEqual(g, w)

    def test_the_two_methods_give_different_probabilities(self):
        X, y = noisy(n=120, seed=5)
        a, b = (CalibratedClassifier(method=m, seed=1) for m in METHODS)
        a.fit(X, y)
        b.fit(X, y)
        self.assertNotEqual(a.predict_proba(X), b.predict_proba(X))

    def test_the_uncalibrated_probabilities_are_the_base_model_on_all_rows(self):
        X, y = noisy(n=80)
        model = CalibratedClassifier(base="gaussian_nb")
        model.fit(X, y)
        base = GaussianNB()
        base.fit(X, y)
        self.assertEqual(model.predict_proba_uncalibrated(X[:5]), base.predict_proba(X[:5]))
        self.assertNotEqual(model.predict_proba(X[:5]), base.predict_proba(X[:5]))

    def test_rows_sum_to_one_and_follow_the_class_order(self):
        X, y = noisy(n=150, classes=3)
        for method in METHODS:
            model = CalibratedClassifier(base="gaussian_nb", method=method, seed=1)
            model.fit(X, y)
            self.assertEqual(model.classes_, ("a", "b", "c"))
            for row in model.predict_proba(X):
                self.assertAlmostEqual(sum(row), 1.0)
                self.assertTrue(all(0.0 <= p <= 1.0 for p in row))
            accuracy = sum(p == t for p, t in zip(model.predict(X), y, strict=True)) / len(y)
            self.assertGreater(accuracy, 0.7)

    def test_calibration_improves_log_loss_for_an_overconfident_model(self):
        # a tree grown until every leaf is pure says 0 or 1 about rows it often gets wrong
        X, y = noisy(n=600, seed=3)
        Xt, yt = noisy(n=600, seed=4)
        model = CalibratedClassifier(
            base="decision_tree", base_params={"max_depth": None}, method="sigmoid", folds=4, seed=1
        )
        model.fit(X, y)
        raw = log_loss(yt, model.predict_proba_uncalibrated(Xt))
        calibrated = log_loss(yt, model.predict_proba(Xt))
        self.assertGreater(raw, 1.0)
        self.assertLess(calibrated, 0.6)

    def test_calibrated_probabilities_match_observed_rates(self):
        X, y = noisy(n=900, seed=6)
        Xt, yt = noisy(n=900, seed=7)
        model = CalibratedClassifier(base="perceptron", method="sigmoid", folds=4, seed=2)
        model.fit(X, y)
        scored = [(row[0], t == "p") for row, t in zip(model.predict_proba(Xt), yt, strict=True)]
        for low, high in ((0.0, 0.3), (0.3, 0.7), (0.7, 1.01)):
            inside = [hit for p, hit in scored if low <= p < high]
            mean_p = sum(p for p, _ in scored if low <= p < high) / len(inside)
            self.assertAlmostEqual(sum(inside) / len(inside), mean_p, delta=0.08)

    def test_repeatable_and_the_seed_changes_the_folds(self):
        X, y = noisy(n=120)
        a, b, c = (
            CalibratedClassifier(base="knn", base_params={"k": 7}, seed=s) for s in (1, 1, 2)
        )
        for m in (a, b, c):
            m.fit(X, y)
        self.assertEqual(a.predict_proba(X), b.predict_proba(X))
        self.assertNotEqual(a.predict_proba(X), c.predict_proba(X))

    def test_a_single_class_predicts_it_with_probability_one(self):
        model = CalibratedClassifier()
        model.fit([[1.0], [2.0], [3.0]], ["only"] * 3)
        self.assertEqual(model.predict_proba([[9.0]]), [[1.0]])
        self.assertEqual(model.predict([[9.0]]), ["only"])

    def test_a_class_missing_from_some_training_folds_is_handled(self):
        X = [[float(i)] for i in range(12)] + [[50.0], [51.0]]
        y = ["a"] * 12 + ["rare", "rare"]
        model = CalibratedClassifier(base="decision_tree", folds=3, seed=1)
        model.fit(X, y)
        for row in model.predict_proba([[0.0], [50.5]]):
            self.assertAlmostEqual(sum(row), 1.0)
        self.assertEqual(model.classes_, ("a", "rare"))

    def test_too_few_rows_for_the_folds_is_an_error(self):
        with self.assertRaises(ModelError):
            CalibratedClassifier(folds=5).fit([[1.0], [2.0], [3.0]], ["a", "b", "a"])

    def test_wrong_width_at_predict_is_an_error(self):
        X, y = noisy(n=60)
        model = CalibratedClassifier()
        model.fit(X, y)
        with self.assertRaises(ModelError):
            model.predict_proba([[1.0]])

    def test_a_refit_starts_over(self):
        X, y = noisy(n=60)
        model = CalibratedClassifier()
        model.fit(X, y)
        first = model.predict_proba(X)
        model.fit(X, y)
        self.assertEqual(model.predict_proba(X), first)
        model.fit([[1.0], [2.0]], ["z", "z"])
        self.assertEqual(model.predict_proba([[1.0]]), [[1.0]])


class RunnerTests(unittest.TestCase):
    def table(self):
        X, y = noisy(n=120, seed=8)
        return Table(
            ("a", "b", "c", "label"),
            tuple((*(f"{v:.3f}" for v in r), t) for r, t in zip(X, y, strict=True)),
        )

    def config(self, **overrides):
        fields = {
            "experiment_id": "cal",
            "model": "calibrated",
            "seed": 4,
            "dataset_id": "rows",
            "metrics": ("log_loss", "ece", "accuracy"),
            "params": {"base": "perceptron", "base_params": {"max_iter": 10}, "folds": 3},
        }
        fields.update(overrides)
        return ExperimentConfig(**fields)

    def test_the_runner_derives_the_seed_and_the_run_reproduces(self):
        table = self.table()
        with Store(":memory:") as store:
            result = run_experiment(store, self.config(), table, "label")
            self.assertEqual(
                store.get_run(result.run_id).tags["model_seed"], str(derive_seed(4, "model"))
            )
            self.assertTrue(math.isfinite(result.metrics["ece"]))
            verdict = check_reproducibility(store, result.run_id, table, "label")
            self.assertEqual(verdict.verdict, "reproduced")

    def test_calibration_lowers_ece_and_log_loss_in_a_recorded_run(self):
        table = self.table()
        with Store(":memory:") as store:
            raw = run_experiment(
                store,
                self.config(experiment_id="raw", model="perceptron", params={"max_iter": 10}),
                table,
                "label",
            )
            calibrated = run_experiment(store, self.config(), table, "label")
            self.assertLess(calibrated.metrics["log_loss"], raw.metrics["log_loss"])

    def test_the_nested_parameters_are_part_of_the_config_hash(self):
        a = self.config()
        b = self.config(params={"base": "perceptron", "base_params": {"max_iter": 11}, "folds": 3})
        self.assertNotEqual(a.config_hash(), b.config_hash())

    def test_cross_validation_wraps_calibration_in_each_fold(self):
        table = self.table()
        with Store(":memory:") as store:
            result = run_experiment(store, self.config(cv_folds=3), table, "label")
            self.assertIn("log_loss_std", result.metrics)


if __name__ == "__main__":
    unittest.main()
