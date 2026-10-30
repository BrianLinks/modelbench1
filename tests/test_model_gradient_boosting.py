import math
import random
import unittest

from modelbench.config import ExperimentConfig
from modelbench.datasets import Table
from modelbench.errors import ModelError, SeedError
from modelbench.models import REGISTRY, REGRESSORS, create_model, create_regressor
from modelbench.models.decision_tree_regressor import DecisionTreeRegressor
from modelbench.models.gradient_boosting import GradientBoostingClassifier, softmax
from modelbench.models.gradient_boosting_regressor import (
    MAX_STAGES,
    GradientBoostingRegressor,
    check_learning_rate,
    check_subsample,
    stage_rows,
)
from modelbench.reproducibility import check_reproducibility
from modelbench.runner import run_experiment
from modelbench.seeding import derive_seed, rng_for
from modelbench.store import Store

ERRORS = (ModelError, SeedError)


def noisy_classes(n=200, seed=1, classes=2):
    rng = random.Random(seed)
    X = [[rng.uniform(-3, 3) for _ in range(3)] for _ in range(n)]
    if classes == 2:
        y = ["p" if r[0] + r[1] * r[2] + rng.gauss(0, 0.5) > 0 else "q" for r in X]
    else:
        y = ["a" if r[0] < -1 else "b" if r[0] < 1 else "c" for r in X]
    return X, y


def curve(n=200, seed=2):
    rng = random.Random(seed)
    X = [[rng.uniform(-3, 3) for _ in range(3)] for _ in range(n)]
    return X, [r[0] ** 2 + 2 * r[1] + rng.gauss(0, 0.3) for r in X]


class ParameterTests(unittest.TestCase):
    def test_learning_rate_and_subsample_ranges(self):
        for good in (0.01, 0.5, 1, 1.0):
            self.assertEqual(check_learning_rate(good), float(good))
            self.assertEqual(check_subsample(good), float(good))
        for bad in (0, 0.0, -0.1, 1.01, float("nan"), float("inf"), True, "0.1", None):
            with self.subTest(bad=bad):
                with self.assertRaises(ModelError):
                    check_learning_rate(bad)
                with self.assertRaises(ModelError):
                    check_subsample(bad)

    def test_registered_and_parameters_round_trip(self):
        self.assertIs(REGISTRY["gradient_boosting"], GradientBoostingClassifier)
        self.assertIs(REGRESSORS["gradient_boosting_regressor"], GradientBoostingRegressor)
        for cls in (GradientBoostingClassifier, GradientBoostingRegressor):
            model = cls(n_estimators=7, learning_rate=0.3, max_depth=2, subsample=0.8, seed=5)
            self.assertEqual(
                model.get_params(),
                {
                    "n_estimators": 7,
                    "learning_rate": 0.3,
                    "max_depth": 2,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "subsample": 0.8,
                    "seed": 5,
                },
            )
            self.assertEqual(cls().get_params()["n_estimators"], 100)
            self.assertEqual(cls().get_params()["max_depth"], 3)
            self.assertEqual(cls().get_params()["learning_rate"], 0.1)

    def test_bad_parameters_fail_at_construction(self):
        for kwargs in (
            {"n_estimators": 0},
            {"n_estimators": MAX_STAGES + 1},
            {"n_estimators": 1.5},
            {"learning_rate": 0},
            {"learning_rate": 2},
            {"max_depth": 0},
            {"max_depth": None},
            {"min_samples_split": 1},
            {"min_samples_leaf": 0},
            {"subsample": 0},
            {"subsample": 1.5},
            {"seed": -3},
        ):
            for cls in (GradientBoostingClassifier, GradientBoostingRegressor):
                with self.subTest(kwargs=kwargs, cls=cls.name), self.assertRaises(ERRORS):
                    cls(**kwargs)

    def test_unknown_parameters_and_use_before_fit(self):
        with self.assertRaises(ModelError):
            create_model("gradient_boosting", trees=3)
        with self.assertRaises(ModelError):
            create_regressor("gradient_boosting_regressor", trees=3)
        with self.assertRaises(ModelError):
            GradientBoostingClassifier().predict([[1.0]])
        with self.assertRaises(ModelError):
            GradientBoostingRegressor().predict([[1.0]])
        with self.assertRaises(ModelError):
            GradientBoostingRegressor().staged_predict([[1.0]])


class StageRowsTests(unittest.TestCase):
    def test_everything_without_subsampling(self):
        self.assertEqual(stage_rows(5, 1.0, 1, 0, "p"), [0, 1, 2, 3, 4])

    def test_a_seeded_sorted_share(self):
        rows = stage_rows(10, 0.5, 3, 4, "gb")
        self.assertEqual(rows, sorted(rng_for(3, "gb", 4).sample(range(10), 5)))
        self.assertEqual(len(set(rows)), 5)
        self.assertEqual(rows, stage_rows(10, 0.5, 3, 4, "gb"))
        self.assertNotEqual(rows, stage_rows(10, 0.5, 3, 5, "gb"))
        self.assertNotEqual(rows, stage_rows(10, 0.5, 4, 4, "gb"))

    def test_never_fewer_than_two_rows_and_never_more_than_there_are(self):
        self.assertEqual(len(stage_rows(3, 0.1, 1, 0, "p")), 2)
        self.assertEqual(len(stage_rows(100, 0.01, 1, 0, "p")), 2)
        self.assertEqual(stage_rows(1, 0.5, 1, 0, "p"), [0])
        self.assertEqual(len(stage_rows(10, 0.96, 1, 0, "p")), 10)


class RegressorTests(unittest.TestCase):
    STEP_X = [[0.0], [1.0], [2.0], [3.0]]
    STEP_Y = [0.0, 0.0, 10.0, 10.0]

    def test_one_full_stage_fits_a_step_exactly(self):
        model = GradientBoostingRegressor(n_estimators=1, learning_rate=1.0, max_depth=1)
        model.fit(self.STEP_X, self.STEP_Y)
        self.assertEqual(model.predict(self.STEP_X), self.STEP_Y)
        self.assertEqual(model.train_loss_, [0.0])

    def test_the_error_shrinks_by_the_learning_rate_each_stage(self):
        model = GradientBoostingRegressor(n_estimators=4, learning_rate=0.5, max_depth=1)
        model.fit(self.STEP_X, self.STEP_Y)
        # start at 5; every stage removes half of the remaining error of 5
        for stage, loss in enumerate(model.train_loss_, start=1):
            self.assertAlmostEqual(loss, 25 * 0.25**stage)
        staged = model.staged_predict(self.STEP_X)
        self.assertEqual(len(staged), 4)
        self.assertAlmostEqual(staged[0][0], 2.5)
        self.assertAlmostEqual(staged[0][3], 7.5)
        self.assertAlmostEqual(staged[1][0], 1.25)
        self.assertEqual(staged[-1], model.predict(self.STEP_X))

    def test_constant_targets_predict_the_constant(self):
        model = GradientBoostingRegressor(n_estimators=5)
        model.fit([[1.0], [2.0], [3.0]], [4.0, 4.0, 4.0])
        self.assertEqual(model.predict([[100.0]]), [4.0])
        self.assertEqual(model.train_loss_, [0.0] * 5)

    def test_the_start_is_the_mean_when_trees_cannot_split(self):
        model = GradientBoostingRegressor(n_estimators=3, min_samples_split=100)
        model.fit([[1.0], [2.0], [3.0], [4.0]], [1.0, 2.0, 3.0, 10.0])
        self.assertEqual(model.predict([[0.0]]), [4.0])

    def test_training_loss_never_rises_without_subsampling(self):
        X, y = curve()
        model = GradientBoostingRegressor(n_estimators=40, learning_rate=0.3, seed=1)
        model.fit(X, y)
        losses = model.train_loss_
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(losses, losses[1:], strict=False)))
        self.assertLess(losses[-1], losses[0] / 5)

    def test_reported_loss_matches_the_final_predictions(self):
        X, y = curve(n=80)
        model = GradientBoostingRegressor(n_estimators=10, seed=2)
        model.fit(X, y)
        mse = sum((p - t) ** 2 for p, t in zip(model.predict(X), y, strict=True)) / len(y)
        self.assertAlmostEqual(model.train_loss_[-1], mse)

    def test_matches_a_hand_built_two_stage_model(self):
        X, y = curve(n=50)
        model = GradientBoostingRegressor(n_estimators=2, learning_rate=0.4, max_depth=2)
        model.fit(X, y)
        mean = sum(y) / len(y)
        first = DecisionTreeRegressor(max_depth=2)
        first.fit(X, [t - mean for t in y])
        after_one = [mean + 0.4 * v for v in first.predict(X)]
        second = DecisionTreeRegressor(max_depth=2)
        second.fit(X, [t - s for t, s in zip(y, after_one, strict=True)])
        queries = [[0.0, 0.0, 0.0], [2.0, -1.0, 1.0], [-3.0, 3.0, 0.5]]
        for got, q in zip(model.predict(queries), queries, strict=True):
            want = mean + 0.4 * first.predict([q])[0] + 0.4 * second.predict([q])[0]
            self.assertAlmostEqual(got, want)

    def test_beats_one_tree_held_out(self):
        X, y = curve(n=300, seed=4)
        single = DecisionTreeRegressor(max_depth=3)
        boosted = GradientBoostingRegressor(n_estimators=60, seed=1)
        single.fit(X[:150], y[:150])
        boosted.fit(X[:150], y[:150])

        def sse(model):
            return sum((p - t) ** 2 for p, t in zip(model.predict(X[150:]), y[150:], strict=True))

        self.assertLess(sse(boosted), sse(single) / 2)

    def test_seed_matters_only_when_subsampling(self):
        X, y = curve(n=60)
        a = GradientBoostingRegressor(n_estimators=8, seed=1)
        b = GradientBoostingRegressor(n_estimators=8, seed=2)
        a.fit(X, y)
        b.fit(X, y)
        self.assertEqual(a.predict(X), b.predict(X))
        c, d, e = (
            GradientBoostingRegressor(n_estimators=8, subsample=0.5, seed=s) for s in (1, 1, 2)
        )
        for m in (c, d, e):
            m.fit(X, y)
        self.assertEqual(c.predict(X), d.predict(X))
        self.assertNotEqual(c.predict(X), e.predict(X))
        self.assertNotEqual(a.predict(X), c.predict(X))

    def test_a_refit_starts_over(self):
        X, y = curve(n=40)
        model = GradientBoostingRegressor(n_estimators=5)
        model.fit(X, y)
        first = model.predict(X)
        model.fit(X, y)
        self.assertEqual(model.predict(X), first)
        self.assertEqual(len(model.train_loss_), 5)


class ClassifierTests(unittest.TestCase):
    def test_softmax(self):
        self.assertEqual(softmax([0.0, 0.0]), [0.5, 0.5])
        row = softmax([1000.0, 0.0, -1000.0])  # would overflow without subtracting the maximum
        self.assertEqual(row, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(sum(softmax([1.0, 2.0, 3.0])), 1.0)
        self.assertAlmostEqual(softmax([0.0, math.log(3.0)])[1], 0.75)

    def test_one_full_stage_on_a_balanced_pair_by_hand(self):
        # priors 1/2; residuals for class a are +-1/2; a stump puts the two a rows in one leaf,
        # whose Newton value is (1/2) * 1 / (2 * 1/4) = 1, and the other leaf gets -1
        model = GradientBoostingClassifier(n_estimators=1, learning_rate=1.0, max_depth=1)
        model.fit([[0.0], [1.0], [2.0], [3.0]], ["a", "a", "b", "b"])
        want = 1 / (1 + math.exp(-2))
        low, high = model.predict_proba([[0.5], [2.5]])
        self.assertAlmostEqual(low[0], want)
        self.assertAlmostEqual(low[1], 1 - want)
        self.assertAlmostEqual(high[1], want)
        self.assertAlmostEqual(model.train_loss_[0], -math.log(want))

    def test_a_second_stage_works_from_the_updated_probabilities(self):
        # after stage one the two a rows have p(a) = q = 1 / (1 + e^-2); their residual is 1 - q,
        # so stage two's Newton value is (1/2) * 2(1 - q) / (2(1 - q) q) = 1 / (2q) and the score
        # gap between a and b grows by 1 / q
        model = GradientBoostingClassifier(n_estimators=2, learning_rate=1.0, max_depth=1)
        model.fit([[0.0], [1.0], [2.0], [3.0]], ["a", "a", "b", "b"])
        q = 1 / (1 + math.exp(-2))
        want = 1 / (1 + math.exp(-(2 + 1 / q)))
        self.assertAlmostEqual(model.predict_proba([[0.5]])[0][0], want)
        self.assertAlmostEqual(model.predict_proba([[2.5]])[0][1], want)
        self.assertAlmostEqual(model.train_loss_[1], -math.log(want))

    def test_trees_that_cannot_split_leave_the_class_priors(self):
        model = GradientBoostingClassifier(n_estimators=5, min_samples_split=100)
        model.fit([[0.0], [1.0], [2.0], [3.0]], ["a", "a", "b", "c"])
        self.assertEqual(model.classes_, ("a", "b", "c"))
        for row in model.predict_proba([[7.0]]):
            for got, want in zip(row, (0.5, 0.25, 0.25), strict=True):
                self.assertAlmostEqual(got, want)

    def test_a_three_class_stage_matches_an_independent_calculation(self):
        X, y = noisy_classes(n=30, classes=3, seed=5)
        model = GradientBoostingClassifier(n_estimators=1, learning_rate=0.5, max_depth=2)
        model.fit(X, y)
        classes = sorted(set(y))
        n = len(y)
        prior = [y.count(c) / n for c in classes]
        want = [[math.log(p) for p in prior] for _ in range(n)]
        probs = [softmax(row) for row in want]
        queries = [[0.0, 0.0, 0.0], [-2.0, 1.0, 0.0], [2.5, -1.0, 2.0]]
        scores = [[math.log(p) for p in prior] for _ in queries]
        for c, label in enumerate(classes):
            r = [(1.0 if t == label else 0.0) - probs[i][c] for i, t in enumerate(y)]
            tree = DecisionTreeRegressor(max_depth=2)
            tree.fit(X, r)
            leaves = tree.apply(X)
            value = {}
            for leaf in set(leaves):
                members = [i for i, lf in enumerate(leaves) if lf == leaf]
                top = sum(r[i] for i in members)
                bottom = sum(abs(r[i]) * (1 - abs(r[i])) for i in members)
                value[leaf] = (2 / 3) * top / bottom
            for q, row in zip(queries, scores, strict=True):
                row[c] += 0.5 * value[tree.apply([q])[0]]
        for got, row in zip(model.predict_proba(queries), scores, strict=True):
            for g, w in zip(got, softmax(row), strict=True):
                self.assertAlmostEqual(g, w)

    def test_probabilities_are_proper_and_follow_the_class_order(self):
        X, y = noisy_classes(classes=3)
        model = GradientBoostingClassifier(n_estimators=15, seed=1)
        model.fit(X, y)
        self.assertEqual(model.classes_, ("a", "b", "c"))
        for row in model.predict_proba(X):
            self.assertAlmostEqual(sum(row), 1.0)
            self.assertTrue(all(0.0 <= p <= 1.0 for p in row))
        self.assertGreater(
            sum(p == t for p, t in zip(model.predict(X), y, strict=True)) / len(y), 0.95
        )

    def test_reported_loss_matches_the_final_probabilities(self):
        X, y = noisy_classes(n=120)
        model = GradientBoostingClassifier(n_estimators=12, seed=1)
        model.fit(X, y)
        proba = model.predict_proba(X)
        slot = {"p": 0, "q": 1}
        loss = -sum(math.log(row[slot[t]]) for row, t in zip(proba, y, strict=True)) / len(y)
        self.assertAlmostEqual(model.train_loss_[-1], loss)
        self.assertEqual(len(model.train_loss_), 12)
        self.assertLess(model.train_loss_[-1], model.train_loss_[0])

    def test_beats_one_tree_on_noisy_held_out_data(self):
        X, y = noisy_classes(n=400, seed=3)
        boosted = GradientBoostingClassifier(n_estimators=40, seed=1)
        boosted.fit(X[:200], y[:200])
        majority = max(set(y), key=y[200:].count)
        base = sum(t == majority for t in y[200:]) / 200
        accuracy = sum(p == t for p, t in zip(boosted.predict(X[200:]), y[200:], strict=True)) / 200
        self.assertGreater(accuracy, base + 0.1)

    def test_separable_data_saturates_without_nan(self):
        X = [[float(i)] for i in range(20)] + [[float(100 + i)] for i in range(20)]
        y = ["a"] * 20 + ["b"] * 20
        model = GradientBoostingClassifier(n_estimators=60, learning_rate=1.0, seed=1)
        model.fit(X, y)
        for row in model.predict_proba([[0.0], [119.0], [50.0]]):
            self.assertTrue(all(math.isfinite(p) for p in row))
            self.assertAlmostEqual(sum(row), 1.0)
        self.assertEqual(model.predict([[-5.0], [200.0]]), ["a", "b"])
        self.assertTrue(math.isfinite(model.train_loss_[-1]))
        self.assertLess(model.train_loss_[-1], 1e-6)

    def test_a_single_class(self):
        model = GradientBoostingClassifier(n_estimators=3)
        model.fit([[1.0], [2.0]], ["only", "only"])
        self.assertEqual(model.predict_proba([[5.0]]), [[1.0]])
        self.assertEqual(model.predict([[5.0]]), ["only"])
        self.assertEqual(model.train_loss_, [])

    def test_a_rare_class_is_still_predicted_somewhere(self):
        X = [[float(i)] for i in range(30)] + [[100.0], [101.0]]
        y = ["a"] * 30 + ["b", "b"]
        model = GradientBoostingClassifier(n_estimators=30, seed=1)
        model.fit(X, y)
        self.assertEqual(model.predict([[100.5]]), ["b"])

    def test_integer_labels_keep_their_numeric_order(self):
        model = GradientBoostingClassifier(n_estimators=5)
        model.fit([[0.0], [1.0], [10.0], [11.0]], [10, 10, 2, 2])
        self.assertEqual(model.classes_, (2, 10))
        self.assertEqual(model.predict([[0.5], [10.5]]), [10, 2])

    def test_seed_matters_only_when_subsampling(self):
        X, y = noisy_classes(n=80)
        a, b = (GradientBoostingClassifier(n_estimators=6, seed=s) for s in (1, 2))
        a.fit(X, y)
        b.fit(X, y)
        self.assertEqual(a.predict_proba(X), b.predict_proba(X))
        c, d = (GradientBoostingClassifier(n_estimators=6, subsample=0.5, seed=s) for s in (1, 2))
        c.fit(X, y)
        d.fit(X, y)
        self.assertNotEqual(c.predict_proba(X), d.predict_proba(X))

    def test_wrong_width_at_predict_is_an_error(self):
        model = GradientBoostingClassifier(n_estimators=2)
        model.fit([[0.0, 1.0], [5.0, 6.0]], ["a", "b"])
        with self.assertRaises(ModelError):
            model.predict([[1.0]])

    def test_a_refit_starts_over(self):
        X, y = noisy_classes(n=60)
        model = GradientBoostingClassifier(n_estimators=4)
        model.fit(X, y)
        first = model.predict_proba(X)
        model.fit(X, y)
        self.assertEqual(model.predict_proba(X), first)
        self.assertEqual(len(model.train_loss_), 4)
        model.fit([[1.0], [2.0]], ["z", "z"])
        self.assertEqual(model.train_loss_, [])


class RunnerTests(unittest.TestCase):
    def table(self):
        X, y = noisy_classes(n=90, seed=6)
        return Table(
            ("a", "b", "c", "label"),
            tuple((*(f"{v:.3f}" for v in r), t) for r, t in zip(X, y, strict=True)),
        )

    def config(self, **overrides):
        fields = {
            "experiment_id": "boost",
            "model": "gradient_boosting",
            "seed": 4,
            "dataset_id": "rows",
            "metrics": ("accuracy", "log_loss"),
            "params": {"n_estimators": 10},
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
            self.assertTrue(math.isfinite(result.metrics["log_loss"]))
            verdict = check_reproducibility(store, result.run_id, table, "label")
            self.assertEqual(verdict.verdict, "reproduced")

    def test_cross_validation(self):
        table = self.table()
        with Store(":memory:") as store:
            result = run_experiment(store, self.config(cv_folds=3), table, "label")
            self.assertIn("log_loss_std", result.metrics)

    def test_the_regressor_runs(self):
        X, y = curve(n=80)
        table = Table(
            ("a", "b", "c", "y"),
            tuple((*(f"{v:.3f}" for v in r), f"{t:.3f}") for r, t in zip(X, y, strict=True)),
        )
        config = self.config(
            experiment_id="reg", model="gradient_boosting_regressor", metrics=("r2",),
            params={"n_estimators": 30},
        )  # fmt: skip
        with Store(":memory:") as store:
            self.assertGreater(run_experiment(store, config, table, "y").metrics["r2"], 0.8)


if __name__ == "__main__":
    unittest.main()
