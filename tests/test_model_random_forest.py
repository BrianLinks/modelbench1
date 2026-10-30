import math
import random
import unittest

from modelbench.config import ExperimentConfig
from modelbench.datasets import Table
from modelbench.errors import ModelError, SeedError
from modelbench.models import REGISTRY, REGRESSORS, create_model
from modelbench.models.bagged_trees import MAX_ESTIMATORS, BaggedTrees, bootstrap_indices
from modelbench.models.decision_tree import DecisionTreeClassifier
from modelbench.models.decision_tree_regressor import DecisionTreeRegressor
from modelbench.models.random_forest import (
    RandomForest,
    check_max_features,
    feature_chooser,
    features_per_split,
)
from modelbench.models.random_forest_regressor import RandomForestRegressor
from modelbench.reproducibility import check_reproducibility
from modelbench.runner import run_experiment
from modelbench.seeding import derive_seed, rng_for
from modelbench.store import Store


def informative_second_feature(n=60, seed=1):
    """Only column 1 decides the class; columns 0 and 2 are noise."""
    rng = random.Random(seed)
    X = [[rng.uniform(0, 1), float(i % 2), rng.uniform(0, 1)] for i in range(n)]
    return X, ["odd" if row[1] else "even" for row in X]


def noisy(seed=1, n=120, width=5):
    rng = random.Random(seed)
    X = [[rng.uniform(-3, 3) for _ in range(width)] for _ in range(n)]
    y = ["p" if r[0] + r[1] + rng.uniform(-1.5, 1.5) > 0 else "q" for r in X]
    return X, y


class MaxFeaturesTests(unittest.TestCase):
    def test_accepted_values(self):
        for value in (None, "sqrt", "log2", 1, 7, 0.5, 1.0, 0.01):
            self.assertEqual(check_max_features(value), value)

    def test_rejected_values(self):
        for value in (0, -1, 0.0, -0.5, 1.5, float("nan"), float("inf"), "all", "SQRT", True, [2]):
            with self.subTest(value=value), self.assertRaises(ModelError):
                check_max_features(value)

    def test_features_per_split_by_hand(self):
        cases = [
            (None, 9, 9),
            ("sqrt", 9, 3),
            ("sqrt", 10, 3),
            ("sqrt", 7, 2),  # 2.65 is cut down, not rounded
            ("sqrt", 8, 2),
            ("sqrt", 1, 1),
            ("log2", 8, 3),
            ("log2", 6, 2),  # 2.58 is cut down, not rounded
            ("log2", 7, 2),
            ("log2", 2, 1),
            ("log2", 1, 1),  # log2(1) is 0, but a split needs a feature
            (4, 9, 4),
            (50, 9, 9),  # capped at the number of features
            (0.5, 10, 5),
            (0.34, 5, 2),  # 1.7 rounds to 2
            (0.01, 10, 1),  # never fewer than one
            (1.0, 6, 6),
        ]
        for value, width, want in cases:
            with self.subTest(value=value, width=width):
                self.assertEqual(features_per_split(value, width), want)


class FeatureChooserTests(unittest.TestCase):
    def test_draws_sorted_distinct_features_of_the_right_count(self):
        choose = feature_chooser("sqrt", 3, 0, "random_forest")
        for _ in range(20):
            picked = choose(16)
            self.assertEqual(len(picked), 4)
            self.assertEqual(picked, sorted(set(picked)))
            self.assertTrue(all(0 <= f < 16 for f in picked))

    def test_is_a_pure_function_of_seed_tree_and_call_number(self):
        def run(seed, tree, purpose="random_forest"):
            choose = feature_chooser(2, seed, tree, purpose)
            return [choose(10) for _ in range(6)]

        self.assertEqual(run(3, 0), run(3, 0))
        self.assertNotEqual(run(3, 0), run(4, 0))
        self.assertNotEqual(run(3, 0), run(3, 1))
        self.assertNotEqual(run(3, 0), run(3, 0, "random_forest_regressor"))
        self.assertEqual(
            run(3, 0)[0], sorted(rng_for(3, "random_forest", 0, "features").sample(range(10), 2))
        )

    def test_every_feature_is_eventually_offered(self):
        choose = feature_chooser(1, 5, 0, "random_forest")
        self.assertEqual({choose(6)[0] for _ in range(200)}, set(range(6)))

    def test_all_features_when_asked(self):
        self.assertEqual(feature_chooser(None, 1, 0, "p")(4), [0, 1, 2, 3])


class TreeHookTests(unittest.TestCase):
    """The tree looks only where its chooser says, and falls back when that finds nothing."""

    def data(self):
        # column 0 separates the classes perfectly; column 1 only partly
        X = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [10.0, 1.0], [11.0, 0.0], [12.0, 1.0]]
        return X, ["a", "a", "a", "b", "b", "b"]

    def test_classifier_uses_the_chosen_feature(self):
        X, y = self.data()
        tree = DecisionTreeClassifier(max_depth=1)
        tree._feature_chooser = lambda width: [1]
        tree.fit(X, y)
        self.assertTrue(tree.describe().startswith("if x1 <= 0.5"))
        free = DecisionTreeClassifier(max_depth=1)
        free.fit(X, y)
        self.assertTrue(free.describe().startswith("if x0 <= 6"))

    def test_classifier_falls_back_when_the_chosen_features_cannot_split(self):
        X = [[5.0, 0.0], [5.0, 1.0], [5.0, 2.0], [5.0, 9.0]]  # column 0 is constant
        y = ["a", "a", "b", "b"]
        tree = DecisionTreeClassifier()
        tree._feature_chooser = lambda width: [0]
        tree.fit(X, y)
        self.assertEqual(tree.predict(X), y)
        self.assertIn("x1", tree.describe())

    def test_regressor_uses_the_chosen_feature_and_falls_back(self):
        X = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [10.0, 1.0], [11.0, 0.0], [12.0, 1.0]]
        y = [0.0, 0.0, 0.0, 9.0, 9.0, 9.0]
        chosen = DecisionTreeRegressor(max_depth=1)
        chosen._feature_chooser = lambda width: [1]
        chosen.fit(X, y)
        self.assertTrue(chosen.describe().startswith("if x1 <= 0.5"))
        constant = [[5.0, float(i)] for i in range(6)]
        fallback = DecisionTreeRegressor(max_depth=1)
        fallback._feature_chooser = lambda width: [0]
        fallback.fit(constant, y)
        self.assertIn("x1", fallback.describe())

    def test_the_chooser_is_not_a_parameter(self):
        tree = DecisionTreeClassifier()
        tree._feature_chooser = lambda width: [0]
        self.assertEqual(tree.get_params(), DecisionTreeClassifier().get_params())
        self.assertEqual(
            DecisionTreeRegressor().get_params(),
            {"max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1},
        )


class RandomForestTests(unittest.TestCase):
    def test_registered_and_built_by_name(self):
        self.assertIs(REGISTRY["random_forest"], RandomForest)
        self.assertIs(REGRESSORS["random_forest_regressor"], RandomForestRegressor)
        self.assertIsInstance(create_model("random_forest", n_estimators=3), RandomForest)

    def test_parameters_round_trip(self):
        model = RandomForest(n_estimators=7, max_depth=3, max_features=0.5, seed=4)
        self.assertEqual(
            model.get_params(),
            {
                "n_estimators": 7,
                "max_depth": 3,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": 0.5,
                "seed": 4,
            },
        )
        again = create_model("random_forest", **model.get_params())
        X, y = noisy()
        model.fit(X, y)
        again.fit(X, y)
        self.assertEqual(model.predict_proba(X), again.predict_proba(X))

    def test_defaults(self):
        params = RandomForest().get_params()
        self.assertEqual(
            (params["n_estimators"], params["max_depth"], params["max_features"]),
            (50, None, "sqrt"),
        )
        regressor = RandomForestRegressor().get_params()
        self.assertEqual((regressor["max_features"], regressor["max_depth"]), (1.0, None))

    def test_bad_parameters_fail_at_construction(self):
        for kwargs in (
            {"n_estimators": 0},
            {"n_estimators": MAX_ESTIMATORS + 1},
            {"n_estimators": 2.5},
            {"max_depth": 0},
            {"min_samples_split": 1},
            {"min_samples_leaf": 0},
            {"max_features": 0},
            {"max_features": "most"},
            {"seed": -1},
            {"seed": "x"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((ModelError, SeedError)):
                    RandomForest(**kwargs)
                with self.assertRaises((ModelError, SeedError)):
                    RandomForestRegressor(**kwargs)

    def test_unknown_parameters_are_reported_by_the_factory(self):
        with self.assertRaises(ModelError):
            create_model("random_forest", n_trees=3)

    def test_use_before_fit_is_an_error(self):
        with self.assertRaises(ModelError):
            RandomForest().predict([[1.0]])
        with self.assertRaises(ModelError):
            RandomForestRegressor().predict([[1.0]])

    def test_same_seed_same_forest_and_other_seeds_differ(self):
        X, y = noisy()
        a, b, c = (RandomForest(n_estimators=10, seed=s) for s in (1, 1, 2))
        for model in (a, b, c):
            model.fit(X, y)
        self.assertEqual(a.predict_proba(X), b.predict_proba(X))
        self.assertNotEqual(a.predict_proba(X), c.predict_proba(X))

    def test_matches_a_hand_built_forest(self):
        X, y = noisy(n=40, width=4)
        model = RandomForest(n_estimators=3, max_depth=2, max_features=2, seed=11)
        model.fit(X, y)
        trees = []
        for i in range(3):
            rows = bootstrap_indices(len(X), 11, i, "random_forest")
            tree = DecisionTreeClassifier(max_depth=2)
            tree._feature_chooser = feature_chooser(2, 11, i, "random_forest")
            tree.fit([X[j] for j in rows], [y[j] for j in rows])
            trees.append(tree)
        for query in ([0.0] * 4, [2.0, 1.0, -1.0, 0.5], [-2.0, -2.0, 2.0, 2.0]):
            per_tree = [t.predict_proba([query])[0] for t in trees]
            want = [sum(p[c] for p in per_tree) / 3 for c in range(2)]
            for got, expected in zip(model.predict_proba([query])[0], want, strict=True):
                self.assertAlmostEqual(got, expected)

    def test_differs_from_bagged_trees_with_the_same_seed(self):
        X, y = noisy()
        forest = RandomForest(n_estimators=10, max_depth=5, max_features=None, seed=3)
        bagged = BaggedTrees(n_estimators=10, max_depth=5, seed=3)
        forest.fit(X, y)
        bagged.fit(X, y)
        self.assertNotEqual(forest.predict_proba(X), bagged.predict_proba(X))  # own samples

    def test_finds_the_one_informative_feature_even_with_one_feature_per_split(self):
        X, y = informative_second_feature()
        model = RandomForest(n_estimators=15, max_features=1, seed=2)
        model.fit(X, y)
        self.assertEqual(model.predict(X), y)

    def test_separable_clusters_and_probability_rows(self):
        X = [[float(i), 0.0] for i in range(10)] + [[float(100 + i), 0.0] for i in range(10)]
        y = ["a"] * 10 + ["b"] * 10
        model = RandomForest(n_estimators=20, seed=1)
        model.fit(X, y)
        self.assertEqual(model.predict([[-5.0, 0.0], [500.0, 0.0]]), ["a", "b"])
        for row in model.predict_proba(X):
            self.assertAlmostEqual(sum(row), 1.0)

    def test_a_single_feature_still_works(self):
        model = RandomForest(n_estimators=5, seed=1)
        model.fit([[0.0], [1.0], [10.0], [11.0]], ["a", "a", "b", "b"])
        self.assertEqual(model.predict([[0.5], [10.5]]), ["a", "b"])

    def test_a_single_class_predicts_it(self):
        model = RandomForest(n_estimators=4)
        model.fit([[1.0], [2.0]], ["only", "only"])
        self.assertEqual(model.predict([[5.0]]), ["only"])

    def test_wrong_width_at_predict_is_an_error(self):
        model = RandomForest(n_estimators=2)
        model.fit([[0.0, 1.0], [5.0, 6.0]], ["a", "b"])
        with self.assertRaises(ModelError):
            model.predict([[1.0]])


class OutOfBagTests(unittest.TestCase):
    def test_one_tree_scores_its_own_held_out_rows(self):
        X, y = noisy(seed=4, n=50, width=3)
        model = RandomForest(n_estimators=1, max_depth=3, max_features=2, seed=6)
        model.fit(X, y)
        rows = bootstrap_indices(len(X), 6, 0, "random_forest")
        tree = DecisionTreeClassifier(max_depth=3)
        tree._feature_chooser = feature_chooser(2, 6, 0, "random_forest")
        tree.fit([X[i] for i in rows], [y[i] for i in rows])
        held_out = [i for i in range(len(X)) if i not in set(rows)]
        self.assertGreater(len(held_out), 5)
        predicted = tree.predict([X[i] for i in held_out])
        want = sum(p == y[i] for p, i in zip(predicted, held_out, strict=True)) / len(held_out)
        self.assertEqual(model.oob_score_, want)

    def test_is_none_when_every_tree_saw_every_row(self):
        model = RandomForest(n_estimators=3)
        model.fit([[1.0]], ["a"])
        self.assertIsNone(model.oob_score_)

    def test_is_a_fraction_that_tracks_difficulty(self):
        easy = RandomForest(n_estimators=30, seed=1)
        X = [[float(i)] for i in range(30)] + [[float(100 + i)] for i in range(30)]
        easy.fit(X, ["a"] * 30 + ["b"] * 30)
        self.assertEqual(easy.oob_score_, 1.0)
        rng = random.Random(3)
        hard = RandomForest(n_estimators=30, seed=1)
        hard.fit([[rng.random()] for _ in range(80)], [rng.choice("ab") for _ in range(80)])
        self.assertTrue(0.2 < hard.oob_score_ < 0.8)

    def test_an_exact_tie_goes_to_the_first_class(self):
        class Undecided(BaggedTrees):
            def _aligned(self, tree, X):
                return [[0.5, 0.5] for _ in X]

        model = Undecided(n_estimators=6, seed=3)
        X = [[float(i)] for i in range(40)]
        y = ["a" if i % 4 else "b" for i in range(40)]  # 30 a, 10 b
        model.fit(X, y)
        covered = {
            i for i in range(40) if any(i not in bootstrap_indices(40, 3, t) for t in range(6))
        }
        want = sum(y[i] == "a" for i in covered) / len(covered)
        self.assertEqual(model.oob_score_, want)

    def test_bagged_trees_report_it_too(self):
        X, y = noisy(seed=2, n=50, width=2)
        model = BaggedTrees(n_estimators=1, max_depth=2, seed=8)
        model.fit(X, y)
        rows = bootstrap_indices(len(X), 8, 0)
        tree = DecisionTreeClassifier(max_depth=2)
        tree.fit([X[i] for i in rows], [y[i] for i in rows])
        held_out = [i for i in range(len(X)) if i not in set(rows)]
        predicted = tree.predict([X[i] for i in held_out])
        want = sum(p == y[i] for p, i in zip(predicted, held_out, strict=True)) / len(held_out)
        self.assertEqual(model.oob_score_, want)

    def test_a_refit_replaces_the_score(self):
        model = RandomForest(n_estimators=1, seed=1)
        model.fit([[1.0]], ["a"])
        self.assertIsNone(model.oob_score_)
        X, y = noisy(n=40)
        model.fit(X, y)
        self.assertIsNotNone(model.oob_score_)

    def test_bootstrap_default_stream_is_unchanged(self):
        # bagged trees written before the purpose argument existed must keep their samples
        rng = rng_for(5, "bagged_trees", 2)
        self.assertEqual(bootstrap_indices(12, 5, 2), [rng.randrange(12) for _ in range(12)])


class RegressorTests(unittest.TestCase):
    def data(self, n=400, seed=2):
        rng = random.Random(seed)
        X = [[rng.uniform(-3, 3) for _ in range(4)] for _ in range(n)]
        y = [r[0] ** 2 + 2 * r[1] + rng.gauss(0, 0.2) for r in X]
        return X, y

    def test_matches_a_hand_built_forest(self):
        X, y = self.data(n=40)
        model = RandomForestRegressor(n_estimators=3, max_depth=3, max_features=2, seed=5)
        model.fit(X, y)
        trees = []
        for i in range(3):
            rows = bootstrap_indices(len(X), 5, i, "random_forest_regressor")
            tree = DecisionTreeRegressor(max_depth=3)
            tree._feature_chooser = feature_chooser(2, 5, i, "random_forest_regressor")
            tree.fit([X[j] for j in rows], [y[j] for j in rows])
            trees.append(tree)
        queries = [[0.0] * 4, [1.0, 2.0, -1.0, 0.5], [-2.0, 1.0, 2.0, -2.0]]
        for got, query in zip(model.predict(queries), queries, strict=True):
            self.assertAlmostEqual(got, sum(t.predict([query])[0] for t in trees) / 3)

    def test_fits_a_smooth_function_better_than_one_tree(self):
        X, y = self.data()
        single = DecisionTreeRegressor(max_depth=None)
        forest = RandomForestRegressor(n_estimators=40, max_features=0.5, seed=1)
        single.fit(X[:200], y[:200])
        forest.fit(X[:200], y[:200])

        def sse(model):
            return sum((p - t) ** 2 for p, t in zip(model.predict(X[200:]), y[200:], strict=True))

        self.assertLess(sse(forest), sse(single))

    def test_one_tree_oob_r2_by_hand(self):
        X, y = self.data(n=40)
        model = RandomForestRegressor(n_estimators=1, max_depth=3, max_features=2, seed=9)
        model.fit(X, y)
        rows = bootstrap_indices(len(X), 9, 0, "random_forest_regressor")
        tree = DecisionTreeRegressor(max_depth=3)
        tree._feature_chooser = feature_chooser(2, 9, 0, "random_forest_regressor")
        tree.fit([X[i] for i in rows], [y[i] for i in rows])
        held_out = [i for i in range(len(X)) if i not in set(rows)]
        truth = [y[i] for i in held_out]
        predicted = tree.predict([X[i] for i in held_out])
        mean = sum(truth) / len(truth)
        want = 1 - sum((p - t) ** 2 for p, t in zip(predicted, truth, strict=True)) / sum(
            (t - mean) ** 2 for t in truth
        )
        self.assertAlmostEqual(model.oob_score_, want)

    def test_several_trees_average_before_scoring(self):
        X, y = self.data(n=40)
        model = RandomForestRegressor(n_estimators=3, max_depth=3, max_features=2, seed=2)
        model.fit(X, y)
        sums = [0.0] * len(X)
        votes = [0] * len(X)
        for t in range(3):
            rows = bootstrap_indices(len(X), 2, t, "random_forest_regressor")
            tree = DecisionTreeRegressor(max_depth=3)
            tree._feature_chooser = feature_chooser(2, 2, t, "random_forest_regressor")
            tree.fit([X[i] for i in rows], [y[i] for i in rows])
            for i in set(range(len(X))) - set(rows):
                sums[i] += tree.predict([X[i]])[0]
                votes[i] += 1
        scored = [i for i in range(len(X)) if votes[i]]
        self.assertTrue(any(votes[i] > 1 for i in scored))
        mean = sum(y[i] for i in scored) / len(scored)
        want = 1 - sum((y[i] - sums[i] / votes[i]) ** 2 for i in scored) / sum(
            (y[i] - mean) ** 2 for i in scored
        )
        self.assertAlmostEqual(model.oob_score_, want)

    def test_oob_is_none_for_constant_targets_and_tiny_data(self):
        model = RandomForestRegressor(n_estimators=10, seed=1)
        model.fit([[float(i)] for i in range(20)], [3.0] * 20)
        self.assertIsNone(model.oob_score_)
        self.assertEqual(model.predict([[4.0]]), [3.0])
        model.fit([[1.0]], [2.0])
        self.assertIsNone(model.oob_score_)

    def test_same_seed_same_predictions(self):
        X, y = self.data(n=60)
        a, b = (RandomForestRegressor(n_estimators=8, seed=3) for _ in range(2))
        a.fit(X, y)
        b.fit(X, y)
        self.assertEqual(a.predict(X), b.predict(X))
        c = RandomForestRegressor(n_estimators=8, seed=4)
        c.fit(X, y)
        self.assertNotEqual(a.predict(X), c.predict(X))

    def test_predictions_are_means_of_training_targets_so_stay_in_range(self):
        X, y = self.data(n=60)
        model = RandomForestRegressor(n_estimators=10, seed=2)
        model.fit(X, y)
        for value in model.predict([[100.0] * 4, [-100.0] * 4]):
            self.assertTrue(min(y) <= value <= max(y))
            self.assertTrue(math.isfinite(value))


class RunnerTests(unittest.TestCase):
    def table(self):
        rng = random.Random(7)
        rows = []
        for _ in range(80):
            a, b = rng.uniform(-3, 3), rng.uniform(-3, 3)
            rows.append((f"{a:.3f}", f"{b:.3f}", "p" if a + b > 0 else "q"))
        return Table(("a", "b", "label"), tuple(rows))

    def config(self, **overrides):
        fields = {
            "experiment_id": "forest",
            "model": "random_forest",
            "seed": 4,
            "dataset_id": "rows",
            "metrics": ("accuracy",),
            "params": {"n_estimators": 8},
        }
        fields.update(overrides)
        return ExperimentConfig(**fields)

    def test_the_runner_derives_the_seed_and_the_run_reproduces(self):
        table = self.table()
        with Store(":memory:") as store:
            result = run_experiment(store, self.config(), table, "label")
            run = store.get_run(result.run_id)
            self.assertEqual(run.tags["model_seed"], str(derive_seed(4, "model")))
            self.assertGreater(result.metrics["accuracy"], 0.8)
            verdict = check_reproducibility(store, result.run_id, table, "label")
            self.assertEqual(verdict.verdict, "reproduced")

    def test_cross_validation_gives_each_fold_its_own_forest_seed(self):
        table = self.table()
        with Store(":memory:") as store:
            result = run_experiment(store, self.config(cv_folds=3), table, "label")
            self.assertIn("accuracy_std", result.metrics)
            again = run_experiment(
                store, self.config(experiment_id="forest-2", cv_folds=3), table, "label"
            )
            self.assertEqual(result.metrics, again.metrics)

    def test_regressor_runs_with_regression_metrics(self):
        rng = random.Random(1)
        rows = tuple(
            (f"{x:.3f}", f"{3 * x + 1:.3f}") for x in (rng.uniform(0, 5) for _ in range(60))
        )
        table = Table(("x", "y"), rows)
        config = self.config(
            experiment_id="reg", model="random_forest_regressor", metrics=("r2", "mae")
        )
        with Store(":memory:") as store:
            result = run_experiment(store, config, table, "y")
            self.assertGreater(result.metrics["r2"], 0.9)


if __name__ == "__main__":
    unittest.main()
