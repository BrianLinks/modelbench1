import random
import unittest

from modelbench.config import ExperimentConfig
from modelbench.datasets import Table
from modelbench.errors import ModelError, SeedError
from modelbench.models import create_model
from modelbench.models.bagged_trees import MAX_ESTIMATORS, BaggedTrees, bootstrap_indices
from modelbench.models.decision_tree import DecisionTreeClassifier
from modelbench.reproducibility import check_reproducibility
from modelbench.runner import run_experiment
from modelbench.seeding import derive_seed
from modelbench.store import Store


def two_clusters(n=10):
    X = [[float(i)] for i in range(n)] + [[float(100 + i)] for i in range(n)]
    return X, ["a"] * n + ["b"] * n


def noisy(seed=1, n=60):
    rng = random.Random(seed)
    X = [[rng.uniform(-3, 3), rng.uniform(-3, 3)] for _ in range(n)]
    y = ["p" if a + b + rng.uniform(-1.5, 1.5) > 0 else "q" for a, b in X]
    return X, y


class BootstrapTests(unittest.TestCase):
    def test_draws_n_rows_with_replacement_and_is_repeatable(self):
        rows = bootstrap_indices(30, 5, 0)
        self.assertEqual(len(rows), 30)
        self.assertTrue(all(0 <= i < 30 for i in rows))
        self.assertLess(len(set(rows)), 30)  # 30 draws from 30 almost surely repeat a row
        self.assertEqual(rows, bootstrap_indices(30, 5, 0))

    def test_each_tree_and_each_seed_gets_its_own_sample(self):
        self.assertNotEqual(bootstrap_indices(30, 5, 0), bootstrap_indices(30, 5, 1))
        self.assertNotEqual(bootstrap_indices(30, 5, 0), bootstrap_indices(30, 6, 0))

    def test_the_average_sample_covers_about_two_thirds_of_the_rows(self):
        covered = [len(set(bootstrap_indices(200, 3, i))) / 200 for i in range(40)]
        self.assertAlmostEqual(sum(covered) / 40, 1 - 1 / 2.718281828, delta=0.02)


class AveragingTests(unittest.TestCase):
    def test_matches_a_hand_built_average_of_trees(self):
        X, y = noisy()
        model = BaggedTrees(n_estimators=4, max_depth=2, seed=9)
        model.fit(X, y)
        trees = []
        for i in range(4):
            rows = bootstrap_indices(len(X), 9, i)
            tree = DecisionTreeClassifier(max_depth=2)
            tree.fit([X[j] for j in rows], [y[j] for j in rows])
            trees.append(tree)
        queries = [[0.0, 0.0], [2.0, 2.0], [-2.0, -2.0], [1.0, -1.0]]
        for got, query in zip(model.predict_proba(queries), queries, strict=True):
            per_tree = [t.predict_proba([query])[0] for t in trees]
            want = [sum(p[c] for p in per_tree) / 4 for c in range(2)]
            for g, w in zip(got, want, strict=True):
                self.assertAlmostEqual(g, w)

    def test_probabilities_are_multiples_of_one_over_n_for_pure_trees(self):
        X, y = two_clusters()
        model = BaggedTrees(n_estimators=8, seed=2)
        model.fit(X, y)
        for row in model.predict_proba([[3.0], [104.0], [50.0]]):
            self.assertAlmostEqual(sum(row), 1.0)
            for share in row:
                self.assertAlmostEqual(share * 8, round(share * 8))

    def test_separable_clusters_are_classified_perfectly(self):
        X, y = two_clusters()
        model = BaggedTrees(n_estimators=10, seed=1)
        model.fit(X, y)
        self.assertEqual(model.predict(X), y)
        self.assertEqual(model.predict([[-50.0], [500.0]]), ["a", "b"])

    def test_a_tree_that_misses_a_class_counts_it_as_zero(self):
        X = [[float(i)] for i in range(5)] + [[50.0]]
        y = ["a"] * 5 + ["b"]
        # find a seed whose first bootstrap sample never draws the lone "b" row
        seed = next(s for s in range(200) if 5 not in bootstrap_indices(6, s, 0))
        model = BaggedTrees(n_estimators=1, seed=seed)
        model.fit(X, y)
        self.assertEqual(model.classes_, ("a", "b"))
        self.assertEqual(model.predict_proba([[50.0]]), [[1.0, 0.0]])

    def test_a_missing_middle_class_leaves_the_others_in_their_own_columns(self):
        X = [[0.0], [1.0], [2.0], [50.0], [100.0], [101.0]]
        y = ["a", "a", "a", "b", "c", "c"]
        seed = next(s for s in range(300) if 3 not in bootstrap_indices(6, s, 0))
        model = BaggedTrees(n_estimators=1, seed=seed)
        model.fit(X, y)
        self.assertEqual(model.classes_, ("a", "b", "c"))
        self.assertEqual(model.predict_proba([[100.0]]), [[0.0, 0.0, 1.0]])
        self.assertEqual(model.predict_proba([[0.0]]), [[1.0, 0.0, 0.0]])

    def test_a_rare_class_still_shows_up_when_some_trees_see_it(self):
        X = [[float(i)] for i in range(8)] + [[50.0]]
        y = ["a"] * 8 + ["b"]
        model = BaggedTrees(n_estimators=40, seed=4)
        model.fit(X, y)
        share_b = model.predict_proba([[50.0]])[0][1]
        seen = sum(8 in bootstrap_indices(9, 4, i) for i in range(40))
        self.assertAlmostEqual(share_b, seen / 40)  # every tree that saw it predicts b there
        self.assertGreater(share_b, 0.0)
        self.assertLess(share_b, 1.0)

    def test_integer_and_string_labels(self):
        X, _ = two_clusters()
        model = BaggedTrees(n_estimators=3)
        model.fit(X, [2] * 10 + [10] * 10)
        self.assertEqual(model.classes_, (2, 10))
        self.assertEqual(model.predict([[0.0], [105.0]]), [2, 10])

    def test_averaging_beats_a_single_deep_tree_on_noisy_held_out_data(self):
        X, y = noisy(seed=3, n=300)
        train_X, test_X, train_y, test_y = X[:150], X[150:], y[:150], y[150:]
        single = DecisionTreeClassifier(max_depth=None)
        single.fit(train_X, train_y)
        bagged = BaggedTrees(n_estimators=40, max_depth=None, seed=1)
        bagged.fit(train_X, train_y)

        def accuracy(model):
            hits = sum(p == t for p, t in zip(model.predict(test_X), test_y, strict=True))
            return hits / len(test_y)

        self.assertGreater(accuracy(bagged), accuracy(single))


class DeterminismTests(unittest.TestCase):
    def test_same_seed_same_predictions_whatever_happened_before(self):
        X, y = noisy()
        first = BaggedTrees(n_estimators=10, seed=3)
        first.fit(X, y)
        second = BaggedTrees(n_estimators=10, seed=3)
        second.fit(X, y)
        random.seed(99)  # the global generator must not matter
        queries = [[a / 2, b / 2] for a in range(-6, 7) for b in range(-6, 7)]
        self.assertEqual(first.predict_proba(queries), second.predict_proba(queries))

    def test_refitting_gives_the_same_model(self):
        X, y = noisy()
        model = BaggedTrees(n_estimators=6, seed=3)
        model.fit(X, y)
        before = model.predict_proba(X)
        model.fit(X, y)
        self.assertEqual(model.predict_proba(X), before)

    def test_different_seeds_can_give_different_models(self):
        X, y = noisy()
        a = BaggedTrees(n_estimators=10, seed=1)
        a.fit(X, y)
        b = BaggedTrees(n_estimators=10, seed=2)
        b.fit(X, y)
        self.assertNotEqual(a.predict_proba(X), b.predict_proba(X))


class ParameterTests(unittest.TestCase):
    def test_defaults_and_round_trip(self):
        model = create_model("bagged_trees")
        self.assertEqual(
            model.get_params(),
            {
                "n_estimators": 25,
                "max_depth": 5,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "seed": 0,
            },
        )
        rebuilt = create_model("bagged_trees", **model.get_params())
        self.assertEqual(rebuilt.get_params(), model.get_params())

    def test_tree_parameters_reach_the_trees(self):
        X, y = noisy()
        stump = BaggedTrees(n_estimators=3, max_depth=1)
        stump.fit(X, y)
        self.assertTrue(all(t.depth_ == 1 for t in stump._trees))
        leafy = BaggedTrees(n_estimators=3, max_depth=None, min_samples_leaf=20)
        leafy.fit(X, y)
        self.assertTrue(all(t.n_leaves_ <= 3 for t in leafy._trees))

    def test_invalid_parameters_fail_at_construction(self):
        for params in (
            {"n_estimators": 0},
            {"n_estimators": 2.5},
            {"n_estimators": True},
            {"n_estimators": MAX_ESTIMATORS + 1},
            {"max_depth": 0},
            {"min_samples_split": 1},
            {"min_samples_leaf": 0},
        ):
            with self.subTest(params), self.assertRaises(ModelError):
                BaggedTrees(**params)
        for seed in (-1, 2**32, True, 1.5):
            with self.subTest(seed=seed), self.assertRaises(SeedError):
                BaggedTrees(seed=seed)
        BaggedTrees(n_estimators=MAX_ESTIMATORS)

    def test_unknown_parameters_and_unfitted_use(self):
        with self.assertRaises(ModelError):
            create_model("bagged_trees", k=3)
        with self.assertRaisesRegex(ModelError, "not been fitted"):
            BaggedTrees().predict_proba([[1.0]])

    def test_wrong_feature_count(self):
        X, y = two_clusters()
        model = BaggedTrees(n_estimators=2)
        model.fit(X, y)
        with self.assertRaisesRegex(ModelError, "1 features"):
            model.predict([[1.0, 2.0]])


class RunnerIntegrationTests(unittest.TestCase):
    def test_the_runner_seeds_the_model_and_the_run_reproduces(self):
        X, y = noisy(n=40)
        table = Table(
            ("a", "b", "label"), tuple((str(r[0]), str(r[1]), t) for r, t in zip(X, y, strict=True))
        )
        config = ExperimentConfig(
            experiment_id="bag", model="bagged_trees", seed=5, dataset_id="noisy",
            metrics=("accuracy",), params={"n_estimators": 5, "max_depth": 3},
        )  # fmt: skip
        env = {"python": "3.12.1", "lib:numpy": None}
        with Store(":memory:") as store:
            result = run_experiment(store, config, table, "label", environment=env)
            tags = store.get_run(result.run_id).tags
            self.assertEqual(tags["model_seed"], str(derive_seed(5, "model")))
            report = check_reproducibility(store, result.run_id, table, "label", environment=env)
            self.assertTrue(report.reproduced, report.detail)


if __name__ == "__main__":
    unittest.main()
