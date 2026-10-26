import math
import unittest

from modelbench.config import ExperimentConfig
from modelbench.datasets import Table
from modelbench.errors import ModelError, SeedError
from modelbench.models import create_model
from modelbench.models.perceptron import MAX_ITER_LIMIT, Perceptron, epoch_order
from modelbench.reproducibility import check_reproducibility
from modelbench.runner import run_experiment
from modelbench.seeding import derive_seed
from modelbench.store import Store


def clusters(labels=("a", "b", "c"), per=6):
    X, y = [], []
    for c, label in enumerate(labels):
        for i in range(per):
            X.append([10.0 * c + i * 0.1, 5.0 - 3.0 * c + (i % 2) * 0.1])
            y.append(label)
    return X, y


class EpochOrderTests(unittest.TestCase):
    def test_a_permutation_that_repeats_for_the_same_inputs(self):
        order = epoch_order(20, 3, 0)
        self.assertEqual(sorted(order), list(range(20)))
        self.assertEqual(order, epoch_order(20, 3, 0))

    def test_depends_on_seed_and_epoch(self):
        self.assertNotEqual(epoch_order(20, 3, 0), epoch_order(20, 3, 1))
        self.assertNotEqual(epoch_order(20, 3, 0), epoch_order(20, 4, 0))


class HandComputedTests(unittest.TestCase):
    """Two rows, x = 0 (class a) and x = 1 (class b).

    Standardised they are -1 and +1 (mean 0.5, population std 0.5). With seed 0 the first epoch
    visits row 0 then row 1. Row 0 ties at zero and is right (a is first). Row 1 is a mistake:
    w_b = 1, w_a = -1, bias_b = 1, bias_a = -1. The second epoch makes no mistake, so training
    stops after 2 epochs and 4 steps. Summed over the 4 steps the weights are -3 (a) and 3 (b),
    and the biases -3 and 3, so the averages are w = -0.75 / 0.75 and bias = -0.75 / 0.75.
    On the original scale that is coef -1.5 / 1.5 and intercept 0 / 0.
    """

    def setUp(self):
        self.assertEqual(epoch_order(2, 0, 0), [0, 1])  # the premise of the arithmetic above
        self.model = Perceptron(max_iter=20, seed=0)
        self.model.fit([[0.0], [1.0]], ["a", "b"])

    def test_converges_after_two_epochs(self):
        self.assertTrue(self.model.converged_)
        self.assertEqual(self.model.n_iter_, 2)

    def test_averaged_weights_on_the_original_scale(self):
        self.assertAlmostEqual(self.model.coef_[0][0], -1.5)
        self.assertAlmostEqual(self.model.coef_[1][0], 1.5)
        self.assertAlmostEqual(self.model.intercept_[0], 0.0)
        self.assertAlmostEqual(self.model.intercept_[1], 0.0)

    def test_scores_become_softmax_probabilities(self):
        # at x = 1 the scores are -1.5 and 1.5
        p_a, p_b = self.model.predict_proba([[1.0]])[0]
        self.assertAlmostEqual(p_b, 1 / (1 + math.exp(-3)))
        self.assertAlmostEqual(p_a + p_b, 1.0)
        # at x = 0 both scores are 0: a tie goes to the first class, with probability one half
        self.assertEqual(self.model.predict([[0.0]]), ["a"])
        self.assertEqual(self.model.predict_proba([[0.0]])[0], [0.5, 0.5])

    def test_predictions(self):
        self.assertEqual(self.model.predict([[1.0], [5.0], [-5.0]]), ["b", "b", "a"])


class LearningTests(unittest.TestCase):
    def test_separable_classes_are_learned_exactly(self):
        X, y = clusters()
        model = Perceptron(seed=1)
        model.fit(X, y)
        self.assertTrue(model.converged_)
        self.assertEqual(model.predict(X), y)
        self.assertEqual(model.classes_, ("a", "b", "c"))
        self.assertEqual(len(model.coef_), 3)
        self.assertEqual(len(model.coef_[0]), 2)

    def test_xor_is_not_separable_so_it_runs_every_epoch(self):
        model = Perceptron(max_iter=7, seed=0)
        model.fit([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], ["a", "a", "b", "b"])
        self.assertFalse(model.converged_)
        self.assertEqual(model.n_iter_, 7)

    def test_a_constant_feature_is_ignored(self):
        X = [[0.0, 7.0], [1.0, 7.0], [0.1, 7.0], [0.9, 7.0]]
        model = Perceptron(seed=0)
        model.fit(X, ["a", "b", "a", "b"])
        self.assertEqual(model.predict([[0.0, 7.0], [1.0, 7.0]]), ["a", "b"])
        self.assertAlmostEqual(model.coef_[0][1], -model.coef_[1][1])  # no preference either way
        self.assertTrue(all(math.isfinite(v) for row in model.coef_ for v in row))

    def test_same_seed_same_model_and_refit_resets(self):
        X, y = clusters()
        a, b = Perceptron(seed=5), Perceptron(seed=5)
        a.fit(X, y)
        b.fit(X, y)
        self.assertEqual(a.coef_, b.coef_)
        self.assertEqual(a.intercept_, b.intercept_)
        a.fit([[0.0], [1.0]], ["a", "b"])
        self.assertEqual(len(a.coef_[0]), 1)
        self.assertEqual(a.classes_, ("a", "b"))

    def test_probabilities_are_ordered_like_the_scores(self):
        X, y = clusters()
        model = Perceptron(seed=1)
        model.fit(X, y)
        for row, guess in zip(model.predict_proba(X), model.predict(X), strict=True):
            self.assertAlmostEqual(sum(row), 1.0)
            self.assertEqual(model.classes_[row.index(max(row))], guess)

    def test_extreme_scores_do_not_overflow(self):
        model = Perceptron(seed=0)
        model.fit([[0.0], [1.0]], ["a", "b"])
        row = model.predict_proba([[1e9]])[0]
        self.assertEqual(row, [0.0, 1.0])


class ParameterTests(unittest.TestCase):
    def test_validation(self):
        for bad in (True, 0, -1, MAX_ITER_LIMIT + 1, 2.5, "5", None):
            with self.subTest(max_iter=bad), self.assertRaises(ModelError):
                Perceptron(max_iter=bad)
        Perceptron(max_iter=MAX_ITER_LIMIT)
        Perceptron(max_iter=1)
        for bad in (True, 1.5, "1", None, -1):
            with self.subTest(seed=bad), self.assertRaises(SeedError):
                Perceptron(seed=bad)

    def test_registry_and_params_round_trip(self):
        model = create_model("perceptron", max_iter=5, seed=9)
        self.assertEqual(model.get_params(), {"max_iter": 5, "seed": 9})
        self.assertEqual(Perceptron().get_params(), {"max_iter": 20, "seed": 0})
        with self.assertRaises(ModelError):
            create_model("perceptron", learning_rate=0.1)

    def test_weights_need_a_fitted_model(self):
        model = Perceptron()
        with self.assertRaises(ModelError):
            _ = model.coef_
        with self.assertRaises(ModelError):
            _ = model.intercept_


class RunnerIntegrationTests(unittest.TestCase):
    def test_the_runner_seeds_the_model_and_the_run_reproduces(self):
        X, y = clusters(per=8)
        table = Table(
            ("a", "b", "label"), tuple((str(r[0]), str(r[1]), t) for r, t in zip(X, y, strict=True))
        )
        config = ExperimentConfig(
            experiment_id="perc", model="perceptron", seed=4, dataset_id="clusters",
            metrics=("accuracy", "f1"), params={"max_iter": 10}, stratify=True,
        )  # fmt: skip
        env = {"python": "3.12.1", "lib:numpy": None}
        with Store(":memory:") as store:
            result = run_experiment(store, config, table, "label", environment=env)
            self.assertEqual(result.metrics["accuracy"], 1.0)
            tags = store.get_run(result.run_id).tags
            self.assertEqual(tags["model_seed"], str(derive_seed(4, "model")))
            report = check_reproducibility(store, result.run_id, table, "label", environment=env)
            self.assertTrue(report.reproduced, report.detail)


if __name__ == "__main__":
    unittest.main()
