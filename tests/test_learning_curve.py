import math
import unittest
from unittest import mock

from modelbench.config import ExperimentConfig
from modelbench.datasets import Table, kfold_indices, split_indices
from modelbench.errors import AnalysisError, MetricError
from modelbench.learning_curve import (
    DEFAULT_FRACTIONS,
    MAX_POINTS,
    check_fractions,
    learning_curve,
)
from modelbench.models import create_model
from modelbench.runner import build_model, run_experiment, score_classifier
from modelbench.seeding import rng_for
from modelbench.store import Store

ENV = {"python": "3.12.1", "lib:numpy": None}


def squares(n=20):
    return Table(("x", "y"), tuple((str(i), repr(float(i * i))) for i in range(n)))


def blobs(n=20):
    rows = [(f"{i * 0.1:.1f}", "0", "a") for i in range(n)]
    rows += [(f"{10 + i * 0.1:.1f}", "5", "b") for i in range(n)]
    return Table(("x1", "x2", "label"), tuple(rows))


def mean_config(**overrides):
    fields = {
        "experiment_id": "lc",
        "model": "mean",
        "seed": 3,
        "dataset_id": "squares",
        "metrics": ("mse",),
        "params": {},
        "test_fraction": 0.25,
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


def knn_config(**overrides):
    fields = {
        "experiment_id": "lc",
        "model": "knn",
        "seed": 3,
        "dataset_id": "blobs",
        "metrics": ("accuracy",),
        "params": {"k": 1},
        "stratify": True,
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


class HandCheckedTests(unittest.TestCase):
    def test_a_mean_model_scores_are_the_subset_variance_and_the_test_error(self):
        table = squares()
        y = [float(i * i) for i in range(20)]
        train, test = split_indices(20, 0.25, 3)  # 15 training rows, 5 test rows
        order = list(range(len(train)))
        rng_for(3, "learning-curve", 0).shuffle(order)

        curve = learning_curve(mean_config(), table, "y", fractions=(0.2, 0.6, 1.0))

        self.assertEqual((curve.metric, curve.folds), ("mse", 1))
        self.assertEqual([p.n_train for p in curve.points], [3, 9, 15])
        self.assertEqual([p.fraction for p in curve.points], [0.2, 0.6, 1.0])
        for point, size in zip(curve.points, (3, 9, 15), strict=True):
            subset = [y[train[i]] for i in order[:size]]
            centre = math.fsum(subset) / size
            train_mse = math.fsum((v - centre) ** 2 for v in subset) / size
            test_mse = math.fsum((y[i] - centre) ** 2 for i in test) / len(test)
            self.assertAlmostEqual(point.train_score, train_mse, places=6)
            self.assertAlmostEqual(point.test_score, test_mse, places=6)
            self.assertEqual(point.test_std, 0.0)  # one split: nothing to spread over

    def test_every_fold_orders_its_own_training_rows(self):
        table = squares()
        y = [float(i * i) for i in range(20)]
        folds = kfold_indices(20, 4, 3)
        expected = []
        for number, (train, test) in enumerate(folds):
            order = list(range(len(train)))
            rng_for(3, "learning-curve", number).shuffle(order)
            subset = [y[train[i]] for i in order[: round(0.4 * len(train))]]
            centre = math.fsum(subset) / len(subset)
            expected.append(math.fsum((y[i] - centre) ** 2 for i in test) / len(test))
        curve = learning_curve(mean_config(cv_folds=4), table, "y", fractions=(0.4,))
        self.assertAlmostEqual(curve.points[0].test_score, math.fsum(expected) / 4, places=6)
        self.assertGreater(max(expected) - min(expected), 1.0)  # the folds really differ

    def test_each_fold_builds_its_own_model(self):
        seen = []

        def spy(config, fold):
            seen.append(fold)
            return build_model(config, fold)

        with mock.patch("modelbench.learning_curve.build_model", spy):
            learning_curve(mean_config(cv_folds=3), squares(), "y", fractions=(0.5, 1.0))
            self.assertEqual(seen, [0, 0, 1, 1, 2, 2])  # fold by fold, a fresh model per size
            seen.clear()
            learning_curve(mean_config(), squares(), "y", fractions=(0.5, 1.0))
            self.assertEqual(seen, [None, None])

    def test_the_last_point_is_what_a_normal_run_scores(self):
        table = squares()
        curve = learning_curve(mean_config(), table, "y")
        with Store(":memory:") as store:
            result = run_experiment(store, mean_config(), table, "y", environment=ENV)
        self.assertEqual([p.fraction for p in curve.points], list(DEFAULT_FRACTIONS))
        self.assertAlmostEqual(curve.points[-1].test_score, result.metrics["mse"], places=9)
        self.assertEqual(curve.points[-1].n_train, result.n_train)

    def test_cross_validation_averages_the_folds(self):
        table = squares()
        config = mean_config(cv_folds=4)
        curve = learning_curve(config, table, "y", fractions=(0.5, 1.0))
        with Store(":memory:") as store:
            result = run_experiment(store, config, table, "y", environment=ENV)
        self.assertEqual(curve.folds, 4)
        self.assertAlmostEqual(curve.points[-1].test_score, result.metrics["mse"], places=9)
        self.assertAlmostEqual(curve.points[-1].test_std, result.metrics["mse_std"], places=9)
        self.assertEqual(curve.points[-1].n_train, result.n_train)

    def test_a_one_nearest_neighbour_always_knows_its_own_training_rows(self):
        curve = learning_curve(knn_config(), blobs(), "label")
        self.assertEqual([p.train_score for p in curve.points], [1.0] * 5)
        self.assertEqual(curve.points[-1].test_score, 1.0)
        self.assertEqual([p.n_train for p in curve.points], [6, 13, 19, 26, 32])


class OneClassSubsetTests(unittest.TestCase):
    """A small training subset can hold a single class; metrics that need two must say so."""

    def table(self):
        rows = [(f"{i * 0.01:.2f}", "b" if i < 4 else "a") for i in range(40)]
        return Table(("x", "label"), tuple(rows))

    def test_a_metric_that_needs_two_classes_gives_a_clear_error(self):
        for metric in ("roc_auc", "average_precision", "brier_score", "ece"):
            config = ExperimentConfig(
                "e", "knn", 1, "d", (metric,), {"k": 1}, test_fraction=0.5, stratify=True
            )
            with self.subTest(metric), self.assertRaises(AnalysisError) as caught:
                learning_curve(config, self.table(), "label", fractions=[0.05, 0.5, 1.0])
            message = str(caught.exception)
            self.assertIn("fraction 0.05", message)
            self.assertIn(
                f"{metric} needs exactly two classes in the training data, got 1", message
            )

    def test_scoring_directly_raises_a_library_error_not_a_value_error(self):
        model = create_model("majority")
        model.fit([[0.0], [1.0]], ["a", "a"])
        with self.assertRaisesRegex(MetricError, "needs exactly two classes"):
            score_classifier(["roc_auc"], model, [[0.0]], ["a"], ["a"])


class OptionsAndErrorsTests(unittest.TestCase):
    def test_repeatable_and_seed_dependent(self):
        table = squares()
        first = learning_curve(mean_config(), table, "y", fractions=(0.2, 0.6))
        self.assertEqual(first, learning_curve(mean_config(), table, "y", fractions=(0.2, 0.6)))
        other = learning_curve(mean_config(seed=4), table, "y", fractions=(0.2, 0.6))
        self.assertNotEqual(first, other)

    def test_fraction_validation(self):
        for bad in ([], [0], [-0.1], [1.01], [0.5, 0.5], [0.6, 0.4], [True], ["0.5"],
                    [float("nan")], [float("inf")], [0.5] * (MAX_POINTS + 1)):  # fmt: skip
            with self.subTest(fractions=bad), self.assertRaises(AnalysisError):
                check_fractions(bad)
        self.assertEqual(check_fractions([1]), (1.0,))
        self.assertEqual(check_fractions([i / 20 for i in range(1, 21)])[-1], 1.0)
        self.assertEqual(len(check_fractions([i / 20 for i in range(1, 21)])), MAX_POINTS)

    def test_the_metric_must_be_one_the_config_has(self):
        with self.assertRaisesRegex(AnalysisError, "not one of the config's metrics"):
            learning_curve(mean_config(), squares(), "y", metric="mae")
        curve = learning_curve(
            mean_config(metrics=("mse", "mae")), squares(), "y", metric="mae", fractions=(1.0,)
        )
        self.assertEqual(curve.metric, "mae")

    def test_a_model_that_cannot_train_on_a_small_subset_is_reported_with_its_size(self):
        config = knn_config(params={"k": 5})
        with self.assertRaisesRegex(AnalysisError, r"at fraction 0\.1 \(3 training rows\)"):
            learning_curve(config, blobs(), "label", fractions=(0.1, 1.0))


if __name__ == "__main__":
    unittest.main()
