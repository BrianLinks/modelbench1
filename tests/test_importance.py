import math
import unittest

from modelbench.config import ExperimentConfig
from modelbench.datasets import Table, kfold_indices, make_version, split_indices
from modelbench.errors import AnalysisError
from modelbench.importance import MAX_REPEATS, permutation_importance
from modelbench.preprocess import PreprocessSpec
from modelbench.runner import run_experiment
from modelbench.seeding import rng_for
from modelbench.store import Store

ENV = {"python": "3.12.1", "lib:numpy": None}


def line_table(n=40):
    """y = 3 * x1 + 2 exactly; x2 is unrelated to y."""
    rows = [(str(i), str((i * 7) % 11), repr(3.0 * i + 2.0)) for i in range(n)]
    return Table(("x1", "x2", "y"), tuple(rows))


def blobs_table():
    rows = [(f"{i * 0.1:.1f}", "0", "a") for i in range(20)]
    rows += [(f"{10 + i * 0.1:.1f}", "0", "b") for i in range(20)]
    return Table(("x1", "x2", "label"), tuple(rows))


def ridge_config(**overrides):
    fields = {
        "experiment_id": "imp",
        "model": "linear_regression",
        "seed": 5,
        "dataset_id": "line",
        "metrics": ("mse", "r2"),
        "params": {},
        "test_fraction": 0.25,
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


def knn_config(**overrides):
    fields = {
        "experiment_id": "imp",
        "model": "knn",
        "seed": 5,
        "dataset_id": "blobs",
        "metrics": ("accuracy",),
        "params": {"k": 3},
        "stratify": True,
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


class ImportanceTestCase(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)

    def run_it(self, config, table, label):
        return run_experiment(self.store, config, table, label, environment=ENV).run_id

    def importance(self, run_id, table, label, **kwargs):
        return permutation_importance(self.store, run_id, table, label, **kwargs)


class HandCheckedTests(ImportanceTestCase):
    def test_the_importance_of_a_used_feature_is_the_error_its_shuffle_causes(self):
        table = line_table()
        run_id = self.run_it(ridge_config(), table, "y")
        result = self.importance(run_id, table, "y", metric="mse", repeats=3, seed=9)

        # The model fits y = 3 * x1 + 2 exactly, so the baseline mse is 0 and shuffling x1 among
        # the test rows gives mean((3 * (x1[shuffled] - x1))**2). Re-derive that here.
        _, test = split_indices(40, 0.25, 5)
        x1 = [float(i) for i in test]
        expected = []
        for repeat in range(3):
            order = list(range(len(test)))
            rng_for(9, "permutation", 0, 0, repeat).shuffle(order)
            expected.append(
                math.fsum((3 * (x1[order[i]] - x1[i])) ** 2 for i in range(len(test))) / len(test)
            )
        mean = math.fsum(expected) / 3
        std = math.sqrt(math.fsum((v - mean) ** 2 for v in expected) / 3)

        by_name = {f.feature: f for f in result.features}
        self.assertAlmostEqual(by_name["x1"].importance, mean, places=4)
        self.assertAlmostEqual(by_name["x1"].std, std, places=4)
        self.assertAlmostEqual(by_name["x2"].importance, 0.0, places=6)  # the model ignores it
        self.assertAlmostEqual(by_name["x2"].std, 0.0, places=6)
        self.assertEqual([f.feature for f in result.features], ["x1", "x2"])
        self.assertEqual((result.metric, result.repeats, result.run_id), ("mse", 3, run_id))
        self.assertFalse(result.higher_is_better)
        self.assertAlmostEqual(result.baseline, 0.0, places=6)

    def test_higher_is_better_metrics_report_the_drop_as_positive(self):
        table = line_table()
        run_id = self.run_it(ridge_config(), table, "y")
        result = self.importance(run_id, table, "y", metric="r2")
        self.assertTrue(result.higher_is_better)
        by_name = {f.feature: f.importance for f in result.features}
        self.assertGreater(by_name["x1"], 0.5)
        self.assertAlmostEqual(by_name["x2"], 0.0, places=6)
        self.assertAlmostEqual(result.baseline, 1.0, places=6)

    def test_a_classifier_that_relies_on_one_feature(self):
        table = blobs_table()
        run_id = self.run_it(knn_config(), table, "label")
        result = self.importance(run_id, table, "label", repeats=10)
        by_name = {f.feature: f for f in result.features}
        self.assertEqual(by_name["x2"].importance, 0.0)  # a constant column cannot matter
        self.assertEqual(by_name["x2"].std, 0.0)
        self.assertGreater(by_name["x1"].importance, 0.2)
        self.assertEqual(result.features[0].feature, "x1")
        self.assertEqual(result.metric, "accuracy")  # the run's first metric by default
        self.assertEqual(result.baseline, 1.0)

    def test_ties_are_ordered_by_name(self):
        rows = [(f"{i * 0.1:.1f}", "0", "0", "a") for i in range(20)]
        rows += [(f"{10 + i * 0.1:.1f}", "0", "0", "b") for i in range(20)]
        table = Table(("x1", "ba", "ab", "label"), tuple(rows))
        run_id = self.run_it(knn_config(), table, "label")
        result = self.importance(run_id, table, "label")
        self.assertEqual([f.feature for f in result.features], ["x1", "ab", "ba"])

    def test_the_run_must_be_rebuilt_from_the_columns_it_used(self):
        rows = [(f"{(i % 5) * 0.1:.1f}", "0", "a") for i in range(20)]
        rows += [(f"{(i % 5) * 0.1:.1f}", "10", "b") for i in range(20)]
        table = Table(("x1", "x2", "label"), tuple(rows))
        run_id = self.run_it(knn_config(), table, "label")
        self.assertEqual(self.importance(run_id, table, "label").baseline, 1.0)
        with self.assertRaisesRegex(AnalysisError, "does not give the predictions"):
            self.importance(run_id, table, "label", feature_columns=["x1"])  # a different model


class RepeatabilityTests(ImportanceTestCase):
    def test_same_inputs_same_answer_and_the_seed_matters(self):
        table = line_table()
        run_id = self.run_it(ridge_config(), table, "y")
        first = self.importance(run_id, table, "y", metric="mse", seed=1)
        again = self.importance(run_id, table, "y", metric="mse", seed=1)
        other = self.importance(run_id, table, "y", metric="mse", seed=2)
        self.assertEqual(first, again)
        self.assertNotEqual(first.features[0].importance, other.features[0].importance)

    def test_a_seed_of_zero_is_a_seed_not_a_request_for_the_default(self):
        table = line_table()
        run_id = self.run_it(ridge_config(), table, "y")
        zero = self.importance(run_id, table, "y", metric="mse", seed=0)
        self.assertNotEqual(zero, self.importance(run_id, table, "y", metric="mse"))
        self.assertEqual(zero, self.importance(run_id, table, "y", metric="mse", seed=0))

    def test_every_fold_is_shuffled_with_its_own_generator(self):
        table = line_table()
        run_id = self.run_it(ridge_config(cv_folds=3), table, "y")
        result = self.importance(run_id, table, "y", metric="mse", repeats=1, seed=4)
        expected = []
        for number, (_, test) in enumerate(kfold_indices(40, 3, 5)):
            x1 = [float(i) for i in test]
            order = list(range(len(test)))
            rng_for(4, "permutation", number, 0, 0).shuffle(order)
            expected.append(
                math.fsum((3 * (x1[order[i]] - x1[i])) ** 2 for i in range(len(test))) / len(test)
            )
        by_name = {f.feature: f.importance for f in result.features}
        self.assertAlmostEqual(by_name["x1"], math.fsum(expected) / 3, places=4)

    def test_folds_with_different_one_hot_columns_cannot_be_combined(self):
        rows = [
            (str(i % 7), "rare" if i == 0 else "common", "a" if i % 2 else "b") for i in range(30)
        ]
        table = Table(("n", "c", "label"), tuple(rows))
        config = knn_config(
            cv_folds=3, stratify=False, preprocess=PreprocessSpec(categorical=("c",))
        )
        run_id = self.run_it(config, table, "label")  # the run itself is fine
        with self.assertRaisesRegex(AnalysisError, "different input features"):
            self.importance(run_id, table, "label")

    def test_the_default_seed_is_the_runs_seed(self):
        table = line_table()
        run_id = self.run_it(ridge_config(), table, "y")
        self.assertEqual(
            self.importance(run_id, table, "y"), self.importance(run_id, table, "y", seed=5)
        )

    def test_cross_validated_runs_are_analysed_over_every_fold(self):
        table = line_table()
        run_id = self.run_it(ridge_config(cv_folds=4), table, "y")
        result = self.importance(run_id, table, "y", metric="mse", repeats=2)
        self.assertEqual(self.store.final_metrics(run_id)["mse"], result.baseline)
        by_name = {f.feature: f.importance for f in result.features}
        self.assertGreater(by_name["x1"], 100.0)  # shuffling the real signal costs a lot
        self.assertAlmostEqual(by_name["x2"], 0.0, places=6)


class RefusalTests(ImportanceTestCase):
    def setUp(self):
        super().setUp()
        self.table = line_table()
        self.run_id = self.run_it(ridge_config(), self.table, "y")

    def test_bad_options(self):
        for bad in (0, -1, MAX_REPEATS + 1, True, 2.5, "3"):
            with self.subTest(repeats=bad), self.assertRaisesRegex(AnalysisError, "repeats"):
                self.importance(self.run_id, self.table, "y", repeats=bad)
        self.importance(self.run_id, self.table, "y", repeats=MAX_REPEATS)
        with self.assertRaisesRegex(AnalysisError, "not one of the run's metrics"):
            self.importance(self.run_id, self.table, "y", metric="mae")

    def test_only_completed_runs(self):
        dataset = make_version("line", self.table, "y")
        failed = self.store.create_run(ridge_config(), dataset, ENV)
        self.store.fail_run(failed, "boom")
        with self.assertRaisesRegex(AnalysisError, "failed"):
            self.importance(failed, self.table, "y")

    def test_the_data_must_be_the_runs_dataset(self):
        other = Table(("x1", "x2", "y"), (*line_table().rows[:-1], ("1", "1", "1.0")))
        with self.assertRaisesRegex(AnalysisError, "not the dataset version"):
            self.importance(self.run_id, other, "y")

    def test_a_refit_that_does_not_match_the_run_is_refused(self):
        dataset = make_version("line", self.table, "y")
        unhashed = self.store.create_run(ridge_config(), dataset, ENV)
        self.store.log_metrics(unhashed, {"mse": 123.0, "r2": 0.5})
        self.store.finish_run(unhashed)
        with self.assertRaisesRegex(AnalysisError, "no predictions hash"):
            self.importance(unhashed, self.table, "y", metric="mse")
        liar = self.store.create_run(ridge_config(), dataset, ENV)
        self.store.log_metrics(liar, {"mse": 123.0, "r2": 0.5})
        self.store.set_tag(liar, "predictions_sha256", "0" * 64)
        self.store.finish_run(liar)
        with self.assertRaisesRegex(AnalysisError, "does not give the predictions"):
            self.importance(liar, self.table, "y", metric="mse")

    def test_unknown_run(self):
        with self.assertRaises(Exception) as caught:
            self.importance("run-999999", self.table, "y")
        self.assertIn("run-999999", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
