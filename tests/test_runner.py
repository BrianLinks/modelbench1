import hashlib
import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from modelbench import confusion
from modelbench.config import ExperimentConfig
from modelbench.datasets import Table, kfold_indices, make_version, split_indices
from modelbench.errors import DatasetError, ModelError, RunnerError, StoreError
from modelbench.models import REGRESSORS, create_model
from modelbench.models.knn import KNNClassifier
from modelbench.models.mean import MeanRegressor
from modelbench.reproducibility import check_reproducibility
from modelbench.runner import _check_metrics, _mean_and_std, columns_signature, run_experiment
from modelbench.seeding import derive_seed
from modelbench.store import Store

ENV = {"python": "3.12.1", "lib:numpy": None}


def blobs(n_a=20, n_b=20):
    """Two well-separated clusters: class a near the origin, class b far away."""
    rows = [(f"{i * 0.1:.1f}", "0", "a") for i in range(n_a)]
    rows += [(f"{10 + i * 0.1:.1f}", "5", "b") for i in range(n_b)]
    return Table(("x1", "x2", "label"), tuple(rows))


def three_class_table():
    rows = []
    for label, centre in (("a", 0), ("b", 10), ("c", 20)):
        rows += [(f"{centre + i * 0.1:.1f}", "0", label) for i in range(10)]
    return Table(("x1", "x2", "label"), tuple(rows))


def make_config(**overrides):
    fields = {
        "experiment_id": "exp-a",
        "model": "knn",
        "seed": 7,
        "dataset_id": "blobs",
        "metrics": ("accuracy",),
        "params": {"k": 3},
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


def expected_split(config, table):
    labels = table.column("label")
    return split_indices(
        len(table), config.test_fraction, config.seed, labels if config.stratify else None
    )


class RunnerTestCase(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)

    def run_it(self, config=None, table=None, **kwargs):
        kwargs.setdefault("environment", ENV)
        return run_experiment(
            self.store, config or make_config(), table or blobs(), "label", **kwargs
        )

    def assertNoRunsCreated(self):
        self.assertEqual(self.store.list_runs(), [])


class RecordingTests(RunnerTestCase):
    def test_separated_clusters_score_perfectly_and_everything_is_recorded(self):
        config = make_config(
            metrics=("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"),
            stratify=True,
        )
        result = self.run_it(config)
        for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            self.assertEqual(result.metrics[name], 1.0, name)
        self.assertAlmostEqual(result.metrics["log_loss"], 0.0, places=9)

        run = self.store.get_run(result.run_id)
        self.assertEqual(run.status, "completed")
        self.assertEqual(run.experiment_id, "exp-a")
        self.assertEqual(run.seed, 7)
        self.assertEqual(run.params, {"k": 3})
        self.assertEqual(run.config_hash, config.config_hash())
        self.assertEqual(run.dataset_version, make_version("blobs", blobs()).version)
        self.assertEqual(run.environment, ENV)
        self.assertEqual(self.store.final_metrics(result.run_id), result.metrics)
        self.assertEqual((result.n_train, result.n_test), (32, 8))
        self.assertEqual(run.tags["n_train"], "32")
        self.assertEqual(run.tags["n_test"], "8")
        self.assertEqual(run.tags["predictions_sha256"], result.predictions_sha256)
        self.assertFalse(result.reused)

    def test_split_sizes_follow_the_config(self):
        result = self.run_it(make_config(test_fraction=0.25))
        self.assertEqual((result.n_train, result.n_test), (30, 10))

    def test_features_can_be_restricted_to_chosen_columns(self):
        result = self.run_it(feature_columns=["x1"])
        self.assertEqual(result.metrics["accuracy"], 1.0)

    def test_numeric_formats_are_accepted(self):
        table = Table(
            ("x", "label"),
            (("1e0", "a"), (" 2 ", "a"), ("+3.5", "a"), (".25", "b"), ("-7", "b"), ("100.", "b")),
        )
        result = self.run_it(make_config(test_fraction=0.5, params={"k": 1}), table)
        self.assertEqual(result.n_train + result.n_test, 6)


class IndependentKnownAnswerTests(RunnerTestCase):
    """The majority model's scores can be worked out by hand from the split labels alone."""

    def setUp(self):
        super().setUp()
        self.table = blobs(n_a=30, n_b=10)  # 30 a, 10 b
        self.config = make_config(
            model="majority",
            params={},
            metrics=("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"),
        )
        train, test = expected_split(self.config, self.table)
        labels = self.table.column("label")
        self.train_counts = Counter(labels[i] for i in train)
        self.test_counts = Counter(labels[i] for i in test)
        self.n_train, self.n_test = len(train), len(test)
        self.assertGreater(self.test_counts["b"], 0)  # the checks below are not vacuous
        self.assertGreater(self.train_counts["a"], self.train_counts["b"])
        self.result = self.run_it(self.config, self.table)

    def test_accuracy_is_the_majority_share_of_the_test_split(self):
        self.assertAlmostEqual(self.result.metrics["accuracy"], self.test_counts["a"] / self.n_test)

    def test_macro_precision_recall_f1_for_a_constant_predictor(self):
        share_a = self.test_counts["a"] / self.n_test
        f1_a = 2 * share_a * 1.0 / (share_a + 1.0)
        self.assertAlmostEqual(self.result.metrics["precision"], share_a / 2)
        self.assertAlmostEqual(self.result.metrics["recall"], 0.5)
        self.assertAlmostEqual(self.result.metrics["f1"], f1_a / 2)

    def test_log_loss_is_the_cross_entropy_of_the_training_priors(self):
        p_a = self.train_counts["a"] / self.n_train
        p_b = self.train_counts["b"] / self.n_train
        expected = (
            -(self.test_counts["a"] * math.log(p_a) + self.test_counts["b"] * math.log(p_b))
            / self.n_test
        )
        self.assertAlmostEqual(self.result.metrics["log_loss"], expected)

    def test_constant_scores_give_an_auc_of_one_half(self):
        self.assertEqual(self.result.metrics["roc_auc"], 0.5)


class NewModelTests(RunnerTestCase):
    CASES = (
        ("decision_tree", {"max_depth": 3}),
        ("logistic_regression", {"max_iter": 300}),
    )

    def test_models_run_score_reuse_and_reproduce_through_the_runner(self):
        from modelbench.reproducibility import check_reproducibility

        metrics = ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss")
        for name, params in self.CASES:
            with self.subTest(model=name):
                config = make_config(
                    experiment_id=f"exp-{name.replace('_', '-')}",
                    model=name, params=params, metrics=metrics, stratify=True,
                )  # fmt: skip
                result = self.run_it(config)
                self.assertEqual(result.metrics["accuracy"], 1.0)
                self.assertEqual(result.metrics["roc_auc"], 1.0)
                self.assertTrue(math.isfinite(result.metrics["log_loss"]))
                again = self.run_it(config, reuse_existing=True)
                self.assertEqual((again.reused, again.run_id), (True, result.run_id))
                report = check_reproducibility(
                    self.store, result.run_id, blobs(), "label", environment=ENV
                )
                self.assertTrue(report.reproduced, msg=report.detail)

    def test_a_parameter_the_model_does_not_have_is_rejected_before_a_run_exists(self):
        with self.assertRaises(ModelError):
            self.run_it(make_config(model="decision_tree", params={"k": 3}))
        with self.assertRaises(ModelError):
            self.run_it(make_config(model="logistic_regression", params={"max_depth": 3}))
        self.assertNoRunsCreated()


class DeterminismTests(RunnerTestCase):
    def test_same_inputs_give_identical_results_in_separate_stores(self):
        config = make_config(model="stratified", params={}, metrics=("accuracy", "f1"))
        outcomes = []
        for _ in range(2):
            with Store(":memory:", clock=lambda: "2026-01-01T00:00:00Z") as store:
                result = run_experiment(store, config, blobs(), "label", environment=ENV)
                record = store.get_run(result.run_id)
                outcomes.append((result, record, store.get_metrics(result.run_id)))
        self.assertEqual(outcomes[0], outcomes[1])

    def test_a_different_seed_changes_the_split_and_therefore_the_predictions(self):
        a = self.run_it(make_config(seed=1))
        b = self.run_it(make_config(seed=2))
        self.assertNotEqual(a.predictions_sha256, b.predictions_sha256)

    def test_random_models_get_a_seed_derived_from_the_experiment_seed(self):
        config = make_config(model="stratified", params={}, seed=11)
        result = self.run_it(config)
        tags = self.store.get_run(result.run_id).tags
        self.assertEqual(tags["model_seed"], str(derive_seed(11, "model")))
        self.assertEqual(self.store.get_run(result.run_id).params, {})  # params are left as given

    def test_an_explicit_model_seed_wins(self):
        config = make_config(model="stratified", params={"seed": 99}, seed=11)
        result = self.run_it(config)
        self.assertEqual(self.store.get_run(result.run_id).tags["model_seed"], "99")

    def test_models_without_a_seed_get_no_model_seed_tag(self):
        result = self.run_it(make_config(model="gaussian_nb", params={}))
        self.assertNotIn("model_seed", self.store.get_run(result.run_id).tags)


class ArtifactTests(RunnerTestCase):
    def test_predictions_file_matches_the_recorded_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_it(artifact_dir=tmp)
            (artifact,) = self.store.get_artifacts(result.run_id)
            path = Path(tmp) / result.run_id / "predictions.json"
            self.assertEqual(artifact.path, str(path))
            data = path.read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), artifact.sha256)
            self.assertEqual(artifact.sha256, result.predictions_sha256)
            self.assertEqual(artifact.size_bytes, len(data))
            body = json.loads(data)
            self.assertEqual(len(body["test_indices"]), result.n_test)
            self.assertEqual(len(body["y_true"]), result.n_test)
            self.assertEqual(len(body["y_pred"]), result.n_test)
            expected_indices = expected_split(make_config(), blobs())[1]
            self.assertEqual(body["test_indices"], expected_indices)

    def test_no_artifact_directory_means_no_files_and_no_artifact_records(self):
        result = self.run_it()
        self.assertEqual(self.store.get_artifacts(result.run_id), [])

    def test_the_digest_matches_an_independent_computation(self):
        table = blobs(n_a=30, n_b=10)
        config = make_config(model="majority", params={}, metrics=("accuracy",))
        train, test = expected_split(config, table)
        labels = table.column("label")
        counts = Counter(labels[i] for i in train)
        self.assertGreater(counts["a"], counts["b"])  # so the majority model predicts "a"
        text = json.dumps(
            {
                "test_indices": test,
                "y_true": [labels[i] for i in test],
                "y_pred": ["a"] * len(test),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.assertEqual(self.run_it(config, table).predictions_sha256, expected)

    def test_the_digest_is_the_same_with_or_without_writing_files(self):
        plain = self.run_it(make_config(experiment_id="exp-plain"))
        with tempfile.TemporaryDirectory() as tmp:
            written = self.run_it(make_config(experiment_id="exp-written"), artifact_dir=tmp)
        self.assertEqual(plain.predictions_sha256, written.predictions_sha256)


class ReuseTests(RunnerTestCase):
    def test_identical_completed_run_is_reused_not_repeated(self):
        first = self.run_it(reuse_existing=True)
        second = self.run_it(reuse_existing=True)
        self.assertFalse(first.reused)
        self.assertTrue(second.reused)
        self.assertEqual(second.run_id, first.run_id)
        self.assertEqual(second.metrics, first.metrics)
        self.assertEqual(
            (second.n_train, second.n_test, second.predictions_sha256),
            (first.n_train, first.n_test, first.predictions_sha256),
        )
        self.assertEqual(len(self.store.list_runs()), 1)

    def test_reuse_is_off_by_default(self):
        self.run_it()
        self.run_it()
        self.assertEqual(len(self.store.list_runs()), 2)

    def test_changed_settings_or_data_or_label_are_not_reused(self):
        self.run_it(reuse_existing=True)
        self.run_it(make_config(params={"k": 5}), reuse_existing=True)
        self.run_it(make_config(experiment_id="exp-b"), reuse_existing=True)
        self.run_it(table=blobs(n_a=21), reuse_existing=True)
        self.assertEqual(len(self.store.list_runs()), 4)

    def test_failed_runs_are_never_reused(self):
        config = make_config(params={"k": 40})  # more neighbours than training rows
        with self.assertRaises(ModelError):
            self.run_it(config, reuse_existing=True)
        with self.assertRaises(ModelError):
            self.run_it(config, reuse_existing=True)
        self.assertEqual([r.status for r in self.store.list_runs()], ["failed", "failed"])

    def test_a_failed_run_is_not_reused_even_if_it_recorded_every_tag(self):
        # an artifact-write failure happens after the tags are set, so the failed run looks
        # complete to a check that only inspects tags
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "not-a-directory"
            blocker.write_text("x")
            with self.assertRaises(OSError):
                self.run_it(artifact_dir=blocker, reuse_existing=True)
        (failed,) = self.store.list_runs()
        self.assertEqual(failed.status, "failed")
        for tag in ("n_train", "n_test", "predictions_sha256"):
            self.assertIn(tag, failed.tags)  # the precondition this test depends on
        fresh = self.run_it(reuse_existing=True)
        self.assertFalse(fresh.reused)
        self.assertNotEqual(fresh.run_id, failed.run_id)
        self.assertEqual([r.status for r in self.store.list_runs()], ["completed", "failed"])

    def test_invalid_input_still_raises_even_when_a_reusable_run_exists(self):
        self.run_it(reuse_existing=True)
        with self.assertRaises(RunnerError):
            self.run_it(reuse_existing=True, feature_columns=["nope"])


class FailureRecordingTests(RunnerTestCase):
    def test_a_failure_during_training_is_recorded_and_reraised(self):
        with self.assertRaises(ModelError):
            self.run_it(make_config(params={"k": 40}))
        (run,) = self.store.list_runs()
        self.assertEqual(run.status, "failed")
        self.assertTrue(run.error.startswith("ModelError:"))
        self.assertIn("k=40", run.error)
        self.assertEqual(self.store.get_metrics(run.run_id), [])
        self.assertIsNotNone(run.finished_at)

    def test_a_failure_while_scoring_is_recorded(self):
        boom = RuntimeError("scoring blew up")
        with (
            mock.patch("modelbench.runner.m.accuracy", side_effect=boom),
            self.assertRaises(RuntimeError),
        ):
            self.run_it()
        (run,) = self.store.list_runs()
        self.assertEqual(run.status, "failed")
        self.assertEqual(run.error, "RuntimeError: scoring blew up")
        self.assertEqual(self.store.get_metrics(run.run_id), [])

    def test_an_artifact_write_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "not-a-directory"
            blocker.write_text("x")
            with self.assertRaises(OSError):
                self.run_it(artifact_dir=blocker)
        (run,) = self.store.list_runs()
        self.assertEqual(run.status, "failed")
        self.assertIn("not-a-directory", run.error)  # the error names the offending path


class ValidationBeforeRunTests(RunnerTestCase):
    def test_regression_metrics_are_rejected(self):
        for name in ("mae", "mse", "rmse", "r2"):
            with self.assertRaises(RunnerError, msg=name):
                self.run_it(make_config(metrics=("accuracy", name)))
        self.assertNoRunsCreated()

    def test_roc_auc_needs_exactly_two_training_classes(self):
        with self.assertRaises(RunnerError):
            self.run_it(make_config(metrics=("roc_auc",)), three_class_table())
        self.assertNoRunsCreated()

    def test_roc_auc_needs_both_classes_in_the_test_split(self):
        table = Table(("x", "label"), (("0", "a"), ("1", "a"), ("2", "a"), ("9", "b")))
        labels = table.column("label")
        for seed in range(50):
            config = make_config(
                model="majority", params={}, metrics=("roc_auc",), test_fraction=0.25, seed=seed
            )
            train, test = expected_split(config, table)
            if {labels[i] for i in test} == {"a"} and {labels[i] for i in train} == {"a", "b"}:
                with self.assertRaises(RunnerError) as ctx:
                    self.run_it(config, table)
                self.assertIn("test split", str(ctx.exception))
                self.assertNoRunsCreated()
                return
        self.fail("no seed gave a single-class test split")  # pragma: no cover

    def test_log_loss_needs_every_test_label_seen_in_training(self):
        table = Table(("x", "label"), tuple((str(i), "a") for i in range(9)) + (("50", "b"),))
        # find a seed whose test split contains the lone "b" row
        for seed in range(50):
            config = make_config(model="majority", params={}, metrics=("log_loss",), seed=seed)
            train, test = expected_split(config, table)
            if 9 in test:
                with self.assertRaises(RunnerError):
                    self.run_it(config, table)
                self.assertNoRunsCreated()
                return
        self.fail("no seed put the rare label in the test split")  # pragma: no cover

    def test_unknown_model_and_bad_model_parameters(self):
        with self.assertRaises(ModelError):
            self.run_it(make_config(model="no-such-model", params={}))
        with self.assertRaises(ModelError):
            self.run_it(make_config(params={"k": 0}))
        with self.assertRaises(ModelError):
            self.run_it(make_config(params={"unknown": 1}))
        self.assertNoRunsCreated()

    def test_bad_columns(self):
        for kwargs in (
            {"feature_columns": ["nope"]},
            {"feature_columns": ["label", "x1"]},
            {"feature_columns": ["x1", "x1"]},
            {"feature_columns": []},
        ):
            with self.assertRaises(RunnerError, msg=str(kwargs)):
                self.run_it(**kwargs)
        with self.assertRaises(RunnerError):
            run_experiment(self.store, make_config(), blobs(), "no-label", environment=ENV)
        self.assertNoRunsCreated()

    def test_non_numeric_features_report_the_row_and_column(self):
        for bad in ("abc", "", "nan", "inf", "1_0", "0x10", "1,5"):
            table = Table(("x", "label"), (("1", "a"), (bad, "a"), ("3", "b"), ("4", "b")))
            with self.assertRaises(RunnerError, msg=repr(bad)) as ctx:
                self.run_it(make_config(test_fraction=0.5, params={"k": 1}), table)
            self.assertIn("row 2", str(ctx.exception))
            self.assertIn("'x'", str(ctx.exception))
        self.assertNoRunsCreated()

    def test_non_ascii_digits_are_not_numbers(self):
        # float() would happily read Arabic-Indic digits, so the check must be stricter
        table = Table(("x", "label"), (("1", "a"), ("١٢", "a"), ("3", "b"), ("4", "b")))
        with self.assertRaises(RunnerError) as ctx:
            self.run_it(make_config(test_fraction=0.5, params={"k": 1}), table)
        self.assertIn("row 2", str(ctx.exception))
        self.assertNoRunsCreated()

    def test_an_empty_label_is_rejected(self):
        table = Table(("x", "label"), (("1", "a"), ("2", " "), ("3", "b"), ("4", "b")))
        with self.assertRaises(RunnerError) as ctx:
            self.run_it(make_config(test_fraction=0.5, params={"k": 1}), table)
        self.assertIn("row 2", str(ctx.exception))
        self.assertNoRunsCreated()

    def test_impossible_splits(self):
        tiny = Table(("x", "label"), (("1", "a"),))
        with self.assertRaises(DatasetError):
            self.run_it(table=tiny)
        singletons = Table(("x", "label"), (("1", "a"), ("2", "b"), ("3", "c")))
        with self.assertRaises(DatasetError):
            self.run_it(make_config(stratify=True, params={"k": 1}), singletons)
        self.assertNoRunsCreated()

    def test_a_pinned_dataset_version_must_match(self):
        with self.assertRaises(StoreError):
            self.run_it(make_config(dataset_version="0" * 12))
        self.assertNoRunsCreated()
        pinned = make_config(dataset_version=make_version("blobs", blobs()).version)
        self.assertEqual(self.run_it(pinned).metrics["accuracy"], 1.0)

    def test_a_wrong_dataset_id_in_the_config_is_not_silently_accepted(self):
        # the dataset is registered under the config's dataset id, so a pinned version
        # for a different table cannot match
        other = blobs(n_a=19)
        pinned = make_config(dataset_version=make_version("blobs", blobs()).version)
        with self.assertRaises(StoreError):
            self.run_it(pinned, other)


class StratifyTests(RunnerTestCase):
    def test_stratified_splits_keep_class_balance_in_the_test_set(self):
        table = blobs(n_a=30, n_b=10)
        config = make_config(stratify=True, params={"k": 1})
        _, test = expected_split(config, table)
        labels = table.column("label")
        self.assertEqual(Counter(labels[i] for i in test), {"a": 6, "b": 2})
        self.assertEqual(self.run_it(config, table).n_test, 8)


def line_table(n=40, slope=2.0, intercept=1.0):
    rows = [(str(i), repr(slope * i + intercept)) for i in range(n)]
    return Table(("x", "y"), tuple(rows))


class RegressionRunTests(RunnerTestCase):
    def run_reg(self, config, table=None, **kwargs):
        kwargs.setdefault("environment", ENV)
        return run_experiment(self.store, config, table or line_table(), "y", **kwargs)

    def reg_config(self, **overrides):
        fields = {
            "model": "linear_regression",
            "params": {},
            "metrics": ("mae", "mse", "rmse", "r2"),
        }
        fields.update(overrides)
        return make_config(**fields)

    def test_exact_line_is_recovered_and_recorded(self):
        result = self.run_reg(self.reg_config())
        self.assertAlmostEqual(result.metrics["mae"], 0.0, places=8)
        self.assertAlmostEqual(result.metrics["rmse"], 0.0, places=8)
        self.assertAlmostEqual(result.metrics["r2"], 1.0, places=8)
        self.assertEqual((result.n_train, result.n_test), (32, 8))
        record = self.store.get_run(result.run_id)
        self.assertEqual(record.status, "completed")
        self.assertEqual(set(self.store.final_metrics(result.run_id)), {"mae", "mse", "rmse", "r2"})

    def test_mean_baseline_matches_a_hand_computation(self):
        config = self.reg_config(model="mean")
        table = line_table()
        result = self.run_reg(config, table)
        train_idx, test_idx = split_indices(len(table), config.test_fraction, config.seed, None)
        targets = [float(row[1]) for row in table.rows]
        mean = sum(targets[i] for i in train_idx) / len(train_idx)
        errors = [targets[i] - mean for i in test_idx]
        self.assertAlmostEqual(result.metrics["mae"], sum(abs(e) for e in errors) / len(errors))
        self.assertAlmostEqual(result.metrics["mse"], sum(e * e for e in errors) / len(errors))
        self.assertAlmostEqual(result.metrics["rmse"], math.sqrt(result.metrics["mse"]))
        self.assertLess(result.metrics["r2"], 0.2)  # the mean explains almost nothing of a line

    def test_predictions_are_hashed_as_floats_and_repeat_exactly(self):
        config = self.reg_config()
        with tempfile.TemporaryDirectory() as tmp:
            first = self.run_reg(config, artifact_dir=tmp)
            second = self.run_reg(config)
            saved = json.loads((Path(tmp) / first.run_id / "predictions.json").read_text())
        self.assertEqual(first.predictions_sha256, second.predictions_sha256)
        self.assertEqual(first.metrics, second.metrics)
        self.assertTrue(all(isinstance(v, float) for v in saved["y_true"]))
        self.assertTrue(all(isinstance(v, float) for v in saved["y_pred"]))
        self.assertEqual(len(saved["y_pred"]), 8)

    def test_reuse_existing_works_for_regression(self):
        config = self.reg_config()
        first = self.run_reg(config)
        again = self.run_reg(config, reuse_existing=True)
        self.assertTrue(again.reused)
        self.assertEqual(again.run_id, first.run_id)
        self.assertEqual(len(self.store.list_runs()), 1)

    def test_numeric_target_formats(self):
        table = Table(
            ("x", "y"),
            tuple((str(i), cell) for i, cell in enumerate(
                [" 1 ", "2.5", "-3", "+4e0", ".5", "6.", "7", "8", "9", "10"]
            )),
        )  # fmt: skip
        result = self.run_reg(self.reg_config(model="mean", metrics=("mae",)), table)
        self.assertEqual(result.n_train + result.n_test, 10)

    def test_bad_targets_are_rejected_before_any_run(self):
        for bad in ("abc", "", "nan", "inf", "1e999", "１２", "1,5"):
            rows = list(line_table(10).rows)
            rows[4] = ("4", bad)
            with self.assertRaises(RunnerError, msg=repr(bad)):
                self.run_reg(self.reg_config(), Table(("x", "y"), tuple(rows)))
        self.assertNoRunsCreated()

    def test_metric_kind_must_match_the_model_kind(self):
        for metrics in (("accuracy",), ("mae", "f1"), ("log_loss",)):
            with self.assertRaises(RunnerError, msg=str(metrics)):
                self.run_reg(self.reg_config(metrics=metrics))
        for metrics in (("mae",), ("accuracy", "r2")):
            with self.assertRaises(RunnerError, msg=str(metrics)):
                self.run_it(make_config(metrics=metrics))
        self.assertNoRunsCreated()

    def test_stratify_is_rejected_for_regression(self):
        with self.assertRaisesRegex(RunnerError, "stratify"):
            self.run_reg(self.reg_config(stratify=True))
        self.assertNoRunsCreated()

    def test_model_errors_fail_the_run_and_are_recorded(self):
        table = Table(("a", "b", "y"), tuple((str(i), str(2 * i), str(i)) for i in range(20)))
        with self.assertRaises(ModelError):  # a and b are perfectly collinear, l2 is 0
            run_experiment(self.store, self.reg_config(), table, "y", environment=ENV)
        (run,) = self.store.list_runs()
        self.assertEqual(run.status, "failed")
        self.assertIn("l2", run.error)

    def test_unknown_parameter_and_unknown_model(self):
        with self.assertRaises(ModelError):
            self.run_reg(self.reg_config(params={"k": 3}))
        with self.assertRaises(ModelError):
            self.run_reg(self.reg_config(model="no_such"))
        self.assertNoRunsCreated()

    def test_ridge_parameter_changes_the_result(self):
        plain = self.run_reg(self.reg_config(experiment_id="p"))
        ridge = self.run_reg(self.reg_config(experiment_id="r", params={"l2": 5.0}))
        self.assertNotEqual(plain.predictions_sha256, ridge.predictions_sha256)
        self.assertGreater(ridge.metrics["mae"], plain.metrics["mae"])

    def test_metric_errors_name_the_problem(self):
        with self.assertRaisesRegex(RunnerError, "do not apply to a regression"):
            self.run_reg(self.reg_config(metrics=("accuracy",)))
        with self.assertRaisesRegex(RunnerError, "do not apply to a classification"):
            self.run_it(make_config(metrics=("rmse",)))
        for kind in ("classification", "regression"):
            with self.assertRaisesRegex(RunnerError, "unsupported metric"):
                _check_metrics(("not_a_metric",), kind)

    def test_regressor_with_a_seed_parameter_gets_a_derived_seed(self):
        class Seeded(MeanRegressor):
            name = "seeded_mean"

            def __init__(self, seed=0):
                super().__init__()
                self.seed = seed

            def get_params(self):
                return {"seed": self.seed}

        with mock.patch.dict(REGRESSORS, {"seeded_mean": Seeded}):
            result = self.run_reg(self.reg_config(model="seeded_mean", metrics=("mae",)))
        record = self.store.get_run(result.run_id)
        self.assertEqual(record.tags["model_seed"], str(derive_seed(7, "model")))


class ExtraMetricTests(RunnerTestCase):
    def test_separable_data_scores_perfectly_on_every_new_metric(self):
        names = (
            "precision_micro", "precision_weighted", "recall_micro", "recall_weighted",
            "f1_micro", "f1_weighted", "balanced_accuracy", "mcc", "average_precision",
        )  # fmt: skip
        result = self.run_it(make_config(metrics=names, stratify=True))
        self.assertEqual(set(result.metrics), set(names))
        for name, value in result.metrics.items():
            self.assertAlmostEqual(value, 1.0, msg=name)

    def test_majority_baseline_values_follow_from_its_accuracy(self):
        names = (
            "accuracy", "precision_micro", "recall_micro", "f1_micro", "recall_weighted",
            "precision_weighted", "f1_weighted", "balanced_accuracy", "mcc",
        )  # fmt: skip
        config = make_config(model="majority", params={}, metrics=names)
        scores = self.run_it(config).metrics
        acc = scores["accuracy"]
        self.assertEqual(acc, 0.375)  # the seeded split: 3 of 8 test rows are the majority class
        for name in ("precision_micro", "recall_micro", "f1_micro", "recall_weighted"):
            self.assertAlmostEqual(scores[name], acc, msg=name)
        self.assertAlmostEqual(scores["precision_weighted"], acc * acc)  # only that class scores
        f1_of_predicted = 2 * acc * 1.0 / (acc + 1.0)  # its recall is 1
        self.assertAlmostEqual(scores["f1_weighted"], f1_of_predicted * acc)
        self.assertAlmostEqual(scores["balanced_accuracy"], 0.5)
        self.assertEqual(scores["mcc"], 0.0)

    def test_each_score_metric_calls_its_own_function(self):
        with (
            mock.patch("modelbench.runner.m.average_precision", return_value=0.25) as ap,
            mock.patch("modelbench.runner.m.roc_auc", return_value=0.75) as auc,
        ):
            scores = self.run_it(make_config(metrics=("roc_auc", "average_precision"))).metrics
        self.assertEqual(scores, {"roc_auc": 0.75, "average_precision": 0.25})
        self.assertEqual((ap.call_count, auc.call_count), (1, 1))

    def test_average_precision_needs_two_classes_like_roc_auc(self):
        with self.assertRaises(RunnerError):
            self.run_it(make_config(metrics=("average_precision",)), three_class_table())
        self.assertNoRunsCreated()

    def test_new_metrics_are_rejected_for_regression_models(self):
        with self.assertRaisesRegex(RunnerError, "do not apply to a regression"):
            self.run_it(make_config(model="mean", params={}, metrics=("mcc",)), line_table())
        self.assertNoRunsCreated()


class ConfusionMatrixTagTests(RunnerTestCase):
    def test_classification_runs_record_the_test_matrix(self):
        result = self.run_it()
        tag = self.store.get_run(result.run_id).tags["confusion_matrix"]
        self.assertEqual(tag, '{"labels":["a","b"],"matrix":[[5,0],[0,3]]}')  # 8 test rows

    def test_the_matrix_matches_the_stored_predictions(self):
        config = make_config(model="majority", params={})
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_it(config, artifact_dir=tmp)
            saved = json.loads((Path(tmp) / result.run_id / "predictions.json").read_text())
        cm = confusion.from_tag(self.store.get_run(result.run_id).tags["confusion_matrix"])
        pairs = list(zip(saved["y_true"], saved["y_pred"], strict=True))
        for i, true in enumerate(cm.labels):
            for j, pred in enumerate(cm.labels):
                self.assertEqual(cm.matrix[i][j], pairs.count((true, pred)))
        self.assertEqual(cm.total, 8)

    def test_a_label_predicted_but_absent_from_the_test_split_still_gets_a_row(self):
        table = three_class_table()
        result = self.run_it(make_config(model="majority", params={}), table)
        cm = confusion.from_tag(self.store.get_run(result.run_id).tags["confusion_matrix"])
        self.assertEqual(cm.total, result.n_test)
        self.assertEqual(cm.labels, ("a", "b", "c")[: len(cm.labels)])

    def test_a_training_class_missing_from_the_test_split_keeps_its_row(self):
        rows = [(str(i), "a") for i in range(20)] + [(str(i + 20), "b") for i in range(20)]
        rows += [("100", "c"), ("101", "c")]
        table = Table(("x", "label"), tuple(rows))
        for seed in range(200):
            config = make_config(model="majority", params={}, test_fraction=0.1, seed=seed)
            train, test = expected_split(config, table)
            labels = table.column("label")
            if "c" in {labels[i] for i in train} and "c" not in {labels[i] for i in test}:
                result = self.run_it(config, table)
                cm = confusion.from_tag(self.store.get_run(result.run_id).tags["confusion_matrix"])
                self.assertEqual(cm.labels, ("a", "b", "c"))
                self.assertEqual(cm.matrix[2], (0, 0, 0))  # no true c rows, and none predicted
                self.assertEqual(cm.total, result.n_test)
                return
        self.fail("no seed left class c out of the test split")  # pragma: no cover

    def test_regression_runs_have_no_matrix(self):
        run = RegressionRunTests()
        run.setUp()
        self.addCleanup(run.store.close)
        result = run.run_reg(run.reg_config())
        self.assertNotIn("confusion_matrix", run.store.get_run(result.run_id).tags)

    def test_a_matrix_too_big_for_a_tag_is_left_out_without_failing_the_run(self):
        with mock.patch("modelbench.runner.confusion.to_tag", return_value=None):
            result = self.run_it()
        self.assertNotIn("confusion_matrix", self.store.get_run(result.run_id).tags)
        self.assertEqual(self.store.get_run(result.run_id).status, "completed")


class ReuseColumnsTests(RunnerTestCase):
    def table(self):
        # a separates the classes perfectly; b is noise that does not
        rows = [(f"{i * 0.1:.1f}", str((i * 7) % 5), "a") for i in range(20)]
        rows += [(f"{10 + i * 0.1:.1f}", str((i * 3) % 5), "b") for i in range(20)]
        return Table(("a", "b", "label"), tuple(rows))

    def test_different_feature_columns_are_a_different_run(self):
        table, config = self.table(), make_config(stratify=True)
        first = self.run_it(config, table, feature_columns=["a"], reuse_existing=True)
        second = self.run_it(config, table, feature_columns=["b"], reuse_existing=True)
        fresh = self.run_it(config, table, feature_columns=["b"])
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertFalse(second.reused)
        self.assertEqual(second.metrics, fresh.metrics)
        self.assertNotEqual(first.metrics["accuracy"], second.metrics["accuracy"])

    def test_the_same_columns_are_still_reused_however_they_are_named(self):
        table, config = self.table(), make_config(stratify=True)
        first = self.run_it(config, table, reuse_existing=True)  # default: every non-label column
        again = self.run_it(config, table, feature_columns=["a", "b"], reuse_existing=True)
        self.assertTrue(again.reused)
        self.assertEqual(again.run_id, first.run_id)
        reordered = self.run_it(config, table, feature_columns=["b", "a"], reuse_existing=True)
        self.assertFalse(reordered.reused)  # the column order is part of how the model sees rows

    def test_the_columns_are_recorded_as_a_tag(self):
        table, config = self.table(), make_config()
        one = self.run_it(config, table, feature_columns=["a"])
        two = self.run_it(config, table, feature_columns=["b"])
        tags = [self.store.get_run(r.run_id).tags["columns_sha256"] for r in (one, two)]
        self.assertEqual(len(set(tags)), 2)
        self.assertEqual(tags[0], columns_signature(table, "label", ["a"]))

    def test_a_run_made_before_the_tag_existed_is_never_reused(self):
        table, config = self.table(), make_config()
        first = self.run_it(config, table)
        with self.store._conn:  # simulate a database written by an older version
            self.store._conn.execute(
                "DELETE FROM tags WHERE run_id = ? AND key = 'columns_sha256'", (first.run_id,)
            )
        again = self.run_it(config, table, reuse_existing=True)
        self.assertFalse(again.reused)
        self.assertNotEqual(again.run_id, first.run_id)


class CrossValidationRunTests(RunnerTestCase):
    def cv_config(self, **overrides):
        fields = {"cv_folds": 4, "stratify": True, "metrics": ("accuracy", "balanced_accuracy")}
        fields.update(overrides)
        return make_config(**fields)

    def test_mean_and_std_by_hand(self):
        mean, std = _mean_and_std([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(mean, 2.5)
        self.assertAlmostEqual(std, math.sqrt(1.25))  # deviations -1.5 -0.5 0.5 1.5: 5 / 4
        mean, std = _mean_and_std([0.5, 0.5, 0.5])
        self.assertEqual((mean, std), (0.5, 0.0))

    def test_majority_baseline_on_balanced_folds_scores_exactly_one_half(self):
        # 20 + 20 rows, stratified into 4 folds of 5 + 5; each training set is 15 + 15, the tie
        # goes to the first class, so every fold predicts all "a": accuracy 5/10.
        config = self.cv_config(model="majority", params={})
        result = self.run_it(config)
        self.assertEqual(result.metrics["accuracy"], 0.5)
        self.assertEqual(result.metrics["accuracy_std"], 0.0)
        self.assertEqual(result.metrics["balanced_accuracy"], 0.5)
        self.assertEqual(set(result.metrics), {"accuracy", "accuracy_std", "balanced_accuracy",
                                               "balanced_accuracy_std"})  # fmt: skip
        record = self.store.get_run(result.run_id)
        self.assertEqual(
            record.tags["confusion_matrix"], '{"labels":["a","b"],"matrix":[[20,0],[20,0]]}'
        )

    def test_per_fold_metrics_are_a_series_and_the_mean_is_their_average(self):
        config = self.cv_config(model="knn", params={"k": 3}, stratify=False, cv_folds=5)
        result = self.run_it(config)
        points = [p for p in self.store.get_metrics(result.run_id) if p.name == "fold/accuracy"]
        self.assertEqual([p.step for p in points], [0, 1, 2, 3, 4])
        mean, std = _mean_and_std([p.value for p in points])
        self.assertAlmostEqual(result.metrics["accuracy"], mean)
        self.assertAlmostEqual(result.metrics["accuracy_std"], std)
        summary = [p for p in self.store.get_metrics(result.run_id) if p.name == "accuracy"]
        self.assertEqual([p.step for p in summary], [0])
        self.assertEqual(self.store.final_metrics(result.run_id)["accuracy"], mean)

    def test_every_row_is_tested_once_and_recorded_in_the_predictions_file(self):
        config = self.cv_config(model="knn", params={"k": 3})
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_it(config, artifact_dir=tmp)
            saved = json.loads((Path(tmp) / result.run_id / "predictions.json").read_text())
        folds = saved["folds"]
        self.assertEqual(len(folds), 4)
        tested = sorted(i for f in folds for i in f["test_indices"])
        self.assertEqual(tested, list(range(40)))
        expected = kfold_indices(40, 4, config.seed, blobs().column("label"))
        self.assertEqual([f["test_indices"] for f in folds], [t for _, t in expected])
        labels = blobs().column("label")
        for fold in folds:
            self.assertEqual(fold["y_true"], [labels[i] for i in fold["test_indices"]])

    def test_tags_describe_the_folds(self):
        result = self.run_it(self.cv_config(model="majority", params={}))
        tags = self.store.get_run(result.run_id).tags
        self.assertEqual(tags["cv_folds"], "4")
        self.assertEqual(tags["n_test"], "40")  # every row is tested once
        self.assertEqual(tags["n_train"], "30")  # the largest training set
        self.assertEqual((result.n_train, result.n_test), (30, 40))
        self.assertNotIn("model_seed", tags)  # each fold has its own

    def test_n_train_is_the_largest_training_set_when_folds_are_uneven(self):
        # 41 rows in 4 folds: test sizes 11, 10, 10, 10, so training sets are 30, 31, 31, 31.
        table = Table(("x", "label"), tuple((str(i), "a" if i % 2 else "b") for i in range(41)))
        result = self.run_it(self.cv_config(model="majority", params={}, stratify=False), table)
        self.assertEqual((result.n_train, result.n_test), (31, 41))

    def test_each_fold_gets_its_own_model_seed(self):
        seeds = []
        real = create_model

        def spy(name, **params):
            seeds.append(params["seed"])
            return real(name, **params)

        with mock.patch("modelbench.runner.create_model", spy):
            self.run_it(self.cv_config(model="stratified", params={}))
        self.assertEqual(len(seeds), 5)  # one check before the run, then one per fold
        self.assertEqual(len(set(seeds[1:])), 4)
        self.assertEqual(seeds[1], derive_seed(7, "model-fold-0"))
        self.assertEqual(seeds[0], derive_seed(7, "model"))

    def test_running_twice_gives_identical_results(self):
        config = self.cv_config()
        first, second = self.run_it(config), self.run_it(config)
        self.assertEqual(first.predictions_sha256, second.predictions_sha256)
        self.assertEqual(first.metrics, second.metrics)

    def test_reuse_returns_the_summary_metrics_only(self):
        config = self.cv_config()
        first = self.run_it(config)
        again = self.run_it(config, reuse_existing=True)
        self.assertTrue(again.reused)
        self.assertEqual(again.metrics, first.metrics)
        self.assertFalse(any(name.startswith("fold/") for name in again.metrics))

    def test_a_cv_run_can_be_checked_for_reproducibility(self):
        result = self.run_it(self.cv_config())
        report = check_reproducibility(self.store, result.run_id, blobs(), "label", environment=ENV)
        self.assertTrue(report.reproduced, report.detail)

    def test_a_cv_run_survives_export_and_import(self):
        from modelbench.export import export_json
        from modelbench.importer import import_runs_json

        result = self.run_it(self.cv_config())
        text = export_json(self.store, [result.run_id])
        other = Store(":memory:")
        self.addCleanup(other.close)
        report = import_runs_json(other, text)
        self.assertEqual(len(report.imported), 1)
        new_id = report.imported[0][1]
        self.assertEqual(other.get_config(new_id).cv_folds, 4)
        self.assertEqual(other.final_metrics(new_id), self.store.final_metrics(result.run_id))
        self.assertEqual(other.get_run(new_id).tags["cv_folds"], "4")

    def test_regression_cross_validation(self):
        config = make_config(
            model="linear_regression", params={}, metrics=("mae", "r2"), cv_folds=5,
            dataset_id="line",
        )  # fmt: skip
        result = run_experiment(self.store, config, line_table(), "y", environment=ENV)
        self.assertAlmostEqual(result.metrics["mae"], 0.0, places=8)
        self.assertAlmostEqual(result.metrics["r2"], 1.0, places=8)
        self.assertAlmostEqual(result.metrics["r2_std"], 0.0, places=8)
        self.assertNotIn("confusion_matrix", self.store.get_run(result.run_id).tags)
        self.assertEqual(result.n_test, 40)

    def test_problems_are_found_before_any_run_exists(self):
        table = Table(("x", "label"), (("0", "a"), ("1", "a"), ("2", "a"), ("3", "a"), ("9", "b")))
        with self.assertRaisesRegex(RunnerError, "test split"):  # a fold without class b
            self.run_it(self.cv_config(metrics=("roc_auc",), stratify=False, cv_folds=5), table)
        with self.assertRaises(DatasetError):
            self.run_it(self.cv_config(cv_folds=50), table)  # more folds than rows
        self.assertNoRunsCreated()

    def test_a_failure_in_a_later_fold_fails_the_run(self):
        calls = []
        real_fit = KNNClassifier.fit

        def flaky(self, X, y):
            calls.append(1)
            if len(calls) == 3:
                raise ModelError("fold exploded")
            return real_fit(self, X, y)

        with (
            mock.patch.object(KNNClassifier, "fit", flaky),
            self.assertRaisesRegex(ModelError, "fold exploded"),
        ):
            self.run_it(self.cv_config())
        (run,) = self.store.list_runs()
        self.assertEqual(run.status, "failed")
        self.assertEqual(self.store.final_metrics(run.run_id), {})


if __name__ == "__main__":
    unittest.main()
