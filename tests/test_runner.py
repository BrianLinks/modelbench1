import hashlib
import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from modelbench.config import ExperimentConfig
from modelbench.datasets import Table, make_version, split_indices
from modelbench.errors import DatasetError, ModelError, RunnerError, StoreError
from modelbench.runner import run_experiment
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
        expected = -(
            self.test_counts["a"] * math.log(p_a) + self.test_counts["b"] * math.log(p_b)
        ) / self.n_test
        self.assertAlmostEqual(self.result.metrics["log_loss"], expected)

    def test_constant_scores_give_an_auc_of_one_half(self):
        self.assertEqual(self.result.metrics["roc_auc"], 0.5)


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
        with mock.patch("modelbench.runner.m.accuracy", side_effect=boom):
            with self.assertRaises(RuntimeError):
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


if __name__ == "__main__":
    unittest.main()
