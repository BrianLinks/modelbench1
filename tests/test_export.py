import csv
import io
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modelbench.config import ExperimentConfig
from modelbench.datasets import Table
from modelbench.errors import ExportError, StoreError
from modelbench.export import (
    FORMAT,
    VERSION,
    export_json,
    export_metrics_csv,
    export_runs,
    write_text_atomic,
)
from modelbench.runner import run_experiment
from modelbench.store import Store

ENV = {"python": "3.12.1", "lib:numpy": None}


def blobs():
    rows = [(f"{i * 0.1:.1f}", "a") for i in range(20)]
    rows += [(f"{10 + i * 0.1:.1f}", "b") for i in range(20)]
    return Table(("x", "label"), tuple(rows))


def make_config(**overrides):
    fields = {
        "experiment_id": "exp-a",
        "model": "knn",
        "seed": 7,
        "dataset_id": "blobs",
        "metrics": ("accuracy", "f1"),
        "params": {"k": 3},
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


class ExportTestCase(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:", clock=lambda: "2026-01-01T00:00:00Z")
        self.addCleanup(self.store.close)
        self.table = blobs()

    def run_it(self, **overrides):
        return run_experiment(
            self.store, make_config(**overrides), self.table, "label", environment=ENV
        ).run_id


class ExportRunsTests(ExportTestCase):
    def test_document_header_and_order(self):
        a, b = self.run_it(), self.run_it(seed=8)
        doc = export_runs(self.store, [b, a])
        self.assertEqual((doc["format"], doc["version"]), ("modelbench-runs", 1))
        self.assertEqual((FORMAT, VERSION), ("modelbench-runs", 1))
        self.assertEqual([r["run_id"] for r in doc["runs"]], [b, a])

    def test_run_carries_everything_needed_to_read_it_without_the_database(self):
        run_id = self.run_it()
        self.store.add_note(run_id, "looks fine")
        run = export_runs(self.store, [run_id])["runs"][0]
        record = self.store.get_run(run_id)
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["seed"], 7)
        self.assertEqual(run["config"], make_config().to_dict())
        self.assertEqual(run["config_hash"], make_config().config_hash())
        self.assertEqual(run["dataset"]["version"], record.dataset_version)
        self.assertEqual(run["dataset"]["n_rows"], 40)
        self.assertEqual(run["dataset"]["columns"], ["x", "label"])
        self.assertEqual(run["dataset"]["label_column"], "label")
        self.assertEqual(run["environment"], ENV)
        self.assertEqual(run["tags"], dict(sorted(record.tags.items())))
        self.assertIn("predictions_sha256", run["tags"])
        self.assertEqual(
            {(m["name"], m["step"]): m["value"] for m in run["metrics"]},
            {("accuracy", 0): 1.0, ("f1", 0): 1.0},
        )
        self.assertEqual(
            run["notes"], [{"created_at": "2026-01-01T00:00:00Z", "text": "looks fine"}]
        )
        self.assertEqual(run["artifacts"], [])
        self.assertIsNone(run["error"])

    def test_artifacts_and_failures_are_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_id = run_experiment(
                self.store, make_config(), self.table, "label", environment=ENV, artifact_dir=tmp
            ).run_id
            artifact = export_runs(self.store, [run_id])["runs"][0]["artifacts"][0]
            self.assertEqual(artifact["name"], "predictions")
            self.assertEqual(len(artifact["sha256"]), 64)
            self.assertTrue(artifact["path"].endswith("predictions.json"))
        dataset = self.store.get_dataset("blobs", self.store.get_run(run_id).dataset_version)
        failed = self.store.create_run(make_config(), dataset, ENV)
        self.store.fail_run(failed, "boom")
        run = export_runs(self.store, [failed])["runs"][0]
        self.assertEqual((run["status"], run["error"], run["metrics"]), ("failed", "boom", []))

    def test_multi_step_metrics_are_all_exported(self):
        run_id = self.run_it()
        dataset = self.store.get_dataset("blobs", self.store.get_run(run_id).dataset_version)
        live = self.store.create_run(make_config(), dataset, ENV)
        for step, loss in enumerate((0.9, 0.5, 0.25)):
            self.store.log_metric(live, "loss", loss, step=step)
        points = export_runs(self.store, [live])["runs"][0]["metrics"]
        self.assertEqual([(p["step"], p["value"]) for p in points], [(0, 0.9), (1, 0.5), (2, 0.25)])

    def test_json_text_is_stable_and_parses_back(self):
        run_id = self.run_it()
        text = export_json(self.store, [run_id])
        self.assertEqual(text, export_json(self.store, [run_id]))
        self.assertTrue(text.endswith("}\n"))
        self.assertEqual(json.loads(text), export_runs(self.store, [run_id]))
        self.assertLess(text.index('"config"'), text.index('"run_id"'))  # sorted keys

    def test_bad_run_lists(self):
        run_id = self.run_it()
        for bad in ([], (), [run_id, run_id], run_id, "run-9"):
            with self.subTest(bad=bad), self.assertRaises(ExportError):
                export_runs(self.store, bad)
        with self.assertRaises(StoreError):
            export_runs(self.store, [run_id, "run-000099"])


class MetricsCsvTests(ExportTestCase):
    def parse(self, text):
        return list(csv.reader(io.StringIO(text)))

    def test_one_row_per_metric_point(self):
        a = self.run_it()
        b = self.run_it(experiment_id="exp-b", metrics=("accuracy",))
        rows = self.parse(export_metrics_csv(self.store, [a, b]))
        self.assertEqual(rows[0], ["run_id", "experiment_id", "metric", "step", "value"])
        self.assertEqual(
            rows[1:],
            [
                [a, "exp-a", "accuracy", "0", "1.0"],
                [a, "exp-a", "f1", "0", "1.0"],
                [b, "exp-b", "accuracy", "0", "1.0"],
            ],
        )

    def test_values_round_trip_exactly(self):
        run_id = self.run_it()
        dataset = self.store.get_dataset("blobs", self.store.get_run(run_id).dataset_version)
        live = self.store.create_run(make_config(), dataset, ENV)
        self.store.log_metric(live, "loss", 0.1 + 0.2)
        rows = self.parse(export_metrics_csv(self.store, [live]))
        self.assertEqual(float(rows[1][4]), 0.1 + 0.2)

    def test_run_without_metrics_gives_header_only(self):
        run_id = self.run_it()
        dataset = self.store.get_dataset("blobs", self.store.get_run(run_id).dataset_version)
        empty = self.store.create_run(make_config(), dataset, ENV)
        self.assertEqual(len(self.parse(export_metrics_csv(self.store, [empty]))), 1)

    def test_uses_unix_line_endings_and_validates_ids(self):
        run_id = self.run_it()
        self.assertNotIn("\r", export_metrics_csv(self.store, [run_id]))
        with self.assertRaises(ExportError):
            export_metrics_csv(self.store, [])
        with self.assertRaises(ExportError):
            export_metrics_csv(self.store, [run_id, run_id])


class WriteTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_writes_text_exactly(self):
        target = self.dir / "out.json"
        write_text_atomic(target, "a\nb\n")
        self.assertEqual(target.read_bytes(), b"a\nb\n")
        self.assertEqual([p.name for p in self.dir.iterdir()], ["out.json"])

    def test_non_ascii_text_is_utf8_and_newlines_are_untouched(self):
        target = self.dir / "out.txt"
        write_text_atomic(target, "héllo\r\nworld")
        self.assertEqual(target.read_bytes(), "héllo\r\nworld".encode())

    def test_text_is_written_without_newline_translation(self):
        real = os.fdopen
        with mock.patch("modelbench.export.os.fdopen", wraps=real) as spy:
            write_text_atomic(self.dir / "out.csv", "a\n")
        self.assertEqual(spy.call_args.kwargs["newline"], "")
        self.assertEqual(spy.call_args.kwargs["encoding"], "utf-8")

    def test_replaces_an_existing_file(self):
        target = self.dir / "out.json"
        target.write_text("old")
        write_text_atomic(target, "new")
        self.assertEqual(target.read_text(), "new")

    def test_file_is_owner_only(self):
        target = self.dir / "out.json"
        write_text_atomic(target, "x")
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)

    def test_failed_write_keeps_the_original_and_leaves_no_temp_file(self):
        target = self.dir / "out.json"
        target.write_text("original")
        with (
            mock.patch("modelbench.export.os.replace", side_effect=OSError("disk full")),
            self.assertRaisesRegex(ExportError, "disk full"),
        ):
            write_text_atomic(target, "new")
        self.assertEqual(target.read_text(), "original")
        self.assertEqual([p.name for p in self.dir.iterdir()], ["out.json"])

    def test_missing_directory(self):
        with self.assertRaises(ExportError):
            write_text_atomic(self.dir / "nope" / "out.json", "x")
        self.assertEqual(list(self.dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
