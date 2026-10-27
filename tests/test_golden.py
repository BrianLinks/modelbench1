"""A frozen sample of the export format (`modelbench-runs`, version 1).

`tests/golden/export_v1.json` is what `export_json` writes for two small, fully deterministic
runs. If the format changes by accident this test fails; if it changes on purpose, bump the
format version, keep the old file importable, and regenerate the file with
`MODELBENCH_UPDATE_GOLDEN=1 python -m unittest tests.test_golden`.
"""

import json
import os
import unittest
from itertools import count
from pathlib import Path

from modelbench.config import ExperimentConfig
from modelbench.datasets import Table
from modelbench.export import export_json
from modelbench.importer import import_runs_json
from modelbench.runner import run_experiment
from modelbench.store import Store

GOLDEN = Path(__file__).resolve().parent / "golden" / "export_v1.json"
ENV = {"python": "3.12.0", "platform": "test"}


def ticking_clock():
    ticks = count()
    return lambda: f"2026-01-01T00:00:{next(ticks):02d}Z"


def classification_table():
    rows = [(f"{i * 0.5:.1f}", "1", "a") for i in range(8)]
    rows += [(f"{10 + i * 0.5:.1f}", "2", "b") for i in range(8)]
    return Table(("x1", "x2", "label"), tuple(rows))


def regression_table():
    return Table(("x", "y"), tuple((str(i), repr(3.0 * i + 2.0)) for i in range(12)))


def build_store():
    store = Store(":memory:", clock=ticking_clock())
    knn = ExperimentConfig(
        experiment_id="golden-knn", model="knn", seed=11, dataset_id="blobs",
        metrics=("accuracy", "f1"), params={"k": 3}, stratify=True,
    )  # fmt: skip
    line = ExperimentConfig(
        experiment_id="golden-line", model="linear_regression", seed=11, dataset_id="line",
        metrics=("mae", "r2"), params={}, cv_folds=3,
    )  # fmt: skip
    run_experiment(store, knn, classification_table(), "label", environment=ENV)
    run_experiment(store, line, regression_table(), "y", environment=ENV)
    store.add_note("run-000001", "frozen for the golden test")
    return store


class GoldenExportTests(unittest.TestCase):
    def setUp(self):
        self.store = build_store()
        self.addCleanup(self.store.close)

    def test_the_export_matches_the_frozen_file_byte_for_byte(self):
        text = export_json(self.store, ["run-000001", "run-000002"])
        if os.environ.get("MODELBENCH_UPDATE_GOLDEN") == "1":
            GOLDEN.parent.mkdir(exist_ok=True)
            GOLDEN.write_text(text, encoding="utf-8", newline="\n")
        self.assertEqual(text, GOLDEN.read_bytes().decode("utf-8"))

    def test_the_frozen_file_has_the_documented_shape(self):
        document = json.loads(GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(document["format"], "modelbench-runs")
        self.assertEqual(document["version"], 1)
        first, second = document["runs"]
        self.assertEqual(first["status"], "completed")
        self.assertEqual(first["metrics"][0]["name"], "accuracy")
        self.assertEqual(first["tags"]["n_test"], "4")
        self.assertEqual(second["config"]["cv_folds"], 3)
        self.assertEqual(second["tags"]["cv_folds"], "3")
        self.assertEqual({m["name"] for m in second["metrics"]} >= {"fold/mae", "mae_std"}, True)

    def test_the_frozen_file_still_imports_and_round_trips(self):
        other = Store(":memory:", clock=ticking_clock())
        self.addCleanup(other.close)
        report = import_runs_json(other, GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(len(report.imported), 2)
        for old, new in report.imported:
            self.assertEqual(other.final_metrics(new), self.store.final_metrics(old))
            self.assertEqual(other.get_run(new).config_hash, self.store.get_run(old).config_hash)
        again = import_runs_json(other, GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual((len(again.imported), len(again.skipped)), (0, 2))


if __name__ == "__main__":
    unittest.main()
