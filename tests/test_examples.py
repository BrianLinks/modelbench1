"""Run every file in examples/ through the command line, as the examples README describes."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from modelbench.cli import main
from modelbench.config import ExperimentConfig

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


class ExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls._tmp.name)
        cls.db = str(cls.dir / "runs.db")
        cls.blobs = cls.dir / "blobs.csv"
        cls.linear = cls.dir / "linear.csv"
        for args in (
            ["sample", "blobs", "--seed", "1", "--rows-per-class", "40", "-o", cls.blobs],
            ["sample", "linear", "--seed", "1", "--rows", "120", "--noise", "3", "-o", cls.linear],
        ):
            code, _, err = cls.cli(*args, db=False)
            assert code == 0, err

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def cli(cls, *args, db=True):
        argv = (["--db", cls.db] if db else []) + [str(a) for a in args]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def example(self, name):
        return EXAMPLES / name

    def test_every_json_file_is_used_by_the_readme(self):
        readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
        files = sorted(p.name for p in EXAMPLES.glob("*.json"))
        self.assertGreaterEqual(len(files), 5)
        for name in files:
            self.assertIn(f"examples/{name}", readme, name)

    def test_the_single_config_files_are_valid_configs(self):
        for name in ("knn.json", "forest-cv.json"):
            config = ExperimentConfig.from_file(self.example(name))
            self.assertEqual(config.dataset_id, "blobs")

    def test_run_a_single_experiment(self):
        code, out, err = self.cli(
            "run", self.example("knn.json"), "--data", self.blobs, "--label", "label", "--json"
        )
        self.assertEqual((code, err), (0, ""))
        data = json.loads(out)
        self.assertEqual(set(data["metrics"]), {"accuracy", "f1", "balanced_accuracy"})
        self.assertGreater(data["metrics"]["accuracy"], 0.9)  # the clusters are well separated

    def test_expanding_window_validation_of_the_linear_model(self):
        code, out, err = self.cli(
            "run",
            self.example("linear-timeseries.json"),
            "--data",
            self.linear,
            "--label",
            "y",
            "--json",
        )
        self.assertEqual((code, err), (0, ""))
        data = json.loads(out)
        self.assertEqual(
            data["n_test"], 96
        )  # 4 folds over 5 blocks of 24 rows; block 0 only trains
        self.assertIn("rmse_std", data["metrics"])

    def test_grouped_validation_keeps_subjects_together(self):
        lines = self.blobs.read_text(encoding="utf-8").splitlines()
        grouped = self.dir / "subjects.csv"
        body = [f"s{i % 10},{line}" for i, line in enumerate(lines[1:])]
        grouped.write_text("\n".join(["subject," + lines[0], *body]) + "\n", encoding="utf-8")
        code, out, err = self.cli(
            "run",
            self.example("subjects-grouped.json"),
            "--data",
            grouped,
            "--label",
            "label",
            "--json",
        )
        self.assertEqual((code, err), (0, ""))
        data = json.loads(out)
        self.assertEqual(data["n_test"], 120)  # every row of every subject is tested once
        self.assertGreater(data["metrics"]["accuracy"], 0.9)

    def test_cross_validate_the_forest(self):
        code, out, err = self.cli(
            "run",
            self.example("forest-cv.json"),
            "--data",
            self.blobs,
            "--label",
            "label",
            "--json",
        )
        self.assertEqual((code, err), (0, ""))
        data = json.loads(out)
        self.assertEqual(data["n_test"], 120)  # every row is tested once
        self.assertIn("accuracy_std", data["metrics"])

    def test_the_sweep_and_the_search_rank_their_runs(self):
        code, out, err = self.cli(
            "sweep",
            self.example("knn-sweep.json"),
            "--data",
            self.blobs,
            "--label",
            "label",
            "--json",
        )
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(len(json.loads(out)["entries"]), 8)
        code, out, err = self.cli(
            "search",
            self.example("knn-search.json"),
            "--data",
            self.blobs,
            "--label",
            "label",
            "--json",
        )
        self.assertEqual((code, err), (0, ""))
        self.assertTrue(1 <= len(json.loads(out)["entries"]) <= 8)

    def test_the_halving_example_narrows_twelve_forests_to_one(self):
        code, out, err = self.cli(
            "halve",
            self.example("forest-halving.json"),
            "--data",
            self.blobs,
            "--label",
            "label",
            "--json",
        )
        self.assertEqual((code, err), (0, ""))
        data = json.loads(out)
        self.assertEqual([r["resource_value"] for r in data["rounds"]], [3, 9, 27])
        sizes = [len(r["entries"]) for r in data["rounds"]]
        self.assertEqual(sizes[1], -(-sizes[0] // 3))
        self.assertEqual(sizes[2], -(-sizes[1] // 3))
        self.assertLessEqual(sizes[0], 12)
        self.assertEqual(data["best"], data["rounds"][-1]["entries"][0]["run_id"])

    def test_the_nested_example_reports_an_honest_estimate(self):
        code, out, err = self.cli(
            "nested", self.example("knn-nested.json"), "--data", self.blobs, "--label", "label",
            "--json",
        )  # fmt: skip
        self.assertEqual((code, err), (0, ""))
        data = json.loads(out)
        self.assertEqual(len(data["outer"]), 5)
        self.assertEqual(len(data["candidates"]), 10)
        self.assertEqual(sum(data["choice_counts"]), 5)
        self.assertEqual(sum(f["test_size"] for f in data["outer"]), 120)
        self.assertEqual(set(data["metrics"]), {"accuracy", "f1"})
        self.assertGreater(data["estimate"]["mean"], 0.9)

    def test_the_selection_example_picks_columns_and_reports_the_optimism_warning(self):
        code, out, err = self.cli(
            "select", self.example("linear-select.json"), "--data", self.linear, "--label", "y",
            "--json",
        )  # fmt: skip
        self.assertEqual((code, err), (0, ""))
        data = json.loads(out)
        self.assertEqual(data["direction"], "forward")
        self.assertFalse(data["higher_is_better"])
        self.assertTrue(data["selected"])
        self.assertLessEqual(data["score"], data["steps"][0]["score"])
        code, text, _ = self.cli(
            "select", self.example("linear-select.json"), "--data", self.linear, "--label", "y"
        )  # fmt: skip
        self.assertIn("the score is optimistic", text)

    def test_the_ridge_sweep_prefers_a_small_penalty(self):
        code, out, err = self.cli(
            "sweep",
            self.example("ridge-sweep.json"),
            "--data",
            self.linear,
            "--label",
            "y",
            "--json",
        )
        self.assertEqual((code, err), (0, ""))
        data = json.loads(out)
        best = next(e for e in data["entries"] if e["run_id"] == data["best"])
        self.assertLessEqual(best["params"]["l2"], 1)
        worst = max(e["metrics"]["mae"] for e in data["entries"])
        self.assertGreater(worst, best["metrics"]["mae"])

    def test_a_stored_example_run_reproduces_and_reports(self):
        self.cli("run", self.example("knn.json"), "--data", self.blobs, "--label", "label")
        listing = json.loads(self.cli("list", "--experiment", "blobs-knn", "--json")[1])
        run_id = listing[0]["run_id"]
        code, out, _ = self.cli("repro", run_id, "--data", self.blobs, "--label", "label")
        self.assertEqual(code, 0, out)
        target = self.dir / "report.html"
        code, out, _ = self.cli("report", "blobs-knn", "-o", target)
        self.assertEqual(code, 0)
        self.assertIn("blobs-knn", target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
