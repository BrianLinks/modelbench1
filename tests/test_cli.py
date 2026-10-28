import contextlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from modelbench.cli import _safe, build_parser, main
from modelbench.store import MIGRATIONS, Store

SRC = str(Path(__file__).resolve().parent.parent / "src")


def blobs_csv():
    lines = ["x1,x2,label"]
    lines += [f"{i * 0.1:.1f},0,a" for i in range(20)]
    lines += [f"{10 + i * 0.1:.1f},5,b" for i in range(20)]
    return "\n".join(lines) + "\n"


def config_text(**overrides):
    config = {
        "experiment_id": "exp-a",
        "model": "knn",
        "seed": 7,
        "dataset_id": "blobs",
        "metrics": ["accuracy", "f1"],
        "params": {"k": 3},
    }
    config.update(overrides)
    return json.dumps(config)


class CliTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.db = str(self.dir / "runs.db")
        self.data = self.dir / "blobs.csv"
        self.data.write_text(blobs_csv())
        self.config = self.dir / "config.json"
        self.config.write_text(config_text())

    def cli(self, *args, db=True):
        argv = (["--db", self.db] if db else []) + [str(a) for a in args]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(argv)
            except SystemExit as exc:  # argparse exits on bad arguments
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def run_experiment(self, config=None, **extra):
        path = config or self.config
        code, out, err = self.cli(
            "run", path, "--data", self.data, "--label", "label", "--json", *extra.get("args", [])
        )
        self.assertEqual((code, err), (0, ""), msg=out)
        return json.loads(out)


class RunAndListTests(CliTestCase):
    def test_run_prints_text_and_records_the_run(self):
        code, out, err = self.cli("run", self.config, "--data", self.data, "--label", "label")
        self.assertEqual((code, err), (0, ""))
        lines = out.splitlines()
        self.assertEqual(lines[0], "run-000001")
        self.assertEqual(lines[1], "train 32  test 8")
        self.assertEqual(lines[2:], ["accuracy  1.0000", "f1  1.0000"])

    def test_run_json(self):
        result = self.run_experiment()
        self.assertEqual(result["run_id"], "run-000001")
        self.assertEqual(result["metrics"], {"accuracy": 1.0, "f1": 1.0})
        self.assertEqual((result["n_train"], result["n_test"], result["reused"]), (32, 8, False))
        self.assertEqual(len(result["predictions_sha256"]), 64)

    def test_reuse_flag(self):
        self.run_experiment()
        again = self.run_experiment(args=["--reuse"])
        self.assertTrue(again["reused"])
        self.assertEqual(again["run_id"], "run-000001")
        fresh = self.run_experiment()
        self.assertEqual(fresh["run_id"], "run-000002")

    def test_features_and_artifact_dir(self):
        art = self.dir / "art"
        code, out, err = self.cli(
            "run", self.config, "--data", self.data, "--label", "label",
            "--features", "x1", "--artifact-dir", art, "--json",
        )  # fmt: skip
        self.assertEqual((code, err), (0, ""))
        self.assertTrue((art / "run-000001" / "predictions.json").is_file())

    def test_list_shows_newest_first(self):
        self.run_experiment()
        self.run_experiment()
        code, out, _ = self.cli("list")
        self.assertEqual(code, 0)
        lines = out.splitlines()
        header = ["RUN", "EXPERIMENT", "MODEL", "STATUS", "STARTED", "METRICS"]
        self.assertEqual(lines[0].split(), header)
        self.assertEqual([line.split()[0] for line in lines[1:]], ["run-000002", "run-000001"])
        self.assertIn("accuracy=1.0000 f1=1.0000", lines[1])
        self.assertIn("knn", lines[1])

    def test_list_filters_and_json(self):
        self.run_experiment()
        other = self.dir / "other.json"
        other.write_text(config_text(experiment_id="exp-b"))
        self.run_experiment(config=other)
        records = json.loads(self.cli("list", "--experiment", "exp-b", "--json")[1])
        self.assertEqual([r["run_id"] for r in records], ["run-000002"])
        self.assertEqual(records[0]["model"], "knn")
        self.assertEqual(json.loads(self.cli("list", "--status", "failed", "--json")[1]), [])
        self.assertEqual(self.cli("list", "--status", "failed")[1], "no runs\n")

    def test_list_limit(self):
        for _ in range(3):
            self.run_experiment()
        records = json.loads(self.cli("list", "--limit", "2", "--json")[1])
        self.assertEqual(len(records), 2)
        code, _, err = self.cli("list", "--limit", "0")
        self.assertEqual(code, 2)
        self.assertIn("limit", err)


class ShowCompareTests(CliTestCase):
    def test_show_text_and_json(self):
        self.run_experiment()
        self.cli("note", "run-000001", "a first note")
        code, out, _ = self.cli("show", "run-000001")
        self.assertEqual(code, 0)
        self.assertIn("run-000001  completed  experiment exp-a", out)
        self.assertIn("dataset blobs@", out)
        self.assertIn("(40 rows)", out)
        self.assertIn('model   knn  seed 7  params {"k": 3}', out)
        self.assertIn("metric  accuracy (step 0)  1.0000", out)
        self.assertIn("note    ", out)
        self.assertIn("a first note", out)
        record = json.loads(self.cli("show", "run-000001", "--json")[1])
        self.assertEqual(record["notes"][0]["text"], "a first note")
        self.assertEqual(record["status"], "completed")

    def test_show_escapes_control_characters_in_stored_text(self):
        self.run_experiment()
        self.cli("note", "run-000001", "bad \x1b[31mred\x1b[0m\x07 text")
        _, out, _ = self.cli("show", "run-000001")
        self.assertNotIn("\x1b", out)
        self.assertNotIn("\x07", out)
        self.assertIn("\\x1b[31mred", out)

    def test_show_failed_run_prints_its_error_and_json_keys_are_sorted(self):
        self.run_experiment()
        with Store(self.db) as store:
            version = store.get_run("run-000001").dataset_version
            failed = store.create_run(
                store.get_config("run-000001"), store.get_dataset("blobs", version), {}
            )
            store.fail_run(failed, "boom \x1b[31m")
        out = self.cli("show", failed)[1]
        self.assertIn("error   boom \\x1b[31m", out)
        keys = list(json.loads(self.cli("show", failed, "--json")[1]))
        self.assertEqual(keys, sorted(keys))

    def test_show_unknown_run(self):
        self.run_experiment()
        code, out, err = self.cli("show", "run-000099")
        self.assertEqual((code, out), (2, ""))
        self.assertTrue(err.startswith("error: "))
        self.assertIn("run-000099", err)

    def test_compare_text(self):
        self.run_experiment()
        worse = self.dir / "majority.json"
        worse.write_text(config_text(model="majority", params={}))
        self.run_experiment(config=worse)
        code, out, err = self.cli("compare", "run-000001", "run-000002")
        self.assertEqual((code, err), (0, ""))
        self.assertIn("baseline run-000001", out)
        self.assertIn("rank accuracy: 1. run-000001 (1.0000), 2. run-000002 (0.3750)", out)
        self.assertIn("run-000002 vs run-000001", out)
        self.assertIn("accuracy: 1.0000 -> 0.3750 (-0.6250) worse", out)
        self.assertIn("same config False, dataset True, seed True", out)

    def test_compare_marks_a_metric_a_run_does_not_have(self):
        self.run_experiment()
        accuracy_only = self.dir / "acc.json"
        accuracy_only.write_text(config_text(metrics=["accuracy"]))
        self.run_experiment(config=accuracy_only)
        out = self.cli("compare", "run-000001", "run-000002")[1]
        self.assertIn("run-000002  1.0000    -", out)

    def test_compare_json_and_metric_selection(self):
        self.run_experiment()
        self.run_experiment()
        code, out, _ = self.cli(
            "compare", "run-000001", "run-000002", "--metrics", "accuracy", "--json"
        )
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(data["baseline"], "run-000001")
        self.assertEqual(list(data["values"]["run-000002"]), ["accuracy"])

    def test_compare_errors_are_reported_not_raised(self):
        self.run_experiment()
        code, _, err = self.cli("compare", "run-000001", "run-000099")
        self.assertEqual(code, 2)
        self.assertTrue(err.startswith("error: "))
        code, _, err = self.cli("compare", "run-000001", "--metrics", "a,,b")
        self.assertEqual(code, 2)
        self.assertIn("--metrics", err)


class ReproNoteExportRecoverTests(CliTestCase):
    def test_repro_reproduced_exit_zero(self):
        self.run_experiment()
        code, out, err = self.cli("repro", "run-000001", "--data", self.data, "--label", "label")
        self.assertEqual((code, err), (0, ""))
        self.assertIn("run-000001: reproduced", out)
        self.assertIn("accuracy: 1.0000 -> 1.0000  ok", out)

    def test_repro_not_reproduced_exit_one(self):
        self.run_experiment()
        with Store(self.db) as store, store._conn:
            store._conn.execute(
                "UPDATE tags SET value = ? WHERE key = 'predictions_sha256'", ("0" * 64,)
            )
        code, out, err = self.cli("repro", "run-000001", "--data", self.data, "--label", "label")
        self.assertEqual((code, err), (1, ""))
        self.assertIn("not_reproduced", out)
        self.assertIn("predictions differ", out)

    def test_repro_json_and_changed_data(self):
        self.run_experiment()
        report = json.loads(
            self.cli("repro", "run-000001", "--data", self.data, "--label", "label", "--json")[1]
        )
        self.assertEqual(report["verdict"], "reproduced")
        changed = self.dir / "changed.csv"
        changed.write_text(blobs_csv().replace("0.0,0,a", "9.9,0,a"))
        code, out, _ = self.cli("repro", "run-000001", "--data", changed, "--label", "label")
        self.assertEqual(code, 1)
        self.assertIn("dataset_changed", out)

    def test_repro_bad_tolerance(self):
        self.run_experiment()
        code, _, err = self.cli(
            "repro", "run-000001", "--data", self.data, "--label", "label", "--tolerance", "-1"
        )
        self.assertEqual(code, 2)
        self.assertIn("tolerance", err)

    def test_dataset_manifest_prints_deterministic_json(self):
        code, out, err = self.cli(
            "dataset", self.data, "--id", "blobs", "--label", "label", "--manifest", db=False
        )
        self.assertEqual((code, err), (0, ""))
        manifest = json.loads(out)
        self.assertEqual(manifest["format"], "modelbench-manifest")
        self.assertEqual((manifest["n_rows"], manifest["labels"]), (40, {"a": 20, "b": 20}))
        self.assertEqual([c["name"] for c in manifest["columns"]], ["x1", "x2", "label"])
        self.assertEqual(manifest["columns"][1]["max"], 5.0)
        again = self.cli(
            "dataset", self.data, "--id", "blobs", "--label", "label", "--manifest", db=False
        )
        self.assertEqual(out, again[1])
        info = json.loads(self.cli("dataset", self.data, "--id", "blobs", "--json", db=False)[1])
        self.assertEqual(manifest["version"], info["version"])

    def test_dataset_manifest_can_be_written_to_a_file(self):
        target = self.dir / "manifest.json"
        code, out, _ = self.cli(
            "dataset", self.data, "--id", "blobs", "--manifest", "-o", target, db=False
        )
        self.assertEqual(code, 0)
        self.assertIn("wrote the manifest of blobs", out)
        self.assertEqual(json.loads(target.read_text())["dataset_id"], "blobs")
        self.assertNotIn("labels", json.loads(target.read_text()))
        code, _, err = self.cli("dataset", self.data, "--id", "blobs", "-o", target, db=False)
        self.assertEqual(code, 2)
        self.assertIn("only goes with --manifest", err)

    def test_note_command(self):
        self.run_experiment()
        code, out, _ = self.cli("note", "run-000001", "hello")
        self.assertEqual((code, out), (0, "added a note to run-000001\n"))
        code, _, err = self.cli("note", "run-000001", "   ")
        self.assertEqual(code, 2)
        self.assertIn("note", err)

    def test_note_command_for_an_experiment(self):
        self.run_experiment()
        code, out, _ = self.cli("note", "--experiment", "exp-a", "the baseline sweep")
        self.assertEqual((code, out), (0, "added a note to experiment exp-a\n"))
        code, _, err = self.cli("note", "--experiment", "nope", "x")
        self.assertEqual(code, 2)
        self.assertIn("no experiment nope", err)
        code, _, err = self.cli("note", "exp-a", "x")  # an experiment id is not a run id
        self.assertEqual(code, 2)
        with Store(self.db, read_only=True) as store:
            notes = [n.text for n in store.get_experiment_notes("exp-a")]
        self.assertEqual(notes, ["the baseline sweep"])

    def test_export_json_to_stdout_and_file(self):
        self.run_experiment()
        code, out, _ = self.cli("export", "run-000001")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["format"], "modelbench-runs")
        target = self.dir / "out.json"
        code, msg, _ = self.cli("export", "run-000001", "-o", target)
        self.assertEqual(code, 0)
        self.assertEqual(msg, f"wrote 1 run(s) to {target}\n")
        self.assertEqual(target.read_text(), out)

    def test_export_csv(self):
        self.run_experiment()
        code, out, _ = self.cli("export", "run-000001", "--format", "csv")
        self.assertEqual(code, 0)
        self.assertEqual(out.splitlines()[0], "run_id,experiment_id,metric,step,value")
        self.assertIn("run-000001,exp-a,accuracy,0,1.0", out)

    def test_export_errors(self):
        self.run_experiment()
        code, _, err = self.cli("export", "run-000001", "run-000001")
        self.assertEqual(code, 2)
        self.assertIn("repeat", err)
        code, _, err = self.cli("export", "run-000001", "-o", self.dir / "nope" / "x.json")
        self.assertEqual(code, 2)
        self.assertTrue(err.startswith("error: "))

    def test_recover_respects_the_cutoff(self):
        self.run_experiment()
        ten_minutes_ago = (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        with Store(self.db, clock=lambda: ten_minutes_ago) as store:
            version = store.get_run("run-000001").dataset_version
            run_id = store.create_run(
                store.get_config("run-000001"), store.get_dataset("blobs", version), {}
            )
        self.assertEqual(self.cli("recover", "--older-than", "3600")[1], "no abandoned runs\n")
        out = self.cli("recover", "--older-than", "60")[1]
        self.assertIn(f"marked 1 abandoned run(s) as failed: {run_id}", out)

    def test_recover(self):
        self.run_experiment()
        with Store(self.db, clock=lambda: "2000-01-01T00:00:00Z") as store:
            run_id = store.create_run(
                store.get_config("run-000001"),
                store.get_dataset("blobs", store.get_run("run-000001").dataset_version),
                {},
            )
        code, out, _ = self.cli("recover", "--older-than", "3600")
        self.assertEqual(code, 0)
        self.assertIn(f"marked 1 abandoned run(s) as failed: {run_id}", out)
        self.assertEqual(self.cli("recover", "--older-than", "3600")[1], "no abandoned runs\n")
        self.assertIn("failed", self.cli("show", run_id)[1])


class SampleAndDemoTests(CliTestCase):
    def test_sample_writes_csv_to_stdout_or_a_file(self):
        from modelbench.samples import generate, table_to_csv

        args = ("sample", "moons", "--seed", "5", "--rows-per-class", "10")
        code, out, err = self.cli(*args, db=False)
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(out, table_to_csv(generate("moons", 5, rows_per_class=10)))
        target = self.dir / "moons.csv"
        code, msg, _ = self.cli(
            "sample", "moons", "--seed", "5", "--rows-per-class", "10", "-o", target, db=False
        )
        self.assertEqual((code, msg), (0, f"wrote 20 rows to {target}\n"))
        self.assertEqual(target.read_text(), out)

    def test_sample_options_reach_the_generator(self):
        _, out, _ = self.cli("sample", "blobs", "--classes", "2", "--features", "3",
                             "--rows-per-class", "4", "--separation", "10", "--spread", "0",
                             db=False)  # fmt: skip
        lines = out.splitlines()
        self.assertEqual(lines[0], "x1,x2,x3,label")
        self.assertEqual(len(lines), 9)
        self.assertEqual({line.split(",")[-1] for line in lines[1:]}, {"c0", "c1"})

    def test_sample_rejects_options_the_dataset_does_not_take_and_unknown_names(self):
        code, out, err = self.cli("sample", "moons", "--classes", "3", db=False)
        self.assertEqual((code, out), (2, ""))
        self.assertIn("does not take", err)
        code, _, err = self.cli("sample", "spirals", db=False)
        self.assertEqual(code, 2)
        code, _, err = self.cli("sample", "xor", "--noise", "nan", db=False)
        self.assertEqual(code, 2)
        self.assertIn("noise", err)
        self.assertEqual(self.cli("sample", "xor", "--seed", "-1", db=False)[0], 2)

    def test_demo_fills_the_database_and_is_idempotent(self):
        code, out, err = self.cli("demo", "--rows-per-class", "30")
        self.assertEqual((code, err), (0, ""))
        lines = out.splitlines()
        self.assertEqual(lines[0].split(), ["DATASET", "MODEL", "PARAMS", "ACCURACY", "F1", "RUN"])
        self.assertEqual(len(lines), 29)  # header, 27 runs, summary
        self.assertEqual(lines[-1], f"27 runs (27 new, 0 reused) in {self.db}")
        self.assertIn('{"k":5}', out)
        again = self.cli("demo", "--rows-per-class", "30")[1]
        self.assertTrue(again.splitlines()[-1].startswith("27 runs (0 new, 27 reused)"))
        records = json.loads(self.cli("list", "--limit", "100", "--json")[1])
        self.assertEqual(len(records), 27)

    def test_defaults_match_the_documentation(self):
        parser = build_parser()
        sample = parser.parse_args(["sample", "xor"])
        self.assertEqual((sample.seed, sample.output, sample.rows_per_class), (42, None, None))
        demo = parser.parse_args(["demo"])
        self.assertEqual((demo.seed, demo.rows_per_class), (42, 100))
        _, out, _ = self.cli("sample", "xor", "--rows-per-class", "5", db=False)
        from modelbench.samples import generate, table_to_csv

        self.assertEqual(out, table_to_csv(generate("xor", 42, rows_per_class=5)))

    def test_demo_table_shows_a_dash_for_models_without_parameters(self):
        out = self.cli("demo", "--rows-per-class", "30")[1]
        majority = [line.split() for line in out.splitlines() if " majority " in line]
        self.assertEqual(len(majority), 3)
        self.assertTrue(all(cells[2] == "-" for cells in majority))
        knn = [line.split() for line in out.splitlines() if " knn " in line]
        self.assertEqual(len(knn), 9)
        self.assertEqual({cells[2] for cells in knn}, {'{"k":1}', '{"k":5}', '{"k":15}'})

    def test_demo_json_reports_which_runs_were_reused(self):
        first = json.loads(self.cli("demo", "--rows-per-class", "30", "--json")[1])
        second = json.loads(self.cli("demo", "--rows-per-class", "30", "--json")[1])
        self.assertEqual({r["reused"] for r in first}, {False})
        self.assertEqual({r["reused"] for r in second}, {True})

    def test_demo_json_and_bad_arguments(self):
        runs = json.loads(self.cli("demo", "--rows-per-class", "30", "--json")[1])
        self.assertEqual(len(runs), 27)
        self.assertEqual({r["dataset"] for r in runs}, {"moons", "xor", "blobs"})
        self.assertTrue(all(0.0 <= r["metrics"]["accuracy"] <= 1.0 for r in runs))
        code, _, err = self.cli("demo", "--rows-per-class", "5")
        self.assertEqual(code, 2)
        self.assertIn("rows_per_class", err)

    def test_demo_runs_can_be_reproduced_from_a_regenerated_sample(self):
        runs = json.loads(self.cli("demo", "--rows-per-class", "30", "--json")[1])
        knn = next(r for r in runs if r["dataset"] == "moons" and r["params"] == {"k": 5})
        csv_path = self.dir / "moons.csv"
        self.cli("sample", "moons", "--seed", "42", "--rows-per-class", "30", "-o", csv_path,
                 db=False)  # fmt: skip
        code, out, err = self.cli("repro", knn["run_id"], "--data", csv_path, "--label", "label")
        self.assertEqual((code, err), (0, ""), msg=out)
        self.assertIn(f"{knn['run_id']}: reproduced", out)
        other = self.dir / "other.csv"
        self.cli("sample", "moons", "--seed", "43", "--rows-per-class", "30", "-o", other,
                 db=False)  # fmt: skip
        code, out, _ = self.cli("repro", knn["run_id"], "--data", other, "--label", "label")
        self.assertEqual(code, 1)
        self.assertIn("dataset_changed", out)


class ConsoleEncodingTests(CliTestCase):
    def test_text_the_console_cannot_encode_is_replaced_not_fatal(self):
        self.run_experiment()
        self.cli("note", "run-000001", "checked \u2713 caf\u00e9")
        console = io.TextIOWrapper(io.BytesIO(), encoding="ascii", write_through=True)
        with mock.patch.object(sys, "stdout", console):
            code = main(["--db", self.db, "show", "run-000001"])
        self.assertEqual(code, 0)
        written = console.buffer.getvalue()
        self.assertIn(b"checked ? caf?", written)
        self.assertNotIn(b"Traceback", written)


class ServeCommandTests(CliTestCase):
    def test_serve_passes_its_options_through(self):
        self.run_experiment()
        with mock.patch("modelbench.cli.serve") as served:
            code, out, err = self.cli("serve", "--port", "9001")
            self.assertEqual((code, out, err), (0, "", ""))
            served.assert_called_once_with(self.db, "127.0.0.1", 9001, allow_network=False)
            served.reset_mock()
            self.cli("serve", "--host", "0.0.0.0", "--allow-network")
            served.assert_called_once_with(self.db, "0.0.0.0", 8765, allow_network=True)

    def test_serve_needs_an_existing_database_and_does_not_create_one(self):
        code, out, err = self.cli("serve")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("no database", err)
        self.assertFalse(Path(self.db).exists())

    def test_serve_refuses_a_network_address_without_the_flag(self):
        self.run_experiment()
        code, out, err = self.cli("serve", "--host", "0.0.0.0", "--port", "0")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("no login", err)

    def test_serve_rejects_a_bad_port(self):
        self.run_experiment()
        code, _, err = self.cli("serve", "--port", "70000")
        self.assertEqual(code, 2)
        self.assertIn("port", err)

    def test_serve_does_not_upgrade_an_old_database(self):
        old = self.dir / "old.db"
        conn = sqlite3.connect(old)
        conn.executescript(f"BEGIN;{MIGRATIONS[0]}PRAGMA user_version = 1;COMMIT;")
        conn.close()
        code, _, err = self.cli("--db", old, "serve", db=False)
        self.assertEqual(code, 2)
        self.assertIn("schema version", err)
        check = sqlite3.connect(old)
        self.addCleanup(check.close)
        self.assertEqual(check.execute("PRAGMA user_version").fetchone()[0], 1)


class ErrorHandlingTests(CliTestCase):
    def test_missing_database_is_not_created_by_read_commands(self):
        for args in (["list"], ["show", "run-000001"], ["compare", "run-000001"],
                     ["note", "run-000001", "x"], ["export", "run-000001"],
                     ["recover", "--older-than", "5"]):  # fmt: skip
            with self.subTest(args=args):
                code, out, err = self.cli(*args)
                self.assertEqual((code, out), (2, ""))
                self.assertIn("no database", err)
        self.assertFalse(Path(self.db).exists())

    def test_file_that_is_not_a_database(self):
        Path(self.db).write_text("this is plain text, not an SQLite file " * 5)
        code, out, err = self.cli("list")
        self.assertEqual((code, out), (2, ""))
        self.assertTrue(err.startswith("error: "))
        self.assertNotIn("Traceback", err)

    def test_bad_config_and_data(self):
        missing = self.dir / "missing.json"
        code, _, err = self.cli("run", missing, "--data", self.data, "--label", "label")
        self.assertEqual(code, 2)
        self.assertIn("config", err)
        no_csv = self.dir / "no.csv"
        code, _, err = self.cli("run", self.config, "--data", no_csv, "--label", "label")
        self.assertEqual(code, 2)
        code, _, err = self.cli("run", self.config, "--data", self.data, "--label", "nope")
        self.assertEqual(code, 2)
        self.assertIn("nope", err)
        code, _, err = self.cli("run", self.config, "--data", self.data, "--label", "label",
                                "--features", "x1,,x2")  # fmt: skip
        self.assertEqual(code, 2)
        self.assertIn("--features", err)

    def test_impossible_metric_is_rejected_before_a_run_exists(self):
        bad = self.dir / "roc.json"
        bad.write_text(config_text(metrics=["roc_auc"], model="majority", params={}))
        # Three classes make roc_auc impossible; the check happens before a run exists.
        three = self.dir / "three.csv"
        three.write_text("x,label\n" + "\n".join(f"{i},{'abc'[i % 3]}" for i in range(30)) + "\n")
        code, _, err = self.cli("run", bad, "--data", three, "--label", "label")
        self.assertEqual(code, 2)
        self.assertIn("roc_auc", err)
        self.assertFalse(Path(self.db).exists() and "run-000001" in self.cli("list")[1])

    def test_error_messages_are_escaped(self):
        hostile = self.dir / "x\x1b[31m.json"
        code, _, err = self.cli("run", hostile, "--data", self.data, "--label", "label")
        self.assertEqual(code, 2)
        self.assertNotIn("\x1b", err)
        self.assertIn("\\x1b[31m", err)

    def test_sqlite_errors_at_runtime_are_reported_not_raised(self):
        self.run_experiment()
        locked = sqlite3.OperationalError("database is locked")
        with mock.patch("modelbench.cli.Store.list_runs", side_effect=locked):
            code, out, err = self.cli("list")
        self.assertEqual((code, out), (2, ""))
        self.assertEqual(err, "error: database is locked\n")

    def test_file_system_errors_are_reported_and_the_run_is_marked_failed(self):
        not_a_directory = self.dir / "file.txt"
        not_a_directory.write_text("x")
        code, out, err = self.cli(
            "run", self.config, "--data", self.data, "--label", "label",
            "--artifact-dir", not_a_directory,
        )  # fmt: skip
        self.assertEqual((code, out), (2, ""))
        self.assertTrue(err.startswith("error: "))
        self.assertNotIn("Traceback", err)
        self.assertIn("failed", self.cli("show", "run-000001")[1])

    def test_argparse_errors_exit_two(self):
        for args in (["bogus"], ["run"], ["list", "--status", "weird"], []):
            with self.subTest(args=args):
                code, out, err = self.cli(*args)
                self.assertEqual(code, 2)
                self.assertNotEqual(err, "")

    def test_sample_linear_options(self):
        code, out, _ = self.cli("sample", "linear", "--rows", "3", "--features", "1",
                                "--noise", "0", db=False)  # fmt: skip
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertEqual(lines[0], "x1,y")
        self.assertEqual(len(lines), 4)
        for line in lines[1:]:
            x, y = (float(v) for v in line.split(","))
            self.assertAlmostEqual(y, 4.0 + x, delta=2e-6)

    def test_models_and_dataset_commands(self):
        code, out, _ = self.cli("models", db=False)
        self.assertEqual(code, 0)
        self.assertEqual(
            out.splitlines(),
            [
                "bagged_trees  classification",
                "decision_tree  classification",
                "decision_tree_regressor  regression",
                "gaussian_nb  classification",
                "knn  classification",
                "knn_regressor  regression",
                "linear_regression  regression",
                "logistic_regression  classification",
                "majority  classification",
                "mean  regression",
                "perceptron  classification",
                "stratified  classification",
            ],
        )
        code, out, _ = self.cli("dataset", self.data, "--id", "blobs", "--label", "label", db=False)
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("blobs@"))
        self.assertIn("rows     40", out)
        self.assertIn("columns  x1, x2, label", out)
        self.assertIn("label    label", out)
        info = json.loads(self.cli("dataset", self.data, "--id", "blobs", "--json", db=False)[1])
        self.assertEqual((info["n_rows"], info["label_column"]), (40, None))
        code, _, err = self.cli("dataset", self.data, "--id", "Bad Id", db=False)
        self.assertEqual(code, 2)


class ShowMatrixTests(CliTestCase):
    def test_show_prints_the_matrix_as_a_grid_instead_of_a_raw_tag(self):
        self.run_experiment()
        code, out, err = self.cli("show", "run-000001")
        self.assertEqual((code, err), (0, ""))
        self.assertIn("matrix  true/pred  a b", out)
        self.assertIn("matrix  a          5 0", out)
        self.assertIn("matrix  b          0 3", out)
        self.assertNotIn("confusion_matrix=", out)
        self.assertIn("tag     n_test=8", out)

    def test_show_json_keeps_the_raw_tag(self):
        self.run_experiment()
        run = json.loads(self.cli("show", "run-000001", "--json")[1])
        self.assertIn("confusion_matrix", run["tags"])

    def test_a_damaged_tag_does_not_break_show(self):
        self.run_experiment()
        with Store(self.db) as store:
            store._conn.execute("UPDATE tags SET value = '{oops' WHERE key = 'confusion_matrix'")
            store._conn.commit()
        code, out, _ = self.cli("show", "run-000001")
        self.assertEqual(code, 0)
        self.assertIn("matrix  unreadable:", out)


class ImportCommandTests(CliTestCase):
    def setUp(self):
        super().setUp()
        self.run_experiment()
        self.run_experiment(self._config(model="majority", params={}))
        self.export_path = self.dir / "runs.json"
        code, _, err = self.cli("export", "run-000001", "run-000002", "-o", self.export_path)
        self.assertEqual((code, err), (0, ""))
        self.other = str(self.dir / "other.db")

    def _config(self, **overrides):
        path = self.dir / f"c{len(list(self.dir.iterdir()))}.json"
        path.write_text(config_text(**overrides))
        return path

    def cli_other(self, *args):
        original, self.db = self.db, self.other
        try:
            return self.cli(*args)
        finally:
            self.db = original

    def test_import_creates_the_database_and_lists_the_mapping(self):
        code, out, err = self.cli_other("import", self.export_path)
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(
            out.splitlines(),
            [
                "run-000001 -> run-000001",
                "run-000002 -> run-000002",
                "imported 2 run(s), skipped 0 already present",
            ],
        )
        shown = json.loads(self.cli_other("show", "run-000001", "--json")[1])
        self.assertEqual(shown["tags"]["imported_from"], "run-000001")

    def test_a_second_import_skips_everything(self):
        self.cli_other("import", self.export_path)
        code, out, _ = self.cli_other("import", self.export_path)
        self.assertEqual(code, 0)
        self.assertIn("run-000001 already present as run-000001", out)
        self.assertTrue(out.endswith("imported 0 run(s), skipped 2 already present\n"))
        self.assertEqual(len(json.loads(self.cli_other("list", "--json")[1])), 2)

    def test_json_output(self):
        code, out, _ = self.cli_other("import", self.export_path, "--json")
        data = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(data["dry_run"], False)
        self.assertEqual(
            data["imported"],
            [
                {"source": "run-000001", "run_id": "run-000001"},
                {"source": "run-000002", "run_id": "run-000002"},
            ],
        )
        self.assertEqual(data["skipped"], [])
        again = json.loads(self.cli_other("import", self.export_path, "--json")[1])
        self.assertEqual(again["skipped"][0], {"source": "run-000001", "existing": "run-000001"})

    def test_dry_run_never_creates_or_changes_a_database(self):
        code, out, _ = self.cli_other("import", self.export_path, "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("run-000001 (new)", out)
        self.assertTrue(out.endswith("would import 2 run(s), skipped 0 already present\n"))
        self.assertFalse(Path(self.other).exists())
        data = json.loads(self.cli_other("import", self.export_path, "--dry-run", "--json")[1])
        self.assertEqual(data["dry_run"], True)
        self.assertEqual(data["imported"][0], {"source": "run-000001", "run_id": None})
        self.cli_other("import", self.export_path)
        out = self.cli_other("import", self.export_path, "--dry-run")[1]
        self.assertTrue(out.endswith("would import 0 run(s), skipped 2 already present\n"))

    def test_bad_files_exit_with_code_2_and_change_nothing(self):
        bad = self.dir / "bad.json"
        bad.write_text('{"format": "modelbench-runs", "version": 1, "runs": [1]}')
        for target in (bad, self.dir / "missing.json"):
            code, out, err = self.cli_other("import", target)
            self.assertEqual((code, out), (2, ""))
            self.assertNotEqual(err, "")
        self.assertFalse(Path(self.other).exists())  # no empty database left behind
        self.assertEqual(len(json.loads(self.cli("list", "--json")[1])), 2)
        self.assertEqual(self.cli_other("import", self.export_path)[0], 0)

    def test_help_lists_the_command(self):
        self.assertIn("import", build_parser().format_help())


class SafeTextTests(unittest.TestCase):
    def test_escapes_only_control_characters(self):
        self.assertEqual(_safe("plain text"), "plain text")
        self.assertEqual(_safe("café ✓"), "café ✓")
        self.assertEqual(_safe("a\nb\tc"), "a\\nb\\tc")
        self.assertEqual(_safe("\x1b[2J"), "\\x1b[2J")
        self.assertEqual(_safe("\x7f"), "\\x7f")
        self.assertEqual(_safe(None), "None")

    def test_every_command_has_help(self):
        parser = build_parser()
        with (
            contextlib.redirect_stdout(io.StringIO()) as out,
            self.assertRaises(SystemExit),
        ):
            parser.parse_args(["--help"])
        commands = ("run", "list", "show", "compare", "repro", "note", "export", "serve")
        for name in (*commands, "sample", "demo"):
            self.assertIn(name, out.getvalue())
        self.assertIn("recover", out.getvalue())


class EntryPointTests(unittest.TestCase):
    def test_python_dash_m_runs_and_returns_exit_codes(self):
        env = dict(os.environ, PYTHONPATH=SRC)
        ok = subprocess.run(
            [sys.executable, "-m", "modelbench", "models"], capture_output=True, text=True, env=env
        )
        self.assertEqual((ok.returncode, ok.stderr), (0, ""))
        self.assertIn("knn", ok.stdout)
        with tempfile.TemporaryDirectory() as tmp:
            bad = subprocess.run(
                [sys.executable, "-m", "modelbench", "--db", str(Path(tmp) / "x.db"), "list"],
                capture_output=True, text=True, env=env,
            )  # fmt: skip
        self.assertEqual(bad.returncode, 2)
        self.assertIn("no database", bad.stderr)


class SweepCommandTests(CliTestCase):
    def sweep_file(self, grid, name="sweep.json", **config):
        document = {"config": json.loads(config_text(params={}, **config)), "grid": grid}
        path = self.dir / name
        path.write_text(json.dumps(document))
        return path

    def sweep(self, grid, *extra, **config):
        return self.cli(
            "sweep",
            self.sweep_file(grid, **config),
            "--data",
            self.data,
            "--label",
            "label",
            *extra,
        )

    def test_text_output_ranks_the_combinations(self):
        code, out, err = self.sweep({"k": [1, 3]})
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(
            out.splitlines(),
            [
                "sweep exp-a: 2 combinations, ranked by accuracy (higher is better)",
                "",
                "rank  run         params  accuracy  f1",
                "1     run-000001  k=1     1.0000    1.0000",
                "1     run-000002  k=3     1.0000    1.0000",
                "",
                "best run-000001  k=1",
            ],
        )
        self.assertEqual(len(json.loads(self.cli("list", "--json")[1])), 2)

    def test_failures_are_listed_and_set_exit_code_one(self):
        code, out, err = self.sweep({"k": [0, 1]})
        self.assertEqual((code, err), (1, ""))
        lines = out.splitlines()
        self.assertEqual(lines[3], "1     run-000001  k=1     1.0000    1.0000")
        self.assertEqual(lines[4], "-     -           k=0     -         -")
        self.assertEqual(lines[5], "failed  k=0: k must be a positive integer, got 0")
        self.assertEqual(lines[-1], "best run-000001  k=1")

    def test_when_nothing_succeeds_there_is_no_best(self):
        code, out, _ = self.sweep({"k": [0]})
        self.assertEqual(code, 1)
        self.assertEqual(out.splitlines()[-1], "no run succeeded")

    def test_running_again_reuses_runs_unless_told_not_to(self):
        self.sweep({"k": [1, 3]})
        code, out, _ = self.sweep({"k": [1, 3]})
        self.assertEqual(code, 0)
        self.assertIn("2 combinations (2 reused), ranked", out.splitlines()[0])
        self.assertEqual(len(json.loads(self.cli("list", "--json")[1])), 2)
        self.sweep({"k": [1, 3]}, "--no-reuse")
        self.assertEqual(len(json.loads(self.cli("list", "--json")[1])), 4)

    def test_rank_by_and_direction(self):
        code, out, _ = self.sweep({"k": [1]}, "--rank-by", "f1")
        self.assertEqual(code, 0)
        self.assertIn("ranked by f1 (higher is better)", out)
        code, _, err = self.sweep({"k": [1]}, "--rank-by", "recall")
        self.assertEqual(code, 2)
        self.assertIn("cannot rank by 'recall'", err)

    def test_a_lower_is_better_metric_is_announced_as_such(self):
        code, out, _ = self.sweep(
            {"k": [1, 3]}, "--rank-by", "log_loss", metrics=["accuracy", "log_loss"]
        )
        self.assertEqual(code, 0)
        self.assertIn("ranked by log_loss (lower is better)", out)

    def test_json_output(self):
        code, out, _ = self.sweep({"k": [0, 1]}, "--json")
        self.assertEqual(code, 1)
        data = json.loads(out)
        self.assertEqual(data["experiment_id"], "exp-a")
        self.assertEqual(data["rank_metric"], "accuracy")
        self.assertEqual(data["best"], "run-000001")
        bad, good = data["entries"]
        self.assertEqual((bad["params"], bad["run_id"], bad["rank"]), ({"k": 0}, None, None))
        self.assertIn("positive integer", bad["error"])
        self.assertEqual(
            (good["params"], good["run_id"], good["rank"]), ({"k": 1}, "run-000001", 1)
        )
        self.assertEqual(good["metrics"], {"accuracy": 1.0, "f1": 1.0})
        self.assertFalse(good["reused"])

    def test_stop_on_error(self):
        code, out, _ = self.sweep({"k": [0, 1]}, "--stop-on-error")
        self.assertEqual(code, 1)
        self.assertEqual(out.splitlines()[-1], "no run succeeded")

    def test_cross_validated_sweep_creates_cv_runs(self):
        code, out, _ = self.sweep({"k": [1]}, cv_folds=4)
        self.assertEqual(code, 0)
        shown = json.loads(self.cli("show", "run-000001", "--json")[1])
        self.assertEqual(shown["tags"]["cv_folds"], "4")

    def test_stored_text_cannot_inject_terminal_escapes(self):
        code, out, _ = self.sweep({"metric": ["\x1b[31mred"]})
        self.assertEqual(code, 1)
        self.assertNotIn("\x1b", out)
        self.assertIn("\\x1b[31mred", out)

    def test_bad_input_is_an_error(self):
        code, _, err = self.cli(
            "sweep", self.dir / "nope.json", "--data", self.data, "--label", "label"
        )
        self.assertEqual(code, 2)
        self.assertIn("cannot read sweep file", err)
        code, _, err = self.cli("sweep", self.sweep_file({"k": [1]}), "--label", "label")
        self.assertEqual(code, 2)  # --data is required


class ImportanceCommandTests(CliTestCase):
    def setUp(self):
        super().setUp()
        lines = ["x1,x2,y"] + [f"{i},{(i * 7) % 11},{3.0 * i + 2.0}" for i in range(40)]
        self.line = self.dir / "line.csv"
        self.line.write_text("\n".join(lines) + "\n")
        path = self.dir / "ridge.json"
        path.write_text(
            config_text(
                model="linear_regression", params={}, metrics=["mse", "r2"], dataset_id="line",
                test_fraction=0.25, seed=5,
            )
        )  # fmt: skip
        code, out, err = self.cli("run", path, "--data", self.line, "--label", "y")
        self.assertEqual((code, err), (0, ""))

    def importance(self, *extra):
        return self.cli("importance", "run-000001", "--data", self.line, "--label", "y", *extra)

    def test_text_output_lists_features_best_first(self):
        code, out, err = self.importance("--metric", "mse", "--repeats", "3")
        self.assertEqual((code, err), (0, ""))
        lines = out.splitlines()
        self.assertEqual(
            lines[0],
            "run-000001: permutation importance for mse (lower is better), "
            "3 shuffle(s) per feature, baseline 0.0000",
        )
        self.assertEqual(lines[2].split(), ["rank", "feature", "importance", "std"])
        self.assertEqual(lines[3].split()[:2], ["1", "x1"])
        self.assertEqual(lines[4].split(), ["2", "x2", "0.0000", "0.0000"])

    def test_json_output(self):
        code, out, _ = self.importance("--json", "--seed", "9")
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual((data["run_id"], data["metric"]), ("run-000001", "mse"))
        self.assertFalse(data["higher_is_better"])
        self.assertEqual(data["repeats"], 5)
        self.assertEqual([f["feature"] for f in data["features"]], ["x1", "x2"])
        self.assertGreater(data["features"][0]["importance"], 100.0)

    def test_errors(self):
        code, _, err = self.importance("--metric", "mae")
        self.assertEqual(code, 2)
        self.assertIn("not one of the run's metrics", err)
        code, _, err = self.importance("--repeats", "0")
        self.assertEqual(code, 2)
        self.assertIn("repeats", err)
        code, _, err = self.cli("importance", "run-000009", "--data", self.line, "--label", "y")
        self.assertEqual(code, 2)
        self.assertIn("run-000009", err)
        other = self.dir / "other.csv"
        other.write_text("x1,x2,y\n1,1,1\n2,2,2\n3,3,3\n")
        code, _, err = self.cli("importance", "run-000001", "--data", other, "--label", "y")
        self.assertEqual(code, 2)
        self.assertIn("not the dataset version", err)


class LearningCurveCommandTests(CliTestCase):
    def curve(self, *extra, **config):
        path = self.dir / "lc.json"
        path.write_text(config_text(params={"k": 1}, stratify=True, **config))
        return self.cli("learning-curve", path, "--data", self.data, "--label", "label", *extra)

    def test_text_output(self):
        code, out, err = self.curve("--fractions", "0.25,0.5,1")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(
            out.splitlines(),
            [
                "learning curve for exp-a: accuracy, 1 split",
                "",
                "fraction  n_train  train   test",
                "0.25      8        1.0000  1.0000",
                "0.5       16       1.0000  1.0000",
                "1         32       1.0000  1.0000",
            ],
        )

    def test_cross_validation_adds_the_spread_column(self):
        code, out, _ = self.curve("--fractions", "1", cv_folds=4)
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertEqual(lines[0], "learning curve for exp-a: accuracy, 4 folds")
        self.assertEqual(lines[2].split(), ["fraction", "n_train", "train", "test", "test_std"])
        self.assertEqual(lines[3].split(), ["1", "30", "1.0000", "1.0000", "0.0000"])

    def test_two_folds_are_enough_for_the_spread_column(self):
        _, out, _ = self.curve("--fractions", "1", cv_folds=2)
        self.assertIn("test_std", out.splitlines()[2])
        self.assertIn("2 folds", out.splitlines()[0])

    def test_json_output_and_metric_choice(self):
        code, out, _ = self.curve(
            "--json", "--metric", "f1", "--fractions", "0.5,1", metrics=["accuracy", "f1"]
        )
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual((data["metric"], data["folds"]), ("f1", 1))
        self.assertEqual([p["fraction"] for p in data["points"]], [0.5, 1.0])
        self.assertEqual(data["points"][1]["n_train"], 32)

    def test_errors(self):
        for fractions, message in (("abc", "comma-separated numbers"), ("0.5,0.5", "increasing"),
                                   ("1.5", "(0, 1]"), ("0.5,,1", "no empty")):  # fmt: skip
            code, _, err = self.curve("--fractions", fractions)
            self.assertEqual(code, 2, fractions)
            self.assertIn(message, err)
        code, _, err = self.curve("--metric", "mcc")
        self.assertEqual(code, 2)
        self.assertIn("not one of the config's metrics", err)


class SearchCommandTests(CliTestCase):
    def search_file(self, space=None, trials=5, **config):
        document = {
            "config": json.loads(config_text(params={}, **config)),
            "space": space
            or {"k": {"int": [1, 9]}, "weights": {"choice": ["uniform", "distance"]}},
            "trials": trials,
        }
        path = self.dir / "search.json"
        path.write_text(json.dumps(document))
        return path

    def search(self, *extra, **kwargs):
        return self.cli(
            "search", self.search_file(**kwargs), "--data", self.data, "--label", "label", *extra
        )

    def test_text_output_lists_the_trials(self):
        code, out, err = self.search()
        self.assertEqual((code, err), (0, ""))
        lines = out.splitlines()
        self.assertTrue(lines[0].startswith("search exp-a: "), lines[0])
        self.assertTrue(lines[0].endswith(" trials, ranked by accuracy (higher is better)"))
        self.assertTrue(lines[-1].startswith("best run-"))
        self.assertEqual(
            len(json.loads(self.cli("list", "--json")[1])), int(lines[0].split(": ")[1].split()[0])
        )

    def test_the_same_file_draws_the_same_runs(self):
        first = json.loads(self.search("--json")[1])
        again = json.loads(self.search("--json")[1])
        self.assertEqual(
            [e["params"] for e in first["entries"]], [e["params"] for e in again["entries"]]
        )
        self.assertTrue(all(e["reused"] for e in again["entries"]))

    def test_failures_set_exit_code_one(self):
        code, out, _ = self.search(space={"k": {"int": [100, 101]}})
        self.assertEqual(code, 1)
        self.assertEqual(out.splitlines()[-1], "no run succeeded")

    def test_bad_files_are_errors(self):
        code, _, err = self.cli(
            "search", self.dir / "nope.json", "--data", self.data, "--label", "label"
        )
        self.assertEqual(code, 2)
        self.assertIn("cannot read search file", err)
        code, _, err = self.search(trials=0)
        self.assertEqual(code, 2)
        self.assertIn('"trials" must be a whole number', err)


class ReportCommandTests(CliTestCase):
    def setUp(self):
        super().setUp()
        self.run_experiment()
        (self.dir / "m.json").write_text(config_text(model="majority", params={}))
        self.run_experiment(self.dir / "m.json")

    def test_markdown_to_stdout(self):
        code, out, err = self.cli("report", "exp-a", "--format", "md")
        self.assertEqual((code, err), (0, ""))
        lines = out.splitlines()
        self.assertEqual(lines[0], "# Experiment exp-a")
        self.assertIn("Ranked by accuracy (higher is better).", out)
        self.assertEqual(lines[6].split(" | ")[1], "run-000001 (best)")

    def test_html_to_a_file_has_no_script(self):
        target = self.dir / "report.html"
        code, out, _ = self.cli("report", "exp-a", "-o", target)
        self.assertEqual(code, 0)
        self.assertEqual(out, f"wrote the report on 2 run(s) to {target}\n")
        html = target.read_text(encoding="utf-8")
        self.assertIn("<title>Experiment exp-a - ModelBench report</title>", html)
        self.assertNotIn("<script", html)

    def test_metric_choice_and_errors(self):
        code, out, _ = self.cli("report", "exp-a", "--format", "md", "--metric", "f1")
        self.assertIn("Ranked by f1", out)
        code, _, err = self.cli("report", "exp-a", "--metric", "nope")
        self.assertEqual(code, 2)
        self.assertIn("is not a metric of these runs", err)
        code, _, err = self.cli("report", "missing")
        self.assertEqual(code, 2)
        self.assertIn("has no completed runs", err)

    def test_the_same_runs_give_the_same_bytes(self):
        first = self.cli("report", "exp-a")[1]
        self.assertEqual(first, self.cli("report", "exp-a")[1])


if __name__ == "__main__":
    unittest.main()
