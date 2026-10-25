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

    def test_note_command(self):
        self.run_experiment()
        code, out, _ = self.cli("note", "run-000001", "hello")
        self.assertEqual((code, out), (0, "added a note to run-000001\n"))
        code, _, err = self.cli("note", "run-000001", "   ")
        self.assertEqual(code, 2)
        self.assertIn("note", err)

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
                store.get_config("run-000001"), store.get_dataset(
                    "blobs", store.get_run("run-000001").dataset_version
                ), {}
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
                "decision_tree  classification",
                "gaussian_nb  classification",
                "knn  classification",
                "linear_regression  regression",
                "logistic_regression  classification",
                "majority  classification",
                "mean  regression",
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
        with contextlib.redirect_stdout(io.StringIO()) as out:
            with self.assertRaises(SystemExit):
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


if __name__ == "__main__":
    unittest.main()
