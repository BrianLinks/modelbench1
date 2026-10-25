import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modelbench import importer
from modelbench.config import ExperimentConfig
from modelbench.datasets import Table, make_version
from modelbench.errors import ModelBenchError, RunImportError, StoreError
from modelbench.export import export_json, export_runs
from modelbench.importer import (
    IMPORT_TAG,
    import_file,
    import_runs_json,
    parse_runs,
)
from modelbench.run_comparison import compare_runs
from modelbench.runner import run_experiment
from modelbench.store import ArtifactRecord, Store

ENV = {"python": "3.12.1", "lib:numpy": None}


def table():
    rows = [(f"{i * 0.1:.1f}", "0", "a") for i in range(20)]
    rows += [(f"{10 + i * 0.1:.1f}", "5", "b") for i in range(20)]
    return Table(("x1", "x2", "label"), tuple(rows))


def line_table():
    return Table(("x", "y"), tuple((str(i), repr(2.0 * i + 1.0)) for i in range(30)))


def ticking_clock():
    state = {"n": 0}

    def clock():
        state["n"] += 1
        return f"2026-03-01T10:{state['n'] // 60:02d}:{state['n'] % 60:02d}Z"

    return clock


def config(**overrides):
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


def build_source(artifact_dir=None):
    """A store holding a completed run with a note and artifact, a regression run, a failed run."""
    store = Store(":memory:", clock=ticking_clock())
    first = run_experiment(
        store, config(), table(), "label", environment=ENV, artifact_dir=artifact_dir
    )
    store.add_note(first.run_id, "first attempt")
    store.add_note(first.run_id, "second thought")
    reg = run_experiment(
        store,
        config(experiment_id="reg", model="linear_regression", params={}, metrics=("mae", "r2"),
               dataset_id="line"),
        line_table(), "y", environment=ENV,
    )  # fmt: skip
    failed = store.create_run(
        config(experiment_id="bad", model="majority", params={}),
        make_version("blobs", table(), "label"),
        ENV,
    )
    store.set_tag(failed, "reason", "demo")
    store.fail_run(failed, "ValueError: boom")
    return store, [first.run_id, reg.run_id, failed]


class StoreCase(unittest.TestCase):
    """Test case that closes every store it opens."""

    def dest(self, offset=0):
        clock = ticking_clock()
        store = Store(":memory:", clock=lambda: clock() if offset == 0 else clock()[:-3] + "59Z")
        self.addCleanup(store.close)
        return store

    def source_store(self, artifact_dir=None):
        store, ids = build_source(artifact_dir)
        self.addCleanup(store.close)
        return store, ids


def normalised(doc_run):
    """A run's export with the fields that legitimately change on import removed."""
    run = copy.deepcopy(doc_run)
    run.pop("run_id")
    run["tags"].pop(IMPORT_TAG, None)
    return run


class RoundTripTests(StoreCase):
    def setUp(self):
        self.source, self.ids = self.source_store()
        self.text = export_json(self.source, self.ids)

    def test_every_field_survives_the_round_trip(self):
        dest = self.dest()
        report = import_runs_json(dest, self.text)
        self.assertEqual([s for s, _ in report.imported], self.ids)
        self.assertEqual(report.skipped, ())
        for source_id, new_id in report.imported:
            before = export_runs(self.source, [source_id])["runs"][0]
            after = export_runs(dest, [new_id])["runs"][0]
            self.assertEqual(normalised(after), normalised(before), source_id)
            self.assertEqual(after["tags"][IMPORT_TAG], source_id)

    def test_status_timestamps_notes_and_errors_are_kept_exactly(self):
        dest = self.dest()
        report = dict(import_runs_json(dest, self.text).imported)
        completed = dest.get_run(report[self.ids[0]])
        original = self.source.get_run(self.ids[0])
        self.assertEqual(completed.status, "completed")
        self.assertEqual(
            (completed.started_at, completed.finished_at),
            (original.started_at, original.finished_at),
        )
        self.assertEqual(dest.get_notes(completed.run_id), self.source.get_notes(self.ids[0]))
        failed = dest.get_run(report[self.ids[2]])
        self.assertEqual((failed.status, failed.error), ("failed", "ValueError: boom"))
        self.assertEqual(failed.tags["reason"], "demo")

    def test_new_ids_continue_the_destination_sequence(self):
        dest = self.dest(offset=1)  # a different clock, so "mine" is not a duplicate
        run_experiment(dest, config(experiment_id="mine"), table(), "label", environment=ENV)
        new_ids = [n for _, n in import_runs_json(dest, self.text).imported]
        self.assertEqual(new_ids, ["run-000002", "run-000003", "run-000004"])

    def test_importing_twice_adds_nothing(self):
        dest = self.dest()
        import_runs_json(dest, self.text)
        before = len(dest.list_runs())
        again = import_runs_json(dest, self.text)
        self.assertEqual(again.imported, ())
        self.assertEqual([s for s, _ in again.skipped], self.ids)
        self.assertEqual(len(dest.list_runs()), before)

    def test_only_new_runs_are_added_on_a_second_import(self):
        dest = self.dest()
        import_runs_json(dest, export_json(self.source, self.ids[:2]))
        report = import_runs_json(dest, self.text)
        self.assertEqual([s for s, _ in report.imported], [self.ids[2]])
        self.assertEqual([s for s, _ in report.skipped], self.ids[:2])

    def test_the_original_run_in_its_own_database_counts_as_present(self):
        report = import_runs_json(self.source, self.text)
        self.assertEqual(report.imported, ())
        self.assertEqual(dict(report.skipped), {i: i for i in self.ids})

    def test_a_run_that_differs_in_one_metric_is_not_a_duplicate(self):
        dest = self.dest()
        import_runs_json(dest, self.text)
        doc = json.loads(self.text)
        doc["runs"][0]["metrics"][0]["value"] = 0.123
        report = import_runs_json(dest, json.dumps(doc))
        self.assertEqual([s for s, _ in report.imported], [self.ids[0]])

    def test_dry_run_reports_and_writes_nothing(self):
        dest = self.dest()
        report = import_runs_json(dest, self.text, dry_run=True)
        self.assertEqual([s for s, _ in report.imported], self.ids)
        self.assertTrue(all(new == "" for _, new in report.imported))
        self.assertEqual(dest.list_runs(), [])

    def test_the_original_source_survives_a_second_hop(self):
        middle, last = self.dest(offset=1), self.dest()
        run_experiment(middle, config(experiment_id="mine"), table(), "label", environment=ENV)
        import_runs_json(middle, self.text)
        ids = [r.run_id for r in reversed(middle.list_runs())][1:]
        self.assertEqual(ids[0], "run-000002")  # the ids differ from the source's
        import_runs_json(last, export_json(middle, ids))
        tags = [last.get_run(r.run_id).tags[IMPORT_TAG] for r in reversed(last.list_runs())]
        self.assertEqual(tags, self.ids)

    def test_a_run_with_different_start_time_is_not_a_duplicate(self):
        dest = self.dest()
        import_runs_json(dest, self.text)
        doc = json.loads(self.text)
        doc["runs"][0]["started_at"] = "2026-03-01T09:00:00Z"
        report = import_runs_json(dest, json.dumps(doc))
        self.assertEqual([s for s, _ in report.imported], [self.ids[0]])

    def test_a_dataset_shared_by_two_runs_is_registered_once_at_the_first_insert(self):
        dest = self.dest()
        import_runs_json(dest, self.text)
        rows = dest._conn.execute("SELECT dataset_id, registered_at FROM datasets").fetchall()
        stamps = {r["dataset_id"]: r["registered_at"] for r in rows}
        self.assertEqual(stamps, {"blobs": "2026-03-01T10:00:01Z", "line": "2026-03-01T10:00:02Z"})

    def test_imported_runs_work_with_the_rest_of_the_tool(self):
        dest = self.dest()
        a = import_runs_json(dest, self.text).imported[0][1]
        b = run_experiment(
            dest, config(model="majority", params={}), table(), "label", environment=ENV
        ).run_id
        result = compare_runs(dest, [a, b])
        self.assertEqual(result.baseline, a)

    def test_artifact_metadata_is_imported(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, ids = self.source_store(artifact_dir=tmp)
            dest = self.dest()
            new = import_runs_json(dest, export_json(source, ids)).imported[0][1]
        (artifact,) = dest.get_artifacts(new)
        self.assertEqual(artifact, source.get_artifacts(ids[0])[0])


class RejectionTests(StoreCase):
    def setUp(self):
        source, ids = self.source_store()
        self.doc = json.loads(export_json(source, ids))

    def assertRejected(self, doc, exc=ModelBenchError, text=None):
        dest = self.dest()
        with self.assertRaises(exc):
            import_runs_json(dest, text if text is not None else json.dumps(doc))
        self.assertEqual(dest.list_runs(), [])
        row = dest._conn.execute("SELECT COUNT(*) AS n FROM datasets").fetchone()
        self.assertEqual(row["n"], 0)

    def edit(self, change):
        doc = copy.deepcopy(self.doc)
        change(doc)
        return doc

    def test_wrong_envelope(self):
        for change in (
            lambda d: d.update(format="other"),
            lambda d: d.update(version=2),
            lambda d: d.update(version=True),
            lambda d: d.update(version="1"),
            lambda d: d.update(runs=[]),
            lambda d: d.update(runs={}),
            lambda d: d.update(extra=1),
            lambda d: d.pop("runs"),
        ):
            with self.subTest(change=change):
                self.assertRejected(self.edit(change), RunImportError)
        for text in ("[]", "null", "3", '"x"'):
            self.assertRejected(None, RunImportError, text)

    def test_not_json_or_not_finite(self):
        for text in ("", "{", "not json", '{"format": NaN}', '{"a": Infinity}', '{"a": -Infinity}'):
            self.assertRejected(None, RunImportError, text)
        self.assertRejected(None, RunImportError, "[" * 100_000)

    def test_run_shape(self):
        for change in (
            lambda d: d["runs"].__setitem__(0, []),
            lambda d: d["runs"][0].pop("notes"),
            lambda d: d["runs"][0].update(surprise=1),
            lambda d: d["runs"][0].update(run_id="Bad Id"),
            lambda d: d["runs"][1].update(run_id=d["runs"][0]["run_id"]),
            lambda d: d["runs"][0].update(metrics={}),
            lambda d: d["runs"][0]["metrics"].__setitem__(0, 5),
            lambda d: d["runs"][0]["metrics"][0].update(extra=1),
            lambda d: d["runs"][0]["artifacts"].append(5),
            lambda d: d["runs"][0].update(notes="x"),
            lambda d: d["runs"][0].update(tags=[]),
            lambda d: d["runs"][0].update(environment=[]),
            lambda d: d["runs"][0]["environment"].update(python=3),
        ):
            with self.subTest(change=change):
                self.assertRejected(self.edit(change))

    def test_the_run_must_agree_with_its_config(self):
        for change in (
            lambda d: d["runs"][0].update(config_hash="0" * 64),
            lambda d: d["runs"][0].update(seed=8),
            lambda d: d["runs"][0].update(seed=True),
            lambda d: d["runs"][0].update(experiment_id="other"),
            lambda d: d["runs"][0]["config"].update(seed=8),
            lambda d: d["runs"][0]["config"].update(unknown_key=1),
            lambda d: d["runs"][0]["config"].update(metrics=["bogus"]),
            lambda d: d["runs"][0]["config"].update(dataset_id="other"),
            lambda d: d["runs"][0]["config"].update(dataset_version="f" * 12),
        ):
            with self.subTest(change=change):
                self.assertRejected(self.edit(change))

    def test_dataset_fields(self):
        for change in (
            lambda d: d["runs"][0]["dataset"].update(sha256="xyz"),
            lambda d: d["runs"][0]["dataset"].update(sha256="A" * 64),
            lambda d: d["runs"][0]["dataset"].update(version="abc"),
            lambda d: d["runs"][0]["dataset"].update(n_rows=-1),
            lambda d: d["runs"][0]["dataset"].update(n_rows=True),
            lambda d: d["runs"][0]["dataset"].update(n_rows=1.5),
            lambda d: d["runs"][0]["dataset"].update(label_column="nope"),
            lambda d: d["runs"][0]["dataset"].update(columns="ab"),
            lambda d: d["runs"][0]["dataset"].update(columns=[1, 2]),
            lambda d: d["runs"][0]["dataset"].update(dataset_id="Bad Id"),
            lambda d: d["runs"][0]["dataset"].pop("version"),
        ):
            with self.subTest(change=change):
                self.assertRejected(self.edit(change))

    def test_status_and_timestamps(self):
        for change in (
            lambda d: d["runs"][0].update(status="running"),
            lambda d: d["runs"][0].update(status="done"),
            lambda d: d["runs"][0].update(error="should not be here"),
            lambda d: d["runs"][2].update(error=None),
            lambda d: d["runs"][2].update(error=""),
            lambda d: d["runs"][0].update(started_at="yesterday"),
            lambda d: d["runs"][0].update(finished_at=None),
            lambda d: d["runs"][0].update(finished_at="2000-01-01T00:00:00Z"),
            lambda d: d["runs"][0]["notes"][0].update(created_at="soon"),
            lambda d: d["runs"][0]["notes"][0].update(text="   "),
        ):
            with self.subTest(change=change):
                self.assertRejected(self.edit(change))

    def test_metrics_artifacts_and_tags(self):
        for change in (
            lambda d: d["runs"][0]["metrics"][0].update(value=True),
            lambda d: d["runs"][0]["metrics"][0].update(value="0.5"),
            lambda d: d["runs"][0]["metrics"][0].update(value=None),
            lambda d: d["runs"][0]["metrics"][0].update(step=-1),
            lambda d: d["runs"][0]["metrics"][0].update(step=1.5),
            lambda d: d["runs"][0]["metrics"][0].update(name="bad name"),
            lambda d: d["runs"][0]["metrics"].append(dict(d["runs"][0]["metrics"][0])),
            lambda d: d["runs"][0]["artifacts"].append(
                {"name": "a", "kind": "k", "sha256": "short", "size_bytes": 1, "path": None}),
            lambda d: d["runs"][0]["artifacts"].extend([
                {"name": "a", "kind": "k", "sha256": "0" * 64, "size_bytes": 1, "path": None}] * 2),
            lambda d: d["runs"][0]["artifacts"].append(
                {"name": "a", "kind": "k", "sha256": "0" * 64, "size_bytes": -1, "path": None}),
            lambda d: d["runs"][0]["tags"].update({"bad tag": "x"}),
            lambda d: d["runs"][0]["tags"].update({"ok": 5}),
            lambda d: d["runs"][0]["tags"].update({"ok": "x" * 2001}),
        ):  # fmt: skip
            with self.subTest(change=change):
                self.assertRejected(self.edit(change))

    def test_seed_true_is_not_the_number_one(self):
        doc = self.edit(lambda d: (d["runs"][0]["config"].update(seed=1),
                                   d["runs"][0].update(seed=True)))  # fmt: skip
        doc["runs"][0]["config_hash"] = ExperimentConfig.from_dict(
            doc["runs"][0]["config"]
        ).config_hash()
        self.assertRejected(doc, RunImportError)
        doc["runs"][0]["seed"] = 1
        self.assertEqual(len(import_runs_json(self.dest(), json.dumps(doc)).imported), 3)

    def test_columns_must_be_strings_even_without_a_label(self):
        def change(d):
            d["runs"][0]["dataset"].update(columns=[1, 2], label_column=None)

        self.assertRejected(self.edit(change), RunImportError)

    def test_environment_size_is_limited(self):
        def change(d):
            d["runs"][0]["environment"] = {f"k{i}": "v" for i in range(201)}

        self.assertRejected(self.edit(change), RunImportError)

    def test_numbers_too_large_for_a_float_are_not_finite(self):
        text = json.dumps(self.doc).replace('"value": 1.0', '"value": 1e999', 1)
        self.assertIn("1e999", text)
        self.assertRejected(None, RunImportError, text)

    def test_literal_nan_and_infinity_are_named_in_the_error(self):
        text = json.dumps(self.doc).replace('"value": 1.0', '"value": NaN', 1)
        with self.assertRaisesRegex(RunImportError, "NaN"):
            import_runs_json(self.dest(), text)

    def test_duplicate_metric_points_and_artifacts_are_named(self):
        run = parse_runs(self.doc)[0][1]
        point = run.metrics[0]
        dest = self.dest()
        doubled = type(run)(**{**run.__dict__, "metrics": run.metrics + (point,)})
        with self.assertRaisesRegex(StoreError, "appears twice"):
            dest.import_runs([doubled])
        artifact = ArtifactRecord("p", "predictions", "0" * 64, 1, None)
        twice = type(run)(**{**run.__dict__, "artifacts": (artifact, artifact)})
        with self.assertRaisesRegex(StoreError, "appears twice"):
            dest.import_runs([twice])

    def test_limits_have_their_documented_values(self):
        self.assertEqual(
            (importer.MAX_RUNS, importer.MAX_METRIC_POINTS, importer.MAX_FILE_BYTES),
            (10_000, 100_000, 50_000_000),
        )

    def test_one_bad_run_rejects_the_whole_document(self):
        doc = self.edit(lambda d: d["runs"][2]["tags"].update({"bad tag": "x"}))
        self.assertRejected(doc, StoreError)

    def test_a_database_error_mid_import_rolls_everything_back(self):
        dest = self.dest()
        original = dest._insert_imported
        calls = []

        def flaky(run, register):
            calls.append(1)
            if len(calls) == 2:
                raise sqlite3.OperationalError("disk full")
            return original(run, register)

        with (
            mock.patch.object(dest, "_insert_imported", flaky),
            self.assertRaisesRegex(StoreError, "nothing was stored"),
        ):
            import_runs_json(dest, json.dumps(self.doc))
        self.assertEqual(dest.list_runs(), [])
        self.assertEqual(dest._conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0], 0)
        self.assertEqual(dest._conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0], 0)
        self.assertEqual(len(import_runs_json(dest, json.dumps(self.doc)).imported), 3)

    def test_a_dataset_with_a_different_hash_is_refused(self):
        dest = self.dest()
        run = parse_runs(self.doc)[0][1]
        other = type(run.dataset)(
            run.dataset.dataset_id, run.dataset.version, "1" * 64,
            run.dataset.n_rows, run.dataset.columns, run.dataset.label_column,
        )  # fmt: skip
        dest.register_dataset(other)
        with self.assertRaisesRegex(StoreError, "different content hash"):
            import_runs_json(dest, json.dumps(self.doc))
        self.assertEqual(dest.list_runs(), [])

    def test_import_runs_refuses_unfinished_or_inconsistent_runs_directly(self):
        run = parse_runs(self.doc)[0][1]
        dest = self.dest()
        for fields in ({"status": "running"}, {"started_at": "x"}, {"error": "oops"}):
            bad = type(run)(**{**run.__dict__, **fields})
            with self.subTest(fields=fields), self.assertRaises(StoreError):
                dest.import_runs([bad])
        self.assertEqual(dest.import_runs([]), [])


class FileTests(StoreCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        source, ids = self.source_store()
        self.text = export_json(source, ids)

    def test_reads_a_file_with_or_without_a_byte_order_mark(self):
        plain = self.text.encode()
        for name, data in (("plain.json", plain), ("bom.json", b"\xef\xbb\xbf" + plain)):
            path = self.dir / name
            path.write_bytes(data)
            report = import_file(self.dest(), path)
            self.assertEqual(len(report.imported), 3, name)

    def test_unreadable_files(self):
        with self.assertRaisesRegex(RunImportError, "cannot read"):
            import_file(self.dest(), self.dir / "missing.json")
        with self.assertRaisesRegex(RunImportError, "cannot read"):
            import_file(self.dest(), self.dir)
        bad = self.dir / "bad.json"
        bad.write_bytes(b"\xff\xfe\x00")
        with self.assertRaisesRegex(RunImportError, "UTF-8"):
            import_file(self.dest(), bad)
        big = self.dir / "big.json"
        big.write_text(self.text)
        with (
            mock.patch.object(importer, "MAX_FILE_BYTES", 10),
            self.assertRaisesRegex(RunImportError, "larger than"),
        ):
            import_file(self.dest(), big)

    def test_limits(self):
        with (
            mock.patch.object(importer, "MAX_RUNS", 2),
            self.assertRaisesRegex(RunImportError, "more than 2 runs"),
        ):
            import_runs_json(self.dest(), self.text)
        with (
            mock.patch.object(importer, "MAX_METRIC_POINTS", 1),
            self.assertRaisesRegex(RunImportError, "points"),
        ):
            import_runs_json(self.dest(), self.text)

    def test_text_must_be_a_string(self):
        with self.assertRaises(RunImportError):
            import_runs_json(self.dest(), b"{}")


class FindByTagTests(StoreCase):
    def test_finds_runs_in_creation_order(self):
        store = Store(":memory:")
        self.addCleanup(store.close)
        ids = [
            run_experiment(
                store, config(experiment_id=f"e{i}"), table(), "label", environment=ENV
            ).run_id
            for i in range(3)
        ]
        self.assertEqual(store.find_runs_by_tag("n_test", "8"), ids)
        self.assertEqual(store.find_runs_by_tag("n_test", "9"), [])
        with self.assertRaises(StoreError):
            store.find_runs_by_tag("bad tag", "x")


if __name__ == "__main__":
    unittest.main()
