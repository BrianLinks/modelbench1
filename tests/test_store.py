import json
import math
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from modelbench.config import ExperimentConfig
from modelbench.datasets import Table, make_version
from modelbench.errors import IdentifierError, StoreError
from modelbench.store import MAX_LIST_LIMIT, MIGRATIONS, Store

TABLE = Table(("x", "label"), (("1", "a"), ("2", "b"), ("3", "a"), ("4", "b")))
DATASET = make_version("iris", TABLE, label_column="label")
ENV = {"python": "3.12.1", "lib:numpy": None}
SHA = "a" * 64


def make_config(**overrides):
    fields = {
        "experiment_id": "exp-a",
        "model": "knn",
        "seed": 1,
        "dataset_id": "iris",
        "metrics": ("accuracy", "f1"),
        "params": {"k": 3, "weights": "uniform"},
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


def fixed_clock():
    ticks = iter(range(1, 100000))
    return lambda: f"2026-01-01T00:00:{next(ticks) % 60:02d}Z"


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "runs.db"
        self.store = Store(self.path, clock=lambda: "2026-01-01T00:00:00Z")
        self.addCleanup(self.store.close)

    def new_run(self, **config_overrides):
        return self.store.create_run(make_config(**config_overrides), DATASET, ENV)


class MigrationTests(StoreTestCase):
    def test_version_is_set_and_survives_reopening(self):
        self.assertEqual(self.store.schema_version, len(MIGRATIONS))
        self.store.close()
        reopened = Store(self.path)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.schema_version, len(MIGRATIONS))

    def test_expected_tables_exist(self):
        names = {
            r[0]
            for r in self.store._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        self.assertTrue(
            {"datasets", "experiments", "runs", "params", "metrics", "artifacts", "tags", "notes"}
            <= names
        )

    def test_foreign_keys_are_enforced(self):
        self.assertEqual(self.store._conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)

    def test_newer_database_is_refused(self):
        future = Path(self._tmp.name) / "future.db"
        conn = sqlite3.connect(future)
        conn.execute(f"PRAGMA user_version = {len(MIGRATIONS) + 3}")
        conn.commit()
        conn.close()
        with self.assertRaises(StoreError):
            Store(future)

    def test_file_that_is_not_a_database_raises_store_error(self):
        junk = Path(self._tmp.name) / "junk.db"
        junk.write_text("this is plain text, not an SQLite database " * 4)
        with self.assertRaisesRegex(StoreError, "junk.db"):
            Store(junk)

    def test_connection_is_closed_when_opening_fails(self):
        class Recording(sqlite3.Connection):
            instances = []

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.was_closed = False
                Recording.instances.append(self)

            def close(self):
                self.was_closed = True
                super().close()

        future = Path(self._tmp.name) / "future2.db"
        conn = sqlite3.connect(future)
        conn.execute(f"PRAGMA user_version = {len(MIGRATIONS) + 1}")
        conn.commit()
        conn.close()
        real_connect = sqlite3.connect
        with (
            mock.patch(
                "modelbench.store.sqlite3.connect",
                lambda *a, **k: real_connect(*a, factory=Recording, **k),
            ),
            self.assertRaises(StoreError),
        ):
            Store(future)
        self.assertEqual(len(Recording.instances), 1)
        self.assertTrue(Recording.instances[0].was_closed)

    def test_unopenable_path_raises_store_error(self):
        with self.assertRaises(StoreError):
            Store(Path(self._tmp.name) / "no-such-dir" / "x.db")

    def test_context_manager_closes_the_connection(self):
        with Store(Path(self._tmp.name) / "ctx.db") as store:
            store.list_runs()
        with self.assertRaises(sqlite3.ProgrammingError):
            store.list_runs()


class DatasetTests(StoreTestCase):
    def test_round_trip(self):
        self.store.register_dataset(DATASET)
        self.assertEqual(self.store.get_dataset("iris", DATASET.version), DATASET)

    def test_registering_twice_is_a_noop(self):
        self.store.register_dataset(DATASET)
        self.store.register_dataset(DATASET)
        count = self.store._conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        self.assertEqual(count, 1)

    def test_same_version_with_different_content_is_refused(self):
        self.store.register_dataset(DATASET)
        with self.assertRaises(StoreError):
            self.store.register_dataset(replace(DATASET, sha256="b" * 64))

    def test_unknown_dataset(self):
        with self.assertRaises(StoreError):
            self.store.get_dataset("iris", "0" * 12)


class CreateRunTests(StoreTestCase):
    def test_ids_are_sequential_and_zero_padded(self):
        ids = [self.new_run() for _ in range(3)]
        self.assertEqual(ids, ["run-000001", "run-000002", "run-000003"])

    def test_record_contents(self):
        run_id = self.new_run()
        run = self.store.get_run(run_id)
        config = make_config()
        self.assertEqual(run.run_id, "run-000001")
        self.assertEqual(run.experiment_id, "exp-a")
        self.assertEqual(run.config_hash, config.config_hash())
        self.assertEqual((run.dataset_id, run.dataset_version), ("iris", DATASET.version))
        self.assertEqual(run.seed, 1)
        self.assertEqual(run.status, "running")
        self.assertEqual(run.started_at, "2026-01-01T00:00:00Z")
        self.assertIsNone(run.finished_at)
        self.assertIsNone(run.error)
        self.assertEqual(run.environment, ENV)
        self.assertEqual(run.params, {"k": 3, "weights": "uniform"})
        self.assertEqual(run.tags, {})

    def test_nested_params_round_trip(self):
        params = {"layers": [8, 4], "opt": {"lr": 0.5, "nesterov": True}, "none": None}
        run_id = self.new_run(params=params)
        self.assertEqual(self.store.get_run(run_id).params, params)

    def test_stored_config_rebuilds_the_original(self):
        config = make_config(stratify=True, test_fraction=0.3)
        run_id = self.store.create_run(config, DATASET, ENV)
        self.assertEqual(self.store.get_config(run_id), config)

    def test_dataset_is_registered_automatically(self):
        self.new_run()
        self.assertEqual(self.store.get_dataset("iris", DATASET.version), DATASET)

    def test_mismatched_dataset_id_is_refused_and_leaves_nothing_behind(self):
        with self.assertRaises(StoreError):
            self.new_run(dataset_id="wine")
        self.assertEqual(self.store.list_runs(), [])

    def test_pinned_dataset_version_must_match(self):
        with self.assertRaises(StoreError):
            self.new_run(dataset_version="0" * 12)
        run_id = self.new_run(dataset_version=DATASET.version)
        self.assertEqual(self.store.get_run(run_id).dataset_version, DATASET.version)

    def test_runs_of_one_experiment_share_one_experiment_row(self):
        self.new_run()
        self.new_run()
        count = self.store._conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        self.assertEqual(count, 1)

    def test_unknown_and_malformed_run_ids(self):
        with self.assertRaises(StoreError):
            self.store.get_run("run-999999")
        for bad in ("../x", "RUN-1", "", "a b"):
            with self.assertRaises(IdentifierError, msg=repr(bad)):
                self.store.get_run(bad)


class LifecycleTests(StoreTestCase):
    def test_finish_run(self):
        run_id = self.new_run()
        self.store.finish_run(run_id)
        run = self.store.get_run(run_id)
        self.assertEqual((run.status, run.finished_at), ("completed", "2026-01-01T00:00:00Z"))

    def test_fail_run_records_and_truncates_the_error(self):
        run_id = self.new_run()
        self.store.fail_run(run_id, "x" * 5000)
        run = self.store.get_run(run_id)
        self.assertEqual(run.status, "failed")
        self.assertEqual(len(run.error), 2000)

    def test_a_finished_run_cannot_be_finished_failed_or_changed(self):
        for end in (self.store.finish_run, lambda r: self.store.fail_run(r, "boom")):
            run_id = self.new_run()
            end(run_id)
            with self.assertRaises(StoreError):
                self.store.finish_run(run_id)
            with self.assertRaises(StoreError):
                self.store.fail_run(run_id, "again")
            with self.assertRaises(StoreError):
                self.store.log_metric(run_id, "accuracy", 0.5)
            with self.assertRaises(StoreError):
                self.store.add_artifact(run_id, "m", "model", SHA, 1)
            with self.assertRaises(StoreError):
                self.store.set_tag(run_id, "note", "x")

    def test_status_does_not_change_after_a_refused_operation(self):
        run_id = self.new_run()
        self.store.fail_run(run_id, "boom")
        with self.assertRaises(StoreError):
            self.store.finish_run(run_id)
        self.assertEqual(self.store.get_run(run_id).status, "failed")


class MetricTests(StoreTestCase):
    def test_series_is_ordered_by_name_then_step(self):
        run_id = self.new_run()
        for step in (2, 0, 1):
            self.store.log_metric(run_id, "loss", 1.0 / (step + 1), step)
        self.store.log_metric(run_id, "accuracy", 0.9, 0)
        points = self.store.get_metrics(run_id)
        self.assertEqual(
            [(p.name, p.step) for p in points],
            [("accuracy", 0), ("loss", 0), ("loss", 1), ("loss", 2)],
        )
        self.assertEqual(points[3].value, 1 / 3)

    def test_final_metrics_use_the_highest_step_even_if_logged_out_of_order(self):
        run_id = self.new_run()
        self.store.log_metric(run_id, "loss", 0.1, step=5)
        self.store.log_metric(run_id, "loss", 0.9, step=1)
        self.store.log_metrics(run_id, {"accuracy": 0.8, "f1": 0.7})
        self.assertEqual(
            self.store.final_metrics(run_id), {"accuracy": 0.8, "f1": 0.7, "loss": 0.1}
        )

    def test_integers_are_stored_as_floats(self):
        run_id = self.new_run()
        self.store.log_metric(run_id, "count", 3)
        self.assertEqual(self.store.final_metrics(run_id), {"count": 3.0})

    def test_history_cannot_be_rewritten(self):
        run_id = self.new_run()
        self.store.log_metric(run_id, "accuracy", 0.5)
        with self.assertRaises(StoreError):
            self.store.log_metric(run_id, "accuracy", 0.6)
        self.assertEqual(self.store.final_metrics(run_id), {"accuracy": 0.5})

    def test_batch_is_atomic(self):
        run_id = self.new_run()
        self.store.log_metric(run_id, "accuracy", 0.5)
        with self.assertRaises(StoreError):
            self.store.log_metrics(run_id, {"f1": 0.4, "accuracy": 0.9})
        self.assertEqual(self.store.final_metrics(run_id), {"accuracy": 0.5})

    def test_invalid_values_steps_and_names_store_nothing(self):
        run_id = self.new_run()
        for value in (float("nan"), float("inf"), -math.inf, True, "0.5", None, [0.5]):
            with self.assertRaises(StoreError, msg=repr(value)):
                self.store.log_metric(run_id, "accuracy", value)
        for step in (-1, 1.5, True, "0", None):
            with self.assertRaises(StoreError, msg=repr(step)):
                self.store.log_metric(run_id, "accuracy", 0.5, step=step)
        for name in ("", " x", "a b", "../x", "x/", "x.", "a" * 65, "x;y", None, 5):
            with self.assertRaises(StoreError, msg=repr(name)):
                self.store.log_metric(run_id, name, 0.5)
        with self.assertRaises(StoreError):
            self.store.log_metrics(run_id, {})
        self.assertEqual(self.store.get_metrics(run_id), [])

    def test_a_bad_value_in_a_batch_stores_none_of_it(self):
        run_id = self.new_run()
        with self.assertRaises(StoreError):
            self.store.log_metrics(run_id, {"good": 1.0, "bad": float("nan")})
        self.assertEqual(self.store.get_metrics(run_id), [])

    def test_namespaced_metric_names_are_allowed(self):
        run_id = self.new_run()
        self.store.log_metric(run_id, "test/accuracy", 0.5)
        self.assertEqual(self.store.final_metrics(run_id), {"test/accuracy": 0.5})

    def test_unknown_run(self):
        with self.assertRaises(StoreError):
            self.store.log_metric("run-000042", "accuracy", 0.5)
        with self.assertRaises(StoreError):
            self.store.get_metrics("run-000042")


class ArtifactAndTagTests(StoreTestCase):
    def test_artifacts_round_trip_in_name_order(self):
        run_id = self.new_run()
        self.store.add_artifact(run_id, "weights", "model", SHA, 1024, path="out/weights.json")
        self.store.add_artifact(run_id, "confusion", "report", "b" * 64, 0)
        artifacts = self.store.get_artifacts(run_id)
        self.assertEqual([a.name for a in artifacts], ["confusion", "weights"])
        self.assertEqual(artifacts[1].path, "out/weights.json")
        self.assertIsNone(artifacts[0].path)
        self.assertEqual(artifacts[1].size_bytes, 1024)

    def test_artifact_validation(self):
        run_id = self.new_run()
        bad_calls = [
            ("m", "model", "A" * 64, 1, None),  # uppercase hex
            ("m", "model", "a" * 63, 1, None),
            ("m", "model", SHA, -1, None),
            ("m", "model", SHA, True, None),
            ("m", "model", SHA, 1.5, None),
            ("m", "model", SHA, 1, "p" * 501),
            ("m", "model", SHA, 1, 7),
            ("bad name", "model", SHA, 1, None),
            ("m", "bad kind", SHA, 1, None),
        ]
        for args in bad_calls:
            with self.assertRaises(StoreError, msg=repr(args)):
                self.store.add_artifact(run_id, *args)
        self.assertEqual(self.store.get_artifacts(run_id), [])

    def test_duplicate_artifact_name_is_refused(self):
        run_id = self.new_run()
        self.store.add_artifact(run_id, "weights", "model", SHA, 1)
        with self.assertRaises(StoreError):
            self.store.add_artifact(run_id, "weights", "model", "b" * 64, 2)
        self.assertEqual(len(self.store.get_artifacts(run_id)), 1)

    def test_tags_set_replace_and_appear_on_the_run(self):
        run_id = self.new_run()
        self.store.set_tag(run_id, "stage", "baseline")
        self.store.set_tag(run_id, "stage", "tuned")
        self.store.set_tag(run_id, "note", "first try")
        self.assertEqual(self.store.get_run(run_id).tags, {"stage": "tuned", "note": "first try"})

    def test_tag_validation(self):
        run_id = self.new_run()
        for key, value in (("bad key", "x"), ("", "x"), ("k", 5), ("k", "x" * 2001), ("k", None)):
            with self.assertRaises(StoreError, msg=repr((key, value))):
                self.store.set_tag(run_id, key, value)
        self.assertEqual(self.store.get_run(run_id).tags, {})


class QueryTests(StoreTestCase):
    def test_list_runs_newest_first_with_filters(self):
        a1 = self.new_run(experiment_id="exp-a")
        b1 = self.new_run(experiment_id="exp-b")
        a2 = self.new_run(experiment_id="exp-a")
        self.store.finish_run(a1)
        self.assertEqual([r.run_id for r in self.store.list_runs()], [a2, b1, a1])
        self.assertEqual([r.run_id for r in self.store.list_runs(experiment_id="exp-a")], [a2, a1])
        self.assertEqual([r.run_id for r in self.store.list_runs(status="completed")], [a1])
        self.assertEqual(
            [r.run_id for r in self.store.list_runs(experiment_id="exp-a", status="running")], [a2]
        )
        self.assertEqual([r.run_id for r in self.store.list_runs(limit=1)], [a2])

    def test_list_runs_validates_arguments(self):
        for limit in (0, -1, MAX_LIST_LIMIT + 1, True, 1.5, "5"):
            with self.assertRaises(StoreError, msg=repr(limit)):
                self.store.list_runs(limit=limit)
        with self.assertRaises(StoreError):
            self.store.list_runs(status="paused")

    def test_listing_hydrates_params_and_tags_for_each_run(self):
        first = self.new_run(params={"k": 1})
        second = self.new_run(params={"k": 2})
        self.store.set_tag(second, "stage", "x")
        runs = {r.run_id: r for r in self.store.list_runs()}
        self.assertEqual(runs[first].params, {"k": 1})
        self.assertEqual((runs[second].params, runs[second].tags), ({"k": 2}, {"stage": "x"}))

    def test_find_runs_by_config(self):
        a = self.new_run(experiment_id="exp-a")
        b = self.new_run(experiment_id="exp-b")  # same settings, different label
        c = self.new_run(params={"k": 9})
        digest = make_config().config_hash()
        self.assertEqual(self.store.find_runs_by_config(digest), [a, b])
        self.assertEqual(self.store.find_runs_by_config(digest, DATASET.version), [a, b])
        self.assertEqual(self.store.find_runs_by_config(digest, "0" * 12), [])
        other_hash = make_config(params={"k": 9}).config_hash()
        self.assertEqual(self.store.find_runs_by_config(other_hash), [c])
        self.assertEqual(self.store.find_runs_by_config("f" * 64), [])


class NoteTests(StoreTestCase):
    def test_notes_keep_order_and_timestamps(self):
        run_id = self.new_run()
        self.store.add_note(run_id, "first")
        self.store.add_note(run_id, "second\nwith a second line")
        notes = self.store.get_notes(run_id)
        self.assertEqual([n.text for n in notes], ["first", "second\nwith a second line"])
        self.assertEqual(notes[0].created_at, "2026-01-01T00:00:00Z")

    def test_notes_can_be_added_in_every_status(self):
        running = self.new_run()
        done = self.new_run()
        self.store.finish_run(done)
        failed = self.new_run()
        self.store.fail_run(failed, "boom")
        for run_id in (running, done, failed):
            self.store.add_note(run_id, "after the fact")
            self.assertEqual(len(self.store.get_notes(run_id)), 1)

    def test_notes_belong_to_their_run(self):
        a, b = self.new_run(), self.new_run()
        self.store.add_note(a, "only a")
        self.assertEqual(self.store.get_notes(b), [])

    def test_note_validation(self):
        run_id = self.new_run()
        for bad in ("", "   \n", None, 5, "x" * 10_001):
            with self.subTest(bad=bad), self.assertRaises(StoreError):
                self.store.add_note(run_id, bad)
        self.assertEqual(self.store.get_notes(run_id), [])
        self.store.add_note(run_id, "x" * 10_000)

    def test_unknown_run(self):
        with self.assertRaises(StoreError):
            self.store.add_note("run-000042", "hello")
        with self.assertRaises(StoreError):
            self.store.get_notes("run-000042")

    def test_database_from_before_notes_is_upgraded_and_keeps_its_data(self):
        old = Path(self._tmp.name) / "old.db"
        conn = sqlite3.connect(old)
        conn.executescript(f"BEGIN;{MIGRATIONS[0]}PRAGMA user_version = 1;COMMIT;")
        conn.close()
        with Store(old, clock=lambda: "2026-01-01T00:00:00Z") as store:
            self.assertEqual(store.schema_version, len(MIGRATIONS))
        # Rewind to version 1 with data in every older table, then upgrade again.
        conn = sqlite3.connect(old)
        conn.execute("DROP TABLE notes")
        conn.execute("DROP TABLE experiment_notes")
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        conn.close()
        with Store(old, clock=lambda: "2026-01-01T00:00:00Z") as store:
            run_id = store.create_run(make_config(), DATASET, ENV)
            store.set_tag(run_id, "stage", "baseline")
            store.log_metric(run_id, "accuracy", 0.5)
            store.add_artifact(run_id, "weights", "model", SHA, 3)
        conn = sqlite3.connect(old)
        conn.execute("DROP TABLE notes")
        conn.execute("DROP TABLE experiment_notes")
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        conn.close()
        with Store(old) as store:
            self.assertEqual(store.schema_version, len(MIGRATIONS))
            self.assertEqual(store.get_run(run_id).tags, {"stage": "baseline"})
            self.assertEqual(store.final_metrics(run_id), {"accuracy": 0.5})
            self.assertEqual(len(store.get_artifacts(run_id)), 1)
            store.add_note(run_id, "works after upgrade")
            self.assertEqual([n.text for n in store.get_notes(run_id)], ["works after upgrade"])


class ExperimentNoteTests(StoreTestCase):
    def test_notes_keep_order_and_timestamps(self):
        self.new_run()  # makes experiment "exp-a"
        self.store.add_experiment_note("exp-a", "why this exists")
        self.store.add_experiment_note("exp-a", "second\nline")
        notes = self.store.get_experiment_notes("exp-a")
        self.assertEqual([n.text for n in notes], ["why this exists", "second\nline"])
        self.assertEqual(notes[0].created_at, "2026-01-01T00:00:00Z")

    def test_each_experiment_has_its_own_notes_apart_from_run_notes(self):
        first = self.new_run()
        self.store.create_run(make_config(experiment_id="exp-b"), DATASET, ENV)
        self.store.add_experiment_note("exp-a", "about a")
        self.store.add_note(first, "about the run")
        self.assertEqual([n.text for n in self.store.get_experiment_notes("exp-a")], ["about a"])
        self.assertEqual(self.store.get_experiment_notes("exp-b"), [])
        self.assertEqual([n.text for n in self.store.get_notes(first)], ["about the run"])

    def test_unknown_experiments_and_bad_text_are_refused(self):
        self.new_run()
        for call in (
            lambda: self.store.add_experiment_note("nope", "x"),
            lambda: self.store.get_experiment_notes("nope"),
            lambda: self.store.add_experiment_note("exp-a", "   "),
            lambda: self.store.add_experiment_note("exp-a", "x" * 10_001),
            lambda: self.store.add_experiment_note("exp-a", 5),
        ):
            with self.assertRaises(StoreError):
                call()
        with self.assertRaises(IdentifierError):
            self.store.get_experiment_notes("not valid!")
        self.assertEqual(self.store.get_experiment_notes("exp-a"), [])

    def test_a_database_made_by_0_2_0_is_upgraded_and_keeps_its_runs(self):
        old = Path(self._tmp.name) / "v2.db"
        conn = sqlite3.connect(old)
        conn.executescript(f"BEGIN;{MIGRATIONS[0]}{MIGRATIONS[1]}PRAGMA user_version = 2;COMMIT;")
        conn.close()
        with Store(old, clock=lambda: "2026-01-01T00:00:00Z") as store:
            run_id = store.create_run(make_config(), DATASET, ENV)
            store.add_note(run_id, "old note")
        conn = sqlite3.connect(old)
        conn.execute("DROP TABLE experiment_notes")
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
        conn.close()
        with Store(old) as store:
            self.assertEqual(store.schema_version, 3)
            self.assertEqual([n.text for n in store.get_notes(run_id)], ["old note"])
            store.add_experiment_note("exp-a", "new")
            self.assertEqual([n.text for n in store.get_experiment_notes("exp-a")], ["new"])

    def test_a_read_only_store_reads_but_cannot_write_them(self):
        self.new_run()
        self.store.add_experiment_note("exp-a", "kept")
        reader = Store(self.path, read_only=True)
        self.addCleanup(reader.close)
        self.assertEqual([n.text for n in reader.get_experiment_notes("exp-a")], ["kept"])
        with self.assertRaises(sqlite3.OperationalError):  # SQLite itself refuses the write
            reader.add_experiment_note("exp-a", "refused")


class ReadOnlyTests(StoreTestCase):
    def open_read_only(self, path=None):
        store = Store(path or self.path, read_only=True)
        self.addCleanup(store.close)
        return store

    def test_reads_work_and_writes_are_refused_by_sqlite(self):
        run_id = self.new_run()
        self.store.log_metric(run_id, "accuracy", 0.5)
        self.store.close()
        ro = self.open_read_only()
        self.assertEqual(ro.final_metrics(run_id), {"accuracy": 0.5})
        self.assertEqual(ro.get_run(run_id).status, "running")
        with self.assertRaises(sqlite3.OperationalError):
            ro.log_metric(run_id, "loss", 1.0)
        with self.assertRaises(sqlite3.OperationalError):
            ro.add_note(run_id, "nope")
        self.assertEqual(ro.final_metrics(run_id), {"accuracy": 0.5})

    def test_does_not_create_a_missing_file(self):
        missing = Path(self._tmp.name) / "missing.db"
        with self.assertRaises(StoreError):
            Store(missing, read_only=True)
        self.assertFalse(missing.exists())

    def test_database_at_an_older_schema_is_not_migrated(self):
        old = Path(self._tmp.name) / "old.db"
        conn = sqlite3.connect(old)
        conn.executescript(f"BEGIN;{MIGRATIONS[0]}PRAGMA user_version = 1;COMMIT;")
        conn.close()
        with self.assertRaisesRegex(StoreError, "schema version 1"):
            Store(old, read_only=True)
        check = sqlite3.connect(old)
        self.addCleanup(check.close)
        self.assertEqual(check.execute("PRAGMA user_version").fetchone()[0], 1)

    def test_not_a_database_and_path_with_special_characters(self):
        junk = Path(self._tmp.name) / "junk.db"
        junk.write_text("plain text, definitely not sqlite " * 5)
        with self.assertRaises(StoreError):
            Store(junk, read_only=True)
        odd = Path(self._tmp.name) / "my runs #1 %20 x=y&z.db"
        with Store(odd) as writable:
            writable.create_run(make_config(), DATASET, ENV)
        ro = self.open_read_only(odd)
        self.assertEqual(len(ro.list_runs()), 1)


class ScaleAndIntegrityTests(unittest.TestCase):
    def test_listing_the_maximum_page_works_beyond_the_sql_variable_chunk_size(self):
        with Store(":memory:") as store:
            for _ in range(MAX_LIST_LIMIT):
                store.create_run(make_config(), DATASET, ENV)
            runs = store.list_runs(limit=MAX_LIST_LIMIT)
            self.assertEqual(len(runs), MAX_LIST_LIMIT)
            self.assertTrue(all(r.params == {"k": 3, "weights": "uniform"} for r in runs))
            self.assertEqual(runs[0].run_id, f"run-{MAX_LIST_LIMIT:06d}")

    def test_deleting_a_run_cascades_to_its_children(self):
        with Store(":memory:") as store:
            run_id = store.create_run(make_config(), DATASET, ENV)
            store.log_metric(run_id, "accuracy", 0.5)
            store.add_artifact(run_id, "m", "model", SHA, 1)
            store.set_tag(run_id, "k", "v")
            store._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            for table in ("params", "metrics", "artifacts", "tags"):
                count = store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                self.assertEqual(count, 0, table)

    def test_a_run_cannot_reference_an_unregistered_dataset(self):
        with Store(":memory:") as store, self.assertRaises(sqlite3.IntegrityError):
            store._conn.execute(
                "INSERT INTO experiments (experiment_id, created_at) VALUES ('e', 't')"
            )
            store._conn.execute(
                "INSERT INTO runs (experiment_id, config_hash, config_json, dataset_id,"
                " dataset_version, seed, status, started_at, environment_json)"
                " VALUES ('e', 'h', '{}', 'nope', 'v', 1, 'running', 't', '{}')"
            )

    def test_data_persists_across_connections(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.db"
            with Store(path) as store:
                run_id = store.create_run(make_config(), DATASET, ENV)
                store.log_metric(run_id, "accuracy", 0.75)
                store.finish_run(run_id)
            with Store(path) as store:
                self.assertEqual(store.get_run(run_id).status, "completed")
                self.assertEqual(store.final_metrics(run_id), {"accuracy": 0.75})
                self.assertEqual(store.create_run(make_config(), DATASET, ENV), "run-000002")

    def test_identical_operations_with_a_fixed_clock_give_identical_records(self):
        def build():
            with Store(":memory:", clock=fixed_clock()) as store:
                run_id = store.create_run(make_config(), DATASET, ENV)
                store.log_metrics(run_id, {"accuracy": 0.5, "f1": 0.25})
                store.set_tag(run_id, "stage", "x")
                store.finish_run(run_id)
                return store.get_run(run_id), store.get_metrics(run_id)

        self.assertEqual(build(), build())

    def test_config_json_is_canonical(self):
        with Store(":memory:") as store:
            run_id = store.create_run(make_config(), DATASET, ENV)
            text = store._conn.execute(
                "SELECT config_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            self.assertEqual(text, json.dumps(json.loads(text), sort_keys=True))


if __name__ == "__main__":
    unittest.main()
