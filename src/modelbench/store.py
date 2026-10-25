"""SQLite run store: datasets, runs, parameters, metric series, artifact metadata, and tags.

Rules the store enforces:
- A run starts 'running' and ends exactly once as 'completed' or 'failed'. Only running runs
  accept metrics, artifacts, and tags.
- A metric is written once per (run, name, step); rewriting history is an error.
- Every run records the dataset version, the seed, the full configuration, and the environment
  it executed in, so a later reproducibility check has everything it needs.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .datasets import DatasetVersion
from .errors import StoreError
from .ids import validate_id

MIGRATIONS: list[str] = [
    """
    CREATE TABLE datasets (
        dataset_id    TEXT NOT NULL,
        version       TEXT NOT NULL,
        sha256        TEXT NOT NULL,
        n_rows        INTEGER NOT NULL,
        columns_json  TEXT NOT NULL,
        label_column  TEXT,
        registered_at TEXT NOT NULL,
        PRIMARY KEY (dataset_id, version)
    );
    CREATE TABLE experiments (
        experiment_id TEXT PRIMARY KEY,
        created_at    TEXT NOT NULL
    );
    CREATE TABLE runs (
        seq              INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id           TEXT UNIQUE,
        experiment_id    TEXT NOT NULL REFERENCES experiments(experiment_id),
        config_hash      TEXT NOT NULL,
        config_json      TEXT NOT NULL,
        dataset_id       TEXT NOT NULL,
        dataset_version  TEXT NOT NULL,
        seed             INTEGER NOT NULL,
        status           TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
        started_at       TEXT NOT NULL,
        finished_at      TEXT,
        error            TEXT,
        environment_json TEXT NOT NULL,
        FOREIGN KEY (dataset_id, dataset_version) REFERENCES datasets(dataset_id, version)
    );
    CREATE INDEX idx_runs_experiment ON runs(experiment_id);
    CREATE INDEX idx_runs_config ON runs(config_hash);
    CREATE TABLE params (
        run_id     TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        key        TEXT NOT NULL,
        value_json TEXT NOT NULL,
        PRIMARY KEY (run_id, key)
    );
    CREATE TABLE metrics (
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        name   TEXT NOT NULL,
        step   INTEGER NOT NULL CHECK (step >= 0),
        value  REAL NOT NULL,
        PRIMARY KEY (run_id, name, step)
    );
    CREATE TABLE artifacts (
        run_id     TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        name       TEXT NOT NULL,
        kind       TEXT NOT NULL,
        sha256     TEXT NOT NULL,
        size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
        path       TEXT,
        PRIMARY KEY (run_id, name)
    );
    CREATE TABLE tags (
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        key    TEXT NOT NULL,
        value  TEXT NOT NULL,
        PRIMARY KEY (run_id, key)
    );
    """,
    """
    CREATE TABLE notes (
        seq        INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id     TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        created_at TEXT NOT NULL,
        text       TEXT NOT NULL
    );
    CREATE INDEX idx_notes_run ON notes(run_id);
    """,
]

STATUSES = ("running", "completed", "failed")
MAX_LIST_LIMIT = 1000
_IN_CHUNK = 400
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-/]{0,63}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_TEXT = 2000
_MAX_NOTE = 10_000
_TIMESTAMP_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    experiment_id: str
    config_hash: str
    dataset_id: str
    dataset_version: str
    seed: int
    status: str
    started_at: str
    finished_at: str | None
    error: str | None
    environment: dict[str, Any]
    params: dict[str, Any]
    tags: dict[str, str]


@dataclass(frozen=True)
class MetricPoint:
    name: str
    step: int
    value: float


@dataclass(frozen=True)
class ArtifactRecord:
    name: str
    kind: str
    sha256: str
    size_bytes: int
    path: str | None


@dataclass(frozen=True)
class NoteRecord:
    created_at: str
    text: str


def _check_name(kind: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or _NAME_RE.fullmatch(value) is None
        or ".." in value
        or value.endswith(("/", "."))
    ):
        raise StoreError(
            f"invalid {kind} name {value!r}: use 1-64 characters from letters, digits, "
            "'_', '.', '-' and '/', starting with a letter or digit"
        )
    return value


def _check_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StoreError(f"metric value must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise StoreError(f"metric value must be finite, got {value!r}")
    return number


def _check_step(step: object) -> int:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise StoreError(f"step must be a non-negative integer, got {step!r}")
    return step


def _check_tag(key: object, value: object) -> tuple[str, str]:
    _check_name("tag", key)
    if not isinstance(value, str) or len(value) > _MAX_TEXT:
        raise StoreError(f"tag value must be a string of at most {_MAX_TEXT} characters")
    return str(key), value


def _check_artifact(
    name: object, kind: object, sha256: object, size_bytes: object, path: object
) -> None:
    _check_name("artifact", name)
    _check_name("artifact kind", kind)
    if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
        raise StoreError("sha256 must be 64 lowercase hex characters")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
        raise StoreError("size_bytes must be a non-negative integer")
    if path is not None and (not isinstance(path, str) or len(path) > 500):
        raise StoreError("path must be a string of at most 500 characters")


def _check_note(text: object) -> str:
    if not isinstance(text, str) or text.strip() == "" or len(text) > _MAX_NOTE:
        raise StoreError(f"a note must be non-empty text of at most {_MAX_NOTE} characters")
    return text


def _check_timestamp(name: str, value: object) -> str:
    if not isinstance(value, str) or _TIMESTAMP_RE.fullmatch(value) is None:
        raise StoreError(f"{name} must look like 2026-01-31T09:30:00Z, got {value!r}")
    return value


@dataclass(frozen=True)
class ImportedRun:
    """A finished run to insert as it was recorded elsewhere, with its original timestamps."""

    config: ExperimentConfig
    dataset: DatasetVersion
    environment: dict[str, Any]
    status: str
    started_at: str
    finished_at: str
    error: str | None
    metrics: tuple[MetricPoint, ...]
    artifacts: tuple[ArtifactRecord, ...]
    tags: dict[str, str]
    notes: tuple[NoteRecord, ...]


class Store:
    """Wrapper over one SQLite database. Pass `clock` to make timestamps deterministic."""

    def __init__(
        self, path: str | Path, clock: Callable[[], str] = utc_now, *, read_only: bool = False
    ) -> None:
        """Open (and migrate) the database at `path`.

        With `read_only=True` the file must already exist at the current schema version; it is
        opened so that SQLite itself refuses every write, and nothing is migrated or created.
        """
        self._clock = clock
        try:
            if read_only:
                uri = Path(path).resolve().as_uri() + "?mode=ro"
                self._conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            else:
                self._conn = sqlite3.connect(str(path), timeout=5.0)
        except sqlite3.Error as exc:
            raise StoreError(f"cannot open database {path}: {exc}") from exc
        try:
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            if read_only:
                if self.schema_version != len(MIGRATIONS):
                    raise StoreError(
                        f"database {path} has schema version {self.schema_version}, "
                        f"not {len(MIGRATIONS)}; open it with a writable command to upgrade it"
                    )
            else:
                self._migrate()
        except BaseException as exc:
            self._conn.close()  # do not leak the connection when opening fails
            if isinstance(exc, sqlite3.Error) and not isinstance(exc, StoreError):
                raise StoreError(f"cannot use {path} as a run database: {exc}") from exc
            raise

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def _migrate(self) -> None:
        current = self.schema_version
        if current > len(MIGRATIONS):
            raise StoreError(
                f"database schema version {current} is newer than this program supports "
                f"({len(MIGRATIONS)})"
            )
        for target in range(current, len(MIGRATIONS)):
            script = f"BEGIN;\n{MIGRATIONS[target]}\nPRAGMA user_version = {target + 1};\nCOMMIT;"
            try:
                self._conn.executescript(script)
            except sqlite3.Error as exc:
                self._conn.rollback()
                raise StoreError(f"migration {target + 1} failed: {exc}") from exc

    # -- datasets -----------------------------------------------------------------------------

    def register_dataset(self, dataset: DatasetVersion) -> None:
        """Record a dataset version. Registering the same version again is a no-op."""
        row = self._conn.execute(
            "SELECT sha256 FROM datasets WHERE dataset_id = ? AND version = ?",
            (dataset.dataset_id, dataset.version),
        ).fetchone()
        if row is not None:
            if row["sha256"] != dataset.sha256:
                raise StoreError(
                    f"dataset {dataset.dataset_id}@{dataset.version} is already registered "
                    "with a different content hash"
                )
            return
        with self._conn:
            self._conn.execute(
                "INSERT INTO datasets (dataset_id, version, sha256, n_rows, columns_json,"
                " label_column, registered_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    dataset.dataset_id,
                    dataset.version,
                    dataset.sha256,
                    dataset.n_rows,
                    json.dumps(list(dataset.columns)),
                    dataset.label_column,
                    self._clock(),
                ),
            )

    def get_dataset(self, dataset_id: str, version: str) -> DatasetVersion:
        row = self._conn.execute(
            "SELECT * FROM datasets WHERE dataset_id = ? AND version = ?", (dataset_id, version)
        ).fetchone()
        if row is None:
            raise StoreError(f"no dataset {dataset_id}@{version}")
        return DatasetVersion(
            dataset_id=row["dataset_id"],
            version=row["version"],
            sha256=row["sha256"],
            n_rows=row["n_rows"],
            columns=tuple(json.loads(row["columns_json"])),
            label_column=row["label_column"],
        )

    # -- runs ---------------------------------------------------------------------------------

    def create_run(
        self,
        config: ExperimentConfig,
        dataset: DatasetVersion,
        environment: Mapping[str, Any],
    ) -> str:
        """Start a run and return its id (`run-000001`, `run-000002`, ...)."""
        if config.dataset_id != dataset.dataset_id:
            raise StoreError(
                f"config uses dataset {config.dataset_id!r} but {dataset.dataset_id!r} was given"
            )
        if config.dataset_version is not None and config.dataset_version != dataset.version:
            raise StoreError(
                f"config pins dataset version {config.dataset_version} "
                f"but {dataset.version} was given"
            )
        self.register_dataset(dataset)
        now = self._clock()
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO experiments (experiment_id, created_at) VALUES (?, ?)",
                (config.experiment_id, now),
            )
            cursor = self._conn.execute(
                "INSERT INTO runs (experiment_id, config_hash, config_json, dataset_id,"
                " dataset_version, seed, status, started_at, environment_json)"
                " VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?)",
                (
                    config.experiment_id,
                    config.config_hash(),
                    json.dumps(config.to_dict(), sort_keys=True),
                    dataset.dataset_id,
                    dataset.version,
                    config.seed,
                    now,
                    json.dumps(dict(environment), sort_keys=True),
                ),
            )
            seq = cursor.lastrowid
            run_id = f"run-{seq:06d}"
            self._conn.execute("UPDATE runs SET run_id = ? WHERE seq = ?", (run_id, seq))
            self._conn.executemany(
                "INSERT INTO params (run_id, key, value_json) VALUES (?, ?, ?)",
                [
                    (run_id, key, json.dumps(value, sort_keys=True))
                    for key, value in sorted(config.params.items())
                ],
            )
        return run_id

    def _run_row(self, run_id: str) -> sqlite3.Row:
        validate_id("run", run_id)
        row: sqlite3.Row | None = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise StoreError(f"no run {run_id}")
        return row

    def _require_running(self, run_id: str) -> None:
        status = self._run_row(run_id)["status"]
        if status != "running":
            raise StoreError(f"run {run_id} is {status}; only running runs can be changed")

    def get_run(self, run_id: str) -> RunRecord:
        return self._hydrate([self._run_row(run_id)])[0]

    def get_config(self, run_id: str) -> ExperimentConfig:
        return ExperimentConfig.from_dict(json.loads(self._run_row(run_id)["config_json"]))

    def list_runs(
        self,
        experiment_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[RunRecord]:
        """Newest runs first."""
        valid_limit = isinstance(limit, int) and not isinstance(limit, bool)
        if not valid_limit or not 1 <= limit <= MAX_LIST_LIMIT:
            raise StoreError(f"limit must be between 1 and {MAX_LIST_LIMIT}")
        if status is not None and status not in STATUSES:
            raise StoreError(f"status must be one of {STATUSES}")
        clauses, args = [], []
        if experiment_id is not None:
            clauses.append("experiment_id = ?")
            args.append(experiment_id)
        if status is not None:
            clauses.append("status = ?")
            args.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM runs {where} ORDER BY seq DESC LIMIT ?", (*args, limit)
        ).fetchall()
        return self._hydrate(rows)

    def find_runs_by_config(
        self, config_hash: str, dataset_version: str | None = None
    ) -> list[str]:
        """Ids of runs with identical settings (and optionally dataset version), oldest first."""
        sql = "SELECT run_id FROM runs WHERE config_hash = ?"
        args: list[Any] = [config_hash]
        if dataset_version is not None:
            sql += " AND dataset_version = ?"
            args.append(dataset_version)
        return [r["run_id"] for r in self._conn.execute(sql + " ORDER BY seq", args)]

    def stale_running(self, started_before: str) -> list[str]:
        """Ids of runs still `running` that started strictly before `started_before`.

        `started_before` is a UTC timestamp in the store's own format (`2026-01-31T12:00:00Z`).
        Oldest first.
        """
        if not isinstance(started_before, str) or not _TIMESTAMP_RE.fullmatch(started_before):
            raise StoreError("started_before must look like 2026-01-31T12:00:00Z")
        rows = self._conn.execute(
            "SELECT run_id FROM runs WHERE status = 'running' AND started_at < ? ORDER BY seq",
            (started_before,),
        )
        return [r["run_id"] for r in rows]

    def _hydrate(self, rows: Sequence[sqlite3.Row]) -> list[RunRecord]:
        ids = [r["run_id"] for r in rows]
        params: dict[str, dict[str, Any]] = {i: {} for i in ids}
        tags: dict[str, dict[str, str]] = {i: {} for i in ids}
        for start in range(0, len(ids), _IN_CHUNK):  # keep under SQLite's variable limit
            chunk = ids[start : start + _IN_CHUNK]
            marks = ",".join("?" * len(chunk))
            for r in self._conn.execute(
                f"SELECT run_id, key, value_json FROM params WHERE run_id IN ({marks})", chunk
            ):
                params[r["run_id"]][r["key"]] = json.loads(r["value_json"])
            for r in self._conn.execute(
                f"SELECT run_id, key, value FROM tags WHERE run_id IN ({marks})", chunk
            ):
                tags[r["run_id"]][r["key"]] = r["value"]
        return [
            RunRecord(
                run_id=r["run_id"],
                experiment_id=r["experiment_id"],
                config_hash=r["config_hash"],
                dataset_id=r["dataset_id"],
                dataset_version=r["dataset_version"],
                seed=r["seed"],
                status=r["status"],
                started_at=r["started_at"],
                finished_at=r["finished_at"],
                error=r["error"],
                environment=json.loads(r["environment_json"]),
                params=params[r["run_id"]],
                tags=tags[r["run_id"]],
            )
            for r in rows
        ]

    def finish_run(self, run_id: str) -> None:
        self._require_running(run_id)
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET status = 'completed', finished_at = ? WHERE run_id = ?",
                (self._clock(), run_id),
            )

    def fail_run(self, run_id: str, error: str) -> None:
        self._require_running(run_id)
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET status = 'failed', finished_at = ?, error = ? WHERE run_id = ?",
                (self._clock(), str(error)[:_MAX_TEXT], run_id),
            )

    # -- metrics ------------------------------------------------------------------------------

    def log_metric(self, run_id: str, name: str, value: float, step: int = 0) -> None:
        self.log_metrics(run_id, {name: value}, step)

    def log_metrics(self, run_id: str, values: Mapping[str, float], step: int = 0) -> None:
        """Write several metrics at one step atomically: all are stored or none are."""
        self._require_running(run_id)
        step = _check_step(step)
        rows = [
            (run_id, _check_name("metric", name), step, _check_value(value))
            for name, value in values.items()
        ]
        if not rows:
            raise StoreError("no metrics given")
        try:
            with self._conn:
                self._conn.executemany(
                    "INSERT INTO metrics (run_id, name, step, value) VALUES (?, ?, ?, ?)", rows
                )
        except sqlite3.IntegrityError as exc:
            raise StoreError(f"metric already logged for run {run_id} at step {step}") from exc

    def get_metrics(self, run_id: str) -> list[MetricPoint]:
        """All points, ordered by metric name then step."""
        self._run_row(run_id)
        rows = self._conn.execute(
            "SELECT name, step, value FROM metrics WHERE run_id = ? ORDER BY name, step",
            (run_id,),
        )
        return [MetricPoint(r["name"], r["step"], r["value"]) for r in rows]

    def final_metrics(self, run_id: str) -> dict[str, float]:
        """The value at each metric's highest step."""
        latest: dict[str, float] = {}
        for point in self.get_metrics(run_id):
            latest[point.name] = point.value  # points arrive in step order per name
        return latest

    # -- artifacts and tags -------------------------------------------------------------------

    def add_artifact(
        self,
        run_id: str,
        name: str,
        kind: str,
        sha256: str,
        size_bytes: int,
        path: str | None = None,
    ) -> None:
        """Record metadata about a file a run produced. The file itself is not stored here."""
        self._require_running(run_id)
        _check_artifact(name, kind, sha256, size_bytes, path)
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO artifacts (run_id, name, kind, sha256, size_bytes, path)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, name, kind, sha256, size_bytes, path),
                )
        except sqlite3.IntegrityError as exc:
            raise StoreError(f"artifact {name!r} already recorded for run {run_id}") from exc

    def get_artifacts(self, run_id: str) -> list[ArtifactRecord]:
        self._run_row(run_id)
        rows = self._conn.execute(
            "SELECT name, kind, sha256, size_bytes, path FROM artifacts"
            " WHERE run_id = ? ORDER BY name",
            (run_id,),
        )
        return [
            ArtifactRecord(r["name"], r["kind"], r["sha256"], r["size_bytes"], r["path"])
            for r in rows
        ]

    def add_note(self, run_id: str, text: str) -> None:
        """Attach a free-text note to a run in any status. Notes are append-only."""
        self._run_row(run_id)
        _check_note(text)
        with self._conn:
            self._conn.execute(
                "INSERT INTO notes (run_id, created_at, text) VALUES (?, ?, ?)",
                (run_id, self._clock(), text),
            )

    def get_notes(self, run_id: str) -> list[NoteRecord]:
        """Notes in the order they were added."""
        self._run_row(run_id)
        rows = self._conn.execute(
            "SELECT created_at, text FROM notes WHERE run_id = ? ORDER BY seq", (run_id,)
        )
        return [NoteRecord(r["created_at"], r["text"]) for r in rows]

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        """Set or replace a tag on a running run."""
        self._require_running(run_id)
        _check_tag(key, value)
        with self._conn:
            self._conn.execute(
                "INSERT INTO tags (run_id, key, value) VALUES (?, ?, ?)"
                " ON CONFLICT (run_id, key) DO UPDATE SET value = excluded.value",
                (run_id, key, value),
            )

    def find_runs_by_tag(self, key: str, value: str) -> list[str]:
        """Run ids carrying the tag `key` = `value`, oldest first."""
        _check_tag(key, value)
        rows = self._conn.execute(
            "SELECT t.run_id FROM tags t JOIN runs r ON r.run_id = t.run_id"
            " WHERE t.key = ? AND t.value = ? ORDER BY r.seq",
            (key, value),
        )
        return [r["run_id"] for r in rows]

    def import_runs(self, runs: Sequence[ImportedRun]) -> list[str]:
        """Insert finished runs recorded elsewhere. All are stored or none are.

        Every run gets a new id from this database and keeps its own timestamps, status, metrics,
        artifacts, tags, and notes. Everything is validated before the first write, and a dataset
        version already registered with a different content hash is refused.
        """
        checked = [self._check_imported(run) for run in runs]
        ids: list[str] = []
        try:
            with self._conn:
                for run, dataset_row in zip(runs, checked, strict=True):
                    ids.append(self._insert_imported(run, dataset_row))
        except sqlite3.Error as exc:
            raise StoreError(f"import failed, nothing was stored: {exc}") from exc
        return ids

    def _check_imported(self, run: ImportedRun) -> bool:
        """Validate one run; return True if its dataset still has to be registered."""
        config, dataset = run.config, run.dataset
        if config.dataset_id != dataset.dataset_id:
            raise StoreError(
                f"config uses dataset {config.dataset_id!r} but {dataset.dataset_id!r} was given"
            )
        if config.dataset_version is not None and config.dataset_version != dataset.version:
            raise StoreError(
                f"config pins dataset version {config.dataset_version} "
                f"but {dataset.version} was given"
            )
        if run.status not in ("completed", "failed"):
            raise StoreError(f"only completed or failed runs can be imported, got {run.status!r}")
        started = _check_timestamp("started_at", run.started_at)
        finished = _check_timestamp("finished_at", run.finished_at)
        if finished < started:
            raise StoreError("finished_at is before started_at")
        if run.status == "failed":
            if not isinstance(run.error, str) or not run.error:
                raise StoreError("a failed run needs an error message")
        elif run.error is not None:
            raise StoreError("a completed run cannot have an error message")
        seen: set[tuple[str, int]] = set()
        for point in run.metrics:
            _check_name("metric", point.name)
            _check_step(point.step)
            _check_value(point.value)
            if (point.name, point.step) in seen:
                raise StoreError(f"metric {point.name!r} appears twice at step {point.step}")
            seen.add((point.name, point.step))
        names: set[str] = set()
        for a in run.artifacts:
            _check_artifact(a.name, a.kind, a.sha256, a.size_bytes, a.path)
            if a.name in names:
                raise StoreError(f"artifact {a.name!r} appears twice")
            names.add(a.name)
        for key, value in run.tags.items():
            _check_tag(key, value)
        for note in run.notes:
            _check_timestamp("note created_at", note.created_at)
            _check_note(note.text)
        existing = self._conn.execute(
            "SELECT sha256 FROM datasets WHERE dataset_id = ? AND version = ?",
            (dataset.dataset_id, dataset.version),
        ).fetchone()
        if existing is not None and existing["sha256"] != dataset.sha256:
            raise StoreError(
                f"dataset {dataset.dataset_id}@{dataset.version} is already registered "
                "with a different content hash"
            )
        return existing is None

    def _insert_imported(self, run: ImportedRun, register: bool) -> str:
        config, dataset = run.config, run.dataset
        now = self._clock()
        if register:
            self._conn.execute(
                "INSERT OR IGNORE INTO datasets (dataset_id, version, sha256, n_rows,"
                " columns_json, label_column, registered_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    dataset.dataset_id,
                    dataset.version,
                    dataset.sha256,
                    dataset.n_rows,
                    json.dumps(list(dataset.columns)),
                    dataset.label_column,
                    now,
                ),
            )
        self._conn.execute(
            "INSERT OR IGNORE INTO experiments (experiment_id, created_at) VALUES (?, ?)",
            (config.experiment_id, now),
        )
        cursor = self._conn.execute(
            "INSERT INTO runs (experiment_id, config_hash, config_json, dataset_id,"
            " dataset_version, seed, status, started_at, finished_at, error, environment_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                config.experiment_id,
                config.config_hash(),
                json.dumps(config.to_dict(), sort_keys=True),
                dataset.dataset_id,
                dataset.version,
                config.seed,
                run.status,
                run.started_at,
                run.finished_at,
                run.error,
                json.dumps(dict(run.environment), sort_keys=True),
            ),
        )
        run_id = f"run-{cursor.lastrowid:06d}"
        self._conn.execute("UPDATE runs SET run_id = ? WHERE seq = ?", (run_id, cursor.lastrowid))
        self._conn.executemany(
            "INSERT INTO params (run_id, key, value_json) VALUES (?, ?, ?)",
            [(run_id, k, json.dumps(v, sort_keys=True)) for k, v in sorted(config.params.items())],
        )
        self._conn.executemany(
            "INSERT INTO metrics (run_id, name, step, value) VALUES (?, ?, ?, ?)",
            [(run_id, p.name, p.step, float(p.value)) for p in run.metrics],
        )
        self._conn.executemany(
            "INSERT INTO artifacts (run_id, name, kind, sha256, size_bytes, path)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [(run_id, a.name, a.kind, a.sha256, a.size_bytes, a.path) for a in run.artifacts],
        )
        self._conn.executemany(
            "INSERT INTO tags (run_id, key, value) VALUES (?, ?, ?)",
            [(run_id, k, v) for k, v in sorted(run.tags.items())],
        )
        self._conn.executemany(
            "INSERT INTO notes (run_id, created_at, text) VALUES (?, ?, ?)",
            [(run_id, n.created_at, n.text) for n in run.notes],
        )
        return run_id
