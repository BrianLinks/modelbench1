"""Import runs from a `modelbench-runs` JSON document (the output of `export_json`).

The document is checked completely before anything is written: its format and version, the
shape of every run, and that each run's config hash, seed, experiment, and dataset agree with the
config inside it. Runs are then inserted in one transaction, so a bad document changes nothing.

Every imported run gets a new id from the receiving database and keeps its original timestamps,
metrics, artifacts, tags, and notes. A tag `imported_from` records the id it had where it was
exported (an existing `imported_from` tag is kept, so the original source survives re-exports).
A run already present - same config, dataset version, timestamps, status, and metrics - is
skipped, so importing the same file twice adds nothing.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .datasets import DatasetVersion
from .errors import ConfigError, IdentifierError, RunImportError
from .export import FORMAT, VERSION
from .ids import validate_id
from .store import ArtifactRecord, ImportedRun, MetricPoint, NoteRecord, Store

MAX_FILE_BYTES = 50_000_000
MAX_RUNS = 10_000
MAX_METRIC_POINTS = 100_000
IMPORT_TAG = "imported_from"

_RUN_KEYS = frozenset(
    {
        "run_id", "experiment_id", "status", "started_at", "finished_at", "error", "seed",
        "config", "config_hash", "dataset", "environment", "tags", "metrics", "artifacts",
        "notes",
    }
)  # fmt: skip
_DATASET_KEYS = frozenset({"dataset_id", "version", "sha256", "n_rows", "columns", "label_column"})
_METRIC_KEYS = frozenset({"name", "step", "value"})
_ARTIFACT_KEYS = frozenset({"name", "kind", "sha256", "size_bytes", "path"})
_NOTE_KEYS = frozenset({"created_at", "text"})


@dataclass(frozen=True)
class ImportReport:
    imported: tuple[tuple[str, str], ...]  # (id in the document, new id)
    skipped: tuple[tuple[str, str], ...]  # (id in the document, id of the matching existing run)


def _reject_constant(name: str) -> Any:
    raise RunImportError(f"the document contains {name}, which is not valid JSON data")


def parse_document(text: str) -> Any:
    if not isinstance(text, str):
        raise RunImportError("the document must be text")
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise RunImportError(f"the document is not valid JSON: {exc}") from exc
    except RecursionError:
        raise RunImportError("the document is nested too deeply") from None


def _object(value: object, where: str, keys: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RunImportError(f"{where} must be an object")
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing:
        raise RunImportError(f"{where} is missing {', '.join(missing)}")
    if extra:
        raise RunImportError(f"{where} has unknown key(s) {', '.join(extra)}")
    return value


def _list(value: object, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise RunImportError(f"{where} must be a list")
    return value


def _parse_dataset(raw: object, where: str) -> DatasetVersion:
    data = _object(raw, where, _DATASET_KEYS)
    columns = _list(data["columns"], f"{where}.columns")
    if not all(isinstance(c, str) for c in columns):
        raise RunImportError(f"{where}.columns must be a list of strings")
    n_rows = data["n_rows"]
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 0:
        raise RunImportError(f"{where}.n_rows must be a non-negative integer")
    label = data["label_column"]
    if label is not None and label not in columns:
        raise RunImportError(f"{where}.label_column {label!r} is not one of the columns")
    version, sha = data["version"], data["sha256"]
    for field_name, value, size in (("version", version, 12), ("sha256", sha, 64)):
        if (
            not isinstance(value, str)
            or len(value) != size
            or any(ch not in "0123456789abcdef" for ch in value)
        ):
            raise RunImportError(f"{where}.{field_name} must be {size} lowercase hex characters")
    if sha[: len(version)] != version:
        raise RunImportError(
            f"{where}.version must be the first {len(version)} characters of sha256"
        )
    try:
        validate_id("dataset", data["dataset_id"])
    except IdentifierError as exc:
        raise RunImportError(f"{where}.dataset_id: {exc}") from exc
    return DatasetVersion(data["dataset_id"], version, sha, n_rows, tuple(columns), label)


def _parse_environment(raw: object, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or len(raw) > 200:
        raise RunImportError(f"{where} must be an object with at most 200 entries")
    for key, value in raw.items():
        if value is not None and not isinstance(value, str):
            raise RunImportError(f"{where}.{key} must be a string or null")
    return dict(raw)


def _parse_run(raw: object, where: str) -> tuple[str, ImportedRun]:
    data = _object(raw, where, _RUN_KEYS)
    source_id = data["run_id"]
    try:
        validate_id("run", source_id)
        config = ExperimentConfig.from_dict(data["config"])
    except (IdentifierError, ConfigError) as exc:
        raise RunImportError(f"{where}: {exc}") from exc
    if config.experiment_id != data["experiment_id"]:
        raise RunImportError(f"{where}: experiment_id does not match the config")
    seed = data["seed"]
    if isinstance(seed, bool) or seed != config.seed:
        raise RunImportError(f"{where}: seed does not match the config")
    if data["config_hash"] != config.config_hash():
        raise RunImportError(f"{where}: config_hash does not match the config")
    dataset = _parse_dataset(data["dataset"], f"{where}.dataset")
    metrics_raw = _list(data["metrics"], f"{where}.metrics")
    if len(metrics_raw) > MAX_METRIC_POINTS:
        raise RunImportError(f"{where}.metrics has more than {MAX_METRIC_POINTS} points")
    metrics = []
    for i, item in enumerate(metrics_raw):
        point = _object(item, f"{where}.metrics[{i}]", _METRIC_KEYS)
        value = point["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RunImportError(f"{where}.metrics[{i}].value must be a number")
        if not math.isfinite(value):
            raise RunImportError(f"{where}.metrics[{i}].value must be finite")
        metrics.append(MetricPoint(point["name"], point["step"], float(value)))
    artifacts = []
    for i, item in enumerate(_list(data["artifacts"], f"{where}.artifacts")):
        a = _object(item, f"{where}.artifacts[{i}]", _ARTIFACT_KEYS)
        artifacts.append(
            ArtifactRecord(a["name"], a["kind"], a["sha256"], a["size_bytes"], a["path"])
        )
    notes = []
    for i, item in enumerate(_list(data["notes"], f"{where}.notes")):
        n = _object(item, f"{where}.notes[{i}]", _NOTE_KEYS)
        notes.append(NoteRecord(n["created_at"], n["text"]))
    tags = data["tags"]
    if not isinstance(tags, dict):
        raise RunImportError(f"{where}.tags must be an object")
    tags = dict(tags)
    tags.setdefault(IMPORT_TAG, source_id)
    run = ImportedRun(
        config=config,
        dataset=dataset,
        environment=_parse_environment(data["environment"], f"{where}.environment"),
        status=data["status"],
        started_at=data["started_at"],
        finished_at=data["finished_at"],
        error=data["error"],
        metrics=tuple(metrics),
        artifacts=tuple(artifacts),
        tags=tags,
        notes=tuple(notes),
    )
    return source_id, run


def parse_runs(document: object) -> list[tuple[str, ImportedRun]]:
    """Validate a decoded document and return `(id in the document, run)` pairs."""
    top = _object(document, "the document", frozenset({"format", "version", "runs"}))
    if top["format"] != FORMAT:
        raise RunImportError(f"not a {FORMAT} document (format is {top['format']!r})")
    version = top["version"]
    if isinstance(version, bool) or version != VERSION:
        raise RunImportError(f"unsupported version {version!r}; this build reads version {VERSION}")
    runs = _list(top["runs"], "runs")
    if not runs:
        raise RunImportError("the document contains no runs")
    if len(runs) > MAX_RUNS:
        raise RunImportError(f"the document has more than {MAX_RUNS} runs")
    parsed = [_parse_run(raw, f"runs[{i}]") for i, raw in enumerate(runs)]
    ids = [source for source, _ in parsed]
    if len(set(ids)) != len(ids):
        raise RunImportError("run ids in the document must not repeat")
    return parsed


def _signature(
    config_hash: str,
    dataset_version: str,
    status: str,
    started_at: str,
    finished_at: str | None,
    metrics: Mapping[tuple[str, int], float],
) -> tuple[Any, ...]:
    return (
        config_hash,
        dataset_version,
        status,
        started_at,
        finished_at,
        tuple(sorted(metrics.items())),
    )


def _existing_match(store: Store, run: ImportedRun) -> str | None:
    wanted = _signature(
        run.config.config_hash(),
        run.dataset.version,
        run.status,
        run.started_at,
        run.finished_at,
        {(p.name, p.step): p.value for p in run.metrics},
    )
    for run_id in store.find_runs_by_config(run.config.config_hash(), run.dataset.version):
        record = store.get_run(run_id)
        found = _signature(
            record.config_hash,
            record.dataset_version,
            record.status,
            record.started_at,
            record.finished_at,
            {(p.name, p.step): p.value for p in store.get_metrics(run_id)},
        )
        if found == wanted:
            return run_id
    return None


def import_runs_json(store: Store, text: str, *, dry_run: bool = False) -> ImportReport:
    """Import the runs in an export document. With `dry_run`, validate and report only."""
    parsed = parse_runs(parse_document(text))
    imported: list[tuple[str, ImportedRun]] = []
    skipped: list[tuple[str, str]] = []
    for source, run in parsed:
        match = _existing_match(store, run)
        if match is not None:
            skipped.append((source, match))
        else:
            imported.append((source, run))
    if dry_run:
        store.check_import([run for _, run in imported])  # the same checks a real import makes
        return ImportReport(tuple((source, "") for source, _ in imported), tuple(skipped))
    new_ids = store.import_runs([run for _, run in imported]) if imported else []
    return ImportReport(
        tuple((source, new) for (source, _), new in zip(imported, new_ids, strict=True)),
        tuple(skipped),
    )


def read_import_file(path: str | Path) -> str:
    """Read a JSON export from `path` (UTF-8, at most 50 MB)."""
    target = Path(path)
    try:
        if target.stat().st_size > MAX_FILE_BYTES:
            raise RunImportError(f"{target} is larger than {MAX_FILE_BYTES} bytes")
        return target.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise RunImportError(f"cannot read {target}: {exc}") from exc
    except UnicodeDecodeError:
        raise RunImportError(f"{target} is not valid UTF-8") from None


def import_file(store: Store, path: str | Path, *, dry_run: bool = False) -> ImportReport:
    """Read a JSON export from `path` and import it."""
    return import_runs_json(store, read_import_file(path), dry_run=dry_run)
