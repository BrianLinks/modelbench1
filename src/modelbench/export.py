"""Export stored runs as a self-describing JSON document or a flat metrics CSV.

The JSON layout (`format: "modelbench-runs"`, `version: 1`) is the shared result-record shape:
each run carries its full config, dataset version, environment, final status, every metric point,
artifact metadata, tags, and notes, so another tool can read it without the database.
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .errors import ExportError
from .store import Store

FORMAT = "modelbench-runs"
VERSION = 1


def _check_run_ids(run_ids: Sequence[str]) -> list[str]:
    if isinstance(run_ids, str):
        raise ExportError("run_ids must be a list of run ids, not a single string")
    ids = list(run_ids)
    if not ids:
        raise ExportError("no runs to export")
    if len(set(ids)) != len(ids):
        raise ExportError("run ids must not repeat")
    return ids


def export_runs(store: Store, run_ids: Sequence[str]) -> dict[str, Any]:
    """Build the JSON-serialisable export for `run_ids`, in the order given."""
    runs = []
    for run_id in _check_run_ids(run_ids):
        record = store.get_run(run_id)
        dataset = store.get_dataset(record.dataset_id, record.dataset_version)
        runs.append(
            {
                "run_id": record.run_id,
                "experiment_id": record.experiment_id,
                "status": record.status,
                "started_at": record.started_at,
                "finished_at": record.finished_at,
                "error": record.error,
                "seed": record.seed,
                "config": store.get_config(run_id).to_dict(),
                "config_hash": record.config_hash,
                "dataset": {
                    "dataset_id": dataset.dataset_id,
                    "version": dataset.version,
                    "sha256": dataset.sha256,
                    "n_rows": dataset.n_rows,
                    "columns": list(dataset.columns),
                    "label_column": dataset.label_column,
                },
                "environment": record.environment,
                "tags": dict(sorted(record.tags.items())),
                "metrics": [
                    {"name": p.name, "step": p.step, "value": p.value}
                    for p in store.get_metrics(run_id)
                ],
                "artifacts": [
                    {
                        "name": a.name,
                        "kind": a.kind,
                        "sha256": a.sha256,
                        "size_bytes": a.size_bytes,
                        "path": a.path,
                    }
                    for a in store.get_artifacts(run_id)
                ],
                "notes": [
                    {"created_at": n.created_at, "text": n.text} for n in store.get_notes(run_id)
                ],
            }
        )
    return {"format": FORMAT, "version": VERSION, "runs": runs}


def export_json(store: Store, run_ids: Sequence[str]) -> str:
    """The export as stable, indented JSON text (sorted keys, trailing newline)."""
    return json.dumps(export_runs(store, run_ids), indent=2, sort_keys=True) + "\n"


def export_metrics_csv(store: Store, run_ids: Sequence[str]) -> str:
    """One row per metric point: run_id, experiment_id, metric, step, value."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["run_id", "experiment_id", "metric", "step", "value"])
    for run_id in _check_run_ids(run_ids):
        experiment_id = store.get_run(run_id).experiment_id
        for point in store.get_metrics(run_id):
            writer.writerow([run_id, experiment_id, point.name, point.step, repr(point.value)])
    return out.getvalue()


def write_text_atomic(path: str | Path, text: str) -> None:
    """Write `text` to `path` via a temporary file, so a failure never leaves a partial file.

    The file is created owner-readable only (mode 0600), as temporary files are.
    """
    target = Path(path)
    try:
        fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=".tmp")
    except OSError as exc:
        raise ExportError(f"cannot write {target}: {exc}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp_name, target)
    except OSError as exc:
        Path(tmp_name).unlink(missing_ok=True)
        raise ExportError(f"cannot write {target}: {exc}") from exc
