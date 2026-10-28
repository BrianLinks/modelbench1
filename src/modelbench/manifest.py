"""A dataset manifest: a deterministic summary of a table, kept next to a dataset version.

The manifest says what is in the data without shipping the data: the version hash, the row
count, and for each column its kind, missing cells, distinct values and simple statistics. The
same table always gives the same manifest, byte for byte, so two manifests can be compared or
stored in a repository. A column is `numeric` when every non-missing cell is a finite number and
`text` otherwise; the missing-cell rules are those of preprocessing (`preprocess.is_missing`).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

from .datasets import Table, make_version
from .preprocess import NUMBER_RE, is_missing

FORMAT = "modelbench-manifest"
FORMAT_VERSION = 1
TOP_VALUES = 5


def _numbers(cells: list[str]) -> list[float] | None:
    """The cells as floats, or None when any non-missing cell is not a finite number."""
    out: list[float] = []
    for cell in cells:
        text = cell.strip()
        if NUMBER_RE.fullmatch(text) is None:
            return None
        value = float(text)
        if not math.isfinite(value):
            return None
        out.append(value)
    return out


def _describe(name: str, cells: list[str]) -> dict[str, Any]:
    present = [c for c in cells if not is_missing(c)]
    info: dict[str, Any] = {
        "name": name,
        "missing": len(cells) - len(present),
        "distinct": len({c.strip() for c in present}),
    }
    numbers = _numbers(present) if present else None
    if numbers is not None:
        mean = math.fsum(numbers) / len(numbers)
        variance = math.fsum((v - mean) ** 2 for v in numbers) / len(numbers)
        info.update(
            kind="numeric",
            min=min(numbers),
            max=max(numbers),
            mean=mean,
            std=math.sqrt(variance),
        )
    else:
        counts = Counter(c.strip() for c in present)
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        info.update(kind="text", top=[[value, count] for value, count in ranked[:TOP_VALUES]])
    return dict(sorted(info.items()))


def build_manifest(
    dataset_id: str, table: Table, label_column: str | None = None
) -> dict[str, Any]:
    """The manifest of `table` as a plain dictionary (see `manifest_json`)."""
    version = make_version(dataset_id, table, label_column)
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "dataset_id": version.dataset_id,
        "version": version.version,
        "sha256": version.sha256,
        "n_rows": version.n_rows,
        "label_column": version.label_column,
        "columns": [_describe(name, table.column(name)) for name in table.columns],
    }
    if label_column is not None:
        counts = Counter(c.strip() for c in table.column(label_column))
        manifest["labels"] = dict(sorted(counts.items()))
    return manifest


def manifest_json(manifest: dict[str, Any]) -> str:
    """Canonical text for a manifest: sorted keys, two-space indent, trailing newline."""
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"
