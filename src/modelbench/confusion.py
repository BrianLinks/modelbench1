"""Store a confusion matrix in a run tag and read it back.

A classification run records its test-split confusion matrix as the tag `confusion_matrix`
(compact JSON: `{"labels": [...], "matrix": [[...], ...]}`). Tags are limited to 2000
characters, so a matrix too large to fit is simply not recorded. Reading is strict: a tag that
was edited into something else is an error, not a guess.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from .errors import MetricError
from .metrics import ConfusionMatrix

TAG = "confusion_matrix"
MAX_TAG_LENGTH = 2000  # the store's limit for a tag value


def to_tag(cm: ConfusionMatrix) -> str | None:
    """The tag text for `cm`, or None if it does not fit in a tag."""
    labels = [str(label) for label in cm.labels]
    text = json.dumps(
        {"labels": labels, "matrix": [list(r) for r in cm.matrix]}, separators=(",", ":")
    )
    return text if len(text) <= MAX_TAG_LENGTH else None


def from_tag(text: str) -> ConfusionMatrix:
    """Parse a `confusion_matrix` tag. Raises MetricError if it is not a square count table."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        raise MetricError("the confusion matrix tag is not valid JSON") from None
    if not isinstance(data, dict) or set(data) != {"labels", "matrix"}:
        raise MetricError("the confusion matrix tag must hold exactly labels and matrix")
    labels, matrix = data["labels"], data["matrix"]
    if not isinstance(labels, list) or not labels or not all(isinstance(x, str) for x in labels):
        raise MetricError("confusion matrix labels must be a non-empty list of strings")
    if len(set(labels)) != len(labels):
        raise MetricError("confusion matrix labels must be unique")
    size = len(labels)
    if not isinstance(matrix, list) or len(matrix) != size:
        raise MetricError("the confusion matrix must have one row per label")
    rows: list[tuple[int, ...]] = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != size:
            raise MetricError("the confusion matrix must be square")
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, int) or cell < 0:
                raise MetricError("confusion matrix cells must be non-negative integers")
        rows.append(tuple(row))
    return ConfusionMatrix(tuple(labels), tuple(rows))


def row_fractions(cm: ConfusionMatrix) -> list[list[float]]:
    """Each row divided by its total (that class's recall breakdown); empty rows stay zero."""
    out: list[list[float]] = []
    for row in cm.matrix:
        total = sum(row)
        out.append([cell / total if total else 0.0 for cell in row])
    return out


def format_text(cm: ConfusionMatrix, indent: str = "") -> list[str]:
    """A plain-text grid: rows are true labels, columns are predicted labels."""
    names: Sequence[str] = [str(label) for label in cm.labels]
    corner = "true/pred"
    first = max(len(corner), *(len(n) for n in names))
    cell = max(len(n) for n in names)
    cell = max(cell, *(len(str(c)) for r in cm.matrix for c in r))
    lines = [f"{indent}{corner:<{first}}  " + " ".join(f"{n:>{cell}}" for n in names)]
    for name, row in zip(names, cm.matrix, strict=True):
        lines.append(f"{indent}{name:<{first}}  " + " ".join(f"{c:>{cell}}" for c in row))
    return lines
