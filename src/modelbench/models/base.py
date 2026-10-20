"""Shared classifier interface and input validation for the pure-Python baseline models."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from typing import Any, ClassVar

from ..errors import ModelError
from ..metrics import ordered_labels

Label = Hashable
Matrix = list[list[float]]


def check_features(X: Sequence[Sequence[float]], where: str = "X") -> Matrix:
    """Validate a rectangular, finite numeric table and return it as lists of floats."""
    if isinstance(X, (str, bytes)) or not isinstance(X, Sequence):
        raise ModelError(f"{where} must be a sequence of rows")
    if len(X) == 0:
        raise ModelError(f"{where} must not be empty")
    rows: Matrix = []
    width: int | None = None
    for position, row in enumerate(X):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ModelError(f"{where}[{position}] must be a sequence of numbers")
        if width is None:
            width = len(row)
            if width == 0:
                raise ModelError(f"{where} rows must have at least one feature")
        elif len(row) != width:
            raise ModelError(f"{where}[{position}] has {len(row)} features, expected {width}")
        values: list[float] = []
        for column, cell in enumerate(row):
            if isinstance(cell, bool) or not isinstance(cell, (int, float)):
                raise ModelError(f"{where}[{position}][{column}] must be a number, got {cell!r}")
            number = float(cell)
            if not math.isfinite(number):
                raise ModelError(f"{where}[{position}][{column}] must be finite, got {cell!r}")
            values.append(number)
        rows.append(values)
    return rows


def check_labels(y: Sequence[Label], n_rows: int) -> list[Label]:
    if isinstance(y, (str, bytes)) or not isinstance(y, Sequence):
        raise ModelError("y must be a sequence of labels")
    if len(y) != n_rows:
        raise ModelError(f"X has {n_rows} rows but y has {len(y)} labels")
    for position, label in enumerate(y):
        if label is None:
            raise ModelError(f"y[{position}] is None; labels must be hashable values")
        try:
            hash(label)
        except TypeError:
            raise ModelError(f"y[{position}] is not hashable: {label!r}") from None
    return list(y)


class Classifier(ABC):
    """A classifier with a fit / predict / predict_proba interface.

    `classes_` lists the training labels in sorted order (first-seen order if they cannot be
    sorted). Probability rows follow that order. Ties between classes always resolve to the
    class that comes first in `classes_`, so predictions never depend on dict or set ordering.
    """

    name: ClassVar[str]

    def __init__(self) -> None:
        self._classes: tuple[Label, ...] | None = None
        self._n_features: int | None = None

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """The constructor arguments needed to rebuild an equivalent unfitted model."""

    @abstractmethod
    def _fit(self, X: Matrix, y: list[Label]) -> None: ...

    @abstractmethod
    def _predict_proba(self, X: Matrix) -> Matrix: ...

    @property
    def classes_(self) -> tuple[Label, ...]:
        if self._classes is None:
            raise ModelError(f"{self.name} has not been fitted yet")
        return self._classes

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[Label]) -> None:
        rows = check_features(X)
        labels = check_labels(y, len(rows))
        self._classes = None  # a failed refit must not leave a half-trained model usable
        self._n_features = None
        classes = ordered_labels(labels)
        self._fit(rows, labels)
        self._classes = classes
        self._n_features = len(rows[0])

    def _check_predict_input(self, X: Sequence[Sequence[float]]) -> Matrix:
        _ = self.classes_  # raises if not fitted
        rows = check_features(X)
        if len(rows[0]) != self._n_features:
            raise ModelError(
                f"model was fitted with {self._n_features} features, got {len(rows[0])}"
            )
        return rows

    def predict_proba(self, X: Sequence[Sequence[float]]) -> Matrix:
        return self._predict_proba(self._check_predict_input(X))

    def predict(self, X: Sequence[Sequence[float]]) -> list[Label]:
        classes = self.classes_
        out: list[Label] = []
        for row in self._predict_proba(self._check_predict_input(X)):
            best = 0
            for index in range(1, len(row)):
                if row[index] > row[best]:
                    best = index
            out.append(classes[best])
        return out


def class_priors(labels: Sequence[Label], classes: Sequence[Label]) -> list[float]:
    """Fraction of `labels` equal to each class, in the order of `classes`."""
    counts = {c: 0 for c in classes}
    for label in labels:
        counts[label] += 1
    return [counts[c] / len(labels) for c in classes]
