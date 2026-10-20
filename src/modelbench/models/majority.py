"""Majority-class baseline: always predicts the most frequent training label."""

from __future__ import annotations

from typing import Any

from ..metrics import ordered_labels
from .base import Classifier, Label, Matrix, class_priors


class MajorityClassifier(Classifier):
    """Predicts the most common training class; ties go to the first class in `classes_`.

    `predict_proba` returns the training class priors for every row, which makes this the
    reference point for log loss as well as accuracy.
    """

    name = "majority"

    def __init__(self) -> None:
        super().__init__()
        self._priors: list[float] = []

    def get_params(self) -> dict[str, Any]:
        return {}

    def _fit(self, X: Matrix, y: list[Label]) -> None:
        self._priors = class_priors(y, ordered_labels(y))

    def _predict_proba(self, X: Matrix) -> Matrix:
        return [list(self._priors) for _ in X]
