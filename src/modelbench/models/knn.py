"""k-nearest-neighbours classifier with deterministic tie-breaking."""

from __future__ import annotations

import math
from typing import Any

from ..errors import ModelError
from ..metrics import ordered_labels
from .base import Classifier, Label, Matrix

METRICS = ("euclidean", "manhattan")
WEIGHTINGS = ("uniform", "distance")


def _euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def _manhattan(a: list[float], b: list[float]) -> float:
    return math.fsum(abs(x - y) for x, y in zip(a, b, strict=True))


class KNNClassifier(Classifier):
    """Votes among the `k` closest training rows.

    Determinism rules:
    - Neighbours at equal distance are ranked by training-row position (earlier rows first).
    - Vote ties resolve to the first class in `classes_` (see `Classifier`).
    - With weights="distance" each neighbour votes 1/distance. If any chosen neighbour sits at
      distance zero, only the zero-distance neighbours vote (each with weight 1).
    Features are used as given: scale them first if their units differ.
    """

    name = "knn"

    def __init__(self, k: int = 5, metric: str = "euclidean", weights: str = "uniform") -> None:
        super().__init__()
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ModelError(f"k must be a positive integer, got {k!r}")
        if metric not in METRICS:
            raise ModelError(f"metric must be one of {METRICS}, got {metric!r}")
        if weights not in WEIGHTINGS:
            raise ModelError(f"weights must be one of {WEIGHTINGS}, got {weights!r}")
        self.k = k
        self.metric = metric
        self.weights = weights
        self._train_X: Matrix = []
        self._train_y: list[Label] = []
        self._class_index: dict[Label, int] = {}

    def get_params(self) -> dict[str, Any]:
        return {"k": self.k, "metric": self.metric, "weights": self.weights}

    def _fit(self, X: Matrix, y: list[Label]) -> None:
        if self.k > len(X):
            raise ModelError(f"k={self.k} is larger than the {len(X)} training rows")
        self._train_X = [list(row) for row in X]
        self._train_y = list(y)
        self._class_index = {c: i for i, c in enumerate(ordered_labels(y))}

    def _predict_proba(self, X: Matrix) -> Matrix:
        distance = _euclidean if self.metric == "euclidean" else _manhattan
        n_classes = len(self._class_index)
        output: Matrix = []
        for query in X:
            distances = [distance(query, row) for row in self._train_X]
            order = sorted(range(len(distances)), key=lambda i: (distances[i], i))
            chosen = order[: self.k]
            if self.weights == "uniform":
                votes_for = [(i, 1.0) for i in chosen]
            else:
                exact = [i for i in chosen if distances[i] == 0.0]
                if exact:
                    votes_for = [(i, 1.0) for i in exact]
                else:
                    votes_for = [(i, 1.0 / distances[i]) for i in chosen]
            tally = [0.0] * n_classes
            for index, weight in votes_for:
                tally[self._class_index[self._train_y[index]]] += weight
            total = math.fsum(tally)
            output.append([votes / total for votes in tally])
        return output
