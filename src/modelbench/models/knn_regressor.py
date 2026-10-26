"""k-nearest-neighbours regressor with deterministic tie-breaking."""

from __future__ import annotations

import math
from typing import Any

from ..errors import ModelError
from .base import Matrix, Regressor
from .knn import METRICS, WEIGHTINGS, _euclidean, _manhattan


class KNNRegressor(Regressor):
    """Predicts the mean target of the `k` closest training rows.

    The neighbour rules match `KNNClassifier`:
    - Neighbours at equal distance are ranked by training-row position (earlier rows first).
    - With weights="distance" each neighbour counts 1/distance in a weighted mean. If any chosen
      neighbour sits at distance zero, only the zero-distance neighbours count, equally.
    Features are used as given: scale them first if their units differ.
    """

    name = "knn_regressor"

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
        self._train_y: list[float] = []

    def get_params(self) -> dict[str, Any]:
        return {"k": self.k, "metric": self.metric, "weights": self.weights}

    def _fit(self, X: Matrix, y: list[float]) -> None:
        if self.k > len(X):
            raise ModelError(f"k={self.k} is larger than the {len(X)} training rows")
        self._train_X = [list(row) for row in X]
        self._train_y = list(y)

    def _predict(self, X: Matrix) -> list[float]:
        distance = _euclidean if self.metric == "euclidean" else _manhattan
        output: list[float] = []
        for query in X:
            distances = [distance(query, row) for row in self._train_X]
            order = sorted(range(len(distances)), key=lambda i: (distances[i], i))
            chosen = order[: self.k]
            if self.weights == "uniform":
                weighted = [(i, 1.0) for i in chosen]
            else:
                exact = [i for i in chosen if distances[i] == 0.0]
                weighted = [(i, 1.0) for i in exact] or [(i, 1.0 / distances[i]) for i in chosen]
            total = math.fsum(w for _, w in weighted)
            output.append(math.fsum(w * self._train_y[i] for i, w in weighted) / total)
        return output
