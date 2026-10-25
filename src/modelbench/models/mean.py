"""Mean baseline regressor: always predicts the training mean."""

from __future__ import annotations

import math
from typing import Any

from .base import Matrix, Regressor


class MeanRegressor(Regressor):
    """Predicts the mean of the training targets for every row (the reference point for R²)."""

    name = "mean"

    def __init__(self) -> None:
        super().__init__()
        self._mean = 0.0

    def get_params(self) -> dict[str, Any]:
        return {}

    def _fit(self, X: Matrix, y: list[float]) -> None:
        self._mean = math.fsum(y) / len(y)

    def _predict(self, X: Matrix) -> list[float]:
        return [self._mean] * len(X)
