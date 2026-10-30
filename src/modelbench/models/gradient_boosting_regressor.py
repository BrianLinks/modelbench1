"""Gradient boosting for regression: regression trees fitted one after another to the errors."""

from __future__ import annotations

import math
from typing import Any

from ..errors import ModelError
from ..seeding import check_seed, rng_for
from .base import Matrix, Regressor
from .decision_tree import _check_int
from .decision_tree_regressor import DecisionTreeRegressor

MAX_STAGES = 1000


def check_learning_rate(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(f"learning_rate must be a number in (0, 1], got {value!r}")
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ModelError(f"learning_rate must be in (0, 1], got {value!r}")
    return float(value)


def check_subsample(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(f"subsample must be a number in (0, 1], got {value!r}")
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ModelError(f"subsample must be in (0, 1], got {value!r}")
    return float(value)


def stage_rows(n: int, subsample: float, seed: int, stage: int, purpose: str) -> list[int]:
    """The training rows stage `stage` is fitted to: all of them, or a seeded share, in order."""
    if subsample >= 1.0:
        return list(range(n))
    count = max(2, round(subsample * n))
    count = min(count, n)
    return sorted(rng_for(seed, purpose, stage).sample(range(n), count))


class GradientBoostingRegressor(Regressor):
    """Squared-error gradient boosting.

    The model starts at the mean target. Each of `n_estimators` stages fits a small regression
    tree (`max_depth`, `min_samples_split`, `min_samples_leaf`) to the current residuals and adds
    `learning_rate` times its prediction. With `subsample` below 1 each stage sees a seeded random
    share of the rows (stochastic gradient boosting). `train_loss_` lists the mean squared error
    on the training rows after each stage, so a flat tail shows more stages would not help.
    """

    name = "gradient_boosting_regressor"
    _purpose = "gradient_boosting_regressor"

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_estimators = _check_int("n_estimators", n_estimators, 1)
        if self.n_estimators > MAX_STAGES:
            raise ModelError(f"n_estimators must be at most {MAX_STAGES}, got {n_estimators}")
        self.learning_rate = check_learning_rate(learning_rate)
        self.max_depth = _check_int("max_depth", max_depth, 1)
        self._tree_params = {
            "max_depth": self.max_depth,
            "min_samples_split": _check_int("min_samples_split", min_samples_split, 2),
            "min_samples_leaf": _check_int("min_samples_leaf", min_samples_leaf, 1),
        }
        self.subsample = check_subsample(subsample)
        self.seed = check_seed(seed)
        self._start = 0.0
        self._trees: list[DecisionTreeRegressor] = []
        self.train_loss_: list[float] = []

    def get_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            **self._tree_params,
            "subsample": self.subsample,
            "seed": self.seed,
        }

    def _fit(self, X: Matrix, y: list[float]) -> None:
        n = len(X)
        start = math.fsum(y) / n
        scores = [start] * n
        trees: list[DecisionTreeRegressor] = []
        losses: list[float] = []
        for stage in range(self.n_estimators):
            rows = stage_rows(n, self.subsample, self.seed, stage, self._purpose)
            tree = DecisionTreeRegressor(**self._tree_params)
            tree.fit([X[i] for i in rows], [y[i] - scores[i] for i in rows])
            for i, step in enumerate(tree.predict(X)):
                scores[i] += self.learning_rate * step
            trees.append(tree)
            losses.append(math.fsum((y[i] - scores[i]) ** 2 for i in range(n)) / n)
        self._start = start
        self._trees = trees
        self.train_loss_ = losses

    def _predict(self, X: Matrix) -> list[float]:
        scores = [self._start] * len(X)
        for tree in self._trees:
            for i, step in enumerate(tree.predict(X)):
                scores[i] += self.learning_rate * step
        return scores

    def staged_predict(self, X: Matrix) -> list[list[float]]:
        """Predictions after 1, 2, ... `n_estimators` stages, to see how many are worth keeping."""
        if not self.is_fitted:
            raise ModelError(f"{self.name} has not been fitted yet")
        scores = [self._start] * len(X)
        out: list[list[float]] = []
        for tree in self._trees:
            for i, step in enumerate(tree.predict(X)):
                scores[i] += self.learning_rate * step
            out.append(list(scores))
        return out
