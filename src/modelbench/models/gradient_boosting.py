"""Gradient boosting for classification with the multinomial log loss."""

from __future__ import annotations

import math
from typing import Any

from ..errors import ModelError
from ..metrics import ordered_labels
from ..seeding import check_seed
from .base import Classifier, Label, Matrix
from .decision_tree import _check_int
from .decision_tree_regressor import DecisionTreeRegressor
from .gradient_boosting_regressor import (
    MAX_STAGES,
    check_learning_rate,
    check_subsample,
    stage_rows,
)

_FLOOR = 1e-12  # a leaf whose second derivative is smaller than this keeps its value at zero


def softmax(scores: list[float]) -> list[float]:
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    total = math.fsum(exps)
    return [e / total for e in exps]


class GradientBoostingClassifier(Classifier):
    """Boosted regression trees on the gradient of the multinomial log loss.

    Scores start at the log of the class frequencies. Each of `n_estimators` stages fits one
    regression tree per class to that class's residual (1 for its rows, 0 for others, minus the
    current probability), then sets each leaf to Friedman's one-step Newton value
    `(K - 1) / K * sum(residual) / sum(|residual| * (1 - |residual|))` and adds `learning_rate`
    times the tree to the class score. `predict_proba` is a softmax of the scores. With
    `subsample` below 1 each stage sees a seeded random share of the rows. `train_loss_` lists the
    mean log loss on the training rows after each stage.
    """

    name = "gradient_boosting"
    _purpose = "gradient_boosting"

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
        self._start: list[float] = []
        self._stages: list[list[DecisionTreeRegressor]] = []
        self.train_loss_: list[float] = []

    def get_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            **self._tree_params,
            "subsample": self.subsample,
            "seed": self.seed,
        }

    def _fit(self, X: Matrix, y: list[Label]) -> None:
        classes = ordered_labels(y)
        k = len(classes)
        n = len(X)
        position = {label: i for i, label in enumerate(classes)}
        target = [position[label] for label in y]
        counts = [target.count(c) for c in range(k)]
        start = [math.log(c / n) for c in counts]
        self._start = start
        self._stages = []
        self.train_loss_ = []
        if k == 1:  # nothing to learn: the only class has probability one
            return
        scores = [list(start) for _ in range(n)]
        for stage in range(self.n_estimators):
            probabilities = [softmax(row) for row in scores]
            rows = stage_rows(n, self.subsample, self.seed, stage, self._purpose)
            trees: list[DecisionTreeRegressor] = []
            for c in range(k):
                residual = {i: (1.0 if target[i] == c else 0.0) - probabilities[i][c] for i in rows}
                tree = DecisionTreeRegressor(**self._tree_params)
                tree.fit([X[i] for i in rows], [residual[i] for i in rows])
                leaf_of = tree.apply([X[i] for i in rows])
                top = [0.0] * tree.n_leaves_
                bottom = [0.0] * tree.n_leaves_
                for i, leaf in zip(rows, leaf_of, strict=True):
                    r = residual[i]
                    top[leaf] += r
                    bottom[leaf] += abs(r) * (1.0 - abs(r))
                factor = (k - 1) / k
                tree.set_leaf_values(
                    [
                        factor * t / b if b > _FLOOR else 0.0
                        for t, b in zip(top, bottom, strict=True)
                    ]
                )
                for i, step in enumerate(tree.predict(X)):
                    scores[i][c] += self.learning_rate * step
                trees.append(tree)
            self._stages.append(trees)
            loss = math.fsum(
                -math.log(max(softmax(scores[i])[target[i]], 1e-300)) for i in range(n)
            )
            self.train_loss_.append(loss / n)

    def _scores(self, X: Matrix) -> Matrix:
        scores = [list(self._start) for _ in X]
        for trees in self._stages:
            for c, tree in enumerate(trees):
                for i, step in enumerate(tree.predict(X)):
                    scores[i][c] += self.learning_rate * step
        return scores

    def _predict_proba(self, X: Matrix) -> Matrix:
        return [softmax(row) for row in self._scores(X)]
