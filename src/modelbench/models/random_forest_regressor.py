"""A random forest for regression: bootstrap trees that see random features at each split."""

from __future__ import annotations

import math
from typing import Any

from ..errors import ModelError
from ..seeding import check_seed
from .bagged_trees import MAX_ESTIMATORS, bootstrap_indices
from .base import Matrix, Regressor
from .decision_tree import _check_int
from .decision_tree_regressor import DecisionTreeRegressor
from .random_forest import MaxFeatures, check_max_features, feature_chooser

PURPOSE = "random_forest_regressor"


class RandomForestRegressor(Regressor):
    """Averages `n_estimators` regression trees, each fitted to a bootstrap sample.

    Every split looks at `max_features` randomly chosen features (the options are those of
    `RandomForest`). The default, 1.0, looks at all of them, which is plain bagging; a smaller
    value decorrelates the trees but needs enough features, and with few of them it can do worse
    than one tree, so tune it. Repeatable per
    `seed`. `oob_score_` is R² of the out-of-bag predictions: each training row is predicted by
    the trees that did not see it, and rows every tree saw are left out. It is None when fewer
    than two rows are out of bag or their targets are all equal.
    """

    name = "random_forest_regressor"

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_estimators = _check_int("n_estimators", n_estimators, 1)
        if self.n_estimators > MAX_ESTIMATORS:
            raise ModelError(f"n_estimators must be at most {MAX_ESTIMATORS}, got {n_estimators}")
        self._tree_params = DecisionTreeRegressor(
            max_depth, min_samples_split, min_samples_leaf
        ).get_params()
        self.max_features = check_max_features(max_features)
        self.seed = check_seed(seed)
        self._trees: list[DecisionTreeRegressor] = []
        self.oob_score_: float | None = None

    def get_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            **self._tree_params,
            "max_features": self.max_features,
            "seed": self.seed,
        }

    def _fit(self, X: Matrix, y: list[float]) -> None:
        trees: list[DecisionTreeRegressor] = []
        sums = [0.0] * len(X)
        votes = [0] * len(X)
        for estimator in range(self.n_estimators):
            rows = bootstrap_indices(len(X), self.seed, estimator, PURPOSE)
            tree = DecisionTreeRegressor(**self._tree_params)
            tree._feature_chooser = feature_chooser(
                self.max_features, self.seed, estimator, PURPOSE
            )
            tree.fit([X[i] for i in rows], [y[i] for i in rows])
            trees.append(tree)
            seen = set(rows)
            held_out = [i for i in range(len(X)) if i not in seen]
            if not held_out:
                continue
            for i, value in zip(held_out, tree.predict([X[i] for i in held_out]), strict=True):
                sums[i] += value
                votes[i] += 1
        self._trees = trees
        scored = [i for i in range(len(X)) if votes[i]]
        self.oob_score_ = None
        if len(scored) >= 2:
            truth = [y[i] for i in scored]
            mean = math.fsum(truth) / len(truth)
            total = math.fsum((t - mean) ** 2 for t in truth)
            if total > 0:
                residual = math.fsum((y[i] - sums[i] / votes[i]) ** 2 for i in scored)
                self.oob_score_ = 1.0 - residual / total

    def _predict(self, X: Matrix) -> list[float]:
        per_tree = [tree.predict(X) for tree in self._trees]
        n = len(per_tree)
        return [math.fsum(column) / n for column in zip(*per_tree, strict=True)]
