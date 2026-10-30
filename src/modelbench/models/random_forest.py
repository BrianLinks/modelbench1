"""Random forests: bagged trees that look at a random subset of features at every split."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from ..errors import ModelError
from ..seeding import rng_for
from .bagged_trees import BaggedTrees
from .decision_tree import DecisionTreeClassifier

MaxFeatures = int | float | str | None


def check_max_features(value: object) -> MaxFeatures:
    """`None` (every feature), "sqrt", "log2", a whole number, or a fraction in (0, 1]."""
    if value is None:
        return None
    if value == "sqrt":
        return "sqrt"
    if value == "log2":
        return "log2"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(
            'max_features must be None, "sqrt", "log2", a whole number or a fraction, '
            f"got {value!r}"
        )
    if isinstance(value, int):
        if value < 1:
            raise ModelError(f"max_features must be at least 1 when a whole number, got {value}")
        return value
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ModelError(f"max_features must be in (0, 1] when a fraction, got {value!r}")
    return value


def features_per_split(value: MaxFeatures, n_features: int) -> int:
    """How many features one split may look at: always between 1 and `n_features`."""
    if value is None:
        count = n_features
    elif value == "sqrt":
        count = int(math.sqrt(n_features))
    elif value == "log2":
        count = int(math.log2(n_features))
    elif isinstance(value, int):
        count = value
    else:
        count = round(float(value) * n_features)
    return max(1, min(n_features, count))


def feature_chooser(
    value: MaxFeatures, seed: int, estimator: int, purpose: str
) -> Callable[[int], Sequence[int]]:
    """A function that draws the features for each split of one tree, in the order splits are
    tried. It depends only on the seed, the tree's position and how many splits came before."""
    rng = rng_for(seed, purpose, estimator, "features")

    def choose(n_features: int) -> list[int]:
        return sorted(rng.sample(range(n_features), features_per_split(value, n_features)))

    return choose


class RandomForest(BaggedTrees):
    """Bagged decision trees whose splits each see `max_features` randomly chosen features.

    Everything else is as in `BaggedTrees`: `n_estimators` trees, each fitted to a bootstrap
    sample, class fractions averaged, repeatable per `seed`. `max_features` is "sqrt" (the
    default), "log2", `None` (all features, which is plain bagging with a different random
    stream), a whole number, or a fraction of the features. If the chosen features offer no valid
    split, the remaining features are tried so a node is never left unsplit for that reason.
    `oob_score_` is the accuracy on training rows using only trees that did not see them.
    """

    name = "random_forest"
    _purpose = "random_forest"

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = "sqrt",
        seed: int = 0,
    ) -> None:
        super().__init__(n_estimators, max_depth, min_samples_split, min_samples_leaf, seed)
        self.max_features = check_max_features(max_features)

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        seed = params.pop("seed")
        return {**params, "max_features": self.max_features, "seed": seed}

    def _new_tree(self, estimator: int, n_features: int) -> DecisionTreeClassifier:
        tree = super()._new_tree(estimator, n_features)
        tree._feature_chooser = feature_chooser(
            self.max_features, self.seed, estimator, self._purpose
        )
        return tree
