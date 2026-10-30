"""Bagged decision trees: many trees, each fitted to a bootstrap sample, with averaged votes."""

from __future__ import annotations

import math
from typing import Any

from ..errors import ModelError
from ..metrics import ordered_labels
from ..seeding import check_seed, rng_for
from .base import Classifier, Label, Matrix
from .decision_tree import DecisionTreeClassifier, _check_int

MAX_ESTIMATORS = 500


def bootstrap_indices(
    n: int, seed: int, estimator: int, purpose: str = "bagged_trees"
) -> list[int]:
    """The rows (with repeats) that tree number `estimator` is fitted to: `n` draws from `n`."""
    rng = rng_for(seed, purpose, estimator)
    return [rng.randrange(n) for _ in range(n)]


class BaggedTrees(Classifier):
    """Averages the class fractions of `n_estimators` decision trees.

    Each tree sees `n` rows drawn with replacement from the `n` training rows, using a generator
    derived from `seed` and the tree's position, so a model with the same parameters and data
    always gives the same trees. A bootstrap sample can miss a rare class; that tree then gives
    that class a fraction of zero. This is bagging only: every split still looks at every
    feature. Parameters other than `n_estimators` and `seed` configure the trees as in
    `DecisionTreeClassifier`.
    """

    name = "bagged_trees"
    _purpose = "bagged_trees"  # names the random stream, so another ensemble draws its own samples

    def __init__(
        self,
        n_estimators: int = 25,
        max_depth: int | None = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_estimators = _check_int("n_estimators", n_estimators, 1)
        if self.n_estimators > MAX_ESTIMATORS:
            raise ModelError(f"n_estimators must be at most {MAX_ESTIMATORS}, got {n_estimators}")
        # the trees check these, but a bad value should fail at construction, not at fit
        self._tree_params = DecisionTreeClassifier(
            max_depth, min_samples_split, min_samples_leaf
        ).get_params()
        self.seed = check_seed(seed)
        self._trees: list[DecisionTreeClassifier] = []
        self._class_position: dict[Label, int] = {}
        self.oob_score_: float | None = None

    def get_params(self) -> dict[str, Any]:
        return {"n_estimators": self.n_estimators, **self._tree_params, "seed": self.seed}

    def _new_tree(self, estimator: int, n_features: int) -> DecisionTreeClassifier:
        """The unfitted tree for position `estimator`; ensembles that restrict splits override."""
        return DecisionTreeClassifier(**self._tree_params)

    def _fit(self, X: Matrix, y: list[Label]) -> None:
        trees: list[DecisionTreeClassifier] = []
        samples: list[list[int]] = []
        for estimator in range(self.n_estimators):
            rows = bootstrap_indices(len(X), self.seed, estimator, self._purpose)
            tree = self._new_tree(estimator, len(X[0]))
            tree.fit([X[i] for i in rows], [y[i] for i in rows])
            trees.append(tree)
            samples.append(rows)
        self._trees = trees
        self._class_position = {label: i for i, label in enumerate(ordered_labels(y))}
        self.oob_score_ = self._out_of_bag_accuracy(X, y, samples)

    def _aligned(self, tree: DecisionTreeClassifier, X: Matrix) -> Matrix:
        """`tree`'s class fractions for `X`, in the ensemble's class order."""
        slots = [self._class_position[label] for label in tree.classes_]
        full = [[0.0] * len(self._class_position) for _ in X]
        for r, fractions in enumerate(tree.predict_proba(X)):
            for slot, fraction in zip(slots, fractions, strict=True):
                full[r][slot] = fraction
        return full

    def _out_of_bag_accuracy(
        self, X: Matrix, y: list[Label], samples: list[list[int]]
    ) -> float | None:
        """Accuracy on each training row using only the trees that never saw it.

        A row every tree saw is left out; None when no row is out of bag for any tree.
        """
        width = len(self._class_position)
        totals = [[0.0] * width for _ in X]
        votes = [0] * len(X)
        for tree, rows in zip(self._trees, samples, strict=True):
            seen = set(rows)
            held_out = [i for i in range(len(X)) if i not in seen]
            if not held_out:
                continue
            for i, row in zip(held_out, self._aligned(tree, [X[i] for i in held_out]), strict=True):
                votes[i] += 1
                for c in range(width):
                    totals[i][c] += row[c]
        classes = ordered_labels(y)
        scored = [i for i in range(len(X)) if votes[i]]
        if not scored:
            return None
        right = 0
        for i in scored:
            best = max(range(width), key=lambda c: (totals[i][c], -c))
            right += classes[best] == y[i]
        return right / len(scored)

    def _predict_proba(self, X: Matrix) -> Matrix:
        width = len(self._class_position)
        aligned = [self._aligned(tree, X) for tree in self._trees]
        n = len(self._trees)
        return [
            [math.fsum(m[r][c] for m in aligned) / n for c in range(width)] for r in range(len(X))
        ]
