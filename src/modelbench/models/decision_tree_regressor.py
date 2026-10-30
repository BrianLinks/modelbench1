"""A CART regression tree that minimises squared error, built without recursion.

The rules match `DecisionTreeClassifier`: splits are `x[feature] <= threshold` with the threshold
halfway between two neighbouring distinct training values; ties go to the lowest feature index
and then the lowest threshold, so the tree depends on the data and never on row order; a node is
split whenever a valid split exists and its targets are not all equal. A split is chosen to
minimise the summed squared error of its two sides, which is the same as maximising
`sum_left**2 / n_left + sum_right**2 / n_right` of the targets measured from the node mean.
Leaves predict the mean target of their training rows.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from ..errors import ModelError
from .base import Matrix, Regressor, check_features
from .decision_tree import _check_int

_TIE_TOLERANCE = 1e-12  # relative to the node's total squared deviation


class _Node:
    __slots__ = ("feature", "leaf_id", "left", "mean", "n", "right", "threshold")

    def __init__(self, mean: float, n: int) -> None:
        self.mean = mean
        self.n = n
        self.feature = -1  # -1 marks a leaf
        self.leaf_id = -1  # position among the leaves once the tree is built, else -1
        self.threshold = 0.0
        self.left: _Node | None = None
        self.right: _Node | None = None


class DecisionTreeRegressor(Regressor):
    """Parameters: `max_depth` (None for unlimited), `min_samples_split`, `min_samples_leaf`."""

    name = "decision_tree_regressor"

    def __init__(
        self,
        max_depth: int | None = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ) -> None:
        super().__init__()
        self.max_depth = None if max_depth is None else _check_int("max_depth", max_depth, 1)
        self.min_samples_split = _check_int("min_samples_split", min_samples_split, 2)
        self.min_samples_leaf = _check_int("min_samples_leaf", min_samples_leaf, 1)
        self._root: _Node | None = None
        self.depth_ = 0
        self.n_leaves_ = 0
        self._leaf_nodes: list[_Node] = []
        # Set by ensembles that look at a subset of features at each split; not a parameter.
        self._feature_chooser: Callable[[int], Sequence[int]] | None = None

    def get_params(self) -> dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
        }

    # -- fitting ------------------------------------------------------------------------------

    def _fit(self, X: Matrix, y: list[float]) -> None:
        def make(rows: list[int]) -> _Node:
            return _Node(math.fsum(y[i] for i in rows) / len(rows), len(rows))

        everything = list(range(len(X)))
        root = make(everything)
        stack: list[tuple[_Node, list[int], int]] = [(root, everything, 0)]
        depth = 0
        leaves = 0
        leaf_nodes: list[_Node] = []
        while stack:
            node, rows, level = stack.pop()
            depth = max(depth, level)
            split = None
            constant = all(y[i] == y[rows[0]] for i in rows)
            allowed = self.max_depth is None or level < self.max_depth
            if not constant and allowed and node.n >= self.min_samples_split:
                split = self._split_for(X, y, rows, node.mean)
            if split is None:
                node.leaf_id = leaves
                leaf_nodes.append(node)
                leaves += 1
                continue
            node.feature, node.threshold = split
            left_rows = [i for i in rows if X[i][node.feature] <= node.threshold]
            right_rows = [i for i in rows if X[i][node.feature] > node.threshold]
            node.left = make(left_rows)
            node.right = make(right_rows)
            stack.append((node.right, right_rows, level + 1))
            stack.append((node.left, left_rows, level + 1))
        self._root = root
        self.depth_ = depth
        self.n_leaves_ = leaves
        self._leaf_nodes = leaf_nodes

    def _split_for(
        self, X: Matrix, y: list[float], rows: list[int], mean: float
    ) -> tuple[int, float] | None:
        width = len(X[0])
        if self._feature_chooser is None:
            return self._best_split(X, y, rows, mean, range(width))
        chosen = sorted(set(self._feature_chooser(width)))
        found = self._best_split(X, y, rows, mean, chosen)
        if found is not None:
            return found
        rest = [f for f in range(width) if f not in set(chosen)]
        return self._best_split(X, y, rows, mean, rest) if rest else None

    def _best_split(
        self,
        X: Matrix,
        y: list[float],
        rows: list[int],
        mean: float,
        features: Sequence[int],
    ) -> tuple[int, float] | None:
        m = len(rows)
        spread = math.fsum((y[i] - mean) ** 2 for i in rows)
        total = math.fsum(y[i] - mean for i in rows)
        best: tuple[int, float] | None = None
        best_score = 0.0
        for feature in features:
            order = sorted(rows, key=lambda i: (X[i][feature], i))
            sum_left = 0.0
            for position in range(m - 1):
                sum_left += y[order[position]] - mean
                here, there = X[order[position]][feature], X[order[position + 1]][feature]
                n_left = position + 1
                n_right = m - n_left
                if here == there or n_left < self.min_samples_leaf:
                    continue
                if n_right < self.min_samples_leaf:
                    break
                sum_right = total - sum_left
                score = sum_left * sum_left / n_left + sum_right * sum_right / n_right
                if best is None or score > best_score + _TIE_TOLERANCE * spread:
                    middle = here + (there - here) / 2.0
                    best = (feature, middle if middle < there else here)
                    best_score = score
        return best

    # -- prediction ---------------------------------------------------------------------------

    def _leaf_for(self, row: list[float]) -> _Node:
        node = self._root
        assert node is not None
        while node.left is not None and node.right is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node

    def _predict(self, X: Matrix) -> list[float]:
        return [self._leaf_for(row).mean for row in X]

    def apply(self, X: Sequence[Sequence[float]]) -> list[int]:
        """The leaf each row lands in, as a number from 0 to `n_leaves_ - 1`.

        Leaves are numbered in the order the builder finished them, which depends only on the
        data. Used by boosting to set its own leaf values.
        """
        rows = self._checked_rows(X)
        return [self._leaf_for(row).leaf_id for row in rows]

    def set_leaf_values(self, values: Sequence[float]) -> None:
        """Replace what each leaf predicts, one finite number per leaf in `apply` order."""
        if self._root is None:
            raise ModelError(f"{self.name} has not been fitted yet")
        if len(values) != self.n_leaves_:
            raise ModelError(f"expected {self.n_leaves_} leaf values, got {len(values)}")
        numbers = [float(v) for v in values]
        if not all(math.isfinite(v) for v in numbers):
            raise ModelError("leaf values must be finite")
        for node, value in zip(self._leaf_nodes, numbers, strict=True):
            node.mean = value

    def _checked_rows(self, X: Sequence[Sequence[float]]) -> Matrix:
        if self._n_features is None:
            raise ModelError(f"{self.name} has not been fitted yet")
        rows = check_features(X)
        if len(rows[0]) != self._n_features:
            raise ModelError(
                f"model was fitted with {self._n_features} features, got {len(rows[0])}"
            )
        return rows

    def describe(self) -> str:
        """The fitted tree as indented rules, one line per node."""
        if self._root is None:
            return ""
        lines: list[str] = []
        stack: list[tuple[_Node, int, str]] = [(self._root, 0, "")]
        while stack:
            node, level, prefix = stack.pop()
            pad = "  " * level
            if node.left is None or node.right is None:
                lines.append(f"{pad}{prefix}predict {node.mean:.6g} (n={node.n})")
                continue
            lines.append(f"{pad}{prefix}if x{node.feature} <= {node.threshold:.6g}:")
            stack.append((node.right, level + 1, "else: "))
            stack.append((node.left, level + 1, ""))
        return "\n".join(lines)
