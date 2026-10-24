"""A CART decision tree classifier using Gini impurity, built without recursion.

Splits are `x[feature] <= threshold` with the threshold halfway between two neighbouring distinct
training values. The best split minimises the weighted Gini impurity of the two sides; ties go to
the lowest feature index and then the lowest threshold, so the tree depends on the data and
never on the order of the rows. A node is split whenever a valid split exists and it is not pure,
even if the split does not lower the impurity at that node (some patterns, such as XOR, need
that). Leaves predict the class fractions of their training rows.
"""

from __future__ import annotations

from typing import Any

from ..errors import ModelError
from ..metrics import ordered_labels
from .base import Classifier, Label, Matrix

_TIE_TOLERANCE = 1e-12  # relative to the node size; scores closer than this count as equal


def _check_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ModelError(f"{name} must be a whole number of at least {minimum}, got {value!r}")
    return value


class _Node:
    __slots__ = ("counts", "feature", "left", "n", "right", "threshold")

    def __init__(self, counts: list[int]) -> None:
        self.counts = counts
        self.n = sum(counts)
        self.feature = -1  # -1 marks a leaf
        self.threshold = 0.0
        self.left: _Node | None = None
        self.right: _Node | None = None


class DecisionTreeClassifier(Classifier):
    """Parameters: `max_depth` (None for unlimited), `min_samples_split`, `min_samples_leaf`."""

    name = "decision_tree"

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
        self._classes_list: tuple[Label, ...] = ()
        self.depth_ = 0
        self.n_leaves_ = 0

    def get_params(self) -> dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
        }

    # -- fitting ------------------------------------------------------------------------------

    def _fit(self, X: Matrix, y: list[Label]) -> None:
        classes = ordered_labels(y)
        index = {label: i for i, label in enumerate(classes)}
        target = [index[label] for label in y]
        k = len(classes)

        def counts_of(rows: list[int]) -> list[int]:
            counts = [0] * k
            for i in rows:
                counts[target[i]] += 1
            return counts

        root = _Node(counts_of(list(range(len(X)))))
        stack: list[tuple[_Node, list[int], int]] = [(root, list(range(len(X))), 0)]
        depth = 0
        leaves = 0
        while stack:
            node, rows, level = stack.pop()
            depth = max(depth, level)
            split = None
            pure = max(node.counts) == node.n
            allowed = self.max_depth is None or level < self.max_depth
            if not pure and allowed and node.n >= self.min_samples_split:
                split = self._best_split(X, target, rows, node.counts)
            if split is None:
                leaves += 1
                continue
            node.feature, node.threshold = split
            left_rows = [i for i in rows if X[i][node.feature] <= node.threshold]
            right_rows = [i for i in rows if X[i][node.feature] > node.threshold]
            node.left = _Node(counts_of(left_rows))
            node.right = _Node(counts_of(right_rows))
            stack.append((node.right, right_rows, level + 1))
            stack.append((node.left, left_rows, level + 1))
        self._root = root
        self._classes_list = classes
        self.depth_ = depth
        self.n_leaves_ = leaves

    def _best_split(
        self, X: Matrix, target: list[int], rows: list[int], totals: list[int]
    ) -> tuple[int, float] | None:
        m = len(rows)
        best: tuple[int, float] | None = None
        best_score = 0.0
        for feature in range(len(X[0])):
            order = sorted(rows, key=lambda i: (X[i][feature], i))
            left = [0] * len(totals)
            right = list(totals)
            sq_left, sq_right = 0, sum(c * c for c in right)
            for position in range(m - 1):
                c = target[order[position]]
                sq_left += 2 * left[c] + 1  # (left[c] + 1)^2 - left[c]^2
                left[c] += 1
                sq_right -= 2 * right[c] - 1  # right[c]^2 - (right[c] - 1)^2
                right[c] -= 1
                here, there = X[order[position]][feature], X[order[position + 1]][feature]
                n_left = position + 1
                n_right = m - n_left
                if here == there or n_left < self.min_samples_leaf:
                    continue
                if n_right < self.min_samples_leaf:
                    break
                score = m - sq_left / n_left - sq_right / n_right  # m * weighted Gini impurity
                if best is None or score < best_score - _TIE_TOLERANCE * m:
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

    def _predict_proba(self, X: Matrix) -> Matrix:
        out: Matrix = []
        for row in X:
            leaf = self._leaf_for(row)
            out.append([c / leaf.n for c in leaf.counts])
        return out

    def describe(self) -> str:
        """The fitted tree as indented rules, one line per node."""
        _ = self.classes_
        lines: list[str] = []
        stack: list[tuple[_Node, int, str]] = [(self._root, 0, "")]  # type: ignore[list-item]
        while stack:
            node, level, prefix = stack.pop()
            pad = "  " * level
            if node.left is None or node.right is None:
                top = max(range(len(node.counts)), key=lambda i: (node.counts[i], -i))
                lines.append(f"{pad}{prefix}predict {self._classes_list[top]!r} (n={node.n})")
                continue
            lines.append(f"{pad}{prefix}if x{node.feature} <= {node.threshold:.6g}:")
            stack.append((node.right, level + 1, "else: "))
            stack.append((node.left, level + 1, ""))
        return "\n".join(lines)
