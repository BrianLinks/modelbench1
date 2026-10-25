"""Pure-Python evaluation metrics.

Conventions:
- Classification inputs are sequences of hashable labels. Confusion-matrix rows are true labels
  and columns are predicted labels.
- A ratio whose denominator is zero (for example precision for a class that was never predicted)
  is reported as 0.0.
- Inputs are validated strictly: empty, mismatched, non-finite, or boolean-as-number inputs raise
  MetricError rather than producing a misleading number.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import MetricError

Label = Hashable
AVERAGES = ("macro", "micro", "weighted", "binary")
_DEFAULT_EPS = 1e-15
_PROB_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ConfusionMatrix:
    """`matrix[i][j]` counts samples whose true label is `labels[i]` and prediction `labels[j]`."""

    labels: tuple[Label, ...]
    matrix: tuple[tuple[int, ...], ...]

    @property
    def total(self) -> int:
        return sum(sum(row) for row in self.matrix)


@dataclass(frozen=True)
class PRF:
    """Precision, recall, F1, and the number of true samples behind them."""

    precision: float
    recall: float
    f1: float
    support: int


# -- validation helpers ----------------------------------------------------------------------


def _check_pair(y_true: Sequence[Any], y_pred: Sequence[Any]) -> None:
    if len(y_true) != len(y_pred):
        raise MetricError(f"length mismatch: {len(y_true)} true values, {len(y_pred)} predictions")
    if len(y_true) == 0:
        raise MetricError("inputs must not be empty")


def ordered_labels(*sequences: Sequence[Label]) -> tuple[Label, ...]:
    seen: dict[Label, None] = {}
    for seq in sequences:
        for value in seq:
            seen.setdefault(value, None)
    try:
        return tuple(sorted(seen))  # type: ignore[type-var]
    except TypeError:  # mixed, unorderable labels: keep first-seen order
        return tuple(seen)


def _resolve_labels(
    y_true: Sequence[Label], y_pred: Sequence[Label], labels: Sequence[Label] | None
) -> tuple[Label, ...]:
    if labels is None:
        return ordered_labels(y_true, y_pred)
    resolved = tuple(labels)
    if len(set(resolved)) != len(resolved):
        raise MetricError("labels must be unique")
    known = set(resolved)
    for value in (*y_true, *y_pred):
        if value not in known:
            raise MetricError(f"value {value!r} is not in the given labels")
    return resolved


def _numbers(values: Sequence[Any], name: str) -> list[float]:
    out: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MetricError(f"{name} must contain numbers, found {value!r}")
        number = float(value)
        if not math.isfinite(number):
            raise MetricError(f"{name} must be finite, found {value!r}")
        out.append(number)
    return out


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _ratio(2 * precision * recall, precision + recall)


# -- classification --------------------------------------------------------------------------


def confusion_matrix(
    y_true: Sequence[Label], y_pred: Sequence[Label], labels: Sequence[Label] | None = None
) -> ConfusionMatrix:
    """Count (true, predicted) pairs. Label order is sorted when sortable, else first-seen."""
    _check_pair(y_true, y_pred)
    resolved = _resolve_labels(y_true, y_pred, labels)
    index = {label: i for i, label in enumerate(resolved)}
    counts = [[0] * len(resolved) for _ in resolved]
    for t, p in zip(y_true, y_pred, strict=True):
        counts[index[t]][index[p]] += 1
    return ConfusionMatrix(resolved, tuple(tuple(row) for row in counts))


def accuracy(y_true: Sequence[Label], y_pred: Sequence[Label]) -> float:
    _check_pair(y_true, y_pred)
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    return correct / len(y_true)


def per_class_metrics(
    y_true: Sequence[Label], y_pred: Sequence[Label], labels: Sequence[Label] | None = None
) -> dict[Label, PRF]:
    """Precision, recall, F1, and support for every label."""
    cm = confusion_matrix(y_true, y_pred, labels)
    result: dict[Label, PRF] = {}
    for i, label in enumerate(cm.labels):
        tp = cm.matrix[i][i]
        support = sum(cm.matrix[i])
        predicted = sum(row[i] for row in cm.matrix)
        precision = _ratio(tp, predicted)
        recall = _ratio(tp, support)
        result[label] = PRF(precision, recall, _f1(precision, recall), support)
    return result


def precision_recall_f1(
    y_true: Sequence[Label],
    y_pred: Sequence[Label],
    average: str = "macro",
    labels: Sequence[Label] | None = None,
    positive_label: Label = 1,
) -> PRF:
    """Aggregate precision, recall, and F1.

    macro: unweighted mean over labels. weighted: mean weighted by support. micro: computed from
    pooled counts. binary: the metrics of `positive_label` alone.
    """
    if average not in AVERAGES:
        raise MetricError(f"average must be one of {AVERAGES}, got {average!r}")
    classes = per_class_metrics(y_true, y_pred, labels)
    total_support = sum(c.support for c in classes.values())

    if average == "binary":
        if positive_label not in classes:
            raise MetricError(f"positive_label {positive_label!r} not found among labels")
        return classes[positive_label]

    if average == "micro":
        cm = confusion_matrix(y_true, y_pred, labels)
        tp = sum(cm.matrix[i][i] for i in range(len(cm.labels)))
        # Every sample is one true label and one predicted label, so pooled fp == pooled fn.
        precision = recall = _ratio(tp, cm.total)
        return PRF(precision, recall, _f1(precision, recall), total_support)

    if average == "macro":
        weights = [1.0] * len(classes)
    else:
        weights = [float(c.support) for c in classes.values()]
    weight_sum = sum(weights)
    if weight_sum == 0:
        return PRF(0.0, 0.0, 0.0, total_support)
    pairs = list(zip(weights, classes.values(), strict=True))
    precision = math.fsum(w * c.precision for w, c in pairs) / weight_sum
    recall = math.fsum(w * c.recall for w, c in pairs) / weight_sum
    f1 = math.fsum(w * c.f1 for w, c in pairs) / weight_sum
    return PRF(precision, recall, f1, total_support)


def balanced_accuracy(y_true: Sequence[Label], y_pred: Sequence[Label]) -> float:
    """Mean recall over the labels that occur in `y_true`, so a rare class counts as much as a
    common one. A model that always predicts the majority class scores 1 / (number of classes)."""
    cm = confusion_matrix(y_true, y_pred)
    recalls = [
        cm.matrix[i][i] / support
        for i in range(len(cm.labels))
        if (support := sum(cm.matrix[i])) > 0
    ]
    return math.fsum(recalls) / len(recalls)


def matthews_corrcoef(y_true: Sequence[Label], y_pred: Sequence[Label]) -> float:
    """Matthews correlation coefficient in [-1, 1] (Gorodkin's multiclass form).

    1 is perfect, 0 is no better than chance, -1 is total disagreement. When one side predicts
    or contains a single class the denominator is zero and the score is 0.0.
    """
    cm = confusion_matrix(y_true, y_pred)
    n = cm.total
    correct = sum(cm.matrix[i][i] for i in range(len(cm.labels)))
    true_counts = [sum(row) for row in cm.matrix]
    pred_counts = [sum(row[j] for row in cm.matrix) for j in range(len(cm.labels))]
    cross = sum(t * p for t, p in zip(true_counts, pred_counts, strict=True))
    denominator = math.sqrt(
        (n * n - sum(p * p for p in pred_counts)) * (n * n - sum(t * t for t in true_counts))
    )
    return _ratio(correct * n - cross, denominator)


def average_precision(
    y_true: Sequence[Label], y_score: Sequence[float], positive_label: Label = 1
) -> float:
    """Area under the precision-recall curve as a step sum: sum of (recall gain) x precision.

    Samples with equal scores are taken together, so the result does not depend on their order.
    """
    _check_pair(y_true, y_score)
    if len(set(y_true)) > 2:
        raise MetricError("average_precision supports at most two distinct labels")
    scores = _numbers(y_score, "y_score")
    is_positive = [t == positive_label for t in y_true]
    n_pos = sum(is_positive)
    if n_pos == 0:
        raise MetricError("average_precision needs at least one positive sample")
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = seen = 0
    previous_recall = 0.0
    total = 0.0
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[position]]:
            end += 1
        for k in range(position, end + 1):
            tp += is_positive[order[k]]
        seen = end + 1
        recall = tp / n_pos
        total += (recall - previous_recall) * (tp / seen)
        previous_recall = recall
        position = end + 1
    return total


def roc_auc(y_true: Sequence[Label], y_score: Sequence[float], positive_label: Label = 1) -> float:
    """Area under the ROC curve for binary labels, via average ranks (ties count half)."""
    _check_pair(y_true, y_score)
    if len(set(y_true)) > 2:
        raise MetricError("roc_auc supports at most two distinct labels")
    scores = _numbers(y_score, "y_score")
    is_positive = [t == positive_label for t in y_true]
    n_pos = sum(is_positive)
    n_neg = len(is_positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise MetricError("roc_auc needs at least one positive and one negative sample")

    order = sorted(range(len(scores)), key=scores.__getitem__)
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[start]]:
            end += 1
        average_rank = (start + end) / 2 + 1  # ranks are 1-based
        for k in range(start, end + 1):
            ranks[order[k]] = average_rank
        start = end + 1

    positive_rank_sum = math.fsum(r for r, pos in zip(ranks, is_positive, strict=True) if pos)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def log_loss(
    y_true: Sequence[Label],
    y_prob: Sequence[Any],
    labels: Sequence[Label] | None = None,
    eps: float = _DEFAULT_EPS,
) -> float:
    """Mean negative log-likelihood of the true label.

    Each `y_prob` entry is a row of probabilities aligned with `labels` (default: the sorted
    distinct values of y_true), or, for two labels, a single number meaning P(labels[1]).
    """
    _check_pair(y_true, y_prob)
    if not 0 < eps < 0.5:
        raise MetricError("eps must be in (0, 0.5)")
    resolved = tuple(labels) if labels is not None else ordered_labels(y_true)
    if len(set(resolved)) != len(resolved):
        raise MetricError("labels must be unique")
    index = {label: i for i, label in enumerate(resolved)}

    losses: list[float] = []
    for truth, entry in zip(y_true, y_prob, strict=True):
        if truth not in index:
            raise MetricError(f"true value {truth!r} is not in the labels")
        if isinstance(entry, (int, float)) and not isinstance(entry, bool):
            if len(resolved) != 2:
                raise MetricError(
                    "single-number probabilities require exactly two labels; "
                    "pass labels=[negative, positive]"
                )
            p = _numbers([entry], "y_prob")[0]
            row = [1.0 - p, p]
        else:
            row = _numbers(list(entry), "y_prob")
        if len(row) != len(resolved):
            raise MetricError(f"probability row has {len(row)} entries for {len(resolved)} labels")
        if any(v < 0 or v > 1 for v in row) or abs(math.fsum(row) - 1.0) > _PROB_TOLERANCE:
            raise MetricError("each probability row must lie in [0, 1] and sum to 1")
        chosen = min(max(row[index[truth]], eps), 1.0 - eps)
        losses.append(-math.log(chosen))
    return math.fsum(losses) / len(losses)


# -- regression ------------------------------------------------------------------------------


def _regression_inputs(
    y_true: Sequence[float], y_pred: Sequence[float]
) -> tuple[list[float], list[float]]:
    _check_pair(y_true, y_pred)
    return _numbers(y_true, "y_true"), _numbers(y_pred, "y_pred")


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    t, p = _regression_inputs(y_true, y_pred)
    return math.fsum(abs(a - b) for a, b in zip(t, p, strict=True)) / len(t)


def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    t, p = _regression_inputs(y_true, y_pred)
    return math.fsum((a - b) ** 2 for a, b in zip(t, p, strict=True)) / len(t)


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return math.sqrt(mse(y_true, y_pred))


def r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Coefficient of determination.

    With constant y_true the score is 1.0 if predictions are exact and 0.0 otherwise.
    """
    t, p = _regression_inputs(y_true, y_pred)
    mean = math.fsum(t) / len(t)
    ss_res = math.fsum((a - b) ** 2 for a, b in zip(t, p, strict=True))
    ss_tot = math.fsum((a - mean) ** 2 for a in t)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot
