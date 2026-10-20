"""Gaussian naive Bayes classifier, computed in log space."""

from __future__ import annotations

import math
from typing import Any

from ..errors import ModelError
from ..metrics import ordered_labels
from .base import Classifier, Label, Matrix

_LOG_2PI = math.log(2.0 * math.pi)


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _variance(values: list[float], mean: float) -> float:
    """Population variance (divides by n, not n - 1)."""
    return math.fsum((v - mean) ** 2 for v in values) / len(values)


class GaussianNB(Classifier):
    """Models each feature as a normal distribution per class, assuming independent features.

    A small amount of variance, `var_smoothing` times the largest feature variance in the whole
    training set, is added to every per-class variance so constant features cannot cause a
    division by zero. All sums use `math.fsum`, so the fitted model does not depend on the order
    of the training rows.
    """

    name = "gaussian_nb"

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        super().__init__()
        valid = isinstance(var_smoothing, (int, float)) and not isinstance(var_smoothing, bool)
        if not valid or not math.isfinite(var_smoothing) or var_smoothing <= 0:
            raise ModelError(
                f"var_smoothing must be a positive finite number, got {var_smoothing!r}"
            )
        self.var_smoothing = float(var_smoothing)
        self._log_priors: list[float] = []
        self._means: list[list[float]] = []
        self._variances: list[list[float]] = []

    def get_params(self) -> dict[str, Any]:
        return {"var_smoothing": self.var_smoothing}

    def _fit(self, X: Matrix, y: list[Label]) -> None:
        classes = ordered_labels(y)
        n_features = len(X[0])
        columns = [[row[j] for row in X] for j in range(n_features)]
        overall = max(_variance(col, _mean(col)) for col in columns)
        epsilon = self.var_smoothing * (overall if overall > 0 else 1.0)

        log_priors: list[float] = []
        means: list[list[float]] = []
        variances: list[list[float]] = []
        for label in classes:
            members = [row for row, target in zip(X, y, strict=True) if target == label]
            log_priors.append(math.log(len(members) / len(X)))
            class_columns = [[row[j] for row in members] for j in range(n_features)]
            class_means = [_mean(col) for col in class_columns]
            means.append(class_means)
            variances.append(
                [
                    _variance(col, m) + epsilon
                    for col, m in zip(class_columns, class_means, strict=True)
                ]
            )
        self._log_priors, self._means, self._variances = log_priors, means, variances

    def _predict_proba(self, X: Matrix) -> Matrix:
        output: Matrix = []
        for row in X:
            joint = []
            for log_prior, means, variances in zip(
                self._log_priors, self._means, self._variances, strict=True
            ):
                terms = [
                    -0.5 * (_LOG_2PI + math.log(var)) - (x - mean) ** 2 / (2.0 * var)
                    for x, mean, var in zip(row, means, variances, strict=True)
                ]
                joint.append(log_prior + math.fsum(terms))
            top = max(joint)
            weights = [math.exp(value - top) for value in joint]
            total = math.fsum(weights)
            output.append([w / total for w in weights])
        return output
