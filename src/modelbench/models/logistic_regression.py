"""Multinomial logistic regression trained by accelerated gradient descent.

Features are standardised internally (mean 0, population standard deviation 1; constant features
are left at 0), so the result does not depend on the units of the inputs. The objective is the
mean cross-entropy plus `(l2 / 2)` times the squared weights (the intercepts are not penalised).
The step size is derived from a proven bound on the curvature of that objective, so training
cannot diverge and there is no learning rate to tune, and the momentum is the constant that is
optimal for a strongly convex objective, which `l2 > 0` guarantees. Training stops when the largest
gradient entry falls below `tol` or after `max_iter` iterations; `converged_` and `n_iter_` report
which. Every classification, binary included, uses the softmax form with one weight row per class.

This is a baseline for small tables: the cost is `max_iter * rows * classes * features`.
"""

from __future__ import annotations

import math
from typing import Any

from ..errors import ModelError
from ..metrics import ordered_labels
from .base import Classifier, Label, Matrix

MAX_ITER_LIMIT = 100_000


def _positive(name: str, value: object, minimum: float) -> float:
    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not ok or not math.isfinite(value) or value < minimum:  # type: ignore[arg-type]
        raise ModelError(f"{name} must be a finite number of at least {minimum:g}, got {value!r}")
    return float(value)  # type: ignore[arg-type]


def _softmax(scores: list[float]) -> list[float]:
    top = max(scores)
    weights = [math.exp(s - top) for s in scores]
    total = math.fsum(weights)
    return [w / total for w in weights]


class LogisticRegression(Classifier):
    """Parameters: `l2` (penalty strength, at least 1e-8), `max_iter`, `tol`."""

    name = "logistic_regression"

    def __init__(self, l2: float = 0.01, max_iter: int = 500, tol: float = 1e-6) -> None:
        super().__init__()
        self.l2 = _positive("l2", l2, 1e-8)
        if isinstance(max_iter, bool) or not isinstance(max_iter, int):
            raise ModelError(f"max_iter must be a whole number, got {max_iter!r}")
        if not 1 <= max_iter <= MAX_ITER_LIMIT:
            raise ModelError(f"max_iter must be between 1 and {MAX_ITER_LIMIT}, got {max_iter}")
        self.max_iter = max_iter
        self.tol = _positive("tol", tol, 1e-300)
        self._mean: list[float] = []
        self._scale: list[float] = []
        self._weights: list[list[float]] = []  # standardised space, one row per class
        self._bias: list[float] = []
        self.n_iter_ = 0
        self.converged_ = False

    def get_params(self) -> dict[str, Any]:
        return {"l2": self.l2, "max_iter": self.max_iter, "tol": self.tol}

    @property
    def coef_(self) -> list[list[float]]:
        """Weights per class on the original feature scale, rows in `classes_` order."""
        _ = self.classes_
        return [[w / s for w, s in zip(row, self._scale, strict=True)] for row in self._weights]

    @property
    def intercept_(self) -> list[float]:
        """Intercepts per class on the original feature scale."""
        _ = self.classes_
        return [
            b - math.fsum(w * m / s for w, m, s in zip(row, self._mean, self._scale, strict=True))
            for b, row in zip(self._bias, self._weights, strict=True)
        ]

    def _fit(self, X: Matrix, y: list[Label]) -> None:
        classes = ordered_labels(y)
        index = {label: i for i, label in enumerate(classes)}
        target = [index[label] for label in y]
        n, d, k = len(X), len(X[0]), len(classes)

        mean = [math.fsum(row[j] for row in X) / n for j in range(d)]
        scale = []
        for j in range(d):
            variance = math.fsum((row[j] - mean[j]) ** 2 for row in X) / n
            scale.append(math.sqrt(variance) if variance > 0 else 1.0)
        Z = [[(row[j] - mean[j]) / scale[j] for j in range(d)] for row in X]

        lipschitz = 0.5 * (d + 1) + self.l2  # bound on the curvature of the objective
        step = 1.0 / lipschitz
        root_l, root_m = math.sqrt(lipschitz), math.sqrt(self.l2)
        momentum = (root_l - root_m) / (root_l + root_m)

        W = [[0.0] * d for _ in range(k)]
        b = [0.0] * k
        lookahead_w = [row[:] for row in W]
        lookahead_b = b[:]
        converged = False
        iterations = 0
        for iterations in range(1, self.max_iter + 1):
            grad_w = [[self.l2 * w for w in row] for row in lookahead_w]
            grad_b = [0.0] * k
            for z, t in zip(Z, target, strict=True):
                scores = [
                    lookahead_b[c] + sum(w * v for w, v in zip(lookahead_w[c], z, strict=True))
                    for c in range(k)
                ]
                probs = _softmax(scores)
                for c in range(k):
                    err = (probs[c] - (1.0 if c == t else 0.0)) / n
                    grad_b[c] += err
                    row = grad_w[c]
                    for j in range(d):
                        row[j] += err * z[j]
            biggest = max(max(abs(g) for row in grad_w for g in row), max(abs(g) for g in grad_b))
            if biggest < self.tol:
                converged = True
                W, b = lookahead_w, lookahead_b
                break
            new_w = [
                [v - step * g for v, g in zip(lw, gw, strict=True)]
                for lw, gw in zip(lookahead_w, grad_w, strict=True)
            ]
            new_b = [v - step * g for v, g in zip(lookahead_b, grad_b, strict=True)]
            lookahead_w = [
                [nv + momentum * (nv - ov) for nv, ov in zip(nrow, orow, strict=True)]
                for nrow, orow in zip(new_w, W, strict=True)
            ]
            lookahead_b = [nv + momentum * (nv - ov) for nv, ov in zip(new_b, b, strict=True)]
            W, b = new_w, new_b
        self._mean, self._scale, self._weights, self._bias = mean, scale, W, b
        self.n_iter_ = iterations
        self.converged_ = converged

    def _predict_proba(self, X: Matrix) -> Matrix:
        out: Matrix = []
        for row in X:
            z = [(v - m) / s for v, m, s in zip(row, self._mean, self._scale, strict=True)]
            out.append(
                _softmax(
                    [
                        bias + sum(w * v for w, v in zip(weights, z, strict=True))
                        for weights, bias in zip(self._weights, self._bias, strict=True)
                    ]
                )
            )
        return out
