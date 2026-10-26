"""Multiclass averaged perceptron.

Features are standardised internally (mean 0, population standard deviation 1; constant features
are left at 0). Each class has a weight vector and a bias. For every training row, in an order
reshuffled each epoch from a generator derived from `seed`, the model scores each class and, if
the best-scoring class (ties to the first class) is not the true one, adds the row to the true
class's weights and subtracts it from the predicted class's. Training stops after an epoch with no
mistakes (`converged_`, which only happens when the classes are linearly separable) or after
`max_iter` epochs. The weights used for prediction are the average over every step of training,
which makes the result much less sensitive to the order of the last few rows.

A perceptron has no probabilities. `predict_proba` returns the softmax of the class scores: it is
ordered like the scores and sums to one, but it is not calibrated, so `log_loss` is only a rough
guide. Because the scale of the weights never matters to the argmax, there is no learning rate.
"""

from __future__ import annotations

import math
from typing import Any

from ..errors import ModelError
from ..metrics import ordered_labels
from ..seeding import check_seed, rng_for
from .base import Classifier, Label, Matrix

MAX_ITER_LIMIT = 10_000


def epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    """The order in which training rows are visited in `epoch` (counting from 0)."""
    order = list(range(n))
    rng_for(seed, "perceptron", epoch).shuffle(order)
    return order


def _softmax(scores: list[float]) -> list[float]:
    top = max(scores)
    weights = [math.exp(s - top) for s in scores]
    total = math.fsum(weights)
    return [w / total for w in weights]


class Perceptron(Classifier):
    """Parameters: `max_iter` (epochs, 1 to 10000) and `seed` (the visiting order)."""

    name = "perceptron"

    def __init__(self, max_iter: int = 20, seed: int = 0) -> None:
        super().__init__()
        if isinstance(max_iter, bool) or not isinstance(max_iter, int):
            raise ModelError(f"max_iter must be a whole number, got {max_iter!r}")
        if not 1 <= max_iter <= MAX_ITER_LIMIT:
            raise ModelError(f"max_iter must be between 1 and {MAX_ITER_LIMIT}, got {max_iter}")
        self.max_iter = max_iter
        self.seed = check_seed(seed)
        self._mean: list[float] = []
        self._scale: list[float] = []
        self._weights: list[list[float]] = []  # averaged, standardised space, one row per class
        self._bias: list[float] = []
        self.n_iter_ = 0
        self.converged_ = False

    def get_params(self) -> dict[str, Any]:
        return {"max_iter": self.max_iter, "seed": self.seed}

    @property
    def coef_(self) -> list[list[float]]:
        """Averaged weights per class on the original feature scale, in `classes_` order."""
        _ = self.classes_
        return [[w / s for w, s in zip(row, self._scale, strict=True)] for row in self._weights]

    @property
    def intercept_(self) -> list[float]:
        """Averaged intercepts per class on the original feature scale."""
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

        W = [[0.0] * d for _ in range(k)]
        b = [0.0] * k
        total_w = [[0.0] * d for _ in range(k)]
        total_b = [0.0] * k
        steps = 0
        converged = False
        epochs = 0
        for epoch in range(self.max_iter):
            epochs = epoch + 1
            mistakes = 0
            for i in epoch_order(n, self.seed, epoch):
                z, t = Z[i], target[i]
                scores = [b[c] + sum(w * v for w, v in zip(W[c], z, strict=True)) for c in range(k)]
                guess = 0
                for c in range(1, k):
                    if scores[c] > scores[guess]:
                        guess = c
                if guess != t:
                    mistakes += 1
                    for j in range(d):
                        W[t][j] += z[j]
                        W[guess][j] -= z[j]
                    b[t] += 1.0
                    b[guess] -= 1.0
                for c in range(k):
                    row, acc = W[c], total_w[c]
                    for j in range(d):
                        acc[j] += row[j]
                    total_b[c] += b[c]
                steps += 1
            if mistakes == 0:
                converged = True
                break
        self._mean, self._scale = mean, scale
        self._weights = [[v / steps for v in row] for row in total_w]
        self._bias = [v / steps for v in total_b]
        self.n_iter_ = epochs
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
