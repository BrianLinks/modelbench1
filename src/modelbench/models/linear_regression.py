"""Ridge / ordinary least squares regression solved in closed form.

Features are standardised and the target is centred, so the intercept is the training mean and
the L2 penalty treats every feature on the same scale. The normal equations are solved by
Cholesky decomposition; constant columns get weight zero, and a rank-deficient design with no
penalty is reported as an error rather than solved with arbitrary numbers.
"""

from __future__ import annotations

import math
from typing import Any

from ..errors import ModelError
from .base import Matrix, Regressor

PIVOT_TOLERANCE = 1e-10  # standardised diagonal entries are 1, so this is a relative test


def cholesky_solve(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve A x = b for symmetric positive definite A. Raises ModelError if A is singular."""
    n = len(b)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = math.fsum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                pivot = A[i][i] - total
                if pivot <= PIVOT_TOLERANCE:
                    raise ModelError(
                        "features are linearly dependent; set l2 > 0 or remove a feature"
                    )
                L[i][j] = math.sqrt(pivot)
            else:
                L[i][j] = (A[i][j] - total) / L[j][j]
    z = [0.0] * n
    for i in range(n):
        z[i] = (b[i] - math.fsum(L[i][k] * z[k] for k in range(i))) / L[i][i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (z[i] - math.fsum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
    return x


class LinearRegression(Regressor):
    """Minimises mean squared error plus `l2` times the squared standardised weights."""

    name = "linear_regression"

    def __init__(self, l2: float = 0.0) -> None:
        super().__init__()
        if isinstance(l2, bool) or not isinstance(l2, (int, float)):
            raise ModelError(f"l2 must be a number, got {l2!r}")
        if not math.isfinite(l2) or l2 < 0:
            raise ModelError(f"l2 must be finite and >= 0, got {l2!r}")
        self.l2 = float(l2)
        self._means: list[float] = []
        self._scales: list[float] = []
        self._weights: list[float] = []
        self._intercept = 0.0

    def get_params(self) -> dict[str, Any]:
        return {"l2": self.l2}

    @property
    def coef_(self) -> list[float]:
        """Weights in the original feature units."""
        if not self.is_fitted:
            raise ModelError(f"{self.name} has not been fitted yet")
        return [w / s for w, s in zip(self._weights, self._scales, strict=True)]

    @property
    def intercept_(self) -> float:
        if not self.is_fitted:
            raise ModelError(f"{self.name} has not been fitted yet")
        shift = math.fsum(c * mu for c, mu in zip(self.coef_, self._means, strict=True))
        return self._intercept - shift

    def _fit(self, X: Matrix, y: list[float]) -> None:
        n, d = len(X), len(X[0])
        means = [math.fsum(row[j] for row in X) / n for j in range(d)]
        sds = [math.sqrt(math.fsum((row[j] - means[j]) ** 2 for row in X) / n) for j in range(d)]
        # a column that is constant up to rounding carries no information
        active = [j for j in range(d) if sds[j] > 1e-12 * max(1.0, abs(means[j]))]
        target_mean = math.fsum(y) / n
        centred = [value - target_mean for value in y]
        Z = [[(row[j] - means[j]) / sds[j] for j in active] for row in X]
        k = len(active)
        gram = [
            [math.fsum(r[a] * r[c] for r in Z) / n + (self.l2 if a == c else 0.0) for c in range(k)]
            for a in range(k)
        ]
        rhs = [math.fsum(r[a] * t for r, t in zip(Z, centred, strict=True)) / n for a in range(k)]
        solved = cholesky_solve(gram, rhs) if k else []
        weights = [0.0] * d
        for position, j in enumerate(active):
            weights[j] = solved[position]
        self._means = means
        self._scales = [sds[j] if j in active else 1.0 for j in range(d)]
        self._weights = weights
        self._intercept = target_mean

    def _predict(self, X: Matrix) -> list[float]:
        out: list[float] = []
        for row in X:
            total = math.fsum(
                w * (v - mu) / s
                for w, v, mu, s in zip(self._weights, row, self._means, self._scales, strict=True)
            )
            out.append(self._intercept + total)
        return out
