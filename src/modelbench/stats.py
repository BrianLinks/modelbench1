"""Significance tests with no dependencies: a paired t-test and an exact McNemar test.

Both answer "is the difference between two models bigger than chance would explain?", which a
bare comparison of two scores cannot. They are about the data you gave them, not about the
future: with only a handful of folds, or a small test set, a p-value is a rough guide.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .errors import StatsError

_TINY = 1e-300
_EPSILON = 1e-15
_MAX_ITERATIONS = 500


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Lentz's evaluation of the continued fraction behind the incomplete beta function."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > _TINY else _TINY)
    h = d
    for m in range(1, _MAX_ITERATIONS + 1):
        m2 = 2 * m
        step = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + step * d
        d = 1.0 / (d if abs(d) > _TINY else _TINY)
        c = 1.0 + step / c
        c = c if abs(c) > _TINY else _TINY
        h *= d * c
        step = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + step * d
        d = 1.0 / (d if abs(d) > _TINY else _TINY)
        c = 1.0 + step / c
        c = c if abs(c) > _TINY else _TINY
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPSILON:
            return h
    raise StatsError("the incomplete beta function did not converge")  # pragma: no cover


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b) for a, b > 0 and 0 <= x <= 1."""
    if not (a > 0 and b > 0) or not 0.0 <= x <= 1.0:
        raise StatsError(f"incomplete beta needs a, b > 0 and 0 <= x <= 1, got {a}, {b}, {x}")
    if x in (0.0, 1.0):
        return x
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_two_sided_p(t: float, df: float) -> float:
    """P(|T| >= |t|) for Student's t with `df` degrees of freedom."""
    if not math.isfinite(t) and not math.isinf(t):
        raise StatsError("t must not be NaN")
    if df <= 0 or not math.isfinite(df):
        raise StatsError(f"degrees of freedom must be positive and finite, got {df}")
    if math.isinf(t):
        return 0.0
    return regularized_incomplete_beta(df / 2.0, 0.5, df / (df + t * t))


@dataclass(frozen=True)
class TTestResult:
    n: int
    mean_difference: float  # mean of a - b
    std_error: float
    t: float
    df: int
    p_value: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean_difference": self.mean_difference,
            "std_error": self.std_error,
            "t": self.t,
            "df": self.df,
            "p_value": self.p_value,
        }


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> TTestResult:
    """Two-sided paired t-test of the differences `a[i] - b[i]` (sample standard deviation).

    A common use is the per-fold scores of two models on the same cross-validation folds. If
    every difference is identical and non-zero the spread is zero and the p-value is 0; if they
    are all zero, `t` is 0 and the p-value is 1. Fold scores overlap in their training data, so
    treat the p-value as optimistic.
    """
    if len(a) != len(b):
        raise StatsError(f"the two samples must pair up, got {len(a)} and {len(b)} values")
    n = len(a)
    if n < 2:
        raise StatsError("a paired t-test needs at least two pairs")
    for value in (*a, *b):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise StatsError(f"values must be finite numbers, got {value!r}")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    mean = math.fsum(diffs) / n
    variance = math.fsum((d - mean) ** 2 for d in diffs) / (n - 1)
    std_error = math.sqrt(variance / n)
    if std_error == 0.0:
        if mean == 0.0:
            return TTestResult(n, 0.0, 0.0, 0.0, n - 1, 1.0)
        return TTestResult(n, mean, 0.0, math.copysign(math.inf, mean), n - 1, 0.0)
    t = mean / std_error
    return TTestResult(n, mean, std_error, t, n - 1, student_t_two_sided_p(t, n - 1))


@dataclass(frozen=True)
class McNemarResult:
    only_a_correct: int
    only_b_correct: int
    p_value: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "only_a_correct": self.only_a_correct,
            "only_b_correct": self.only_b_correct,
            "p_value": self.p_value,
        }


def mcnemar_exact(
    y_true: Sequence[Any], pred_a: Sequence[Any], pred_b: Sequence[Any]
) -> McNemarResult:
    """Exact two-sided McNemar test on the rows where exactly one model is right.

    Under "the models are equally accurate", each such row is a fair coin flip between them, so
    the p-value is the two-sided binomial tail with p = 1/2 over the `b + c` discordant rows
    (1 when there are none).
    """
    if not len(y_true) == len(pred_a) == len(pred_b):
        raise StatsError("y_true and both prediction lists must be the same length")
    if len(y_true) == 0:
        raise StatsError("McNemar's test needs at least one row")
    only_a = sum(1 for t, x, y in zip(y_true, pred_a, pred_b, strict=True) if x == t and y != t)
    only_b = sum(1 for t, x, y in zip(y_true, pred_a, pred_b, strict=True) if y == t and x != t)
    n = only_a + only_b
    if n == 0:
        return McNemarResult(0, 0, 1.0)
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1))
    # integer / integer true division, so huge counts never pass through a float
    return McNemarResult(only_a, only_b, min(1.0, (2 * tail) / 2**n))
