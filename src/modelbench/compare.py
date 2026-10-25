"""Comparison primitives: parameter diffs, metric deltas, rankings, and a paired bootstrap.

These are pure functions over plain data. Comparing stored runs is built on top of them.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import CompareError
from .seeding import check_seed, rng_for

# Which direction is better for each known metric. Names may carry a prefix such as "test/".
HIGHER_IS_BETTER: dict[str, bool] = {
    "accuracy": True,
    "precision": True,
    "recall": True,
    "f1": True,
    "roc_auc": True,
    "precision_micro": True,
    "precision_weighted": True,
    "recall_micro": True,
    "recall_weighted": True,
    "f1_micro": True,
    "f1_weighted": True,
    "balanced_accuracy": True,
    "mcc": True,
    "average_precision": True,
    "r2": True,
    "log_loss": False,
    "mae": False,
    "mse": False,
    "rmse": False,
}
MIN_RESAMPLES = 100
MAX_RESAMPLES = 100_000


def higher_is_better(metric: str) -> bool:
    """Direction for a metric, from its base name (the part after the last '/')."""
    base = metric.rsplit("/", 1)[-1]
    try:
        return HIGHER_IS_BETTER[base]
    except KeyError:
        raise CompareError(
            f"unknown direction for metric {metric!r}; pass higher_is_better explicitly"
        ) from None


# -- parameters ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamDiff:
    key: str
    kind: str  # "added" (only in new), "removed" (only in old), or "changed"
    old: Any
    new: Any


def flatten_params(params: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested objects into dotted keys. Lists and scalars are kept as values."""
    flat: dict[str, Any] = {}
    for key, value in params.items():
        if not isinstance(key, str):
            raise CompareError(f"parameter names must be strings, got {key!r}")
        path = f"{prefix}{key}"
        pieces = flatten_params(value, f"{path}.") if isinstance(value, Mapping) and value else {}
        for sub_key, sub_value in (pieces or {path: value}).items():
            if sub_key in flat:
                raise CompareError(f"parameter name {sub_key!r} is ambiguous after flattening")
            flat[sub_key] = sub_value
    return flat


def _same(a: Any, b: Any) -> bool:
    """Equality that, like the config hash, tells 1 from 1.0 and True from 1."""
    try:
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    except TypeError as exc:
        raise CompareError(f"parameter values must be JSON-compatible: {exc}") from exc


def diff_params(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[ParamDiff]:
    """Differences between two parameter sets, sorted by key. Equal parameters are omitted."""
    flat_old, flat_new = flatten_params(old), flatten_params(new)
    diffs: list[ParamDiff] = []
    for key in sorted(set(flat_old) | set(flat_new)):
        if key not in flat_new:
            diffs.append(ParamDiff(key, "removed", flat_old[key], None))
        elif key not in flat_old:
            diffs.append(ParamDiff(key, "added", None, flat_new[key]))
        elif not _same(flat_old[key], flat_new[key]):
            diffs.append(ParamDiff(key, "changed", flat_old[key], flat_new[key]))
    return diffs


# -- metrics ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricDelta:
    name: str
    old: float
    new: float
    delta: float  # new - old
    relative: float | None  # delta / |old|, or None when old is 0
    improved: bool | None  # None when the metric's direction is unknown or nothing changed


@dataclass(frozen=True)
class MetricComparison:
    deltas: tuple[MetricDelta, ...]
    only_in_old: tuple[str, ...]
    only_in_new: tuple[str, ...]


def _check_metric_values(values: Mapping[str, float], where: str) -> None:
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CompareError(f"{where}[{name!r}] must be a number, got {value!r}")
        if not math.isfinite(value):
            raise CompareError(f"{where}[{name!r}] must be finite, got {value!r}")


def resolve_direction(metric: str, directions: Mapping[str, bool] | None) -> bool:
    """An explicit direction if one was given for this metric, else the built-in table."""
    if directions is not None and metric in directions:
        return directions[metric]
    return higher_is_better(metric)


def compare_metrics(
    old: Mapping[str, float],
    new: Mapping[str, float],
    directions: Mapping[str, bool] | None = None,
) -> MetricComparison:
    """Per-metric change from `old` to `new` for metrics present in both, sorted by name.

    `directions` maps a metric name to True (higher is better) or False, and takes priority over
    the built-in table; metrics with no known direction get `improved=None`.
    """
    _check_metric_values(old, "old")
    _check_metric_values(new, "new")
    deltas: list[MetricDelta] = []
    for name in sorted(set(old) & set(new)):
        before, after = float(old[name]), float(new[name])
        delta = after - before
        try:
            better = resolve_direction(name, directions)
        except CompareError:
            improved = None
        else:
            improved = None if delta == 0 else (delta > 0) == better
        relative = None if before == 0 else delta / abs(before)
        deltas.append(MetricDelta(name, before, after, delta, relative, improved))
    return MetricComparison(
        tuple(deltas),
        tuple(sorted(set(old) - set(new))),
        tuple(sorted(set(new) - set(old))),
    )


@dataclass(frozen=True)
class RankedRun:
    rank: int
    run_id: str
    value: float


@dataclass(frozen=True)
class Ranking:
    metric: str
    ranked: tuple[RankedRun, ...]
    missing: tuple[str, ...]  # runs that did not report the metric


def rank_runs(
    results: Mapping[str, Mapping[str, float]],
    metric: str,
    higher_is_better_override: bool | None = None,
) -> Ranking:
    """Rank runs by one metric, best first.

    Runs with exactly equal values share a rank and the next rank is skipped (1, 2, 2, 4).
    Within a tie, runs are ordered by run id so the output never depends on input order.
    """
    better_high = (
        higher_is_better(metric) if higher_is_better_override is None else higher_is_better_override
    )
    values: dict[str, float] = {}
    missing: list[str] = []
    for run_id, metrics in results.items():
        if metric not in metrics:
            missing.append(run_id)
            continue
        _check_metric_values({metric: metrics[metric]}, run_id)
        values[run_id] = float(metrics[metric])
    sign = -1.0 if better_high else 1.0
    order = sorted(values, key=lambda run_id: (sign * values[run_id], run_id))
    ranked: list[RankedRun] = []
    for position, run_id in enumerate(order, start=1):
        rank = ranked[-1].rank if ranked and values[run_id] == ranked[-1].value else position
        ranked.append(RankedRun(rank, run_id, values[run_id]))
    return Ranking(metric, tuple(ranked), tuple(sorted(missing)))


# -- paired bootstrap ------------------------------------------------------------------------


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """The q-th quantile (0 <= q <= 1) by linear interpolation between closest ranks."""
    if not sorted_values:
        raise CompareError("percentile of an empty sequence")
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not 0 <= q <= 1:
        raise CompareError(f"q must be between 0 and 1, got {q!r}")
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


@dataclass(frozen=True)
class BootstrapResult:
    mean_diff: float  # mean of (a - b) over the observed samples
    ci_low: float
    ci_high: float
    confidence: float
    n_samples: int
    n_resamples: int
    prob_a_better: float  # share of resamples where a's mean beats b's


def paired_bootstrap(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile bootstrap interval for the mean per-example difference a - b.

    Examples are resampled with replacement as pairs, so the pairing between the two models on
    each example is kept. Indices come from `int(rng.random() * n)` on a generator derived from
    `seed`, which depends only on Python's seeded Mersenne Twister. Higher scores must be better
    (use 0/1 correctness, or negated errors).
    """
    if len(scores_a) != len(scores_b):
        raise CompareError(f"score lists differ in length: {len(scores_a)} vs {len(scores_b)}")
    n = len(scores_a)
    if n < 2:
        raise CompareError("need at least 2 paired examples")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int):
        raise CompareError(f"n_resamples must be an integer, got {n_resamples!r}")
    if not MIN_RESAMPLES <= n_resamples <= MAX_RESAMPLES:
        raise CompareError(f"n_resamples must be between {MIN_RESAMPLES} and {MAX_RESAMPLES}")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise CompareError(f"confidence must be a number, got {confidence!r}")
    if not 0 < confidence < 1:
        raise CompareError(f"confidence must be strictly between 0 and 1, got {confidence}")
    check_seed(seed)
    _check_metric_values(dict(enumerate(scores_a)), "scores_a")  # type: ignore[arg-type]
    _check_metric_values(dict(enumerate(scores_b)), "scores_b")  # type: ignore[arg-type]

    diffs = [float(a) - float(b) for a, b in zip(scores_a, scores_b, strict=True)]
    rng = rng_for(seed, "bootstrap")
    means: list[float] = []
    for _ in range(n_resamples):
        total = math.fsum(diffs[int(rng.random() * n)] for _ in range(n))
        means.append(total / n)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    return BootstrapResult(
        mean_diff=math.fsum(diffs) / n,
        ci_low=percentile(means, tail),
        ci_high=percentile(means, 1.0 - tail),
        confidence=float(confidence),
        n_samples=n,
        n_resamples=n_resamples,
        prob_a_better=sum(1 for m in means if m > 0) / n_resamples,
    )
