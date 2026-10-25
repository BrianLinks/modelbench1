"""Compare stored runs: deltas against a baseline, rankings, and what differs between them."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .compare import (
    MetricComparison,
    ParamDiff,
    Ranking,
    compare_metrics,
    diff_params,
    rank_runs,
    resolve_direction,
)
from .environment import EnvDifference, compare_environments
from .errors import CompareError
from .store import RunRecord, Store


@dataclass(frozen=True)
class RunDiff:
    """How one run differs from the baseline run."""

    run_id: str
    params: tuple[ParamDiff, ...]
    metrics: MetricComparison
    same_config: bool  # identical settings (the config hash ignores the experiment label)
    same_dataset: bool  # same dataset id and version
    same_seed: bool
    environment: tuple[EnvDifference, ...]


@dataclass(frozen=True)
class RunComparison:
    baseline: str
    run_ids: tuple[str, ...]
    values: dict[str, dict[str, float]]  # run id -> metric -> final value (selected metrics)
    diffs: tuple[RunDiff, ...]  # one per non-baseline run, in the order given
    rankings: tuple[Ranking, ...]  # one per metric with a known direction, sorted by metric
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable form with a stable key order."""
        return {
            "baseline": self.baseline,
            "run_ids": list(self.run_ids),
            "values": {run_id: dict(sorted(v.items())) for run_id, v in self.values.items()},
            "diffs": [_diff_to_dict(d) for d in self.diffs],
            "rankings": [
                {
                    "metric": r.metric,
                    "ranked": [
                        {"rank": x.rank, "run_id": x.run_id, "value": x.value} for x in r.ranked
                    ],
                    "missing": list(r.missing),
                }
                for r in self.rankings
            ],
            "warnings": list(self.warnings),
        }


def _diff_to_dict(diff: RunDiff) -> dict[str, Any]:
    return {
        "run_id": diff.run_id,
        "same_config": diff.same_config,
        "same_dataset": diff.same_dataset,
        "same_seed": diff.same_seed,
        "params": [{"key": p.key, "kind": p.kind, "old": p.old, "new": p.new} for p in diff.params],
        "metrics": [
            {
                "name": m.name,
                "old": m.old,
                "new": m.new,
                "delta": m.delta,
                "relative": m.relative,
                "improved": m.improved,
            }
            for m in diff.metrics.deltas
        ],
        "only_in_baseline": list(diff.metrics.only_in_old),
        "only_in_run": list(diff.metrics.only_in_new),
        "environment": [
            {"key": e.key, "old": e.old, "new": e.new, "impact": e.impact} for e in diff.environment
        ],
    }


def compare_runs(
    store: Store,
    run_ids: Sequence[str],
    metrics: Sequence[str] | None = None,
    directions: Mapping[str, bool] | None = None,
) -> RunComparison:
    """Compare completed runs. The first id is the baseline.

    `metrics` limits the comparison to the named metrics (default: every metric any run
    reported). `directions` supplies higher-is-better flags for custom metric names. Metrics
    with no known direction are compared but not ranked, and a warning says so.
    """
    ids = _check_ids(run_ids)
    records = {run_id: store.get_run(run_id) for run_id in ids}
    unfinished = {i: r.status for i, r in records.items() if r.status != "completed"}
    if unfinished:
        listing = ", ".join(f"{i} is {status}" for i, status in unfinished.items())
        raise CompareError(f"only completed runs can be compared: {listing}")

    reported = {run_id: store.final_metrics(run_id) for run_id in ids}
    selected = _select_metrics(reported, metrics)
    values = {
        run_id: {name: value for name, value in found.items() if name in selected}
        for run_id, found in reported.items()
    }

    baseline, others = ids[0], ids[1:]
    diffs = tuple(
        _diff_against(
            records[baseline], records[run_id], values[baseline], values[run_id], directions
        )
        for run_id in others
    )

    warnings: list[str] = []
    versions = {(r.dataset_id, r.dataset_version) for r in records.values()}
    if len(versions) > 1:
        shown = ", ".join(f"{d}@{v}" for d, v in sorted(versions))
        warnings.append(f"runs used different datasets or dataset versions: {shown}")
    for metric in sorted(selected):
        lacking = [run_id for run_id in ids if metric not in values[run_id]]
        if lacking:
            warnings.append(f"metric {metric!r} was not reported by: {', '.join(lacking)}")

    rankings: list[Ranking] = []
    for metric in sorted(selected):
        try:
            direction = resolve_direction(metric, directions)
        except CompareError:
            warnings.append(f"metric {metric!r} was not ranked: its direction is unknown")
            continue
        rankings.append(rank_runs(values, metric, higher_is_better_override=direction))

    return RunComparison(baseline, ids, values, diffs, tuple(rankings), tuple(warnings))


def _check_ids(run_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(run_ids, (str, bytes)) or not isinstance(run_ids, Sequence):
        raise CompareError("run_ids must be a sequence of run ids")
    ids = tuple(run_ids)
    if len(ids) < 2:
        raise CompareError("need at least two runs to compare")
    if len(set(ids)) != len(ids):
        raise CompareError("run ids must be distinct")
    return ids


def _select_metrics(
    values: Mapping[str, Mapping[str, float]], requested: Sequence[str] | None
) -> set[str]:
    available = {name for found in values.values() for name in found}
    if requested is None:
        return available
    if isinstance(requested, (str, bytes)):
        raise CompareError("metrics must be a sequence of metric names")
    absent = sorted(set(requested) - available)
    if absent:
        raise CompareError(f"no compared run reported: {', '.join(absent)}")
    return set(requested)


def _diff_against(
    base: RunRecord,
    other: RunRecord,
    base_values: Mapping[str, float],
    other_values: Mapping[str, float],
    directions: Mapping[str, bool] | None,
) -> RunDiff:
    return RunDiff(
        run_id=other.run_id,
        params=tuple(diff_params(base.params, other.params)),
        metrics=compare_metrics(base_values, other_values, directions),
        same_config=base.config_hash == other.config_hash,
        same_dataset=(base.dataset_id, base.dataset_version)
        == (other.dataset_id, other.dataset_version),
        same_seed=base.seed == other.seed,
        environment=tuple(compare_environments(base.environment, other.environment)),
    )
