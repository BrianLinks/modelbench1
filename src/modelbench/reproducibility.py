"""Check that a stored run can be reproduced, and clean up runs a crash left behind.

`check_reproducibility` re-runs a completed run's stored configuration on the same data in a
scratch in-memory store, so checking never adds to the run history. It compares the SHA-256 of
the predictions and every stored metric, and lists environment differences so a mismatch can be
explained rather than only reported.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .datasets import Table, make_version
from .environment import EnvDifference, compare_environments, environment_fingerprint
from .errors import ModelBenchError, ReproError
from .runner import run_experiment
from .store import Store

REPRODUCED = "reproduced"
NOT_REPRODUCED = "not_reproduced"
DATASET_CHANGED = "dataset_changed"
RERUN_FAILED = "rerun_failed"
_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class MetricCheck:
    name: str
    original: float
    rerun: float | None
    difference: float | None
    within_tolerance: bool


@dataclass(frozen=True)
class ReproReport:
    run_id: str
    verdict: str
    tolerance: float
    predictions_match: bool | None
    metrics: tuple[MetricCheck, ...]
    environment_differences: tuple[EnvDifference, ...]
    explained_by_environment: bool
    detail: str

    @property
    def reproduced(self) -> bool:
        return self.verdict == REPRODUCED

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "verdict": self.verdict,
            "tolerance": self.tolerance,
            "predictions_match": self.predictions_match,
            "metrics": [
                {
                    "name": c.name,
                    "original": c.original,
                    "rerun": c.rerun,
                    "difference": c.difference,
                    "within_tolerance": c.within_tolerance,
                }
                for c in self.metrics
            ],
            "environment_differences": [
                {"key": d.key, "old": d.old, "new": d.new, "impact": d.impact}
                for d in self.environment_differences
            ],
            "explained_by_environment": self.explained_by_environment,
            "detail": self.detail,
        }


def check_tolerance(tolerance: object) -> float:
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ReproError(f"tolerance must be a number, got {tolerance!r}")
    value = float(tolerance)
    if not math.isfinite(value) or value < 0:
        raise ReproError(f"tolerance must be finite and not negative, got {tolerance!r}")
    return value


def check_reproducibility(
    store: Store,
    run_id: str,
    table: Table,
    label_column: str,
    *,
    feature_columns: Sequence[str] | None = None,
    tolerance: float = 0.0,
    environment: Mapping[str, str | None] | None = None,
) -> ReproReport:
    """Re-run `run_id` on `table` and compare against what the run recorded.

    Reproduced means the predictions hash is identical and every stored metric differs by at
    most `tolerance` (absolute). A different predictions hash is never reproduced, even when the
    metrics agree, because the model behaved differently. If `table` is not the dataset version
    the run used, nothing is re-run and the verdict is `dataset_changed`. Only completed runs
    that recorded a predictions hash can be checked. `environment` defaults to the current one.
    """
    tol = check_tolerance(tolerance)
    record = store.get_run(run_id)
    if record.status != "completed":
        raise ReproError(f"run {run_id} is {record.status}; only completed runs can be checked")
    expected_hash = record.tags.get("predictions_sha256")
    if expected_hash is None:
        raise ReproError(f"run {run_id} recorded no predictions hash, so it cannot be checked")
    original = store.final_metrics(run_id)
    config = store.get_config(run_id)
    current_env = dict(environment) if environment is not None else environment_fingerprint()
    env_diffs = tuple(compare_environments(record.environment, current_env))

    version = make_version(config.dataset_id, table, label_column)
    if version.version != record.dataset_version:
        return ReproReport(
            run_id,
            DATASET_CHANGED,
            tol,
            None,
            tuple(MetricCheck(n, v, None, None, False) for n, v in sorted(original.items())),
            env_diffs,
            False,
            f"the run used dataset version {record.dataset_version} "
            f"but the data given is {version.version}",
        )

    scratch = Store(":memory:")
    try:
        try:
            result = run_experiment(
                scratch,
                config,
                table,
                label_column,
                feature_columns=feature_columns,
                environment=current_env,
            )
        except ModelBenchError as exc:
            return ReproReport(
                run_id,
                RERUN_FAILED,
                tol,
                None,
                tuple(MetricCheck(n, v, None, None, False) for n, v in sorted(original.items())),
                env_diffs,
                bool(env_diffs),
                f"the re-run failed: {type(exc).__name__}: {exc}",
            )
    finally:
        scratch.close()

    checks = []
    for name, old in sorted(original.items()):
        new = result.metrics.get(name)
        if new is None:
            checks.append(MetricCheck(name, old, None, None, False))
            continue
        diff = abs(new - old)
        checks.append(MetricCheck(name, old, new, diff, diff <= tol))
    predictions_match = result.predictions_sha256 == expected_hash
    ok = predictions_match and all(c.within_tolerance for c in checks)
    if ok:
        detail = "predictions and all metrics match"
    elif not predictions_match:
        detail = "the predictions differ from the original run"
    else:
        detail = "predictions match but a metric is outside the tolerance"
    return ReproReport(
        run_id,
        REPRODUCED if ok else NOT_REPRODUCED,
        tol,
        predictions_match,
        tuple(checks),
        env_diffs,
        (not ok) and bool(env_diffs),
        detail,
    )


def recover_stale_runs(
    store: Store, older_than_seconds: float, *, now: datetime | None = None
) -> list[str]:
    """Mark runs still `running` after `older_than_seconds` as failed and return their ids.

    A run left `running` by a crash can never finish, so it is closed with an explanatory error.
    Runs younger than the cutoff are left alone because they may still be executing; timestamps
    have one-second resolution. `now` must be timezone-aware UTC (default: the current time).
    """
    if (
        isinstance(older_than_seconds, bool)
        or not isinstance(older_than_seconds, (int, float))
        or not math.isfinite(older_than_seconds)
        or older_than_seconds <= 0
    ):
        raise ReproError(
            f"older_than_seconds must be a positive number, got {older_than_seconds!r}"
        )
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None or now.utcoffset() != timedelta(0):
        raise ReproError("now must be a timezone-aware UTC datetime")
    cutoff = (now - timedelta(seconds=older_than_seconds)).strftime(_TIME_FORMAT)
    stale = store.stale_running(cutoff)
    for run_id in stale:
        store.fail_run(run_id, f"abandoned: still running after {older_than_seconds:g} seconds")
    return stale
