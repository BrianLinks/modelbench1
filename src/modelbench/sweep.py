"""Run one experiment over every combination of a parameter grid and rank the results.

A sweep file is JSON with two keys: `config`, an ordinary experiment config, and `grid`, a map
from parameter name to a non-empty list of values. Every combination becomes one run of the same
experiment, in the deterministic order `expand_grid` gives. By default a combination that was
already run with identical settings on the same dataset version is reused instead of repeated,
so a sweep can be re-run or extended cheaply.

A combination that fails (for example `k` larger than the training set) is recorded in the
result with its error and the sweep carries on; `stop_on_error` stops at the first failure.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compare import Ranking, higher_is_better, rank_runs
from .config import ExperimentConfig, expand_config
from .datasets import Table
from .errors import ModelBenchError, SweepError
from .runner import run_experiment
from .store import Store

MAX_SWEEP_FILE_BYTES = 1_000_000


@dataclass(frozen=True)
class SweepEntry:
    params: dict[str, Any]
    run_id: str | None  # None when it failed before any run was recorded (bad parameters)
    reused: bool
    metrics: dict[str, float]
    error: str | None


@dataclass(frozen=True)
class SweepResult:
    experiment_id: str
    rank_metric: str
    entries: tuple[SweepEntry, ...]  # in grid order
    ranking: Ranking  # best first, over the entries that produced a run

    @property
    def failed(self) -> tuple[SweepEntry, ...]:
        return tuple(e for e in self.entries if e.error is not None)

    def best(self) -> SweepEntry | None:
        """The top-ranked entry (ties go to the lowest run id), or None if nothing succeeded."""
        if not self.ranking.ranked:
            return None
        top = self.ranking.ranked[0].run_id
        return next(e for e in self.entries if e.run_id == top)

    def ranked_entries(self) -> list[tuple[int, SweepEntry]]:
        """`(rank, entry)` best first, then failed entries (rank 0) in grid order."""
        by_run = {e.run_id: e for e in self.entries if e.error is None}
        out = [(r.rank, by_run[r.run_id]) for r in self.ranking.ranked]
        out += [(0, e) for e in self.entries if e.error is not None]
        return out


def _newest_run_id(store: Store) -> str | None:
    newest = store.list_runs(limit=1)
    return newest[0].run_id if newest else None


def load_sweep(path: str | Path) -> tuple[ExperimentConfig, dict[str, list[Any]]]:
    """Read a sweep file: `{"config": {...}, "grid": {"param": [values, ...]}}`."""
    target = Path(path)
    try:
        if target.stat().st_size > MAX_SWEEP_FILE_BYTES:
            raise SweepError(f"{target} is larger than {MAX_SWEEP_FILE_BYTES} bytes")
        document = json.loads(target.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise SweepError(f"cannot read sweep file {target}: {exc}") from exc
    except UnicodeDecodeError:
        raise SweepError(f"sweep file {target} is not valid UTF-8") from None
    except json.JSONDecodeError as exc:
        raise SweepError(f"sweep file {target} is not valid JSON: {exc}") from exc
    return parse_sweep(document)


def parse_sweep(document: object) -> tuple[ExperimentConfig, dict[str, list[Any]]]:
    if not isinstance(document, dict):
        raise SweepError("a sweep file must be a JSON object")
    extra = sorted(set(document) - {"config", "grid"})
    if extra:
        raise SweepError(f"unknown key(s) in sweep file: {', '.join(extra)}")
    if "config" not in document or "grid" not in document:
        raise SweepError('a sweep file needs both "config" and "grid"')
    grid = document["grid"]
    if not isinstance(grid, dict) or not grid:
        raise SweepError('"grid" must be an object with at least one parameter')
    for name, values in grid.items():
        if not isinstance(values, list) or not values:
            raise SweepError(f"grid[{name!r}] must be a non-empty list")
    return ExperimentConfig.from_dict(document["config"]), dict(grid)


def run_sweep(
    store: Store,
    config: ExperimentConfig,
    grid: Mapping[str, Sequence[Any]],
    table: Table,
    label_column: str,
    *,
    feature_columns: Sequence[str] | None = None,
    artifact_dir: str | Path | None = None,
    reuse_existing: bool = True,
    rank_by: str | None = None,
    stop_on_error: bool = False,
    environment: Mapping[str, str | None] | None = None,
) -> SweepResult:
    """Run every grid combination of `config`. Ranks by `rank_by` (default: the first metric)."""
    metric = rank_by if rank_by is not None else config.metrics[0]
    reported = set(config.metrics)
    if config.cv_folds is not None:  # cross-validation also reports each metric's spread
        reported |= {f"{name}_std" for name in config.metrics}
    if metric not in reported:
        raise SweepError(f"cannot rank by {metric!r}; the config reports {sorted(reported)}")
    higher_is_better(metric)  # fail now, not after the runs, if the direction is unknown
    configs = expand_config(config, grid)
    entries: list[SweepEntry] = []
    for variant in configs:
        newest_before = _newest_run_id(store)
        try:
            result = run_experiment(
                store,
                variant,
                table,
                label_column,
                feature_columns=feature_columns,
                artifact_dir=artifact_dir,
                reuse_existing=reuse_existing,
                environment=environment,
            )
        except ModelBenchError as exc:
            newest_after = _newest_run_id(store)
            failed_run = newest_after if newest_after != newest_before else None
            entries.append(SweepEntry(dict(variant.params), failed_run, False, {}, str(exc)))
            if stop_on_error:
                break
            continue
        entries.append(
            SweepEntry(dict(variant.params), result.run_id, result.reused, result.metrics, None)
        )
    ranking = rank_runs(
        {e.run_id: e.metrics for e in entries if e.error is None and e.run_id}, metric
    )
    return SweepResult(config.experiment_id, metric, tuple(entries), ranking)
