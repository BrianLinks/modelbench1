"""Run one experiment over a parameter grid, or random search it, and rank the results.

A sweep file is JSON with two keys: `config`, an ordinary experiment config, and `grid`, a map
from parameter name to a non-empty list of values. Every combination becomes one run of the same
experiment, in the deterministic order `expand_grid` gives. By default a combination that was
already run with identical settings on the same dataset version is reused instead of repeated,
so a sweep can be re-run or extended cheaply.

A combination that fails (for example `k` larger than the training set) is recorded in the
result with its error and the sweep carries on; `stop_on_error` stops at the first failure.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .compare import Ranking, higher_is_better, rank_runs
from .config import ExperimentConfig, check_json_value, expand_config
from .datasets import Table
from .errors import ConfigError, ModelBenchError, SweepError
from .runner import run_experiment
from .seeding import rng_for
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


def _read_json(target: Path, what: str) -> object:
    try:
        if target.stat().st_size > MAX_SWEEP_FILE_BYTES:
            raise SweepError(f"{target} is larger than {MAX_SWEEP_FILE_BYTES} bytes")
        return json.loads(target.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise SweepError(f"cannot read {what} {target}: {exc}") from exc
    except UnicodeDecodeError:
        raise SweepError(f"{what} {target} is not valid UTF-8") from None
    except json.JSONDecodeError as exc:
        raise SweepError(f"{what} {target} is not valid JSON: {exc}") from exc


def load_sweep(path: str | Path) -> tuple[ExperimentConfig, dict[str, list[Any]]]:
    """Read a sweep file: `{"config": {...}, "grid": {"param": [values, ...]}}`."""
    return parse_sweep(_read_json(Path(path), "sweep file"))


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


def _check_rank_metric(config: ExperimentConfig, rank_by: str | None) -> str:
    metric = rank_by if rank_by is not None else config.metrics[0]
    reported = set(config.metrics)
    if config.cv_folds is not None:  # cross-validation also reports each metric's spread
        reported |= {f"{name}_std" for name in config.metrics}
    if metric not in reported:
        raise SweepError(f"cannot rank by {metric!r}; the config reports {sorted(reported)}")
    higher_is_better(metric)  # fail now, not after the runs, if the direction is unknown
    return metric


def _run_variants(
    store: Store,
    experiment_id: str,
    variants: Sequence[ExperimentConfig],
    metric: str,
    table: Table,
    label_column: str,
    *,
    feature_columns: Sequence[str] | None,
    artifact_dir: str | Path | None,
    reuse_existing: bool,
    stop_on_error: bool,
    environment: Mapping[str, str | None] | None,
) -> SweepResult:
    entries: list[SweepEntry] = []
    for variant in variants:
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
    return SweepResult(experiment_id, metric, tuple(entries), ranking)


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
    metric = _check_rank_metric(config, rank_by)
    return _run_variants(
        store,
        config.experiment_id,
        expand_config(config, grid),
        metric,
        table,
        label_column,
        feature_columns=feature_columns,
        artifact_dir=artifact_dir,
        reuse_existing=reuse_existing,
        stop_on_error=stop_on_error,
        environment=environment,
    )


# -- random search ---------------------------------------------------------------------------

MAX_TRIALS = 1000
SPACE_KINDS = ("int", "uniform", "log_uniform", "choice")


def _bound(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SweepError(f"{where} must be a finite number, got {value!r}")
    return float(value)


def check_space(space: object) -> dict[str, tuple[str, Any]]:
    """Validate a search space and return `{name: (kind, argument)}` in sorted name order.

    Each parameter is an object with exactly one key: `{"int": [lo, hi]}` (both ends possible),
    `{"uniform": [lo, hi]}`, `{"log_uniform": [lo, hi]}` (0 < lo < hi) or `{"choice": [values]}`.
    """
    if not isinstance(space, dict) or not space:
        raise SweepError('"space" must be an object with at least one parameter')
    checked: dict[str, tuple[str, Any]] = {}
    for name in sorted(space):
        spec = space[name]
        where = f"space[{name!r}]"
        if not isinstance(spec, dict) or len(spec) != 1:
            raise SweepError(f"{where} must be an object with one of {SPACE_KINDS}")
        kind, arg = next(iter(spec.items()))
        if kind not in SPACE_KINDS:
            raise SweepError(f"{where}: unknown kind {kind!r}; use one of {SPACE_KINDS}")
        if kind == "choice":
            if not isinstance(arg, list) or not arg:
                raise SweepError(f"{where}: choice needs a non-empty list")
            try:
                check_json_value(arg, where)
            except ConfigError as exc:
                raise SweepError(str(exc)) from exc
            checked[name] = (kind, arg)
            continue
        if not isinstance(arg, list) or len(arg) != 2:
            raise SweepError(f"{where}: {kind} needs [low, high]")
        low, high = _bound(arg[0], f"{where} low"), _bound(arg[1], f"{where} high")
        if kind == "int":
            whole = all(float(v).is_integer() and not isinstance(v, bool) for v in arg)
            if not whole or low > high:
                raise SweepError(f"{where}: int needs whole numbers with low <= high")
            checked[name] = (kind, (int(low), int(high)))
            continue
        if not low < high:
            raise SweepError(f"{where}: {kind} needs low < high")
        if kind == "log_uniform" and low <= 0:
            raise SweepError(f"{where}: log_uniform needs low > 0")
        checked[name] = (kind, (low, high))
    return checked


def sample_params(space: Mapping[str, tuple[str, Any]], seed: int, trial: int) -> dict[str, Any]:
    """The parameters for one trial. Depends only on the arguments: trial n is always the same."""
    rng = rng_for(seed, "search", trial)
    out: dict[str, Any] = {}
    for name in sorted(space):  # a fixed order, so adding a trial never changes an earlier one
        kind, arg = space[name]
        if kind == "int":
            out[name] = rng.randint(arg[0], arg[1])
        elif kind == "uniform":
            out[name] = float(f"{rng.uniform(arg[0], arg[1]):.6g}")
        elif kind == "log_uniform":
            out[name] = float(f"{math.exp(rng.uniform(math.log(arg[0]), math.log(arg[1]))):.6g}")
        else:
            out[name] = copy.deepcopy(rng.choice(arg))
    return out


def load_search(path: str | Path) -> tuple[ExperimentConfig, dict[str, Any], int]:
    """Read a search file: `{"config": {...}, "space": {...}, "trials": N}`."""
    document = _read_json(Path(path), "search file")
    if not isinstance(document, dict):
        raise SweepError("a search file must be a JSON object")
    extra = sorted(set(document) - {"config", "space", "trials"})
    if extra:
        raise SweepError(f"unknown key(s) in search file: {', '.join(extra)}")
    if not {"config", "space", "trials"} <= set(document):
        raise SweepError('a search file needs "config", "space" and "trials"')
    trials = document["trials"]
    if isinstance(trials, bool) or not isinstance(trials, int) or not 1 <= trials <= MAX_TRIALS:
        raise SweepError(f'"trials" must be a whole number from 1 to {MAX_TRIALS}')
    check_space(document["space"])
    return ExperimentConfig.from_dict(document["config"]), dict(document["space"]), trials


def run_search(
    store: Store,
    config: ExperimentConfig,
    space: Mapping[str, Any],
    trials: int,
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
    """Random search: run `trials` parameter sets drawn from `space`, seeded by `config.seed`.

    Trial n depends only on the seed and n. A trial that repeats an earlier one (likely in a
    small space) is skipped, so the result can have fewer entries than `trials`. A parameter may
    not be in both `space` and the config's own `params`.
    """
    if isinstance(trials, bool) or not isinstance(trials, int) or not 1 <= trials <= MAX_TRIALS:
        raise SweepError(f"trials must be a whole number from 1 to {MAX_TRIALS}")
    checked = check_space(space)
    clash = sorted(set(checked) & set(config.params))
    if clash:
        raise SweepError(f"parameters in both the space and the config: {', '.join(clash)}")
    metric = _check_rank_metric(config, rank_by)
    variants: list[ExperimentConfig] = []
    seen: set[str] = set()
    for trial in range(trials):
        sampled = sample_params(checked, config.seed, trial)
        key = json.dumps(sampled, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        variants.append(replace(config, params={**config.params, **sampled}))
    return _run_variants(
        store,
        config.experiment_id,
        variants,
        metric,
        table,
        label_column,
        feature_columns=feature_columns,
        artifact_dir=artifact_dir,
        reuse_existing=reuse_existing,
        stop_on_error=stop_on_error,
        environment=environment,
    )
