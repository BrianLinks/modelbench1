"""Successive halving: spend little on many candidates, more on the few that look best.

Random search runs every draw at full cost. Halving starts many draws at a small budget, keeps
the best `1 / eta` of them, raises the budget by a factor of `eta`, and repeats until the last
survivors have run at the full budget. The budget is one numeric parameter of the model that you
name, such as `n_estimators` for a forest or `max_iter` for the perceptron.

Candidates are drawn exactly as in `run_search` (candidate n depends only on the seed and n), so
a halving run and a search with the same seed and space look at the same parameter sets. Every
step is an ordinary recorded run of the same experiment, so each can be shown, compared and
reproduced, and a rerun reuses the runs that already exist. A candidate that fails is dropped.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .datasets import Table
from .errors import SweepError
from .store import Store
from .sweep import (
    MAX_TRIALS,
    SweepEntry,
    SweepResult,
    _check_rank_metric,
    _read_json,
    _run_variants,
    check_space,
    sample_params,
)

MAX_ETA = 10
MAX_RESOURCE = 1_000_000


@dataclass(frozen=True)
class HalvingSpec:
    resource: str  # the model parameter that is the budget
    min_resource: int
    max_resource: int
    eta: int = 3


@dataclass(frozen=True)
class HalvingRound:
    resource_value: int
    result: SweepResult
    kept: tuple[str, ...]  # run ids that go on to the next round; empty in the last round


@dataclass(frozen=True)
class HalvingResult:
    experiment_id: str
    rank_metric: str
    resource: str
    rounds: tuple[HalvingRound, ...]

    def best(self) -> SweepEntry | None:
        """The top entry of the last round that produced a result, or None if none did."""
        for finished in reversed(self.rounds):
            best = finished.result.best()
            if best is not None:
                return best
        return None

    @property
    def n_runs(self) -> int:
        return sum(len(r.result.entries) for r in self.rounds)


def _whole(value: object, where: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise SweepError(f"{where} must be a whole number from {low} to {high}, got {value!r}")
    return value


def check_spec(spec: HalvingSpec) -> HalvingSpec:
    if not isinstance(spec.resource, str) or not spec.resource:
        raise SweepError("halving resource must be the name of a model parameter")
    low = _whole(spec.min_resource, "min_resource", 1, MAX_RESOURCE)
    high = _whole(spec.max_resource, "max_resource", 1, MAX_RESOURCE)
    if low > high:
        raise SweepError(f"min_resource ({low}) must not exceed max_resource ({high})")
    _whole(spec.eta, "eta", 2, MAX_ETA)
    return spec


def budgets(spec: HalvingSpec) -> list[int]:
    """The budget of each round: `min`, `min * eta`, ... while within `max`, and `max` last."""
    check_spec(spec)
    values = [spec.min_resource]
    while values[-1] * spec.eta <= spec.max_resource:
        values.append(values[-1] * spec.eta)
    if values[-1] != spec.max_resource:
        values.append(spec.max_resource)
    return values


def survivors_after(n: int, eta: int) -> int:
    """How many of `n` candidates go on to the next round: a share of `1 / eta`, at least one."""
    return math.ceil(n / eta)


def load_halving(path: str | Path) -> tuple[ExperimentConfig, dict[str, Any], int, HalvingSpec]:
    """Read `{"config": {...}, "space": {...}, "candidates": N, "halving": {...}}`."""
    document = _read_json(Path(path), "halving file")
    if not isinstance(document, dict):
        raise SweepError("a halving file must be a JSON object")
    keys = {"config", "space", "candidates", "halving"}
    extra = sorted(set(document) - keys)
    if extra:
        raise SweepError(f"unknown key(s) in halving file: {', '.join(extra)}")
    if not keys <= set(document):
        raise SweepError('a halving file needs "config", "space", "candidates" and "halving"')
    block = document["halving"]
    if not isinstance(block, dict):
        raise SweepError('"halving" must be an object')
    wanted = {"resource", "min_resource", "max_resource"}
    unknown = sorted(set(block) - wanted - {"eta"})
    if unknown or not wanted <= set(block):
        raise SweepError(
            '"halving" needs resource, min_resource and max_resource, and may have eta'
            + (f"; unknown: {', '.join(unknown)}" if unknown else "")
        )
    spec = check_spec(
        HalvingSpec(
            block["resource"], block["min_resource"], block["max_resource"], block.get("eta", 3)
        )
    )
    candidates = _whole(document["candidates"], '"candidates"', 1, MAX_TRIALS)
    check_space(document["space"])
    return ExperimentConfig.from_dict(document["config"]), dict(document["space"]), candidates, spec


def run_halving(
    store: Store,
    config: ExperimentConfig,
    space: Mapping[str, Any],
    candidates: int,
    spec: HalvingSpec,
    table: Table,
    label_column: str,
    *,
    feature_columns: Sequence[str] | None = None,
    artifact_dir: str | Path | None = None,
    reuse_existing: bool = True,
    rank_by: str | None = None,
    environment: Mapping[str, str | None] | None = None,
) -> HalvingResult:
    """Run successive halving over `candidates` draws from `space`, ranked by `rank_by`.

    The resource may be in neither `space` nor the config's own `params`, and a parameter may
    not be in both. Draws that repeat an earlier one are skipped, so there can be fewer
    candidates than requested.
    """
    check_spec(spec)
    candidates = _whole(candidates, "candidates", 1, MAX_TRIALS)
    checked = check_space(space)
    if spec.resource in checked or spec.resource in config.params:
        raise SweepError(
            f"the halving resource {spec.resource!r} must not also be set in the space "
            "or the config"
        )
    clash = sorted(set(checked) & set(config.params))
    if clash:
        raise SweepError(f"parameters in both the space and the config: {', '.join(clash)}")
    metric = _check_rank_metric(config, rank_by)

    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for trial in range(candidates):
        sampled = sample_params(checked, config.seed, trial)
        key = json.dumps(sampled, sort_keys=True)
        if key not in seen:
            seen.add(key)
            pool.append(sampled)

    rounds: list[HalvingRound] = []
    schedule = budgets(spec)
    for position, budget in enumerate(schedule):
        variants = [
            replace(config, params={**config.params, **sampled, spec.resource: budget})
            for sampled in pool
        ]
        result = _run_variants(
            store,
            config.experiment_id,
            variants,
            metric,
            table,
            label_column,
            feature_columns=feature_columns,
            artifact_dir=artifact_dir,
            reuse_existing=reuse_existing,
            stop_on_error=False,
            environment=environment,
        )
        last = position == len(schedule) - 1
        ranked = [r.run_id for r in result.ranking.ranked]
        keep = [] if last else ranked[: survivors_after(len(pool), spec.eta)]
        rounds.append(HalvingRound(budget, result, tuple(keep)))
        if last or not keep:
            break
        by_run = {e.run_id: e.params for e in result.entries if e.error is None}
        pool = [by_run[run_id] for run_id in keep]  # the budget and fixed values are re-applied
    return HalvingResult(config.experiment_id, metric, spec.resource, tuple(rounds))
