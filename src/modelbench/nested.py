"""Nested cross-validation: choose parameters and estimate the score of that choice honestly.

Picking the best of many parameter sets by cross-validated score and then reporting that score
flatters the winner: it was chosen *because* it scored well on those very folds. Nested
cross-validation keeps the two jobs apart:

- the outer loop splits the table into `outer_folds` train/test pairs;
- inside each outer training set an inner `inner_folds`-fold cross-validation scores every
  candidate parameter set, and the best by the selection metric is chosen (ties go to the
  earlier candidate);
- that choice is refitted on the whole outer training set and scored once on the outer test fold,
  which no choice ever saw.

The mean of the outer test scores estimates how well *the whole procedure* (search, then fit)
will do on new data. The mean of the chosen candidates' inner scores is what plain selection
would have reported; its excess over the outer mean is the optimism. Nothing here is stored:
splits, seeds and fits are deterministic, so the same inputs always give the same result.

Models that take a `seed` get one derived from the config's seed and a fold number: the refit
for outer fold `i` uses `i`, as cross-validation does, and inner fit `j` of outer fold `i` uses
`(i + 1) * 1000 + j`, so no two fits share a stream.

Candidates come from a grid (`expand_grid`) or from `trials` seeded draws from a search space
(`sample_params`, as in `search`). Preprocessing is fitted on each training split only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .compare import higher_is_better
from .config import ExperimentConfig, expand_grid
from .datasets import MAX_FOLDS, Table, kfold_indices
from .errors import ModelBenchError, SweepError
from .models import model_kind
from .runner import (
    Fold,
    _check_metrics,
    _check_split_for_metrics,
    _evaluate,
    _features_and_labels,
    _mean_and_std,
    _prepare_folds,
    build_model,
)
from .seeding import derive_seed
from .sweep import MAX_TRIALS, _read_json, check_space, sample_params

MAX_CANDIDATES = 1000
MAX_FITS = 20_000  # candidates x inner folds x outer folds, plus the outer refits


@dataclass(frozen=True)
class NestedSpec:
    config: ExperimentConfig
    grid: dict[str, list[Any]] | None
    space: dict[str, Any] | None
    trials: int | None
    outer_folds: int
    inner_folds: int


@dataclass(frozen=True)
class OuterFold:
    number: int  # from 0
    train_size: int
    test_size: int
    chosen: int  # index into NestedResult.candidates
    inner_scores: tuple[float | None, ...]  # per candidate: mean over inner folds, None if failed
    test_scores: dict[str, float]  # every configured metric on the outer test fold

    @property
    def inner_score(self) -> float:
        score = self.inner_scores[self.chosen]
        assert score is not None
        return score


@dataclass(frozen=True)
class NestedResult:
    experiment_id: str
    metric: str
    higher_is_better: bool
    candidates: tuple[dict[str, Any], ...]
    outer: tuple[OuterFold, ...]

    def outer_scores(self, metric: str | None = None) -> list[float]:
        name = metric or self.metric
        return [fold.test_scores[name] for fold in self.outer]

    @property
    def estimate(self) -> tuple[float, float]:
        """Mean and population standard deviation of the outer test scores of the metric."""
        return _mean_and_std(self.outer_scores())

    @property
    def selection_score(self) -> float:
        """The mean inner score of the chosen candidates: what plain selection would report."""
        return math.fsum(f.inner_score for f in self.outer) / len(self.outer)

    @property
    def optimism(self) -> float:
        """How much plain selection overstated the score, in the metric's own unit. Positive means
        overstated, whichever way the metric points; near zero or negative means little bias."""
        gap = self.selection_score - self.estimate[0]
        return gap if self.higher_is_better else -gap

    def choice_counts(self) -> list[int]:
        counts = [0] * len(self.candidates)
        for fold in self.outer:
            counts[fold.chosen] += 1
        return counts

    def metric_summary(self) -> dict[str, tuple[float, float]]:
        """`(mean, std)` over the outer folds of every configured metric."""
        names = list(self.outer[0].test_scores)
        return {n: _mean_and_std(self.outer_scores(n)) for n in names}


def _count(value: object, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise SweepError(f"{name} must be a whole number from {low} to {high}, got {value!r}")
    return value


def load_nested(path: str | Path) -> NestedSpec:
    """Read `{"config": {...}, "grid": {...} | "space": {...} + "trials": N, "outer_folds": K,
    "inner_folds": K}`."""
    document = _read_json(Path(path), "nested file")
    if not isinstance(document, dict):
        raise SweepError("a nested file must be a JSON object")
    allowed = {"config", "grid", "space", "trials", "outer_folds", "inner_folds"}
    extra = sorted(set(document) - allowed)
    if extra:
        raise SweepError(f"unknown key(s) in nested file: {', '.join(extra)}")
    for key in ("config", "outer_folds", "inner_folds"):
        if key not in document:
            raise SweepError(f'a nested file needs "{key}"')
    has_grid, has_space = "grid" in document, "space" in document
    if has_grid == has_space:
        raise SweepError('a nested file needs exactly one of "grid" or "space" (with "trials")')
    grid: dict[str, list[Any]] | None = None
    space: dict[str, Any] | None = None
    trials: int | None = None
    if has_grid:
        if "trials" in document:
            raise SweepError('"trials" goes with "space", not "grid"')
        raw = document["grid"]
        if not isinstance(raw, dict) or not raw:
            raise SweepError('"grid" must be an object with at least one parameter')
        for name, values in raw.items():
            if not isinstance(values, list) or not values:
                raise SweepError(f"grid[{name!r}] must be a non-empty list")
        grid = dict(raw)
    else:
        if "trials" not in document:
            raise SweepError('"space" needs "trials"')
        trials = _count(document["trials"], '"trials"', 1, MAX_TRIALS)
        check_space(document["space"])
        space = dict(document["space"])
    config = ExperimentConfig.from_dict(document["config"])
    if config.cv_folds is not None:
        raise SweepError(
            'the config must not set "cv_folds"; nested cross-validation sets the folds'
        )
    return NestedSpec(
        config,
        grid,
        space,
        trials,
        _count(document["outer_folds"], '"outer_folds"', 2, MAX_FOLDS),
        _count(document["inner_folds"], '"inner_folds"', 2, MAX_FOLDS),
    )


def candidate_params(
    config: ExperimentConfig,
    grid: Mapping[str, Sequence[Any]] | None,
    space: Mapping[str, Any] | None,
    trials: int | None,
) -> list[dict[str, Any]]:
    """The parameter sets to choose between, in a fixed order and without repeats."""
    if (grid is None) == (space is None):
        raise SweepError("give exactly one of a grid or a search space")
    if grid is not None:
        found = expand_grid(config.params, grid)
    else:
        assert space is not None and trials is not None
        checked = check_space(space)
        clash = sorted(set(checked) & set(config.params))
        if clash:
            raise SweepError(f"parameters in both the space and the config: {', '.join(clash)}")
        found = []
        seen: set[str] = set()
        for trial in range(_count(trials, "trials", 1, MAX_TRIALS)):
            params = {**config.params, **sample_params(checked, config.seed, trial)}
            key = json.dumps(params, sort_keys=True)
            if key not in seen:
                seen.add(key)
                found.append(params)
    if len(found) > MAX_CANDIDATES:
        raise SweepError(f"{len(found)} candidates; the limit is {MAX_CANDIDATES}")
    return found


def run_nested(
    config: ExperimentConfig,
    candidates: Sequence[Mapping[str, Any]],
    table: Table,
    label_column: str,
    *,
    outer_folds: int,
    inner_folds: int,
    feature_columns: Sequence[str] | None = None,
    rank_by: str | None = None,
) -> NestedResult:
    """Run nested cross-validation of `candidates` (parameter sets) for `config`'s model."""
    if config.cv_folds is not None:
        raise SweepError("the config must not set cv_folds")
    outer_folds = _count(outer_folds, "outer_folds", 2, MAX_FOLDS)
    inner_folds = _count(inner_folds, "inner_folds", 2, MAX_FOLDS)
    if not candidates:
        raise SweepError("there are no candidates")
    if len(candidates) > MAX_CANDIDATES:
        raise SweepError(f"{len(candidates)} candidates; the limit is {MAX_CANDIDATES}")
    fits = outer_folds * (len(candidates) * inner_folds + 1)
    if fits > MAX_FITS:
        raise SweepError(f"{fits} model fits are needed; the limit is {MAX_FITS}")
    metric = rank_by if rank_by is not None else config.metrics[0]
    if metric not in config.metrics:
        raise SweepError(f"cannot select by {metric!r}; the config reports {list(config.metrics)}")
    larger = higher_is_better(metric)

    kind = model_kind(config.model)
    _check_metrics(config.metrics, kind)
    if kind == "regression" and config.stratify:
        raise SweepError("stratify applies to classification; regression has no classes")
    variants = [replace(config, params=dict(p)) for p in candidates]
    for variant in variants:  # a bad parameter set is an error now, not a skipped candidate
        build_model(variant)

    names, X, cells, y = _features_and_labels(
        table, label_column, feature_columns, kind == "regression", config
    )
    labels = y if config.stratify else None
    n = len(y)
    outer_splits = kfold_indices(n, outer_folds, config.seed, labels)

    plans: list[tuple[list[tuple[list[int], list[int]]], list[int]]] = []
    for number, (train_idx, _) in enumerate(outer_splits):
        if len(train_idx) < inner_folds:
            raise SweepError(
                f"outer training set {number} has {len(train_idx)} rows, fewer than "
                f"inner_folds={inner_folds}"
            )
        inner = kfold_indices(
            len(train_idx),
            inner_folds,
            derive_seed(config.seed, "nested-inner", number),
            [y[i] for i in train_idx] if config.stratify else None,
        )
        mapped = [([train_idx[p] for p in tr], [train_idx[p] for p in te]) for tr, te in inner]
        plans.append((mapped, train_idx))
    if kind == "classification":  # every split must be scoreable before any model is fitted
        for (train_idx, test_idx), (mapped, _) in zip(outer_splits, plans, strict=True):
            _check_split_for_metrics(
                config.metrics, [y[i] for i in train_idx], [y[i] for i in test_idx]
            )
            for tr, te in mapped:
                _check_split_for_metrics(config.metrics, [y[i] for i in tr], [y[i] for i in te])

    outer_results: list[OuterFold] = []
    for number, (outer_split, (mapped, _)) in enumerate(zip(outer_splits, plans, strict=True)):
        inner_prepared = _prepare_folds(config, names, X, cells, y, mapped)
        inner_scores: list[float | None] = []
        for variant in variants:
            inner_scores.append(_inner_mean(variant, inner_prepared, number, metric))
        usable = [i for i, s in enumerate(inner_scores) if s is not None]
        if not usable:
            raise SweepError(f"every candidate failed in the inner folds of outer fold {number}")
        chosen = usable[0]
        for i in usable[1:]:
            a, b = inner_scores[i], inner_scores[chosen]
            assert a is not None and b is not None
            if (a > b) if larger else (a < b):
                chosen = i
        fold = _prepare_folds(config, names, X, cells, y, [outer_split])[0]
        model, _ = build_model(variants[chosen], number)
        scores = _evaluate(variants[chosen], model, fold).scores
        outer_results.append(
            OuterFold(
                number,
                len(fold.train_y),
                len(fold.test_y),
                chosen,
                tuple(inner_scores),
                scores,
            )
        )
    return NestedResult(
        config.experiment_id,
        metric,
        larger,
        tuple(dict(v.params) for v in variants),
        tuple(outer_results),
    )


def _inner_mean(
    variant: ExperimentConfig, folds: Sequence[Fold], outer_number: int, metric: str
) -> float | None:
    """A candidate's mean score over the inner folds, or None if it cannot be fitted on them."""
    values: list[float] = []
    for inner_number, fold in enumerate(folds):
        model, _ = build_model(variant, (outer_number + 1) * 1000 + inner_number)
        try:
            values.append(_evaluate(variant, model, fold).scores[metric])
        except ModelBenchError:
            return None
    return math.fsum(values) / len(values)
