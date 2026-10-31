"""Greedy feature selection scored by cross-validation.

Forward selection starts with no features and, at each step, adds the one column that most
improves the cross-validated score; it stops when the best addition improves the score by no
more than `tolerance`, or when `max_features` columns are chosen. Backward elimination starts
with every column and, at each step, drops the one whose removal gives the best score; it
continues while the score after dropping is no more than `tolerance` below the score before,
and stops at `min_features`.

Every candidate subset is scored on the same folds (`kfold_indices` with the config's seed, so
the comparison is fair), with the same model, parameters, preprocessing and model seeds, and
the mean of the chosen metric over the folds is its score. Ties go to the column that comes
first in the feature order: forward adds it, backward drops it. The first forward step always
adds the best single column, since there is no score to beat yet; `max_features` applies to
forward selection and `min_features` to backward elimination. Nothing is stored.

The score of the final subset is *optimistic*: the subset was chosen because it scored well on
these very folds, and the more columns there are to choose between, the larger the effect. Use
`nested` to estimate the honest score of a procedure that includes a choice, or check the chosen
subset on data the search never saw.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compare import higher_is_better
from .config import ExperimentConfig
from .datasets import MAX_FOLDS, Table, kfold_indices
from .errors import SweepError
from .models import model_kind
from .runner import (
    _check_metrics,
    _check_split_for_metrics,
    _evaluate,
    _feature_names,
    _features_and_labels,
    _prepare_folds,
    build_model,
)
from .sweep import _read_json

DIRECTIONS = ("forward", "backward")
MAX_EVALUATIONS = 20_000  # subsets scored x folds


@dataclass(frozen=True)
class SelectionStep:
    """One accepted move: `feature` was added (forward) or dropped (backward)."""

    feature: str
    score: float  # the score of the subset after the move
    tried: dict[str, float]  # every column tried at this step and the score it would have given


@dataclass(frozen=True)
class SelectionResult:
    experiment_id: str
    direction: str
    metric: str
    higher_is_better: bool
    folds: int
    features: tuple[str, ...]  # every column that was considered, in order
    start_score: float | None  # the starting subset's score: all columns (backward) or None
    steps: tuple[SelectionStep, ...]
    selected: tuple[str, ...]  # in feature order
    score: float | None  # the selected subset's score; None if forward chose nothing
    all_score: float  # the score with every column, for comparison


@dataclass(frozen=True)
class SelectionSpec:
    config: ExperimentConfig
    folds: int
    direction: str
    max_features: int | None
    min_features: int
    tolerance: float


def _whole(value: object, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise SweepError(f"{name} must be a whole number from {low} to {high}, got {value!r}")
    return value


def _tolerance(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SweepError(f"tolerance must be a number, got {value!r}")
    if not math.isfinite(value) or value < 0:
        raise SweepError(f"tolerance must be zero or more, got {value!r}")
    return float(value)


def load_selection(path: str | Path) -> SelectionSpec:
    """Read `{"config": {...}, "folds": K, "direction": "forward"|"backward", ...}`."""
    document = _read_json(Path(path), "selection file")
    if not isinstance(document, dict):
        raise SweepError("a selection file must be a JSON object")
    allowed = {"config", "folds", "direction", "max_features", "min_features", "tolerance"}
    extra = sorted(set(document) - allowed)
    if extra:
        raise SweepError(f"unknown key(s) in selection file: {', '.join(extra)}")
    for key in ("config", "folds"):
        if key not in document:
            raise SweepError(f'a selection file needs "{key}"')
    config = ExperimentConfig.from_dict(document["config"])
    max_features = document.get("max_features")
    return SelectionSpec(
        config,
        _whole(document["folds"], '"folds"', 2, MAX_FOLDS),
        _direction(document.get("direction", "forward")),
        None if max_features is None else _whole(max_features, '"max_features"', 1, 10_000),
        _whole(document.get("min_features", 1), '"min_features"', 1, 10_000),
        _tolerance(document.get("tolerance", 0.0)),
    )


def _direction(value: object) -> str:
    if value not in DIRECTIONS:
        raise SweepError(f"direction must be one of {DIRECTIONS}, got {value!r}")
    assert isinstance(value, str)
    return value


def run_selection(
    config: ExperimentConfig,
    table: Table,
    label_column: str,
    *,
    folds: int,
    direction: str = "forward",
    feature_columns: Sequence[str] | None = None,
    max_features: int | None = None,
    min_features: int = 1,
    tolerance: float = 0.0,
    rank_by: str | None = None,
) -> SelectionResult:
    """Select feature columns for `config`'s model by cross-validated score."""
    if config.cv_folds is not None:
        raise SweepError("the config must not set cv_folds; selection takes its own folds")
    folds = _whole(folds, "folds", 2, MAX_FOLDS)
    direction = _direction(direction)
    tolerance = _tolerance(tolerance)
    names = _feature_names(table, label_column, feature_columns)
    if max_features is not None:
        max_features = _whole(max_features, "max_features", 1, len(names))
    min_features = _whole(min_features, "min_features", 1, len(names))
    if max_features is not None and min_features > max_features:
        raise SweepError("min_features must not exceed max_features")
    metric = rank_by if rank_by is not None else config.metrics[0]
    if metric not in config.metrics:
        raise SweepError(f"cannot select by {metric!r}; the config reports {list(config.metrics)}")
    larger = higher_is_better(metric)
    kind = model_kind(config.model)
    _check_metrics(config.metrics, kind)
    if config.time_ordered:
        raise SweepError("time_ordered is not supported here; these folds are shuffled")
    if kind == "regression" and config.stratify:
        raise SweepError("stratify applies to classification; regression has no classes")
    build_model(config)  # a bad model or parameter fails before anything is scored

    n_rows = len(table)
    width = len(names)
    worst_case = (width * (width + 1) // 2 + 1) * folds
    if worst_case > MAX_EVALUATIONS:
        raise SweepError(
            f"up to {worst_case} model fits would be needed; the limit is {MAX_EVALUATIONS}"
        )
    _, X_all, cells_all, y = _features_and_labels(
        table, label_column, names, kind == "regression", config
    )
    splits = kfold_indices(n_rows, folds, config.seed, y if config.stratify else None)
    if kind == "classification":
        for train_idx, test_idx in splits:
            _check_split_for_metrics(
                config.metrics, [y[i] for i in train_idx], [y[i] for i in test_idx]
            )

    position = {name: i for i, name in enumerate(names)}
    cache: dict[tuple[str, ...], float] = {}

    def score_of(subset: tuple[str, ...]) -> float:
        ordered = tuple(sorted(subset, key=position.__getitem__))
        if ordered in cache:
            return cache[ordered]
        columns = [position[c] for c in ordered]
        X = [[row[c] for c in columns] for row in X_all] if X_all else []
        cells = [[row[c] for c in columns] for row in cells_all] if cells_all is not None else None
        prepared = _prepare_folds(config, ordered, X, cells, y, splits)
        values = []
        for number, fold in enumerate(prepared):
            model, _ = build_model(config, number)
            values.append(_evaluate(config, model, fold).scores[metric])
        cache[ordered] = math.fsum(values) / len(values)
        return cache[ordered]

    def better(a: float, b: float) -> bool:
        return a > b if larger else a < b

    all_score = score_of(tuple(names))
    steps: list[SelectionStep] = []
    if direction == "forward":
        chosen: list[str] = []
        current: float | None = None
        limit = max_features if max_features is not None else width
        while len(chosen) < limit:
            tried = {c: score_of((*chosen, c)) for c in names if c not in chosen}
            best = next(iter(tried))
            for column, value in tried.items():
                if better(value, tried[best]):
                    best = column
            gain = tried[best] - current if current is not None else None
            if gain is not None and not _exceeds(gain, tolerance, larger):
                break
            chosen.append(best)
            current = tried[best]
            steps.append(SelectionStep(best, current, tried))
        selected = tuple(c for c in names if c in chosen)
        return SelectionResult(
            config.experiment_id, direction, metric, larger, folds, tuple(names),
            None, tuple(steps), selected, current, all_score,
        )  # fmt: skip
    kept = list(names)
    current = all_score
    floor = max(min_features, 1)
    while len(kept) > floor:
        tried = {c: score_of(tuple(k for k in kept if k != c)) for c in kept}
        best = next(iter(tried))
        for column, value in tried.items():
            if better(value, tried[best]):
                best = column
        if not _within(tried[best] - current, tolerance, larger):
            break
        kept.remove(best)
        current = tried[best]
        steps.append(SelectionStep(best, current, tried))
    return SelectionResult(
        config.experiment_id, direction, metric, larger, folds, tuple(names),
        all_score, tuple(steps), tuple(kept), current, all_score,
    )  # fmt: skip


def _exceeds(change: float, tolerance: float, larger: bool) -> bool:
    """Whether a score moved in the good direction by more than `tolerance`."""
    return (change if larger else -change) > tolerance


def _within(change: float, tolerance: float, larger: bool) -> bool:
    """Whether a score got worse by no more than `tolerance` (or did not get worse)."""
    return (-change if larger else change) <= tolerance


def describe(result: SelectionResult) -> dict[str, Any]:
    """The result as plain data, for JSON output."""
    return {
        "experiment_id": result.experiment_id,
        "direction": result.direction,
        "metric": result.metric,
        "higher_is_better": result.higher_is_better,
        "folds": result.folds,
        "features": list(result.features),
        "all_score": result.all_score,
        "start_score": result.start_score,
        "selected": list(result.selected),
        "score": result.score,
        "steps": [{"feature": s.feature, "score": s.score, "tried": s.tried} for s in result.steps],
    }
