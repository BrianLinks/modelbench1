"""Learning curves: how the score changes as the model sees more training rows.

For each fold of the config (the single split, or each cross-validation fold) the training rows
are put in a seeded order and the model is fitted on the first `fraction` of them, for each
fraction in turn, so every smaller training set is contained in the next. Each fit is scored on
the rows it was trained on and on the fold's untouched test rows. Training far below the test
score means the model has not learned enough to generalise; a large gap that does not close as
the training set grows means more data is the cure and, if both scores sit low and flat, more
data will not help.

Preprocessing is fitted once on the whole training split of each fold (as in a normal run), not
refitted on the smaller subsets. Nothing is stored; a curve is an analysis, not a run.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .config import ExperimentConfig
from .datasets import Table
from .errors import AnalysisError, ModelBenchError
from .models.base import Classifier, Regressor
from .runner import Fold, build_model, load_folds, score_classifier, score_regressor
from .seeding import rng_for

MAX_POINTS = 20
DEFAULT_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class CurvePoint:
    fraction: float
    n_train: int  # the largest training subset over the folds
    train_score: float  # mean over folds
    test_score: float  # mean over folds
    test_std: float  # population standard deviation over folds (0 with a single split)


@dataclass(frozen=True)
class LearningCurve:
    metric: str
    folds: int
    points: tuple[CurvePoint, ...]


def check_fractions(fractions: Sequence[float]) -> tuple[float, ...]:
    values = tuple(fractions)
    if not 1 <= len(values) <= MAX_POINTS:
        raise AnalysisError(f"give between 1 and {MAX_POINTS} fractions")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AnalysisError(f"fractions must be numbers, got {value!r}")
        if not math.isfinite(value) or not 0 < value <= 1:
            raise AnalysisError(f"each fraction must be in (0, 1], got {value}")
    if any(b <= a for a, b in zip(values, values[1:], strict=False)):
        raise AnalysisError("fractions must be strictly increasing")
    return tuple(float(v) for v in values)


def _score(model: Classifier | Regressor, X: list[list[float]], y: list[Any], metric: str) -> float:
    if isinstance(model, Classifier):
        labels: list[Any] = list(y)
        return score_classifier([metric], model, X, labels, model.predict(X))[metric]
    targets = [float(v) for v in y]
    return score_regressor([metric], targets, model.predict(X))[metric]


def _subset(fold: Fold, order: list[int], fraction: float) -> tuple[list[list[float]], list[Any]]:
    size = max(1, round(fraction * len(order)))
    chosen = order[:size]
    return [fold.train_X[i] for i in chosen], [fold.train_y[i] for i in chosen]


def learning_curve(
    config: ExperimentConfig,
    table: Table,
    label_column: str,
    *,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    metric: str | None = None,
    feature_columns: Sequence[str] | None = None,
) -> LearningCurve:
    """Train and test scores of `config` at each fraction of the training rows."""
    steps = check_fractions(fractions)
    chosen = metric if metric is not None else config.metrics[0]
    if chosen not in config.metrics:
        raise AnalysisError(f"{chosen!r} is not one of the config's metrics {list(config.metrics)}")
    folds = load_folds(config, table, label_column, feature_columns)
    cv = config.cv_folds is not None
    train_scores: list[list[float]] = [[] for _ in steps]
    test_scores: list[list[float]] = [[] for _ in steps]
    sizes = [0] * len(steps)
    for number, fold in enumerate(folds):
        order = list(range(len(fold.train_y)))
        rng_for(config.seed, "learning-curve", number).shuffle(order)
        for position, fraction in enumerate(steps):
            X, y = _subset(fold, order, fraction)
            model, _ = build_model(config, number if cv else None)
            try:
                model.fit(X, y)
                train_scores[position].append(_score(model, X, y, chosen))
                test_scores[position].append(_score(model, fold.test_X, fold.test_y, chosen))
            except ModelBenchError as exc:
                raise AnalysisError(
                    f"at fraction {fraction} ({len(y)} training rows): {exc}"
                ) from exc
            sizes[position] = max(sizes[position], len(y))
    points = []
    for position, fraction in enumerate(steps):
        test = test_scores[position]
        mean = math.fsum(test) / len(test)
        std = math.sqrt(math.fsum((v - mean) ** 2 for v in test) / len(test))
        train = train_scores[position]
        points.append(
            CurvePoint(fraction, sizes[position], math.fsum(train) / len(train), mean, std)
        )
    return LearningCurve(chosen, len(folds), tuple(points))
