"""Permutation feature importance for a stored run.

How much does the model's score drop when one feature's values are shuffled among the test
rows? A feature the model relies on drops the score a lot; one it ignores changes nothing. The
model is rebuilt with `runner.refit` (splits, preprocessing and model seeds are deterministic),
checked against the predictions hash the run recorded, and then each feature of each test split
is shuffled `repeats` times with a generator derived from the seed, the split, the feature and
the repeat.

Importance is the score before minus the score after for metrics where higher is better, and
after minus before where lower is better, so a positive number always means "the shuffle hurt".
The spread is the population standard deviation over every shuffle in every split. Features are
the model's inputs after preprocessing, so a one-hot encoded column is one feature.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

from .compare import higher_is_better
from .datasets import Table, make_version
from .errors import AnalysisError
from .models.base import Classifier
from .runner import FittedFold, predictions_payload, refit, score_classifier, score_regressor
from .seeding import check_seed, rng_for
from .store import Store

MAX_REPEATS = 100


@dataclass(frozen=True)
class FeatureImportance:
    feature: str
    importance: float
    std: float


@dataclass(frozen=True)
class ImportanceResult:
    run_id: str
    metric: str
    higher_is_better: bool
    baseline: float  # the metric as the run recorded it (the mean over folds in cross-validation)
    repeats: int
    features: tuple[FeatureImportance, ...]  # most important first, ties by name


def _score(fitted: FittedFold, metric: str, test_X: list[list[float]]) -> float:
    model = fitted.model
    if isinstance(model, Classifier):
        return score_classifier([metric], model, test_X, fitted.fold.test_y, model.predict(test_X))[
            metric
        ]
    targets: list[float] = [float(v) for v in fitted.fold.test_y]
    return score_regressor([metric], targets, model.predict(test_X))[metric]


def permutation_importance(
    store: Store,
    run_id: str,
    table: Table,
    label_column: str,
    *,
    feature_columns: Sequence[str] | None = None,
    metric: str | None = None,
    repeats: int = 5,
    seed: int | None = None,
) -> ImportanceResult:
    """Importance of every input feature for one completed run, on the data the run used."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= MAX_REPEATS:
        raise AnalysisError(f"repeats must be a whole number from 1 to {MAX_REPEATS}")
    record = store.get_run(run_id)
    if record.status != "completed":
        raise AnalysisError(f"run {run_id} is {record.status}; only completed runs can be analysed")
    config = store.get_config(run_id)
    chosen = metric if metric is not None else config.metrics[0]
    if chosen not in config.metrics:
        raise AnalysisError(f"{chosen!r} is not one of the run's metrics {list(config.metrics)}")
    base_seed = check_seed(seed if seed is not None else config.seed)
    if make_version(config.dataset_id, table, label_column).version != record.dataset_version:
        raise AnalysisError(f"the data given is not the dataset version run {run_id} used")

    fitted = refit(config, table, label_column, feature_columns)
    expected_hash = record.tags.get("predictions_sha256")
    if expected_hash is None:
        raise AnalysisError(f"run {run_id} recorded no predictions hash, so it cannot be analysed")
    parts = [(f.fold.test_idx, f.fold.test_y, f.model.predict(f.fold.test_X)) for f in fitted]
    payload = predictions_payload(config.cv_folds is not None, parts)
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise AnalysisError(
            f"re-fitting run {run_id} does not give the predictions it recorded; the data, the "
            "feature columns, or the environment differ, so the importances would be about "
            "a different model"
        )
    recorded = store.final_metrics(run_id)[chosen]
    before = [_score(f, chosen, f.fold.test_X) for f in fitted]
    sign = 1.0 if higher_is_better(chosen) else -1.0

    names = fitted[0].fold.feature_names
    if any(f.fold.feature_names != names for f in fitted):  # e.g. a category only in some folds
        raise AnalysisError("the folds have different input features, so they cannot be combined")
    drops: dict[str, list[float]] = {name: [] for name in names}
    for number, part in enumerate(fitted):
        for column, name in enumerate(names):
            values = [row[column] for row in part.fold.test_X]
            for repeat in range(repeats):
                order = list(range(len(values)))
                rng_for(base_seed, "permutation", number, column, repeat).shuffle(order)
                shuffled = [
                    [*row[:column], values[order[i]], *row[column + 1 :]]
                    for i, row in enumerate(part.fold.test_X)
                ]
                drops[name].append(sign * (before[number] - _score(part, chosen, shuffled)))
    ranked = []
    for name, values in drops.items():
        mean = math.fsum(values) / len(values)
        std = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / len(values))
        ranked.append(FeatureImportance(name, mean, std))
    ranked.sort(key=lambda f: (-f.importance, f.feature))
    return ImportanceResult(
        run_id, chosen, higher_is_better(chosen), recorded, repeats, tuple(ranked)
    )
