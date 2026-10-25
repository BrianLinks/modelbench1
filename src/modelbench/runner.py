"""Run one experiment: split a dataset, fit a model, score it, and record everything.

Everything that can be checked before work starts is checked first, so bad input raises without
leaving a run behind. Once a run exists, any failure marks it `failed` with the error recorded
and is then re-raised. A hard interruption (for example Ctrl-C or a killed process) leaves the
run in `running`.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import confusion
from . import metrics as m
from .config import ExperimentConfig
from .datasets import DatasetVersion, Table, kfold_indices, make_version, split_indices
from .environment import environment_fingerprint
from .errors import RunnerError
from .models import REGISTRY, REGRESSORS, create_model, create_regressor, model_kind
from .models.base import Classifier, Label, Regressor
from .seeding import derive_seed
from .store import Store

CLASSIFICATION_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "log_loss",
    "precision_micro",
    "precision_weighted",
    "recall_micro",
    "recall_weighted",
    "f1_micro",
    "f1_weighted",
    "balanced_accuracy",
    "mcc",
    "average_precision",
)
SCORE_NEEDS_TWO_CLASSES = ("roc_auc", "average_precision")
REGRESSION_METRICS = ("mae", "mse", "rmse", "r2")
REUSE_TAGS = ("n_train", "n_test", "predictions_sha256")
_NUMBER_RE = re.compile(r"[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?")  # ASCII digits only


@dataclass(frozen=True)
class RunResult:
    run_id: str
    metrics: dict[str, float]
    n_train: int
    n_test: int
    predictions_sha256: str
    reused: bool = False


def run_experiment(
    store: Store,
    config: ExperimentConfig,
    table: Table,
    label_column: str,
    *,
    feature_columns: Sequence[str] | None = None,
    environment: Mapping[str, str | None] | None = None,
    artifact_dir: str | Path | None = None,
    reuse_existing: bool = False,
) -> RunResult:
    """Run `config` on `table` and record it in `store`.

    Features are the numeric columns in `feature_columns` (default: every column except the
    label). Precision, recall, and F1 are macro-averaged over every label that appears in the
    test split's true or predicted values, so a label the model predicts but the test split never
    contains counts as zero. ROC-AUC needs exactly two training labels and both labels present in
    the test split; the second label (in sorted order) is scored as positive. With
    `reuse_existing=True`, a completed run of the same experiment with identical settings and
    dataset version is returned instead of running again.
    """
    kind = model_kind(config.model)
    _check_metrics(config.metrics, kind)
    if kind == "regression" and config.stratify:
        raise RunnerError("stratify applies to classification; regression has no classes")
    X, y = _features_and_labels(
        table, label_column, feature_columns, numeric_target=kind == "regression"
    )
    dataset = make_version(config.dataset_id, table, label_column)
    labels_for_split = y if config.stratify else None
    if config.cv_folds is None:
        splits = [split_indices(len(table), config.test_fraction, config.seed, labels_for_split)]
    else:
        splits = kfold_indices(len(table), config.cv_folds, config.seed, labels_for_split)
    for train_idx, test_idx in splits:  # every fold must be scoreable before a run exists
        _check_split_for_metrics(
            config.metrics, [y[i] for i in train_idx], [y[i] for i in test_idx]
        )  # no-op for regression
    _build_model(config)  # reject unknown models and parameters before a run exists

    if reuse_existing:
        earlier = _find_reusable(store, config, dataset)
        if earlier is not None:
            return earlier

    env = dict(environment) if environment is not None else environment_fingerprint()
    run_id = store.create_run(config, dataset, env)
    try:
        return _execute(store, run_id, config, X, y, splits, artifact_dir)
    except Exception as exc:
        store.fail_run(run_id, f"{type(exc).__name__}: {exc}")
        raise


@dataclass(frozen=True)
class _Evaluation:
    scores: dict[str, float]
    test_idx: list[int]
    y_true: list[Any]
    y_pred: list[Any]


def _evaluate(
    config: ExperimentConfig,
    model: Classifier | Regressor,
    X: list[list[float]],
    y: list[Label],
    train_idx: list[int],
    test_idx: list[int],
) -> _Evaluation:
    train_y: list[Any] = [y[i] for i in train_idx]  # labels, or floats for a regressor
    model.fit([X[i] for i in train_idx], train_y)
    test_X = [X[i] for i in test_idx]
    y_true: list[Any] = [y[i] for i in test_idx]
    y_pred: list[Any] = list(model.predict(test_X))
    if isinstance(model, Regressor):
        scores = _score_regression(config.metrics, y_true, y_pred)
    else:
        scores = _score(config.metrics, model, test_X, y_true, y_pred)
    return _Evaluation(scores, test_idx, y_true, y_pred)


def _mean_and_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and population standard deviation (divide by the number of folds)."""
    mean = math.fsum(values) / len(values)
    return mean, math.sqrt(math.fsum((v - mean) ** 2 for v in values) / len(values))


def _execute(
    store: Store,
    run_id: str,
    config: ExperimentConfig,
    X: list[list[float]],
    y: list[Label],
    splits: list[tuple[list[int], list[int]]],
    artifact_dir: str | Path | None,
) -> RunResult:
    cv = config.cv_folds is not None
    evaluations: list[_Evaluation] = []
    model: Classifier | Regressor | None = None
    model_seed: int | None = None
    for fold, (train_idx, test_idx) in enumerate(splits):
        model, seed = _build_model(config, fold if cv else None)
        model_seed = seed if not cv else None
        evaluations.append(_evaluate(config, model, X, y, train_idx, test_idx))
    assert model is not None  # a run always has at least one split

    y_true: list[Any] = [v for e in evaluations for v in e.y_true]
    y_pred: list[Any] = [v for e in evaluations for v in e.y_pred]
    if cv:
        payload_data: dict[str, Any] = {
            "folds": [
                {"test_indices": e.test_idx, "y_true": e.y_true, "y_pred": e.y_pred}
                for e in evaluations
            ]
        }
    else:
        only = evaluations[0]
        payload_data = {"test_indices": only.test_idx, "y_true": only.y_true, "y_pred": only.y_pred}
    payload = json.dumps(payload_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()

    scores: dict[str, float]
    if cv:
        scores = {}
        for fold, evaluation in enumerate(evaluations):
            store.log_metrics(
                run_id, {f"fold/{name}": v for name, v in evaluation.scores.items()}, step=fold
            )
        for name in config.metrics:
            mean, std = _mean_and_std([e.scores[name] for e in evaluations])
            scores[name], scores[f"{name}_std"] = mean, std
        store.log_metrics(run_id, scores)
        n_train = max(len(train) for train, _ in splits)
        n_test = len(y_true)  # every row is tested exactly once
        store.set_tag(run_id, "cv_folds", str(len(splits)))
    else:
        scores = evaluations[0].scores
        store.log_metrics(run_id, scores)
        n_train, n_test = len(splits[0][0]), len(splits[0][1])
    store.set_tag(run_id, "n_train", str(n_train))
    store.set_tag(run_id, "n_test", str(n_test))
    store.set_tag(run_id, "predictions_sha256", digest)
    if not isinstance(model, Regressor):
        known = list(model.classes_) if not cv else []
        labels = m.ordered_labels(known, y_true, y_pred)
        matrix_tag = confusion.to_tag(m.confusion_matrix(y_true, y_pred, labels))
        if matrix_tag is not None:
            store.set_tag(run_id, confusion.TAG, matrix_tag)
    if model_seed is not None:
        store.set_tag(run_id, "model_seed", str(model_seed))
    if artifact_dir is not None:
        folder = Path(artifact_dir) / run_id
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "predictions.json"
        target.write_bytes(payload)
        store.add_artifact(
            run_id, "predictions", "predictions", digest, len(payload), path=str(target)
        )
    store.finish_run(run_id)
    return RunResult(run_id, scores, n_train, n_test, digest)


# -- checks made before any run is created ---------------------------------------------------


def _check_metrics(names: Sequence[str], kind: str = "classification") -> None:
    allowed, other = (
        (CLASSIFICATION_METRICS, REGRESSION_METRICS)
        if kind == "classification"
        else (REGRESSION_METRICS, CLASSIFICATION_METRICS)
    )
    wrong_kind = [n for n in names if n in other]
    if wrong_kind:
        raise RunnerError(f"metric(s) {wrong_kind} do not apply to a {kind} model")
    unknown = [n for n in names if n not in allowed]
    if unknown:
        raise RunnerError(f"unsupported metric(s) {unknown}")


def _features_and_labels(
    table: Table,
    label_column: str,
    feature_columns: Sequence[str] | None,
    numeric_target: bool = False,
) -> tuple[list[list[float]], list[Label]]:
    if label_column not in table.columns:
        raise RunnerError(f"label column {label_column!r} is not in the dataset")
    if feature_columns is None:
        names = [c for c in table.columns if c != label_column]
    else:
        names = list(feature_columns)
        missing = [c for c in names if c not in table.columns]
        if missing:
            raise RunnerError(f"feature column(s) not in the dataset: {missing}")
        if label_column in names:
            raise RunnerError("the label column cannot also be a feature")
        if len(set(names)) != len(names):
            raise RunnerError("feature columns must not repeat")
    if not names:
        raise RunnerError("no feature columns to train on")
    positions = [table.columns.index(c) for c in names]
    X: list[list[float]] = []
    for row_number, row in enumerate(table.rows, start=1):
        values: list[float] = []
        for name, position in zip(names, positions, strict=True):
            cell = row[position].strip()
            if _NUMBER_RE.fullmatch(cell) is None or not math.isfinite(float(cell)):
                raise RunnerError(
                    f"row {row_number}, column {name!r}: {row[position]!r} is not a finite number"
                )
            values.append(float(cell))
        X.append(values)
    label_position = table.columns.index(label_column)
    y: list[Label] = []
    for row_number, row in enumerate(table.rows, start=1):
        label = row[label_position]
        if label.strip() == "":
            raise RunnerError(f"row {row_number}: label is empty")
        if numeric_target:
            text = label.strip()
            if _NUMBER_RE.fullmatch(text) is None or not math.isfinite(float(text)):
                raise RunnerError(f"row {row_number}: target {label!r} is not a finite number")
            y.append(float(text))
        else:
            y.append(label)
    return X, y


def _check_split_for_metrics(
    names: Sequence[str], train_labels: Sequence[Label], test_labels: Sequence[Label]
) -> None:
    for needs_two in SCORE_NEEDS_TWO_CLASSES:
        if needs_two not in names:
            continue
        if len(set(train_labels)) != 2:
            raise RunnerError(
                f"{needs_two} needs exactly two classes in the training split, "
                f"found {len(set(train_labels))}"
            )
        if len(set(test_labels)) != 2:
            raise RunnerError(
                f"{needs_two} needs both classes in the test split, found {len(set(test_labels))}"
            )
    if "log_loss" in names:
        unseen = sorted(set(test_labels) - set(train_labels), key=str)
        if unseen:
            raise RunnerError(f"log_loss needs every test label in training; unseen: {unseen}")


def _build_model(
    config: ExperimentConfig, fold: int | None = None
) -> tuple[Classifier | Regressor, int | None]:
    params = dict(config.params)
    cls = REGISTRY.get(config.model) or REGRESSORS.get(config.model)
    model_seed: int | None = None
    if cls is not None and "seed" in inspect.signature(cls.__init__).parameters:
        if "seed" not in params:
            purpose = "model" if fold is None else f"model-fold-{fold}"
            params["seed"] = derive_seed(config.seed, purpose)
        model_seed = params["seed"]
    if config.model in REGRESSORS:
        return create_regressor(config.model, **params), model_seed
    return create_model(config.model, **params), model_seed


# -- scoring and reuse -----------------------------------------------------------------------


def _score(
    names: Sequence[str],
    model: Classifier,
    test_X: list[list[float]],
    y_true: list[Label],
    y_pred: list[Label],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    proba: list[list[float]] | None = None
    for name in names:
        if name == "accuracy":
            scores[name] = m.accuracy(y_true, y_pred)
        elif name in ("precision", "recall", "f1"):
            scores[name] = getattr(m.precision_recall_f1(y_true, y_pred, average="macro"), name)
        elif name == "balanced_accuracy":
            scores[name] = m.balanced_accuracy(y_true, y_pred)
        elif name == "mcc":
            scores[name] = m.matthews_corrcoef(y_true, y_pred)
        elif name.rsplit("_", 1)[-1] in ("micro", "weighted") and name != "log_loss":
            base, average = name.rsplit("_", 1)
            scores[name] = getattr(m.precision_recall_f1(y_true, y_pred, average=average), base)
        else:
            proba = proba if proba is not None else model.predict_proba(test_X)
            if name in SCORE_NEEDS_TWO_CLASSES:
                _, positive = model.classes_
                column = [row[1] for row in proba]
                function = m.roc_auc if name == "roc_auc" else m.average_precision
                scores[name] = function(y_true, column, positive_label=positive)
            else:
                scores[name] = m.log_loss(y_true, proba, labels=model.classes_)
    return scores


def _score_regression(
    names: Sequence[str], y_true: list[float], y_pred: list[float]
) -> dict[str, float]:
    functions = {"mae": m.mae, "mse": m.mse, "rmse": m.rmse, "r2": m.r2}
    return {name: functions[name](y_true, y_pred) for name in names}


def _find_reusable(
    store: Store, config: ExperimentConfig, dataset: DatasetVersion
) -> RunResult | None:
    for run_id in reversed(store.find_runs_by_config(config.config_hash(), dataset.version)):
        record = store.get_run(run_id)
        if (
            record.status == "completed"
            and record.experiment_id == config.experiment_id
            and all(tag in record.tags for tag in REUSE_TAGS)
        ):
            return RunResult(
                run_id,
                {k: v for k, v in store.final_metrics(run_id).items() if not k.startswith("fold/")},
                int(record.tags["n_train"]),
                int(record.tags["n_test"]),
                record.tags["predictions_sha256"],
                reused=True,
            )
    return None
