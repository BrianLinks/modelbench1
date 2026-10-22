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

from . import metrics as m
from .config import ExperimentConfig
from .datasets import DatasetVersion, Table, make_version, split_indices
from .environment import environment_fingerprint
from .errors import RunnerError
from .models import REGISTRY, create_model
from .models.base import Classifier, Label
from .seeding import derive_seed
from .store import Store

CLASSIFICATION_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss")
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
    _check_metrics(config.metrics)
    X, y = _features_and_labels(table, label_column, feature_columns)
    dataset = make_version(config.dataset_id, table, label_column)
    train_idx, test_idx = split_indices(
        len(table), config.test_fraction, config.seed, y if config.stratify else None
    )
    train_labels = [y[i] for i in train_idx]
    test_labels = [y[i] for i in test_idx]
    _check_split_for_metrics(config.metrics, train_labels, test_labels)
    model, model_seed = _build_model(config)

    if reuse_existing:
        earlier = _find_reusable(store, config, dataset)
        if earlier is not None:
            return earlier

    env = dict(environment) if environment is not None else environment_fingerprint()
    run_id = store.create_run(config, dataset, env)
    try:
        return _execute(
            store, run_id, config, model, model_seed, X, y, train_idx, test_idx, artifact_dir
        )
    except Exception as exc:
        store.fail_run(run_id, f"{type(exc).__name__}: {exc}")
        raise


def _execute(
    store: Store,
    run_id: str,
    config: ExperimentConfig,
    model: Classifier,
    model_seed: int | None,
    X: list[list[float]],
    y: list[Label],
    train_idx: list[int],
    test_idx: list[int],
    artifact_dir: str | Path | None,
) -> RunResult:
    model.fit([X[i] for i in train_idx], [y[i] for i in train_idx])
    test_X = [X[i] for i in test_idx]
    y_true = [y[i] for i in test_idx]
    y_pred = model.predict(test_X)
    scores = _score(config.metrics, model, test_X, y_true, y_pred)

    payload = json.dumps(
        {"test_indices": test_idx, "y_true": y_true, "y_pred": y_pred},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()

    store.log_metrics(run_id, scores)
    store.set_tag(run_id, "n_train", str(len(train_idx)))
    store.set_tag(run_id, "n_test", str(len(test_idx)))
    store.set_tag(run_id, "predictions_sha256", digest)
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
    return RunResult(run_id, scores, len(train_idx), len(test_idx), digest)


# -- checks made before any run is created ---------------------------------------------------


def _check_metrics(names: Sequence[str]) -> None:
    unsupported = [n for n in names if n in REGRESSION_METRICS]
    if unsupported:
        raise RunnerError(
            f"metric(s) {unsupported} apply to regression, but only classification models exist"
        )
    unknown = [n for n in names if n not in CLASSIFICATION_METRICS]
    if unknown:
        raise RunnerError(f"unsupported metric(s) {unknown}")


def _features_and_labels(
    table: Table, label_column: str, feature_columns: Sequence[str] | None
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
        y.append(label)
    return X, y


def _check_split_for_metrics(
    names: Sequence[str], train_labels: Sequence[Label], test_labels: Sequence[Label]
) -> None:
    if "roc_auc" in names:
        if len(set(train_labels)) != 2:
            raise RunnerError(
                f"roc_auc needs exactly two classes in the training split, "
                f"found {len(set(train_labels))}"
            )
        if len(set(test_labels)) != 2:
            raise RunnerError(
                f"roc_auc needs both classes in the test split, found {len(set(test_labels))}"
            )
    if "log_loss" in names:
        unseen = sorted(set(test_labels) - set(train_labels))
        if unseen:
            raise RunnerError(f"log_loss needs every test label in training; unseen: {unseen}")


def _build_model(config: ExperimentConfig) -> tuple[Classifier, int | None]:
    params = dict(config.params)
    cls = REGISTRY.get(config.model)
    model_seed: int | None = None
    if cls is not None and "seed" in inspect.signature(cls.__init__).parameters:
        if "seed" not in params:
            params["seed"] = derive_seed(config.seed, "model")
        model_seed = params["seed"]
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
        else:
            proba = proba if proba is not None else model.predict_proba(test_X)
            if name == "roc_auc":
                _, positive = model.classes_
                scores[name] = m.roc_auc(y_true, [row[1] for row in proba], positive_label=positive)
            else:
                scores[name] = m.log_loss(y_true, proba, labels=model.classes_)
    return scores


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
                store.final_metrics(run_id),
                int(record.tags["n_train"]),
                int(record.tags["n_test"]),
                record.tags["predictions_sha256"],
                reused=True,
            )
    return None
