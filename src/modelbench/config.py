"""Validated experiment configurations and deterministic parameter grids."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .datasets import MAX_FOLDS
from .errors import ConfigError
from .ids import validate_id
from .preprocess import PreprocessSpec
from .seeding import check_seed

KNOWN_METRICS = (
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
    "brier_score",
    "ece",
    "mae",
    "mse",
    "rmse",
    "r2",
)
MAX_GRID_SIZE = 10_000
_MAX_DEPTH = 16
_VERSION_RE = re.compile(r"[0-9a-f]{12}")
_CONFIG_KEYS = {
    "experiment_id",
    "model",
    "seed",
    "dataset_id",
    "metrics",
    "params",
    "dataset_version",
    "test_fraction",
    "stratify",
    "cv_folds",
    "preprocess",
}
_REQUIRED_KEYS = ("experiment_id", "model", "seed", "dataset_id", "metrics")


def check_json_value(value: object, where: str, depth: int = 0) -> None:
    """Require a value that survives a JSON round trip unchanged.

    Allowed: None, bool, int, str, finite float, lists of those, and dicts with string keys.
    Tuples are rejected because they would come back as lists.
    """
    if depth > _MAX_DEPTH:
        raise ConfigError(f"{where}: nesting is deeper than {_MAX_DEPTH} levels")
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(f"{where}: numbers must be finite, got {value}")
        return
    if isinstance(value, list):
        for position, item in enumerate(value):
            check_json_value(item, f"{where}[{position}]", depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigError(f"{where}: keys must be strings, got {key!r}")
            check_json_value(item, f"{where}.{key}", depth + 1)
        return
    raise ConfigError(f"{where}: unsupported value of type {type(value).__name__}")


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything that determines an experiment's results, plus its label (`experiment_id`)."""

    experiment_id: str
    model: str
    seed: int
    dataset_id: str
    metrics: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)
    dataset_version: str | None = None
    test_fraction: float = 0.2
    stratify: bool = False
    cv_folds: int | None = None
    preprocess: PreprocessSpec | None = None

    def __post_init__(self) -> None:
        validate_id("experiment", self.experiment_id)
        validate_id("model", self.model)
        validate_id("dataset", self.dataset_id)
        check_seed(self.seed)
        if self.dataset_version is not None and (
            not isinstance(self.dataset_version, str)
            or _VERSION_RE.fullmatch(self.dataset_version) is None
        ):
            raise ConfigError("dataset_version must be 12 lowercase hex characters or null")
        fraction = self.test_fraction
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            raise ConfigError(f"test_fraction must be a number, got {fraction!r}")
        if not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ConfigError(f"test_fraction must be strictly between 0 and 1, got {fraction}")
        if not isinstance(self.stratify, bool):
            raise ConfigError("stratify must be true or false")
        folds = self.cv_folds
        if folds is not None and (
            isinstance(folds, bool) or not isinstance(folds, int) or not 2 <= folds <= MAX_FOLDS
        ):
            raise ConfigError(f"cv_folds must be null or a whole number from 2 to {MAX_FOLDS}")
        if self.preprocess is not None and not isinstance(self.preprocess, PreprocessSpec):
            raise ConfigError("preprocess must be a PreprocessSpec or null")
        if not isinstance(self.metrics, tuple) or not self.metrics:
            raise ConfigError("metrics must be a non-empty tuple of metric names")
        unknown = [m for m in self.metrics if m not in KNOWN_METRICS]
        if unknown:
            raise ConfigError(f"unknown metric(s) {unknown}; known: {list(KNOWN_METRICS)}")
        if len(set(self.metrics)) != len(self.metrics):
            raise ConfigError("metrics must not repeat")
        if not isinstance(self.params, dict):
            raise ConfigError("params must be an object")
        check_json_value(self.params, "params")
        object.__setattr__(self, "params", copy.deepcopy(self.params))

    def to_dict(self) -> dict[str, Any]:
        data = {
            "experiment_id": self.experiment_id,
            "model": self.model,
            "seed": self.seed,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "metrics": list(self.metrics),
            "params": copy.deepcopy(self.params),
            "test_fraction": self.test_fraction,
            "stratify": self.stratify,
        }
        if self.cv_folds is not None:  # absent when unused, so older configs keep their hash
            data["cv_folds"] = self.cv_folds
        if self.preprocess is not None:
            data["preprocess"] = self.preprocess.to_dict()
        return data

    def canonical_json(self) -> str:
        """Stable text form of the settings that affect results (everything but the label)."""
        data = self.to_dict()
        del data["experiment_id"]
        return json.dumps(
            data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )

    def config_hash(self) -> str:
        """SHA-256 of `canonical_json`. Equal settings hash equal whatever the experiment label.

        Note 1 and 1.0 serialise differently and therefore hash differently.
        """
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, data: object) -> ExperimentConfig:
        if not isinstance(data, dict):
            raise ConfigError("config must be a JSON object")
        extra = sorted(set(data) - _CONFIG_KEYS)
        if extra:
            raise ConfigError(f"unknown key(s) in config: {', '.join(extra)}")
        for key in _REQUIRED_KEYS:
            if key not in data:
                raise ConfigError(f"config is missing required key {key!r}")
        metrics = data["metrics"]
        if not isinstance(metrics, list):
            raise ConfigError("metrics must be a list of metric names")
        return cls(
            experiment_id=data["experiment_id"],
            model=data["model"],
            seed=data["seed"],
            dataset_id=data["dataset_id"],
            metrics=tuple(metrics),
            params=data.get("params", {}),
            dataset_version=data.get("dataset_version"),
            test_fraction=data.get("test_fraction", 0.2),
            stratify=data.get("stratify", False),
            cv_folds=data.get("cv_folds"),
            preprocess=(
                PreprocessSpec.from_dict(data["preprocess"])
                if data.get("preprocess") is not None
                else None
            ),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentConfig:
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigError(f"cannot read config file {path}: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise ConfigError(f"config file {path} is not valid UTF-8") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"config file {path} is not valid JSON: {exc}") from exc
        return cls.from_dict(document)


def expand_grid(base: Mapping[str, Any], grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Every combination of the grid's values merged onto `base`, in a deterministic order.

    Keys are visited in sorted order and each key's values in the order given, so the first key
    varies slowest. A key may not appear in both `base` and `grid`.
    """
    check_json_value(dict(base), "base")
    overlap = sorted(set(base) & set(grid))
    if overlap:
        raise ConfigError(f"keys in both base and grid: {', '.join(overlap)}")
    keys = sorted(grid)
    choices: list[list[Any]] = []
    total = 1
    for key in keys:
        values = grid[key]
        if not isinstance(values, list) or not values:
            raise ConfigError(f"grid[{key!r}] must be a non-empty list")
        check_json_value(values, f"grid[{key!r}]")
        choices.append(values)
        total *= len(values)
        if total > MAX_GRID_SIZE:
            raise ConfigError(f"grid has more than {MAX_GRID_SIZE} combinations")
    results: list[dict[str, Any]] = []
    for combination in itertools.product(*choices):
        merged = copy.deepcopy(dict(base))
        merged.update(copy.deepcopy(dict(zip(keys, combination, strict=True))))
        results.append(dict(sorted(merged.items())))
    return results


def expand_config(
    config: ExperimentConfig, grid: Mapping[str, Sequence[Any]]
) -> list[ExperimentConfig]:
    """One config per grid combination, all sharing the experiment id and other settings."""
    return [replace(config, params=params) for params in expand_grid(config.params, grid)]
