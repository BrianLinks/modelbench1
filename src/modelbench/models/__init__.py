"""Pure-Python baseline classifiers and a name-based registry."""

from __future__ import annotations

from typing import Any

from ..errors import ModelError
from .bagged_trees import BaggedTrees
from .base import Classifier, Regressor
from .calibration import CalibratedClassifier
from .decision_tree import DecisionTreeClassifier
from .decision_tree_regressor import DecisionTreeRegressor
from .gradient_boosting import GradientBoostingClassifier
from .gradient_boosting_regressor import GradientBoostingRegressor
from .knn import KNNClassifier
from .knn_regressor import KNNRegressor
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .majority import MajorityClassifier
from .mean import MeanRegressor
from .naive_bayes import GaussianNB
from .perceptron import Perceptron
from .random_forest import RandomForest
from .random_forest_regressor import RandomForestRegressor
from .stratified import StratifiedClassifier

REGISTRY: dict[str, type[Classifier]] = {
    MajorityClassifier.name: MajorityClassifier,
    StratifiedClassifier.name: StratifiedClassifier,
    KNNClassifier.name: KNNClassifier,
    GaussianNB.name: GaussianNB,
    DecisionTreeClassifier.name: DecisionTreeClassifier,
    LogisticRegression.name: LogisticRegression,
    BaggedTrees.name: BaggedTrees,
    Perceptron.name: Perceptron,
    RandomForest.name: RandomForest,
    GradientBoostingClassifier.name: GradientBoostingClassifier,
    CalibratedClassifier.name: CalibratedClassifier,
}

REGRESSORS: dict[str, type[Regressor]] = {
    MeanRegressor.name: MeanRegressor,
    LinearRegression.name: LinearRegression,
    KNNRegressor.name: KNNRegressor,
    DecisionTreeRegressor.name: DecisionTreeRegressor,
    RandomForestRegressor.name: RandomForestRegressor,
    GradientBoostingRegressor.name: GradientBoostingRegressor,
}


def model_kind(name: str) -> str:
    """ "classification" or "regression" for a registered model name; ModelError if unknown."""
    if isinstance(name, str):
        if name in REGISTRY:
            return "classification"
        if name in REGRESSORS:
            return "regression"
    raise ModelError(f"unknown model {name!r}; available: {list(all_models())}")


def available_models() -> tuple[str, ...]:
    return tuple(sorted(REGISTRY))


def available_regressors() -> tuple[str, ...]:
    return tuple(sorted(REGRESSORS))


def all_models() -> tuple[str, ...]:
    return tuple(sorted(set(REGISTRY) | set(REGRESSORS)))


def create_regressor(name: str, **params: Any) -> Regressor:
    """Build a regression model by name. Unknown names and unknown parameters are errors."""
    try:
        cls = REGRESSORS[name]
    except (KeyError, TypeError):
        raise ModelError(
            f"unknown regressor {name!r}; available: {list(available_regressors())}"
        ) from None
    try:
        return cls(**params)
    except TypeError as exc:
        raise ModelError(f"invalid parameters for model {name!r}: {exc}") from exc


def create_model(name: str, **params: Any) -> Classifier:
    """Build a model by registry name. Unknown names and unknown parameters are errors."""
    try:
        cls = REGISTRY[name]
    except (KeyError, TypeError):
        raise ModelError(f"unknown model {name!r}; available: {list(available_models())}") from None
    try:
        return cls(**params)
    except TypeError as exc:
        raise ModelError(f"invalid parameters for model {name!r}: {exc}") from exc


__all__ = [
    "REGISTRY",
    "REGRESSORS",
    "BaggedTrees",
    "CalibratedClassifier",
    "Classifier",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "GaussianNB",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "KNNClassifier",
    "KNNRegressor",
    "LinearRegression",
    "LogisticRegression",
    "MajorityClassifier",
    "MeanRegressor",
    "Perceptron",
    "RandomForest",
    "RandomForestRegressor",
    "Regressor",
    "StratifiedClassifier",
    "all_models",
    "available_models",
    "available_regressors",
    "create_model",
    "create_regressor",
    "model_kind",
]
