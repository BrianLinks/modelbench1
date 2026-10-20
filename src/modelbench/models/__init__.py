"""Pure-Python baseline classifiers and a name-based registry."""

from __future__ import annotations

from typing import Any

from ..errors import ModelError
from .base import Classifier
from .majority import MajorityClassifier
from .stratified import StratifiedClassifier

REGISTRY: dict[str, type[Classifier]] = {
    MajorityClassifier.name: MajorityClassifier,
    StratifiedClassifier.name: StratifiedClassifier,
}


def available_models() -> tuple[str, ...]:
    return tuple(sorted(REGISTRY))


def create_model(name: str, **params: Any) -> Classifier:
    """Build a model by registry name. Unknown names and unknown parameters are errors."""
    try:
        cls = REGISTRY[name]
    except (KeyError, TypeError):
        raise ModelError(f"unknown model {name!r}; available: {list(available_models())}") from None
    try:
        return cls(**params)  # type: ignore[call-arg]
    except TypeError as exc:
        raise ModelError(f"invalid parameters for model {name!r}: {exc}") from exc


__all__ = [
    "REGISTRY",
    "Classifier",
    "MajorityClassifier",
    "StratifiedClassifier",
    "available_models",
    "create_model",
]
