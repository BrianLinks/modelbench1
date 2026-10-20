"""Stratified random baseline: guesses labels in proportion to their training frequency."""

from __future__ import annotations

from typing import Any

from ..metrics import ordered_labels
from ..seeding import check_seed, rng_for
from .base import Classifier, Label, Matrix, class_priors


class StratifiedClassifier(Classifier):
    """Samples each prediction from the training class priors using a seeded generator.

    Every call to `predict` starts a fresh generator from `seed`, so the same model gives the
    same answer every time, and row i's prediction does not depend on how many rows follow it.
    Features are ignored. `predict_proba` returns the priors, like the majority baseline.
    """

    name = "stratified"

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed = check_seed(seed)
        self._priors: list[float] = []

    def get_params(self) -> dict[str, Any]:
        return {"seed": self.seed}

    def _fit(self, X: Matrix, y: list[Label]) -> None:
        self._priors = class_priors(y, ordered_labels(y))

    def _predict_proba(self, X: Matrix) -> Matrix:
        return [list(self._priors) for _ in X]

    def predict(self, X: Any) -> list[Label]:
        classes = self.classes_
        rows = self._check_predict_input(X)
        rng = rng_for(self.seed, "stratified", "predict")
        cumulative: list[float] = []
        total = 0.0
        for prior in self._priors:
            total += prior
            cumulative.append(total)
        out: list[Label] = []
        for _ in rows:
            draw = rng.random()
            chosen = len(classes) - 1  # guards against the running sum ending just below 1.0
            for index, bound in enumerate(cumulative):
                if draw < bound:
                    chosen = index
                    break
            out.append(classes[chosen])
        return out
