"""Probability calibration: wrap any classifier so its probabilities mean what they say.

Models such as boosted trees, naive Bayes or the perceptron often rank rows well but give
probabilities that are too confident or too timid. `CalibratedClassifier` fits the wrapped model
on k-1 folds of the training data, collects its probabilities for the held-out fold, and learns a
monotone map from those out-of-fold probabilities to how often each class really occurred. The
map is Platt's sigmoid or isotonic regression, fitted separately for each class (one against
the rest); calibrated rows are then renormalised to sum to one. The wrapped model used for
prediction is refitted on all the training rows.
"""

from __future__ import annotations

import bisect
import inspect
import math
from collections.abc import Sequence
from typing import Any

from ..datasets import kfold_indices
from ..errors import DatasetError, ModelError
from ..metrics import ordered_labels
from ..seeding import check_seed, derive_seed
from .base import Classifier, Label, Matrix

METHODS = ("sigmoid", "isotonic")
MAX_FOLDS = 20
_EPS = 1e-6


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _softplus(z: float) -> float:
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


def smoothed_targets(present: Sequence[int]) -> list[float]:
    """Platt's smoothed 0/1 outcomes: `(N+ + 1) / (N+ + 2)` if present, `1 / (N- + 2)` if not."""
    positives = sum(present)
    negatives = len(present) - positives
    high = (positives + 1.0) / (positives + 2.0)
    low = 1.0 / (negatives + 2.0)
    return [high if t else low for t in present]


class SigmoidCalibrator:
    """Platt scaling: `p -> sigmoid(a * logit(p) + b)`, fitted by Newton's method.

    The 0/1 targets are smoothed as Platt suggested, `(N+ + 1) / (N+ + 2)` for positives and
    `1 / (N- + 2)` for negatives, so a class that is always or never present gives a finite map.
    """

    def __init__(self) -> None:
        self.a = 1.0
        self.b = 0.0

    def fit(self, probabilities: Sequence[float], present: Sequence[int]) -> SigmoidCalibrator:
        if len(probabilities) != len(present) or not probabilities:
            raise ModelError("calibration needs one outcome per probability and at least one row")
        scores = [_logit(p) for p in probabilities]
        positives = sum(present)
        negatives = len(present) - positives
        targets = smoothed_targets(present)

        def loss(a: float, b: float) -> float:
            return math.fsum(
                _softplus(a * s + b) - t * (a * s + b) for s, t in zip(scores, targets, strict=True)
            )

        a, b = 0.0, math.log((negatives + 1.0) / (positives + 1.0))
        current = loss(a, b)
        for _ in range(100):
            ga = gb = haa = hab = hbb = 0.0
            for s, t in zip(scores, targets, strict=True):
                p = _sigmoid(a * s + b)
                d = p - t
                w = p * (1.0 - p)
                ga += d * s
                gb += d
                haa += w * s * s
                hab += w * s
                hbb += w
            haa += 1e-9
            hbb += 1e-9
            det = haa * hbb - hab * hab
            if det <= 0.0:
                break
            step_a = (hbb * ga - hab * gb) / det
            step_b = (haa * gb - hab * ga) / det
            if abs(step_a) < 1e-10 and abs(step_b) < 1e-10:
                break
            size = 1.0
            while size > 1e-12:  # back off until the loss falls
                candidate = loss(a - size * step_a, b - size * step_b)
                if candidate <= current + 1e-14:
                    break
                size /= 2.0
            else:
                break
            a, b, current = a - size * step_a, b - size * step_b, candidate
        self.a, self.b = a, b
        return self

    def predict(self, probability: float) -> float:
        return _sigmoid(self.a * _logit(probability) + self.b)


class IsotonicCalibrator:
    """Isotonic regression by pooling adjacent violators, applied by linear interpolation.

    Outcomes are smoothed first, as in `SigmoidCalibrator` (present rows count as
    `(N+ + 1) / (N+ + 2)`, absent rows as `1 / (N- + 2)`), so the map never reaches exactly 0 or 1
    and one confident mistake cannot make the log loss explode. Equal probabilities are pooled.
    Between two fitted blocks the map is a straight line from the end of one to the start of the
    next; outside the fitted range it stays at the first or last level. Isotonic maps need more
    rows than sigmoid ones: with a few hundred rows, prefer `sigmoid`.
    """

    def __init__(self) -> None:
        self._xs: list[float] = []
        self._ys: list[float] = []

    def fit(self, probabilities: Sequence[float], present: Sequence[int]) -> IsotonicCalibrator:
        if len(probabilities) != len(present) or not probabilities:
            raise ModelError("calibration needs one outcome per probability and at least one row")
        targets = smoothed_targets(present)
        order = sorted(range(len(probabilities)), key=lambda i: (probabilities[i], i))
        # blocks of [first x, last x, weighted sum of outcomes, weight]
        blocks: list[list[float]] = []
        for i in order:
            x, t = probabilities[i], targets[i]
            if blocks and blocks[-1][1] == x:
                blocks[-1][2] += t
                blocks[-1][3] += 1.0
            else:
                blocks.append([x, x, t, 1.0])
            while len(blocks) > 1 and (
                blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]
            ):
                last = blocks.pop()
                blocks[-1][1] = last[1]
                blocks[-1][2] += last[2]
                blocks[-1][3] += last[3]
        xs: list[float] = []
        ys: list[float] = []
        for low, high, total, weight in blocks:
            level = total / weight
            xs.append(low)
            ys.append(level)
            if high != low:
                xs.append(high)
                ys.append(level)
        self._xs, self._ys = xs, ys
        return self

    def predict(self, probability: float) -> float:
        xs, ys = self._xs, self._ys
        if not xs:
            raise ModelError("the calibrator has not been fitted")
        if probability <= xs[0]:
            return ys[0]
        if probability >= xs[-1]:
            return ys[-1]
        right = bisect.bisect_right(xs, probability)
        x0, x1, y0, y1 = xs[right - 1], xs[right], ys[right - 1], ys[right]
        return y0 + (y1 - y0) * (probability - x0) / (x1 - x0)


class CalibratedClassifier(Classifier):
    """`base` (a classifier name) with `base_params`, calibrated with `method` over `folds` folds.

    The out-of-fold split is stratified and seeded. If `base` takes a `seed` that `base_params`
    does not set, each fit of it gets its own seed derived from `seed`, so the whole model is
    repeatable. `predict_proba_uncalibrated` gives the wrapped model's own probabilities, for
    comparison.
    """

    name = "calibrated"

    def __init__(
        self,
        base: str = "gaussian_nb",
        base_params: dict[str, Any] | None = None,
        method: str = "sigmoid",
        folds: int = 3,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(base, str):
            raise ModelError(f"base must be a model name, got {base!r}")
        if base_params is None:
            base_params = {}
        if not isinstance(base_params, dict):
            raise ModelError(f"base_params must be an object, got {base_params!r}")
        if method not in METHODS:
            raise ModelError(f"method must be one of {METHODS}, got {method!r}")
        if isinstance(folds, bool) or not isinstance(folds, int) or not 2 <= folds <= MAX_FOLDS:
            raise ModelError(f"folds must be a whole number from 2 to {MAX_FOLDS}, got {folds!r}")
        self.base = base
        self.base_params = dict(base_params)
        self.method = method
        self.folds = folds
        self.seed = check_seed(seed)
        self._make_base("check")  # unknown names and bad parameters fail now, not at fit
        self._base_model: Classifier | None = None
        self._calibrators: list[SigmoidCalibrator | IsotonicCalibrator] = []

    def get_params(self) -> dict[str, Any]:
        return {
            "base": self.base,
            "base_params": dict(self.base_params),
            "method": self.method,
            "folds": self.folds,
            "seed": self.seed,
        }

    def _make_base(self, purpose: str) -> Classifier:
        from . import REGISTRY, create_model  # a late import: the registry holds this class

        if self.base == self.name:
            raise ModelError("a calibrated model cannot wrap another calibrated model")
        cls = REGISTRY.get(self.base)
        if cls is None:
            raise ModelError(f"base must be a classifier; available: {sorted(REGISTRY)}")
        params = dict(self.base_params)
        if "seed" in inspect.signature(cls.__init__).parameters and "seed" not in params:
            params["seed"] = derive_seed(self.seed, "calibrated-base", purpose)
        return create_model(self.base, **params)

    @staticmethod
    def _aligned(model: Classifier, X: Matrix, classes: tuple[Label, ...]) -> Matrix:
        slot = {label: i for i, label in enumerate(classes)}
        out = [[0.0] * len(classes) for _ in X]
        for r, row in enumerate(model.predict_proba(X)):
            for label, p in zip(model.classes_, row, strict=True):
                out[r][slot[label]] = p
        return out

    def _fit(self, X: Matrix, y: list[Label]) -> None:
        classes = ordered_labels(y)
        self._calibrators = []
        if len(classes) == 1:
            self._base_model = self._make_base("final")
            self._base_model.fit(X, y)
            return
        try:
            splits = kfold_indices(len(X), self.folds, derive_seed(self.seed, "calibration"), y)
        except DatasetError as exc:
            raise ModelError(f"cannot calibrate: {exc}") from exc
        held_out = [[0.0] * len(classes) for _ in X]
        for number, (train, test) in enumerate(splits):
            model = self._make_base(f"fold-{number}")
            model.fit([X[i] for i in train], [y[i] for i in train])
            rows = self._aligned(model, [X[i] for i in test], classes)
            for i, row in zip(test, rows, strict=True):
                held_out[i] = row
        for c, label in enumerate(classes):
            probabilities = [row[c] for row in held_out]
            present = [1 if t == label else 0 for t in y]
            calibrator: SigmoidCalibrator | IsotonicCalibrator = (
                SigmoidCalibrator() if self.method == "sigmoid" else IsotonicCalibrator()
            )
            self._calibrators.append(calibrator.fit(probabilities, present))
        final = self._make_base("final")
        final.fit(X, y)
        self._base_model = final

    def _predict_proba(self, X: Matrix) -> Matrix:
        assert self._base_model is not None
        raw = self._aligned(self._base_model, X, self.classes_)
        if not self._calibrators:
            return raw
        out: Matrix = []
        for row in raw:
            mapped = [cal.predict(p) for cal, p in zip(self._calibrators, row, strict=True)]
            total = math.fsum(mapped)
            out.append([m / total for m in mapped] if total > 0 else [1 / len(row)] * len(row))
        return out

    def predict_proba_uncalibrated(self, X: Sequence[Sequence[float]]) -> Matrix:
        """The wrapped model's own probabilities for `X`, before calibration."""
        rows = self._check_predict_input(X)
        assert self._base_model is not None
        return self._aligned(self._base_model, rows, self.classes_)
