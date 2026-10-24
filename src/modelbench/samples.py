"""Seeded synthetic datasets and a one-call demo that fills a store with runs.

Every generator is a pure function of its arguments: the same seed gives the same table, byte
for byte, on any machine. Values are rounded to six decimals, so the last-digit differences that
`sin` and `cos` can show between platforms cannot change a dataset's content hash.

Datasets (labels are strings):

- `blobs`: Gaussian clusters, one per class, `c0`, `c1`, ... (any number of features)
- `moons`: two interleaving half circles, `a` and `b`; a straight line cannot separate them
- `xor`: four clusters whose label is the sign of x1 * x2, `neg` or `pos`; no single straight
  line separates them either
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .config import ExperimentConfig, expand_config
from .datasets import Table
from .errors import ModelBenchError
from .runner import RunResult, run_experiment
from .seeding import check_seed, rng_for
from .store import Store

MAX_ROWS = 100_000
MAX_FEATURES = 50
MAX_CLASSES = 20
MAX_CELLS = 1_000_000  # rows x features
DEMO_MAX_ROWS_PER_CLASS = 2000
DECIMALS = 6
BASELINE_NOTE = "baseline: always predicts the most common class"


class SamplesError(ModelBenchError, ValueError):
    """An unknown dataset name, an option that dataset does not take, or an invalid value."""


def _number(value: float) -> str:
    text = f"{value:.{DECIMALS}f}"
    return "0." + "0" * DECIMALS if text == "-0." + "0" * DECIMALS else text


def _check_count(name: str, value: object, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise SamplesError(f"{name} must be a whole number from {minimum} to {maximum}")
    return value


def _check_float(name: str, value: object, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SamplesError(f"{name} must be a number from {minimum:g} to {maximum:g}")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise SamplesError(f"{name} must be a number from {minimum:g} to {maximum:g}")
    return number


def _shuffled(rows: list[tuple[str, ...]], seed: int, name: str) -> tuple[tuple[str, ...], ...]:
    rng = rng_for(seed, name, "order")
    order = list(range(len(rows)))
    rng.shuffle(order)
    return tuple(rows[i] for i in order)


def make_blobs(
    seed: int,
    rows_per_class: int = 50,
    classes: int = 3,
    features: int = 2,
    separation: float = 3.0,
    spread: float = 1.0,
) -> Table:
    """`classes` Gaussian clusters with `rows_per_class` rows each, in shuffled order.

    The centre of class k in feature j is `separation * ((k + j) % classes)`, so the first
    feature alone already orders the classes. `spread` is the standard deviation in every
    feature.
    """
    seed = check_seed(seed)
    per_class = _check_count("rows_per_class", rows_per_class, 2, MAX_ROWS)
    n_classes = _check_count("classes", classes, 2, MAX_CLASSES)
    n_features = _check_count("features", features, 1, MAX_FEATURES)
    if per_class * n_classes > MAX_ROWS:
        raise SamplesError(f"at most {MAX_ROWS} rows in total")
    if per_class * n_classes * n_features > MAX_CELLS:
        raise SamplesError(f"at most {MAX_CELLS} values (rows x features) in total")
    gap = _check_float("separation", separation, 0.0, 1e6)
    sd = _check_float("spread", spread, 0.0, 1e6)
    rng = rng_for(seed, "blobs", "values")
    rows: list[tuple[str, ...]] = []
    for k in range(n_classes):
        for _ in range(per_class):
            cells = [
                _number(gap * ((k + j) % n_classes) + rng.gauss(0.0, sd))
                for j in range(n_features)
            ]
            rows.append((*cells, f"c{k}"))
    columns = tuple(f"x{j + 1}" for j in range(n_features)) + ("label",)
    return Table(columns, _shuffled(rows, seed, "blobs"))


def make_moons(seed: int, rows_per_class: int = 100, noise: float = 0.2) -> Table:
    """Two half circles of radius 1: class `a` on top, class `b` flipped and shifted by (1, 0.5).

    Without noise every `a` row satisfies x^2 + y^2 = 1 and every `b` row (x - 1)^2 +
    (y - 0.5)^2 = 1, with angles spread evenly over the half circle.
    """
    seed = check_seed(seed)
    per_class = _check_count("rows_per_class", rows_per_class, 2, MAX_ROWS // 2)
    sd = _check_float("noise", noise, 0.0, 1e3)
    rng = rng_for(seed, "moons", "values")
    rows: list[tuple[str, ...]] = []
    for i in range(per_class):
        angle = math.pi * i / (per_class - 1)
        top = (math.cos(angle), math.sin(angle))
        bottom = (1.0 - math.cos(angle), 0.5 - math.sin(angle))
        for label, (x, y) in (("a", top), ("b", bottom)):
            rows.append((_number(x + rng.gauss(0.0, sd)), _number(y + rng.gauss(0.0, sd)), label))
    return Table(("x1", "x2", "label"), _shuffled(rows, seed, "moons"))


def make_xor(seed: int, rows_per_class: int = 100, noise: float = 0.3) -> Table:
    """Clusters at (1, 1) and (-1, -1) labelled `pos`, and at (1, -1) and (-1, 1) labelled `neg`.

    Each class gets `rows_per_class` rows, split evenly between its two clusters (the extra row
    of an odd count goes to the first cluster).
    """
    seed = check_seed(seed)
    per_class = _check_count("rows_per_class", rows_per_class, 2, MAX_ROWS // 2)
    sd = _check_float("noise", noise, 0.0, 1e3)
    rng = rng_for(seed, "xor", "values")
    rows: list[tuple[str, ...]] = []
    clusters = (
        ("pos", ((1.0, 1.0), (-1.0, -1.0))),
        ("neg", ((1.0, -1.0), (-1.0, 1.0))),
    )
    for label, centres in clusters:
        for i in range(per_class):
            cx, cy = centres[i % 2]
            rows.append((_number(cx + rng.gauss(0.0, sd)), _number(cy + rng.gauss(0.0, sd)), label))
    return Table(("x1", "x2", "label"), _shuffled(rows, seed, "xor"))


GENERATORS: dict[str, Callable[..., Table]] = {
    "blobs": make_blobs,
    "moons": make_moons,
    "xor": make_xor,
}
OPTIONS: dict[str, tuple[str, ...]] = {
    "blobs": ("rows_per_class", "classes", "features", "separation", "spread"),
    "moons": ("rows_per_class", "noise"),
    "xor": ("rows_per_class", "noise"),
}


def available_samples() -> tuple[str, ...]:
    return tuple(sorted(GENERATORS))


def generate(name: str, seed: int, **options: Any) -> Table:
    """Build the named dataset. Options the dataset does not take are an error, not ignored."""
    if not isinstance(name, str) or name not in GENERATORS:
        raise SamplesError(f"unknown sample {name!r}; available: {list(available_samples())}")
    unknown = sorted(set(options) - set(OPTIONS[name]))
    if unknown:
        raise SamplesError(
            f"{name} does not take {', '.join(unknown)}; its options are {', '.join(OPTIONS[name])}"
        )
    return GENERATORS[name](seed, **options)


def table_to_csv(table: Table) -> str:
    """The table as CSV text (header, then rows) that `read_csv` reads back unchanged.

    Generated cells never contain commas, quotes, or newlines, so no quoting is needed; a table
    that does is refused rather than written wrongly.
    """
    lines = [",".join(table.columns)]
    for number, row in enumerate(table.rows, start=1):
        for cell in (*table.columns, *row):
            if any(ch in cell for ch in ',"\r\n'):
                raise SamplesError(f"row {number} has a cell that needs CSV quoting: {cell!r}")
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


# -- demo -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoRun:
    dataset: str
    model: str
    params: Mapping[str, Any]
    result: RunResult


def _demo_configs(dataset: str, seed: int, binary: bool) -> list[ExperimentConfig]:
    metrics = ("accuracy", "f1", "log_loss") + (("roc_auc",) if binary else ())

    def base(model: str, **params: Any) -> ExperimentConfig:
        return ExperimentConfig(
            f"demo-{dataset}", model, seed=seed, dataset_id=f"demo-{dataset}",
            metrics=metrics, params=params, stratify=True,
        )  # fmt: skip

    configs = [base("majority"), base("gaussian_nb"), base("logistic_regression", max_iter=300)]
    configs += expand_config(base("knn"), {"k": [1, 5, 15]})
    configs += expand_config(base("decision_tree"), {"max_depth": [1, 3, 6]})
    return configs


def run_demo(store: Store, seed: int = 42, rows_per_class: int = 100) -> list[DemoRun]:
    """Run every demo model on `moons`, `xor`, and `blobs`; identical runs are reused.

    Calling it again on the same store with the same arguments adds nothing. Baseline runs
    get a note saying what they are (added again if a crash lost it), so the dashboard has
    something to show. `rows_per_class` is limited to 2000 to keep the demo quick.
    """
    seed = check_seed(seed)
    rows_per_class = _check_count(
        "rows_per_class", rows_per_class, 20, DEMO_MAX_ROWS_PER_CLASS
    )
    tables = {
        "moons": (make_moons(seed, rows_per_class), True),
        "xor": (make_xor(seed, rows_per_class), True),
        "blobs": (make_blobs(seed, rows_per_class, classes=3), False),
    }
    runs: list[DemoRun] = []
    for name, (table, binary) in tables.items():
        for config in _demo_configs(name, seed, binary):
            result = run_experiment(store, config, table, "label", reuse_existing=True)
            if config.model == "majority" and not store.get_notes(result.run_id):
                store.add_note(result.run_id, BASELINE_NOTE)
            runs.append(DemoRun(name, config.model, dict(config.params), result))
    return runs
