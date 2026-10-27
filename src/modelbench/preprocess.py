"""Preprocessing for tabular features: missing values, categories, and scaling.

A `PreprocessSpec` is part of the experiment config. The data is parsed once (so a bad cell is
reported before any run exists), then for every training split a `Preprocessor` is fitted on the
training rows only and applied to both the training and the test rows, so nothing about the test
rows leaks into the features.

Order of work, for each feature column in the order given:

1. Cells in `categorical` columns are text; every other column must be a finite number or a
   missing value (empty, NA, N/A, NaN, null, None, or ?, in any case).
2. Missing numbers are filled by `impute`: "error" (the default: a missing number is an error),
   "mean", "median" or "constant" (`fill_value`). The statistic comes from the training rows.
   With `missing_indicator`, a 0/1 column follows each numeric column that had a missing value
   in the training rows.
3. Each categorical column becomes one 0/1 column per category seen in training, categories in
   sorted order; missing text is its own category. A category not seen in training is all zeros.
   A column with more than `max_categories` categories is an error.
4. `scale` applies to the original numeric columns after imputing: "standardize" (mean 0,
   population standard deviation 1) or "minmax" (0 to 1). A constant column becomes 0.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .errors import PreprocessError

NUMBER_RE = re.compile(r"[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?")  # ASCII digits only
MISSING_TOKENS = frozenset({"", "na", "n/a", "nan", "null", "none", "?"})
IMPUTE_STRATEGIES = ("error", "mean", "median", "constant")
SCALERS = ("none", "standardize", "minmax")
MAX_CATEGORIES_LIMIT = 1000
MISSING_CATEGORY = "<missing>"

Cell = float | str | None  # a number, text, or None for missing


def is_missing(text: str) -> bool:
    return text.strip().lower() in MISSING_TOKENS


@dataclass(frozen=True)
class PreprocessSpec:
    categorical: tuple[str, ...] = ()
    impute: str = "error"
    fill_value: float = 0.0
    scale: str = "none"
    missing_indicator: bool = False
    max_categories: int = 50

    def __post_init__(self) -> None:
        if not isinstance(self.categorical, tuple) or not all(
            isinstance(c, str) and c for c in self.categorical
        ):
            raise PreprocessError("categorical must be a list of column names")
        if len(set(self.categorical)) != len(self.categorical):
            raise PreprocessError("categorical must not repeat a column")
        if self.impute not in IMPUTE_STRATEGIES:
            raise PreprocessError(f"impute must be one of {IMPUTE_STRATEGIES}, got {self.impute!r}")
        fill = self.fill_value
        if isinstance(fill, bool) or not isinstance(fill, (int, float)) or not math.isfinite(fill):
            raise PreprocessError(f"fill_value must be a finite number, got {fill!r}")
        if self.scale not in SCALERS:
            raise PreprocessError(f"scale must be one of {SCALERS}, got {self.scale!r}")
        if not isinstance(self.missing_indicator, bool):
            raise PreprocessError("missing_indicator must be true or false")
        limit = self.max_categories
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise PreprocessError(f"max_categories must be a whole number, got {limit!r}")
        if not 1 <= limit <= MAX_CATEGORIES_LIMIT:
            raise PreprocessError(f"max_categories must be from 1 to {MAX_CATEGORIES_LIMIT}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "categorical": list(self.categorical),
            "impute": self.impute,
            "fill_value": self.fill_value,
            "scale": self.scale,
            "missing_indicator": self.missing_indicator,
            "max_categories": self.max_categories,
        }

    @classmethod
    def from_dict(cls, data: object) -> PreprocessSpec:
        if not isinstance(data, dict):
            raise PreprocessError("preprocess must be a JSON object")
        known = {
            "categorical", "impute", "fill_value", "scale", "missing_indicator", "max_categories"
        }  # fmt: skip
        extra = sorted(set(data) - known)
        if extra:
            raise PreprocessError(f"unknown key(s) in preprocess: {', '.join(extra)}")
        columns = data.get("categorical", [])
        if not isinstance(columns, list):
            raise PreprocessError("categorical must be a list of column names")
        values = {k: v for k, v in data.items() if k != "categorical"}
        return cls(categorical=tuple(columns), **values)


def parse_cells(
    rows: Sequence[Sequence[str]], names: Sequence[str], spec: PreprocessSpec
) -> list[list[Cell]]:
    """Turn raw text cells into numbers, text, or None (missing). `rows` hold only the features.

    Raises PreprocessError naming the (1-based) row and column of the first unusable cell.
    """
    unknown = [c for c in spec.categorical if c not in names]
    if unknown:
        raise PreprocessError(f"categorical column(s) are not features: {unknown}")
    categorical = [name in spec.categorical for name in names]
    out: list[list[Cell]] = []
    for number, row in enumerate(rows, start=1):
        parsed: list[Cell] = []
        for name, is_cat, cell in zip(names, categorical, row, strict=True):
            if is_cat:
                parsed.append(None if is_missing(cell) else cell.strip())
            elif is_missing(cell):
                parsed.append(None)
            else:
                text = cell.strip()
                if NUMBER_RE.fullmatch(text) is None or not math.isfinite(float(text)):
                    raise PreprocessError(
                        f"row {number}, column {name!r}: {cell!r} is not a finite number"
                    )
                parsed.append(float(text))
        out.append(parsed)
    return out


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


@dataclass(frozen=True)
class _Numeric:
    name: str
    fill: float
    indicator: bool
    shift: float
    scale: float


@dataclass(frozen=True)
class _Categories:
    name: str
    levels: tuple[str | None, ...]  # sorted, with None (missing) last


class Preprocessor:
    """Fitted preprocessing: `Preprocessor.fit(...)`, then `transform(rows)`."""

    def __init__(self, spec: PreprocessSpec, columns: list[_Numeric | _Categories]) -> None:
        self.spec = spec
        self._columns = columns

    @classmethod
    def fit(
        cls, spec: PreprocessSpec, names: Sequence[str], rows: Sequence[Sequence[Cell]]
    ) -> Preprocessor:
        if not rows:
            raise PreprocessError("cannot fit preprocessing on zero rows")
        columns: list[_Numeric | _Categories] = []
        for j, name in enumerate(names):
            values = [row[j] for row in rows]
            if name in spec.categorical:
                seen = {v for v in values if v is not None}
                if len(seen) > spec.max_categories:
                    raise PreprocessError(
                        f"column {name!r} has {len(seen)} categories in the training rows; "
                        f"the limit is {spec.max_categories}"
                    )
                levels: list[str | None] = [*sorted(s for s in seen if isinstance(s, str))]
                if any(v is None for v in values):
                    levels.append(None)
                columns.append(_Categories(name, tuple(levels)))
                continue
            present = [v for v in values if isinstance(v, float)]
            missing = len(present) < len(values)
            if missing and spec.impute == "error":
                raise PreprocessError(
                    f"column {name!r} has missing values; set impute to mean, median or constant"
                )
            if not present and spec.impute != "constant":
                raise PreprocessError(f"column {name!r} has no values in the training rows")
            if spec.impute == "mean" and present:
                fill = math.fsum(present) / len(present)
            elif spec.impute == "median" and present:
                fill = _median(present)
            else:
                fill = float(spec.fill_value)
            filled = [fill if v is None else v for v in values if not isinstance(v, str)]
            shift, scale = 0.0, 1.0
            if spec.scale == "standardize":
                shift = math.fsum(filled) / len(filled)
                variance = math.fsum((v - shift) ** 2 for v in filled) / len(filled)
                scale = math.sqrt(variance) if variance > 0 else 1.0
            elif spec.scale == "minmax":
                shift = min(filled)
                scale = (max(filled) - shift) or 1.0
            columns.append(_Numeric(name, fill, missing and spec.missing_indicator, shift, scale))
        return cls(spec, columns)

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        for column in self._columns:
            if isinstance(column, _Categories):
                names += [
                    f"{column.name}={MISSING_CATEGORY if level is None else level}"
                    for level in column.levels
                ]
            else:
                names.append(column.name)
                if column.indicator:
                    names.append(f"{column.name}_missing")
        return names

    def transform(self, rows: Sequence[Sequence[Cell]]) -> list[list[float]]:
        out: list[list[float]] = []
        for row in rows:
            features: list[float] = []
            for column, cell in zip(self._columns, row, strict=True):
                if isinstance(column, _Categories):
                    features += [1.0 if cell == level else 0.0 for level in column.levels]
                else:
                    value = column.fill if cell is None or isinstance(cell, str) else cell
                    features.append((value - column.shift) / column.scale)
                    if column.indicator:
                        features.append(1.0 if cell is None else 0.0)
            out.append(features)
        return out


def fit_transform_split(
    spec: PreprocessSpec,
    names: Sequence[str],
    cells: Sequence[Sequence[Cell]],
    train_idx: Sequence[int],
    test_idx: Sequence[int],
) -> tuple[list[list[float]], list[list[float]], list[str]]:
    """Fit on the training rows and return `(train features, test features, feature names)`."""
    pre = Preprocessor.fit(spec, names, [cells[i] for i in train_idx])
    return (
        pre.transform([cells[i] for i in train_idx]),
        pre.transform([cells[i] for i in test_idx]),
        pre.feature_names,
    )
