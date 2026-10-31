"""Dataset tables, content-hash versions, and deterministic train/test splits.

A dataset version is a hash of the table's content (column names and every cell, in order), so
the same data always gets the same version no matter where the file lives or how it was encoded.
Row order is part of the content: splits are index-based, so a reordered table is a new version.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .errors import DatasetError
from .ids import validate_id
from .seeding import check_seed, rng_for

VERSION_LENGTH = 12


@dataclass(frozen=True)
class Table:
    """Rectangular string data: named columns and rows of equal width."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not self.columns:
            raise DatasetError("table must have at least one column")
        if any(not name.strip() for name in self.columns):
            raise DatasetError("column names must not be empty")
        if len(set(self.columns)) != len(self.columns):
            raise DatasetError("column names must be unique")
        width = len(self.columns)
        for number, row in enumerate(self.rows, start=1):
            if len(row) != width:
                raise DatasetError(f"row {number} has {len(row)} cells, expected {width}")

    def __len__(self) -> int:
        return len(self.rows)

    def column(self, name: str) -> list[str]:
        try:
            position = self.columns.index(name)
        except ValueError:
            raise DatasetError(f"no column named {name!r}") from None
        return [row[position] for row in self.rows]

    def select(self, indices: Sequence[int]) -> Table:
        """A new table holding the given rows, in the given order."""
        return Table(self.columns, tuple(self.rows[i] for i in indices))


@dataclass(frozen=True)
class DatasetVersion:
    dataset_id: str
    version: str
    sha256: str
    n_rows: int
    columns: tuple[str, ...]
    label_column: str | None = None


def resolve_input_file(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve to an absolute regular file, optionally requiring it to live inside base_dir."""
    resolved = Path(path).expanduser().resolve()
    if base_dir is not None:
        base = Path(base_dir).expanduser().resolve()
        if not resolved.is_relative_to(base):
            raise DatasetError(f"{path} is outside the allowed directory {base}")
    if not resolved.is_file():
        raise DatasetError(f"{path} is not a readable file")
    return resolved


def read_csv(path: str | Path, base_dir: str | Path | None = None) -> Table:
    """Read a UTF-8 CSV file with a header. Ragged rows are errors, not silently padded."""
    resolved = resolve_input_file(path, base_dir)
    try:
        with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = [cell.strip() for cell in next(reader)]
            except StopIteration:
                raise DatasetError(f"{resolved.name} is empty") from None
            rows: list[tuple[str, ...]] = []
            for record in reader:
                if not record:
                    continue  # blank line
                if len(record) != len(header):
                    raise DatasetError(
                        f"{resolved.name} line {reader.line_num}: expected {len(header)} "
                        f"cells, found {len(record)}"
                    )
                rows.append(tuple(record))
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{resolved.name} is not valid UTF-8: {exc.reason}") from exc
    except csv.Error as exc:
        raise DatasetError(f"{resolved.name} is not valid CSV: {exc}") from exc
    except OSError as exc:
        raise DatasetError(f"cannot read {resolved}: {exc}") from exc
    return Table(tuple(header), tuple(rows))


def table_digest(table: Table) -> str:
    """SHA-256 of the table's canonical JSON form: columns, then rows, in order."""
    document = {"columns": list(table.columns), "rows": [list(row) for row in table.rows]}
    text = json.dumps(document, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def make_version(dataset_id: str, table: Table, label_column: str | None = None) -> DatasetVersion:
    validate_id("dataset", dataset_id)
    if len(table) == 0:
        raise DatasetError("dataset has no rows")
    if label_column is not None and label_column not in table.columns:
        raise DatasetError(f"label column {label_column!r} is not in the dataset")
    digest = table_digest(table)
    return DatasetVersion(
        dataset_id=dataset_id,
        version=digest[:VERSION_LENGTH],
        sha256=digest,
        n_rows=len(table),
        columns=table.columns,
        label_column=label_column,
    )


def _check_fraction(test_fraction: object) -> float:
    if isinstance(test_fraction, bool) or not isinstance(test_fraction, (int, float)):
        raise DatasetError(f"test_fraction must be a number, got {test_fraction!r}")
    if not math.isfinite(test_fraction) or not 0 < test_fraction < 1:
        raise DatasetError(f"test_fraction must be strictly between 0 and 1, got {test_fraction}")
    return float(test_fraction)


def split_indices(
    n: int,
    test_fraction: float,
    seed: int,
    stratify: Sequence[Hashable] | None = None,
) -> tuple[list[int], list[int]]:
    """Deterministically split row indices 0..n-1 into sorted (train, test) lists.

    The result depends only on the arguments. Without `stratify` the test set has
    max(1, min(n - 1, round(n * test_fraction))) rows. With `stratify` (one label per row) each
    class is split on its own so class proportions carry over; a class with a single row stays in
    train. Uses Python's seeded Mersenne Twister, which the environment fingerprint covers.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise DatasetError(f"need at least 2 rows to split, got {n!r}")
    fraction = _check_fraction(test_fraction)
    check_seed(seed)

    if stratify is None:
        rng = rng_for(seed, "split", "random")
        order = list(range(n))
        rng.shuffle(order)
        n_test = max(1, min(n - 1, round(n * fraction)))
        return sorted(order[n_test:]), sorted(order[:n_test])

    if len(stratify) != n:
        raise DatasetError(f"stratify has {len(stratify)} labels for {n} rows")
    groups: dict[Hashable, list[int]] = {}
    for index, label in enumerate(stratify):
        groups.setdefault(label, []).append(index)
    rng = rng_for(seed, "split", "stratified")
    train: list[int] = []
    test: list[int] = []
    for label in sorted(groups, key=lambda k: (type(k).__name__, repr(k))):
        members = groups[label]
        rng.shuffle(members)
        size = len(members)
        n_test = 0 if size == 1 else max(1, min(size - 1, round(size * fraction)))
        test.extend(members[:n_test])
        train.extend(members[n_test:])
    if not test:
        raise DatasetError("cannot stratify: every class has a single row")
    return sorted(train), sorted(test)


MAX_FOLDS = 50


def kfold_indices(
    n: int,
    folds: int,
    seed: int,
    stratify: Sequence[Hashable] | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Deterministic k-fold splits: a list of sorted (train, test) index lists, one per fold.

    Every row is in exactly one test fold. Fold sizes differ by at most one. With `stratify`
    (one label per row) each class is shuffled and dealt across the folds in turn, so every class
    count also differs by at most one between folds. The result depends only on the arguments.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise DatasetError(f"need at least 2 rows to split, got {n!r}")
    if isinstance(folds, bool) or not isinstance(folds, int) or not 2 <= folds <= MAX_FOLDS:
        raise DatasetError(f"folds must be a whole number from 2 to {MAX_FOLDS}, got {folds!r}")
    if folds > n:
        raise DatasetError(f"cannot make {folds} folds from {n} rows")
    check_seed(seed)

    assignment = [0] * n
    if stratify is None:
        rng = rng_for(seed, "kfold", "random")
        order = list(range(n))
        rng.shuffle(order)
        for position, index in enumerate(order):
            assignment[index] = position % folds
    else:
        if len(stratify) != n:
            raise DatasetError(f"stratify has {len(stratify)} labels for {n} rows")
        groups: dict[Hashable, list[int]] = {}
        for index, label in enumerate(stratify):
            groups.setdefault(label, []).append(index)
        rng = rng_for(seed, "kfold", "stratified")
        dealt = 0
        for label in sorted(groups, key=lambda k: (type(k).__name__, repr(k))):
            members = groups[label]
            rng.shuffle(members)
            for index in members:
                assignment[index] = dealt % folds
                dealt += 1

    result: list[tuple[list[int], list[int]]] = []
    for fold in range(folds):
        test = [i for i in range(n) if assignment[i] == fold]
        train = [i for i in range(n) if assignment[i] != fold]
        result.append((train, test))
    return result


def ordered_split(n: int, test_fraction: float) -> tuple[list[int], list[int]]:
    """Split rows 0..n-1 by position: the first rows train, the last `test_fraction` test.

    The test set has max(1, min(n - 1, round(n * test_fraction))) rows, the same size rule as
    `split_indices`. Nothing is shuffled, so the test rows always come after the training rows.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise DatasetError(f"need at least 2 rows to split, got {n!r}")
    fraction = _check_fraction(test_fraction)
    n_test = max(1, min(n - 1, round(n * fraction)))
    return list(range(n - n_test)), list(range(n - n_test, n))


def expanding_window_indices(n: int, folds: int) -> list[tuple[list[int], list[int]]]:
    """Expanding-window splits for ordered rows: `folds` (train, test) pairs.

    The rows are cut, in order, into folds + 1 contiguous blocks whose sizes differ by at most
    one (the earlier blocks take the extra rows). Fold i trains on blocks 0..i and tests on block
    i + 1, so every test row comes after every training row and the training set only grows.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise DatasetError(f"need at least 2 rows to split, got {n!r}")
    if isinstance(folds, bool) or not isinstance(folds, int) or not 2 <= folds <= MAX_FOLDS:
        raise DatasetError(f"folds must be a whole number from 2 to {MAX_FOLDS}, got {folds!r}")
    blocks = folds + 1
    if blocks > n:
        raise DatasetError(f"cannot cut {n} rows into {blocks} blocks for {folds} folds")
    size, extra = divmod(n, blocks)
    edges = [0]
    for block in range(blocks):
        edges.append(edges[-1] + size + (1 if block < extra else 0))
    return [
        (list(range(edges[fold + 1])), list(range(edges[fold + 1], edges[fold + 2])))
        for fold in range(folds)
    ]


def split_table(
    table: Table,
    test_fraction: float,
    seed: int,
    stratify_column: str | None = None,
) -> tuple[Table, Table]:
    """Split a table into (train, test) tables. See `split_indices` for the rules."""
    labels = table.column(stratify_column) if stratify_column is not None else None
    train, test = split_indices(len(table), test_fraction, seed, labels)
    return table.select(train), table.select(test)
