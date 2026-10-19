"""Deterministic seeding.

Python's built-in `hash()` is randomised per process for strings, so sub-seeds are derived with
SHA-256 instead. Derived seeds depend only on the base seed and the parts given, never on the
interpreter, platform, or process.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import random
from dataclasses import dataclass

from .errors import SeedError

MAX_SEED = 2**32 - 1  # the largest seed numpy and scikit-learn accept
OPTIONAL_GLOBAL_SEEDED = ("numpy",)


def check_seed(seed: object) -> int:
    """Return `seed` if it is an integer in [0, 2**32 - 1], otherwise raise SeedError."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SeedError(f"seed must be an integer, got {type(seed).__name__}")
    if not 0 <= seed <= MAX_SEED:
        raise SeedError(f"seed must be between 0 and {MAX_SEED}, got {seed}")
    return seed


def derive_seed(base: int, *parts: str | int) -> int:
    """Derive a stable sub-seed from a base seed and labels, e.g. ("split",) or ("fold", 3)."""
    check_seed(base)
    for part in parts:
        if isinstance(part, bool) or not isinstance(part, (str, int)):
            raise SeedError(f"seed parts must be strings or integers, got {part!r}")
    payload = json.dumps([base, *parts], separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("ascii")).digest()
    return int.from_bytes(digest[:4], "big")


def rng_for(base: int, *parts: str | int) -> random.Random:
    """A private random generator for one purpose. It never touches the global `random` state."""
    return random.Random(derive_seed(base, *parts))


@dataclass(frozen=True)
class SeedReport:
    """Which libraries `seed_everything` seeded and which were not installed."""

    seed: int
    seeded: tuple[str, ...]
    skipped: tuple[str, ...]


def seed_everything(seed: int) -> SeedReport:
    """Seed Python's global `random` and numpy's global generator when numpy is installed.

    scikit-learn has no global seed: pass `derive_seed(seed, ...)` as each estimator's
    `random_state`. Reproducibility of string-keyed set ordering additionally needs the
    PYTHONHASHSEED environment variable fixed before the interpreter starts; the environment
    fingerprint records it.
    """
    check_seed(seed)
    random.seed(seed)
    seeded = ["random"]
    skipped: list[str] = []
    for name in OPTIONAL_GLOBAL_SEEDED:
        try:
            module = importlib.import_module(name)
        except ImportError:
            skipped.append(name)
            continue
        module.random.seed(seed)
        seeded.append(name)
    return SeedReport(seed, tuple(seeded), tuple(skipped))
