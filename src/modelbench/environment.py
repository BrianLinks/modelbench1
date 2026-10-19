"""Environment fingerprints and drift detection.

A fingerprint records the facts that can change numerical results between two machines or two
installs. Comparing fingerprints lets the reproducibility checker explain a mismatch instead of
just reporting it.
"""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata

DEFAULT_LIBRARIES = ("numpy", "scikit-learn")

_IMPACT = {
    "python": "a different Python version can change random number streams and float formatting",
    "implementation": "a different interpreter can change numeric and ordering behaviour",
    "system": "operating systems can differ in math library results",
    "machine": "CPU architecture can change floating-point results",
    "pythonhashseed": "unfixed hash seeds make string-keyed set ordering vary between processes",
}
_LIBRARY_IMPACT = "a different library version can change algorithm results"


@dataclass(frozen=True)
class EnvDifference:
    key: str
    old: str | None
    new: str | None
    impact: str


def environment_fingerprint(
    libraries: Sequence[str] = DEFAULT_LIBRARIES,
) -> dict[str, str | None]:
    """Collect interpreter, platform, and library versions. Missing libraries map to None."""
    info: dict[str, str | None] = {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
    for name in libraries:
        try:
            info[f"lib:{name}"] = metadata.version(name)
        except metadata.PackageNotFoundError:
            info[f"lib:{name}"] = None
    return info


def compare_environments(
    old: Mapping[str, str | None], new: Mapping[str, str | None]
) -> list[EnvDifference]:
    """Differences between two fingerprints, sorted by key. A key missing on one side is None."""
    differences: list[EnvDifference] = []
    for key in sorted(set(old) | set(new)):
        before, after = old.get(key), new.get(key)
        if before == after:
            continue
        impact = _LIBRARY_IMPACT if key.startswith("lib:") else _IMPACT.get(key, "unknown effect")
        differences.append(EnvDifference(key, before, after, impact))
    return differences
