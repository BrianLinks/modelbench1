"""Print the changelog section for a release and check the version numbers agree.

    python scripts/release_notes.py v0.3.0

The argument is a tag (`v0.3.0`) or a bare version. The script fails (exit code 1) unless the
version in `pyproject.toml` and in `src/modelbench/__init__.py` both equal it and CHANGELOG.md has
a non-empty `## [0.3.0] - date` section; otherwise it prints that section's body, ready to use as
the text of a GitHub release. Exit code 2 means a file could not be read.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")


def normalise(tag: str) -> str:
    """`v0.3.0` or `0.3.0` to `0.3.0`; anything else is an error."""
    version = tag[1:] if tag.startswith("v") else tag
    if not _VERSION.match(version):
        raise ValueError(f"{tag!r} is not a version like v1.2.3")
    return version


def declared_versions(root: Path = ROOT) -> dict[str, str]:
    """The version each file declares, by file name."""
    found: dict[str, str] = {}
    patterns = {
        "pyproject.toml": (root / "pyproject.toml", r'^version\s*=\s*"([^"]+)"'),
        "__init__.py": (
            root / "src" / "modelbench" / "__init__.py",
            r'^__version__\s*=\s*"([^"]+)"',
        ),
    }
    for name, (path, pattern) in patterns.items():
        match = re.search(pattern, path.read_text(encoding="utf-8"), re.MULTILINE)
        if match is None:
            raise ValueError(f"no version found in {path}")
        found[name] = match.group(1)
    return found


def changelog_section(text: str, version: str) -> str:
    """The body under `## [version] - date`, without surrounding blank lines."""
    heading = re.compile(
        rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}\s*$", re.MULTILINE
    )
    start = heading.search(text)
    if start is None:
        raise ValueError(f"CHANGELOG.md has no dated section for {version}")
    rest = text[start.end() :]
    following = re.search(r"^## ", rest, re.MULTILINE)
    body = (rest[: following.start()] if following else rest).strip()
    if not body:
        raise ValueError(f"the CHANGELOG.md section for {version} is empty")
    return body


def release_notes(tag: str, root: Path = ROOT) -> str:
    version = normalise(tag)
    mismatched = {n: v for n, v in declared_versions(root).items() if v != version}
    if mismatched:
        detail = ", ".join(f"{name} says {value}" for name, value in sorted(mismatched.items()))
        raise ValueError(f"tag {tag} does not match the declared version: {detail}")
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    return changelog_section(text, version)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: release_notes.py TAG", file=sys.stderr)
        return 2
    try:
        notes = release_notes(args[0])
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
