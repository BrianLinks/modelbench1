"""Check commit messages against the Conventional Commits style used by this repository.

    python scripts/check_commits.py main..HEAD      # every commit in a revision range
    python scripts/check_commits.py --message-file .git/COMMIT_EDITMSG   # one message (hook)

A header is `type(scope)!: subject`, where the type is one of TYPES, the scope is optional, and
the subject is non-empty, does not end with a full stop, and keeps the whole header within 72
characters. A blank line must separate the header from any body. Merge commits and the
automatic messages git writes for `fixup!`, `squash!` and `amend!` are skipped, as are the
release commits, whose whole header is `Release X.Y.Z`. Exit code 0
means every message passed, 1 means at least one failed, 2 means the input could not be read.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence

TYPES = (
    "feat",
    "fix",
    "docs",
    "test",
    "chore",
    "refactor",
    "perf",
    "ci",
    "build",
    "style",
    "revert",
)
MAX_HEADER = 72
_HEADER = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[a-z0-9][a-z0-9._/-]*)\))?(?P<bang>!)?: (?P<subject>.*)$"
)
_AUTOMATIC = ("Merge ", "fixup! ", "squash! ", "amend! ")
_RELEASE = re.compile(r"^Release \d+\.\d+\.\d+$")


def check_message(message: str) -> list[str]:
    """The problems with one commit message; an empty list means it is fine."""
    lines = message.strip("\n").splitlines()
    if not lines or not lines[0].strip():
        return ["the message is empty"]
    header = lines[0]
    if header.startswith(_AUTOMATIC) or _RELEASE.match(header):
        return []
    problems: list[str] = []
    found = _HEADER.match(header)
    if found is None:
        problems.append("the header must look like `type(scope): subject`")
    else:
        if found["type"] not in TYPES:
            problems.append(f"unknown type {found['type']!r}; use one of {', '.join(TYPES)}")
        subject = found["subject"]
        if not subject.strip():
            problems.append("the subject is empty")
        elif subject != subject.strip():
            problems.append("the subject has leading or trailing spaces")
        if subject.rstrip().endswith("."):
            problems.append("the subject must not end with a full stop")
    if len(header) > MAX_HEADER:
        problems.append(f"the header is {len(header)} characters; the limit is {MAX_HEADER}")
    if len(lines) > 1 and lines[1].strip():
        problems.append("a blank line must follow the header")
    return problems


def commit_messages(revision_range: str) -> list[tuple[str, str]]:
    """`(short hash, message)` for every commit in the range, oldest first."""
    try:
        out = subprocess.run(
            ["git", "log", "--reverse", "--format=%h%x00%B%x01", revision_range],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise SystemExit(f"cannot read commits for {revision_range!r}: {detail.strip()}") from exc
    commits = []
    for chunk in out.split("\x01"):
        if chunk.strip():
            short, _, message = chunk.lstrip("\n").partition("\x00")
            commits.append((short, message))
    return commits


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("revision_range", nargs="?", help="a git revision range, e.g. main..HEAD")
    group.add_argument("--message-file", help="check the message in this file")
    args = parser.parse_args(argv)
    if args.message_file:
        try:
            with open(args.message_file, encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            print(f"error: cannot read {args.message_file}: {exc}", file=sys.stderr)
            return 2
        # git leaves its own `#` comment lines in the editor file
        message = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
        found = {"message": check_message(message)}
    else:
        found = {short: check_message(msg) for short, msg in commit_messages(args.revision_range)}
    failed = {key: problems for key, problems in found.items() if problems}
    for key, problems in failed.items():
        for problem in problems:
            print(f"{key}: {problem}")
    if failed:
        print(f"{len(failed)} of {len(found)} message(s) need fixing", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
