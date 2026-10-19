"""Identifier validation. Identifiers may end up in file names and URLs, so the rules are strict."""

from __future__ import annotations

import re

from .errors import IdentifierError

KINDS = ("dataset", "experiment", "run", "model")
_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")


def validate_id(kind: str, value: object) -> str:
    """Return `value` if it is a valid identifier of the given kind, otherwise raise.

    Valid identifiers are 1 to 64 characters from a-z, 0-9, '.', '_' and '-', start with a letter
    or digit, never contain '..', and do not end with '.'.
    """
    if kind not in KINDS:
        raise IdentifierError(f"unknown identifier kind {kind!r}; expected one of {KINDS}")
    if not isinstance(value, str):
        raise IdentifierError(f"{kind} id must be a string, got {type(value).__name__}")
    if _ID_RE.fullmatch(value) is None or ".." in value or value.endswith("."):
        raise IdentifierError(
            f"invalid {kind} id {value!r}: use 1-64 characters from a-z, 0-9, '.', '_' or '-', "
            "starting with a letter or digit, with no '..' and no trailing '.'"
        )
    return value


def is_valid_id(kind: str, value: object) -> bool:
    try:
        validate_id(kind, value)
    except IdentifierError:
        return False
    return True
