# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately to the maintainer (open a private
security advisory on the repository) rather than a public issue. Include steps to
reproduce and the version affected. Expect an acknowledgement within a few days.

## Scope and design notes

- ModelBench works on local files and a local SQLite database. It makes no network calls.
- Identifiers (dataset, experiment, run) are validated against a strict pattern before use.
- Import paths can be confined to a base directory to block path traversal.
- No user-supplied string is ever passed to `eval`, `exec`, or `pickle`; artifacts are JSON.
- The dashboard server binds to localhost by default.
