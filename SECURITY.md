# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately to the maintainer (open a private
security advisory on the repository) rather than a public issue. Include steps to
reproduce and the version affected. Expect an acknowledgement within a few days.

## Scope and design notes

- ModelBench works on local files and a local SQLite database. It makes no network calls.
- Identifiers (dataset, experiment, run) are validated against a strict pattern before use.
- The library can confine CSV reads to a base directory (`read_csv(..., base_dir=...)`) to block
  path traversal; the command line reads the path you give it.
- No user-supplied string is ever passed to `eval`, `exec`, or `pickle`; artifacts are JSON.
- The dashboard server binds to localhost by default.

See [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) for what is protected, what is not, and which
tests keep the claims true.
