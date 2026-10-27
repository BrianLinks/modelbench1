# Threat model

ModelBench is a local tool: you run it on your own machine, on your own data, and it makes no
network calls of its own. This page says what it protects, what it does not, and which checks in
the test suite keep the claims true.

## What is trusted

- **You and your machine.** Anyone who can run `modelbench` or open the database file can read
  and change every run. The database is not encrypted and has no users.
- **Your Python and the standard library.** There are no runtime dependencies to audit.

## What is not trusted

These inputs may come from someone else, so each is parsed strictly and bounded.

| Input | What can go wrong | What ModelBench does |
|---|---|---|
| CSV data file | ragged rows, bad encoding, a path that escapes a folder | Ragged rows and non-UTF-8 files are errors, not repaired. `read_csv(..., base_dir=...)` refuses paths that resolve outside the folder, including `..` tricks (the command line reads the path you give it) |
| Config, sweep and search files | huge or nested JSON, unknown keys, wrong types | Unknown keys and wrong types are errors; nesting depth, grid size (10,000), trials (1,000) and sweep-file size (1 MB) are capped |
| A JSON export from someone else | bad structure, `NaN`, thousands of fake runs, hostile text | `import` validates the whole document first, rejects `NaN` and `Infinity`, caps the file (50 MB), runs (10,000) and metric points (100,000), and writes everything or nothing in one transaction |
| Text stored in the database (notes, errors, paths) | terminal escape sequences, HTML | The command line prints it with control characters escaped; the dashboard HTML-escapes everything |
| Artifact paths in an import | a path like `/etc/passwd` | Stored as text only. ModelBench never opens an imported path |
| Requests to the dashboard | DNS rebinding, writes, huge URLs | See below |

No input is ever passed to `eval`, `exec`, `compile`, `pickle` or a shell. Artifacts are JSON.

## The dashboard

`modelbench serve` is the only part that listens on a network port.

- It binds to loopback unless you pass `--host` with another address *and* `--allow-network`.
  There is no login, so anyone who can reach the port can read every run, note and path.
- While bound to loopback it answers 403 to any request whose `Host` header is not a loopback
  name, which blocks DNS-rebinding from a web page you have open.
- It opens the database read-only (SQLite refuses writes) and accepts only GET and HEAD.
- Pages carry no script, load nothing from outside, and send a Content-Security-Policy that
  forbids scripts, remote loads and framing.
- Request targets longer than 2,000 characters are refused.

## Known limits (not defended)

- **Resource use.** A large CSV or a large `k`, tree depth or number of trials takes as long as
  it takes; there is no timeout, and a CSV has no size cap. The caps above stop accidents, not a
  determined attacker with shell access, who does not need an exploit anyway.
- **Local file permissions.** The database and reports are created with the process's default
  permissions (exports and reports are written owner-only through a temporary file; the database
  is not). Keep sensitive runs somewhere only you can read.
- **Tampering you can't detect.** Dataset versions and prediction hashes show that results were
  reproduced, not who produced them. There are no signatures.
- **Dashboard on a shared machine.** Another local user can connect to the loopback port while it
  runs. Do not run it on a machine where that matters.
- **Sample and example data** is generated from seeds; nothing sensitive is bundled.

## What the tests check

`tests/test_security_claims.py` scans the source for imports and calls and fails if any module
other than the dashboard imports a networking module, if any module imports `pickle`,
`subprocess` or similar, or calls `eval`, `exec` or `compile`, or if a runtime dependency is
added. The same file checks the folder confinement, that an imported artifact path is kept as
text, and the dashboard's host check. The importer, dashboard and CLI tests cover the rest.

To report a vulnerability see [SECURITY.md](../SECURITY.md).
