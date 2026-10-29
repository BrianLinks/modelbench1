# Architecture

ModelBench records machine-learning experiments in a local SQLite file and compares them. It uses
only the Python standard library and has no network access. This page describes how the pieces
fit together; the decisions behind them are in [docs/adr](adr).

## Layers

Each layer imports only from its own layer or the layers above it in this list.

1. **Foundations**: `errors` (one hierarchy rooted at `ModelBenchError`), `ids` (strict
   identifiers), `seeding` (derived seeds), `metrics`, `environment` (fingerprints and drift).
2. **Inputs**: `datasets` (CSV reading, content-hash versions, seeded splits), `manifest`
   (a deterministic per-column summary of a dataset) and `config` (validated experiment configs,
   config hashes, parameter grids).
3. **Models**: `models/` holds classifiers and regressors behind two small interfaces
   (`Classifier`, `Regressor`) and two name registries (`REGISTRY`, `REGRESSORS`).
4. **Store**: `store` is the only module that touches SQLite.
5. **Runner**: `runner.run_experiment` splits, fits, scores, and records one run.
6. **Analysis and exchange**: `compare`, `run_comparison`, `reproducibility`, `export`,
   `importer`, `sweep`, `importance`, `learning_curve`,
   `samples`.
7. **Interfaces**: `cli` (the `modelbench` command) and `dashboard` (read-only web view).

## What happens in one run

1. The config is validated and the model's kind (classification or regression) picks which
   metrics are allowed. Everything that can be checked is checked before any run exists.
2. The table becomes numeric features plus labels (or numeric targets), then a seeded split.
   The dataset gets a version from a hash of its content.
3. `Store.create_run` writes the config, dataset version, seed, and environment fingerprint, and
   the run starts as `running`.
4. The model is fitted on the training rows and scored on the test rows (or, with `cv_folds`, on
   each of k folds in turn; see [ADR 6](adr/0006-cross-validation-records-folds-and-a-summary.md)).
   Predictions are hashed.
5. Metrics, tags (`n_train`, `n_test`, `predictions_sha256`, `model_seed`) and an optional
   predictions file are recorded and the run becomes `completed`. Any failure after step 3 marks
   the run `failed` with the error and re-raises. A killed process leaves `running`, which
   `recover` can close later.

## Reproducibility

`check_reproducibility` re-runs a stored run in a scratch database from its stored config and a
dataset the caller supplies, then compares the dataset version, the predictions hash, and the
metrics. If something differs it lists environment differences (Python version, platform, library
versions) that may explain it.

## Moving runs between databases

`export_json` writes a versioned document (`modelbench-runs`, version 1). `importer` validates a
whole document before writing, inserts all runs in one transaction with new ids, and skips runs
that are already present. The metrics CSV is for analysis and cannot be imported.

## The dashboard

`modelbench serve` opens the database read-only and renders complete HTML pages and a JSON API on
the server. One dependency-free script (`static/dashboard.js`, served from a fixed allowlist of
one path) adds sorting, filtering, chart read-outs and similar conveniences on top of those pages;
the pages work without it. The server binds to loopback unless told otherwise, checks the `Host`
header, sends a strict Content-Security-Policy that allows scripts from itself only, and escapes
everything it prints. The script's logic is plain functions tested under Node, and the whole page
behaviour is tested in a real browser when Playwright is installed (see
[ADR 9](adr/0009-a-small-same-origin-script-for-the-dashboard.md)).

## Testing approach

Tests use `unittest`, with one test file per module in `tests/`. Known-answer tests are computed by hand
(an exact line, a hand-built confusion matrix). New code is checked with mutation testing:
small deliberate breakages are applied one at a time and the suite must fail for each. A
survivor is either a missing test or an equivalent change; equivalent ones are written down in
the commit message rather than "fixed".
