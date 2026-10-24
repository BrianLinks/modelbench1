# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added
- Project scaffold: packaging, lint/type-check/test CI, Dependabot, security policy.
- Metrics: confusion matrix, accuracy, precision/recall/F1, ROC-AUC, log loss, MAE, MSE, RMSE, R2.
- Identifier validation and an error hierarchy rooted at `ModelBenchError`.
- Seeding: derived sub-seeds, global seeding report, environment fingerprint and comparison.
- Datasets: content-hash versions, strict CSV reader with path confinement, seeded splits.
- Experiment configs: strict loading, canonical hash, deterministic parameter grids.
- Run store: SQLite schema with migrations, run lifecycle, metric series, artifact metadata, tags.
- Models: classifier interface with strict input validation and a name registry.
- Baselines: majority class, seeded stratified random, k-NN, and Gaussian naive Bayes.
- Comparison: parameter diffs, metric deltas with direction, competition ranking, and a seeded
  paired bootstrap for per-example score differences.
- `compare_runs`: baseline-relative comparison of completed runs with config, dataset, seed, and
  environment differences, per-metric rankings, and a JSON-serialisable result.
- `run_experiment`: seeded split, fit, scoring, and recording of a classification run, with
  validation before any run is created, failure recording, optional predictions artifact, and
  reuse of identical completed runs.
- `check_reproducibility`: re-runs a stored run in a scratch store and reports whether the
  predictions hash and metrics (within a tolerance) match, with environment drift listed.
- `recover_stale_runs` and `Store.stale_running`: close runs a crash left in `running`.
- Run notes (`Store.add_note`, `get_notes`) with a schema migration that keeps existing data.
- Export: `export_json` (`modelbench-runs` version 1), `export_metrics_csv`, and
  `write_text_atomic`.
- Models: `decision_tree` (CART, Gini) and `logistic_regression` (multinomial, L2, accelerated
  gradient descent with standardised features).
- Dashboard: `modelbench serve` runs a read-only local web dashboard (run list, run detail with
  charts, comparison, JSON API) with a loopback-only default, Host-header check, and a strict
  Content-Security-Policy. `Store(read_only=True)` opens a database that SQLite will not write.
- Command line: `run`, `list`, `show`, `compare`, `repro`, `note`, `export`, `recover`,
  `dataset`, and `models`, with `--json` output and `python -m modelbench`.

### Changed
- `metrics.ordered_labels` is now public so metrics and models share one label ordering.

### Fixed
- Float overflow in any model (for example `gaussian_nb` with features near 1e300) raised a raw
  `OverflowError`; it is now a `ModelError`.
- Opening a file that is not a SQLite database now raises `StoreError` instead of a raw
  `sqlite3.DatabaseError`.
- Run store no longer leaks its SQLite connection when opening fails (for example when the
  database was written by a newer version).
