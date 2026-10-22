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

### Changed
- `metrics.ordered_labels` is now public so metrics and models share one label ordering.

### Fixed
- Run store no longer leaks its SQLite connection when opening fails (for example when the
  database was written by a newer version).
