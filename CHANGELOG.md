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

### Fixed
- Run store no longer leaks its SQLite connection when opening fails (for example when the
  database was written by a newer version).
