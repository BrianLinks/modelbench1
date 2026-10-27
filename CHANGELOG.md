# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added
- `importance` command and `permutation_importance`: seeded permutation feature importance for a
  stored run, rebuilt with the new `runner.refit` and checked against the run's predictions hash.
- Preprocessing (`preprocess` in the config): one-hot categories, mean/median/constant imputation
  with optional missing-value indicators, standardising and min-max scaling, all fitted on the
  training rows of each fold only. Runs record `n_features`.
- Metrics `brier_score` and `ece`, and the functions `roc_curve`, `precision_recall_curve`,
  `calibration_bins` and `expected_calibration_error`, checked against hand-computed points and
  the area identities with `roc_auc` and `average_precision`.
- `stats` module: paired t-test and exact McNemar test.
- A Python interface guide (`docs/API.md`) whose examples are run by the test suite.

## [0.2.0] - 2026-10-26

### Added
- Metrics: `balanced_accuracy`, `matthews_corrcoef`, `average_precision`, and micro and weighted
  precision, recall, and F1 as experiment metrics, with hand-computed tests.
- Confusion matrix per classification run (`confusion_matrix` tag), printed by `show` and drawn as
  a heatmap on the dashboard's run page.
- k-fold cross-validation (`cv_folds`): per-fold metric series, mean and standard deviation,
  pooled confusion matrix, per-fold model seeds, stratified folds.
- `sweep` command: runs every combination of a parameter grid, ranks them, reuses stored runs and
  reports failed combinations without stopping (`--stop-on-error` to stop).
- Dashboard experiment page (`/experiment/<id>`): one parameter plotted against one metric, with
  the best run highlighted.
- `knn_regressor`: mean target of the k nearest rows, with the classifier's tie and weighting rules.
- `decision_tree_regressor`: CART regression tree with the classifier's deterministic split rules.
- `bagged_trees`: bootstrap-aggregated decision trees with seeded, repeatable samples.
- `perceptron`: multiclass averaged perceptron with a seeded visiting order and standardised features.
- Seeded property tests over metrics, splits and folds, configs and grids, every model, and run
  reproducibility.
- ADR 6 on how cross-validation is recorded.

### Fixed
- Reproducibility checks no longer compare per-fold metric series the re-run does not return.

## [0.1.0] - 2026-10-25

### Added
- Architecture overview and five decision records in `docs/`.
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
- Samples: seeded `blobs`, `moons`, and `xor` generators, `run_demo` (27 runs, idempotent),
  and the `sample` and `demo` commands.
- Dashboard: `modelbench serve` runs a read-only local web dashboard (run list, run detail with
  charts, comparison, JSON API) with a loopback-only default, Host-header check, and a strict
  Content-Security-Policy. `Store(read_only=True)` opens a database that SQLite will not write.
- Regression: `Regressor` interface, `mean` baseline and `linear_regression` (ridge, closed form),
  a `REGRESSORS` registry with `model_kind`, and regression runs in `run_experiment` (numeric
  targets, regression metrics only, no stratify). `modelbench models` now shows each model's kind.
- Samples: seeded `linear` regression generator (`make_linear`, known weights via
  `linear_weights`) and the `sample linear --rows --features --noise` options.
- Import: `import_file` and `import_runs_json` read a `modelbench-runs` JSON export with full
  validation, add runs in one transaction with new ids and original timestamps, tag them
  `imported_from`, and skip runs already present. `Store.import_runs`, `Store.find_runs_by_tag`,
  and the `import` command (`--dry-run`, `--json`).
- Command line: `run`, `list`, `show`, `compare`, `repro`, `note`, `export`, `recover`,
  `dataset`, and `models`, with `--json` output and `python -m modelbench`.

### Changed
- `metrics.ordered_labels` is now public so metrics and models share one label ordering.

### Fixed
- The dashboard now claims its port exclusively on Windows. Python's HTTP server sets
  `SO_REUSEADDR`, which on Windows lets another process bind a port that is already listening.
- Printing text the console cannot encode (for example a note containing a check mark on a
  legacy Windows code page) no longer crashes the command line; unencodable characters print as
  `?`.
- Float overflow in any model (for example `gaussian_nb` with features near 1e300) raised a raw
  `OverflowError`; it is now a `ModelError`.
- Opening a file that is not a SQLite database now raises `StoreError` instead of a raw
  `sqlite3.DatabaseError`.
- Run store no longer leaks its SQLite connection when opening fails (for example when the
  database was written by a newer version).
