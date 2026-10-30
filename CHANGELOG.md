# Changelog

All notable changes to this project are documented here.

## [Unreleased]

## [0.6.0] - 2026-10-30

### Added
- `modelbench select`: forward and backward feature selection scored by cross-validation on shared
  folds, listing what every candidate column would have scored at each step.
- `modelbench nested`: nested cross-validation over a grid or a seeded search space, reporting the
  honest outer estimate, the inner score plain selection would give, the optimism between them,
  and how often each candidate was chosen.

## [0.5.0] - 2026-10-30

### Added
- `modelbench halve`: successive halving over a seeded search space, with one model parameter
  (such as `n_estimators`) as the budget; each step is an ordinary recorded run.
- `calibrated`: wraps any classifier with Platt or isotonic probability calibration fitted on
  out-of-fold predictions, seeded and repeatable.
- `gradient_boosting` and `gradient_boosting_regressor`: gradient-boosted regression trees for
  classification (multinomial log loss, Newton leaf values) and squared-error regression, with
  `learning_rate`, `subsample`, a per-stage training loss and staged predictions.
- `DecisionTreeRegressor.apply` and `set_leaf_values`.
- `random_forest` and `random_forest_regressor`: bagged trees that look at a random subset of
  features at every split (seeded, repeatable), with an out-of-bag score. `bagged_trees` reports
  `oob_score_` too.

## [0.4.0] - 2026-10-29

### Added
- Dashboard script (`static/dashboard.js`, ES5, no dependencies): sort the run list by any column
  or metric, filter it by text, read the value under the pointer on charts, click a line in the
  all-parameters chart to open the run, switch the confusion matrix between counts and percent of
  row, and apply experiment-page choices at once. Every page is still complete without it. The
  Content-Security-Policy now allows `script-src 'self'`; inline scripts stay forbidden
  ([ADR 9](docs/adr/0009-a-small-same-origin-script-for-the-dashboard.md)).
- Tests for the script: logic under Node, a source scan that forbids `innerHTML`, `eval`, network
  calls, storage and external URLs, and browser tests (Playwright, optional) with a CI job that
  requires them.
- `requirements-dev.txt`: exact pins for pytest, ruff, mypy and their dependencies (Python 3.12 and
  newer), used as constraints by CI and the release workflow, with a test that keeps it consistent
  with `pyproject.toml`.
- `docs/REPRODUCIBILITY.md`: what a run depends on, how `repro` decides, what can break reproducibility, and what to do when it does.

## [0.3.1] - 2026-10-28

### Fixed
- `--reuse` and sweeps no longer return a stored run that used different feature columns: runs now
  record a `columns_sha256` tag (label column plus ordered feature columns) and reuse needs it to
  match. Runs stored by older versions lack the tag and are not reused.
- `import --dry-run` now makes every check the real import makes (timestamps, tag values, metric
  names, artifacts, notes), so a document it accepts is a document the import accepts.
- The importer rejects a dataset whose `version` is not the first 12 characters of its `sha256`,
  and one import naming the same dataset version with two different hashes.
- A learning curve (or any scoring) with `roc_auc`, `average_precision`, `brier_score` or `ece` on a
  training subset with a single class now reports a clear error instead of a traceback.

## [0.3.0] - 2026-10-28

### Added
- A test that every import in the package is the standard library or the package itself, and that
  no module refers to a paid service, so the standalone rule in ADR 7 is checked, not just stated.
- Parallel-coordinates chart on the dashboard's experiment page and in reports, for experiments
  where two or more parameters vary: one line per run, the best run highlighted.
- Dataset manifests (`dataset --manifest`): row count, version hash, per-column kind, missing
  cells, distinct values and statistics, and label counts, as canonical JSON.
- Experiment notes (`note --experiment`, schema migration 3): free text about a whole experiment,
  shown on the dashboard's experiment page and in the report. They stay in the database and are
  not part of the export format.
- `examples/`: ready-made config, sweep, search and cross-validation files, run end to end by
  the test suite.
- A frozen export file (`tests/golden/export_v1.json`) that the export must match byte for byte
  and the importer must still accept.
- Release workflow: pushing a `vX.Y.Z` tag runs the checks and publishes a GitHub release with
  the changelog section as notes; `scripts/release_notes.py` refuses a tag that does not match the
  declared version.
- CI tests on Windows and Python 3.10 to 3.13.
- `search` command and `run_search`: seeded random search over `int`, `uniform`, `log_uniform`
  and `choice` parameters, ranked and reported like a sweep.
- `report` command and `report` module: a standalone, script-free HTML or Markdown report on
  one experiment's runs, deterministic for the same database contents.
- `learning-curve` command and `learning_curve`: train and test score at growing fractions of
  the training rows, per fold when cross-validating.
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
