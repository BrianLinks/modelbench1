# ModelBench

Local experiment tracking and model comparison, with no online ML platform and no paid API.

**Status: version 0.4.0.** What it contains:

- `modelbench.metrics`: classification and regression metrics
- `modelbench.ids`: strict dataset, experiment, run, and model identifiers
- `modelbench.seeding`: SHA-256 derived sub-seeds and global seeding
- `modelbench.environment`: environment fingerprints and drift comparison
- `modelbench.datasets`: content-hash dataset versions, CSV reading, seeded and stratified splits
- `modelbench.manifest`: a deterministic per-column summary of a dataset (`dataset --manifest`)
- `modelbench.config`: validated experiment configs, config hashes, parameter grids
- `modelbench.store`: SQLite run store (datasets, runs, params, metric series, artifact metadata,
  tags) with migrations and a strict run lifecycle
- `modelbench.models`: pure-Python classifiers behind one interface and a name registry:
  `majority`, `stratified`, `knn`, `gaussian_nb`, `decision_tree`, `logistic_regression`,
  `bagged_trees`, `random_forest`, `gradient_boosting`, `calibrated`,
  `perceptron`
- `modelbench.compare`: parameter diffs, metric deltas, ranking, and a paired bootstrap
- `modelbench.run_comparison`: `compare_runs` over completed runs in the store
- `modelbench.runner`: `run_experiment` splits a dataset, fits a model, scores it, and records
  the run
- `modelbench.reproducibility`: re-run a stored run and compare it, and close runs a crash left
  in `running`
- `modelbench.export`: export runs as JSON or a metrics CSV, written atomically
- `modelbench.sweep`: run a parameter grid as one experiment and rank the runs
- `modelbench.importer`: import a JSON export into a store, all or nothing, skipping runs already there
- `modelbench.cli`: the `modelbench` command (also `python -m modelbench`)
- `modelbench.dashboard`: a read-only local web dashboard (`modelbench serve`)
- `modelbench.samples`: seeded synthetic datasets and a demo that fills a database with runs

Regression models (`mean`, `linear_regression`, `knn_regressor`, `decision_tree_regressor`, `random_forest_regressor`,
`gradient_boosting_regressor`) run
through the same runner, with `mae`, `mse`, `rmse` and `r2`.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the pieces fit and
[docs/adr](docs/adr) for the decisions behind them.

## Running an experiment

```python
from modelbench.config import ExperimentConfig
from modelbench.datasets import Table
from modelbench.runner import run_experiment
from modelbench.store import Store

table = Table(("x", "label"), tuple((str(i), "a" if i < 10 else "b") for i in range(20)))
config = ExperimentConfig(
    "demo", "knn", seed=1, dataset_id="demo-data", metrics=("accuracy", "f1"), params={"k": 3}
)
store = Store("runs.db")  # or `with Store("runs.db") as store:` to close it for you
result = run_experiment(store, config, table, "label")
result.metrics  # {'accuracy': 1.0, 'f1': 1.0}
run_id = result.run_id  # 'run-000001'
```

The Python blocks in this README run in order, in one session, and are executed by the test suite.
The full Python interface is in [docs/API.md](docs/API.md).

What it does and records:

- Features are every column except the label, or only the columns you pass as
  `feature_columns`. Every feature cell must be a plain ASCII number, so `1_000`, `nan`, and
  non-ASCII digits are rejected with the row and column.
- The split, the model's random seed (derived from the experiment seed unless you set one in
  `params`), and the data all come from the config, so the same inputs give the same run.
- Metrics, the train and test sizes, and a SHA-256 of the test indices, true labels, and
  predictions are stored. Pass `artifact_dir` to also write the predictions as JSON, with the file
  recorded as an artifact.
- Input is validated before any run is created, including metrics the data cannot support
  (ROC-AUC needs exactly two classes in both splits; log loss needs every test label seen in
  training). Metrics must suit the model: `mae`, `mse`, `rmse` and `r2` for a regressor and the
  classification metrics for a classifier; the wrong kind is rejected.
- A failure after the run starts marks it `failed` with the error and re-raises. A hard
  interruption (Ctrl-C, a killed process) leaves it `running`.
- `reuse_existing=True` returns an earlier completed run of the same experiment, settings, and
  dataset version instead of running again.

Precision, recall, and F1 are macro-averaged over every label in the test split's true or
predicted values. ROC-AUC treats the second label in sorted order as positive.

## Checking reproducibility

```python
from modelbench.reproducibility import check_reproducibility, recover_stale_runs

report = check_reproducibility(store, run_id, table, "label", tolerance=1e-9)
report.verdict  # 'reproduced'
report.explained_by_environment  # False
recover_stale_runs(store, older_than_seconds=3600)  # ids of runs it closed
```

`check_reproducibility` re-runs the run's stored config on the data you pass, in a scratch
in-memory store, so the check never adds to your run history.

- `reproduced`: the predictions hash is identical and every stored metric is within `tolerance`
  (absolute, inclusive; default 0). Different predictions are never reproduced, even when the
  metrics agree.
- `not_reproduced`: something differs. `environment_differences` lists changed Python, platform,
  and library versions with what each can affect, and `explained_by_environment` is true when
  there is a mismatch and at least one such difference. It says the drift is a candidate cause,
  not a proven one.
- `dataset_changed`: the data is not the dataset version the run used, so nothing is re-run.
- `rerun_failed`: the re-run raised a ModelBench error, which is reported in `detail`.
- Only completed runs that recorded a predictions hash can be checked; anything else raises
  `ReproError`. Pass the same `feature_columns` the original run used.

`recover_stale_runs` marks runs still `running` after `older_than_seconds` as failed with an
explanatory error. It cannot tell a crashed run from a slow one, so choose a cutoff longer than
your longest run. Timestamps have one-second resolution and the cutoff is exclusive.

## Command line

```
modelbench [--db runs.db] <command> ...
```

| Command | What it does |
|---|---|
| `run CONFIG.json --data F.csv --label COL` | run an experiment (`--features a,b`, `--artifact-dir D`, `--reuse`) |
| `sweep SWEEP.json --data F.csv --label COL` | run every combination of a parameter grid and rank them (`--rank-by`, `--no-reuse`, `--stop-on-error`) |
| `importance RUN --data F.csv --label COL` | permutation feature importance for a stored run (`--metric`, `--repeats`, `--seed`) |
| `report EXPERIMENT` | a standalone report on one experiment's completed runs: HTML with charts, or `--format md` (`--metric`, `-o FILE`) |
| `learning-curve CONFIG.json --data F.csv --label COL` | train and test score at growing training-set sizes (`--fractions`, `--metric`) |
| `search SEARCH.json --data F.csv --label COL` | seeded random search over parameter ranges, ranked like `sweep` (same options) |
| `list` | runs, newest first (`--experiment`, `--status`, `--limit`) |
| `show RUN` | one run in full: config, metrics, tags, files, notes |
| `compare RUN RUN ...` | the first run is the baseline (`--metrics a,b`) |
| `repro RUN --data F.csv --label COL` | re-run and check it reproduces (`--tolerance`) |
| `note RUN TEXT` | attach a note to a run; `note --experiment EXP TEXT` attaches one to a whole experiment |
| `export RUN ... ` | JSON, or `--format csv`; `-o FILE` writes atomically |
| `import FILE.json` | add runs from a JSON export (`--dry-run`, `--json`); creates the database if needed |
| `sample NAME` | write a seeded synthetic dataset as CSV (`--seed`, `-o FILE`, size and noise options) |
| `demo` | run every model on three sample datasets and record the runs (`--seed`, `--rows-per-class`) |
| `serve` | read-only dashboard at http://127.0.0.1:8765/ (`--port`, `--host`, `--allow-network`) |
| `recover --older-than SECONDS` | mark long-running unfinished runs as failed |
| `dataset F.csv --id ID [--label COL]` | show a CSV's content-hash version; `--manifest` prints a per-column summary as JSON instead (`-o FILE` writes it) |
| `models` | list the available models |

`run`, `sweep`, `search`, `importance`, `learning-curve`, `list`, `show`, `compare`, `repro`, and `dataset` also take `--json`. The database
defaults to `modelbench.db`; only `run` creates it, and every other command reports a missing
database instead of making an empty one. A config file is the JSON form of an experiment:

```json
{"experiment_id": "demo", "model": "knn", "seed": 7, "dataset_id": "blobs",
 "metrics": ["accuracy", "f1"], "params": {"k": 3}}
```

A session on two clusters of 20 points:

```
$ modelbench run knn.json --data blobs.csv --label label
run-000001
train 32  test 8
accuracy  1.0000
f1  1.0000
$ modelbench run majority.json --data blobs.csv --label label
$ modelbench compare run-000001 run-000002
baseline run-000001

RUN         accuracy  f1
run-000001  1.0000    1.0000
run-000002  0.3750    0.2727
rank accuracy: 1. run-000001 (1.0000), 2. run-000002 (0.3750)
...
$ modelbench repro run-000001 --data blobs.csv --label label
run-000001: reproduced
```

Exit codes: 0 on success; 1 when `repro` finds the run is not reproduced (a result, not an
error); 2 for bad arguments or any error, with one `error: ...` line on stderr. Text taken from
the database (notes, error messages) is printed with control characters escaped, so a stored
note cannot send terminal escape sequences. Ctrl-C during `run` leaves the run `running`;
`recover` closes it later.

## Sweeps

A sweep file holds an ordinary config and a grid of parameter values:

```json
{"config": {"experiment_id": "ridge", "model": "linear_regression", "seed": 1,
            "dataset_id": "lin", "metrics": ["r2", "mae"], "params": {}},
 "grid": {"l2": [0, 0.01, 0.1, 1, 10, 100]}}
```

```
$ modelbench sweep ridge.json --data lin.csv --label y
sweep ridge: 6 combinations, ranked by r2 (higher is better)

rank  run         params   r2       mae
1     run-000003  l2=0.1   0.8469   2.2651
2     run-000002  l2=0.01  0.8389   2.2750
...
best run-000003  l2=0.1
```

Every combination is a normal run of the same experiment, in grid order (the first key, in
alphabetical order, varies slowest). Runs are ranked by the config's first metric, or by
`--rank-by`; a cross-validated sweep can also rank by `<metric>_std`. A combination that fails,
such as `k` larger than the training set, is listed with its error and the sweep carries on
(`--stop-on-error` stops instead) and the exit code is 1. Combinations already stored with
identical settings on the same data are reused, so a sweep can be re-run or extended cheaply
(`--no-reuse` repeats them). A grid key must not also be in the config's `params`.

### Random search

When a grid would be too large, `modelbench search` draws parameter sets at random:

```json
{"config": {"experiment_id": "knn-search", "model": "knn", "seed": 5, "dataset_id": "blobs",
            "metrics": ["accuracy"], "params": {}},
 "space": {"k": {"int": [1, 15]}, "weights": {"choice": ["uniform", "distance"]}},
 "trials": 20}
```

Each parameter is one of `{"int": [lo, hi]}` (both ends included), `{"uniform": [lo, hi]}`,
`{"log_uniform": [lo, hi]}` (for scales such as `l2`; `lo` must be above 0) or
`{"choice": [values]}`. Trial *n* is drawn from the config's `seed` and *n* alone, so the same
file always gives the same runs, and raising `trials` only adds draws after the earlier ones
(which are reused). A draw that repeats an earlier one is skipped, so a small space can give
fewer runs than `trials`. Everything else works as for `sweep`: ranking, `--rank-by`, failed
draws listed with their errors, exit code 1 if any failed.

The dashboard's experiment page (`/experiment/<id>`, linked from the run list and run pages)
plots one parameter against one metric for the experiment's completed runs and highlights the
best run. Pick the parameter and metric with the form; positive values spanning three orders of
magnitude get a log axis (a zero is kept), and text parameters get one slot per value. When two or
more parameters vary, a parallel-coordinates chart below it draws one line per run across up to five
parameters and the metric, with the best run highlighted; the same chart is in `report` files.
Experiment notes (`note --experiment`) are listed above the charts.

## Reports

`modelbench report EXPERIMENT -o report.html` writes one file you can mail, archive or commit. It
ranks the experiment's completed runs by a metric (`--metric`, default the newest run's first),
charts that metric by run and against each parameter that differs between runs, and shows the
best run in full: seed, dataset, confusion matrix, tags, environment and notes. `--format md`
gives the same tables as Markdown without charts. The HTML has no script and no outside
resources, and the output depends only on what is in the database, so the same runs always give
the same bytes. Failed runs are counted, not listed.

## Feature importance

```
$ modelbench importance run-000001 --data lin.csv --label y --metric mse
run-000001: permutation importance for mse (lower is better), 5 shuffle(s) per feature, baseline 10.2412

rank  feature  importance  std
1     x2       84.7212     25.6420
2     x1       19.7277     8.2195
```

Shuffling a feature among the test rows breaks its link to the target. The importance is how
much the metric got worse (a positive number always means the shuffle hurt, whichever way the
metric points), and `std` is the spread over every shuffle in every fold. The model is rebuilt
from the stored config and checked against the run's predictions hash first, so give the same
CSV and `--features` the run used; anything else is refused instead of describing another
model. Features are the model's inputs after preprocessing, so one-hot columns count one by
one, and a cross-validated run whose folds have different one-hot columns cannot be analysed.
The shuffles are seeded (`--seed`, default the run's seed), so the answer repeats.

## Learning curves

```
$ modelbench learning-curve tree.json --data moons.csv --label label
learning curve for tree: accuracy, 1 split

fraction  n_train  train   test
0.2       32       1.0000  0.7500
0.4       64       1.0000  0.8500
0.6       96       1.0000  0.8750
0.8       128      1.0000  0.7500
1         160      1.0000  0.8500
```

The model is fitted on the first `fraction` of a seeded shuffle of the training rows (each
smaller set is inside the next) and scored on the rows it learned and on the untouched test
rows. A deep tree scores 1.0 on its own rows at every size, so the gap to the test column is the
overfitting. With `cv_folds` the curve is averaged over the folds and a `test_std` column shows
the spread. Nothing is stored, and preprocessing is fitted once on each fold's whole training
set. Fractions must be increasing, in (0, 1], at most 20 of them. A model that cannot train on a
small subset (a `k` larger than the rows) stops the curve with the size at fault.

See [examples/](examples) for ready-made config, sweep and search files with the commands to run them.

## Sample data and demo

Four seeded generators produce tables with no download and no network:

| Name | What it is |
|---|---|
| `blobs` | Gaussian clusters, one per class (`c0`, `c1`, ...), any number of features |
| `linear` | a regression table: `y = 4 + 1·x1 - 2·x2 + 3·x3 ...` plus Gaussian `noise` (`--rows`, `--features`, `--noise`); the target column is `y` |
| `moons` | two interleaving half circles (`a`, `b`); a straight line cannot separate them |
| `xor` | four clusters labelled by the sign of x1 times x2 (`neg`, `pos`); no straight line separates them either |

```
modelbench sample moons --seed 42 --rows-per-class 100 -o moons.csv
modelbench --db demo.db demo --rows-per-class 30
```

The demo runs the three classification datasets. To try regression, generate `linear` and run a
config with `"model": "linear_regression"` and `"metrics": ["mae", "r2"]` on it with `--label y`.

```
DATASET  MODEL                PARAMS            ACCURACY  F1      RUN
moons    majority             -                 0.5000    0.3333  run-000001
moons    gaussian_nb          -                 0.7500    0.7483  run-000002
moons    logistic_regression  {"max_iter":300}  0.6667    0.6571  run-000003
moons    knn                  {"k":1}           0.9167    0.9161  run-000004
...
xor      knn                  {"k":5}           1.0000    1.0000  run-000014
...
27 runs (27 new, 0 reused) in demo.db
```

`demo` runs a majority baseline, Gaussian naive Bayes, logistic regression, k-NN with k of 1, 5,
and 15, and decision trees of depth 1, 3, and 6 on each dataset (27 runs). Run it again with the
same arguments and nothing is added: identical runs are reused. Then `modelbench --db demo.db
serve` shows the comparison. At 100 rows per class and seed 42, logistic regression stays under
0.9 accuracy on `moons` and under 0.65 on `xor`, a one-level tree stays under 0.65 on `xor`, and
k-NN with k of 5 scores 1.0 on `xor`: the datasets show where a linear model runs out.

Every demo run can be checked. Generate the same data again and ask `repro`:

```
modelbench sample moons --seed 42 --rows-per-class 30 -o moons.csv
modelbench --db demo.db repro run-000005 --data moons.csv --label label   # reproduced
```

Generated values are rounded to six decimals, so the last-digit differences that `sin` and
`cos` can show between platforms cannot change a dataset's content hash. Pinned golden hashes in
the tests, and the whole suite, pass on Python 3.11, 3.12, and 3.13. The demo is limited to 2000
rows per class to stay quick, and `blobs` to 100,000 rows and 1,000,000 values.

## Dashboard

```
modelbench --db runs.db serve            # http://127.0.0.1:8765/
```

Pages: the run list (filter by experiment and status, tick runs to compare), an experiment
page plotting a parameter against a metric, one run in full
(metrics, a line chart for any metric logged over several steps, tags, files, environment,
notes), and a comparison with a bar chart per metric, rankings, and parameter and environment
differences. For the comparison page the oldest selected run is the baseline unless you pick
another. JSON is available at `/api/runs`, `/api/run/<id>`, and `/api/compare?run=...` (there
the first `run` is the baseline). The pages load nothing from outside and work in light and dark
mode.

Every page is complete HTML with inline SVG charts and works with scripts blocked. One small
script, [`dashboard.js`](src/modelbench/static/dashboard.js), served by the dashboard itself and
with no dependencies, adds:

- on the run list: a filter box (every word must match), sorting by any column header (click, or
  Enter or Space on the focused header), and sorting by any recorded metric;
- on charts: the value under the pointer is shown below the chart and the nearest dot or line is
  highlighted; in the all-parameters chart, clicking a line opens that run;
- on a confusion matrix: a button that switches between counts and percent of each row;
- on the experiment page: choosing a parameter or metric applies it at once.

The script makes no requests, never builds HTML from text, and only navigates to `/run/<id>`
pages. See [ADR 9](docs/adr/0009-a-small-same-origin-script-for-the-dashboard.md).

What it will and will not do:

- It is read-only. The database is opened in SQLite's read-only mode, so a write is refused by
  SQLite itself, and a database at an older schema is reported rather than upgraded. Only GET and
  HEAD are accepted.
- It listens on the loopback interface only. While it does, requests whose `Host` header names
  anything else are answered with 403, which blocks DNS-rebinding attacks from a web page you
  have open. `--host` with a non-loopback address is refused unless you add `--allow-network`,
  because there is no login: anyone who can reach the port can read every run, note, and path.
- Every value from the database or the URL is HTML-escaped, and the Content-Security-Policy
  allows scripts only from the dashboard itself (no inline scripts), and forbids remote loads and
  framing.
- Stop it with Ctrl-C.

## Preprocessing

By default every feature must be a finite number. Add a `preprocess` block to the config to
handle text categories, missing values, and scales:

```json
{"experiment_id": "houses", "model": "decision_tree", "seed": 1, "dataset_id": "houses",
 "metrics": ["accuracy"],
 "preprocess": {"categorical": ["city"], "impute": "median", "scale": "standardize",
                "missing_indicator": true}}
```

- `categorical`: columns that hold text. Each becomes one 0/1 column per category seen in the
  training rows (sorted; missing text is its own category). A category the training rows never
  saw is all zeros. `max_categories` (default 50) stops a free-text column from exploding.
- `impute`: what to do with a missing number (empty, NA, N/A, NaN, null, None, or ?):
  `error` (the default), `mean`, `median`, or `constant` with `fill_value`.
  `missing_indicator` adds a 0/1 column after each numeric column that had a missing training value.
- `scale`: `none`, `standardize` (population standard deviation), or `minmax`, for the numeric
  columns.

Every statistic is learned from the training rows only, in each fold under cross-validation, and
applied to the test rows, so nothing leaks. Bad cells and over-wide columns are reported before a
run exists. Runs record an `n_features` tag with the width after encoding. A config without
`preprocess` behaves and hashes exactly as before.

## Cross-validation

Set `cv_folds` (2 to 50) in the config to score a model with k-fold cross-validation instead of a
single split. Rows are shuffled with the run's seed and dealt into folds (stratified by class when
`stratify` is true). Every row is tested exactly once, and each fold gets its own model seed.

- `fold/<metric>` is the per-fold series (step = fold number).
- `<metric>` is the mean over folds and `<metric>_std` the population standard deviation; for
  `_std` metrics smaller is better, so `compare` ranks them that way.
- Tags: `cv_folds`, `n_test` (all rows) and `n_train` (the largest training set). The confusion
  matrix is pooled over all folds, and `predictions.json` holds one entry per fold.
- Reproducibility checks compare the predictions hash and the summary metrics.

## Confusion matrices

Every classification run records its test-split confusion matrix as the tag `confusion_matrix`
(compact JSON, rows are true labels and columns predicted labels; it includes every class seen
in training, even one that never shows up in the test split). A matrix that would not fit in a
2000-character tag is left out and the run is unaffected. `modelbench show RUN` prints it as a
grid, and the dashboard's run page draws it as a heatmap shaded by each cell's share of its row,
so the diagonal reads as per-class recall. The tag travels with `export` and `import` like any
other. Regression runs have none.

## Notes and export

```python
from modelbench.export import export_json, export_metrics_csv, write_text_atomic

store.add_note(run_id, "baseline for the k sweep")  # allowed in any run status
store.add_experiment_note("demo", "why this experiment exists")  # once it has a run
write_text_atomic("runs.json", export_json(store, [run_id]))
export_metrics_csv(store, [run_id])
# run_id,experiment_id,metric,step,value
# run-000001,demo,accuracy,0,1.0
```

- Notes are free text (up to 10,000 characters), append-only, and can be added to running,
  completed, and failed runs. Tags stay system-managed and can only be set while a run is running.
- `export_json` writes `{"format": "modelbench-runs", "version": 1, "runs": [...]}` with sorted
  keys. Each run holds its config and hash, dataset version, environment, status and error, every
  metric point, artifact metadata, tags, and notes, so another tool can read it without the
  database. Runs appear in the order you pass them; an empty list or a repeated id is an error.
- `export_metrics_csv` writes one row per metric point with exact float values.
- `write_text_atomic` writes through a temporary file and replaces the target, so a failure keeps
  the old file. The file is created owner-readable only (mode 0600).

### Importing

```
modelbench --db other.db import runs.json            # creates other.db if needed
modelbench --db other.db import runs.json --dry-run  # check and report, write nothing
```

- Only the JSON export can be imported. The metrics CSV has no configs, so it cannot rebuild runs.
- The whole file is checked before anything is written: format and version, the shape of every
  run, and that each run's config hash, seed, experiment, and dataset agree with its config. One
  bad run rejects the file, and the runs go in as a single transaction, so nothing is half-imported.
- Imported runs get new ids from the receiving database and keep their timestamps, status,
  metrics, artifacts, tags, and notes. A tag `imported_from` holds the id the run had where it was
  exported; if the run was already imported once, the first source is kept.
- A run already present (same config, dataset version, status, timestamps, and metrics) is
  skipped, so importing a file twice adds nothing. Running `import` on the run's own database
  skips everything too.
- Unfinished (`running`) runs are refused, and a dataset version already stored with a different
  content hash stops the import. Files up to 50 MB and 10,000 runs are accepted.
- Artifact files are not copied: only their metadata, including the original path, travels.
- In Python: `modelbench.importer.import_file(store, path)` or `import_runs_json(store, text)`,
  each returning an `ImportReport` with `imported` and `skipped` pairs.

## Comparing runs

`compare_runs(store, [baseline_id, other_id, ...])` compares completed runs. For each run after
the first it reports parameter differences, metric deltas against the baseline (with
whether each change is an improvement), and whether the config, dataset version, seed, and
environment match. It also ranks all runs per metric.

- Ranking: exact ties share a rank and the next rank is skipped (1, 2, 2, 4).
- Direction: accuracy, precision, recall, F1, ROC-AUC, and R² are higher-is-better; log loss, MAE,
  MSE, and RMSE are lower-is-better. A metric name may carry a prefix such as `test/`. For any
  other metric pass `directions={"name": True or False}`; without it the metric is compared but
  not ranked, and a warning says so.
- Comparing runs on different dataset versions produces a warning, not an error.

### Paired bootstrap

`paired_bootstrap(scores_a, scores_b, seed=...)` gives a percentile confidence interval for the
mean per-example difference between two models scored on the same examples.

```python
from modelbench.compare import paired_bootstrap

a = [1] * 90 + [0] * 10  # model A is right on 90 of 100 examples
b = [1] * 70 + [0] * 30  # model B is right on 70 of the same 100
result = paired_bootstrap(a, b, seed=3)
result.mean_diff  # 0.2
result.ci_low > 0  # True: the 95% interval excludes zero
```

It works on per-example scores where higher is better: 0/1 correctness for accuracy, or negated
errors for MAE/RMSE-style comparisons. It cannot give an interval for metrics that are not
averages over examples, such as F1 or ROC-AUC. The result is repeatable for a given seed.

## Models

All classifiers share `fit(X, y)`, `predict(X)`, and `predict_proba(X)`. Probability rows follow
`classes_`, the sorted training labels, and ties always go to the first class in that order.

| Name | Behaviour |
|---|---|
| `majority` | Most frequent training class; probabilities are the class priors |
| `stratified` | Samples labels from the priors with a seeded generator; repeatable per `seed` |
| `knn` | `k`, `metric` (euclidean, manhattan), `weights` (uniform, distance); equal distances rank by training-row position |
| `gaussian_nb` | Per-class normal distributions per feature, log space, `var_smoothing` |
| `decision_tree` | CART with Gini impurity: `max_depth` (default 5, `None` for unlimited), `min_samples_split`, `min_samples_leaf`; leaves predict class fractions |
| `bagged_trees` | Averages the class fractions of `n_estimators` (default 25, at most 500) decision trees, each fitted to a bootstrap sample drawn from a generator derived from `seed` (the runner derives it from the run seed, as for `stratified`); takes the tree parameters. Bagging only: every split sees every feature |
| `random_forest` | `bagged_trees` whose every split looks at `max_features` randomly chosen features ("sqrt" by default, "log2", `None`, a whole number or a fraction); `n_estimators` defaults to 50, `max_depth` to unlimited. If the chosen features give no valid split the rest are tried. `oob_score_` is the accuracy on training rows using only trees that never saw them. The runner derives `seed` from the run seed |
| `gradient_boosting` | Boosted regression trees on the multinomial log loss: `n_estimators` (default 100, at most 1000), `learning_rate` (default 0.1, in (0, 1]), `max_depth` (default 3), `min_samples_split`, `min_samples_leaf`, `subsample` (default 1.0; below 1 each stage sees a seeded random share of the rows) and `seed`. One tree per class per stage with Friedman's Newton leaf values; probabilities are a softmax of the scores, so `log_loss` is meaningful. `train_loss_` lists the training log loss after each stage. `seed` only matters when `subsample` is below 1 (the runner still derives and records it) |
| `calibrated` | Wraps any classifier (`base`, with `base_params`) so its probabilities match observed frequencies. The base is fitted on `folds` (default 3) stratified, seeded folds; its held-out probabilities train one calibrator per class (`method`: `sigmoid`, Platt scaling, the default, or `isotonic`, which needs more rows), and the rows are renormalised. Prediction uses the base refitted on all rows. If the base takes a `seed` it gets one derived from this model's `seed`. `predict_proba_uncalibrated` returns the base's own probabilities. Outcomes are smoothed as Platt suggested, so probabilities are never exactly 0 or 1 |
| `perceptron` | Multiclass averaged perceptron: `max_iter` epochs (default 20, at most 10000) and `seed` for the visiting order (derived from the run seed by the runner). Features are standardised; the weights used are the average over all steps; `converged_` is true only when an epoch makes no mistakes (separable data). No learning rate exists because only the weights' direction matters. `predict_proba` is a softmax of the scores, ordered correctly but not calibrated, so `log_loss` is only a rough guide |
| `logistic_regression` | Multinomial (softmax) with an L2 penalty: `l2` (default 0.01), `max_iter`, `tol`; features are standardised internally |

`bagged_trees` also reports `oob_score_`. Its random samples are unchanged from earlier versions.

Regression models share `fit(X, y)` and `predict(X)` with numeric targets and are used with the
metrics `mae`, `mse`, `rmse` and `r2`. A model's kind decides which metrics are allowed, and
`stratify` is rejected for regression. `modelbench models` lists every model with its kind.

| Name | Behaviour |
|---|---|
| `mean` | Predicts the training mean; the baseline for `r2` |
| `knn_regressor` | Mean target of the `k` nearest rows (`k`, `metric`, `weights` as for `knn`); ties go to the earlier training row; an exact match overrides distance weighting. Scale features first |
| `decision_tree_regressor` | CART regression tree minimising squared error: `max_depth` (default 5, `null` for unlimited), `min_samples_split`, `min_samples_leaf`. Same deterministic split rules as `decision_tree`; leaves predict the mean target; `describe()` prints the rules |
| `random_forest_regressor` | Bootstrap regression trees whose splits look at `max_features` random features (default 1.0, all of them, which is plain bagging; smaller values can hurt when there are few features); `n_estimators` defaults to 50; predictions are the mean of the trees; `oob_score_` is the out-of-bag R² (None when it cannot be computed) |
| `gradient_boosting_regressor` | Squared-error boosting from the mean target: same parameters as `gradient_boosting`; each stage fits a tree to the residuals. `train_loss_` lists the training MSE per stage (it never rises when `subsample` is 1) and `staged_predict(X)` gives predictions after each stage |
| `linear_regression` | Closed-form least squares (Cholesky) on standardised features: `l2` (default 0). Constant columns get zero weight; collinear features with `l2=0` are an error that suggests `l2>0`. `coef_` and `intercept_` are in original units |

```python
from modelbench.models import create_model

model = create_model("knn", k=3)
model.fit([[0.0], [1.0], [10.0], [11.0]], ["a", "a", "b", "b"])
model.predict([[0.5], [10.5]])  # ['a', 'b']
```

k-NN uses features as given, so scale them first if their units differ.

```python
tree = create_model("decision_tree", max_depth=2)
tree.fit([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]], ["a", "a", "a", "b", "b", "b"])
print(tree.describe())
# if x0 <= 6.5:
#   predict 'a' (n=3)
#   else: predict 'b' (n=3)
```

Decision tree details: splits are `x <= threshold` with the threshold halfway between two
neighbouring distinct training values; equal-quality splits go to the lowest feature index and
then the lowest threshold, so the tree does not depend on row order. A node is split even when
no split lowers its impurity (XOR needs that), until it is pure, too small, or at `max_depth`.
`depth_`, `n_leaves_`, and `describe()` report the fitted tree.

Logistic regression details: the objective is mean cross-entropy plus `l2 / 2` times the squared
(standardised) weights, intercepts unpenalised. It is solved by accelerated gradient descent
with a step size taken from a bound on the objective's curvature, so there is no learning rate
to set. Training stops when the largest gradient entry is below `tol`, or after `max_iter`
iterations; `converged_` and `n_iter_` say which, and `coef_` and `intercept_` give the result
on the original feature scale. The penalty shrinks the weights, so even separable data can
leave a point misclassified at the default `l2`; lower it for a sharper boundary. Results
depend on row order only at rounding level and are identical for identical input. It is meant
for small tables, since the cost is `max_iter` x rows x classes x features.

Every model reports a float overflow (for example features near 1e300) as a `ModelError`
asking you to rescale, never as a raw `OverflowError`.

## Run store rules

- A run starts `running` and ends once as `completed` or `failed`. Only running runs accept
  metrics, artifacts, and tags.
- A metric is written once per (run, name, step). Rewriting history is an error, and a batch of
  metrics is stored all-or-nothing.
- Each run records its config hash, dataset version, seed, and environment fingerprint.
- Artifact records hold metadata only (name, kind, SHA-256, size, optional path), not file contents.

## Reproducibility notes

The full guide, with what each verdict means and what can break a run, is in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

- Sub-seeds come from SHA-256, never from Python's per-process `hash()`.
- Splits use Python's seeded Mersenne Twister. Results can differ across Python or library
  versions, so `environment_fingerprint()` records them and `compare_environments()` explains
  differences.
- Fix `PYTHONHASHSEED` before starting Python if anything depends on string-set ordering.
- A dataset version covers content and row order, so reordering rows creates a new version.

## Metrics

Pure-Python implementations with no dependencies:

- classification: accuracy, balanced accuracy, Matthews correlation (MCC), confusion matrix,
  precision, recall, and F1 (per class, macro, micro, weighted), ROC-AUC and average precision
  (both binary), log loss, Brier score and expected calibration error (binary), and the ROC
  and precision-recall curves and calibration bins behind them
- regression: MAE, MSE, RMSE, R²

In an experiment config the metric names are `accuracy`, `precision`, `recall`, `f1` (macro),
`precision_micro`, `precision_weighted`, `recall_micro`, `recall_weighted`, `f1_micro`,
`f1_weighted`, `balanced_accuracy`, `mcc`, `roc_auc`, `average_precision`, `log_loss`,
`brier_score`, `ece` (expected calibration error, 10 bins) for
classifiers and `mae`, `mse`, `rmse`, `r2` for regressors. `roc_auc` and `average_precision` need
exactly two classes, with both present in the test split; `brier_score` and `ece` need exactly
two classes in the training split (lower is better for both; they score the probability of
the second class). Balanced accuracy averages recall over
the classes present in the true labels, so a majority-class guess scores 1 / (number of classes).

Every formula is tested against small hand-computed examples.

```python
from modelbench.metrics import accuracy, precision_recall_f1

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
accuracy(y_true, y_pred)  # 0.3333...
precision_recall_f1(y_true, y_pred, average="macro")
```

## Development

```
pip install -e ".[dev]" -c requirements-dev.txt
python -m unittest discover -s tests
```

`requirements-dev.txt` pins the exact tool versions CI uses (Python 3.12 and newer); leave out
`-c requirements-dev.txt` to take the newest releases.

The dashboard script has its own tests: `node --test tests/js/dashboard.test.js` (Node.js 18 or
newer; also run by the normal test suite when `node` is on the path), and tests that drive the
pages in Chromium: `pip install -e ".[browser]"`, `playwright install chromium`, then
`python -m unittest tests.test_dashboard_browser`. Without Playwright those are skipped.

Python 3.10 or newer. The package has no runtime dependencies.

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat(cli): add sweep`). `python scripts/check_commits.py main..HEAD` checks a range, CI checks
every new commit, and `git config core.hooksPath .githooks` checks each message as you commit.

Without installing anything, run from the project folder. On Linux and macOS:

```
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m modelbench demo
```

On Windows (PowerShell), where the command is `python` or `py`, not `python3`:

```
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
python -m modelbench demo
```

Besides the example-based tests, `tests/test_properties.py` checks invariants over generated
inputs (metric ranges and symmetries, splits and folds partitioning the rows, every classifier's
probabilities, reproducibility of runs). Each case comes from a fixed seed, so a failure
repeats exactly and names its case.

The whole suite passes on Windows, Linux, and Python 3.11 to 3.13. To unzip on Windows use
`Expand-Archive modelbench.zip .` in PowerShell.

## License

MIT
