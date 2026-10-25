# ModelBench

Local experiment tracking and model comparison, with no online ML platform and no paid API.

**Status: early development.** Implemented so far:

- `modelbench.metrics`: classification and regression metrics
- `modelbench.ids`: strict dataset, experiment, run, and model identifiers
- `modelbench.seeding`: SHA-256 derived sub-seeds and global seeding
- `modelbench.environment`: environment fingerprints and drift comparison
- `modelbench.datasets`: content-hash dataset versions, CSV reading, seeded and stratified splits
- `modelbench.config`: validated experiment configs, config hashes, parameter grids
- `modelbench.store`: SQLite run store (datasets, runs, params, metric series, artifact metadata,
  tags) with migrations and a strict run lifecycle
- `modelbench.models`: pure-Python classifiers behind one interface and a name registry:
  `majority`, `stratified`, `knn`, `gaussian_nb`, `decision_tree`, `logistic_regression`
- `modelbench.compare`: parameter diffs, metric deltas, ranking, and a paired bootstrap
- `modelbench.run_comparison`: `compare_runs` over completed runs in the store
- `modelbench.runner`: `run_experiment` splits a dataset, fits a model, scores it, and records
  the run
- `modelbench.reproducibility`: re-run a stored run and compare it, and close runs a crash left
  in `running`
- `modelbench.export`: export runs as JSON or a metrics CSV, written atomically
- `modelbench.cli`: the `modelbench` command (also `python -m modelbench`)
- `modelbench.dashboard`: a read-only local web dashboard (`modelbench serve`)
- `modelbench.samples`: seeded synthetic datasets and a demo that fills a database with runs

Linear regression (and regression metrics in the runner) and import are planned and not built
yet.

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
with Store("runs.db") as store:
    result = run_experiment(store, config, table, "label")
    result.metrics   # {'accuracy': 1.0, 'f1': 1.0}
```

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
  training). Regression metrics are rejected because only classifiers exist so far.
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
report.verdict                    # 'reproduced'
report.explained_by_environment   # False
recover_stale_runs(store, older_than_seconds=3600)   # ids of runs it closed
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
| `list` | runs, newest first (`--experiment`, `--status`, `--limit`) |
| `show RUN` | one run in full: config, metrics, tags, files, notes |
| `compare RUN RUN ...` | the first run is the baseline (`--metrics a,b`) |
| `repro RUN --data F.csv --label COL` | re-run and check it reproduces (`--tolerance`) |
| `note RUN TEXT` | attach a note |
| `export RUN ... ` | JSON, or `--format csv`; `-o FILE` writes atomically |
| `sample NAME` | write a seeded synthetic dataset as CSV (`--seed`, `-o FILE`, size and noise options) |
| `demo` | run every model on three sample datasets and record the runs (`--seed`, `--rows-per-class`) |
| `serve` | read-only dashboard at http://127.0.0.1:8765/ (`--port`, `--host`, `--allow-network`) |
| `recover --older-than SECONDS` | mark long-running unfinished runs as failed |
| `dataset F.csv --id ID [--label COL]` | show a CSV's content-hash version |
| `models` | list the available models |

`run`, `list`, `show`, `compare`, `repro`, and `dataset` also take `--json`. The database
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

Pages: the run list (filter by experiment and status, tick runs to compare), one run in full
(metrics, a line chart for any metric logged over several steps, tags, files, environment,
notes), and a comparison with a bar chart per metric, rankings, and parameter and environment
differences. For the comparison page the oldest selected run is the baseline unless you pick
another. JSON is available at `/api/runs`, `/api/run/<id>`, and `/api/compare?run=...` (there
the first `run` is the baseline). The pages use no script and no outside resources, and work in
light and dark mode.

What it will and will not do:

- It is read-only. The database is opened in SQLite's read-only mode, so a write is refused by
  SQLite itself, and a database at an older schema is reported rather than upgraded. Only GET and
  HEAD are accepted.
- It listens on the loopback interface only. While it does, requests whose `Host` header names
  anything else are answered with 403, which blocks DNS-rebinding attacks from a web page you
  have open. `--host` with a non-loopback address is refused unless you add `--allow-network`,
  because there is no login: anyone who can reach the port can read every run, note, and path.
- Every value from the database or the URL is HTML-escaped, and the Content-Security-Policy
  forbids scripts, remote loads, and framing.
- Stop it with Ctrl-C.

## Notes and export

```python
from modelbench.export import export_json, export_metrics_csv, write_text_atomic

store.add_note(run_id, "baseline for the k sweep")   # allowed in any run status
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
- Importing a document back into a store is not built yet.

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

a = [1] * 90 + [0] * 10   # model A is right on 90 of 100 examples
b = [1] * 70 + [0] * 30   # model B is right on 70 of the same 100
result = paired_bootstrap(a, b, seed=3)
result.mean_diff           # 0.2
result.ci_low > 0          # True: the 95% interval excludes zero
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
| `logistic_regression` | Multinomial (softmax) with an L2 penalty: `l2` (default 0.01), `max_iter`, `tol`; features are standardised internally |

Regression models share `fit(X, y)` and `predict(X)` with numeric targets and are used with the
metrics `mae`, `mse`, `rmse` and `r2`. A model's kind decides which metrics are allowed, and
`stratify` is rejected for regression. `modelbench models` lists every model with its kind.

| Name | Behaviour |
|---|---|
| `mean` | Predicts the training mean; the baseline for `r2` |
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

- Sub-seeds come from SHA-256, never from Python's per-process `hash()`.
- Splits use Python's seeded Mersenne Twister. Results can differ across Python or library
  versions, so `environment_fingerprint()` records them and `compare_environments()` explains
  differences.
- Fix `PYTHONHASHSEED` before starting Python if anything depends on string-set ordering.
- A dataset version covers content and row order, so reordering rows creates a new version.

## Metrics

Pure-Python implementations with no dependencies:

- classification: accuracy, confusion matrix, precision, recall, and F1 (per class, macro, micro,
  weighted), ROC-AUC (binary), log loss
- regression: MAE, RMSE, R²

Every formula is tested against small hand-computed examples.

```python
from modelbench.metrics import accuracy, precision_recall_f1

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
accuracy(y_true, y_pred)                       # 0.3333...
precision_recall_f1(y_true, y_pred, average="macro")
```

## Development

```
pip install -e ".[dev]"
python -m unittest discover -s tests
```

Python 3.10 or newer. The package has no runtime dependencies.

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

The whole suite passes on Windows, Linux, and Python 3.11 to 3.13. To unzip on Windows use
`Expand-Archive modelbench.zip .` in PowerShell.

## License

MIT
