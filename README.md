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
  `majority`, `stratified`, `knn`, `gaussian_nb`
- `modelbench.compare`: parameter diffs, metric deltas, ranking, and a paired bootstrap
- `modelbench.run_comparison`: `compare_runs` over completed runs in the store
- `modelbench.runner`: `run_experiment` splits a dataset, fits a model, scores it, and records
  the run
- `modelbench.reproducibility`: re-run a stored run and compare it, and close runs a crash left
  in `running`

Linear/logistic regression and a decision tree, import/export, the CLI, and the dashboard are
planned and not built yet.

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

```python
from modelbench.models import create_model

model = create_model("knn", k=3)
model.fit([[0.0], [1.0], [10.0], [11.0]], ["a", "a", "b", "b"])
model.predict([[0.5], [10.5]])  # ['a', 'b']
```

k-NN uses features as given, so scale them first if their units differ.

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

## License

MIT
