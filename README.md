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

Linear/logistic regression and a decision tree, an experiment runner, reproducibility checks,
import/export, the CLI, and the dashboard are planned and not built yet. The store has no recovery
for runs left in `running` after a crash; that belongs with the reproducibility checker.

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
