# Python interface

Everything the command line does is available as plain Python functions. Each block below runs
in order, in one session, in an empty folder; `tests/test_docs.py` executes them and the
`assert` lines are real checks. Nothing here needs a network connection or a paid service.

## Data and configuration

A dataset is a `Table` of strings: named columns and rows of equal width. `read_csv` builds one
from a UTF-8 file (a ragged row is an error, not silently padded). An experiment is an
`ExperimentConfig`, validated when it is created, so a bad value fails before anything runs.

```python
from modelbench.config import ExperimentConfig
from modelbench.datasets import Table

rows = [(f"{i * 0.1:.1f}", "0", "a") for i in range(20)]
rows += [(f"{10 + i * 0.1:.1f}", "5", "b") for i in range(20)]
table = Table(("x1", "x2", "label"), tuple(rows))

config = ExperimentConfig(
    experiment_id="docs",
    model="knn",
    seed=7,
    dataset_id="clusters",
    metrics=("accuracy", "f1"),
    params={"k": 3},
)
assert config.config_hash() == ExperimentConfig.from_dict(config.to_dict()).config_hash()
```

The config hash covers every setting that changes results and ignores the `experiment_id`
label, so two experiments with the same settings share a hash. `test_fraction` (default 0.2),
`stratify` (default false), `cv_folds` (default none, 2 to 50) and `time_ordered` (default false)
shape the evaluation.

## Running an experiment

`run_experiment` splits the table, fits the model, scores it, and records everything in a
`Store`. The result carries the metrics, the training and test sizes, and the predictions hash.

```python
from modelbench.runner import run_experiment
from modelbench.store import Store

store = Store("runs.db")  # created if missing; ":memory:" keeps it in RAM
result = run_experiment(store, config, table, "label")
assert result.run_id == "run-000001"
assert result.metrics == {"accuracy": 1.0, "f1": 1.0}
assert (result.n_train, result.n_test) == (32, 8)

again = run_experiment(store, config, table, "label", reuse_existing=True)
assert again.reused and again.run_id == result.run_id
```

Add `cv_folds` for cross-validation. The run's `accuracy` is then the mean over folds,
`accuracy_std` the spread, and each fold's score is stored as the series `fold/accuracy`.

```python
from dataclasses import replace

cv = replace(config, experiment_id="docs-cv", cv_folds=4, stratify=True)
cv_result = run_experiment(store, cv, table, "label")
assert set(cv_result.metrics) == {"accuracy", "accuracy_std", "f1", "f1_std"}
assert cv_result.n_test == 40  # every row is tested exactly once
```

A regression model uses numeric targets and regression metrics:

```python
line = Table(("x", "y"), tuple((str(i), repr(2.0 * i + 1.0)) for i in range(30)))
regression = ExperimentConfig(
    "docs-line", "linear_regression", seed=1, dataset_id="line", metrics=("r2", "mae"), params={}
)
fit = run_experiment(store, regression, line, "y")
assert abs(fit.metrics["r2"] - 1.0) < 1e-9
```

Text categories and missing values need a `preprocess` setting. It is fitted on each training
split only, so the test rows never influence it.

```python
from modelbench.preprocess import PreprocessSpec

towns = Table(
    ("town", "size", "label"),
    tuple(
        (("Oslo", "Rome", "Lima")[i % 3], "NA" if i % 5 == 0 else str(i), "abc"[i % 3])
        for i in range(30)
    ),
)
prepared = ExperimentConfig(
    "docs-towns",
    "decision_tree",
    seed=1,
    dataset_id="towns",
    metrics=("accuracy",),
    params={},
    stratify=True,
    preprocess=PreprocessSpec(categorical=("town",), impute="median", scale="standardize"),
)
towns_result = run_experiment(store, prepared, towns, "label")
assert towns_result.metrics["accuracy"] == 1.0  # the town decides the label
assert store.get_run(towns_result.run_id).tags["n_features"] == "4"  # 3 towns + size
```

## Sweeps

`run_sweep` runs every combination of a parameter grid as one experiment and ranks the runs by a
metric. A combination that fails is recorded with its error and the sweep carries on.

```python
from modelbench.sweep import run_sweep

sweep_config = ExperimentConfig(
    "docs-sweep", "knn", seed=7, dataset_id="clusters", metrics=("accuracy",), params={}
)
sweep = run_sweep(store, sweep_config, {"k": [0, 1, 3]}, table, "label")
assert [e.params for e in sweep.entries] == [{"k": 0}, {"k": 1}, {"k": 3}]
assert sweep.failed[0].error is not None  # k = 0 is not a valid neighbour count
best = sweep.best()
assert best is not None and best.params == {"k": 1}  # ties go to the earliest run
```

## Feature importance

`permutation_importance` rebuilds the model of a stored run, checks it against the run's
predictions hash, then shuffles one feature at a time among the test rows and reports how much
the metric got worse. It refuses data that is not the run's own, so the answer is always about
the model that actually ran.

```python
from modelbench.importance import permutation_importance

importance = permutation_importance(store, result.run_id, table, "label", repeats=5)
assert importance.metric == "accuracy" and importance.baseline == 1.0
assert {f.feature for f in importance.features} == {"x1", "x2"}
assert importance.features[0].importance > 0  # shuffling the most important feature hurts
```

## Learning curves

`learning_curve` fits a config on growing fractions of its training rows and scores each fit on
the rows it learned and on the test rows. Nothing is stored.

```python
from modelbench.learning_curve import learning_curve

curve = learning_curve(
    ExperimentConfig("docs-curve", "knn", 7, "clusters", ("accuracy",), {"k": 1}),
    table,
    "label",
    fractions=(0.5, 1.0),
)
assert [p.n_train for p in curve.points] == [16, 32]
assert [p.train_score for p in curve.points] == [1.0, 1.0]  # a 1-NN knows its own rows
assert curve.points[-1].test_score == 1.0
```

## Reading the store

```python
record = store.get_run(result.run_id)
assert record.status == "completed" and record.params == {"k": 3}
assert store.final_metrics(result.run_id)["accuracy"] == 1.0
assert record.tags["n_test"] == "8"

ids = [r.run_id for r in store.list_runs("docs")]
assert ids == [result.run_id]
assert store.find_runs_by_config(config.config_hash(), record.dataset_version) == [result.run_id]

store.add_note(result.run_id, "baseline")  # notes are allowed in any run status
assert [n.text for n in store.get_notes(result.run_id)] == ["baseline"]
```

Metrics are write-once and a finished run cannot change, so what you read back is what was
recorded. Pass `read_only=True` to `Store` to open a database that must not be modified.

## Comparing runs

```python
from modelbench.run_comparison import compare_runs

majority = replace(config, experiment_id="docs-baseline", model="majority", params={})
baseline = run_experiment(store, majority, table, "label")
comparison = compare_runs(store, [baseline.run_id, result.run_id])
assert comparison.baseline == baseline.run_id
assert comparison.rankings[0].ranked[0].run_id == result.run_id  # kNN beats the baseline
assert comparison.to_dict()["values"][result.run_id]["accuracy"] == 1.0
```

`paired_bootstrap` in `modelbench.compare` gives a confidence interval for the difference between
two models scored on the same examples.

## Checking reproducibility

```python
from modelbench.reproducibility import check_reproducibility

report = check_reproducibility(store, result.run_id, table, "label")
assert report.reproduced and report.predictions_match
```

The run is repeated in a scratch database and compared with what was stored. Changed data gives
the verdict `dataset_changed` without re-running.

## Exporting and importing

```python
from modelbench.export import export_json
from modelbench.importer import import_runs_json

document = export_json(store, [result.run_id, baseline.run_id])
other = Store(":memory:")
report = import_runs_json(other, document)
assert len(report.imported) == 2 and not report.skipped
assert other.final_metrics("run-000001") == store.final_metrics(result.run_id)
assert len(import_runs_json(other, document).skipped) == 2  # importing twice adds nothing
```

An import is validated completely before anything is written and then inserted in one
transaction, so a bad document leaves the database untouched.

## Models

Models are looked up by name. Classifiers have `fit`, `predict` and `predict_proba`;
regressors have `fit` and `predict`.

```python
from modelbench.models import all_models, create_model, create_regressor, model_kind

assert model_kind("knn") == "classification" and model_kind("linear_regression") == "regression"
assert "perceptron" in all_models()

model = create_model("knn", k=1)
model.fit([[0.0], [1.0], [10.0], [11.0]], ["a", "a", "b", "b"])
assert model.predict([[0.4], [10.6]]) == ["a", "b"]
assert model.predict_proba([[0.4]])[0] == [1.0, 0.0]

ridge = create_regressor("linear_regression", l2=0.0)
ridge.fit([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0])
assert abs(ridge.predict([[3.0]])[0] - 7.0) < 1e-9
```

## Errors

Every error the package raises on purpose derives from `ModelBenchError`; most are also
`ValueError`, and `StoreError` is not. Catch the base class to handle all of them.

```python
from modelbench.errors import ConfigError, ModelBenchError

try:
    ExperimentConfig("docs", "knn", seed=1, dataset_id="d", metrics=("nonsense",), params={})
except ModelBenchError as error:
    assert isinstance(error, ConfigError) and "nonsense" in str(error)
else:
    raise AssertionError("an unknown metric must be rejected")

store.close()
other.close()
```
