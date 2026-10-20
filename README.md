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

Baseline models, run comparison, reproducibility checks, import/export, the CLI, and the dashboard
are planned and not built yet. The store has no recovery for runs left in `running` after a crash;
that belongs with the reproducibility checker.

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
