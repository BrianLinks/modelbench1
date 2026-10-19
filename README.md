# ModelBench

Local experiment tracking and model comparison, with no online ML platform and no paid API.

**Status: early development.** Implemented so far: the metrics module (`modelbench.metrics`).
Dataset versioning, run tracking, baseline models, comparison, reproducibility checks, and the
dashboard are planned and not built yet.

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
python -m unittest discover -s tests -t .
```

Python 3.10 or newer. The package has no runtime dependencies.

## License

MIT
