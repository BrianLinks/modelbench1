# 6. Cross-validation records the folds and a summary

Status: accepted

## Context
A single train/test split on a small table can rank two models the wrong way round by luck.
Cross-validation reduces that, but the rest of the tool (the store, `compare`, `repro`,
export and import) is built around one value per metric per run.

## Decision
- `cv_folds` in the config turns a run into k-fold cross-validation; the run is still one run.
- Each fold's score is stored as the series `fold/<metric>` (step = fold number). The run's
  headline `<metric>` is the mean over folds and `<metric>_std` the population standard
  deviation, both at step 0, so every existing consumer sees ordinary metrics.
- Every row is tested exactly once. `n_test` is the number of rows and `n_train` the largest
  training set. The confusion matrix is pooled over folds, and the predictions hash covers
  every fold's predictions in fold order.
- Each fold gets its own model seed derived from the run seed and the fold number.
- For `_std` metrics smaller is better. `repro` compares the summary metrics and the
  predictions hash, not the per-fold series, which the hash already covers.
- `cv_folds` is part of the config hash only when set, so hashes of runs made before it
  existed do not change.

## Consequences
- Stored runs from version 0.1.0 stay valid and keep their hashes.
- Population (not sample) standard deviation describes these k folds themselves, and is
  not an estimate of the spread of a larger population; it should not be read as a
  confidence interval.
- Folds from a stratified run differ in class counts by at most one, so very rare classes
  can still be missing from a fold; metrics that need both classes (`roc_auc`) are rejected
  before the run starts when that happens.
