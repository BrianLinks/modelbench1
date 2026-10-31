# Reproducibility guide

ModelBench aims for one thing: given the same data, the same config, and the same software, a run
gives the same predictions, and `repro` can show it. This guide says what that rests on, what can
break it, and how to read the result when it does.

## What a run depends on

| Input | How it is pinned |
|---|---|
| The data | A dataset version: the first 12 characters of a SHA-256 over the column names and every row in order. Row order is part of the content, so reordering rows is a new version. The dataset id is not hashed, and neither is the choice of label column; that choice is recorded with the columns below. |
| The settings | The config hash: SHA-256 of the canonical config (model, params, seed, split settings, metrics, `cv_folds`, `time_ordered`, preprocessing). The experiment label is not part of it. `1` and `1.0` hash differently. |
| Every random choice | One seed in the config. Each purpose draws from its own generator, derived as SHA-256 of the seed and a purpose name, so adding a new use of randomness never shifts an existing one. |
| The columns | The label and the ordered feature columns, recorded in the `columns_sha256` tag. |
| The software | An environment fingerprint stored with the run (below). |

The purposes are `split` (train/test), `kfold` (cross-validation folds), `model` or `model-fold-N`
(a model's own seed, recorded as the `model_seed` tag), `learning-curve`, and `permutation`
(feature importance). None of them uses Python's `hash()`, which changes between processes.

The result of a run is summarised by the `predictions_sha256` tag: a hash of every prediction in
order (for cross-validation, every fold's predictions in fold order). Two runs with the same
hash made the same predictions.

## Checking a run

```
modelbench repro run-000032 --data moons.csv --label label
```

The run's stored config is run again on the data you pass, in a scratch in-memory store, so the
check never adds to your history. Pass the same `--features` the original run used. Exit code 0
means reproduced, 1 means not reproduced, 2 means the check could not be made (a bad argument, or
a run that is unfinished or has no predictions hash).

A run reproduces only when the predictions hash is identical and every stored summary metric is
within `--tolerance` (absolute, default 0). Different predictions are never reproduced, even when
the metrics agree, because the model behaved differently. Per-fold series are covered by the hash
and are not compared one by one.

| Verdict | Meaning |
|---|---|
| `reproduced` | Same predictions, metrics within tolerance. |
| `not_reproduced` | Something differs. The report lists each metric's old and new value. |
| `dataset_changed` | The data you passed is not the version the run used. Nothing was re-run. |
| `rerun_failed` | The re-run raised an error, which is shown. |

## What can break it

These are the real causes of a run that does not reproduce on the same data.

- **A different Python version.** The generator behind `random` is stable for plain draws, but
  helpers built on it (`shuffle`, `sample`, `choice`) have changed between releases, so a split can
  differ. This changes the predictions hash.
- **A different operating system or CPU.** Functions such as `exp` and `log` come from the
  platform's math library and can differ in the last digit. Models that use them (logistic
  regression, naive Bayes, `log_loss`, the perceptron's softmax) can then differ by a tiny amount.
  This usually leaves predictions and the hash alone but moves metrics by around 1e-12, so
  pass a small `--tolerance` such as `1e-9` when comparing across machines.
- **A different interpreter.** PyPy and CPython can differ in numeric and ordering behaviour.
- **An unfixed `PYTHONHASHSEED`.** Anything that iterates a set of strings can change order from
  one process to the next. ModelBench sorts where order matters, but set the variable if your own
  code around it depends on set order.
- **Changed data or columns.** A re-ordered or edited file is a new dataset version; different
  feature columns give a different `columns_sha256`.

The fingerprint stores `python`, `implementation`, `system`, `machine`, `pythonhashseed`, and the
installed version of numpy and scikit-learn (ModelBench imports neither; they are recorded because
a user's own code may). `repro` compares the stored fingerprint with the current one. When a run
does not reproduce and the two differ, the report says `explained_by_environment` and lists each
difference with what it can affect. That is a candidate cause, not a proof: the tool cannot tell
whether the difference was the reason.

## If a run does not reproduce

1. Read the verdict. `dataset_changed` means you passed different data: find the file the run
   used (`modelbench show RUN` prints the dataset version; `modelbench dataset FILE --id ID`
   prints a file's version).
2. Compare the environment differences. A different Python minor version or operating system is
   the usual reason.
3. If only metrics moved, and the predictions hash matched, raise `--tolerance` to the precision
   you care about. If the hash differs, the model's predictions really changed.
4. If nothing differs, treat it as a bug: re-run the check once more, then look at the run's
   `model_seed` tag and config with `show`.

## What ModelBench does not promise

It does not promise bit-identical floating point across machines, and it does not claim a run
reproduces until `repro` has shown it. A reproduced result on one machine says the same inputs give
the same outputs there; the fingerprint is how you tell when a second machine is a different
setup.
