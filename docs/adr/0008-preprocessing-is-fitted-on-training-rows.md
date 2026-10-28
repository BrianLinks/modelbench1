# 8. Preprocessing is part of the config and fitted on training rows only

Status: accepted

## Context
Real tables have missing cells, text categories and columns on different scales. Cleaning them
by hand before every run hides decisions outside the record, and fitting a scaler or an
imputer on the whole table lets information about the test rows leak into the features.

## Decision
- A `preprocess` block in the config says what to do: which columns are categorical, how to
  impute, whether to add missing-value indicators, and which scaling to use.
- Every cell is parsed once before any run exists, so a bad value is reported up front.
- For each training split (each fold, when cross-validating) the preprocessor is fitted on the
  training rows only and then applied to the training and the test rows. A category never seen in
  training becomes all zeros rather than an error.
- The block is part of the config hash only when it is set, so configs written before it existed
  keep their hashes.
- Runs record `n_features`, the width of the data after preprocessing.

## Consequences
- Two runs differing only in preprocessing have different config hashes and are compared like any
  other pair of runs.
- Permutation importance counts each one-hot column as a feature, which the README says.
- Unusual cleaning (joins, parsing dates, outlier rules) stays outside the tool, on purpose.
