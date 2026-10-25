# 1. Standard library only

Status: accepted

## Context
The tool must run on a laptop with nothing installed, with no paid service and no network.
Numerical libraries would make models faster but would also tie results to library versions.

## Decision
The package depends on nothing outside the Python standard library. Models (k-NN, Gaussian
naive Bayes, a CART tree, logistic regression, linear regression) are written in plain Python.
Development tools (pytest, ruff, mypy) are optional extras.

## Consequences
- Installation is trivial and results do not depend on a BLAS build.
- Models are slow on large data: sample datasets and the demo are capped to keep runs quick.
- The environment fingerprint still records `numpy` and `scikit-learn` versions if present,
  because users may compare these runs with runs made elsewhere.
