# 5. A versioned export format and all-or-nothing import

Status: accepted

## Context
Results need to move between machines, and a half-imported file would leave a database that
looks complete but is not.

## Decision
Export is a self-describing JSON document (`format`, `version`, runs with full configs). Import
validates the entire document first, including that each run's config hash, seed, experiment,
and dataset agree with its config, then inserts everything in one transaction. New ids are
assigned by the receiving database and the old id is kept in an `imported_from` tag. A run is
considered already present when its config hash, dataset version, status, timestamps, and metrics
all match, which makes importing the same file twice a no-op.

## Consequences
- One invalid run rejects the whole file, which is stricter than skipping it, but never leaves
  partial data.
- Artifact files are not transported, only their metadata.
- A change to the format requires a new `version` and an explicit reader for the old one.
