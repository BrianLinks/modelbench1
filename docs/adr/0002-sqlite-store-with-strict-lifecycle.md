# 2. SQLite store with a strict run lifecycle

Status: accepted

## Context
Experiment results are only useful if they cannot be quietly rewritten after the fact.

## Decision
Runs live in one SQLite file with migrations tracked by `PRAGMA user_version`. A run starts
`running` and ends exactly once as `completed` or `failed`; only running runs accept metrics,
artifacts, and tags. A metric is written once per (run, name, step). Notes are append-only and
allowed in any status. The dashboard opens the file read-only at the SQLite level.

## Consequences
- History cannot be edited through the API, so comparisons stay trustworthy.
- A crash can leave a run in `running`; `recover` closes those explicitly.
- Importing needs its own insert path (`Store.import_runs`) because imported runs are already
  finished and keep their original timestamps.
