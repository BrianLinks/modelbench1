# Database schema

ModelBench keeps everything in one SQLite file (default `modelbench.db`). The schema version is
stored in SQLite's `PRAGMA user_version` and equals the number of entries in
`store.MIGRATIONS` (currently 3). Opening a database writable applies any missing migration, each
inside its own transaction; the read-only dashboard refuses a database at an older version
instead of upgrading it. Foreign keys are enforced (`PRAGMA foreign_keys = ON`).

`tests/test_schema_doc.py` compares the tables and columns below with a freshly created
database, so this page cannot drift from the code.

## Tables

### datasets

One row per dataset version. The version is a hash of the file's content, so a changed file is a
new version and old runs keep pointing at what they actually used.

| column | type | null | meaning |
|---|---|---|---|
| dataset_id | TEXT | no | name given by the user (part of the primary key) |
| version | TEXT | no | 12 hex characters of the content hash (part of the primary key) |
| sha256 | TEXT | no | SHA-256 of the dataset content |
| n_rows | INTEGER | no | number of data rows |
| columns_json | TEXT | no | JSON list of column names |
| label_column | TEXT | yes | the label or target column, when one was named |
| registered_at | TEXT | no | UTC timestamp, `YYYY-MM-DDTHH:MM:SSZ` |

### experiments

A label that groups runs. It is created the first time a run uses it.

| column | type | null | meaning |
|---|---|---|---|
| experiment_id | TEXT | yes | the experiment name (primary key) |
| created_at | TEXT | no | UTC timestamp |

### runs

One row per run. `seq` is the insertion order; `run_id` is `run-` plus `seq` zero-padded to six
digits and is filled in by the same transaction that creates the row.

| column | type | null | meaning |
|---|---|---|---|
| seq | INTEGER | yes | autoincrement primary key |
| run_id | TEXT | yes | unique id such as `run-000012` |
| experiment_id | TEXT | no | references `experiments` |
| config_hash | TEXT | no | hash of the settings that affect results (ignores the experiment label) |
| config_json | TEXT | no | the full experiment config as JSON |
| dataset_id | TEXT | no | with `dataset_version`, references `datasets` |
| dataset_version | TEXT | no | the version the run used |
| seed | INTEGER | no | the run seed |
| status | TEXT | no | `running`, `completed` or `failed` (enforced by a CHECK constraint) |
| started_at | TEXT | no | UTC timestamp |
| finished_at | TEXT | yes | set when the run completes or fails |
| error | TEXT | yes | the error text of a failed run |
| environment_json | TEXT | no | JSON fingerprint of the Python version and installed libraries |

A run moves only `running` to `completed` or `running` to `failed`; once finished it cannot be
changed.

### params

The config's top-level `params`, one row per key.

| column | type | null | meaning |
|---|---|---|---|
| run_id | TEXT | no | references `runs`, deleted with the run |
| key | TEXT | no | parameter name (primary key with `run_id`) |
| value_json | TEXT | no | the value as JSON |

### metrics

A series per metric name. A single-split run writes each metric at step 0. A cross-validated
run writes `fold/<metric>` at steps 0 to k-1 and the mean and spread at step 0. Rows are
write-once: logging the same name and step again is an error.

| column | type | null | meaning |
|---|---|---|---|
| run_id | TEXT | no | references `runs`, deleted with the run |
| name | TEXT | no | metric name |
| step | INTEGER | no | zero or more (primary key with `run_id` and `name`) |
| value | REAL | no | a finite number |

### artifacts

Metadata about files a run produced (for example `predictions.json`). The files themselves live
on disk, not in the database.

| column | type | null | meaning |
|---|---|---|---|
| run_id | TEXT | no | references `runs`, deleted with the run |
| name | TEXT | no | artifact name (primary key with `run_id`) |
| kind | TEXT | no | what sort of file it is |
| sha256 | TEXT | no | SHA-256 of the file content |
| size_bytes | INTEGER | no | zero or more |
| path | TEXT | yes | where the file was written |

### tags

Free-form key and value facts about a run, such as `n_train`, `n_test`, `predictions_sha256`,
`model_seed`, `cv_folds`, `time_ordered`, `group_column`, and `confusion_matrix`. A tag value is at most 2000 characters.

| column | type | null | meaning |
|---|---|---|---|
| run_id | TEXT | no | references `runs`, deleted with the run |
| key | TEXT | no | tag name (primary key with `run_id`) |
| value | TEXT | no | tag value |

### notes

Text notes attached to a run, allowed in any run status (added in migration 2).

| column | type | null | meaning |
|---|---|---|---|
| seq | INTEGER | yes | autoincrement primary key; notes are listed in this order |
| run_id | TEXT | no | references `runs`, deleted with the run |
| created_at | TEXT | no | UTC timestamp |
| text | TEXT | no | the note, at most 10000 characters |

### experiment_notes

Text notes attached to an experiment, which exists once it has a run (added in migration 3).

| column | type | null | meaning |
|---|---|---|---|
| seq | INTEGER | yes | autoincrement primary key; notes are listed in this order |
| experiment_id | TEXT | no | references `experiments`, deleted with the experiment |
| created_at | TEXT | no | UTC timestamp |
| text | TEXT | no | the note, at most 10000 characters |

## Indexes

| index | on | used for |
|---|---|---|
| idx_runs_experiment | runs(experiment_id) | listing an experiment's runs |
| idx_runs_config | runs(config_hash) | finding earlier runs with identical settings (`--reuse`, sweeps) |
| idx_notes_run | notes(run_id) | reading a run's notes |
| idx_experiment_notes_experiment | experiment_notes(experiment_id) | reading an experiment's notes |

## Changing the schema

Append a new SQL script to `store.MIGRATIONS`; never edit an existing one. Update this page and
the test will tell you if a table or column is missing from it.
