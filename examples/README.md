# Examples

Config files for the commands in the main README. The data is generated, not stored: every
example uses a seeded sample dataset, so nothing here needs a download or an account.

Run these from the project folder (use `python -m modelbench` if the `modelbench` command is
not installed; on Windows PowerShell set `$env:PYTHONPATH = "src"` first when running from source).

```
modelbench sample blobs --seed 1 --rows-per-class 40 -o blobs.csv
modelbench sample linear --seed 1 --rows 120 --noise 3 -o linear.csv

modelbench run examples/knn.json --data blobs.csv --label label
modelbench run examples/forest-cv.json --data blobs.csv --label label
modelbench sweep examples/knn-sweep.json --data blobs.csv --label label
modelbench search examples/knn-search.json --data blobs.csv --label label
modelbench halve examples/forest-halving.json --data blobs.csv --label label
modelbench nested examples/knn-nested.json --data blobs.csv --label label
modelbench sweep examples/ridge-sweep.json --data linear.csv --label y

modelbench list
modelbench report blobs-knn-sweep -o report.html
modelbench serve
```

| File | What it shows |
|---|---|
| `knn.json` | one experiment: a k-NN classifier with three metrics |
| `forest-cv.json` | 5-fold stratified cross-validation of bagged trees; per-fold scores, mean and spread |
| `knn-sweep.json` | a parameter grid, ranked |
| `knn-search.json` | seeded random search over a range of `k` and a choice of weighting |
| `forest-halving.json` | successive halving: 12 forests start at 3 trees, the best third get 9, the best of those 27 |
| `knn-nested.json` | nested cross-validation: pick `k` and the weighting inside each outer fold, then score the pick on rows it never saw |
| `ridge-sweep.json` | a ridge-penalty sweep on a regression dataset (use `--rank-by mae` for the error) |

Everything is deterministic: run a command twice and you get the same numbers, and
`modelbench repro RUN --data blobs.csv --label label` checks that a stored run still reproduces.
