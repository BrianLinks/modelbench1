"""The ordered and grouped split modes must work in every command that rebuilds a run's splits."""

import unittest

from modelbench.config import ExperimentConfig
from modelbench.datasets import Table
from modelbench.importance import permutation_importance
from modelbench.learning_curve import learning_curve
from modelbench.reproducibility import check_reproducibility
from modelbench.runner import refit, run_experiment
from modelbench.store import Store

ENV = {"python": "3.12.1", "lib:numpy": None}

# x drives the label, noise does not; "subject" (0 to 5) repeats every 6 rows
ROWS = tuple(
    (str(i * 0.5), str((i * 7919) % 13), str(i % 6), "lo" if i < 24 else "hi") for i in range(48)
)
TABLE = Table(("x", "noise", "subject", "label"), ROWS)

MODES = {
    "time_ordered": {"time_ordered": True, "test_fraction": 0.25},
    "time_ordered+cv": {"time_ordered": True, "cv_folds": 3},
    "time_ordered+gap": {"time_ordered": True, "time_gap": 2, "test_fraction": 0.25},
    "group": {"group_column": "subject", "test_fraction": 0.34},
    "group+cv": {"group_column": "subject", "cv_folds": 3},
}


def config(**overrides):
    fields = {
        "experiment_id": "modes",
        "model": "knn",
        "seed": 3,
        "dataset_id": "rows",
        "metrics": ("accuracy",),
        "params": {"k": 3},
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


class EveryCommandSeesTheSameSplitsTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)

    def test_refit_rebuilds_the_run_exactly(self):
        for name, extra in MODES.items():
            with self.subTest(mode=name):
                cfg = config(**extra)
                result = run_experiment(self.store, cfg, TABLE, "label", environment=ENV)
                fitted = refit(cfg, TABLE, "label")
                tested = sorted(i for f in fitted for i in f.fold.test_idx)
                self.assertEqual(len(tested), result.n_test)
                self.assertEqual(len(set(tested)), len(tested))

    def test_reproducibility_check_reproduces_every_mode(self):
        for name, extra in MODES.items():
            with self.subTest(mode=name):
                cfg = config(experiment_id=f"r-{name.replace('+', '-').replace('_', '-')}", **extra)
                result = run_experiment(self.store, cfg, TABLE, "label", environment=ENV)
                report = check_reproducibility(
                    self.store, result.run_id, TABLE, "label", environment=ENV
                )
                self.assertEqual(report.verdict, "reproduced")

    def test_importance_lists_the_right_features_in_every_mode(self):
        for name, extra in MODES.items():
            with self.subTest(mode=name):
                cfg = config(
                    experiment_id=f"i-{name.replace('+', '-').replace('_', '-')}",
                    model="knn",
                    **extra,
                )
                result = run_experiment(self.store, cfg, TABLE, "label", environment=ENV)
                importance = permutation_importance(
                    self.store, result.run_id, TABLE, "label", repeats=3
                )
                names = [f.feature for f in importance.features]
                # the group column is not a feature; without grouping it is an ordinary column
                grouped = "group_column" in extra
                expected = {"x", "noise"} | (set() if grouped else {"subject"})
                self.assertEqual(set(names), expected)

    def test_learning_curve_runs_in_every_mode(self):
        for name, extra in MODES.items():
            with self.subTest(mode=name):
                curve = learning_curve(config(**extra), TABLE, "label", fractions=(0.5, 1.0))
                self.assertEqual(len(curve.points), 2)
                self.assertLessEqual(curve.points[0].n_train, curve.points[1].n_train)


if __name__ == "__main__":
    unittest.main()
