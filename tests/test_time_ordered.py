import json
import tempfile
import unittest
from pathlib import Path

from modelbench.config import ExperimentConfig
from modelbench.datasets import (
    MAX_FOLDS,
    Table,
    expanding_window_indices,
    ordered_split,
    split_indices,
)
from modelbench.errors import ConfigError, DatasetError, SweepError
from modelbench.nested import run_nested
from modelbench.runner import load_folds, run_experiment
from modelbench.selection import run_selection
from modelbench.store import Store

ENV = {"python": "3.12.1", "lib:numpy": None}


def make(**overrides):
    fields = {
        "experiment_id": "ts",
        "model": "knn",
        "seed": 5,
        "dataset_id": "series",
        "metrics": ("accuracy",),
        "params": {"k": 1},
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


def drifting(n=40):
    """Row i has x = i; the label flips from a to b halfway through."""
    rows = [(str(i), "a" if i < n // 2 else "b") for i in range(n)]
    return Table(("x", "label"), tuple(rows))


class OrderedSplitTests(unittest.TestCase):
    def test_by_hand(self):
        self.assertEqual(ordered_split(10, 0.2), ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9]))
        self.assertEqual(ordered_split(5, 0.5), ([0, 1, 2], [3, 4]))  # round(2.5) = 2 test rows

    def test_the_test_set_is_never_empty_and_never_everything(self):
        self.assertEqual(ordered_split(10, 0.001), (list(range(9)), [9]))
        self.assertEqual(ordered_split(10, 0.999), ([0], list(range(1, 10))))

    def test_test_size_matches_the_shuffled_split(self):
        for n in (2, 7, 33, 100):
            for fraction in (0.1, 0.25, 0.5, 0.9):
                with self.subTest(n=n, fraction=fraction):
                    _, shuffled_test = split_indices(n, fraction, 1)
                    train, test = ordered_split(n, fraction)
                    self.assertEqual(len(test), len(shuffled_test))
                    self.assertEqual(train + test, list(range(n)))

    def test_validation(self):
        for n in (1, 0, True, 3.0):
            with self.subTest(n=n), self.assertRaises(DatasetError):
                ordered_split(n, 0.2)
        for fraction in (0, 1, -0.1, 1.5):
            with self.subTest(fraction=fraction), self.assertRaises(DatasetError):
                ordered_split(10, fraction)


class ExpandingWindowTests(unittest.TestCase):
    def test_by_hand_even_blocks(self):
        # 12 rows, 3 folds -> 4 blocks of 3
        self.assertEqual(
            expanding_window_indices(12, 3),
            [
                ([0, 1, 2], [3, 4, 5]),
                ([0, 1, 2, 3, 4, 5], [6, 7, 8]),
                (list(range(9)), [9, 10, 11]),
            ],
        )

    def test_by_hand_uneven_blocks_give_the_extra_rows_to_the_earlier_blocks(self):
        # 10 rows, 2 folds -> 3 blocks of 4, 3, 3
        self.assertEqual(
            expanding_window_indices(10, 2),
            [([0, 1, 2, 3], [4, 5, 6]), ([0, 1, 2, 3, 4, 5, 6], [7, 8, 9])],
        )

    def test_properties_for_many_shapes(self):
        for n in range(3, 40):
            for folds in range(2, min(MAX_FOLDS, n - 1) + 1):
                splits = expanding_window_indices(n, folds)
                self.assertEqual(len(splits), folds)
                tested = []
                previous_train = 0
                for train, test in splits:
                    self.assertEqual(train, list(range(len(train))))
                    self.assertEqual(test, list(range(len(train), len(train) + len(test))))
                    self.assertGreater(len(train), previous_train)  # the window only grows
                    self.assertTrue(max(train) < min(test))  # no future rows in training
                    previous_train = len(train)
                    tested.extend(test)
                self.assertEqual(tested, list(range(splits[0][0][-1] + 1, n)))
                sizes = [len(t) for _, t in splits]
                self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_validation(self):
        for n, folds in [(1, 2), (2, 2), (5, 5), (10, 1), (10, 51), (10, True), (10, 2.0)]:
            with self.subTest(n=n, folds=folds), self.assertRaises(DatasetError):
                expanding_window_indices(n, folds)
        self.assertEqual(len(expanding_window_indices(3, 2)), 2)  # exactly one row per block


class ConfigTests(unittest.TestCase):
    def test_default_leaves_the_dict_and_hash_alone(self):
        config = make()
        self.assertFalse(config.time_ordered)
        self.assertNotIn("time_ordered", config.to_dict())
        self.assertNotIn("time_ordered", config.canonical_json())
        self.assertEqual(ExperimentConfig.from_dict(config.to_dict()), config)

    def test_set_flag_is_hashed_and_round_trips(self):
        config = make(time_ordered=True)
        self.assertIs(config.to_dict()["time_ordered"], True)
        self.assertNotEqual(config.config_hash(), make().config_hash())
        self.assertNotEqual(config.config_hash(), make(time_ordered=True, cv_folds=3).config_hash())
        self.assertEqual(ExperimentConfig.from_dict(config.to_dict()), config)

    def test_validation(self):
        for bad in (1, 0, "yes", None):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                make(time_ordered=bad)
        with self.assertRaises(ConfigError):
            make(time_ordered=True, stratify=True)
        self.assertTrue(make(time_ordered=True, stratify=False).time_ordered)

    def test_unknown_keys_are_still_rejected(self):
        with self.assertRaises(ConfigError):
            ExperimentConfig.from_dict(make().to_dict() | {"time_order": True})


class LoadFoldsTests(unittest.TestCase):
    def test_single_split_trains_on_the_past_and_tests_on_the_end(self):
        table = drifting(40)
        (fold,) = load_folds(make(time_ordered=True), table, "label", None)
        self.assertEqual(fold.test_idx, list(range(32, 40)))
        self.assertEqual([row[0] for row in fold.train_X], [float(i) for i in range(32)])
        self.assertEqual(fold.train_y, ["a"] * 20 + ["b"] * 12)
        self.assertEqual(fold.test_y, ["b"] * 8)

    def test_the_seed_does_not_change_an_ordered_split(self):
        table = drifting(40)
        first = load_folds(make(time_ordered=True, seed=1), table, "label", None)
        second = load_folds(make(time_ordered=True, seed=999), table, "label", None)
        self.assertEqual(first, second)

    def test_expanding_windows_with_cv_folds(self):
        table = drifting(40)
        folds = load_folds(make(time_ordered=True, cv_folds=4), table, "label", None)
        expected = expanding_window_indices(40, 4)
        self.assertEqual([f.test_idx for f in folds], [t for _, t in expected])
        self.assertEqual([len(f.train_y) for f in folds], [len(t) for t, _ in expected])
        for fold in folds:
            self.assertLess(max(row[0] for row in fold.train_X), min(row[0] for row in fold.test_X))

    def test_unordered_configs_are_unchanged(self):
        table = drifting(40)
        (fold,) = load_folds(make(), table, "label", None)
        _, test = split_indices(40, 0.2, 5)
        self.assertEqual(fold.test_idx, test)

    def test_preprocessing_is_fitted_on_the_past_only(self):
        values = [0, 2, 4, 6, 100, 100, 100, 100, 100, 100]
        rows = tuple((str(v), "a" if i % 2 else "b") for i, v in enumerate(values))
        table = Table(("x", "label"), rows)
        config = make(
            time_ordered=True,
            test_fraction=0.3,
            preprocess=_standardize(),
        )
        (fold,) = load_folds(config, table, "label", None)
        # train rows are 0, 2, 4, 6, 100, 100, 100: the test rows' 100 must not shift the scaling
        train_values = [0, 2, 4, 6, 100, 100, 100]
        mean = sum(train_values) / 7
        self.assertAlmostEqual(sum(row[0] for row in fold.train_X) / 7, 0.0, places=9)
        std = (sum((v - mean) ** 2 for v in train_values) / 7) ** 0.5
        self.assertAlmostEqual(fold.test_X[0][0], (100 - mean) / std, places=9)


def _standardize():
    from modelbench.preprocess import PreprocessSpec

    return PreprocessSpec(scale="standardize")


class RunTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)

    def run_it(self, config, table, **kwargs):
        return run_experiment(self.store, config, table, "label", environment=ENV, **kwargs)

    def test_a_drifting_series_scores_worse_ordered_than_shuffled(self):
        table = drifting(40)
        shuffled = self.run_it(make(model="majority", params={}), table)
        ordered = self.run_it(make(model="majority", params={}, time_ordered=True), table)
        # train = rows 0..31 (20 a, 12 b) -> predicts a; test = rows 32..39, all b
        self.assertEqual(ordered.metrics["accuracy"], 0.0)
        self.assertGreater(shuffled.metrics["accuracy"], 0.0)
        self.assertEqual((ordered.n_train, ordered.n_test), (32, 8))

    def test_expanding_window_cv_by_hand(self):
        # folds over 40 rows in 5 blocks of 8: tests are rows 8-15, 16-23, 24-31, 32-39.
        # The majority of each past is "a" until it holds more b's: past = 0..7 (all a),
        # 0..15 (all a), 0..23 (20 a, 4 b), 0..31 (20 a, 12 b) -> always "a".
        # Test labels: rows 8-15 a, 16-23 b... rows < 20 are a, so 8-15 all a, 16-23 = 4 a + 4 b,
        # 24-31 all b, 32-39 all b -> accuracies 1, 0.5, 0, 0.
        config = make(model="majority", params={}, time_ordered=True, cv_folds=4)
        result = self.run_it(config, drifting(40))
        self.assertAlmostEqual(result.metrics["accuracy"], (1 + 0.5 + 0 + 0) / 4)
        points = [p for p in self.store.get_metrics(result.run_id) if p.name == "fold/accuracy"]
        self.assertEqual([p.value for p in points], [1.0, 0.5, 0.0, 0.0])

    def test_predictions_file_lists_the_ordered_test_rows(self):
        config = make(time_ordered=True, cv_folds=3)
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_it(config, drifting(40), artifact_dir=tmp)
            saved = json.loads((Path(tmp) / result.run_id / "predictions.json").read_text())
        self.assertEqual(
            [f["test_indices"] for f in saved["folds"]],
            [t for _, t in expanding_window_indices(40, 3)],
        )

    def test_identical_ordered_runs_are_reused_and_differ_from_shuffled(self):
        table = drifting(40)
        config = make(time_ordered=True)
        first = self.run_it(config, table, reuse_existing=True)
        again = self.run_it(config, table, reuse_existing=True)
        self.assertTrue(again.reused)
        self.assertEqual(again.run_id, first.run_id)
        other = self.run_it(make(), table, reuse_existing=True)
        self.assertFalse(other.reused)

    def test_row_order_changes_the_dataset_version_so_reordering_is_visible(self):
        table = drifting(40)
        reversed_table = Table(table.columns, tuple(reversed(table.rows)))
        config = make(time_ordered=True)
        first = self.run_it(config, table)
        second = self.run_it(config, reversed_table)
        self.assertNotEqual(
            self.store.get_run(first.run_id).dataset_version,
            self.store.get_run(second.run_id).dataset_version,
        )


class OtherCommandsRefuseTests(unittest.TestCase):
    def test_nested_and_selection_say_so_instead_of_shuffling(self):
        table = drifting(40)
        cfg = make(time_ordered=True)
        with self.assertRaises(SweepError) as raised:
            run_nested(cfg, [{"k": 1}, {"k": 3}], table, "label", outer_folds=3, inner_folds=2)
        self.assertIn("time_ordered", str(raised.exception))
        with self.assertRaises(SweepError) as raised:
            run_selection(cfg, table, "label", folds=3)
        self.assertIn("time_ordered", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
