import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from modelbench.config import ExperimentConfig
from modelbench.datasets import Table, group_kfold_indices, group_split
from modelbench.errors import ConfigError, DatasetError, RunnerError, SweepError
from modelbench.nested import run_nested
from modelbench.runner import COLUMNS_TAG, columns_signature, load_folds, run_experiment
from modelbench.seeding import rng_for
from modelbench.selection import run_selection
from modelbench.store import Store

ENV = {"python": "3.12.1", "lib:numpy": None}


def make(**overrides):
    fields = {
        "experiment_id": "grp",
        "model": "knn",
        "seed": 9,
        "dataset_id": "people",
        "metrics": ("accuracy",),
        "params": {"k": 1},
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


def interleaved(sizes):
    """Group names repeated by size, in an interleaved row order."""
    remaining = dict(sizes)
    out = []
    while any(remaining.values()):
        for name in sizes:
            if remaining[name]:
                out.append(name)
                remaining[name] -= 1
    return out


SIZES = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}


def people(per_group=6, groups=8):
    """`groups` subjects with `per_group` rows each; the label follows x."""
    rows = []
    for i in range(per_group * groups):
        subject = f"s{i % groups}"
        x = i * 0.5
        rows.append((str(x), subject, "lo" if x < per_group * groups / 4 else "hi"))
    return Table(("x", "subject", "label"), tuple(rows))


class GroupSplitTests(unittest.TestCase):
    def test_every_group_is_on_one_side_for_many_shapes(self):
        for seed in range(6):
            for sizes in (SIZES, {"a": 1, "b": 1}, {"a": 7, "b": 1, "c": 1, "d": 3}):
                groups = interleaved(sizes)
                for fraction in (0.1, 0.3, 0.5, 0.8):
                    train, test = group_split(groups, fraction, seed)
                    with self.subTest(seed=seed, sizes=sizes, fraction=fraction):
                        self.assertEqual(sorted(train + test), list(range(len(groups))))
                        self.assertEqual(train, sorted(train))
                        self.assertEqual(test, sorted(test))
                        self.assertFalse({groups[i] for i in train} & {groups[i] for i in test})
                        self.assertTrue(train and test)

    def test_matches_an_independent_recomputation(self):
        groups = interleaved(SIZES)
        order = ["A", "B", "C", "D", "E"]  # first appearance order
        rng_for(3, "split", "group").shuffle(order)
        target = max(1, min(14, round(15 * 0.3)))  # 4 rows
        picked, rows = [], 0
        for name in order[:-1]:
            if rows >= target:
                break
            picked.append(name)
            rows += SIZES[name]
        test = [i for i, g in enumerate(groups) if g in picked]
        train = [i for i, g in enumerate(groups) if g not in picked]
        self.assertEqual(group_split(groups, 0.3, 3), (train, test))

    def test_the_test_set_reaches_the_target_size_unless_groups_are_bigger(self):
        groups = interleaved({"a": 10, "b": 10, "c": 10, "d": 10})
        for seed in range(5):
            _, test = group_split(groups, 0.25, seed)
            self.assertEqual(len(test), 10)  # one whole group
        _, test = group_split(groups, 0.05, 1)
        self.assertEqual(len(test), 10)  # target is 2 rows, the smallest unit is a group

    def test_the_last_group_is_never_taken(self):
        groups = interleaved({"a": 3, "b": 3})
        train, test = group_split(groups, 0.99, 1)
        self.assertEqual((len(train), len(test)), (3, 3))

    def test_seed_changes_the_choice_and_is_repeatable(self):
        groups = interleaved({f"g{i}": 3 for i in range(12)})
        results = {tuple(group_split(groups, 0.25, s)[1]) for s in range(10)}
        self.assertGreater(len(results), 3)
        self.assertEqual(group_split(groups, 0.25, 4), group_split(groups, 0.25, 4))

    def test_validation(self):
        with self.assertRaises(DatasetError):
            group_split(["a"], 0.2, 1)
        with self.assertRaises(DatasetError):
            group_split(["a", "a", "a"], 0.2, 1)  # one group cannot be split
        with self.assertRaises(DatasetError):
            group_split(["a", "b", ""], 0.2, 1)
        with self.assertRaises(DatasetError):
            group_split(["a", "b", " "], 0.2, 1)
        for fraction in (0, 1, True):
            with self.assertRaises(DatasetError):
                group_split(["a", "b"], fraction, 1)


class GroupKFoldTests(unittest.TestCase):
    def test_by_hand(self):
        # sizes 5,4,3,2,1 into 2 folds, largest first into the lightest fold:
        # A->0 (5), B->1 (4), C->1 (7), D->0 (7), E->0 on the tie (8): fold 0 = A,D,E; fold 1 = B,C
        groups = interleaved(SIZES)
        splits = group_kfold_indices(groups, 2, seed=1)
        tests = [{groups[i] for i in test} for _, test in splits]
        self.assertEqual(tests, [{"A", "D", "E"}, {"B", "C"}])
        self.assertEqual([len(t) for _, t in splits], [8, 7])

    def test_each_group_is_tested_once_and_never_in_its_own_training_set(self):
        for seed in range(4):
            groups = interleaved({f"g{i}": 1 + i % 4 for i in range(14)})
            for folds in (2, 3, 5, 14):
                splits = group_kfold_indices(groups, folds, seed)
                with self.subTest(seed=seed, folds=folds):
                    tested = Counter(i for _, test in splits for i in test)
                    self.assertEqual(set(tested), set(range(len(groups))))
                    self.assertEqual(set(tested.values()), {1})
                    for train, test in splits:
                        self.assertTrue(test)
                        self.assertEqual(sorted(train + test), list(range(len(groups))))
                        self.assertFalse({groups[i] for i in train} & {groups[i] for i in test})

    def test_fold_sizes_are_balanced(self):
        groups = interleaved({f"g{i}": 4 for i in range(12)})
        sizes = [len(t) for _, t in group_kfold_indices(groups, 4, 2)]
        self.assertEqual(sizes, [12, 12, 12, 12])

    def test_equal_sized_groups_are_ordered_by_seed(self):
        groups = interleaved({f"g{i}": 2 for i in range(8)})
        results = {
            tuple(map(tuple, (t for _, t in group_kfold_indices(groups, 4, s)))) for s in range(8)
        }
        self.assertGreater(len(results), 2)
        self.assertEqual(group_kfold_indices(groups, 4, 5), group_kfold_indices(groups, 4, 5))

    def test_validation(self):
        groups = interleaved(SIZES)
        with self.assertRaises(DatasetError):
            group_kfold_indices(groups, 6, 1)  # five groups, six folds
        self.assertEqual(len(group_kfold_indices(groups, 5, 1)), 5)
        for folds in (1, 51, True, 2.0):
            with self.assertRaises(DatasetError):
                group_kfold_indices(groups, folds, 1)
        with self.assertRaises(DatasetError):
            group_kfold_indices(["a", "", "b"], 2, 1)


class ConfigTests(unittest.TestCase):
    def test_default_is_absent_from_the_dict_and_hash(self):
        config = make()
        self.assertIsNone(config.group_column)
        self.assertNotIn("group_column", config.to_dict())

    def test_set_value_is_hashed_and_round_trips(self):
        config = make(group_column="subject")
        self.assertEqual(config.to_dict()["group_column"], "subject")
        self.assertNotEqual(config.config_hash(), make().config_hash())
        self.assertNotEqual(config.config_hash(), make(group_column="other").config_hash())
        self.assertEqual(ExperimentConfig.from_dict(config.to_dict()), config)

    def test_validation(self):
        for bad in ("", "  ", 3, True, "x" * 201):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                make(group_column=bad)
        with self.assertRaises(ConfigError):
            make(group_column="subject", stratify=True)
        with self.assertRaises(ConfigError):
            make(group_column="subject", time_ordered=True)


class LoadFoldsTests(unittest.TestCase):
    def test_single_split_keeps_subjects_apart_and_drops_the_group_column_from_features(self):
        table = people()
        (fold,) = load_folds(make(group_column="subject"), table, "label", None)
        self.assertEqual(fold.feature_names, ["x"])
        subjects = table.column("subject")
        test_subjects = {subjects[i] for i in fold.test_idx}
        train_rows = set(range(len(table))) - set(fold.test_idx)
        self.assertFalse(test_subjects & {subjects[i] for i in train_rows})
        self.assertEqual(len(fold.test_idx) % 6, 0)  # whole subjects only

    def test_cv_tests_every_subject_exactly_once(self):
        table = people()
        folds = load_folds(make(group_column="subject", cv_folds=4), table, "label", None)
        subjects = table.column("subject")
        self.assertEqual(sorted(i for f in folds for i in f.test_idx), list(range(len(table))))
        for fold in folds:
            test = {subjects[i] for i in fold.test_idx}
            train = {subjects[i] for i in set(range(len(table))) - set(fold.test_idx)}
            self.assertFalse(test & train)
            self.assertEqual(len(test), 2)

    def test_the_config_seed_picks_the_splits(self):
        table = people()
        groups = table.column("subject")
        (single,) = load_folds(make(group_column="subject", seed=4), table, "label", None)
        self.assertEqual(single.test_idx, group_split(groups, 0.2, 4)[1])
        folds = load_folds(make(group_column="subject", cv_folds=4, seed=4), table, "label", None)
        self.assertEqual(
            [f.test_idx for f in folds], [t for _, t in group_kfold_indices(groups, 4, 4)]
        )

    def test_a_numeric_group_column_still_cannot_be_a_feature_or_the_label(self):
        rows = tuple((str(i), str(i % 4), "lo" if i < 12 else "hi") for i in range(24))
        table = Table(("x", "site", "label"), rows)
        config = make(group_column="site")
        (fold,) = load_folds(config, table, "label", None)
        self.assertEqual(fold.feature_names, ["x"])
        with self.assertRaises(RunnerError) as raised:
            load_folds(config, table, "label", ["x", "site"])
        self.assertIn("also be a feature", str(raised.exception))
        with self.assertRaises(RunnerError) as raised:
            load_folds(make(group_column="x"), table, "x", ["site"])
        self.assertIn("also be the label", str(raised.exception))

    def test_explicit_features_are_kept_and_cannot_include_the_group(self):
        table = people()
        (fold,) = load_folds(make(group_column="subject"), table, "label", ["x"])
        self.assertEqual(fold.feature_names, ["x"])
        with self.assertRaises(RunnerError):
            load_folds(make(group_column="subject"), table, "label", ["x", "subject"])

    def test_missing_or_label_group_column_is_an_error(self):
        table = people()
        with self.assertRaises(RunnerError):
            load_folds(make(group_column="nope"), table, "label", None)
        with self.assertRaises(RunnerError):
            load_folds(make(group_column="label"), table, "label", None)

    def test_without_the_setting_the_group_like_column_is_an_ordinary_feature_error(self):
        # "s0" is not a number, so the default feature list cannot be used when it is included
        with self.assertRaises(RunnerError):
            load_folds(make(), people(), "label", None)

    def test_too_few_groups_for_the_folds(self):
        with self.assertRaises(DatasetError):
            load_folds(make(group_column="subject", cv_folds=9), people(), "label", None)


class RunTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)

    def test_run_records_the_tag_and_reuses_identical_runs(self):
        table = people()
        config = make(group_column="subject", cv_folds=4)
        first = run_experiment(self.store, config, table, "label", environment=ENV)
        self.assertEqual(self.store.get_run(first.run_id).tags["group_column"], "subject")
        again = run_experiment(
            self.store, config, table, "label", environment=ENV, reuse_existing=True
        )
        self.assertTrue(again.reused)
        self.assertEqual(first.n_test, len(table))

    def test_predictions_file_lists_whole_subjects_per_fold(self):
        table = people()
        subjects = table.column("subject")
        with tempfile.TemporaryDirectory() as tmp:
            result = run_experiment(
                self.store,
                make(group_column="subject", cv_folds=4),
                table,
                "label",
                environment=ENV,
                artifact_dir=tmp,
            )
            saved = json.loads((Path(tmp) / result.run_id / "predictions.json").read_text())
        for fold in saved["folds"]:
            self.assertEqual(len(fold["test_indices"]), 12)
            self.assertEqual(len({subjects[i] for i in fold["test_indices"]}), 2)

    def test_signature_matches_the_resolved_features(self):
        table = people()
        run = run_experiment(
            self.store, make(group_column="subject"), table, "label", environment=ENV
        )
        stored = self.store.get_run(run.run_id).tags
        self.assertEqual(stored[COLUMNS_TAG], columns_signature(table, "label", ["x"]))

    def test_the_run_counts_cover_the_whole_table(self):
        table = people()
        result = run_experiment(
            self.store, make(group_column="subject"), table, "label", environment=ENV
        )
        self.assertEqual(result.n_test + result.n_train, len(table))
        self.assertEqual(result.n_test % 6, 0)


class OtherCommandsRefuseTests(unittest.TestCase):
    def test_nested_and_selection_refuse_groups(self):
        table = people()
        cfg = make(group_column="subject")
        with self.assertRaises(SweepError) as raised:
            run_nested(cfg, [{"k": 1}, {"k": 3}], table, "label", outer_folds=3, inner_folds=2)
        self.assertIn("group_column", str(raised.exception))
        with self.assertRaises(SweepError) as raised:
            run_selection(cfg, table, "label", folds=3)
        self.assertIn("group_column", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
