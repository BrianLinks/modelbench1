import hashlib
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from modelbench.datasets import (
    MAX_FOLDS,
    Table,
    kfold_indices,
    make_version,
    read_csv,
    resolve_input_file,
    split_indices,
    split_table,
    table_digest,
)
from modelbench.errors import DatasetError, IdentifierError, SeedError

SMALL = Table(("a", "b"), (("1", "2"), ("3", "4")))


class TableTests(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(DatasetError):
            Table((), ())
        with self.assertRaises(DatasetError):
            Table(("a", "a"), ())
        with self.assertRaises(DatasetError):
            Table(("a", " "), ())
        with self.assertRaises(DatasetError):
            Table(("a", "b"), (("1",),))

    def test_column_and_select(self):
        self.assertEqual(SMALL.column("b"), ["2", "4"])
        with self.assertRaises(DatasetError):
            SMALL.column("zzz")
        picked = SMALL.select([1, 0, 1])
        self.assertEqual(picked.rows, (("3", "4"), ("1", "2"), ("3", "4")))
        self.assertEqual(len(picked), 3)


class FileTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def write(self, name, text, encoding="utf-8"):
        path = self.dir / name
        path.write_bytes(text.encode(encoding))
        return path


class ReadCsvTests(FileTestCase):
    def test_reads_header_and_rows(self):
        table = read_csv(self.write("a.csv", "x, y \n1,2\n3,4\n"))
        self.assertEqual(table.columns, ("x", "y"))
        self.assertEqual(table.rows, (("1", "2"), ("3", "4")))

    def test_bom_blank_lines_and_quoted_newlines(self):
        table = read_csv(self.write("a.csv", '﻿x,note\n1,"two\nlines"\n\n2,ok\n'))
        self.assertEqual(table.rows, (("1", "two\nlines"), ("2", "ok")))

    def test_header_only_file_gives_an_empty_table(self):
        self.assertEqual(len(read_csv(self.write("a.csv", "x,y\n"))), 0)

    def test_ragged_row_reports_the_line(self):
        with self.assertRaises(DatasetError) as ctx:
            read_csv(self.write("a.csv", "x,y\n1,2\n3\n"))
        self.assertIn("line 3", str(ctx.exception))

    def test_bad_files(self):
        with self.assertRaises(DatasetError):
            read_csv(self.write("empty.csv", ""))
        with self.assertRaises(DatasetError):
            read_csv(self.write("dup.csv", "a,a\n1,2\n"))
        with self.assertRaises(DatasetError):
            read_csv(self.write("latin.csv", "a\ncafé\n", encoding="latin-1"))
        with self.assertRaises(DatasetError):
            read_csv(self.dir / "missing.csv")
        with self.assertRaises(DatasetError):
            read_csv(self.dir)

    def test_base_dir_blocks_path_traversal(self):
        inside = self.dir / "data"
        inside.mkdir()
        good = inside / "ok.csv"
        good.write_text("a\n1\n")
        outside = self.write("secret.csv", "a\n1\n")
        self.assertEqual(resolve_input_file(good, inside), good.resolve())
        with self.assertRaises(DatasetError):
            read_csv(outside, base_dir=inside)
        with self.assertRaises(DatasetError):
            read_csv(inside / ".." / "secret.csv", base_dir=inside)


class VersionTests(FileTestCase):
    def test_digest_matches_an_independent_hash_of_the_canonical_text(self):
        literal = b'{"columns":["a","b"],"rows":[["1","2"],["3","4"]]}'
        self.assertEqual(table_digest(SMALL), hashlib.sha256(literal).hexdigest())

    def test_version_fields(self):
        v = make_version("iris", SMALL, label_column="b")
        self.assertEqual(v.version, v.sha256[:12])
        self.assertEqual((v.dataset_id, v.n_rows, v.columns), ("iris", 2, ("a", "b")))
        self.assertEqual(v.label_column, "b")

    def test_same_content_gives_same_version_regardless_of_file_encoding(self):
        lf = read_csv(self.write("lf.csv", "a,b\n1,2\n3,4\n"))
        crlf = read_csv(self.write("crlf.csv", "﻿a,b\r\n1,2\r\n3,4\r\n"))
        self.assertEqual(make_version("d", lf).version, make_version("d", crlf).version)

    def test_any_content_change_changes_the_version(self):
        base = make_version("d", SMALL).version
        changed_cell = Table(("a", "b"), (("1", "2"), ("3", "5")))
        reordered_rows = Table(("a", "b"), (("3", "4"), ("1", "2")))
        reordered_cols = Table(("b", "a"), (("2", "1"), ("4", "3")))
        extra_row = Table(("a", "b"), (*SMALL.rows, ("5", "6")))
        for other in (changed_cell, reordered_rows, reordered_cols, extra_row):
            self.assertNotEqual(make_version("d", other).version, base)

    def test_cell_boundaries_cannot_collide(self):
        one = Table(("a", "b"), (("1,2", "3"),))
        two = Table(("a", "b"), (("1", "2,3"),))
        self.assertNotEqual(table_digest(one), table_digest(two))

    def test_validation(self):
        with self.assertRaises(IdentifierError):
            make_version("Bad Id", SMALL)
        with self.assertRaises(DatasetError):
            make_version("d", Table(("a",), ()))
        with self.assertRaises(DatasetError):
            make_version("d", SMALL, label_column="nope")


class SplitTests(unittest.TestCase):
    def check_partition(self, n, train, test):
        self.assertEqual(sorted(train + test), list(range(n)))
        self.assertEqual(train, sorted(train))
        self.assertEqual(test, sorted(test))
        self.assertTrue(train and test)

    def test_partition_sizes_and_determinism(self):
        train, test = split_indices(10, 0.3, seed=42)
        self.check_partition(10, train, test)
        self.assertEqual(len(test), 3)
        self.assertEqual((train, test), split_indices(10, 0.3, seed=42))

    def test_different_seeds_give_different_splits(self):
        splits = {tuple(split_indices(50, 0.2, seed=s)[1]) for s in range(10)}
        self.assertGreater(len(splits), 5)

    def test_small_inputs_always_keep_both_sides_non_empty(self):
        for n in (2, 3):
            for fraction in (0.01, 0.5, 0.99):
                train, test = split_indices(n, fraction, seed=1)
                self.check_partition(n, train, test)

    def test_global_random_state_is_irrelevant(self):
        expected = split_indices(30, 0.25, seed=7)
        random.seed(999)
        random.random()
        self.assertEqual(split_indices(30, 0.25, seed=7), expected)

    def test_stratified_split_preserves_class_proportions(self):
        labels = ["a"] * 70 + ["b"] * 30
        train, test = split_indices(100, 0.2, seed=3, stratify=labels)
        self.check_partition(100, train, test)
        counts = Counter(labels[i] for i in test)
        self.assertEqual((counts["a"], counts["b"]), (14, 6))
        self.assertEqual((train, test), split_indices(100, 0.2, seed=3, stratify=labels))

    def test_stratified_singleton_class_stays_in_train(self):
        labels = ["a"] * 9 + ["b"]
        train, test = split_indices(10, 0.3, seed=1, stratify=labels)
        self.assertIn(9, train)
        self.assertNotIn(9, test)
        self.assertTrue(test)

    def test_stratified_with_only_singletons_is_an_error(self):
        with self.assertRaises(DatasetError):
            split_indices(3, 0.5, seed=1, stratify=["a", "b", "c"])

    def test_invalid_arguments(self):
        for n in (0, 1, -3, True, 2.5):
            with self.assertRaises(DatasetError, msg=repr(n)):
                split_indices(n, 0.2, seed=1)
        for fraction in (0, 1, -0.1, 1.5, True, "0.2", float("nan"), float("inf")):
            with self.assertRaises(DatasetError, msg=repr(fraction)):
                split_indices(10, fraction, seed=1)
        with self.assertRaises(DatasetError):
            split_indices(10, 0.2, seed=1, stratify=["a"] * 9)
        with self.assertRaises(ValueError):
            split_indices(10, 0.2, seed=-1)

    def test_partition_property_over_random_sizes(self):
        rng = random.Random(20261019)
        for _ in range(100):
            n = rng.randint(2, 80)
            fraction = rng.uniform(0.05, 0.95)
            seed = rng.randint(0, 10_000)
            self.check_partition(n, *split_indices(n, fraction, seed))
            labels = [rng.choice("abc") for _ in range(n)]
            try:
                train, test = split_indices(n, fraction, seed, stratify=labels)
            except DatasetError:
                self.assertTrue(all(c == 1 for c in Counter(labels).values()))
                continue
            self.check_partition(n, train, test)


class SplitTableTests(unittest.TestCase):
    TABLE = Table(
        ("id", "label"),
        tuple((str(i), "pos" if i % 3 == 0 else "neg") for i in range(30)),
    )

    def test_rows_are_partitioned_exactly(self):
        train, test = split_table(self.TABLE, 0.3, seed=5)
        self.assertEqual(train.columns, self.TABLE.columns)
        self.assertEqual(sorted(train.rows + test.rows), sorted(self.TABLE.rows))
        self.assertEqual(len(test), 9)

    def test_stratify_by_column(self):
        train, test = split_table(self.TABLE, 0.3, seed=5, stratify_column="label")
        original = Counter(self.TABLE.column("label"))
        self.assertEqual(Counter(train.column("label")) + Counter(test.column("label")), original)
        self.assertEqual(Counter(test.column("label"))["pos"], 3)  # round(10 * 0.3)

    def test_unknown_stratify_column(self):
        with self.assertRaises(DatasetError):
            split_table(self.TABLE, 0.3, seed=5, stratify_column="nope")


class KFoldTests(unittest.TestCase):
    def test_two_rows_two_folds_by_hand(self):
        folds = kfold_indices(2, 2, seed=1)
        self.assertEqual(sorted(folds), [([0], [1]), ([1], [0])])

    def test_fold_sizes_differ_by_at_most_one_and_cover_every_row_once(self):
        folds = kfold_indices(10, 3, seed=4)
        self.assertEqual(sorted(len(test) for _, test in folds), [3, 3, 4])
        seen = [i for _, test in folds for i in test]
        self.assertEqual(sorted(seen), list(range(10)))
        for train, test in folds:
            self.assertEqual(train, sorted(train))
            self.assertEqual(test, sorted(test))
            self.assertEqual(sorted(train + test), list(range(10)))
            self.assertFalse(set(train) & set(test))

    def test_deterministic_and_seed_dependent(self):
        self.assertEqual(kfold_indices(30, 5, 7), kfold_indices(30, 5, 7))
        self.assertNotEqual(kfold_indices(30, 5, 7), kfold_indices(30, 5, 8))

    def test_global_random_state_is_untouched(self):
        random.seed(5)
        expected = [random.random() for _ in range(3)]
        random.seed(5)
        kfold_indices(30, 5, 7)
        self.assertEqual([random.random() for _ in range(3)], expected)

    def test_stratified_folds_keep_class_balance(self):
        labels = ["a"] * 6 + ["b"] * 6
        for _, test in kfold_indices(12, 3, 2, labels):
            self.assertEqual(Counter(labels[i] for i in test), {"a": 2, "b": 2})

    def test_stratified_counts_differ_by_at_most_one_with_uneven_classes(self):
        labels = ["a"] * 7 + ["b"] * 5 + ["c"] * 1
        folds = kfold_indices(13, 3, 9, labels)
        for cls in "abc":
            counts = [sum(labels[i] == cls for i in test) for _, test in folds]
            self.assertLessEqual(max(counts) - min(counts), 1, cls)
        sizes = [len(test) for _, test in folds]
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_random_inputs_always_partition(self):
        rng = random.Random(11)
        for _ in range(60):
            n = rng.randint(2, 40)
            k = rng.randint(2, min(n, 12))
            labels = [rng.choice("xyz") for _ in range(n)] if rng.random() < 0.5 else None
            folds = kfold_indices(n, k, rng.randint(0, 99), labels)
            self.assertEqual(len(folds), k)
            tests = sorted(i for _, test in folds for i in test)
            self.assertEqual(tests, list(range(n)))
            self.assertTrue(all(test and train for train, test in folds))

    def test_validation(self):
        bad = [
            dict(n=1, folds=2), dict(n=True, folds=2), dict(n=10, folds=1),
            dict(n=10, folds=51), dict(n=10, folds=True), dict(n=10, folds=2.0),
            dict(n=3, folds=4), dict(n=4, folds=2, stratify=["a"] * 3),
        ]  # fmt: skip
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(DatasetError):
                kfold_indices(seed=1, **kwargs)
        with self.assertRaises(SeedError):
            kfold_indices(4, 2, seed=-1)
        self.assertEqual(MAX_FOLDS, 50)


if __name__ == "__main__":
    unittest.main()
