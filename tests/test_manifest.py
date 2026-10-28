import json
import math
import unittest

from modelbench.datasets import Table, make_version
from modelbench.errors import DatasetError
from modelbench.manifest import FORMAT, TOP_VALUES, build_manifest, manifest_json


def table():
    return Table(
        ("x", "grade", "label"),
        (("1", "a", "u"), ("2", "b", "u"), ("3", "a", "v"), ("4", "", "v")),
    )


class ManifestTests(unittest.TestCase):
    def test_the_header_matches_the_dataset_version(self):
        manifest = build_manifest("demo", table(), "label")
        version = make_version("demo", table(), "label")
        self.assertEqual(manifest["format"], FORMAT)
        self.assertEqual(manifest["format_version"], 1)
        self.assertEqual(manifest["version"], version.version)
        self.assertEqual(manifest["sha256"], version.sha256)
        self.assertEqual((manifest["n_rows"], manifest["label_column"]), (4, "label"))

    def test_a_numeric_column_by_hand(self):
        # 1, 2, 3, 4: mean 2.5, population variance (2.25 + 0.25 + 0.25 + 2.25) / 4 = 1.25
        x = build_manifest("demo", table(), "label")["columns"][0]
        self.assertEqual(
            x,
            {
                "name": "x", "kind": "numeric", "missing": 0, "distinct": 4,
                "min": 1.0, "max": 4.0, "mean": 2.5, "std": math.sqrt(1.25),
            },
        )  # fmt: skip

    def test_a_text_column_counts_missing_cells_and_lists_the_commonest_values(self):
        grade = build_manifest("demo", table(), "label")["columns"][1]
        self.assertEqual(
            grade,
            {"name": "grade", "kind": "text", "missing": 1, "distinct": 2,
             "top": [["a", 2], ["b", 1]]},
        )  # fmt: skip

    def test_label_counts_are_sorted_by_label(self):
        rows = (("1", "z"), ("2", "a"), ("3", "z"))
        manifest = build_manifest("d", Table(("x", "y"), rows), "y")
        self.assertEqual(manifest["labels"], {"a": 1, "z": 2})
        self.assertEqual(list(manifest["labels"]), ["a", "z"])
        self.assertNotIn("labels", build_manifest("d", Table(("x", "y"), rows)))

    def test_labels_that_differ_only_in_spacing_count_together(self):
        manifest = build_manifest(
            "d", Table(("x", "y"), (("1", "u"), ("2", " u "), ("3", "v"))), "y"
        )
        self.assertEqual(manifest["labels"], {"u": 2, "v": 1})

    def test_missing_tokens_are_the_ones_preprocessing_uses(self):
        rows = tuple((c,) for c in ("1", "NA", "n/a", "?", " ", "5"))
        column = build_manifest("d", Table(("x",), rows))["columns"][0]
        self.assertEqual((column["missing"], column["kind"], column["mean"]), (4, "numeric", 3.0))

    def test_a_column_with_one_non_number_is_text(self):
        for bad in ("abc", "inf", "1e999", "1,5", "0x10"):
            rows = (("1",), ("2",), (bad,))
            column = build_manifest("d", Table(("x",), rows))["columns"][0]
            self.assertEqual(column["kind"], "text", bad)
            self.assertNotIn("mean", column)

    def test_an_all_missing_column_is_text_with_no_values(self):
        column = build_manifest("d", Table(("x", "y"), (("", "1"), ("na", "2"))))["columns"][0]
        self.assertEqual(
            column, {"name": "x", "kind": "text", "missing": 2, "distinct": 0, "top": []}
        )

    def test_top_values_break_ties_alphabetically_and_stop_at_the_limit(self):
        names = [f"v{i}" for i in range(TOP_VALUES + 3)]
        rows = tuple((n,) for n in reversed(names))  # every value once
        top = build_manifest("d", Table(("x",), rows))["columns"][0]["top"]
        self.assertEqual([v for v, _ in top], names[:TOP_VALUES])
        self.assertEqual(build_manifest("d", Table(("x",), rows))["columns"][0]["distinct"], 8)

    def test_whitespace_around_cells_does_not_create_new_values(self):
        rows = (("a",), (" a ",), ("b",))
        column = build_manifest("d", Table(("x",), rows))["columns"][0]
        self.assertEqual((column["distinct"], column["top"]), (2, [["a", 2], ["b", 1]]))

    def test_the_same_table_gives_identical_text_and_valid_json(self):
        first = manifest_json(build_manifest("demo", table(), "label"))
        second = manifest_json(build_manifest("demo", table(), "label"))
        self.assertEqual(first, second)
        self.assertTrue(first.endswith("}\n"))
        self.assertEqual(json.loads(first)["n_rows"], 4)
        self.assertLess(first.index('"columns"'), first.index('"version"'))  # sorted keys

    def test_changing_a_cell_changes_the_version_and_the_statistics(self):
        other = Table(
            table().columns, (("1", "a", "u"), ("2", "b", "u"), ("3", "a", "v"), ("9", "", "v"))
        )
        a, b = build_manifest("d", table()), build_manifest("d", other)
        self.assertNotEqual(a["version"], b["version"])
        self.assertNotEqual(a["columns"][0]["max"], b["columns"][0]["max"])

    def test_an_empty_table_is_refused(self):
        with self.assertRaisesRegex(DatasetError, "no rows"):
            build_manifest("d", Table(("x",), ()))
        with self.assertRaises(DatasetError):
            build_manifest("d", table(), "nope")


if __name__ == "__main__":
    unittest.main()
