import math
import random
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from modelbench.datasets import make_version, read_csv
from modelbench.errors import SeedError
from modelbench.models import create_regressor
from modelbench.samples import (
    BASELINE_NOTE,
    DEMO_MAX_ROWS_PER_CLASS,
    LINEAR_INTERCEPT,
    SamplesError,
    _number,
    available_samples,
    generate,
    linear_weights,
    make_blobs,
    make_linear,
    make_moons,
    make_xor,
    run_demo,
    table_to_csv,
)
from modelbench.store import Store

CELL = re.compile(r"-?[0-9]+\.[0-9]{6}")


def floats(table, columns=("x1", "x2")):
    positions = [table.columns.index(c) for c in columns]
    return [[float(row[i]) for i in positions] for row in table.rows]


class FormattingTests(unittest.TestCase):
    def test_numbers_have_six_decimals_and_never_negative_zero(self):
        self.assertEqual(_number(1.0), "1.000000")
        self.assertEqual(_number(-2.5), "-2.500000")
        self.assertEqual(_number(-1e-9), "0.000000")
        self.assertEqual(_number(-0.0), "0.000000")
        self.assertEqual(_number(1.23456749), "1.234567")
        self.assertEqual(_number(1.23456751), "1.234568")

    def test_every_generated_cell_follows_the_format(self):
        for name in available_samples():
            table = generate(name, 3)
            for row in table.rows:
                for cell in row[:-1]:
                    self.assertRegex(cell, CELL)
                    self.assertNotEqual(cell, "-0.000000")


class BlobsTests(unittest.TestCase):
    def test_without_spread_every_row_is_its_class_centre(self):
        table = make_blobs(1, rows_per_class=4, classes=3, features=2, separation=3.0, spread=0.0)
        self.assertEqual(table.columns, ("x1", "x2", "label"))
        centres = {"c0": ("0.000000", "3.000000"), "c1": ("3.000000", "6.000000"),
                   "c2": ("6.000000", "0.000000")}  # fmt: skip
        self.assertEqual(len(table), 12)
        for *cells, label in table.rows:
            self.assertEqual(tuple(cells), centres[label])

    def test_classes_are_balanced_and_columns_follow_the_options(self):
        table = make_blobs(2, rows_per_class=7, classes=4, features=5)
        self.assertEqual(table.columns, ("x1", "x2", "x3", "x4", "x5", "label"))
        self.assertEqual(Counter(table.column("label")), {f"c{k}": 7 for k in range(4)})

    def test_spread_and_separation_have_their_effect(self):
        def class_spread(table, label):
            values = [float(r[0]) for r in table.rows if r[-1] == label]
            mean = sum(values) / len(values)
            return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

        wide = class_spread(make_blobs(3, rows_per_class=400, spread=2.0), "c0")
        narrow = class_spread(make_blobs(3, rows_per_class=400, spread=0.5), "c0")
        self.assertAlmostEqual(wide, 2.0, delta=0.2)
        self.assertAlmostEqual(narrow, 0.5, delta=0.05)
        far = make_blobs(3, rows_per_class=50, separation=10.0, spread=0.0)
        self.assertEqual(sorted({float(r[0]) for r in far.rows}), [0.0, 10.0, 20.0])


class MoonsTests(unittest.TestCase):
    def test_without_noise_points_lie_on_the_two_circles(self):
        table = make_moons(1, rows_per_class=25, noise=0.0)
        self.assertEqual(Counter(table.column("label")), {"a": 25, "b": 25})
        for (x, y), label in zip(floats(table), table.column("label"), strict=True):
            if label == "a":
                self.assertAlmostEqual(math.hypot(x, y), 1.0, delta=2e-6)
                self.assertGreaterEqual(y, -1e-6)
            else:
                self.assertAlmostEqual(math.hypot(x - 1.0, y - 0.5), 1.0, delta=2e-6)
                self.assertLessEqual(y, 0.5 + 1e-6)

    def test_the_arc_end_points_are_included(self):
        rows = set(make_moons(1, rows_per_class=9, noise=0.0).rows)
        for expected in (("1.000000", "0.000000", "a"), ("-1.000000", "0.000000", "a"),
                         ("0.000000", "0.500000", "b"), ("2.000000", "0.500000", "b")):  # fmt: skip
            self.assertIn(expected, rows)

    def test_noise_moves_the_points(self):
        quiet = make_moons(1, rows_per_class=50, noise=0.0)
        noisy = make_moons(1, rows_per_class=50, noise=0.3)
        self.assertNotEqual(quiet.rows, noisy.rows)


class XorTests(unittest.TestCase):
    def test_without_noise_rows_sit_on_the_four_corners(self):
        table = make_xor(1, rows_per_class=5, noise=0.0)
        counts = Counter(table.rows)
        self.assertEqual(
            counts,
            {
                ("1.000000", "1.000000", "pos"): 3,
                ("-1.000000", "-1.000000", "pos"): 2,
                ("1.000000", "-1.000000", "neg"): 3,
                ("-1.000000", "1.000000", "neg"): 2,
            },
        )

    def test_the_label_is_the_sign_of_the_product_up_to_noise(self):
        table = make_xor(4, rows_per_class=300, noise=0.2)
        labels = table.column("label")
        wrong = sum(
            (x * y > 0) != (label == "pos")
            for (x, y), label in zip(floats(table), labels, strict=True)
        )
        self.assertLessEqual(wrong, 6)  # a 0.2 spread rarely crosses an axis from a corner


class LinearTests(unittest.TestCase):
    def test_weights_follow_the_documented_pattern(self):
        self.assertEqual(linear_weights(7), [1.0, -2.0, 3.0, -4.0, 5.0, -1.0, 2.0])
        self.assertEqual(LINEAR_INTERCEPT, 4.0)

    def test_without_noise_every_row_lies_on_the_plane(self):
        table = make_linear(2, rows=50, features=3, noise=0.0)
        self.assertEqual(table.columns, ("x1", "x2", "x3", "y"))
        self.assertEqual(len(table), 50)
        weights = linear_weights(3)
        for *x, y in ([float(c) for c in row] for row in table.rows):
            self.assertTrue(all(-5.0 <= v <= 5.0 for v in x))
            self.assertAlmostEqual(
                y, 4.0 + sum(w * v for w, v in zip(weights, x, strict=True)), delta=2e-6
            )
        for row in table.rows:
            self.assertTrue(all(CELL.fullmatch(cell) for cell in row), row)

    def test_noise_has_the_requested_spread(self):
        table = make_linear(3, rows=4000, features=1, noise=2.0)
        residuals = [float(y) - (4.0 + float(x)) for x, y in table.rows]
        mean = sum(residuals) / len(residuals)
        sd = math.sqrt(sum((r - mean) ** 2 for r in residuals) / len(residuals))
        self.assertAlmostEqual(sd, 2.0, delta=0.15)
        self.assertAlmostEqual(mean, 0.0, delta=0.15)

    def test_a_fitted_model_recovers_the_true_weights(self):
        table = make_linear(5, rows=600, features=3, noise=1.0)
        X = [[float(c) for c in row[:3]] for row in table.rows]
        y = [float(row[3]) for row in table.rows]
        model = create_regressor("linear_regression")
        model.fit(X, y)
        for got, want in zip(model.coef_, linear_weights(3), strict=True):
            self.assertAlmostEqual(got, want, delta=0.1)
        self.assertAlmostEqual(model.intercept_, 4.0, delta=0.2)

    def test_csv_round_trip_and_defaults(self):
        table = make_linear(1)
        self.assertEqual((table.columns, len(table)), (("x1", "x2", "y"), 200))
        self.assertEqual(generate("linear", 1), table)


class DeterminismTests(unittest.TestCase):
    def test_same_seed_same_table_different_seed_different_table(self):
        for name in available_samples():
            self.assertEqual(generate(name, 5), generate(name, 5), name)
            self.assertNotEqual(generate(name, 5).rows, generate(name, 6).rows, name)

    def test_rows_are_shuffled_not_grouped_by_class(self):
        for name in ("blobs", "moons", "xor"):
            labels = generate(name, 5).column("label")
            self.assertNotEqual(labels, sorted(labels), name)

    def test_global_random_state_is_untouched(self):
        random.seed(99)
        expected = [random.random() for _ in range(3)]
        random.seed(99)
        for name in available_samples():
            generate(name, 5)
        self.assertEqual([random.random() for _ in range(3)], expected)

    def test_content_hashes_are_pinned(self):
        # Golden values: if one changes, a generator changed or the random stream differs on
        # this Python build, and every stored run on that dataset would stop reproducing.
        golden = {
            "blobs": "588e5e0914fd",
            "linear": "edee70153b88",
            "moons": "438b1454664d",
            "xor": "389fe30a9431",
        }
        for name, version in golden.items():
            options = {"rows": 10} if name == "linear" else {"rows_per_class": 10}
            table = generate(name, 1, **options)
            target = "y" if name == "linear" else "label"
            self.assertEqual(make_version("d", table, target).version, version, name)


class ValidationTests(unittest.TestCase):
    def test_unknown_name_and_options(self):
        with self.assertRaisesRegex(SamplesError, "unknown sample"):
            generate("spirals", 1)
        with self.assertRaisesRegex(SamplesError, "does not take"):
            generate("moons", 1, classes=3)
        with self.assertRaisesRegex(SamplesError, "does not take"):
            generate("blobs", 1, noise=0.1)
        self.assertEqual(available_samples(), ("blobs", "linear", "moons", "xor"))

    def test_bad_values(self):
        bad = [
            ("blobs", {"rows_per_class": 1}), ("blobs", {"rows_per_class": True}),
            ("blobs", {"features": True}), ("blobs", {"classes": True}),
            ("moons", {"rows_per_class": True}), ("xor", {"rows_per_class": True}),
            ("blobs", {"rows_per_class": 2.0}), ("blobs", {"classes": 1}),
            ("blobs", {"classes": 21}), ("blobs", {"features": 0}), ("blobs", {"features": 51}),
            ("blobs", {"separation": -1.0}), ("blobs", {"spread": float("nan")}),
            ("blobs", {"spread": float("inf")}), ("blobs", {"spread": "1"}),
            ("blobs", {"rows_per_class": 90_000, "classes": 3}),
            ("moons", {"noise": -0.1}), ("moons", {"noise": True}),
            ("moons", {"rows_per_class": 1}),
            ("blobs", {"rows_per_class": 20_000, "classes": 3, "features": 50}),
            ("xor", {"noise": float("nan")}), ("xor", {"rows_per_class": 60_000}),
            ("linear", {"rows": 1}), ("linear", {"rows": True}), ("linear", {"features": 0}),
            ("linear", {"features": 51}), ("linear", {"features": True}),
            ("linear", {"noise": -1.0}), ("linear", {"noise": float("inf")}),
            ("linear", {"rows": 100_001}), ("linear", {"rows": 50_000, "features": 50}),
            ("linear", {"rows_per_class": 10}),
        ]  # fmt: skip
        for name, options in bad:
            with self.subTest(name=name, options=options), self.assertRaises(SamplesError):
                generate(name, 1, **options)

    def test_bad_seed(self):
        for seed in (-1, 2**32, True, 1.5, "1"):
            with self.subTest(seed=seed), self.assertRaises(SeedError):
                generate("moons", seed)


class CsvTests(unittest.TestCase):
    def test_round_trip_through_read_csv(self):
        table = generate("blobs", 8, rows_per_class=6)
        text = table_to_csv(table)
        self.assertTrue(text.startswith("x1,x2,label\n"))
        self.assertTrue(text.endswith("\n"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blobs.csv"
            path.write_text(text, encoding="utf-8")
            self.assertEqual(read_csv(path), table)

    def test_cells_that_would_need_quoting_are_refused(self):
        from modelbench.datasets import Table

        for cell in ("a,b", 'say "hi"', "line\nbreak"):
            with self.subTest(cell=cell), self.assertRaises(SamplesError):
                table_to_csv(Table(("x", "label"), ((cell, "a"),)))
        with self.assertRaises(SamplesError):
            table_to_csv(Table(("x,y", "label"), (("1", "a"),)))


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)

    def test_runs_every_model_on_every_dataset(self):
        runs = run_demo(self.store, seed=42)
        self.assertEqual(len(runs), 27)
        self.assertEqual(Counter(r.dataset for r in runs), {"moons": 9, "xor": 9, "blobs": 9})
        self.assertEqual(
            Counter(r.model for r in runs),
            {
                "majority": 3,
                "gaussian_nb": 3,
                "logistic_regression": 3,
                "knn": 9,
                "decision_tree": 9,
            },  # fmt: skip
        )
        stored = self.store.list_runs(limit=100)
        self.assertEqual(len(stored), 27)
        self.assertTrue(all(r.status == "completed" for r in stored))
        experiments = {r.experiment_id for r in stored}
        self.assertEqual(experiments, {"demo-moons", "demo-xor", "demo-blobs"})

    def test_running_it_again_adds_nothing(self):
        first = run_demo(self.store, seed=42)
        second = run_demo(self.store, seed=42)
        self.assertTrue(all(r.result.reused for r in second))
        self.assertEqual([r.result.run_id for r in second], [r.result.run_id for r in first])
        self.assertEqual(len(self.store.list_runs(limit=100)), 27)

    def test_a_different_seed_makes_a_new_set_of_runs(self):
        run_demo(self.store, seed=42)
        run_demo(self.store, seed=43)
        self.assertEqual(len(self.store.list_runs(limit=100)), 54)

    def test_only_the_baselines_carry_a_note_and_only_once(self):
        runs = run_demo(self.store, seed=42)
        run_demo(self.store, seed=42)
        for r in runs:
            notes = self.store.get_notes(r.result.run_id)
            if r.model == "majority":
                self.assertEqual([n.text for n in notes], [BASELINE_NOTE])
            else:
                self.assertEqual(notes, [])

    def test_the_datasets_teach_what_they_are_meant_to(self):
        acc = {(r.dataset, r.model, tuple(r.params.items())): r.result.metrics["accuracy"]
               for r in run_demo(self.store, seed=42)}  # fmt: skip
        self.assertAlmostEqual(acc[("blobs", "majority", ())], 1 / 3, places=6)
        self.assertEqual(acc[("blobs", "knn", (("k", 5),))], 1.0)
        self.assertLess(acc[("moons", "logistic_regression", (("max_iter", 300),))], 0.9)
        self.assertGreater(acc[("moons", "knn", (("k", 5),))], 0.9)
        self.assertLess(acc[("xor", "logistic_regression", (("max_iter", 300),))], 0.65)
        self.assertEqual(acc[("xor", "knn", (("k", 5),))], 1.0)
        self.assertLess(acc[("xor", "decision_tree", (("max_depth", 1),))], 0.65)
        self.assertEqual(acc[("xor", "decision_tree", (("max_depth", 6),))], 1.0)

    def test_binary_datasets_also_report_roc_auc(self):
        runs = run_demo(self.store, seed=42)
        for r in runs:
            self.assertEqual("roc_auc" in r.result.metrics, r.dataset != "blobs", r.dataset)

    def test_a_baseline_note_lost_to_a_crash_is_restored_exactly_once(self):
        runs = run_demo(self.store, seed=42)
        baseline = next(r for r in runs if r.model == "majority").result.run_id
        with self.store._conn:
            self.store._conn.execute("DELETE FROM notes WHERE run_id = ?", (baseline,))
        self.assertEqual(self.store.get_notes(baseline), [])
        run_demo(self.store, seed=42)
        run_demo(self.store, seed=42)
        self.assertEqual([n.text for n in self.store.get_notes(baseline)], [BASELINE_NOTE])

    def test_demo_size_is_capped_at_two_thousand_rows_per_class(self):
        self.assertEqual(DEMO_MAX_ROWS_PER_CLASS, 2000)
        with self.assertRaises(SamplesError):
            run_demo(self.store, rows_per_class=2001)
        self.assertEqual(self.store.list_runs(), [])

    def test_rejects_bad_arguments_without_creating_runs(self):
        for kwargs in ({"seed": -1}, {"rows_per_class": 5}, {"rows_per_class": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises((SamplesError, SeedError)):
                run_demo(self.store, **kwargs)
        self.assertEqual(self.store.list_runs(), [])


if __name__ == "__main__":
    unittest.main()
