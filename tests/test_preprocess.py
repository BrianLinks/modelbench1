import math
import unittest

from modelbench.config import ExperimentConfig
from modelbench.datasets import Table
from modelbench.errors import ConfigError, PreprocessError, RunnerError
from modelbench.preprocess import (
    Preprocessor,
    PreprocessSpec,
    fit_transform_split,
    is_missing,
    parse_cells,
)
from modelbench.reproducibility import check_reproducibility
from modelbench.runner import run_experiment
from modelbench.store import Store

ENV = {"python": "3.12.1", "lib:numpy": None}
NAMES = ["age", "city", "income"]
ROWS = [
    ["20", "Paris", "100"],
    ["30", "Rome", "NA"],
    ["40", "Paris", "300"],
    ["", "", "500"],
]


def cells(spec, rows=ROWS, names=NAMES):
    return parse_cells(rows, names, spec)


class SpecTests(unittest.TestCase):
    def test_defaults_and_round_trip(self):
        spec = PreprocessSpec()
        self.assertEqual(
            spec.to_dict(),
            {
                "categorical": [],
                "impute": "error",
                "fill_value": 0.0,
                "scale": "none",
                "missing_indicator": False,
                "max_categories": 50,
            },
        )
        full = PreprocessSpec(("a", "b"), "median", 2.5, "minmax", True, 7)
        self.assertEqual(PreprocessSpec.from_dict(full.to_dict()), full)

    def test_rejects_bad_settings(self):
        bad = [
            {"categorical": "city"},
            {"categorical": ["a", "a"]},
            {"categorical": [""]},
            {"impute": "mode"},
            {"fill_value": True},
            {"fill_value": float("inf")},
            {"fill_value": "0"},
            {"scale": "log"},
            {"missing_indicator": 1},
            {"max_categories": 0},
            {"max_categories": 1001},
            {"max_categories": True},
            {"max_categories": 2.5},
            {"bogus": 1},
        ]
        for data in bad:
            with self.subTest(data), self.assertRaises(PreprocessError):
                PreprocessSpec.from_dict(data)
        with self.assertRaises(PreprocessError):
            PreprocessSpec.from_dict([])
        PreprocessSpec.from_dict({"max_categories": 1000})
        PreprocessSpec.from_dict({"max_categories": 1})


class ParseTests(unittest.TestCase):
    def test_missing_tokens(self):
        for token in ("", " ", "NA", "n/a", "NaN", "null", "None", "?", " na "):
            self.assertTrue(is_missing(token), token)
        for token in ("0", "no", "-", "nan1"):
            self.assertFalse(is_missing(token), token)

    def test_numbers_text_and_missing(self):
        spec = PreprocessSpec(categorical=("city",), impute="mean")
        self.assertEqual(
            cells(spec),
            [
                [20.0, "Paris", 100.0],
                [30.0, "Rome", None],
                [40.0, "Paris", 300.0],
                [None, None, 500.0],
            ],
        )

    def test_a_bad_number_names_its_row_and_column(self):
        spec = PreprocessSpec(categorical=("city",))
        rows = [["1", "a", "2"], ["x", "a", "2"]]
        with self.assertRaisesRegex(PreprocessError, r"row 2, column 'age': 'x' is not a finite"):
            parse_cells(rows, NAMES, spec)
        for text in ("inf", "1e999", "١٢", "1,5", "0x10"):
            with self.subTest(text), self.assertRaises(PreprocessError):
                parse_cells([["1", "a", text]], NAMES, spec)

    def test_categorical_columns_must_be_features(self):
        with self.assertRaisesRegex(PreprocessError, "not features"):
            parse_cells([["1", "a", "2"]], NAMES, PreprocessSpec(categorical=("zip",)))

    def test_text_in_a_categorical_column_is_kept_as_is(self):
        spec = PreprocessSpec(categorical=("city",))
        got = parse_cells([["1", "  New York ", "2"]], NAMES, spec)
        self.assertEqual(got[0][1], "New York")


class FitTransformTests(unittest.TestCase):
    def fit(self, **settings):
        spec = PreprocessSpec(categorical=("city",), **settings)
        return Preprocessor.fit(spec, NAMES, cells(spec))

    def test_mean_imputation_and_one_hot_with_a_missing_category(self):
        pre = self.fit(impute="mean")
        self.assertEqual(
            pre.feature_names, ["age", "city=Paris", "city=Rome", "city=<missing>", "income"]
        )
        spec = PreprocessSpec(categorical=("city",), impute="mean")
        got = pre.transform(cells(spec))
        self.assertEqual(
            got,
            [
                [20.0, 1.0, 0.0, 0.0, 100.0],
                [30.0, 0.0, 1.0, 0.0, 300.0],  # income filled with (100 + 300 + 500) / 3
                [40.0, 1.0, 0.0, 0.0, 300.0],
                [30.0, 0.0, 0.0, 1.0, 500.0],  # age filled with (20 + 30 + 40) / 3
            ],
        )

    def test_median_uses_the_middle_or_the_average_of_the_two_middle_values(self):
        spec = PreprocessSpec(impute="median")
        rows = [["1"], ["2"], ["10"], ["50"], ["NA"]]
        pre = Preprocessor.fit(spec, ["x"], parse_cells(rows, ["x"], spec))
        self.assertEqual(pre.transform([[None]]), [[6.0]])  # (2 + 10) / 2
        odd = Preprocessor.fit(spec, ["x"], parse_cells([["1"], ["9"], ["4"], ["NA"]], ["x"], spec))
        self.assertEqual(odd.transform([[None]]), [[4.0]])

    def test_constant_imputation(self):
        spec = PreprocessSpec(impute="constant", fill_value=-1.5)
        pre = Preprocessor.fit(spec, ["x"], parse_cells([["1"], ["NA"]], ["x"], spec))
        self.assertEqual(pre.transform([[None], [3.0]]), [[-1.5], [3.0]])

    def test_a_missing_number_is_an_error_by_default(self):
        spec = PreprocessSpec()
        with self.assertRaisesRegex(PreprocessError, "column 'x' has missing values"):
            Preprocessor.fit(spec, ["x"], parse_cells([["1"], ["NA"]], ["x"], spec))

    def test_a_column_with_no_values_cannot_be_imputed_from_data(self):
        spec = PreprocessSpec(impute="mean")
        with self.assertRaisesRegex(PreprocessError, "no values in the training rows"):
            Preprocessor.fit(spec, ["x"], parse_cells([["NA"], ["NA"]], ["x"], spec))
        constant = PreprocessSpec(impute="constant", fill_value=4.0)
        pre = Preprocessor.fit(constant, ["x"], parse_cells([["NA"]], ["x"], constant))
        self.assertEqual(pre.transform([[None]]), [[4.0]])

    def test_standardize_uses_the_population_standard_deviation(self):
        spec = PreprocessSpec(categorical=("city",), impute="mean", scale="standardize")
        pre = Preprocessor.fit(spec, NAMES, cells(spec))
        got = pre.transform(cells(spec))
        # age filled: 20, 30, 40, 30 -> mean 30, variance (100 + 0 + 100 + 0) / 4 = 50
        self.assertAlmostEqual(got[0][0], -10 / math.sqrt(50))
        self.assertAlmostEqual(got[2][0], 10 / math.sqrt(50))
        self.assertEqual(got[1][0], 0.0)
        # income filled: 100, 300, 300, 500 -> mean 300, variance 20000
        self.assertAlmostEqual(got[0][4], -200 / math.sqrt(20000))
        self.assertEqual(got[0][1:4], [1.0, 0.0, 0.0])  # one-hot columns are not scaled

    def test_minmax_scales_to_the_training_range(self):
        spec = PreprocessSpec(categorical=("city",), impute="mean", scale="minmax")
        pre = Preprocessor.fit(spec, NAMES, cells(spec))
        got = pre.transform(cells(spec))
        self.assertEqual([row[0] for row in got], [0.0, 0.5, 1.0, 0.5])
        outside = pre.transform([[60.0, "Paris", 100.0]])
        self.assertEqual(outside[0][0], 2.0)  # values beyond the training range are not clipped

    def test_a_constant_column_becomes_zero_when_scaling(self):
        for scale in ("standardize", "minmax"):
            spec = PreprocessSpec(scale=scale)
            pre = Preprocessor.fit(spec, ["x"], parse_cells([["5"], ["5"]], ["x"], spec))
            self.assertEqual(pre.transform([[5.0]]), [[0.0]], scale)

    def test_missing_indicator_only_for_columns_with_missing_training_values(self):
        pre = self.fit(impute="mean", missing_indicator=True)
        self.assertEqual(
            pre.feature_names,
            ["age", "age_missing", "city=Paris", "city=Rome", "city=<missing>", "income",
             "income_missing"],
        )  # fmt: skip
        spec = PreprocessSpec(categorical=("city",), impute="mean", missing_indicator=True)
        got = pre.transform(cells(spec))
        self.assertEqual([row[1] for row in got], [0.0, 0.0, 0.0, 1.0])
        self.assertEqual([row[6] for row in got], [0.0, 1.0, 0.0, 0.0])
        flat = PreprocessSpec(impute="mean", missing_indicator=True)
        clean = Preprocessor.fit(flat, ["a"], parse_cells([["1"], ["2"]], ["a"], flat))
        self.assertEqual(clean.feature_names, ["a"])  # no missing seen, so no indicator column

    def test_an_unseen_category_is_all_zeros_and_a_missing_one_without_training_is_too(self):
        spec = PreprocessSpec(categorical=("c",))
        pre = Preprocessor.fit(spec, ["c"], parse_cells([["a"], ["b"]], ["c"], spec))
        self.assertEqual(pre.feature_names, ["c=a", "c=b"])
        self.assertEqual(
            pre.transform([["z"], [None], ["a"]]), [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
        )

    def test_too_many_categories(self):
        spec = PreprocessSpec(categorical=("c",), max_categories=2)
        rows = [["a"], ["b"], ["c"]]
        with self.assertRaisesRegex(PreprocessError, "3 categories .* limit is 2"):
            Preprocessor.fit(spec, ["c"], parse_cells(rows, ["c"], spec))
        Preprocessor.fit(spec, ["c"], parse_cells(rows[:2], ["c"], spec))

    def test_categories_are_sorted_not_in_order_of_appearance(self):
        spec = PreprocessSpec(categorical=("c",))
        pre = Preprocessor.fit(spec, ["c"], parse_cells([["b"], ["a"], ["c"]], ["c"], spec))
        self.assertEqual(pre.feature_names, ["c=a", "c=b", "c=c"])

    def test_no_rows(self):
        with self.assertRaises(PreprocessError):
            Preprocessor.fit(PreprocessSpec(), ["x"], [])


class NoLeakageTests(unittest.TestCase):
    def test_statistics_come_from_the_training_rows_only(self):
        spec = PreprocessSpec(impute="mean", scale="minmax")
        data = parse_cells([["0"], ["10"], ["1000"], ["NA"]], ["x"], spec)
        train, test, names = fit_transform_split(spec, ["x"], data, [0, 1], [2, 3])
        self.assertEqual(train, [[0.0], [1.0]])
        self.assertEqual(test, [[100.0], [0.5]])  # 1000 / 10; NA filled with the train mean, 5
        self.assertEqual(names, ["x"])


def city_table():
    rows = []
    for i in range(30):
        city = ("Paris", "Rome", "Oslo")[i % 3]
        label = {"Paris": "a", "Rome": "b", "Oslo": "c"}[city]
        noise = "NA" if i % 7 == 0 else str((i * 37) % 11)
        rows.append((city, noise, label))
    return Table(("city", "noise", "label"), tuple(rows))


def make_config(**overrides):
    fields = {
        "experiment_id": "pp",
        "model": "decision_tree",
        "seed": 5,
        "dataset_id": "cities",
        "metrics": ("accuracy",),
        "params": {},
        "stratify": True,
        "preprocess": PreprocessSpec(categorical=("city",), impute="median", scale="standardize"),
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


class ConfigTests(unittest.TestCase):
    def test_round_trip_and_hash(self):
        config = make_config()
        again = ExperimentConfig.from_dict(config.to_dict())
        self.assertEqual(again, config)
        self.assertEqual(again.config_hash(), config.config_hash())
        plain = make_config(preprocess=None)
        self.assertNotIn("preprocess", plain.to_dict())  # absent when unused: old hashes hold
        self.assertNotEqual(plain.config_hash(), config.config_hash())
        other = make_config(preprocess=PreprocessSpec(categorical=("city",), impute="mean"))
        self.assertNotEqual(other.config_hash(), config.config_hash())

    def test_bad_values(self):
        with self.assertRaises(PreprocessError):
            ExperimentConfig.from_dict({**make_config().to_dict(), "preprocess": {"scale": "x"}})
        with self.assertRaises(ConfigError):
            make_config(preprocess={"scale": "none"})


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)

    def test_without_preprocessing_text_is_still_rejected(self):
        with self.assertRaisesRegex(RunnerError, "not a finite number"):
            run_experiment(
                self.store, make_config(preprocess=None), city_table(), "label", environment=ENV
            )

    def test_categories_and_missing_values_are_handled_and_recorded(self):
        result = run_experiment(self.store, make_config(), city_table(), "label", environment=ENV)
        self.assertEqual(result.metrics["accuracy"], 1.0)  # the city decides the label
        tags = self.store.get_run(result.run_id).tags
        self.assertEqual(tags["n_features"], "4")  # three cities and the noise column
        self.assertEqual(self.store.get_config(result.run_id).preprocess.scale, "standardize")

    def test_cross_validation_fits_preprocessing_in_every_fold(self):
        config = make_config(cv_folds=3)
        result = run_experiment(self.store, config, city_table(), "label", environment=ENV)
        self.assertEqual(result.metrics["accuracy"], 1.0)
        self.assertEqual(result.n_test, 30)

    def test_bad_cells_are_found_before_a_run_exists(self):
        bad = Table(("city", "noise", "label"), (("a", "x", "p"), ("b", "1", "q")))
        with self.assertRaisesRegex(PreprocessError, "row 1, column 'noise'"):
            run_experiment(self.store, make_config(), bad, "label", environment=ENV)
        self.assertEqual(self.store.list_runs(), [])

    def test_a_training_problem_is_found_before_a_run_exists(self):
        config = make_config(preprocess=PreprocessSpec(categorical=("city",), impute="error"))
        with self.assertRaisesRegex(PreprocessError, "missing values"):
            run_experiment(self.store, config, city_table(), "label", environment=ENV)
        self.assertEqual(self.store.list_runs(), [])

    def test_a_category_only_in_the_test_rows_does_not_break_prediction(self):
        rows = [("a", "1", "x")] * 6 + [("b", "2", "y")] * 6 + [("c", "3", "x")]
        table = Table(("city", "noise", "label"), tuple(rows))
        config = make_config(
            stratify=False, test_fraction=0.5, params={"max_depth": 2}, seed=2,
            preprocess=PreprocessSpec(categorical=("city",), scale="none"),
        )  # fmt: skip
        result = run_experiment(self.store, config, table, "label", environment=ENV)
        self.assertGreaterEqual(result.metrics["accuracy"], 0.0)

    def test_a_preprocessed_run_reproduces(self):
        table = city_table()
        result = run_experiment(self.store, make_config(), table, "label", environment=ENV)
        report = check_reproducibility(self.store, result.run_id, table, "label", environment=ENV)
        self.assertTrue(report.reproduced, report.detail)

    def test_n_features_is_the_widest_fold(self):
        rows = [("a", "1", "x")] * 5 + [("b", "2", "y")] * 5 + [("c", "3", "x")]
        table = Table(("city", "noise", "label"), tuple(rows))
        config = make_config(
            stratify=False, cv_folds=3, seed=1, params={"max_depth": 2},
            preprocess=PreprocessSpec(categorical=("city",)),
        )  # fmt: skip
        result = run_experiment(self.store, config, table, "label", environment=ENV)
        # the fold that tests the lone "c" row trains on two cities (3 columns); the others
        # train on all three (4 columns)
        self.assertEqual(self.store.get_run(result.run_id).tags["n_features"], "4")

    def test_plain_runs_do_not_get_an_n_features_tag(self):
        table = Table(("x", "label"), tuple((str(i), "a" if i < 10 else "b") for i in range(20)))
        config = make_config(preprocess=None, dataset_id="plain")
        result = run_experiment(self.store, config, table, "label", environment=ENV)
        self.assertNotIn("n_features", self.store.get_run(result.run_id).tags)


if __name__ == "__main__":
    unittest.main()
