import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modelbench import selection
from modelbench.config import ExperimentConfig
from modelbench.datasets import Table, kfold_indices
from modelbench.errors import ModelBenchError, SweepError
from modelbench.metrics import accuracy
from modelbench.models import KNNClassifier
from modelbench.selection import describe, load_selection, run_selection


def signal_table(n=90, seed=5, noise_columns=3, noise=0.6):
    """Columns a and b carry the class; the rest are noise."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b = rng.uniform(-3, 3), rng.uniform(-3, 3)
        label = "p" if a + b + rng.gauss(0, noise) > 0 else "q"
        rows.append(
            (
                f"{a:.3f}",
                f"{b:.3f}",
                *(f"{rng.uniform(0, 1):.3f}" for _ in range(noise_columns)),
                label,
            )
        )
    columns = ("a", "b", *(f"n{i}" for i in range(noise_columns)), "label")
    return Table(columns, tuple(rows))


def config(**overrides):
    fields = {
        "experiment_id": "sel",
        "model": "knn",
        "seed": 3,
        "dataset_id": "rows",
        "metrics": ("accuracy",),
        "params": {"k": 5},
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


def by_hand(table, direction, folds=4, seed=3, tolerance=0.0, max_features=None, min_features=1):
    """A slow re-implementation of the greedy search using only public pieces."""
    names = [c for c in table.columns if c != "label"]
    y = [r[-1] for r in table.rows]
    splits = kfold_indices(len(y), folds, seed, None)

    def score(subset):
        positions = [table.columns.index(c) for c in subset]
        X = [[float(r[p]) for p in positions] for r in table.rows]
        values = []
        for train, test in splits:
            model = KNNClassifier(k=5)
            model.fit([X[i] for i in train], [y[i] for i in train])
            values.append(accuracy([y[i] for i in test], model.predict([X[i] for i in test])))
        return sum(values) / len(values)

    steps = []
    if direction == "forward":
        chosen, current = [], None
        while len(chosen) < (max_features or len(names)):
            tried = {c: score(chosen + [c]) for c in names if c not in chosen}
            best = max(tried, key=lambda c: (tried[c], -names.index(c)))
            if current is not None and not tried[best] - current > tolerance:
                break
            chosen.append(best)
            current = tried[best]
            steps.append((best, current, tried))
        return [c for c in names if c in chosen], current, steps
    kept, current = list(names), score(names)
    while len(kept) > min_features:
        tried = {c: score([k for k in kept if k != c]) for c in kept}
        best = max(tried, key=lambda c: (tried[c], -kept.index(c)))
        if not current - tried[best] <= tolerance:
            break
        kept.remove(best)
        current = tried[best]
        steps.append((best, current, tried))
    return kept, current, steps


class AgainstAHandBuiltSearchTests(unittest.TestCase):
    def check(self, direction, **kwargs):
        table = signal_table(n=72)
        result = run_selection(config(), table, "label", folds=4, direction=direction, **kwargs)
        selected, score, steps = by_hand(table, direction, **kwargs)
        self.assertEqual(list(result.selected), selected)
        self.assertAlmostEqual(result.score, score)
        self.assertEqual([s.feature for s in result.steps], [s[0] for s in steps])
        for got, want in zip(result.steps, steps, strict=True):
            self.assertAlmostEqual(got.score, want[1])
            self.assertEqual(list(got.tried), list(want[2]))
            for column, value in want[2].items():
                self.assertAlmostEqual(got.tried[column], value)
        return result

    def test_forward(self):
        result = self.check("forward")
        self.assertIn("a", result.selected)
        self.assertIn("b", result.selected)

    def test_backward(self):
        result = self.check("backward")
        self.assertIn("a", result.selected)
        self.assertIn("b", result.selected)

    def test_with_limits_and_tolerances(self):
        self.check("forward", max_features=2)
        self.check("forward", tolerance=0.05)
        self.check("backward", min_features=3)
        self.check("backward", tolerance=0.04)

    def test_the_all_columns_score_is_reported(self):
        table = signal_table(n=72)
        result = run_selection(config(), table, "label", folds=4)
        names = [c for c in table.columns if c != "label"]
        _, _, steps = by_hand(table, "backward", min_features=len(names))
        self.assertEqual(steps, [])
        backward = run_selection(config(), table, "label", folds=4, direction="backward")
        self.assertAlmostEqual(result.all_score, backward.all_score)
        self.assertAlmostEqual(backward.start_score, backward.all_score)
        self.assertIsNone(result.start_score)


class BehaviourTests(unittest.TestCase):
    def test_the_informative_columns_are_found_first(self):
        table = signal_table(n=120, noise=0.3)
        result = run_selection(config(), table, "label", folds=5)
        self.assertEqual({s.feature for s in result.steps[:2]}, {"a", "b"})
        first = result.steps[0].tried
        self.assertEqual(max(first, key=first.get), result.steps[0].feature)
        self.assertGreater(result.score, 0.85)

    def test_the_selected_columns_come_back_in_table_order(self):
        table = signal_table(n=120, noise=0.3)
        result = run_selection(config(), table, "label", folds=5)
        self.assertEqual(list(result.selected), [c for c in table.columns if c in result.selected])

    def test_a_large_tolerance_stops_forward_after_one_column_and_backward_at_the_floor(self):
        table = signal_table(n=72)
        forward = run_selection(config(), table, "label", folds=4, tolerance=1.0)
        self.assertEqual(len(forward.selected), 1)
        backward = run_selection(
            config(), table, "label", folds=4, direction="backward", tolerance=1.0
        )
        self.assertEqual(len(backward.selected), 1)
        floor = run_selection(
            config(), table, "label", folds=4, direction="backward", tolerance=1.0, min_features=3
        )
        self.assertEqual(len(floor.selected), 3)

    def test_max_features_caps_forward(self):
        result = run_selection(config(), signal_table(n=72), "label", folds=4, max_features=1)
        self.assertEqual(len(result.selected), 1)
        self.assertEqual(len(result.steps), 1)

    def test_ties_go_to_the_first_column_in_order(self):
        rng = random.Random(1)
        rows = []
        for _ in range(60):
            x = rng.uniform(-3, 3)
            rows.append((f"{x:.3f}", f"{x:.3f}", "p" if x > 0 else "q"))
        table = Table(("first", "second", "label"), tuple(rows))  # identical columns
        forward = run_selection(config(), table, "label", folds=4)
        self.assertEqual(forward.selected, ("first",))
        backward = run_selection(config(), table, "label", folds=4, direction="backward")
        self.assertEqual(backward.selected, ("second",))  # the first of the pair is dropped

    def test_lower_is_better_metrics_minimise(self):
        rng = random.Random(2)
        rows = []
        for _ in range(80):
            x = rng.uniform(0, 5)
            rows.append(
                (f"{x:.3f}", f"{rng.uniform(0, 5):.3f}", f"{3 * x + rng.gauss(0, 0.5):.3f}")
            )
        table = Table(("x", "noise", "y"), tuple(rows))
        result = run_selection(
            config(model="linear_regression", metrics=("mae",), params={}), table, "y", folds=4
        )
        self.assertFalse(result.higher_is_better)
        self.assertEqual(result.selected[0], "x")
        first = result.steps[0]
        self.assertEqual(first.score, min(first.tried.values()))
        self.assertLess(result.score, 1.0)

    def test_a_columns_subset_can_be_given(self):
        table = signal_table(n=72)
        result = run_selection(config(), table, "label", folds=4, feature_columns=["a", "n0", "n1"])
        self.assertEqual(result.features, ("a", "n0", "n1"))
        self.assertTrue(set(result.selected) <= {"a", "n0", "n1"})

    def test_every_subset_is_scored_on_the_same_folds_and_never_twice(self):
        table = signal_table(n=60)
        fits = []
        real = KNNClassifier.fit

        def spy(self, X, y):
            fits.append(len(X[0]))
            return real(self, X, y)

        with (
            mock.patch.object(selection, "kfold_indices", wraps=kfold_indices) as splitter,
            mock.patch.object(KNNClassifier, "fit", spy),
        ):
            result = run_selection(config(), table, "label", folds=3, max_features=2)
        self.assertEqual(splitter.call_count, 1)
        # the all-columns score, then 5 + 4 candidates over two steps, three folds each
        self.assertEqual(len(fits), (1 + 5 + 4) * 3)
        self.assertEqual(len(result.steps), 2)

    def test_the_result_depends_only_on_the_inputs(self):
        table = signal_table(n=72)
        a = run_selection(config(), table, "label", folds=4)
        self.assertEqual(a, run_selection(config(), table, "label", folds=4))
        other = run_selection(config(seed=9), table, "label", folds=4)
        self.assertNotEqual(a.steps, other.steps)

    def test_stratified_folds_are_used_when_asked(self):
        rows = tuple((f"{i * 0.37 % 5:.2f}", f"{i % 3}", "a" if i % 5 else "b") for i in range(50))
        table = Table(("x", "z", "label"), rows)
        labels = [r[-1] for r in rows]
        with mock.patch.object(selection, "kfold_indices", wraps=kfold_indices) as splitter:
            run_selection(config(stratify=True), table, "label", folds=5)
        self.assertEqual(splitter.call_args.args[3], labels)
        with mock.patch.object(selection, "kfold_indices", wraps=kfold_indices) as splitter:
            run_selection(config(), table, "label", folds=5)
        self.assertIsNone(splitter.call_args.args[3])

    def test_preprocessing_works_when_a_subset_leaves_a_categorical_column_out(self):
        rng = random.Random(4)
        rows = tuple(
            (f"{rng.uniform(0, 100):.1f}", rng.choice("uv"), "p" if rng.random() < 0.5 else "q")
            for _ in range(48)
        )
        table = Table(("size", "kind", "label"), rows)
        cfg = ExperimentConfig.from_dict(
            {"experiment_id": "s", "model": "knn", "seed": 1, "dataset_id": "d",
             "metrics": ["accuracy"], "params": {"k": 3},
             "preprocess": {"categorical": ["kind"], "scale": "standardize"}}
        )  # fmt: skip
        result = run_selection(cfg, table, "label", folds=3)
        self.assertEqual(set(result.features), {"size", "kind"})

    def test_describe_is_plain_data(self):
        result = run_selection(config(), signal_table(n=72), "label", folds=4)
        data = json.loads(json.dumps(describe(result)))
        self.assertEqual(data["selected"], list(result.selected))
        self.assertEqual(data["direction"], "forward")
        self.assertEqual(len(data["steps"]), len(result.steps))
        self.assertEqual(set(data["steps"][0]), {"feature", "score", "tried"})
        self.assertIsNone(data["start_score"])


class DirectionAndCacheTests(unittest.TestCase):
    def test_step_acceptance_rules_by_hand(self):
        exceeds, within = selection._exceeds, selection._within
        # higher is better: a gain must be bigger than the tolerance
        self.assertTrue(exceeds(0.06, 0.05, True))
        self.assertFalse(exceeds(0.05, 0.05, True))
        self.assertFalse(exceeds(-0.2, 0.05, True))
        # lower is better: a fall must be bigger than the tolerance
        self.assertTrue(exceeds(-0.06, 0.05, False))
        self.assertFalse(exceeds(-0.05, 0.05, False))
        self.assertFalse(exceeds(0.2, 0.05, False))
        # a drop is allowed up to and including the tolerance
        self.assertTrue(within(-0.05, 0.05, True))
        self.assertFalse(within(-0.06, 0.05, True))
        self.assertTrue(within(0.3, 0.0, True))
        self.assertTrue(within(0.05, 0.05, False))
        self.assertFalse(within(0.06, 0.05, False))
        self.assertTrue(within(-0.3, 0.0, False))

    def regression_table(self):
        rng = random.Random(2)
        rows = []
        for _ in range(80):
            x = rng.uniform(0, 5)
            rows.append(
                (f"{x:.3f}", f"{rng.uniform(0, 5):.3f}", f"{3 * x + rng.gauss(0, 0.5):.3f}")
            )
        return Table(("x", "noise", "y"), tuple(rows))

    def test_a_lower_is_better_metric_honours_tolerance_in_both_directions(self):
        cfg = config(model="linear_regression", metrics=("mae",), params={})
        table = self.regression_table()
        loose = run_selection(cfg, table, "y", folds=4, tolerance=100.0)
        self.assertEqual(loose.selected, ("x",))  # nothing after the first column beats 100
        strict = run_selection(cfg, table, "y", folds=4, tolerance=0.0)
        self.assertEqual(strict.selected[0], "x")
        backward = run_selection(cfg, table, "y", folds=4, direction="backward", tolerance=0.0)
        self.assertEqual(backward.selected, ("x",))  # dropping the noise column does not hurt
        keep_all = run_selection(
            cfg, table, "y", folds=4, direction="backward", tolerance=0.0, min_features=2
        )
        self.assertEqual(keep_all.selected, ("x", "noise"))

    def test_seeded_models_get_the_fold_seeds_cross_validation_uses(self):
        table = signal_table(n=60, noise_columns=1)
        cfg = config(model="random_forest", params={"n_estimators": 4, "max_depth": 3})
        result = run_selection(cfg, table, "label", folds=3, max_features=1)
        y = [r[-1] for r in table.rows]
        from modelbench.models import RandomForest
        from modelbench.seeding import derive_seed

        expected = {}
        for column in ("a", "b", "n0"):
            p = table.columns.index(column)
            X = [[float(r[p])] for r in table.rows]
            scores = []
            for number, (train, test) in enumerate(kfold_indices(60, 3, 3, None)):
                model = RandomForest(
                    n_estimators=4, max_depth=3, seed=derive_seed(3, f"model-fold-{number}")
                )
                model.fit([X[i] for i in train], [y[i] for i in train])
                scores.append(accuracy([y[i] for i in test], model.predict([X[i] for i in test])))
            expected[column] = sum(scores) / 3
        for column, value in expected.items():
            self.assertAlmostEqual(result.steps[0].tried[column], value)

    def test_a_subset_is_never_scored_twice(self):
        rng = random.Random(6)
        rows = []
        for _ in range(60):
            a, b = rng.uniform(-3, 3), rng.uniform(-3, 3)
            rows.append((f"{a:.3f}", f"{b:.3f}", "p" if a + 2 * b > 0 else "q"))
        table = Table(("a", "b", "label"), tuple(rows))
        fits = []
        real = KNNClassifier.fit

        def spy(self, X, y):
            fits.append(1)
            return real(self, X, y)

        with mock.patch.object(KNNClassifier, "fit", spy):
            result = run_selection(config(), table, "label", folds=3)
        self.assertEqual([s.feature for s in result.steps], ["b", "a"])  # b, then a: the set (b, a)
        # the all-columns score is computed once and reused when the second step reaches it
        self.assertEqual(len(fits), (1 + 2) * 3)


class ValidationTests(unittest.TestCase):
    def run_with(self, **kwargs):
        table = kwargs.pop("table", signal_table(n=48))
        cfg = kwargs.pop("cfg", config())
        args = {"folds": 3}
        args.update(kwargs)
        return run_selection(cfg, table, "label", **args)

    def test_bad_options(self):
        cases = [
            {"folds": 1},
            {"folds": 51},
            {"folds": 2.5},
            {"direction": "sideways"},
            {"tolerance": -0.1},
            {"tolerance": float("nan")},
            {"tolerance": "0"},
            {"max_features": 0},
            {"max_features": 99},
            {"min_features": 0},
            {"min_features": 99},
            {"max_features": 1, "min_features": 2},
            {"rank_by": "f1"},
            {"feature_columns": ["nope"]},
            {"feature_columns": ["label"]},
            {"feature_columns": []},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ModelBenchError):
                self.run_with(**kwargs)

    def test_the_config_must_leave_the_folds_to_selection(self):
        with self.assertRaises(SweepError):
            self.run_with(cfg=config(cv_folds=3))

    def test_a_bad_model_or_metric_fails_before_anything_is_scored(self):
        with mock.patch.object(KNNClassifier, "fit") as fit:
            with self.assertRaises(ModelBenchError):
                self.run_with(cfg=config(params={"nope": 1}))
            with self.assertRaises(ModelBenchError):
                self.run_with(cfg=config(metrics=("mae",)))
        fit.assert_not_called()

    def test_stratify_is_rejected_for_regression(self):
        table = Table(("x", "y"), tuple((str(i), str(2.0 * i)) for i in range(30)))
        cfg = config(model="linear_regression", metrics=("mae",), params={}, stratify=True)
        with self.assertRaises(SweepError):
            run_selection(cfg, table, "y", folds=3)

    def test_a_fold_that_cannot_be_scored_is_reported_before_any_fit(self):
        rows = tuple((str(i), str(i % 3), "a" if i < 27 else "b") for i in range(30))
        table = Table(("x", "z", "label"), rows)
        with mock.patch.object(KNNClassifier, "fit") as fit, self.assertRaises(ModelBenchError):
            run_selection(config(metrics=("roc_auc",)), table, "label", folds=3)
        fit.assert_not_called()

    def test_too_many_columns_for_the_fit_budget(self):
        columns = tuple(f"c{i}" for i in range(250))
        rows = tuple((*(str(i % 7) for _ in columns), "a" if i % 2 else "b") for i in range(12))
        table = Table((*columns, "label"), rows)
        with self.assertRaises(SweepError) as raised:
            run_selection(config(), table, "label", folds=3)
        self.assertIn("limit", str(raised.exception))


class FileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "selection.json"

    def document(self, **changes):
        document = {
            "config": {"experiment_id": "s", "model": "knn", "seed": 1, "dataset_id": "d",
                       "metrics": ["accuracy"], "params": {}},
            "folds": 4,
        }  # fmt: skip
        document.update(changes)
        return document

    def load(self, document):
        self.path.write_text(json.dumps(document))
        return load_selection(self.path)

    def test_defaults_and_overrides(self):
        spec = self.load(self.document())
        self.assertEqual(
            (spec.folds, spec.direction, spec.max_features, spec.min_features, spec.tolerance),
            (4, "forward", None, 1, 0.0),
        )
        spec = self.load(
            self.document(direction="backward", max_features=3, min_features=2, tolerance=0.01)
        )
        self.assertEqual(
            (spec.direction, spec.max_features, spec.min_features, spec.tolerance),
            ("backward", 3, 2, 0.01),
        )

    def test_bad_files(self):
        no_folds = self.document()
        del no_folds["folds"]
        cases = {
            "not an object": [],
            "unknown key": self.document(extra=1),
            "no folds": no_folds,
            "folds too small": self.document(folds=1),
            "bad direction": self.document(direction="both"),
            "bad max": self.document(max_features=0),
            "null min": self.document(min_features=None),
            "negative tolerance": self.document(tolerance=-1),
            "text tolerance": self.document(tolerance="x"),
        }
        for name, document in cases.items():
            with self.subTest(name=name), self.assertRaises(ModelBenchError):
                self.load(document)

    def test_an_unreadable_file(self):
        with self.assertRaises(SweepError):
            load_selection(self.path)


if __name__ == "__main__":
    unittest.main()
