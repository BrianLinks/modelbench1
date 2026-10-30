import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modelbench.config import ExperimentConfig
from modelbench.datasets import Table, kfold_indices
from modelbench.errors import ModelBenchError, ModelError, SweepError
from modelbench.metrics import accuracy
from modelbench.models import KNNClassifier, RandomForest
from modelbench.nested import (
    MAX_CANDIDATES,
    NestedResult,
    OuterFold,
    candidate_params,
    load_nested,
    run_nested,
)
from modelbench.seeding import derive_seed
from modelbench.sweep import check_space, sample_params


def noisy_table(n=60, seed=3, noise=1.5, extra=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b = rng.uniform(-3, 3), rng.uniform(-3, 3)
        label = "p" if a + b + rng.gauss(0, noise) > 0 else "q"
        rows.append(
            (f"{a:.3f}", f"{b:.3f}", *(f"{rng.uniform(0, 1):.3f}" for _ in range(extra)), label)
        )
    columns = ("a", "b", *(f"n{i}" for i in range(extra)), "label")
    return Table(columns, tuple(rows))


def config(**overrides):
    fields = {
        "experiment_id": "nested",
        "model": "knn",
        "seed": 4,
        "dataset_id": "rows",
        "metrics": ("accuracy",),
        "params": {},
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


KS = [{"k": 1}, {"k": 3}, {"k": 7}, {"k": 15}]


class IndependentRecomputationTests(unittest.TestCase):
    """The result must equal a slow, separate re-implementation that uses only the public pieces."""

    def expected(self, table, candidates, outer_k, inner_k, seed):
        X = [[float(r[0]), float(r[1])] for r in table.rows]
        y = [r[-1] for r in table.rows]
        out = []
        for number, (train, test) in enumerate(kfold_indices(len(y), outer_k, seed, None)):
            inner = kfold_indices(
                len(train), inner_k, derive_seed(seed, "nested-inner", number), None
            )
            means = []
            for params in candidates:
                scores = []
                for tr, te in inner:
                    rows_tr = [train[p] for p in tr]
                    rows_te = [train[p] for p in te]
                    model = KNNClassifier(**params)
                    model.fit([X[i] for i in rows_tr], [y[i] for i in rows_tr])
                    pred = model.predict([X[i] for i in rows_te])
                    scores.append(accuracy([y[i] for i in rows_te], pred))
                means.append(sum(scores) / len(scores))
            chosen = max(range(len(candidates)), key=lambda i: (means[i], -i))
            model = KNNClassifier(**candidates[chosen])
            model.fit([X[i] for i in train], [y[i] for i in train])
            pred = model.predict([X[i] for i in test])
            out.append((chosen, means[chosen], accuracy([y[i] for i in test], pred)))
        return out

    def test_matches_by_hand_for_several_shapes(self):
        table = noisy_table(n=48)
        for outer_k, inner_k, seed in [(4, 3, 4), (3, 2, 9), (6, 4, 1)]:
            with self.subTest(outer=outer_k, inner=inner_k, seed=seed):
                result = run_nested(
                    config(seed=seed), KS, table, "label", outer_folds=outer_k, inner_folds=inner_k
                )
                want = self.expected(table, KS, outer_k, inner_k, seed)
                got = [(f.chosen, f.inner_score, f.test_scores["accuracy"]) for f in result.outer]
                self.assertEqual(len(got), outer_k)
                for (c1, i1, t1), (c2, i2, t2) in zip(got, want, strict=True):
                    self.assertEqual(c1, c2)
                    self.assertAlmostEqual(i1, i2)
                    self.assertAlmostEqual(t1, t2)

    def test_stratified_splits_match_by_hand(self):
        rows = tuple(
            (f"{(i * 7) % 11 + (i % 5) * 0.1:.2f}", f"{i % 4}", "a" if i % 4 else "b")
            for i in range(48)
        )
        table = Table(("x", "z", "label"), rows)
        X = [[float(r[0]), float(r[1])] for r in rows]
        y = [r[-1] for r in rows]
        seed, candidates = 4, [{"k": 1}, {"k": 5}]
        result = run_nested(
            config(stratify=True), candidates, table, "label", outer_folds=4, inner_folds=3
        )
        for number, (train, test) in enumerate(kfold_indices(48, 4, seed, y)):
            inner = kfold_indices(
                len(train), 3, derive_seed(seed, "nested-inner", number), [y[i] for i in train]
            )
            means = []
            for params in candidates:
                scores = []
                for tr, te in inner:
                    a, b = [train[p] for p in tr], [train[p] for p in te]
                    model = KNNClassifier(**params)
                    model.fit([X[i] for i in a], [y[i] for i in a])
                    scores.append(accuracy([y[i] for i in b], model.predict([X[i] for i in b])))
                means.append(sum(scores) / 3)
            chosen = max(range(2), key=lambda i: (means[i], -i))
            model = KNNClassifier(**candidates[chosen])
            model.fit([X[i] for i in train], [y[i] for i in train])
            want = accuracy([y[i] for i in test], model.predict([X[i] for i in test]))
            fold = result.outer[number]
            self.assertEqual(fold.chosen, chosen)
            self.assertAlmostEqual(fold.inner_score, means[chosen])
            self.assertAlmostEqual(fold.test_scores["accuracy"], want)

    def test_seeded_models_get_the_documented_fold_seeds(self):
        table = noisy_table(n=45)
        X = [[float(r[0]), float(r[1])] for r in table.rows]
        y = [r[-1] for r in table.rows]
        seed = 4
        candidates = [{"n_estimators": 3, "max_depth": 2}, {"n_estimators": 4}]

        def forest(params, fold):
            return RandomForest(**params, seed=derive_seed(seed, f"model-fold-{fold}"))

        result = run_nested(
            config(model="random_forest"), candidates, table, "label", outer_folds=3, inner_folds=2
        )
        for number, (train, test) in enumerate(kfold_indices(45, 3, seed, None)):
            inner = kfold_indices(len(train), 2, derive_seed(seed, "nested-inner", number), None)
            means = []
            for params in candidates:
                scores = []
                for j, (tr, te) in enumerate(inner):
                    a, b = [train[p] for p in tr], [train[p] for p in te]
                    model = forest(params, (number + 1) * 1000 + j)
                    model.fit([X[i] for i in a], [y[i] for i in a])
                    scores.append(accuracy([y[i] for i in b], model.predict([X[i] for i in b])))
                means.append(sum(scores) / 2)
            chosen = max(range(2), key=lambda i: (means[i], -i))
            model = forest(candidates[chosen], number)
            model.fit([X[i] for i in train], [y[i] for i in train])
            want = accuracy([y[i] for i in test], model.predict([X[i] for i in test]))
            fold = result.outer[number]
            self.assertEqual(fold.chosen, chosen)
            self.assertAlmostEqual(fold.inner_score, means[chosen])
            self.assertAlmostEqual(fold.test_scores["accuracy"], want)

    def test_summary_numbers_follow_from_the_folds(self):
        table = noisy_table(n=48)
        result = run_nested(config(), KS, table, "label", outer_folds=4, inner_folds=3)
        tests = [f.test_scores["accuracy"] for f in result.outer]
        mean = sum(tests) / 4
        std = (sum((t - mean) ** 2 for t in tests) / 4) ** 0.5
        self.assertAlmostEqual(result.estimate[0], mean)
        self.assertAlmostEqual(result.estimate[1], std)
        inner = sum(f.inner_score for f in result.outer) / 4
        self.assertAlmostEqual(result.selection_score, inner)
        self.assertAlmostEqual(result.optimism, inner - mean)
        self.assertEqual(sum(result.choice_counts()), 4)
        self.assertEqual([f.train_size + f.test_size for f in result.outer], [48] * 4)


class NoLeakageTests(unittest.TestCase):
    def test_selection_never_sees_an_outer_test_row_and_the_final_fit_uses_the_whole_train_set(
        self,
    ):
        # each row has a unique first feature, so a fit's rows can be read off its inputs
        rng = random.Random(1)
        rows = tuple((str(i), f"{rng.uniform(0, 1):.3f}", "p" if i % 3 else "q") for i in range(36))
        table = Table(("id", "x", "label"), rows)
        fits = []
        real = KNNClassifier.fit

        def spy(self, X, y):
            fits.append(sorted(int(r[0]) for r in X))
            return real(self, X, y)

        candidates = [{"k": 1}, {"k": 3}]
        with mock.patch.object(KNNClassifier, "fit", spy):
            run_nested(config(seed=2), candidates, table, "label", outer_folds=3, inner_folds=2)
        per_outer = len(candidates) * 2 + 1
        self.assertEqual(len(fits), 3 * per_outer)
        for number, (train, test) in enumerate(kfold_indices(36, 3, 2, None)):
            block = fits[number * per_outer : (number + 1) * per_outer]
            for rows_used in block[:-1]:  # candidate fits inside the inner folds
                self.assertTrue(set(rows_used) <= set(train))
                self.assertFalse(set(rows_used) & set(test))
                self.assertLess(len(rows_used), len(train))
            self.assertEqual(block[-1], train)  # the refit uses all of the outer training set


class RunNestedBehaviourTests(unittest.TestCase):
    def test_the_result_is_a_pure_function_of_its_inputs(self):
        table = noisy_table()
        args = (config(), KS, table, "label")
        a = run_nested(*args, outer_folds=3, inner_folds=3)
        b = run_nested(*args, outer_folds=3, inner_folds=3)
        self.assertEqual(a, b)
        c = run_nested(config(seed=5), KS, table, "label", outer_folds=3, inner_folds=3)
        self.assertNotEqual(a.outer, c.outer)

    def test_ties_go_to_the_earlier_candidate(self):
        table = noisy_table(n=30)
        same = [{"k": 3}, {"k": 3, "weights": "uniform"}, {"k": 3, "metric": "euclidean"}]
        result = run_nested(config(), same, table, "label", outer_folds=3, inner_folds=2)
        self.assertEqual({f.chosen for f in result.outer}, {0})

    def test_lower_is_better_metrics_pick_the_smallest_and_flip_the_optimism_sign(self):
        rng = random.Random(2)
        rows = tuple(
            (f"{x:.3f}", f"{3 * x + rng.gauss(0, 1):.3f}")
            for x in (rng.uniform(0, 5) for _ in range(60))
        )
        table = Table(("x", "y"), rows)
        result = run_nested(
            config(model="knn_regressor", metrics=("mae",)),
            [{"k": 1}, {"k": 5}, {"k": 25}],
            table,
            "y",
            outer_folds=4,
            inner_folds=3,
        )
        self.assertFalse(result.higher_is_better)
        for fold in result.outer:
            scores = [s for s in fold.inner_scores if s is not None]
            self.assertEqual(fold.inner_score, min(scores))
        gap = result.selection_score - result.estimate[0]
        self.assertAlmostEqual(result.optimism, -gap)

    def test_a_candidate_that_cannot_be_fitted_is_skipped_not_chosen(self):
        table = noisy_table(n=30)
        result = run_nested(
            config(), [{"k": 1}, {"k": 500}], table, "label", outer_folds=3, inner_folds=2
        )
        for fold in result.outer:
            self.assertIsNone(fold.inner_scores[1])
            self.assertEqual(fold.chosen, 0)

    def test_when_every_candidate_fails_it_is_an_error(self):
        with self.assertRaises(SweepError) as raised:
            run_nested(
                config(), [{"k": 500}], noisy_table(n=30), "label", outer_folds=3, inner_folds=2
            )
        self.assertIn("outer fold 0", str(raised.exception))

    def test_a_bad_parameter_set_is_rejected_before_any_model_is_fitted(self):
        with mock.patch.object(KNNClassifier, "fit") as fit, self.assertRaises(ModelError):
            run_nested(
                config(),
                [{"k": 3}, {"nope": 1}],
                noisy_table(),
                "label",
                outer_folds=3,
                inner_folds=2,
            )
        fit.assert_not_called()

    def test_other_metrics_are_scored_on_the_outer_folds(self):
        result = run_nested(
            config(metrics=("accuracy", "f1")), KS, noisy_table(), "label",
            outer_folds=3, inner_folds=3,
        )  # fmt: skip
        for fold in result.outer:
            self.assertEqual(set(fold.test_scores), {"accuracy", "f1"})
        self.assertEqual(set(result.metric_summary()), {"accuracy", "f1"})

    def test_selecting_by_another_metric(self):
        table = noisy_table()
        cfg = config(metrics=("accuracy", "f1"))
        by_f1 = run_nested(cfg, KS, table, "label", outer_folds=3, inner_folds=3, rank_by="f1")
        self.assertEqual(by_f1.metric, "f1")
        for fold in by_f1.outer:
            best = max(s for s in fold.inner_scores if s is not None)
            self.assertEqual(fold.inner_score, best)

    def test_stratification_balances_the_outer_test_folds(self):
        rows = tuple(
            (f"{i * 0.1:.1f}", f"{(i % 7) * 0.3:.1f}", "a" if i % 5 else "b") for i in range(50)
        )
        table = Table(("x", "z", "label"), rows)  # 40 a, 10 b
        result = run_nested(
            config(stratify=True),
            [{"k": 1}, {"k": 3}],
            table,
            "label",
            outer_folds=5,
            inner_folds=2,
        )
        self.assertEqual([f.test_size for f in result.outer], [10] * 5)

    def test_preprocessing_is_applied_per_split(self):
        rng = random.Random(4)
        rows = tuple(
            (
                f"{rng.uniform(0, 100):.2f}",
                rng.choice(["u", "v"]),
                "p" if rng.random() < 0.5 else "q",
            )
            for _ in range(40)
        )
        table = Table(("size", "kind", "label"), rows)
        cfg = ExperimentConfig.from_dict(
            {
                "experiment_id": "pre", "model": "knn", "seed": 1, "dataset_id": "t",
                "metrics": ["accuracy"], "params": {},
                "preprocess": {"categorical": ["kind"], "scale": "standardize"},
            }
        )  # fmt: skip
        result = run_nested(cfg, [{"k": 1}, {"k": 5}], table, "label", outer_folds=4, inner_folds=2)
        self.assertEqual(len(result.outer), 4)

    def test_selection_overstates_the_score_when_there_is_nothing_to_find(self):
        # random labels: any candidate's true accuracy is 0.5, yet picking the best of many
        # looks better than that on the inner folds and falls back on the outer folds
        rng = random.Random(8)
        rows = tuple(
            (f"{rng.uniform(0, 1):.3f}", f"{rng.uniform(0, 1):.3f}", rng.choice("pq"))
            for _ in range(80)
        )
        table = Table(("a", "b", "label"), rows)
        candidates = [{"k": k, "weights": w} for k in (1, 2, 3, 4, 5, 6, 7, 9, 11, 13)
                      for w in ("uniform", "distance")]  # fmt: skip
        gaps = []
        for seed in range(1, 7):
            result = run_nested(
                config(seed=seed), candidates, table, "label", outer_folds=4, inner_folds=3
            )
            gaps.append(result.optimism)
        self.assertGreater(sum(gaps) / len(gaps), 0.02)
        self.assertGreater(sum(1 for g in gaps if g > 0), 4)


class ValidationTests(unittest.TestCase):
    def run_with(self, **kwargs):
        args = {"outer_folds": 3, "inner_folds": 2}
        args.update(kwargs)
        cfg = args.pop("cfg", config())
        candidates = args.pop("candidates", KS)
        return run_nested(cfg, candidates, args.pop("table", noisy_table(n=30)), "label", **args)

    def test_fold_counts(self):
        for outer, inner in [(1, 2), (3, 1), (51, 2), (3, 51), (True, 2), (3, 2.5)]:
            with self.subTest(outer=outer, inner=inner), self.assertRaises(SweepError):
                self.run_with(outer_folds=outer, inner_folds=inner)

    def test_the_inner_folds_must_fit_in_the_outer_training_set(self):
        with self.assertRaises(SweepError):
            self.run_with(table=noisy_table(n=8), outer_folds=4, inner_folds=7)

    def test_the_config_must_leave_the_folds_to_nested(self):
        with self.assertRaises(SweepError):
            self.run_with(cfg=config(cv_folds=3))

    def test_candidates(self):
        with self.assertRaises(SweepError):
            self.run_with(candidates=[])
        with self.assertRaises(SweepError):
            self.run_with(candidates=[{"k": 1}] * (MAX_CANDIDATES + 1))
        with self.assertRaises(SweepError):
            self.run_with(
                candidates=[{"k": i + 1} for i in range(500)], outer_folds=50, inner_folds=50
            )

    def test_the_selection_metric_must_be_reported(self):
        with self.assertRaises(SweepError):
            self.run_with(rank_by="f1")
        with self.assertRaises(SweepError):
            self.run_with(cfg=config(metrics=("accuracy", "f1")), rank_by="accuracy_std")

    def test_metrics_must_match_the_model_kind(self):
        with self.assertRaises(ModelBenchError) as raised:
            self.run_with(cfg=config(metrics=("mae",)))
        self.assertIn("mae", str(raised.exception))

    def test_stratify_is_rejected_for_regression(self):
        table = Table(("x", "y"), tuple((str(i), str(i * 2.0)) for i in range(30)))
        with self.assertRaises(SweepError):
            run_nested(
                config(model="knn_regressor", metrics=("mae",), stratify=True),
                [{"k": 1}], table, "y", outer_folds=3, inner_folds=2,
            )  # fmt: skip

    def test_a_split_that_cannot_be_scored_is_reported_before_any_fit(self):
        rows = tuple((str(i), "a" if i < 28 else "b") for i in range(30))
        table = Table(("x", "label"), rows)
        with mock.patch.object(KNNClassifier, "fit") as fit, self.assertRaises(ModelBenchError):
            run_nested(
                config(metrics=("roc_auc",)), [{"k": 1}], table, "label",
                outer_folds=3, inner_folds=2,
            )  # fmt: skip
        fit.assert_not_called()


class CandidateTests(unittest.TestCase):
    def test_a_grid_expands_in_a_fixed_order_on_top_of_the_config_params(self):
        cfg = config(params={"metric": "euclidean"})
        found = candidate_params(cfg, {"weights": ["uniform", "distance"], "k": [1, 3]}, None, None)
        self.assertEqual(
            found,
            [
                {"k": 1, "metric": "euclidean", "weights": "uniform"},
                {"k": 1, "metric": "euclidean", "weights": "distance"},
                {"k": 3, "metric": "euclidean", "weights": "uniform"},
                {"k": 3, "metric": "euclidean", "weights": "distance"},
            ],
        )

    def test_a_space_draws_what_search_draws_without_repeats(self):
        space = {"k": {"int": [1, 3]}, "weights": {"choice": ["uniform", "distance"]}}
        found = candidate_params(config(seed=6), None, space, 12)
        checked = check_space(space)
        drawn = []
        for trial in range(12):
            sampled = sample_params(checked, 6, trial)
            if sampled not in drawn:
                drawn.append(sampled)
        self.assertEqual(found, drawn)
        self.assertLessEqual(len(found), 6)  # only six distinct sets exist
        self.assertEqual(len({json.dumps(p, sort_keys=True) for p in found}), len(found))

    def test_config_params_are_kept_in_drawn_candidates(self):
        found = candidate_params(
            config(params={"metric": "manhattan"}), None, {"k": {"int": [1, 9]}}, 3
        )
        self.assertTrue(all(p["metric"] == "manhattan" for p in found))

    def test_errors(self):
        grid, space = {"k": [1]}, {"k": {"int": [1, 3]}}
        for args in [
            (config(), None, None, None),
            (config(), grid, space, 3),
            (config(params={"k": 3}), None, space, 3),
            (config(), None, space, 0),
            (config(), None, space, True),
        ]:
            with self.subTest(args=args), self.assertRaises(ModelBenchError):
                candidate_params(*args)

    def test_too_many_candidates(self):
        with self.assertRaises(ModelBenchError):
            candidate_params(
                config(), {"k": list(range(1, 40)), "p": list(range(1, 40))}, None, None
            )


class ResultMathTests(unittest.TestCase):
    def result(self, larger):
        folds = (
            OuterFold(0, 8, 4, 1, (0.5, 0.9), {"m": 0.6}),
            OuterFold(1, 8, 4, 1, (0.7, 0.8), {"m": 0.7}),
            OuterFold(2, 8, 4, 0, (0.75, None), {"m": 0.8}),
        )
        return NestedResult("e", "m", larger, ({"a": 1}, {"a": 2}), folds)

    def test_higher_is_better(self):
        result = self.result(True)
        self.assertAlmostEqual(result.estimate[0], 0.7)
        self.assertAlmostEqual(result.estimate[1], (0.02 / 3) ** 0.5)
        self.assertAlmostEqual(result.selection_score, (0.9 + 0.8 + 0.75) / 3)
        self.assertAlmostEqual(result.optimism, (0.9 + 0.8 + 0.75) / 3 - 0.7)
        self.assertEqual(result.choice_counts(), [1, 2])

    def test_lower_is_better_flips_the_sign_of_the_gap(self):
        result = self.result(False)
        self.assertAlmostEqual(result.optimism, -((0.9 + 0.8 + 0.75) / 3 - 0.7))


class FileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "nested.json"

    def document(self, **changes):
        document = {
            "config": {"experiment_id": "n", "model": "knn", "seed": 1, "dataset_id": "d",
                       "metrics": ["accuracy"], "params": {}},
            "grid": {"k": [1, 3]},
            "outer_folds": 4,
            "inner_folds": 3,
        }  # fmt: skip
        document.update(changes)
        return document

    def load(self, document):
        self.path.write_text(json.dumps(document))
        return load_nested(self.path)

    def test_a_grid_file_and_a_space_file(self):
        spec = self.load(self.document())
        self.assertEqual((spec.grid, spec.space, spec.trials), ({"k": [1, 3]}, None, None))
        self.assertEqual((spec.outer_folds, spec.inner_folds), (4, 3))
        document = self.document(space={"k": {"int": [1, 5]}}, trials=6)
        del document["grid"]
        spec = self.load(document)
        self.assertEqual((spec.grid, spec.trials), (None, 6))
        self.assertEqual(spec.space, {"k": {"int": [1, 5]}})

    def test_bad_files(self):
        grid_gone = self.document()
        del grid_gone["grid"]
        no_config = self.document()
        del no_config["config"]
        cases = {
            "not an object": [],
            "unknown key": self.document(extra=1),
            "no config": no_config,
            "neither grid nor space": grid_gone,
            "both": self.document(space={"k": {"int": [1, 3]}}, trials=3),
            "trials with grid": self.document(trials=3),
            "space without trials": {**grid_gone, "space": {"k": {"int": [1, 3]}}},
            "empty grid": self.document(grid={}),
            "empty list": self.document(grid={"k": []}),
            "bad space": {**grid_gone, "space": {"k": {"bogus": 1}}, "trials": 3},
            "bad trials": {**grid_gone, "space": {"k": {"int": [1, 3]}}, "trials": 0},
            "outer too small": self.document(outer_folds=1),
            "inner not whole": self.document(inner_folds=2.5),
            "missing outer": {k: v for k, v in self.document().items() if k != "outer_folds"},
            "cv_folds set": self.document(
                config={"experiment_id": "n", "model": "knn", "seed": 1, "dataset_id": "d",
                        "metrics": ["accuracy"], "params": {}, "cv_folds": 3}
            ),
        }  # fmt: skip
        for name, document in cases.items():
            with self.subTest(name=name), self.assertRaises(ModelBenchError):
                self.load(document)

    def test_an_unreadable_file(self):
        with self.assertRaises(SweepError):
            load_nested(self.path)


if __name__ == "__main__":
    unittest.main()
