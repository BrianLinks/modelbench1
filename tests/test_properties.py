"""Seeded property tests: invariants that must hold for many generated inputs.

Each test draws its cases from a generator derived from a fixed seed, so a failure is exactly
repeatable; the failing case's index and seed are in the message. No third-party library is used.
"""

import itertools
import math
import random
import unittest

from modelbench import metrics as m
from modelbench.config import ExperimentConfig, expand_config, expand_grid
from modelbench.datasets import Table, kfold_indices, split_indices
from modelbench.models import REGISTRY, REGRESSORS, create_model, create_regressor
from modelbench.reproducibility import check_reproducibility
from modelbench.runner import run_experiment
from modelbench.seeding import derive_seed
from modelbench.store import Store

CASES = 120
ENV = {"python": "3.12.1", "lib:numpy": None}


def cases(name, count=CASES):
    """Yield `(index, rng)` for `count` independent, reproducible generators."""
    for index in range(count):
        yield index, random.Random(derive_seed(20261026, name, index))


def labels_for(rng, n, k):
    """n labels over k classes, every class present at least once."""
    names = [f"c{i}" for i in range(k)]
    out = [names[i % k] for i in range(k)] + [rng.choice(names) for _ in range(n - k)]
    rng.shuffle(out)
    return out


def feature_rows(rng, n, d):
    return [[round(rng.uniform(-5, 5), 3) for _ in range(d)] for _ in range(n)]


class MetricProperties(unittest.TestCase):
    def pair(self, rng):
        k = rng.randint(2, 4)
        n = rng.randint(max(k, 4), 40)
        truth = labels_for(rng, n, k)
        names = sorted(set(truth))
        pred = [rng.choice(names) if rng.random() < 0.5 else t for t in truth]
        return truth, pred

    def test_confusion_matrix_accounts_for_every_row(self):
        for i, rng in cases("confusion"):
            truth, pred = self.pair(rng)
            cm = m.confusion_matrix(truth, pred)
            total = sum(sum(row) for row in cm.matrix)
            self.assertEqual(total, len(truth), i)
            for label, row in zip(cm.labels, cm.matrix, strict=True):
                self.assertEqual(sum(row), truth.count(label), i)
            trace = sum(cm.matrix[j][j] for j in range(len(cm.labels)))
            self.assertAlmostEqual(m.accuracy(truth, pred), trace / len(truth), msg=str(i))

    def test_scores_stay_in_their_ranges(self):
        for i, rng in cases("ranges"):
            truth, pred = self.pair(rng)
            for name, value, lo, hi in (
                ("accuracy", m.accuracy(truth, pred), 0, 1),
                ("balanced", m.balanced_accuracy(truth, pred), 0, 1),
                ("mcc", m.matthews_corrcoef(truth, pred), -1, 1),
                ("f1", m.precision_recall_f1(truth, pred, "macro").f1, 0, 1),
                ("f1w", m.precision_recall_f1(truth, pred, "weighted").f1, 0, 1),
            ):
                self.assertTrue(lo - 1e-12 <= value <= hi + 1e-12, f"{name} {value} case {i}")

    def test_micro_f1_equals_accuracy_for_single_label_data(self):
        for i, rng in cases("micro"):
            truth, pred = self.pair(rng)
            self.assertAlmostEqual(
                m.precision_recall_f1(truth, pred, "micro").f1, m.accuracy(truth, pred), msg=str(i)
            )

    def test_perfect_predictions_score_one(self):
        for i, rng in cases("perfect"):
            truth, _ = self.pair(rng)
            self.assertEqual(m.accuracy(truth, truth), 1.0, i)
            self.assertAlmostEqual(m.balanced_accuracy(truth, truth), 1.0, msg=str(i))
            self.assertAlmostEqual(m.matthews_corrcoef(truth, truth), 1.0, msg=str(i))
            self.assertAlmostEqual(m.precision_recall_f1(truth, truth, "macro").f1, 1.0)

    def test_row_order_does_not_matter(self):
        for i, rng in cases("permute"):
            truth, pred = self.pair(rng)
            order = list(range(len(truth)))
            rng.shuffle(order)
            t2, p2 = [truth[j] for j in order], [pred[j] for j in order]
            for fn in (m.accuracy, m.balanced_accuracy, m.matthews_corrcoef):
                self.assertAlmostEqual(fn(truth, pred), fn(t2, p2), msg=f"{fn.__name__} {i}")
            self.assertAlmostEqual(
                m.precision_recall_f1(truth, pred, "macro").f1,
                m.precision_recall_f1(t2, p2, "macro").f1,
            )

    def test_renaming_the_classes_does_not_matter(self):
        for i, rng in cases("rename"):
            truth, pred = self.pair(rng)
            names = sorted(set(truth))
            mapping = {name: f"z{len(names) - j}" for j, name in enumerate(names)}
            t2, p2 = [mapping[x] for x in truth], [mapping[x] for x in pred]
            for fn in (m.accuracy, m.balanced_accuracy, m.matthews_corrcoef):
                self.assertAlmostEqual(fn(truth, pred), fn(t2, p2), msg=f"{fn.__name__} {i}")

    def test_a_constant_predictor_has_known_scores(self):
        for i, rng in cases("constant"):
            truth, _ = self.pair(rng)
            names = sorted(set(truth))
            guess = rng.choice(names)
            pred = [guess] * len(truth)
            n, k, hits = len(truth), len(names), truth.count(guess)
            self.assertAlmostEqual(m.balanced_accuracy(truth, pred), 1 / k, msg=str(i))
            self.assertEqual(m.matthews_corrcoef(truth, pred), 0.0, i)
            macro = m.precision_recall_f1(truth, pred, "macro")
            self.assertAlmostEqual(macro.recall, 1 / k, msg=str(i))
            self.assertAlmostEqual(macro.precision, hits / n / k, msg=str(i))
            # only the guessed class has a non-zero F1: 2 * precision * 1 / (precision + 1)
            p = hits / n
            self.assertAlmostEqual(macro.f1, 2 * p / (p + 1) / k, msg=str(i))

    def test_regression_metric_identities(self):
        for i, rng in cases("regression"):
            n = rng.randint(3, 30)
            truth = [round(rng.uniform(-10, 10), 3) for _ in range(n)]
            pred = [round(t + rng.gauss(0, 2), 3) for t in truth]
            self.assertAlmostEqual(m.rmse(truth, pred) ** 2, m.mse(truth, pred), msg=str(i))
            self.assertLessEqual(m.mae(truth, pred), m.rmse(truth, pred) + 1e-12, i)
            self.assertLessEqual(m.r2(truth, pred), 1.0 + 1e-12, i)
            self.assertEqual(m.mse(truth, truth), 0.0)
            shift = rng.uniform(0.5, 3)
            shifted = [t + shift for t in truth]
            self.assertAlmostEqual(m.mse(truth, shifted), shift**2, msg=str(i))
            self.assertAlmostEqual(m.mae(truth, shifted), shift, msg=str(i))


class SplitProperties(unittest.TestCase):
    def test_a_split_partitions_the_rows(self):
        for i, rng in cases("split"):
            n = rng.randint(4, 80)
            fraction = rng.choice([0.1, 0.2, 0.25, 0.5, 0.7])
            seed = rng.randint(0, 10**6)
            strat = labels_for(rng, n, rng.randint(2, 3)) if rng.random() < 0.5 else None
            try:
                train, test = split_indices(n, fraction, seed, strat)
            except ValueError:  # a split can be impossible (e.g. a class of one row)
                continue
            self.assertEqual(sorted(train + test), list(range(n)), i)
            self.assertEqual(train, sorted(train))
            self.assertEqual(test, sorted(test))
            self.assertEqual((train, test), split_indices(n, fraction, seed, strat), i)

    def test_folds_partition_the_rows_and_stay_balanced(self):
        for i, rng in cases("kfold"):
            k = rng.randint(2, 3)
            n = rng.randint(12, 80)
            folds = rng.randint(2, 6)
            seed = rng.randint(0, 10**6)
            classes = labels_for(rng, n, k) if rng.random() < 0.5 else None
            result = kfold_indices(n, folds, seed, classes)
            self.assertEqual(result, kfold_indices(n, folds, seed, classes), i)
            tests = [t for _, t in result]
            self.assertEqual(sorted(itertools.chain(*tests)), list(range(n)), i)
            sizes = [len(t) for t in tests]
            self.assertLessEqual(max(sizes) - min(sizes), 1, i)
            for train, test in result:
                self.assertEqual(sorted(train + test), list(range(n)), i)
                self.assertFalse(set(train) & set(test), i)
            if classes is not None:
                for label in set(classes):
                    counts = [sum(classes[j] == label for j in t) for t in tests]
                    self.assertLessEqual(max(counts) - min(counts), 1, f"{label} case {i}")


class ConfigProperties(unittest.TestCase):
    def random_config(self, rng, **overrides):
        fields = {
            "experiment_id": rng.choice(["a", "exp-1", "x.y_z"]),
            "model": "knn",
            "seed": rng.randint(0, 10**9),
            "dataset_id": "d",
            "metrics": tuple(rng.sample(["accuracy", "f1", "recall", "mcc"], rng.randint(1, 4))),
            "params": {"k": rng.randint(1, 9), "weights": rng.choice(["uniform", "distance"])},
            "test_fraction": rng.choice([0.2, 0.25, 0.3]),
            "stratify": rng.random() < 0.5,
            "cv_folds": rng.choice([None, 2, 5, 10]),
        }
        fields.update(overrides)
        return ExperimentConfig(**fields)

    def test_round_trip_keeps_the_config_and_its_hash(self):
        for i, rng in cases("config"):
            config = self.random_config(rng)
            again = ExperimentConfig.from_dict(config.to_dict())
            self.assertEqual(again, config, i)
            self.assertEqual(again.config_hash(), config.config_hash(), i)

    def test_the_hash_ignores_the_label_but_not_the_settings(self):
        for i, rng in cases("hash"):
            config = self.random_config(rng)
            relabelled = ExperimentConfig.from_dict({**config.to_dict(), "experiment_id": "other"})
            self.assertEqual(relabelled.config_hash(), config.config_hash(), i)
            reseeded = ExperimentConfig.from_dict({**config.to_dict(), "seed": config.seed + 1})
            self.assertNotEqual(reseeded.config_hash(), config.config_hash(), i)
            params = {**config.params, "k": config.params["k"] + 1}
            changed = ExperimentConfig.from_dict({**config.to_dict(), "params": params})
            self.assertNotEqual(changed.config_hash(), config.config_hash(), i)

    def test_a_grid_has_one_distinct_entry_per_combination(self):
        for i, rng in cases("grid"):
            grid = {
                f"p{j}": rng.sample(range(100), rng.randint(1, 4)) for j in range(rng.randint(1, 3))
            }
            expanded = expand_grid({"fixed": 1}, grid)
            self.assertEqual(len(expanded), math.prod(len(v) for v in grid.values()), i)
            self.assertEqual(len({tuple(sorted(e.items())) for e in expanded}), len(expanded), i)
            self.assertTrue(all(e["fixed"] == 1 for e in expanded))
            config = ConfigProperties().random_config(rng, params={"fixed": 1})
            configs = expand_config(config, grid)
            self.assertEqual([c.params for c in configs], expanded, i)
            self.assertEqual(expanded, expand_grid({"fixed": 1}, grid), i)


class ModelProperties(unittest.TestCase):
    def data(self, rng):
        k = rng.randint(2, 3)
        n = rng.randint(14, 40)
        d = rng.randint(1, 3)
        X = feature_rows(rng, n, d)
        return X, labels_for(rng, n, k), feature_rows(rng, 6, d)

    def build(self, name, rng):
        cls = REGISTRY[name]
        model = create_model(name)
        if "seed" in model.get_params():
            model = cls(seed=rng.randint(0, 1000))
        return model

    def test_every_classifier_gives_valid_probabilities_and_labels(self):
        for name in sorted(REGISTRY):
            for i, rng in cases(f"clf-{name}", 20):
                X, y, queries = self.data(rng)
                model = self.build(name, rng)
                model.fit(X, y)
                where = f"{name} case {i}"
                for row in model.predict_proba(queries):
                    self.assertEqual(len(row), len(model.classes_), where)
                    self.assertAlmostEqual(sum(row), 1.0, msg=where)
                    self.assertTrue(all(0 <= p <= 1 for p in row), where)
                predictions = model.predict(queries)
                self.assertEqual(len(predictions), len(queries), where)
                self.assertTrue(set(predictions) <= set(y), where)

    def test_refitting_with_the_same_parameters_gives_the_same_predictions(self):
        for name in sorted(REGISTRY):
            for i, rng in cases(f"refit-{name}", 10):
                X, y, queries = self.data(rng)
                first = self.build(name, rng)
                first.fit(X, y)
                second = REGISTRY[name](**first.get_params())
                second.fit(X, y)
                self.assertEqual(first.predict(queries), second.predict(queries), f"{name} {i}")
                first.fit(X, y)  # fitting again on the same data changes nothing
                self.assertEqual(first.predict(queries), second.predict(queries), f"{name} {i}")

    def test_every_regressor_predicts_finite_numbers(self):
        for name in sorted(REGRESSORS):
            for i, rng in cases(f"reg-{name}", 20):
                n, d = rng.randint(14, 40), rng.randint(1, 3)
                X = feature_rows(rng, n, d)
                y = [round(sum(r) + rng.gauss(0, 1), 3) for r in X]
                model = create_regressor(name)
                model.fit(X, y)
                out = model.predict(feature_rows(rng, 5, d))
                self.assertEqual(len(out), 5)
                self.assertTrue(all(math.isfinite(v) for v in out), f"{name} {i}")

    def test_averaging_regressors_never_leave_the_range_of_the_targets(self):
        for name in ("mean", "knn_regressor", "decision_tree_regressor"):
            for i, rng in cases(f"range-{name}", 20):
                n, d = rng.randint(14, 40), rng.randint(1, 3)
                X = feature_rows(rng, n, d)
                y = [round(rng.uniform(-20, 20), 3) for _ in X]
                model = create_regressor(name)
                model.fit(X, y)
                for value in model.predict(feature_rows(rng, 8, d) + [[1e6] * d]):
                    self.assertTrue(min(y) - 1e-9 <= value <= max(y) + 1e-9, f"{name} {i}")

    def test_linear_regression_recovers_a_noiseless_line(self):
        for i, rng in cases("linear", 40):
            d = rng.randint(1, 3)
            weights = [round(rng.uniform(-3, 3), 2) for _ in range(d)]
            intercept = round(rng.uniform(-5, 5), 2)
            X = feature_rows(rng, rng.randint(d + 5, 40), d)
            y = [intercept + sum(w * v for w, v in zip(weights, r, strict=True)) for r in X]
            model = create_regressor("linear_regression")
            model.fit(X, y)
            pred = model.predict(X)
            self.assertLess(max(abs(a - b) for a, b in zip(pred, y, strict=True)), 1e-6, i)


class RunProperties(unittest.TestCase):
    def table(self, rng):
        k = rng.randint(2, 3)
        n = rng.randint(30, 60)
        X = feature_rows(rng, n, 2)
        # classes separated by a shift so scores are not all chance, but overlapping a bit
        y = labels_for(rng, n, k)
        rows = tuple(
            (str(r[0] + 4 * int(label[1:])), str(r[1]), label)
            for r, label in zip(X, y, strict=True)
        )
        return Table(("a", "b", "label"), rows)

    def test_every_classifier_run_reproduces(self):
        for name in sorted(REGISTRY):
            for i, rng in cases(f"repro-{name}", 3):
                table = self.table(rng)
                config = ExperimentConfig(
                    experiment_id="p", model=name, seed=rng.randint(0, 10**6), dataset_id="d",
                    metrics=("accuracy", "f1"), params={}, stratify=True,
                    cv_folds=rng.choice([None, 3]),
                )  # fmt: skip
                with Store(":memory:") as store:
                    result = run_experiment(store, config, table, "label", environment=ENV)
                    report = check_reproducibility(
                        store, result.run_id, table, "label", environment=ENV
                    )
                    self.assertTrue(report.reproduced, f"{name} case {i}: {report.detail}")
                    self.assertTrue(0.0 <= result.metrics["accuracy"] <= 1.0)

    def test_a_cross_validated_run_tests_every_row_once(self):
        for i, rng in cases("cv-rows", 15):
            table = self.table(rng)
            folds = rng.randint(2, 5)
            config = ExperimentConfig(
                experiment_id="p", model="majority", seed=rng.randint(0, 10**6), dataset_id="d",
                metrics=("accuracy",), params={}, stratify=rng.random() < 0.5, cv_folds=folds,
            )  # fmt: skip
            with Store(":memory:") as store:
                result = run_experiment(store, config, table, "label", environment=ENV)
            self.assertEqual(result.n_test, len(table), i)
            self.assertLess(result.n_train, len(table), i)


if __name__ == "__main__":
    unittest.main()
