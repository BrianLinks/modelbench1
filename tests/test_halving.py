import json
import random
import tempfile
import unittest
from pathlib import Path

from modelbench.config import ExperimentConfig
from modelbench.datasets import Table
from modelbench.errors import SweepError
from modelbench.halving import (
    MAX_ETA,
    HalvingSpec,
    budgets,
    check_spec,
    load_halving,
    run_halving,
    survivors_after,
)
from modelbench.store import Store
from modelbench.sweep import check_space, run_search, sample_params

ENV = {"python": "3.12.1", "lib:numpy": None}
SPACE = {
    "max_depth": {"int": [1, 8]},
    "min_samples_leaf": {"int": [1, 6]},
    "max_features": {"choice": ["sqrt", "log2", 1.0]},
}


def table(n=80, seed=3):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b = rng.uniform(-3, 3), rng.uniform(-3, 3)
        rows.append((f"{a:.3f}", f"{b:.3f}", "p" if a + b + rng.uniform(-1, 1) > 0 else "q"))
    return Table(("a", "b", "label"), tuple(rows))


def config(**overrides):
    fields = {
        "experiment_id": "halve",
        "model": "random_forest",
        "seed": 9,
        "dataset_id": "rows",
        "metrics": ("accuracy", "f1"),
        "params": {},
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


def distinct_draws(seed, trials, space=SPACE):
    """The parameter sets random search draws, in order, without repeats."""
    checked = check_space(space)
    drawn = []
    for trial in range(trials):
        sampled = sample_params(checked, seed, trial)
        if sampled not in drawn:
            drawn.append(sampled)
    return drawn


SPEC = HalvingSpec("n_estimators", 2, 18, 3)


class ScheduleTests(unittest.TestCase):
    def test_budgets_by_hand(self):
        cases = [
            ((1, 27, 3), [1, 3, 9, 27]),
            ((1, 20, 3), [1, 3, 9, 20]),  # the last round is always the full budget
            ((2, 18, 3), [2, 6, 18]),
            ((5, 5, 3), [5]),
            ((2, 100, 10), [2, 20, 100]),
            ((1, 2, 2), [1, 2]),
            ((3, 4, 2), [3, 4]),
            ((1, 8, 2), [1, 2, 4, 8]),
        ]
        for (low, high, eta), want in cases:
            with self.subTest(low=low, high=high, eta=eta):
                self.assertEqual(budgets(HalvingSpec("n", low, high, eta)), want)

    def test_survivors_are_a_share_of_one_over_eta_rounded_up_and_never_zero(self):
        for n, eta, want in [(9, 3, 3), (10, 3, 4), (1, 3, 1), (2, 10, 1), (7, 2, 4), (3, 3, 1)]:
            with self.subTest(n=n, eta=eta):
                self.assertEqual(survivors_after(n, eta), want)

    def test_bad_specs(self):
        for spec in (
            HalvingSpec("", 1, 9),
            HalvingSpec(3, 1, 9),
            HalvingSpec("n", 0, 9),
            HalvingSpec("n", 1, 0),
            HalvingSpec("n", 10, 9),
            HalvingSpec("n", 1.5, 9),
            HalvingSpec("n", True, 9),
            HalvingSpec("n", 1, 9, 1),
            HalvingSpec("n", 1, 9, MAX_ETA + 1),
            HalvingSpec("n", 1, 9, 2.0),
            HalvingSpec("n", 1, 1_000_001),
        ):
            with self.subTest(spec=spec), self.assertRaises(SweepError):
                check_spec(spec)

    def test_a_good_spec_passes_through(self):
        self.assertIs(check_spec(SPEC), SPEC)
        check_spec(HalvingSpec("n", 1, 1, MAX_ETA))


class RunTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)
        self.table = table()

    def halve(self, candidates=9, spec=SPEC, space=SPACE, **kwargs):
        return run_halving(
            self.store, config(**kwargs.pop("cfg", {})), space, candidates, spec, self.table,
            "label", environment=ENV, **kwargs,
        )  # fmt: skip

    def test_round_sizes_and_budgets_follow_the_schedule(self):
        result = self.halve()
        pool = len(distinct_draws(9, 9))
        self.assertEqual(pool, 8)  # one of the nine draws repeats an earlier one
        self.assertEqual([r.resource_value for r in result.rounds], [2, 6, 18])
        self.assertEqual([len(r.result.entries) for r in result.rounds], [8, 3, 1])
        self.assertEqual(result.n_runs, 12)
        self.assertEqual(result.resource, "n_estimators")
        self.assertEqual(result.rank_metric, "accuracy")
        for finished in result.rounds:
            for entry in finished.result.entries:
                self.assertEqual(entry.params["n_estimators"], finished.resource_value)
                run = self.store.get_run(entry.run_id)
                self.assertEqual(run.params["n_estimators"], finished.resource_value)
                self.assertEqual(run.experiment_id, "halve")

    def test_the_first_round_draws_exactly_what_random_search_draws(self):
        result = self.halve(candidates=9)
        drawn = distinct_draws(9, 9)
        first = [{k: v for k, v in e.params.items() if k != "n_estimators"}
                 for e in result.rounds[0].result.entries]  # fmt: skip
        self.assertEqual(first, drawn)

    def test_survivors_are_the_best_ranked_of_each_round_and_carry_their_params(self):
        result = self.halve()
        for before, after in zip(result.rounds, result.rounds[1:], strict=False):
            ranked = [r.run_id for r in before.result.ranking.ranked]
            self.assertEqual(list(before.kept), ranked[: survivors_after(len(ranked), 3)])
            by_run = {e.run_id: e.params for e in before.result.entries}
            want = [
                {k: v for k, v in by_run[run_id].items() if k != "n_estimators"}
                for run_id in before.kept
            ]
            got = [
                {k: v for k, v in e.params.items() if k != "n_estimators"}
                for e in after.result.entries
            ]
            self.assertEqual(got, want)
        self.assertEqual(result.rounds[-1].kept, ())

    def test_the_best_is_the_top_of_the_last_round(self):
        result = self.halve()
        last = result.rounds[-1].result
        self.assertEqual(result.best().run_id, last.best().run_id)

    def test_a_rerun_reuses_every_run_and_changes_nothing(self):
        first = self.halve()
        before = len(self.store.list_runs(limit=100))
        again = self.halve()
        self.assertEqual(len(self.store.list_runs(limit=100)), before)
        for a, b in zip(first.rounds, again.rounds, strict=True):
            self.assertEqual(a.kept, b.kept)
            self.assertTrue(all(e.reused for e in b.result.entries))
            self.assertEqual(
                [e.run_id for e in a.result.entries], [e.run_id for e in b.result.entries]
            )

    def test_reuse_can_be_turned_off(self):
        self.halve()
        before = len(self.store.list_runs(limit=100))
        self.halve(reuse_existing=False)
        self.assertEqual(len(self.store.list_runs(limit=100)), 2 * before)

    def test_a_larger_pool_extends_the_first_round_without_changing_earlier_draws(self):
        small = self.halve(candidates=4)
        large = self.halve(candidates=9)
        head = [e.params for e in large.rounds[0].result.entries][
            : len(small.rounds[0].result.entries)
        ]
        self.assertEqual([e.params for e in small.rounds[0].result.entries], head)

    def test_a_single_budget_is_just_a_search(self):
        spec = HalvingSpec("n_estimators", 5, 5)
        result = self.halve(candidates=6, spec=spec)
        self.assertEqual(len(result.rounds), 1)
        self.assertEqual(result.rounds[0].kept, ())
        search = run_search(
            self.store, config(experiment_id="plain", params={"n_estimators": 5}), SPACE, 6,
            self.table, "label", environment=ENV,
        )  # fmt: skip
        self.assertEqual(
            [e.metrics for e in result.rounds[0].result.entries],
            [e.metrics for e in search.entries],
        )

    def test_ranking_uses_the_chosen_metric(self):
        result = self.halve(rank_by="f1")
        self.assertEqual(result.rank_metric, "f1")
        ranked = result.rounds[0].result.ranking.ranked
        scores = [
            e.metrics["f1"]
            for r in ranked
            for e in result.rounds[0].result.entries
            if e.run_id == r.run_id
        ]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_lower_is_better_metrics_keep_the_smallest(self):
        result = self.halve(cfg={"metrics": ("log_loss",)})
        first = result.rounds[0].result
        losses = {e.run_id: e.metrics["log_loss"] for e in first.entries}
        kept = result.rounds[0].kept
        self.assertTrue(
            all(min(losses[r] for r in losses if r not in kept) >= losses[k] - 1e-12 for k in kept)
        )

    def test_candidates_that_fail_are_dropped_and_reported(self):
        space = {"max_features": {"choice": [1.0, "bogus"]}, "max_depth": {"int": [2, 4]}}
        result = self.halve(candidates=8, space=space)
        first = result.rounds[0].result
        self.assertTrue(first.failed)
        self.assertTrue(all("bogus" in e.params.get("max_features", "") for e in first.failed))
        ok = [e for e in first.entries if e.error is None]
        self.assertTrue(ok)
        for finished in result.rounds[1:]:
            for entry in finished.result.entries:
                self.assertNotEqual(entry.params["max_features"], "bogus")

    def test_when_every_candidate_fails_there_is_no_best_and_no_second_round(self):
        space = {"max_features": {"choice": ["bogus"]}}
        result = self.halve(candidates=3, space=space)
        self.assertEqual(len(result.rounds), 1)
        self.assertIsNone(result.best())
        self.assertEqual(result.rounds[0].kept, ())

    def test_best_falls_back_to_the_last_round_that_produced_a_result(self):
        # a round with a result, then a round where nothing can be built
        result = self.halve(spec=HalvingSpec("n_estimators", 2, 18, 3), candidates=3)
        self.assertIsNotNone(result.best())

    def test_parameter_clashes_are_rejected_before_anything_runs(self):
        cases = [
            (SPACE | {"n_estimators": {"int": [1, 3]}}, {}),  # the budget is in the space
            (SPACE, {"params": {"n_estimators": 5}}),  # the budget is fixed in the config
            (SPACE, {"params": {"max_depth": 3}}),  # a space parameter is fixed in the config
        ]
        for space, cfg in cases:
            with self.subTest(space=space, cfg=cfg), self.assertRaises(SweepError):
                self.halve(space=space, cfg=cfg)
        self.assertEqual(self.store.list_runs(limit=5), [])

    def test_bad_candidate_counts_and_metrics_are_rejected(self):
        for candidates in (0, 1001, 2.5, True):
            with self.subTest(candidates=candidates), self.assertRaises(SweepError):
                self.halve(candidates=candidates)
        with self.assertRaises(SweepError):
            self.halve(rank_by="nope")
        self.assertEqual(self.store.list_runs(limit=5), [])

    def test_fixed_parameters_stay_fixed_in_every_round(self):
        space = {"max_depth": {"int": [1, 8]}, "min_samples_leaf": {"int": [1, 6]}}
        result = self.halve(space=space, cfg={"params": {"max_features": 1.0}})
        for finished in result.rounds:
            for entry in finished.result.entries:
                self.assertEqual(entry.params["max_features"], 1.0)

    def test_every_step_is_an_ordinary_reproducible_run(self):
        from modelbench.reproducibility import check_reproducibility

        result = self.halve(candidates=3)
        run_id = result.best().run_id
        self.assertEqual(
            check_reproducibility(self.store, run_id, self.table, "label").verdict, "reproduced"
        )


class FileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "halving.json"

    def document(self, **changes):
        document = {
            "config": json.loads(
                json.dumps({"experiment_id": "h", "model": "random_forest", "seed": 1,
                            "dataset_id": "d", "metrics": ["accuracy"], "params": {}})
            ),
            "space": SPACE,
            "candidates": 9,
            "halving": {"resource": "n_estimators", "min_resource": 2, "max_resource": 18},
        }  # fmt: skip
        document.update(changes)
        return document

    def load(self, document):
        self.path.write_text(json.dumps(document))
        return load_halving(self.path)

    def test_a_good_file(self):
        loaded, space, candidates, spec = self.load(self.document())
        self.assertEqual(loaded.model, "random_forest")
        self.assertEqual(space, SPACE)
        self.assertEqual(candidates, 9)
        self.assertEqual(spec, HalvingSpec("n_estimators", 2, 18, 3))
        *_, spec = self.load(
            self.document(halving={"resource": "n", "min_resource": 1, "max_resource": 8, "eta": 2})
        )
        self.assertEqual(spec.eta, 2)

    def test_bad_files(self):
        bad = {
            "not an object": [],
            "unknown key": self.document(extra=1),
            "missing key": {k: v for k, v in self.document().items() if k != "space"},
            "halving not an object": self.document(halving=[]),
            "missing resource": self.document(halving={"min_resource": 1, "max_resource": 3}),
            "unknown halving key": self.document(
                halving={"resource": "n", "min_resource": 1, "max_resource": 3, "rate": 2}
            ),
            "bad eta": self.document(
                halving={"resource": "n", "min_resource": 1, "max_resource": 3, "eta": 1}
            ),
            "min above max": self.document(
                halving={"resource": "n", "min_resource": 5, "max_resource": 3}
            ),
            "zero candidates": self.document(candidates=0),
            "float candidates": self.document(candidates=2.5),
            "too many candidates": self.document(candidates=1001),
            "empty space": self.document(space={}),
            "bad space kind": self.document(space={"k": {"normal": [0, 1]}}),
        }
        for name, document in bad.items():
            with self.subTest(name=name), self.assertRaises(SweepError):
                self.load(document)

    def test_unreadable_files(self):
        with self.assertRaises(SweepError):
            load_halving(self.path)  # does not exist
        self.path.write_text("{not json")
        with self.assertRaises(SweepError):
            load_halving(self.path)


if __name__ == "__main__":
    unittest.main()
