import hashlib
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from modelbench.config import (
    KNOWN_METRICS,
    MAX_GRID_SIZE,
    ExperimentConfig,
    check_json_value,
    expand_config,
    expand_grid,
)
from modelbench.errors import ConfigError, IdentifierError, SeedError


def make(**overrides):
    fields = {
        "experiment_id": "exp-a",
        "model": "knn",
        "seed": 1,
        "dataset_id": "iris",
        "metrics": ("accuracy",),
        "params": {"k": 3},
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


class ConstructionTests(unittest.TestCase):
    def test_defaults(self):
        cfg = make()
        self.assertEqual((cfg.test_fraction, cfg.stratify, cfg.dataset_version), (0.2, False, None))

    def test_is_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            make().seed = 5

    def test_identifier_and_seed_validation(self):
        with self.assertRaises(IdentifierError):
            make(experiment_id="Bad Id")
        with self.assertRaises(IdentifierError):
            make(model="../x")
        with self.assertRaises(IdentifierError):
            make(dataset_id="")
        with self.assertRaises(SeedError):
            make(seed=-1)
        with self.assertRaises(SeedError):
            make(seed=1.5)

    def test_test_fraction_validation(self):
        for bad in (0, 1, -0.2, 1.2, True, "0.2", None, float("nan")):
            with self.assertRaises(ConfigError, msg=repr(bad)):
                make(test_fraction=bad)

    def test_dataset_version_validation(self):
        self.assertEqual(make(dataset_version="0123456789ab").dataset_version, "0123456789ab")
        for bad in ("0123456789AB", "short", "0123456789abc", 12, ""):
            with self.assertRaises(ConfigError, msg=repr(bad)):
                make(dataset_version=bad)

    def test_metric_validation(self):
        for bad in ((), ("accuracy", "accuracy"), ("bleu",), ["accuracy"], "accuracy"):
            with self.assertRaises(ConfigError, msg=repr(bad)):
                make(metrics=bad)
        self.assertEqual(make(metrics=KNOWN_METRICS).metrics, KNOWN_METRICS)

    def test_stratify_must_be_a_real_boolean(self):
        for bad in (1, "yes", None):
            with self.assertRaises(ConfigError, msg=repr(bad)):
                make(stratify=bad)

    def test_params_are_copied_so_outside_mutation_cannot_change_the_config(self):
        params = {"layers": [1, 2]}
        cfg = make(params=params)
        params["layers"].append(3)
        self.assertEqual(cfg.params, {"layers": [1, 2]})
        cfg.to_dict()["params"]["layers"].append(9)
        self.assertEqual(cfg.params, {"layers": [1, 2]})


class JsonValueTests(unittest.TestCase):
    def test_accepts_json_compatible_values(self):
        check_json_value({"a": [1, 2.5, "x", None, True, {"b": []}]}, "params")

    def test_rejects_everything_else(self):
        bad = [
            {"a": float("nan")},
            {"a": float("inf")},
            {"a": (1, 2)},
            {"a": {1, 2}},
            {1: "x"},
            {"a": b"bytes"},
            {"a": object()},
        ]
        for value in bad:
            with self.assertRaises(ConfigError, msg=repr(value)):
                check_json_value(value, "params")

    def test_error_names_the_location(self):
        with self.assertRaises(ConfigError) as ctx:
            check_json_value({"a": [1, {"b": float("nan")}]}, "params")
        self.assertIn("params.a[1].b", str(ctx.exception))

    def test_depth_is_limited(self):
        value = current = {}
        for _ in range(30):
            current["x"] = {}
            current = current["x"]
        with self.assertRaises(ConfigError):
            check_json_value(value, "params")


class HashTests(unittest.TestCase):
    def test_hash_matches_an_independent_hash_of_the_canonical_text(self):
        literal = (
            b'{"dataset_id":"iris","dataset_version":null,"metrics":["accuracy"],'
            b'"model":"knn","params":{"k":3},"seed":1,"stratify":false,"test_fraction":0.2}'
        )
        self.assertEqual(make().config_hash(), hashlib.sha256(literal).hexdigest())

    def test_label_does_not_change_the_hash_but_settings_do(self):
        base = make()
        self.assertEqual(base.config_hash(), make(experiment_id="other").config_hash())
        changes = [
            {"seed": 2},
            {"model": "tree"},
            {"params": {"k": 4}},
            {"dataset_id": "wine"},
            {"dataset_version": "0123456789ab"},
            {"test_fraction": 0.3},
            {"stratify": True},
            {"metrics": ("accuracy", "f1")},
        ]
        for change in changes:
            self.assertNotEqual(base.config_hash(), make(**change).config_hash(), change)

    def test_param_key_order_does_not_matter(self):
        a = make(params={"a": 1, "b": 2})
        b = make(params={"b": 2, "a": 1})
        self.assertEqual(a.config_hash(), b.config_hash())

    def test_metric_order_does_matter(self):
        a = make(metrics=("accuracy", "f1"))
        b = make(metrics=("f1", "accuracy"))
        self.assertNotEqual(a.config_hash(), b.config_hash())


class LoadingTests(unittest.TestCase):
    DOC = {
        "experiment_id": "exp-a",
        "model": "knn",
        "seed": 7,
        "dataset_id": "iris",
        "metrics": ["accuracy", "f1"],
        "params": {"k": 5},
    }

    def test_round_trip(self):
        cfg = ExperimentConfig.from_dict(self.DOC)
        again = ExperimentConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
        self.assertEqual(cfg, again)
        self.assertEqual(cfg.config_hash(), again.config_hash())

    def test_unknown_and_missing_keys(self):
        with self.assertRaises(ConfigError):
            ExperimentConfig.from_dict({**self.DOC, "surprise": 1})
        for key in ("experiment_id", "model", "seed", "dataset_id", "metrics"):
            doc = {k: v for k, v in self.DOC.items() if k != key}
            with self.assertRaises(ConfigError, msg=key):
                ExperimentConfig.from_dict(doc)

    def test_wrong_shapes(self):
        for bad in ([], "x", None, {**self.DOC, "metrics": "accuracy"}, {**self.DOC, "params": []}):
            with self.assertRaises(ConfigError, msg=repr(bad)):
                ExperimentConfig.from_dict(bad)

    def test_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.json"
            good.write_text(json.dumps(self.DOC), encoding="utf-8")
            self.assertEqual(ExperimentConfig.from_file(good).seed, 7)
            broken = Path(tmp) / "broken.json"
            broken.write_text("{nope", encoding="utf-8")
            binary = Path(tmp) / "binary.json"
            binary.write_bytes(b"\xff\xfe\x00")
            for bad in (broken, binary, Path(tmp) / "missing.json"):
                with self.assertRaises(ConfigError, msg=str(bad)):
                    ExperimentConfig.from_file(bad)


class GridTests(unittest.TestCase):
    def test_cartesian_product_in_deterministic_order(self):
        grid = {"b": [10, 20], "a": [1, 2, 3]}
        result = expand_grid({"fixed": "x"}, grid)
        self.assertEqual(len(result), 6)
        # keys are visited in sorted order ("a" varies slowest), values in given order
        self.assertEqual(
            [(r["a"], r["b"]) for r in result],
            [(1, 10), (1, 20), (2, 10), (2, 20), (3, 10), (3, 20)],
        )
        self.assertTrue(all(r["fixed"] == "x" for r in result))
        self.assertEqual(result, expand_grid({"fixed": "x"}, {"a": [1, 2, 3], "b": [10, 20]}))

    def test_empty_grid_returns_the_base(self):
        self.assertEqual(expand_grid({"k": 1}, {}), [{"k": 1}])

    def test_results_do_not_share_mutable_state(self):
        result = expand_grid({"layers": [1]}, {"lr": [0.1, 0.2]})
        result[0]["layers"].append(2)
        self.assertEqual(result[1]["layers"], [1])

    def test_invalid_grids(self):
        with self.assertRaises(ConfigError):
            expand_grid({"k": 1}, {"k": [1, 2]})  # overlap
        with self.assertRaises(ConfigError):
            expand_grid({}, {"k": []})
        with self.assertRaises(ConfigError):
            expand_grid({}, {"k": (1, 2)})
        with self.assertRaises(ConfigError):
            expand_grid({}, {"k": [float("nan")]})

    def test_size_limit(self):
        side = int(MAX_GRID_SIZE**0.5) + 1
        with self.assertRaises(ConfigError):
            expand_grid({}, {"a": list(range(side)), "b": list(range(side))})

    def test_expand_config_gives_validated_variants_with_distinct_hashes(self):
        configs = expand_config(make(params={"metric": "l2"}), {"k": [1, 3, 5]})
        self.assertEqual([c.params["k"] for c in configs], [1, 3, 5])
        self.assertEqual({c.experiment_id for c in configs}, {"exp-a"})
        self.assertEqual(len({c.config_hash() for c in configs}), 3)
        self.assertEqual(replace(configs[0], params={"metric": "l2", "k": 1}), configs[0])


if __name__ == "__main__":
    unittest.main()
