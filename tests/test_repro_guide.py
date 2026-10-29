"""The reproducibility guide must describe what the code does; these tests check its claims."""

import re
import unittest
from pathlib import Path

from modelbench import reproducibility
from modelbench.config import ExperimentConfig
from modelbench.datasets import Table, make_version
from modelbench.environment import DEFAULT_LIBRARIES, environment_fingerprint
from modelbench.runner import columns_signature

ROOT = Path(__file__).resolve().parent.parent
GUIDE = (ROOT / "docs" / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
SOURCE = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "src" / "modelbench").glob("*.py")}


def config(**overrides):
    fields = {
        "experiment_id": "a",
        "model": "knn",
        "seed": 1,
        "dataset_id": "d",
        "metrics": ("accuracy",),
        "params": {"k": 3},
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


class GuideNamesWhatExistsTests(unittest.TestCase):
    def test_every_verdict_is_described(self):
        for verdict in (
            reproducibility.REPRODUCED,
            reproducibility.NOT_REPRODUCED,
            reproducibility.DATASET_CHANGED,
            reproducibility.RERUN_FAILED,
        ):
            self.assertIn(f"`{verdict}`", GUIDE)

    def test_every_fingerprint_field_is_described(self):
        for key in environment_fingerprint():
            name = key.removeprefix("lib:")
            self.assertIn(name, GUIDE, key)
        for library in DEFAULT_LIBRARIES:
            self.assertIn(library, GUIDE)

    def test_every_seed_purpose_exists_in_the_code(self):
        purposes = {
            "split": "datasets.py",
            "kfold": "datasets.py",
            "model-fold-": "runner.py",
            "learning-curve": "learning_curve.py",
            "permutation": "importance.py",
        }
        for purpose, module in purposes.items():
            self.assertIn(purpose, GUIDE)
            self.assertIn(f'"{purpose}', SOURCE[module], purpose)
        self.assertIn('"model"', SOURCE["runner.py"])

    def test_the_tags_it_mentions_are_written_by_the_runner(self):
        for tag in re.findall(r"`((?:predictions|columns)_sha256|model_seed)`", GUIDE):
            self.assertIn(f'"{tag}"', SOURCE["runner.py"], tag)

    def test_the_links_in_the_readme_resolve(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("docs/REPRODUCIBILITY.md", readme)
        self.assertTrue((ROOT / "docs" / "REPRODUCIBILITY.md").is_file())


class GuideBehaviourClaimsTests(unittest.TestCase):
    def test_reordering_rows_is_a_new_dataset_version(self):
        rows = (("1", "a"), ("2", "b"), ("3", "a"))
        same = make_version("d", Table(("x", "label"), rows), "label")
        again = make_version("d", Table(("x", "label"), rows), "label")
        swapped = make_version("d", Table(("x", "label"), rows[::-1]), "label")
        self.assertEqual(same.version, again.version)
        self.assertNotEqual(same.version, swapped.version)

    def test_the_label_choice_is_recorded_with_the_columns_not_in_the_version(self):
        table = Table(("x", "y"), (("1", "2"), ("3", "4")))
        self.assertEqual(
            make_version("d", table, "x").version, make_version("d", table, "y").version
        )
        self.assertNotEqual(
            columns_signature(table, "x", None), columns_signature(table, "y", None)
        )

    def test_the_experiment_label_is_not_in_the_config_hash(self):
        self.assertEqual(config().config_hash(), config(experiment_id="other").config_hash())

    def test_one_and_one_point_zero_hash_differently(self):
        self.assertNotEqual(
            config(params={"k": 1}).config_hash(), config(params={"k": 1.0}).config_hash()
        )

    def test_the_cv_folds_setting_is_in_the_config_hash(self):
        self.assertNotEqual(config().config_hash(), config(cv_folds=3).config_hash())


if __name__ == "__main__":
    unittest.main()
