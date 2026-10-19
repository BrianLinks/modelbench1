import platform
import sys
import unittest
from importlib import metadata
from unittest import mock

from modelbench.environment import compare_environments, environment_fingerprint


class FingerprintTests(unittest.TestCase):
    def test_contains_interpreter_and_platform_facts(self):
        info = environment_fingerprint(libraries=())
        major, minor, micro = sys.version_info[:3]
        self.assertEqual(info["python"], f"{major}.{minor}.{micro}")
        self.assertEqual(info["implementation"], platform.python_implementation())
        self.assertEqual(info["system"], platform.system())
        self.assertEqual(info["machine"], platform.machine())
        self.assertIn("pythonhashseed", info)

    def test_records_pythonhashseed_when_set(self):
        with mock.patch.dict("os.environ", {"PYTHONHASHSEED": "0"}):
            self.assertEqual(environment_fingerprint(())["pythonhashseed"], "0")
        with mock.patch.dict("os.environ", clear=True):
            self.assertIsNone(environment_fingerprint(())["pythonhashseed"])

    def test_library_versions_and_missing_libraries(self):
        def fake_version(name):
            if name == "numpy":
                return "9.9.9"
            raise metadata.PackageNotFoundError(name)

        with mock.patch("modelbench.environment.metadata.version", side_effect=fake_version):
            info = environment_fingerprint(("numpy", "not-installed"))
        self.assertEqual(info["lib:numpy"], "9.9.9")
        self.assertIsNone(info["lib:not-installed"])

    def test_values_are_strings_or_none(self):
        for value in environment_fingerprint().values():
            self.assertTrue(value is None or isinstance(value, str))


class CompareTests(unittest.TestCase):
    BASE = {
        "python": "3.12.1",
        "system": "Linux",
        "pythonhashseed": "0",
        "lib:numpy": "1.26.0",
    }

    def test_identical_fingerprints_have_no_differences(self):
        self.assertEqual(compare_environments(self.BASE, dict(self.BASE)), [])

    def test_changed_values_are_reported_with_an_impact(self):
        new = {**self.BASE, "python": "3.13.0", "lib:numpy": "2.0.0"}
        diffs = {d.key: d for d in compare_environments(self.BASE, new)}
        self.assertEqual(set(diffs), {"python", "lib:numpy"})
        self.assertEqual((diffs["python"].old, diffs["python"].new), ("3.12.1", "3.13.0"))
        self.assertIn("library", diffs["lib:numpy"].impact)
        self.assertIn("Python", diffs["python"].impact)

    def test_keys_missing_on_one_side_are_differences(self):
        new = {k: v for k, v in self.BASE.items() if k != "lib:numpy"}
        (diff,) = compare_environments(self.BASE, new)
        self.assertEqual((diff.key, diff.old, diff.new), ("lib:numpy", "1.26.0", None))

    def test_installing_a_library_counts(self):
        old = {**self.BASE, "lib:numpy": None}
        (diff,) = compare_environments(old, self.BASE)
        self.assertEqual((diff.old, diff.new), (None, "1.26.0"))

    def test_hash_seed_difference_explains_set_ordering(self):
        (diff,) = compare_environments(self.BASE, {**self.BASE, "pythonhashseed": None})
        self.assertIn("set ordering", diff.impact)

    def test_output_is_sorted_by_key(self):
        new = {"python": "3", "system": "Mac", "pythonhashseed": "1", "lib:numpy": "2"}
        keys = [d.key for d in compare_environments(self.BASE, new)]
        self.assertEqual(keys, sorted(keys))


if __name__ == "__main__":
    unittest.main()
