import random
import sys
import types
import unittest
from unittest import mock

from modelbench.errors import SeedError
from modelbench.seeding import MAX_SEED, check_seed, derive_seed, rng_for, seed_everything


class CheckSeedTests(unittest.TestCase):
    def test_accepts_range_endpoints(self):
        self.assertEqual(check_seed(0), 0)
        self.assertEqual(check_seed(MAX_SEED), MAX_SEED)

    def test_rejects_out_of_range_and_wrong_types(self):
        for bad in (-1, MAX_SEED + 1, 1.0, "7", None, True, False):
            with self.assertRaises(SeedError, msg=repr(bad)):
                check_seed(bad)


class DeriveSeedTests(unittest.TestCase):
    def test_golden_values_pin_the_derivation(self):
        # 1207492688 is int("47f8e050", 16), the first four bytes of
        # SHA-256 of the exact text [42,"split"], checked independently with hashlib.
        # If these change, every stored experiment's splits and runs change with them.
        self.assertEqual(derive_seed(0), 3502022929)
        self.assertEqual(derive_seed(42), 3094027937)
        self.assertEqual(derive_seed(42, "split"), 1207492688)
        self.assertEqual(derive_seed(42, "fold", 3), 2927506931)

    def test_deterministic_and_in_range(self):
        for base in (0, 1, 42, MAX_SEED):
            for parts in ((), ("a",), ("a", 1), (7, "b")):
                value = derive_seed(base, *parts)
                self.assertEqual(value, derive_seed(base, *parts))
                self.assertTrue(0 <= value <= MAX_SEED)

    def test_parts_and_order_and_types_matter(self):
        self.assertNotEqual(derive_seed(42, "a"), derive_seed(42, "b"))
        self.assertNotEqual(derive_seed(42, "a", "b"), derive_seed(42, "b", "a"))
        self.assertNotEqual(derive_seed(42, 3), derive_seed(42, "3"))
        self.assertNotEqual(derive_seed(42, "a,b"), derive_seed(42, "a", "b"))
        self.assertNotEqual(derive_seed(1, "x"), derive_seed(2, "x"))

    def test_invalid_inputs(self):
        with self.assertRaises(SeedError):
            derive_seed(-5, "x")
        for part in (1.5, None, True, ("tuple",)):
            with self.assertRaises(SeedError, msg=repr(part)):
                derive_seed(1, part)

    def test_no_collisions_across_many_labels(self):
        seeds = {derive_seed(42, "fold", i) for i in range(2000)}
        self.assertEqual(len(seeds), 2000)


class RngTests(unittest.TestCase):
    def test_same_labels_give_identical_streams(self):
        first, second = rng_for(9, "model"), rng_for(9, "model")
        self.assertEqual([first.random() for _ in range(20)], [second.random() for _ in range(20)])

    def test_different_labels_give_different_streams(self):
        self.assertNotEqual(rng_for(9, "a").random(), rng_for(9, "b").random())

    def test_private_generator_leaves_global_state_alone(self):
        random.seed(1)
        expected = random.random()
        random.seed(1)
        rng_for(7, "x").random()
        self.assertEqual(random.random(), expected)


class SeedEverythingTests(unittest.TestCase):
    def test_seeds_global_random_reproducibly(self):
        with mock.patch.dict(sys.modules, {"numpy": None}):
            seed_everything(5)
            first = [random.random() for _ in range(3)]
            seed_everything(5)
            second = [random.random() for _ in range(3)]
        self.assertEqual(first, second)
        self.assertEqual(first[0], random.Random(5).random())

    def test_reports_numpy_as_skipped_when_missing(self):
        with mock.patch.dict(sys.modules, {"numpy": None}):
            report = seed_everything(1)
        self.assertEqual((report.seed, report.seeded, report.skipped), (1, ("random",), ("numpy",)))

    def test_seeds_numpy_when_present(self):
        calls = []
        fake = types.ModuleType("numpy")
        fake.random = types.SimpleNamespace(seed=calls.append)
        with mock.patch.dict(sys.modules, {"numpy": fake}):
            report = seed_everything(123)
        self.assertEqual(calls, [123])
        self.assertEqual(report.seeded, ("random", "numpy"))
        self.assertEqual(report.skipped, ())

    def test_invalid_seed_changes_nothing(self):
        random.seed(3)
        expected = random.random()
        random.seed(3)
        with self.assertRaises(SeedError):
            seed_everything(-1)
        self.assertEqual(random.random(), expected)


if __name__ == "__main__":
    unittest.main()
