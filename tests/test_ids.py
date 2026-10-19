import unittest

from modelbench.errors import IdentifierError, ModelBenchError
from modelbench.ids import KINDS, is_valid_id, validate_id


class ValidIdTests(unittest.TestCase):
    def test_accepts_typical_identifiers(self):
        for value in ("iris", "iris-v2", "run_001", "exp.baseline", "a", "0abc", "a" * 64):
            self.assertEqual(validate_id("run", value), value)

    def test_every_kind_uses_the_same_rules(self):
        for kind in KINDS:
            self.assertTrue(is_valid_id(kind, "ok-id"))
            self.assertFalse(is_valid_id(kind, "Not OK"))


class InvalidIdTests(unittest.TestCase):
    def test_rejects_unsafe_or_malformed_identifiers(self):
        bad = [
            "",
            "A",  # uppercase
            "has space",
            "../etc/passwd",
            "a/b",
            "a\\b",
            "a..b",
            "trailing.",
            "-leading",
            ".hidden",
            "_x",
            "a" * 65,
            "naïve",
            "tab\tname",
            "line\nbreak",
            "semi;colon",
        ]
        for value in bad:
            with self.assertRaises(IdentifierError, msg=repr(value)):
                validate_id("dataset", value)

    def test_newline_suffix_is_not_swallowed_by_the_pattern(self):
        # a regex using `$` instead of fullmatch would accept this
        with self.assertRaises(IdentifierError):
            validate_id("run", "good\n")

    def test_non_strings_rejected(self):
        for value in (None, 5, 1.5, b"bytes", ["a"], True):
            with self.assertRaises(IdentifierError, msg=repr(value)):
                validate_id("experiment", value)

    def test_unknown_kind_rejected(self):
        with self.assertRaises(IdentifierError):
            validate_id("galaxy", "ok")

    def test_error_hierarchy(self):
        with self.assertRaises(ModelBenchError):
            validate_id("run", "")
        with self.assertRaises(ValueError):
            validate_id("run", "")

    def test_message_names_the_kind_and_value(self):
        with self.assertRaises(IdentifierError) as ctx:
            validate_id("experiment", "Bad Name")
        self.assertIn("experiment", str(ctx.exception))
        self.assertIn("Bad Name", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
