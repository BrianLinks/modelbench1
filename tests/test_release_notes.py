import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import release_notes  # noqa: E402
from release_notes import changelog_section, declared_versions, normalise  # noqa: E402

CHANGELOG = """# Changelog

## [Unreleased]

### Added
- something pending

## [0.3.0] - 2026-11-01

### Added
- the new thing

### Fixed
- the old thing

## [0.2.0] - 2026-10-26

### Added
- earlier
"""


def make_root(root, version="0.3.0", init_version=None, changelog=CHANGELOG):
    (root / "src" / "modelbench").mkdir(parents=True)
    (root / "pyproject.toml").write_text(f'[project]\nname = "x"\nversion = "{version}"\n')
    (root / "src" / "modelbench" / "__init__.py").write_text(
        f'"""doc"""\n\n__version__ = "{init_version or version}"\n'
    )
    (root / "CHANGELOG.md").write_text(changelog)


class NormaliseTests(unittest.TestCase):
    def test_tags_and_bare_versions(self):
        self.assertEqual(normalise("v0.3.0"), "0.3.0")
        self.assertEqual(normalise("10.20.30"), "10.20.30")

    def test_anything_else_is_rejected(self):
        for bad in ("", "v", "0.3", "v0.3.0-rc1", "release-1.2.3", "V1.2.3", "1.2.3.4", "v1.2.x"):
            with self.subTest(bad), self.assertRaises(ValueError):
                normalise(bad)


class SectionTests(unittest.TestCase):
    def test_the_body_stops_at_the_next_release(self):
        body = changelog_section(CHANGELOG, "0.3.0")
        self.assertEqual(body, "### Added\n- the new thing\n\n### Fixed\n- the old thing")

    def test_the_last_section_runs_to_the_end(self):
        self.assertEqual(changelog_section(CHANGELOG, "0.2.0"), "### Added\n- earlier")

    def test_missing_undated_or_empty_sections_fail(self):
        with self.assertRaisesRegex(ValueError, "no dated section for 0.4.0"):
            changelog_section(CHANGELOG, "0.4.0")
        with self.assertRaisesRegex(ValueError, "no dated section for"):
            changelog_section(CHANGELOG, "Unreleased")
        with self.assertRaisesRegex(ValueError, "empty"):
            changelog_section("## [1.0.0] - 2026-01-01\n\n## [0.9.0] - 2025-01-01\n- x\n", "1.0.0")

    def test_a_version_is_not_matched_as_a_prefix(self):
        text = "## [0.3.0] - 2026-01-01\n- a\n\n## [0.30.0] - 2026-02-01\n- b\n"
        self.assertEqual(changelog_section(text, "0.3.0"), "- a")
        self.assertEqual(changelog_section(text, "0.30.0"), "- b")


class ReleaseNotesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_matching_versions_give_the_notes(self):
        make_root(self.root)
        self.assertEqual(
            declared_versions(self.root), {"pyproject.toml": "0.3.0", "__init__.py": "0.3.0"}
        )
        notes = release_notes.release_notes("v0.3.0", self.root)
        self.assertTrue(notes.startswith("### Added"))

    def test_either_file_out_of_step_fails_and_names_it(self):
        make_root(self.root, version="0.3.0", init_version="0.2.0")
        with self.assertRaisesRegex(ValueError, "__init__.py says 0.2.0"):
            release_notes.release_notes("v0.3.0", self.root)
        with self.assertRaisesRegex(ValueError, "pyproject.toml says 0.3.0"):
            release_notes.release_notes("v0.2.0", self.root)

    def test_a_missing_version_line_is_an_error(self):
        make_root(self.root)
        (self.root / "pyproject.toml").write_text("[project]\nname = 'x'\n")
        with self.assertRaisesRegex(ValueError, "no version found"):
            declared_versions(self.root)

    def test_the_real_repository_is_consistent(self):
        versions = declared_versions()
        self.assertEqual(len(set(versions.values())), 1, versions)
        version = next(iter(versions.values()))
        self.assertTrue(release_notes.release_notes(f"v{version}"))


class MainTests(unittest.TestCase):
    def run_main(self, args):
        import contextlib
        import io

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = release_notes.main(args)
        return code, out.getvalue(), err.getvalue()

    def test_usage_and_failures_have_distinct_exit_codes(self):
        self.assertEqual(self.run_main([])[0], 2)
        self.assertEqual(self.run_main(["a", "b"])[0], 2)
        code, out, err = self.run_main(["v999.0.0"])
        self.assertEqual((code, out), (1, ""))
        self.assertIn("does not match", err)
        code, _, err = self.run_main(["nonsense"])
        self.assertEqual(code, 1)
        self.assertIn("not a version", err)

    def test_success_prints_the_notes(self):
        version = next(iter(declared_versions().values()))
        code, out, err = self.run_main([f"v{version}"])
        self.assertEqual((code, err), (0, ""))
        self.assertTrue(out.strip())


class WorkflowTests(unittest.TestCase):
    """The workflows cannot run here, so check the pieces they depend on."""

    ROOT = Path(__file__).resolve().parent.parent

    def test_the_release_workflow_uses_scripts_that_exist(self):
        text = (self.ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
        self.assertIn("scripts/release_notes.py", text)
        self.assertTrue((self.ROOT / "scripts" / "release_notes.py").is_file())
        self.assertIn("--verify-tag", text)
        self.assertIn("contents: write", text)

    def test_the_test_matrix_covers_windows_and_every_supported_python(self):
        text = (self.ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn("windows-latest", text)
        for version in ("3.10", "3.11", "3.12", "3.13"):
            self.assertIn(f'"{version}"', text)
        self.assertIn("fail-fast: false", text)


if __name__ == "__main__":
    unittest.main()
