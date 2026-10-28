"""ModelBench stands alone: standard library only, no sibling project, no paid service."""

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "modelbench"
PAID_HOSTS = ("api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com")


def top_level_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0], node.lineno


class StandaloneTests(unittest.TestCase):
    def test_there_are_files_to_scan(self):
        self.assertGreater(len(list(PACKAGE.rglob("*.py"))), 30)

    def test_every_import_is_the_standard_library_or_the_package_itself(self):
        allowed = set(sys.stdlib_module_names) | {"modelbench"}
        for path in sorted(PACKAGE.rglob("*.py")):
            for name, line in top_level_imports(path):
                self.assertIn(name, allowed, f"{path.relative_to(ROOT)}:{line} imports {name}")

    def test_nothing_refers_to_a_paid_service(self):
        for path in sorted(PACKAGE.rglob("*.py")):
            text = path.read_text(encoding="utf-8").lower()
            for host in PAID_HOSTS:
                self.assertNotIn(host, text, f"{path.relative_to(ROOT)} mentions {host}")

    def test_the_declared_dependencies_stay_empty(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("dependencies = []", text)


if __name__ == "__main__":
    unittest.main()
