"""Run the Python examples in the documentation, so the docs cannot describe code that is gone."""

import contextlib
import io
import os
import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


def python_blocks(path):
    text = path.read_text(encoding="utf-8")
    return [(text[: m.start()].count("\n") + 2, m.group(1)) for m in FENCE.finditer(text)]


def run_blocks(path):
    """Run a file's blocks in order in one namespace and one empty folder; return the output."""
    namespace = {"__name__": "__docs__"}
    output = io.StringIO()
    previous = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            for line, code in python_blocks(path):
                try:
                    with contextlib.redirect_stdout(output):
                        exec(compile(code, f"{path.name}:{line}", "exec"), namespace)  # noqa: S102
                except Exception as exc:
                    raise AssertionError(
                        f"{path.name} line {line}: {type(exc).__name__}: {exc}"
                    ) from exc
        finally:
            os.chdir(previous)
    return namespace, output.getvalue()


class DocumentedExamplesTests(unittest.TestCase):
    def test_the_api_guide_runs(self):
        blocks = python_blocks(ROOT / "docs" / "API.md")
        self.assertGreaterEqual(len(blocks), 10)
        namespace, _ = run_blocks(ROOT / "docs" / "API.md")
        self.assertEqual(namespace["result"].metrics, {"accuracy": 1.0, "f1": 1.0})

    def test_the_readme_python_blocks_run_in_order(self):
        blocks = python_blocks(ROOT / "README.md")
        self.assertGreaterEqual(len(blocks), 6)
        namespace, output = run_blocks(ROOT / "README.md")
        # the values the README quotes in comments
        self.assertEqual(namespace["run_id"], "run-000001")
        self.assertEqual(
            namespace["store"].final_metrics("run-000001"), {"accuracy": 1.0, "f1": 1.0}
        )
        self.assertEqual(namespace["report"].verdict, "reproduced")
        self.assertFalse(namespace["report"].explained_by_environment)
        self.assertEqual(namespace["model"].predict([[0.5], [10.5]]), ["a", "b"])
        self.assertIn("if x0 <= 6.5:\n  predict 'a' (n=3)\n  else: predict 'b' (n=3)", output)
        boot = namespace["result"]  # the bootstrap block reuses the name
        self.assertAlmostEqual(boot.mean_diff, 0.2)
        self.assertGreater(boot.ci_low, 0)

    def test_the_fences_are_balanced(self):
        for name in ("README.md", "docs/API.md", "docs/ARCHITECTURE.md", "docs/SCHEMA.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertEqual(text.count("```") % 2, 0, name)


if __name__ == "__main__":
    unittest.main()
