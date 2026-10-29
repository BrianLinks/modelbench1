"""The dashboard script: what it may not do, and a run of its logic tests under Node."""

import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "src" / "modelbench" / "static" / "dashboard.js"

FORBIDDEN = {
    "innerHTML": r"\binnerHTML\b",
    "outerHTML": r"\bouterHTML\b",
    "insertAdjacentHTML": r"\binsertAdjacentHTML\b",
    "document.write": r"\bdocument\s*\.\s*write",
    "eval": r"\beval\s*\(",
    "Function constructor": r"\bnew\s+Function\b|\bFunction\s*\(",
    "string timers": r"\bset(?:Timeout|Interval)\s*\(\s*['\"`]",
    "fetch": r"\bfetch\s*\(",
    "XMLHttpRequest": r"\bXMLHttpRequest\b",
    "WebSocket": r"\bWebSocket\b",
    "EventSource": r"\bEventSource\b",
    "sendBeacon": r"\bsendBeacon\b",
    "importScripts": r"\bimportScripts\b",
    "dynamic import": r"\bimport\s*\(",
    "storage": r"\b(?:localStorage|sessionStorage|indexedDB)\b",
    "cookies": r"\bdocument\s*\.\s*cookie\b",
    "a URL": r"https?://",
}


class ScriptSafetyTests(unittest.TestCase):
    def setUp(self):
        # comments may mention anything; only code is checked
        text = SCRIPT.read_text("utf-8")
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        self.source = re.sub(r"^\s*//.*$", "", text, flags=re.M)

    def test_the_script_has_none_of_the_forbidden_constructs(self):
        for name, pattern in FORBIDDEN.items():
            with self.subTest(name=name):
                self.assertIsNone(re.search(pattern, self.source), name)

    def test_the_script_only_navigates_through_the_run_path_check(self):
        for match in re.finditer(r"location\s*\.\s*(assign|replace|href)", self.source):
            line = self.source[: match.start()].count("\n") + 1
            context = "\n".join(self.source.splitlines()[max(0, line - 2) : line])
            self.assertIn("isRunPath", context, f"line {line}")

    def test_the_script_is_plain_es5_so_any_browser_can_run_it(self):
        for name, pattern in {
            "arrow function": r"=>",
            "let/const": r"\b(?:let|const)\s",
            "template literal": r"`",
            "class": r"\bclass\s+\w+\s*[{]",
        }.items():
            with self.subTest(name=name):
                self.assertIsNone(re.search(pattern, self.source), name)

    def test_the_script_is_listed_as_package_data(self):
        text = (ROOT / "pyproject.toml").read_text("utf-8")
        self.assertIn('modelbench = ["static/*.js"]', text)


@unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
class NodeLogicTests(unittest.TestCase):
    def test_the_javascript_logic_tests_pass(self):
        result = subprocess.run(
            ["node", "--test", str(ROOT / "tests" / "js" / "dashboard.test.js")],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=ROOT,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertRegex(result.stdout, r"# fail 0\b")
        self.assertNotRegex(result.stdout, r"# pass 0\b")


if __name__ == "__main__":
    unittest.main()
