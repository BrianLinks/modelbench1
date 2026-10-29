"""The pinned development tools stay consistent with pyproject.toml and CI."""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([0-9][^;\s]*)(?:\s*;\s*(.+))?$")


def pins():
    found = {}
    for line in LOCK.splitlines():
        if line.strip() and not line.startswith("#"):
            match = PIN.match(line.strip())
            assert match, f"not an exact pin: {line!r}"
            found[match.group(1).lower().replace("_", "-")] = (match.group(2), match.group(3))
    return found


def version_tuple(text):
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


class DevLockTests(unittest.TestCase):
    def test_every_line_is_an_exact_pin_limited_to_python_3_12_and_newer(self):
        found = pins()
        self.assertGreaterEqual(len(found), 8)
        for name, (_, marker) in found.items():
            self.assertIsNotNone(marker, name)
            self.assertIn('python_version >= "3.12"', marker, name)

    def test_the_tools_pyproject_asks_for_are_pinned_at_or_above_its_minimum(self):
        found = pins()
        wanted = re.search(r"^dev = \[(.*?)\]", PYPROJECT, re.M | re.S).group(1)
        for name, minimum in re.findall(r'"([A-Za-z0-9_.-]+)>=([0-9.]+)"', wanted):
            self.assertIn(name.lower(), found, name)
            pinned = version_tuple(found[name.lower()][0])
            self.assertGreaterEqual(pinned, version_tuple(minimum), name)

    def test_ci_and_the_release_workflow_install_with_the_lock_as_constraints(self):
        for name in ("ci.yml", "release.yml"):
            text = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
            self.assertRegex(text, r'pip install -e "\.\[dev\]"[^\n]* -c requirements-dev\.txt')

    def test_every_workflow_install_of_the_package_uses_the_lock(self):
        installs = 0
        for path in (ROOT / ".github" / "workflows").glob("*.yml"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if "pip install -e" in line:
                    installs += 1
                    self.assertIn("-c requirements-dev.txt", line, f"{path.name}: {line.strip()}")
        self.assertGreaterEqual(installs, 2)

    def test_the_lock_names_no_runtime_dependency(self):
        self.assertIn("dependencies = []", PYPROJECT)
        self.assertNotIn("modelbench", {name for name in pins()})


if __name__ == "__main__":
    unittest.main()
