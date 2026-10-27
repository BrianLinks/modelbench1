"""Check the claims in SECURITY.md and docs/THREAT_MODEL.md that can be checked mechanically."""

import ast
import json
import tempfile
import unittest
from pathlib import Path

from modelbench.dashboard import handle
from modelbench.datasets import read_csv
from modelbench.errors import DatasetError
from modelbench.export import export_json
from modelbench.importer import import_runs_json
from modelbench.store import Store
from test_golden import GOLDEN  # the frozen export is a convenient valid document

SRC = Path(__file__).resolve().parent.parent / "src" / "modelbench"
ROOT = Path(__file__).resolve().parent.parent


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def called_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


class SourceScanTests(unittest.TestCase):
    FILES = sorted(SRC.rglob("*.py"))

    def test_there_are_files_to_scan(self):
        self.assertGreater(len(self.FILES), 30)

    def test_no_dynamic_code_execution_or_unsafe_deserialisation(self):
        forbidden_modules = {"pickle", "marshal", "shelve", "subprocess", "ctypes"}
        forbidden_calls = {"eval", "exec", "compile", "__import__"}
        for path in self.FILES:
            self.assertFalse(imported_modules(path) & forbidden_modules, path.name)
            self.assertFalse(called_names(path) & forbidden_calls, path.name)

    def test_only_the_dashboard_touches_the_network(self):
        network = {"socket", "socketserver", "http.server", "http.client", "urllib.request",
                   "ssl", "ftplib", "smtplib", "xmlrpc", "requests", "httpx"}  # fmt: skip
        for path in self.FILES:
            found = {m for m in imported_modules(path) if m.split(".")[0] in
                     {n.split(".")[0] for n in network} and m != "urllib.parse"}  # fmt: skip
            if path.name != "dashboard.py":
                self.assertFalse(found, f"{path.name} imports {sorted(found)}")

    def test_the_package_has_no_runtime_dependencies(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("dependencies = []", text)


class InputHandlingTests(unittest.TestCase):
    def test_a_csv_outside_the_allowed_folder_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            inside = Path(tmp) / "a.csv"
            inside.write_text("x,label\n1,a\n")
            outside = Path(other) / "b.csv"
            outside.write_text("x,label\n1,a\n")
            self.assertEqual(len(read_csv(inside, base_dir=tmp)), 1)
            with self.assertRaises(DatasetError):
                read_csv(outside, base_dir=tmp)
            sneaky = Path(tmp) / ".." / Path(other).name / "b.csv"
            with self.assertRaises(DatasetError):
                read_csv(sneaky, base_dir=tmp)

    def test_an_imported_artifact_path_is_data_and_is_never_opened_or_run(self):
        document = json.loads(GOLDEN.read_text(encoding="utf-8"))
        document["runs"][0]["artifacts"] = [
            {
                "name": "predictions",
                "kind": "predictions",
                "sha256": "0" * 64,
                "size_bytes": 5,
                "path": "/etc/passwd\x1b[31m<script>alert(1)</script>",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "r.db"
            with Store(db) as store:
                report = import_runs_json(store, json.dumps(document))
                self.assertEqual(len(report.imported), 2)
                exported = export_json(store, [report.imported[0][1]])
                self.assertIn("/etc/passwd", exported)  # kept as text, nothing more
            page = handle(db, f"/run/{report.imported[0][1]}", "localhost:8765", True).body
            self.assertNotIn("<script>alert(1)", page)

    def test_the_dashboard_refuses_to_write_and_to_answer_foreign_hosts(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "r.db"
            Store(db).close()
            self.assertEqual(handle(db, "/", "evil.example", True).status, 403)
            self.assertEqual(handle(db, "/", "localhost:8765", True).status, 200)


if __name__ == "__main__":
    unittest.main()
