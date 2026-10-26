import re
import unittest
from pathlib import Path

from modelbench.store import MIGRATIONS, Store

DOC = Path(__file__).resolve().parent.parent / "docs" / "SCHEMA.md"


def documented_tables():
    """`{table: {column: (type, nullable)}}` from the `### table` sections of SCHEMA.md."""
    tables = {}
    current = None
    for line in DOC.read_text(encoding="utf-8").splitlines():
        heading = re.fullmatch(r"### (\w+)", line)
        if heading:
            current = heading.group(1)
            tables[current] = {}
            continue
        if line.startswith("## "):
            current = None
        row = re.match(r"\| (\w+) \| (TEXT|INTEGER|REAL) \| (yes|no) \|", line)
        if row and current:
            tables[current][row.group(1)] = (row.group(2), row.group(3) == "yes")
    return tables


class SchemaDocTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)
        self.conn = self.store._conn  # the only way to ask SQLite about its own schema

    def actual_tables(self):
        names = [
            r[0]
            for r in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        out = {}
        for name in names:
            columns = self.conn.execute(f"PRAGMA table_info({name})").fetchall()
            # a column may hold NULL when it is not declared NOT NULL; an INTEGER PRIMARY KEY
            # or TEXT PRIMARY KEY on its own is documented as nullable because SQLite allows it
            out[name] = {c[1]: (c[2], not c[3]) for c in columns}
        return out

    def test_every_table_and_column_is_documented_with_its_type_and_nullability(self):
        self.assertEqual(documented_tables(), self.actual_tables())

    def test_the_documented_migration_count_is_current(self):
        text = DOC.read_text(encoding="utf-8")
        self.assertIn(f"(currently {len(MIGRATIONS)})", text)
        self.assertEqual(self.store.schema_version, len(MIGRATIONS))

    def test_every_index_is_documented(self):
        indexes = {
            r[0]
            for r in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"
            )
        }
        text = DOC.read_text(encoding="utf-8")
        for name in indexes:
            self.assertIn(f"| {name} |", text)
        self.assertEqual(len(indexes), len(re.findall(r"^\| idx_\w+ \|", text, re.M)))

    def test_cascade_claims_hold(self):
        for table in ("params", "metrics", "artifacts", "tags", "notes"):
            actions = [r[6] for r in self.conn.execute(f"PRAGMA foreign_key_list({table})")]
            self.assertEqual(actions, ["CASCADE"], table)

    def test_status_values_match_the_check_constraint(self):
        sql = self.conn.execute("SELECT sql FROM sqlite_master WHERE name = 'runs'").fetchone()[0]
        for status in ("running", "completed", "failed"):
            self.assertIn(f"'{status}'", sql)


if __name__ == "__main__":
    unittest.main()
