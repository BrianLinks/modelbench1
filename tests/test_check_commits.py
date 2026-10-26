import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_commits  # noqa: E402
from check_commits import MAX_HEADER, TYPES, check_message, commit_messages, main  # noqa: E402


class CheckMessageTests(unittest.TestCase):
    def test_accepted_headers(self):
        for message in (
            "feat: add sweeps",
            "fix(store): reject a negative step",
            "feat(cli)!: rename --db to --database",
            "docs: explain the schema\n\nA body paragraph.\n- a list item",
            "chore(ci): run on windows",
            "test(models.knn): cover ties",
            "revert: undo the release workflow",
        ):
            with self.subTest(message=message):
                self.assertEqual(check_message(message), [])

    def test_every_listed_type_is_accepted(self):
        for kind in TYPES:
            self.assertEqual(check_message(f"{kind}: x"), [], kind)

    def test_rejected_headers_name_the_problem(self):
        cases = {
            "Add sweeps": "must look like",
            "feat add sweeps": "must look like",
            "feat:add sweeps": "must look like",
            "Feat: add sweeps": "must look like",
            "feature: add sweeps": "unknown type 'feature'",
            "feat(): add sweeps": "must look like",
            "feat(Store): add": "must look like",
            "feat: ": "the subject is empty",
            "feat:  padded": "leading or trailing spaces",
            "feat: ends with a stop.": "full stop",
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                problems = check_message(message)
                self.assertTrue(any(expected in p for p in problems), problems)

    def test_header_length_limit(self):
        ok = "feat: " + "x" * (MAX_HEADER - 6)
        self.assertEqual(len(ok), MAX_HEADER)
        self.assertEqual(check_message(ok), [])
        problems = check_message(ok + "x")
        self.assertEqual(len(problems), 1)
        self.assertIn(f"is {MAX_HEADER + 1} characters", problems[0])

    def test_a_blank_line_must_follow_the_header(self):
        self.assertEqual(check_message("fix: a\n\nbody"), [])
        self.assertEqual(check_message("fix: a\n"), [])
        problems = check_message("fix: a\nbody right away")
        self.assertEqual(problems, ["a blank line must follow the header"])

    def test_empty_messages(self):
        for message in ("", "\n\n", "   \nbody"):
            with self.subTest(message=message):
                self.assertTrue(check_message(message))

    def test_git_generated_headers_are_skipped(self):
        for message in ("Merge branch 'x' into main", "fixup! feat: a", "squash! fix: b"):
            self.assertEqual(check_message(message), [], message)

    def test_several_problems_are_all_reported(self):
        problems = check_message("Added a very " + "long " * 20 + "thing.")
        self.assertGreaterEqual(len(problems), 2)


class GitTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name)
        self._cwd = os.getcwd()
        self.addCleanup(os.chdir, self._cwd)
        os.chdir(self.repo)
        self.git("init", "-q")
        self.git("config", "user.email", "t@example.com")
        self.git("config", "user.name", "T")
        self.git("config", "commit.gpgsign", "false")

    def git(self, *args):
        subprocess.run(["git", *args], check=True, capture_output=True, text=True)

    def commit(self, message):
        (self.repo / "f.txt").write_text(message)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)

    def run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_commits_come_back_oldest_first_with_full_messages(self):
        self.commit("feat: first")
        self.commit("fix: second\n\nwith a body")
        found = commit_messages("HEAD")
        self.assertEqual([m.splitlines()[0] for _, m in found], ["feat: first", "fix: second"])
        self.assertIn("with a body", found[1][1])
        self.assertEqual(len(found[0][0]), 7)

    def test_a_range_only_checks_the_new_commits(self):
        self.commit("Old style message")
        self.git("tag", "base")
        self.commit("feat: new work")
        self.assertEqual(self.run_main("base..HEAD"), (0, "", ""))
        code, out, err = self.run_main("HEAD")
        self.assertEqual(code, 1)
        self.assertIn("must look like", out)
        self.assertIn("1 of 2 message(s) need fixing", err)

    def test_an_unknown_range_is_an_error(self):
        self.commit("feat: a")
        with self.assertRaises(SystemExit) as caught:
            commit_messages("nope..HEAD")
        self.assertIn("cannot read commits", str(caught.exception))

    def test_message_file_ignores_git_comment_lines(self):
        path = self.repo / "msg"
        path.write_text("fix: handle empty input\n# Please enter the commit message\n# more\n")
        self.assertEqual(self.run_main("--message-file", str(path)), (0, "", ""))
        path.write_text("oops\n# comment\n")
        code, out, _ = self.run_main("--message-file", str(path))
        self.assertEqual(code, 1)
        self.assertIn("must look like", out)

    def test_unreadable_message_file(self):
        code, _, err = self.run_main("--message-file", str(self.repo / "missing"))
        self.assertEqual(code, 2)
        self.assertIn("cannot read", err)

    def test_the_hook_script_calls_the_checker(self):
        hook = Path(check_commits.__file__).resolve().parent.parent / ".githooks" / "commit-msg"
        text = hook.read_text()
        self.assertIn("scripts/check_commits.py --message-file", text)
        self.assertTrue(text.startswith("#!/bin/sh"))


if __name__ == "__main__":
    unittest.main()
