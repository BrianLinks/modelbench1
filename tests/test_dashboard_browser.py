"""Drives the dashboard in a real browser. Skipped unless Playwright and Chromium are installed.

The package has no dependencies; Playwright is something a developer may add to run these.
"""

import os
import random
import tempfile
import threading
import unittest
from pathlib import Path

from modelbench.config import ExperimentConfig
from modelbench.dashboard import make_server
from modelbench.datasets import Table
from modelbench.runner import run_experiment
from modelbench.store import Store
from modelbench.sweep import run_sweep

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

# CI sets this so that a missing browser fails the build instead of skipping the tests.
REQUIRED = os.environ.get("MODELBENCH_REQUIRE_BROWSER_TESTS") == "1"

ENV = {"python": "3.12.1", "lib:numpy": None}


def noisy_blobs():
    rng = random.Random(5)
    rows = []
    for label, centre in (("a", 0.0), ("b", 1.5)):
        for _ in range(60):
            rows.append((f"{rng.gauss(centre, 1.0):.3f}", f"{rng.gauss(centre, 1.0):.3f}", label))
    return Table(("x1", "x2", "label"), tuple(rows))


def config(**overrides):
    fields = {
        "experiment_id": "sweep",
        "model": "knn",
        "seed": 3,
        "dataset_id": "noisy",
        "metrics": ("accuracy", "f1"),
        "params": {},
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


class PlaywrightPresenceTests(unittest.TestCase):
    @unittest.skipUnless(REQUIRED, "browser tests are optional here")
    def test_playwright_is_installed_when_the_browser_tests_are_required(self):
        self.assertIsNotNone(sync_playwright)


@unittest.skipIf(sync_playwright is None, "Playwright is not installed")
class BrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.db = Path(cls._tmp.name) / "runs.db"
        table = noisy_blobs()
        with Store(cls.db) as store:
            run_sweep(
                store, config(), {"k": [1, 5, 25], "weights": ["uniform", "distance"]}, table,
                "label", environment=ENV,
            )  # fmt: skip
            run_experiment(
                store,
                config(experiment_id="base", model="majority"),
                table,
                "label",
                environment=ENV,
            )
        cls.server = make_server(cls.db, "127.0.0.1", 0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        try:
            cls.playwright = sync_playwright().start()
            cls.browser = cls.playwright.chromium.launch()
        except Exception as exc:  # no browser binary installed
            cls.server.shutdown()
            cls.server.server_close()
            cls._tmp.cleanup()
            if REQUIRED:
                raise
            raise unittest.SkipTest(f"Chromium is not available: {exc}") from exc

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(5)
        cls._tmp.cleanup()

    def open(self, path, javascript=True):
        context = self.browser.new_context(java_script_enabled=javascript)
        self.addCleanup(context.close)
        page = context.new_page()
        self.problems = []
        page.on("console", lambda m: self.problems.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: self.problems.append(str(e)))
        page.goto(self.base + path)
        return page

    def run_ids(self, page):
        return page.eval_on_selector_all(
            "table[data-runs] tbody tr:not([hidden]) td:nth-child(2)",
            "cells => cells.map(c => c.textContent.trim())",
        )

    def accuracy_of(self, page, run_id):
        text = page.eval_on_selector(
            f"table[data-runs] tbody tr:has(a[href='/run/{run_id}']) td:last-child",
            "c => c.textContent",
        )
        return float(text.split("accuracy=")[1].split()[0])

    def test_the_index_filters_and_sorts_runs_and_the_script_raises_no_errors(self):
        page = self.open("/")
        every = self.run_ids(page)
        self.assertEqual(len(every), 7)
        self.assertEqual(page.inner_text(".toolbar span"), "7 of 7 runs shown")

        page.fill(".toolbar input", "majority")
        self.assertEqual(len(self.run_ids(page)), 1)
        self.assertEqual(page.inner_text(".toolbar span"), "1 of 7 runs shown")
        page.fill(".toolbar input", "KNN distance")  # terms are anded, case is ignored
        self.assertEqual(page.inner_text(".toolbar span"), "0 of 7 runs shown")
        page.fill(".toolbar input", "")
        self.assertEqual(self.run_ids(page), every)

        page.select_option(".toolbar select", "accuracy")
        by_accuracy = [self.accuracy_of(page, r) for r in self.run_ids(page)]
        self.assertEqual(by_accuracy, sorted(by_accuracy, reverse=True))
        self.assertGreater(len(set(by_accuracy)), 2)  # the data really does separate the runs
        page.click(".toolbar button")
        by_accuracy = [self.accuracy_of(page, r) for r in self.run_ids(page)]
        self.assertEqual(by_accuracy, sorted(by_accuracy))

        page.click("th[data-sort]:nth-child(2)")  # Run column
        self.assertEqual(page.get_attribute("th[data-sort]:nth-child(2)", "aria-sort"), "ascending")
        self.assertEqual(self.run_ids(page), sorted(every))
        page.click("th[data-sort]:nth-child(2)")
        self.assertEqual(
            page.get_attribute("th[data-sort]:nth-child(2)", "aria-sort"), "descending"
        )
        self.assertEqual(self.run_ids(page), sorted(every, reverse=True))
        self.assertEqual(page.get_attribute("th[data-sort]:nth-child(3)", "aria-sort"), "none")
        self.assertEqual(self.problems, [])

    def test_sorting_a_text_column_keeps_filters_applied(self):
        page = self.open("/")
        page.fill(".toolbar input", "knn")
        page.click("th[data-sort]:nth-child(4)")  # Model
        shown = self.run_ids(page)
        self.assertEqual(len(shown), 6)
        self.assertEqual(page.inner_text(".toolbar span"), "6 of 7 runs shown")

    def test_a_column_can_be_sorted_from_the_keyboard(self):
        page = self.open("/")
        page.focus("th[data-sort]:nth-child(2)")
        page.keyboard.press("Enter")
        self.assertEqual(page.get_attribute("th[data-sort]:nth-child(2)", "aria-sort"), "ascending")
        page.keyboard.press("Space")
        self.assertEqual(
            page.get_attribute("th[data-sort]:nth-child(2)", "aria-sort"), "descending"
        )

    def test_choosing_a_filter_in_the_form_submits_it(self):
        page = self.open("/experiment/sweep")
        page.select_option("select[name=metric]", "f1")
        page.wait_for_url("**metric=f1**")
        self.assertIn("<h2>f1 by", page.content())

    def test_a_chart_dot_shows_its_value_under_the_pointer(self):
        page = self.open("/experiment/sweep?param=k&metric=accuracy")
        box = self.dot_position(page, "svg[data-kind=points] circle[data-readout]")
        page.mouse.move(box["x"], box["y"])
        shown = page.inner_text("svg[data-kind=points] + .readout")
        expected = page.get_attribute("svg[data-kind=points] circle.active", "data-readout")
        self.assertEqual(shown, expected)
        self.assertRegex(shown, r"^run-\d+: \d\.\d{4}$")
        page.mouse.move(2, 2)  # leave the chart
        self.assertEqual(
            page.inner_text("svg[data-kind=points] + .readout"),
            "Move over the chart for the value under the pointer.",
        )
        self.assertEqual(page.locator("circle.active").count(), 0)
        self.assertEqual(self.problems, [])

    def dot_position(self, page, selector, index=0):
        return page.evaluate(
            """([selector, index]) => {
              const dot = document.querySelectorAll(selector)[index];
              const svg = dot.ownerSVGElement;
              const p = svg.createSVGPoint();
              p.x = +dot.getAttribute('cx'); p.y = +dot.getAttribute('cy');
              const q = p.matrixTransform(svg.getScreenCTM());
              svg.scrollIntoView({block: 'center'});
              return {x: q.x, y: q.y};
            }""",
            [selector, index],
        )

    def test_parallel_lines_highlight_on_hover_and_open_the_run_on_click(self):
        page = self.open("/experiment/sweep")
        self.assertGreater(page.locator("svg[data-kind=parallel]").count(), 0)
        target = page.evaluate(
            """() => {
              const svg = document.querySelector('svg[data-kind=parallel]');
              const line = svg.querySelector('polyline[data-readout]');
              svg.scrollIntoView({block: 'center'});
              const pts = line.getAttribute('points').trim().split(/\\s+/)[0].split(',');
              const p = svg.createSVGPoint(); p.x = +pts[0]; p.y = +pts[1];
              const q = p.matrixTransform(svg.getScreenCTM());
              return {x: q.x, y: q.y, id: line.getAttribute('data-readout')};
            }"""
        )
        page.mouse.move(target["x"], target["y"])
        highlighted = page.get_attribute("svg[data-kind=parallel] .active", "data-readout")
        shown = page.inner_text("svg[data-kind=parallel] + .readout")
        self.assertEqual(shown, highlighted)
        self.assertTrue(page.evaluate("document.querySelector('svg.pc-focus') !== null"))
        page.mouse.click(target["x"], target["y"])
        page.wait_for_url(f"**/run/{highlighted}")
        self.assertIn(highlighted, page.inner_text("h1"))

    def test_a_chart_never_follows_a_link_that_is_not_a_run_page(self):
        page = self.open("/experiment/sweep")
        target = page.evaluate(
            """() => {
              const svg = document.querySelector('svg[data-kind=parallel]');
              svg.scrollIntoView({block: 'center'});
              for (const m of svg.querySelectorAll('[data-href]')) {
                m.setAttribute('data-href', '//evil.example/run/x');
              }
              const line = svg.querySelector('polyline[data-readout]');
              const pts = line.getAttribute('points').trim().split(/\\s+/)[0].split(',');
              const p = svg.createSVGPoint(); p.x = +pts[0]; p.y = +pts[1];
              const q = p.matrixTransform(svg.getScreenCTM());
              return {x: q.x, y: q.y};
            }"""
        )
        url = page.url
        page.mouse.click(target["x"], target["y"])
        page.wait_for_timeout(300)
        self.assertEqual(page.url, url)

    def test_the_confusion_matrix_toggles_between_counts_and_percent_of_row(self):
        page = self.open("/run/run-000001")
        before = page.eval_on_selector_all(
            "svg[data-kind=heatmap] text[data-count]", "t => t.map(x => x.textContent)"
        )
        counts = page.eval_on_selector_all(
            "svg[data-kind=heatmap] text[data-count]", "t => t.map(x => x.dataset.count)"
        )
        self.assertEqual(before, counts)
        page.click("button:text('show percent of row')")
        percent = page.eval_on_selector_all(
            "svg[data-kind=heatmap] text[data-count]", "t => t.map(x => x.textContent)"
        )
        self.assertTrue(all(p.endswith("%") for p in percent), percent)
        rows = [percent[i : i + 2] for i in range(0, len(percent), 2)]
        for row in rows:  # each row's shares add to about 100
            self.assertAlmostEqual(sum(int(p[:-1]) for p in row), 100, delta=1)
        page.click("button:text('show counts')")
        self.assertEqual(
            page.eval_on_selector_all(
                "svg[data-kind=heatmap] text[data-count]", "t => t.map(x => x.textContent)"
            ),
            counts,
        )

    def test_pages_work_with_scripts_turned_off(self):
        page = self.open("/", javascript=False)
        self.assertEqual(page.locator("table[data-runs] tbody tr").count(), 7)
        self.assertEqual(page.locator(".toolbar").count(), 0)
        page = self.open("/experiment/sweep", javascript=False)
        self.assertGreater(page.locator("svg[data-kind=parallel] polyline").count(), 0)

    def test_the_policy_blocks_an_injected_script(self):
        page = self.open("/")
        blocked = page.evaluate(
            """() => new Promise(resolve => {
              document.addEventListener(
                'securitypolicyviolation', e => resolve(e.violatedDirective));
              const s = document.createElement('script');
              s.textContent = 'window.injected = true';
              document.head.appendChild(s);
              setTimeout(() => resolve('not blocked'), 1000);
            })"""
        )
        self.assertTrue(blocked.startswith("script-src"), blocked)
        self.assertIsNone(page.evaluate("window.injected"))


if __name__ == "__main__":
    unittest.main()
