import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modelbench.config import ExperimentConfig
from modelbench.datasets import Table, make_version
from modelbench.errors import AnalysisError, CompareError
from modelbench.report import (
    MAX_PLOTS,
    _md,
    collect,
    render_html,
    render_markdown,
    write_report,
)
from modelbench.runner import run_experiment
from modelbench.store import Store
from modelbench.sweep import run_sweep

ENV = {"python": "3.12.1", "lib:numpy": None}
XSS = '<script>alert("x")</script>'


def line_table(n=40):
    return Table(("x", "y"), tuple((str(i), repr(2.0 * i + 1.0)) for i in range(n)))


def blobs():
    rows = [(f"{i * 0.1:.1f}", "a") for i in range(20)] + [
        (f"{10 + i * 0.1:.1f}", "b") for i in range(20)
    ]
    return Table(("x", "label"), tuple(rows))


def ridge_config(**overrides):
    fields = {
        "experiment_id": "ridge",
        "model": "linear_regression",
        "seed": 5,
        "dataset_id": "line",
        "metrics": ("r2", "mae"),
        "params": {},
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


class ReportTestCase(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)
        # run-000001..3: l2 = 0, 1e6, 100 (so the ranking by r2 is 1, 3, 2)
        run_sweep(
            self.store, ridge_config(), {"l2": [0.0, 1e6, 100.0]}, line_table(), "y",
            environment=ENV,
        )  # fmt: skip
        failed = self.store.create_run(ridge_config(), make_version("line", line_table(), "y"), ENV)
        self.store.fail_run(failed, "boom")  # run-000004


class CollectTests(ReportTestCase):
    def test_runs_are_ranked_best_first_and_the_best_is_found(self):
        data = collect(self.store, "ridge")
        self.assertEqual(data.rank_metric, "r2")  # the config's first metric
        self.assertTrue(data.higher_is_better)
        self.assertEqual([r.run_id for r in data.runs], ["run-000001", "run-000003", "run-000002"])
        self.assertEqual(data.metric_names, ("mae", "r2"))
        self.assertEqual(data.varying, ("l2",))
        self.assertEqual(data.failed, 1)
        self.assertEqual(data.best.run_id, "run-000001")
        self.assertEqual(data.best.seed, 5)
        self.assertTrue(data.best.dataset.startswith("line@"))
        self.assertTrue(data.best.dataset.endswith(", 40 rows"))
        self.assertEqual(data.best.environment, ENV)
        self.assertEqual(data.runs[0].params, {"l2": 0.0})
        self.assertEqual(data.runs[0].model, "linear_regression")

    def test_a_lower_is_better_metric_reverses_the_direction(self):
        data = collect(self.store, "ridge", "mae")
        self.assertFalse(data.higher_is_better)
        self.assertEqual(data.runs[0].run_id, "run-000001")
        self.assertEqual(data.runs[-1].run_id, "run-000002")

    def test_errors(self):
        with self.assertRaisesRegex(AnalysisError, "has no completed runs"):
            collect(self.store, "nope")
        with self.assertRaisesRegex(AnalysisError, "not a metric of these runs"):
            collect(self.store, "ridge", "accuracy")

    def test_runs_that_lack_the_metric_come_last(self):
        for extra in ({"a": 1}, {"a": 2}):
            run_id = self.store.create_run(
                ridge_config(experiment_id="mixed", params=extra),
                make_version("line", line_table(), "y"),
                ENV,
            )
            self.store.log_metrics(run_id, {"mae": 1.0, **({"r2": 0.5} if extra["a"] == 2 else {})})
            self.store.finish_run(run_id)
        data = collect(self.store, "mixed", "r2")
        self.assertEqual([r.run_id for r in data.runs], ["run-000006", "run-000005"])
        self.assertEqual(data.best.run_id, "run-000006")

    def test_a_metric_without_a_known_direction_has_no_best_run(self):
        with mock.patch("modelbench.report.higher_is_better", side_effect=CompareError("?")):
            data = collect(self.store, "ridge")
        self.assertIsNone(data.higher_is_better)
        self.assertIsNone(data.best)
        self.assertEqual([r.run_id for r in data.runs], ["run-000001", "run-000002", "run-000003"])
        self.assertNotIn("(best)", render_html(data))
        self.assertIn("no known direction", render_markdown(data))


class HtmlTests(ReportTestCase):
    def test_a_self_contained_page(self):
        html = render_html(collect(self.store, "ridge"))
        self.assertIn("<h1>Experiment ridge</h1>", html)
        self.assertIn("3 completed run(s) of linear_regression, 1 failed run(s) not shown.", html)
        self.assertIn("Ranked by r2 (higher is better).", html)
        self.assertEqual(html.count("(best)"), 1)
        self.assertNotIn("<script", html)
        self.assertNotIn('href="', html)  # no links back to a dashboard
        self.assertNotIn("http", html)
        self.assertEqual(html.count("<svg"), 2)  # the bar chart and the l2 plot
        self.assertIn("<h2>r2 by l2</h2>", html)
        self.assertIn("<h2>Best run: run-000001</h2>", html)
        self.assertIn("lib:numpy", html)  # the environment table
        rows = re.findall(r"<tr><td>(\d)</td><td>(run-\d+)", html)
        self.assertEqual(rows, [("1", "run-000001"), ("2", "run-000003"), ("3", "run-000002")])

    def test_the_same_runs_give_the_same_bytes(self):
        data = collect(self.store, "ridge")
        self.assertEqual(render_html(data), render_html(collect(self.store, "ridge")))
        self.assertEqual(render_markdown(data), render_markdown(collect(self.store, "ridge")))

    def test_a_classification_report_draws_the_confusion_matrix_and_escapes_notes(self):
        config = ExperimentConfig("knn", "knn", 7, "blobs", ("accuracy",), {"k": 3})
        result = run_experiment(self.store, config, blobs(), "label", environment=ENV)
        self.store.add_note(result.run_id, f"note {XSS}")
        html = render_html(collect(self.store, "knn"))
        self.assertIn("<h2>Confusion matrix</h2>", html)
        self.assertEqual(html.count("<svg"), 2)  # the bar chart and the heatmap
        self.assertNotIn(XSS, html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("<h2>Tags</h2>", html)
        self.assertNotIn("confusion_matrix", html)  # drawn, not dumped as a raw tag

    def test_at_most_four_parameters_are_plotted(self):
        for i in range(2):
            params = {f"p{n}": i for n in range(MAX_PLOTS + 1)}
            run_id = self.store.create_run(
                ridge_config(experiment_id="wide", params=params),
                make_version("line", line_table(), "y"),
                ENV,
            )
            self.store.log_metrics(run_id, {"r2": 0.1 * i})
            self.store.finish_run(run_id)
        data = collect(self.store, "wide")
        self.assertEqual(len(data.varying), MAX_PLOTS + 1)
        html = render_html(data)
        self.assertEqual(len(re.findall(r"<h2>r2 by p\d</h2>", html)), MAX_PLOTS)
        self.assertIn("Not plotted: p4.", html)

    def test_runs_without_the_rank_metric_come_last(self):
        for name, metrics in (
            ("a", {"mae": 9.0}),
            ("b", {"r2": -0.9, "mae": 1.0}),
            ("c", {"r2": -0.1}),
        ):
            run_id = self.store.create_run(
                ridge_config(experiment_id="gaps", params={"tag": name}),
                make_version("line", line_table(), "y"),
                ENV,
            )
            self.store.log_metrics(run_id, metrics)
            self.store.finish_run(run_id)
        data = collect(self.store, "gaps", "r2")
        self.assertEqual([r.params["tag"] for r in data.runs], ["c", "b", "a"])
        self.assertEqual(data.best.run_id, data.runs[0].run_id)
        self.assertIn("<td>3</td><td>run-", render_html(data))

    def test_only_the_best_twenty_bars_are_drawn(self):
        for i in range(22):
            run_id = self.store.create_run(
                ridge_config(experiment_id="many", params={"i": i}),
                make_version("line", line_table(), "y"),
                ENV,
            )
            self.store.log_metrics(run_id, {"r2": i / 100})
            self.store.finish_run(run_id)
        html = render_html(collect(self.store, "many"))
        self.assertIn("r2 by run (the best 20)", html)
        self.assertEqual(html.count('class="bar"'), 20)


class MarkdownTests(ReportTestCase):
    def test_tables_without_charts(self):
        lines = render_markdown(collect(self.store, "ridge")).splitlines()
        self.assertEqual(lines[0], "# Experiment ridge")
        self.assertEqual(lines[4], "| # | Run | Model | Params | mae | r2 |")
        self.assertEqual(lines[5], "|---|---|---|---|---|---|")
        first = lines[6].split(" | ")
        self.assertEqual(first[:4], ["| 1", "run-000001 (best)", "linear_regression", "l2=0.0"])
        self.assertIn("## Best run: run-000001", lines)
        self.assertIn("- seed: 5", lines)
        self.assertTrue(lines[-1].startswith("Made by ModelBench "))

    def test_cell_text_cannot_break_the_table(self):
        self.assertEqual(_md("a|b\nc\\d"), "a\\|b c\\\\d")

    def test_the_confusion_matrix_is_a_code_block(self):
        config = ExperimentConfig("knn", "knn", 7, "blobs", ("accuracy",), {"k": 3})
        run_experiment(self.store, config, blobs(), "label", environment=ENV)
        text = render_markdown(collect(self.store, "knn"))
        self.assertIn("```\ntrue/pred", text)


class WriteTests(ReportTestCase):
    def test_writes_either_format_and_refuses_others(self):
        with tempfile.TemporaryDirectory() as tmp:
            html_path, md_path = Path(tmp) / "r.html", Path(tmp) / "r.md"
            data = write_report(self.store, "ridge", html_path)
            write_report(self.store, "ridge", md_path, fmt_name="md", metric="mae")
            self.assertEqual(html_path.read_text(), render_html(data))
            self.assertTrue(md_path.read_text().startswith("# Experiment ridge"))
            self.assertIn("Ranked by mae (lower is better).", md_path.read_text())
            with self.assertRaisesRegex(AnalysisError, "format must be one of"):
                write_report(self.store, "ridge", Path(tmp) / "r.pdf", fmt_name="pdf")
            self.assertFalse((Path(tmp) / "r.pdf").exists())
            with self.assertRaises(AnalysisError):
                write_report(self.store, "nope", Path(tmp) / "x.html")
            self.assertFalse((Path(tmp) / "x.html").exists())


if __name__ == "__main__":
    unittest.main()
