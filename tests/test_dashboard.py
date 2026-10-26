import hashlib
import http.client
import io
import json
import math
import os
import re
import socket
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

from modelbench import dashboard
from modelbench.config import ExperimentConfig
from modelbench.dashboard import (
    bar_chart,
    handle,
    is_allowed_host,
    is_loopback_host,
    line_chart,
    make_server,
    scatter_chart,
    serve,
)
from modelbench.datasets import Table, make_version
from modelbench.errors import DashboardError, StoreError
from modelbench.runner import run_experiment
from modelbench.store import Store
from modelbench.sweep import run_sweep

ENV = {"python": "3.12.1", "lib:numpy": None}
XSS = '<script>alert("x")</script>'


def blobs():
    rows = [(f"{i * 0.1:.1f}", "a") for i in range(20)]
    rows += [(f"{10 + i * 0.1:.1f}", "b") for i in range(20)]
    return Table(("x", "label"), tuple(rows))


def config(**overrides):
    fields = {
        "experiment_id": "exp-a",
        "model": "knn",
        "seed": 7,
        "dataset_id": "blobs",
        "metrics": ("accuracy", "f1"),
        "params": {"k": 3},
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


class DashboardTestCase(unittest.TestCase):
    """Three completed runs, one failed, one live run with a multi-step metric."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Path(self._tmp.name) / "runs.db"
        table = blobs()
        with Store(self.db) as store:
            run_experiment(store, config(), table, "label", environment=ENV)  # run-000001
            run_experiment(
                store, config(model="majority", params={}), table, "label", environment=ENV
            )  # run-000002
            run_experiment(
                store,
                config(experiment_id="exp-b", params={"k": 5}),
                table,
                "label",
                environment={"python": "3.13.0", "lib:numpy": None},
            )  # run-000003 (fmt: skip)
            dataset = store.get_dataset("blobs", store.get_run("run-000001").dataset_version)
            failed = store.create_run(config(), dataset, ENV)  # run-000004
            store.fail_run(failed, f"boom {XSS}")
            live = store.create_run(config(experiment_id="exp-live"), dataset, ENV)  # run-000005
            for step, loss in enumerate((0.9, 0.5, 0.25)):
                store.log_metric(live, "loss", loss, step=step)
            store.add_note(live, f"note {XSS}\nsecond line")

    def get(self, target, host="localhost:8765", loopback=True):
        return handle(self.db, target, host, loopback)


class IndexTests(DashboardTestCase):
    def test_lists_runs_newest_first_with_links_and_metrics(self):
        response = self.get("/")
        self.assertEqual(response.status, 200)
        self.assertEqual(response.content_type, "text/html; charset=utf-8")
        body = response.body
        order = [body.index(f'href="/run/run-00000{n}"') for n in (5, 4, 3, 2, 1)]
        self.assertEqual(order, sorted(order))  # run-000005 appears first
        self.assertIn("accuracy=1.0000 f1=1.0000", body)
        self.assertIn("accuracy=0.3750 f1=0.2727", body)
        self.assertIn('class="good">completed', body)
        self.assertIn('class="bad">failed', body)
        self.assertIn('name="run" value="run-000001"', body)

    def test_filters(self):
        body = self.get("/?experiment=exp-b").body
        self.assertIn("run-000003", body)
        self.assertNotIn("run-000001", body)
        body = self.get("/?status=failed").body
        self.assertIn("run-000004", body)
        self.assertNotIn("run-000003", body)
        self.assertIn("No runs match", self.get("/?experiment=nothing").body)

    def test_limit(self):
        body = self.get("/?limit=2").body
        self.assertEqual(body.count('name="run" value='), 2)

    def test_bad_query_values_are_400_not_500(self):
        for target in ("/?status=weird", "/?limit=0", "/?limit=201", "/?limit=abc",
                       "/?limit=%D9%A1", "/?status=failed&status=completed"):  # fmt: skip
            with self.subTest(target=target):
                response = self.get(target)
                self.assertEqual(response.status, 400)
                self.assertIn("Bad request", response.body)

    def test_query_text_is_escaped(self):
        body = self.get("/?experiment=" + "%22%3E%3Cscript%3Ealert(1)%3C/script%3E").body
        self.assertNotIn("<script>alert(1)", body)
        self.assertIn("&quot;&gt;&lt;script&gt;", body)


class RunPageTests(DashboardTestCase):
    def test_completed_run(self):
        body = self.get("/run/run-000001").body
        self.assertIn("<h1>run-000001</h1>", body)
        expected = ["exp-a", "knn", "seed", "{&quot;k&quot;: 3}", "blobs@", "40 rows"]
        expected += ["accuracy", "1.0000", "predictions_sha256", "python", "3.12.1"]
        for text in expected:
            self.assertIn(text, body)
        self.assertNotIn("<polyline", body)  # one point per metric: no line chart
        self.assertIn('href="/api/run/run-000001"', body)

    def test_failed_run_shows_escaped_error(self):
        body = self.get("/run/run-000004").body
        self.assertIn("boom &lt;script&gt;", body)
        self.assertNotIn(XSS, body)

    def test_multi_step_metric_gets_a_line_chart_and_notes_are_escaped(self):
        body = self.get("/run/run-000005").body
        self.assertIn("<svg", body)
        self.assertIn("loss by step", body)
        self.assertIn("0.2500", body)
        self.assertIn("<td>2</td><td>3</td>", body)  # final step 2, three points
        self.assertIn("note &lt;script&gt;", body)
        self.assertNotIn("<script", body)
        self.assertIn("second line", body)

    def test_unknown_and_invalid_ids(self):
        self.assertEqual(self.get("/run/run-000099").status, 404)
        for target in ("/run/", "/run/..", "/run/run-1/extra", "/run/%00", "/run/%ff"):
            with self.subTest(target=target):
                self.assertIn(self.get(target).status, (400, 404))
        self.assertEqual(self.get("/run/%ff").status, 400)


class CompareTests(DashboardTestCase):
    def target(self, *ids, extra=""):
        return "/compare?" + "&".join(f"run={i}" for i in ids) + extra

    def test_html_compare_defaults_to_the_oldest_as_baseline(self):
        body = self.get(self.target("run-000002", "run-000001")).body
        self.assertIn("baseline: run-000001", body)
        ranking = "rank by <strong>accuracy</strong>: 1. run-000001 (1.0000), 2. run-000002"
        self.assertIn(ranking, body)
        self.assertEqual(body.count("<svg"), 2)  # accuracy and f1 bar charts
        self.assertIn('class="num bad">-0.6250 worse', body)
        self.assertIn("use as baseline instead", body)

    def test_baseline_parameter(self):
        ids = ("run-000001", "run-000002")
        body = self.get(self.target(*ids, extra="&baseline=run-000002")).body
        self.assertIn("baseline: run-000002", body)
        self.assertIn('class="num good">+0.6250 better', body)
        self.assertEqual(self.get(self.target(*ids, extra="&baseline=run-000003")).status, 400)

    def test_environment_and_parameter_differences_are_shown(self):
        body = self.get(self.target("run-000001", "run-000003")).body
        self.assertIn("environment python: 3.12.1 to 3.13.0", body)
        self.assertIn("<td>k</td><td>changed</td><td>3</td><td>5</td>", body)

    def test_bad_selections(self):
        self.assertEqual(self.get("/compare").status, 400)
        one = self.get(self.target("run-000001"))
        self.assertEqual(one.status, 400)
        self.assertIn("at least two", one.body)
        many = self.get(self.target(*["run-000001"] * 21))
        self.assertEqual(many.status, 400)
        self.assertIn("at most 20", many.body)
        self.assertEqual(self.get(self.target("run-000001", "run-000099")).status, 404)
        failed_run = self.get(self.target("run-000001", "run-000004"))
        self.assertEqual(failed_run.status, 400)

    def test_api_compare_keeps_the_given_order(self):
        target = self.target("run-000002", "run-000001").replace("/compare", "/api/compare")
        response = self.get(target)
        self.assertEqual(response.content_type, "application/json; charset=utf-8")
        self.assertEqual(json.loads(response.body)["baseline"], "run-000002")


class ApiTests(DashboardTestCase):
    def test_runs_and_run(self):
        self.assertEqual(
            json.loads(self.get("/api/runs").body),
            ["run-000005", "run-000004", "run-000003", "run-000002", "run-000001"],
        )
        self.assertEqual(json.loads(self.get("/api/runs?status=failed").body), ["run-000004"])
        run = json.loads(self.get("/api/run/run-000001").body)
        self.assertEqual(run["run_id"], "run-000001")
        self.assertIn("config", run)
        self.assertEqual(run["status"], "completed")

    def test_errors_are_json_on_api_paths(self):
        for target, status in (("/api/run/run-000099", 404), ("/api/runs?status=x", 400),
                               ("/api/compare", 400), ("/api/nothing", 404)):  # fmt: skip
            with self.subTest(target=target):
                response = self.get(target)
                self.assertEqual(response.status, status)
                self.assertEqual(response.content_type, "application/json; charset=utf-8")
                self.assertIn("error", json.loads(response.body))

    def test_unknown_page_is_404(self):
        response = self.get("/nope")
        self.assertEqual(response.status, 404)
        self.assertEqual(response.content_type, "text/html; charset=utf-8")


class RequestHandlingTests(DashboardTestCase):
    def test_host_allowlist(self):
        for host in ("localhost", "localhost:8765", "127.0.0.1:80", "[::1]:8765", "::1",
                     "LOCALHOST", None):  # fmt: skip
            with self.subTest(host=host):
                self.assertTrue(is_allowed_host(host, True))
        for host in ("evil.example", "evil.example:8765", "localhost.evil.example",
                     "127.0.0.1.evil.example", "", "[evil]:80", "0.0.0.0"):  # fmt: skip
            with self.subTest(host=host):
                self.assertFalse(is_allowed_host(host, True))
        self.assertTrue(is_allowed_host("anything.example", False))

    def test_foreign_host_is_forbidden_before_any_database_access(self):
        response = self.get("/", host="evil.example")
        self.assertEqual(response.status, 403)
        self.assertNotIn("run-000001", response.body)
        self.assertEqual(self.get("/api/runs", host="evil.example").status, 403)
        self.assertEqual(self.get("/", host="evil.example", loopback=False).status, 200)

    def test_malformed_and_oversized_targets(self):
        self.assertEqual(self.get("/" + "a" * 2001).status, 400)
        many = "/?" + "&".join(f"k{i}=1" for i in range(101))
        self.assertEqual(self.get(many).status, 400)
        self.assertEqual(self.get("/%ff%fe").status, 400)

    def test_internal_errors_do_not_leak_details(self):
        with (
            mock.patch("modelbench.dashboard.route", side_effect=RuntimeError("secret path /etc")),
            mock.patch("sys.stderr", new_callable=io.StringIO) as err,
        ):
            response = self.get("/")
        self.assertEqual(response.status, 500)
        self.assertNotIn("secret", response.body)
        self.assertIn("RuntimeError", err.getvalue())

    def test_missing_database_is_404_not_created(self):
        missing = Path(self._tmp.name) / "none.db"
        self.assertEqual(handle(missing, "/", "localhost", True).status, 404)
        self.assertFalse(missing.exists())


class ChartTests(unittest.TestCase):
    def test_bar_chart_scales_and_escapes(self):
        svg = bar_chart(XSS, [("run-1", 1.0), ("<b>", 0.5)])
        self.assertNotIn("<script>", svg)
        self.assertNotIn("<b>", svg)
        self.assertIn("&lt;b&gt;", svg)
        widths = [float(w) for w in re.findall(r'class="bar"[^>]* width="([0-9.]+)"', svg)]
        self.assertAlmostEqual(widths[0], 2 * widths[1], places=0)
        self.assertIn("1.0000", svg)
        lefts = [float(x) for x in re.findall(r'class="bar" x="(-?[0-9.]+)"', svg)]
        self.assertEqual(len(lefts), 2)
        for left, width in zip(lefts, widths, strict=True):
            self.assertGreaterEqual(left, 120)  # inside the plot area, not off the canvas
            self.assertLessEqual(left + width, 560 - 80 + 0.5)

    def test_bar_chart_negative_and_zero_values(self):
        svg = bar_chart("r2", [("a", -0.5), ("b", 0.5)])
        xs = [float(x) for x in re.findall(r'class="bar" x="([0-9.]+)"', svg)]
        zero = float(re.search(r'class="zero" x1="([0-9.]+)"', svg).group(1))
        self.assertLess(xs[0], zero)
        self.assertAlmostEqual(xs[1], zero, places=0)
        flat = bar_chart("zeros", [("a", 0.0), ("b", 0.0)])
        self.assertIn("<svg", flat)
        self.assertNotIn("nan", flat.lower())

    def test_bar_chart_empty(self):
        self.assertEqual(bar_chart("none", []), "")

    def test_line_chart(self):
        self.assertEqual(line_chart("t", [(0, 1.0)]), "")
        svg = line_chart("loss", [(0, 0.9), (1, 0.5), (2, 0.25)])
        self.assertEqual(svg.count("<circle"), 3)
        self.assertIn("step 0", svg)
        self.assertIn("step 2", svg)
        coords = [float(v) for v in re.findall(r'cy="([0-9.]+)"', svg)]
        self.assertTrue(coords[0] < coords[1] < coords[2])  # falling loss is drawn falling
        flat = line_chart("flat", [(0, 1.0), (5, 1.0)])
        self.assertNotIn("nan", flat.lower())


class ServerTests(DashboardTestCase):
    def start(self, **kwargs):
        server = make_server(self.db, "127.0.0.1", 0, **kwargs)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop():
            server.shutdown()
            server.server_close()
            thread.join(5)

        self.addCleanup(stop)
        return server.server_address[1]

    def request(self, port, method, path, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.addCleanup(conn.close)
        conn.request(method, path, headers=headers or {})
        response = conn.getresponse()
        return response, response.read()

    def test_get_serves_pages_with_security_headers(self):
        port = self.start()
        response, body = self.request(port, "GET", "/")
        self.assertEqual(response.status, 200)
        self.assertIn(b"run-000001", body)
        self.assertEqual(response.getheader("Content-Length"), str(len(body)))
        self.assertIn("default-src 'none'", response.getheader("Content-Security-Policy"))
        self.assertNotIn("script-src", response.getheader("Content-Security-Policy"))
        self.assertEqual(response.getheader("X-Content-Type-Options"), "nosniff")
        self.assertEqual(response.getheader("X-Frame-Options"), "DENY")
        self.assertEqual(response.getheader("Cache-Control"), "no-store")
        self.assertNotIn("Python", response.getheader("Server"))

    def test_head_has_headers_but_no_body(self):
        port = self.start()
        get, body = self.request(port, "GET", "/run/run-000001")
        head, empty = self.request(port, "HEAD", "/run/run-000001")
        self.assertEqual(empty, b"")
        self.assertEqual(head.getheader("Content-Length"), str(len(body)))

    def test_head_sends_no_bytes_after_the_headers(self):
        port = self.start()
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            sock.sendall(b"HEAD / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            data = b""
            while chunk := sock.recv(65536):
                data += chunk
        head, _, rest = data.partition(b"\r\n\r\n")
        self.assertTrue(head.startswith(b"HTTP/1.0 200") or head.startswith(b"HTTP/1.1 200"))
        self.assertEqual(rest, b"")

    def test_server_does_not_wait_for_open_connections_on_exit(self):
        self.assertTrue(dashboard._Server.daemon_threads)

    def test_write_methods_are_refused(self):
        port = self.start()
        for method in ("POST", "PUT", "DELETE", "PATCH"):
            with self.subTest(method=method):
                response, body = self.request(port, method, "/", {"Content-Length": "0"})
                self.assertEqual(response.status, 405)
                self.assertEqual(response.getheader("Allow"), "GET, HEAD")
        response, body = self.request(port, "POST", "/api/runs", {"Content-Length": "0"})
        self.assertEqual(response.getheader("Content-Type"), "application/json; charset=utf-8")

    def test_foreign_host_header_is_forbidden_over_the_wire(self):
        port = self.start()
        response, body = self.request(port, "GET", "/", {"Host": "evil.example"})
        self.assertEqual(response.status, 403)
        self.assertNotIn(b"run-000001", body)
        ok, _ = self.request(port, "GET", "/", {"Host": f"localhost:{port}"})
        self.assertEqual(ok.status, 200)

    def test_serving_never_changes_the_database_file(self):
        before = hashlib.sha256(self.db.read_bytes()).hexdigest()
        port = self.start()
        for path in ("/", "/run/run-000001", "/run/run-000005", "/api/runs",
                     "/compare?run=run-000001&run=run-000002"):  # fmt: skip
            self.request(port, "GET", path)
        self.assertEqual(hashlib.sha256(self.db.read_bytes()).hexdigest(), before)
        self.assertEqual([p.name for p in self.db.parent.iterdir()], ["runs.db"])

    def test_refuses_non_loopback_without_opt_in(self):
        for host in ("127.0.0.1", "localhost", "::1", "127.0.0.5"):
            self.assertTrue(is_loopback_host(host), host)
        for host in ("0.0.0.0", "192.168.1.5", "example.com", ""):
            self.assertFalse(is_loopback_host(host), host)
        with self.assertRaisesRegex(DashboardError, "no login"):
            make_server(self.db, "0.0.0.0", 0)
        server = make_server(self.db, "0.0.0.0", 0, allow_network=True)
        self.addCleanup(server.server_close)
        self.assertFalse(server.loopback_only)

    def test_bad_arguments(self):
        for port in (-1, 65536, "80", True, None):
            with self.subTest(port=port), self.assertRaises(DashboardError):
                make_server(self.db, "127.0.0.1", port)
        with self.assertRaises(StoreError):
            make_server(Path(self._tmp.name) / "none.db", "127.0.0.1", 0)
        busy = make_server(self.db, "127.0.0.1", 0)
        self.addCleanup(busy.server_close)
        with self.assertRaisesRegex(DashboardError, "cannot listen"):
            make_server(self.db, "127.0.0.1", busy.server_address[1])

    def test_the_port_is_claimed_exclusively_on_windows_only(self):
        self.assertIs(dashboard.reuse_address_is_safe("nt"), False)
        for system in ("posix", "java", ""):
            self.assertIs(dashboard.reuse_address_is_safe(system), True)
        self.assertEqual(
            dashboard._Server.allow_reuse_address, dashboard.reuse_address_is_safe(os.name)
        )
        for system, expected in (("nt", True), ("posix", False)):
            calls = []
            real = socket.socket.setsockopt

            def spy(sock, level, option, value, *rest, calls=calls, real=real):
                calls.append((level, option, value))
                if option != 4242:  # the stand-in for SO_EXCLUSIVEADDRUSE is not a real option
                    real(sock, level, option, value, *rest)

            with (
                mock.patch.object(dashboard, "os", types.SimpleNamespace(name=system)),
                mock.patch.object(socket, "SO_EXCLUSIVEADDRUSE", 4242, create=True),
                mock.patch.object(socket.socket, "setsockopt", spy),
            ):
                server = make_server(self.db, "127.0.0.1", 0)
            server.server_close()
            self.assertEqual((socket.SOL_SOCKET, 4242, 1) in calls, expected, system)

    def test_serve_announces_and_closes_on_interrupt(self):
        out = io.StringIO()
        created = []
        real_make = dashboard.make_server

        def recording_make(*args, **kwargs):
            created.append(real_make(*args, **kwargs))
            return created[-1]

        with (
            mock.patch.object(dashboard._Server, "serve_forever", side_effect=KeyboardInterrupt),
            mock.patch("modelbench.dashboard.make_server", recording_make),
        ):
            serve(self.db, "127.0.0.1", 0, out=out)
        self.assertIn("serving", out.getvalue())
        self.assertRegex(out.getvalue(), r"http://127\.0\.0\.1:\d+/")
        self.assertEqual(created[0].socket.fileno(), -1)  # the listening socket was closed


class ScatterChartTests(unittest.TestCase):
    def circles(self, svg):
        return [
            (float(x), float(y))
            for x, y in re.findall(r'<circle class="dot[^"]*" cx="([0-9.]+)" cy="([0-9.]+)"', svg)
        ]

    def test_numeric_axis_puts_the_extremes_on_the_plot_edges(self):
        svg = scatter_chart("t", [(1, 0.0, "a"), (3, 1.0, "b")], "k")
        # plot area: x from 56 to 544, y from 176 (bottom) up to 12 (top)
        self.assertEqual(self.circles(svg), [(56.0, 176.0), (544.0, 12.0)])
        self.assertIn(">0.0000<", svg)
        self.assertIn(">1.0000<", svg)

    def test_the_last_label_is_pulled_back_inside_the_chart(self):
        svg = scatter_chart("t", [(1, 0.0, "a"), (3, 1.0, "b")], "k")
        self.assertIn('x="544.0" y="192" text-anchor="end">3</text>', svg)
        self.assertIn('x="56.0" y="192" text-anchor="middle">1</text>', svg)

    def test_every_distinct_value_gets_a_tick_label(self):
        svg = scatter_chart("t", [(1, 0.0, "a"), (2, 1.0, "b"), (2, 0.5, "c"), (4, 0.2, "d")], "k")
        ticks = re.findall(r'text-anchor="(?:middle|end)">([0-9.e-]+)</text>', svg)
        self.assertEqual([t for t in ticks if "." not in t], ["1", "2", "4"])

    def test_a_wide_range_with_a_zero_uses_a_shifted_log_scale(self):
        svg = scatter_chart("t", [(0, 0.1, "a"), (1, 0.2, "b"), (1000, 0.3, "c")], "l2")
        xs = [x for x, _ in self.circles(svg)]
        expected = 56 + math.log10(2) / math.log10(1001) * 488
        self.assertEqual(xs[0], 56.0)
        self.assertAlmostEqual(xs[1], expected, places=1)
        self.assertEqual(xs[2], 544.0)
        self.assertIn("l2 (log scale)", svg)

    def test_a_narrow_range_stays_linear(self):
        svg = scatter_chart("t", [(1, 0.1, "a"), (10, 0.2, "b"), (100, 0.3, "c")], "k")
        self.assertNotIn("log scale", svg)
        self.assertAlmostEqual(self.circles(svg)[1][0], 56 + 9 / 99 * 488, places=1)

    def test_non_numbers_get_one_slot_each_in_sorted_order(self):
        svg = scatter_chart("t", [("b", 0.1, "x"), ("a", 0.2, "y")], "weights")
        self.assertEqual([x for x, _ in self.circles(svg)], [422.0, 178.0])
        mixed = scatter_chart("t", [(True, 0.1, "x"), (3, 0.2, "y")], "p")  # a bool is not a number
        self.assertIn(">True<", mixed)
        self.assertEqual(len(self.circles(mixed)), 2)

    def test_degenerate_inputs_are_centred_and_never_nan(self):
        self.assertEqual(scatter_chart("t", [], "k"), "")
        one = scatter_chart("t", [(5, 0.5, "a")], "k")
        self.assertEqual(self.circles(one), [(300.0, 94.0)])
        flat = scatter_chart("t", [(1, 0.5, "a"), (2, 0.5, "b")], "k")
        self.assertEqual({y for _, y in self.circles(flat)}, {94.0})
        for svg in (one, flat):
            self.assertNotIn("nan", svg.lower())

    def test_infinite_x_values_fall_back_to_categories(self):
        svg = scatter_chart("t", [(float("inf"), 0.1, "a"), (1.0, 0.2, "b")], "p")
        self.assertNotIn("nan", svg.lower())
        self.assertNotIn("inf</text>x", svg)
        self.assertEqual(len(self.circles(svg)), 2)

    def test_only_the_best_dot_is_highlighted_and_text_is_escaped(self):
        svg = scatter_chart(XSS, [(1, 0.1, XSS), (2, 0.9, "ok")], XSS, best=1)
        self.assertEqual(svg.count('class="dot best"'), 1)
        self.assertLess(svg.index('class="dot"'), svg.index('class="dot best"'))
        self.assertNotIn("<script>", svg)
        self.assertIn("&lt;script&gt;", svg)


class ExperimentPageTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Path(self._tmp.name) / "sweep.db"
        line = Table(("x", "y"), tuple((str(i), repr(2.0 * i + 1.0)) for i in range(40)))
        with Store(self.db) as store:
            ridge = config(
                experiment_id="ridge", model="linear_regression", dataset_id="line", params={},
                metrics=("r2", "mae"),
            )  # fmt: skip
            run_sweep(store, ridge, {"l2": [0.0, 1e6, 100.0]}, line, "y", environment=ENV)
            knn = config(experiment_id="knn-w", params={"k": 3})
            run_sweep(
                store, knn, {"weights": ["uniform", "distance"]}, blobs(), "label", environment=ENV
            )
            lone = config(experiment_id="lone", params={"k": 3})
            run_experiment(store, lone, blobs(), "label", environment=ENV)

    def get(self, target):
        return handle(self.db, target, "localhost:8765", True)

    def test_the_best_run_is_highlighted_and_runs_are_listed_best_first(self):
        response = self.get("/experiment/ridge")
        self.assertEqual(response.status, 200)
        body = response.body
        self.assertIn("3 completed run(s)", body)
        self.assertIn("<h2>r2 by l2</h2>", body)  # the config's first metric, the first parameter
        self.assertEqual(body.count('class="dot best"'), 1)
        self.assertRegex(body, r'class="dot best"[^>]*><title>run-000001: 1\.0000</title>')
        listed = re.findall(r'<td><a href="/run/(run-\d+)">', body)
        self.assertEqual(listed, ["run-000001", "run-000003", "run-000002"])  # l2 = 0, 100, 1e6

    def test_lower_is_better_metrics_list_the_smallest_first(self):
        body = self.get("/experiment/ridge?param=l2&metric=mae").body
        self.assertIn("<h2>mae by l2</h2>", body)
        self.assertRegex(body, r'class="dot best"[^>]*><title>run-000001: 0\.0000</title>')
        listed = re.findall(r'<td><a href="/run/(run-\d+)">', body)
        self.assertEqual(listed, ["run-000001", "run-000003", "run-000002"])

    def test_a_text_parameter_is_plotted_by_category(self):
        body = self.get("/experiment/knn-w").body
        self.assertIn("<h2>accuracy by weights</h2>", body)
        self.assertIn(">distance<", body)
        self.assertIn(">uniform<", body)
        self.assertIn('<option value="weights" selected>', body)

    def test_the_form_offers_only_varying_parameters_and_recorded_metrics(self):
        body = self.get("/experiment/ridge").body
        self.assertIn('<option value="l2" selected>l2</option>', body)
        self.assertIn('<option value="mae">mae</option>', body)
        self.assertNotIn("fold/", body)

    def test_nothing_varies_in_a_single_run_experiment(self):
        response = self.get("/experiment/lone")
        self.assertEqual(response.status, 200)
        self.assertIn("nothing to plot", response.body)
        self.assertNotIn('<svg class="chart"', response.body)
        self.assertIn("run-000006", response.body)  # still listed

    def test_unknown_experiment_and_bad_choices(self):
        none = self.get("/experiment/nope")
        self.assertEqual(none.status, 200)
        self.assertIn("No completed runs.", none.body)
        bad = self.get("/experiment/ridge?param=zzz")
        self.assertEqual(bad.status, 400)
        self.assertIn("param must be one of: l2", bad.body)
        self.assertEqual(self.get("/experiment/ridge?metric=zzz").status, 400)
        self.assertEqual(self.get("/experiment/ridge?metric=r2&metric=mae").status, 400)

    def test_the_experiment_name_in_the_url_is_escaped(self):
        body = self.get("/experiment/%3Cscript%3Ealert(1)%3C/script%3E").body
        self.assertNotIn("<script>alert(1)", body)
        self.assertIn("&lt;script&gt;", body)

    def test_failed_runs_are_left_out(self):
        with Store(self.db) as store:
            dataset = store.get_dataset("blobs", store.get_run("run-000006").dataset_version)
            failed = store.create_run(config(experiment_id="lone"), dataset, ENV)
            store.fail_run(failed, "boom")
        self.assertIn("1 completed run(s)", self.get("/experiment/lone").body)

    def test_the_index_and_run_pages_link_to_the_experiment(self):
        self.assertIn('<a href="/experiment/ridge">ridge</a>', self.get("/").body)
        self.assertIn('<a href="/experiment/ridge">ridge</a>', self.get("/run/run-000001").body)

    def test_the_page_carries_no_script(self):
        body = self.get("/experiment/ridge").body
        self.assertNotIn("<script", body)


class RegressionRunsTests(unittest.TestCase):
    def test_every_page_and_api_works_for_regression_runs(self):
        table = Table(("x", "y"), tuple((str(i), repr(2.0 * i + 1.0)) for i in range(30)))
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "reg.db"
            with Store(db) as store:
                for model in ("linear_regression", "mean"):
                    run_experiment(
                        store,
                        config(model=model, params={}, metrics=("mae", "rmse", "r2"),
                               dataset_id="line"),
                        table, "y", environment=ENV,
                    )  # fmt: skip
            host = "127.0.0.1:8765"
            for target in ("/", "/run/run-000001", "/compare?run=run-000001&run=run-000002",
                           "/api/runs", "/api/run/run-000002",
                           "/api/compare?run=run-000001&run=run-000002"):  # fmt: skip
                response = handle(db, target, host, True)
                self.assertEqual(response.status, 200, target)
            page = handle(db, "/compare?run=run-000001&run=run-000002", host, True).body
            for name in ("mae", "rmse", "r2"):
                self.assertIn(name, page)
            api = handle(db, "/api/compare?run=run-000001&run=run-000002", host, True)
            data = json.loads(api.body)
            self.assertEqual(data["baseline"], "run-000001")
            self.assertAlmostEqual(data["values"]["run-000001"]["r2"], 1.0, places=8)
            self.assertLess(data["values"]["run-000002"]["r2"], 0.2)


class ConfusionHeatmapTests(unittest.TestCase):
    def cm(self):
        from modelbench.metrics import ConfusionMatrix

        return ConfusionMatrix(("a", "b"), ((3, 1), (0, 4)))

    def test_one_cell_per_entry_with_counts_and_row_shares(self):
        svg = dashboard.confusion_heatmap(self.cm())
        self.assertEqual(svg.count('class="cell"'), 4)
        self.assertIn('viewBox="0 0 210 156"', svg)  # 110 + 2 * 46 + 8 wide, 56 + 2 * 46 + 8 high
        for count in ("3", "1", "0", "4"):
            self.assertIn('class="count"', svg)
            self.assertRegex(svg, rf'text-anchor="middle">{count}</text>')
        self.assertIn("true a, predicted a: 3 (75% of the row)", svg)
        self.assertIn("true b, predicted b: 4 (100% of the row)", svg)
        self.assertIn('fill-opacity="0.080"', svg)  # the empty cell is faint, not invisible

    def test_shade_follows_the_row_share(self):
        svg = dashboard.confusion_heatmap(self.cm())
        # 75% of the row -> 0.08 + 0.92 * 0.75 = 0.77; 100% -> 1.0
        self.assertIn('fill-opacity="0.770"', svg)
        self.assertIn('fill-opacity="1.000"', svg)

    def test_labels_are_escaped_and_long_ones_shortened(self):
        from modelbench.metrics import ConfusionMatrix

        hostile = XSS + "x" * 30
        svg = dashboard.confusion_heatmap(ConfusionMatrix((hostile, "b"), ((1, 0), (0, 1))))
        self.assertNotIn("<script>", svg)
        self.assertIn("&lt;script&gt;", svg)
        self.assertIn("...", svg)  # shortened on the axis, whole in the tooltip

    def test_run_page_draws_it_and_hides_the_raw_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "m.db"
            with Store(db) as store:
                run_experiment(store, config(), blobs(), "label", environment=ENV)
            page = handle(db, "/run/run-000001", "127.0.0.1:8765", True).body
        self.assertIn("<h2>Confusion matrix</h2>", page)
        self.assertEqual(page.count('class="cell"'), 4)
        self.assertNotIn("confusion_matrix</th>", page)  # shown as a picture, not a JSON tag

    def test_a_damaged_tag_is_reported_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "m.db"
            with Store(db) as store:
                run_id = store.create_run(config(), make_version("blobs", blobs(), "label"), ENV)
                store.set_tag(run_id, "confusion_matrix", "{broken")
                store.finish_run(run_id)
            response = handle(db, f"/run/{run_id}", "127.0.0.1:8765", True)
        self.assertEqual(response.status, 200)
        self.assertIn('class="bad"', response.body)
        self.assertIn("not valid JSON", response.body)


if __name__ == "__main__":
    unittest.main()
