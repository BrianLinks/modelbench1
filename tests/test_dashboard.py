import hashlib
import http.client
import io
import json
import re
import socket
import tempfile
import threading
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
    serve,
)
from modelbench.datasets import Table
from modelbench.errors import DashboardError, StoreError
from modelbench.runner import run_experiment
from modelbench.store import Store

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
                store, config(experiment_id="exp-b", params={"k": 5}), table, "label",
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
        self.assertNotIn("<svg", body)  # one point per metric: no line chart
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
        with mock.patch("modelbench.dashboard.route", side_effect=RuntimeError("secret path /etc")):
            with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
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
        for left, width in zip(lefts, widths):
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

    def test_serve_announces_and_closes_on_interrupt(self):
        out = io.StringIO()
        created = []
        real_make = dashboard.make_server

        def recording_make(*args, **kwargs):
            created.append(real_make(*args, **kwargs))
            return created[-1]

        with mock.patch.object(dashboard._Server, "serve_forever", side_effect=KeyboardInterrupt):
            with mock.patch("modelbench.dashboard.make_server", recording_make):
                serve(self.db, "127.0.0.1", 0, out=out)
        self.assertIn("serving", out.getvalue())
        self.assertRegex(out.getvalue(), r"http://127\.0\.0\.1:\d+/")
        self.assertEqual(created[0].socket.fileno(), -1)  # the listening socket was closed


if __name__ == "__main__":
    unittest.main()
