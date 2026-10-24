"""A read-only local dashboard served with the standard library.

Pages are rendered on the server as plain HTML with inline SVG charts and no script. The
database is opened read-only for every request, so the dashboard cannot change a run. Safety
choices:

- It binds to the loopback interface unless told otherwise, and while bound there it rejects
  requests whose Host header names anything but a loopback name (a DNS rebinding defence).
- Every value taken from the database or the URL is HTML-escaped, and a strict
  Content-Security-Policy forbids scripts, remote resources, and framing.
- Only GET and HEAD are accepted.
"""

from __future__ import annotations

import html
import ipaddress
import json
import os
import socket
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import parse_qs, unquote, urlencode, urlsplit

from .errors import DashboardError, ModelBenchError, StoreError
from .export import export_runs
from .run_comparison import RunComparison, compare_runs
from .store import STATUSES, Store

DEFAULT_PORT = 8765
LOOPBACK_NAMES = ("localhost", "127.0.0.1", "::1")
MAX_LIST = 200
DEFAULT_LIST = 50
MAX_COMPARE = 20
MAX_TARGET_LENGTH = 2000
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; "
        "form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


class NotFound(DashboardError):
    """The path does not name a page."""


@dataclass(frozen=True)
class Response:
    status: int
    content_type: str
    body: str


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: float) -> str:
    return f"{value:.4f}"


# -- charts ---------------------------------------------------------------------------------


def bar_chart(title: str, items: Sequence[tuple[str, float]], width: int = 560) -> str:
    """Horizontal bars, one per (label, value). Negative values extend left of the zero line."""
    if not items:
        return ""
    label_w, value_w, row_h = 120, 80, 26
    area = width - label_w - value_w
    lo = min(0.0, min(v for _, v in items))
    hi = max(0.0, max(v for _, v in items))
    span = (hi - lo) or 1.0

    def x(v: float) -> float:
        return label_w + (v - lo) / span * area

    height = row_h * len(items) + 8
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="{width}" role="img" '
        f'aria-label="{esc(title)}"><title>{esc(title)}</title>'
    ]
    zero = x(0.0)
    for i, (label, value) in enumerate(items):
        y = 4 + i * row_h
        left, right = sorted((zero, x(value)))
        parts.append(
            f'<text class="axis" x="{label_w - 8}" y="{y + 15}" text-anchor="end">'
            f"{esc(label)}</text>"
            f'<rect class="bar" x="{left:.1f}" y="{y + 3}" width="{max(right - left, 1):.1f}" '
            f'height="{row_h - 8}"><title>{esc(label)}: {fmt(value)}</title></rect>'
            f'<text class="axis" x="{label_w + area + 8}" y="{y + 15}">{fmt(value)}</text>'
        )
    parts.append(
        f'<line class="zero" x1="{zero:.1f}" y1="2" x2="{zero:.1f}" y2="{height - 2}"/></svg>'
    )
    return "".join(parts)


def line_chart(title: str, points: Sequence[tuple[int, float]], width: int = 560) -> str:
    """A value-over-step line with the first and last values and the range labelled."""
    if len(points) < 2:
        return ""
    height, left, right, top, bottom = 170, 56, 16, 12, 28
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    x_span = (x_hi - x_lo) or 1
    y_span = (y_hi - y_lo) or 1.0

    def px(step: int) -> float:
        return left + (step - x_lo) / x_span * (width - left - right)

    def py(value: float) -> float:
        return top + (1 - (value - y_lo) / y_span) * (height - top - bottom)

    path = " ".join(f"{px(s):.1f},{py(v):.1f}" for s, v in points)
    dots = "".join(
        f'<circle class="dot" cx="{px(s):.1f}" cy="{py(v):.1f}" r="3">'
        f"<title>step {s}: {fmt(v)}</title></circle>"
        for s, v in points
    )
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="{width}" role="img" '
        f'aria-label="{esc(title)}"><title>{esc(title)}</title>'
        f'<line class="zero" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>'
        f'<line class="zero" x1="{left}" y1="{height - bottom}" x2="{width - right}" '
        f'y2="{height - bottom}"/>'
        f'<polyline class="line" points="{path}"/>{dots}'
        f'<text class="axis" x="{left - 6}" y="{top + 9}" text-anchor="end">{fmt(y_hi)}</text>'
        f'<text class="axis" x="{left - 6}" y="{height - bottom}" text-anchor="end">'
        f"{fmt(y_lo)}</text>"
        f'<text class="axis" x="{left}" y="{height - 8}">step {x_lo}</text>'
        f'<text class="axis" x="{width - right}" y="{height - 8}" text-anchor="end">'
        f"step {x_hi}</text></svg>"
    )


# -- pages ----------------------------------------------------------------------------------

_CSS = """
:root{--bg:#fff;--fg:#1d2430;--muted:#5c6675;--line:#d9dee6;--accent:#1f5fbf;--bar:#3b7dd8;
--bad:#b3261e;--ok:#1b6e3c;--card:#f6f8fb}
@media (prefers-color-scheme:dark){:root{--bg:#14181f;--fg:#e6e9ef;--muted:#9aa4b2;
--line:#2c3441;--accent:#7aa8ee;--bar:#5b94e6;--bad:#f0817a;--ok:#6fcf97;--card:#1b212b}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:16px}
nav{border-bottom:1px solid var(--line);padding:10px 16px}
nav a{margin-right:16px}a{color:var(--accent)}
h1{font-size:1.4rem;margin:.6em 0}h2{font-size:1.1rem;margin:1.4em 0 .4em}
table{border-collapse:collapse;width:100%;display:block;overflow-x:auto}
th,td{text-align:left;padding:4px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
td.num{font-variant-numeric:tabular-nums}
table+table,p+table{margin-top:1em}form{margin:.6em 0 1em}
.muted{color:var(--muted)}.good{color:var(--ok)}.bad{color:var(--bad)}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 12px}
.chart{max-width:100%;height:auto}.bar,.dot{fill:var(--bar)}
.line{fill:none;stroke:var(--bar);stroke-width:2}.zero{stroke:var(--muted);stroke-width:1}
.axis{fill:var(--muted);font-size:12px}
pre{white-space:pre-wrap;word-break:break-word;margin:0}
"""


def page(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{esc(title)} - ModelBench</title><style>{_CSS}</style></head><body>"
        '<nav><a href="/">ModelBench</a><span class="muted">read-only dashboard</span></nav>'
        f"<main>{body}</main></body></html>"
    )


def _query_one(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    if len(values) > 1:
        raise DashboardError(f"{key!r} may be given only once")
    return values[0] or None


def _limit(query: dict[str, list[str]]) -> int:
    raw = _query_one(query, "limit")
    if raw is None:
        return DEFAULT_LIST
    if not raw.isascii() or not raw.isdigit() or not 1 <= int(raw) <= MAX_LIST:
        raise DashboardError(f"limit must be a whole number from 1 to {MAX_LIST}")
    return int(raw)


def _status(query: dict[str, list[str]]) -> str | None:
    status = _query_one(query, "status")
    if status is not None and status not in STATUSES:
        raise DashboardError(f"status must be one of {', '.join(STATUSES)}")
    return status


def _status_class(status: str) -> str:
    return {"completed": "good", "failed": "bad"}.get(status, "muted")


def render_index(store: Store, query: dict[str, list[str]]) -> str:
    experiment = _query_one(query, "experiment")
    status = _status(query)
    runs = store.list_runs(experiment, status, _limit(query))
    options = "".join(
        f'<option value="{s}"{" selected" if s == status else ""}>{s}</option>' for s in STATUSES
    )
    form = (
        '<form method="get" action="/"><label>experiment '
        f'<input name="experiment" value="{esc(experiment or "")}"></label> '
        f'<label>status <select name="status"><option value="">any</option>{options}</select>'
        '</label> <button>Filter</button></form>'
    )
    if not runs:
        return page("Runs", f"<h1>Runs</h1>{form}<p>No runs match.</p>")
    rows = []
    for run in runs:
        model = store.get_config(run.run_id).model
        metrics = " ".join(
            f"{esc(k)}={fmt(v)}" for k, v in sorted(store.final_metrics(run.run_id).items())
        )
        rows.append(
            f'<tr><td><input type="checkbox" name="run" value="{esc(run.run_id)}" '
            f'aria-label="compare {esc(run.run_id)}"></td>'
            f'<td><a href="/run/{esc(run.run_id)}">{esc(run.run_id)}</a></td>'
            f"<td>{esc(run.experiment_id)}</td><td>{esc(model)}</td>"
            f'<td class="{_status_class(run.status)}">{esc(run.status)}</td>'
            f"<td>{esc(run.started_at)}</td><td>{metrics}</td></tr>"
        )
    table = (
        '<form method="get" action="/compare"><table><thead><tr><th></th><th>Run</th>'
        "<th>Experiment</th><th>Model</th><th>Status</th><th>Started</th><th>Metrics</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        '<p><button>Compare selected runs</button> '
        '<span class="muted">the oldest selected run is the baseline</span></p>'
        "</form>"
    )
    return page("Runs", f"<h1>Runs</h1>{form}{table}")


def _kv(rows: Sequence[tuple[str, str]]) -> str:
    cells = "".join(f"<tr><th>{esc(k)}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<table>{cells}</table>"


def render_run(store: Store, run_id: str) -> str:
    run = export_runs(store, [run_id])["runs"][0]
    dataset, config = run["dataset"], run["config"]
    summary = [
        ("status", f'<span class="{_status_class(run["status"])}">{esc(run["status"])}</span>'),
        ("experiment", esc(run["experiment_id"])),
        ("started", esc(run["started_at"])),
        ("finished", esc(run["finished_at"] or "-")),
        ("model", esc(config["model"])),
        ("seed", esc(run["seed"])),
        ("params", f"<pre>{esc(json.dumps(config['params'], sort_keys=True))}</pre>"),
        ("dataset", esc(f"{dataset['dataset_id']}@{dataset['version']}, {dataset['n_rows']} rows")),
        ("config hash", esc(run["config_hash"])),
    ]
    if run["error"]:
        summary.insert(1, ("error", f'<span class="bad">{esc(run["error"])}</span>'))
    body = [f"<h1>{esc(run_id)}</h1>", _kv(summary)]

    series: dict[str, list[tuple[int, float]]] = {}
    for point in run["metrics"]:
        series.setdefault(point["name"], []).append((point["step"], point["value"]))
    if series:
        body.append("<h2>Metrics</h2>")
        rows = "".join(
            f"<tr><td>{esc(name)}</td><td class=\"num\">{fmt(pts[-1][1])}</td>"
            f"<td>{esc(pts[-1][0])}</td><td>{len(pts)}</td></tr>"
            for name, pts in sorted(series.items())
        )
        body.append(
            "<table><thead><tr><th>Metric</th><th>Final value</th><th>At step</th>"
            f"<th>Points</th></tr></thead><tbody>{rows}</tbody></table>"
        )
        for name, pts in sorted(series.items()):
            chart = line_chart(f"{name} by step", pts)
            if chart:
                body.append(f"<h2>{esc(name)} over steps</h2>{chart}")
    if run["tags"]:
        body.append("<h2>Tags</h2>" + _kv([(k, esc(v)) for k, v in run["tags"].items()]))
    if run["artifacts"]:
        rows = "".join(
            f"<tr><td>{esc(a['name'])}</td><td>{esc(a['kind'])}</td>"
            f"<td>{esc(a['size_bytes'])}</td><td>{esc(a['sha256'])}</td></tr>"
            for a in run["artifacts"]
        )
        body.append(
            "<h2>Files</h2><table><thead><tr><th>Name</th><th>Kind</th><th>Bytes</th>"
            f"<th>SHA-256</th></tr></thead><tbody>{rows}</tbody></table>"
        )
    body.append("<h2>Environment</h2>" + _kv([(k, esc(v)) for k, v in run["environment"].items()]))
    if run["notes"]:
        notes = "".join(
            f'<div class="card"><span class="muted">{esc(n["created_at"])}</span>'
            f"<pre>{esc(n['text'])}</pre></div>"
            for n in run["notes"]
        )
        body.append(f"<h2>Notes</h2>{notes}")
    body.append(f'<p><a href="/api/run/{esc(run_id)}">JSON</a></p>')
    return page(run_id, "".join(body))


def _render_diff(result: RunComparison) -> str:
    out = []
    for diff in result.diffs:
        lines = [
            f"<h2>{esc(diff.run_id)} vs {esc(result.baseline)}</h2>",
            f'<p class="muted">same config: {diff.same_config}, same dataset: '
            f"{diff.same_dataset}, same seed: {diff.same_seed}</p>",
        ]
        if diff.params:
            rows = "".join(
                f"<tr><td>{esc(p.key)}</td><td>{esc(p.kind)}</td>"
                f"<td>{esc(repr(p.old))}</td><td>{esc(repr(p.new))}</td></tr>"
                for p in diff.params
            )
            lines.append(
                "<table><thead><tr><th>Parameter</th><th>Change</th><th>Baseline</th>"
                f"<th>This run</th></tr></thead><tbody>{rows}</tbody></table>"
            )
        if diff.metrics.deltas:
            rows = ""
            for m in diff.metrics.deltas:
                css = {True: "good", False: "bad", None: "muted"}[m.improved]
                word = {True: "better", False: "worse", None: ""}[m.improved]
                rows += (
                    f"<tr><td>{esc(m.name)}</td><td class=\"num\">{fmt(m.old)}</td>"
                    f'<td class="num">{fmt(m.new)}</td>'
                    f'<td class="num {css}">{m.delta:+.4f} {word}</td></tr>'
                )
            lines.append(
                "<table><thead><tr><th>Metric</th><th>Baseline</th><th>This run</th>"
                f"<th>Change</th></tr></thead><tbody>{rows}</tbody></table>"
            )
        for e in diff.environment:
            lines.append(
                f'<p class="muted">environment {esc(e.key)}: {esc(e.old)} to {esc(e.new)}</p>'
            )
        out.append("".join(lines))
    return "".join(out)


def _check_compare_ids(query: dict[str, list[str]], html_page: bool) -> list[str]:
    """Run ids to compare. The first is the baseline; the web page defaults to the oldest."""
    ids = list(query.get("run", []))
    if len(ids) < 2:
        raise DashboardError("select at least two runs to compare")
    if len(ids) > MAX_COMPARE:
        raise DashboardError(f"at most {MAX_COMPARE} runs can be compared at once")
    if html_page:
        ids.sort()  # ids are zero-padded, so this is oldest first
    baseline = _query_one(query, "baseline")
    if baseline is not None:
        if baseline not in ids:
            raise DashboardError("baseline must be one of the selected runs")
        ids.remove(baseline)
        ids.insert(0, baseline)
    return ids


def render_compare(store: Store, query: dict[str, list[str]]) -> str:
    result = compare_runs(store, _check_compare_ids(query, html_page=True))
    names = sorted({n for values in result.values.values() for n in values})
    head = "".join(f"<th>{esc(n)}</th>" for n in names)
    rows = "".join(
        f'<tr><td><a href="/run/{esc(rid)}">{esc(rid)}</a></td>'
        + "".join(
            f'<td class="num">{fmt(result.values[rid][n]) if n in result.values[rid] else "-"}</td>'
            for n in names
        )
        + "</tr>"
        for rid in result.run_ids
    )
    chosen = [("run", r) for r in result.run_ids]
    others = " ".join(
        f'<a href="/compare?{esc(urlencode([*chosen, ("baseline", r)]))}">{esc(r)}</a>'
        for r in result.run_ids
        if r != result.baseline
    )
    body = [
        "<h1>Compare runs</h1>",
        f'<p class="muted">baseline: {esc(result.baseline)}'
        f"{' &middot; use as baseline instead: ' + others if others else ''}</p>",
        f"<table><thead><tr><th>Run</th>{head}</tr></thead><tbody>{rows}</tbody></table>",
    ]
    for ranking in result.rankings:
        ranked = ", ".join(f"{r.rank}. {esc(r.run_id)} ({fmt(r.value)})" for r in ranking.ranked)
        body.append(f"<p>rank by <strong>{esc(ranking.metric)}</strong>: {ranked}</p>")
    for name in names:
        items = [
            (rid, result.values[rid][name])
            for rid in result.run_ids
            if name in result.values[rid]
        ]
        body.append(f"<h2>{esc(name)}</h2>{bar_chart(name, items)}")
    body.append(_render_diff(result))
    for warning in result.warnings:
        body.append(f'<p class="bad">warning: {esc(warning)}</p>')
    return page("Compare", "".join(body))


# -- routing --------------------------------------------------------------------------------


def _json_response(data: Any) -> Response:
    return Response(200, "application/json; charset=utf-8", json.dumps(data, sort_keys=True) + "\n")


def route(store: Store, path: str, query: dict[str, list[str]]) -> Response:
    """Map a request path to a response. Raises ModelBenchError for bad input or unknown runs."""
    if path == "/":
        return Response(200, "text/html; charset=utf-8", render_index(store, query))
    if path.startswith("/run/"):
        return Response(200, "text/html; charset=utf-8", render_run(store, path[5:]))
    if path == "/compare":
        return Response(200, "text/html; charset=utf-8", render_compare(store, query))
    if path == "/api/runs":
        runs = store.list_runs(_query_one(query, "experiment"), _status(query), _limit(query))
        return _json_response([r.run_id for r in runs])
    if path.startswith("/api/run/"):
        return _json_response(export_runs(store, [path[9:]])["runs"][0])
    if path == "/api/compare":
        ids = _check_compare_ids(query, html_page=False)
        return _json_response(compare_runs(store, ids).to_dict())
    raise NotFound(f"no page at {path}")


def _error_page(status: int, message: str, api: bool = False) -> Response:
    if api:
        return Response(
            status, "application/json; charset=utf-8", json.dumps({"error": message}) + "\n"
        )
    title = {400: "Bad request", 403: "Forbidden", 404: "Not found", 405: "Not allowed"}.get(
        status, "Error"
    )
    body = f"<h1>{esc(title)}</h1><p>{esc(message)}</p>"
    return Response(status, "text/html; charset=utf-8", page(title, body))


def is_allowed_host(header: str | None, loopback_only: bool) -> bool:
    """Whether the Host header names a loopback host (always true when not restricted)."""
    if not loopback_only or header is None:
        return True
    host = header.strip().lower()
    if host.startswith("["):
        host = host[1 : host.find("]")] if "]" in host else host
    elif host.count(":") == 1:
        host = host.split(":", 1)[0]
    return host in LOOPBACK_NAMES


def handle(
    db_path: str | Path, target: str, host_header: str | None, loopback_only: bool
) -> Response:
    """Everything a GET does, without sockets: host check, parsing, routing, error mapping."""
    if not is_allowed_host(host_header, loopback_only):
        return _error_page(403, "this server only answers requests addressed to localhost")
    if len(target) > MAX_TARGET_LENGTH:
        return _error_page(400, "request target is too long")
    try:
        parts = urlsplit(target)
        query = parse_qs(parts.query, max_num_fields=100)
        path = unquote(parts.path, errors="strict")
    except (ValueError, UnicodeDecodeError):
        return _error_page(400, "malformed request")
    api = path.startswith("/api/")
    try:
        with Store(db_path, read_only=True) as store:
            return route(store, path, query)
    except (StoreError, NotFound) as exc:
        return _error_page(404, str(exc), api)
    except ModelBenchError as exc:
        return _error_page(400, str(exc), api)
    except Exception as exc:  # never leak internals to the browser
        print(f"dashboard error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return _error_page(500, "internal error", api)


# -- server ---------------------------------------------------------------------------------


def is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def reuse_address_is_safe(system: str) -> bool:
    """Whether SO_REUSEADDR may be set. On Windows it lets another process bind a port that is
    already listening, so a second program could share this server's port; there the port is
    claimed exclusively instead."""
    return system != "nt"


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = reuse_address_is_safe(os.name)
    db_path: Path
    loopback_only: bool
    verbose: bool = False

    def server_bind(self) -> None:
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)  # exists on Windows only
        if not reuse_address_is_safe(os.name) and exclusive is not None:
            self.socket.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        super().server_bind()


class _V6Server(_Server):
    address_family = socket.AF_INET6


class _Handler(BaseHTTPRequestHandler):
    server: _Server  # type: ignore[assignment]
    server_version = "ModelBench"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        if self.server.verbose:
            super().log_message(format, *args)

    def _reply(
        self, response: Response, with_body: bool = True, extra: dict[str, str] | None = None
    ) -> None:
        body = response.body.encode("utf-8")
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in {**SECURITY_HEADERS, **(extra or {})}.items():
            self.send_header(name, value)
        self.end_headers()
        if with_body:
            self.wfile.write(body)

    def _get(self, with_body: bool) -> None:
        server = self.server
        response = handle(server.db_path, self.path, self.headers.get("Host"), server.loopback_only)
        self._reply(response, with_body)

    def do_GET(self) -> None:
        self._get(True)

    def do_HEAD(self) -> None:
        self._get(False)

    def _refuse(self) -> None:
        response = _error_page(405, "this dashboard is read-only", self.path.startswith("/api/"))
        self._reply(response, True, {"Allow": "GET, HEAD"})

    do_POST = do_PUT = do_DELETE = do_PATCH = _refuse  # type: ignore[assignment]


def make_server(
    db_path: str | Path,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    *,
    allow_network: bool = False,
) -> ThreadingHTTPServer:
    """Create (but do not start) the dashboard server. Port 0 picks a free port."""
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise DashboardError(f"port must be a whole number from 0 to 65535, got {port!r}")
    loopback = is_loopback_host(host)
    if not loopback and not allow_network:
        raise DashboardError(
            f"{host!r} is not a loopback address; the dashboard has no login, so anyone who can "
            "reach it can read every run. Pass allow_network=True (--allow-network) to accept that."
        )
    with Store(db_path, read_only=True):  # fail now, not on the first request
        pass
    cls = _V6Server if ":" in host else _Server
    try:
        server = cls((host, port), _Handler)
    except OSError as exc:
        raise DashboardError(f"cannot listen on {host}:{port}: {exc}") from exc
    server.db_path = Path(db_path)
    server.loopback_only = loopback
    return server


def serve(
    db_path: str | Path,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    *,
    allow_network: bool = False,
    out: TextIO | None = None,
) -> None:
    """Run the dashboard until interrupted (Ctrl-C), then close it cleanly."""
    server = make_server(db_path, host, port, allow_network=allow_network)
    name, bound_port = server.server_address[:2]
    shown = f"[{name}]" if ":" in str(name) else str(name)
    message = f"serving {db_path} at http://{shown}:{bound_port}/ (Ctrl-C to stop)"
    print(message, file=out or sys.stdout, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
