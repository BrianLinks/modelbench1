"""A standalone report on one experiment: every completed run ranked, with charts.

The report is one file with no script, no outside resources and no links back to a running
dashboard, so it can be mailed, archived or committed. HTML uses the dashboard's styling and
inline SVG charts; Markdown has the same tables without charts. The output depends only on the
store's contents (no clock, no paths), so the same runs always give the same bytes.

Contents: the runs ranked by one metric, a bar chart of that metric, one plot per parameter that
differs between runs, and the best run in full (settings, dataset, confusion matrix, environment
and notes). Failed runs are counted, not listed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__, confusion
from .compare import flatten_params, higher_is_better
from .dashboard import CSS, bar_chart, confusion_heatmap, esc, fmt, scatter_chart
from .errors import AnalysisError, CompareError, MetricError
from .export import write_text_atomic
from .store import Store

MAX_RUNS = 200
MAX_PLOTS = 4
MAX_BARS = 20
FORMATS = ("html", "md")


@dataclass(frozen=True)
class ReportRun:
    run_id: str
    model: str
    params: dict[str, Any]
    metrics: dict[str, float]


@dataclass(frozen=True)
class BestRun:
    run_id: str
    seed: int
    dataset: str
    config_hash: str
    environment: dict[str, Any]
    tags: dict[str, str]
    matrix_tag: str | None
    notes: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ReportData:
    experiment_id: str
    rank_metric: str
    higher_is_better: bool | None  # None when the metric has no known direction
    runs: tuple[ReportRun, ...]  # best first; runs without the rank metric last
    failed: int
    metric_names: tuple[str, ...]
    varying: tuple[str, ...]  # parameters whose value differs between runs
    best: BestRun | None


def _param_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def collect(store: Store, experiment_id: str, metric: str | None = None) -> ReportData:
    """Gather everything the report shows. Raises AnalysisError if nothing completed."""
    records = store.list_runs(experiment_id, "completed", MAX_RUNS)
    if not records:
        raise AnalysisError(f"experiment {experiment_id!r} has no completed runs")
    runs: dict[str, ReportRun] = {}
    for record in records:
        config = store.get_config(record.run_id)
        try:
            params = flatten_params(config.params)
        except CompareError:
            params = dict(config.params)
        summary = {
            k: v for k, v in store.final_metrics(record.run_id).items() if not k.startswith("fold/")
        }
        runs[record.run_id] = ReportRun(record.run_id, config.model, params, summary)
    names = tuple(sorted({n for r in runs.values() for n in r.metrics}))
    if not names:
        raise AnalysisError(f"the completed runs of {experiment_id!r} recorded no metrics")
    newest_first = store.get_config(records[0].run_id).metrics
    default = next((m for m in newest_first if m in names), names[0])
    chosen = metric if metric is not None else default
    if chosen not in names:
        raise AnalysisError(f"{chosen!r} is not a metric of these runs; they have {list(names)}")
    try:
        high: bool | None = higher_is_better(chosen)
    except CompareError:
        high = None
    sign = -1.0 if high else 1.0

    def order(run: ReportRun) -> tuple[int, float, str]:
        if chosen not in run.metrics:
            return (1, 0.0, run.run_id)
        return (0, sign * run.metrics[chosen] if high is not None else 0.0, run.run_id)

    ranked = tuple(sorted(runs.values(), key=order))
    seen: dict[str, set[str]] = {}
    for run in ranked:
        for key, value in run.params.items():
            seen.setdefault(key, set()).add(_param_text(value))
    varying = tuple(sorted(k for k, values in seen.items() if len(values) > 1))
    best_run = ranked[0] if high is not None and chosen in ranked[0].metrics else None
    best = None
    if best_run is not None:
        record = next(r for r in records if r.run_id == best_run.run_id)
        dataset = store.get_dataset(record.dataset_id, record.dataset_version)
        best = BestRun(
            record.run_id,
            record.seed,
            f"{dataset.dataset_id}@{dataset.version}, {dataset.n_rows} rows",
            record.config_hash,
            dict(record.environment),
            {k: v for k, v in record.tags.items() if k != confusion.TAG},
            record.tags.get(confusion.TAG),
            tuple((n.created_at, n.text) for n in store.get_notes(record.run_id)),
        )
    failed = len(store.list_runs(experiment_id, "failed", MAX_RUNS))
    return ReportData(experiment_id, chosen, high, ranked, failed, names, varying, best)


def _header_line(data: ReportData) -> str:
    direction = {True: "higher is better", False: "lower is better", None: "no known direction"}
    models = sorted({r.model for r in data.runs})
    failed = f", {data.failed} failed run(s) not shown" if data.failed else ""
    return (
        f"{len(data.runs)} completed run(s) of {', '.join(models)}{failed}. "
        f"Ranked by {data.rank_metric} ({direction[data.higher_is_better]})."
    )


def _params_text(run: ReportRun) -> str:
    return " ".join(f"{k}={_param_text(v)}" for k, v in run.params.items()) or "-"


def render_html(data: ReportData) -> str:
    title = f"Experiment {data.experiment_id}"
    body = [f"<h1>{esc(title)}</h1>", f'<p class="muted">{esc(_header_line(data))}</p>']
    head = "".join(f"<th>{esc(n)}</th>" for n in data.metric_names)
    rows = []
    for position, run in enumerate(data.runs, start=1):
        cells = "".join(
            f'<td class="num">{fmt(run.metrics[n]) if n in run.metrics else "-"}</td>'
            for n in data.metric_names
        )
        mark = " (best)" if data.best is not None and run.run_id == data.best.run_id else ""
        rows.append(
            f"<tr><td>{position}</td><td>{esc(run.run_id)}{mark}</td><td>{esc(run.model)}</td>"
            f"<td>{esc(_params_text(run))}</td>{cells}</tr>"
        )
    body.append(
        "<h2>Runs</h2><table><thead><tr><th>#</th><th>Run</th><th>Model</th><th>Params</th>"
        f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )
    scored = [r for r in data.runs if data.rank_metric in r.metrics]
    bars = [(r.run_id, r.metrics[data.rank_metric]) for r in scored[:MAX_BARS]]
    if bars:
        shown = f" (the best {MAX_BARS})" if len(scored) > MAX_BARS else ""
        body.append(f"<h2>{esc(data.rank_metric)} by run{shown}</h2>")
        body.append(bar_chart(data.rank_metric, bars))
    for param in data.varying[:MAX_PLOTS]:
        points = [
            (r.params[param], r.metrics[data.rank_metric], r.run_id)
            for r in scored
            if param in r.params
        ]
        best_index = next(
            (
                i
                for i, (_, _, rid) in enumerate(points)
                if data.best is not None and rid == data.best.run_id
            ),
            None,
        )
        heading = f"{data.rank_metric} by {param}"
        body.append(f"<h2>{esc(heading)}</h2>{scatter_chart(heading, points, param, best_index)}")
    if len(data.varying) > MAX_PLOTS:
        hidden = ", ".join(data.varying[MAX_PLOTS:])
        body.append(f'<p class="muted">Not plotted: {esc(hidden)}.</p>')
    if data.best is not None:
        body.append(_best_html(data.best))
    body.append(f'<p class="muted">Made by ModelBench {esc(__version__)}.</p>')
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{esc(title)} - ModelBench report</title><style>{CSS}</style></head>"
        f"<body><main>{''.join(body)}</main></body></html>"
    )


def _kv_html(rows: list[tuple[str, str]]) -> str:
    return (
        "<table>"
        + "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in rows)
        + ("</table>")
    )


def _best_html(best: BestRun) -> str:
    parts = [
        f"<h2>Best run: {esc(best.run_id)}</h2>",
        _kv_html(
            [("seed", str(best.seed)), ("dataset", best.dataset), ("config hash", best.config_hash)]
        ),
    ]
    if best.matrix_tag is not None:
        try:
            parts.append(
                "<h2>Confusion matrix</h2>" + confusion_heatmap(confusion.from_tag(best.matrix_tag))
            )
        except MetricError as exc:
            parts.append(f'<p class="bad">confusion matrix unreadable: {esc(exc)}</p>')
    if best.tags:
        parts.append("<h2>Tags</h2>" + _kv_html(list(best.tags.items())))
    parts.append(
        "<h2>Environment</h2>" + _kv_html([(k, str(v)) for k, v in best.environment.items()])
    )
    for created, text in best.notes:
        parts.append(
            f'<div class="card"><span class="muted">{esc(created)}</span>'
            f"<pre>{esc(text)}</pre></div>"
        )
    return "".join(parts)


def _md(text: object) -> str:
    return str(text).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def render_markdown(data: ReportData) -> str:
    lines = [f"# Experiment {_md(data.experiment_id)}", "", _md(_header_line(data)), ""]
    header = ["#", "Run", "Model", "Params", *data.metric_names]
    lines.append("| " + " | ".join(_md(h) for h in header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for position, run in enumerate(data.runs, start=1):
        mark = " (best)" if data.best is not None and run.run_id == data.best.run_id else ""
        cells = [
            str(position),
            _md(run.run_id) + mark,
            _md(run.model),
            _md(_params_text(run)),
            *(fmt(run.metrics[n]) if n in run.metrics else "-" for n in data.metric_names),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    if data.best is not None:
        best = data.best
        lines += ["", f"## Best run: {_md(best.run_id)}", ""]
        lines += [f"- seed: {best.seed}", f"- dataset: {_md(best.dataset)}"]
        lines.append(f"- config hash: {best.config_hash}")
        if best.matrix_tag is not None:
            try:
                grid = confusion.format_text(confusion.from_tag(best.matrix_tag), "")
                lines += ["", "Confusion matrix:", "", "```", *grid, "```"]
            except MetricError as exc:
                lines.append(f"- confusion matrix unreadable: {_md(exc)}")
        lines += ["", "Environment:", ""]
        lines += [f"- {_md(k)}: {_md(v)}" for k, v in best.environment.items()]
        for created, text in best.notes:
            lines += ["", f"Note ({_md(created)}): {_md(text)}"]
    lines += ["", f"Made by ModelBench {__version__}.", ""]
    return "\n".join(lines)


def write_report(
    store: Store,
    experiment_id: str,
    path: str | Path,
    *,
    fmt_name: str = "html",
    metric: str | None = None,
) -> ReportData:
    """Write the report for `experiment_id` to `path` atomically."""
    if fmt_name not in FORMATS:
        raise AnalysisError(f"format must be one of {FORMATS}, got {fmt_name!r}")
    data = collect(store, experiment_id, metric)
    text = render_html(data) if fmt_name == "html" else render_markdown(data)
    write_text_atomic(path, text)
    return data
