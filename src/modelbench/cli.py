"""Command-line interface: `modelbench [--db PATH] <command> ...`.

Exit codes: 0 success; 1 for a negative result that is not an error (`repro` when the run is not
reproduced); 2 for bad arguments, a ModelBench error, or a file system error, with a one-line
message on stderr.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .dashboard import DEFAULT_PORT, serve
from .datasets import make_version, read_csv
from .errors import CliError, ModelBenchError
from .export import export_json, export_metrics_csv, export_runs, write_text_atomic
from .models import all_models, model_kind
from .reproducibility import check_reproducibility, recover_stale_runs
from .run_comparison import RunComparison, compare_runs
from .runner import run_experiment
from .samples import available_samples, generate, run_demo, table_to_csv
from .store import STATUSES, Store

DEFAULT_DB = "modelbench.db"


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _safe(text: object) -> str:
    """Make stored text safe to print: control characters (terminal escapes) are shown escaped."""
    return "".join(
        ch if unicodedata.category(ch) != "Cc" else ch.encode("unicode_escape").decode("ascii")
        for ch in str(text)
    )


def _split_list(text: str | None, what: str) -> list[str] | None:
    if text is None:
        return None
    items = [part.strip() for part in text.split(",")]
    if not all(items):
        raise CliError(f"{what} must be a comma-separated list with no empty names")
    return items


def _open_existing(path: str) -> Store:
    if not Path(path).is_file():
        raise CliError(f"no database at {path}; `modelbench run` creates one")
    return Store(path)


def _table(rows: Sequence[Sequence[str]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip()
        for row in rows
    )


# -- commands ---------------------------------------------------------------------------------


def cmd_models(args: argparse.Namespace) -> int:
    print("\n".join(f"{name}  {model_kind(name)}" for name in all_models()))
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    table = read_csv(args.file)
    version = make_version(args.id, table, args.label)
    info = {
        "dataset_id": version.dataset_id,
        "version": version.version,
        "sha256": version.sha256,
        "n_rows": version.n_rows,
        "columns": list(version.columns),
        "label_column": version.label_column,
    }
    if args.json:
        print(_json(info))
    else:
        print(f"{version.dataset_id}@{version.version}")
        print(f"sha256   {version.sha256}")
        print(f"rows     {version.n_rows}")
        print(f"columns  {', '.join(version.columns)}")
        print(f"label    {version.label_column}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = ExperimentConfig.from_file(args.config)
    table = read_csv(args.data)
    with Store(args.db) as store:
        result = run_experiment(
            store,
            config,
            table,
            args.label,
            feature_columns=_split_list(args.features, "--features"),
            artifact_dir=args.artifact_dir,
            reuse_existing=args.reuse,
        )
    if args.json:
        print(
            _json(
                {
                    "run_id": result.run_id,
                    "reused": result.reused,
                    "metrics": result.metrics,
                    "n_train": result.n_train,
                    "n_test": result.n_test,
                    "predictions_sha256": result.predictions_sha256,
                }
            )
        )
        return 0
    note = " (reused an identical earlier run)" if result.reused else ""
    print(f"{result.run_id}{note}")
    print(f"train {result.n_train}  test {result.n_test}")
    for name, value in sorted(result.metrics.items()):
        print(f"{name}  {_fmt(value)}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    with _open_existing(args.db) as store:
        runs = store.list_runs(args.experiment, args.status, args.limit)
        rows = [("RUN", "EXPERIMENT", "MODEL", "STATUS", "STARTED", "METRICS")]
        records: list[dict[str, Any]] = []
        for run in runs:
            metrics = store.final_metrics(run.run_id)
            model = store.get_config(run.run_id).model
            summary = " ".join(f"{k}={_fmt(v)}" for k, v in sorted(metrics.items()))
            rows.append(
                (run.run_id, run.experiment_id, model, run.status, run.started_at, summary)
            )
            records.append(
                {
                    "run_id": run.run_id,
                    "experiment_id": run.experiment_id,
                    "model": model,
                    "status": run.status,
                    "started_at": run.started_at,
                    "metrics": metrics,
                }
            )
    if args.json:
        print(_json(records))
    elif records:
        print(_table(rows))
    else:
        print("no runs")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    with _open_existing(args.db) as store:
        run = export_runs(store, [args.run_id])["runs"][0]
    if args.json:
        print(_json(run))
        return 0
    print(f"{run['run_id']}  {run['status']}  experiment {run['experiment_id']}")
    print(f"started {run['started_at']}  finished {run['finished_at']}")
    if run["error"]:
        print(f"error   {_safe(run['error'])}")
    dataset = run["dataset"]
    print(f"dataset {dataset['dataset_id']}@{dataset['version']} ({dataset['n_rows']} rows)")
    config = run["config"]
    params = _safe(json.dumps(config["params"], sort_keys=True))
    print(f"model   {config['model']}  seed {config['seed']}  params {params}")
    for point in run["metrics"]:
        print(f"metric  {point['name']} (step {point['step']})  {_fmt(point['value'])}")
    for key, value in run["tags"].items():
        print(f"tag     {_safe(key)}={_safe(value)}")
    for artifact in run["artifacts"]:
        print(f"file    {artifact['name']}  {artifact['size_bytes']} bytes  {artifact['sha256']}")
    for note in run["notes"]:
        print(f"note    {note['created_at']}  {_safe(note['text'])}")
    return 0


def _render_comparison(result: RunComparison) -> str:
    lines = [f"baseline {result.baseline}"]
    names = sorted({name for values in result.values.values() for name in values})
    if names:
        rows = [("RUN", *names)]
        for run_id in result.run_ids:
            values = result.values[run_id]
            rows.append((run_id, *(_fmt(values[n]) if n in values else "-" for n in names)))
        lines += ["", _table(rows)]
    for ranking in result.rankings:
        ranked = ", ".join(f"{r.rank}. {r.run_id} ({_fmt(r.value)})" for r in ranking.ranked)
        lines.append(f"rank {ranking.metric}: {ranked}")
    for diff in result.diffs:
        lines += ["", f"{diff.run_id} vs {result.baseline}"]
        lines.append(
            f"  same config {diff.same_config}, dataset {diff.same_dataset}, seed {diff.same_seed}"
        )
        for p in diff.params:
            lines.append(f"  param {p.key}: {p.kind} {p.old!r} -> {p.new!r}")
        for m in diff.metrics.deltas:
            verdict = {True: "better", False: "worse", None: ""}[m.improved]
            change = f"{_fmt(m.old)} -> {_fmt(m.new)} ({m.delta:+.4f}) {verdict}"
            lines.append(f"  {m.name}: {change}".rstrip())
        for e in diff.environment:
            lines.append(f"  environment {_safe(e.key)}: {_safe(e.old)} -> {_safe(e.new)}")
    for warning in result.warnings:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)


def cmd_compare(args: argparse.Namespace) -> int:
    with _open_existing(args.db) as store:
        result = compare_runs(store, args.run_ids, _split_list(args.metrics, "--metrics"))
    print(_json(result.to_dict()) if args.json else _render_comparison(result))
    return 0


def cmd_repro(args: argparse.Namespace) -> int:
    table = read_csv(args.data)
    with _open_existing(args.db) as store:
        report = check_reproducibility(
            store,
            args.run_id,
            table,
            args.label,
            feature_columns=_split_list(args.features, "--features"),
            tolerance=args.tolerance,
        )
    if args.json:
        print(_json(report.to_dict()))
    else:
        print(f"{report.run_id}: {report.verdict}")
        print(report.detail)
        for check in report.metrics:
            rerun = "-" if check.rerun is None else _fmt(check.rerun)
            flag = "ok" if check.within_tolerance else "DIFFERS"
            print(f"  {check.name}: {_fmt(check.original)} -> {rerun}  {flag}")
        for diff in report.environment_differences:
            old, new = _safe(diff.old), _safe(diff.new)
            print(f"  environment {_safe(diff.key)}: {old} -> {new} ({diff.impact})")
        if report.explained_by_environment:
            print("the environment differences above are a possible cause")
    return 0 if report.reproduced else 1


def cmd_note(args: argparse.Namespace) -> int:
    with _open_existing(args.db) as store:
        store.add_note(args.run_id, args.text)
    print(f"added a note to {args.run_id}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    with _open_existing(args.db) as store:
        text = (
            export_metrics_csv(store, args.run_ids)
            if args.format == "csv"
            else export_json(store, args.run_ids)
        )
    if args.output:
        write_text_atomic(args.output, text)
        print(f"wrote {len(args.run_ids)} run(s) to {args.output}")
    else:
        sys.stdout.write(text)
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    with _open_existing(args.db) as store:
        closed = recover_stale_runs(store, args.older_than)
    if closed:
        print(f"marked {len(closed)} abandoned run(s) as failed: {', '.join(closed)}")
    else:
        print("no abandoned runs")
    return 0


SAMPLE_OPTIONS = ("rows_per_class", "classes", "features", "separation", "spread", "noise")


def cmd_sample(args: argparse.Namespace) -> int:
    given = {k: getattr(args, k) for k in SAMPLE_OPTIONS if getattr(args, k) is not None}
    table = generate(args.name, args.seed, **given)
    text = table_to_csv(table)
    if args.output:
        write_text_atomic(args.output, text)
        print(f"wrote {len(table)} rows to {args.output}")
    else:
        sys.stdout.write(text)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        runs = run_demo(store, args.seed, args.rows_per_class)
    if args.json:
        print(
            _json(
                [
                    {
                        "dataset": r.dataset,
                        "model": r.model,
                        "params": dict(r.params),
                        "run_id": r.result.run_id,
                        "reused": r.result.reused,
                        "metrics": r.result.metrics,
                    }
                    for r in runs
                ]
            )
        )
        return 0
    rows = [("DATASET", "MODEL", "PARAMS", "ACCURACY", "F1", "RUN")]
    for r in runs:
        params = json.dumps(dict(r.params), sort_keys=True, separators=(",", ":"))
        rows.append(
            (
                r.dataset,
                r.model,
                params if r.params else "-",
                _fmt(r.result.metrics["accuracy"]),
                _fmt(r.result.metrics["f1"]),
                r.result.run_id,
            )
        )
    print(_table(rows))
    reused = sum(r.result.reused for r in runs)
    print(f"{len(runs)} runs ({len(runs) - reused} new, {reused} reused) in {args.db}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    if not Path(args.db).is_file():  # serving never creates or upgrades a database
        raise CliError(f"no database at {args.db}; `modelbench run` creates one")
    serve(args.db, args.host, args.port, allow_network=args.allow_network)
    return 0


# -- parser -----------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modelbench", description="Local experiment tracking and model comparison."
    )
    parser.add_argument("--db", default=DEFAULT_DB, help=f"run database (default {DEFAULT_DB})")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    def add(
        name: str, func: Any, help_text: str, json_flag: bool = False
    ) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.set_defaults(func=func)
        if json_flag:
            p.add_argument("--json", action="store_true", help="print JSON instead of text")
        return p

    add("models", cmd_models, "list the available models")

    p = add("dataset", cmd_dataset, "show the content-hash version of a CSV file", True)
    p.add_argument("file")
    p.add_argument("--id", required=True, help="dataset identifier")
    p.add_argument("--label", help="label column")

    p = add("run", cmd_run, "run an experiment from a JSON config file", True)
    p.add_argument("config", help="experiment config (JSON)")
    p.add_argument("--data", required=True, help="CSV file")
    p.add_argument("--label", required=True, help="label column")
    p.add_argument("--features", help="comma-separated feature columns (default: all others)")
    p.add_argument("--artifact-dir", help="also write the predictions here")
    p.add_argument("--reuse", action="store_true", help="reuse an identical completed run")

    p = add("list", cmd_list, "list runs, newest first", True)
    p.add_argument("--experiment")
    p.add_argument("--status", choices=STATUSES)
    p.add_argument("--limit", type=int, default=20)

    p = add("show", cmd_show, "show one run in full", True)
    p.add_argument("run_id")

    p = add("compare", cmd_compare, "compare completed runs; the first is the baseline", True)
    p.add_argument("run_ids", nargs="+")
    p.add_argument("--metrics", help="comma-separated metrics to compare")

    p = add("repro", cmd_repro, "re-run a run and check it reproduces (exit 1 if not)", True)
    p.add_argument("run_id")
    p.add_argument("--data", required=True, help="the CSV the run used")
    p.add_argument("--label", required=True)
    p.add_argument("--features")
    p.add_argument("--tolerance", type=float, default=0.0, help="absolute metric tolerance")

    p = add("note", cmd_note, "attach a note to a run")
    p.add_argument("run_id")
    p.add_argument("text")

    p = add("export", cmd_export, "export runs as JSON or a metrics CSV")
    p.add_argument("run_ids", nargs="+")
    p.add_argument("--format", choices=("json", "csv"), default="json")
    p.add_argument("-o", "--output", help="write to a file instead of stdout")

    p = add("sample", cmd_sample, "write a seeded synthetic dataset as CSV")
    p.add_argument("name", choices=available_samples())
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-o", "--output", help="write to a file instead of stdout")
    p.add_argument("--rows-per-class", dest="rows_per_class", type=int)
    p.add_argument("--classes", type=int, help="blobs only")
    p.add_argument("--features", type=int, help="blobs only")
    p.add_argument("--separation", type=float, help="blobs only")
    p.add_argument("--spread", type=float, help="blobs only")
    p.add_argument("--noise", type=float, help="moons and xor only")

    p = add("demo", cmd_demo, "run every model on three sample datasets", True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rows-per-class", dest="rows_per_class", type=int, default=100)

    p = add("serve", cmd_serve, "serve a read-only dashboard of the runs")
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (default loopback)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument(
        "--allow-network",
        action="store_true",
        help="allow a non-loopback --host; the dashboard has no login",
    )

    p = add("recover", cmd_recover, "mark long-running unfinished runs as failed")
    p.add_argument("--older-than", type=float, required=True, metavar="SECONDS")
    return parser


def _tolerate_unencodable_output() -> None:
    """Print `?` for characters the console cannot encode instead of crashing (Windows
    consoles often use a legacy code page, and notes may hold any text)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    _tolerate_unencodable_output()
    args = build_parser().parse_args(argv)
    try:
        code: int = args.func(args)
    except (ModelBenchError, OSError, sqlite3.Error) as exc:
        print(f"error: {_safe(exc)}", file=sys.stderr)
        return 2
    return code
