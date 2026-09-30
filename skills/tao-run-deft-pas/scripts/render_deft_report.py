#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a deterministic PAS DEFT HTML report from audited disk state.

The HTML rendering layer is deterministic and performs no workload actions.
It uses ``audit_deft_run`` first, so invoke it with the PAS runtime that
provides YAML and parquet validation dependencies. It does not infer completed
work from prose or directory names. A loop-end render additionally requires a
terminal ``loop_stop`` event, including terminal FAILED runs.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import pathlib
import re
import sys
import tempfile
import warnings
from typing import Any, Iterable

import audit_deft_run
import metric_contract


REPORT_NAME = "DEFT_Loop_Report.html"
_SENSITIVE_KEY_PARTS = (
    "password",
    "secret",
    "token",
    "credential",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "ngc_key",
    "hf_token",
)


class ReportError(ValueError):
    """A report cannot be rendered safely from the available evidence."""


_ACCEPTED_AUDIT_STATUSES = frozenset({"IN_PROGRESS", "FAILED", "COMPLETE"})


def _audited_report(results_dir: pathlib.Path) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = audit_deft_run.audit(results_dir, require_complete=False)
    if not isinstance(report, dict):
        raise ReportError("audit returned a non-object report")

    status = report.get("status")
    errors = report.get("errors")
    if not isinstance(errors, list):
        raise ReportError("audit report has an invalid errors field")
    if status not in _ACCEPTED_AUDIT_STATUSES:
        detail = "; ".join(str(item) for item in errors[:3])
        if status == "INVALID":
            raise ReportError(
                f"audit rejected the run: {detail or 'invalid disk state'}"
            )
        raise ReportError(f"audit returned unsupported status {status!r}")
    if errors:
        detail = "; ".join(str(item) for item in errors[:3])
        raise ReportError(f"audit returned errors with status {status}: {detail}")
    if not isinstance(report.get("terminal"), bool):
        raise ReportError("audit report has an invalid terminal field")
    log_entries = report.get("log_entries")
    if (
        isinstance(log_entries, bool)
        or not isinstance(log_entries, int)
        or log_entries < 0
    ):
        raise ReportError("audit report has an invalid log_entries field")
    if not isinstance(report.get("warnings"), list):
        raise ReportError("audit report has an invalid warnings field")
    return report


def _canonical_snapshot(results_dir: pathlib.Path) -> tuple[bytes, bool, bytes]:
    state_bytes = (results_dir / "deft_state.json").read_bytes()
    log_path = results_dir / "loop_log.jsonl"
    log_exists = log_path.exists()
    log_bytes = log_path.read_bytes() if log_exists else b""
    return state_bytes, log_exists, log_bytes


def _parse_snapshot(
    snapshot: tuple[bytes, bool, bytes]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state_bytes, _, log_bytes = snapshot
    try:
        state = json.loads(state_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"deft_state.json is invalid JSON: {exc}") from exc
    if not isinstance(state, dict):
        raise ReportError("deft_state.json root must be an object")

    entries: list[dict[str, Any]] = []
    try:
        lines = log_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ReportError(f"loop_log.jsonl is not UTF-8: {exc}") from exc
    for line_number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ReportError(
                f"loop_log.jsonl line {line_number} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(entry, dict):
            raise ReportError(
                f"loop_log.jsonl line {line_number} must be a JSON object"
            )
        entries.append(entry)
    return state, entries


def _iteration_sort_key(label: str) -> tuple[int, int, str]:
    if label == "baseline":
        return (0, 0, label)
    match = re.fullmatch(r"iter([1-9][0-9]*)", label)
    if match:
        return (1, int(match.group(1)), label)
    return (2, 0, label)


def _phase_dir(results_dir: pathlib.Path, label: str) -> pathlib.Path | None:
    if label == "baseline":
        return results_dir / "zs"
    match = re.fullmatch(r"iter([1-9][0-9]*)", label)
    if match:
        return results_dir / f"iter_{match.group(1)}"
    return None


def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _format_duration(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _format_delta(value: float, reference: float | None) -> str:
    if reference is None:
        return "—"
    return f"{value - reference:+.6g}"


def _is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _redact_config(value: Any) -> Any:
    """Return a recursively redacted copy of untrusted configuration data."""

    if isinstance(value, dict):
        return {
            key: "[redacted]" if _is_sensitive_key(str(key)) else _redact_config(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redact_config(child) for child in value]
    return value


def _flatten_config(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        rows: list[tuple[str, str]] = []
        for key in sorted(value, key=str):
            name = f"{prefix}.{key}" if prefix else str(key)
            if _is_sensitive_key(str(key)):
                rows.append((name, "[redacted]"))
            else:
                rows.extend(_flatten_config(value[key], name))
        return rows
    if isinstance(value, list):
        if not value:
            return [(prefix or "config", "[]")]
        rows: list[tuple[str, str]] = []
        for index, child in enumerate(value):
            name = f"{prefix}[{index}]" if prefix else f"config[{index}]"
            rows.extend(_flatten_config(child, name))
        return rows
    return [(prefix or "config", _format_value(value))]


def _metric_rows(
    state: dict[str, Any],
    entries: list[dict[str, Any]],
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    successful_evaluates = {
        str(entry.get("iteration"))
        for entry in entries
        if entry.get("stage") == "evaluate" and entry.get("status") == "ok"
    }
    iterations = state.get("iterations", {})
    rows: list[dict[str, Any]] = []
    if not isinstance(iterations, dict):
        return rows, None
    for label in sorted(iterations, key=_iteration_sort_key):
        if label not in successful_evaluates:
            continue
        info = iterations[label]
        if not isinstance(info, dict):
            continue
        result = metric_contract.result_from_iteration(info, contract)
        if result is None:
            continue
        passed, _ = metric_contract.result_passes(contract, result)
        rows.append(
            {
                "label": label,
                "value": float(result["value"]),
                "passed": passed,
                "evidence_path": result.get("evidence_path"),
            }
        )
    if not rows:
        return rows, None
    minimizing = contract["op"] in metric_contract.MINIMIZING_OPERATORS
    best = min(rows, key=lambda row: row["value"]) if minimizing else max(
        rows, key=lambda row: row["value"]
    )
    return rows, str(best["label"])


def _run_status(
    audit_report: dict[str, Any],
    state: dict[str, Any],
    contract: dict[str, Any],
    metric_rows: list[dict[str, Any]],
) -> tuple[str, str]:
    audited = str(audit_report.get("status", "UNKNOWN"))
    reason = state.get("loop_stop_reason")
    if audited == "FAILED" or reason == "hard_stop":
        return "FAILED", "The audited run ended at a hard stop."
    if not audit_report.get("terminal"):
        return "IN PROGRESS", str(audit_report.get("next_action", "continue workflow"))
    if reason == "kpi_met" or any(row["passed"] for row in metric_rows):
        return "MET", "The approved metric contract passed."
    if contract.get("target") is None:
        return "COMPLETE", "The iteration budget completed without a KPI target."
    return "TARGET NOT MET", "The iteration budget completed before the target passed."


def _state_as_of(state: dict[str, Any], entries: list[dict[str, Any]]) -> str:
    if entries and isinstance(entries[-1].get("ts"), str):
        return str(entries[-1]["ts"])
    return _format_value(state.get("started_at"))


def _resolve_path(value: str, *, base: pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _collect_paths(
    value: Any,
    prefix: str = "",
    *,
    base: pathlib.Path,
) -> list[tuple[str, pathlib.Path]]:
    paths: list[tuple[str, pathlib.Path]] = []
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            child = value[key]
            name = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_collect_paths(child, name, base=base))
    elif isinstance(value, str):
        field = prefix.rsplit(".", 1)[-1]
        path_fields = getattr(audit_deft_run, "PATH_FIELDS", set())
        if field in path_fields or field in {"evidence_path", "stats_json"}:
            paths.append((prefix or "path", _resolve_path(value, base=base)))
    return paths


def _dedupe_paths(
    paths: Iterable[tuple[str, pathlib.Path]],
) -> list[tuple[str, pathlib.Path]]:
    unique: list[tuple[str, pathlib.Path]] = []
    seen: set[str] = set()
    for label, path in paths:
        rendered = str(path)
        if rendered in seen:
            continue
        seen.add(rendered)
        unique.append((label, path))
    return unique


def _path_html(
    label: str, path: pathlib.Path, *, exists_override: bool | None = None
) -> str:
    expanded = path.expanduser()
    exists = expanded.exists() if exists_override is None else exists_override
    display = html.escape(str(expanded))
    name = html.escape(label)
    state = "exists" if exists else "missing"
    css_class = "path-ok" if exists else "path-missing"
    if expanded.is_absolute():
        try:
            target = html.escape(expanded.resolve().as_uri(), quote=True)
            display = f'<a href="{target}"><code>{display}</code></a>'
        except (OSError, ValueError):
            display = f"<code>{display}</code>"
    else:
        display = f"<code>{display}</code>"
    return f"<li><span class=\"field\">{name}</span>: {display} <span class=\"{css_class}\">{state}</span></li>"


def _table(headers: list[str], rows: list[list[str]], *, empty: str) -> str:
    if not rows:
        return f'<p class="empty">{html.escape(empty)}</p>'
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<div class=\"table-wrap\"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _render_html(
    *,
    results_dir: pathlib.Path,
    trigger: str,
    state: dict[str, Any],
    entries: list[dict[str, Any]],
    audit_report: dict[str, Any],
) -> tuple[str, str, str | None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        contract = metric_contract.contract_from_state(state)
        metric_rows, best_label = _metric_rows(state, entries, contract)
    status, status_detail = _run_status(audit_report, state, contract, metric_rows)
    status_class = {
        "FAILED": "failed",
        "MET": "success",
        "IN PROGRESS": "progress",
        "TARGET NOT MET": "neutral",
        "COMPLETE": "success",
    }.get(status, "neutral")

    state_as_of = _state_as_of(state, entries)
    target = "none" if contract["target"] is None else _format_value(contract["target"])
    config_rows = [
        ["workflow", html.escape(_format_value(state.get("workflow")))],
        ["started_at", html.escape(_format_value(state.get("started_at")))],
        ["results_dir", f"<code>{html.escape(str(results_dir))}</code>"],
        ["max_iterations", html.escape(_format_value(state.get("max_iterations")))],
        ["current_iteration", html.escape(_format_value(state.get("current_iteration")))],
        ["loop_stop_reason", html.escape(_format_value(state.get("loop_stop_reason")))],
    ]
    for name, value in _flatten_config(_redact_config(state.get("config", {}))):
        config_rows.append([html.escape(name), html.escape(value)])

    baseline_value = next(
        (row["value"] for row in metric_rows if row["label"] == "baseline"), None
    )
    previous: float | None = None
    kpi_rows: list[list[str]] = []
    for row in metric_rows:
        evidence = row.get("evidence_path")
        evidence_html = "—"
        if isinstance(evidence, str) and evidence:
            evidence_path = _resolve_path(evidence, base=results_dir)
            evidence_html = _path_html("metric", evidence_path)[4:-5]
        gate = "no target" if contract["target"] is None else (
            "met" if row["passed"] else "not met"
        )
        marker = " <strong class=\"best\">best</strong>" if row["label"] == best_label else ""
        kpi_rows.append(
            [
                html.escape(str(row["label"])) + marker,
                html.escape(_format_value(row["value"])),
                html.escape(_format_delta(row["value"], baseline_value)),
                html.escape(_format_delta(row["value"], previous)),
                html.escape(gate),
                evidence_html,
            ]
        )
        previous = row["value"]

    notes: list[str] = []
    iteration_rows: list[list[str]] = []
    iterations = state.get("iterations", {})
    if isinstance(iterations, dict):
        for label in sorted(iterations, key=_iteration_sort_key):
            info = iterations[label]
            if not isinstance(info, dict):
                continue
            artifact_paths = _collect_paths(info, base=results_dir)
            phase = _phase_dir(results_dir, label)
            if phase is not None:
                summary_path = phase / "iteration_summary.json"
                if summary_path.exists():
                    artifact_paths.append(("iteration_summary", summary_path))
            artifact_paths = _dedupe_paths(artifact_paths)
            if artifact_paths:
                items = "".join(_path_html(name, path) for name, path in artifact_paths)
                artifacts_html = (
                    f"<details><summary>{len(artifact_paths)} evidence path(s)</summary>"
                    f"<ul class=\"paths\">{items}</ul></details>"
                )
            else:
                artifacts_html = "—"
            metric = next(
                (row for row in metric_rows if row["label"] == label), None
            )
            metric_text = _format_value(metric["value"]) if metric else "—"
            iteration_rows.append(
                [
                    html.escape(label),
                    html.escape(_format_value(info.get("status"))),
                    html.escape(_format_value(info.get("stage_completed"))),
                    html.escape(metric_text),
                    artifacts_html,
                ]
            )

    timeline_rows = [
        [
            html.escape(_format_value(entry.get("seq"))),
            html.escape(_format_value(entry.get("iteration"))),
            html.escape(_format_value(entry.get("stage"))),
            html.escape(_format_value(entry.get("status"))),
            html.escape(_format_duration(entry.get("duration_s"))),
            html.escape(_format_value(entry.get("summary"))),
        ]
        for entry in entries
    ]

    canonical_paths = [
        ("deft_state", results_dir / "deft_state.json"),
        ("loop_log", results_dir / "loop_log.jsonl"),
        ("report", results_dir / REPORT_NAME),
    ]
    canonical_html = "".join(
        _path_html(name, path, exists_override=True if name == "report" else None)
        for name, path in canonical_paths
    )
    warning_items = [str(item) for item in audit_report.get("warnings", [])]
    warning_items.extend(notes)
    warnings_html = ""
    if warning_items:
        warnings_html = (
            "<section><h2>Warnings</h2><ul>"
            + "".join(f"<li>{html.escape(item)}</li>" for item in warning_items)
            + "</ul></section>"
        )

    selection_direction = "lowest" if contract["op"] in metric_contract.MINIMIZING_OPERATORS else "highest"
    best_summary = (
        f"{html.escape(best_label)} ({selection_direction} value under operator "
        f"{html.escape(contract['op'])})"
        if best_label
        else "none yet"
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PAS DEFT Loop Report</title>
<style>
:root {{ color-scheme: light dark; --bg:#f5f7fa; --card:#fff; --text:#17202a; --muted:#5d6d7e; --line:#d5dbe3; --accent:#2563eb; --good:#137333; --bad:#b42318; --warn:#8a4b08; }}
@media (prefers-color-scheme:dark) {{ :root {{ --bg:#111827; --card:#1f2937; --text:#f3f4f6; --muted:#b8c0cc; --line:#465164; --accent:#8ab4ff; --good:#76d58a; --bad:#ff8f87; --warn:#ffd080; }} }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,-apple-system,sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:24px; }} h1 {{ margin:0 0 4px; }} h2 {{ margin:0 0 12px; font-size:18px; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:10px; margin:14px 0; padding:16px; }}
.banner {{ border-left:7px solid var(--accent); }} .banner.failed {{ border-left-color:var(--bad); }} .banner.success {{ border-left-color:var(--good); }} .banner.neutral {{ border-left-color:var(--warn); }}
.status {{ font-size:22px; font-weight:750; }} .muted,.empty {{ color:var(--muted); }} .meta {{ display:flex; gap:16px; flex-wrap:wrap; margin-top:8px; }}
.table-wrap {{ overflow:auto; }} table {{ width:100%; border-collapse:collapse; }} th,td {{ border-bottom:1px solid var(--line); padding:8px; text-align:left; vertical-align:top; }} th {{ color:var(--muted); font-weight:650; white-space:nowrap; }}
code {{ overflow-wrap:anywhere; }} a {{ color:var(--accent); }} ul.paths {{ margin:8px 0 0; padding-left:20px; }} .field,.best {{ font-weight:700; }} .best,.path-ok {{ color:var(--good); }} .path-missing {{ color:var(--bad); font-weight:650; }}
footer {{ color:var(--muted); margin:18px 2px; }}
</style>
</head>
<body><main>
<section class="banner {status_class}">
  <h1>PAS DEFT Loop Report</h1>
  <div class="status">{html.escape(status)}</div>
  <div>{html.escape(status_detail)}</div>
  <div class="meta"><span>trigger: <code>{html.escape(trigger)}</code></span><span>audit: <code>{html.escape(str(audit_report.get('status')))}</code></span><span>state as of: <code>{html.escape(state_as_of)}</code></span></div>
</section>
<section><h2>Run contract</h2>
  <p><strong>{html.escape(contract['metric_name'])}</strong> ({html.escape(contract['query_type'])}) {html.escape(contract['op'])} {html.escape(target)}. Best: {best_summary}.</p>
  {_table(["Field", "Value"], config_rows, empty="No run configuration recorded.")}
</section>
<section><h2>KPI trend</h2>
  {_table(["Iteration", "Value", "Δ baseline", "Δ previous", "Gate", "Evidence"], kpi_rows, empty="No successfully committed evaluate result yet.")}
</section>
<section><h2>Iterations and evidence</h2>
  {_table(["Iteration", "Status", "Last stage", "KPI", "Artifacts"], iteration_rows, empty="No iteration state recorded.")}
</section>
<section><h2>Stage timeline</h2>
  {_table(["Seq", "Iteration", "Stage", "Status", "Duration", "Summary"], timeline_rows, empty="No stage events committed yet.")}
</section>
<section><h2>Canonical evidence</h2><ul class="paths">{canonical_html}</ul></section>
{warnings_html}
<footer>Rendered from audited <code>deft_state.json</code> and <code>loop_log.jsonl</code>. No report value is completion evidence by itself.</footer>
</main></body></html>
"""
    return document, status, best_label


def _write_atomic(path: pathlib.Path, content: str) -> int:
    encoded = content.encode("utf-8")
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return len(encoded)


def render(results_dir: pathlib.Path, trigger: str) -> tuple[pathlib.Path, int, str, str | None]:
    results_dir = results_dir.expanduser().resolve()
    _audited_report(results_dir)
    snapshot = _canonical_snapshot(results_dir)
    state, entries = _parse_snapshot(snapshot)
    audit_report = _audited_report(results_dir)
    if _canonical_snapshot(results_dir) != snapshot:
        raise ReportError("canonical state changed during audit; rerun the renderer")
    if len(entries) != audit_report["log_entries"]:
        raise ReportError("audit log count does not match the stable snapshot")
    if trigger == "loop-end" and not audit_report["terminal"]:
        raise ReportError("loop-end render requires a terminal loop_stop event")
    if trigger == "iteration-complete":
        completed_labels = {
            str(entry.get("iteration"))
            for entry in entries
            if entry.get("stage") == "evaluate"
            and entry.get("status") == "ok"
            and re.fullmatch(r"iter[1-9][0-9]*", str(entry.get("iteration")))
        }
        iterations = state.get("iterations")
        if not completed_labels or not isinstance(iterations, dict) or not any(
            isinstance(iterations.get(label), dict)
            and iterations[label].get("status") == "complete"
            for label in completed_labels
        ):
            raise ReportError(
                "iteration-complete render requires a successfully committed "
                "iterN/evaluate stage"
            )

    document, status, best_label = _render_html(
        results_dir=results_dir,
        trigger=trigger,
        state=state,
        entries=entries,
        audit_report=audit_report,
    )
    if _canonical_snapshot(results_dir) != snapshot:
        raise ReportError("canonical state changed while rendering; rerun the renderer")
    output = results_dir / REPORT_NAME
    size = _write_atomic(output, document)
    return output, size, status, best_label


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--trigger",
        required=True,
        choices=("iteration-complete", "loop-end"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output, size, status, best_label = render(args.results_dir, args.trigger)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"render_deft_report: {exc}", file=sys.stderr)
        return 2
    best = best_label or "none"
    print(
        f"render_deft_report: wrote {output.name} "
        f"({size} bytes, status={status}, best={best})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
