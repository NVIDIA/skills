#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the NVIDIA-styled Cosmos3 DEFT AOI report from disk evidence."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html as html_lib
import json
import math
import os
import pathlib
import re
import sys
import tempfile
from typing import Any

from metric_contract import (
    contract_from_state,
    pick_best,
    render_target,
    result_from_iteration,
    result_passes,
)


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE_PATH = SKILL_ROOT / "references" / "DEFT_Loop_Report.html"
REPORT_NAME = "DEFT_Loop_Report.html"


def _escape(value: Any) -> str:
    return html_lib.escape(str(value), quote=True)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "not available"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            return "not available"
        if number.is_integer():
            return f"{int(number):,}"
        return f"{number:.{digits}g}"
    text = str(value)
    return _escape(text) if text else "not available"


def _label_key(label: str) -> tuple[int, int]:
    if label == "baseline":
        return (0, 0)
    match = re.fullmatch(r"iter([1-9][0-9]*)", label)
    return (1, int(match.group(1))) if match else (2, 0)


def _read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_json(path_value: Any) -> Any:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = pathlib.Path(path_value)
    if not path.is_file():
        return None
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def _json_records(path_value: Any) -> list[dict[str, Any]]:
    payload = _optional_json(path_value)
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _label_counts(records: list[dict[str, Any]]) -> tuple[int, int]:
    ok = ng = 0
    for record in records:
        conversations = record.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            continue
        final = conversations[-1]
        if not isinstance(final, dict):
            continue
        value = str(final.get("value", "")).strip()
        if value == "OK":
            ok += 1
        elif value == "NG":
            ng += 1
    return ok, ng


def _human_prompt(record: dict[str, Any]) -> str | None:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return None
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("from", "")).strip().lower() not in {"human", "user"}:
            continue
        value = turn.get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _prompt_examples(state: dict[str, Any], *, limit: int = 3) -> str:
    config = state.get("config", {})
    if not isinstance(config, dict):
        config = {}
    annotations = config.get("annotations", {})
    if not isinstance(annotations, dict):
        annotations = {}

    examples: dict[str, dict[str, Any]] = {}
    for role in ("proxy", "benchmark", "mining"):
        for record in _json_records(annotations.get(role)):
            prompt = _human_prompt(record)
            if prompt is None:
                continue
            example = examples.setdefault(prompt, {"roles": [], "records": 0})
            if role not in example["roles"]:
                example["roles"].append(role)
            example["records"] += 1

    if not examples:
        return (
            '<div class="notice warn"><strong>Prompt examples are not available yet.</strong> '
            "They appear after the recorded annotation files are staged.</div>"
        )

    rows: list[str] = []
    for prompt, metadata in list(examples.items())[:limit]:
        roles = " · ".join(str(role).title() for role in metadata["roles"])
        record_count = int(metadata["records"])
        preview = prompt if len(prompt) <= 600 else prompt[:599] + "…"
        truncation = (
            '<span class="prompt-truncated">Preview truncated at 600 characters</span>'
            if len(prompt) > 600
            else ""
        )
        rows.append(
            '<div class="prompt-example">'
            '<div class="prompt-meta">'
            f'<span class="prompt-role">{_escape(roles)}</span>'
            f'<span class="badge muted">{record_count:,} RECORDS</span>'
            "</div>"
            f'<div class="prompt-text">{_escape(preview)}</div>'
            f"{truncation}"
            '<div class="prompt-response"><span>Exact assistant output</span>'
            '<code>OK</code><span>or</span><code>NG</code></div>'
            "</div>"
        )
    return "\n".join(rows)


def _csv_rows(path_value: Any) -> list[dict[str, str]]:
    if not isinstance(path_value, str) or not path_value:
        return []
    path = pathlib.Path(path_value)
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return []


def _metric_candidates(
    state: dict[str, Any], contract: dict[str, Any]
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    iterations = state.get("iterations", {})
    if not isinstance(iterations, dict):
        raise ValueError("state.iterations must be an object")
    result: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for label in sorted(iterations, key=_label_key):
        phase = iterations[label]
        if not isinstance(phase, dict):
            continue
        metric = result_from_iteration(phase, contract)
        if metric is not None:
            result.append((label, phase, metric))
    return result


def _duration(value: Any) -> str:
    try:
        total = int(value)
    except (TypeError, ValueError):
        return "not recorded"
    if total <= 0:
        return "not recorded"
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _metric_summary_html(
    result: dict[str, Any] | None, contract: dict[str, Any]
) -> str:
    if result is None:
        return "not available"
    detail = (
        f"{_escape(contract['display_name'])} = {_fmt(result.get('value'))}"
        f"{_escape(str(contract.get('unit', '')))}"
    )
    values = result.get("constraints", {})
    if isinstance(values, dict):
        constraints: list[str] = []
        for constraint in contract.get("constraints", []):
            value = values.get(constraint.get("name"))
            if value is None:
                continue
            constraints.append(
                f"{_escape(constraint.get('display_name', constraint.get('name')))} "
                f"{_fmt(value)}{_escape(str(constraint.get('unit', '')))}"
            )
        if constraints:
            detail += " @ " + ", ".join(constraints)
    return detail


def _benchmark_summary_html(state: dict[str, Any]) -> str:
    config = state.get("config", {})
    if not isinstance(config, dict):
        config = {}
    annotations = config.get("annotations", {})
    if not isinstance(annotations, dict):
        annotations = {}
    records = _json_records(annotations.get("benchmark"))
    if not records:
        return "Benchmark dataset not available"
    ok, ng = _label_counts(records)
    return f"Benchmark: {len(records):,} rows: {ok:,} OK / {ng:,} NG"


def _recorded_duration_summary(
    entries: list[dict[str, Any]], completed_iterations: int
) -> str:
    timed = [entry for entry in entries if entry.get("stage") != "loop_stop"]
    durations = [
        int(entry["duration_sec"])
        for entry in timed
        if isinstance(entry.get("duration_sec"), int)
        and not isinstance(entry.get("duration_sec"), bool)
        and int(entry["duration_sec"]) > 0
    ]
    if not durations:
        return f"{completed_iterations} iterations · duration not recorded"
    total = sum(durations)
    missing = len(timed) - len(durations)
    if missing:
        return (
            f"{completed_iterations} iterations · {_duration(total)} recorded · "
            f"{missing} stage duration{'s' if missing != 1 else ''} missing"
        )
    if completed_iterations <= 0:
        return f"0 iterations · {_duration(total)} recorded"
    average = max(1, round(total / completed_iterations))
    return (
        f"{completed_iterations} iters × ~{_duration(average)} = "
        f"{_duration(total)} total time"
    )


def _sdg_summary_html(
    state: dict[str, Any], entries: list[dict[str, Any]]
) -> str:
    iterations = state.get("iterations", {})
    if not isinstance(iterations, dict):
        return "not available"
    generated: list[int] = []
    for label in sorted(iterations, key=_label_key):
        if label == "baseline" or not isinstance(iterations[label], dict):
            continue
        phase = iterations[label]
        if phase.get("anomalygen_skipped"):
            generated.append(0)
        elif phase.get("anomalygen_sdg_csv"):
            generated.append(len(_csv_rows(phase.get("anomalygen_sdg_csv"))))
    if not generated:
        return "not available"
    total = sum(generated)
    average = round(total / len(generated))
    detail = f"{average:,} images/iter · {total:,} total"
    durations = [
        int(entry["duration_sec"])
        for entry in entries
        if entry.get("stage") == "anomalygen"
        and isinstance(entry.get("duration_sec"), int)
        and not isinstance(entry.get("duration_sec"), bool)
        and int(entry["duration_sec"]) > 0
    ]
    if durations:
        detail += f" · {_duration(round(sum(durations) / len(durations)))} avg SDG time/iter"
    else:
        detail += " · SDG duration not recorded"
    return detail


def _assemble_counts(phase: dict[str, Any]) -> tuple[int | None, int | None]:
    summary = _optional_json(phase.get("assemble_summary"))
    total = _first_number(summary, ("output_records",))
    unique = None
    if isinstance(summary, dict):
        unique_targets = summary.get("unique_target_images")
        if isinstance(unique_targets, dict):
            unique = _first_number(unique_targets, ("new_after_dedup",))
    if total is None and phase.get("combined_training_json"):
        total = len(_json_records(phase.get("combined_training_json")))
    return (
        int(total) if isinstance(total, (int, float)) else None,
        int(unique) if isinstance(unique, (int, float)) else None,
    )


def _growth_rows(state: dict[str, Any]) -> str:
    rows = [
        '<tr><td><strong>Baseline</strong></td><td class="num">0</td>'
        '<td class="num">0</td><td class="num">0</td><td class="num">0</td>'
        '<td class="num">—</td></tr>'
    ]
    previous_total = 0
    iterations = state.get("iterations", {})
    if not isinstance(iterations, dict):
        return "\n".join(rows)
    for label in sorted(iterations, key=_label_key):
        phase = iterations[label]
        if label == "baseline" or not isinstance(phase, dict):
            continue
        mining_summary = _optional_json(phase.get("mining_summary"))
        raw = _first_number(
            mining_summary,
            ("raw_candidates", "candidate_count", "input_rows", "raw_rows"),
        )
        if raw is None and phase.get("data_mining_skipped"):
            raw = 0
        sdg_generated = (
            0
            if phase.get("anomalygen_skipped")
            else (
                len(_csv_rows(phase.get("anomalygen_sdg_csv")))
                if phase.get("anomalygen_sdg_csv")
                else None
            )
        )
        total, _batch_unique = _assemble_counts(phase)
        delta = total - previous_total if total is not None else None
        new_unique = delta if delta is not None and delta >= 0 else None
        delta_html = (
            "—"
            if delta is None
            else (f"+{delta:,}" if delta > 0 else f"{delta:,}")
        )
        rows.append(
            f'<tr><td><strong>{_escape(label.title())}</strong></td>'
            f'<td class="num">{_fmt(raw)}</td>'
            f'<td class="num">{_fmt(sdg_generated)}</td>'
            f'<td class="num">{_fmt(new_unique)}</td>'
            f'<td class="num">{_fmt(total)}</td>'
            f'<td class="num">{delta_html}</td></tr>'
        )
        if total is not None:
            previous_total = total
    return "\n".join(rows)


def _run_summary_rows(
    state: dict[str, Any],
    contract: dict[str, Any],
    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]],
    entries: list[dict[str, Any]],
) -> str:
    config = state.get("config", {})
    if not isinstance(config, dict):
        config = {}
    training = config.get("training", {})
    if not isinstance(training, dict):
        training = {}
    baseline = next(
        (result for label, _, result in candidates if label == "baseline"), None
    )
    end_label, end_result = (
        (candidates[-1][0], candidates[-1][2])
        if candidates
        else (None, None)
    )
    completed = sum(1 for label, _, _ in candidates if label != "baseline")
    gpu = f"{_fmt(training.get('num_gpus'))}x {_fmt(training.get('gpu_model'))}"
    baseline_detail = _metric_summary_html(baseline, contract)
    if baseline is not None:
        baseline_detail += f" ({_benchmark_summary_html(state)})"
    end_detail = _metric_summary_html(end_result, contract)
    if end_label is not None and end_result is not None:
        end_detail += f" ({_escape(end_label.title())})"
    details = (
        ("Prompt/Goal", f"Run DEFT loop · KPI: {_escape(render_target(contract))}"),
        ("Model", f"NVIDIA TAO Cosmos Reason 3 · {_fmt(config.get('base_model'))}"),
        ("GPU", gpu),
        ("Baseline (pre-DEFT)", baseline_detail),
        ("Data Routing", "AnomalyGen SDG &amp; k-NN Mining"),
        ("Iterations × Time", _escape(_recorded_duration_summary(entries, completed))),
        ("SDG Images", _escape(_sdg_summary_html(state, entries))),
        ("End Result", end_detail),
    )
    return "\n".join(
        f'<tr><td><strong>{_escape(item)}</strong></td><td>{detail}</td></tr>'
        for item, detail in details
    )


def _dataset_rows(state: dict[str, Any]) -> str:
    config = state.get("config", {})
    if not isinstance(config, dict):
        config = {}
    annotations = config.get("annotations", {})
    if not isinstance(annotations, dict):
        annotations = {}
    purposes = {
        "proxy": "RCCA and routing only; never gates the loop",
        "benchmark": "Frozen stop-gate evaluation only",
        "mining": "Candidate source for mined training pairs",
    }
    rows: list[str] = []
    for role in ("proxy", "benchmark", "mining"):
        path = annotations.get(role)
        records = _json_records(path)
        ok, ng = _label_counts(records)
        rows.append(
            f'<tr><td><strong>{_escape(role.title())}</strong></td><td>{_escape(purposes[role])}</td>'
            f'<td class="num">{len(records):,}</td><td class="num">{ok:,} / {ng:,}</td><td class="path">{_fmt(path)}</td></tr>'
        )
    iterations = state.get("iterations", {})
    if isinstance(iterations, dict):
        for label in sorted(iterations, key=_label_key):
            if label == "baseline" or not isinstance(iterations[label], dict):
                continue
            phase = iterations[label]
            producers = (
                ("Mined real pairs", phase.get("mined_sharegpt_json")),
                ("AnomalyGen synthetic pairs", phase.get("anomalygen_sharegpt_json")),
            )
            for producer, path in producers:
                if not path and producer.startswith("AnomalyGen") and phase.get("anomalygen_skipped"):
                    rows.append(
                        f'<tr><td><strong>{_escape(label.title())} · AnomalyGen</strong></td><td>Documented skip</td><td class="num">0</td><td class="num">0 / 0</td><td>skipped by committed stage</td></tr>'
                    )
                    continue
                if not path:
                    continue
                records = _json_records(path)
                ok, ng = _label_counts(records)
                rows.append(
                    f'<tr><td><strong>{_escape(label.title())} · {_escape(producer)}</strong></td><td>Generated Train producer</td>'
                    f'<td class="num">{len(records):,}</td><td class="num">{ok:,} / {ng:,}</td><td class="path">{_fmt(path)}</td></tr>'
                )
    return "\n".join(rows)


def _metric_rows(
    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]],
    contract: dict[str, Any],
    best_label: str | None,
) -> str:
    rows: list[str] = []
    for label, _, result in candidates:
        metrics = result.get("metrics", {})
        confusion = result.get("confusion", {})
        constraints = result.get("constraints", {})
        if not isinstance(metrics, dict):
            metrics = {}
        if not isinstance(confusion, dict):
            confusion = {}
        if not isinstance(constraints, dict):
            constraints = {}
        passed, failures = result_passes(contract, result)
        verdict = (
            '<span class="badge good">MET</span>'
            if passed
            else f'<span class="badge warn">GAP · {_escape(", ".join(failures))}</span>'
        )
        rows.append(
            f'<tr class="{"best" if label == best_label else ""}"><td><strong>{_escape(label.title())}</strong></td><td>Frozen Benchmark</td>'
            f'<td class="num">{_fmt(metrics.get("accuracy"))}</td><td class="num">{_fmt(metrics.get("recall_ng", result.get("value") if contract["name"] == "recall_ng" else None))}</td>'
            f'<td class="num">{_fmt(metrics.get("precision_ng"))}</td><td class="num">{_fmt(metrics.get("f1_ng"))}</td>'
            f'<td class="num">{_fmt(confusion.get("fn_ng_to_ok_false_accept"))}</td><td class="num">{_fmt(confusion.get("fp_ok_to_ng_false_reject"))}</td>'
            f'<td class="num">{_fmt(constraints.get("unknown_predictions"))}</td><td>{verdict}</td></tr>'
        )
    return "\n".join(rows) or '<tr><td colspan="10">No frozen Benchmark metric has been committed yet.</td></tr>'


def _stage_rows(entries: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for entry in entries:
        status = str(entry.get("status", "unknown"))
        badge = {"ok": "good", "skipped": "muted"}.get(status, "error")
        rows.append(
            f'<tr><td class="num">{_fmt(entry.get("seq"))}</td><td>{_fmt(entry.get("iter"))}</td><td>{_fmt(entry.get("stage"))}</td>'
            f'<td><span class="badge {badge}">{_escape(status.upper())}</span></td><td class="num">{_escape(_duration(entry.get("duration_sec")))}</td>'
            f'<td>{_fmt(entry.get("summary"))}</td></tr>'
        )
    return "\n".join(rows) or '<tr><td colspan="6">No stage has been committed yet.</td></tr>'


def _first_number(payload: Any, names: tuple[str, ...]) -> Any:
    if not isinstance(payload, dict):
        return None
    for name in names:
        value = payload.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    for value in payload.values():
        if isinstance(value, dict):
            found = _first_number(value, names)
            if found is not None:
                return found
    return None


def _defect_breakdown(rows: list[dict[str, str]]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        raw = row.get("defect_type") or row.get("anomaly_type") or row.get("reconstructed_image") or "unknown"
        name = pathlib.Path(raw).stem.split("_")[0].split("+")[-1] or "unknown"
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return "not available"
    return ", ".join(f"{_escape(name)}: {count}" for name, count in sorted(counts.items()))


def _augmentation_rows(state: dict[str, Any]) -> str:
    config = state.get("config", {})
    if not isinstance(config, dict):
        config = {}
    anomalygen = config.get("anomalygen", {})
    if not isinstance(anomalygen, dict):
        anomalygen = {}
    requested = anomalygen.get("num_SDG")
    iterations = state.get("iterations", {})
    if not isinstance(iterations, dict):
        return '<tr><td colspan="9">No augmentation has been committed yet.</td></tr>'
    rows: list[str] = []
    previous_train = 0
    for label in sorted(iterations, key=_label_key):
        if label == "baseline" or not isinstance(iterations[label], dict):
            continue
        phase = iterations[label]
        sdg_rows = _csv_rows(phase.get("anomalygen_sdg_csv"))
        generated = 0 if phase.get("anomalygen_skipped") else (len(sdg_rows) if sdg_rows else None)
        summary = _optional_json(phase.get("mining_summary"))
        raw = _first_number(summary, ("raw_candidates", "candidate_count", "input_rows", "raw_rows"))
        kept = phase.get("mining_mined_count")
        if kept is None:
            kept = _first_number(summary, ("kept_rows", "kept_count", "output_rows"))
        train_count, new_unique = _assemble_counts(phase)
        if new_unique is None and train_count is not None:
            new_unique = max(train_count - previous_train, 0)
        if train_count is not None:
            previous_train = train_count
        allocated = phase.get("anomalygen_amp_allocated")
        if not isinstance(allocated, int) or isinstance(allocated, bool):
            allocation = _optional_json(phase.get("anomalygen_allocation_json"))
            if allocation and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for value in allocation.values()
            ):
                allocated = sum(allocation.values())
            else:
                allocated = None
        rows.append(
            f'<tr><td><strong>{_escape(label.title())}</strong></td><td class="num">{_fmt(requested)}</td><td class="num">{_fmt(allocated)}</td>'
            f'<td class="num">{_fmt(generated)}</td><td>{_defect_breakdown(sdg_rows)}</td><td class="num">{_fmt(raw)}</td>'
            f'<td class="num">{_fmt(kept)}</td><td class="num">{_fmt(new_unique)}</td><td class="num">{_fmt(train_count)}</td></tr>'
        )
    return "\n".join(rows) or '<tr><td colspan="9">No augmentation has been committed yet.</td></tr>'


def _terminal_iteration(label: str, phase: dict[str, Any], state: dict[str, Any], contract: dict[str, Any]) -> bool:
    result = result_from_iteration(phase, contract)
    if result is None:
        return False
    passed = result_passes(contract, result)[0]
    match = re.fullmatch(r"iter([1-9][0-9]*)", label)
    reached_max = bool(match and int(match.group(1)) >= int(state.get("max_iterations", 0)))
    return passed or reached_max


def _artifact_rows(state: dict[str, Any], contract: dict[str, Any]) -> str:
    config = state.get("config", {})
    if not isinstance(config, dict):
        config = {}
    rows = [
        f'<tr><td>Baseline</td><td>Base model reference</td><td><span class="badge good">RECORDED</span></td><td class="path">{_fmt(config.get("base_model"))}</td></tr>'
    ]
    fields = (
        ("best_ckpt_path", "Checkpoint"),
        ("benchmark_results_json", "Benchmark results"),
        ("benchmark_metrics_summary", "Benchmark metrics"),
        ("proxy_results_json", "Proxy results"),
        ("proxy_gaps_summary", "Proxy RCCA"),
        ("mining_targets_json", "Routing targets"),
        ("anomalygen_sdg_csv", "AnomalyGen SDG_result.csv"),
        ("anomalygen_allocation_json", "AnomalyGen AMP allocation"),
        ("anomalygen_sharegpt_json", "AnomalyGen ShareGPT"),
        ("mining_mined_parquet", "Mined candidates"),
        ("mining_summary", "Mining filter summary"),
        ("combined_training_json", "Assembled Train JSON"),
        ("validation_report", "Validation report"),
    )
    iterations = state.get("iterations", {})
    if not isinstance(iterations, dict):
        return "\n".join(rows)
    for label in sorted(iterations, key=_label_key):
        phase = iterations[label]
        if not isinstance(phase, dict):
            continue
        terminal = _terminal_iteration(label, phase, state, contract)
        for field, display in fields:
            value = phase.get(field)
            if value:
                path = pathlib.Path(str(value))
                badge = "good" if path.exists() else "warn"
                state_text = "AVAILABLE" if path.exists() else "RECORDED"
                rows.append(
                    f'<tr><td>{_escape(label.title())}</td><td>{_escape(display)}</td><td><span class="badge {badge}">{state_text}</span></td><td class="path">{_fmt(value)}</td></tr>'
                )
            elif field in {"proxy_results_json", "proxy_gaps_summary"} and terminal:
                rows.append(
                    f'<tr><td>{_escape(label.title())}</td><td>{_escape(display)}</td><td><span class="badge muted">NOT RUN</span></td><td>not run (terminal iteration)</td></tr>'
                )
        if phase.get("anomalygen_skipped"):
            rows.append(
                f'<tr><td>{_escape(label.title())}</td><td>AnomalyGen</td><td><span class="badge muted">SKIPPED</span></td><td>documented skip: driving Proxy RCCA recorded zero false accepts</td></tr>'
            )
    return "\n".join(rows)


def _warnings(entries: list[dict[str, Any]]) -> str:
    messages: list[tuple[str, str]] = []
    for entry in entries:
        if entry.get("status") == "error":
            messages.append(
                ("error", f"{entry.get('iter')}/{entry.get('stage')}: {entry.get('summary')}")
            )
    if not messages:
        return '<div class="notice"><strong>No committed hard stops.</strong> No error event is recorded in deft_state.json.</div>'
    return "\n".join(
        f'<div class="notice {kind}"><strong>{"Hard stop" if kind == "error" else "Warning"}:</strong> {_escape(message)}</div>'
        for kind, message in messages
    )


def _metric_chart(
    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]],
    contract: dict[str, Any],
) -> str:
    if not candidates:
        return '<div class="notice">Trend appears after the first frozen Benchmark metric is committed.</div>'
    width, height = 720, 240
    left, right, top, bottom = 54, 18, 18, 44
    values = [float(result["value"]) for _, _, result in candidates]
    target = float(contract["target"])
    low, high = min([target, *values]), max([target, *values])
    padding = max((high - low) * 0.15, abs(high or 1) * 0.06, 0.02)
    low -= padding
    high += padding
    chart_w, chart_h = width - left - right, height - top - bottom

    def x(index: int) -> float:
        return left + chart_w * index / max(len(values) - 1, 1)

    def y(value: float) -> float:
        return top + (high - value) / (high - low) * chart_h

    grid: list[str] = []
    for index in range(5):
        value = low + (high - low) * index / 4
        ypos = y(value)
        grid.append(f'<line x1="{left}" y1="{ypos:.1f}" x2="{width-right}" y2="{ypos:.1f}" stroke="rgba(255,255,255,.08)"/>')
        grid.append(f'<text x="{left-8}" y="{ypos+4:.1f}" fill="#858585" font-size="11" text-anchor="end">{_escape(f"{value:.3g}")}</text>')
    target_y = y(target)
    points = " ".join(f"{x(index):.1f},{y(value):.1f}" for index, value in enumerate(values))
    marks: list[str] = []
    for index, ((label, _, result), value) in enumerate(zip(candidates, values)):
        xpos, ypos = x(index), y(value)
        passed = result_passes(contract, result)[0]
        color = "#76b900" if passed else "#ffbc01"
        marks.append(f'<circle cx="{xpos:.1f}" cy="{ypos:.1f}" r="6" fill="{color}" stroke="#232323" stroke-width="3"/>')
        marks.append(f'<text x="{xpos:.1f}" y="{ypos-13:.1f}" fill="{color}" font-size="12" font-weight="700" text-anchor="middle">{_escape(f"{value:.4g}")}</text>')
        marks.append(f'<text x="{xpos:.1f}" y="{height-12}" fill="#c2c2c2" font-size="12" text-anchor="middle">{_escape(label.title())}</text>')
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Benchmark KPI trend">'
        + "".join(grid)
        + f'<line x1="{left}" y1="{target_y:.1f}" x2="{width-right}" y2="{target_y:.1f}" stroke="#c2262d" stroke-width="1.5" stroke-dasharray="6 5"/>'
        + f'<text x="{width-right}" y="{target_y-7:.1f}" fill="#ef8085" font-size="11" text-anchor="end">Target {_escape(f"{target:.4g}")}</text>'
        + (f'<polyline points="{points}" fill="none" stroke="#76b900" stroke-width="2.5"/>' if len(values) > 1 else "")
        + "".join(marks)
        + "</svg>"
    )


def _atomic_write(path: pathlib.Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def render(results_dir: pathlib.Path) -> pathlib.Path:
    results_dir = results_dir.expanduser().resolve()
    state_path = results_dir / "deft_state.json"
    if not state_path.is_file():
        raise ValueError(f"state file not found: {state_path}")
    state = _read_json(state_path)
    if not isinstance(state, dict):
        raise ValueError("deft_state.json root must be an object")
    contract = contract_from_state(state)
    raw_entries = state.get("events", [])
    if not isinstance(raw_entries, list):
        raise ValueError("state.events must be an array")
    entries = [entry for entry in raw_entries if isinstance(entry, dict)]
    stored_status = str(state.get("status", "")).lower()
    if stored_status not in {"in_progress", "complete", "failed"}:
        stored_status = (
            "failed"
            if any(entry.get("status") == "error" for entry in entries)
            else (
                "complete"
                if entries and entries[-1].get("stage") == "loop_stop"
                else "in_progress"
            )
        )
    candidates = _metric_candidates(state, contract)
    best_label: str | None = None
    best_result: dict[str, Any] | None = None
    if candidates:
        best_label, _, best_result = pick_best(candidates, contract)
    metric_passed = bool(best_result and result_passes(contract, best_result)[0])
    terminal = stored_status in {"complete", "failed"}
    run_status = stored_status.upper()

    if run_status == "FAILED":
        banner = (
            '<div class="kpi-banner error"><div class="icon">!</div>'
            '<div class="content"><div class="title">RUN ENDED AT A HARD STOP</div>'
            '<div class="body">Review the latest error event in deft_state.json before retrying the failed stage.</div>'
            "</div></div>"
        )
    elif terminal and metric_passed:
        label = best_label.title() if best_label else "Best result"
        banner = (
            '<div class="kpi-banner"><div class="icon">✓</div>'
            '<div class="content"><div class="title">KPI MET</div>'
            f'<div class="body">{_escape(label)} satisfies the frozen Benchmark contract.</div>'
            "</div></div>"
        )
    else:
        banner = ""

    config = state.get("config", {})
    if not isinstance(config, dict):
        config = {}
    training = config.get("training", {})
    if not isinstance(training, dict):
        training = {}
    num_gpus = training.get("num_gpus")
    num_nodes = training.get("num_nodes")
    compute_shape = (
        f"{_fmt(num_nodes)} node(s) · {_fmt(num_gpus)} GPU(s) · "
        f"{_fmt(training.get('gpu_model'))}"
    )
    completed_iterations = sum(1 for label, _, _ in candidates if label != "baseline")
    benchmark_hash = (
        config.get("evaluation", {}).get("benchmark", {}).get("sha256")
        if isinstance(config.get("evaluation"), dict)
        and isinstance(config.get("evaluation", {}).get("benchmark"), dict)
        else None
    )
    last_ts = entries[-1].get("ts") if entries else "not available"
    best_metric = (
        f"{_fmt(best_result['value'])} ({_escape(contract['display_name'])})"
        if best_result
        else "not available"
    )
    values = {
        "GENERATED_DATE": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "KPI_TARGET": _escape(render_target(contract)),
        "ITERATIONS_RUN": str(completed_iterations),
        "MAX_ITERATIONS": _fmt(state.get("max_iterations")),
        "RUN_STATUS": _escape(run_status),
        "KPI_BANNER_HTML": banner,
        "PLATFORM": _fmt(config.get("platform")),
        "MODEL": _fmt(config.get("base_model")),
        "ANNOTATION_MODE": _fmt(config.get("annotation_mode", "bare_okng")),
        "COMPUTE_SHAPE": compute_shape,
        "STARTED_AT": _fmt(state.get("started_at")),
        "FINISHED_AT": _fmt(last_ts),
        "BEST_ITERATION": _escape(best_label.title() if best_label else "not available"),
        "BEST_METRIC": best_metric,
        "RUN_SUMMARY_ROWS_HTML": _run_summary_rows(
            state, contract, candidates, entries
        ),
        "GROWTH_ROWS_HTML": _growth_rows(state),
        "DATASET_ROWS_HTML": _dataset_rows(state),
        "BENCHMARK_HASH": _fmt(benchmark_hash),
        "PROMPT_EXAMPLES_HTML": _prompt_examples(state),
        "METRIC_CHART_HTML": _metric_chart(candidates, contract),
        "AUGMENTATION_ROWS_HTML": _augmentation_rows(state),
        "METRIC_ROWS_HTML": _metric_rows(candidates, contract, best_label),
        "STAGE_ROWS_HTML": _stage_rows(entries),
        "ARTIFACT_ROWS_HTML": _artifact_rows(state, contract),
        "WARNING_ROWS_HTML": _warnings(entries),
    }
    rendered = TEMPLATE_PATH.read_text(encoding="utf-8")
    for name, value in values.items():
        rendered = rendered.replace("{{ " + name + " }}", str(value))
    remaining = sorted(set(re.findall(r"\{\{\s+[A-Z0-9_]+\s+\}\}", rendered)))
    if remaining:
        raise ValueError("unfilled report placeholders: " + ", ".join(remaining))
    required = (
        "NVIDIA TAO · DEFT AOI",
        "Run Configuration &amp; Outcome",
        "Training Set Growth",
        "Benchmark KPI Trend",
        "Dataset Isolation",
        "Prompt Examples",
        "Iteration Metrics",
        "Pipeline Execution",
        "Augmentation Volume",
        "Artifacts",
        "Hard Stops / Warnings",
        "--nvidia-green: #76b900",
    )
    missing = [token for token in required if token not in rendered]
    if missing:
        raise ValueError("rendered report is missing required content: " + ", ".join(missing))
    output = results_dir / REPORT_NAME
    _atomic_write(output, rendered)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--require-terminal",
        action="store_true",
        help="Refuse to render unless deft_state.json records a terminal status.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        state = _read_json(
            args.results_dir.expanduser().resolve() / "deft_state.json"
        )
        if not isinstance(state, dict):
            raise ValueError("deft_state.json root must be an object")
        if args.require_terminal and state.get("status") not in {"complete", "failed"}:
            raise ValueError(
                "--require-terminal requested but deft_state.json is not terminal"
            )
        output = render(args.results_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"render_report: {exc}", file=sys.stderr)
        return 2
    print(f"render_report: wrote {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
