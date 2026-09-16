#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the NVIDIA-styled DEFT AOI HTML report from canonical disk state.

The renderer is intentionally deterministic and stdlib-first.  It is called by
``commit_stage.py`` after every successful commit, so report freshness no
longer depends on an agent remembering an end-of-loop rendering task.
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import html as html_lib
import io
import json
import math
import os
import pathlib
import re
import sys
import tempfile
from typing import Any, Iterable

from metric_contract import (
    MINIMIZING_OPERATORS,
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
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            return "—"
        if number.is_integer():
            return f"{int(number):,}"
        return f"{number:.{digits}g}"
    return _escape(value)


def _display_unit(unit: str) -> str:
    if not unit:
        return ""
    return unit if unit == "%" else f" {unit}"


def _json_for_html(value: Any) -> str:
    """Serialize JSON safely for an inline script element."""
    return (
        json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _label_key(label: str) -> tuple[int, int]:
    if label == "baseline":
        return (0, 0)
    match = re.fullmatch(r"iter([1-9][0-9]*)", label)
    return (1, int(match.group(1))) if match else (2, 0)


def _read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value:
        return None
    path = pathlib.Path(value)
    if not path.is_file():
        return None
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _csv_row_count(path_value: Any) -> int | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = pathlib.Path(path_value)
    if not path.is_file():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return None


def _csv_rows(path: pathlib.Path) -> list[dict[str, str]]:
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
    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for label in sorted(iterations, key=_label_key):
        phase = iterations[label]
        if not isinstance(phase, dict):
            continue
        result = result_from_iteration(phase, contract)
        if result is not None:
            candidates.append((label, phase, result))
    return candidates


def _chart_extent(values: Iterable[float], target: float) -> tuple[float, float, list[float]]:
    points = [float(target), *(float(value) for value in values)]
    low, high = min(points), max(points)
    if math.isclose(low, high):
        padding = max(abs(low) * 0.1, 0.1)
    else:
        padding = (high - low) * 0.15
    low -= padding
    high += padding
    steps = [low + (high - low) * index / 4 for index in range(5)]
    return low, high, [round(value, 6) for value in steps]


def _format_duration(seconds: Any) -> str:
    try:
        total = int(seconds)
    except (TypeError, ValueError):
        return "—"
    if total <= 0:
        return "not recorded"
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _metric_summary_html(
    result: dict[str, Any] | None, contract: dict[str, Any]
) -> str:
    if result is None:
        return "not available"
    detail = (
        f"{_escape(contract['display_name'])} = "
        f"{_fmt(result.get('value'))}{_escape(_display_unit(contract['unit']))}"
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
                f"{_fmt(value)}{_escape(_display_unit(str(constraint.get('unit', ''))))}"
            )
        if constraints:
            detail += " @ " + ", ".join(constraints)
    return detail


def _dataset_summary_html(state: dict[str, Any]) -> str:
    config = state.get("config", {})
    if not isinstance(config, dict):
        config = {}
    rows = _csv_rows(pathlib.Path(str(config.get("kpi_test_csv", ""))))
    if not rows:
        return "dataset not available"
    counts: dict[str, int] = {}
    for row in rows:
        label = str(
            row.get("label", row.get("object_name", "Unlabeled"))
        ).strip() or "Unlabeled"
        counts[label] = counts.get(label, 0) + 1
    ordered = sorted(
        counts.items(),
        key=lambda item: (item[0].upper() not in {"PASS", "OK"}, item[0].lower()),
    )
    breakdown = " / ".join(
        f"{count:,} {_escape(label)}" for label, count in ordered
    )
    return f"{len(rows):,} rows: {breakdown}"


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
            f"{completed_iterations} iterations · {_format_duration(total)} recorded · "
            f"{missing} stage duration{'s' if missing != 1 else ''} missing"
        )
    if completed_iterations <= 0:
        return f"0 iterations · {_format_duration(total)} recorded"
    average = max(1, round(total / completed_iterations))
    return (
        f"{completed_iterations} iters × ~{_format_duration(average)} = "
        f"{_format_duration(total)} total time"
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
        count = (
            0
            if phase.get("anomalygen_skipped")
            else _csv_row_count(phase.get("anomalygen_sdg_csv"))
        )
        if isinstance(count, int):
            generated.append(count)
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
        detail += f" · {_format_duration(round(sum(durations) / len(durations)))} avg SDG time/iter"
    else:
        detail += " · SDG duration not recorded"
    return detail


def _mining_raw_count(phase: dict[str, Any]) -> int | None:
    summary = phase.get("mining_summary")
    if not isinstance(summary, str) or not summary:
        return 0 if phase.get("data_mining_skipped") else None
    rows = _csv_rows(pathlib.Path(summary))
    if not rows:
        return None
    try:
        value = int(rows[0]["candidate_count"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _growth_rows(state: dict[str, Any]) -> str:
    config = state.get("config", {})
    if not isinstance(config, dict):
        config = {}
    baseline_total = _csv_row_count(config.get("training_csv"))
    rows = [
        '<tr><td><strong>Baseline</strong></td><td class="num">0</td>'
        '<td class="num">0</td><td class="num">0</td>'
        f'<td class="num">{_fmt(baseline_total)}</td>'
        '<td class="num">—</td></tr>'
    ]
    previous_total = baseline_total
    iterations = state.get("iterations", {})
    if not isinstance(iterations, dict):
        return "\n".join(rows)
    for label in sorted(iterations, key=_label_key):
        phase = iterations[label]
        if label == "baseline" or not isinstance(phase, dict):
            continue
        total = _csv_row_count(phase.get("combined_training_csv"))
        sdg_generated = (
            0
            if phase.get("anomalygen_skipped")
            else _csv_row_count(phase.get("anomalygen_sdg_csv"))
        )
        delta = (
            total - previous_total
            if isinstance(total, int) and isinstance(previous_total, int)
            else None
        )
        new_unique = delta if isinstance(delta, int) and delta >= 0 else None
        delta_html = (
            "—"
            if delta is None
            else (f"+{delta:,}" if delta > 0 else f"{delta:,}")
        )
        rows.append(
            f'<tr><td><strong>{_escape(label.title())}</strong></td>'
            f'<td class="num">{_fmt(_mining_raw_count(phase))}</td>'
            f'<td class="num">{_fmt(sdg_generated)}</td>'
            f'<td class="num">{_fmt(new_unique)}</td>'
            f'<td class="num">{_fmt(total)}</td>'
            f'<td class="num">{delta_html}</td></tr>'
        )
        if isinstance(total, int):
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
    baseline = next(
        (result for label, _, result in candidates if label == "baseline"), None
    )
    end_label, end_result = (
        (candidates[-1][0], candidates[-1][2])
        if candidates
        else (None, None)
    )
    completed = sum(1 for label, _, _ in candidates if label != "baseline")
    gpu = f"{_fmt(config.get('num_gpus'))}x {_fmt(config.get('gpu_model'))}"
    baseline_detail = _metric_summary_html(baseline, contract)
    if baseline is not None:
        baseline_detail += f" ({_dataset_summary_html(state)})"
    end_detail = _metric_summary_html(end_result, contract)
    if end_label is not None and end_result is not None:
        end_detail += f" ({_escape(end_label.title())})"
    details = (
        ("Prompt/Goal", f"Run DEFT loop · KPI: {_escape(render_target(contract))}"),
        ("Model", "NVIDIA TAO Visual ChangeNet classification"),
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


def _iteration_rows(
    state: dict[str, Any],
    contract: dict[str, Any],
    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]],
    best_label: str | None,
) -> tuple[str, str, str]:
    baseline_value = candidates[0][2]["value"] if candidates else None
    minimizes = contract["operator"] in MINIMIZING_OPERATORS
    table_rows: list[str] = []
    cards: list[str] = []
    pool_rows: list[str] = []
    for label, phase, result in candidates:
        value = float(result["value"])
        delta = None if baseline_value is None or label == "baseline" else value - float(baseline_value)
        delta_class = ""
        if delta is not None:
            improved = delta < 0 if minimizes else delta > 0
            delta_class = "pos" if improved else ("neg" if delta != 0 else "")
        training_rows = _csv_row_count(
            phase.get("combined_training_csv")
            or (state.get("config", {}) if isinstance(state.get("config"), dict) else {}).get("training_csv")
        )
        synthetic_rows = _csv_row_count(phase.get("anomalygen_sdg_csv")) or 0
        mined_rows = phase.get("mining_mined_count")
        if not isinstance(mined_rows, int):
            mined_rows = 0
        ratio = (
            synthetic_rows / training_rows * 100
            if isinstance(training_rows, int) and training_rows > 0
            else None
        )
        passed, failures = result_passes(contract, result)
        row_class = "best" if label == best_label else ("regress" if delta_class == "neg" else "")
        badge = '<span class="badge best">★ BEST</span>' if label == best_label else ""
        threshold = phase.get("threshold", result.get("threshold"))
        ratio_html = _fmt(ratio)
        if ratio is not None:
            ratio_html += "%"
            if ratio > 50:
                ratio_html = f'<span class="badge warn">{ratio_html} ⚠</span>'
        note = "KPI met" if passed else ("Constraints: " + ", ".join(failures) if failures else "Evaluated")
        table_rows.append(
            f'<tr class="{row_class}">'
            f'<td><strong>{_escape(label.title())}</strong> {badge}</td>'
            f'<td class="num {"pos" if label == best_label else ""}">{_fmt(value)}{_escape(_display_unit(contract["unit"]))}</td>'
            f'<td class="num {delta_class}">{"—" if delta is None else _fmt(delta)}</td>'
            f'<td class="num">{_fmt(threshold)}</td>'
            f'<td class="num">{_fmt(training_rows)}</td>'
            f'<td class="num">{_fmt(synthetic_rows)}</td>'
            f'<td>{ratio_html}</td>'
            f'<td>{_escape(note)}</td></tr>'
        )
        card_class = "best" if label == best_label else ("regress" if delta_class == "neg" else "")
        tag = "★ Best" if label == best_label else ("Regression" if delta_class == "neg" else "Evaluated")
        cards.append(
            f'<div class="iter-card {card_class}"><span class="iter-tag">{_escape(tag)}</span>'
            f'<div class="iter-title">{_escape(label.title())}</div>'
            f'<div class="iter-metric">{_fmt(value)}{_escape(_display_unit(contract["unit"]))}</div>'
            '<ul>'
            f'<li>Stage: {_escape(phase.get("stage_completed", "evaluate"))}</li>'
            f'<li>Training rows: {_fmt(training_rows)}</li>'
            f'<li>Synthetic / mined: {_fmt(synthetic_rows)} / {_fmt(mined_rows)}</li>'
            f'<li class="{"" if passed else "warn"}">{_escape(note)}</li>'
            '</ul></div>'
        )
        if label != "baseline":
            pool_rows.append(
                f'<tr><td><strong>{_escape(label.title())}</strong></td>'
                f'<td class="num">{_fmt(synthetic_rows)}</td>'
                f'<td class="num">{_fmt(mined_rows)}</td>'
                f'<td class="num pos">{_fmt(synthetic_rows + mined_rows)}</td></tr>'
            )
    if not table_rows:
        table_rows.append('<tr><td colspan="8">No completed evaluation yet.</td></tr>')
    if not cards:
        cards.append('<div class="iter-card"><span class="iter-tag">Pending</span><div class="iter-title">No completed iteration</div><div class="iter-metric">—</div></div>')
    if not pool_rows:
        pool_rows.append('<tr><td colspan="4">No augmentation has been committed yet.</td></tr>')
    return "\n".join(table_rows), "\n".join(cards), "\n".join(pool_rows)


def _dataset_card(state: dict[str, Any]) -> str:
    config = state.get("config", {})
    if not isinstance(config, dict):
        config = {}
    path = pathlib.Path(str(config.get("kpi_test_csv", "")))
    rows = _csv_rows(path)
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("label", row.get("object_name", "Unlabeled"))).strip() or "Unlabeled"
        counts[label] = counts.get(label, 0) + 1
    if not rows:
        return '<p class="info-text">KPI dataset details are not available yet. The report will refresh automatically after evaluation artifacts are committed.</p>'
    body = [
        f'<p class="info-text"><strong>{len(rows):,}</strong> evaluation rows from <code>{_escape(path.name)}</code>. Counts are read directly from the frozen KPI CSV.</p>',
        '<table class="data-table"><thead><tr><th>Category</th><th>Rows</th><th>Share</th></tr></thead><tbody>',
    ]
    for label, count in sorted(counts.items()):
        body.append(f'<tr><td>{_escape(label)}</td><td class="num">{count:,}</td><td class="num">{count / len(rows) * 100:.1f}%</td></tr>')
    body.append('</tbody></table>')
    return "\n".join(body)


def _context_cards(state: dict[str, Any], contract: dict[str, Any]) -> tuple[str, str]:
    config = state.get("config", {})
    if not isinstance(config, dict):
        config = {}
    evaluator = contract.get("evaluator", {})
    evaluator_name = evaluator.get("id") or evaluator.get("producer") or evaluator.get("path") or evaluator.get("type", "configured evaluator")
    constraints = contract.get("constraints", [])
    constraint_text = ", ".join(
        f"{item.get('display_name', item.get('name'))} {item.get('operator')} {item.get('target'):g}{_display_unit(str(item.get('unit', '')))}"
        for item in constraints
    ) or "none"
    problem = (
        '<p class="info-text">Improve NVIDIA TAO Visual ChangeNet PCB inspection against the approved customer KPI without changing the evaluation contract.</p>'
        '<ul class="context-list">'
        f'<li><strong>Primary gate:</strong> {_escape(render_target(contract))}</li>'
        f'<li><strong>Evaluator:</strong> {_escape(evaluator_name)}</li>'
        f'<li><strong>Secondary constraints:</strong> {_escape(constraint_text)}</li>'
        f'<li><strong>Iteration budget:</strong> {_fmt(state.get("max_iterations"))}</li>'
        '</ul>'
    )
    mining = config.get("mining_filter", {})
    if not isinstance(mining, dict):
        mining = {}
    approach = (
        '<p class="info-text">The DEFT loop evaluates the current checkpoint, identifies root causes, routes weak samples, expands the training set with mined real images and AnomalyGen defects, then fine-tunes and re-evaluates.</p>'
        '<div class="insight">'
        f'<strong>Run policy:</strong> up to {_fmt(state.get("max_iterations"))} iterations; '
        f'k-NN metric <code>{_escape(mining.get("metric", "cosine"))}</code> with minimum similarity <code>{_fmt(mining.get("min_similarity"))}</code>. '
        'Every number shown here is rebuilt from canonical state and committed artifacts.'
        '</div>'
    )
    return problem, approach


def _diagnostic_rows(
    contract: dict[str, Any], best_result: dict[str, Any] | None
) -> str:
    rows: list[str] = []
    if best_result is not None:
        passed, _ = result_passes(contract, best_result)
        rows.append(
            f'<tr><td>{_escape(contract["display_name"])}</td><td class="num">{_fmt(best_result["value"])}{_escape(_display_unit(contract["unit"]))}</td>'
            f'<td class="num">{_escape(contract["operator"])} {_fmt(contract["target"])}</td>'
            f'<td><span class="{"check" if passed else "warn"}">{"✓" if passed else "△"}</span></td></tr>'
        )
        values = best_result.get("constraints", {})
        if isinstance(values, dict):
            for constraint in contract.get("constraints", []):
                value = values.get(constraint["name"])
                rows.append(
                    f'<tr><td>{_escape(constraint["display_name"])}</td><td class="num">{_fmt(value)}{_escape(_display_unit(constraint["unit"]))}</td>'
                    f'<td class="num">{_escape(constraint["operator"])} {_fmt(constraint["target"])}</td>'
                    f'<td>{"recorded" if value is not None else "not available"}</td></tr>'
                )
    return "\n".join(rows) or '<tr><td colspan="4">Evaluator diagnostics are not available yet.</td></tr>'


def _rca_rows(results_dir: pathlib.Path, best_label: str | None) -> tuple[str, str]:
    if best_label is None:
        empty = '<tr><td colspan="4">RCA artifacts are not available yet.</td></tr>'
        return empty, empty
    base = results_dir / best_label
    score_candidates = [
        base / "rca_results" / "score_distribution.csv",
        base / "inference" / "threshold_metrics.csv",
    ]
    score_rows = next((_csv_rows(path) for path in score_candidates if path.is_file()), [])
    rendered_score: list[str] = []
    for row in score_rows[:8]:
        values = list(row.values())
        values += ["—"] * (4 - len(values))
        rendered_score.append('<tr>' + ''.join(f'<td>{_escape(value)}</td>' for value in values[:4]) + '</tr>')
    defect_rows = _csv_rows(base / "rca_results" / "defect_type_rows.csv")
    rendered_defect: list[str] = []
    for row in defect_rows[:20]:
        values = list(row.values())
        values += ["—"] * (4 - len(values))
        rendered_defect.append('<tr>' + ''.join(f'<td>{_escape(value)}</td>' for value in values[:4]) + '</tr>')
    empty_score = '<tr><td colspan="4">Score-distribution artifact is not available.</td></tr>'
    empty_defect = '<tr><td colspan="4">Per-defect RCA artifact is not available.</td></tr>'
    return "\n".join(rendered_score) or empty_score, "\n".join(rendered_defect) or empty_defect


def _thumbnail_data_uri(path: pathlib.Path) -> str | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.thumbnail((256, 256))
            if image.mode not in ("RGB", "L"):
                background = Image.new("RGB", image.size, "white")
                if "A" in image.getbands():
                    background.paste(image, mask=image.getchannel("A"))
                else:
                    background.paste(image)
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")
            payload = io.BytesIO()
            image.save(payload, format="JPEG", quality=86, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(payload.getvalue()).decode("ascii")
    except (ImportError, OSError, ValueError):
        return None


def _sample_html(
    results_dir: pathlib.Path,
    state: dict[str, Any],
    best_label: str | None,
) -> str:
    labels = [
        label
        for label in sorted(state.get("iterations", {}), key=_label_key, reverse=True)
        if label != "baseline"
    ]
    if best_label in labels:
        labels.remove(best_label)
        labels.insert(0, best_label)
    selected: tuple[pathlib.Path, pathlib.Path, str] | None = None
    for label in labels:
        number = label.removeprefix("iter")
        ok_dir = results_dir / label / "dataset" / "images" / f"synthetic_iter{number}_ok"
        ng_dir = results_dir / label / "dataset" / "images" / f"synthetic_iter{number}_ng"
        for ok_path in sorted(path for path in ok_dir.glob("*") if path.is_file()):
            ng_path = ng_dir / ok_path.name
            if ng_path.is_file():
                selected = (ok_path, ng_path, label)
                break
        if selected:
            break
    placeholder = '<div class="sample-img-placeholder">No image</div>'
    if selected:
        ok_uri = _thumbnail_data_uri(selected[0])
        ng_uri = _thumbnail_data_uri(selected[1])
        ok_html = f'<img class="sample-img" src="{ok_uri}" alt="AnomalyGen input OK">' if ok_uri else placeholder
        ng_html = f'<img class="sample-img" src="{ng_uri}" alt="AnomalyGen output synthetic NG">' if ng_uri else placeholder
        label = selected[2]
    else:
        ok_html = ng_html = placeholder
        label = "No completed synthetic pair"
    generated = 0
    if selected:
        phase = state.get("iterations", {}).get(selected[2], {})
        generated = _csv_row_count(phase.get("anomalygen_sdg_csv")) or 0
    return (
        f'<p class="info-text"><strong>AnomalyGen:</strong> enabled · num_SDG: {_fmt(generated)}</p>'
        f'<div class="sample-iter-block"><h3 class="sub-title">{_escape(label.title())}</h3><div class="sample-strip">'
        '<div class="sample-col"><div class="sample-col-title">AnomalyGen Input (OK / normal)</div>'
        f'{ok_html}</div><div class="sample-col"><div class="sample-col-title">AnomalyGen Output (synthetic NG)</div>{ng_html}</div>'
        '</div></div>'
    )


def _recommendations(
    terminal: bool, passed: bool, failures: list[str], *, failed: bool = False
) -> str:
    if failed:
        items = [
            ("Review the hard stop", "Use the latest error event in deft_state.json; do not infer recovery from this report."),
            ("Resume from recorded state", "Repair the failed stage, then retry it and commit the new result to deft_state.json."),
        ]
    elif not terminal:
        items = [
            ("Keep the loop running", "The report refreshes automatically after every committed stage."),
            ("Review live evidence", "Use the iteration table and augmentation pool to spot regressions early."),
        ]
    elif passed:
        items = [
            ("Promote the best checkpoint", "Validate the recorded checkpoint in the deployment environment."),
            ("Archive the evidence", "Keep this self-contained report with deft_state.json and the recorded artifacts."),
        ]
    else:
        detail = ", ".join(failures) if failures else "the remaining KPI gap"
        items = [
            ("Continue from the best checkpoint", f"Prioritize {detail} in the next approved run."),
            ("Inspect augmentation balance", "Review mined versus synthetic volume before increasing the iteration budget."),
        ]
    return "\n".join(
        f'<div class="reco"><div class="num-badge">{index}</div><div class="reco-body"><div class="reco-title">{_escape(title)}</div><div class="reco-desc">{_escape(description)}</div></div></div>'
        for index, (title, description) in enumerate(items, 1)
    )


def _load_source_template() -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    doc_start = template.index("<!--\n====")
    outer_close = template.index("-->\n<html")
    return template[:doc_start] + template[outer_close + 3 :]


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


def render(
    results_dir: pathlib.Path,
) -> pathlib.Path:
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
    terminal = stored_status in {"complete", "failed"}
    run_status = stored_status.upper()
    candidates = _metric_candidates(state, contract)
    best_label: str | None = None
    best_phase: dict[str, Any] = {}
    best_result: dict[str, Any] | None = None
    if candidates:
        best_label, best_phase, best_result = pick_best(candidates, contract)

    passed = False
    failures: list[str] = []
    if best_result is not None:
        passed, failures = result_passes(contract, best_result)
    unit = _display_unit(contract["unit"])
    completed_iterations = sum(1 for label, _, _ in candidates if label != "baseline")
    iteration_rows, iter_cards, pool_rows = _iteration_rows(
        state, contract, candidates, best_label
    )
    problem_html, approach_html = _context_cards(state, contract)
    score_rows, defect_rows = _rca_rows(results_dir, best_label)
    low, high, steps = _chart_extent(
        (result["value"] for _, _, result in candidates), contract["target"]
    )
    metric_data = [
        {
            "label": "Baseline" if label == "baseline" else label.title(),
            "value": float(result["value"]),
            "color": "#76b900" if label == best_label else "#b0b0b0",
        }
        for label, _, result in candidates
    ]

    if run_status == "FAILED":
        final_status = "RUN ENDED AT A HARD STOP"
        final_class = ""
        banner = (
            '<div class="kpi-banner"><div class="icon">!</div><div class="content">'
            '<div class="title">Run ended at a hard stop</div>'
            '<div class="body">Review the latest error event in deft_state.json before retrying the failed stage.</div></div></div>'
        )
    elif not terminal:
        final_status = "IN PROGRESS"
        final_class = ""
        banner = ""
    elif passed:
        final_status = "MET"
        final_class = "green"
        banner = (
            '<div class="kpi-banner" style="background:rgba(118,185,0,0.12);border-color:rgba(118,185,0,0.4);">'
            '<div class="icon" style="background:var(--nvidia-green);color:#000">✓</div><div class="content">'
            '<div class="title" style="color:var(--nvidia-green)">KPI MET</div>'
            f'<div class="body">{_escape(best_label.title() if best_label else "Best result")} achieved <strong>{_escape(render_target(contract))}</strong>.</div></div></div>'
        )
    elif best_result is not None:
        value = float(best_result["value"])
        gap = value - contract["target"] if contract["operator"] in MINIMIZING_OPERATORS else contract["target"] - value
        gap_unit = "pp" if contract["unit"] == "%" else unit
        final_status = f"{_fmt(abs(gap))}{gap_unit} from target"
        final_class = ""
        banner = ""
    else:
        final_status = "NO EVALUATION"
        final_class = ""
        banner = ""

    rca_insight = (
        f'<strong>Best evaluated phase:</strong> {_escape(best_label.title())} at {_fmt(best_result["value"])}{_escape(unit)}.'
        if best_label and best_result
        else "RCA will appear after the first evaluated checkpoint is committed."
    )
    threshold_insight = (
        f'Operating threshold: <code>{_fmt(best_phase.get("threshold"))}</code>. The primary KPI remains {_escape(render_target(contract))}.'
    )
    generated_date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    values = {
        "GENERATED_DATE": generated_date,
        "KPI_TARGET": _escape(render_target(contract)),
        "PRIMARY_METRIC_LABEL": _escape(contract["display_name"]),
        "PRIMARY_METRIC_UNIT": _escape(unit),
        "PRIMARY_METRIC_UNIT_JSON": _json_for_html(unit),
        "METRIC_DIRECTION_LABEL": "Lower is better" if contract["operator"] in MINIMIZING_OPERATORS else "Higher is better",
        "METRIC_DATA_JSON": _json_for_html(metric_data),
        "METRIC_TARGET_VALUE": _fmt(contract["target"]),
        "METRIC_Y_MIN": _fmt(low, 8),
        "METRIC_Y_MAX": _fmt(high, 8),
        "METRIC_Y_STEPS_JSON": _json_for_html(steps),
        "METRIC_MINIMIZES_JSON": json.dumps(contract["operator"] in MINIMIZING_OPERATORS),
        "MAX_ITERATIONS": _fmt(state.get("max_iterations")),
        "ITERATIONS_RUN": str(completed_iterations),
        "KPI_BANNER_HTML": banner,
        "RUN_SUMMARY_ROWS_HTML": _run_summary_rows(
            state, contract, candidates, entries
        ),
        "GROWTH_ROWS_HTML": _growth_rows(state),
        "PROBLEM_STATEMENT_HTML": problem_html,
        "KPI_DATASET_HTML": _dataset_card(state),
        "APPROACH_HTML": approach_html,
        "ITERATION_TABLE_ROWS_HTML": iteration_rows,
        "MINING_POOL_ROWS_HTML": pool_rows,
        "RCA_INSIGHT_HTML": rca_insight,
        "THRESHOLD_INSIGHT_HTML": threshold_insight,
        "SCORE_DIST_ROWS_HTML": score_rows,
        "EVALUATOR_DIAGNOSTIC_ROWS_HTML": _diagnostic_rows(contract, best_result),
        "DEFECT_TYPE_ROWS_HTML": defect_rows,
        "BEST_ITER_LABEL": _escape(best_label.title() if best_label else "Pending"),
        "DATA_SAMPLES_HTML": _sample_html(results_dir, state, best_label),
        "ITER_CARDS_HTML": iter_cards,
        "RECOMMENDATIONS_HTML": _recommendations(
            terminal,
            passed and run_status != "FAILED",
            failures,
            failed=run_status == "FAILED",
        ),
        "FINAL_KPI_STATUS": _escape(final_status),
        "FINAL_KPI_STATUS_CLASS": final_class,
        "FINAL_ITER_COUNT_LABEL": f"{completed_iterations} / {_fmt(state.get('max_iterations'))}",
        "BEST_METRIC_VALUE": _fmt(best_result.get("value") if best_result else None),
        "BEST_THRESHOLD": _fmt(best_phase.get("threshold")),
        "BEST_CHECKPOINT": _escape(best_phase.get("best_ckpt_path", "not available")),
    }
    template = _load_source_template()
    rendered = template
    for name, value in values.items():
        rendered = rendered.replace("{{ " + name + " }}", str(value))
    remaining = sorted(set(re.findall(r"\{\{\s+[A-Z0-9_]+\s+\}\}", rendered)))
    if remaining:
        raise ValueError("unfilled report placeholders: " + ", ".join(remaining))
    required = (
        "DEFT Loop Final Report",
        "Run Configuration &amp; Outcome",
        "Training Set Growth",
        "Progress Overview",
        "Per-Iteration Results",
        "Final Status",
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
