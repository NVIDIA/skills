#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render a USD Performance Tuning Report template.

The renderer intentionally supports only the small Jinja-compatible subset used
by the report templates: variable interpolation, for loops, if/else blocks, and
simple equality checks. It has no third-party runtime dependencies.

Block tags that sit alone on a line are stripped of that line's leading
whitespace and its trailing newline, matching Jinja's ``lstrip_blocks`` +
``trim_blocks``. Without that, every ``{% for %}`` iteration carried the newline
that followed the tag, so each rendered table row was separated by a blank line —
which ends the table in Markdown, turning the report's tables into loose runs of
pipe-delimited text.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


TEMPLATE_DIR = Path(__file__).resolve().parent
DEFAULT_FIXTURE = TEMPLATE_DIR / "optimization-report.design-fixture.json"
DEFAULT_TEMPLATE = TEMPLATE_DIR / "optimization-report.html.template"
DEFAULT_OUTPUT = TEMPLATE_DIR / "optimization-report.preview.html"
TAG_RE = re.compile(r"{%\s*(.*?)\s*%}", re.DOTALL)
VAR_RE = re.compile(r"{{\s*(.*?)\s*}}", re.DOTALL)
#: A block tag alone on its line: nothing but whitespace before it, nothing but
#: whitespace and the newline after it. A tag sharing a line with content (the
#: HTML template's inline ``{% if group.caveat %}…{% endif %}``) does not match
#: and keeps its surrounding text exactly.
STANDALONE_TAG_RE = re.compile(r"^[ \t]*(\{%[^\n]*?%\})[ \t]*\n", re.MULTILINE)


def _score_percent(score: object) -> int:
    if not isinstance(score, (int, float)):
        return 0
    return max(0, min(100, round(float(score) * 10)))


def _score_display(score: object) -> str:
    if not isinstance(score, (int, float)):
        return "N/A"
    return f"{float(score):.1f}/10"


def _flag(value: object) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "not recorded"


def _count(value: object) -> str:
    return f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else "not recorded"


def _gate_summary_rows(report: dict) -> list[dict]:
    """One line each for the machine-readable gate blocks the report carries.

    ``footprint`` used to be the only block with a rendered section, so a reader
    saw the disk numbers and none of the correctness state. These rows put the
    preservation, coverage, reuse, and fidelity gates in the deliverable without
    the agent having to restate them in free text. ``safety_gate`` gets its own
    section rather than a row — it is the one that can invalidate the run.
    """
    rows: list[dict] = []

    pres = report.get("preservation")
    if isinstance(pres, dict):
        rows.append({
            "name": "Preservation (silent loss)",
            "value": (
                f"{_count(pres.get('rendered_mesh_count'))} rendered meshes; "
                f"{_count(pres.get('dangling'))} dangling bindings; "
                f"distinct geometry bytes preserved: {_flag(pres.get('distinct_geometry_bytes_preserved'))}; "
                f"bounds preserved: {_flag(pres.get('bounds_preserved'))}"
            ),
        })

    coverage = report.get("target_coverage")
    if isinstance(coverage, dict):
        entries = [e for e in (coverage.get("entries") or []) if isinstance(e, Mapping)]
        tally: dict[str, int] = {}
        for entry in entries:
            tally[str(entry.get("disposition", "unrecorded"))] = (
                tally.get(str(entry.get("disposition", "unrecorded")), 0) + 1
            )
        breakdown = ", ".join(f"{name} {count}" for name, count in sorted(tally.items()))
        rows.append({
            "name": "Phase-4 target coverage",
            "value": (
                f"{len(entries)} target(s); ledger complete: {_flag(coverage.get('complete'))}"
                + (f" ({breakdown})" if breakdown else "")
            ),
        })

    structural = report.get("structural_summary")
    if isinstance(structural, dict):
        rows.append({
            "name": "Reuse and descent",
            "value": (
                f"reuse measured: {_flag(structural.get('reuse_measured'))}; "
                f"{_count(structural.get('unique_prototype_count'))} prototypes; "
                f"descent converged: {_flag(structural.get('frontier_descent_converged'))}"
            ),
        })

    fidelity = report.get("fidelity")
    if isinstance(fidelity, dict):
        band = fidelity.get("band")
        rows.append({
            "name": "Fidelity",
            "value": (
                f"max deviation within band: {_flag(fidelity.get('max_deviation_within_band'))}"
                + (f" (band {band})" if band else "")
            ),
        })

    return rows


def build_context(report: dict) -> dict:
    context = dict(report)
    score = report.get("optimization_score")
    context["score_degrees"] = round(float(score) * 36, 1) if isinstance(score, (int, float)) else 0
    context.setdefault("executive_summary", "")
    context.setdefault("score_scope_label", "Stage Optimization Score")
    context.setdefault("reasoning", "")
    context["reasoning_paragraphs"] = [
        paragraph.strip()
        for paragraph in str(context["reasoning"]).split("\n\n")
        if paragraph.strip()
    ]

    groups = []
    for group in report.get("metric_groups", []):
        item = dict(group)
        item["score_percent"] = _score_percent(item.get("score"))
        item["score_display"] = _score_display(item.get("score"))
        groups.append(item)
    context["metric_groups"] = groups

    metrics = []
    for metric in report.get("metrics", []):
        item = dict(metric)
        item.setdefault("display_name", item.get("name", ""))
        item.setdefault("unit", "")
        item.setdefault("evidence_type", "direct")
        metrics.append(item)
    context["metrics"] = metrics

    measurement_context = report.get("measurement_context", {})
    context["measurement_context_items"] = [
        {"name": key.replace("_", " ").title(), "value": "N/A" if value is None else value}
        for key, value in measurement_context.items()
    ]
    runtime_profiling = report.get("runtime_profiling", {})
    context["runtime_profiling_items"] = [
        {"name": key.replace("_", " ").title(), "value": "N/A" if value is None else value}
        for key, value in runtime_profiling.items()
    ]
    context["gate_summary_rows"] = _gate_summary_rows(report)
    return context


def should_autoescape(template_name: str | None) -> bool:
    return bool(template_name and template_name.endswith(".html.template"))


def _resolve(name: str, context: Mapping[str, Any]) -> Any:
    name = name.strip()
    if not name:
        return ""
    if (name.startswith('"') and name.endswith('"')) or (name.startswith("'") and name.endswith("'")):
        return name[1:-1]
    value: Any = context
    for part in name.split("."):
        if isinstance(value, Mapping):
            value = value.get(part, "")
        else:
            value = getattr(value, part, "")
    return value


def _eval_condition(expression: str, context: Mapping[str, Any]) -> bool:
    expression = expression.strip()
    if expression.startswith("not "):
        return not _eval_condition(expression[4:], context)
    for operator in ("==", "!="):
        if operator in expression:
            left, right = expression.split(operator, 1)
            result = _resolve(left, context) == _resolve(right, context)
            return result if operator == "==" else not result
    return bool(_resolve(expression, context))


def _find_endfor(template: str, pos: int) -> tuple[int, int]:
    depth = 1
    for match in TAG_RE.finditer(template, pos):
        command = match.group(1).strip()
        if command.startswith("for "):
            depth += 1
        elif command == "endfor":
            depth -= 1
            if depth == 0:
                return match.start(), match.end()
    raise ValueError("missing {% endfor %} in report template")


def _find_if_parts(template: str, pos: int) -> tuple[int | None, int | None, int, int]:
    depth = 1
    else_start: int | None = None
    else_end: int | None = None
    for match in TAG_RE.finditer(template, pos):
        command = match.group(1).strip()
        if command.startswith("if "):
            depth += 1
        elif command == "else" and depth == 1:
            else_start = match.start()
            else_end = match.end()
        elif command == "endif":
            depth -= 1
            if depth == 0:
                return else_start, else_end, match.start(), match.end()
    raise ValueError("missing {% endif %} in report template")


def _render_variables(text: str, context: Mapping[str, Any], autoescape: bool) -> str:
    def replace(match: re.Match[str]) -> str:
        value = _resolve(match.group(1), context)
        if value is None:
            value = "N/A"
        rendered = str(value)
        return html.escape(rendered, quote=True) if autoescape else rendered

    return VAR_RE.sub(replace, text)


def _render_block(template: str, context: Mapping[str, Any], autoescape: bool) -> str:
    rendered: list[str] = []
    pos = 0

    while True:
        match = TAG_RE.search(template, pos)
        if not match:
            rendered.append(_render_variables(template[pos:], context, autoescape))
            break

        rendered.append(_render_variables(template[pos:match.start()], context, autoescape))
        command = match.group(1).strip()

        if command.startswith("for "):
            loop_target, _, expression = command[4:].partition(" in ")
            if not loop_target or not expression:
                raise ValueError(f"unsupported for expression: {command}")
            end_start, end_end = _find_endfor(template, match.end())
            body = template[match.end():end_start]
            for item in _resolve(expression, context) or []:
                child_context = dict(context)
                child_context[loop_target.strip()] = item
                rendered.append(_render_block(body, child_context, autoescape))
            pos = end_end
            continue

        if command.startswith("if "):
            else_start, else_end, endif_start, endif_end = _find_if_parts(template, match.end())
            if else_start is None:
                true_body = template[match.end():endif_start]
                false_body = ""
            else:
                true_body = template[match.end():else_start]
                false_body = template[else_end:endif_start]
            rendered.append(
                _render_block(
                    true_body if _eval_condition(command[3:], context) else false_body,
                    context,
                    autoescape,
                )
            )
            pos = endif_end
            continue

        if command in {"else", "endif", "endfor"}:
            raise ValueError(f"unexpected template tag: {command}")

        raise ValueError(f"unsupported template tag: {command}")

    return "".join(rendered)


def trim_block_tag_lines(template: str) -> str:
    """Collapse each standalone block tag onto the text around it.

    Jinja's ``lstrip_blocks`` + ``trim_blocks``. A ``{% for %}`` tag on its own
    line otherwise donates that line's newline to every iteration of the loop
    body, so an N-row table renders as N rows each preceded by a blank line —
    and a blank line ends a Markdown table.
    """
    return STANDALONE_TAG_RE.sub(lambda match: match.group(1), template)


def render_template(template: str, context: Mapping[str, Any], *, autoescape: bool) -> str:
    return _render_block(trim_block_tag_lines(template), context, autoescape)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = json.loads(args.fixture.read_text(encoding="utf-8"))
    rendered = render_template(
        args.template.read_text(encoding="utf-8"),
        build_context(report),
        autoescape=should_autoescape(args.template.name),
    )
    args.output.write_text(rendered, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
