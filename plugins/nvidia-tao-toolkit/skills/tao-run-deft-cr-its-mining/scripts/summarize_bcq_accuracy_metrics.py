#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize baseline and iteration BCQ accuracy metrics for one DEFT run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from workflow_common import absolute_path, atomic_write_json


ITERATION_DIR = re.compile(r"^iter_(\d+)$")
REQUIRED_METRICS = (
    "total_samples",
    "accuracy",
    "balanced_accuracy",
    "false_positives",
    "false_negatives",
    "unparseable_predictions",
)


def load_metrics(path: Path) -> dict[str, Any]:
    """Load and validate one computed metrics artifact."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    missing = [field for field in REQUIRED_METRICS if field not in payload]
    if missing:
        raise ValueError(f"{path}: missing required metrics: {', '.join(missing)}")
    return payload


def collect_run_metrics(run_dir: Path) -> list[dict[str, Any]]:
    """Collect baseline and all completed iteration metric files in order."""
    baseline_path = run_dir / "baseline" / "evaluate" / "bcq_accuracy_metrics.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"baseline metrics do not exist: {baseline_path}")

    evaluations = [{"evaluation": "Baseline", "iteration": 0, **load_metrics(baseline_path)}]
    iteration_dirs: list[tuple[int, Path]] = []
    for child in run_dir.iterdir():
        match = ITERATION_DIR.match(child.name)
        if child.is_dir() and match:
            iteration_dirs.append((int(match.group(1)), child))
    for iteration, iteration_dir in sorted(iteration_dirs):
        metrics_path = iteration_dir / "evaluate" / "bcq_accuracy_metrics.json"
        if metrics_path.is_file():
            evaluations.append(
                {
                    "evaluation": f"Iteration {iteration}",
                    "iteration": iteration,
                    **load_metrics(metrics_path),
                }
            )
    return evaluations


def render_markdown(evaluations: list[dict[str, Any]]) -> str:
    """Render a compact accuracy table for the agent's final report."""
    lines = [
        "# BCQ Accuracy Report",
        "",
        "| Evaluation | Accuracy | Balanced accuracy | False positives | False negatives | Unparseable predictions | Samples |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metrics in evaluations:
        lines.append(
            "| {evaluation} | {accuracy:.2%} | {balanced_accuracy:.2%} | "
            "{false_positives} | {false_negatives} | {unparseable_predictions} | "
            "{total_samples} |".format(**metrics)
        )
    return "\n".join(lines) + "\n"


def write_report(
    run_dir: Path,
    output_markdown: Path,
    output_json: Path,
) -> list[dict[str, Any]]:
    """Collect run metrics and write Markdown and machine-readable summaries."""
    run_dir = absolute_path(run_dir)
    output_markdown = absolute_path(output_markdown)
    output_json = absolute_path(output_json)
    if not run_dir.is_dir():
        raise NotADirectoryError(f"run directory does not exist: {run_dir}")
    evaluations = collect_run_metrics(run_dir)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(render_markdown(evaluations), encoding="utf-8")
    atomic_write_json(
        output_json,
        {
            "schema_version": 1,
            "run_dir": str(run_dir),
            "evaluations": evaluations,
        },
    )
    return evaluations


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    run_dir = absolute_path(args.run_dir)
    output_markdown = absolute_path(args.output_markdown or run_dir / "bcq_accuracy_report.md")
    output_json = absolute_path(args.output_json or run_dir / "bcq_accuracy_summary.json")
    evaluations = write_report(run_dir, output_markdown, output_json)
    print(f"report_markdown: {output_markdown}")
    print(f"report_json: {output_json}")
    print(f"evaluations: {len(evaluations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
