#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Summarize a downloaded Video Data Augmentation run without loading media."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_CANDIDATE_FILES = ("output.mp4", "caption.txt", "metadata.json")


class SummaryError(ValueError):
    """Raised when a downloaded run cannot be interpreted."""


def _file_count(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file()) if path.is_dir() else 0


def _evaluator_state(metadata_path: Path) -> str:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unreadable"
    evaluator = metadata.get("cosmos_evaluator") if isinstance(metadata, dict) else None
    if not isinstance(evaluator, dict):
        return "not_recorded"
    if evaluator.get("operational_success") is not True:
        return "operational_failure"
    return "passed" if evaluator.get("quality_passed") is True else "rejected"


def summarize(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise SummaryError(f"Results path is not a directory: {root}")
    cosmos_root = root / "cosmos"
    if not cosmos_root.is_dir():
        raise SummaryError(f"No cosmos directory found under {root}")

    candidates: list[dict[str, Any]] = []
    evaluator_states: Counter[str] = Counter()
    per_video: Counter[str] = Counter()
    for video_dir in sorted(path for path in cosmos_root.iterdir() if path.is_dir()):
        for candidate_dir in sorted(path for path in video_dir.iterdir() if path.is_dir()):
            missing = [
                name for name in REQUIRED_CANDIDATE_FILES if not (candidate_dir / name).is_file()
            ]
            state = (
                _evaluator_state(candidate_dir / "metadata.json")
                if "metadata.json" not in missing
                else "missing_metadata"
            )
            evaluator_states[state] += 1
            per_video[video_dir.name] += 1
            candidates.append(
                {
                    "video_key": video_dir.name,
                    "augmentation_index": candidate_dir.name,
                    "missing": missing,
                    "evaluator": state,
                }
            )

    report_path = root / "output" / "report.json"
    report_present = report_path.is_file()
    report_valid = False
    if report_present:
        try:
            report_valid = isinstance(json.loads(report_path.read_text(encoding="utf-8")), dict)
        except (OSError, json.JSONDecodeError):
            report_valid = False

    return {
        "candidate_directories": len(candidates),
        "complete_candidates": sum(1 for item in candidates if not item["missing"]),
        "incomplete_candidates": sum(1 for item in candidates if item["missing"]),
        "candidates_per_video": dict(sorted(per_video.items())),
        "evaluator_states": dict(sorted(evaluator_states.items())),
        "auto_labeling_input_files": _file_count(root / "auto_labeling_input"),
        "auto_labeling_output_files": _file_count(root / "auto_labeling_output"),
        "training_export_input_files": _file_count(
            root / "training_export" / "auto_labeling_input"
        ),
        "training_export_output_files": _file_count(
            root / "training_export" / "auto_labeling_output"
        ),
        "report_present": report_present,
        "report_valid": report_valid,
        "downloaded_files": _file_count(root),
        "candidates": candidates,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize downloaded Video Data Augmentation results."
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = summarize(Path(args.results_dir))
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(f"Candidate directories: {summary['candidate_directories']}")
            print(f"Complete candidates: {summary['complete_candidates']}")
            print(f"Incomplete candidates: {summary['incomplete_candidates']}")
            print(f"Auto-labeling input files: {summary['auto_labeling_input_files']}")
            print(f"Auto-labeling output files: {summary['auto_labeling_output_files']}")
            print(f"Training-export input files: {summary['training_export_input_files']}")
            print(f"Training-export output files: {summary['training_export_output_files']}")
            print(f"Evaluator states: {summary['evaluator_states']}")
            print(f"Report present/valid: {summary['report_present']}/{summary['report_valid']}")
        return 0
    except (OSError, json.JSONDecodeError, SummaryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
