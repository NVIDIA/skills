#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a TAO Data Services VLM BCQ gap-analysis spec."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml


def absolute_path(value: str | Path) -> Path:
    """Expand a path and return an absolute path without resolving symlinks."""
    return Path(os.path.abspath(os.path.expanduser(str(value))))


def prepare_spec(
    predictions_json: Path,
    videos_dir: Path | None,
    results_dir: Path,
    output_spec: Path,
) -> dict[str, Any]:
    """Validate inputs and write the gap-analysis YAML spec."""
    predictions_json = absolute_path(predictions_json)
    results_dir = absolute_path(results_dir)
    output_spec = absolute_path(output_spec)
    if not predictions_json.is_file():
        raise FileNotFoundError(f"predictions JSON does not exist: {predictions_json}")

    resolved_videos_dir = ""
    if videos_dir is not None:
        videos_dir = absolute_path(videos_dir)
        if not videos_dir.is_dir():
            raise NotADirectoryError(f"videos directory does not exist: {videos_dir}")
        resolved_videos_dir = str(videos_dir)

    spec = {
        "predictions_json": str(predictions_json),
        "videos_dir": resolved_videos_dir,
        "results_dir": str(results_dir),
    }
    output_spec.parent.mkdir(parents=True, exist_ok=True)
    with output_spec.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(spec, handle, sort_keys=False)
    return spec


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-json", required=True, type=Path)
    parser.add_argument("--videos-dir", type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-spec", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """Write one VLM BCQ gap-analysis spec."""
    args = parse_args()
    spec = prepare_spec(args.predictions_json, args.videos_dir, args.results_dir, args.output_spec)
    print(f"Wrote VLM BCQ spec: {absolute_path(args.output_spec)}")
    print(f"Expected gaps JSONL: {Path(spec['results_dir']) / 'kpi_gaps.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
