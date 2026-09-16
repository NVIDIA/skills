#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Summarize downloaded anomaly-dataset outputs without loading media into memory."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


class SummaryError(ValueError):
    """Raised when downloaded results cannot be interpreted."""


def find_dataset_manifest(root: Path) -> Path:
    matches = sorted(root.rglob("dataset.json"))
    if not matches:
        raise SummaryError(f"No dataset.json found under {root}")
    if len(matches) > 1:
        raise SummaryError(
            "Multiple dataset.json files found; summarize one workflow download at a time"
        )
    return matches[0]


def summarize(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise SummaryError(f"Results path is not a directory: {root}")
    data_path = find_dataset_manifest(root)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SummaryError("dataset.json must contain an object")
    metadata = data.get("metadata")
    entries = data.get("entries")
    if not isinstance(metadata, dict) or not isinstance(entries, list):
        raise SummaryError("dataset.json requires metadata and entries")

    input_keys: set[str] = set()
    augmentation_counts: Counter[str] = Counter()
    invalid_entries = 0
    for entry in entries:
        if not isinstance(entry, dict):
            invalid_entries += 1
            continue
        input_key = entry.get("input_key")
        if input_key is not None:
            input_keys.add(str(input_key))
            augmentation_counts[str(input_key)] += 1

    relative_manifest = data_path.relative_to(root).as_posix()
    return {
        "manifest": relative_manifest,
        "entries": len(entries),
        "input_images": len(input_keys),
        "invalid_entries": invalid_entries,
        "reported_total_scenes": metadata.get("total_scenes"),
        "reported_original_inputs": metadata.get("original_inputs"),
        "augmentations_per_input": dict(sorted(augmentation_counts.items())),
        "downloaded_files": sum(1 for path in root.rglob("*") if path.is_file()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize downloaded SDG Event Video Generation results."
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON."
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = summarize(Path(args.results_dir))
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(f"Manifest: {summary['manifest']}")
            print(f"Entries (scenes): {summary['entries']}")
            print(f"Input images: {summary['input_images']}")
            print(f"Downloaded files: {summary['downloaded_files']}")
            if summary["invalid_entries"]:
                print(f"Invalid entries: {summary['invalid_entries']}")
            print(f"Reported total scenes: {summary['reported_total_scenes']}")
            print(f"Reported original inputs: {summary['reported_original_inputs']}")
        return 0
    except (OSError, json.JSONDecodeError, SummaryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
