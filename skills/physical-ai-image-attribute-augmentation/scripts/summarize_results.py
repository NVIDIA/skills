#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Summarize downloaded augmented-dataset outputs without loading media into memory."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class SummaryError(ValueError):
    """Raised when downloaded results cannot be interpreted."""


def find_augmented_data(root: Path) -> Path:
    matches = sorted(root.rglob("augmented_data.json"))
    if not matches:
        raise SummaryError(f"No augmented_data.json found under {root}")
    if len(matches) > 1:
        raise SummaryError(
            "Multiple augmented_data.json files found; summarize one workflow download at a time"
        )
    return matches[0]


def summarize(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise SummaryError(f"Results path is not a directory: {root}")
    data_path = find_augmented_data(root)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SummaryError("augmented_data.json must contain an object")
    metadata = data.get("metadata")
    entries = data.get("entries")
    if not isinstance(metadata, dict) or not isinstance(entries, list):
        raise SummaryError("augmented_data.json requires metadata and entries")

    source_ids: set[str] = set()
    attributes: dict[str, Counter[str]] = defaultdict(Counter)
    query_counts: Counter[str] = Counter()
    invalid_entries = 0
    for entry in entries:
        if not isinstance(entry, dict):
            invalid_entries += 1
            continue
        person_id = entry.get("source_person_key") or entry.get("person_id")
        if person_id is not None:
            source_ids.add(str(person_id))
        selected = entry.get("selected_attributes")
        if isinstance(selected, dict):
            for name, value in selected.items():
                attributes[str(name)][str(value)] += 1
        queries = entry.get("queries")
        if isinstance(queries, dict):
            for level, values in queries.items():
                if isinstance(values, list):
                    query_counts[str(level)] += len(values)

    relative_manifest = data_path.relative_to(root).as_posix()
    return {
        "manifest": relative_manifest,
        "entries": len(entries),
        "source_person_ids": len(source_ids),
        "invalid_entries": invalid_entries,
        "reported_total_ids": metadata.get("total_ids"),
        "reported_total_scenes": metadata.get("total_scenes"),
        "query_counts": dict(sorted(query_counts.items())),
        "attribute_counts": {
            name: dict(sorted(counts.items()))
            for name, counts in sorted(attributes.items())
        },
        "downloaded_files": sum(1 for path in root.rglob("*") if path.is_file()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize downloaded SDG Image Attribute Augmentation results."
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
            print(f"Entries: {summary['entries']}")
            print(f"Source person IDs: {summary['source_person_ids']}")
            print(f"Downloaded files: {summary['downloaded_files']}")
            if summary["invalid_entries"]:
                print(f"Invalid entries: {summary['invalid_entries']}")
            for level, count in summary["query_counts"].items():
                print(f"{level.title()} queries: {count}")
        return 0
    except (OSError, json.JSONDecodeError, SummaryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
