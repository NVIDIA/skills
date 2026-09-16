#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write generated and failed PAIDF handoff JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from paidf_common import input_records, read_jsonl, write_jsonl


def mappings_by_media_path(path_map: Path) -> dict[str, dict[str, Any]]:
    """Index path_map.jsonl by each media path alias used by input rows."""
    mappings: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(read_jsonl(path_map), start=1):
        key = record.get("absolute_media_path")
        if not isinstance(key, str):
            raise ValueError(f"{path_map}: record {index} missing absolute_media_path")
        mappings[key] = record
        for alias_key in ("host_media_path", "container_media_path"):
            alias = record.get(alias_key)
            if isinstance(alias, str):
                mappings[alias] = record
    return mappings


def main() -> None:
    """Write one generated or failed row for each input JSONL row."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--path-map", required=True, type=Path)
    parser.add_argument("--generated-jsonl", required=True, type=Path)
    parser.add_argument("--failed-jsonl", required=True, type=Path)
    args = parser.parse_args()

    mappings = mappings_by_media_path(args.path_map)
    generated_records: list[dict[str, str]] = []
    failed_records: list[dict[str, str]] = []

    for record in input_records(args.input_jsonl):
        media_path = record["media_path"]
        if media_path not in mappings:
            raise ValueError(f"{args.path_map}: no path mapping for {media_path!r}")
        mapped = mappings[media_path]
        generated_path = mapped["host_generated_video_path"]
        if not Path(generated_path).exists():
            failed_records.append(
                {
                    "id": record["id"],
                    "original_media_path": mapped["host_media_path"],
                    "expected_generated_video_path": generated_path,
                    "error": "missing_generated_video",
                }
            )
            continue
        generated_records.append(
            {
                "id": record["id"],
                "original_media_path": mapped["host_media_path"],
                "generated_video_path": generated_path,
            }
        )

    write_jsonl(args.generated_jsonl, generated_records)
    write_jsonl(args.failed_jsonl, failed_records)
    print(f"Wrote generated videos handoff: {args.generated_jsonl}")
    print(f"Wrote failed videos handoff: {args.failed_jsonl}")
    print(f"Generated rows: {len(generated_records)}")
    print(f"Failed rows: {len(failed_records)}")


if __name__ == "__main__":
    main()
