#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble monotonic bare OK/NG Cosmos3 AOI training JSON."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

from validate_sharegpt import load_records, prompt_and_label


def _load_json_list(path: pathlib.Path) -> list[dict]:
    data = load_records(path)
    for index, record in enumerate(data):
        images = record.get("images")
        if not isinstance(images, list) or len(images) != 2:
            raise ValueError(
                f"{path}[{index}]: images must contain [AOI, golden_reference]"
            )
        prompt_and_label(record, context=f"{path}[{index}]")
    return data


def _assistant_label(record: dict) -> str:
    return prompt_and_label(record, context="record")[1]


def _pair_key(record: dict) -> tuple[str, ...]:
    images = record.get("images")
    return tuple(images) if isinstance(images, list) else tuple()


def _target_key(record: dict) -> str:
    images = record.get("images")
    if not isinstance(images, list) or not images:
        return ""
    return str(images[0])


def _unique_target_images(records: list[dict]) -> set[str]:
    """Return distinct, non-empty target image paths (the first image in each pair)."""
    return {target for record in records if (target := _target_key(record))}


def assemble(
    seed_path: pathlib.Path | None,
    new_paths: list[pathlib.Path],
    *,
    dedupe: bool,
    validation_paths: list[pathlib.Path],
) -> tuple[list[dict], dict]:
    if not new_paths:
        raise ValueError("at least one --new-json input is required")
    seed = _load_json_list(seed_path) if seed_path is not None else []
    seed_targets = _unique_target_images(seed)
    sources: list[tuple[pathlib.Path, list[dict]]] = []
    if seed_path is not None:
        sources.append((seed_path, seed))
    new_records: list[dict] = []
    for path in new_paths:
        records = _load_json_list(path)
        sources.append((path, records))
        new_records.extend(records)

    validation_targets: dict[str, pathlib.Path] = {}
    for validation_path in validation_paths:
        for record in _load_json_list(validation_path):
            key = _target_key(record)
            if key:
                validation_targets[key] = validation_path

    merged: list[dict] = []
    provenance: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    duplicates = 0
    leakage: list[dict] = []
    labels: Counter[str] = Counter()

    for source_path, records in sources:
        for index, record in enumerate(records):
            key = _pair_key(record)
            if dedupe and key and key in seen:
                duplicates += 1
                continue
            if key:
                seen.add(key)
            target_key = _target_key(record)
            if validation_targets and target_key in validation_targets:
                leakage.append(
                    {
                        "source": str(source_path),
                        "index": index,
                        "target": target_key,
                        "evaluation_split": str(validation_targets[target_key]),
                    }
                )
            merged.append(record)
            labels[_assistant_label(record)] += 1
            provenance.append({"source": str(source_path), "source_index": index})

    if leakage:
        raise ValueError(f"train/evaluation leakage detected: {leakage[:5]}")

    new_input_targets = _unique_target_images(new_records)
    output_targets = _unique_target_images(merged)

    summary = {
        "seed": str(seed_path) if seed_path is not None else None,
        "new_inputs": [str(p) for p in new_paths],
        "output_records": len(merged),
        "mode": "bare_okng",
        "dedupe": dedupe,
        "dedupe_key": "media",
        "label_source": "assistant",
        "duplicates_skipped": duplicates,
        "unique_target_images": {
            "seed": len(seed_targets),
            "new_inputs": len(new_input_targets),
            "new_after_dedup": len(output_targets - seed_targets),
            "output_total": len(output_targets),
        },
        "labels": dict(labels),
        "validation_jsons": [str(path) for path in validation_paths],
        "provenance": provenance,
    }
    return merged, summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--previous-json",
        type=pathlib.Path,
        help="Previous iteration training JSON. Omit for iter1.",
    )
    parser.add_argument(
        "--new-json",
        action="append",
        default=[],
        type=pathlib.Path,
        help="Current real Mining or synthetic AnomalyGen records; repeat for each producer.",
    )
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--summary", default=None, type=pathlib.Path)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument(
        "--validation-json",
        action="append",
        default=[],
        type=pathlib.Path,
        help="Evaluation JSON excluded from training. Repeat for Proxy val and Benchmark test.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        merged, summary = assemble(
            args.previous_json,
            args.new_json,
            dedupe=args.dedupe,
            validation_paths=args.validation_json,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(merged, indent=2) + "\n")
        summary_path = args.summary or args.output.with_name("assemble_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"assemble_training_json: {exc}", file=sys.stderr)
        return 2

    print(
        f"assemble_training_json: wrote {len(merged)} records to {args.output}; "
        f"duplicates_skipped={summary['duplicates_skipped']}"
    )
    print(f"assemble_training_json: wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
