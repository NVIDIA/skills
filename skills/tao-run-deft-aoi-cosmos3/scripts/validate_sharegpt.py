#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Validate bare OK/NG ShareGPT image-pair annotations for Cosmos3 AOI."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter
from typing import Any


VALID_LABELS = {"OK", "NG"}
HUMAN_ROLES = {"human", "user"}
ASSISTANT_ROLES = {"gpt", "assistant"}
_SAFE_ID = re.compile(r"[A-Za-z0-9._@-]+")


def load_records(path: pathlib.Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}: expected one JSON array; JSONL is not supported: {exc.msg}"
        ) from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path}: expected a non-empty JSON array")
    if not all(isinstance(record, dict) for record in payload):
        raise ValueError(f"{path}: every record must be an object")
    return payload


def _turn_role(turn: dict[str, Any]) -> Any:
    return turn.get("from", turn.get("role"))


def _turn_value(turn: dict[str, Any]) -> Any:
    return turn.get("value", turn.get("content"))


def prompt_and_label(record: dict[str, Any], *, context: str) -> tuple[str, str]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(f"{context}: conversations must be a list")
    prompt: str | None = None
    label: str | None = None
    for turn in conversations:
        if not isinstance(turn, dict):
            raise ValueError(f"{context}: every conversation turn must be an object")
        role = _turn_role(turn)
        value = _turn_value(turn)
        if role in HUMAN_ROLES and prompt is None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{context}: human prompt must be non-empty")
            prompt = value.strip()
        if role in ASSISTANT_ROLES:
            if not isinstance(value, str):
                raise ValueError(f"{context}: assistant response must be a string")
            label = value.strip().upper()
    if prompt is None:
        raise ValueError(f"{context}: missing human prompt")
    if label not in VALID_LABELS:
        raise ValueError(
            f"{context}: bare mode requires the final assistant response to be "
            f"exactly OK or NG, got {label!r}"
        )
    return prompt, label


def resolve_image(path_text: str, media_root: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path_text).expanduser()
    if not path.is_absolute():
        path = media_root / path
    return path.resolve()


def validate_records(
    records: list[dict[str, Any]],
    *,
    media_root: pathlib.Path,
    require_files: bool,
    require_id: bool = False,
) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    targets: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    duplicate_targets: list[str] = []
    duplicate_pairs: list[list[str]] = []
    ids: set[str] = set()

    for index, record in enumerate(records):
        context = f"record[{index}]"
        images = record.get("images")
        if not isinstance(images, list) or len(images) != 2:
            raise ValueError(
                f"{context}: images must contain exactly [AOI, golden_reference]"
            )
        if not all(isinstance(image, str) and image.strip() for image in images):
            raise ValueError(f"{context}: image paths must be non-empty strings")
        resolved = tuple(
            str(resolve_image(image, media_root)) for image in images
        )
        if require_files:
            missing = [path for path in resolved if not pathlib.Path(path).is_file()]
            if missing:
                raise ValueError(f"{context}: missing image file(s): {missing}")
        if resolved[0] in targets:
            duplicate_targets.append(resolved[0])
        targets.add(resolved[0])
        if resolved in pairs:
            duplicate_pairs.append(list(resolved))
        pairs.add(resolved)
        # Framework evaluation writes per-sample artifacts keyed by ``id``, so
        # evaluation splits need a unique filesystem-safe value. Training does
        # not consume this field.
        record_id = record.get("id")
        if require_id or record_id is not None:
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(
                    f"{context}: id must be a non-empty string for evaluation "
                    "annotations"
                )
            if not _SAFE_ID.fullmatch(record_id):
                raise ValueError(
                    f"{context}: id must be filesystem-safe (letters, digits, "
                    f"dot, dash, underscore, @); got {record_id!r}"
                )
            if record_id in ids:
                raise ValueError(f"{context}: duplicate id {record_id!r}")
            ids.add(record_id)
        _, label = prompt_and_label(record, context=context)
        labels[label] += 1
        video_fps = record.get("video_fps")
        if video_fps is not None and (
            not isinstance(video_fps, (int, float))
            or isinstance(video_fps, bool)
            or video_fps <= 0
        ):
            raise ValueError(f"{context}: video_fps must be positive when present")

    if duplicate_targets:
        raise ValueError(
            "target AOI images must be unique; duplicates="
            f"{sorted(set(duplicate_targets))[:5]}"
        )
    if duplicate_pairs:
        raise ValueError(
            f"image pairs must be unique; duplicates={duplicate_pairs[:5]}"
        )
    return {
        "mode": "bare_okng",
        "records": len(records),
        "labels": dict(sorted(labels.items())),
        "unique_target_images": len(targets),
        "require_files": require_files,
        "unique_ids": len(ids),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=pathlib.Path)
    parser.add_argument("--media-root", required=True, type=pathlib.Path)
    parser.add_argument("--require-files", action="store_true")
    parser.add_argument(
        "--require-id",
        action="store_true",
        help="Require a unique, filesystem-safe id per record for Framework "
        "Proxy and Benchmark evaluation.",
    )
    parser.add_argument("--summary", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        summary = validate_records(
            load_records(args.annotations),
            media_root=args.media_root.expanduser().resolve(),
            require_files=args.require_files,
            require_id=args.require_id,
        )
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"validate_sharegpt: {exc}", file=sys.stderr)
        return 2
    print(
        "validate_sharegpt: OK "
        f"mode=bare_okng records={summary['records']} labels={summary['labels']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
