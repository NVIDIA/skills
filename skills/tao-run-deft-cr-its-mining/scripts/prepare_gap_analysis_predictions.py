#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rewrite Cosmos Reason prediction IDs to LLaVA video paths for gap analysis."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def load_json_array(path: Path, label: str) -> list[dict[str, Any]]:
    """Read a JSON array and require every item to be an object."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{label} must be a JSON array: {path}")
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: {label} item {index} is not a JSON object")
    return payload


def resolve_media_path(video: str, media_dir: Path) -> str:
    """Resolve a LLaVA video value against its configured media directory."""
    expanded = os.path.expanduser(video)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(str(media_dir), expanded))


def annotation_video_lookup(
    annotations: list[dict[str, Any]],
    annotations_path: Path,
    media_dir: Path,
) -> tuple[dict[str, str], set[str]]:
    """Build the annotation-id-to-video mapping used to patch predictions."""
    lookup: dict[str, str] = {}
    video_paths: set[str] = set()
    for index, annotation in enumerate(annotations, start=1):
        annotation_id = annotation.get("id")
        video = annotation.get("video")
        if not isinstance(annotation_id, str) or not annotation_id:
            raise ValueError(
                f"{annotations_path}: annotation item {index} is missing non-empty 'id'"
            )
        if not isinstance(video, str) or not video:
            raise ValueError(
                f"{annotations_path}: annotation item {index} is missing non-empty 'video'"
            )
        video_path = resolve_media_path(video, media_dir)
        existing = lookup.get(annotation_id)
        if existing is not None and existing != video_path:
            raise ValueError(
                f"{annotations_path}: annotation id {annotation_id!r} maps to conflicting "
                f"video paths: {existing!r} and {video_path!r}"
            )
        lookup[annotation_id] = video_path
        video_paths.add(video_path)
    return lookup, video_paths


def prepare_predictions(
    predictions: list[dict[str, Any]],
    predictions_path: Path,
    annotation_lookup: dict[str, str],
    annotation_video_paths: set[str],
    media_dir: Path,
) -> list[dict[str, Any]]:
    """Replace prediction annotation IDs with their resolved video paths."""
    prepared: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions, start=1):
        video_id = prediction.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(
                f"{predictions_path}: prediction item {index} is missing non-empty 'video_id'"
            )

        # Temporary compatibility patch until vlm_bcq_gap_analysis resolves
        # Cosmos Reason annotation IDs through the LLaVA annotations itself.
        video_path = annotation_lookup.get(video_id)
        if video_path is None:
            candidate = resolve_media_path(video_id, media_dir)
            if candidate not in annotation_video_paths:
                raise ValueError(
                    f"{predictions_path}: prediction item {index} video_id {video_id!r} "
                    "does not match an annotation id or annotation video path"
                )
            video_path = candidate

        output = dict(prediction)
        output["video_id"] = video_path
        prepared.append(output)
    return prepared


def write_predictions(
    results_json: Path,
    annotations_json: Path,
    media_dir: Path,
    output_json: Path,
) -> list[dict[str, Any]]:
    """Load, patch, and write the predictions consumed by gap analysis."""
    predictions = load_json_array(results_json, "predictions")
    annotations = load_json_array(annotations_json, "annotations")
    lookup, video_paths = annotation_video_lookup(annotations, annotations_json, media_dir)
    prepared = prepare_predictions(
        predictions,
        results_json,
        lookup,
        video_paths,
        media_dir,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(prepared, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return prepared


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", required=True, type=Path)
    parser.add_argument("--annotations-json", required=True, type=Path)
    parser.add_argument("--media-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    results_json = Path(os.path.abspath(os.path.expanduser(str(args.results_json))))
    annotations_json = Path(os.path.abspath(os.path.expanduser(str(args.annotations_json))))
    media_dir = Path(os.path.abspath(os.path.expanduser(str(args.media_dir))))
    output_json = Path(os.path.abspath(os.path.expanduser(str(args.output_json))))
    if not results_json.is_file():
        raise FileNotFoundError(f"results JSON does not exist: {results_json}")
    if not annotations_json.is_file():
        raise FileNotFoundError(f"annotations JSON does not exist: {annotations_json}")
    if not media_dir.is_dir():
        raise NotADirectoryError(f"media directory does not exist: {media_dir}")

    prepared = write_predictions(results_json, annotations_json, media_dir, output_json)
    print(f"Wrote gap-analysis predictions: {output_json}")
    print(f"Prediction rows: {len(prepared)}")
    print(f"Unique video paths: {len({row['video_id'] for row in prepared})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
