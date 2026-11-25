#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare config.yaml and path_map.jsonl for PAIDF Cosmos Predict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paidf_common import (
    DEFAULT_GENERATION_SETTINGS,
    absolute_path,
    build_path_map,
    dump_yaml,
    ensure_container_writable_directory,
    ensure_readable_directory,
    ensure_writable_file_target,
    normalize_media_dir,
    read_generation_settings,
    settings_section,
    write_jsonl,
)


def paidf_config(
    mappings: list[dict[str, Any]],
    endpoint: str,
    caption_model: str,
    caption_parameters: dict[str, Any],
    augmentation: dict[str, Any],
    prompt: str,
) -> dict[str, Any]:
    """Assemble the PAIDF config dictionary from prepared inputs and settings."""
    return {
        "data": [
            {
                "inputs": {"rgb": item["container_media_path"]},
                "output": {
                    "video": item["container_generated_video_path"],
                    "caption": item["container_caption_path"],
                    "metadata": item["container_metadata_path"],
                },
            }
            for item in mappings
        ],
        "endpoints": {"vlm": {"url": endpoint, "model": caption_model}},
        "captioning": {
            "vlm": {
                "parser": "instruct",
                "user_prompt": prompt,
                "parameters": caption_parameters,
            }
        },
        "augmentation": augmentation,
    }


def main() -> None:
    """Parse inputs, write config.yaml/path_map.jsonl, and pre-create outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--vlm-captioning-endpoint", required=True)
    parser.add_argument("--generation-settings", type=Path, default=DEFAULT_GENERATION_SETTINGS)
    parser.add_argument("--paidf-num-gpus", type=int, required=True)
    parser.add_argument("--caption-prompt-file", type=Path, required=True)
    parser.add_argument("--media-dir", required=True)
    args = parser.parse_args()
    if args.paidf_num_gpus < 1:
        parser.error("--paidf-num-gpus must be >= 1")

    output_dir = absolute_path(args.output_dir)
    config_path = output_dir / "config.yaml"
    path_map_path = output_dir / "path_map.jsonl"

    media_dir = normalize_media_dir(args.media_dir)
    ensure_readable_directory(media_dir, "media dir")

    ensure_container_writable_directory(output_dir, "PAIDF output")
    ensure_container_writable_directory(output_dir / "generated" / "videos", "generated videos")
    ensure_container_writable_directory(output_dir / "generated" / "metadata", "generated metadata")
    ensure_container_writable_directory(output_dir / "captions", "captions")
    ensure_writable_file_target(config_path, "PAIDF config")
    ensure_writable_file_target(path_map_path, "path map")

    settings = read_generation_settings(args.generation_settings)
    vlm_settings = settings_section(settings, "vlm_captioning")
    caption_model = vlm_settings.get("model")
    if not isinstance(caption_model, str) or not caption_model:
        raise ValueError("Generation settings field 'vlm_captioning.model' must be a string")
    caption_parameters = dict(settings_section(settings, "vlm_captioning.parameters"))
    augmentation = json.loads(json.dumps(settings_section(settings, "paidf.augmentation")))

    local_parameters = augmentation.get("local_parameters")
    if not isinstance(local_parameters, dict):
        raise ValueError(
            "Generation settings field 'paidf.augmentation.local_parameters' must be an object"
        )
    local_parameters["num_processes"] = args.paidf_num_gpus

    if not args.caption_prompt_file.exists():
        raise FileNotFoundError(f"Caption prompt file not found: {args.caption_prompt_file}")
    prompt = args.caption_prompt_file.read_text(encoding="utf-8")
    mappings = build_path_map(args.input_jsonl, output_dir, media_dir)
    config = paidf_config(
        mappings,
        args.vlm_captioning_endpoint,
        caption_model,
        caption_parameters,
        augmentation,
        prompt,
    )
    config_path.write_text("\n".join(dump_yaml(config)) + "\n", encoding="utf-8")
    write_jsonl(path_map_path, mappings)

    metadata = {
        "vlm_captioning_endpoint": args.vlm_captioning_endpoint,
        "vlm_captioning_model": caption_model,
        "paidf_num_gpus": args.paidf_num_gpus,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote PAIDF config: {config_path}")
    print(f"Wrote deterministic path map: {path_map_path}")
    print(f"Wrote run metadata: {output_dir / 'run_metadata.json'}")
    print(f"Media dir mounted 1:1: {media_dir}")
    print(f"Unique media files: {len(mappings)}")


if __name__ == "__main__":
    main()
