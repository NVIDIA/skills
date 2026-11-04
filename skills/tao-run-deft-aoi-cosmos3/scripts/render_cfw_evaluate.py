#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Render a single-GPU Framework two-image bare-label evaluation TOML."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import tempfile
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 in supported TAO host environments
    import tomli as tomllib


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_scalar(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {type(value).__name__}")


def dump_toml(value: dict[str, Any]) -> str:
    lines: list[str] = []

    def emit(table: dict[str, Any], prefix: tuple[str, ...]) -> None:
        scalars = [(key, item) for key, item in table.items() if not isinstance(item, dict)]
        children = [(key, item) for key, item in table.items() if isinstance(item, dict)]
        if prefix:
            if lines and lines[-1]:
                lines.append("")
            lines.append("[" + ".".join(prefix) + "]")
        lines.extend(f"{key} = {_scalar(item)}" for key, item in scalars)
        for key, child in children:
            emit(child, (*prefix, key))

    emit(value, ())
    return "\n".join(lines).rstrip() + "\n"


def build_config(
    *,
    annotation_path: str,
    media_root: str,
    model_path: str,
    results_dir: str,
    framework_config: str = "",
    action_model_dir: str = "",
    base_model_path: str = "",
) -> dict[str, Any]:
    if framework_config and not action_model_dir:
        raise ValueError("a DCP evaluation requires --action-model-dir")
    if action_model_dir and not framework_config:
        raise ValueError("--action-model-dir requires --framework-config")
    if framework_config and not base_model_path:
        raise ValueError("a DCP evaluation requires --base-model-path")
    return {
        "num_gpus": 1,
        "results_dir": results_dir,
        "task": {"type": ""},
        "dataset": {
            "annotation_path": annotation_path,
            "media_dir": media_root,
            "system_prompt": (
                "Compare the AOI image with its golden reference. "
                "Respond with exactly OK or NG."
            ),
        },
        "model": {
            "model_name": model_path,
            "dtype": "bfloat16",
            "max_length": 40960,
            "tp_size": 1,
            "enable_lora": False,
            "base_model_path": "",
            "config_file": framework_config,
            "export_dir": action_model_dir,
            "vit_checkpoint_path": base_model_path,
            "attn_implementation": "sdpa",
        },
        "evaluation": {
            "answer_type": "freeform",
            "num_processes": 1,
            "skip_saved": False,
            "seed": 42,
            "limit": -1,
            "shard_strategy": "stride",
            "shard_id": 0,
            "batch_size": 1,
            "barrier_timeout_seconds": 14400,
            "soft_accuracy": {"enabled": False, "f1_threshold": 0.8},
        },
        "vision": {
            "num_frames": 1,
            "video_decoder": "torchcodec-cuda-on-demand",
            "video_cache_size": 0,
            "frame_transfer": "host_rgb",
            "process_threads": 8,
            "decoder_cache_size": 4,
            "decoder_threads": 1,
            "decoder_device": "cuda",
            "dataloader_num_workers": 1,
            "dataloader_prefetch_factor": 2,
            "dataloader_multiprocessing_context": "spawn",
            "dataloader_persistent_workers": True,
        },
        "generation": {
            "max_retries": 10,
            "max_tokens": 4,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
        "metrics": {"names": []},
        "results": {
            "save_individual_results": True,
            "save_confusion_matrix": False,
            "save_metrics_summary": False,
        },
    }


def _absolute(value: pathlib.Path, flag: str) -> str:
    resolved = value.expanduser().resolve()
    if not resolved.is_absolute():
        raise ValueError(f"{flag} must resolve to an absolute path")
    return str(resolved)


def atomic_write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-path", required=True, type=pathlib.Path)
    parser.add_argument("--media-root", required=True, type=pathlib.Path)
    parser.add_argument("--model-path", required=True, type=pathlib.Path)
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--framework-config",
        type=pathlib.Path,
        help=(
            "Framework Train's saved Hydra config.yaml beside the DCP; "
            "never the input SFT TOML."
        ),
    )
    parser.add_argument("--action-model-dir", type=pathlib.Path)
    parser.add_argument("--base-model-path", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = build_config(
            annotation_path=_absolute(args.annotation_path, "--annotation-path"),
            media_root=_absolute(args.media_root, "--media-root"),
            model_path=_absolute(args.model_path, "--model-path"),
            results_dir=_absolute(args.results_dir, "--results-dir"),
            framework_config=(
                _absolute(args.framework_config, "--framework-config")
                if args.framework_config else ""
            ),
            action_model_dir=(
                _absolute(args.action_model_dir, "--action-model-dir")
                if args.action_model_dir else ""
            ),
            base_model_path=(
                _absolute(args.base_model_path, "--base-model-path")
                if args.base_model_path else ""
            ),
        )
        rendered = dump_toml(config)
        tomllib.loads(rendered)
        atomic_write(args.output.expanduser().resolve(), rendered)
    except (OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"render_cfw_evaluate: {exc}", file=os.sys.stderr)
        return 2
    print(args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
