#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve and verify the evaluator-loadable Cosmos-RL epoch export.

Training status reports the native policy checkpoint.  Cosmos evaluation loads
the sibling Hugging Face export under ``safetensors/epoch_N``.  This helper is
designed to run from the target compute frame and emits a manifest that binds
the native event to the exact verified action checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import tempfile
from pathlib import Path
from typing import Any


class CheckpointError(ValueError):
    """A deterministic checkpoint handoff failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _epoch_from_path(path: Path) -> int | None:
    matches = re.findall(r"(?:^|/)epoch_(\d+)(?:/|$)", str(path))
    return int(matches[-1]) if matches else None


def resolve_action_checkpoint(source: Path, epoch: int | None) -> tuple[Path, int]:
    inferred = _epoch_from_path(source)
    if epoch is not None and inferred is not None and epoch != inferred:
        raise CheckpointError(
            f"requested epoch {epoch} conflicts with source checkpoint epoch {inferred}: {source}"
        )
    selected_epoch = epoch if epoch is not None else inferred
    if selected_epoch is None:
        raise CheckpointError(
            f"cannot infer epoch from source checkpoint; supply --epoch: {source}"
        )

    if source.name == f"epoch_{selected_epoch}" and source.parent.name == "safetensors":
        return source, selected_epoch

    parts = source.parts
    try:
        checkpoint_index = max(
            index for index, value in enumerate(parts) if value == "checkpoints"
        )
    except ValueError as exc:
        raise CheckpointError(
            "native Cosmos-RL checkpoint must contain a checkpoints/epoch_N segment: "
            f"{source}"
        ) from exc
    if checkpoint_index + 1 >= len(parts) or parts[checkpoint_index + 1] != f"epoch_{selected_epoch}":
        raise CheckpointError(
            f"source checkpoint does not identify epoch_{selected_epoch}: {source}"
        )
    run_root = Path(*parts[:checkpoint_index])
    if source.is_absolute():
        run_root = Path("/") / run_root
    return run_root / "safetensors" / f"epoch_{selected_epoch}", selected_epoch


def _validate_safetensors(path: Path) -> None:
    if not path.is_file():
        raise CheckpointError(f"missing safetensors file: {path}")
    size = path.stat().st_size
    if size < 10:
        raise CheckpointError(f"truncated safetensors file: {path} ({size} bytes)")
    with path.open("rb") as stream:
        header_bytes = stream.read(8)
        if len(header_bytes) != 8:
            raise CheckpointError(f"truncated safetensors header: {path}")
        header_size = struct.unpack("<Q", header_bytes)[0]
        if header_size <= 2 or header_size > size - 8:
            raise CheckpointError(
                f"invalid safetensors header length in {path}: {header_size}"
            )
        try:
            header = json.loads(stream.read(header_size))
        except json.JSONDecodeError as exc:
            raise CheckpointError(f"invalid safetensors JSON header in {path}: {exc}") from exc
    if not isinstance(header, dict) or not any(key != "__metadata__" for key in header):
        raise CheckpointError(f"safetensors contains no tensor index: {path}")


def _require_contained_file(checkpoint: Path, path: Path) -> None:
    try:
        path.resolve().relative_to(checkpoint.resolve())
    except ValueError:
        raise CheckpointError(f"checkpoint file escapes its epoch directory: {path}")


def _validate_dense(checkpoint: Path) -> list[Path]:
    config_path = checkpoint / "config.json"
    config = _load_object(config_path)
    if not config.get("model_type"):
        raise CheckpointError(f"dense checkpoint config has no model_type: {config_path}")
    index_path = checkpoint / "model.safetensors.index.json"
    files = [config_path]
    if index_path.is_file():
        index = _load_object(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise CheckpointError(f"dense checkpoint weight_map is empty: {index_path}")
        names = sorted(set(weight_map.values()))
        if not all(
            isinstance(name, str)
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts
            for name in names
        ):
            raise CheckpointError(f"dense checkpoint index has an unsafe weight path: {index_path}")
        weights = [checkpoint / name for name in names]
        files.append(index_path)
    else:
        weights = sorted(checkpoint.glob("*.safetensors"))
    if not weights:
        raise CheckpointError(f"dense checkpoint has no safetensors weights: {checkpoint}")
    for path in weights:
        _require_contained_file(checkpoint, path)
        _validate_safetensors(path)
    return files + weights


def _validate_peft(checkpoint: Path) -> list[Path]:
    config_path = checkpoint / "adapter_config.json"
    weights_path = checkpoint / "adapter_model.safetensors"
    config = _load_object(config_path)
    if not config.get("peft_type") or not isinstance(config.get("r"), int) or config["r"] <= 0:
        raise CheckpointError(f"adapter config is missing peft_type or positive rank: {config_path}")
    _require_contained_file(checkpoint, weights_path)
    _validate_safetensors(weights_path)
    model_config = checkpoint / "config.json"
    return [config_path, weights_path] + ([model_config] if model_config.is_file() else [])


def verify(
    source_checkpoint: str,
    training_mode: str,
    epoch: int | None,
    *,
    base_model: bool = False,
) -> dict[str, Any]:
    source = Path(source_checkpoint).expanduser()
    if not source.exists():
        raise CheckpointError(f"source checkpoint is inaccessible from the compute frame: {source}")
    if base_model:
        if epoch is not None:
            raise CheckpointError("--base-model cannot be combined with --epoch")
        if training_mode != "dense":
            raise CheckpointError("a Cosmos-RL baseline model must use dense training mode")
        action_checkpoint, selected_epoch = source, None
    else:
        action_checkpoint, selected_epoch = resolve_action_checkpoint(source, epoch)
    if not action_checkpoint.is_dir():
        raise CheckpointError(
            f"evaluator-loadable epoch export is missing: {action_checkpoint}"
        )
    if training_mode == "dense":
        files = _validate_dense(action_checkpoint)
        checkpoint_kind = (
            "hf_dense_base_model_safetensors"
            if base_model
            else "hf_dense_safetensors"
        )
    elif training_mode == "peft":
        files = _validate_peft(action_checkpoint)
        checkpoint_kind = "hf_peft_adapter_safetensors"
    else:
        raise CheckpointError(f"unsupported Cosmos-RL training mode: {training_mode}")
    inventory = [
        {
            "path": str(path.relative_to(action_checkpoint)),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(files)
    ]
    return {
        "schema_version": 1,
        "status": "VERIFIED",
        "backend": "cosmos-rl",
        "source_checkpoint": source_checkpoint,
        "action_model_path": str(action_checkpoint),
        "epoch": selected_epoch,
        "training_mode": training_mode,
        "checkpoint_kind": checkpoint_kind,
        "base_model": base_model,
        "files": inventory,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-mode", choices=("dense", "peft"), required=True)
    parser.add_argument("--epoch", type=int)
    parser.add_argument(
        "--base-model",
        action="store_true",
        help="Verify an existing dense Hugging Face model directory for AutoML baseline evaluation.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.epoch is not None and args.epoch <= 0:
            raise CheckpointError("epoch must be positive")
        result = verify(
            args.checkpoint,
            args.training_mode,
            args.epoch,
            base_model=args.base_model,
        )
        _atomic_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (CheckpointError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
