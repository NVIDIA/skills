#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan, verify, and run the Framework checkpoint pre-action export.

Cosmos Framework training writes a PyTorch Distributed Checkpoint (DCP), while
TAO evaluate and inference actions load a Hugging Face safetensors directory.
This checked-in skill helper makes that conversion an idempotent, provenance-
checked stage.  The tensor conversion itself deliberately remains owned by
``cosmos_framework.scripts.export_vlm_dcp`` in the Framework repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cosmos_common import (
    WorkflowError,
    path_identity,
    planned_path_identity,
    sha256_file,
)

FRAMEWORK_ACTIONS = {"export", "evaluate", "inference", "inference_microservice"}
EXPORTER_MODULE = "cosmos_framework.scripts.export_vlm_dcp"


def _weight_files(root: Path) -> list[Path]:
    files = sorted(root.glob("*.safetensors"))
    index = root / "model.safetensors.index.json"
    if index.is_file():
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
            files.extend(root / name for name in sorted(set(payload.get("weight_map", {}).values())))
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            raise WorkflowError(f"invalid safetensors index: {index}: {exc}") from exc
    return sorted(set(files))


def is_hf_checkpoint(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file() and bool(_weight_files(path))


def dcp_metadata(path: Path) -> Path | None:
    for candidate in (path / "model" / ".metadata", path / ".metadata"):
        if candidate.is_file():
            return candidate
    matches = sorted(path.rglob("*.distcp")) if path.is_dir() else []
    if matches:
        for parent in (matches[0].parent, *matches[0].parents):
            candidate = parent / ".metadata"
            if candidate.is_file() and (parent == path or path in parent.parents):
                return candidate
    return None


def infer_config_file(checkpoint: Path) -> Path:
    candidates = (
        checkpoint.parent.parent / "config.yaml",
        checkpoint.parent.parent / "config.json",
        checkpoint.parent / "config.yaml",
        checkpoint.parent / "config.json",
        checkpoint / "config.yaml",
        checkpoint / "config.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise WorkflowError(
        "Framework DCP checkpoint configuration could not be inferred; supply --config-file "
        f"for {checkpoint}"
    )


def _normalized_source(value: str) -> str:
    if value.startswith("hf_model://"):
        return value.removeprefix("hf_model://").strip("/")
    if value.startswith("hf://"):
        return value.removeprefix("hf://").removeprefix("models/").strip("/")
    candidate = Path(value).expanduser()
    return str(candidate.resolve()) if candidate.exists() else value


def _safetensors_file_tensor_keys(path: Path) -> set[str]:
    """Read tensor names from a safetensors header without loading tensor data."""
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            encoded_length = stream.read(8)
            if len(encoded_length) != 8:
                raise WorkflowError(f"safetensors file has no complete header: {path}")
            header_length = int.from_bytes(encoded_length, "little", signed=False)
            if header_length <= 0 or header_length > min(100 * 1024 * 1024, size - 8):
                raise WorkflowError(
                    f"safetensors file has an invalid header length: {path}"
                )
            encoded_header = stream.read(header_length)
    except OSError as exc:
        raise WorkflowError(f"cannot read safetensors header: {path}: {exc}") from exc
    try:
        header = json.loads(encoded_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"invalid safetensors header JSON: {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise WorkflowError(f"safetensors header must be a JSON object: {path}")
    tensor_entries = {
        name: value for name, value in header.items() if name != "__metadata__"
    }
    if not tensor_entries or not all(
        isinstance(name, str) and name and isinstance(value, dict)
        for name, value in tensor_entries.items()
    ):
        raise WorkflowError(f"safetensors header has no valid tensor entries: {path}")
    return set(tensor_entries)


def _safetensors_tensor_keys(root: Path) -> set[str] | None:
    """Return verified tensor keys for indexed or single-file safetensors."""
    index = root / "model.safetensors.index.json"
    expected_keys: set[str] | None = None
    if index.is_file():
        payload = _read_json(index, "safetensors index")
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map or not all(
            isinstance(name, str)
            and name
            and isinstance(filename, str)
            and filename
            for name, filename in weight_map.items()
        ):
            raise WorkflowError(f"safetensors index has an invalid weight_map: {index}")
        expected_keys = set(weight_map)
        weight_files = sorted({root / filename for filename in weight_map.values()})
    else:
        weight_files = sorted(root.glob("*.safetensors"))
    if not weight_files:
        return None
    actual_keys: set[str] = set()
    for weight_file in weight_files:
        try:
            resolved_weight = weight_file.resolve(strict=True)
            resolved_weight.relative_to(root.resolve())
        except (OSError, ValueError) as exc:
            raise WorkflowError(
                f"safetensors weight path escapes or is missing below {root}: {weight_file}"
            ) from exc
        keys = _safetensors_file_tensor_keys(resolved_weight)
        duplicates = actual_keys & keys
        if duplicates:
            raise WorkflowError(
                "duplicate tensor keys across safetensors files: "
                f"{sorted(duplicates)[:10]}"
            )
        actual_keys.update(keys)
    if expected_keys is not None and actual_keys != expected_keys:
        raise WorkflowError(
            "safetensors index tensor keys disagree with shard headers: "
            f"missing={sorted(expected_keys - actual_keys)[:10]}, "
            f"unexpected={sorted(actual_keys - expected_keys)[:10]}"
        )
    return actual_keys


def _base_model_fingerprint(path: Path) -> str:
    names = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "chat_template.json",
    )
    files = {
        name: sha256_file(path / name)
        for name in names
        if (path / name).is_file()
    }
    files.update({item.name: sha256_file(item) for item in sorted(path.glob("*.safetensors"))})
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _default_export_dir(checkpoint: Path, config_file: Path, metadata: Path) -> Path:
    fingerprint = hashlib.sha256(
        (sha256_file(metadata) + sha256_file(config_file)).encode("ascii")
    ).hexdigest()[:16]
    if checkpoint.parent.name == "checkpoints":
        root = checkpoint.parent.parent / "hf_exports"
    else:
        root = checkpoint.parent / "hf_exports"
    return (root / f"{checkpoint.name}-{fingerprint}").resolve()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkflowError(f"{label} must contain a JSON object: {path}")
    return value


def verify_export(
    *,
    checkpoint_path: str,
    config_file: str,
    export_dir: str,
    base_model_path_or_uri: str = "",
    base_model_revision: str = "",
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    config = Path(config_file).expanduser().resolve(strict=True)
    output = Path(export_dir).expanduser().resolve(strict=True)
    metadata = dcp_metadata(checkpoint)
    if metadata is None:
        raise WorkflowError(f"no Framework DCP metadata found below {checkpoint}")
    if not is_hf_checkpoint(output):
        raise WorkflowError(f"Framework export is missing config.json or safetensors weights: {output}")
    weights = _weight_files(output)
    missing_weights = [str(item) for item in weights if not item.is_file()]
    if missing_weights:
        raise WorkflowError(f"Framework export index references missing weights: {missing_weights[:5]}")

    manifest = _read_json(output / "export_manifest.json", "Framework export manifest")
    checkpoint_record = _read_json(output / "checkpoint.json", "Framework checkpoint record")
    expected_checkpoint = str(checkpoint)
    recorded_checkpoint = _normalized_source(str(manifest.get("checkpoint", "")))
    if recorded_checkpoint != expected_checkpoint:
        raise WorkflowError(
            f"Framework export checkpoint mismatch: expected {expected_checkpoint}, found {recorded_checkpoint}"
        )
    if _normalized_source(str(checkpoint_record.get("checkpoint_path", ""))) != expected_checkpoint:
        raise WorkflowError("checkpoint.json does not identify the requested Framework DCP checkpoint")
    if manifest.get("format") != "cosmos-framework-vlm-dcp":
        raise WorkflowError(f"unsupported Framework export format: {manifest.get('format')!r}")
    if manifest.get("checkpoint_metadata_sha256") != sha256_file(metadata):
        raise WorkflowError("Framework export DCP metadata fingerprint is stale")
    if _normalized_source(str(manifest.get("config", ""))) != str(config):
        raise WorkflowError("Framework export configuration path does not match the requested configuration")
    if manifest.get("config_sha256") != sha256_file(config):
        raise WorkflowError("Framework export configuration fingerprint is stale")
    if base_model_path_or_uri:
        expected_base = _normalized_source(base_model_path_or_uri)
        recorded_base = _normalized_source(str(manifest.get("base_model_path_or_uri", "")))
        if recorded_base != expected_base:
            raise WorkflowError(
                f"Framework export base model mismatch: expected {expected_base}, found {recorded_base}"
            )
        local_base = Path(base_model_path_or_uri).expanduser()
        if local_base.exists():
            recorded_fingerprint = manifest.get("base_model_fingerprint", {}).get("sha256")
            if recorded_fingerprint != _base_model_fingerprint(local_base.resolve()):
                raise WorkflowError("Framework export base model fingerprint is stale")
            base_keys = _safetensors_tensor_keys(local_base.resolve())
            export_keys = _safetensors_tensor_keys(output)
            if base_keys is None or export_keys is None:
                raise WorkflowError(
                    "Framework export tensor-key verification requires safetensors "
                    "weights in both the local base checkpoint and export"
                )
            if base_keys != export_keys:
                missing = sorted(base_keys - export_keys)[:10]
                unexpected = sorted(export_keys - base_keys)[:10]
                raise WorkflowError(
                    "Framework export tensor key set differs from the base checkpoint: "
                    f"missing={missing}, unexpected={unexpected}"
                )
    if base_model_revision and manifest.get("base_model_revision") != base_model_revision:
        raise WorkflowError("Framework export base model revision does not match the requested revision")
    return {
        "schema_version": 1,
        "status": "VERIFIED",
        "backend": "cosmos-framework",
        "ok": True,
        "source_checkpoint": str(checkpoint),
        "action_model_path": str(output),
        "checkpoint": path_identity(str(checkpoint)),
        "config": path_identity(str(config)),
        "export": path_identity(str(output)),
        "checkpoint_metadata_sha256": sha256_file(metadata),
        "config_sha256": sha256_file(config),
        "weight_files": [str(item.relative_to(output)) for item in weights],
        "manifest": manifest,
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.action not in FRAMEWORK_ACTIONS:
        raise WorkflowError(f"unsupported Framework action: {args.action}")
    if not args.checkpoint_path:
        raise WorkflowError("--checkpoint-path is required")
    supplied = Path(args.checkpoint_path).expanduser()
    if not supplied.exists():
        if supplied.is_absolute() or args.checkpoint_path.startswith((".", "~")):
            raise WorkflowError(f"checkpoint path does not exist: {args.checkpoint_path}")
        if not args.base_model_revision:
            raise WorkflowError("an immutable revision is required for a model URI/identifier")
        return {
            "schema_version": 1,
            "backend": "cosmos-framework",
            "action": args.action,
            "checkpoint_kind": "model_uri",
            "checkpoint": {"original": args.checkpoint_path, "resolved": None},
            "export_required": False,
            "export_state": "not_applicable",
            "action_model_path": args.checkpoint_path,
            "base_model_revision": args.base_model_revision,
            "pre_action": None,
        }

    checkpoint = supplied.resolve()
    if is_hf_checkpoint(checkpoint):
        return {
            "schema_version": 1,
            "backend": "cosmos-framework",
            "action": args.action,
            "checkpoint_kind": "hf_safetensors",
            "checkpoint": path_identity(args.checkpoint_path),
            "export_required": False,
            "export_state": "already_hf",
            "action_model_path": str(checkpoint),
            "base_model_revision": args.base_model_revision or None,
            "pre_action": None,
        }

    metadata = dcp_metadata(checkpoint)
    if metadata is None:
        raise WorkflowError(
            f"model input is neither a complete HF safetensors directory nor a Framework DCP checkpoint: {checkpoint}"
        )
    config = Path(args.config_file).expanduser().resolve(strict=True) if args.config_file else infer_config_file(checkpoint)
    export = (
        Path(args.export_dir).expanduser().resolve()
        if args.export_dir
        else _default_export_dir(checkpoint, config, metadata)
    )
    export_command = [
        args.python_executable,
        "-m",
        EXPORTER_MODULE,
        "--checkpoint-path",
        str(checkpoint),
        "--config-file",
        str(config),
        "--output-dir",
        str(export),
        "--dtype",
        args.dtype,
    ]
    if args.base_model_path_or_uri:
        export_command.extend(
            ["--base-model-path-or-uri", _normalized_source(args.base_model_path_or_uri)]
        )
    if args.base_model_revision:
        export_command.extend(["--base-model-revision", args.base_model_revision])

    export_state = "missing"
    validation_error = None
    if export.exists():
        try:
            verify_export(
                checkpoint_path=str(checkpoint),
                config_file=str(config),
                export_dir=str(export),
                base_model_path_or_uri=args.base_model_path_or_uri,
                base_model_revision=args.base_model_revision,
            )
            export_state = "verified_complete"
        except WorkflowError as exc:
            export_state = "stale_or_incomplete"
            validation_error = str(exc)
    return {
        "schema_version": 1,
        "backend": "cosmos-framework",
        "action": args.action,
        "checkpoint_kind": "framework_dcp",
        "checkpoint": path_identity(args.checkpoint_path),
        "checkpoint_metadata": path_identity(str(metadata)),
        "checkpoint_metadata_sha256": sha256_file(metadata),
        "config": path_identity(str(config)),
        "config_sha256": sha256_file(config),
        "base_model_path_or_uri": args.base_model_path_or_uri or None,
        "base_model_revision": args.base_model_revision or None,
        "export": path_identity(str(export)) if export.exists() else planned_path_identity(str(export)),
        "export_required": export_state != "verified_complete",
        "export_state": export_state,
        "export_validation_error": validation_error,
        "action_model_path": str(export),
        "pre_action": {
            "owner": "cosmos-framework",
            "module": EXPORTER_MODULE,
            "argv": export_command,
            "command": shlex.join(export_command),
            "resources": {"tasks": 1, "gpus": 1, "distributed": False},
            "child_exit_code_required": True,
            "idempotency": "skip only after manifest, DCP metadata, config, base-model, and weight validation",
        },
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_export(args: argparse.Namespace) -> dict[str, Any]:
    plan = build_plan(args)
    if not plan["export_required"]:
        result = {
            **plan,
            "pre_action_result": "reused" if plan.get("export") else "not_applicable",
        }
        if plan.get("checkpoint_kind") == "framework_dcp":
            verification = verify_export(
                checkpoint_path=plan["checkpoint"]["resolved"],
                config_file=plan["config"]["resolved"],
                export_dir=plan["action_model_path"],
                base_model_path_or_uri=args.base_model_path_or_uri,
                base_model_revision=args.base_model_revision,
            )
            result.update(
                {
                    "status": "VERIFIED",
                    "source_checkpoint": plan["checkpoint"]["resolved"],
                    "action_model_path": plan["action_model_path"],
                    "verification": verification,
                }
            )
        return result
    if plan["checkpoint_kind"] != "framework_dcp":
        raise WorkflowError("only a Framework DCP checkpoint can require export")

    final_output = Path(plan["action_model_path"])
    temporary = final_output.with_name(f".{final_output.name}.partial-{os.getpid()}-{time.time_ns()}")
    command = list(plan["pre_action"]["argv"])
    output_index = command.index("--output-dir") + 1
    command[output_index] = str(temporary)
    full_command = [*args.command_prefix, *command]
    completed = subprocess.run(full_command, check=False)
    if completed.returncode:
        raise WorkflowError(f"Framework checkpoint exporter failed with child exit code {completed.returncode}")
    verified = verify_export(
        checkpoint_path=plan["checkpoint"]["resolved"],
        config_file=plan["config"]["resolved"],
        export_dir=str(temporary),
        base_model_path_or_uri=args.base_model_path_or_uri,
        base_model_revision=args.base_model_revision,
    )

    displaced = None
    if final_output.exists():
        displaced = final_output.with_name(f"{final_output.name}.invalid-{time.time_ns()}")
        final_output.rename(displaced)
    temporary.rename(final_output)
    marker = {
        "schema_version": 1,
        "checkpoint": plan["checkpoint"],
        "checkpoint_metadata_sha256": plan["checkpoint_metadata_sha256"],
        "config": plan["config"],
        "config_sha256": plan["config_sha256"],
        "base_model_path_or_uri": args.base_model_path_or_uri or None,
        "base_model_revision": args.base_model_revision or None,
        "export": path_identity(str(final_output)),
        "export_manifest_sha256": sha256_file(final_output / "export_manifest.json"),
        "timestamp_ns": time.time_ns(),
    }
    _atomic_json(final_output / ".tao_export_complete", marker)
    return {
        **build_plan(args),
        "status": "VERIFIED",
        "source_checkpoint": plan["checkpoint"]["resolved"],
        "action_model_path": str(final_output),
        "pre_action_result": "exported",
        "displaced_invalid_export": str(displaced) if displaced else None,
        "verification": {
            **verified,
            "action_model_path": str(final_output),
            "export": path_identity(str(final_output)),
        },
    }


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--action", choices=sorted(FRAMEWORK_ACTIONS), required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-file", default="")
    parser.add_argument("--export-dir", default="")
    parser.add_argument("--base-model-path-or-uri", default="")
    parser.add_argument("--base-model-revision", default="")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--python-executable", default="/workspace/.venv/bin/python")
    parser.add_argument("--output", type=Path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="verb", required=True)
    for verb in ("plan", "prepare"):
        child = subparsers.add_parser(verb)
        _common_parser(child)
        if verb == "prepare":
            child.add_argument(
                "--command-prefix",
                default="",
                help="Optional shell-quoted runner prefix, such as a Docker or srun container command",
            )
    verify = subparsers.add_parser("verify")
    verify.add_argument("--checkpoint-path", required=True)
    verify.add_argument("--config-file", required=True)
    verify.add_argument("--export-dir", required=True)
    verify.add_argument("--base-model-path-or-uri", default="")
    verify.add_argument("--base-model-revision", default="")
    verify.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if hasattr(args, "command_prefix"):
        args.command_prefix = shlex.split(args.command_prefix)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.verb == "verify":
            result = verify_export(
                checkpoint_path=args.checkpoint_path,
                config_file=args.config_file,
                export_dir=args.export_dir,
                base_model_path_or_uri=args.base_model_path_or_uri,
                base_model_revision=args.base_model_revision,
            )
        elif args.verb == "prepare":
            result = prepare_export(args)
        else:
            result = build_plan(args)
        if args.output:
            _atomic_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, json.JSONDecodeError, WorkflowError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
