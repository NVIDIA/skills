#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the workflow baseline checkpoint for the pinned Cosmos-RL runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from workflow_common import (
    absolute_path,
    atomic_write_json,
    checkpoint_model_type,
    existing_absolute_path,
    load_json_object,
    load_yaml,
    path_in_workspace,
    require_mapping,
    require_string,
    validate_qwen3_vl_checkpoint,
)


DEFAULT_ARCHITECTURE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_ARCHITECTURE_REVISION = "main"
IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{40,64}$")
SHA256_DIGEST = re.compile(r"^sha256:[0-9a-fA-F]{64}$")


def default_model_preparation_script() -> Path:
    """Locate the converter owned by the installed Cosmos Reason model skill."""
    skills_root = Path(__file__).resolve().parents[3]
    return (
        skills_root
        / "models"
        / "tao-finetune-cosmos-reason"
        / "scripts"
        / "prepare_cosmos3_vlm_checkpoint.py"
    )


def checkpoint_fingerprint(path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    """Fingerprint checkpoint content and return per-file integrity metadata.

    The walk is recursive because a native Cosmos3 Omni checkpoint keeps its
    shards in per-component subdirectories (``transformer/``,
    ``vision_encoder/``, ``vae/``, ``sound_tokenizer/``) rather than at the top
    level; its own config.json declares this as ``allow_patterns_overrides:
    ["*/*.safetensors"]``. Files are keyed by their checkpoint-relative path,
    not their basename, because that layout repeats basenames across
    components (``diffusion_pytorch_model.safetensors`` appears in both
    ``vae/`` and ``sound_tokenizer/``, ``config.json`` five times).
    """
    digest = hashlib.sha256()
    model_files = sorted(
        file
        for file in path.rglob("*")
        # ``.cache`` is Hugging Face download bookkeeping, not model content;
        # excluding it keeps the fingerprint stable across fetch methods.
        if file.is_file()
        and file.suffix in {".json", ".jinja", ".safetensors", ".txt"}
        and ".cache" not in file.relative_to(path).parts
    )
    weights = [file for file in model_files if file.suffix == ".safetensors"]
    if not weights:
        raise FileNotFoundError(f"baseline checkpoint has no safetensors weights: {path}")
    file_metadata: dict[str, dict[str, Any]] = {}
    for model_file in model_files:
        relative = str(model_file.relative_to(path))
        digest.update(relative.encode("utf-8"))
        file_digest = hashlib.sha256()
        with model_file.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
                file_digest.update(block)
        file_metadata[relative] = {
            "size": model_file.stat().st_size,
            "sha256": file_digest.hexdigest(),
        }
    return digest.hexdigest(), file_metadata


def source_fingerprint(path: Path) -> str:
    """Fingerprint checkpoint metadata and safetensors by content."""
    return checkpoint_fingerprint(path)[0]


def remote_model_id(value: str) -> str | None:
    """Return a Hugging Face model id, or None for an existing local directory."""
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"architecture model path does not exist: {path}")
        return None
    model_id = value.removeprefix("hf_model://")
    if "/" not in model_id:
        raise ValueError(
            "architecture model must be an existing local directory or a Hugging Face model id"
        )
    return model_id


def resolve_architecture_revision(model: str, requested_revision: str) -> str:
    """Resolve a remote Hugging Face revision to an immutable commit SHA."""
    model_id = remote_model_id(model)
    if model_id is None:
        return ""
    revision = requested_revision or DEFAULT_ARCHITECTURE_REVISION
    if IMMUTABLE_REVISION.fullmatch(revision):
        return revision.lower()

    model_url = urllib.parse.quote(model_id, safe="/")
    revision_url = urllib.parse.quote(revision, safe="")
    request = urllib.request.Request(
        f"https://huggingface.co/api/models/{model_url}/revision/{revision_url}",
        headers={"Accept": "application/json"},
    )
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # nosec B310 - https endpoint from skill configuration
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"could not resolve immutable Hugging Face revision for {model_id}@{revision}: {exc}"
        ) from exc
    resolved = payload.get("sha") if isinstance(payload, dict) else None
    if not isinstance(resolved, str) or not IMMUTABLE_REVISION.fullmatch(resolved):
        raise RuntimeError(
            f"Hugging Face returned no immutable commit SHA for {model_id}@{revision}"
        )
    return resolved.lower()


def validate_image_digest(value: str) -> None:
    """Require the immutable digest format recorded by the model-preparation contract."""
    if not SHA256_DIGEST.fullmatch(value):
        raise ValueError("--runtime-image-digest must be an immutable sha256:<64-hex> digest")


def verify_image_digest(image: str, expected_digest: str) -> None:
    """Verify that the local Cosmos-RL image matches its declared digest."""
    validate_image_digest(expected_digest)
    if shutil.which("docker") is None:
        raise RuntimeError("docker is required to verify the Cosmos-RL image")
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise RuntimeError(f"unexpected docker image inspect output for {image}")
    record = payload[0]
    observed = {record.get("Id")}
    for repo_digest in record.get("RepoDigests", []) or []:
        if isinstance(repo_digest, str) and "@" in repo_digest:
            observed.add(repo_digest.rsplit("@", 1)[-1])
    if expected_digest not in observed:
        values = sorted(value for value in observed if isinstance(value, str))
        raise ValueError(
            f"Cosmos-RL image digest mismatch for {image}: expected "
            f"{expected_digest}, observed {values}"
        )


def conversion_cache_fingerprint(
    prepared: Path,
    *,
    source: Path,
    architecture_model: str,
    architecture_revision: str,
    runtime_image: str,
    runtime_image_digest: str,
) -> str | None:
    """Return the content fingerprint when a cached conversion is exact and intact."""
    try:
        validate_qwen3_vl_checkpoint(prepared)
        provenance = load_json_object(prepared / "tao_conversion_provenance.json")
    except (FileNotFoundError, NotADirectoryError, ValueError, json.JSONDecodeError):
        return None
    base_model = provenance.get("base_model")
    architecture = provenance.get("architecture_model")
    preparation_runtime = provenance.get("preparation_runtime")
    if (
        not isinstance(base_model, dict)
        or not isinstance(architecture, dict)
        or not isinstance(preparation_runtime, dict)
    ):
        return None
    recorded_prepared = provenance.get("prepared")
    if not isinstance(recorded_prepared, dict):
        return None
    recorded_files = recorded_prepared.get("files")
    if not isinstance(recorded_files, list) or not recorded_files:
        return None
    fingerprint, current_files = checkpoint_fingerprint(prepared)
    recorded_names: set[str] = set()
    for record in recorded_files:
        if not isinstance(record, dict):
            return None
        name = record.get("name")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in recorded_names
        ):
            return None
        recorded_names.add(name)
        current = current_files.get(name)
        if current is None or current != {
            "size": record.get("size"),
            "sha256": record.get("sha256"),
        }:
            return None
    current_provenance_scope = {
        name
        for name in current_files
        if Path(name).parent == Path(".")
        and name != "tao_conversion_provenance.json"
        and Path(name).suffix in {".json", ".safetensors", ".jinja"}
    }
    if recorded_names != current_provenance_scope:
        return None
    source_values = {base_model.get("original"), base_model.get("resolved")}
    matches = bool({str(source), str(source.resolve())} & source_values) and (
        architecture.get("original") == architecture_model
        and provenance.get("architecture_model_revision") == architecture_revision
        and preparation_runtime.get("image") == runtime_image
        and preparation_runtime.get("image_digest") == runtime_image_digest
    )
    return fingerprint if matches else None


def conversion_fingerprint(
    *,
    source_fingerprint_value: str,
    architecture_model: str,
    architecture_revision: str,
    runtime_image_digest: str,
) -> str:
    """Fingerprint every input that can change converted checkpoint contents."""
    payload = {
        "source_fingerprint": source_fingerprint_value,
        "architecture_model": architecture_model,
        "architecture_revision": architecture_revision,
        "backend": "cosmos-rl",
        "runtime_image_digest": runtime_image_digest,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def validate_with_runtime(checkpoint: Path, runtime_image: str) -> dict[str, str]:
    """Require the pinned Cosmos-RL image to construct the prepared Qwen3-VL model."""
    if shutil.which("docker") is None:
        raise RuntimeError("docker is required to validate the prepared Cosmos Reason model")
    runtime_user = os.environ.get("USER") or os.environ.get("LOGNAME") or "tao"
    validation = "\n".join(
        (
            "from accelerate import init_empty_weights",
            "from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor",
            "config = AutoConfig.from_pretrained('/model', local_files_only=True)",
            "assert config.model_type == 'qwen3_vl', config.model_type",
            "AutoProcessor.from_pretrained('/model', local_files_only=True)",
            "with init_empty_weights():",
            "    AutoModelForImageTextToText.from_config(config)",
            "print('prepared checkpoint loads as qwen3_vl')",
        )
    )
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            "HOME=/tmp",
            "-e",
            f"USER={runtime_user}",
            "-e",
            f"LOGNAME={runtime_user}",
            "-e",
            "XDG_CACHE_HOME=/tmp/.cache",
            "-e",
            "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor",
            "-v",
            f"{checkpoint}:/model:ro",
            "--entrypoint",
            "python",
            runtime_image,
            "-c",
            validation,
        ],
        check=True,
    )
    return {"image": runtime_image, "status": "passed"}


def prepare_model(
    *,
    workspace: Path,
    workflow_yaml: Path,
    run_dir: Path,
    runtime_image: str,
    runtime_image_digest: str,
    model_preparation_script: Path,
    architecture_model: str,
    architecture_revision: str,
) -> dict[str, Any]:
    """Prepare or reuse a Qwen3-VL baseline and write its provenance manifest."""
    config = load_yaml(workflow_yaml)
    cosmos_reason = require_mapping(config, "cosmos_reason")
    source = existing_absolute_path(
        require_string(cosmos_reason, "cosmos_reason.baseline_model_path"),
        workspace,
        "cosmos_reason.baseline_model_path",
        "dir",
    )
    model_type = checkpoint_model_type(source)
    if model_type not in {"qwen3_vl", "cosmos3_omni"}:
        raise ValueError(
            "unsupported Cosmos Reason baseline model_type "
            f"{model_type!r}; expected 'qwen3_vl' or 'cosmos3_omni'"
        )
    fingerprint = source_fingerprint(source)
    verify_image_digest(runtime_image, runtime_image_digest)
    preparation_command: list[str] | None = None
    resolved_architecture_revision = ""
    resolved_conversion_fingerprint = ""
    prepared_fingerprint_value = ""

    if model_type == "qwen3_vl":
        prepared = source
        preparation = "reused_qwen3_vl"
        prepared_fingerprint_value = fingerprint
    elif model_type == "cosmos3_omni":
        if not model_preparation_script.is_file():
            raise FileNotFoundError(
                f"Cosmos Reason model-preparation helper does not exist: {model_preparation_script}"
            )
        resolved_architecture_revision = resolve_architecture_revision(
            architecture_model,
            architecture_revision,
        )
        resolved_conversion_fingerprint = conversion_fingerprint(
            source_fingerprint_value=fingerprint,
            architecture_model=architecture_model,
            architecture_revision=resolved_architecture_revision,
            runtime_image_digest=runtime_image_digest,
        )
        prepared = (
            workspace
            / "model"
            / "prepared"
            / fingerprint[:16]
            / resolved_conversion_fingerprint[:16]
            / "prepared"
        )
        preparation_command = [
            sys.executable,
            str(model_preparation_script),
            "--backend",
            "cosmos-rl",
            "--base-model-path-or-uri",
            str(source),
            "--base-model-identity",
            str(source.resolve()),
            "--vlm-architecture-model-path-or-uri",
            architecture_model,
            "--vlm-architecture-model-identity",
            architecture_model,
            "--output-path",
            str(prepared),
            "--cache-dir",
            str(workspace / "hf_cache"),
            "--runtime-image",
            runtime_image,
            "--runtime-image-digest",
            runtime_image_digest,
        ]
        if resolved_architecture_revision:
            preparation_command.extend(
                ["--vlm-architecture-model-revision", resolved_architecture_revision]
            )
        prepared_fingerprint_value = conversion_cache_fingerprint(
            prepared,
            source=source,
            architecture_model=architecture_model,
            architecture_revision=resolved_architecture_revision,
            runtime_image=runtime_image,
            runtime_image_digest=runtime_image_digest,
        ) or ""
        if prepared_fingerprint_value:
            preparation = "reused_converted_cosmos3_omni"
        else:
            if prepared.exists():
                preparation_command.append("--force")
            subprocess.run(preparation_command, check=True)
            preparation = "converted_cosmos3_omni"
    validation = validate_qwen3_vl_checkpoint(prepared)
    runtime_validation = validate_with_runtime(prepared, runtime_image)
    manifest = {
        "schema_version": 1,
        "status": "ready",
        "preparation": preparation,
        "source_model_path": str(source),
        "source_model_type": model_type,
        "source_fingerprint": fingerprint,
        "prepared_model_path": str(prepared),
        "prepared_fingerprint": prepared_fingerprint_value or source_fingerprint(prepared),
        "prepared_validation": validation,
        "architecture_model": architecture_model if model_type == "cosmos3_omni" else None,
        "architecture_revision": resolved_architecture_revision or None,
        "conversion_fingerprint": resolved_conversion_fingerprint or None,
        "runtime_image": runtime_image,
        "runtime_image_digest": runtime_image_digest,
        "runtime_validation": runtime_validation,
        "preparation_command": preparation_command,
        "preparation_command_executed": preparation == "converted_cosmos3_omni",
    }
    output_path = run_dir / "baseline" / "model_preparation.json"
    atomic_write_json(output_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--workflow-yaml", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--runtime-image-digest", required=True)
    parser.add_argument(
        "--model-preparation-script",
        type=Path,
        default=default_model_preparation_script(),
    )
    parser.add_argument("--architecture-model", default=DEFAULT_ARCHITECTURE_MODEL)
    parser.add_argument("--architecture-revision", default=DEFAULT_ARCHITECTURE_REVISION)
    return parser.parse_args()


def main() -> int:
    """Prepare the configured baseline and report the selected checkpoint."""
    args = parse_args()
    workspace = absolute_path(args.workspace)
    workflow_yaml = absolute_path(args.workflow_yaml)
    run_dir = absolute_path(args.run_dir)
    if not workspace.is_dir():
        raise NotADirectoryError(f"workspace does not exist: {workspace}")
    if not workflow_yaml.is_file():
        raise FileNotFoundError(f"workflow YAML does not exist: {workflow_yaml}")
    path_in_workspace(workflow_yaml, workspace, "workflow YAML")
    path_in_workspace(run_dir, workspace, "run directory")

    manifest = prepare_model(
        workspace=workspace,
        workflow_yaml=workflow_yaml,
        run_dir=run_dir,
        runtime_image=args.runtime_image,
        runtime_image_digest=args.runtime_image_digest,
        model_preparation_script=absolute_path(args.model_preparation_script),
        architecture_model=args.architecture_model,
        architecture_revision=args.architecture_revision,
    )
    print(f"prepared_model: {manifest['prepared_model_path']}")
    print(f"model_preparation: {run_dir / 'baseline' / 'model_preparation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
