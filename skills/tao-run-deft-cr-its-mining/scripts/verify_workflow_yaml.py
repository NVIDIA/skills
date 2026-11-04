#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the high-level DEFT CR ITS mining workflow YAML."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from workflow_common import (
    MODALITY_CHOICES,
    absolute_path,
    checkpoint_model_type,
    existing_absolute_path,
    load_yaml,
    optional_embedding_parquets,
    optional_bool,
    require_mapping,
    require_string,
    validate_qwen3_vl_checkpoint,
)


HF_MODEL_PREFIX = "hf_model://"


def require_positive_int(section: dict[str, Any], dotted_key: str) -> int:
    """Return a required positive integer value from a section."""
    key = dotted_key.rsplit(".", 1)[-1]
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"missing required positive integer: {dotted_key}")
    return value


def validate_optional_checkpoint(mining: dict[str, Any], workspace: Path) -> str | None:
    """Validate the optional local checkpoint path or remote Hugging Face model id."""
    value = mining.get("cosmos_embed_checkpoint_path")
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError("mining.cosmos_embed_checkpoint_path must be a string or null")
    configured_path = Path(os.path.expanduser(value))
    if configured_path.is_absolute():
        return str(
            existing_absolute_path(
                value,
                workspace,
                "mining.cosmos_embed_checkpoint_path",
                "path",
            )
        )
    model_id = value[len(HF_MODEL_PREFIX) :] if value.startswith(HF_MODEL_PREFIX) else value
    if model_id.startswith(".") or "/" not in model_id:
        raise ValueError(
            "mining.cosmos_embed_checkpoint_path must be an absolute local path, "
            "a Hugging Face model id like nvidia/Cosmos-Embed1-224p, or null"
        )
    return value


def validate_cosmos_embed_template(path: Path) -> int:
    """Return the positive GPU count declared by a Cosmos Embed inference template."""
    template = load_yaml(path)
    inference = template.get("inference")
    if not isinstance(inference, dict):
        raise ValueError(f"{path}: missing required object 'inference'")
    num_gpus = inference.get("num_gpus")
    if not isinstance(num_gpus, int) or isinstance(num_gpus, bool) or num_gpus < 1:
        raise ValueError(f"{path}: inference.num_gpus must be a positive integer")
    return num_gpus


def load_llava_annotations(annotation_path: Path) -> list[dict[str, Any]]:
    """Read a LLaVA annotation file and require a list of JSON objects."""
    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{annotation_path}: expected a JSON array of LLaVA annotation items")
    if not payload:
        raise ValueError(f"{annotation_path}: annotation file has no items")
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{annotation_path}: item {index} is not a JSON object")
    return payload


def resolve_llava_media_path(media_value: str, media_dir: Path) -> Path:
    """Resolve a LLaVA video field using the configured dataset media directory."""
    if os.path.isabs(media_value):
        return Path(os.path.normpath(os.path.expanduser(media_value)))
    return Path(os.path.normpath(os.path.join(str(media_dir), media_value)))


def validate_annotations_match_media_dir(
    annotation_path: Path,
    media_dir: Path,
    dataset_key: str,
) -> int:
    """Require every LLaVA video item to resolve to an existing file."""
    items = load_llava_annotations(annotation_path)
    missing: list[tuple[int, Path]] = []
    annotation_ids: set[str] = set()
    for index, item in enumerate(items, start=1):
        annotation_id = item.get("id")
        if not isinstance(annotation_id, str) or not annotation_id.strip():
            raise ValueError(f"{annotation_path}: item {index} is missing non-empty field 'id'")
        if annotation_id in annotation_ids:
            raise ValueError(f"{annotation_path}: duplicate annotation id: {annotation_id!r}")
        annotation_ids.add(annotation_id)
        media_value = item.get("video")
        if not isinstance(media_value, str) or not media_value.strip():
            raise ValueError(f"{annotation_path}: item {index} is missing non-empty field 'video'")
        media_path = resolve_llava_media_path(media_value, media_dir)
        if not media_path.is_file():
            missing.append((index, media_path))

    if missing:
        examples = ", ".join(f"item {index}: {path}" for index, path in missing[:5])
        extra = "" if len(missing) <= 5 else f", ... {len(missing) - 5} more"
        raise FileNotFoundError(
            f"{dataset_key} annotations are not compatible with {dataset_key}.media_dir; "
            f"{len(missing)} referenced media file(s) were not found after resolving relative "
            f"'video' values against {media_dir}: {examples}{extra}"
        )
    return len(items)


def validate_workflow_config(config: dict[str, Any], workspace: Path) -> dict[str, Any]:
    """Validate required workflow fields and return resolved values for logging."""
    if not workspace.is_dir():
        raise NotADirectoryError(f"workspace must be an existing directory: {workspace}")

    run = require_mapping(config, "run")
    run_name = run.get("name")
    if run_name is not None and (not isinstance(run_name, str) or not run_name.strip()):
        raise ValueError("run.name must be a non-empty string or null")
    max_iterations = require_positive_int(run, "run.max_iterations")

    kpi_dataset = require_mapping(config, "kpi_dataset")
    kpi_annotations = existing_absolute_path(
        require_string(kpi_dataset, "kpi_dataset.annotations_path"),
        workspace,
        "kpi_dataset.annotations_path",
        "file",
    )
    kpi_media_dir = existing_absolute_path(
        require_string(kpi_dataset, "kpi_dataset.media_dir"),
        workspace,
        "kpi_dataset.media_dir",
        "dir",
    )
    kpi_annotation_count = validate_annotations_match_media_dir(
        kpi_annotations,
        kpi_media_dir,
        "kpi_dataset",
    )

    train_dataset = require_mapping(config, "train_dataset")
    train_annotations = existing_absolute_path(
        require_string(train_dataset, "train_dataset.annotations_path"),
        workspace,
        "train_dataset.annotations_path",
        "file",
    )
    train_media_dir = existing_absolute_path(
        require_string(train_dataset, "train_dataset.media_dir"),
        workspace,
        "train_dataset.media_dir",
        "dir",
    )
    train_annotation_count = validate_annotations_match_media_dir(
        train_annotations,
        train_media_dir,
        "train_dataset",
    )

    cosmos_reason = require_mapping(config, "cosmos_reason")
    baseline_model = existing_absolute_path(
        require_string(cosmos_reason, "cosmos_reason.baseline_model_path"),
        workspace,
        "cosmos_reason.baseline_model_path",
        "dir",
    )
    baseline_model_type = checkpoint_model_type(baseline_model)
    if baseline_model_type == "qwen3_vl":
        validate_qwen3_vl_checkpoint(baseline_model)
    elif baseline_model_type != "cosmos3_omni":
        raise ValueError(
            "cosmos_reason.baseline_model_path has unsupported model_type "
            f"{baseline_model_type!r}; expected 'qwen3_vl' or 'cosmos3_omni'"
        )
    base_evaluate_toml = existing_absolute_path(
        require_string(cosmos_reason, "cosmos_reason.base_evaluate_toml"),
        workspace,
        "cosmos_reason.base_evaluate_toml",
        "file",
    )
    base_train_toml = existing_absolute_path(
        require_string(cosmos_reason, "cosmos_reason.base_train_toml"),
        workspace,
        "cosmos_reason.base_train_toml",
        "file",
    )
    continual_model = optional_bool(cosmos_reason, "cosmos_reason.continual_model", False)

    mining = require_mapping(config, "mining")
    embeddings_spec_template = existing_absolute_path(
        require_string(mining, "mining.embeddings_spec_template"),
        workspace,
        "mining.embeddings_spec_template",
        "file",
    )
    embeddings_num_gpus = validate_cosmos_embed_template(embeddings_spec_template)
    embeddings_modality = require_string(mining, "mining.embeddings_modality")
    if embeddings_modality not in MODALITY_CHOICES:
        choices = ", ".join(sorted(MODALITY_CHOICES))
        raise ValueError(f"mining.embeddings_modality must be one of: {choices}")
    mining_spec_template = existing_absolute_path(
        require_string(mining, "mining.mining_spec_template"),
        workspace,
        "mining.mining_spec_template",
        "file",
    )
    mine_unique_only = optional_bool(mining, "mining.mine_unique_only", True)
    cosmos_embed_checkpoint = validate_optional_checkpoint(mining, workspace)
    embedding_parquets = optional_embedding_parquets(mining, workspace, embeddings_modality)

    return {
        "workspace": str(workspace),
        "run_name": run_name,
        "max_iterations": max_iterations,
        "kpi_annotations": str(kpi_annotations),
        "kpi_media_dir": str(kpi_media_dir),
        "kpi_annotation_count": kpi_annotation_count,
        "train_annotations": str(train_annotations),
        "train_media_dir": str(train_media_dir),
        "train_annotation_count": train_annotation_count,
        "baseline_model": str(baseline_model),
        "baseline_model_type": baseline_model_type,
        "base_evaluate_toml": str(base_evaluate_toml),
        "base_train_toml": str(base_train_toml),
        "continual_model": continual_model,
        "embeddings_spec_template": str(embeddings_spec_template),
        "embeddings_num_gpus": embeddings_num_gpus,
        "embeddings_modality": embeddings_modality,
        "cosmos_embed_checkpoint": cosmos_embed_checkpoint,
        "kpi_embeddings_parquet": str(embedding_parquets["kpi"]) if embedding_parquets["kpi"] else None,
        "train_embeddings_parquet": str(embedding_parquets["train"]) if embedding_parquets["train"] else None,
        "mining_spec_template": str(mining_spec_template),
        "mine_unique_only": mine_unique_only,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-yaml", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """Validate workflow.yaml and print resolved paths."""
    args = parse_args()
    workflow_yaml = absolute_path(args.workflow_yaml)
    workspace = absolute_path(args.workspace)
    try:
        if not workflow_yaml.is_file():
            raise FileNotFoundError(f"workflow YAML does not exist: {workflow_yaml}")
        resolved = validate_workflow_config(load_yaml(workflow_yaml), workspace)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("workflow.yaml is valid")
    for key, value in resolved.items():
        if value is not None:
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
