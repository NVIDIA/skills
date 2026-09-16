#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare mined annotations and configuration for Cosmos Reason training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from workflow_common import (
    absolute_path,
    dump_toml,
    existing_absolute_path,
    load_json_array,
    load_toml,
    load_yaml,
    path_in_workspace,
    require_mapping,
    require_string,
    write_json_array,
)


def build_llava_records(
    mined_neighbors_parquet: Path,
    train_embeddings_parquet: Path,
    train_lookup_parquet: Path,
) -> list[dict[str, Any]]:
    """Recover mined source modalities and return their LLaVA records."""
    neighbors = pd.read_parquet(mined_neighbors_parquet)
    source_col = "source_filepath" if "source_filepath" in neighbors.columns else "filepath"
    if source_col not in neighbors.columns:
        raise ValueError(f"{mined_neighbors_parquet}: missing source filepath column")

    embeddings = pd.read_parquet(train_embeddings_parquet)
    required_embedding_columns = {"filepath", "modality"}
    missing = required_embedding_columns - set(embeddings.columns)
    if missing:
        raise ValueError(f"{train_embeddings_parquet}: missing required columns: {sorted(missing)}")
    source_metadata = embeddings[["filepath", "modality"]].drop_duplicates()
    conflicts = source_metadata.groupby("filepath")["modality"].nunique()
    conflicting_paths = conflicts[conflicts > 1].index.astype(str).tolist()
    if conflicting_paths:
        raise ValueError(
            f"{train_embeddings_parquet}: source filepaths map to multiple modalities: "
            f"{conflicting_paths[:5]}"
        )

    mined = neighbors[[source_col]].drop_duplicates().rename(columns={source_col: "source_filepath"})
    mined = mined.merge(
        source_metadata,
        left_on="source_filepath",
        right_on="filepath",
        how="left",
        validate="one_to_one",
    )
    missing_metadata = mined[mined["modality"].isna()]["source_filepath"].astype(str).tolist()
    if missing_metadata:
        raise ValueError(
            "mined source filepaths are absent from the train embeddings parquet: "
            f"{missing_metadata[:5]}"
        )
    unsupported = sorted(set(mined["modality"].astype(str)) - {"text", "video"})
    if unsupported:
        raise ValueError(f"unsupported source modalities in {train_embeddings_parquet}: {unsupported}")

    lookup = pd.read_parquet(train_lookup_parquet)
    required = {"filepath", "annotation_id", "video_path", "question", "answer"}
    missing = required - set(lookup.columns)
    if missing:
        raise ValueError(f"{train_lookup_parquet}: missing required columns: {sorted(missing)}")

    matched: list[pd.DataFrame] = []
    text_sources = mined[mined["modality"] == "text"]
    if not text_sources.empty:
        matched.append(
            text_sources.merge(lookup, left_on="source_filepath", right_on="filepath", how="inner")
        )
    video_sources = mined[mined["modality"] == "video"]
    if not video_sources.empty:
        matched.append(
            video_sources.merge(lookup, left_on="source_filepath", right_on="video_path", how="inner")
        )
    merged = pd.concat(matched, ignore_index=True) if matched else pd.DataFrame()
    if merged.empty:
        raise RuntimeError("no mined neighbors joined to train lookup rows")
    unmatched = sorted(
        set(mined["source_filepath"].astype(str)) - set(merged["source_filepath"].astype(str))
    )
    if unmatched:
        raise ValueError(f"mined source filepaths did not match the train lookup: {unmatched[:5]}")

    records: list[dict[str, Any]] = []
    for _, row in merged.drop_duplicates(subset=["annotation_id"]).iterrows():
        records.append(
            {
                "id": str(row["annotation_id"]),
                "video": str(row["video_path"]),
                "conversations": [
                    {"from": "human", "value": f"<video>\n{row['question']}"},
                    {"from": "gpt", "value": str(row["answer"])},
                ],
            }
        )
    return records


def annotation_id(record: dict[str, Any], source: Path, index: int) -> str:
    """Return a required LLaVA annotation id."""
    value = record.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: item {index} is missing non-empty 'id'")
    return value


def assemble_annotations(
    previous_annotations: Path | None,
    mined_annotations: Path,
) -> list[dict[str, Any]]:
    """Merge previous and current mined annotations, deduplicated by LLaVA id."""
    sources = [path for path in (previous_annotations, mined_annotations) if path is not None]
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"annotation source does not exist: {source}")
        for index, record in enumerate(load_json_array(source), start=1):
            record_id = annotation_id(record, source, index)
            if record_id in seen:
                continue
            records.append(record)
            seen.add(record_id)
    return records


def patch_train_config(
    config: dict[str, Any],
    *,
    train_dir: Path,
    train_annotations: Path,
    train_media_dir: Path,
    val_annotations: Path,
    val_media_dir: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Patch run-specific fields in a Cosmos Reason training config."""
    config["results_dir"] = str(train_dir)
    config.setdefault("train", {})
    config["train"]["output_dir"] = str(train_dir)
    config.setdefault("policy", {})
    config["policy"]["model_name_or_path"] = str(checkpoint_path)
    config.setdefault("custom", {})
    config["custom"].setdefault("train_dataset", {})
    config["custom"]["train_dataset"]["annotation_path"] = str(train_annotations)
    config["custom"]["train_dataset"]["media_path"] = str(train_media_dir)
    config["custom"].setdefault("val_dataset", {})
    config["custom"]["val_dataset"]["annotation_path"] = str(val_annotations)
    config["custom"]["val_dataset"]["media_path"] = str(val_media_dir)
    return config


def positive_int(mapping: dict[str, Any], key: str, section: str) -> int:
    """Read a positive integer training parameter."""
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{section}.{key} must be a positive integer")
    return value


def validate_expected_optimizer_steps(config: dict[str, Any], sample_count: int) -> int:
    """Require the assembled dataset and training parameters to produce a step."""
    train = config.get("train")
    policy = config.get("policy")
    if not isinstance(train, dict) or not isinstance(policy, dict):
        raise ValueError("training config must contain [train] and [policy]")
    train_policy = train.get("train_policy")
    parallelism = policy.get("parallelism")
    if not isinstance(train_policy, dict) or not isinstance(parallelism, dict):
        raise ValueError(
            "training config must contain [train.train_policy] and [policy.parallelism]"
        )

    epochs = positive_int(train, "epoch", "train")
    batch_per_replica = positive_int(train, "train_batch_per_replica", "train")
    dp_shard_size = positive_int(parallelism, "dp_shard_size", "policy.parallelism")
    dp_replicate_size = positive_int(
        parallelism,
        "dp_replicate_size",
        "policy.parallelism",
    )
    drop_last = train_policy.get("dataloader_drop_last", True)
    if not isinstance(drop_last, bool):
        raise ValueError("train.train_policy.dataloader_drop_last must be true or false")

    effective_batch = batch_per_replica * dp_shard_size * dp_replicate_size
    if drop_last:
        steps_per_epoch = sample_count // effective_batch
    else:
        steps_per_epoch = (sample_count + effective_batch - 1) // effective_batch
    expected_steps = epochs * steps_per_epoch
    if expected_steps < 1:
        raise ValueError(
            "assembled training data and the configured Cosmos-RL parameters produce zero "
            f"optimizer steps: samples={sample_count}, epoch={epochs}, "
            f"train_batch_per_replica={batch_per_replica}, dp_shard_size={dp_shard_size}, "
            f"dp_replicate_size={dp_replicate_size}, dataloader_drop_last={drop_last}. "
            "Stop before training and tell the user that the training parameters likely need "
            "to be tuned for this iteration's dataset size and GPU count."
        )
    return expected_steps


def training_checkpoint(
    config: dict[str, Any],
    workspace: Path,
    run_dir: Path,
    iteration: int,
) -> Path:
    """Select the baseline or previous evaluated checkpoint for this iteration."""
    cosmos_reason = require_mapping(config, "cosmos_reason")
    continual_model = cosmos_reason.get("continual_model", False)
    if not isinstance(continual_model, bool):
        raise ValueError("cosmos_reason.continual_model must be true or false")
    if iteration == 1 or not continual_model:
        return existing_absolute_path(
            require_string(cosmos_reason, "cosmos_reason.baseline_model_path"),
            workspace,
            "cosmos_reason.baseline_model_path",
            "path",
        )

    previous_evaluate_toml = run_dir / f"iter_{iteration - 1}" / "evaluate" / "specs" / "evaluate.toml"
    if not previous_evaluate_toml.is_file():
        raise FileNotFoundError(
            f"previous iteration evaluation config does not exist: {previous_evaluate_toml}"
        )
    previous_config = load_toml(previous_evaluate_toml)
    model = previous_config.get("model")
    checkpoint_value = model.get("model_name") if isinstance(model, dict) else None
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError(f"{previous_evaluate_toml}: missing model.model_name")
    checkpoint_path = absolute_path(checkpoint_value)
    path_in_workspace(checkpoint_path, run_dir, "previous iteration checkpoint")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"previous iteration checkpoint does not exist: {checkpoint_path}")
    return checkpoint_path


def generate_train_toml(
    workspace: Path,
    workflow_yaml: Path,
    run_dir: Path,
    *,
    iteration: int,
    train_annotations: Path,
    checkpoint_path: Path,
) -> Path:
    """Write the Cosmos Reason training TOML for one iteration."""
    config = load_yaml(workflow_yaml)
    kpi_dataset = require_mapping(config, "kpi_dataset")
    train_dataset = require_mapping(config, "train_dataset")
    cosmos_reason = require_mapping(config, "cosmos_reason")
    base_train_toml = existing_absolute_path(
        require_string(cosmos_reason, "cosmos_reason.base_train_toml"),
        workspace,
        "cosmos_reason.base_train_toml",
        "file",
    )
    train_media_dir = existing_absolute_path(
        require_string(train_dataset, "train_dataset.media_dir"),
        workspace,
        "train_dataset.media_dir",
        "dir",
    )
    val_annotations = existing_absolute_path(
        require_string(kpi_dataset, "kpi_dataset.annotations_path"),
        workspace,
        "kpi_dataset.annotations_path",
        "file",
    )
    val_media_dir = existing_absolute_path(
        require_string(kpi_dataset, "kpi_dataset.media_dir"),
        workspace,
        "kpi_dataset.media_dir",
        "dir",
    )
    train_dir = run_dir / f"iter_{iteration}" / "train"
    output_path = train_dir / "specs" / "train.toml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patched = patch_train_config(
        load_toml(base_train_toml),
        train_dir=train_dir,
        train_annotations=train_annotations,
        train_media_dir=train_media_dir,
        val_annotations=val_annotations,
        val_media_dir=val_media_dir,
        checkpoint_path=checkpoint_path,
    )
    sample_count = len(load_json_array(train_annotations))
    expected_steps = validate_expected_optimizer_steps(patched, sample_count)
    output_path.write_text(dump_toml(patched), encoding="utf-8")
    print(f"training samples: {sample_count}")
    print(f"expected optimizer steps: {expected_steps}")
    return output_path


def prepare_training(
    workspace: Path,
    workflow_yaml: Path,
    run_dir: Path,
    iteration: int,
) -> dict[str, Path]:
    """Build current/accumulated annotations and the training TOML."""
    if iteration < 1:
        raise ValueError("iteration must be >= 1")
    iteration_dir = run_dir / f"iter_{iteration}"
    mined_annotations = iteration_dir / "mining" / "mined_train_annotations.json"
    train_annotations = iteration_dir / "train" / "train_annotations.json"

    mined_records = build_llava_records(
        iteration_dir / "mining" / "mined_neighbors.parquet",
        run_dir / "embedding_parquets" / "train" / "embeddings.parquet",
        run_dir / "cosmos_embed_output" / "train" / "lookup.parquet",
    )
    write_json_array(mined_annotations, mined_records)

    previous_annotations = None
    if iteration > 1:
        previous_annotations = run_dir / f"iter_{iteration - 1}" / "train" / "train_annotations.json"
    assembled_records = assemble_annotations(previous_annotations, mined_annotations)
    write_json_array(train_annotations, assembled_records)

    config = load_yaml(workflow_yaml)
    checkpoint_path = training_checkpoint(config, workspace, run_dir, iteration)
    train_toml = generate_train_toml(
        workspace,
        workflow_yaml,
        run_dir,
        iteration=iteration,
        train_annotations=train_annotations,
        checkpoint_path=checkpoint_path,
    )
    return {
        "mined_annotations": mined_annotations,
        "train_annotations": train_annotations,
        "checkpoint": checkpoint_path,
        "toml": train_toml,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--workflow-yaml", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--iteration", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    """Prepare all Cosmos Reason training inputs for one iteration."""
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

    outputs = prepare_training(workspace, workflow_yaml, run_dir, args.iteration)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
