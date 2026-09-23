#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare Cosmos Embed inference specs and lookup files from workflow.yaml.

Cosmos Embed can race when multiple GPU workers try to download the same Hugging Face
checkpoint during startup. This script resolves remote HF checkpoints before inference
spec generation, so every generated spec points at an already-local checkpoint path.
"""

from __future__ import annotations

import argparse
import copy
import os
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd

from workflow_common import (
    MODALITY_CHOICES,
    absolute_path,
    atomic_write_parquet,
    clean_question,
    dataset_modalities,
    existing_absolute_path,
    load_json_array,
    load_yaml,
    optional_embedding_parquets,
    path_in_workspace,
    require_mapping,
    require_string,
    write_yaml,
)

DEFAULT_COSMOS_EMBED_MODEL = "nvidia/Cosmos-Embed1-224p"
HF_MODEL_PREFIX = "hf_model://"


def hf_cache_dir(workspace: Path) -> Path:
    """Return the workflow Hugging Face cache directory, creating it if needed."""
    cache_dir = workspace / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not cache_dir.is_dir():
        raise NotADirectoryError(f"HF cache is not a directory: {cache_dir}")
    if not os.access(cache_dir, os.W_OK):
        raise PermissionError(f"HF cache is not writable: {cache_dir}")
    return cache_dir


def download_hf_checkpoint(model_id: str, workspace: Path) -> Path:
    """Download or reuse a remote Hugging Face checkpoint under workspace/hf_cache."""
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required to download a remote Cosmos Embed checkpoint. "
            "Install it or set mining.cosmos_embed_checkpoint_path to a local absolute path."
        ) from exc
    cache_dir = hf_cache_dir(workspace)
    return Path(snapshot_download(model_id, cache_dir=str(cache_dir)))  # nosec B615 - pinned model_id from skill config


def normalize_hf_model_id(value: str) -> str:
    """Return the raw Hugging Face model id from supported remote forms."""
    if value.startswith(HF_MODEL_PREFIX):
        value = value[len(HF_MODEL_PREFIX) :]
    if "/" not in value or value.startswith("."):
        raise ValueError(
            "mining.cosmos_embed_checkpoint_path must be an absolute local path, "
            "a Hugging Face model id like nvidia/Cosmos-Embed1-224p, or null"
        )
    return value


def resolve_checkpoint(mining: dict[str, Any], workspace: Path) -> Path:
    """Resolve the Cosmos Embed checkpoint from workflow.yaml to a local path."""
    value = mining.get("cosmos_embed_checkpoint_path")
    if value in (None, ""):
        return download_hf_checkpoint(DEFAULT_COSMOS_EMBED_MODEL, workspace)
    if not isinstance(value, str):
        raise ValueError("mining.cosmos_embed_checkpoint_path must be a string or null")
    candidate = Path(os.path.expanduser(value))
    if candidate.is_absolute():
        return existing_absolute_path(value, workspace, "mining.cosmos_embed_checkpoint_path", "path")
    return download_hf_checkpoint(normalize_hf_model_id(value), workspace)


def load_template(path: Path) -> dict[str, Any]:
    """Load and validate the bundled Cosmos Embed inference template."""
    payload = load_yaml(path)
    if not isinstance(payload.get("inference"), dict):
        raise ValueError(f"{path}: missing object field 'inference'")
    if not isinstance(payload["inference"].get("query"), dict):
        raise ValueError(f"{path}: missing object field 'inference.query'")
    if not isinstance(payload.get("model"), dict):
        raise ValueError(f"{path}: missing object field 'model'")
    num_gpus = payload["inference"].get("num_gpus")
    if not isinstance(num_gpus, int) or isinstance(num_gpus, bool) or num_gpus < 1:
        raise ValueError(f"{path}: inference.num_gpus must be a positive integer")
    return payload


def llava_video_path(item: dict[str, Any], source: Path, index: int) -> str:
    """Return the LLaVA video path for an annotation item."""
    value = item.get("video")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: item {index} is missing non-empty field 'video'")
    return value


def llava_annotation_id(item: dict[str, Any], source: Path, index: int) -> str:
    """Return the stable LLaVA annotation identifier."""
    value = item.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: item {index} is missing non-empty field 'id'")
    return value


def llava_question(item: dict[str, Any], source: Path, index: int) -> str:
    """Extract the first human question from a LLaVA annotation item."""
    conversations = item.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"{source}: item {index} is missing non-empty 'conversations'")
    first = conversations[0]
    if not isinstance(first, dict):
        raise ValueError(f"{source}: item {index} first conversation must be an object")
    value = first.get("value")
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{source}: item {index} first conversation field 'value' must be a non-empty string"
        )
    return value


def llava_answer(item: dict[str, Any]) -> str:
    """Return the first assistant answer from a LLaVA item, or an empty string."""
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        return ""
    for turn in conversations:
        if isinstance(turn, dict) and turn.get("from") == "gpt" and isinstance(turn.get("value"), str):
            return turn["value"]
    return ""


def resolve_media_path(media_path: str, media_dir: Path) -> str:
    """Resolve a LLaVA video path against the dataset media directory."""
    if os.path.isabs(media_path):
        return os.path.normpath(os.path.expanduser(media_path))
    return os.path.normpath(os.path.join(str(media_dir), media_path))


def collect_embedding_inputs(
    annotation_json: Path,
    media_dir: Path,
    output_dir: Path,
) -> tuple[list[str], list[str], Path]:
    """Collect text/video queries and write lookup rows used by mining joins."""
    items = load_json_array(annotation_json)
    if not items:
        raise ValueError(f"{annotation_json}: no annotation items")

    question_dir = output_dir / "results" / "text" / "questions"
    question_dir.mkdir(parents=True, exist_ok=True)
    questions: list[str] = []
    videos_by_path: dict[str, None] = {}
    lookup_rows: list[dict[str, Any]] = []
    annotation_ids: set[str] = set()
    for index, item in enumerate(items):
        annotation_id = llava_annotation_id(item, annotation_json, index + 1)
        if annotation_id in annotation_ids:
            raise ValueError(f"{annotation_json}: duplicate annotation id: {annotation_id!r}")
        annotation_ids.add(annotation_id)
        question = clean_question(llava_question(item, annotation_json, index + 1))
        video_path = resolve_media_path(llava_video_path(item, annotation_json, index + 1), media_dir)
        question_path = question_dir / f"q_{index:05d}.txt"
        question_path.write_text(question, encoding="utf-8")

        questions.append(question)
        videos_by_path.setdefault(video_path, None)
        lookup_rows.append(
            {
                "filepath": str(question_path),
                "annotation_id": annotation_id,
                "video_path": video_path,
                "item_index": index,
                "question": question,
                "answer": llava_answer(item),
            }
        )

    lookup_path = output_dir / "lookup.parquet"
    atomic_write_parquet(pd.DataFrame(lookup_rows), lookup_path)
    return questions, list(videos_by_path), lookup_path


def build_spec(
    template: dict[str, Any],
    mode: str,
    output_dir: Path,
    checkpoint_path: Path,
    questions: list[str],
    videos: list[str],
) -> dict[str, Any]:
    """Create one Cosmos Embed inference spec for the requested modality."""
    spec = copy.deepcopy(template)
    spec_results_dir = output_dir / "results" / mode
    spec["results_dir"] = str(spec_results_dir)
    spec["inference"]["mode"] = mode
    spec["inference"]["checkpoint"] = str(checkpoint_path)
    spec["inference"]["query"]["input_texts"] = questions
    spec["inference"]["query"]["input_videos"] = videos
    spec["inference"]["save_dataset_pkl"] = str(spec_results_dir / "embeddings.pkl")
    return spec


def stage_provided_embedding_parquet(
    source: Path,
    parquet_dir: Path,
    lookup_path: Path,
) -> Path:
    """Stage a combined parquet, remapping text identifiers to this run's lookup."""
    destination = parquet_dir / "embeddings.parquet"
    embeddings = pd.read_parquet(source)
    text_mask = embeddings["modality"].astype(str) == "text"
    if text_mask.any():
        lookup = pd.read_parquet(lookup_path, columns=["filepath", "question"])
        filepath_queues: dict[str, deque[str]] = {}
        for _, row in lookup.iterrows():
            filepath_queues.setdefault(str(row["question"]), deque()).append(str(row["filepath"]))

        remapped: list[str] = []
        for filepath in embeddings.loc[text_mask, "filepath"].astype(str):
            question_path = Path(filepath)
            if not question_path.is_file():
                raise FileNotFoundError(
                    f"provided text embedding filepath does not exist: {question_path}"
                )
            question = clean_question(question_path.read_text(encoding="utf-8"))
            candidates = filepath_queues.get(question)
            if not candidates:
                raise ValueError(
                    f"provided text embedding question has no unconsumed annotation match: {question!r}"
                )
            remapped.append(candidates.popleft())
        unmatched = sum(len(candidates) for candidates in filepath_queues.values())
        if unmatched:
            raise ValueError(
                f"provided text embeddings are missing {unmatched} annotation question occurrence(s)"
            )
        embeddings.loc[text_mask, "filepath"] = remapped

    atomic_write_parquet(embeddings, destination)
    return destination


def write_specs(
    annotation_json: Path,
    media_dir: Path,
    output_dir: Path,
    parquet_dir: Path,
    checkpoint_path: Path | None,
    required_modes: list[str],
    spec_template: Path,
    provided_parquet: Path | None,
) -> list[Path]:
    """Stage a complete provided parquet or write every required inference spec."""
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    specs_dir = output_dir / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    questions, videos, lookup_path = collect_embedding_inputs(annotation_json, media_dir, output_dir)

    if provided_parquet is not None:
        staged_path = stage_provided_embedding_parquet(provided_parquet, parquet_dir, lookup_path)
        print(f"staged combined embeddings parquet: {provided_parquet} -> {staged_path}")
        print(f"lookup_parquet: {lookup_path}")
        print(f"media_dir: {media_dir}")
        print(f"output_dir: {output_dir}")
        print(f"parquet_dir: {parquet_dir}")
        return []

    if checkpoint_path is None:
        raise ValueError("missing Cosmos Embed checkpoint for generated embeddings")
    template = load_template(spec_template)

    written: list[Path] = []
    for mode in required_modes:
        (output_dir / "results" / mode).mkdir(parents=True, exist_ok=True)
        spec = build_spec(
            template=template,
            mode=mode,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            questions=questions if mode == "text" else [],
            videos=videos if mode == "video" else [],
        )
        spec_path = specs_dir / f"inference_{mode}.yaml"
        write_yaml(spec_path, spec)
        written.append(spec_path)
        count = len(questions) if mode == "text" else len(videos)
        print(f"wrote {count} {mode} queries -> {spec_path}")
        print(f"{mode}_num_gpus: {spec['inference']['num_gpus']}")

    print(f"lookup_parquet: {lookup_path}")
    print(f"media_dir: {media_dir}")
    print(f"output_dir: {output_dir}")
    print(f"parquet_dir: {parquet_dir}")
    return written


def workflow_dataset_paths(
    config: dict[str, Any],
    dataset: str,
    workspace: Path,
) -> tuple[Path, Path]:
    """Return annotation and media paths for one configured workflow dataset."""
    section_key = "kpi_dataset" if dataset == "kpi" else "train_dataset"
    section = require_mapping(config, section_key)
    annotation_json = existing_absolute_path(
        require_string(section, f"{section_key}.annotations_path"),
        workspace,
        f"{section_key}.annotations_path",
        "file",
    )
    media_dir = existing_absolute_path(
        require_string(section, f"{section_key}.media_dir"),
        workspace,
        f"{section_key}.media_dir",
        "dir",
    )
    return annotation_json, media_dir


def workflow_mining_config(
    config: dict[str, Any],
    workspace: Path,
) -> tuple[Path, str, Path | None, dict[str, Path | None]]:
    """Return embedding setup inputs, resolving a checkpoint only when needed."""
    mining = require_mapping(config, "mining")
    spec_template = existing_absolute_path(
        require_string(mining, "mining.embeddings_spec_template"),
        workspace,
        "mining.embeddings_spec_template",
        "file",
    )
    embedding_modality = require_string(mining, "mining.embeddings_modality")
    if embedding_modality not in MODALITY_CHOICES:
        choices = ", ".join(MODALITY_CHOICES)
        raise ValueError(f"mining.embeddings_modality must be one of: {choices}")
    provided_parquets = optional_embedding_parquets(mining, workspace, embedding_modality)
    needs_generation = any(provided_parquets[dataset] is None for dataset in ("kpi", "train"))
    checkpoint_path = resolve_checkpoint(mining, workspace) if needs_generation else None
    return spec_template, embedding_modality, checkpoint_path, provided_parquets


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--workflow-yaml", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Generate Cosmos Embed specs and lookup artifacts for one dataset."""
    args = parse_args()
    workspace = absolute_path(args.workspace)
    workflow_yaml = absolute_path(args.workflow_yaml)
    if not workspace.is_dir():
        raise NotADirectoryError(f"workspace does not exist: {workspace}")
    if not workflow_yaml.is_file():
        raise FileNotFoundError(f"workflow YAML does not exist: {workflow_yaml}")
    path_in_workspace(workflow_yaml, workspace, "workflow YAML")

    config = load_yaml(workflow_yaml)
    run_dir = absolute_path(args.run_dir)
    path_in_workspace(run_dir, workspace, "run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    spec_template, embedding_modality, checkpoint_path, provided_parquets = workflow_mining_config(
        config,
        workspace,
    )
    for dataset in ("kpi", "train"):
        annotation_json, media_dir = workflow_dataset_paths(config, dataset, workspace)
        output_dir = run_dir / "cosmos_embed_output" / dataset
        parquet_dir = run_dir / "embedding_parquets" / dataset
        write_specs(
            annotation_json=annotation_json,
            media_dir=media_dir,
            output_dir=output_dir,
            parquet_dir=parquet_dir,
            checkpoint_path=checkpoint_path,
            required_modes=dataset_modalities(dataset, embedding_modality),
            spec_template=spec_template,
            provided_parquet=provided_parquets[dataset],
        )
    print(f"run_dir: {run_dir}")


if __name__ == "__main__":
    main()
