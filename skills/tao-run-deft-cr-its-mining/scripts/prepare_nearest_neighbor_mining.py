#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare one per-iteration mining target and nearest-neighbor spec."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from workflow_common import (
    absolute_path,
    atomic_write_parquet,
    clean_question,
    existing_absolute_path,
    load_yaml,
    modality_list,
    normalize_media_path,
    optional_bool,
    path_in_workspace,
    read_jsonl,
    require_mapping,
    require_string,
    write_yaml,
)


def weak_samples(gaps_jsonl: Path) -> list[dict[str, str]]:
    """Read gap-analysis rows as unique weak `(video_path, question)` samples."""
    samples: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(read_jsonl(gaps_jsonl), start=1):
        video_id = record.get("video_id")
        question = record.get("question")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"{gaps_jsonl}: record {index} is missing non-empty 'video_id'")
        if not isinstance(question, str) or not question:
            raise ValueError(f"{gaps_jsonl}: record {index} is missing non-empty 'question'")
        key = (normalize_media_path(video_id), clean_question(question))
        if key in seen:
            continue
        seen.add(key)
        samples.append({"video_path": key[0], "question": key[1]})
    return samples


def build_text_target(
    gaps_jsonl: Path,
    kpi_embeddings_parquet: Path,
    kpi_lookup_parquet: Path,
    output_target_parquet: Path,
) -> int:
    """Filter KPI text embeddings to the failed `(video, question)` rows."""
    target = text_target_dataframe(gaps_jsonl, kpi_embeddings_parquet, kpi_lookup_parquet)
    atomic_write_parquet(target, output_target_parquet)
    return len(target)


def text_target_dataframe(
    gaps_jsonl: Path,
    kpi_embeddings_parquet: Path,
    kpi_lookup_parquet: Path,
) -> pd.DataFrame:
    """Return KPI text embedding rows matching failed `(video, question)` pairs."""
    samples = weak_samples(gaps_jsonl)
    lookup = pd.read_parquet(kpi_lookup_parquet)
    for column in ("filepath", "video_path", "question"):
        if column not in lookup.columns:
            raise ValueError(f"{kpi_lookup_parquet}: missing required column {column!r}")
    weak_pairs = {(sample["video_path"], sample["question"]) for sample in samples}
    lookup_pairs = lookup.assign(
        _video_path=lookup["video_path"].map(normalize_media_path),
        _question=lookup["question"].map(clean_question),
    )
    matched_lookup = lookup_pairs[
        lookup_pairs.apply(lambda row: (row["_video_path"], row["_question"]) in weak_pairs, axis=1)
    ]
    question_paths = set(matched_lookup["filepath"].tolist())
    embeddings = pd.read_parquet(kpi_embeddings_parquet)
    require_embedding_columns(embeddings, kpi_embeddings_parquet)
    target = embeddings[
        (embeddings["modality"] == "text") & embeddings["filepath"].isin(question_paths)
    ].reset_index(drop=True)
    if target.empty:
        raise RuntimeError("no KPI text embeddings matched the weak gap questions")
    return target


def build_video_target(
    gaps_jsonl: Path,
    kpi_embeddings_parquet: Path,
    output_target_parquet: Path,
) -> int:
    """Filter KPI video embeddings to unique failed videos."""
    target = video_target_dataframe(gaps_jsonl, kpi_embeddings_parquet)
    atomic_write_parquet(target, output_target_parquet)
    return len(target)


def video_target_dataframe(gaps_jsonl: Path, kpi_embeddings_parquet: Path) -> pd.DataFrame:
    """Return KPI video embedding rows matching unique failed videos."""
    weak_videos = {sample["video_path"] for sample in weak_samples(gaps_jsonl)}
    embeddings = pd.read_parquet(kpi_embeddings_parquet)
    require_embedding_columns(embeddings, kpi_embeddings_parquet)
    target = embeddings[
        (embeddings["modality"] == "video")
        & embeddings["filepath"].map(lambda value: normalize_media_path(str(value)) in weak_videos)
    ].reset_index(drop=True)
    if target.empty:
        raise RuntimeError("no KPI video embeddings matched the weak gap videos")
    return target


def require_embedding_columns(frame: pd.DataFrame, path: Path) -> None:
    """Require the shared columns used by consolidated embedding parquets."""
    missing = {"filepath", "embedding", "modality"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns: {sorted(missing)}")


def build_iteration_target(
    gaps_jsonl: Path,
    kpi_embeddings_parquet: Path,
    kpi_lookup_parquet: Path,
    modes: list[str],
    output_target_parquet: Path,
) -> int:
    """Write one target parquet containing rows from the selected modalities."""
    frames: list[pd.DataFrame] = []
    for mode in modes:
        if mode == "text":
            frames.append(
                text_target_dataframe(gaps_jsonl, kpi_embeddings_parquet, kpi_lookup_parquet)
            )
        elif mode == "video":
            frames.append(video_target_dataframe(gaps_jsonl, kpi_embeddings_parquet))
        else:
            raise ValueError(f"unsupported target modality: {mode}")
    target = pd.concat(frames, ignore_index=True)
    atomic_write_parquet(target, output_target_parquet)
    return len(target)


def build_nearest_neighbors_spec(
    template: dict[str, Any],
    source_parquet: Path,
    target_parquet: Path,
    output_parquet: Path,
) -> dict[str, Any]:
    """Patch source, target, and output paths into a nearest-neighbor spec template."""
    spec = dict(template)
    spec["source_parquet"] = str(source_parquet)
    spec["target_parquet"] = str(target_parquet)
    spec["output_parquet"] = str(output_parquet)
    return spec


def filter_source_pool(source_parquet: Path, mined_log_parquet: Path, output_filtered_parquet: Path) -> int:
    """Drop source rows whose filepath was already mined in an earlier iteration."""
    source = pd.read_parquet(source_parquet)
    if "filepath" not in source.columns:
        raise ValueError(f"{source_parquet}: missing required column 'filepath'")
    if mined_log_parquet.is_file():
        mined_log = pd.read_parquet(mined_log_parquet)
        if "filepath" not in mined_log.columns:
            raise ValueError(f"{mined_log_parquet}: missing required column 'filepath'")
        excluded = set(mined_log["filepath"].astype(str).tolist())
        filtered = source[~source["filepath"].astype(str).isin(excluded)].reset_index(drop=True)
    else:
        filtered = source.reset_index(drop=True)
    atomic_write_parquet(filtered, output_filtered_parquet)
    return len(filtered)


def prepare_nearest_neighbor_mining(
    workspace: Path,
    workflow_yaml: Path,
    run_dir: Path,
    iteration: int,
    gaps_jsonl: Path,
) -> list[Path]:
    """Prepare one target parquet and nearest-neighbor spec for all requested modalities."""
    config = load_yaml(workflow_yaml)
    mining = require_mapping(config, "mining")
    modality = require_string(mining, "mining.embeddings_modality")
    mine_unique_only = optional_bool(mining, "mining.mine_unique_only", True)
    template_path = existing_absolute_path(
        require_string(mining, "mining.mining_spec_template"),
        workspace,
        "mining.mining_spec_template",
        "file",
    )
    template = load_yaml(template_path)
    path_in_workspace(gaps_jsonl, run_dir, "gaps JSONL")
    mined_log_parquet = run_dir / "mining" / "mined_paths_log.parquet"
    mining_dir = run_dir / f"iter_{iteration}" / "mining"
    source_parquet = run_dir / "embedding_parquets" / "train" / "embeddings.parquet"
    kpi_parquet = run_dir / "embedding_parquets" / "kpi" / "embeddings.parquet"
    target_parquet = mining_dir / "target.parquet"
    output_parquet = mining_dir / "mined_neighbors.parquet"
    spec_path = mining_dir / "nearest_neighbors.yaml"
    for label, path in (("source parquet", source_parquet), ("KPI parquet", kpi_parquet)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    source = pd.read_parquet(source_parquet)
    require_embedding_columns(source, source_parquet)
    source_modes = set(source["modality"].astype(str))
    if source_modes != {"text", "video"}:
        raise ValueError(
            f"{source_parquet}: train source must contain text and video modalities; "
            f"found {sorted(source_modes)}"
        )
    kpi_lookup = run_dir / "cosmos_embed_output" / "kpi" / "lookup.parquet"
    if not kpi_lookup.is_file():
        raise FileNotFoundError(f"KPI lookup parquet does not exist: {kpi_lookup}")
    rows = build_iteration_target(
        gaps_jsonl,
        kpi_parquet,
        kpi_lookup,
        modality_list(modality),
        target_parquet,
    )
    effective_source_parquet = source_parquet
    if mine_unique_only:
        effective_source_parquet = mining_dir / "filtered_source.parquet"
        source_rows = filter_source_pool(source_parquet, mined_log_parquet, effective_source_parquet)
        print(f"wrote {source_rows} source rows after mined-log filtering -> {effective_source_parquet}")
    write_yaml(
        spec_path,
        build_nearest_neighbors_spec(template, effective_source_parquet, target_parquet, output_parquet),
    )
    print(f"wrote {rows} target rows -> {target_parquet}")
    print(f"wrote nearest-neighbor spec -> {spec_path}")
    return [spec_path]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--workflow-yaml", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--gaps-jsonl", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    workspace = absolute_path(args.workspace)
    workflow_yaml = absolute_path(args.workflow_yaml)
    run_dir = absolute_path(args.run_dir)
    gaps_jsonl = absolute_path(args.gaps_jsonl)
    if args.iteration < 1:
        raise ValueError("--iteration must be >= 1")
    if not gaps_jsonl.is_file():
        raise FileNotFoundError(f"gaps JSONL does not exist: {gaps_jsonl}")
    prepare_nearest_neighbor_mining(
        workspace,
        workflow_yaml,
        run_dir,
        args.iteration,
        gaps_jsonl,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
