#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consolidate Cosmos Embed modality outputs into one mining parquet."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from workflow_common import MODALITY_CHOICES, atomic_write_parquet, load_yaml, modality_list


def load_metadata(path: Path) -> dict[str, Any]:
    """Read one Cosmos Embed embeddings metadata JSON object."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def npy_path(inference_dir: Path, metadata: dict[str, Any]) -> Path:
    """Resolve the NPY embedding array referenced by Cosmos Embed metadata."""
    value = metadata.get("npy_file")
    if not isinstance(value, str) or not value:
        raise ValueError("Cosmos Embed metadata is missing non-empty 'npy_file'")
    path = Path(value)
    if not path.is_absolute():
        path = inference_dir / path
    return path


def text_filepath_queues(output_dir: Path) -> dict[str, deque[str]] | None:
    """Group staged question-file paths by embedded text, preserving occurrence order."""
    lookup_path = output_dir / "lookup.parquet"
    if not lookup_path.exists():
        return None
    lookup = pd.read_parquet(lookup_path)
    missing = {"filepath", "question"} - set(lookup.columns)
    if missing:
        raise ValueError(f"{lookup_path}: missing required columns: {sorted(missing)}")

    queues: dict[str, deque[str]] = defaultdict(deque)
    for index, row in lookup.iterrows():
        question = row["question"]
        filepath = row["filepath"]
        if not isinstance(question, str) or not question:
            raise ValueError(f"{lookup_path}: row {index} has invalid question: {question!r}")
        if not isinstance(filepath, str) or not filepath:
            raise ValueError(f"{lookup_path}: row {index} has invalid filepath: {filepath!r}")
        queues[question].append(filepath)
    return dict(queues)


def inference_dir_from_spec(output_dir: Path, mode: str) -> Path:
    """Return the inference output directory declared by a generated spec."""
    spec_path = output_dir / "specs" / f"inference_{mode}.yaml"
    if not spec_path.is_file():
        raise FileNotFoundError(f"Cosmos Embed inference spec is missing: {spec_path}")
    spec = load_yaml(spec_path)
    results_dir = spec.get("results_dir")
    if not isinstance(results_dir, str) or not results_dir:
        raise ValueError(f"{spec_path}: results_dir must be a non-empty string")
    results_path = Path(results_dir).expanduser()
    if not results_path.is_absolute():
        raise ValueError(f"{spec_path}: results_dir must be absolute: {results_path}")
    return results_path / "inference"


def raw_embeddings_dataframe(mode: str, output_dir: Path) -> pd.DataFrame:
    """Convert one modality's Cosmos Embed JSON/NPY output to a dataframe."""
    inference_dir = inference_dir_from_spec(output_dir, mode)
    metadata_path = inference_dir / f"{mode}_embeddings.json"
    metadata = load_metadata(metadata_path)
    embeddings = np.load(npy_path(inference_dir, metadata))
    results = metadata.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{metadata_path}: expected list field 'results'")

    text_queues = text_filepath_queues(output_dir) if mode == "text" else None
    id_key = "video_path" if mode == "video" else "text"
    filepaths: list[str] = []
    rows: list[list[float]] = []
    for index, record in enumerate(results, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{metadata_path}: result {index} is not an object")
        npy_row = record.get("npy_row")
        if not isinstance(npy_row, int):
            raise ValueError(f"{metadata_path}: result {index} missing integer 'npy_row'")
        if npy_row < 0 or npy_row >= len(embeddings):
            raise ValueError(
                f"{metadata_path}: result {index} npy_row {npy_row} exceeds "
                f"embedding rows 0..{len(embeddings) - 1}"
            )
        if mode == "text" and text_queues is not None:
            text = record.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError(f"{metadata_path}: result {index} missing non-empty 'text'")
            queue = text_queues.get(text)
            if not queue:
                raise ValueError(
                    f"{metadata_path}: result {index} text has no unconsumed lookup row: {text!r}"
                )
            identifier = queue.popleft()
        else:
            identifier = record.get(id_key)
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(f"{metadata_path}: result {index} missing non-empty {id_key!r}")
        filepaths.append(identifier)
        rows.append(embeddings[npy_row].tolist())

    if text_queues is not None:
        unmatched_count = sum(len(queue) for queue in text_queues.values())
        if unmatched_count:
            examples = [text for text, queue in text_queues.items() if queue][:3]
            raise ValueError(
                f"{metadata_path}: {unmatched_count} text lookup rows were not returned by "
                f"Cosmos Embed; example questions: {examples!r}"
            )

    print(f"{mode}: loaded {len(filepaths)} rows x {embeddings.shape[1]} dims")
    return pd.DataFrame({"filepath": filepaths, "embedding": rows, "modality": mode})


def consolidate_embeddings(
    output_dir: Path,
    parquet_dir: Path,
    embedding_modality: str,
) -> Path:
    """Gather required modality outputs into one mining-ready parquet."""
    frames: list[pd.DataFrame] = []
    for mode in modality_list(embedding_modality):
        frames.append(raw_embeddings_dataframe(mode, output_dir))
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise RuntimeError("no Cosmos Embed rows were available to consolidate")
    output_path = parquet_dir / "embeddings.parquet"
    atomic_write_parquet(combined, output_path)
    counts = combined.groupby("modality").size().to_dict()
    print(f"wrote {len(combined)} consolidated rows {counts} -> {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--parquet-dir", required=True, type=Path)
    parser.add_argument("--embedding-modality", required=True, choices=MODALITY_CHOICES)
    return parser.parse_args()


def main() -> None:
    """Consolidate selected modality outputs into one parquet."""
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    parquet_dir = args.parquet_dir.expanduser().resolve()
    consolidate_embeddings(output_dir, parquet_dir, args.embedding_modality)


if __name__ == "__main__":
    main()
