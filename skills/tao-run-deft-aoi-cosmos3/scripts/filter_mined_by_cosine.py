#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""Filter mined source paths by recomputed maximum cosine similarity to target embeddings."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import tempfile
from typing import Iterable


def _path_key(value: object) -> str:
    return str(value).replace("\\", "/").rstrip("/")


def _vector(value: object, *, context: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{context}: embedding must be a non-empty list")
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: embedding contains a non-numeric value") from exc
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{context}: embedding contains a non-finite value")
    return vector


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = list(left)
    right_values = list(right)
    if len(left_values) != len(right_values):
        raise ValueError(
            f"embedding dimension mismatch: source={len(left_values)}, target={len(right_values)}"
        )
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("zero-norm embedding cannot be compared with cosine similarity")
    similarity = sum(a * b for a, b in zip(left_values, right_values)) / (
        left_norm * right_norm
    )
    return max(-1.0, min(1.0, similarity))


def filter_mined_records(
    *,
    mined_rows: list[dict],
    source_rows: list[dict],
    target_rows: list[dict],
    min_similarity: float,
    filepath_column: str = "filepath",
    source_embedding_column: str = "embedding",
    target_embedding_column: str = "embedding",
    label_column: str = "label",
    filter_by_label: bool = False,
) -> tuple[list[int], list[dict]]:
    if not -1.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between -1 and 1")
    if not mined_rows:
        raise ValueError("mined parquet has no rows")
    if not source_rows or not target_rows:
        raise ValueError("source and target embedding parquets must both be non-empty")

    source_by_path: dict[str, dict] = {}
    for index, row in enumerate(source_rows):
        if filepath_column not in row:
            raise ValueError(f"source row {index}: missing {filepath_column!r}")
        key = _path_key(row[filepath_column])
        if key in source_by_path:
            raise ValueError(f"source embeddings contain duplicate filepath: {key}")
        source_by_path[key] = row

    target_vectors: list[tuple[str, object, list[float]]] = []
    for index, row in enumerate(target_rows):
        if filepath_column not in row or target_embedding_column not in row:
            raise ValueError(
                f"target row {index}: missing {filepath_column!r} or {target_embedding_column!r}"
            )
        if filter_by_label and label_column not in row:
            raise ValueError(f"target row {index}: missing label column {label_column!r}")
        target_vectors.append(
            (
                str(row[filepath_column]),
                row.get(label_column),
                _vector(row[target_embedding_column], context=f"target row {index}"),
            )
        )

    kept_indices: list[int] = []
    audit_rows: list[dict] = []
    for index, mined in enumerate(mined_rows):
        if filepath_column not in mined:
            raise ValueError(f"mined row {index}: missing {filepath_column!r}")
        filepath = str(mined[filepath_column])
        source = source_by_path.get(_path_key(filepath))
        if source is None:
            raise ValueError(f"mined row {index}: filepath not found in source embeddings: {filepath}")
        if source_embedding_column not in source:
            raise ValueError(
                f"source filepath {filepath}: missing embedding column {source_embedding_column!r}"
            )
        if filter_by_label and label_column not in source:
            raise ValueError(f"source filepath {filepath}: missing label column {label_column!r}")
        source_vector = _vector(
            source[source_embedding_column], context=f"source filepath {filepath}"
        )
        source_label = source.get(label_column)
        candidates = [
            (target_path, target_vector)
            for target_path, target_label, target_vector in target_vectors
            if not filter_by_label or target_label == source_label
        ]
        if not candidates:
            raise ValueError(f"source filepath {filepath}: no target embedding with matching label")
        scored = [
            (cosine_similarity(source_vector, target_vector), target_path)
            for target_path, target_vector in candidates
        ]
        best_similarity, best_target = max(scored, key=lambda item: item[0])
        kept = best_similarity >= min_similarity
        if kept:
            kept_indices.append(index)
        audit_rows.append(
            {
                "filepath": filepath,
                "max_cosine_similarity": best_similarity,
                "matched_target_filepath": best_target,
                "kept": kept,
            }
        )
    return kept_indices, audit_rows


def _load_parquet(path: pathlib.Path) -> tuple[object, list[dict]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError(
            "pyarrow is required; install it in the selected CR3 Python environment with "
            "`python -m pip install pyarrow`"
        ) from exc
    table = pq.read_table(path)
    return table, table.to_pylist()


def _write_outputs(
    *,
    mined_table: object,
    kept_indices: list[int],
    audit_rows: list[dict],
    output: pathlib.Path,
    summary_path: pathlib.Path,
    min_similarity: float,
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("pyarrow is required to write filtered parquet output") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    filtered = mined_table.take(pa.array(kept_indices, type=pa.int64()))
    kept_audit = [audit_rows[index] for index in kept_indices]
    filtered = filtered.append_column(
        "max_cosine_similarity",
        pa.array([row["max_cosine_similarity"] for row in kept_audit], type=pa.float64()),
    )
    filtered = filtered.append_column(
        "matched_target_filepath",
        pa.array([row["matched_target_filepath"] for row in kept_audit], type=pa.string()),
    )
    fd, tmp_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    os.close(fd)
    try:
        pq.write_table(filtered, tmp_name)
        os.replace(tmp_name, output)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    similarities = [row["max_cosine_similarity"] for row in audit_rows]
    summary = {
        "metric": "cosine",
        "min_similarity": min_similarity,
        "input_rows": len(audit_rows),
        "kept_rows": len(kept_indices),
        "dropped_rows": len(audit_rows) - len(kept_indices),
        "similarity_min": min(similarities),
        "similarity_max": max(similarities),
        "similarity_mean": sum(similarities) / len(similarities),
        "output": str(output),
        "rows": audit_rows,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mined-parquet", required=True, type=pathlib.Path)
    parser.add_argument("--source-embeddings", required=True, type=pathlib.Path)
    parser.add_argument("--target-embeddings", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--summary", default=None, type=pathlib.Path)
    parser.add_argument("--min-similarity", default=0.9, type=float)
    parser.add_argument("--filepath-column", default="filepath")
    parser.add_argument("--source-embedding-column", default="embedding")
    parser.add_argument("--target-embedding-column", default="embedding")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--filter-by-label", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.output.resolve() == args.mined_parquet.resolve():
            raise ValueError("--output must differ from --mined-parquet so raw mining output is preserved")
        mined_table, mined_rows = _load_parquet(args.mined_parquet)
        _, source_rows = _load_parquet(args.source_embeddings)
        _, target_rows = _load_parquet(args.target_embeddings)
        kept_indices, audit_rows = filter_mined_records(
            mined_rows=mined_rows,
            source_rows=source_rows,
            target_rows=target_rows,
            min_similarity=args.min_similarity,
            filepath_column=args.filepath_column,
            source_embedding_column=args.source_embedding_column,
            target_embedding_column=args.target_embedding_column,
            label_column=args.label_column,
            filter_by_label=args.filter_by_label,
        )
        summary_path = args.summary or args.output.with_name("cosine_filter_summary.json")
        _write_outputs(
            mined_table=mined_table,
            kept_indices=kept_indices,
            audit_rows=audit_rows,
            output=args.output,
            summary_path=summary_path,
            min_similarity=args.min_similarity,
        )
    except (OSError, ValueError) as exc:
        print(f"filter_mined_by_cosine: {exc}", file=sys.stderr)
        return 2
    print(
        f"filter_mined_by_cosine: {len(mined_rows)} -> {len(kept_indices)} rows "
        f"at cosine >= {args.min_similarity}; wrote {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
