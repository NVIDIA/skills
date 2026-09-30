# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact platform-neutral TAO argv contracts for the PAS DEFT workflow."""

from __future__ import annotations

import hashlib
import json
import re
import pathlib
from typing import Any

from pas_deft.pas_artifacts import PAS_METRICS_AGGREGATE_FILENAME


_MODEL_COMMANDS = {
    "pool_embed",
    "target_embed",
    "viz_weak_embed",
    "viz_mined_embed",
    "viz_previous_embed",
    "train",
    "evaluate",
}


def command_sha256(command: list[str]) -> str:
    """Return a stable digest for an argv vector (without shell parsing)."""
    encoded = json.dumps(
        command, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: pathlib.Path) -> str:
    """Hash one regular, non-symlink file without trusting path aliases."""
    absolute = pathlib.Path(path).expanduser()
    if (
        not absolute.is_absolute()
        or not absolute.is_file()
        or absolute.is_symlink()
        or absolute.resolve() != absolute
    ):
        raise ValueError(f"content-bound input is missing or unsafe: {absolute}")
    digest = hashlib.sha256()
    with absolute.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fresh_output_sha256(paths: list[str]) -> dict[str, str]:
    """Bind the byte content of every regular file produced by an adapter."""
    hashes: dict[str, str] = {}
    for raw in paths:
        path = pathlib.Path(raw)
        if path.is_file() and not path.is_symlink():
            hashes[str(path)] = file_sha256(path)
    return hashes


def validate_content_bound_outputs(
    payload: dict[str, Any], required_outputs: list[pathlib.Path], evidence_name: str
) -> dict[str, str]:
    """Verify that required outputs still have their producer-recorded bytes."""
    hashes = payload.get("fresh_output_sha256")
    if not isinstance(hashes, dict):
        raise ValueError(f"{evidence_name}.fresh_output_sha256 must be an object")
    verified: dict[str, str] = {}
    for path in required_outputs:
        absolute = pathlib.Path(path).expanduser()
        key = str(absolute)
        expected = hashes.get(key)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(
                f"{evidence_name} does not bind a SHA256 digest for {absolute}"
            )
        actual = file_sha256(absolute)
        if actual != expected:
            raise ValueError(
                f"content hash mismatch for {absolute}: expected {expected}, got {actual}"
            )
        verified[key] = actual
    return verified


def derived_spec_evidence(
    name: str, label: str, results_dir: pathlib.Path
) -> tuple[pathlib.Path, pathlib.Path, str] | None:
    """Return the exact generated spec and producer status consumed by an action."""
    if name == "evaluate":
        phase = "zs" if label == "baseline" else f"iter_{_iteration_number(label)}"
        spec_name = "eval_config.yaml"
        producer = "eval-config"
    elif name == "train":
        phase = f"iter_{_iteration_number(label)}"
        spec_name = "train_config.yaml"
        producer = "train-config"
    else:
        return None
    spec = results_dir / phase / "specs" / spec_name
    status = results_dir / phase / "specs" / f"{producer}.host.status.json"
    return spec, status, producer


def _iteration_number(label: str) -> int:
    match = re.fullmatch(r"iter([1-9][0-9]*)", label)
    if not match:
        raise ValueError(f"container command requires iterN, got {label!r}")
    return int(match.group(1))


def expected_container_command(
    name: str, label: str, config: dict[str, Any]
) -> list[str]:
    """Build the one allowed argv vector for a named workflow launch."""
    if name == "pool_embed":
        if label != "baseline":
            raise ValueError("pool_embed is valid only for baseline")
        return [
            "embedding",
            "text_embeddings",
            "-e",
            "/specs/text_embed_spec.yaml",
            "input_parquet=/results/embeddings/source/source_pool.parquet",
            "output_parquet=/results/embeddings/source/embeddings.parquet",
        ]

    if name == "evaluate":
        phase = "zs" if label == "baseline" else f"iter_{_iteration_number(label)}"
        return ["clip", "evaluate", "-e", f"/results/{phase}/specs/eval_config.yaml"]

    number = _iteration_number(label)
    if name == "target_embed":
        return [
            "embedding",
            "text_embeddings",
            "-e",
            "/specs/text_embed_spec.yaml",
            f"input_parquet=/results/iter_{number}/gaps/kpi_gaps.parquet",
            f"output_parquet=/results/iter_{number}/embeddings/target/embeddings.parquet",
        ]
    if name == "knn":
        return [
            "tmm",
            "nearest_neighbors",
            "-e",
            "/specs/mining_spec.yaml",
            "source_parquet=/results/embeddings/source/embeddings.parquet",
            f"target_parquet=/results/iter_{number}/embeddings/target/embeddings.parquet",
            f"output_parquet=/results/iter_{number}/mining/mined_samples.parquet",
            f"topn={config.get('mining_topn')}",
            f"knn_metric={config.get('knn_metric')}",
        ]
    if name == "train":
        return ["clip", "train", "-e", f"/results/iter_{number}/specs/train_config.yaml"]
    if name == "viz_weak_embed":
        return [
            "embedding",
            "image_embeddings",
            "-e",
            "/specs/image_embed_spec.yaml",
            f"input_parquet=/results/iter_{number}/embeddings/viz_weak/input.parquet",
            f"output_parquet=/results/iter_{number}/embeddings/viz_weak/embeddings.parquet",
        ]
    if name == "viz_mined_embed":
        return [
            "embedding",
            "image_embeddings",
            "-e",
            "/specs/image_embed_spec.yaml",
            f"input_parquet=/results/iter_{number}/mining/mined_unique_images.parquet",
            f"output_parquet=/results/iter_{number}/embeddings/augmented/mined_embeddings.parquet",
        ]
    if name == "viz_previous_embed":
        return [
            "embedding",
            "image_embeddings",
            "-e",
            "/specs/image_embed_spec.yaml",
            f"input_parquet=/results/iter_{number}/embeddings/previous/prev_pool.parquet",
            f"output_parquet=/results/iter_{number}/embeddings/previous/embeddings.parquet",
        ]
    raise ValueError(f"unsupported PAS DEFT container command name: {name!r}")


def expected_image_kind(name: str) -> str:
    if name in {"train", "evaluate"}:
        return "pyt"
    if name in {
        "pool_embed",
        "target_embed",
        "knn",
        "viz_weak_embed",
        "viz_mined_embed",
        "viz_previous_embed",
    }:
        return "ds"
    raise ValueError(f"unsupported PAS DEFT container command name: {name!r}")


def expected_hf_forwarding(name: str, config: dict[str, Any]) -> bool:
    """Only model-loading stages receive the approved token environment."""
    return bool(config.get("requires_hf_token")) and name in _MODEL_COMMANDS


def expected_stage_directory(
    name: str, label: str, results_dir: pathlib.Path
) -> pathlib.Path:
    """Return the exact host directory that owns a container status."""
    if name == "pool_embed":
        if label != "baseline":
            raise ValueError("pool_embed is valid only for baseline")
        return results_dir / "embeddings" / "source"
    phase = (
        results_dir / "zs"
        if label == "baseline"
        else results_dir / f"iter_{_iteration_number(label)}"
    )
    if name == "evaluate":
        return phase / "evaluate"
    _iteration_number(label)
    suffixes = {
        "target_embed": pathlib.Path("embeddings/target"),
        "knn": pathlib.Path("mining"),
        "viz_weak_embed": pathlib.Path("embeddings/viz_weak"),
        "viz_mined_embed": pathlib.Path("embeddings/augmented"),
        "viz_previous_embed": pathlib.Path("embeddings/previous"),
        "train": pathlib.Path("train"),
    }
    suffix = suffixes.get(name)
    if suffix is None:
        raise ValueError(f"unsupported PAS DEFT container command name: {name!r}")
    return phase / suffix


def expected_fresh_outputs(
    name: str, label: str, results_dir: pathlib.Path
) -> list[pathlib.Path]:
    """Return the exact files whose recreation proves a container stage."""
    if name == "pool_embed":
        return [results_dir / "embeddings" / "source" / "embeddings.parquet"]
    phase = (
        results_dir / "zs"
        if label == "baseline"
        else results_dir / f"iter_{_iteration_number(label)}"
    )
    if name == "evaluate":
        return [
            phase / "evaluate" / PAS_METRICS_AGGREGATE_FILENAME,
            phase / "evaluate" / "status.json",
        ]
    number = _iteration_number(label)
    fixed = {
        "target_embed": phase / "embeddings" / "target" / "embeddings.parquet",
        "knn": phase / "mining" / "mined_samples.parquet",
        "viz_weak_embed": phase / "embeddings" / "viz_weak" / "embeddings.parquet",
        "viz_mined_embed": phase / "embeddings" / "augmented" / "mined_embeddings.parquet",
        "viz_previous_embed": phase / "embeddings" / "previous" / "embeddings.parquet",
        "train": phase / "train" / "status.json",
    }
    if name not in fixed:
        raise ValueError(
            f"unsupported PAS DEFT container command name for iter{number}: {name!r}"
        )
    return [fixed[name]]
