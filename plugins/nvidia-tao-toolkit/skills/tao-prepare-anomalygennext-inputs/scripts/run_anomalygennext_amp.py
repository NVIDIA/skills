#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build pair-preserving KNN requests and run AnomalyGenNext AMP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image


BRANCHES = {"fn_mask", "same_type_sampled_mask"}
AMP_HF_REPOS = ("Qwen/Qwen3-VL-8B-Instruct",)


def _hf_hub(root: Path) -> Path:
    return root / "hub" if (root / "hub").is_dir() else root


def _complete_transformers_model(directory: Path) -> bool:
    if not (directory / "config.json").is_file():
        return False
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = directory / index_name
        if not index.is_file():
            continue
        try:
            weight_map = json.loads(index.read_text()).get("weight_map", {})
        except (OSError, json.JSONDecodeError):
            return False
        return bool(weight_map) and all(
            (directory / filename).is_file() for filename in set(weight_map.values())
        )
    return any(directory.glob("*.safetensors")) or any(
        directory.glob("pytorch_model*.bin")
    )


def _configured_model_available(root: Path, hub: Path, model_id: str) -> bool:
    configured = Path(model_id).expanduser()
    local = configured.resolve() if configured.is_absolute() else root / configured
    try:
        local.resolve().relative_to(root)
    except ValueError:
        local = root / "__outside_checkpoint_root__"
    if _complete_transformers_model(local):
        return True
    snapshots = hub / f"models--{model_id.replace('/', '--')}" / "snapshots"
    return snapshots.is_dir() and any(
        child.is_dir() and _complete_transformers_model(child)
        for child in snapshots.iterdir()
    )


def _validate_checkpoint_root(
    root: Path, repo: Path, model_id: str = "nvidia/Cosmos3-Nano"
) -> Path:
    resolved = root.expanduser().resolve()
    image_root = (repo / "checkpoints").resolve()
    if resolved != image_root:
        raise ValueError(
            f"checkpoint root must be mounted at {repo}/checkpoints: {resolved}"
        )
    hf_home = resolved / "hf"
    hub = _hf_hub(hf_home)
    missing = []
    if not model_id.strip():
        raise ValueError("AMP model_id must be nonempty")
    for name in AMP_HF_REPOS:
        directory = hub / f"models--{name.replace('/', '--')}"
        if not (directory / "blobs").is_dir() or not (directory / "snapshots").is_dir():
            missing.append(name)
    if not _configured_model_available(resolved, hub, model_id):
        missing.append(
            f"{model_id} (complete checkpoints/{model_id} or Hugging Face snapshot)"
        )
    if missing:
        raise FileNotFoundError(
            "checkpoint root lacks required AMP model assets: "
            + ", ".join(missing)
        )
    sam2 = (
        resolved / "facebook" / "sam2.1-hiera-large" / "sam2.1_hiera_large.pt"
    )
    if not sam2.is_file():
        raise FileNotFoundError(f"checkpoint root lacks SAM2.1 checkpoint: {sam2}")
    return hf_home


def _matrix(values: pd.Series) -> np.ndarray:
    rows = [np.asarray(value, dtype=np.float32).reshape(-1) for value in values]
    if not rows or len({row.size for row in rows}) != 1:
        raise ValueError("embeddings are empty or have inconsistent widths")
    matrix = np.stack(rows)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or np.any(norms == 0):
        raise ValueError("embeddings contain non-finite or zero-norm rows")
    return matrix / norms


def _pair_id(fn_id: str, clean: str) -> str:
    return "pair-" + hashlib.sha256(f"{fn_id}\0{clean}".encode()).hexdigest()[:16]


def _rank_clean_images(
    clean: pd.DataFrame,
    clean_vectors: np.ndarray,
    query: pd.Series,
    query_vector: np.ndarray,
    topn: int,
) -> list[tuple[str, float]]:
    eligible = clean.pool_key.astype(str) == str(query.pool_key)
    if "anomaly_type_eligibility" in clean.columns:
        eligible &= (
            clean.anomaly_type_eligibility.astype(str) == str(query.anomaly_type)
        )
    pool = clean.index[eligible].to_numpy()
    if not len(pool):
        raise ValueError(
            "no clean embeddings for "
            f"pool_key={query.pool_key}, anomaly_type={query.anomaly_type}"
        )

    scores = clean_vectors[pool] @ query_vector
    best_by_filepath: dict[str, float] = {}
    for index, score in zip(pool, scores, strict=True):
        filepath = str(clean.loc[int(index), "filepath"])
        value = float(score)
        if filepath not in best_by_filepath or value > best_by_filepath[filepath]:
            best_by_filepath[filepath] = value
    return sorted(best_by_filepath.items(), key=lambda item: (-item[1], item[0]))[:topn]


def _validate_amp_mask(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"AMP mask is missing: {path}")
    with Image.open(path) as image:
        mask = np.asarray(image)
        canvas = image.size
    if mask.ndim != 2 or set(np.unique(mask).tolist()) != {0, 255}:
        raise ValueError(f"AMP mask must contain exactly binary values 0 and 255: {path}")
    foreground = mask == 255
    count = int(np.count_nonzero(foreground))
    if count == 0 or count == foreground.size:
        raise ValueError(f"AMP mask foreground must be nonempty and non-full: {path}")
    ys, xs = np.nonzero(foreground)
    tight = (int(xs.max()) - int(xs.min()) + 1, int(ys.max()) - int(ys.min()) + 1)
    if tight == canvas:
        raise ValueError(f"AMP mask tight extent fills the canvas: {path}")


def plan(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    manifests, embeddings = root / "manifests", root / "embeddings"
    clean = pd.read_parquet(embeddings / "clean_embeddings.parquet").reset_index(drop=True)
    embedded = pd.read_parquet(embeddings / "fn_embeddings.parquet")
    queries = pd.read_parquet(manifests / "selected_fn_queries.parquet")
    masks = pd.read_parquet(manifests / "mask_selection.parquet")
    for mask_path in sorted(set(masks.mask_path.astype(str))):
        _validate_amp_mask(Path(mask_path))
    for label, frame in (("clean", clean), ("FN", embedded)):
        if frame.empty or not {"filepath", "embedding"}.issubset(frame.columns):
            raise ValueError(f"{label} embeddings lack filepath/embedding")
    if embedded.filepath.astype(str).duplicated().any():
        raise ValueError("FN embedding rows must be unique by filepath")
    queries = queries.merge(embedded[["filepath", "embedding"]], on="filepath",
                            how="left", validate="many_to_one")
    if queries.embedding.isna().any():
        raise ValueError("one or more FN queries have no embedding")
    clean_vectors, query_vectors = _matrix(clean.embedding), _matrix(queries.embedding)
    retrieval = config["retrieval"]
    if retrieval.get("metric", "cosine") != "cosine":
        raise ValueError("only cosine retrieval is supported")
    topn, floor = int(retrieval["candidate_topn"]), float(retrieval["min_similarity"])
    excluded: set[str] = set()
    exclusion = str(retrieval.get("prior_clean_exclusion_manifest") or "")
    if exclusion:
        excluded = set(pd.read_parquet(exclusion).filepath.astype(str))

    candidates, requests = [], []
    for position, query in queries.reset_index(drop=True).iterrows():
        ranked = _rank_clean_images(
            clean, clean_vectors, query, query_vectors[position], topn
        )
        mask_rows = masks[masks.fn_id == query.fn_id]
        if set(mask_rows.branch.astype(str)) != BRANCHES or len(mask_rows) != 2:
            raise ValueError(f"FN {query.fn_id} needs exactly two mask branches")
        for rank, (clean_path, score) in enumerate(ranked, start=1):
            reason = "below_similarity_floor" if score < floor else (
                "prior_iteration_exclusion" if clean_path in excluded else ""
            )
            pair = _pair_id(str(query.fn_id), clean_path)
            candidate = {
                "candidate_id": pair, "fn_id": str(query.fn_id),
                "query_order": int(query.query_order), "dataset_id": str(query.dataset_id),
                "anomaly_type": str(query.anomaly_type), "od_category": str(query.od_category),
                "pool_key": str(query.pool_key), "fn_filepath": str(query.filepath),
                "clean_filepath": clean_path, "neighbor_rank": rank,
                "cosine_similarity": score, "eligible_for_amp": not reason,
                "gate_reason": reason,
            }
            candidates.append(candidate)
            if not reason:
                for mask in mask_rows.sort_values("branch").itertuples():
                    requests.append({
                        "clean_image": clean_path, "defect_type": str(query.anomaly_type),
                        "submask": str(mask.mask_path), "name": f"{pair}__{mask.branch}",
                        "cad_mask": None, "cad_mask_label": None, "n_seeds": 1,
                        "submask_split_largest": False,
                    })
    if not requests:
        raise ValueError("no candidates passed the AMP gate")
    pd.DataFrame(candidates).to_parquet(manifests / "knn_candidates.parquet", index=False)
    amp = root / "amp"
    amp.mkdir(parents=True, exist_ok=True)
    (amp / "amp_samples.json").write_text(json.dumps(requests, indent=2) + "\n")
    return {"candidates": len(candidates), "amp_rows": len(requests),
            "embedding_dim": int(clean_vectors.shape[1])}


def _publish_paths(amp_dir: Path, runtime_root: Path, published_root: Path) -> None:
    source, destination = str(runtime_root.resolve()), str(published_root.resolve())
    if source == destination:
        return
    for path in sorted(amp_dir.rglob("*.json")) + sorted(amp_dir.rglob("*.jsonl")):
        value = path.read_text(encoding="utf-8")
        if source in value:
            path.write_text(value.replace(source, destination), encoding="utf-8")


def run(root: Path, checkpoint_root: Path, pool_dataset_root: Path,
        published_root: Path | None = None,
        repo: Path = Path("/workspace/paidf-anomalygen")) -> dict[str, Any]:
    frozen = root / "prepared_anomalygennext_inputs" / "filtering_config.yaml"
    config = yaml.safe_load(frozen.read_text())
    pool = pool_dataset_root.expanduser().resolve()
    expected_pool = Path(config["pool_dataset_root"]).expanduser().resolve()
    if not pool.is_dir() or expected_pool != pool:
        raise ValueError(
            "pool must be remounted at the compute path frozen during preparation: "
            f"expected {expected_pool}, received {pool}"
        )
    report = plan(root, config)
    amp = config.get("amp") or {}
    model_id = str(amp.get("model_id", "nvidia/Cosmos3-Nano"))
    hf_home = _validate_checkpoint_root(checkpoint_root, repo, model_id)
    command = [sys.executable, "-m", "anomalygen.scripts.auto_mask_placement.roi_place",
               "--input_pair_path", str(root / "amp" / "amp_samples.json"),
               "--defect_desc", str(Path(config["defect_spec"]).resolve()),
               "--output_dir", str(root / "amp"), "--n_seeds", "1",
               "--seed", str(int(amp.get("seed", 43))),
               "--model_id", model_id]
    env = os.environ.copy()
    env.update(HF_HOME=str(hf_home), HF_HUB_CACHE=str(_hf_hub(hf_home)),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    subprocess.run(command, check=True, stdout=sys.stderr, env=env)
    published_root = (published_root or root).resolve()
    _publish_paths(root / "amp", root, published_root)
    testcase = root / "amp" / "testcase.jsonl"
    if not testcase.is_file() or not testcase.read_text().strip():
        raise ValueError("AMP produced no testcase rows")
    report["testcase"] = str(published_root / "amp" / "testcase.jsonl")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--published-root", type=Path)
    parser.add_argument("--pool-dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("/workspace/paidf-anomalygen"))
    args = parser.parse_args()
    result = run(Path(args.prepared_root).resolve(), args.checkpoint_root.resolve(),
                 args.pool_dataset_root.resolve(), args.published_root,
                 args.repo.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
