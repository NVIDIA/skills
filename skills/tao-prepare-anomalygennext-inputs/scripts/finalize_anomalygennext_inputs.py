#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze successful AMP pairs into generation-ready AnomalyGenNext inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image


BRANCHES = {"fn_mask", "same_type_sampled_mask"}


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _amp_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        mask = Path(row["mask_filename"])
        key = (str(row["image_filename"]), mask.name.split("__seed", 1)[0])
        if key in result:
            raise ValueError(f"duplicate AMP output key: {key}")
        result[key] = row
    return result


def _validate_mask(mask_path: Path, image_path: Path) -> str:
    with Image.open(mask_path) as mask_image, Image.open(image_path) as image:
        mask = np.asarray(mask_image.convert("L")) > 0
        if mask.shape != (image.height, image.width):
            raise ValueError("dimension_mismatch")
    pixels = int(mask.sum())
    if pixels == 0:
        raise ValueError("empty")
    if pixels == mask.size:
        raise ValueError("full_image")
    return _sha256(mask_path)


def finalize(root: Path) -> dict[str, Any]:
    prepared = root / "prepared_anomalygennext_inputs"
    config = yaml.safe_load((prepared / "filtering_config.yaml").read_text())
    candidates = pd.read_parquet(root / "manifests" / "knn_candidates.parquet")
    masks = pd.read_parquet(root / "manifests" / "mask_selection.parquet")
    amp = _amp_index(root / "amp" / "testcase.jsonl")
    keep = int(config["retrieval"]["max_neighbors_per_fn"])
    status_rows, selected_rows = [], []
    generator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    provenance: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for fn_id in candidates.sort_values("query_order").fn_id.drop_duplicates():
        retained, used_clean = 0, set()
        group = candidates[candidates.fn_id == fn_id].sort_values("neighbor_rank")
        branch_rows = masks[masks.fn_id == fn_id]
        if len(branch_rows) != 2 or set(branch_rows.branch.astype(str)) != BRANCHES:
            raise ValueError(f"FN {fn_id} needs exactly two mask branches")
        branch_rows = branch_rows.set_index("branch")
        for candidate in group.to_dict("records"):
            status = {**candidate, "selected": False,
                      "selection_reason": str(candidate["gate_reason"])}
            clean = str(candidate["clean_filepath"])
            if retained >= keep:
                status["selection_reason"] = "quota_reached"
            elif not bool(candidate["eligible_for_amp"]):
                pass
            elif clean in used_clean:
                status["selection_reason"] = "duplicate_clean_within_fn"
            else:
                placed = {}
                for branch in sorted(BRANCHES):
                    source = Path(str(branch_rows.loc[branch, "mask_path"]))
                    row = amp.get((clean, source.stem))
                    if row is None:
                        status["selection_reason"] = f"amp_failure:{branch}"
                        break
                    try:
                        _validate_mask(Path(row["mask_filename"]), Path(clean))
                    except (FileNotFoundError, ValueError) as exc:
                        reason = exc.args[0] if isinstance(exc, ValueError) else "missing"
                        status["selection_reason"] = f"invalid_aligned_mask:{branch}:{reason}"
                        break
                    placed[branch] = row
                if len(placed) == 2:
                    pair = str(candidate["candidate_id"])
                    dataset = str(candidate["dataset_id"])
                    selected = {**candidate, "pair_id": pair}
                    for branch, row in placed.items():
                        source = Path(str(branch_rows.loc[branch, "mask_path"]))
                        aligned = prepared / "aligned_masks" / dataset / f"{pair}__{branch}.png"
                        aligned.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(row["mask_filename"], aligned)
                        frozen = {**row, "mask_filename": str(aligned),
                                  "anomaly_type": str(candidate["anomaly_type"]),
                                  "num_generated_images": 1}
                        index = len(generator[dataset])
                        generator[dataset].append(frozen)
                        provenance[dataset].append({
                            "generation_index": index, "pair_id": pair, "fn_id": str(fn_id),
                            "dataset_id": dataset, "anomaly_type": str(candidate["anomaly_type"]),
                            "od_category": str(candidate["od_category"]), "mask_branch": branch,
                            "source_mask": str(source), "source_mask_sha256": _sha256(source),
                            "fn_filepath": str(candidate["fn_filepath"]), "clean_filepath": clean,
                            "aligned_mask": str(aligned), "aligned_mask_sha256": _sha256(aligned),
                            "neighbor_rank": int(candidate["neighbor_rank"]),
                            "cosine_similarity": float(candidate["cosine_similarity"]),
                            "source_tag": config.get("source_tag", "user_provided"),
                        })
                        selected[f"{branch}_aligned_mask"] = str(aligned)
                    selected_rows.append(selected)
                    used_clean.add(clean)
                    retained += 1
                    status.update(selected=True, selection_reason="selected")
            status_rows.append(status)

    status = pd.DataFrame(status_rows)
    selected = pd.DataFrame(selected_rows)
    status.to_parquet(root / "manifests" / "knn_roi_status.parquet", index=False)
    selected.to_parquet(root / "manifests" / "selected_pairs.parquet", index=False)
    plans, unified, artifacts = [], [], []
    for dataset in sorted(generator):
        directory = prepared / "anomalygen_inputs" / dataset
        testcase, provenance_path = directory / "testcase.jsonl", directory / "provenance.jsonl"
        _jsonl(testcase, generator[dataset])
        _jsonl(provenance_path, provenance[dataset])
        types = sorted({row["anomaly_type"] for row in provenance[dataset]})
        mapping = config["datasets"][dataset]
        plans.append({"dataset_id": dataset, "anomaly_types": types,
                      "testcase": str(testcase), "provenance": str(provenance_path),
                      "checkpoint": str(mapping["checkpoint"]), "recipe": str(mapping["recipe"]),
                      "real_root": str(config["pool_dataset_root"]),
                      "requested_rows": len(generator[dataset])})
        unified.extend({**proof, "checkpoint": str(mapping["checkpoint"]),
                        "recipe": str(mapping["recipe"]), "generator_input": request}
                       for proof, request in zip(provenance[dataset], generator[dataset], strict=True))
        artifacts.extend([testcase, provenance_path])
    if not unified:
        raise RuntimeError("prepared inputs contain no AnomalyGenNext generation rows")
    unified_path = prepared / "anomalygen_inputs.jsonl"
    plan_path = prepared / "anomalygen_next_generation_plan.json"
    _jsonl(unified_path, unified)
    _json(plan_path, plans)
    artifacts.extend([unified_path, plan_path, prepared / "filtering_config.yaml",
                      prepared / "input_contract.json"])
    artifacts.extend(sorted((prepared / "aligned_masks").rglob("*.png")))
    manifest = {
        "schema_version": 2, "phase": "prepared_anomalygennext_inputs", "status": "COMPLETE",
        "source_tag": config.get("source_tag", "user_provided"),
        "selected_fn_count": int(selected.fn_id.nunique()), "selected_pair_count": len(selected),
        "generator_row_count": len(unified), "generator_groups": plans,
        "artifacts": [{"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
                      for path in artifacts],
        "skip_counts": dict(Counter(status.selection_reason)), "generation_ready": True,
        "training_pool_mutated": False,
    }
    _json(prepared / "prepared_inputs_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True)
    args = parser.parse_args()
    result = finalize(Path(args.prepared_root).resolve())
    print(json.dumps({key: result[key] for key in
                      ("selected_fn_count", "selected_pair_count", "generator_row_count")},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
