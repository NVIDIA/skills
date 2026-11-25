#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize DEFT AOI false negatives into AnomalyGenNext preparation inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deft_od_aoi_synthesis_contract import (
    resolve_candidate_topn,
    validate_synthesis_contract,
)


FIELDS = ("dataset_id", "texture_id", "defect_class", "fn_mask_source")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_contract(
    synthesis: dict[str, Any], iteration: int | None
) -> tuple[str, dict[str, int] | None, dict[str, Any] | None]:
    mode, _ = validate_synthesis_contract(synthesis)
    if mode == "all_eligible":
        return mode, None, None
    if mode == "generated_per_type_plan":
        if iteration is None or iteration < 1:
            raise ValueError("generated_per_type_plan requires a positive --iteration")
        return mode, None, None
    raise AssertionError(f"unvalidated synthesis.fn_selection.mode: {mode}")


def _generated_plan(
    synthesis: dict[str, Any], rows: list[dict[str, Any]], real_coco: Path,
    output: Path, iteration: int,
) -> tuple[dict[str, int] | None, dict[str, Any] | None, dict[str, Any]]:
    if not real_coco.is_file():
        raise ValueError(f"generated_per_type_plan needs an admitted real COCO: {real_coco}")
    document = json.loads(real_coco.read_text())
    by_kind = Counter(str(row.get("deft_kind") or "") for row in document.get("images", []))
    real_count = by_kind["real_defect"]
    prior_synthetic = by_kind["synthetic_defect"]
    _, images_per_fn = validate_synthesis_contract(synthesis)
    fraction = float(synthesis["cumulative_fraction_of_total_defects"])
    cumulative_limit = int(fraction / (1.0 - fraction) * real_count)
    image_budget = max(0, cumulative_limit - prior_synthetic)
    fn_budget = image_budget // images_per_fn
    available = Counter(str(row["anomaly_type"]) for row in rows)
    selected_total = min(fn_budget, sum(available.values()))
    allocation = {name: 0 for name in available}
    if selected_total:
        total_available = sum(available.values())
        exact = {
            name: selected_total * count / total_available
            for name, count in available.items()
        }
        allocation = {name: min(available[name], math.floor(value))
                      for name, value in exact.items()}
        remaining = selected_total - sum(allocation.values())
        for name in sorted(available, key=lambda item: (-(exact[item] % 1), item)):
            if not remaining:
                break
            if allocation[name] < available[name]:
                allocation[name] += 1
                remaining -= 1
    plan = {name: allocation[name] * images_per_fn for name in sorted(allocation)}
    plan = {name: count for name, count in plan.items() if count}
    evidence = {
        "basis": "fraction_of_total", "fraction": fraction,
        "cumulative_real_images": real_count,
        "prior_synthetic_images": prior_synthetic,
        "cumulative_synthetic_limit": cumulative_limit,
        "new_image_budget": image_budget,
        "images_per_fn": images_per_fn,
        "eligible_fn_count": sum(available.values()),
        "selected_fn_count": sum(allocation.values()),
        "planned_images": sum(plan.values()),
        "unplanned_budget": image_budget - sum(plan.values()),
    }
    if not plan:
        return None, None, evidence
    path = output / "synthetic_plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    contract = {
        "iteration": iteration, "path": str(path.resolve()), "sha256": _sha256(path),
        "counts": plan, "images_per_fn": images_per_fn, "generated": True,
        "budget": evidence,
    }
    return plan, contract, evidence


def _source(images: Path, row: dict[str, Any]) -> Path:
    path = Path(str(row.get("source_path") or "")) if row.get("source_path") else (
        images / str(row["file_name"])
    )
    return path.resolve()


def _image_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index canonical COCO ids and filename-stem ids emitted by gap analysis."""
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        aliases = {str(row.get("id"))}
        file_name = str(row.get("file_name") or "").strip()
        if file_name:
            aliases.add(Path(file_name).stem)
        for alias in aliases:
            if not alias or alias == "None":
                raise ValueError("KPI COCO image lacks an id")
            existing = output.get(alias)
            if existing is not None and existing is not row:
                raise ValueError(f"duplicate KPI image identity: {alias}")
            output[alias] = row
    return output


def _xyxy(value: Any) -> tuple[float, float, float, float]:
    values = tuple(map(float, value))
    if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError(f"invalid xyxy gap box: {value}")
    return values


def _iou(left: tuple[float, ...], annotation: dict[str, Any]) -> float:
    x, y, width, height = map(float, annotation["bbox"])
    right = (x, y, x + width, y + height)
    ix1, iy1, ix2, iy2 = max(left[0], right[0]), max(left[1], right[1]), min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (left[2] - left[0]) * (left[3] - left[1]) + width * height - intersection
    return intersection / union if union else 0.0


def prepare(
    policy_path: Path, strict_gaps: Path, output: Path, iteration: int | None = None,
    real_coco: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    policy = yaml.safe_load(policy_path.read_text())
    synthesis = policy["synthesis"]
    if not synthesis.get("enabled"):
        raise ValueError("synthesis is disabled in the frozen policy")
    retrieval = policy.get("retrieval") or {}
    candidate_topn_source = (
        "synthesis.candidate_topn"
        if "candidate_topn" in synthesis
        else "retrieval.candidate_overfetch"
    )
    try:
        candidate_topn = resolve_candidate_topn(
            synthesis, retrieval.get("candidate_overfetch")
        )
    except ValueError as error:
        if candidate_topn_source == "retrieval.candidate_overfetch":
            raise ValueError(
                "retrieval.candidate_overfetch fallback for missing "
                f"synthesis.candidate_topn is invalid; {error}"
            ) from error
        raise
    candidate_depth = {
        "candidate_topn": candidate_topn,
        "candidate_topn_source": candidate_topn_source,
    }
    selection_mode, plan, plan_contract = _selection_contract(synthesis, iteration)
    routes = synthesis.get("routes") or {}
    pool, defect_spec = Path(str(synthesis["pool_dataset_root"])), Path(str(synthesis["defect_spec"]))
    if not pool.is_dir() or not defect_spec.is_file() or not routes:
        raise ValueError("synthesis pool, defect spec, and routes are required")
    for name, route in routes.items():
        if not Path(str(route.get("checkpoint") or "")).is_file() or not Path(str(route.get("recipe") or "")).is_file():
            raise ValueError(f"route {name} needs resolved checkpoint and recipe; run synthesis bootstrap")
    kpi = policy["sources"]["kpi"]
    images_root = Path(kpi["images"])
    coco = json.loads(Path(kpi["coco"]).read_text())
    images = _image_index(coco["images"])
    annotations: dict[str, list[dict[str, Any]]] = {}
    for row in coco.get("annotations", []):
        annotations.setdefault(str(row["image_id"]), []).append(row)
    gaps = pd.read_parquet(strict_gaps)
    required = {"image_id", "filepath", "gap_type", "bbox", "class"}
    if not required.issubset(gaps):
        raise ValueError(f"strict gaps lack {sorted(required - set(gaps))}")
    rows = []
    skipped_unrouted: Counter[str] = Counter()
    observed_datasets: set[str] = set()
    for gap in gaps[gaps.gap_type.astype(str).str.upper().eq("FN")].to_dict("records"):
        image_id, box = str(gap["image_id"]), _xyxy(gap["bbox"])
        if image_id not in images:
            raise ValueError(f"FN image id is absent from KPI COCO: {image_id}")
        image = images[image_id]
        annotation_image_id = str(image["id"])
        ranked = sorted(((_iou(box, row), row)
                         for row in annotations.get(annotation_image_id, [])),
                        key=lambda item: item[0], reverse=True)
        if not ranked or ranked[0][0] < 0.999:
            raise ValueError(f"FN box has no exact KPI annotation match: {image_id} {box}")
        annotation = ranked[0][1]
        metadata = {}
        for source in (image, image.get("deft_od_aoi", {}), annotation,
                       annotation.get("deft_od_aoi", {})):
            metadata.update({key: source[key] for key in FIELDS if key in source})
        dataset = str(metadata.get("dataset_id") or "").strip()
        if not dataset:
            raise ValueError(
                f"KPI FN {image_id} has no dataset_id; every KPI sample must retain its "
                "real nonempty dataset_id. synthesis.routes is an allowlist, and a missing "
                "dataset_id is malformed metadata, not a synthesis opt-out"
            )
        observed_datasets.add(dataset)
        if dataset not in routes:
            skipped_unrouted[dataset] += 1
            continue
        missing = [key for key in FIELDS[1:] if not str(metadata.get(key) or "").strip()]
        if missing:
            raise ValueError(f"FN metadata is incomplete for {image_id}: {missing}")
        mask = Path(str(metadata["fn_mask_source"])).expanduser().resolve()
        if not mask.is_file():
            raise ValueError(f"FN pixel mask is missing: {mask}")
        metadata["fn_mask_source"] = str(mask)
        source_path = _source(images_root, image)
        gap_path = Path(str(gap["filepath"])).expanduser().resolve()
        try:
            same_file = source_path.samefile(gap_path)
        except OSError:
            same_file = False
        if not same_file:
            raise ValueError(f"gap filepath does not match frozen KPI image: {image_id}")
        rows.append({**gap, "filepath": str(source_path), "split": "kpi", **metadata,
                     "anomaly_type": f"{metadata['texture_id']}+{metadata['defect_class']}"})
    output.mkdir(parents=True)
    if not rows:
        configured_routes = sorted(map(str, routes))
        skipped_count = sum(skipped_unrouted.values())
        report = {
            "status": "SKIPPED", "reason": "no_routed_false_negatives", "fn_count": 0,
            **candidate_depth,
            "skipped_unrouted_fn_count": skipped_count,
            "skipped_unrouted_by_dataset": dict(sorted(skipped_unrouted.items())),
            "observed_dataset_ids": sorted(observed_datasets),
            "configured_route_keys": configured_routes,
            "message": (
                f"Skipped synthesis: all {skipped_count} strict false negatives use "
                f"unrouted dataset IDs {sorted(observed_datasets)}; configured synthesis "
                f"routes are {configured_routes}."
            ),
            "config": "",
        }
        (output / "synthesis_request.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    eligible_count = len(rows)
    per_type: dict[str, dict[str, int]] = {}
    planning = None
    if selection_mode == "generated_per_type_plan":
        if real_coco is None or iteration is None:
            raise ValueError("generated_per_type_plan requires --real-coco and --iteration")
        plan, plan_contract, planning = _generated_plan(
            synthesis, rows, real_coco, output, iteration
        )
        if plan is None:
            report = {
                "status": "SKIPPED", "reason": "no_synthetic_budget", "fn_count": 0,
                **candidate_depth,
                "eligible_fn_count": eligible_count, "selection_mode": selection_mode,
                "planning": planning, "config": "",
            }
            (output / "synthesis_request.json").write_text(json.dumps(report, indent=2) + "\n")
            return report
    if plan is not None and plan_contract is not None:
        frame = pd.DataFrame(rows)
        frame["_image_sort"] = frame["image_id"].map(str)
        frame["_bbox_sort"] = frame["bbox"].map(
            lambda value: json.dumps([float(item) for item in value], separators=(",", ":"))
        )
        selected: list[pd.DataFrame] = []
        images_per_fn = int(plan_contract["images_per_fn"])
        for anomaly_type, requested in plan.items():
            available = frame[frame.anomaly_type.astype(str).eq(anomaly_type)].sort_values(
                ["_image_sort", "filepath", "_bbox_sort"], kind="stable"
            )
            selected_count = min(len(available), requested // images_per_fn)
            if selected_count:
                selected.append(available.head(selected_count))
            frozen_rows = selected_count * images_per_fn
            per_type[anomaly_type] = {
                "eligible_fn_count": int(len(available)),
                "selected_fn_count": selected_count,
                "requested_images": requested,
                "frozen_generator_rows": frozen_rows,
            }
        rows = pd.concat(selected, ignore_index=True).drop(
            columns=["_image_sort", "_bbox_sort"]
        ).to_dict("records")
    normalized = output / "normalized_fn_gaps.parquet"
    pd.DataFrame(rows).to_parquet(normalized, index=False)
    config = {"source_tag": "deft_od_aoi", "gap_parquet": str(normalized),
              "pool_dataset_root": str(pool.resolve()), "defect_spec": str(defect_spec.resolve()),
              "datasets": {name: {"checkpoint": str(Path(route["checkpoint"]).resolve()),
                                    "recipe": str(Path(route["recipe"]).resolve())}
                           for name, route in routes.items()},
              "selection": {"mode": "all_eligible", "datasets": sorted(routes),
                            "mask_sample_seed": 42},
              "embedding": {"model": policy["retrieval"]["model"],
                            "model_path": policy["retrieval"]["model_path"], "batch_size": 64},
              "retrieval": {"metric": "cosine", "candidate_topn": candidate_topn,
                            "max_neighbors_per_fn": int(synthesis["max_neighbors_per_fn"]),
                            "min_similarity": float(synthesis["min_similarity"]),
                            "prior_clean_exclusion_manifest": ""},
              "amp": {"model_id": synthesis["amp_model_id"], "seed": 43}}
    if plan_contract is not None:
        config["synthetic_plan"] = plan_contract
    config_path = output / "anomalygen_filtering.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    report = {
        "status": "COMPLETE",
        **candidate_depth,
        "fn_count": len(rows),
        "eligible_fn_count": eligible_count,
        "selection_mode": selection_mode,
        "skipped_unrouted_fn_count": sum(skipped_unrouted.values()),
        "skipped_unrouted_by_dataset": dict(sorted(skipped_unrouted.items())),
        "observed_dataset_ids": sorted(observed_datasets),
        "configured_route_keys": sorted(map(str, routes)),
        "config": str(config_path.resolve()),
    }
    if plan_contract is not None:
        requested_images = sum(plan.values()) if plan is not None else 0
        frozen_generator_rows = sum(
            row["frozen_generator_rows"] for row in per_type.values()
        )
        if frozen_generator_rows != requested_images:
            raise ValueError("frozen generator rows disagree with the synthetic plan")
        report.update({
            "requested_images": requested_images,
            "frozen_generator_rows": frozen_generator_rows,
            "synthetic_plan": plan_contract,
            "per_type": per_type,
        })
    if planning is not None:
        report["planning"] = planning
    (output / "synthesis_request.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--strict-gaps", type=Path, required=True)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--real-coco", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(
        args.policy.resolve(), args.strict_gaps.resolve(), args.output_dir.resolve(),
        args.iteration, args.real_coco.resolve() if args.real_coco else None,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
