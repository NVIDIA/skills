#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admit mined source images and publish cumulative real-only binary COCO."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _vectors(values: pd.Series) -> np.ndarray:
    rows = [np.asarray(value, dtype=np.float32).reshape(-1) for value in values]
    if not rows or len({row.size for row in rows}) != 1:
        raise ValueError("embeddings are empty or width-mismatched")
    matrix = np.stack(rows)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or np.any(norm == 0):
        raise ValueError("embeddings contain non-finite or zero-norm rows")
    return matrix / norm


def _selected(role: str, reason: str, desired: int, excluded: set[str],
              candidate_root: Path, retrieval_root: Path,
              minimum: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mined = retrieval_root / f"mine_{role}" / "final_unique_files.parquet"
    if not mined.is_file():
        raise FileNotFoundError(f"enabled {role} mining output is missing: {mined}")
    chosen = pd.read_parquet(mined)
    if "filepath" not in chosen:
        raise ValueError(f"enabled {role} mining output lacks filepath")
    if chosen.empty:
        return [], {"desired_parents": desired, "mined_crops": 0,
                    "unique_parents": 0, "novel_parents": 0,
                    "selected_parents": 0, "shortfall_parents": desired,
                    "quota_met": desired == 0}
    if desired == 0:
        return [], {"desired_parents": 0, "mined_crops": len(chosen),
                    "unique_parents": 0, "novel_parents": 0,
                    "selected_parents": 0, "shortfall_parents": 0,
                    "quota_met": True}
    candidates = pd.read_parquet(candidate_root / f"{role}_candidate_embeddings.parquet")
    queries = pd.read_parquet(retrieval_root / f"{role}_query_embeddings.parquet")
    required = {"filepath", "source_filepath", "source_image_id", "embedding"}
    if (not required.issubset(candidates)
            or not {"filepath", "embedding", "reason"}.issubset(queries)):
        raise ValueError(f"{role} embedding outputs lack routing columns")
    queries = queries[queries["reason"].astype(str) == reason]
    if desired and queries.empty:
        raise ValueError(f"{role}/{reason} has a positive target but no queries")
    candidates = chosen[["filepath"]].merge(candidates, on="filepath", how="left", validate="one_to_one")
    if candidates.embedding.isna().any():
        raise ValueError(f"{role} mined paths do not match candidate embeddings")
    candidate_vectors, query_vectors = _vectors(candidates.embedding), _vectors(queries.embedding)
    if candidate_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError(f"{role} candidate/query embedding widths differ")
    candidates["similarity"] = (candidate_vectors @ query_vectors.T).max(axis=1)
    candidates = candidates[candidates.similarity >= minimum].sort_values(
        ["similarity", "source_filepath"], ascending=[False, True]
    )
    parents = candidates.drop_duplicates("source_filepath")
    novel = parents[~parents["source_filepath"].map(
        lambda value: str(Path(value).resolve()) in excluded
    )]
    selected = novel.head(desired)
    stats = {"desired_parents": desired, "mined_crops": len(chosen),
             "unique_parents": len(parents), "novel_parents": len(novel),
             "selected_parents": len(selected),
             "shortfall_parents": max(0, desired - len(selected)),
             "quota_met": len(selected) >= desired}
    return selected.to_dict("records"), stats


def _round_robin_selected(role: str, retrieval_root: Path) -> list[dict[str, Any]]:
    mined = retrieval_root / f"mine_{role}" / "final_unique_files.parquet"
    if not mined.is_file():
        raise FileNotFoundError(f"enabled {role} selection output is missing: {mined}")
    chosen = pd.read_parquet(mined)
    required = {"filepath", "source_filepath", "similarity", "query_id"}
    if not required.issubset(chosen):
        raise ValueError(f"enabled {role} round-robin selection output is invalid")
    if chosen.empty:
        return []
    return chosen.drop_duplicates("source_filepath").to_dict("records")


def _source_index(policy: dict[str, Any], role: str) -> dict[str, dict[str, Any]]:
    source = policy["sources"][role]
    images = Path(source["images"])
    coco = json.loads(Path(source["coco"]).read_text())
    annotations: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco.get("annotations", []):
        annotations.setdefault(int(annotation["image_id"]), []).append(annotation)
    result = {}
    for image in coco["images"]:
        path = Path(str(image.get("source_path") or "")) if image.get("source_path") else (
            images / str(image["file_name"])
        )
        metadata = image.get("deft_od_aoi") or {}
        result[str(path.resolve())] = {
            "image": image,
            "dataset_id": str(
                metadata.get("dataset_id") or image.get("dataset_id") or "unknown"
            ),
            "annotations": annotations.get(int(image["id"]), []),
        }
    return result


def _place(source: Path, images: Path, mode: str) -> Path:
    name = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:20] + source.suffix.lower()
    target = images / name
    if target.exists():
        return target
    if mode == "copy":
        shutil.copy2(source, target)
    else:
        os.link(source, target)
    return target


def _synthetic_candidates(document: dict[str, Any], images_root: Path,
                          minimum_area: float, maximum_aspect: float
                          ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    categories = document.get("categories")
    if not isinstance(categories, list) or len(categories) != 1:
        raise ValueError("synthetic COCO must declare exactly one category")
    category_id = categories[0].get("id")
    annotations: dict[int, list[dict[str, Any]]] = {}
    for row in document.get("annotations", []):
        if row.get("category_id") != category_id:
            raise ValueError("synthetic COCO contains an unexpected category")
        annotations.setdefault(int(row["image_id"]), []).append(row)
    quality = collections.Counter({
        "input_images": 0, "input_annotations": 0,
        "rejected_annotations_small": 0, "rejected_annotations_aspect": 0,
        "rejected_annotations_full_frame": 0,
        "rejected_images_no_eligible_boxes": 0,
    })
    candidates = []
    for image in document.get("images", []):
        width, height = int(image["width"]), int(image["height"])
        if width < 1 or height < 1:
            raise ValueError("synthetic COCO image has invalid dimensions")
        quality["input_images"] += 1
        rows = []
        for row in annotations.get(int(image["id"]), []):
            quality["input_annotations"] += 1
            try:
                x, y, box_width, box_height = map(float, row["bbox"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("synthetic COCO contains an invalid bbox") from error
            if (not all(math.isfinite(value) for value in (x, y, box_width, box_height))
                    or min(x, y) < 0 or box_width <= 0 or box_height <= 0
                    or x + box_width > width or y + box_height > height):
                raise ValueError("synthetic COCO bbox is invalid or outside its image")
            if box_width * box_height < minimum_area:
                quality["rejected_annotations_small"] += 1
                continue
            if max(box_width / box_height, box_height / box_width) > maximum_aspect:
                quality["rejected_annotations_aspect"] += 1
                continue
            if x == 0 and y == 0 and box_width == width and box_height == height:
                quality["rejected_annotations_full_frame"] += 1
                continue
            rows.append(row)
        if not rows:
            quality["rejected_images_no_eligible_boxes"] += 1
            continue
        source = Path(str(image.get("source_path") or image["file_name"]))
        if not source.is_absolute():
            source = images_root / source
        if not source.is_file():
            raise FileNotFoundError(f"synthetic image is missing: {source}")
        candidates.append({"source": source.resolve(), "image": image, "rows": rows,
                           "stratum": str(image.get("dataset_id") or "unknown")})
    quality["eligible_images"] = len(candidates)
    quality["eligible_annotations"] = sum(len(row["rows"]) for row in candidates)
    return candidates, dict(quality)


def _stratified_synthetic(candidates: list[dict[str, Any]], limit: int
                          ) -> list[dict[str, Any]]:
    if limit <= 0 or not candidates:
        return []
    unique = {str(row["source"]): row for row in candidates}
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in unique.values():
        groups[row["stratum"]].append(row)
    if limit >= len(unique):
        return [unique[key] for key in sorted(unique)]
    exact = {name: limit * len(rows) / len(unique) for name, rows in groups.items()}
    allocation = {name: math.floor(value) for name, value in exact.items()}
    remaining = limit - sum(allocation.values())
    for name in sorted(groups, key=lambda value: (-(exact[value] % 1), value)):
        if remaining <= 0:
            break
        allocation[name] += 1
        remaining -= 1
    selected = []
    for name, rows in groups.items():
        ranked = sorted(rows, key=lambda row: (
            hashlib.sha256(str(row["source"]).encode()).hexdigest(), str(row["source"]),
        ))
        selected.extend(ranked[:allocation[name]])
    return sorted(selected, key=lambda row: str(row["source"]))


def _synthetic_limit(real_total: int, synthesis: dict[str, Any]) -> tuple[int, str, float]:
    if not synthesis or synthesis.get("enabled") is False:
        return 0, "disabled", 0.0
    if "cumulative_fraction_of_total_defects" in synthesis:
        fraction = float(synthesis["cumulative_fraction_of_total_defects"])
        if not math.isfinite(fraction) or not 0 <= fraction < 1:
            raise ValueError("cumulative_fraction_of_total_defects must be in [0, 1)")
        return int(fraction / (1.0 - fraction) * real_total), "fraction_of_total", fraction
    if "cumulative_fraction_of_real_defects" not in synthesis:
        return 0, "unconfigured", 0.0
    fraction = float(synthesis["cumulative_fraction_of_real_defects"])
    if not math.isfinite(fraction) or fraction < 0:
        raise ValueError("cumulative_fraction_of_real_defects must be finite and nonnegative")
    return int(real_total * fraction), "fraction_of_real", fraction


def admit(policy_path: Path, candidate_root: Path, retrieval_root: Path, output: Path,
          previous_path: Path | None, mode: str, synthetic_coco: Path | None = None,
          synthetic_images: Path | None = None, synthetic_only: bool = False) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    policy = yaml.safe_load(policy_path.read_text())
    manifest = json.loads((retrieval_root / "query_manifest.json").read_text())
    previous = ({"images": [], "annotations": [], "categories": []} if previous_path is None
                else json.loads(previous_path.read_text()))
    existing_sources = {str(row.get("source_path") or Path(str(row["file_name"])).resolve())
                        for row in previous.get("images", [])}
    by_kind = {kind: sum(row.get("deft_kind") == kind for row in previous.get("images", []))
               for kind in ("real_defect", "clean_negative", "synthetic_defect")}
    if synthetic_only and (previous_path is None or synthetic_coco is None):
        raise ValueError("--synthetic-only requires --previous-coco and synthetic inputs")
    enabled = set() if synthetic_only else set(manifest["enabled_roles"])
    indexes = {role: _source_index(policy, role) for role in enabled}
    strategy = ((policy.get("retrieval") or {}).get("selection") or {}).get(
        "strategy", "round_robin_similarity"
    )
    selection_audit = None
    selection_index = None
    targets = manifest.get("admission_targets") or {}
    additions: dict[str, list[dict[str, Any]]] = {role: [] for role in enabled}
    preview: dict[str, Any] = {
        "status": "PASS", "iteration": int(manifest["iteration"]),
        "mining_admission": "skipped" if synthetic_only else "evaluated", "roles": {},
    }
    retrieval_admission = {
        role: {"requested": 0, "deduplicated": 0, "capped": 0}
        for role in ("real", "clean")
    }
    minimum = float(policy["retrieval"]["minimum_similarity"])
    if strategy == "round_robin_similarity":
        selection_report = json.loads(
            (retrieval_root / "round_robin_selection_report.json").read_text()
        )
        if (selection_report.get("status") != "COMPLETE"
                or int(selection_report.get("iteration", -1)) != int(manifest["iteration"])
                or selection_report.get("selection_strategy") != strategy):
            raise ValueError("round-robin selection report is incomplete or mismatched")
        selection_audit = selection_report.get("audit")
        selection_index = retrieval_root / "round_robin_admission_index.npy"
        if (selection_report.get("admission_index") != str(selection_index)
                or not selection_index.is_file()):
            raise ValueError("round-robin admission index is missing or mismatched")
        additions = {
            role: _round_robin_selected(role, retrieval_root) for role in enabled
        }
        for role, rows in additions.items():
            retrieval_admission[role]["requested"] = len(rows)
    else:
        if "real" in enabled:
            excluded = set(existing_sources)
            branches = {}
            for reason in ("fn", "near_miss_fp"):
                requested = int((targets.get("real") or {}).get(reason, 0))
                selected, stats = _selected(
                    "real", reason, requested,
                    excluded, candidate_root, retrieval_root, minimum,
                )
                additions["real"].extend(selected)
                excluded.update(
                    str(Path(row["source_filepath"]).resolve()) for row in selected
                )
                branches[reason] = stats
                retrieval_admission["real"]["requested"] += requested
            preview["roles"]["real"] = {
                "requested_crop_count": int(
                    (manifest.get("requested_crop_counts") or {}).get("real", 0)
                ),
                "branches": branches,
            }
    for role in additions:
        additions[role] = [row for row in additions[role]
                           if str(Path(row["source_filepath"]).resolve()) not in existing_sources]
        retrieval_admission[role]["deduplicated"] = len(additions[role])
    real_total = by_kind["real_defect"] + len(additions.get("real", []))
    clean_limit = int(real_total * float(policy["routing"]["clean_cumulative_cap_per_real"]))
    clean_room = max(0, clean_limit - by_kind["clean_negative"])
    if "clean" in enabled:
        if strategy == "round_robin_similarity":
            additions["clean"] = additions.get("clean", [])[:clean_room]
        else:
            clean_target = (targets.get("clean") or {}).get("background_fp")
            requested = int(clean_target) if clean_target is not None else clean_room
            selected, stats = _selected("clean", "background_fp", requested, existing_sources,
                                        candidate_root, retrieval_root, minimum)
            retrieval_admission["clean"]["requested"] = requested
            retrieval_admission["clean"]["deduplicated"] = len(selected)
            additions["clean"] = selected[:clean_room]
            desired = min(requested, clean_room)
            stats.update({
                "desired_parents": desired,
                "selected_parents": len(additions["clean"]),
                "shortfall_parents": max(0, desired - len(additions["clean"])),
                "quota_met": len(additions["clean"]) >= desired,
            })
            preview["roles"]["clean"] = {
                "requested_crop_count": int(
                    (manifest.get("requested_crop_counts") or {}).get("clean", 0)
                ),
                "branches": {"background_fp": stats},
            }
    for role in additions:
        retrieval_admission[role]["capped"] = len(additions[role])
    for role, rows in additions.items():
        missing = [str(Path(row["source_filepath"]).resolve()) for row in rows
                   if str(Path(row["source_filepath"]).resolve()) not in indexes[role]]
        if missing:
            raise ValueError(
                f"selected {role} source is absent from its frozen COCO: {missing[0]}"
            )
        counts = collections.Counter(
            indexes[role][str(Path(row["source_filepath"]).resolve())]["dataset_id"]
            for row in rows
        )
        preview["roles"].setdefault(role, {})["per_dataset"] = dict(sorted(counts.items()))
    role_status = {
        role: {"status": "SELECTED" if rows else "NO_MATCHES",
               "selected_count": len(rows)}
        for role, rows in additions.items()
    }
    output.mkdir(parents=True)
    _json(output / "admission_preview.json", preview)
    images_root = output / "images"
    images_root.mkdir()
    images, annotations = [], []
    next_image = next_annotation = 1

    def append(source: Path, image: dict[str, Any], rows: list[dict[str, Any]], kind: str,
               similarity: float | None) -> None:
        nonlocal next_image, next_annotation
        target = _place(source, images_root, mode)
        images.append({**image, "id": next_image, "file_name": target.name,
                       "source_path": str(source.resolve()), "deft_kind": kind,
                       "retrieval_similarity": similarity})
        for row in rows:
            annotations.append({**row, "id": next_annotation, "image_id": next_image,
                                "category_id": 1})
            next_annotation += 1
        next_image += 1

    old_annotations: dict[int, list[dict[str, Any]]] = {}
    for row in previous.get("annotations", []):
        old_annotations.setdefault(int(row["image_id"]), []).append(row)
    for image in previous.get("images", []):
        source = Path(str(image.get("source_path") or image["file_name"]))
        if not source.is_file() and previous_path:
            source = previous_path.parent / "images" / Path(str(image["file_name"])).name
        append(source, image, old_annotations.get(int(image["id"]), []),
               str(image["deft_kind"]), image.get("retrieval_similarity"))
    admitted_rows = []
    for role, kind in (("real", "real_defect"), ("clean", "clean_negative")):
        index = indexes.get(role, {})
        for selected in additions.get(role, []):
            source = Path(str(selected["source_filepath"])).resolve()
            if str(source) not in index:
                raise ValueError(f"selected {role} source is absent from its frozen COCO: {source}")
            row = index[str(source)]
            source_annotations = row["annotations"]
            if role == "real" and selected.get("admission_boxes") is not None:
                source_annotations = [
                    {"bbox": [float(value) for value in box],
                     "area": float(box[2] * box[3]), "iscrowd": 0}
                    for box in selected["admission_boxes"]
                ]
            append(source, row["image"], source_annotations, kind,
                   float(selected["similarity"]))
            admitted_rows.append({"source_filepath": str(source), "kind": kind,
                                  "similarity": float(selected["similarity"])})
    synthetic_admitted = 0
    synthetic_quality: dict[str, int] = {}
    synthetic_requested = 0
    admitted_by_stratum: dict[str, int] = {}
    synthetic_limit, fraction_basis, synthetic_fraction = _synthetic_limit(
        real_total, policy.get("synthesis") or {}
    )
    synthetic_room = max(0, synthetic_limit - by_kind["synthetic_defect"])
    warnings = []
    if (policy.get("synthesis") or {}).get("enabled") and synthetic_room == 0:
        warnings.append({
            "code": "SYNTHETIC_ADMISSION_CAP_ZERO",
            "message": (
                "synthetic admission currently has zero room; generation may complete "
                "without admitting any synthetic images"
            ),
            "cumulative_real_images": real_total,
            "configured_fraction": synthetic_fraction,
            "cumulative_synthetic_limit": synthetic_limit,
            "synthetic_images_before_admission": by_kind["synthetic_defect"],
        })
    if bool(synthetic_coco) != bool(synthetic_images):
        raise ValueError("pass both synthetic COCO and synthetic images, or neither")
    if synthetic_coco and synthetic_images:
        document = json.loads(synthetic_coco.read_text())
        admission = policy.get("admission") or {}
        candidates, synthetic_quality = _synthetic_candidates(
            document, synthetic_images,
            float(admission.get("minimum_box_area_px", 64)),
            float(admission.get("maximum_box_aspect", 25.0)),
        )
        unique = {}
        for candidate in candidates:
            key = str(candidate["source"])
            if key in existing_sources:
                continue
            if key in unique and unique[key]["rows"] != candidate["rows"]:
                raise ValueError(f"conflicting duplicate synthetic source: {key}")
            unique.setdefault(key, candidate)
        synthetic_requested = len(unique)
        admitted = _stratified_synthetic(list(unique.values()), synthetic_room)
        admitted_by_stratum = dict(sorted(collections.Counter(
            row["stratum"] for row in admitted
        ).items()))
        for candidate in admitted:
            source = candidate["source"]
            append(source, candidate["image"], candidate["rows"], "synthetic_defect", None)
            existing_sources.add(str(source))
            synthetic_admitted += 1
            admitted_rows.append({"source_filepath": str(source),
                                  "kind": "synthetic_defect", "similarity": None})
        excluded_by_cap = synthetic_requested - synthetic_admitted
        if excluded_by_cap:
            warnings.append({
                "code": "SYNTHETIC_ADMISSION_CAPPED",
                "message": (
                    f"{excluded_by_cap} eligible generated synthetic images were excluded "
                    "by the configured cumulative synthetic-admission cap"
                ),
                "cumulative_real_images": real_total,
                "configured_fraction": synthetic_fraction,
                "cumulative_synthetic_limit": synthetic_limit,
                "synthetic_images_before_admission": by_kind["synthetic_defect"],
            })
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "defect"}]}
    _json(output / "train.json", coco)
    pd.DataFrame(admitted_rows, columns=["source_filepath", "kind", "similarity"]).to_parquet(
        output / "admitted_sources.parquet", index=False
    )
    if selection_index is not None:
        shutil.copy2(selection_index, output / "admission_index.npy")
    report = {"status": "COMPLETE", "iteration": int(manifest["iteration"]),
              "selection_strategy": strategy,
              "selection_audit": selection_audit,
              "selection_admission_counters": (
                  selection_report.get("admission_counters")
                  if strategy == "round_robin_similarity" else None
              ),
              "admission_index": (
                  str(output / "admission_index.npy")
                  if strategy == "round_robin_similarity" else None
              ),
              "role_status": role_status,
              "retained_previous_images": len(previous.get("images", [])),
              "admitted": {"real": len(additions.get("real", [])),
                           "clean": len(additions.get("clean", [])),
                           "synthetic": synthetic_admitted},
              "total_images": len(images), "total_annotations": len(annotations),
              "by_kind": {kind: sum(row["deft_kind"] == kind for row in images)
                          for kind in ("real_defect", "clean_negative", "synthetic_defect")},
              "retrieval_admission": retrieval_admission,
              "synthetic_admission": {
                  "fraction_basis": fraction_basis,
                  "cumulative_limit": synthetic_limit,
                  "requested_new": synthetic_requested,
                  "admitted_new": synthetic_admitted,
                  "excluded_by_cap": synthetic_requested - synthetic_admitted,
                  "configured_fraction": synthetic_fraction,
                  "cumulative_real_images": real_total,
                  "cumulative_synthetic_limit": synthetic_limit,
                  "available_room_before_admission": synthetic_room,
                  "admitted_by_stratum": admitted_by_stratum,
                  "quality_filter": synthetic_quality,
              },
              "warnings": warnings,
              "new_training_images": len(admitted_rows),
              "training_pool_mutated": False}
    _json(output / "admission_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous-coco", type=Path)
    parser.add_argument("--synthetic-coco", type=Path)
    parser.add_argument("--synthetic-images", type=Path)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--link-mode", choices=("copy", "hardlink"), default="copy")
    args = parser.parse_args()
    result = admit(args.policy.resolve(), args.candidate_root.resolve(),
                   args.retrieval_root.resolve(), args.output_dir.resolve(),
                   args.previous_coco.resolve() if args.previous_coco else None, args.link_mode,
                   args.synthetic_coco.resolve() if args.synthetic_coco else None,
                   args.synthetic_images.resolve() if args.synthetic_images else None,
                   args.synthetic_only)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
