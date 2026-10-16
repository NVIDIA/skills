#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare reusable candidate crops or per-iteration DEFT AOI gap queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image, ImageOps, ImageStat


MIN_EMBEDDING_EDGE = 8
PREPROCESSING_PROFILES = {"tight_context", "square_context"}


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _id(*values: Any) -> str:
    return hashlib.sha256("\0".join(map(str, values)).encode()).hexdigest()[:16]


def _source(images: Path, row: dict[str, Any]) -> Path:
    path = Path(str(row.get("source_path") or "")) if row.get("source_path") else (
        images / str(row["file_name"])
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _box(value: Any, width: int, height: int, scale: float = 1.0) -> tuple[int, int, int, int]:
    x, y, w, h = map(float, value)
    if min(x, y) < 0 or w <= 0 or h <= 0:
        raise ValueError(f"invalid xywh box: {value}")
    cx, cy, w, h = x + w / 2, y + h / 2, w * scale, h * scale
    x1, y1 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
    x2, y2 = min(width, int(cx + w / 2 + 0.999)), min(height, int(cy + h / 2 + 0.999))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"box clips empty: {value}")
    return x1, y1, x2, y2


def _gap_box(value: Any, width: int, height: int, scale: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, value)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid xyxy gap box: {value}")
    x1, y1 = max(0.0, min(x1, width)), max(0.0, min(y1, height))
    x2, y2 = max(0.0, min(x2, width)), max(0.0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"gap box clips empty: {value}")
    return _box((x1, y1, x2 - x1, y2 - y1), width, height, scale)


def _minimum_crop_geometry(
    box: tuple[int, int, int, int], width: int, height: int,
    minimum: int = MIN_EMBEDDING_EDGE,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Expand a crop around its center, padding only undersized source axes."""
    if width < 1 or height < 1 or minimum < 1:
        raise ValueError("crop image dimensions and minimum edge must be positive")

    def axis(start: int, end: int, limit: int) -> tuple[int, int, int, int]:
        start, end = max(0, start), min(limit, end)
        if end <= start:
            raise ValueError(f"crop clips empty on axis: {(start, end)} of {limit}")
        if end - start >= minimum:
            return start, end, 0, 0
        center = (start + end) / 2
        if limit >= minimum:
            expanded_start = min(max(0, int(center - minimum / 2)), limit - minimum)
            return expanded_start, expanded_start + minimum, 0, 0
        padding = minimum - limit
        before = min(padding, max(0, round(minimum / 2 - center)))
        return 0, limit, before, padding - before

    x1, x2, left, right = axis(box[0], box[2], width)
    y1, y2, top, bottom = axis(box[1], box[3], height)
    return (x1, y1, x2, y2), (left, top, right, bottom)


def _display_image(source: Path) -> Image.Image:
    with Image.open(source) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def _crop(source: Path, box: tuple[int, int, int, int], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with _display_image(source) as image:
        crop_box, padding = _minimum_crop_geometry(box, image.width, image.height)
        crop = image.crop(crop_box)
        if any(padding):
            mean = tuple(int(round(value)) for value in ImageStat.Stat(crop).mean[:3])
            crop = ImageOps.expand(crop, border=padding, fill=mean)
        crop.save(output, format="PNG")


def _size(source: Path) -> tuple[int, int]:
    with _display_image(source) as image:
        return image.size


def _preprocessing(policy: dict[str, Any]) -> tuple[str, int | None]:
    retrieval = policy["retrieval"]
    # Policies frozen before profiles existed used the original tight-context
    # geometry; new policies explicitly select the square-context default.
    profile = str((retrieval.get("preprocessing") or {}).get("profile", "tight_context"))
    if profile not in PREPROCESSING_PROFILES:
        raise ValueError(
            "retrieval.preprocessing.profile must be tight_context or square_context"
        )
    output_size = None
    if profile == "square_context":
        output_size = int(retrieval.get("output_size", 224))
        if output_size < 1:
            raise ValueError("retrieval.output_size must be positive")
    return profile, output_size


def _save_square(image: Image.Image, output: Path, size: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC).save(
        temporary, format="PNG", optimize=False
    )
    os.replace(temporary, output)


def _mean_pad_square(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    side = max(image.size)
    mean = tuple(int(round(value)) for value in ImageStat.Stat(image).mean[:3])
    canvas = Image.new("RGB", (side, side), mean)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def _square_context_crop(source: Path, bbox: Any, scale: float, output: Path,
                         size: int) -> None:
    x, y, width, height = map(float, bbox)
    if width <= 0 or height <= 0:
        raise ValueError(f"non-positive bbox: {bbox}")
    with _display_image(source) as image:
        side = max(1, int(math.ceil(max(width, height) * scale)))
        center_x, center_y = x + width / 2.0, y + height / 2.0
        left = int(math.floor(center_x - side / 2.0))
        top = int(math.floor(center_y - side / 2.0))
        clipped = (
            max(0, left), max(0, top), min(image.width, left + side),
            min(image.height, top + side),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            raise ValueError(f"bbox context lies outside image: {bbox}")
        visible = image.crop(clipped)
        mean = tuple(int(round(value)) for value in ImageStat.Stat(visible).mean[:3])
        canvas = Image.new("RGB", (side, side), mean)
        canvas.paste(visible, (clipped[0] - left, clipped[1] - top))
        _save_square(canvas, output, size)


def _square_grid_crops(source: Path, grids: list[int], output: Path,
                       image_id: Any, size: int) -> list[tuple[str, Path]]:
    rows = []
    with _display_image(source) as image:
        for grid in grids:
            if grid < 1:
                raise ValueError("clean grid values must be positive")
            for row in range(grid):
                for column in range(grid):
                    box = (
                        round(column * image.width / grid),
                        round(row * image.height / grid),
                        round((column + 1) * image.width / grid),
                        round((row + 1) * image.height / grid),
                    )
                    crop_id = f"clean:{image_id}:g{grid}:r{row}:c{column}"
                    crop = output / "crops" / "clean" / (
                        f"image_{image_id}_g{grid}_r{row}_c{column}.png"
                    )
                    _save_square(_mean_pad_square(image.crop(box)), crop, size)
                    rows.append((crop_id, crop))
    return rows


def _gap_xywh(value: Any, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = _gap_box(value, width, height, 1.0)
    return [x1, y1, x2 - x1, y2 - y1]


def _embedding_spec(policy: dict[str, Any], input_path: Path, output: Path) -> dict[str, Any]:
    retrieval = policy["retrieval"]
    return {"input_parquet": str(input_path), "output_parquet": str(output),
            "model": retrieval["model"], "model_path": retrieval["model_path"],
            "model_config_path": "", "batch_size": 64}


def _kpi_pockets(policy: dict[str, Any]) -> dict[str, dict[str, str]]:
    source = policy["sources"]["kpi"]
    images = Path(source["images"])
    document = json.loads(Path(source["coco"]).read_text())
    result = {}
    for row in document["images"]:
        path = _source(images, row)
        metadata = row.get("deft_od_aoi") or {}
        values = {
            key: str(metadata.get(key) or row.get(key) or "unknown")
            for key in ("dataset_id", "texture_id", "defect_class")
        }
        values["pocket"] = "/".join(values[key] for key in (
            "dataset_id", "texture_id", "defect_class"
        ))
        result[str(path)] = values
    return result


def _path_values(path: Path, label: str) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} exclusion input is missing: {path}")
    frame = pd.read_parquet(path)
    if "filepath" not in frame:
        raise ValueError(f"{label} exclusion input lacks filepath")
    return {str(Path(value).expanduser().resolve()) for value in frame.filepath.astype(str)}


def _history_sources(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.is_file():
        raise FileNotFoundError(f"previous COCO is missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value.get("images"), list):
        raise ValueError("previous COCO lacks images")
    return {
        str(Path(str(row.get("original_source_path") or row.get("source_path")
                          or path.parent / "images" / row["file_name"])).resolve())
        for row in value["images"]
    }


def _candidate_counts(root: Path) -> dict[str, int]:
    path = root / "candidate_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"candidate manifest is missing: {path}")
    value = json.loads(path.read_text())
    counts = value.get("counts")
    if (value.get("status") != "COMPLETE" or not isinstance(counts, dict)
            or set(counts) != {"real", "clean"}):
        raise ValueError("candidate manifest lacks complete real/clean counts")
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0
           for count in counts.values()):
        raise ValueError("candidate manifest counts must be nonnegative integers")
    return counts


def _validate_candidate_manifest(
    policy: dict[str, Any], candidate_root: Path, profile: str, output_size: int | None
) -> None:
    path = candidate_root / "candidate_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"candidate cache lacks manifest: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "COMPLETE":
        raise ValueError("candidate cache manifest is not COMPLETE")
    # Manifests written before profiles existed used the original tight-context
    # geometry; new candidate caches always record their selected profile.
    cached_profile = str(manifest.get("preprocessing_profile") or "tight_context")
    if cached_profile != profile:
        raise ValueError(
            f"candidate cache preprocessing profile {cached_profile!r} "
            f"does not match policy {profile!r}"
        )
    cached = manifest.get("encoder") or {}
    current = policy["retrieval"]
    checks: dict[str, Any] = {
        "model": current.get("model"),
        "model_path": current.get("model_path"),
        "defect_context_scale": current.get("defect_context_scale"),
        "clean_grids": [int(value) for value in current.get("clean_grids", [])],
    }
    if profile == "square_context":
        checks["output_size"] = output_size
    mismatched = [
        key for key, value in checks.items()
        if cached.get(key) != value
    ]
    if mismatched:
        raise ValueError(
            "candidate cache encoder/preprocessing settings do not match policy: "
            + ", ".join(mismatched)
        )


def candidates(policy_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    policy = yaml.safe_load(policy_path.read_text())
    profile, output_size = _preprocessing(policy)
    output.mkdir(parents=True)
    result: dict[str, int] = {}
    role_status: dict[str, dict[str, Any]] = {}
    for role in ("real", "clean"):
        source = policy["sources"][role]
        images = Path(source["images"])
        coco = json.loads(Path(source["coco"]).read_text())
        annotations: dict[int, list[dict[str, Any]]] = {}
        for annotation in coco.get("annotations", []):
            annotations.setdefault(int(annotation["image_id"]), []).append(annotation)
        rows = []
        image_rows = (
            sorted(coco["images"], key=lambda row: str(row["id"]))
            if profile == "square_context" else coco["images"]
        )
        for image_row in image_rows:
            source_path = _source(images, image_row)
            width, height = _size(source_path)
            if role == "real":
                image_annotations = annotations.get(int(image_row["id"]), [])
                if profile == "square_context":
                    image_annotations = sorted(
                        image_annotations, key=lambda row: str(row["id"])
                    )
                for annotation in image_annotations:
                    if profile == "square_context":
                        crop_id = f"defect:{image_row['id']}:{annotation['id']}"
                        crop = output / "crops" / role / (
                            f"image_{image_row['id']}_ann_{annotation['id']}.png"
                        )
                        _square_context_crop(
                            source_path, annotation["bbox"],
                            float(policy["retrieval"]["defect_context_scale"]),
                            crop, output_size,
                        )
                    else:
                        box = _box(annotation["bbox"], width, height,
                                   float(policy["retrieval"]["defect_context_scale"]))
                        crop_id = "real-" + _id(source_path, annotation["id"], box)
                        crop = output / "crops" / role / f"{crop_id}.png"
                        _crop(source_path, box, crop)
                    rows.append({"filepath": str(crop), "source_filepath": str(source_path),
                                 "source_image_id": int(image_row["id"]), "role": role,
                                 "candidate_id": crop_id, "source_bbox": annotation["bbox"]})
            elif role == "clean":
                if profile == "square_context":
                    for crop_id, crop in _square_grid_crops(
                            source_path,
                            [int(value) for value in policy["retrieval"]["clean_grids"]],
                            output, image_row["id"], output_size):
                        rows.append({"filepath": str(crop),
                                     "source_filepath": str(source_path),
                                     "source_image_id": int(image_row["id"]), "role": role,
                                     "candidate_id": crop_id, "source_bbox": None})
                else:
                    for grid in policy["retrieval"]["clean_grids"]:
                        grid = int(grid)
                        if grid < 1:
                            raise ValueError("clean grid values must be positive")
                        for row in range(grid):
                            for column in range(grid):
                                box = (column * width // grid, row * height // grid,
                                       (column + 1) * width // grid,
                                       (row + 1) * height // grid)
                                crop_id = "clean-" + _id(source_path, grid, row, column)
                                crop = output / "crops" / role / f"{crop_id}.png"
                                _crop(source_path, box, crop)
                                rows.append({"filepath": str(crop),
                                             "source_filepath": str(source_path),
                                             "source_image_id": int(image_row["id"]),
                                             "role": role, "candidate_id": crop_id,
                                             "source_bbox": None})
            else:
                raise ValueError(f"unsupported candidate role: {role}")
        frame = pd.DataFrame(rows)
        if frame.empty:
            result[role] = 0
            role_status[role] = {"status": "EXHAUSTED", "candidate_count": 0,
                                 "reason": "empty_source_role"}
            continue
        parquet = output / f"{role}_candidates.parquet"
        frame.to_parquet(parquet, index=False)
        spec = _embedding_spec(policy, parquet, output / f"{role}_candidate_embeddings.parquet")
        (output / f"embed_{role}_candidates.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
        result[role] = len(frame)
        role_status[role] = {"status": "READY", "candidate_count": len(frame),
                             "reason": None}
    warnings = [{"code": "empty_retrieval_candidate_role", "role": role,
                 "message": f"{role} retrieval has zero candidates; embedding and mining are disabled"}
                for role, evidence in role_status.items() if evidence["status"] == "EXHAUSTED"]
    report = {"status": "COMPLETE", "counts": result, "role_status": role_status,
              "warnings": warnings, "encoder": policy["retrieval"],
              "preprocessing_profile": profile}
    _json(output / "candidate_manifest.json", report)
    return report


def _previous_sources(previous_path: Path | None) -> dict[str, set[Path]]:
    result = {"real": set(), "clean": set()}
    if previous_path is None:
        return result
    previous = json.loads(previous_path.read_text())
    kinds = {"real_defect": "real", "clean_negative": "clean"}
    for row in previous.get("images", []):
        role = kinds.get(str(row.get("deft_kind") or ""))
        if role:
            for key in ("source_path", "original_source_path"):
                raw = row.get(key)
                if raw:
                    result[role].add(Path(str(raw)).resolve())
    return result


def queries(policy_path: Path, strict_path: Path, loose_path: Path, iteration: int,
            output: Path, candidate_root: Path, real_factor: int | None,
            previous_coco: Path | None = None,
            exclusion_paths: dict[str, Path] | None = None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    policy = yaml.safe_load(policy_path.read_text())
    profile, output_size = _preprocessing(policy)
    _validate_candidate_manifest(policy, candidate_root, profile, output_size)
    strategy = ((policy.get("retrieval") or {}).get("selection") or {}).get(
        "strategy", "round_robin_similarity"
    )
    routing = policy["routing"]
    round_robin_default = int(routing["round_robin_real_factor_default"])
    strict, loose = pd.read_parquet(strict_path), pd.read_parquet(loose_path)
    required = {"filepath", "gap_type", "bbox", "best_iou"}
    for label, frame in (("strict", strict), ("loose", loose)):
        if not required.issubset(frame.columns):
            raise ValueError(f"{label} gaps lack {sorted(required - set(frame.columns))}")
    events = []
    for gap_index, row in strict[
            strict.gap_type.astype(str).str.upper().eq("FN")].iterrows():
        events.append(("real", "fn", "strict", gap_index, row.to_dict()))
    gap = policy["gap"]
    for gap_index, row in loose[
            loose.gap_type.astype(str).str.upper().eq("FP")].iterrows():
        iou = float(row["best_iou"])
        if iou < gap["background_iou_upper"]:
            events.append(("clean", "background_fp", "loose", gap_index, row.to_dict()))
        elif iou < gap["near_miss_iou_upper"]:
            events.append(("real", "near_miss_fp", "loose", gap_index, row.to_dict()))
    output.mkdir(parents=True)
    pockets = _kpi_pockets(policy)
    history = _history_sources(previous_coco)
    prior_sources = _previous_sources(previous_coco)
    candidate_counts = _candidate_counts(candidate_root)
    configured_exclusions = exclusion_paths or {}
    counts, frames, targets, requested, role_status = {}, {}, {}, {}, {}
    excluded_candidate_crops, excluded_source_images = {}, {}
    warnings = []
    for role in ("real", "clean"):
        rows = []
        for index, (_, reason, gap_pass, gap_index, event) in enumerate(
                item for item in events if item[0] == role):
            source = Path(str(event["filepath"])).resolve()
            if str(source) not in pockets:
                raise ValueError(f"gap image is absent from the frozen KPI role: {source}")
            if role == "real" and "unknown" in pockets[str(source)].values():
                raise ValueError(f"gap image lacks frozen pocket metadata: {source}")
            width, height = _size(source)
            if profile == "square_context":
                box = _gap_xywh(event["bbox"], width, height)
            else:
                box = _gap_box(event["bbox"], width, height,
                               float(policy["retrieval"]["defect_context_scale"]))
            query_id = f"iter{iteration}-{role}-" + _id(source, event["bbox"], reason, index)
            crop = output / "crops" / role / f"{query_id}.png"
            if profile == "square_context":
                _square_context_crop(
                    source, box, float(policy["retrieval"]["defect_context_scale"]),
                    crop, output_size,
                )
            else:
                _crop(source, box, crop)
            row = {"filepath": str(crop), "query_id": query_id, "role": role,
                   "reason": reason, "source_filepath": str(source),
                   "source_bbox": event["bbox"], "best_iou": float(event["best_iou"]),
                   "routing_order_key": (
                       f"{gap_pass}:{gap_index}:"
                       f"{'strict_fn' if reason == 'fn' else reason}"
                   ), **pockets[str(source)]}
            if strategy == "round_robin_similarity" and role == "real":
                if reason == "fn":
                    factor_value = event.get("real_factor")
                    if factor_value is None or pd.isna(factor_value):
                        factor_value = real_factor or round_robin_default
                    row["real_factor"] = int(factor_value)
            rows.append(row)
        counts[role], frames[role] = len(rows), pd.DataFrame(rows)
        if not rows:
            excluded_candidate_crops[role] = 0
            excluded_source_images[role] = 0
            role_status[role] = {"status": "NO_QUERIES", "query_count": 0,
                                 "candidate_count": candidate_counts[role],
                                 "excluded_count": 0,
                                 "remaining_candidate_count": candidate_counts[role]}
            continue
        parquet = output / f"{role}_queries.parquet"
        frames[role].to_parquet(parquet, index=False)
        if candidate_counts[role] == 0:
            exclusion_file = output / f"exclude_{role}_candidates.parquet"
            pd.DataFrame(columns=["filepath"]).to_parquet(exclusion_file, index=False)
            role_status[role] = {
                "status": "EXHAUSTED", "reason": "zero_candidate_count",
                "query_count": len(rows), "candidate_count": 0, "excluded_count": 0,
                "remaining_candidate_count": 0,
                "exclusion_manifest": str(exclusion_file.resolve()),
                "history_source_count": 0, "explicit_exclusion_count": 0,
            }
            excluded_candidate_crops[role] = 0
            excluded_source_images[role] = 0
            warnings.append({
                "code": "retrieval_role_has_no_candidates", "role": role,
                "message": f"{role} queries were routed to an empty candidate role; mining is skipped",
            })
            continue
        candidate_path = candidate_root / f"{role}_candidates.parquet"
        if not candidate_path.is_file():
            raise FileNotFoundError(f"{role} candidate manifest is missing: {candidate_path}")
        candidates = pd.read_parquet(candidate_path)
        required_candidates = {"filepath", "source_filepath"}
        if candidates.empty or not required_candidates.issubset(candidates):
            raise ValueError(
                f"{role} candidate manifest lacks usable {sorted(required_candidates)}"
            )
        candidate_files = {
            str(Path(value).expanduser().resolve()) for value in candidates.filepath.astype(str)
        }
        candidate_sources = {
            str(Path(value).expanduser().resolve()) for value in candidates.source_filepath.astype(str)
        }
        if len(candidate_files) != candidate_counts[role]:
            raise ValueError(f"{role} candidate parquet disagrees with candidate manifest")
        if prior_sources[role] and not {
                str(path) for path in prior_sources[role]} & candidate_sources:
            raise ValueError(
                f"{role} prior sources matched no frozen candidate source paths"
            )
        explicit = set()
        if role in configured_exclusions:
            explicit = _path_values(configured_exclusions[role], role)
            unmatched = explicit - candidate_files - candidate_sources
            if unmatched:
                raise ValueError(
                    f"{role} exclusion input contains {len(unmatched)} paths absent from candidates"
                )
        excluded = candidates[
            candidates.source_filepath.astype(str).map(
                lambda value: str(Path(value).expanduser().resolve()) in history | explicit
            )
            | candidates.filepath.astype(str).map(
                lambda value: str(Path(value).expanduser().resolve()) in explicit
            )
        ]
        exclusion_file = output / f"exclude_{role}_candidates.parquet"
        excluded[["filepath"]].drop_duplicates().to_parquet(exclusion_file, index=False)
        remaining = len(candidate_files - {
            str(Path(value).expanduser().resolve()) for value in excluded.filepath.astype(str)
        })
        excluded_candidate_crops[role] = int(excluded.filepath.nunique())
        excluded_source_images[role] = int(
            candidates.loc[excluded.index, "source_filepath"].astype(str).map(
                lambda value: str(Path(value).expanduser().resolve())
            ).nunique()
        )
        role_status[role] = {
            "status": "READY" if remaining else "EXHAUSTED",
            "query_count": len(rows), "candidate_count": len(candidate_files),
            "excluded_count": int(excluded.filepath.nunique()),
            "remaining_candidate_count": remaining,
            "exclusion_manifest": str(exclusion_file.resolve()),
            "history_source_count": len(history & candidate_sources),
            "explicit_exclusion_count": len(explicit),
        }
        if not remaining:
            continue
        embedded = output / f"{role}_query_embeddings.parquet"
        (output / f"embed_{role}_queries.yaml").write_text(
            yaml.safe_dump(_embedding_spec(policy, parquet, embedded), sort_keys=False)
        )
        if role == "real":
            factor = real_factor or (
                round_robin_default if strategy == "round_robin_similarity"
                else int(routing["real_mine_factor_min"])
            )
            if not int(routing["real_mine_factor_min"]) <= factor <= int(routing["real_mine_factor_max"]):
                raise ValueError("real factor is outside the frozen policy range")
            strict_count = sum(row["reason"] == "fn" for row in rows)
            near = pd.DataFrame(row for row in rows if row["reason"] == "near_miss_fp")
            near_target = (0 if near.empty else sum(
                min(len(group) * int(routing["near_miss_real_factor"]),
                    int(routing["near_miss_real_cap_per_pocket"]))
                for _, group in near.groupby("pocket")
            ))
            targets[role] = {"fn": strict_count * factor, "near_miss_fp": near_target}
        else:
            targets[role] = {"background_fp": len(rows) * int(routing["clean_factor"])}
        desired = sum(targets[role].values())
        if strategy == "max_similarity":
            requested[role] = min(
                remaining, desired * int(policy["retrieval"]["candidate_overfetch"])
            )
            mining = {
                "source_path": str(candidate_root / f"{role}_candidate_embeddings.parquet"),
                "target_path": str(embedded), "output_dir": str(output / f"mine_{role}"),
                "desired_unique_count": requested[role], "allocation_policy": "global",
                "distance_metric": "cosine",
                "candidate_expansion_factor": int(policy["retrieval"]["candidate_overfetch"]),
            }
            if role_status[role]["excluded_count"]:
                mining["exclude_path"] = str(exclusion_file.resolve())
            (output / f"mine_{role}.yaml").write_text(
                yaml.safe_dump(mining, sort_keys=False)
            )
        else:
            # Round-robin depth is governed by round_robin_refill_overfetch.
            requested[role] = min(remaining, desired)
    enabled = [role for role, evidence in role_status.items() if evidence["status"] == "READY"]
    synthesis_pending = bool(policy.get("synthesis", {}).get("enabled")) and any(
        strict.gap_type.astype(str).str.upper().eq("FN")
    )
    report = {"status": "COMPLETE", "iteration": iteration, "query_counts": counts,
              "enabled_roles": enabled, "role_status": role_status,
              "converged": not enabled and not synthesis_pending,
              "synthesis_pending": synthesis_pending,
              "preprocessing_profile": profile,
              "admission_targets": targets, "requested_crop_counts": requested,
              "excluded_candidate_crops": excluded_candidate_crops,
              "excluded_source_images": excluded_source_images,
              "warnings": warnings, "selection_strategy": strategy}
    _json(output / "query_manifest.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    candidate = sub.add_parser("candidates")
    candidate.add_argument("--policy", type=Path, required=True)
    candidate.add_argument("--output-dir", type=Path, required=True)
    query = sub.add_parser("queries")
    query.add_argument("--policy", type=Path, required=True)
    query.add_argument("--strict-gaps", type=Path, required=True)
    query.add_argument("--loose-gaps", type=Path, required=True)
    query.add_argument("--iteration", type=int, required=True)
    query.add_argument("--output-dir", type=Path, required=True)
    query.add_argument("--candidate-root", type=Path, required=True)
    query.add_argument("--real-factor", type=int)
    query.add_argument("--previous-coco", type=Path)
    query.add_argument("--real-exclusions", type=Path)
    query.add_argument("--clean-exclusions", type=Path)
    args = parser.parse_args()
    result = (candidates(args.policy.resolve(), args.output_dir.resolve()) if args.command == "candidates"
              else queries(args.policy.resolve(), args.strict_gaps.resolve(), args.loose_gaps.resolve(),
                           args.iteration, args.output_dir.resolve(), args.candidate_root.resolve(),
                           args.real_factor,
                           args.previous_coco.resolve() if args.previous_coco else None,
                           {role: getattr(args, f"{role}_exclusions").resolve()
                            for role in ("real", "clean")
                            if getattr(args, f"{role}_exclusions", None)}))
    for warning in result.get("warnings", []):
        print(f"WARNING: {warning['message']}", file=sys.stderr)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
