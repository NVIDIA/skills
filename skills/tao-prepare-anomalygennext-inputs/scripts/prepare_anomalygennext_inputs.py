#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze FN masks and embedding inputs for AnomalyGenNext preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
IDENTITY_COLUMNS = {
    "dataset_id", "texture_id", "defect_class", "anomaly_type", "fn_mask_source"
}


class MaskEligibilityError(ValueError):
    """A mask-specific failure that makes only one FN ineligible."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stable_id(*values: Any) -> str:
    text = "\0".join(str(value) for value in values)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path.resolve() for path in root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def _defect_specs(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("defect_type") or row.get("anomaly_type") or row.get("name") or "").strip()
        if not name or name in result:
            raise ValueError("defect_spec rows need unique anomaly_type values")
        if row.get("spatial_dependency") == "text" and not str(
            row.get("roi_prompt_defect_location") or ""
        ).strip():
            raise ValueError(f"text-routed defect lacks roi_prompt_defect_location: {name}")
        result[name] = row
    return result


def _recipe_types(path: Path) -> set[str]:
    value = yaml.safe_load(path.read_text())
    rows = value.get("anomaly_types") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"recipe lacks anomaly_types: {path}")
    return {f"{row[0]}+{row[1]}" for row in rows if isinstance(row, list) and len(row) == 2}


def _foreground(mask_path: Path) -> np.ndarray:
    try:
        with Image.open(mask_path) as image:
            mask = np.asarray(image)
    except OSError as error:
        raise MaskEligibilityError("invalid_mask", f"unreadable mask: {mask_path}") from error
    if mask.ndim == 2:
        return mask > 0
    if mask.ndim == 3 and mask.shape[2] >= 1:
        return np.any(mask[..., :3] > 0, axis=2)
    raise MaskEligibilityError("invalid_mask", f"unsupported mask shape {mask.shape}: {mask_path}")


def _mask_metadata(foreground: np.ndarray, source: Path) -> dict[str, Any]:
    if foreground.ndim != 2:
        raise MaskEligibilityError("invalid_mask", f"mask must be two-dimensional: {source}")
    pixels = int(np.count_nonzero(foreground))
    if pixels == 0:
        raise MaskEligibilityError("empty_mask", f"mask foreground is empty: {source}")
    if pixels == foreground.size:
        raise MaskEligibilityError("full_frame_mask", f"mask foreground fills the frame: {source}")
    ys, xs = np.nonzero(foreground)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    if (x2 - x1, y2 - y1) == (foreground.shape[1], foreground.shape[0]):
        raise MaskEligibilityError(
            "full_tight_extent_mask", f"mask tight extent fills the frame: {source}"
        )
    return {"canvas_width": int(foreground.shape[1]), "canvas_height": int(foreground.shape[0]),
            "foreground_bbox": [x1, y1, x2, y2], "foreground_pixels": pixels}


def _write_binary_mask(foreground: np.ndarray, source: Path, output: Path) -> dict[str, Any]:
    metadata = _mask_metadata(foreground, source)
    binary = np.where(foreground, 255, 0).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary).save(output)
    return metadata


def _isolate_mask(mask_path: Path, image_path: Path, bbox: Any, output: Path) -> dict[str, Any]:
    with Image.open(image_path) as image:
        canvas = (image.width, image.height)
    try:
        foreground = _foreground(mask_path)
    except MaskEligibilityError as error:
        raise MaskEligibilityError("invalid_fn_mask", str(error)) from error
    if foreground.shape != (canvas[1], canvas[0]):
        raise MaskEligibilityError("dimension_mismatch", f"mask/image dimensions differ: {mask_path}")
    try:
        _mask_metadata(foreground, mask_path)
    except MaskEligibilityError as error:
        reason = {
            "empty_mask": "empty_fn_mask",
            "full_frame_mask": "full_frame_fn_mask",
            "full_tight_extent_mask": "full_tight_extent_fn_mask",
        }.get(error.reason, "invalid_fn_mask")
        raise MaskEligibilityError(reason, str(error)) from error
    box = np.asarray(bbox, dtype=float).reshape(-1)
    if box.size != 4 or not np.isfinite(box).all():
        raise MaskEligibilityError("invalid_bbox", f"invalid bbox: {bbox!r}")
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    x2, y2 = min(canvas[0], int(np.ceil(x2))), min(canvas[1], int(np.ceil(y2)))
    if x2 <= x1 or y2 <= y1:
        raise MaskEligibilityError("invalid_bbox", f"empty clipped bbox: {bbox!r}")
    isolated = np.zeros_like(foreground)
    isolated[y1:y2, x1:x2] = foreground[y1:y2, x1:x2]
    if not np.any(isolated):
        raise MaskEligibilityError("bbox_mask_mismatch", f"FN bbox contains no mask pixels: {mask_path}")
    return _write_binary_mask(isolated, mask_path, output)


def _ordered_donors(candidates: list[Path], used: set[Path], seed: int, fn_id: str) -> list[Path]:
    if not candidates:
        return []
    unused = [path for path in candidates if path not in used]
    preferred = unused or candidates
    rng = random.Random(int(_stable_id(seed, fn_id), 16))
    start = rng.randrange(len(preferred))
    ordered = preferred[start:] + preferred[:start]
    return ordered + [path for path in candidates if path not in ordered]


def _select(rows: pd.DataFrame, selection: dict[str, Any]) -> pd.DataFrame:
    datasets = [str(value) for value in selection.get("datasets", [])]
    eligible = rows[rows.dataset_id.astype(str).isin(datasets)].copy()
    eligible = eligible.sort_values(["dataset_id", "image_id", "filepath", "fn_id"])
    mode = selection.get("mode", "all_eligible")
    if mode == "all_eligible":
        selected = eligible
    elif mode == "per_dataset":
        split, count = str(selection["split"]), int(selection.get("per_dataset", 1))
        selected = pd.concat(
            [eligible[(eligible.dataset_id == name) & (eligible.split == split)].head(count)
             for name in datasets], ignore_index=True
        )
        if any(len(eligible[(eligible.dataset_id == name) & (eligible.split == split)]) < count
               for name in datasets):
            raise ValueError("not enough eligible FNs for per_dataset selection")
    else:
        raise ValueError(f"unsupported selection.mode: {mode}")
    if selected.empty:
        raise ValueError("selection produced no eligible false negatives")
    return selected.reset_index(drop=True)


def prepare(config_path: Path, output: Path,
            pool_dataset_root: Path | None = None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    gaps = pd.read_parquet(Path(config["gap_parquet"]).expanduser().resolve())
    required = {"image_id", "filepath", "gap_type", "bbox", "class", "split"} | IDENTITY_COLUMNS
    missing = sorted(required - set(gaps.columns))
    if missing:
        raise ValueError(f"gap parquet lacks normalized columns: {missing}")
    gaps = gaps[gaps.gap_type.astype(str).str.upper().eq("FN")].copy()
    specs = _defect_specs(Path(config["defect_spec"]).expanduser().resolve())
    pool_root = (pool_dataset_root or Path(config["pool_dataset_root"])).expanduser().resolve()
    if not pool_root.is_dir():
        raise FileNotFoundError(f"pool dataset root is missing: {pool_root}")
    config["pool_dataset_root"] = str(pool_root)
    datasets = config.get("datasets") or {}
    recipe_types: dict[str, set[str]] = {}
    for name, dataset in datasets.items():
        checkpoint = Path(dataset["checkpoint"]).expanduser().resolve()
        recipe = Path(dataset["recipe"]).expanduser().resolve()
        if not checkpoint.is_file() or not recipe.is_file():
            raise FileNotFoundError(f"dataset {name} checkpoint/recipe is missing")
        recipe_types[str(name)] = _recipe_types(recipe)

    normalized: list[dict[str, Any]] = []
    for row in gaps.to_dict("records"):
        dataset, texture = str(row["dataset_id"]), str(row["texture_id"])
        defect, anomaly = str(row["defect_class"]), str(row["anomaly_type"])
        if dataset not in datasets or anomaly != f"{texture}+{defect}":
            continue
        image = Path(str(row["filepath"])).expanduser().resolve()
        mask = Path(str(row["fn_mask_source"])).expanduser().resolve()
        clean_dir = pool_root / texture / "clean_image"
        sampled_dir = pool_root / texture / "mask" / defect
        if anomaly not in specs or anomaly not in recipe_types[dataset]:
            continue
        if not image.is_file():
            continue
        bbox_json = json.dumps(np.asarray(row["bbox"]).reshape(-1).tolist(), separators=(",", ":"))
        normalized.append({**row, "filepath": str(image), "fn_mask_source": str(mask),
                           "clean_dir": str(clean_dir), "sampled_dir": str(sampled_dir),
                           "bbox_json": bbox_json,
                           "fn_id": "fn-" + _stable_id(dataset, row["image_id"], image, bbox_json)})
    if not normalized:
        raise ValueError("no eligible normalized false negatives")
    selected = _select(pd.DataFrame(normalized), config["selection"])

    root = output / "prepared_anomalygennext_inputs"
    manifests, specs_dir = output / "manifests", output / "specs"
    masks_root = root / "source_masks"
    for directory in (manifests, specs_dir, masks_root):
        directory.mkdir(parents=True, exist_ok=True)
    (root / "filtering_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    mask_rows, query_rows, clean_rows, prepared_rows, skipped_rows = [], [], [], [], []
    clean_seen: set[tuple[str, str]] = set()
    sampled_used: dict[str, set[Path]] = defaultdict(set)
    seed = int(config["selection"].get("mask_sample_seed", 42))
    for order, row in selected.iterrows():
        fn_id, anomaly = str(row.fn_id), str(row.anomaly_type)
        clean_images = _images(Path(row.clean_dir))
        if not clean_images:
            skipped_rows.append({"fn_id": fn_id, "dataset_id": str(row.dataset_id),
                                 "image_id": str(row.image_id),
                                 "reason": "no_clean_reference_images",
                                 "detail": f"no clean reference images under {row.clean_dir}"})
            continue
        mask_dir = masks_root / fn_id
        isolated = mask_dir / f"{fn_id}__fn_mask.png"
        try:
            isolated_metadata = _isolate_mask(
                Path(row.fn_mask_source), Path(row.filepath), row.bbox, isolated
            )
        except MaskEligibilityError as error:
            skipped_rows.append({"fn_id": fn_id, "dataset_id": str(row.dataset_id),
                                 "image_id": str(row.image_id), "reason": error.reason,
                                 "detail": str(error)})
            shutil.rmtree(mask_dir, ignore_errors=True)
            continue
        candidates = _images(Path(row.sampled_dir))
        sampled_output = mask_dir / f"{fn_id}__same_type_sampled_mask.png"
        sampled = None
        donor_failures: Counter[str] = Counter()
        for candidate in _ordered_donors(candidates, sampled_used[anomaly], seed, fn_id):
            try:
                sampled_metadata = _write_binary_mask(
                    _foreground(candidate), candidate, sampled_output
                )
            except MaskEligibilityError as error:
                donor_failures[error.reason] += 1
                continue
            sampled = candidate
            break
        if sampled is None:
            skipped_rows.append({"fn_id": fn_id, "dataset_id": str(row.dataset_id),
                                 "image_id": str(row.image_id),
                                 "reason": "no_valid_same_type_donor",
                                 "detail": dict(sorted(donor_failures.items()))})
            shutil.rmtree(mask_dir, ignore_errors=True)
            continue
        sampled_used[anomaly].add(sampled)
        prepared_rows.append(row.to_dict())
        for branch, path, metadata in (
            ("fn_mask", isolated, isolated_metadata),
            ("same_type_sampled_mask", sampled_output, sampled_metadata),
        ):
            mask_rows.append({"fn_id": fn_id, "query_order": order, "dataset_id": row.dataset_id,
                              "anomaly_type": anomaly, "branch": branch, "mask_path": str(path),
                              **metadata})
        query_rows.append({"filepath": row.filepath, "fn_id": fn_id, "query_order": order,
                           "dataset_id": row.dataset_id, "pool_key": row.texture_id,
                           "anomaly_type": anomaly, "od_category": row["class"]})
        for clean in clean_images:
            key = (str(clean), anomaly)
            if key not in clean_seen:
                clean_seen.add(key)
                clean_rows.append({"filepath": str(clean), "dataset_id": row.dataset_id,
                                   "pool_key": row.texture_id,
                                   "anomaly_type_eligibility": anomaly})
    skip_counts = dict(sorted(Counter(row["reason"] for row in skipped_rows).items()))
    clean_reference_skips = skip_counts.get("no_clean_reference_images", 0)
    warnings = ([{
        "code": "no_clean_reference_images",
        "fn_count": clean_reference_skips,
        "message": (f"Skipped {clean_reference_skips} routed false negatives because their "
                    "AnomalyGen clean-reference pools contain no images"),
    }] if clean_reference_skips else [])
    contract = {"status": "COMPLETE" if prepared_rows else "SKIPPED",
                "reason": "" if prepared_rows else "no_eligible_false_negatives",
                "source_tag": config.get("source_tag", "user_provided"),
                "selection_candidate_fn_count": len(selected),
                "selected_fn_count": len(prepared_rows),
                "eligible_fn_count": len(prepared_rows),
                "eligible_fn_ids": [str(row["fn_id"]) for row in prepared_rows],
                "skipped_fn_count": len(skipped_rows), "skip_counts": skip_counts,
                "skipped_fns": skipped_rows, "clean_image_count": len(clean_rows),
                "source_mask_count": len(mask_rows), "warnings": warnings,
                "training_pool_mutated": False}
    pool_mount = {"type": "folder", "compute_path": str(pool_root), "read_only": True}
    contract["downstream_inputs"] = {
        "clean_embeddings": {"pool_dataset_root": pool_mount},
        "run_amp": {"pool_dataset_root": pool_mount},
    }
    if not prepared_rows:
        _write_json(root / "input_contract.json", contract)
        return contract

    pd.DataFrame(prepared_rows).to_parquet(manifests / "fn_queries.parquet", index=False)
    query_frame = pd.DataFrame(query_rows)
    query_frame.to_parquet(manifests / "selected_fn_queries.parquet", index=False)
    query_frame.drop_duplicates("filepath").to_parquet(
        manifests / "fn_embedding_inputs.parquet", index=False
    )
    pd.DataFrame(mask_rows).to_parquet(manifests / "mask_selection.parquet", index=False)
    clean_frame = pd.DataFrame(clean_rows)
    clean_frame.to_parquet(manifests / "clean_pool.parquet", index=False)
    embedding = config["embedding"]
    for name, input_path in (("clean", manifests / "clean_pool.parquet"),
                             ("fn", manifests / "fn_embedding_inputs.parquet")):
        spec = {"input_parquet": str(input_path),
                "output_parquet": str(output / "embeddings" / f"{name}_embeddings.parquet"),
                "model": embedding["model"], "model_path": embedding["model_path"],
                "model_config_path": "", "batch_size": int(embedding.get("batch_size", 64))}
        (specs_dir / f"{name}_embeddings.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    _write_json(root / "input_contract.json", contract)
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pool-dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    contract = prepare(Path(args.config).resolve(), Path(args.output_dir).resolve(),
                       args.pool_dataset_root.resolve())
    for warning in contract.get("warnings", []):
        print(f"WARNING: {warning['message']}", file=sys.stderr)
    print(json.dumps(contract, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
