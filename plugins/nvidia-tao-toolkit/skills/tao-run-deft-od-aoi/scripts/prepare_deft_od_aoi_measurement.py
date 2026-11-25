#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write RT-DETR KPI/test inference and dual OD gap-analysis specs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_xyxy(value: Any, image: dict[str, Any]) -> tuple[float, float, float, float]:
    """Validate a COCO xywh box against its image dimensions.
    Return KITTI xyxy coordinates without clipping or rewriting ground truth.
    """
    x, y, width, height = map(float, value)
    image_width = float(image.get("width") or 0)
    image_height = float(image.get("height") or 0)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("KPI normalized COCO image dimensions must be positive")
    if width <= 0 or height <= 0:
        raise ValueError("KPI normalized COCO bbox dimensions must be positive")
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError("KPI normalized COCO bbox exceeds image bounds")
    return x, y, x + width, y + height


def _kitti(coco_path: Path, output: Path) -> dict[str, int]:
    coco = json.loads(coco_path.read_text())
    categories = {int(row["id"]): str(row["name"]) for row in coco.get("categories", [])}
    if set(categories.values()) != {"defect"}:
        raise ValueError("KPI COCO must contain only the defect category")
    images = {int(row["id"]): row for row in coco.get("images", [])}
    stems = [Path(str(row["file_name"])).stem for row in images.values()]
    if not images or len(stems) != len(set(stems)):
        raise ValueError("KPI images need unique stems for KITTI projection")
    labels: dict[int, list[str]] = {image_id: [] for image_id in images}
    for annotation in coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id not in images or categories.get(int(annotation["category_id"])) != "defect":
            raise ValueError("KPI annotation references an unknown image/category")
        x1, y1, x2, y2 = _bounded_xyxy(annotation["bbox"], images[image_id])
        labels[image_id].append(
            f"defect 0.0 0 0.0 {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} 0 0 0 0 0 0 0"
        )
    output.mkdir(parents=True)
    for image_id, image in images.items():
        (output / f"{Path(str(image['file_name'])).stem}.txt").write_text(
            "\n".join(labels[image_id]) + ("\n" if labels[image_id] else "")
        )
    return {"images": len(images), "annotations": sum(map(len, labels.values()))}


def _inference(policy: dict[str, Any], images: str, classmap: Path, checkpoint: Path,
               results: Path) -> dict[str, Any]:
    return {"results_dir": str(results), "model": {"backbone": "resnet_50",
            "train_backbone": True, "num_feature_levels": 3,
            "return_interm_indices": [1, 2, 3]},
            "dataset": {"infer_data_sources": {"image_dir": [images],
                                                 "classmap": str(classmap)},
                        "num_classes": 2, "eval_class_ids": [1], "batch_size": 8,
                        "workers": 4, "remap_mscoco_category": False},
            "inference": {"num_gpus": 1, "gpu_ids": [0], "checkpoint": str(checkpoint),
                          "conf_threshold": float(policy["gap"]["inference_confidence"])}}


def _empty_predictions(coco_path: Path, output: Path) -> int:
    images = json.loads(coco_path.read_text()).get("images", [])
    stems = [Path(str(row["file_name"])).stem for row in images]
    if not images or len(stems) != len(set(stems)):
        raise ValueError("held-out images need unique stems for empty predictions")
    output.mkdir(parents=True)
    for stem in stems:
        (output / f"{stem}.txt").touch()
    return len(stems)


def prepare(policy_path: Path, checkpoint: Path, predictions: Path,
            results_root: Path, output: Path, baseline: bool = False) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    policy = yaml.safe_load(policy_path.read_text())
    baseline_mode = str(policy.get("baseline_mode") or "cold_start")
    if baseline_mode not in {"cold_start", "checkpoint"}:
        raise ValueError("baseline_mode must be cold_start or checkpoint")
    kpi, test = policy["sources"]["kpi"], policy["sources"]["test"]
    for role in (kpi, test):
        if not Path(role["images"]).is_dir() or not Path(role["coco"]).is_file():
            raise ValueError("frozen KPI/test role is missing")
    output.mkdir(parents=True)
    gt = output / "kpi_ground_truth_kitti"
    projection = _kitti(Path(kpi["coco"]), gt)
    cold_start = baseline and baseline_mode == "cold_start"
    spec_names = ["gap_loose.yaml", "gap_strict.yaml"]
    gap_predictions = predictions
    test_predictions = results_root / "test/inference/labels"
    if cold_start:
        gap_predictions = output / "cold_start_predictions/kpi"
        test_predictions = output / "cold_start_predictions/test"
        _empty_predictions(Path(kpi["coco"]), gap_predictions)
        test_count = _empty_predictions(Path(test["coco"]), test_predictions)
    else:
        classmap = output / "inference_classmap.txt"
        classmap.write_text("background\ndefect\n")
        _yaml(output / "kpi_inference.yaml",
              _inference(policy, kpi["images"], classmap, checkpoint, results_root / "kpi"))
        _yaml(output / "test_inference.yaml",
              _inference(policy, test["images"], classmap, checkpoint, results_root / "test"))
        spec_names = ["kpi_inference.yaml", "test_inference.yaml", *spec_names]
        test_count = len(json.loads(Path(test["coco"]).read_text()).get("images", []))
    for kind in ("loose", "strict"):
        gap = policy["gap"]
        spec = {"ground_truth_ann_path": str(gt),
                "inference_ann_path": str(gap_predictions),
                "images_dir": str(Path(kpi["images"]).resolve()),
                "results_dir": str((results_root / f"gap_{kind}").resolve()),
                "kpi": f"kpi_{kind}", "input_format": "kitti",
                "iou_threshold": float(gap["match_iou"]),
                "conf_threshold": float(gap[f"{kind}_confidence"]), "min_area": 0,
                "class_mapping": {}, "weak_thresholds": {},
                "default_recall_threshold": 0.0, "default_precision_threshold": 0.0,
                "default_ap50_threshold": 0.0}
        _yaml(output / f"gap_{kind}.yaml", spec)
    report = {"status": "COMPLETE", "checkpoint": str(checkpoint.resolve()),
              "checkpoint_sha256": _sha(checkpoint),
              "baseline": baseline, "baseline_mode": baseline_mode,
              "cold_start": cold_start,
              "kpi_ground_truth": projection,
              "inference_roles": {
                  "kpi": {"expected_images": projection["images"],
                          "predictions": str(gap_predictions.resolve())},
                  "test": {"expected_images": test_count,
                           "predictions": str(test_predictions.resolve())},
              },
              "specs": {name: str((output / name).resolve()) for name in spec_names}}
    (output / "measurement_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kpi-predictions", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    report = prepare(args.policy.resolve(), args.checkpoint.resolve(),
                     args.kpi_predictions.resolve(), args.results_root.resolve(),
                     args.output_dir.resolve(), args.baseline)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
