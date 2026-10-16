#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image-quality and visual-duplicate admission for round robin."""

from __future__ import annotations

import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deft_od_aoi_round_robin_contract import (
    ADMISSION_INDEX_WIDTH,
    DCT_KEEP,
    DCT_SIZE,
)

_POSITIONS = np.arange(DCT_SIZE)
_BASIS = np.cos(
    np.pi * (2 * _POSITIONS[None, :] + 1) * _POSITIONS[:, None] / (2 * DCT_SIZE)
)
_DEFAULT_POLICY = {
    "duplicate_global_cosine": 0.995,
    "duplicate_defect_cosine": 0.985,
    "duplicate_position_delta": 0.04,
    "clean_duplicate_global_cosine": 0.999,
    "clean_cluster_cosine": 0.97,
    "clean_cluster_minimum_cap": 2,
    "clean_cluster_quota_divisor": 4,
    "duplicate_gt_iou": 0.9,
    "minimum_box_area_px": 64,
    "maximum_box_aspect": 25.0,
}


def _dct(image: Image.Image) -> np.ndarray:
    array = np.asarray(
        image.convert("L").resize((DCT_SIZE, DCT_SIZE), Image.Resampling.BILINEAR),
        dtype=np.float64,
    )
    vector = (_BASIS @ array @ _BASIS.T)[:DCT_KEEP, :DCT_KEEP].ravel()
    vector[0] = 0.0
    vector -= vector.mean()
    norm = np.linalg.norm(vector)
    return vector / norm if norm else np.zeros(DCT_KEEP * DCT_KEEP)


def _signature(path: str, boxes: list[list[float]]) -> np.ndarray | None:
    try:
        with Image.open(path) as image:
            global_signature = _dct(image)
            if not boxes:
                return np.concatenate(
                    [global_signature, global_signature, [-1.0, -1.0, 0.0, 0.0]]
                )
            xyxy = [(x, y, x + width, y + height) for x, y, width, height in boxes]
            x0, y0 = min(box[0] for box in xyxy), min(box[1] for box in xyxy)
            x1, y1 = max(box[2] for box in xyxy), max(box[3] for box in xyxy)
            pad_x, pad_y = 0.2 * (x1 - x0), 0.2 * (y1 - y0)
            crop = image.crop((
                max(0, x0 - pad_x), max(0, y0 - pad_y),
                min(image.width, x1 + pad_x), min(image.height, y1 + pad_y),
            ))
            defect = (
                _dct(crop) if crop.width >= 8 and crop.height >= 8 else global_signature
            )
            position = [
                (x0 + x1) / (2 * max(1, image.width)),
                (y0 + y1) / (2 * max(1, image.height)),
                (x1 - x0) / max(1, image.width),
                (y1 - y0) / max(1, image.height),
            ]
            return np.concatenate([global_signature, defect, position])
    except OSError:
        return None


def _iou(left: list[float], right: list[float]) -> float:
    x, y, width, height = left
    a, b, other_width, other_height = right
    overlap = (
        max(0.0, min(x + width, a + other_width) - max(x, a))
        * max(0.0, min(y + height, b + other_height) - max(y, b))
    )
    union = width * height + other_width * other_height - overlap
    return overlap / union if union > 0 else 0.0


def _screen(path: str, boxes: list[list[float]], policy: dict[str, Any],
            report: Counter[str]) -> list[list[float]]:
    try:
        with Image.open(path) as image:
            image_width, image_height = image.size
    except OSError:
        report["rejected_unreadable_image"] += 1
        return []
    kept = []
    for raw in boxes:
        if len(raw) != 4:
            report["boxes_quarantined"] += 1
            continue
        x, y, width, height = map(float, raw)
        box = [
            max(0.0, x), max(0.0, y),
            min(float(image_width), x + width) - max(0.0, x),
            min(float(image_height), y + height) - max(0.0, y),
        ]
        if all(np.isfinite(raw)) and any(
            not math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-9)
            for left, right in zip(box, (x, y, width, height), strict=True)
        ):
            report["boxes_clipped_to_image"] += 1
        invalid = (
            not np.isfinite(raw).all()
            or box[2] <= 0 or box[3] <= 0
            or box[2] * box[3] < float(policy["minimum_box_area_px"])
            or max(box[2], box[3]) / min(box[2], box[3])
            > float(policy["maximum_box_aspect"])
            or any(
                _iou(box, previous) > float(policy["duplicate_gt_iou"])
                for previous in kept
            )
        )
        if invalid:
            report["boxes_quarantined"] += 1
        else:
            kept.append(box)
    return kept


class RoundRobinAdmission:
    """Screen and deduplicate ranked parents while preserving a cumulative index."""

    def __init__(self, policy: dict[str, Any], previous_index: Path | None) -> None:
        self.policy = {**_DEFAULT_POLICY, **(policy.get("admission") or {})}
        self.pending: list[np.ndarray] = []
        self.report: Counter[str] = Counter()
        if previous_index is not None and not previous_index.is_file():
            raise FileNotFoundError(f"previous admission index is missing: {previous_index}")
        self.index = (
            np.load(previous_index)
            if previous_index
            else np.zeros((0, ADMISSION_INDEX_WIDTH))
        )
        if self.index.ndim != 2 or self.index.shape[1] != ADMISSION_INDEX_WIDTH:
            raise ValueError(f"invalid admission index shape {self.index.shape}")
        self.clean_clusters: list[list[Any]] = []
        self.clean_cluster_cap = 0

    def begin_branch(self, quota: int, clean: bool) -> None:
        if clean:
            self.clean_clusters = []
            self.clean_cluster_cap = max(
                int(self.policy["clean_cluster_minimum_cap"]),
                quota // int(self.policy["clean_cluster_quota_divisor"]),
            )

    def _pool(self) -> np.ndarray:
        if not self.pending:
            return self.index
        return np.vstack([self.index, *(row[None, :] for row in self.pending)])

    def admit(self, ranked: list[dict[str, Any]], quota: int, clean: bool,
              record: Callable[[str], dict[str, Any]]) -> list[dict[str, Any]]:
        admitted: list[dict[str, Any]] = []
        if clean and self.clean_cluster_cap < 1:
            self.begin_branch(quota, clean=True)
        for candidate in ranked:
            if len(admitted) >= quota:
                break
            self.report["checked"] += 1
            source = str(candidate["source_filepath"])
            boxes = record(source)["boxes"]
            if not clean:
                boxes = _screen(source, boxes, self.policy, self.report)
                if not boxes:
                    self.report["rejected_no_valid_boxes"] += 1
                    continue
                candidate = {**candidate, "admission_boxes": boxes}
            signature = _signature(source, boxes)
            if signature is None:
                self.report["rejected_unreadable_image"] += 1
                continue
            pool = self._pool()
            duplicate = False
            if len(pool):
                width = DCT_KEEP * DCT_KEEP
                global_scores = pool[:, :width] @ signature[:width]
                previous_clean = pool[:, 2 * width] < 0
                if clean:
                    duplicate = bool(np.any(
                        previous_clean & (
                            global_scores
                            > float(self.policy["clean_duplicate_global_cosine"])
                        )
                    ))
                else:
                    defect_scores = pool[:, width:2 * width] @ signature[width:2 * width]
                    positions = np.abs(
                        pool[:, 2 * width:2 * width + 2]
                        - signature[2 * width:2 * width + 2]
                    ).max(axis=1)
                    duplicate = bool(np.any(
                        (~previous_clean)
                        & (global_scores > float(self.policy["duplicate_global_cosine"]))
                        & (defect_scores > float(self.policy["duplicate_defect_cosine"]))
                        & (positions < float(self.policy["duplicate_position_delta"]))
                    ))
            if duplicate:
                self.report["rejected_duplicate"] += 1
                continue
            if clean:
                global_signature = signature[:DCT_KEEP * DCT_KEEP]
                cluster = next((item for item in self.clean_clusters
                                if float(item[0] @ global_signature)
                                > float(self.policy["clean_cluster_cosine"])), None)
                if cluster and cluster[1] >= self.clean_cluster_cap:
                    self.report["rejected_cluster_cap"] += 1
                    continue
                if cluster:
                    cluster[1] += 1
                else:
                    self.clean_clusters.append([global_signature, 1])
            admitted.append(candidate)
            self.pending.append(signature)
        self.report["admitted"] += len(admitted)
        return admitted

    def save(self, path: Path) -> None:
        temporary = Path(str(path) + ".tmp.npy")
        np.save(temporary, self._pool())
        os.replace(temporary, path)
