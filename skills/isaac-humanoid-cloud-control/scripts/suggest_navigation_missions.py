#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Suggest occupancy-safe navigation poses beside live Isaac Sim objects.

Usage:
    python3 suggest_navigation_missions.py --inventory FILE --map MAP_YAML [OPTIONS]

Arguments:
    Inventory input and --map are required; clearance, robot pose, count, and JSON
    output are optional.

Output:
    Writes ranked approach poses or a machine-readable suggestion object to stdout.

Exit codes:
    0 means suggestions were produced; 1 means input failed; 2 means none were safe.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image


MISSION_LABELS = {
    "shelf": "walk next to shelf",
    "pallet": "inspect pallet",
    "box": "inspect stored boxes",
    "traffic_cone": "walk to traffic cone",
    "forklift": "walk next to forklift",
    "workbench": "walk to workbench",
    "table": "walk to table",
    "dock": "walk to charging dock",
    "door": "walk to doorway",
}


def _yaml_scalar(text: str, key: str, default: str | None = None) -> str:
    match = re.search(
        rf"(?m)^\s*{re.escape(key)}\s*:\s*['\"]?([^'\"#\r\n]+)", text
    )
    if match:
        return match.group(1).strip()
    if default is not None:
        return default
    raise ValueError(f"map YAML is missing {key!r}")


def _parse_origin(raw: str) -> tuple[float, float, float]:
    values = [value.strip() for value in raw.strip("[] ").split(",")]
    if len(values) != 3:
        raise ValueError("map YAML origin must contain three values")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _load_map(map_yaml: Path) -> dict[str, Any]:
    text = map_yaml.read_text(encoding="utf-8")
    image_value = _yaml_scalar(text, "image")
    image_path = Path(image_value).expanduser()
    if not image_path.is_absolute():
        image_path = map_yaml.parent / image_path
    image = Image.open(image_path).convert("L")
    return {
        "image": image,
        "resolution": float(_yaml_scalar(text, "resolution")),
        "origin": _parse_origin(_yaml_scalar(text, "origin")),
        "negate": int(_yaml_scalar(text, "negate", "0")),
        "free_thresh": float(_yaml_scalar(text, "free_thresh", "0.196")),
    }


def _load_inventory(path: str | None, inline: str | None) -> dict[str, Any]:
    if bool(path) == bool(inline):
        raise ValueError("provide exactly one of --inventory or --inventory-json")
    if path:
        return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return json.loads(inline or "{}")


def _world_to_pixel(
    x: float, y: float, width: int, height: int, resolution: float, origin: tuple[float, float, float]
) -> tuple[int, int]:
    column = round((x - origin[0]) / resolution)
    row = round(height - 1 - (y - origin[1]) / resolution)
    return column, row


def _is_free(pixel: int, negate: int, free_thresh: float) -> bool:
    normalized = pixel / 255.0
    occupancy = normalized if negate else 1.0 - normalized
    return occupancy <= free_thresh


def _has_clearance(
    pixels: Any,
    width: int,
    height: int,
    column: int,
    row: int,
    resolution: float,
    radius: float,
    negate: int,
    free_thresh: float,
) -> bool:
    cells = math.ceil(radius / resolution)
    if (
        column - cells < 0
        or row - cells < 0
        or column + cells >= width
        or row + cells >= height
    ):
        return False
    for y_index in range(row - cells, row + cells + 1):
        for x_index in range(column - cells, column + cells + 1):
            if math.hypot(x_index - column, y_index - row) * resolution > radius:
                continue
            if not _is_free(pixels[x_index, y_index], negate, free_thresh):
                return False
    return True


def _approach_candidates(obj: dict[str, Any], stand_off: float) -> list[tuple[float, float]]:
    center = obj.get("center_m", [])
    size = obj.get("size_m", [])
    if len(center) < 2 or len(size) < 2:
        return []
    x, y = float(center[0]), float(center[1])
    half_x = max(0.0, float(size[0]) * 0.5)
    half_y = max(0.0, float(size[1]) * 0.5)
    return [
        (x + half_x + stand_off, y),
        (x - half_x - stand_off, y),
        (x, y + half_y + stand_off),
        (x, y - half_y - stand_off),
    ]


def _suggestions(
    inventory: dict[str, Any],
    map_data: dict[str, Any],
    stand_off: float,
    footprint_radius: float,
    robot_x: float | None,
    robot_y: float | None,
) -> list[dict[str, Any]]:
    image: Image.Image = map_data["image"]
    width, height = image.size
    pixels = image.load()
    resolution = map_data["resolution"]
    origin = map_data["origin"]
    negate = map_data["negate"]
    free_thresh = map_data["free_thresh"]
    results = []

    for obj in inventory.get("objects", []):
        category = str(obj.get("category", ""))
        if category not in MISSION_LABELS:
            continue
        center = obj.get("center_m", [])
        if len(center) < 2:
            continue
        object_x, object_y = float(center[0]), float(center[1])
        valid = []
        for x, y in _approach_candidates(obj, stand_off):
            column, row = _world_to_pixel(x, y, width, height, resolution, origin)
            if not (0 <= column < width and 0 <= row < height):
                continue
            if not _has_clearance(
                pixels,
                width,
                height,
                column,
                row,
                resolution,
                footprint_radius,
                negate,
                free_thresh,
            ):
                continue
            travel_distance = (
                math.hypot(x - robot_x, y - robot_y)
                if robot_x is not None and robot_y is not None
                else 0.0
            )
            valid.append(
                {
                    "x": x,
                    "y": y,
                    "theta": math.atan2(object_y - y, object_x - x),
                    "travel_distance_m": travel_distance,
                }
            )
        if not valid:
            continue
        pose = min(valid, key=lambda candidate: candidate["travel_distance_m"])
        results.append(
            {
                "mission": MISSION_LABELS[category],
                "category": category,
                "object_name": obj.get("name") or Path(str(obj.get("prim_path", "object"))).name,
                "prim_path": obj.get("prim_path"),
                "classification_source": obj.get("classification_source"),
                "object_center_m": [object_x, object_y],
                "suggested_pose": pose,
                "validated_occupancy_clearance_m": footprint_radius,
                "nav2_reachability_validated": False,
            }
        )

    return sorted(
        results,
        key=lambda item: (
            item["suggested_pose"]["travel_distance_m"],
            item["category"],
            str(item["prim_path"]),
        ),
    )


def _select_diverse(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Prefer one nearby mission per object category, then fill by distance."""
    selected = []
    selected_ids = set()
    seen_categories = set()
    for item in items:
        category = item["category"]
        if category in seen_categories:
            continue
        selected.append(item)
        selected_ids.add(id(item))
        seen_categories.add(category)
        if len(selected) >= limit:
            return selected
    for item in items:
        if id(item) in selected_ids:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--inventory", help="JSON inventory emitted by scene_inventory.py")
    source.add_argument("--inventory-json", help="Inline inventory JSON (primarily for tests)")
    parser.add_argument("--map", dest="map_yaml", required=True)
    parser.add_argument("--stand-off", type=float, default=0.75)
    parser.add_argument("--footprint-radius", type=float, default=0.45)
    parser.add_argument("--robot-x", type=float)
    parser.add_argument("--robot-y", type=float)
    parser.add_argument("--max-suggestions", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    if args.stand_off <= args.footprint_radius:
        parser.error("--stand-off must be greater than --footprint-radius")
    if (args.robot_x is None) != (args.robot_y is None):
        parser.error("provide --robot-x and --robot-y together")
    if args.max_suggestions <= 0:
        parser.error("--max-suggestions must be positive")

    try:
        inventory = _load_inventory(args.inventory, args.inventory_json)
        map_data = _load_map(Path(args.map_yaml).expanduser())
        candidates = _suggestions(
            inventory,
            map_data,
            args.stand_off,
            args.footprint_radius,
            args.robot_x,
            args.robot_y,
        )
        suggestions = _select_diverse(candidates, args.max_suggestions)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Scene mission suggestion failed: {exc}", file=sys.stderr)
        return 1

    payload = {
        "schema_version": 1,
        "stage_identifier": inventory.get("stage_identifier"),
        "suggestions": suggestions,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if not suggestions:
            print("No occupancy-safe scene-object mission suggestions were found.")
            return 2
        print("Scene-based navigation suggestions (Nav2 validation still required):")
        for index, item in enumerate(suggestions, start=1):
            pose = item["suggested_pose"]
            print(
                f"{index}. {item['mission']} [{item['object_name']}] at "
                f"({pose['x']:.3f}, {pose['y']:.3f}, yaw={pose['theta']:.3f}); "
                f"source={item['classification_source']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
