#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inventory navigation-relevant objects in the live Isaac Sim USD stage.

Usage:
    Send this file through isaacsim_send.py to the loopback Isaac Sim server.

Arguments:
    Values arrive through the injected globals documented below.

Output:
    Writes a JSON inventory and reports its path and a compact summary to stdout.

Exit codes:
    The remote runner returns 0 on success and a nonzero status on an exception.

Send this file to an Isaac Sim instance through isaac-sim-remote's
``isaacsim_send.py``. The script does not modify the stage. Injected globals:

``output_path``
    Optional new JSON output path on the simulator host; existing files are
    refused. By default, securely creates a unique temporary file and reports
    its path. An empty string disables file output.
``max_objects``
    Maximum classified objects to emit. Defaults to 200.
"""

if "output_path" not in dir():
    output_path = None
if "max_objects" not in dir():
    max_objects = 200

import json
import math
import tempfile
from pathlib import Path

import omni.usd
from pxr import Usd, UsdGeom


CATEGORY_KEYWORDS = {
    "shelf": ("shelf", "shelving", "rack"),
    "pallet": ("pallet", "palette"),
    "box": ("cardbox", "cardboard", "carton", "crate", "box"),
    "traffic_cone": ("trafficcone", "traffic_cone", "traffic cone", "cone"),
    "forklift": ("forklift",),
    "workbench": ("workbench", "work_bench", "work bench"),
    "table": ("table",),
    "dock": ("chargingdock", "charging_dock", "charging dock", "dock"),
    "door": ("door", "gate"),
}


def _string_value(value):
    try:
        return str(value)
    except Exception:
        return "<unreadable>"


def _semantic_labels(prim):
    labels = {}
    for attribute in prim.GetAttributes():
        name = attribute.GetName()
        lowered = name.lower()
        if "semantic" not in lowered and "label" not in lowered:
            continue
        try:
            value = attribute.Get()
        except Exception:
            continue
        if value is None:
            continue
        rendered = _string_value(value)
        if rendered not in {"", "[]"}:
            labels[name] = rendered
    return labels


def _asset_metadata(prim):
    metadata = []
    for key in ("references", "payload"):
        try:
            value = prim.GetMetadata(key)
        except Exception:
            continue
        if value:
            metadata.append(_string_value(value))
    return metadata


def _classify(prim, labels, assets):
    sources = {
        "semantic": " ".join(labels.values()).lower(),
        "name": prim.GetName().lower(),
        "asset": " ".join(assets).lower(),
    }
    for source_name in ("semantic", "name", "asset"):
        for category, keywords in CATEGORY_KEYWORDS.items():
            text = sources[source_name]
            if any(keyword in text for keyword in keywords):
                return category, source_name
    return None, None


def _current_time_code(stage):
    try:
        import omni.timeline

        seconds = omni.timeline.get_timeline_interface().get_current_time()
        return Usd.TimeCode(seconds * stage.GetTimeCodesPerSecond())
    except Exception:
        return Usd.TimeCode.Default()


def _world_bounds(cache, prim, meters_per_unit):
    try:
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        minimum = aligned.GetMin()
        maximum = aligned.GetMax()
        center = (minimum + maximum) * 0.5
        size = maximum - minimum
        values = [*minimum, *maximum, *center, *size]
        if not all(math.isfinite(float(value)) for value in values):
            return None
        if max(float(value) for value in size) <= 0.0:
            return None
        scale = float(meters_per_unit)
        return {
            "minimum_m": [float(value) * scale for value in minimum],
            "maximum_m": [float(value) * scale for value in maximum],
            "center_m": [float(value) * scale for value in center],
            "size_m": [float(value) * scale for value in size],
        }
    except Exception:
        return None


stage = omni.usd.get_context().get_stage()
if not stage:
    raise RuntimeError("No USD stage is open")

meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
bbox_cache = UsdGeom.BBoxCache(
    _current_time_code(stage), purposes, useExtentsHint=True
)

candidates = []
for prim in stage.Traverse():
    if not prim.IsActive() or prim.IsPseudoRoot():
        continue
    labels = _semantic_labels(prim)
    assets = _asset_metadata(prim)
    category, classification_source = _classify(prim, labels, assets)
    if not category:
        continue
    bounds = _world_bounds(bbox_cache, prim, meters_per_unit)
    if not bounds:
        continue
    candidates.append(
        {
            "prim_path": str(prim.GetPath()),
            "name": prim.GetName(),
            "category": category,
            "classification_source": classification_source,
            "semantic_labels": labels,
            "asset_metadata": assets,
            **bounds,
        }
    )

# Prefer the deepest classified object so a parent such as ``/World/Shelves``
# does not collapse several individually named rack children into one bound.
candidates.sort(key=lambda item: (-item["prim_path"].count("/"), item["prim_path"]))
objects = []
for candidate in candidates:
    path = candidate["prim_path"]
    category = candidate["category"]
    if any(
        category == existing["category"]
        and existing["prim_path"].startswith(path.rstrip("/") + "/")
        for existing in objects
    ):
        continue
    objects.append(candidate)
    if len(objects) >= int(max_objects):
        break

objects.sort(key=lambda item: item["prim_path"])

inventory = {
    "schema_version": 1,
    "stage_identifier": stage.GetRootLayer().identifier,
    "meters_per_unit": float(meters_per_unit),
    "objects": objects,
}

encoded = json.dumps(inventory, indent=2, sort_keys=True)
if output_path is None:
    with tempfile.NamedTemporaryFile(mode="w", prefix="humanoid-inventory-",
                                     suffix=".json", encoding="utf-8", delete=False) as output:
        output.write(encoded + "\n")
        print(f"scene_inventory_path={output.name}")
elif output_path:
    destination = Path(str(output_path)).expanduser()
    with destination.open("x", encoding="utf-8") as output:
        output.write(encoded + "\n")
    print(f"scene_inventory_path={destination}")
print(f"scene_inventory_objects={len(objects)}")
print("SCENE_INVENTORY_JSON=" + json.dumps(inventory, separators=(",", ":")))
