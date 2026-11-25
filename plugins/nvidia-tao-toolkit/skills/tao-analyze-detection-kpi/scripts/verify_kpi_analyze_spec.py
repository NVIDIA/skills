#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a TAO Data Services detection KPI-analyze spec.

Fill `assets/default_kpi_analyze.yaml` — it carries every default this stage
needs — then run this against the result. The template is the only place a
default value lives, so there is nothing that can disagree with it.

Checks the spec against what `analytics kpi_analyze` actually requires:

- `data.input_format` is `KITTI` or `COCO`, uppercase. The same container's
  `gap_analysis object_detection` takes these lowercase.
- every `data.kpi_sources` entry carries all three of `image_dir`,
  `ground_truth_ann_path` and `inference_ann_path`, each absolute and present.
- no `image_dir` ends in `/`. The reported `Sequence Name` is
  `image_dir.split('/')[-2]`, so a trailing slash silently reports the wrong one.
- `data.mapping` exists and is a file.
- thresholds are within [0, 1] and `num_recall_points` is at least 1.

Creates `results_dir` if absent, so a container launched straight after this has
somewhere to write.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import yaml

VALID_FORMATS = {"KITTI", "COCO"}
VALID_PLATFORMS = {"local", "wandb"}
SOURCE_KEYS = ("image_dir", "ground_truth_ann_path", "inference_ann_path")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def validate_config(config: dict[str, Any]) -> None:
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("data must be a mapping.")

    fmt = data.get("input_format")
    if not isinstance(fmt, str) or fmt not in VALID_FORMATS:
        raise ValueError(f"data.input_format must be one of {sorted(VALID_FORMATS)} (uppercase).")

    sources = data.get("kpi_sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("data.kpi_sources must contain at least one source.")
    for idx, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"data.kpi_sources[{idx}] must be a mapping.")
        for key in SOURCE_KEYS:
            value = source.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"data.kpi_sources[{idx}].{key} is required.")
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                raise ValueError(f"data.kpi_sources[{idx}].{key} must be absolute: {path}")
            if not path.exists():
                raise FileNotFoundError(f"data.kpi_sources[{idx}].{key} does not exist: {path}")
        # `Sequence Name` is image_dir.split('/')[-2]; a trailing slash shifts the pick.
        if str(source["image_dir"]).endswith("/"):
            raise ValueError(
                f"data.kpi_sources[{idx}].image_dir must not end with '/': the reported "
                "Sequence Name is derived from the second-to-last path component."
            )

    mapping = data.get("mapping")
    if not isinstance(mapping, str) or not mapping.strip():
        raise ValueError("data.mapping is required.")
    mapping_path = Path(str(mapping)).expanduser()
    if not mapping_path.is_absolute():
        raise ValueError(f"data.mapping must be absolute: {mapping_path}")
    if not mapping_path.is_file():
        raise FileNotFoundError(f"data.mapping does not exist or is not a file: {mapping_path}")

    results_dir = config.get("results_dir")
    if not isinstance(results_dir, str) or not results_dir.strip():
        raise ValueError("results_dir is required.")
    out = Path(str(results_dir)).expanduser()
    if not out.is_absolute():
        raise ValueError(f"results_dir must be absolute: {out}")
    out.mkdir(parents=True, exist_ok=True)

    visualize = config.get("visualize", {})
    if not isinstance(visualize, dict):
        raise ValueError("visualize must be a mapping.")
    platform = visualize.get("platform", "local")
    if not isinstance(platform, str) or platform not in VALID_PLATFORMS:
        raise ValueError(f"visualize.platform must be one of {sorted(VALID_PLATFORMS)}.")
    tag = visualize.get("tag")
    if tag is not None and not isinstance(tag, str):
        raise ValueError("visualize.tag must be a string or null.")

    kpi = config.get("kpi", {})
    if not isinstance(kpi, dict):
        raise ValueError("kpi must be a mapping.")
    for key in ("iou_threshold", "conf_threshold"):
        value = kpi.get(key, 0.5)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"kpi.{key} must be a finite number between 0.0 and 1.0.")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"kpi.{key} must be between 0.0 and 1.0.")
        kpi[key] = float(value)
    recall_points = kpi.get("num_recall_points", 101)
    if isinstance(recall_points, bool) or not isinstance(recall_points, int) or recall_points < 1:
        raise ValueError("kpi.num_recall_points must be an integer of at least 1.")
    for key in ("filter", "is_internal"):
        value = kpi.get(key, False)
        if not isinstance(value, bool):
            raise ValueError(f"kpi.{key} must be true or false.")
    ignore_sqwidth = kpi.get("ignore_sqwidth", 0)
    if (
        isinstance(ignore_sqwidth, bool)
        or not isinstance(ignore_sqwidth, (int, float))
        or not math.isfinite(ignore_sqwidth)
        or ignore_sqwidth < 0
    ):
        raise ValueError("kpi.ignore_sqwidth must be a finite non-negative number.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, help="Path to kpi-analyze YAML.")
    return parser.parse_args()


def main() -> int:
    try:
        spec = Path(parse_args().spec).expanduser().resolve()
        config = load_yaml(spec)
        validate_config(config)
        kpi = config.get("kpi") or {}
        print(f"OK: kpi-analyze spec is valid: {spec}")
        print(f"Sources: {len(config['data']['kpi_sources'])} | format: {config['data']['input_format']}")
        print(f"conf_threshold={kpi.get('conf_threshold')} "
              f"num_recall_points={kpi.get('num_recall_points')} "
              f"ignore_sqwidth={kpi.get('ignore_sqwidth')}")
        print(f"Expected output: {config['results_dir']}/kpi_calc.csv")
        out_dir = Path(str(config["results_dir"])).expanduser().resolve()
        if spec.parent != out_dir:
            print(f"WARNING: the spec is outside results_dir ({spec.parent} vs {out_dir}). "
                  "The run does not copy it, so the finished artifacts will carry no record of "
                  f"the settings that produced them. Author it at {out_dir / spec.name} instead.",
                  file=sys.stderr)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
