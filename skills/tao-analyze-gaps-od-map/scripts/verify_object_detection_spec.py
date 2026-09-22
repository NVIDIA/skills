#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a TAO Data Services object-detection gap-analysis spec.

Fill `assets/default_object_detection.yaml` — it carries every default this stage
needs — then run this against the result. The template is the only place a
default value lives, so there is nothing that can disagree with it.

Checks the spec against what `gap_analysis object_detection` actually requires:

- `input_format` is `kitti` or `coco`, **lowercase**. `analytics kpi_analyze` in
  the same container wants these uppercase; the two stages spell it differently
  and both are correct for themselves.
- the three input paths and `results_dir` are present and absolute. KITTI passes
  a directory here, COCO a json file, so both are accepted.
- `weak_thresholds` entries are mappings of metric to a number in [0, 1].
- every gated class is reported, since an image is weak if **any** class falls
  below its gate and one forgotten entry silently changes the selection.

The `default_*_threshold` values get their own report line. TAO DS defaults
`default_ap50_threshold` to 0.5, so a spec that lists some classes and says
nothing else still gates every class it did not list — which inflates the weak
set and the mining budget derived from it. A non-zero fallback is legal and
occasionally wanted, so it warns rather than fails.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import yaml

VALID_FORMATS = {"kitti", "coco"}
PATH_KEYS = ("ground_truth_ann_path", "inference_ann_path", "images_dir")
METRICS = ("ap50", "recall", "precision")
DEFAULT_KEYS = ("default_ap50_threshold", "default_recall_threshold",
                "default_precision_threshold")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def validate_config(config: dict[str, Any]) -> list[str]:
    """Raise on anything that would fail or silently mis-select. Returns warnings."""
    warnings: list[str] = []

    fmt = config.get("input_format")
    if not isinstance(fmt, str) or fmt not in VALID_FORMATS:
        raise ValueError(
            f"input_format must be one of {sorted(VALID_FORMATS)} (lowercase); got {fmt!r}. "
            "It is never inferred from the path."
        )

    for key in PATH_KEYS:
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} is required.")
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            raise ValueError(f"{key} must be absolute: {path}")
        if not path.exists():
            raise FileNotFoundError(f"{key} does not exist: {path}")

    results_dir = config.get("results_dir")
    if not isinstance(results_dir, str) or not results_dir.strip():
        raise ValueError("results_dir is required.")
    out = Path(str(results_dir)).expanduser()
    if not out.is_absolute():
        raise ValueError(f"results_dir must be absolute: {out}")
    out.mkdir(parents=True, exist_ok=True)

    if not isinstance(config.get("kpi"), str) or not config["kpi"].strip():
        raise ValueError("kpi is required — it labels the rows in the emitted parquets.")

    iou = config.get("iou_threshold", 0.5)
    if isinstance(iou, bool) or not isinstance(iou, (int, float)) or not math.isfinite(iou):
        raise ValueError("iou_threshold must be a finite number within (0, 1].")
    if not 0.0 < iou <= 1.0:
        raise ValueError(f"iou_threshold must be within (0, 1]; got {iou}")
    conf = config.get("conf_threshold", 0.0)
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not math.isfinite(conf):
        raise ValueError("conf_threshold must be a finite number within [0, 1].")
    if not 0.0 <= conf <= 1.0:
        raise ValueError(f"conf_threshold must be within [0, 1]; got {conf}")

    min_area = config.get("min_area", 0)
    if (
        isinstance(min_area, bool)
        or not isinstance(min_area, (int, float))
        or not math.isfinite(min_area)
        or min_area < 0
    ):
        raise ValueError("min_area must be a finite non-negative number.")
    class_mapping = config.get("class_mapping", {})
    if not isinstance(class_mapping, dict):
        raise ValueError("class_mapping must be a mapping.")

    thresholds = config.get("weak_thresholds") or {}
    if not isinstance(thresholds, dict):
        raise ValueError("weak_thresholds must be a mapping of class name to metric gates.")
    if not thresholds:
        warnings.append(
            "weak_thresholds is empty, so no class is gated explicitly and every class "
            "falls back to the default_* values below."
        )
    for name, gates in thresholds.items():
        if not isinstance(gates, dict):
            raise ValueError(
                f"weak_thresholds[{name!r}] must be a mapping such as {{ap50: 0.7}}; got {gates!r}"
            )
        for metric, value in gates.items():
            if metric not in METRICS:
                raise ValueError(
                    f"weak_thresholds[{name!r}] has unknown metric {metric!r}; "
                    f"expected one of {list(METRICS)}"
                )
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"weak_thresholds[{name!r}][{metric!r}] must be a finite number."
                )
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"weak_thresholds[{name!r}][{metric!r}] must be within [0, 1]; got {value}"
                )

    defaults: dict[str, float] = {}
    for key in DEFAULT_KEYS:
        value = config.get(key, 0.0)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"{key} must be a finite number within [0, 1].")
        defaults[key] = float(value)
    gated = [key for key, value in defaults.items() if value > 0.0]
    if gated:
        warnings.append(
            f"{', '.join(gated)} is above zero, so every class absent from weak_thresholds is "
            "also gated. An image is weak if ANY class falls below its gate, so a class you "
            "did not list can select images on its own — which inflates the weak set and any "
            "mining budget derived from it. Set these to 0.0 unless that is intended."
        )
    return warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, help="Path to the gap-analysis YAML.")
    return parser.parse_args()


def main() -> int:
    try:
        spec = Path(parse_args().spec).expanduser().resolve()
        config = load_yaml(spec)
        warnings = validate_config(config)
        out = Path(str(config['results_dir'])).expanduser()

        thresholds = config.get("weak_thresholds") or {}
        print(f"OK: gap-analysis spec is valid: {spec}")
        print(f"format: {config['input_format']} | iou={config.get('iou_threshold', 0.5)} "
              f"conf={config.get('conf_threshold', 0.0)} min_area={config.get('min_area', 0)}")
        if thresholds:
            gates = ", ".join(f"{n} {dict(g)}" for n, g in thresholds.items())
            print(f"gated classes ({len(thresholds)}): {gates}")
        print("fallbacks: " + ", ".join(f"{k.replace('default_', '').replace('_threshold', '')}"
                                        f"={config.get(k, 0.0)}" for k in DEFAULT_KEYS))
        print(f"Expected output: {config['results_dir']}/weak_images.parquet")
        # The run writes four artifacts and not the spec, so a spec kept elsewhere
        # leaves the finished results with no record of what selected them.
        if spec.parent != out.resolve():
            warnings.append(
                f"the spec is outside results_dir ({spec.parent} vs {out}). The run does not "
                "copy it, so the finished artifacts will carry no record of the thresholds "
                f"that produced them. Author it at {out / spec.name} instead."
            )
        for w in warnings:
            print(f"WARNING: {w}", file=sys.stderr)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
