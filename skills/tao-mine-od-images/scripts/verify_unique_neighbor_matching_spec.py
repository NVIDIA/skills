#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a TAO Data Services TMM unique-neighbor-matching spec.

Fill `assets/default_unique_neighbor_matching.yaml` — it carries every default
this stage needs — then run this against the result. The template is the only
place a default value lives, so there is nothing that can disagree with it.

Checks the spec against what `tmm unique_neighbor_matching` actually requires:

- `source_path`, `target_path`, `output_dir` and `desired_unique_count` are
  present; the paths are absolute and the two parquets exist.
- `allocation_policy` and `distance_metric` are values the miner accepts.
- `detection_format` is set whenever a detection file is, since the miner cannot
  infer it.
- `class_stratified` has everything it needs: a `rare_class_list`, both detection
  files, and a format. Without them the policy silently degrades to a global
  match, which mines the wrong images rather than failing.

It also reports the budget and policy, because the mined parquet is a list of
filepaths and records nothing about why those files were chosen.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

VALID_METRICS = {"euclidean", "cosine", "manhattan"}
VALID_POLICIES = {"global", "class_stratified"}
VALID_FORMATS = {"coco", "kitti"}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def validate_config(config: dict[str, Any]) -> None:
    required = ("source_path", "target_path", "output_dir", "desired_unique_count")
    missing = [key for key in required if config.get(key) is None]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    for key in ("source_path", "target_path", "output_dir"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty string.")
    for key in ("source_path", "target_path"):
        p = Path(config[key]).expanduser()
        if not p.is_absolute():
            raise ValueError(f"{key} must be absolute: {p}")
        if not p.exists():
            raise FileNotFoundError(f"{key} does not exist: {p}")

    out = Path(config["output_dir"]).expanduser()
    if not out.is_absolute():
        raise ValueError(f"output_dir must be absolute: {out}")
    out.mkdir(parents=True, exist_ok=True)

    count = config["desired_unique_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("desired_unique_count must be an integer of at least 1.")
    config["desired_unique_count"] = count

    policy = config.get("allocation_policy", "global")
    if not isinstance(policy, str) or policy not in VALID_POLICIES:
        raise ValueError(f"allocation_policy must be one of {sorted(VALID_POLICIES)}.")
    config["allocation_policy"] = policy

    metric = config.get("distance_metric", "euclidean")
    if not isinstance(metric, str) or metric not in VALID_METRICS:
        raise ValueError(f"distance_metric must be one of {sorted(VALID_METRICS)}.")
    config["distance_metric"] = metric

    det_src = config.get("source_detection_file")
    det_tgt = config.get("target_detection_file")
    det_fmt = config.get("detection_format")
    for key, value in (
        ("source_detection_file", det_src),
        ("target_detection_file", det_tgt),
        ("exclude_path", config.get("exclude_path")),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{key} must be a non-empty string or null.")
    if det_fmt is not None and not isinstance(det_fmt, str):
        raise ValueError("detection_format must be a string or null.")
    if (det_src or det_tgt) and not det_fmt:
        raise ValueError("detection_format (coco or kitti) is required when a detection file is set.")
    if det_fmt and det_fmt not in VALID_FORMATS:
        raise ValueError(f"detection_format must be one of {sorted(VALID_FORMATS)}.")

    if policy == "class_stratified":
        if not config.get("rare_class_list"):
            raise ValueError("rare_class_list is required when allocation_policy is class_stratified.")
        if not det_src or not det_tgt:
            raise ValueError("source_detection_file and target_detection_file are required for class_stratified.")
        if not det_fmt:
            raise ValueError("detection_format is required for class_stratified.")

    expansion = config.get("candidate_expansion_factor", 5)
    if isinstance(expansion, bool) or not isinstance(expansion, int) or expansion < 1:
        raise ValueError("candidate_expansion_factor must be an integer of at least 1.")
    for key in (
        "source_embedding_column",
        "target_embedding_column",
        "source_filepath_column",
        "target_filepath_column",
    ):
        value = config.get(key, "embedding" if "embedding" in key else "filepath")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string.")
    rare = config.get("rare_class_list", "")
    if not isinstance(rare, str):
        raise ValueError("rare_class_list must be a string.")
    for key in ("save_embeddings", "visualize"):
        value = config.get(key, False)
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, help="Path to unique-neighbor-matching YAML.")
    return parser.parse_args()


def main() -> int:
    try:
        spec = Path(parse_args().spec).expanduser().resolve()
        config = load_yaml(spec)
        validate_config(config)
        print(f"OK: unique-neighbor-matching spec is valid: {spec}")
        print(f"budget={config['desired_unique_count']} policy={config['allocation_policy']} "
              f"metric={config['distance_metric']}"
              + (f" rare={config['rare_class_list']}" if config.get("rare_class_list") else ""))
        print(f"Output directory: {config['output_dir']}")
        print("Expected artifacts: final_unique_files.parquet, summary.json")
        out_dir = Path(str(config["output_dir"])).expanduser().resolve()
        if spec.parent != out_dir:
            print(f"WARNING: the spec is outside output_dir ({spec.parent} vs {out_dir}). The "
                  "run does not copy it, so the mined set will carry no record of the budget, "
                  "allocation policy or rare-class list that selected it. Author it at "
                  f"{out_dir / spec.name} instead.", file=sys.stderr)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
