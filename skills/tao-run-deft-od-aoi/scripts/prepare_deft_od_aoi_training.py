#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write direct RT-DETR training and deterministic probe specs for DEFT AOI."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import yaml


def _write(path: Path, value: Any) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def _set(value: dict[str, Any], dotted: str, replacement: Any) -> None:
    node = value
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = replacement


def _epochs(training: dict[str, Any], iteration: int, size: int) -> int:
    if iteration <= 2:
        return int(training["early_iterations_epochs"])
    value = round(int(training["early_iterations_epochs"]) * math.sqrt(
        int(training["adaptive_reference_images"]) / size
    ))
    return min(int(training["adaptive_max_epochs"]),
               max(int(training["adaptive_min_epochs"]), value))


def _base(policy: dict[str, Any], train_coco: Path, train_images: Path,
          results: Path, epochs: int) -> dict[str, Any]:
    training, kpi = policy["training"], policy["sources"]["kpi"]
    return {"results_dir": str(results),
            "model": {"backbone": "resnet_50", "train_backbone": True,
                      "num_feature_levels": 3, "return_interm_indices": [1, 2, 3]},
            "dataset": {"train_data_sources": [{"image_dir": str(train_images),
                                                  "json_file": str(train_coco)}],
                        "val_data_sources": {"image_dir": str(kpi["images"]),
                                             "json_file": str(kpi["coco"])},
                        "num_classes": 2, "eval_class_ids": [1],
                        "batch_size": int(training["batch_size"]),
                        "workers": int(training["workers"]),
                        "remap_mscoco_category": False},
            "train": {"num_gpus": int(training["num_gpus"]),
                      "gpu_ids": list(range(int(training["num_gpus"]))),
                      "num_epochs": epochs,
                      "checkpoint_interval": int(training["checkpoint_interval"]),
                      "validation_interval": int(training["validation_interval"]),
                      "pretrained_model_path": str(Path(policy["base_checkpoint"]).resolve()),
                      "optim": {"lr": float(training["base_lr"]),
                                "lr_backbone": float(training["backbone_lr"])}}}


def prepare(policy_path: Path, iteration: int, train_coco: Path, train_images: Path,
            results: Path, output: Path, history_path: Path | None,
            incumbent_path: Path | None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if not train_coco.is_file() or not train_images.is_dir():
        raise ValueError("training COCO/images are missing")
    policy = yaml.safe_load(policy_path.read_text())
    if not 1 <= iteration <= int(policy["max_iterations"]):
        raise ValueError("iteration is outside the frozen policy")
    size = len(json.loads(train_coco.read_text()).get("images", []))
    if size < 1:
        raise ValueError("training COCO has no images")
    training = policy["training"]
    epochs = _epochs(training, iteration, size)
    output.mkdir(parents=True)
    template = _base(policy, train_coco.resolve(), train_images.resolve(), results / "main", epochs)
    probes = (bool(training["probes_enabled"])
              and iteration >= int(training.get("probes_start_iteration", 3)))
    manifest: dict[str, Any] = {"status": "COMPLETE", "iteration": iteration,
                                "train_size": size, "planned_epochs": epochs,
                                "probes_enabled": probes, "probes": []}
    _write(output / "main_template.yaml", template)
    if not probes:
        _write(output / "train.yaml", template)
    else:
        history = json.loads(history_path.read_text()) if history_path else []
        if not isinstance(history, list):
            raise ValueError("history must be a JSON list")
        previous = int(history[-1]["train_size"]) if history else size
        growth = size / max(1, previous)
        incumbent = {"train.optim.lr": float(training["base_lr"]),
                     "train.optim.lr_backbone": float(training["backbone_lr"])}
        if incumbent_path:
            value = json.loads(incumbent_path.read_text())
            if not isinstance(value, dict):
                raise ValueError("incumbent must be a JSON mapping")
            incumbent.update(value)
        scaled = dict(incumbent)
        if growth > float(training["growth_high"]):
            scaled["train.optim.lr"] *= float(training["high_growth_lr_factor"])
        elif growth < float(training["growth_low"]):
            scaled["train.optim.lr"] *= float(training["low_growth_lr_factor"])
        rng = random.Random(int(training["jitter_seed_base"]) + iteration)
        jitter = dict(incumbent)
        jitter["train.optim.lr"] *= rng.uniform(*map(float, training["jitter_lr_factor"]))
        jitter["train.optim.lr_backbone"] *= rng.uniform(
            *map(float, training["jitter_backbone_lr_factor"])
        )
        for index, (name, overrides) in enumerate(
                (("incumbent", incumbent), ("scaled", scaled), ("jitter", jitter))):
            spec = copy.deepcopy(template)
            for key, value in overrides.items():
                _set(spec, key, value)
            spec["results_dir"] = str(results / "probes" / f"p{index}")
            spec["train"]["num_epochs"] = int(training["probe_epochs"])
            spec["train"]["checkpoint_interval"] = int(training["probe_epochs"])
            _write(output / f"probe{index}.yaml", spec)
            manifest["probes"].append({"index": index, "name": name, "overrides": overrides})
        manifest["growth"] = growth
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--train-coco", type=Path, required=True)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--incumbent", type=Path)
    args = parser.parse_args()
    report = prepare(args.policy.resolve(), args.iteration, args.train_coco.resolve(),
                     args.train_images.resolve(), args.results_root.resolve(),
                     args.output_dir.resolve(), args.history.resolve() if args.history else None,
                     args.incumbent.resolve() if args.incumbent else None)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
