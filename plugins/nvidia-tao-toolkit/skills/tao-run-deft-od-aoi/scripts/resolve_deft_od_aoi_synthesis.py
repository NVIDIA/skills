#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve synthesis routes or emit one-time AnomalyGenNext fine-tune requests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


FINETUNE_INPUTS = ("dataset_root", "validation_testcase", "base_checkpoint",
                   "vae_path", "checkpoint_root", "result_handoff")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(policy_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    policy = yaml.safe_load(policy_path.read_text())
    synthesis = policy.get("synthesis", {})
    if not synthesis.get("enabled"):
        raise ValueError("synthesis is disabled")
    output.mkdir(parents=True)
    requests, resolved = [], {}
    for name, route in synthesis.get("routes", {}).items():
        checkpoint, recipe = Path(str(route.get("checkpoint") or "")), Path(str(route.get("recipe") or ""))
        if checkpoint.is_file() and recipe.is_file():
            resolved[name] = {"checkpoint": str(checkpoint.resolve()), "recipe": str(recipe.resolve())}
            continue
        finetune = route.get("finetune") or {}
        missing = [key for key in FINETUNE_INPUTS if not str(finetune.get(key) or "").strip()]
        if missing:
            raise ValueError(f"route {name} lacks checkpoint/recipe and finetune fields: {missing}")
        handoff = Path(str(finetune["result_handoff"])).expanduser()
        if not handoff.is_file():
            request = {"dataset_name": name, **{key: str(Path(str(finetune[key])).expanduser().resolve())
                                                for key in FINETUNE_INPUTS[:-1]},
                       "result_handoff": str(handoff.resolve()),
                       "defect_spec": str(Path(str(finetune.get("defect_spec")
                                                       or synthesis["defect_spec"])).resolve())}
            if finetune.get("recipe_template"):
                request["recipe_template"] = str(Path(finetune["recipe_template"]).resolve())
            for key, value in request.items():
                if key not in {"dataset_name", "result_handoff"} and not Path(value).exists():
                    raise ValueError(f"route {name} fine-tune input is missing: {key}={value}")
            requests.append(request)
            continue
        value = json.loads(handoff.read_text())
        if value.get("status") != "COMPLETE":
            raise ValueError(f"route {name} fine-tune handoff is not COMPLETE")
        checkpoint, recipe = Path(str(value.get("checkpoint") or "")), Path(str(value.get("recipe") or ""))
        if (not checkpoint.is_file() or not recipe.is_file()
                or _sha(checkpoint) != value.get("checkpoint_sha256")
                or _sha(recipe) != value.get("recipe_sha256")):
            raise ValueError(f"route {name} fine-tune handoff failed artifact verification")
        if value.get("dataset_name") != name or not value.get("anomaly_types"):
            raise ValueError(f"route {name} fine-tune handoff identity does not match")
        resolved[name] = {"checkpoint": str(checkpoint.resolve()), "recipe": str(recipe.resolve()),
                          "training_handoff": str(handoff.resolve())}
    if requests:
        report = {"status": "NEEDS_TRAINING", "requests": requests,
                  "message": "Run tao-finetune-anomalygennext once per request, then resolve again."}
        (output / "finetune_requests.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    policy["synthesis"]["routes"] = resolved
    resolved_policy = output / "resolved_synthesis_policy.yaml"
    resolved_policy.write_text(yaml.safe_dump(policy, sort_keys=False))
    report = {"status": "COMPLETE", "resolved_policy": str(resolved_policy.resolve()),
              "routes": resolved}
    (output / "synthesis_resolution.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = resolve(args.policy.resolve(), args.output_dir.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
