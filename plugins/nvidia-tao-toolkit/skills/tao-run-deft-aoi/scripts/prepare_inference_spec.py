# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare inference handoff artifacts at loop end.

Produces two files under ``${RESULTS_DIR}/`` so downstream inference skills can
consume the trained checkpoint without reading ``deft_state.json`` or the
training spec directly:

- ``best_model.json``                — checkpoint, operating threshold, and customer metric
- ``best_model_inference_spec.yaml`` — a ready-to-run TAO inference spec built
                                       from the training spec used for the best
                                       iteration. Model / dataset config is
                                       copied verbatim so it matches the
                                       checkpoint's architecture exactly.

The consumer fills in only data-path overrides (the CSV + images_dir for their
inference set) and the checkpoint/threshold are already wired in.

Library usage:

    from prepare_inference_spec import prepare
    prepare(results_dir=pathlib.Path("/abs/path/results/run_..."))

CLI usage:

    python scripts/prepare_inference_spec.py --results-dir /abs/path/results/run_...

Why both files: ``best_model.json`` is the small contract (5 fields any
consumer can read); ``best_model_inference_spec.yaml`` is the executable
artifact TAO actually runs. Keeping them in sync is this script's job — never
hand-edit either file.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from copy import deepcopy
from typing import Any

import yaml

from metric_contract import contract_from_state, pick_best, result_from_iteration


def _pick_best(
    state: dict[str, Any]
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return the best completed iteration according to the metric direction."""
    contract = contract_from_state(state)
    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for label, info in state.get("iterations", {}).items():
        if not isinstance(info, dict) or info.get("status") != "complete":
            continue
        result = result_from_iteration(info, contract)
        if result is not None:
            candidates.append((label, info, result))
    # Read-only compatibility for runs created before baseline was normalized
    # under state["iterations"]. New writers must never create this shape.
    if "baseline" in state and isinstance(state["baseline"], dict):
        legacy_result = result_from_iteration(state["baseline"], contract)
        if legacy_result is not None and not any(
            label == "baseline" for label, _, _ in candidates
        ):
            candidates.append(("baseline", state["baseline"], legacy_result))
    if not candidates:
        raise RuntimeError(
            "no completed iteration in deft_state.json has the configured metric — "
            "finish an evaluate commit before preparing the handoff"
        )
    label, info, result = pick_best(candidates, contract)
    return label, info, result, contract


CHECKPOINT_MOUNT = "/model/best.pth"


def _build_inference_spec(
    train_spec: dict[str, Any],
    threshold: float | None,
) -> dict[str, Any]:
    """Transform a training spec into a minimal, runnable inference spec.

    Strips train/evaluate/export blocks. Keeps model + dataset architecture
    verbatim so backbone, lighting layout, image size, difference module, and
    concat type all match the checkpoint. The currently pinned TAO 7.1 runtime
    still rebuilds its criterion during non-training actions, so carry only the
    matching ``train.classify.loss`` compatibility stub. TAO 7.2 runtimes that
    include NVIDIA-TAO/tao-pytorch#107 no longer require it.

    The ``inference.checkpoint`` path is the in-container mount point, not the
    host path — consumers mount ``best_model.json["checkpoint"]`` (host) to
    ``CHECKPOINT_MOUNT`` (container). The training spec's
    ``pretrained_backbone_path`` is already an in-container path and is kept
    verbatim. See ``references/prepare-for-inference.md`` for the mount table.
    """
    spec: dict[str, Any] = {
        "encryption_key": train_spec.get("encryption_key", "tlt_encode"),
        "task": train_spec.get("task", "classify"),
        "results_dir": "",  # CONSUMER: override with your output dir
        # Required by the pinned TAO 7.1 runtime. Remove after the documented
        # baseline includes NVIDIA-TAO/tao-pytorch#107.
        "train": {
            "classify": {
                "loss": (
                    train_spec.get("train", {})
                    .get("classify", {})
                    .get("loss", "ce")
                ),
            },
        },
        "model": deepcopy(train_spec["model"]),
        "dataset": {"classify": deepcopy(train_spec["dataset"]["classify"])},
        "inference": {
            "checkpoint": CHECKPOINT_MOUNT,
            "batch_size": 1,
            "results_dir": "",  # CONSUMER: override with your output dir
        },
    }

    # Use an evaluator-selected operating point when supplied. A custom metric
    # may intentionally evaluate the training spec's existing threshold.
    if threshold is not None:
        spec["model"].setdefault("classify", {})["eval_margin"] = float(threshold)

    # Strip training/evaluation data sources; consumer only needs infer_dataset.
    cls = spec["dataset"]["classify"]
    for k in ("train_dataset", "validation_dataset", "test_dataset"):
        cls.pop(k, None)
    cls["infer_dataset"] = {
        "csv_path": "",       # CONSUMER: path to inference CSV
        "images_dir": "",     # CONSUMER: root of images referenced by CSV
    }
    cls["batch_size"] = 1
    cls["workers"] = 1
    # Disable training-time augmentation for inference.
    aug = cls.get("augmentation_config")
    if isinstance(aug, dict):
        aug["augment"] = False

    return spec


def _resolve_backbone(
    train_spec: dict[str, Any],
    state: dict[str, Any],
) -> tuple[str, str, str, bool]:
    """Resolve the host backbone while preserving the legacy AOI mount."""
    backbone_config = train_spec.get("model", {}).get("backbone", {})
    container_path = backbone_config.get("pretrained_backbone_path")
    backbone_type = str(backbone_config.get("type", ""))
    backbone_frozen = bool(backbone_config.get("freeze_backbone", False))
    if not isinstance(container_path, str) or not container_path.strip():
        raise ValueError(
            "training spec has no model.backbone.pretrained_backbone_path"
        )

    configured = pathlib.Path(container_path)
    if configured.suffix not in {".ckpt", ".pth", ".safetensors"}:
        raise ValueError(f"unsupported pretrained backbone file: {container_path}")

    if configured.is_file():
        host_path = configured
    else:
        backbone_dir = pathlib.Path(state["config"]["backbone_weight_dir"])
        host_path = backbone_dir / configured.name

        # Existing AOI mounts may rename the single staged C-RADIO file between
        # host and container. Require the normalized alias; never substitute an
        # unrelated sole checkpoint. DINO profiles must resolve exactly.
        if not host_path.is_file() and "dinov3" not in backbone_type:
            candidates = [
                path
                for suffix in (".ckpt", ".pth", ".safetensors")
                for path in sorted(backbone_dir.glob(f"*{suffix}"))
                if path.is_file() and path.stat().st_size > 0
            ]
            configured_stem = "".join(
                char for char in configured.stem.casefold() if char.isalnum()
            )
            aliases = [
                path
                for path in candidates
                if "".join(
                    char for char in path.stem.casefold() if char.isalnum()
                ) == configured_stem
            ]
            if len(aliases) == 1:
                host_path = aliases[0]

    if not host_path.is_file() or host_path.stat().st_size <= 0:
        raise FileNotFoundError(
            "the exact pretrained backbone referenced by the training spec is "
            f"not staged: {host_path}"
        )

    return (
        str(host_path.resolve()),
        container_path,
        backbone_type,
        backbone_frozen,
    )


def prepare(results_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """Write best_model.json and best_model_inference_spec.yaml.

    Returns a dict mapping artifact name to written path. Raises if state
    or training spec is missing — the caller should treat those as hard stops.
    """
    state_path = results_dir / "deft_state.json"
    state = json.loads(state_path.read_text())

    iter_label, best, metric_result, metric_contract = _pick_best(state)

    train_spec_path = pathlib.Path(
        best.get("training_spec") or state["config"]["specs_file"]
    )
    if not train_spec_path.exists():
        raise FileNotFoundError(f"training spec not found: {train_spec_path}")
    train_spec = yaml.safe_load(train_spec_path.read_text())

    backbone, backbone_container_path, backbone_type, backbone_frozen = (
        _resolve_backbone(train_spec, state)
    )

    threshold = best.get("threshold")
    handoff = {
        "checkpoint": best["best_ckpt_path"],
        "threshold": threshold,
        "metric_contract": metric_contract,
        "metric_result": metric_result,
        "iteration": iter_label,
        "backbone": backbone,
        "backbone_container_path": backbone_container_path,
        "backbone_type": backbone_type,
        "backbone_frozen": backbone_frozen,
        "images_dir": state["config"]["images_dir"],
        "training_spec": str(train_spec_path),
    }
    if metric_contract["name"] == "far_pct":
        handoff["far_pct"] = metric_result["value"]

    inference_spec = _build_inference_spec(
        train_spec=train_spec,
        threshold=threshold,
    )
    inference_spec["model"]["backbone"][
        "pretrained_backbone_path"
    ] = backbone_container_path

    json_path = results_dir / "best_model.json"
    yaml_path = results_dir / "best_model_inference_spec.yaml"

    json_path.write_text(json.dumps(handoff, indent=2) + "\n")
    yaml_path.write_text(yaml.safe_dump(inference_spec, sort_keys=False))

    return {"best_model_json": json_path, "best_model_inference_spec": yaml_path}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--results-dir",
        type=pathlib.Path,
        required=True,
        help="absolute path to the run results directory (contains deft_state.json)",
    )
    args = p.parse_args()

    written = prepare(args.results_dir)
    for name, path in written.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
