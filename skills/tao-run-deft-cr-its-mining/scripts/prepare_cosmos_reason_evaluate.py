#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare baseline or iteration Cosmos Reason evaluation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from workflow_common import (
    absolute_path,
    dump_toml,
    existing_absolute_path,
    load_toml,
    load_yaml,
    path_in_workspace,
    require_mapping,
    require_string,
)


def patch_evaluate_config(
    config: dict[str, Any],
    *,
    results_dir: Path,
    annotation_path: Path,
    media_dir: Path,
    checkpoint_path: str,
) -> dict[str, Any]:
    """Patch run-specific fields in a Cosmos Reason evaluation config."""
    config["results_dir"] = str(results_dir)
    config.setdefault("dataset", {})
    config["dataset"]["annotation_path"] = str(annotation_path)
    config["dataset"]["media_dir"] = str(media_dir)
    config.setdefault("model", {})
    config["model"]["model_name"] = checkpoint_path
    return config


def latest_safetensors_checkpoint(train_dir: Path) -> Path:
    """Return the latest epoch checkpoint from a successfully completed train directory."""
    if not train_dir.is_dir():
        raise NotADirectoryError(f"train directory does not exist: {train_dir}")
    timestamp_re = re.compile(r"^\d{14}$")
    run_dirs = [
        child for child in train_dir.iterdir()
        if child.is_dir() and timestamp_re.match(child.name)
    ]
    if not run_dirs:
        raise FileNotFoundError(f"no timestamped train run directories found under {train_dir}")
    latest_run = max(run_dirs, key=lambda path: path.name)
    safetensors_dir = latest_run / "safetensors"
    if not safetensors_dir.is_dir():
        raise FileNotFoundError(f"safetensors directory not found: {safetensors_dir}")

    epoch_re = re.compile(r"^epoch_(\d+)$")
    epochs: list[tuple[int, Path]] = []
    for child in safetensors_dir.iterdir():
        match = epoch_re.match(child.name)
        if child.is_dir() and match:
            epochs.append((int(match.group(1)), child))
    if not epochs:
        raise FileNotFoundError(f"no epoch_<N> checkpoints found under {safetensors_dir}")
    return max(epochs, key=lambda item: item[0])[1]


def generate_evaluate_toml(
    workspace: Path,
    workflow_yaml: Path,
    run_dir: Path,
    *,
    output_dir: Path,
    checkpoint_path: Path,
) -> Path:
    """Write one evaluation TOML using the selected checkpoint."""
    config = load_yaml(workflow_yaml)
    kpi_dataset = require_mapping(config, "kpi_dataset")
    cosmos_reason = require_mapping(config, "cosmos_reason")
    annotation_path = existing_absolute_path(
        require_string(kpi_dataset, "kpi_dataset.annotations_path"),
        workspace,
        "kpi_dataset.annotations_path",
        "file",
    )
    media_dir = existing_absolute_path(
        require_string(kpi_dataset, "kpi_dataset.media_dir"),
        workspace,
        "kpi_dataset.media_dir",
        "dir",
    )
    base_evaluate_toml = existing_absolute_path(
        require_string(cosmos_reason, "cosmos_reason.base_evaluate_toml"),
        workspace,
        "cosmos_reason.base_evaluate_toml",
        "file",
    )
    path_in_workspace(checkpoint_path, workspace, "evaluation checkpoint")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"evaluation checkpoint does not exist: {checkpoint_path}")
    path_in_workspace(output_dir, run_dir, "evaluation output directory")

    output_path = output_dir / "specs" / "evaluate.toml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patched = patch_evaluate_config(
        load_toml(base_evaluate_toml),
        results_dir=output_dir,
        annotation_path=annotation_path,
        media_dir=media_dir,
        checkpoint_path=str(checkpoint_path),
    )
    output_path.write_text(dump_toml(patched), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--workflow-yaml", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--iteration",
        type=int,
        help="Iteration to evaluate. Omit to prepare baseline evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    """Prepare baseline evaluation or one iteration's evaluation."""
    args = parse_args()
    workspace = absolute_path(args.workspace)
    workflow_yaml = absolute_path(args.workflow_yaml)
    run_dir = absolute_path(args.run_dir)
    if not workspace.is_dir():
        raise NotADirectoryError(f"workspace does not exist: {workspace}")
    if not workflow_yaml.is_file():
        raise FileNotFoundError(f"workflow YAML does not exist: {workflow_yaml}")
    path_in_workspace(workflow_yaml, workspace, "workflow YAML")
    path_in_workspace(run_dir, workspace, "run directory")

    config = load_yaml(workflow_yaml)
    cosmos_reason = require_mapping(config, "cosmos_reason")
    if args.iteration is None:
        output_dir = run_dir / "baseline" / "evaluate"
        checkpoint_path = existing_absolute_path(
            require_string(cosmos_reason, "cosmos_reason.baseline_model_path"),
            workspace,
            "cosmos_reason.baseline_model_path",
            "path",
        )
    else:
        if args.iteration < 1:
            raise ValueError("iteration must be >= 1")
        output_dir = run_dir / f"iter_{args.iteration}" / "evaluate"
        checkpoint_path = latest_safetensors_checkpoint(
            run_dir / f"iter_{args.iteration}" / "train"
        )

    output_path = generate_evaluate_toml(
        workspace,
        workflow_yaml,
        run_dir,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
    )
    print(f"checkpoint: {checkpoint_path}")
    print(f"toml: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
