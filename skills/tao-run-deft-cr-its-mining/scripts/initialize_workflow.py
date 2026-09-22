#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initialize a DEFT CR ITS mining workflow run and its state snapshot."""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import Any

from workflow_common import (
    absolute_path,
    atomic_write_json,
    copy_workflow_yaml_to_run_dir,
    load_yaml,
    path_in_workspace,
    require_mapping,
    workflow_run_dir,
)


def build_state(workspace: Path, workflow_yaml: Path, run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Create the initial workflow state payload."""
    run = require_mapping(config, "run")
    mining = require_mapping(config, "mining")
    cosmos_reason = require_mapping(config, "cosmos_reason")
    return {
        "version": 1,
        "workflow": "tao-run-deft-cr-its-mining",
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "workspace": str(workspace),
        "workflow_yaml": str(workflow_yaml),
        "run_dir": str(run_dir),
        "max_iterations": run["max_iterations"],
        "current_iteration": 0,
        "status": "initialized",
        "embedding_modality": mining["embeddings_modality"],
        "mine_unique_only": bool(mining.get("mine_unique_only", True)),
        "continual_model": bool(cosmos_reason.get("continual_model", False)),
        "baseline_results_json": None,
        "iterations": {},
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--workflow-yaml", required=True, type=Path)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Explicit run directory. Defaults to the directory derived from workflow.yaml.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite the state snapshot; existing loop logs and stage artifacts remain.",
    )
    return parser.parse_args()


def main() -> int:
    """Create run directory, snapshot workflow.yaml, and write deft_state.json."""
    args = parse_args()
    workspace = absolute_path(args.workspace)
    workflow_yaml = absolute_path(args.workflow_yaml)
    if not workspace.is_dir():
        raise NotADirectoryError(f"workspace does not exist: {workspace}")
    if not workflow_yaml.is_file():
        raise FileNotFoundError(f"workflow YAML does not exist: {workflow_yaml}")
    path_in_workspace(workflow_yaml, workspace, "workflow YAML")

    config = load_yaml(workflow_yaml)
    run_dir = absolute_path(args.run_dir) if args.run_dir else workflow_run_dir(config, workspace)
    path_in_workspace(run_dir, workspace, "run directory")
    state_path = run_dir / "deft_state.json"
    if state_path.exists() and not args.force:
        raise FileExistsError(f"{state_path} already exists; pass --force only when restarting")

    copy_workflow_yaml_to_run_dir(workflow_yaml, run_dir)
    atomic_write_json(state_path, build_state(workspace, workflow_yaml, run_dir, config))
    print(f"run_dir: {run_dir}")
    print(f"state_path: {state_path}")
    print(f"workflow_yaml_snapshot: {run_dir / 'workflow.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
