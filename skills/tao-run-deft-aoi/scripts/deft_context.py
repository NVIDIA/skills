#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Print the durable DEFT policy and deterministic next-stage decision."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


AFTER = {
    None: "train",
    "train": "evaluate",
    "rca": "routing",
    "routing": "anomalygen",
    "anomalygen_finetune": "anomalygen",
    "anomalygen": "data_mining",
    "data_mining": "data_merge",
    "data_merge": "train",
}


def _next_stage(state: dict[str, Any]) -> tuple[str, str]:
    iterations = state.get("iterations", {})
    current = int(state.get("current_iteration", 0))
    label = "baseline" if not iterations else (f"iter{current}" if current else "baseline")
    if state.get("status") == "complete":
        return label, "complete"
    if state.get("status") == "failed":
        return label, "halt"
    phase = iterations.get(label, {}) if isinstance(iterations, dict) else {}
    completed = phase.get("stage_completed") if isinstance(phase, dict) else None
    if completed == "evaluate":
        metric = phase.get("metric_result", {})
        if metric.get("passed") is True:
            return label, "finalize"
        if label != "baseline" and current >= int(state.get("max_iterations", 0)):
            return label, "finalize"
        return label, "rca"
    next_stage = AFTER.get(completed, "train")
    if completed == "rca":
        label = f"iter{current + 1}"
    return label, next_stage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=pathlib.Path)
    parser.add_argument("--stage", help="Fail if this is not the recorded next stage")
    args = parser.parse_args(argv)
    try:
        state = json.loads(args.state.expanduser().read_text())
        label, next_stage = _next_stage(state)
        policy = state.get("execution_policy", {})
        print(json.dumps({
            "status": state.get("status"),
            "iteration": label,
            "next_stage": next_stage,
            "network_mode": policy.get("network_mode", "legacy-unset"),
            "python_executable": policy.get("python_executable"),
        }, sort_keys=True))
        if args.stage and args.stage != next_stage:
            raise ValueError(
                f"requested stage {args.stage!r}; durable next_stage is {next_stage!r}"
            )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"deft_context: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
