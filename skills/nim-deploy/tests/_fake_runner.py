# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A fake runner for test_runners.py: a CLI wrapper `fakecli up|down` (not a script; see
runners/__init__.py for the contract)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nd_common as nd


def required_inputs() -> tuple[str, ...]:
    return ("model", "gpus", "port")


def container_name(inputs: dict[str, Any]) -> str:
    return f"fake-{str(inputs['model']).rsplit('/', 1)[-1]}"


def deny_rules() -> list[dict[str, str]]:
    reason = "fakecli nuke removes everything. Fix: ask the user."
    return [{"id": "fakecli-nuke", "pattern": "^fakecli{F} nuke( |$)", "reason": reason}]


def ready_url(inputs: dict[str, Any]) -> str:
    return f"http://localhost:{inputs['port']}"


def render(inputs: dict[str, Any], run_dir: Path) -> list[Path]:
    path = run_dir / "fake-plan.txt"
    nd.write_atomic(
        path, f"up {inputs['model']} on {inputs['gpus']} gpu(s), port {inputs['port']}\n"
    )
    return [path]


def _up(model: Any, port: Any) -> list[str]:
    return ["fakecli", "up", str(model), "--port", str(port)]


def apply_command(state: dict[str, Any]) -> list[str]:
    return _up(nd.input_value(state, "model"), nd.input_value(state, "port"))


def rollback_command(state: dict[str, Any]) -> list[str]:
    return ["fakecli", "down", str(nd.input_value(state, "model"))]


def guard_forms(approval: Any) -> list[list[str]]:
    record = approval.record
    return [_up(record["model"], record["port"]), ["fakecli", "down", str(record["model"])]]
