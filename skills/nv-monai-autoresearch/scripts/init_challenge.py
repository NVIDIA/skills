#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Initialize the NV-MONAI-Autoresearch task, dashboard, and trajectory before discovery."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = SKILL_ROOT / "assets" / "challenge_template"
INITIAL_RECORDS = ("challenge.md", "dashboard.md", "trajectory.md")


def load_request(path: Path) -> dict[str, str]:
    """Accept only the two descriptive task fields; no commands or credentials."""
    request = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(request, dict) or set(request) != {"name", "task"}:
        raise ValueError("request must contain exactly the name and task fields")
    for key, value in request.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty string")
        if any(character in value for character in ("\n", "\r", "|")):
            raise ValueError(f"{key} must be a single line without table separators")
    return request


def initial_record(filename: str, request: dict[str, str]) -> str:
    """Keep the v3.5 fields, clearing example results and job rows."""
    text = (TEMPLATE_ROOT / filename).read_text(encoding="utf-8")
    if filename == "dashboard.md":
        start = text.index("## baseline001_")
        end = text.index("## Human Queue Steering")
        text = text[:start] + text[end:]
    values = {
        "MIAgent harness version": "v3.5 workflow; nv-monai-autoresearch skill 0.1.0",
        "challenge name": request["name"],
        "objective/core problem class": request["task"],
        "Challenge Name": request["name"],
        "Run authorization": "not entered",
        "Run state": "planning",
        "Current loop": "planning",
        "Active baseline": "none",
        "Active usage": "no jobs launched",
        "Heartbeat": "not established; no asynchronous work launched",
        "Final freeze readiness": "not checked",
    }
    lines = []
    table_header = False
    gate_table = False
    for line in text.splitlines():
        if not line.startswith("|"):
            table_header = False
            lines.append(line)
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not table_header:
            table_header = True
            gate_table = cells[0] == "Gate"
            lines.append(line)
        elif all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            lines.append(line)
        elif len(cells) == 2:
            value = values.get(cells[0], "not checked" if gate_table else "unknown")
            lines.append(f"| {cells[0]} | {value} |")
        # Wider tables contain illustrative rows; leave their headers intact.
    return "\n".join(lines) + "\n"


def initialize(request_path: Path, output_dir: Path) -> dict:
    request = load_request(request_path)
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("workspace already exists; resume it instead of reinitializing")
    output_dir = output_dir.absolute()
    resolved_output = output_dir.resolve()
    if resolved_output == SKILL_ROOT or SKILL_ROOT in resolved_output.parents:
        # A source checkout may use its ignored challenge/ directory; a skill
        # installation must never receive generated files under assets/scripts.
        source_workspace = SKILL_ROOT / "challenge"
        if source_workspace not in resolved_output.parents or not (SKILL_ROOT / ".git").exists():
            raise ValueError("choose a workspace outside the installed skill")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".nv-monai-autoresearch-init-", dir=output_dir.parent))
    try:
        for filename in INITIAL_RECORDS:
            (staging / filename).write_text(initial_record(filename, request), encoding="utf-8")
        # Exclusive directory creation protects another session's workspace.
        output_dir.mkdir()
        for filename in INITIAL_RECORDS:
            shutil.move(str(staging / filename), str(output_dir / filename))
    finally:
        shutil.rmtree(staging)
    return {
        "skill": "nv-monai-autoresearch",
        "operation": "initialize",
        "status": "initialized",
        "workspace_path": str(output_dir.resolve()),
        "artifacts": list(INITIAL_RECORDS),
        "phase": "understand",
        "research_completed": False,
        "experiments_launched": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="JSON object with name and task strings")
    parser.add_argument("--output-dir", required=True, type=Path, help="New research workspace")
    args = parser.parse_args()
    try:
        result = initialize(args.request, args.output_dir)
    except (OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "skill": "nv-monai-autoresearch",
                    "operation": "initialize",
                    "status": "error",
                    "error": str(error),
                }
            )
        )
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
