#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Append one stage event to the DEFT CR ITS mining loop log."""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path
from typing import Any

from workflow_common import atomic_write_json


VALID_STATUSES = {"ok", "error", "skipped"}
VALID_STAGES = {
    "validate_workflow",
    "initialize_workflow",
    "prepare_cosmos_reason_model",
    "baseline_evaluate",
    "prepare_cosmos_embed_inference",
    "cosmos_embed",
    "convert_embeddings",
    "gap_analysis",
    "prepare_nearest_neighbor_mining",
    "mine_nearest_neighbors",
    "record_mined_paths",
    "prepare_cosmos_reason_train",
    "train",
    "evaluate",
    "cleanup_cosmos_reason_training",
    "loop_stop",
}


ITERATION_LABEL = re.compile(r"^iter_(\d+)$")


def read_valid_events(log_path: Path, *, warn: bool = True) -> list[dict[str, Any]]:
    """Read valid stage events, warning and continuing past malformed lines."""
    if not log_path.exists():
        return []
    events: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("expected a JSON object")
                seq = event.get("seq")
                if not isinstance(seq, int) or seq < 1:
                    raise ValueError(f"invalid seq: {seq!r}")
            except (json.JSONDecodeError, ValueError) as exc:
                if warn:
                    print(f"WARNING: ignoring malformed log line {log_path}:{line_number}: {exc}", file=sys.stderr)
                continue
            events.append(event)
    return events


def next_seq(log_path: Path, *, warn: bool = False) -> int:
    """Return one greater than the maximum valid sequence number in the log."""
    events = read_valid_events(log_path, warn=warn)
    return max((event["seq"] for event in events), default=0) + 1


def rebuild_state(state_path: Path, log_path: Path) -> dict[str, Any]:
    """Rebuild dynamic workflow state from valid append-only log events."""
    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict):
        raise ValueError(f"{state_path}: expected a JSON object")

    events = sorted(read_valid_events(log_path, warn=False), key=lambda event: event["seq"])
    state["iterations"] = {}
    state["baseline_results_json"] = None
    state["current_iteration"] = 0
    state["status"] = "initialized"
    state.pop("last_event", None)
    latest_by_stage: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        iter_label = event.get("iter")
        stage = event.get("stage")
        if not isinstance(iter_label, str) or not isinstance(stage, str):
            continue
        latest_by_stage[(iter_label, stage)] = event
        iteration_match = ITERATION_LABEL.match(iter_label)
        if iteration_match:
            iteration = int(iteration_match.group(1))
            state["current_iteration"] = max(state["current_iteration"], iteration)
            state["iterations"].setdefault(iter_label, {})[stage] = event
        if stage == "baseline_evaluate" and event.get("status") == "ok":
            for artifact in event.get("artifacts", []):
                if isinstance(artifact, str) and artifact.endswith("results.json"):
                    state["baseline_results_json"] = artifact
                    break
        state["last_event"] = event

    if events:
        last_event = events[-1]
        completed_stop = any(
            event.get("stage") == "loop_stop" and event.get("status") in {"ok", "skipped"}
            for event in latest_by_stage.values()
        )
        if completed_stop:
            state["status"] = "complete"
        elif any(event.get("status") == "error" for event in latest_by_stage.values()):
            state["status"] = "failed"
        else:
            state["status"] = "running"
        state["updated_at"] = last_event.get("ts")
    atomic_write_json(state_path, state)
    return state


def append_stage(
    log_path: Path,
    *,
    iter_label: str,
    stage: str,
    status: str,
    summary: str,
    duration_sec: int,
    artifacts: list[str],
) -> dict[str, Any]:
    """Append a validated stage event and refresh the sibling state snapshot."""
    if stage not in VALID_STAGES:
        raise ValueError(f"stage must be one of {sorted(VALID_STAGES)}, got {stage!r}")
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}, got {status!r}")
    if duration_sec < 0:
        raise ValueError("duration_sec must be >= 0")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "seq": next_seq(log_path),
        "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "iter": iter_label,
        "stage": stage,
        "status": status,
        "summary": summary,
        "duration_sec": duration_sec,
    }
    if artifacts:
        entry["artifacts"] = artifacts
    needs_newline = False
    if log_path.exists() and log_path.stat().st_size:
        with log_path.open("rb") as handle:
            handle.seek(-1, 2)
            needs_newline = handle.read(1) != b"\n"
    with log_path.open("a", encoding="utf-8") as handle:
        if needs_newline:
            handle.write("\n")
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    state_path = log_path.parent / "deft_state.json"
    if state_path.is_file():
        rebuild_state(state_path, log_path)
    return entry


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", required=True, type=Path)
    parser.add_argument("--iter-label", required=True)
    parser.add_argument("--stage", required=True, choices=sorted(VALID_STAGES))
    parser.add_argument("--status", required=True, choices=sorted(VALID_STATUSES))
    parser.add_argument("--summary", required=True)
    parser.add_argument("--duration-sec", required=True, type=int)
    parser.add_argument("--artifact", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    append_stage(
        args.log_path,
        iter_label=args.iter_label,
        stage=args.stage,
        status=args.status,
        summary=args.summary,
        duration_sec=args.duration_sec,
        artifacts=args.artifact,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
