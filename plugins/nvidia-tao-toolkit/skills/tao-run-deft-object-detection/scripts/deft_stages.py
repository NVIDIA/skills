#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared stage contract for the DEFT OD loop.

Single source of truth for phase/stage ordering, required artifacts, and atomic
state I/O. `init_deft_state.py`, `commit_stage.py`, and `audit_deft_run.py` all
import this so the transition table exists once rather than three times.

No CLI.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

PREP_ORDER = ["prep"]
BASELINE_ORDER = ["inference", "kpi_analyze"]
ITER_ORDER = ["gap_analysis", "embed", "mine", "stage", "train", "inference", "kpi_analyze"]
TERMINAL_STAGE = "loop_stop"

ALL_STAGES = sorted(set(PREP_ORDER + BASELINE_ORDER + ITER_ORDER + [TERMINAL_STAGE]))


def derive_rare_classes(pool_counts: dict[str, int],
                        target_classes: list[str]) -> tuple[list[str], str]:
    """Rare target classes from the pool's own annotation counts.

    Below-mean share, not below-median: with one dominant class the median calls
    half the target set rare however lopsided the pool actually is.

    Returns the class list and a human-readable account of the counts it used.
    Only classes the pool actually holds are eligible -- a class with no
    annotations cannot be mined for, so naming it rare would allocate budget that
    nothing can fill.
    """
    backed = {c: pool_counts[c] for c in target_classes if pool_counts.get(c)}
    if not backed:
        return [], "the pool holds no annotations for any target class"
    mean_share = sum(backed.values()) / len(backed)
    rare = sorted(c for c, n in backed.items() if n < mean_share)
    detail = ", ".join(f"{c}={backed[c]}" for c in sorted(backed, key=backed.get))
    return rare, detail


# stage -> {state field: (commit flag, "file"|"dir")}
# Every listed artifact must exist on disk before the stage may be committed.
STAGE_ARTIFACTS: dict[str, dict[str, tuple[str, str]]] = {
    "prep": {
        "pool_odvg_dir": ("--pool-odvg", "dir"),
        "pool_embeddings_parquet": ("--pool-embeddings", "file"),
    },
    "gap_analysis": {
        "weak_images_parquet": ("--weak-images", "file"),
        "gap_report_json": ("--gap-report", "file"),
    },
    "embed": {
        "embeddings_parquet": ("--embeddings-parquet", "file"),
    },
    "mine": {
        "mining_output_parquet": ("--mining-output", "file"),
        "mining_summary_json": ("--mining-summary", "file"),
    },
    "stage": {
        "odvg_jsonl": ("--odvg", "file"),
        "label_map_json": ("--label-map", "file"),
        "staged_images_dir": ("--staged-images-dir", "dir"),
        "exclude_parquet": ("--exclude-parquet", "file"),
    },
    "train": {
        "checkpoint_path": ("--checkpoint", "file"),
        "training_spec": ("--training-spec", "file"),
    },
    "inference": {
        "inference_labels_dir": ("--inference-labels-dir", "dir"),
    },
    "kpi_analyze": {
        "kpi_csv": ("--kpi-csv", "file"),
    },
    TERMINAL_STAGE: {},
}

_ITER_RE = re.compile(r"^iter(\d+)$")


def phase_order(phase: str) -> list[str]:
    """Return the ordered stage list for a phase label."""
    if phase == "prep":
        return PREP_ORDER
    if phase == "baseline":
        return BASELINE_ORDER
    if _ITER_RE.match(phase):
        return ITER_ORDER
    raise ValueError(f"unknown phase label: {phase!r} (expected prep, baseline, or iterN)")


def iter_number(phase: str) -> int | None:
    m = _ITER_RE.match(phase)
    return int(m.group(1)) if m else None


def previous_phase(phase: str) -> str | None:
    """The phase whose inference output this phase's gap_analysis consumes."""
    n = iter_number(phase)
    if n is None:
        return None
    return "baseline" if n == 1 else f"iter{n - 1}"


def next_stage(phase: str, completed: str | None) -> str | None:
    """Next stage within a phase, or None when the phase is finished."""
    order = phase_order(phase)
    if completed is None:
        return order[0]
    if completed not in order:
        raise ValueError(f"stage {completed!r} is not part of phase {phase!r}")
    idx = order.index(completed)
    return order[idx + 1] if idx + 1 < len(order) else None


def validate_transition(phase: str, completed: str | None, stage: str) -> None:
    """Raise unless `stage` is the legal next stage for `phase`."""
    if stage == TERMINAL_STAGE:
        return
    expected = next_stage(phase, completed)
    if expected is None:
        raise ValueError(
            f"phase {phase!r} already completed its final stage "
            f"({phase_order(phase)[-1]!r}); cannot commit {stage!r}"
        )
    if stage != expected:
        raise ValueError(
            f"out-of-order commit for {phase!r}: expected {expected!r}, got {stage!r}. "
            f"Order is: {' -> '.join(phase_order(phase))}"
        )


# ── state I/O ────────────────────────────────────────────────────────────────

def state_path(results_dir: str | Path) -> Path:
    return Path(results_dir) / "deft_state.json"


def log_path(results_dir: str | Path) -> Path:
    return Path(results_dir) / "loop_log.jsonl"


def read_state(results_dir: str | Path) -> dict[str, Any]:
    p = state_path(results_dir)
    if not p.is_file():
        raise FileNotFoundError(f"deft_state.json not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_state_atomic(results_dir: str | Path, state: dict[str, Any]) -> None:
    p = state_path(results_dir)
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, p)


def read_log(results_dir: str | Path) -> list[dict[str, Any]]:
    p = log_path(results_dir)
    if not p.is_file():
        return []
    events = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def write_log_atomic(results_dir: str | Path, events: list[dict[str, Any]]) -> None:
    """Rewrite the whole log. Used for append and for rollback."""
    p = log_path(results_dir)
    tmp = p.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")
    os.replace(tmp, p)


def phase_entry(state: dict[str, Any], phase: str) -> dict[str, Any]:
    return state.setdefault("iterations", {}).setdefault(phase, {})


def check_artifact(path: str, kind: str) -> str | None:
    """Return an error string when the artifact is missing, wrong-kind, or empty."""
    p = Path(path)
    if kind == "dir":
        if not p.is_dir():
            return f"not a directory: {path}"
        if not any(p.iterdir()):
            return f"directory is empty: {path}"
        return None
    if not p.is_file():
        return f"not a file: {path}"
    if p.stat().st_size == 0:
        return f"file is empty: {path}"

    # Structural check by extension. A stage that died mid-write leaves a file that
    # exists and is non-zero but cannot be read by whatever consumes it next.
    suffix = p.suffix.lower()
    try:
        if suffix == ".json":
            with p.open("r", encoding="utf-8") as fh:
                json.load(fh)
        elif suffix == ".jsonl":
            # Every line, not just the first: `any()` over a truthy expression stops
            # at record one and a file whose remainder is corrupt reads as valid.
            records = 0
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        json.loads(line)
                        records += 1
            if not records:
                return f"no records in {path}"
        elif suffix == ".csv":
            with p.open("r", encoding="utf-8") as fh:
                rows = [line for line in fh if line.strip()]
            if len(rows) < 2:
                return f"no data rows (header only) in {path}"
        elif suffix == ".parquet":
            with p.open("rb") as fh:
                if fh.read(4) != b"PAR1":
                    return f"not a parquet file: {path}"
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        return f"unreadable ({type(exc).__name__}): {path}"
    return None
