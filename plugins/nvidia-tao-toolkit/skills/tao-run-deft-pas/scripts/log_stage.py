# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Append a stage entry to results/loop_log.jsonl.

Disk-truth invariant: never trust in-memory seq across turns. Always re-read
the last entry of the log to compute next_seq. Context compaction is invisible
to this writer — there is no "compacted" flag and no detection branch. seq is
strict and gap-free, starting at 0.

`tokens` remains null because runtime-neutral stage scripts cannot measure an
agent harness's context usage.

Library usage:

    from log_stage import append_stage
    import time, pathlib

    t0 = time.monotonic()
    # ... run the stage ...
    append_stage(
        pathlib.Path(f"{RESULTS_DIR}/loop_log.jsonl"),
        iteration="iter1",
        stage="data_mining",
        status="ok",
        summary="mined 250 candidate pairs from 10 gap clusters",
        duration_s=time.monotonic() - t0,
    )

This is an internal module. ``commit_stage.py`` is the only supported writer.
"""

from __future__ import annotations

import datetime
import json
import math
import pathlib

_VALID_STATUSES = {"ok", "error", "skip"}
_VALID_STAGES = {
    "dataset_setup",
    "pool_embed",
    "evaluate",
    "gap_analysis",
    "data_mining",
    "history_select",
    "visualize",
    "train",
    "loop_stop",
}


def next_seq(log_path: pathlib.Path) -> int:
    """Return seq for the next entry: last entry's seq + 1, or 0 if no log yet."""
    if not isinstance(log_path, pathlib.Path):
        raise TypeError(
            f"log_path must be pathlib.Path, got {type(log_path).__name__}"
        )
    if not log_path.exists():
        return 0
    last = None
    with log_path.open() as f:
        for line in f:
            if line.strip():
                last = line
    if last is None:
        return 0
    try:
        prev_seq = json.loads(last)["seq"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise ValueError(
            f"corrupt last line in {log_path}: {exc}; refusing to append"
        ) from exc
    if not isinstance(prev_seq, int) or isinstance(prev_seq, bool):
        raise ValueError(
            f"non-integer seq in last line of {log_path}: {prev_seq!r}"
        )
    return prev_seq + 1


def append_stage(
    log_path: pathlib.Path,
    *,
    iteration: str,
    stage: str,
    status: str,
    summary: str,
    duration_s: float | None = None,
    tokens: int | None = None,
) -> None:
    """Append one stage event. Caller is responsible for measuring duration.

    Raises:
        TypeError: any argument has the wrong type.
        ValueError: any argument is empty, out-of-range, or otherwise invalid.
    """
    if not isinstance(log_path, pathlib.Path):
        raise TypeError(
            f"log_path must be pathlib.Path, got {type(log_path).__name__}"
        )
    if not isinstance(iteration, str) or not iteration:
        raise ValueError(f"iteration must be a non-empty string, got {iteration!r}")
    if not isinstance(stage, str) or not stage:
        raise ValueError(f"stage must be a non-empty string, got {stage!r}")
    if stage not in _VALID_STAGES:
        raise ValueError(
            f"stage must be one of {sorted(_VALID_STAGES)}, got {stage!r}"
        )
    if status not in _VALID_STATUSES:
        raise ValueError(
            f"status must be one of {sorted(_VALID_STATUSES)}, got {status!r}"
        )
    if not isinstance(summary, str) or not summary:
        raise ValueError(f"summary must be a non-empty string, got {summary!r}")
    if duration_s is not None:
        if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
            raise TypeError(
                f"duration_s must be a number or None, got "
                f"{type(duration_s).__name__}"
            )
        if not math.isfinite(duration_s):
            raise ValueError(f"duration_s must be finite, got {duration_s!r}")
        if duration_s < 0:
            raise ValueError(f"duration_s must be >= 0, got {duration_s}")
        duration_s = float(duration_s)
    if tokens is not None:
        if not isinstance(tokens, int) or isinstance(tokens, bool):
            raise TypeError(
                f"tokens must be int or None, got {type(tokens).__name__}"
            )
        if tokens < 0:
            raise ValueError(f"tokens must be >= 0, got {tokens}")

    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "seq": next_seq(log_path),
        "ts": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        ),
        "iteration": iteration,
        "stage": stage,
        "status": status,
        "summary": summary,
        "duration_s": duration_s,
        "tokens": tokens,
    }
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
