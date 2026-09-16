#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diagnose the product Auto-FL campaign continuation decision.

Usage:
    ``python campaign_guard.py [RESULTS] [options]``
Arguments:
    ``RESULTS`` is the campaign TSV ledger; options configure diagnostic caps,
    stop files, plateau, crash, exploration, and family-repeat thresholds.
Output:
    Prints the read-only continuation decision as text or JSON. The campaign
    runner remains the only writer of authoritative campaign state.
Exit codes:
    0 for a completed diagnostic; argparse uses 2 for invalid CLI syntax and
    contract validation errors return a nonzero process status.

The skill runner executes candidates and writes ``results.tsv``. This guard
turns that ledger into a machine-readable decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_HARD_CRASH_THRESHOLD = 6
DEFAULT_MIN_DELTA = 0.0005
DEFAULT_PLATEAU_THRESHOLD = 8
DEFAULT_EXPLORATION_BATCH_SIZE = 3
DEFAULT_FAMILY_REPEAT_LIMIT = 6
DEFAULT_STOP_FILES = ("STOP_AUTOFL", ".nvflare/autofl/STOP")
ATTEMPT_STATUSES = {"candidate", "keep", "discard", "crash"}
SCORED_ATTEMPT_STATUSES = {"keep", "discard"}
LITERATURE_EVENT_STATUSES = {"event", "literature", "checkpoint"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_score(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def normalize_status(row: Dict[str, str]) -> str:
    return (row.get("status", "") or "").strip().lower()


def row_text(row: Dict[str, str]) -> str:
    return " ".join(str(value or "") for value in row.values()).lower()


def is_baseline(row: Dict[str, str]) -> bool:
    return normalize_status(row) == "baseline"


def is_literature_event(row: Dict[str, str]) -> bool:
    status = normalize_status(row)
    if status in LITERATURE_EVENT_STATUSES and "literature" in row_text(row):
        return True
    return "literature" in (row.get("name", "") or "").lower()


def candidate_kind(row: Dict[str, str]) -> str:
    kind = (row.get("candidate_kind", "") or "").strip().lower()
    if kind:
        return kind
    changed = (row.get("changed_files", "") or "").strip().lower()
    return "source_edit" if changed and changed != "none" else "argument_only"


def algorithm_family(row: Dict[str, str]) -> str:
    return (row.get("algorithm_family", "") or "").strip().lower()


def literature_events(rows: List[Dict[str, str]]) -> List[Tuple[int, str]]:
    events = []
    for idx, row in enumerate(rows):
        if is_literature_event(row):
            event_id = (row.get("literature_event_id", "") or "").strip() or f"lit-{len(events) + 1:04d}"
            events.append((idx, event_id))
    return events


def comparable_attempts(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [row for row in rows if normalize_status(row) in ATTEMPT_STATUSES and not is_baseline(row)]


def pending_candidates(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [row for row in rows if normalize_status(row) == "candidate"]


def is_scored_attempt(row: Dict[str, str]) -> bool:
    return normalize_status(row) in SCORED_ATTEMPT_STATUSES and not is_baseline(row)


def is_retained(row: Dict[str, str]) -> bool:
    return normalize_status(row) == "keep" or is_baseline(row)


def scored_rows_with_index(rows: List[Dict[str, str]], keep_row) -> List[Tuple[int, Dict[str, str], float]]:
    scored = []
    for idx, row in enumerate(rows):
        if not keep_row(row):
            continue
        score = parse_score(row.get("score", ""))
        if score is not None:
            scored.append((idx, row, score))
    return scored


def scored_attempts_with_index(rows: List[Dict[str, str]]) -> List[Tuple[int, Dict[str, str], float]]:
    return scored_rows_with_index(rows, is_scored_attempt)


def exploration_batches(rows: List[Dict[str, str]], batch_size: int) -> List[Dict[str, Any]]:
    """Per literature event: linked scored source-edited candidates and the batch-completion row index."""
    events = literature_events(rows)
    scored = scored_attempts_with_index(rows)
    batches = []
    for position, (event_idx, event_id) in enumerate(events):
        next_event_idx = events[position + 1][0] if position + 1 < len(events) else len(rows)
        linked = []
        for row_idx, row, _ in scored:
            if row_idx <= event_idx or candidate_kind(row) != "source_edit":
                continue
            row_event = (row.get("literature_event_id", "") or "").strip()
            if row_event == event_id or (not row_event and row_idx < next_event_idx):
                linked.append(row_idx)
        if batch_size <= 0:
            completion_index: Optional[int] = event_idx
        elif len(linked) >= batch_size:
            completion_index = sorted(linked)[batch_size - 1]
        else:
            completion_index = None
        batches.append(
            {
                "literature_event_id": event_id,
                "event_index": event_idx,
                "required": max(batch_size, 0),
                "completed": len(linked),
                "completion_index": completion_index,
            }
        )
    return batches


def family_repeat_stalled(rows: List[Dict[str, str]], limit: int) -> bool:
    if limit <= 0 or not literature_events(rows):
        return False
    scored = scored_attempts_with_index(rows)
    if len(scored) < limit:
        return False
    recent = [row for _, row, _ in scored[-limit:]]
    families = {algorithm_family(row) for row in recent}
    if len(families) != 1 or not next(iter(families)):
        return False
    return all(candidate_kind(row) == "argument_only" for row in recent)


def validate_mode(mode: str) -> str:
    if mode not in {"min", "max"}:
        raise ValueError(f"mode must be 'min' or 'max', but got {mode!r}")
    return mode


def better(new_score: Optional[float], old_score: Optional[float], mode: str = "max", min_delta: float = 0.0) -> bool:
    mode = validate_mode(mode)
    new_score = parse_score(new_score)
    old_score = parse_score(old_score)
    if new_score is None:
        return False
    if old_score is None:
        return True
    if mode == "min":
        return new_score < old_score - min_delta
    return new_score > old_score + min_delta


def best_score(rows: List[Dict[str, str]], mode: str = "max") -> Optional[float]:
    best = None
    for _, _, score in scored_rows_with_index(rows, is_retained):
        if better(score, best, mode):
            best = score
    return best


def scored_baseline_score(rows: List[Dict[str, str]]) -> Optional[float]:
    for row in rows:
        if not is_baseline(row):
            continue
        score = parse_score(row.get("score", ""))
        if score is not None:
            return score
    return None


def improvement_over_baseline(baseline: Optional[float], best: Optional[float], mode: str = "max") -> Optional[float]:
    """Return a direction-adjusted delta so positive values always mean improvement."""

    mode = validate_mode(mode)
    if baseline is None or best is None:
        return None
    return baseline - best if mode == "min" else best - baseline


def campaign_mode_for_results(results_path: Path) -> str:
    """Read the authoritative objective mode associated with a campaign ledger."""

    state_path = results_path.parent / ".nvflare/autofl/campaign_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot determine objective mode from {state_path}; pass --mode explicitly or repair campaign state"
        ) from exc
    state_mode = state.get("mode") if isinstance(state, dict) else None
    if state_mode not in {"max", "min"}:
        raise ValueError(
            f"campaign state {state_path} does not contain a valid objective mode; "
            "pass --mode explicitly or repair campaign state"
        )
    return state_mode


def plateau_status(
    rows: List[Dict[str, str]],
    threshold: int,
    min_delta: float,
    mode: str = "max",
    exploration_batch_size: int = DEFAULT_EXPLORATION_BATCH_SIZE,
) -> Dict[str, Any]:
    retained = scored_rows_with_index(rows, is_retained)
    scored_attempts = scored_attempts_with_index(rows)
    if threshold <= 0 or not retained:
        return {
            "available": True,
            "recommendation": "continue",
            "scored_since_reset": 0,
            "threshold": threshold,
            "min_delta": min_delta,
        }

    best = None
    best_row_idx = -1
    best_scored_idx = -1
    best_name = ""
    for scored_idx, (row_idx, row, score) in enumerate(retained):
        if better(score, best, mode, min_delta):
            best = score
            best_row_idx = row_idx
            best_scored_idx = scored_idx
            best_name = row.get("name", "")

    batches = exploration_batches(rows, exploration_batch_size)
    last_literature_idx = max((batch["event_index"] for batch in batches), default=-1)
    # The plateau clock resets when a literature exploration batch COMPLETES, not when the
    # review is merely recorded — otherwise recording a review relieves plateau pressure
    # without any source-backed follow-through.
    last_batch_completion_idx = max(
        (batch["completion_index"] for batch in batches if batch["completion_index"] is not None), default=-1
    )
    reset_row_idx = max(best_row_idx, last_batch_completion_idx)
    scored_since_reset = sum(1 for row_idx, _, _ in scored_attempts if row_idx > reset_row_idx)
    recommendation = "literature" if scored_since_reset >= threshold else "continue"
    return {
        "available": True,
        "recommendation": recommendation,
        "best_name": best_name,
        "best_score": best,
        "best_scored_index": best_scored_idx,
        "last_literature_event_index": last_literature_idx,
        "last_batch_completion_index": last_batch_completion_idx,
        "min_delta": min_delta,
        "reset_row_index": reset_row_idx,
        "scored_since_reset": scored_since_reset,
        "threshold": threshold,
    }


def parse_max_candidates(value: Optional[str]) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def parse_max_candidates_arg(value: str) -> int:
    parsed = parse_max_candidates(value)
    if parsed is None:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def existing_stop_files(paths: List[str]) -> List[str]:
    return [path for path in paths if Path(path).exists()]


def repeated_crash_blocker(attempts: List[Dict[str, str]], threshold: int) -> bool:
    if threshold <= 0 or len(attempts) < threshold:
        return False
    return all(normalize_status(row) == "crash" for row in attempts[-threshold:])


def guard_state_for_rows(
    rows: List[Dict[str, str]],
    *,
    results_path: str = "results.tsv",
    max_candidates: Optional[int] = None,
    stop_files: Optional[List[str]] = None,
    plateau_threshold: int = DEFAULT_PLATEAU_THRESHOLD,
    min_delta: float = DEFAULT_MIN_DELTA,
    hard_crash_threshold: int = DEFAULT_HARD_CRASH_THRESHOLD,
    mode: str = "max",
    pending_manifest_count: int = 0,
    exploration_batch_size: int = DEFAULT_EXPLORATION_BATCH_SIZE,
    family_repeat_limit: int = DEFAULT_FAMILY_REPEAT_LIMIT,
) -> Dict[str, Any]:
    if hard_crash_threshold < 0:
        raise ValueError("hard_crash_threshold must be non-negative")
    if exploration_batch_size < 0:
        raise ValueError("exploration_batch_size must be non-negative")
    if family_repeat_limit < 0:
        raise ValueError("family_repeat_limit must be non-negative")
    mode = validate_mode(mode)
    attempts = comparable_attempts(rows)
    pending = pending_candidates(rows)
    cap = max_candidates
    cap_source = "explicit" if cap is not None else "uncapped"
    stop_file_hits = existing_stop_files(stop_files or list(DEFAULT_STOP_FILES))
    batches = exploration_batches(rows, exploration_batch_size)
    active_batch = next((batch for batch in batches if batch["completion_index"] is None), None)
    plateau = plateau_status(rows, plateau_threshold, min_delta, mode, exploration_batch_size)

    decision = "continue"
    reason = "continue"
    next_action = "propose_candidate"
    final_response_allowed = False

    pending_count = len(pending) + pending_manifest_count
    if stop_file_hits:
        decision = "stop"
        if pending_count:
            reason = "manual_stop_pending_candidate"
            next_action = "abandon_candidate"
        else:
            reason = "manual_stop_file"
            next_action = "final_report"
            final_response_allowed = True
    elif pending_count:
        reason = "pending_candidates"
        next_action = "edit_candidate"
    elif cap is not None and len(attempts) >= cap:
        decision = "stop"
        reason = "candidate_cap_exhausted"
        next_action = "final_report"
        final_response_allowed = True
    elif repeated_crash_blocker(attempts, hard_crash_threshold):
        decision = "stop"
        reason = "hard_repeated_crash_blocker"
        next_action = "final_report"
        final_response_allowed = True
    elif active_batch is not None:
        reason = "literature_exploration_batch"
        next_action = "develop_literature_batch"
    elif family_repeat_stalled(rows, family_repeat_limit):
        reason = "family_repeat_limit"
        next_action = "diversify_candidates"
    elif plateau.get("recommendation") == "literature":
        reason = "plateau_literature"
        next_action = "run_literature_loop"

    if final_response_allowed:
        instruction = (
            "Final report is allowed because authoritative campaign state reached a stop condition. "
            "Finalize results.tsv, progress.png, and the refreshed autofl_report.md, then produce a final "
            "answer summarizing baseline vs best score with metric source, failures, friction, commands, "
            "and absolute artifact paths."
        )
    elif next_action == "abandon_candidate":
        instruction = (
            "Manual stop requested. Do not execute the pending candidate. Run the runner abandon action, "
            "then refresh status and generate the final report."
        )
    elif next_action == "edit_candidate":
        instruction = "Do not produce a final answer. Finish the pending candidate, then run the runner status action."
    elif next_action == "run_literature_loop":
        instruction = (
            "Do not produce a final answer. Run the literature loop and record a literature event with "
            "record --literature. Select a workload-appropriate idea (server aggregation, client optimizer, "
            "loss, schedule, or architecture within the fixed budget), then develop it as a source-backed "
            "exploration batch linked via --literature-event."
        )
    elif next_action == "develop_literature_batch":
        remaining = active_batch["required"] - active_batch["completed"]
        instruction = (
            f"Do not produce a final answer. Literature event {active_batch['literature_event_id']} needs "
            f"{remaining} more scored source-backed candidate(s) prepared with --literature-event before normal "
            "candidate flow resumes: a faithful implementation, a tuned variant, and an ablation under the "
            "same comparison budget."
        )
    elif next_action == "diversify_candidates":
        instruction = (
            "Do not produce a final answer. Recent scored attempts are argument-only tuning of a single "
            "algorithm family; prepare a source-backed candidate or switch to a different family next."
        )
    else:
        instruction = "Do not produce a final answer. Propose and prepare the next same-budget candidate now."

    retained_best = best_score(rows, mode)
    baseline = scored_baseline_score(rows)
    return {
        "schema_version": "nvflare.autofl.campaign_state.v1",
        "updated_at": utc_now(),
        "results": results_path,
        "mode": mode,
        "decision": decision,
        "reason": reason,
        "next_action": next_action,
        "final_response_allowed": final_response_allowed,
        "candidate_cap": cap,
        "candidate_cap_source": cap_source,
        "candidate_attempts": len(attempts),
        "remaining_candidates": max(0, cap - len(attempts)) if cap is not None else None,
        "pending_candidates": pending_count,
        "scored_attempts": len(scored_attempts_with_index(rows)),
        "best_score": retained_best,
        "baseline_status": "complete" if baseline is not None else "pending",
        "baseline_score": baseline,
        "improvement": improvement_over_baseline(baseline, retained_best, mode),
        "stop_files": stop_file_hits,
        "plateau": plateau,
        "exploration_batch": active_batch or (batches[-1] if batches else None),
        "required_exploration": (
            "source_backed_exploration" if next_action in {"run_literature_loop", "develop_literature_batch"} else None
        ),
        "agent_instruction": instruction,
    }


def guard_state(
    results_path: Path,
    *,
    max_candidates: Optional[int] = None,
    stop_files: Optional[List[str]] = None,
    plateau_threshold: int = DEFAULT_PLATEAU_THRESHOLD,
    min_delta: float = DEFAULT_MIN_DELTA,
    hard_crash_threshold: int = DEFAULT_HARD_CRASH_THRESHOLD,
    mode: str = "max",
    pending_manifest_count: int = 0,
    exploration_batch_size: int = DEFAULT_EXPLORATION_BATCH_SIZE,
    family_repeat_limit: int = DEFAULT_FAMILY_REPEAT_LIMIT,
) -> Dict[str, Any]:
    return guard_state_for_rows(
        load_rows(results_path),
        results_path=str(results_path),
        max_candidates=max_candidates,
        stop_files=stop_files,
        plateau_threshold=plateau_threshold,
        min_delta=min_delta,
        hard_crash_threshold=hard_crash_threshold,
        mode=mode,
        pending_manifest_count=pending_manifest_count,
        exploration_batch_size=exploration_batch_size,
        family_repeat_limit=family_repeat_limit,
    )


def print_text(state: Dict[str, Any]) -> None:
    for key in [
        "decision",
        "reason",
        "next_action",
        "final_response_allowed",
        "candidate_cap",
        "candidate_cap_source",
        "mode",
        "candidate_attempts",
        "remaining_candidates",
        "pending_candidates",
        "scored_attempts",
        "best_score",
        "baseline_status",
        "baseline_score",
        "improvement",
        "agent_instruction",
    ]:
        value = state.get(key)
        if isinstance(value, bool):
            value = str(value).lower()
        elif value is None:
            value = ""
        print(f"{key}={value}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", default="results.tsv")
    parser.add_argument("--max-candidates", type=parse_max_candidates_arg)
    parser.add_argument("--stop-file", action="append")
    parser.add_argument("--plateau-threshold", type=int, default=DEFAULT_PLATEAU_THRESHOLD)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--hard-crash-threshold", type=int, default=DEFAULT_HARD_CRASH_THRESHOLD)
    parser.add_argument("--exploration-batch-size", type=int, default=DEFAULT_EXPLORATION_BATCH_SIZE)
    parser.add_argument("--family-repeat-limit", type=int, default=DEFAULT_FAMILY_REPEAT_LIMIT)
    parser.add_argument("--mode", choices=["max", "min"])
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args(argv)

    if args.plateau_threshold <= 0:
        raise ValueError("--plateau-threshold must be positive")
    if args.min_delta < 0:
        raise ValueError("--min-delta must be non-negative")
    if args.hard_crash_threshold < 0:
        raise ValueError("--hard-crash-threshold must be non-negative")
    if args.exploration_batch_size < 0:
        raise ValueError("--exploration-batch-size must be non-negative")
    if args.family_repeat_limit < 0:
        raise ValueError("--family-repeat-limit must be non-negative")

    results_path = Path(args.results).resolve()
    mode = args.mode
    if mode is None:
        try:
            mode = campaign_mode_for_results(results_path)
        except ValueError as exc:
            parser.error(str(exc))
    stop_files = args.stop_file if args.stop_file is not None else list(DEFAULT_STOP_FILES)
    resolved_stop_files = [
        str(path if path.is_absolute() else results_path.parent / path) for path in map(Path, stop_files)
    ]
    state = guard_state(
        results_path,
        max_candidates=args.max_candidates,
        stop_files=resolved_stop_files,
        plateau_threshold=args.plateau_threshold,
        min_delta=args.min_delta,
        hard_crash_threshold=args.hard_crash_threshold,
        mode=mode,
        exploration_batch_size=args.exploration_batch_size,
        family_repeat_limit=args.family_repeat_limit,
    )
    if args.format == "json":
        print(json.dumps(state, sort_keys=True))
    else:
        print_text(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
