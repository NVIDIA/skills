#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only audit of one DEFT OD run: state vs log vs what is actually on disk.

Why this exists: the loop's memory is disk, not context. An agent resuming after
compaction — or about to tell the user the loop finished — has no trustworthy
in-context record of what ran. This script rebuilds that record from
``deft_state.json`` and ``loop_log.jsonl``, re-checks every artifact path the run
recorded, and prints the single next action that is safe to take.

It never writes anything. Repairing an inconsistency is the agent's job, and the
only supported repair is re-running the stage through ``commit_stage.py`` — never
hand-editing JSON.

Three questions, in the order the loop asks them:

  DEFT_RUN_STATUS     Is the run internally consistent? Anything else is unsafe.
  next_action         Which stage may run now (or ``loop_stop`` / ``complete``).
  read_before_action  Which ``references/*.md`` overlay to load before doing it.

``--require-complete`` exists because "the report file is there" is not proof the
loop ran. Completion means a committed ``loop_stop``, a ``baseline`` that finished
``kpi_analyze`` — without it there is no baseline mAP and therefore no trend, the
loop's whole deliverable — plus either every configured iteration finishing all of
``ITER_ORDER``, or a ``gap_analysis`` that recorded zero weak images and stopped
there (the documented early stop). The presence of a checkpoint, a KPI csv, or
``DEFT_Loop_Report.md`` counts for nothing here.

``--require-terminal`` means one thing: ``loop_stop`` is committed. A run that
failed but never finalized is *not* terminal — ``next_action`` still says
``loop_stop``, and a report rendered before that has no authoritative outcome.

CLI:

    python3 scripts/audit_deft_run.py --results-dir /abs/results/run_X
    python3 scripts/audit_deft_run.py --results-dir /abs/results/run_X --json
    python3 scripts/audit_deft_run.py --results-dir /abs/results/run_X --require-terminal
    python3 scripts/audit_deft_run.py --results-dir /abs/results/run_X --require-complete

Exit codes:
  0  VALID and every --require-* condition met
  1  INVALID, or a --require-* condition unmet
  2  deft_state.json is missing or unparseable (a special case of INVALID)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from deft_stages import (
    ALL_STAGES,
    BASELINE_ORDER,
    ITER_ORDER,
    PREP_ORDER,
    SCHEMA_VERSION,
    STAGE_ARTIFACTS,
    TERMINAL_STAGE,
    check_artifact,
    iter_number,
    log_path,
    next_stage,
    phase_order,
    previous_phase,
    read_state,
    state_path,
    validate_transition,
)

# next_action -> the overlay the agent must read before running it. Stages absent
# from this table (loop_stop, complete, and any INVALID repair description) have
# no overlay and report "none".
STAGE_OVERLAYS = {
    "prep": "references/prep-source-pool.md",
    "gap_analysis": "references/tao-analyze-gaps-od-map.md",
    "embed": "references/tao-generate-image-embeddings.md",
    "mine": "references/tao-mine-od-images.md",
    "stage": "references/stage-mined-data.md",
    "train": "references/grounding-dino.md",
    "inference": "references/grounding-dino.md",
    "kpi_analyze": "references/tao-analyze-detection-kpi.md",
}

VALID_LOG_STATUSES = {"ok", "error"}
# "stopped": loop_stop was committed on a run this audit does not call complete.
# commit_stage.py writes "complete" only when this audit agrees it is one.
VALID_RUN_STATUSES = {"running", "in_progress", "stopped", "complete", "failed"}

# Optional paths the overlays record alongside a stage's required artifacts.
# commit_stage.py declares one flag per entry and checks it at write time; this
# audit re-checks them on every run, because a recorded path that is no longer on
# disk is exactly as misleading here as a required one. Field -> (flag, kind).
EXTRA_ARTIFACT_FIELDS: dict[str, tuple[str, str]] = {
    # kpi_analyze stdout tee; the only place mAP appears
    "kpi_log": ("--kpi-log", "file"),
    # prep: validate_pool_coco.py's verdict on the converted pool
    "pool_report": ("--pool-report", "file"),
    # stage: merged training manifest
    "combined_manifest": ("--combined-manifest", "file"),
    # stage: post-merge consistency report
    "merge_validation_report": ("--merge-validation-report", "file"),
}

# init_deft_state.py pins the source pool in config; `prep` is the stage that
# produces it when --allow-missing-pool was passed. Same two artifacts as
# STAGE_ARTIFACTS["prep"], under the config names.
POOL_CONFIG_FIELDS = {
    "source_pool_annotations": "dir",
    "source_pool_embeddings": "file",
}

REPAIR_ACTION = "repair the listed disk inconsistencies before running another stage"
FAILED_ACTION = "report the run as FAILED; do not claim completion"
INCOMPLETE_ACTION = (
    "report the run as INCOMPLETE; loop_stop was committed before the configured "
    "iterations finished and before any documented early stop"
)


def _phase_sort_key(phase: str) -> tuple[int, int]:
    """Order phases the way the loop runs them: prep, baseline, iter1, iter2..."""
    if phase == "prep":
        return (0, 0)
    if phase == "baseline":
        return (1, 0)
    number = iter_number(phase)
    return (2, number) if number is not None else (3, 0)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _label_problem(phase: str) -> str | None:
    """Reject iteration labels that are not exactly ``iter<n>``, n >= 1.

    ``deft_stages`` matches ``iter(\\d+)``, so ``iter01`` is a phase distinct from
    ``iter1`` that reports the same iteration number. Left alone it is an alias:
    an extra full iteration that slips past the max_iterations bound, is counted
    in iterations_completed, and desynchronizes ``previous_phase()`` (iter2 then
    reads iter1's labels even though iter01 trained last). One spelling per
    iteration is the only way those three stay consistent.

    ``iter0`` is rejected for a different reason: iterations are numbered from 1,
    and ``previous_phase("iter0")`` answers ``"iter-1"`` — a label nothing can
    resolve.
    """
    number = iter_number(phase)
    if number is None:
        return None
    if number < 1:
        return (
            f"iteration label {phase!r} is out of range; iterations are numbered from 1 "
            "and the zeroth phase is called 'baseline'"
        )
    if phase != f"iter{number}":
        return (
            f"iteration label {phase!r} is not canonical; the only accepted spelling for "
            f"iteration {number} is 'iter{number}'"
        )
    return None


def _load_log(results_dir: Path, errors: list[str]) -> list[dict[str, Any]]:
    """Parse loop_log.jsonl line by line.

    deft_stages.read_log() raises on the first bad line, which would abandon the
    whole audit. An auditor has to survive a corrupt log and report where it is
    corrupt, so it parses per line and keeps going.
    """
    path = log_path(results_dir)
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"loop_log.jsonl:{lineno}: invalid JSON: {exc}")
                continue
            if not isinstance(event, dict):
                errors.append(f"loop_log.jsonl:{lineno}: event must be a JSON object")
                continue
            entries.append(event)
    return entries


def _check_event_schema(
    entries: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> None:
    """seq must be strictly increasing from 1 with no gaps or duplicates."""
    for index, event in enumerate(entries, 1):
        seq = event.get("seq")
        if not _is_int(seq):
            errors.append(f"loop_log entry {index}: seq must be an integer, got {seq!r}")
        elif seq != index:
            errors.append(
                f"loop_log entry {index}: seq={seq} breaks the strictly-increasing-from-1 "
                f"sequence (expected {index}); the log has a gap, a duplicate, or a reorder"
            )
        label = f"seq={seq}" if _is_int(seq) else f"entry {index}"
        if event.get("status") not in VALID_LOG_STATUSES:
            errors.append(
                f"loop_log {label}: status must be one of "
                f"{sorted(VALID_LOG_STATUSES)}, got {event.get('status')!r}"
            )
        summary = event.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            errors.append(f"loop_log {label}: summary must be a non-empty string")
        duration = event.get("duration_sec")
        if not _is_int(duration) or duration < 0:
            errors.append(
                f"loop_log {label}: duration_sec must be a non-negative integer, "
                f"got {duration!r}"
            )
        if not isinstance(event.get("ts"), str) or not event.get("ts"):
            errors.append(f"loop_log {label}: ts must be an ISO-8601 UTC string")
        # context_tokens is a reserved placeholder; it is always 0.
        # Absent is survivable; present-but-wrong means someone edited the log.
        if "context_tokens" not in event:
            warnings.append(f"loop_log {label}: context_tokens is missing")
        elif not _is_int(event["context_tokens"]) or event["context_tokens"] < 0:
            errors.append(
                f"loop_log {label}: context_tokens must be a non-negative integer, "
                f"got {event['context_tokens']!r}"
            )


def _zero_weak_images(info: Any) -> bool:
    """True when this phase's gap_analysis recorded the documented early stop.

    The count wins over the flag whenever both are present: a recorded positive
    count is the measurement, and ``zero_weak_images`` next to it is a
    contradiction, not a second opinion (``_contradictory_zero_weak`` reports it).
    """
    if not isinstance(info, dict):
        return False
    count = info.get("weak_image_count")
    if _is_int(count):
        return count == 0
    return info.get("zero_weak_images") is True


def _contradictory_zero_weak(info: Any) -> bool:
    """True when a phase entry claims zero weak images and a positive count."""
    if not isinstance(info, dict):
        return False
    count = info.get("weak_image_count")
    return info.get("zero_weak_images") is True and _is_int(count) and count > 0


def _pool_exhausted(iterations: dict[str, Any]) -> str | None:
    """The phase whose loop_stop recorded a spent source pool, if any.

    The second documented terminal state. The miner raises rather than returning a
    short result, so a pool that cannot fill the budget ends the run. ``--pool-exhausted``
    on the loop_stop commit asserts it; ``pool_remaining`` is the count and may be
    non-zero, since too few images is exhaustion just as much as none.
    """
    for phase in sorted(iterations, key=_phase_sort_key):
        # Only an iteration can run out of pool: baseline never mines, so a claim
        # recorded there is not evidence of anything.
        if iter_number(phase) is None:
            continue
        info = iterations.get(phase)
        if not isinstance(info, dict):
            continue
        if info.get("pool_exhausted") is True or info.get("pool_remaining") == 0:
            return phase
    return None


def _max_iterations(state: dict[str, Any]) -> tuple[Any, str]:
    """init_deft_state.py freezes it under config; accept a top-level copy too."""
    config = state.get("config")
    if isinstance(config, dict) and "max_iterations" in config:
        return config["max_iterations"], "state.config.max_iterations"
    return state.get("max_iterations"), "state.max_iterations"


def _prep_produced_pool(iterations: dict[str, Any]) -> bool:
    """True when a prep phase already recorded both pool artifacts, still on disk."""
    entry = iterations.get("prep")
    if not isinstance(entry, dict):
        return False
    return all(
        entry.get(field) and check_artifact(str(entry[field]), kind) is None
        for field, (_flag, kind) in STAGE_ARTIFACTS["prep"].items()
    )


def _pool_on_disk(state: dict[str, Any]) -> bool:
    """True when the pool init_deft_state.py pinned is already labeled and embedded."""
    config = state.get("config")
    config = config if isinstance(config, dict) else {}
    return all(
        config.get(field) and check_artifact(str(config[field]), kind) is None
        for field, kind in POOL_CONFIG_FIELDS.items()
    )


def _next_action(
    *,
    state: dict[str, Any],
    iterations: dict[str, Any],
    completed_by_phase: dict[str, str],
    loop_stop_committed: bool,
    run_failed: bool,
    complete: bool,
    max_iterations: int | None,
) -> str:
    """The one stage it is safe to run next, given only what is on disk.

    This walks the run in the order the loop runs it — prep, baseline, iter1..N —
    and answers with the first phase that still owes a stage. It deliberately does
    not key off the last log event's phase: a commit to a lagging phase (the
    baseline kpi_analyze that was skipped, a late prep) is a legal write that says
    nothing about where the run is. Keying off it rewinds next_action to a stage
    commit_stage.py will reject, and the loop's one instruction to the agent
    becomes an instruction it cannot carry out.
    """
    if loop_stop_committed:
        # A finalized hard stop is over, but it is not "complete" — say so, so a
        # resuming agent cannot read this line as permission to claim success.
        if run_failed:
            return FAILED_ACTION
        # Same reasoning for a loop_stop committed early: --require-complete
        # rejects this run, so next_action must not answer "complete" and hand
        # the agent the completion claim that flag exists to withhold.
        return "complete" if complete else INCOMPLETE_ACTION
    if run_failed:
        # A hard stop still has to be finalized: commit loop_stop, then report.
        return TERMINAL_STAGE

    # prep comes first and produces the mining corpus every iteration reads. It is
    # re-checked on every audit, not only on an empty log: the pool can be absent
    # at `mine` on a --allow-missing-pool run whose prep was never committed, and
    # naming `mine` there hands the agent a stage that cannot succeed.
    if not (_pool_on_disk(state) or _prep_produced_pool(iterations)):
        return PREP_ORDER[0]

    def _owed(phase: str) -> str | None:
        try:
            upcoming = next_stage(phase, completed_by_phase.get(phase))
        except ValueError:
            return REPAIR_ACTION
        if upcoming is None:
            return None
        # Zero weak images short-circuits the remaining six stages of the phase.
        if (
            completed_by_phase.get(phase) == "gap_analysis"
            and _zero_weak_images(iterations.get(phase))
        ):
            return TERMINAL_STAGE
        return upcoming

    owed = _owed("baseline")
    if owed is not None:
        return owed
    for number in range(1, (max_iterations or 0) + 1):
        owed = _owed(f"iter{number}")
        if owed is not None:
            return owed
    return TERMINAL_STAGE


def audit(results_dir: Path) -> dict[str, Any]:
    """Cross-check state, log, and disk. Returns the full report; writes nothing."""
    results_dir = results_dir.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []

    state: dict[str, Any] | None = None
    load_failed = False
    try:
        state = read_state(results_dir)
    except FileNotFoundError:
        errors.append(f"deft_state.json not found: {state_path(results_dir)}")
        load_failed = True
    except json.JSONDecodeError as exc:
        errors.append(f"deft_state.json is not valid JSON: {exc}")
        load_failed = True
    if state is not None and not isinstance(state, dict):
        errors.append("deft_state.json root must be a JSON object")
        load_failed = True
        state = None
    if state is None:
        state = {}

    entries = _load_log(results_dir, errors)
    _check_event_schema(entries, errors, warnings)

    if not load_failed and state.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"state.schema_version={state.get('schema_version')!r} does not match "
            f"the schema this audit understands ({SCHEMA_VERSION})"
        )

    recorded_results_dir = state.get("results_dir")
    if recorded_results_dir:
        recorded_path = Path(str(recorded_results_dir)).expanduser()
        if not recorded_path.is_absolute():
            errors.append(f"state.results_dir must be absolute: {recorded_results_dir}")
        elif recorded_path.resolve() != results_dir:
            errors.append(
                f"state.results_dir={recorded_path.resolve()} does not match the audited "
                f"directory {results_dir}"
            )
    elif not load_failed:
        errors.append("state.results_dir is required")

    max_iterations, max_iterations_key = _max_iterations(state)
    if not _is_int(max_iterations) or max_iterations <= 0:
        if not load_failed:
            errors.append(
                f"{max_iterations_key} must be an integer > 0, got {max_iterations!r}"
            )
        max_iterations = None

    current_iteration = state.get("current_iteration")
    if current_iteration is not None and (
        not _is_int(current_iteration) or current_iteration < 0
    ):
        errors.append(
            f"state.current_iteration must be a non-negative integer, "
            f"got {current_iteration!r}"
        )

    run_status = state.get("status")
    if run_status is not None and run_status not in VALID_RUN_STATUSES:
        errors.append(
            f"state.status={run_status!r} is invalid; expected one of "
            f"{sorted(VALID_RUN_STATUSES)}"
        )

    iterations = state.get("iterations")
    if not isinstance(iterations, dict):
        if not load_failed:
            errors.append("state.iterations must be an object")
        iterations = {}

    # ── walk the log: legal ordering, and what each phase actually finished ──
    completed_by_phase: dict[str, str] = {}
    ok_stages_by_phase: dict[str, list[str]] = {}
    seq_of: dict[tuple[str, str], int] = {}
    flagged_phases: set[str] = set()
    first_error_seq: Any = None
    loop_stop_seq: Any = None
    loop_stop_committed = False

    for event in entries:
        seq = event.get("seq")
        phase = str(event.get("iter", ""))
        stage = str(event.get("stage", ""))
        status = event.get("status")
        try:
            phase_order(phase)
        except ValueError as exc:
            errors.append(f"loop_log seq={seq}: {exc}")
            continue
        label_problem = _label_problem(phase)
        if label_problem:
            errors.append(f"loop_log seq={seq}: {label_problem}")
            continue
        if stage not in ALL_STAGES:
            errors.append(
                f"loop_log seq={seq}: stage {stage!r} is not one of {ALL_STAGES}"
            )
            continue
        if loop_stop_seq is not None:
            errors.append(
                f"loop_log seq={seq}: {phase}/{stage} was committed after loop_stop "
                f"(seq={loop_stop_seq}); loop_stop must be the final event"
            )
        if first_error_seq is not None and stage != TERMINAL_STAGE:
            errors.append(
                f"loop_log seq={seq}: {phase}/{stage} follows the status=error commit at "
                f"seq={first_error_seq}; only loop_stop may follow a hard stop"
            )
        if status == "error":
            if first_error_seq is None:
                first_error_seq = seq
            continue  # a failed stage does not advance the phase
        if status != "ok":
            continue  # already reported by _check_event_schema
        if stage == TERMINAL_STAGE:
            loop_stop_seq = seq
            loop_stop_committed = True
            continue

        if stage not in phase_order(phase):
            errors.append(
                f"loop_log seq={seq}: stage {stage!r} is not part of phase {phase!r} "
                f"({' -> '.join(phase_order(phase))})"
            )
            continue
        try:
            validate_transition(phase, completed_by_phase.get(phase), stage)
        except ValueError as exc:
            if phase not in flagged_phases:
                errors.append(f"loop_log seq={seq}: {exc}")
                flagged_phases.add(phase)
        completed_by_phase[phase] = stage
        ok_stages_by_phase.setdefault(phase, []).append(stage)
        seq_of[(phase, stage)] = seq if _is_int(seq) else 0

    if loop_stop_seq is not None and entries and entries[-1].get("stage") != TERMINAL_STAGE:
        warnings.append("loop_stop is committed but is not the last log event")

    # ── cross-phase dependency: iterN/gap_analysis reads prev phase's inference ──
    # An iteration may not start until the phase before it is *finished*, not just
    # far enough along to have written labels. A phase abandoned at inference has
    # no kpi_analyze, and its mAP is the number the next iteration is measured
    # against; without it the run's one deliverable — the trend — cannot exist.
    for phase in sorted(set(list(completed_by_phase) + list(iterations)), key=_phase_sort_key):
        previous = previous_phase(phase)
        if previous is None:
            continue
        gap_seq = seq_of.get((phase, "gap_analysis"))
        if gap_seq is None:
            continue
        try:
            # _label_problem already rejected the labels that make this unresolvable;
            # an auditor still never crashes on the file it is auditing.
            final_stage = phase_order(previous)[-1]
        except ValueError as exc:
            errors.append(f"state.iterations.{phase}: {exc}")
            continue
        previous_seq = seq_of.get((previous, "inference"))
        final_seq = seq_of.get((previous, final_stage))
        if previous_seq is None:
            errors.append(
                f"{phase}/gap_analysis was committed (seq={gap_seq}) but {previous}/inference "
                f"never completed; gap_analysis consumes {previous}'s inference labels"
            )
        elif previous_seq > gap_seq:
            errors.append(
                f"{phase}/gap_analysis (seq={gap_seq}) was committed before "
                f"{previous}/inference (seq={previous_seq}); it cannot have read those labels"
            )
        elif final_seq is None:
            errors.append(
                f"{phase}/gap_analysis was committed (seq={gap_seq}) but {previous} never "
                f"finished {final_stage}; {previous} was abandoned mid-phase, so the run has "
                f"no {previous} mAP to measure {phase} against"
            )
        elif final_seq > gap_seq:
            errors.append(
                f"{phase}/gap_analysis (seq={gap_seq}) was committed before {previous} "
                f"finished {final_stage} (seq={final_seq}); {phase} started on an unfinished phase"
            )

    # ── state must agree with the log about what finished ──
    for phase, stage in completed_by_phase.items():
        info = iterations.get(phase)
        if not isinstance(info, dict):
            errors.append(
                f"loop_log commits {phase}/{stage} but state.iterations.{phase} is missing"
            )
            continue
        recorded = info.get("stage_completed")
        if recorded != stage:
            errors.append(
                f"state.iterations.{phase}.stage_completed={recorded!r} disagrees with the "
                f"log, whose last ok stage for {phase} is {stage!r}"
            )
    for phase in sorted(iterations, key=_phase_sort_key):
        info = iterations[phase]
        if not isinstance(info, dict):
            errors.append(f"state.iterations.{phase} must be an object")
            continue
        try:
            phase_order(phase)
        except ValueError as exc:
            errors.append(f"state.iterations: {exc}")
            continue
        label_problem = _label_problem(phase)
        if label_problem:
            errors.append(f"state.iterations: {label_problem}")
            continue
        if _contradictory_zero_weak(info):
            errors.append(
                f"state.iterations.{phase} records zero_weak_images=true alongside "
                f"weak_image_count={info.get('weak_image_count')}; the early stop and the "
                "measurement contradict each other, so neither can be trusted"
            )
        recorded = info.get("stage_completed")
        if recorded is not None and phase not in completed_by_phase:
            errors.append(
                f"state.iterations.{phase}.stage_completed={recorded!r} but loop_log has no "
                f"matching ok event for {phase}"
            )
        number = iter_number(phase)
        if number is not None and max_iterations is not None and number > max_iterations:
            errors.append(
                f"state.iterations.{phase} exceeds state.max_iterations={max_iterations}"
            )

    # ── every recorded artifact must still be on disk, and be the right kind ──
    recorded_artifacts: dict[str, tuple[str, str]] = dict(EXTRA_ARTIFACT_FIELDS)
    for stage, fields in STAGE_ARTIFACTS.items():
        recorded_artifacts.update(fields)
    for phase in sorted(iterations, key=_phase_sort_key):
        info = iterations[phase]
        if not isinstance(info, dict):
            continue
        for field, (flag, kind) in recorded_artifacts.items():
            value = info.get(field)
            if value in (None, ""):
                continue
            if not Path(str(value)).is_absolute():
                errors.append(
                    f"state.iterations.{phase}.{field} ({flag}) must be an absolute "
                    f"path: {value}"
                )
                continue
            problem = check_artifact(str(value), kind)
            if problem:
                errors.append(f"state.iterations.{phase}.{field} ({flag}) {problem}")
        # A stage the log says finished must have recorded its artifact paths.
        for stage in ok_stages_by_phase.get(phase, []):
            for field, (flag, _kind) in STAGE_ARTIFACTS[stage].items():
                if info.get(field) in (None, ""):
                    errors.append(
                        f"loop_log commits {phase}/{stage} but state.iterations.{phase}."
                        f"{field} ({flag}) was never recorded"
                    )

    highest_iteration = max(
        (
            number
            for phase in set(list(iterations) + list(completed_by_phase))
            if (number := iter_number(phase)) is not None
        ),
        default=0,
    )
    if _is_int(current_iteration) and current_iteration != highest_iteration:
        warnings.append(
            f"state.current_iteration={current_iteration} does not match the highest "
            f"iteration on disk ({highest_iteration})"
        )

    # ── run-level status ──
    error_events = [event for event in entries if event.get("status") == "error"]
    run_failed = bool(error_events) or run_status == "failed"
    if error_events and run_status is not None and run_status != "failed":
        warnings.append(
            f"loop_log records a status=error commit but state.status is {run_status!r}"
        )
    if run_status == "failed" and not error_events:
        warnings.append(
            "state.status is 'failed' but no loop_log event has status=error; the failure "
            "was never recorded as a stage, so nothing on disk says what failed"
        )
    if entries and not (_pool_on_disk(state) or _prep_produced_pool(iterations)):
        pool_config = state.get("config")
        pool_config = pool_config if isinstance(pool_config, dict) else {}
        missing = [
            str(pool_config.get(field))
            for field, kind in POOL_CONFIG_FIELDS.items()
            if not (
                pool_config.get(field)
                and check_artifact(str(pool_config[field]), kind) is None
            )
        ]
        warnings.append(
            "the source pool `mine` reads is not on disk and no prep stage produced it: "
            f"{missing}; prep must run before any iteration reaches `mine`"
        )

    status = "INVALID" if errors else "VALID"
    # Terminal means finalized, and only loop_stop finalizes. A failed run that
    # never committed loop_stop still owes that commit — next_action says so — and
    # a report rendered before it has no authoritative outcome to render.
    terminal = loop_stop_committed

    iterations_completed = sum(
        1
        for phase, stage in completed_by_phase.items()
        if iter_number(phase) is not None and stage == ITER_ORDER[-1]
    )

    complete = False
    completion_reason = ""
    if not loop_stop_committed:
        completion_reason = "no loop_stop event is committed"
    elif error_events:
        completion_reason = "the run committed a status=error hard stop"
    elif run_status == "failed":
        completion_reason = "state.status is 'failed'"
    elif (
        completed_by_phase.get("baseline") == BASELINE_ORDER[-1]
        and not isinstance((iterations.get("baseline") or {}).get("map_value"), (int, float))
    ):
        # --map-value is optional, because a log printing "mAP: nan" has none to
        # record. But the baseline mAP is what every iteration is compared against,
        # so a run without one has produced no trend and is not complete.
        completion_reason = (
            "baseline/kpi_analyze committed no map_value, so there is no baseline mAP to "
            "compare iterations against"
        )
    elif completed_by_phase.get("baseline") != BASELINE_ORDER[-1]:
        # The baseline mAP is the number every iteration is compared against.
        # Without it there is no trend, which is the only thing the loop produces.
        completion_reason = (
            f"baseline never finished {BASELINE_ORDER[-1]}, so the run has no baseline mAP "
            "and no measurable trend"
        )
    elif max_iterations is not None and all(
        completed_by_phase.get(f"iter{n}") == ITER_ORDER[-1]
        for n in range(1, max_iterations + 1)
    ):
        complete = True
        completion_reason = (
            f"all {max_iterations} iterations completed every stage of "
            f"{' -> '.join(ITER_ORDER)}"
        )
    else:
        # The early stop only counts on a phase that actually stopped at
        # gap_analysis. A phase that recorded zero weak images and then went on to
        # embed, mine and train did not take the documented exit, and a zero
        # recorded on any other stage is not gap_analysis's measurement at all.
        early = next(
            (
                phase
                for phase in sorted(iterations, key=_phase_sort_key)
                if completed_by_phase.get(phase) == "gap_analysis"
                and _zero_weak_images(iterations[phase])
            ),
            None,
        )
        exhausted = _pool_exhausted(iterations)
        if early is not None:
            complete = True
            completion_reason = (
                f"documented early stop: {early}/gap_analysis recorded zero weak images"
            )
        elif exhausted is not None and iterations_completed >= 1:
            # A spent pool ends a loop that produced something. With no iteration
            # finished there is no trend, and the claim is unfalsifiable — it would
            # let any abandoned run be relabelled as a documented stop.
            complete = True
            completion_reason = (
                f"documented early stop: the source pool was exhausted at {exhausted} "
                f"(pool_remaining={iterations[exhausted].get('pool_remaining', 'unrecorded')}), "
                f"so no further iteration could mine"
            )
        else:
            completion_reason = (
                f"only {iterations_completed} of {max_iterations} iterations finished "
                f"{ITER_ORDER[-1]}, and the loop was stopped by neither documented early "
                f"stop (zero weak images, or a source pool exhausted after at least one "
                f"completed iteration)"
            )
    if status == "INVALID":
        complete = False
        completion_reason = "the run is INVALID"
    elif run_status == "complete" and not complete:
        # commit_stage.py only writes "complete" once this audit has agreed it is
        # one, so a disagreement means the file was written by something else.
        warnings.append(
            f"state.status is 'complete' but this audit withholds completion: "
            f"{completion_reason}"
        )

    if status == "INVALID":
        next_action = REPAIR_ACTION
    else:
        next_action = _next_action(
            state=state,
            iterations=iterations,
            completed_by_phase=completed_by_phase,
            loop_stop_committed=loop_stop_committed,
            run_failed=run_failed,
            complete=complete,
            max_iterations=max_iterations,
        )

    last_event = entries[-1] if entries else None
    last_committed = (
        f"{last_event.get('iter')}/{last_event.get('stage')}" if last_event else "none"
    )

    return {
        "status": status,
        "results_dir": str(results_dir),
        "schema_version": state.get("schema_version"),
        "max_iterations": max_iterations,
        "iterations_completed": iterations_completed,
        "log_entries": len(entries),
        "last_committed": last_committed,
        "last_status": last_event.get("status") if last_event else None,
        "stage_completed_by_phase": completed_by_phase,
        "next_action": next_action,
        "read_before_action": STAGE_OVERLAYS.get(next_action, "none"),
        "terminal": terminal,
        "loop_stop_committed": loop_stop_committed,
        "run_failed": run_failed,
        "complete": complete,
        "completion_reason": completion_reason,
        "load_failed": load_failed,
        "errors": errors,
        "warnings": warnings,
    }


def _print_text(report: dict[str, Any]) -> None:
    print(f"DEFT_RUN_STATUS={report['status']}")
    print(f"results_dir={report['results_dir']}")
    max_iterations = report["max_iterations"]
    print(
        f"iterations_completed={report['iterations_completed']}/"
        f"{max_iterations if max_iterations is not None else 'unknown'}"
    )
    print(f"last_committed={report['last_committed']}")
    print(f"next_action={report['next_action']}")
    print(f"read_before_action={report['read_before_action']}")
    print(f"terminal={str(report['terminal']).lower()}")
    print(f"complete={str(report['complete']).lower()}")
    # Printed on success as well as failure: "complete" reached by finishing every
    # iteration and "complete" reached by a documented early stop are different
    # runs, and the reason is the only thing that tells them apart.
    print(f"completion_reason={report['completion_reason']}")
    sys.stdout.flush()  # keep the key=value block ahead of stderr when piped
    for warning in report["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    for error in report["errors"]:
        print(f"error: {error}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Absolute path to the run directory holding deft_state.json.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON object instead of key=value lines.",
    )
    parser.add_argument(
        "--require-terminal",
        action="store_true",
        help=(
            "Also require the run to be finalized: loop_stop committed. A properly "
            "finalized FAILED run passes; a run that failed and never committed "
            "loop_stop does not."
        ),
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "Also require genuine completion: a committed loop_stop plus either "
            "max_iterations full iterations, or a documented early stop: zero weak\n"
            "images, or an exhausted source pool."
        ),
    )
    args = parser.parse_args()

    try:
        report = audit(Path(args.results_dir))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_text(report)

    if report["load_failed"]:
        return 2
    if report["status"] == "INVALID":
        return 1
    if args.require_terminal and not report["terminal"]:
        print(
            "error: --require-terminal: the run is not finalized; no loop_stop is "
            f"committed. next_action={report['next_action']}",
            file=sys.stderr,
        )
        return 1
    if args.require_complete and not report["complete"]:
        print(
            f"error: --require-complete: {report['completion_reason']}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
