#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable per-run state for the showcase.

A run writes one manifest early and updates it at every lifecycle transition, so
an interrupted run can still be described and stopped precisely. Cleanup reads
the manifest instead of guessing which containers or processes belong to the
run, which keeps a stop scoped to resources this run recorded.

Motion evidence is recorded when it is observed. A later status command loads
that record rather than resampling a stationary robot and concluding the robot
never moved.
"""

import json
import os
import tempfile
from pathlib import Path

MANIFEST_NAME = "run-manifest.json"
RESULT_NAME = "run-result.json"
MANIFEST_SCHEMA = "mission-control-showcase/run-manifest@1"
RESULT_SCHEMA = "mission-control-showcase/run-result@1"

# Lifecycle states, in order. A run only moves forward.
STATES = (
    "initializing",
    "preflight",
    "starting-cloud",
    "starting-isaac",
    "starting-carter",
    "ready",
    "mission-running",
    "accepted",
    "failed",
    "stopped",
)


def _atomic_write(path: Path, payload: dict) -> None:
    """Write JSON so a reader never observes a partial manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def manifest_path(work_dir) -> Path:
    return Path(work_dir) / MANIFEST_NAME


def result_path(work_dir) -> Path:
    return Path(work_dir) / RESULT_NAME


def load(work_dir) -> dict:
    """Return the manifest, or an empty dict when the run has none."""
    path = manifest_path(work_dir)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def initialize(work_dir, **fields) -> dict:
    """Create the manifest at the start of a run. Never clobbers evidence."""
    existing = load(work_dir)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "state": "initializing",
        "work_dir": str(work_dir),
        "containers": [],
        "isaac": {},
        "ports": {},
        "evidence": {},
        "motion": {},
        "acceptance": {},
    }
    manifest.update(existing)
    manifest.update({k: v for k, v in fields.items() if v is not None})
    manifest.setdefault("run_id", Path(work_dir).name)
    _atomic_write(manifest_path(work_dir), manifest)
    return manifest


VERDICT_STATES = frozenset({"accepted", "failed"})


def _check_transition(current, requested) -> None:
    """Reject a lifecycle write that would lose or contradict the record.

    STATES is ordered and a run only moves forward. Re-writing the current
    state is allowed so callers stay idempotent. Cleanup is always allowed:
    "stopped" is reachable from any state, because a run is routinely stopped
    after it was accepted or failed. What is refused is overwriting a recorded
    verdict with anything other than cleanup, resuming from "stopped", and
    moving backward through the startup stages, each of which would corrupt
    the record that run-status and cleanup read.
    """
    if requested is None:
        return
    if requested not in STATES:
        raise ValueError(f"unknown lifecycle state: {requested}")
    if current is None or current == requested:
        return
    if current not in STATES:
        raise ValueError(f"unknown recorded lifecycle state: {current}")
    if requested == "stopped":
        return
    if current == "stopped":
        raise ValueError(f"cannot resume a stopped run as {requested!r}")
    if current in VERDICT_STATES:
        raise ValueError(
            f"cannot overwrite recorded verdict {current!r} with {requested!r}"
        )
    if STATES.index(requested) < STATES.index(current):
        raise ValueError(
            f"cannot move lifecycle state backward: {current!r} -> {requested!r}"
        )


def update(work_dir, **fields) -> dict:
    """Merge fields into the manifest atomically, enforcing the lifecycle."""
    manifest = load(work_dir) or {"schema": MANIFEST_SCHEMA, "work_dir": str(work_dir)}
    _check_transition(manifest.get("state"), fields.get("state"))
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(manifest.get(key), dict):
            manifest[key] = {**manifest[key], **value}
        else:
            manifest[key] = value
    _atomic_write(manifest_path(work_dir), manifest)
    return manifest


def set_state(work_dir, state: str) -> dict:
    if state not in STATES:
        raise ValueError(f"unknown lifecycle state: {state}")
    return update(work_dir, state=state)


def add_container(work_dir, name: str) -> dict:
    """Record a container this run owns. Only recorded names may be stopped."""
    manifest = load(work_dir)
    names = list(manifest.get("containers") or [])
    if name and name not in names:
        names.append(name)
    return update(work_dir, containers=names)


def record_motion(work_dir, evidence: dict) -> dict:
    """Persist motion evidence, never downgrading a confirmed observation.

    Motion is monotonic: a robot that moved cannot later have not moved. A
    stationary follow-up sample may raise the maxima but must not clear
    `motion_confirmed` or discard the pose that proved it.
    """
    manifest = load(work_dir)
    prior = manifest.get("motion") or {}
    merged = {**prior, **{k: v for k, v in evidence.items() if v is not None}}
    merged["max_translation_m"] = max(
        float(prior.get("max_translation_m") or 0.0),
        float(evidence.get("max_translation_m") or 0.0),
    )
    merged["max_heading_rad"] = max(
        float(prior.get("max_heading_rad") or 0.0),
        float(evidence.get("max_heading_rad") or 0.0),
    )
    merged["motion_confirmed"] = bool(
        prior.get("motion_confirmed") or evidence.get("motion_confirmed")
    )
    if prior.get("moving_pose") is not None:
        merged["moving_pose"] = prior["moving_pose"]
    return update(work_dir, motion=merged)


def motion_status(work_dir, sampled: dict | None = None) -> dict:
    """Classify motion, separating a quiet sample from an absence of evidence.

    Returns one of:
      confirmed-historical  evidence recorded earlier proves motion
      confirmed-sample      this sample alone proves motion
      none-observed         a sample ran and saw no motion, no prior evidence
      no-evidence           nothing has ever been observed
    """
    stored = (load(work_dir) or {}).get("motion") or {}
    if stored.get("motion_confirmed"):
        return {
            "verdict": "confirmed-historical",
            "motion_confirmed": True,
            "source": "run-manifest",
            "max_translation_m": stored.get("max_translation_m"),
            "max_heading_rad": stored.get("max_heading_rad"),
        }
    if sampled and sampled.get("motion_confirmed"):
        return {
            "verdict": "confirmed-sample",
            "motion_confirmed": True,
            "source": "live-sample",
            "max_translation_m": sampled.get("max_translation_m"),
            "max_heading_rad": sampled.get("max_heading_rad"),
        }
    if sampled is not None:
        return {
            "verdict": "none-observed",
            "motion_confirmed": False,
            "source": "live-sample",
            "detail": "no motion during this sample; no earlier evidence recorded",
        }
    return {
        "verdict": "no-evidence",
        "motion_confirmed": False,
        "source": "none",
        "detail": "no motion evidence has been recorded for this run",
    }


def record_isaac_exit(work_dir, pid, returncode, reason: str) -> dict:
    """Record an observed Isaac Sim exit. Absence of this means 'not observed'."""
    return update(
        work_dir,
        isaac={
            "pid": pid,
            "exit_status": returncode,
            "exit_reason": reason,
            "running": False,
        },
    )


def write_result(work_dir, **fields) -> Path:
    """Emit the machine-readable acceptance artifact for this run."""
    manifest = load(work_dir)
    motion = manifest.get("motion") or {}
    result = {
        "schema": RESULT_SCHEMA,
        "run_id": manifest.get("run_id"),
        "work_dir": str(work_dir),
        "mission_id": manifest.get("mission_id"),
        "started_at": manifest.get("started_at"),
        "state": manifest.get("state"),
        "map": manifest.get("map"),
        "robot": manifest.get("robot"),
        "ports": manifest.get("ports"),
        "ros_domain_id": manifest.get("ros_domain_id"),
        "compose_project": manifest.get("compose_project"),
        "containers": manifest.get("containers"),
        "isaac": manifest.get("isaac"),
        "evidence": manifest.get("evidence"),
        "motion": {
            "initial_pose": motion.get("initial_pose"),
            "moving_pose": motion.get("moving_pose"),
            "final_pose": motion.get("final_pose"),
            "max_translation_m": motion.get("max_translation_m"),
            "max_heading_rad": motion.get("max_heading_rad"),
            "motion_confirmed": bool(motion.get("motion_confirmed")),
        },
        "acceptance": manifest.get("acceptance"),
    }
    result.update({k: v for k, v in fields.items() if v is not None})
    path = result_path(work_dir)
    _atomic_write(path, result)
    return path


def accept(work_dir, *, mission_state, robot_state, robot_healthy, oom, processes_healthy):
    """Apply the canonical acceptance criteria and record the verdict.

    Acceptance is strict and never weakened: the mission must be COMPLETED, the
    robot online, healthy and IDLE, motion must be proven, no OOM may have
    occurred, and required processes must be healthy.
    """
    motion = motion_status(work_dir)
    reasons = []
    if mission_state != "COMPLETED":
        reasons.append(f"mission state is {mission_state!r}, expected 'COMPLETED'")
    if robot_state != "IDLE":
        reasons.append(f"robot state is {robot_state!r}, expected 'IDLE'")
    if not robot_healthy:
        reasons.append("robot reported errors or was offline")
    if not motion["motion_confirmed"]:
        reasons.append(f"motion not confirmed ({motion['verdict']})")
    if oom:
        reasons.append("an OOM event was recorded during the run")
    if not processes_healthy:
        reasons.append("required Nova Carter or Nav2 processes were unhealthy")
    verdict = {
        "accepted": not reasons,
        "reasons": reasons,
        "motion": motion,
        "mission_state": mission_state,
        "robot_state": robot_state,
        "oom": bool(oom),
        "processes_healthy": bool(processes_healthy),
    }
    update(work_dir, acceptance=verdict, state="accepted" if not reasons else "failed")
    return verdict


def memory_policy(mem_total, mem_available, swap_free, min_available, min_total):
    """Decide whether host memory suffices, treating swap as a fallback only.

    Swap compensates for a shortfall in RAM; it is not required when RAM is
    already sufficient. A host with abundant available memory and no swap is
    healthy, so zero swap alone is never a failure.

    swap_free is free swap, not configured capacity. Swap another process has
    already consumed cannot cover this run's shortfall, so counting it would
    credit bytes that are not reclaimable.
    """
    effective = mem_available + swap_free
    if mem_available >= min_available:
        return {
            "status": "ok",
            "effective_bytes": effective,
            "detail": "available memory satisfies the showcase without relying on swap",
        }
    if effective >= min_available:
        return {
            "status": "warn",
            "effective_bytes": effective,
            "detail": (
                "available memory is below the requirement and the shortfall is "
                "covered by swap; expect paging under load"
            ),
        }
    return {
        "status": "fail",
        "effective_bytes": effective,
        "detail": (
            "available memory plus swap is below the requirement; free memory or "
            "authorize adding swap before running the showcase"
        ),
    }
