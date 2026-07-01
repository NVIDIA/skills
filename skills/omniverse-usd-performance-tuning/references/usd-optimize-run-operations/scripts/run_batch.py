#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parallel execution harness — resource-aware per-target scheduler.

Turns the Phase-4b batch-mode prose into a real, deterministic runner. The
*agent* still decides max_workers, archetype tags, which op chains, and
intent-gated opt-ins; this runner owns the mechanics it must own deterministically:
spawning standalone single-asset workers, dependency ordering, per-target/per-op
timeouts (killable subprocess), the GPU-cliff guard, status emission, and resume.

Key contracts (see ``status.schema.json`` and
``skills/.../usd-optimize-run-operations/references/batch-mode.md``):

* **The status artifact is the contract.** Everything else (CLI bars, a future
  web dashboard) is a *view* over ``status.json``.
* **``state: done`` is NOT coverage ``disposition: optimized``.** Worker
  completion proves the op ran, not that it changed anything. The runner records
  per-target before/after deltas (when the worker emits a summary) and derives a
  ``disposition`` from those deltas — it never assumes ``done`` ⇒ ``optimized``.
* **Two safety behaviors are encoded:** per-target ``subprocess.run(timeout=...)``
  (a hung worker is killed without stalling the batch), and a GPU-cliff guard
  that skips/warns ``gpu_bound`` targets on a CPU-only host (CUDA read from the
  setup preflight ``runtime_context``, never from SO's own ``hasNvidiaGpu()``).
* **``--resume`` replaces the improvised remainder script:** resuming off
  ``status.json`` re-runs only the unfinished targets.

Workers are standalone single-asset processes (each target file is independent;
optimization/validation never uses Kit; standalone is the sole optimization
runtime). The opt-in Kit->omniperf profiling path is capped separately and sits
outside this fan-out.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as _dt
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

SHARED_FIRST = "shared_first"
DEPENDENT_AFTER = "dependent_after"
INDEPENDENT = "independent"

# Target states.
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
TIMEOUT = "timeout"
SKIPPED_GPU = "skipped_gpu_unavailable"

# Coverage dispositions derived from the before/after outcome (NOT from state).
DISP_OPTIMIZED = "optimized"
DISP_NO_OP = "no_op"
DISP_UNKNOWN = "unknown"

_TERMINAL_STATES = {DONE, SKIPPED_GPU}


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _cuda_available_from_preflight(preflight_path: Path) -> bool:
    """Read the CUDA signal from the setup preflight ``runtime_context``.

    The plan is explicit: detect CUDA from setup preflight / ``runtime_context``,
    NOT from Usd Optimize's own ``hasNvidiaGpu()``. We look for an explicit
    boolean and fail closed (treat as unavailable) when the signal is absent, so
    a CPU-only/WSL host never silently enters a long GPU fallback.
    """
    try:
        data = _read_json(preflight_path)
    except (OSError, ValueError):
        return False
    rc = data.get("runtime_context", data)
    for key in ("cudaAvailable", "cuda_available", "hasCuda", "gpu_available"):
        if isinstance(rc.get(key), bool):
            return rc[key]
    gpu = rc.get("gpu")
    if isinstance(gpu, dict) and isinstance(gpu.get("available"), bool):
        return gpu["available"]
    return False


def _derive_disposition(summary_path: str | None) -> tuple[str, dict | None, dict | None]:
    """Read a worker summary and derive a coverage disposition from deltas.

    ``done`` only means the worker exited 0. A target is ``optimized`` only when
    a real before/after change is recorded; ``mesh_count``/``tris`` unchanged is
    ``no_op`` (the no-op-masquerade guard). Missing summary => ``unknown``, which
    keeps the coverage gate honest rather than fabricating an ``optimized`` mark.
    """
    if not summary_path:
        return DISP_UNKNOWN, None, None
    p = Path(summary_path)
    if not p.is_file():
        return DISP_UNKNOWN, None, None
    try:
        s = _read_json(p)
    except (OSError, ValueError):
        return DISP_UNKNOWN, None, None
    before = s.get("before")
    after = s.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return DISP_UNKNOWN, before if isinstance(before, dict) else None, after if isinstance(after, dict) else None
    changed = any(before.get(k) != after.get(k) for k in set(before) | set(after))
    return (DISP_OPTIMIZED if changed else DISP_NO_OP), before, after


class _Target:
    def __init__(self, spec: dict, index: int):
        self.spec = spec
        self.index = index
        self.path = spec.get("path", f"target_{index}")
        self.role = spec.get("role") or spec.get("target_class") or "monolith"
        self.level = spec.get("level")
        self.archetype = spec.get("archetype")
        self.dep_group = spec.get("dep_group") or spec.get("dependency_group") or INDEPENDENT
        self.gpu_bound = bool(spec.get("gpu_bound", False))
        self.command = spec.get("command") or []
        self.timeout_sec = spec.get("timeout_sec")
        self.summary_path = spec.get("summary_path")
        self.log_path = spec.get("log_path")
        self.state = QUEUED
        self.disposition = DISP_UNKNOWN
        self.started: str | None = None
        self.ended: str | None = None
        self.duration_seconds: float | None = None
        self.exit_code: int | None = None
        self.error: str | None = None
        self.steps_applied: list[str] = list(spec.get("steps_applied", []))
        self.before: dict | None = None
        self.after: dict | None = None

    def to_status(self) -> dict:
        return {
            "path": self.path,
            "role": self.role,
            "level": self.level,
            "archetype": self.archetype,
            "dep_group": self.dep_group,
            "gpu_bound": self.gpu_bound,
            "state": self.state,
            "disposition": self.disposition,
            "steps_applied": self.steps_applied,
            "started": self.started,
            "ended": self.ended,
            "duration_seconds": self.duration_seconds,
            "exit_code": self.exit_code,
            "log_path": self.log_path,
            "summary_path": self.summary_path,
            "before": self.before,
            "after": self.after,
            "error": self.error,
        }


class BatchScheduler:
    def __init__(
        self,
        plan: dict,
        status_path: Path,
        max_workers: int = 4,
        cuda_available: bool = False,
        resume: bool = False,
        progress: bool = False,
    ):
        self.run_id = plan.get("run_id") or "batch"
        self.status_path = status_path
        self.max_workers = max(1, int(max_workers))
        self.cuda_available = cuda_available
        self.progress = progress
        self.started = _utcnow()
        self._lock = threading.Lock()
        self.targets = [_Target(t, i) for i, t in enumerate(plan.get("targets", []))]
        if resume:
            self._apply_resume()

    # -- resume -----------------------------------------------------------
    def _apply_resume(self) -> None:
        if not self.status_path.is_file():
            return
        try:
            prev = _read_json(self.status_path)
        except (OSError, ValueError):
            return
        prev_by_path = {t.get("path"): t for t in prev.get("targets", [])}
        for tgt in self.targets:
            done = prev_by_path.get(tgt.path)
            if done and done.get("state") in _TERMINAL_STATES:
                # Carry the finished result forward; do not re-run.
                tgt.state = done["state"]
                tgt.disposition = done.get("disposition", tgt.disposition)
                tgt.started = done.get("started")
                tgt.ended = done.get("ended")
                tgt.before = done.get("before")
                tgt.after = done.get("after")
                tgt.steps_applied = done.get("steps_applied", tgt.steps_applied)

    # -- execution --------------------------------------------------------
    def _run_one(self, tgt: _Target) -> None:
        # GPU-cliff guard: never enter a long CPU fallback for a gpu_bound op.
        if tgt.gpu_bound and not self.cuda_available:
            with self._lock:
                tgt.state = SKIPPED_GPU
                tgt.error = "gpu_bound op skipped: no CUDA in runtime_context (CPU-only host)"
            self._write_status()
            return
        if not tgt.command:
            with self._lock:
                tgt.state = FAILED
                tgt.error = "no command specified for target"
            self._write_status()
            return
        with self._lock:
            tgt.state = RUNNING
            tgt.started = _utcnow()
        self._write_status()
        stdout_target = None
        rc: int | None = None
        timed_out = False
        start = time.monotonic()
        try:
            if tgt.log_path:
                Path(tgt.log_path).parent.mkdir(parents=True, exist_ok=True)
                stdout_target = open(tgt.log_path, "w", encoding="utf-8")
            proc = subprocess.run(
                tgt.command,
                timeout=tgt.timeout_sec,
                stdout=stdout_target or subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            rc = proc.returncode
            err = None if rc == 0 else f"worker exited with code {rc}"
        except subprocess.TimeoutExpired:
            timed_out = True
            err = f"timeout after {tgt.timeout_sec}s (worker killed)"
        except (OSError, ValueError) as exc:  # spawn failure
            err = f"spawn error: {exc}"
        finally:
            if stdout_target is not None:
                stdout_target.close()
        duration = round(time.monotonic() - start, 3)
        with self._lock:
            tgt.ended = _utcnow()
            tgt.duration_seconds = duration
            tgt.exit_code = rc
            if err is None:
                tgt.state = DONE
                disp, before, after = _derive_disposition(tgt.summary_path)
                tgt.disposition = disp
                tgt.before = before
                tgt.after = after
            elif timed_out:
                # A killed-by-timeout worker is a distinct outcome from a worker
                # that ran to a non-zero exit — the per-rule-timeout hang this
                # scheduler exists to bound. Surface it as its own state, not generic failed.
                tgt.state = TIMEOUT
                tgt.error = err
            else:
                tgt.state = FAILED
                tgt.error = err
        self._write_status()

    def _run_group(self, group: list[_Target]) -> None:
        pending = [t for t in group if t.state not in _TERMINAL_STATES]
        if not pending:
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(self._run_one, t) for t in pending]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()

    def run(self) -> dict:
        # Prototype-first ordering: shared_first group runs to completion before
        # dependent targets, so prototype changes propagate. Independent targets
        # ride with the dependent wave.
        shared = [t for t in self.targets if t.dep_group == SHARED_FIRST]
        rest = [t for t in self.targets if t.dep_group != SHARED_FIRST]
        self._write_status()
        self._run_group(shared)
        self._run_group(rest)
        self._write_status()
        return self.snapshot()

    # -- status -----------------------------------------------------------
    def _overall(self) -> dict:
        counts = {"total": len(self.targets), "done": 0, "failed": 0,
                  "timeout": 0, "running": 0, "queued": 0, "skipped": 0}
        for t in self.targets:
            if t.state == DONE:
                counts["done"] += 1
            elif t.state == FAILED:
                counts["failed"] += 1
            elif t.state == TIMEOUT:
                counts["timeout"] += 1
            elif t.state == RUNNING:
                counts["running"] += 1
            elif t.state == SKIPPED_GPU:
                counts["skipped"] += 1
            else:
                counts["queued"] += 1
        return counts

    def snapshot(self) -> dict:
        return {
            "run_id": self.run_id,
            "started": self.started,
            "max_workers": self.max_workers,
            "cuda_available": self.cuda_available,
            "overall": self._overall(),
            "targets": [t.to_status() for t in self.targets],
        }

    def _write_status(self) -> None:
        # Serialize writes: concurrent workers all emit status, and an atomic
        # replace from a shared temp path would race. Hold the lock and use a
        # per-write unique temp name.
        with self._lock:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            snap = self.snapshot()
            tmp = self.status_path.with_name(
                f"{self.status_path.name}.{threading.get_ident()}.tmp"
            )
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(snap, fh, indent=2)
            tmp.replace(self.status_path)
        if self.progress:
            o = snap["overall"]
            sys.stderr.write(
                f"\r[{self.run_id}] done {o['done']}/{o['total']} "
                f"failed {o['failed']} timeout {o['timeout']} "
                f"skipped {o['skipped']} running {o['running']}  "
            )
            sys.stderr.flush()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Resource-aware per-target batch scheduler.")
    p.add_argument("--plan", required=True, help="Batch plan JSON (targets[] with command/timeout/dep_group/gpu_bound).")
    p.add_argument("--status", help="status.json output path (default: <plan dir>/status.json).")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--resume", action="store_true", help="Reuse an existing status.json; re-run only unfinished targets.")
    p.add_argument("--progress", action="store_true", help="Print a compact progress line to stderr.")
    cuda = p.add_mutually_exclusive_group()
    cuda.add_argument("--cuda", choices=["available", "unavailable"], help="Explicit CUDA signal.")
    cuda.add_argument("--preflight", help="setup-preflight.json to read the runtime_context CUDA signal from.")
    return p


def main(argv: list[str]) -> int:
    args = _build_arg_parser().parse_args(argv)
    plan_path = Path(args.plan)
    plan = _read_json(plan_path)
    status_path = Path(args.status) if args.status else plan_path.parent / "status.json"

    if args.cuda is not None:
        cuda_available = args.cuda == "available"
    elif args.preflight:
        cuda_available = _cuda_available_from_preflight(Path(args.preflight))
    else:
        cuda_available = False  # fail closed: no signal => treat host as CPU-only

    sched = BatchScheduler(
        plan,
        status_path=status_path,
        max_workers=args.max_workers,
        cuda_available=cuda_available,
        resume=args.resume,
        progress=args.progress,
    )
    snap = sched.run()
    if args.progress:
        sys.stderr.write("\n")
    o = snap["overall"]
    # Non-zero exit when any target failed or timed out, so CI / the agent notices.
    return 1 if (o["failed"] or o.get("timeout")) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
