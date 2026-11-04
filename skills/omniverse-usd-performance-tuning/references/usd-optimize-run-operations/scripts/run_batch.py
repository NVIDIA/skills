#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run standalone workers by definition layer, with bounded concurrency.

Jobs start when their explicit dependencies have verified results. Failed work
blocks only its consumers; unrelated jobs can still run. Completion alone
does not establish optimization or preservation.

Resume requires unchanged input/dependency and runtime files, parameters, and
written outputs. Plans without the file inventories still execute, but their
previous results cannot be reused. See operations.md for plan preparation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as _dt
import json
import math
import hashlib
import importlib.util
import os
import shutil
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
BLOCKED = "blocked_dependency"
SKIPPED_GPU = "skipped_gpu_unavailable"

# Coverage dispositions derived from the before/after outcome (NOT from state).
DISP_OPTIMIZED = "optimized"
DISP_NO_OP = "no_op"
DISP_UNKNOWN = "unknown"
DISP_UNCHANGED = "verified_unchanged"

_TERMINAL_STATES = {DONE, FAILED, TIMEOUT, SKIPPED_GPU, BLOCKED}


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


def _summary_stamp(path):
    try:
        stat = Path(path).stat() if path else None
        return (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size) if stat else None
    except OSError:
        return None


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


class _FingerprintCache:
    """Share content hashes within this run; metadata changes invalidate a read."""
    def __init__(self):
        self._values = {}
        self._lock = threading.Lock()
        self.reads = 0
        self.hits = 0

    def digest(self, path):
        resolved = str(Path(path).resolve())
        with self._lock:
            stat = Path(resolved).stat()
            key = (resolved, stat.st_dev, stat.st_ino, stat.st_size,
                   stat.st_mtime_ns, stat.st_ctime_ns)
            if key in self._values:
                self.hits += 1
                return [resolved, self._values[key]]
            with Path(resolved).open('rb') as stream:
                value = hashlib.file_digest(stream, 'sha256').hexdigest()
            after = Path(resolved).stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise OSError('file changed while fingerprinting: ' + resolved)
            self._values[key] = value
            self.reads += 1
            return [resolved, value]


def _file_hashes(paths, cache=None):
    if not isinstance(paths, list) or not paths or any(not isinstance(p, str) for p in paths):
        return None
    cache = cache or _FingerprintCache()
    try:
        return [cache.digest(path) for path in sorted(set(paths))]
    except OSError:
        return None


def _resume_fingerprint(spec, cuda_available, cache=None):
    inputs = _file_hashes(spec.get('input_files'), cache)
    runtime = _file_hashes(spec.get('runtime_files'), cache)
    executable = shutil.which(spec.get('command', [''])[0]) if spec.get('command') else None
    if inputs is None or runtime is None or not executable:
        return None
    executable_hash = _file_hashes([executable, __file__], cache)
    if executable_hash is None:
        return None
    payload = {'spec': spec, 'inputs': inputs, 'runtime': runtime,
               'executable': executable_hash, 'cuda_available': cuda_available,
               'cwd': str(Path.cwd()),
               'environment': {k: os.environ.get(k) for k in (
                   'PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'PXR_PLUGINPATH_NAME', 'USD_PLUGIN_PATH')}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _output_fingerprint(spec, cache=None):
    paths = spec.get('output_files')
    if not isinstance(paths, list) or not paths:
        return None
    return _file_hashes(paths + ([spec['summary_path']] if spec.get('summary_path') else []), cache)


def _definition_results(summary_path, expected):
    """Validate the worker's evidence contract, not geometry in the scheduler."""
    try:
        summary = _read_json(Path(summary_path)) if summary_path else {}
    except (OSError, ValueError, TypeError):
        summary = {}
    results = summary.get('definition_results') if isinstance(summary, dict) else None
    errors = []
    if not isinstance(results, list) or any(not isinstance(r, dict) for r in results):
        return [], ['missing per-definition verification results']
    ids = [r.get('definition_id') for r in results]
    if len(ids) != len(expected) or sorted(str(i) for i in ids) != sorted(expected):
        errors.append('definition results do not match writer ownership')
    helper = Path(__file__).resolve().parents[2] / 'optimization-report/scripts/structure_conformance.py'
    spec = importlib.util.spec_from_file_location('scheduler_conformance', helper)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    for result in results:
        name = str(result.get('definition_id', '<missing>'))
        errors.extend(name + ': ' + message for message in
                      checker.check_definition_result(result, require_categories=True))
    return results, errors


class _Target:
    def __init__(self, spec: dict, index: int):
        self.spec = spec
        self.index = index
        self.path = spec.get("path", f"target_{index}")
        self.id = spec.get("id", self.path)
        self.depends_on = spec.get("depends_on", [])
        self.definition_ids = spec.get("definition_ids", [])
        self.definition_results = []
        self.result_errors = []
        self.decisions_required = []
        self.require_verified_results = bool(spec.get("require_verified_results", False))
        self.resources = spec.get("resources", {})
        self.role = spec.get("role") or spec.get("target_class") or "monolith"
        self.level = spec.get("level")
        self.archetype = spec.get("archetype")
        self.dep_group = spec.get("dep_group") or spec.get("dependency_group") or INDEPENDENT
        self.gpu_bound = bool(spec.get("gpu_bound", False))
        self.command = spec.get("command") or []
        self.timeout_sec = spec.get("timeout_sec")
        self.summary_path = spec.get("summary_path")
        self.log_path = spec.get("log_path")
        self.resume_fingerprint = None
        self.output_fingerprint = None
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
            "id": self.id,
            "depends_on": self.depends_on,
            "definition_ids": self.definition_ids,
            "definition_results": self.definition_results,
            "result_errors": self.result_errors,
            "decisions_required": self.decisions_required,
            "resources": self.resources,
            "resume_fingerprint": self.resume_fingerprint,
            "output_fingerprint": self.output_fingerprint,
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
        self._hash_cache = _FingerprintCache()
        self.resource_limits = plan.get('resource_limits', {})
        self.elapsed_seconds = None
        self.peak_running = 0
        self._validate_plan()
        if resume:
            self._apply_resume()

    def _validate_plan(self):
        ids = [t.id for t in self.targets]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('job IDs must be unique nonempty strings')
        self._by_id = {t.id: t for t in self.targets}
        shared = [t.id for t in self.targets if t.dep_group == SHARED_FIRST]
        writers, definitions = {}, set()
        for tgt in self.targets:
            raw_dependencies = tgt.spec.get('depends_on', [])
            if not isinstance(raw_dependencies, list) or any(not isinstance(d, str) for d in raw_dependencies):
                raise ValueError('depends_on must contain job IDs')
            if 'depends_on' not in tgt.spec and tgt.dep_group == DEPENDENT_AFTER:
                tgt.depends_on = list(shared)
            if len(set(tgt.depends_on)) != len(tgt.depends_on):
                raise ValueError('duplicate job dependency: ' + tgt.id)
            if any(dep not in self._by_id for dep in tgt.depends_on):
                raise ValueError('missing dependency for job ' + tgt.id)
            raw_definitions = tgt.spec.get('definition_ids', [])
            if not isinstance(raw_definitions, list) or any(not isinstance(d, str) or not d for d in raw_definitions):
                raise ValueError('definition_ids must contain nonempty strings')
            if tgt.require_verified_results and not tgt.definition_ids:
                raise ValueError('verified jobs must declare definition_ids')
            for definition in tgt.definition_ids:
                if definition in definitions:
                    raise ValueError('definition has more than one writer: ' + definition)
                definitions.add(definition)
            paths = [tgt.path] + tgt.spec.get('output_files', [])
            paths += [tgt.spec[k] for k in ('summary_path', 'log_path') if tgt.spec.get(k)]
            for path in set(str(Path(p).resolve()) for p in paths):
                if path in writers:
                    raise ValueError('workers must not write the same layer or output file: ' + path)
                writers[path] = tgt.id
        for tgt in self.targets:
            if tgt.require_verified_results and any(
                    not self._by_id[d].require_verified_results for d in tgt.depends_on):
                raise ValueError('verified jobs require verified prerequisites: ' + tgt.id)
        # Kahn's traversal also catches self-dependencies before any mutation.
        pending = {t.id: set(t.depends_on) for t in self.targets}
        while pending:
            ready = {key for key, deps in pending.items() if not deps}
            if not ready:
                raise ValueError('cyclic job dependencies')
            pending = {key: deps - ready for key, deps in pending.items() if key not in ready}
        if not isinstance(self.resource_limits, dict):
            raise ValueError('resource_limits must be an object')
        for name, value in self.resource_limits.items():
            self._check_resource(name, value)
        for tgt in self.targets:
            if not isinstance(tgt.resources, dict):
                raise ValueError('resources must be an object')
            for name, value in tgt.resources.items():
                self._check_resource(name, value)
                if name in self.resource_limits and value > self.resource_limits[name]:
                    raise ValueError('job exceeds resource limit: ' + tgt.id + ' ' + name)
            for name in self.resource_limits:
                if name not in tgt.resources:
                    raise ValueError('job must declare bounded resource: ' + tgt.id + ' ' + name)

    @staticmethod
    def _check_resource(name, value):
        if name not in ('memory_mb', 'gpu_slots'):
            raise ValueError('unknown resource: ' + name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('resource must be a finite nonnegative number: ' + name)
        if name == 'gpu_slots' and int(value) != value:
            raise ValueError('gpu_slots must be an integer')

    # -- resume -----------------------------------------------------------
    def _apply_resume(self) -> None:
        if not self.status_path.is_file():
            return
        try:
            prev = _read_json(self.status_path)
        except (OSError, ValueError):
            return
        if not isinstance(prev, dict) or not isinstance(prev.get('targets'), list):
            return
        prev_by_id = {t.get('id', t.get('path')): t for t in prev['targets'] if isinstance(t, dict)}
        for tgt in self.targets:
            done = prev_by_id.get(tgt.id)
            fingerprint = _resume_fingerprint(tgt.spec, self.cuda_available, self._hash_cache)
            outputs = _output_fingerprint(tgt.spec, self._hash_cache)
            if (done and done.get('state') == DONE and fingerprint is not None
                    and outputs is not None and done.get('resume_fingerprint') == fingerprint
                    and done.get('output_fingerprint') == outputs):
                if tgt.require_verified_results:
                    results, errors = _definition_results(tgt.summary_path, tgt.definition_ids)
                    if errors:
                        continue
                for field in ('state', 'disposition', 'started', 'ended', 'before', 'after',
                              'steps_applied', 'duration_seconds', 'exit_code',
                              'resume_fingerprint', 'output_fingerprint', 'result_errors'):
                    setattr(tgt, field, done.get(field, getattr(tgt, field)))
                if tgt.require_verified_results:
                    tgt.definition_results = results
                else:
                    tgt.definition_results = done.get("definition_results", [])
        # A changed prerequisite invalidates only its transitive consumers.
        changed = True
        while changed:
            changed = False
            for tgt in self.targets:
                if tgt.state == DONE and any(self._by_id[d].state != DONE for d in tgt.depends_on):
                    tgt.state = QUEUED
                    tgt.disposition = DISP_UNKNOWN
                    tgt.before = tgt.after = None
                    tgt.definition_results = []
                    changed = True

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
        tgt.resume_fingerprint = _resume_fingerprint(tgt.spec, self.cuda_available, self._hash_cache)
        # Require a summary written by this invocation; leave earlier artifacts intact.
        summary_before = _summary_stamp(tgt.summary_path)
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
                if _summary_stamp(tgt.summary_path) == summary_before:
                    disp, before, after = DISP_UNKNOWN, None, None
                if tgt.require_verified_results:
                    results, errors = _definition_results(tgt.summary_path, tgt.definition_ids)
                    if _summary_stamp(tgt.summary_path) == summary_before:
                        errors.append('worker did not write fresh verification results')
                    if tgt.spec.get('output_files') and _output_fingerprint(tgt.spec, self._hash_cache) is None:
                        errors.append('required output files are missing')
                    tgt.definition_results, tgt.result_errors = results, errors
                    if errors:
                        tgt.state = FAILED
                        tgt.error = '; '.join(errors)
                        disp = DISP_UNKNOWN
                    else:
                        disp = (DISP_OPTIMIZED if any(r['disposition'] == DISP_OPTIMIZED for r in results)
                                else DISP_UNCHANGED)
                tgt.disposition = disp
                tgt.before = before
                tgt.after = after
                tgt.output_fingerprint = _output_fingerprint(tgt.spec, self._hash_cache)
                if tgt.resume_fingerprint != _resume_fingerprint(tgt.spec, self.cuda_available, self._hash_cache):
                    tgt.resume_fingerprint = None
            elif timed_out:
                # A killed-by-timeout worker is a distinct outcome from a worker
                # that ran to a non-zero exit — the per-rule-timeout hang this
                # scheduler exists to bound. Surface it as its own state, not generic failed.
                tgt.state = TIMEOUT
                tgt.error = err
            else:
                tgt.state = FAILED
                tgt.error = err
            if err is not None and tgt.require_verified_results and _summary_stamp(tgt.summary_path) != summary_before:
                tgt.definition_results, tgt.result_errors = _definition_results(tgt.summary_path, tgt.definition_ids)
                try:
                    summary = _read_json(Path(tgt.summary_path))
                    decisions = summary.get('decisions_required', []) if isinstance(summary, dict) else []
                    if isinstance(decisions, list):
                        tgt.decisions_required = decisions
                except (OSError, ValueError, TypeError):
                    pass
        self._write_status()

    def run(self) -> dict:
        start = time.monotonic()
        self._write_status()
        running = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while True:
                # Propagate a failed prerequisite before selecting new ready work.
                for tgt in self.targets:
                    if tgt.state != QUEUED:
                        continue
                    failed = [d for d in tgt.depends_on
                              if self._by_id[d].state in _TERMINAL_STATES and self._by_id[d].state != DONE]
                    if failed:
                        tgt.state = BLOCKED
                        tgt.error = 'prerequisite did not verify: ' + ', '.join(failed)
                submitted = {t.id for t in running.values()}
                for tgt in self.targets:
                    if len(running) >= self.max_workers:
                        break
                    if tgt.state != QUEUED or tgt.id in submitted:
                        continue
                    if any(self._by_id[d].state != DONE for d in tgt.depends_on):
                        continue
                    if any(sum(t.resources.get(name, 0) for t in running.values()) + tgt.resources.get(name, 0) > limit
                           for name, limit in self.resource_limits.items()):
                        continue
                    running[executor.submit(self._run_one, tgt)] = tgt
                    self.peak_running = max(self.peak_running, len(running))
                if not running:
                    if any(t.state == QUEUED for t in self.targets):
                        # Failure propagation may need another pass when plan order is reversed.
                        continue
                    break
                complete, _ = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in complete:
                    tgt = running.pop(future)
                    try:
                        future.result()
                    except Exception as exc:
                        tgt.state = FAILED
                        tgt.error = 'worker dispatch error: ' + str(exc)
                        tgt.ended = _utcnow()
                self._write_status()
        self.elapsed_seconds = round(time.monotonic() - start, 6)
        self._write_status()
        return self.snapshot()

    # -- status -----------------------------------------------------------
    def _overall(self) -> dict:
        counts = {"total": len(self.targets), "done": 0, "failed": 0,
                  "timeout": 0, "running": 0, "queued": 0, "skipped": 0, "blocked": 0}
        for t in self.targets:
            if t.state == DONE:
                counts["done"] += 1
            elif t.state == FAILED:
                counts["failed"] += 1
            elif t.state == TIMEOUT:
                counts["timeout"] += 1
            elif t.state == RUNNING:
                counts["running"] += 1
            elif t.state == BLOCKED:
                counts["blocked"] += 1
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
            "resource_limits": self.resource_limits,
            "metrics": {"elapsed_seconds": self.elapsed_seconds, "peak_running": self.peak_running,
                        "fingerprint_reads": self._hash_cache.reads, "fingerprint_cache_hits": self._hash_cache.hits},
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
                # Detailed immutable evidence lives in each summary_path. Keep the
                # returned snapshot complete, but never rewrite all evidence per job.
                compact = dict(snap, targets=[dict(row, definition_results=[],
                    definition_result_count=len(row.get("definition_results", [])))
                    for row in snap["targets"]])
                json.dump(compact, fh, indent=2)
            tmp.replace(self.status_path)
            # A compact view lets user-facing progress avoid reading large native evidence.
            outcomes = {}
            reasons = {}
            failures = []
            completed = []
            for target in snap['targets']:
                if target.get('ended'):
                    completed.append(target)
                if target.get('error'):
                    failures.append(target['id']+': '+str(target['error'])[:500])
                for definition in target.get('definition_results', []):
                    if not isinstance(definition, dict):
                        continue
                    assessment = definition.get('assessment')
                    categories = assessment.get('categories') if isinstance(assessment, dict) else None
                    for category in categories if isinstance(categories, list) else []:
                        if not isinstance(category, dict):
                            continue
                        if category.get('status') in ('not_applicable', 'no_benefit', 'excluded'):
                            reason = str(category.get('category', 'operation'))+': '+str(category.get('reason', 'No reason recorded.'))
                            reasons[reason] = reasons.get(reason, 0)+1
                    operations = definition.get('operations')
                    for operation in operations if isinstance(operations, list) else []:
                        if not isinstance(operation, dict) or not isinstance(operation.get('operation'), str):
                            continue
                        counts = outcomes.setdefault(operation['operation'], {})
                        outcome = str(operation.get('status', 'recorded'))
                        counts[outcome] = counts.get(outcome, 0)+1
            last = max(completed, key=lambda row: row['ended'], default={})
            last_operations = [op for record in last.get('definition_results', []) if isinstance(record, dict)
                               for op in (record.get('operations') or [])
                               if isinstance(op, dict) and isinstance(op.get('operation'), str)]
            summary = {'overall': snap['overall'], 'operation_outcomes': outcomes,
                       'running_jobs': [{'id': target['id'], 'progress_path': str(Path(target['summary_path']).with_suffix('.progress.json'))}
                                        for target in snap['targets'] if target['state'] == RUNNING and target.get('summary_path')],
                       'last_operation': last_operations[-1]['operation'] if last_operations else None,
                       'unchanged_reasons': dict(sorted(reasons.items(), key=lambda row: -row[1])[:8]),
                       'failures': failures[:5], 'last_completed': last.get('id'),
                       'last_completed_state': last.get('state')}
            summary_path = self.status_path.with_suffix('.summary.json')
            summary_tmp = summary_path.with_suffix('.json.tmp')
            summary_tmp.write_text(json.dumps(summary), encoding='utf-8')
            summary_tmp.replace(summary_path)
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
    p.add_argument("--plan", required=True, help="Batch plan JSON (targets[] with command, definition_ids, depends_on and resource budgets).")
    p.add_argument("--status", help="status.json output path (default: <plan dir>/status.json).")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--resume", action="store_true", help="Reuse completed targets only when their recorded inputs, runtime and outputs still match.")
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
    incomplete = o["failed"] or any(o.get(status) for status in ("timeout", "blocked", "skipped"))
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
