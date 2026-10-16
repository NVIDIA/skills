#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Four-verb process lifecycle for virtualenv-native TAO jobs.

This is the virtualenv platform's "native CLI" — the role `docker` plays for
the docker skill. It launches a Python script as an argv vector whose first
element is ``<venv>/bin/python`` (no shell, no activation), detached in its own
session, with a durable on-disk lifecycle that survives the launching process:

    submit  --job-dir D --venv V --script S [--arg TOKEN]... -> prints {pid,...}
    status  --job-dir D            -> {status: PENDING|RUNNING|COMPLETE|ERROR|CANCELED|UNKNOWN}
    logs    --job-dir D [--tail N] -> bounded tail of the job log
    cancel  --job-dir D            -> pidfd SIGTERM->SIGKILL for job processes

Job records are NOT written here — the agent owns them via tao_job_record.py
(open binds results_dir BEFORE submit; mark records the states this CLI
reports). The runner's own durable truth lives under ``<job-dir>/.tao_runner/``:
``submit_meta.json``, ``launcher_status.json`` (written by the wrapper the
moment it starts: pid + start marker), ``exit_status.json`` (fsync'd atomic
write on exit), a ``start_authorized`` gate, and a ``cancel_requested`` marker.

Correctness properties:
- The wrapper blocks on the start gate, so a submit that fails bookkeeping can
  abort before the training script ever runs.
- The wrapper, a durable guardian, and the direct workload are persisted with
  start markers before waiting. The guardian anchors the original process group
  even after the workload exec-replaces its submitted command line.
- After the script exits, leftover process-group members are SIGTERM/SIGKILLed
  through Linux pidfds so background DataLoader workers cannot leak and reused
  numeric IDs cannot be signaled.
- Observation is tri-state and fail-closed. Other POSIX systems can prove an
  empty group, but active cancellation is refused without an equivalent
  kernel-held process identity.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from virtualenv_group_supervisor import (
    OwnershipError,
    SupervisionError,
    group_facts,
    supervise,
)

VOCAB_PENDING = "PENDING"
VOCAB_RUNNING = "RUNNING"
VOCAB_COMPLETE = "COMPLETE"
VOCAB_ERROR = "ERROR"
VOCAB_CANCELED = "CANCELED"
VOCAB_UNKNOWN = "UNKNOWN"

RUNNER_DIR_NAME = ".tao_runner"
LOG_TAIL_MAX_BYTES = 2 * 1024 * 1024
LAUNCHER_RECORD_TIMEOUT_SECONDS = 10.0
# STATUS may never declare a launch dead while submit could still legitimately
# be waiting for the launcher record — the grace strictly exceeds that wait.
PENDING_LAUNCH_GRACE_SECONDS = LAUNCHER_RECORD_TIMEOUT_SECONDS + 5.0
GATE_WAIT_TIMEOUT_SECONDS = 30.0  # wrapper gives up if submit dies pre-gate
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
SUPERVISOR_SOURCE_PATH = Path(__file__).resolve().with_name(
    "virtualenv_group_supervisor.py"
)

# ---------------------------------------------------------------------------
# The wrapper written into the job dir and executed as
#   <venv>/bin/python launch_job.py <exit> <launcher> <gate> <cancel>
#       <supervisor> <guardian-release> <command...>
# It must stay self-contained (stdlib only) and portable.
# ---------------------------------------------------------------------------
JOB_WRAPPER_SOURCE = r'''
import json
import os
import signal
import subprocess
import sys
import time
import traceback


GUARDIAN_SOURCE = r"""
import os
import signal
import sys
import time

for name in ("SIGTERM", "SIGINT", "SIGHUP"):
    if hasattr(signal, name):
        signal.signal(getattr(signal, name), signal.SIG_IGN)
release_path = sys.argv[1]
while not os.path.exists(release_path):
    time.sleep(0.02)
"""


def _write_status(path, payload):
    temp_path = f"{path}.{os.getpid()}.tmp"
    with open(temp_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def _probe(argv):
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=5, start_new_session=True,
        env={**os.environ, "LC_ALL": "C", "TZ": "UTC"}, check=False,
    ).stdout


def _start_marker(pid):
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stream:
            stat = stream.read()
        closing = stat.rfind(")")
        fields = stat[closing + 2:].split() if closing >= 0 else []
        return fields[19] if len(fields) > 19 else None
    except OSError:
        pass
    try:
        out = _probe(["ps", "-p", str(pid), "-o", "lstart="]).strip()
        return out or None
    except Exception:
        return None


def _release_guardian(path, guardian):
    open(path, "a", encoding="utf-8").close()
    guardian.wait(timeout=5)


def _release_guardian_reliably(path, guardian, launch_path=None,
                               launch_record=None):
    """Do not publish terminal state while the durable group anchor is live."""
    failed = False
    while True:
        try:
            _release_guardian(path, guardian)
            return failed
        except BaseException as exc:
            failed = True
            if launch_path and launch_record is not None:
                launch_record["cleanup_error"] = (
                    f"Guardian release failed: {type(exc).__name__}: {exc}"
                )
                _write_status(launch_path, launch_record)
            time.sleep(1.0)


def _cancel_timeout(path):
    try:
        with open(path, encoding="utf-8") as stream:
            value = json.load(stream).get("timeout", 1.0)
        return max(0.0, float(value))
    except (OSError, ValueError, TypeError, AttributeError):
        return 1.0


def _supervise(supervisor_path, guardian_pid, guardian_marker, wrapper_marker,
               timeout):
    command = [
        sys.executable, supervisor_path,
        "--pgid", str(os.getpgrp()),
        "--guardian-pid", str(guardian_pid),
        "--guardian-marker", str(guardian_marker),
        "--exclude", f"{guardian_pid}:{guardian_marker}",
        "--exclude", f"{os.getpid()}:{wrapper_marker}",
        "--term-timeout", str(timeout),
        "--kill-timeout", str(max(1.0, timeout)),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True,
        timeout=max(10.0, timeout * 2 + 5.0), start_new_session=True,
        env={**os.environ, "LC_ALL": "C", "TZ": "UTC"}, check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stdout or completed.stderr).strip()
        raise RuntimeError(detail or f"group supervisor exited {completed.returncode}")


def main():
    status_path = sys.argv[1]
    launch_path = sys.argv[2]
    start_gate_path = sys.argv[3]
    cancel_path = sys.argv[4]
    supervisor_path = sys.argv[5]
    guardian_release_path = sys.argv[6]
    command = sys.argv[7:]
    started_at = time.time()
    wrapper_marker = _start_marker(os.getpid())
    guardian = subprocess.Popen(
        [sys.executable, "-c", GUARDIAN_SOURCE, guardian_release_path],
        stdin=subprocess.DEVNULL,
    )
    guardian_marker = _start_marker(guardian.pid)
    if not wrapper_marker or not guardian_marker:
        _release_guardian_reliably(guardian_release_path, guardian)
        _write_status(status_path, {
            "return_code": 127,
            "canceled": False,
            "error": "Could not establish wrapper/guardian process identity",
            "started_at": started_at,
            "finished_at": time.time(),
        })
        return 127
    launch_record = {
        "pid": os.getpid(),
        "process_start_marker": wrapper_marker,
        "guardian_pid": guardian.pid,
        "guardian_start_marker": guardian_marker,
        "started_at": started_at,
    }
    _write_status(launch_path, launch_record)

    gate_timeout = float(os.environ.get("TAO_RUNNER_GATE_TIMEOUT", "30"))
    gate_deadline = time.monotonic() + gate_timeout
    while not os.path.exists(start_gate_path):
        if time.monotonic() > gate_deadline:
            _release_guardian_reliably(
                guardian_release_path, guardian, launch_path, launch_record,
            )
            launch_record.pop("cleanup_error", None)
            _write_status(launch_path, launch_record)
            _write_status(status_path, {
                "return_code": 125,
                "canceled": False,
                "error": "Start was never authorized (submit died before opening the gate)",
                "started_at": started_at,
                "finished_at": time.time(),
            })
            return 125
        if os.path.exists(cancel_path):
            _release_guardian_reliably(
                guardian_release_path, guardian, launch_path, launch_record,
            )
            launch_record.pop("cleanup_error", None)
            _write_status(launch_path, launch_record)
            _write_status(status_path, {
                "return_code": -signal.SIGTERM,
                "canceled": True,
                "error": "Canceled before the script started",
                "started_at": started_at,
                "finished_at": time.time(),
            })
            return 128 + signal.SIGTERM
        time.sleep(0.02)
    if os.path.exists(cancel_path):
        _release_guardian_reliably(
            guardian_release_path, guardian, launch_path, launch_record,
        )
        launch_record.pop("cleanup_error", None)
        _write_status(launch_path, launch_record)
        _write_status(status_path, {
            "return_code": -signal.SIGTERM,
            "canceled": True,
            "error": "Canceled before the script started",
            "started_at": started_at,
            "finished_at": time.time(),
        })
        return 128 + signal.SIGTERM

    error = None
    canceled = False
    try:
        workload = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    except BaseException as exc:
        traceback.print_exc()
        workload = None
        return_code = 127
        error = f"{type(exc).__name__}: {exc}"
    if workload is not None:
        workload_marker = _start_marker(workload.pid)
        if not workload_marker:
            # This Popen object is still the unreaped parent-owned child, so its
            # pid cannot be reused before wait; requesting termination is safe.
            workload.terminate()
            try:
                workload.wait(timeout=2)
            except subprocess.TimeoutExpired:
                workload.kill()
                workload.wait()
            return_code = 127
            error = "Could not establish direct workload process identity"
        else:
            launch_record.update({
                "workload_pid": workload.pid,
                "workload_start_marker": workload_marker,
            })
            _write_status(launch_path, launch_record)
            while workload.poll() is None:
                if os.path.exists(cancel_path):
                    canceled = True
                    break
                time.sleep(0.05)
            if not canceled:
                return_code = workload.returncode

    # The same checked-in supervisor copied by submit owns every cleanup path.
    # Keep wrapper+guardian alive on indeterminate evidence and retry rather
    # than writing a false terminal record while workers may still execute.
    cleanup_failed = False
    while True:
        try:
            timeout = _cancel_timeout(cancel_path) if canceled else 1.0
            _supervise(
                supervisor_path, guardian.pid, guardian_marker, wrapper_marker,
                timeout,
            )
            break
        except BaseException as exc:
            cleanup_failed = True
            launch_record["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            _write_status(launch_path, launch_record)
            time.sleep(1.0)

    if workload is not None and workload.poll() is None:
        workload.wait()
    cleanup_failed = _release_guardian_reliably(
        guardian_release_path, guardian, launch_path, launch_record,
    ) or cleanup_failed
    launch_record.pop("cleanup_error", None)
    _write_status(launch_path, launch_record)
    if canceled:
        return_code = -signal.SIGTERM
        error = "Canceled by process-group supervisor"
    elif cleanup_failed:
        cleanup_error = "Process-group cleanup was temporarily indeterminate"
        error = f"{error}; {cleanup_error}" if error else cleanup_error
        if return_code == 0:
            return_code = 126

    _write_status(status_path, {
        "return_code": return_code,
        "canceled": canceled,
        "error": error,
        "started_at": started_at,
        "finished_at": time.time(),
    })
    return return_code if 0 <= return_code <= 255 else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, payload: dict) -> None:
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def _read_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _process_start_marker(pid: int) -> str | None:
    """Opaque per-process start marker that disambiguates PID reuse."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        closing = stat.rfind(")")
        if closing >= 0:
            fields = stat[closing + 2:].split()
            if len(fields) > 19:
                return fields[19]
        return None
    except OSError:
        pass
    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            capture_output=True, text=True, timeout=5,
            env={**os.environ, "LC_ALL": "C", "TZ": "UTC"},
        ).stdout.strip()
        return out or None
    except Exception:
        return None


def _coerce_pid(value) -> int:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return 0
    return pid if pid > 0 else 0


def _active_process_group_members(
    pgid: int, *, proc_root: Path | None = None,
) -> list[int] | None:
    """Return non-zombie members of ``pgid``; ``None`` means indeterminate.

    ``killpg(pgid, 0)`` only proves that a process-group entry still exists.
    In particular, it succeeds for an unreaped zombie even though that process
    cannot execute or receive a signal.  Lifecycle waits therefore need an
    explicit state-aware membership probe rather than signalability.

    This delegates to the same scanner used by destructive supervision so
    status and cancellation cannot disagree about missing or malformed evidence.
    """
    if pgid <= 0:
        return []

    try:
        facts = group_facts(pgid, proc_root=proc_root)
    except SupervisionError:
        return None
    return [fact.pid for fact in facts if fact.active]


def _identity_matches(pid: int, marker: str, pgid: int) -> bool | None:
    """Tri-state durable identity check for one active group member."""
    try:
        if os.getpgid(pid) != pgid:
            return False
    except ProcessLookupError:
        return False
    except (PermissionError, OSError, ValueError):
        return None
    marker_now = _process_start_marker(pid)
    if marker_now is None:
        # The member may have exited between getpgid and the marker read.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except (PermissionError, OSError, ValueError):
            return None
        return None
    return marker_now == marker


def _capture_group_identities(
    pgid: int, members: list[int],
) -> dict[int, str] | None:
    """Capture start identities for known-active members of an owned group."""
    identities: dict[int, str] = {}
    for member_pid in members:
        try:
            if os.getpgid(member_pid) != pgid:
                continue
        except ProcessLookupError:
            continue
        except (PermissionError, OSError, ValueError):
            return None
        marker = _process_start_marker(member_pid)
        if marker is None:
            # An exited member is benign; a still-live unidentifiable member is
            # not safe to use as authority for a later group signal.
            try:
                os.kill(member_pid, 0)
            except ProcessLookupError:
                continue
            except (PermissionError, OSError, ValueError):
                return None
            return None
        identities[member_pid] = str(marker)
    return identities


def _durably_owned_survivors(
    pgid: int, launcher: dict,
) -> dict[int, str] | None:
    """Return active survivors while a persisted job identity anchors the PGID.

    Guardian ownership is preferred.  The direct workload identity is retained
    as a read-only fallback so an externally killed guardian cannot make a live
    workload look terminal.  ``None`` means active membership could not be
    attributed safely; callers must report UNKNOWN and must not signal.
    """
    members = _active_process_group_members(pgid)
    if members is None:
        return None
    if not members:
        return {}

    anchors: list[tuple[int, str]] = []
    guardian = _guardian_identity(launcher)
    if guardian is not None:
        anchors.append(guardian)
    workload_pid = _coerce_pid(launcher.get("workload_pid"))
    workload_marker = launcher.get("workload_start_marker")
    if workload_pid and isinstance(workload_marker, str) and workload_marker:
        anchors.append((workload_pid, workload_marker))

    owned = False
    indeterminate = False
    for anchor_pid, anchor_marker in anchors:
        match = _identity_matches(anchor_pid, anchor_marker, pgid)
        if match is True:
            owned = True
            break
        if match is None:
            indeterminate = True
    if not owned:
        return None if members or indeterminate else {}

    excluded = {pgid}
    if guardian is not None:
        excluded.add(guardian[0])
    return _capture_group_identities(
        pgid, [member for member in members if member not in excluded]
    )


def _cmdline_probe(pid: int, needle: str) -> bool | None:
    """Tri-state: True/False when the cmdline was read, None when the probe
    itself failed (indeterminate — callers must not treat it as evidence)."""
    proc_path = Path(f"/proc/{pid}/cmdline")
    if proc_path.parent.exists():
        try:
            cmdline = proc_path.read_bytes().split(b"\0")
        except OSError:
            return None
        # A zombie has an empty cmdline — definitively not a live launcher.
        return os.fsencode(needle) in cmdline
    try:
        completed = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    return needle in completed.stdout


def _launcher_state(pid: int, expected_marker: str | None, wrapper_path: str | None) -> str:
    """'running' | 'gone' for STATUS derivation.

    Conservative toward 'running': a dead launcher self-heals on the next poll
    (its exit record, or a definitive probe), but a live job misread as gone
    would be recorded terminal — so 'gone' requires POSITIVE evidence: the pid
    is unsignalable, it is not a session leader (our wrapper always is), its
    retrieved start marker mismatches, or its cmdline definitively lacks the
    wrapper. Indeterminate probes stay 'running'.
    """
    if pid <= 0:
        return "gone"
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return "gone"
    try:
        if os.getpgid(pid) != pid:
            return "gone"  # PID reuse: the wrapper is always a session leader.
    except (OSError, ValueError):
        return "gone"
    if expected_marker:
        marker_now = _process_start_marker(pid)
        if marker_now is not None and marker_now != str(expected_marker):
            return "gone"
    if wrapper_path and _cmdline_probe(pid, wrapper_path) is False:
        return "gone"
    return "running"


def _recorded_launcher_reused(pid: int, expected_marker: str | None) -> bool | None:
    """Whether a still-live numeric PID is provably not the recorded launcher.

    ``False`` includes a process that is definitively gone. ``None`` means the
    identity probe is indeterminate and must not support a terminal result.
    """
    if pid <= 0 or not expected_marker:
        return None
    try:
        process_group = os.getpgid(pid)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError, ValueError):
        return None
    if process_group != pid:
        return True
    marker_now = _process_start_marker(pid)
    if marker_now is None:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except (PermissionError, OSError, ValueError):
            return None
        return None
    return marker_now != str(expected_marker)


def _process_matches(pid: int, expected_marker: str | None, wrapper_path: str | None) -> bool:
    """True only if ``pid`` is POSITIVELY this job's launcher (PID-reuse safe).

    The bias is the inverse of _launcher_state: this gates SIGNALING, so an
    indeterminate probe means "do not kill".
    """
    if pid <= 0 or not expected_marker:
        return False
    try:
        os.kill(pid, 0)
        if os.getpgid(pid) != pid:
            return False
    except (OSError, ValueError):
        return False
    if _process_start_marker(pid) != str(expected_marker):
        return False
    if wrapper_path and _cmdline_probe(pid, wrapper_path) is not True:
        return False
    return True


def _guardian_identity(launcher: dict | None) -> tuple[int, str] | None:
    if not isinstance(launcher, dict):
        return None
    pid = _coerce_pid(launcher.get("guardian_pid"))
    marker = launcher.get("guardian_start_marker")
    if not pid or not isinstance(marker, str) or not marker:
        return None
    return pid, marker


def _supervise_orphan(
    paths: "RunnerPaths", launcher: dict, timeout: float,
) -> dict[str, object]:
    """Terminate an orphaned group through the shared pidfd supervisor.

    The guardian is deliberately excluded and released only after every other
    active member is gone, so the numeric process-group id cannot be reused
    during discovery. No destructive signal in this process uses a raw PID or
    PGID.
    """
    identity = _guardian_identity(launcher)
    if identity is None:
        raise SupervisionError("orphaned job lacks a durable guardian identity")
    guardian_pid, guardian_marker = identity
    result = supervise(
        pgid=_coerce_pid(launcher.get("pid")),
        guardian_pid=guardian_pid,
        guardian_marker=guardian_marker,
        excludes={guardian_pid: guardian_marker},
        term_timeout=max(0.0, timeout),
        kill_timeout=max(1.0, timeout),
    )
    paths.guardian_release.touch()
    release_deadline = time.monotonic() + 5.0
    while time.monotonic() < release_deadline:
        facts = group_facts(_coerce_pid(launcher.get("pid")))
        guardian = next((fact for fact in facts if fact.pid == guardian_pid), None)
        if guardian is None or not guardian.active:
            return result
        if guardian.start_marker != guardian_marker:
            raise OwnershipError("guardian identity changed during release")
        time.sleep(0.02)
    raise SupervisionError("durable guardian did not exit after release")


def _read_tail(path: Path, line_count: int, max_bytes: int = LOG_TAIL_MAX_BYTES) -> str:
    """Read at most ``line_count`` lines without scanning the whole log."""
    if line_count <= 0:
        return ""
    block_size = 64 * 1024
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        remaining = stream.tell()
        chunks: list[bytes] = []
        newline_count = 0
        bytes_read = 0
        while remaining > 0 and newline_count <= line_count and bytes_read < max_bytes:
            size = min(block_size, remaining, max_bytes - bytes_read)
            remaining -= size
            stream.seek(remaining)
            chunk = stream.read(size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")
            bytes_read += len(chunk)
    data = b"".join(reversed(chunks))
    return b"".join(data.splitlines(keepends=True)[-line_count:]).decode(
        "utf-8", errors="replace"
    )


class RunnerPaths:
    """All durable control-file paths for one job dir."""

    def __init__(self, job_dir: Path):
        self.job_dir = job_dir
        self.runner_dir = job_dir / RUNNER_DIR_NAME
        self.wrapper = self.runner_dir / "launch_job.py"
        self.supervisor = self.runner_dir / "supervise_group.py"
        self.submit_meta = self.runner_dir / "submit_meta.json"
        self.launcher_status = self.runner_dir / "launcher_status.json"
        self.exit_status = self.runner_dir / "exit_status.json"
        self.start_gate = self.runner_dir / "start_authorized"
        self.cancel_marker = self.runner_dir / "cancel_requested"
        self.guardian_release = self.runner_dir / "guardian_release"
        self.log_path = job_dir / "logs" / "job.log"


def _fail(message: str, **extra) -> "int":
    print(json.dumps({"result": "error", "message": message, **extra}))
    return 1


def _emit(**payload) -> int:
    print(json.dumps(payload))
    return 0


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

def _validate_venv(venv: str) -> Path:
    path = Path(venv).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Virtual environment does not exist: {path}")
    if not (path / "pyvenv.cfg").is_file():
        raise ValueError(f"Not a Python virtual environment (pyvenv.cfg is missing): {path}")
    python_path = path / "bin" / "python"
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise ValueError(f"Virtual environment Python is not executable: {python_path}")
    return path


def _validate_gpus(gpus: int | None, gpu_ids: str | None) -> list[int] | None:
    if gpus is not None and gpus < 0:
        raise ValueError("--gpus must be a non-negative integer")
    if gpu_ids is None:
        return None
    tokens = [] if not gpu_ids.strip() else gpu_ids.split(",")
    normalized: list[int] = []
    for token in tokens:
        token = token.strip()
        if not re.fullmatch(r"\d+", token):
            raise ValueError("--gpu-ids must be a comma-separated list of non-negative integers")
        normalized.append(int(token))
    if len(set(normalized)) != len(normalized):
        raise ValueError("--gpu-ids must not contain duplicates")
    if gpus is not None and len(normalized) != gpus:
        raise ValueError("--gpus must match the number of --gpu-ids")
    return normalized


def _render_arg(token: str, placeholders: dict[str, str]) -> str:
    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in placeholders:
            raise ValueError(
                f"Unknown placeholder {{{name}}} in --arg {token!r}; "
                f"supported: {', '.join(sorted(placeholders))}"
            )
        return placeholders[name]

    return _PLACEHOLDER_RE.sub(replace, token)


def cmd_submit(args: argparse.Namespace) -> int:
    try:
        venv_path = _validate_venv(args.venv)
        script_path = Path(args.script).expanduser()
        cwd = Path(args.cwd).expanduser().resolve() if args.cwd else Path.cwd()
        if not cwd.is_dir():
            raise ValueError(f"Working directory does not exist: {cwd}")
        if not script_path.is_absolute():
            script_path = cwd / script_path
        script_path = script_path.resolve()
        if not script_path.is_file():
            raise ValueError(f"Python script does not exist: {script_path}")
        gpu_ids = _validate_gpus(args.gpus, args.gpu_ids)
        job_dir = Path(args.job_dir).expanduser().resolve()
        paths = RunnerPaths(job_dir)
        for existing in (paths.launcher_status, paths.exit_status):
            if existing.exists():
                raise ValueError(
                    f"Job dir already has runner state ({existing.name}); "
                    "one submit per job dir — open a new job record for a retry"
                )
        placeholders = {
            "results_dir": str(job_dir),
            "job_id": args.job_id or job_dir.name,
        }
        if args.config_path:
            placeholders["config_path"] = args.config_path
        rendered_args = [_render_arg(token, placeholders) for token in (args.arg or [])]
    except ValueError as exc:
        return _fail(str(exc))

    python_path = venv_path / "bin" / "python"
    paths.runner_dir.mkdir(parents=True, exist_ok=True)
    paths.log_path.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL on submit_meta is the SUBMISSION LOCK: exactly one submit can
    # claim a job dir, even racing concurrently (check-then-act is not enough).
    try:
        meta_fd = os.open(
            paths.submit_meta, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError:
        return _fail(
            "Job dir already has runner state (submit_meta.json); "
            "one submit per job dir — open a new job record for a retry"
        )
    # O_EXCL|O_NOFOLLOW: a pre-planted file or symlink at the wrapper path is
    # an error, never followed and never executed.
    try:
        wrapper_fd = os.open(
            paths.wrapper,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as exc:
        os.close(meta_fd)
        paths.submit_meta.unlink(missing_ok=True)
        return _fail(f"Refusing pre-existing wrapper path {paths.wrapper}: {exc}")
    with os.fdopen(wrapper_fd, "w", encoding="utf-8") as stream:
        stream.write(JOB_WRAPPER_SOURCE)

    supervisor_created = False
    try:
        supervisor_source = SUPERVISOR_SOURCE_PATH.read_bytes()
        supervisor_fd = os.open(
            paths.supervisor,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        supervisor_created = True
        with os.fdopen(supervisor_fd, "wb") as stream:
            stream.write(supervisor_source)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        if supervisor_created:
            paths.supervisor.unlink(missing_ok=True)
        paths.wrapper.unlink(missing_ok=True)
        paths.submit_meta.unlink(missing_ok=True)
        return _fail(f"Could not install group supervisor {paths.supervisor}: {exc}")

    script_argv = [str(python_path), str(script_path), *rendered_args]
    launcher_argv = [
        str(python_path),
        str(paths.wrapper),
        str(paths.exit_status),
        str(paths.launcher_status),
        str(paths.start_gate),
        str(paths.cancel_marker),
        str(paths.supervisor),
        str(paths.guardian_release),
        *script_argv,
    ]

    process_env = os.environ.copy()
    process_env.pop("PYTHONHOME", None)
    for name in args.env or []:
        # Pass credentials by NAME only; the value comes from this process's
        # environment and never lands on argv or in runner files.
        if name in os.environ:
            process_env[name] = os.environ[name]
    process_env["VIRTUAL_ENV"] = str(venv_path)
    current_path = process_env.get("PATH", "")
    process_env["PATH"] = str(venv_path / "bin") + (
        os.pathsep + current_path if current_path else ""
    )
    process_env["TAO_JOB_ID"] = placeholders["job_id"]
    process_env["TAO_RESULTS_ROOT"] = str(job_dir)
    process_env.setdefault("PYTHONUNBUFFERED", "1")
    if gpu_ids is not None:
        process_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)
    elif args.gpus == 0:
        process_env["CUDA_VISIBLE_DEVICES"] = ""

    with os.fdopen(meta_fd, "w", encoding="utf-8") as stream:
        json.dump({
            "job_id": placeholders["job_id"],
            "launch_started_at": time.time(),
            "argv": script_argv,
            "venv_path": str(venv_path),
            "wrapper_path": str(paths.wrapper),
            "supervisor_path": str(paths.supervisor),
            "cwd": str(cwd),
            "log_path": str(paths.log_path),
            "gpu_ids": gpu_ids,
            "env_passthrough": list(args.env or []),
        }, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

    try:
        log_file = paths.log_path.open("ab", buffering=0)
        try:
            process = subprocess.Popen(
                launcher_argv,
                cwd=str(cwd),
                env=process_env,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=True,
            )
        finally:
            log_file.close()  # Popen duplicated the descriptor for the child.
    except OSError as exc:
        return _fail(f"Failed to launch the wrapper: {exc}")
    # Wait for the wrapper's durable identity record, then open the gate. A
    # wrapper that never writes it means the interpreter/env is broken — kill
    # the group and report, before the script has had any chance to run.
    deadline = time.monotonic() + LAUNCHER_RECORD_TIMEOUT_SECONDS
    launcher = None
    while time.monotonic() < deadline:
        launcher = _read_json(paths.launcher_status)
        if launcher and launcher.get("pid"):
            break
        if process.poll() is not None:
            break
        time.sleep(0.05)
    if not launcher or not launcher.get("pid"):
        # The wrapper may not yet have created its durable guardian. Request a
        # gate self-cancel and never signal the numeric Popen pid: a process can
        # fork between this timeout and a signal, and there is no persisted
        # supervision anchor to own those descendants yet.
        _atomic_write_json(paths.cancel_marker, {"timeout": 1.0})
        tail = _read_tail(paths.log_path, 40) if paths.log_path.exists() else ""
        return _fail(
            "Launcher never wrote its durable identity record "
            f"(wrapper exit code: {process.poll()})",
            log_tail=tail,
        )

    paths.start_gate.touch()
    return _emit(
        result="submitted",
        status=VOCAB_RUNNING,
        job_dir=str(job_dir),
        pid=launcher["pid"],
        process_start_marker=launcher.get("process_start_marker"),
        log_path=str(paths.log_path),
        backend_ref=f"pid:{launcher['pid']}",
    )


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def _status_from_exit_record(paths: RunnerPaths) -> tuple[str, str, dict] | None:
    exit_record = _read_json(paths.exit_status)
    if exit_record is None:
        return None
    code = exit_record.get("return_code")
    if not isinstance(code, int) or isinstance(code, bool):
        return None
    canceled = exit_record.get("canceled")
    # Records predating the explicit field retain their historical behavior.
    # New records make the wrapper's observed outcome authoritative so a late
    # cancel marker cannot relabel a natural exit as canceled.
    if canceled is True or (canceled is None and paths.cancel_marker.exists()):
        return VOCAB_CANCELED, "Process group was canceled", {"return_code": code}
    if code == 0:
        return VOCAB_COMPLETE, "Process exited successfully", {"return_code": 0}
    if code < 0:
        return (
            VOCAB_ERROR,
            f"Process terminated by signal {-code}",
            {"return_code": code, "error": exit_record.get("error")},
        )
    return (
        VOCAB_ERROR,
        f"Process exited with code {code}",
        {"return_code": code, "error": exit_record.get("error")},
    )


def _derive_status(paths: RunnerPaths) -> tuple[str, str, dict]:
    """Stateless status derivation from the job dir's durable files."""
    from_record = _status_from_exit_record(paths)
    if from_record is not None:
        return from_record

    canceled = paths.cancel_marker.exists()
    submit_meta = _read_json(paths.submit_meta)
    launcher = _read_json(paths.launcher_status)
    wrapper_path = (submit_meta or {}).get("wrapper_path") or str(paths.wrapper)

    if launcher and _coerce_pid(launcher.get("pid")):
        pid = _coerce_pid(launcher.get("pid"))
        marker = launcher.get("process_start_marker")
        if _launcher_state(pid, marker, wrapper_path) == "running":
            if launcher.get("cleanup_error"):
                return (
                    VOCAB_UNKNOWN,
                    "Process-group cleanup is indeterminate; supervision remains active",
                    {"pid": pid, "cleanup_error": launcher.get("cleanup_error")},
                )
            message = f"Process {pid} is running"
            if canceled:
                message += " (cancel requested)"
            return VOCAB_RUNNING, message, {"pid": pid}
        # TOCTOU guard: the wrapper may have finished BETWEEN our exit-record
        # read and the process check — its fsync'd exit_status write strictly
        # precedes its exit, so with the process now gone, one re-read is
        # definitive: the record either landed or never will.
        from_record = _status_from_exit_record(paths)
        if from_record is not None:
            return from_record
        # A direct workload may have exec-replaced the submitted Python script,
        # so cmdline text is not identity. The durable guardian anchors the
        # original PGID and the persisted workload start marker survives exec.
        survivor_identities = _durably_owned_survivors(pid, launcher)
        if survivor_identities is None:
            return (
                VOCAB_UNKNOWN,
                "Launcher ended and process-group ownership is indeterminate",
                {"pid": pid},
            )
        survivors = sorted(survivor_identities)
        if survivors:
            message = (
                f"Launcher {pid} died but the script group is still running "
                f"(pids {survivors})"
            )
            if canceled:
                message += " (cancel requested)"
            return VOCAB_RUNNING, message, {"pid": pid, "orphaned": True}
        reused = _recorded_launcher_reused(pid, marker)
        if reused is not False:
            return (
                VOCAB_UNKNOWN,
                "Launcher ended and process-group ownership is indeterminate",
                {"pid": pid},
            )
        if canceled:
            return (
                VOCAB_CANCELED,
                "Canceled before a durable exit status was recorded",
                {"pid": pid},
            )
        return (
            VOCAB_ERROR,
            "Launcher ended before a durable exit status was recorded",
            {"pid": pid},
        )

    if submit_meta is not None:
        try:
            launch_age = time.time() - float(submit_meta.get("launch_started_at"))
        except (TypeError, ValueError):
            launch_age = PENDING_LAUNCH_GRACE_SECONDS
        if canceled:
            return VOCAB_CANCELED, "Canceled before the launcher started", {}
        if launch_age < PENDING_LAUNCH_GRACE_SECONDS:
            return VOCAB_PENDING, "Waiting for the durable launcher record", {}
        return VOCAB_ERROR, "Job host ended before the launcher started", {}

    return VOCAB_UNKNOWN, f"No runner state under {paths.runner_dir}", {}


def cmd_status(args: argparse.Namespace) -> int:
    paths = RunnerPaths(Path(args.job_dir).expanduser().resolve())
    status, message, extra = _derive_status(paths)
    return _emit(status=status, message=message, **extra)


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------

def cmd_logs(args: argparse.Namespace) -> int:
    paths = RunnerPaths(Path(args.job_dir).expanduser().resolve())
    try:
        sys.stdout.write(_read_tail(paths.log_path, args.tail))
    except OSError as exc:
        return _fail(f"No readable log at {paths.log_path}: {exc}")
    return 0


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------

def cmd_cancel(args: argparse.Namespace) -> int:
    paths = RunnerPaths(Path(args.job_dir).expanduser().resolve())
    status, message, extra = _derive_status(paths)
    if status in (VOCAB_COMPLETE, VOCAB_ERROR, VOCAB_CANCELED):
        return _emit(result="already_terminal", status=status, message=message, **extra)

    launcher = _read_json(paths.launcher_status)
    submit_meta = _read_json(paths.submit_meta)
    if launcher is None and submit_meta is None:
        return _fail(message, status=VOCAB_UNKNOWN)

    # Mark first so a healthy wrapper performs supervision while its durable
    # guardian anchors the PGID. The timeout value is non-secret control data.
    try:
        _atomic_write_json(
            paths.cancel_marker, {"timeout": max(0.0, args.timeout)},
        )
    except OSError as exc:
        return _fail(
            f"could not persist cancel request: {exc}", status=VOCAB_UNKNOWN,
        )
    pid = _coerce_pid(launcher.get("pid")) if launcher else 0
    marker = (launcher or {}).get("process_start_marker")
    wrapper_path = (submit_meta or {}).get("wrapper_path") or str(paths.wrapper)

    if pid and _process_matches(pid, marker, wrapper_path):
        # No external destructive signal is needed: the wrapper sees the
        # marker, invokes the same copied pidfd supervisor, writes status, and
        # releases the guardian. Wait a bounded interval for that durable path.
        deadline = time.monotonic() + max(3.0, args.timeout * 2 + 2.0)
        while time.monotonic() < deadline:
            status, message, extra = _derive_status(paths)
            if status in (VOCAB_COMPLETE, VOCAB_ERROR, VOCAB_CANCELED):
                result = "canceled" if status == VOCAB_CANCELED else "already_terminal"
                return _emit(result=result, status=status, message=message, **extra)
            time.sleep(0.05)
        status, message, extra = _derive_status(paths)
        if status == VOCAB_UNKNOWN:
            return _fail(message, status=VOCAB_UNKNOWN, **extra)
        return _emit(result="cancel_requested", status=status, message=message, **extra)

    if pid and launcher:
        try:
            _supervise_orphan(paths, launcher, args.timeout)
        except (OSError, OwnershipError, SupervisionError, ValueError) as exc:
            return _fail(str(exc), status=VOCAB_UNKNOWN)
        deadline = time.monotonic() + max(2.0, args.timeout)
        while time.monotonic() < deadline:
            status, message, extra = _derive_status(paths)
            if status == VOCAB_CANCELED:
                return _emit(result="canceled", status=status, message=message, **extra)
            time.sleep(0.05)
        return _fail(
            "guardian did not release after orphan supervision",
            status=VOCAB_UNKNOWN,
        )

    # A pre-launch wrapper will consume the marker if it appears. Without any
    # durable launcher/guardian identity there is nothing safe to signal.
    if status == VOCAB_PENDING:
        return _emit(result="cancel_requested", status=status, message=message, **extra)
    return _fail(message, status=VOCAB_UNKNOWN)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)

    submit = sub.add_parser("submit", help="Launch a Python script in a venv, detached.")
    submit.add_argument("--job-dir", required=True,
                        help="The job's results_dir (bound by tao_job_record open BEFORE submit).")
    submit.add_argument("--venv", required=True, help="Virtualenv root (pyvenv.cfg + bin/python).")
    submit.add_argument("--script", required=True, help="Python script to run.")
    submit.add_argument("--arg", action="append", default=[],
                        help="Script argv token; may use {config_path} {results_dir} {job_id}.")
    submit.add_argument("--job-id", default=None, help="Job-record id (sets TAO_JOB_ID).")
    submit.add_argument("--config-path", default=None,
                        help="Spec file the agent authored; fills {config_path}.")
    submit.add_argument("--cwd", default=None, help="Working directory (default: current).")
    submit.add_argument("--gpus", type=int, default=None,
                        help="GPU count; 0 hides CUDA devices. Metadata, not a reservation.")
    submit.add_argument("--gpu-ids", default=None,
                        help="Comma-separated CUDA device ids -> CUDA_VISIBLE_DEVICES.")
    submit.add_argument("-e", "--env", action="append", default=[],
                        help="Environment variable NAME to pass through (value from this env).")
    submit.set_defaults(func=cmd_submit)

    status = sub.add_parser("status", help="Derive job status from durable state.")
    status.add_argument("--job-dir", required=True)
    status.set_defaults(func=cmd_status)

    logs = sub.add_parser("logs", help="Bounded tail of the job log.")
    logs.add_argument("--job-dir", required=True)
    logs.add_argument("--tail", type=int, default=200)
    logs.set_defaults(func=cmd_logs)

    cancel = sub.add_parser("cancel", help="Cancel the job's whole process group.")
    cancel.add_argument("--job-dir", required=True)
    cancel.add_argument("--timeout", type=float, default=5.0)
    cancel.set_defaults(func=cmd_cancel)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
