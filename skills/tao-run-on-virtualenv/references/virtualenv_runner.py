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
    cancel  --job-dir D            -> SIGTERM->SIGKILL the whole process group

Job records are NOT written here — the agent owns them via tao_job_record.py
(open binds results_dir BEFORE submit; mark records the states this CLI
reports). The runner's own durable truth lives under ``<job-dir>/.tao_runner/``:
``submit_meta.json``, ``launcher_status.json`` (written by the wrapper the
moment it starts: pid + start marker), ``exit_status.json`` (fsync'd atomic
write on exit), a ``start_authorized`` gate, and a ``cancel_requested`` marker.

Correctness properties ported from the reviewed upstream implementation:
- The wrapper blocks on the start gate, so a submit that fails bookkeeping can
  abort before the training script ever runs.
- Process identity is (pid, pgid-leader, process start marker, wrapper path in
  cmdline) — a reused PID is never treated as the job, and cancel can never
  kill an innocent process.
- After the script exits, leftover process-group members are SIGTERM/SIGKILLed
  so background DataLoader workers cannot leak (exit code 126 if any survive
  a clean run).
- Linux is first-class (/proc); on other POSIX systems `ps`/`pgrep` fallbacks
  keep the verbs functional for local smokes.
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

# ---------------------------------------------------------------------------
# The wrapper written into the job dir and executed as
#   <venv>/bin/python launch_job.py <exit_status> <launcher_status> <gate> <cancel> <command...>
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


def _write_status(path, payload):
    temp_path = f"{path}.{os.getpid()}.tmp"
    with open(temp_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def _probe(argv):
    """Run a ps/pgrep probe OUTSIDE our process group so membership scans
    never see the probe itself. Locale/TZ pinned so `ps lstart` markers are
    invocation-invariant."""
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=5, start_new_session=True,
        env={**os.environ, "LC_ALL": "C", "TZ": "UTC"},
    ).stdout


def _start_marker():
    try:
        with open(f"/proc/{os.getpid()}/stat", encoding="utf-8") as stream:
            stat = stream.read()
        fields = stat[stat.rfind(")") + 2:].split()
        return fields[19] if len(fields) > 19 else None
    except OSError:
        pass
    try:
        out = _probe(["ps", "-p", str(os.getpid()), "-o", "lstart="]).strip()
        return out or None
    except Exception:
        return None


def _active_group_members():
    """Return non-zombie members of this launcher's process group (not us)."""
    own_pid = os.getpid()
    group_id = os.getpgrp()
    members = []
    try:
        proc_entries = os.scandir("/proc")
    except OSError:
        proc_entries = None
    if proc_entries is not None:
        with proc_entries:
            for entry in proc_entries:
                if not entry.name.isdigit():
                    continue
                pid = int(entry.name)
                if pid == own_pid:
                    continue
                try:
                    with open(f"/proc/{pid}/stat", encoding="utf-8") as stream:
                        stat = stream.read()
                    fields = stat[stat.rfind(")") + 2:].split()
                    state = fields[0]
                    process_group = int(fields[2])
                except (OSError, ValueError, IndexError):
                    continue
                if process_group == group_id and state != "Z":
                    members.append(pid)
        return members
    # Non-/proc fallback (macOS): pgrep by process group, filter zombies via ps.
    try:
        out = _probe(["pgrep", "-g", str(group_id)]).split()
    except Exception:
        return members
    for token in out:
        try:
            pid = int(token)
        except ValueError:
            continue
        if pid == own_pid:
            continue
        try:
            state = _probe(["ps", "-p", str(pid), "-o", "stat="]).strip()
        except Exception:
            state = ""
        if state and not state.startswith("Z"):
            members.append(pid)
    return members


def _stop_remaining_group_members(timeout=1.0):
    """Prevent a finished script from leaking background workers."""
    members = _active_group_members()
    for pid in members:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    while members and time.monotonic() < deadline:
        time.sleep(0.05)
        members = _active_group_members()
    for pid in members:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    kill_deadline = time.monotonic() + max(1.0, timeout)
    while members and time.monotonic() < kill_deadline:
        time.sleep(0.05)
        members = _active_group_members()
    if members:
        raise RuntimeError(f"process group still has live members: {members}")


def main():
    status_path = sys.argv[1]
    launch_path = sys.argv[2]
    start_gate_path = sys.argv[3]
    cancel_path = sys.argv[4]
    command = sys.argv[5:]
    started_at = time.time()
    _write_status(launch_path, {
        "pid": os.getpid(),
        "process_start_marker": _start_marker(),
        "started_at": started_at,
    })
    # A healthy submit opens the gate within one poll of reading our record;
    # if it died first, give up with a durable record instead of spinning as
    # a phantom RUNNING job forever.
    gate_timeout = float(os.environ.get("TAO_RUNNER_GATE_TIMEOUT", "30"))
    gate_deadline = time.monotonic() + gate_timeout
    while not os.path.exists(start_gate_path):
        if time.monotonic() > gate_deadline:
            _write_status(status_path, {
                "return_code": 125,
                "error": "Start was never authorized (submit died before opening the gate)",
                "started_at": started_at,
                "finished_at": time.time(),
            })
            return 125
        if os.path.exists(cancel_path):
            _write_status(status_path, {
                "return_code": -signal.SIGTERM,
                "error": "Canceled before the script started",
                "started_at": started_at,
                "finished_at": time.time(),
            })
            return 128 + signal.SIGTERM
        time.sleep(0.02)
    if os.path.exists(cancel_path):
        _write_status(status_path, {
            "return_code": -signal.SIGTERM,
            "error": "Canceled before the script started",
            "started_at": started_at,
            "finished_at": time.time(),
        })
        return 128 + signal.SIGTERM
    error = None
    try:
        return_code = subprocess.call(command)
    except BaseException as exc:  # Preserve a durable failure record.
        traceback.print_exc()
        return_code = 127
        error = f"{type(exc).__name__}: {exc}"
    try:
        _stop_remaining_group_members()
    except BaseException as exc:
        traceback.print_exc()
        cleanup_error = f"{type(exc).__name__}: {exc}"
        error = f"{error}; {cleanup_error}" if error else cleanup_error
        if return_code == 0:
            return_code = 126
    _write_status(status_path, {
        "return_code": return_code,
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


def _group_members(pgid: int, identity_hint: str | None) -> list[int]:
    """POSITIVELY-identified surviving members of ``pgid`` (leader excluded).

    Used when the launcher is dead: a surviving training script must not be
    reported terminal or left unkillable. Only members whose cmdline matches
    ``identity_hint`` count — a recycled pgid of foreign processes never does.
    """
    candidates: list[int] = []
    if Path("/proc").exists():
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text(encoding="utf-8")
                fields = stat[stat.rfind(")") + 2:].split()
                if int(fields[2]) == pgid and fields[0] != "Z":
                    candidates.append(int(entry.name))
            except (OSError, ValueError, IndexError):
                continue
    else:
        try:
            out = subprocess.run(
                ["pgrep", "-g", str(pgid)],
                capture_output=True, text=True, timeout=5,
            ).stdout.split()
            candidates = [int(t) for t in out if t.isdigit()]
        except Exception:
            candidates = []
    members = [pid for pid in candidates if pid != pgid]
    if not identity_hint:
        return []
    return [pid for pid in members if _cmdline_probe(pid, identity_hint) is True]


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


def _process_matches(pid: int, expected_marker: str | None, wrapper_path: str | None) -> bool:
    """True only if ``pid`` is POSITIVELY this job's launcher (PID-reuse safe).

    The bias is the inverse of _launcher_state: this gates SIGNALING, so an
    indeterminate probe means "do not kill".
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        if os.getpgid(pid) != pid:
            return False
    except (OSError, ValueError):
        return False
    if expected_marker and _process_start_marker(pid) != str(expected_marker):
        return False
    if wrapper_path and _cmdline_probe(pid, wrapper_path) is not True:
        return False
    return True


def _terminate_process_group(pid: int, timeout: float = 5.0) -> str:
    """SIGTERM then SIGKILL the group. Returns 'gone' | 'terminated' | raises.

    Callers verify process identity first, so this never signals a foreign
    group. PermissionError on a real signal is treated as terminated: it
    happens when only unreaped zombies remain in the group (observed on
    macOS), which cannot be signaled and are already dead.
    """

    def group_exists() -> bool:
        try:
            os.killpg(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "gone"
    except PermissionError:
        return "terminated"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not group_exists():
            return "terminated"
        time.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return "terminated"
    except PermissionError:
        return "terminated"
    kill_deadline = time.monotonic() + max(1.0, timeout)
    while time.monotonic() < kill_deadline:
        if not group_exists():
            return "terminated"
        time.sleep(0.05)
    raise RuntimeError(f"process group {pid} survived SIGKILL")


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
        self.submit_meta = self.runner_dir / "submit_meta.json"
        self.launcher_status = self.runner_dir / "launcher_status.json"
        self.exit_status = self.runner_dir / "exit_status.json"
        self.start_gate = self.runner_dir / "start_authorized"
        self.cancel_marker = self.runner_dir / "cancel_requested"
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

    script_argv = [str(python_path), str(script_path), *rendered_args]
    launcher_argv = [
        str(python_path),
        str(paths.wrapper),
        str(paths.exit_status),
        str(paths.launcher_status),
        str(paths.start_gate),
        str(paths.cancel_marker),
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
        try:
            _terminate_process_group(process.pid)
        except RuntimeError:
            pass
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
    if paths.cancel_marker.exists():
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
        # A dead launcher does NOT mean a dead job: the training script is a
        # same-group child that survives a wrapper-only SIGKILL/OOM. Probe for
        # positively-identified survivors before declaring anything terminal.
        script_hint = None
        argv = (submit_meta or {}).get("argv") or []
        if len(argv) > 1:
            script_hint = argv[1]
        survivors = _group_members(pid, script_hint)
        if survivors:
            message = (
                f"Launcher {pid} died but the script group is still running "
                f"(pids {survivors})"
            )
            if canceled:
                message += " (cancel requested)"
            return VOCAB_RUNNING, message, {"pid": pid, "orphaned": True}
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
    if status == VOCAB_UNKNOWN:
        return _fail(message)

    # Mark first so a wrapper still waiting on the gate self-cancels, then
    # verify identity before signaling anything.
    paths.cancel_marker.touch()
    launcher = _read_json(paths.launcher_status)
    pid = _coerce_pid(launcher.get("pid")) if launcher else 0
    marker = (launcher or {}).get("process_start_marker")
    submit_meta = _read_json(paths.submit_meta)
    wrapper_path = (submit_meta or {}).get("wrapper_path") or str(paths.wrapper)

    signaled = False
    if pid and _process_matches(pid, marker, wrapper_path):
        signaled = True
        try:
            outcome = _terminate_process_group(pid, timeout=args.timeout)
        except RuntimeError as exc:
            return _fail(str(exc), status=VOCAB_UNKNOWN)
    else:
        outcome = "gone"

    if outcome == "gone" and pid:
        # Launcher gone, but the script group may have survived it (orphaned
        # children of a killed wrapper). Kill positively-identified survivors.
        argv = (submit_meta or {}).get("argv") or []
        script_hint = argv[1] if len(argv) > 1 else None
        if _group_members(pid, script_hint):
            signaled = True
            try:
                _terminate_process_group(pid, timeout=args.timeout)
            except RuntimeError as exc:
                return _fail(str(exc), status=VOCAB_UNKNOWN)

    if outcome == "gone" and not signaled:
        # Nothing was signaled by this cancel: any landed exit record is a
        # NATURAL exit that beat the cancel — undo the cancel claim rather
        # than mislabel it (the one exception: the wrapper's gate self-cancel
        # record, which only exists because of OUR marker).
        deadline = time.monotonic() + 2.0
        exit_record = _read_json(paths.exit_status)
        while exit_record is None and time.monotonic() < deadline:
            time.sleep(0.05)
            exit_record = _read_json(paths.exit_status)
        code = (exit_record or {}).get("return_code")
        if isinstance(code, int) and not isinstance(code, bool):
            self_canceled = (
                (exit_record or {}).get("error") == "Canceled before the script started"
            )
            if not self_canceled:
                paths.cancel_marker.unlink(missing_ok=True)
                status, message, extra = _derive_status(paths)
                return _emit(
                    result="already_terminal", status=status, message=message, **extra
                )

    status, message, extra = _derive_status(paths)
    result = "canceled"
    if not signaled and status in (VOCAB_PENDING, VOCAB_RUNNING):
        # Honest reply: the marker is set (a gated wrapper will self-cancel)
        # but nothing live was signaled — the caller should re-run cancel.
        result = "cancel_requested"
    return _emit(result=result, status=status, message=message, **extra)


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
