#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed process-group supervision for virtualenv-native jobs.

The caller supplies a durable guardian identity that remains alive in the job
process group for the whole operation.  On Linux, every destructive signal is
sent through a pidfd, never a numeric PID or PGID.  The live guardian prevents
the numeric group id from being reused while members are discovered, and the
pidfds pin each target across the observation/signal boundary.

On another POSIX platform this helper may prove that no target remains, but it
refuses to signal because there is no equivalent kernel-held process handle in
the Python standard library.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import platform
import select
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


class SupervisionError(RuntimeError):
    """The target group could not be observed or signaled safely."""


class OwnershipError(SupervisionError):
    """The durable guardian no longer owns the recorded process group."""


@dataclass(frozen=True)
class ProcFact:
    pid: int
    state: str
    pgid: int
    start_marker: str

    @property
    def active(self) -> bool:
        return self.state != "Z"


def _proc_fact(pid: int, *, proc_root: Path = Path("/proc")) -> ProcFact | None:
    """Read one Linux identity atomically enough to validate a later pidfd.

    ``None`` means the process is gone.  Any other observation failure is
    indeterminate and therefore raises instead of being converted to absence.
    """
    try:
        raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        try:
            os.getpgid(pid)
        except ProcessLookupError:
            return None
        except OSError:
            pass
        raise SupervisionError(f"cannot read /proc/{pid}/stat: {exc}") from exc
    closing = raw.rfind(")")
    if closing < 0:
        raise SupervisionError(f"malformed /proc/{pid}/stat: missing comm terminator")
    fields = raw[closing + 2 :].split()
    if len(fields) <= 19:
        raise SupervisionError(f"malformed /proc/{pid}/stat: truncated fields")
    try:
        return ProcFact(
            pid=pid,
            state=fields[0],
            pgid=int(fields[2]),
            start_marker=fields[19],
        )
    except (TypeError, ValueError) as exc:
        raise SupervisionError(f"malformed /proc/{pid}/stat: invalid fields") from exc


def _linux_group_facts(
    pgid: int, *, proc_root: Path = Path("/proc"),
) -> list[ProcFact]:
    root = proc_root
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        raise SupervisionError(f"cannot enumerate /proc: {exc}") from exc
    facts: list[ProcFact] = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            fact = _proc_fact(pid, proc_root=proc_root)
        except SupervisionError:
            # An unreadable unrelated process must not poison an otherwise
            # observable same-UID job.  getpgid is the only safe discriminator.
            try:
                observed_group = os.getpgid(pid)
            except ProcessLookupError:
                continue
            except OSError as exc:
                raise SupervisionError(
                    f"cannot determine whether unreadable process {pid} belongs "
                    f"to process group {pgid}: {exc}"
                ) from exc
            if observed_group != pgid:
                continue
            raise
        if fact is not None and fact.pgid == pgid:
            facts.append(fact)
    return sorted(facts, key=lambda item: item.pid)


def _fallback_group_facts(pgid: int) -> list[ProcFact]:
    """Portable empty-group proof; active members cannot be signaled safely."""
    try:
        found = subprocess.run(
            ["pgrep", "-g", str(pgid)],
            capture_output=True,
            text=True,
            timeout=5,
            env={**os.environ, "LC_ALL": "C", "TZ": "UTC"},
            check=False,
        )
    except Exception as exc:
        raise SupervisionError(f"cannot enumerate process group {pgid}: {exc}") from exc
    if found.returncode == 1:
        return []
    if found.returncode != 0:
        raise SupervisionError(
            f"pgrep failed while enumerating process group {pgid} "
            f"(exit {found.returncode})"
        )
    facts: list[ProcFact] = []
    for token in found.stdout.split():
        if not token.isdigit():
            raise SupervisionError(f"pgrep returned an invalid pid: {token!r}")
        pid = int(token)
        try:
            group = os.getpgid(pid)
        except ProcessLookupError:
            continue
        except OSError as exc:
            raise SupervisionError(f"cannot inspect process {pid}: {exc}") from exc
        if group != pgid:
            continue
        try:
            probed = subprocess.run(
                ["ps", "-p", str(pid), "-o", "stat=", "-o", "lstart="],
                capture_output=True,
                text=True,
                timeout=5,
                env={**os.environ, "LC_ALL": "C", "TZ": "UTC"},
                check=False,
            )
        except Exception as exc:
            raise SupervisionError(f"cannot inspect process {pid}: {exc}") from exc
        text = probed.stdout.strip()
        if probed.returncode != 0 or not text:
            try:
                os.getpgid(pid)
            except ProcessLookupError:
                continue
            except OSError as exc:
                raise SupervisionError(f"cannot inspect process {pid}: {exc}") from exc
            raise SupervisionError(f"state of process {pid} is indeterminate")
        state, _, marker = text.partition(" ")
        facts.append(ProcFact(pid, state, pgid, marker.strip()))
    return sorted(facts, key=lambda item: item.pid)


def group_facts(
    pgid: int, *, proc_root: Path | None = None,
) -> list[ProcFact]:
    if pgid <= 0:
        raise SupervisionError("process-group id must be positive")
    if proc_root is not None:
        return _linux_group_facts(pgid, proc_root=proc_root)
    if Path("/proc/self/stat").is_file():
        return _linux_group_facts(pgid)
    return _fallback_group_facts(pgid)


def _pidfd_capable() -> bool:
    return (
        sys.platform.startswith("linux")
        and (hasattr(os, "pidfd_open") or _pidfd_open_syscall_number() is not None)
        and hasattr(signal, "pidfd_send_signal")
        and Path("/proc/self/stat").is_file()
    )


def _pidfd_open_syscall_number() -> int | None:
    """Linux assigned pidfd_open 434 on the supported TAO architectures.

    Some packaged Python builds omit ``os.pidfd_open`` even when the kernel
    supports it.  Calling the kernel API through libc retains the same
    kernel-held identity guarantee; unknown architectures fail closed.
    """
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64", "aarch64", "arm64", "riscv64"}:
        return 434
    return None


def _pidfd_open(pid: int) -> int:
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid, 0)
    number = _pidfd_open_syscall_number()
    if number is None:
        raise OSError(errno.ENOSYS, "pidfd_open syscall number is unknown")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    fd = int(libc.syscall(number, int(pid), 0))
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def _pidfd_dead(fd: int) -> bool:
    poller = select.poll()
    poller.register(fd, select.POLLIN | select.POLLHUP | select.POLLERR)
    return bool(poller.poll(0))


def _open_verified_pidfd(fact: ProcFact, *, expected_marker: str | None = None) -> int | None:
    """Open a kernel process handle and prove it names the observed identity."""
    if not _pidfd_capable():
        raise SupervisionError(
            "safe signaling requires Linux /proc, pidfd_open support, and "
            "signal.pidfd_send_signal"
        )
    if expected_marker is not None and fact.start_marker != expected_marker:
        raise OwnershipError(
            f"process {fact.pid} start marker no longer matches durable identity"
        )
    try:
        fd = _pidfd_open(fact.pid)
    except ProcessLookupError:
        return None
    except OSError as exc:
        raise SupervisionError(f"cannot open pidfd for process {fact.pid}: {exc}") from exc
    try:
        after = _proc_fact(fact.pid)
        if after is None or _pidfd_dead(fd):
            os.close(fd)
            return None
        if (
            after.pid != fact.pid
            or after.pgid != fact.pgid
            or after.start_marker != fact.start_marker
        ):
            raise OwnershipError(
                f"process {fact.pid} changed identity while its pidfd was opened"
            )
        return fd
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _send_pidfd(fd: int, pid: int, sig: int) -> None:
    try:
        signal.pidfd_send_signal(fd, sig, None, 0)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        if _pidfd_dead(fd):
            return
        raise SupervisionError(
            f"permission denied signaling process {pid} through its pidfd"
        ) from exc
    except OSError as exc:
        if _pidfd_dead(fd):
            return
        raise SupervisionError(
            f"failed signaling process {pid} through its pidfd: {exc}"
        ) from exc


def _parse_excludes(values: list[str]) -> dict[int, str]:
    excludes: dict[int, str] = {}
    for value in values:
        pid_text, separator, marker = value.partition(":")
        if not separator or not pid_text.isdigit() or not marker:
            raise SupervisionError(
                f"--exclude must be PID:START_MARKER, got {value!r}"
            )
        excludes[int(pid_text)] = marker
    return excludes


def supervise(
    *,
    pgid: int,
    guardian_pid: int,
    guardian_marker: str,
    excludes: dict[int, str],
    term_timeout: float,
    kill_timeout: float,
) -> dict[str, object]:
    """Terminate every active member except durable, identity-bound excludes."""
    if term_timeout < 0 or kill_timeout < 0:
        raise SupervisionError("timeouts must be non-negative")
    facts = group_facts(pgid)
    guardian = next((item for item in facts if item.pid == guardian_pid), None)
    if guardian is None or not guardian.active:
        raise OwnershipError("durable guardian is gone; refusing group supervision")
    if guardian.pgid != pgid or guardian.start_marker != guardian_marker:
        raise OwnershipError("durable guardian identity no longer owns this group")

    # The guardian is an invariant of this abstraction, not a caller option.
    # Always bind and exclude it even if a future caller omits it from excludes.
    excludes = {**excludes, guardian_pid: guardian_marker}

    if not _pidfd_capable():
        targets = [
            item.pid
            for item in facts
            if item.active
            and not (
                item.pid in excludes
                and excludes[item.pid] == item.start_marker
            )
        ]
        if targets:
            raise SupervisionError(
                "active group members require Linux pidfd supervision; refusing "
                f"to signal numeric pids on {sys.platform}: {targets}"
            )
        return {"result": "empty", "terminated": []}

    guardian_fd = _open_verified_pidfd(guardian, expected_marker=guardian_marker)
    if guardian_fd is None:
        raise OwnershipError("durable guardian exited before supervision started")
    handles: dict[tuple[int, str], int] = {}
    terminated: set[int] = set()

    def close_dead_handles() -> None:
        for identity, fd in list(handles.items()):
            if _pidfd_dead(fd):
                os.close(fd)
                handles.pop(identity, None)
                terminated.add(identity[0])

    def discover() -> int:
        if _pidfd_dead(guardian_fd):
            raise OwnershipError("durable guardian exited during supervision")
        observed = group_facts(pgid)
        if _pidfd_dead(guardian_fd):
            raise OwnershipError("durable guardian exited during group discovery")
        added = 0
        for fact in observed:
            if not fact.active:
                continue
            if fact.pid in excludes and excludes[fact.pid] == fact.start_marker:
                continue
            identity = (fact.pid, fact.start_marker)
            if identity in handles:
                continue
            fd = _open_verified_pidfd(fact)
            if fd is None:
                continue
            if _pidfd_dead(guardian_fd):
                os.close(fd)
                raise OwnershipError("durable guardian exited before target binding")
            handles[identity] = fd
            added += 1
        return added

    try:
        discover()
        for (pid, _), fd in list(handles.items()):
            _send_pidfd(fd, pid, signal.SIGTERM)

        term_deadline = time.monotonic() + term_timeout
        while time.monotonic() < term_deadline:
            close_dead_handles()
            before = set(handles)
            discover()
            for identity, fd in list(handles.items()):
                if identity not in before:
                    _send_pidfd(fd, identity[0], signal.SIGTERM)
            if not handles and discover() == 0:
                return {"result": "terminated", "terminated": sorted(terminated)}
            time.sleep(0.02)

        # Escalation is still pidfd-only.  Repeated discovery catches workers
        # forked during graceful shutdown while the guardian anchors the group.
        kill_deadline = time.monotonic() + kill_timeout
        while True:
            close_dead_handles()
            discover()
            for (pid, _), fd in list(handles.items()):
                _send_pidfd(fd, pid, signal.SIGKILL)
            close_dead_handles()
            if not handles and discover() == 0:
                return {"result": "terminated", "terminated": sorted(terminated)}
            if time.monotonic() >= kill_deadline:
                remaining = sorted({pid for pid, _ in handles})
                raise SupervisionError(
                    f"process group {pgid} still has live members after pidfd "
                    f"SIGKILL: {remaining}"
                )
            time.sleep(0.02)
    finally:
        for fd in handles.values():
            try:
                os.close(fd)
            except OSError:
                pass
        os.close(guardian_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgid", required=True, type=int)
    parser.add_argument("--guardian-pid", required=True, type=int)
    parser.add_argument("--guardian-marker", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--term-timeout", type=float, default=1.0)
    parser.add_argument("--kill-timeout", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = supervise(
            pgid=args.pgid,
            guardian_pid=args.guardian_pid,
            guardian_marker=args.guardian_marker,
            excludes=_parse_excludes(args.exclude),
            term_timeout=args.term_timeout,
            kill_timeout=args.kill_timeout,
        )
    except (OSError, SupervisionError, ValueError) as exc:
        print(json.dumps({"result": "error", "message": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
