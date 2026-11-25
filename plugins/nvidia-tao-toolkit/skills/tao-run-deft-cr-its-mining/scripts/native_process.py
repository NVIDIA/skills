#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own a Linux container command, including workers which create new sessions."""

from contextlib import contextmanager, redirect_stdout
import ctypes
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time


@contextmanager
def live_output():
    """Keep platform backpressure from blocking worker cancellation.

    The batch file is authoritative; a stalled/closed platform log reader may
    miss live bytes. Never buffer an unbounded copy of model output in memory.
    """
    stream = sys.stdout
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError):
        yield  # In-memory streams used by callers/tests cannot block on a pipe.
        return
    blocking = os.get_blocking(fd)

    class LiveOutput:
        dropped = 0

        @property
        def buffer(self):
            return self

        def write(self, data):
            raw = data.encode("utf-8", errors="replace") if isinstance(data, str) else data
            remaining = memoryview(raw)
            while remaining:
                try:
                    count = os.write(fd, remaining)
                except (BlockingIOError, BrokenPipeError):
                    self.dropped += len(remaining)
                    break
                remaining = remaining[count:]
            return len(data)

        def flush(self):
            pass  # os.write is unbuffered.

    try:
        os.set_blocking(fd, False)
        with redirect_stdout(LiveOutput()):
            yield
    finally:
        os.set_blocking(fd, blocking)


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[19], fields[0]  # start time (PID reuse guard), state
    except (FileNotFoundError, ProcessLookupError):
        return None


def children(pid):
    found = set()
    for task in Path(f"/proc/{pid}/task").glob("*/children"):
        try:
            found.update(map(int, task.read_text().split()))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return found


def open_pidfd(pid):
    """Use libc when a portable Python build omits this Linux API."""
    native = getattr(os, "pidfd_open", None)
    if callable(native):
        return native(pid)
    libc = ctypes.CDLL(None, use_errno=True)
    native = getattr(libc, "pidfd_open", None)
    if native is None:
        raise RuntimeError("Native execution requires Linux pidfd_open support")
    native.argtypes = (ctypes.c_int, ctypes.c_uint)
    native.restype = ctypes.c_int
    fd = native(pid, 0)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def signal_pidfd(fd, signum):
    native = getattr(signal, "pidfd_send_signal", None)
    if callable(native):
        return native(fd, signum)
    libc = ctypes.CDLL(None, use_errno=True)
    native = getattr(libc, "pidfd_send_signal", None)
    if native is None:
        raise RuntimeError("Native execution requires Linux pidfd_send_signal support")
    native.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint)
    native.restype = ctypes.c_int
    if native(fd, signum, None, 0):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


@contextmanager
def adopted_descendants():
    """Retain orphaned torchrun/workers until this single-command owner reaps them."""
    if sys.platform != "linux":
        raise RuntimeError("Native execution requires a Linux container with pidfd support")
    # Probe kernel/seccomp support before any child is created.
    probe = open_pidfd(os.getpid())
    try:
        signal_pidfd(probe, 0)
    finally:
        os.close(probe)
    children_file = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children")
    if not children_file.exists():
        raise RuntimeError(
            "Native worker ownership requires /proc task children support "
            "(CONFIG_PROC_CHILDREN)"
        )
    if children(os.getpid()):
        raise RuntimeError("Native worker ownership requires a dedicated process without existing children")
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) or libc.prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), "Cannot enable native-worker subreaping")
    known = {}

    def collect():
        pending = list(children(os.getpid()))
        seen = set()
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            info = identity(pid)
            if info is None:
                continue
            known[pid] = info[0]
            pending.extend(children(pid))
        return {pid: info for pid, start in known.items()
                if (info := identity(pid)) is not None and info[0] == start}

    try:
        yield collect
    finally:
        if libc.prctl(36, previous.value, 0, 0, 0):
            raise OSError(ctypes.get_errno(), "Cannot restore child-subreaper setting")


def send_signal(pid, start, signum):
    """Pin a process before signaling; never signal a recycled PID."""
    try:
        fd = open_pidfd(pid)
    except ProcessLookupError:
        return
    try:
        info = identity(pid)
        if info is not None and info[0] == start:
            signal_pidfd(fd, signum)
    except ProcessLookupError:
        pass
    finally:
        os.close(fd)


def stop_and_reap(child, collect, drain, signum):
    """Bound graceful shutdown, drain its logs, then kill/reap remaining workers."""
    def owned_processes():
        owned = collect()
        # The direct CLI remains ours even if /proc descendant discovery
        # degrades after the pre-launch capability check.
        if (child_info := identity(child.pid)) is not None:
            owned[child.pid] = child_info
        return owned

    started = time.monotonic()
    signaled = set()
    while True:
        owned = owned_processes()
        force = time.monotonic() - started >= 5
        for pid, (start, state) in owned.items():
            if state != "Z" and (force or (pid, start) not in signaled):
                send_signal(pid, start, signal.SIGKILL if force else signum)
                signaled.add((pid, start))
        child.poll()  # Preserve the CLI's exact exit code before reaping workers.
        for pid in owned:
            if pid != child.pid:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass  # Its launcher has not exited/adopted it yet.
        drain(0.05, suppress_errors=True)
        if not owned_processes():
            break
        if time.monotonic() - started > 10:
            raise RuntimeError("Native workers did not stop after SIGKILL")
        time.sleep(0.01)
    child.wait(timeout=1)
    # Reaped descendants have closed every writer, but buffered bytes may remain.
    while drain(0.05, suppress_errors=True):
        pass


def run_child(command, *, stdout, stderr, start_new_session, cancellation):
    """Tee live bytes to the platform and batch log, owning every native worker.

    Called by the single-threaded batch CLI, one native command at a time.
    Session isolation protects it from the native CLI's teardown SIGINT;
    subreaping handles torchrun's further isolated sessions without daemon help.
    """
    cancellation.check()
    with live_output(), adopted_descendants() as collect, selectors.DefaultSelector() as selector:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr,
                                 start_new_session=start_new_session)
        output_errors = []
        dropped_before = getattr(sys.stdout, "dropped", 0)

        def drain(timeout, suppress_errors=False):
            for key, _ in selector.select(timeout):
                block = os.read(key.fd, 65536)
                if not block:
                    selector.unregister(key.fileobj)
                    continue
                for target in (stdout, getattr(sys.stdout, "buffer", sys.stdout)):
                    try:
                        target.write(block if target is stdout or hasattr(sys.stdout, "buffer")
                                     else block.decode("utf-8", errors="replace"))
                        target.flush()
                    except (OSError, ValueError) as exc:
                        if not output_errors:
                            output_errors.append(exc)
                        if not suppress_errors:
                            raise
            return bool(selector.get_map())

        try:
            selector.register(child.stdout, selectors.EVENT_READ)
            while child.poll() is None:
                if cancellation.signal is not None:
                    break
                drain(0.1)
        finally:
            try:
                stop_and_reap(child, collect, drain, cancellation.signal or signal.SIGTERM)
            finally:
                child.stdout.close()
        dropped = getattr(sys.stdout, "dropped", 0) - dropped_before
        if dropped and not output_errors:
            stdout.write(f"\n[wrapper] Platform log reader stalled/closed; {dropped} live bytes "
                         "omitted. Complete native output is retained in this file.\n".encode())
            stdout.flush()
    if output_errors:
        raise output_errors[0]
    return subprocess.CompletedProcess(command, child.returncode)
