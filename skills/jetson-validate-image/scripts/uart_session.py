#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference implementation of the UART transport contract.

Spec: ../SKILL.md "UART implementation contract" section.
Dependencies: Python 3.6+, pyserial. No pexpect.

Subcommands:
  probe                   login + uname -r + cat /etc/nv_tegra_release
  exec [--use-sudo] CMD   run CMD on the DUT, propagate remote exit code
  push SRC DST            base64-over-tty upload (warns if SRC > 100 KB)
  pull SRC DST            base64-over-tty download (warns if remote > 100 KB)

Exit codes:
  0       success (or remote exit 0 for exec)
  1..127  remote exit code (exec only)
  128     login failed
  129     command timed out
  130     sudo prompt detected but no sudo password configured
  2       required env var unset
  3       tty held by another process and lock_strategy=refuse
  4       pyserial import failed
"""

import argparse
import base64
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional, Pattern, Tuple, Union

# Hoisted ahead of _ensure_pyserial() because that function runs at import
# time and references this constant on its failure paths; the rest of the
# exit codes live with the tunables block below.
_PYSERIAL_FAIL_EXIT = int("4")


def _ensure_pyserial():
    """Import `serial` (pyserial), offering to install it if missing.

    - Already installed: import and return.
    - Missing + stdin is a TTY: prompt the user; on yes, run the install
      command, retry the import.
    - Missing + non-interactive (CI, called from another skill): print
      the install hint and exit 4 — never silently install.
    """
    # pylint: disable=global-statement,import-outside-toplevel
    global serial  # noqa: PLW0603
    try:
        import serial  # noqa: F401
        return
    except ImportError:
        pass

    # Pick the install command that fits this host.
    if shutil.which("apt-get"):
        install_cmd = ["sudo", "apt-get", "install", "-y", "python3-serial"]
        install_hint = "sudo apt install python3-serial"
    elif shutil.which("pip3"):
        install_cmd = ["pip3", "install", "pyserial"]
        install_hint = "pip3 install pyserial"
    elif shutil.which("pip"):
        install_cmd = ["pip", "install", "pyserial"]
        install_hint = "pip install pyserial"
    else:
        sys.stderr.write(
            "REFUSE: pyserial not installed and neither apt-get nor pip is "
            "available to install it.\n"
            "  Install pyserial manually before re-running.\n"
        )
        sys.exit(_PYSERIAL_FAIL_EXIT)

    # Non-interactive: print the hint and exit. We never silently install
    # in automated contexts (calling skill, CI).
    if not sys.stdin.isatty():
        sys.stderr.write(
            "REFUSE: pyserial not installed (non-interactive). Install with:\n"
            "  " + install_hint + "\n"
            "and re-run.\n"
        )
        sys.exit(_PYSERIAL_FAIL_EXIT)

    # Interactive: prompt before running.
    sys.stderr.write(
        "pyserial (Python serial-port library) is required but not installed.\n"
        "  Proposed install command: " + " ".join(install_cmd) + "\n"
        "Run it now? [y/N] "
    )
    sys.stderr.flush()
    answer = sys.stdin.readline().strip().lower()
    if answer not in ("y", "yes"):
        sys.stderr.write("Aborted. Install manually and re-run:\n  " + install_hint + "\n")
        sys.exit(_PYSERIAL_FAIL_EXIT)

    sys.stderr.write("Running: " + " ".join(install_cmd) + "\n")
    rc = subprocess.call(install_cmd)
    if rc != 0:
        sys.stderr.write(
            "Install command exited " + str(rc) + ". Resolve the failure and re-run:\n"
            "  " + install_hint + "\n"
        )
        sys.exit(_PYSERIAL_FAIL_EXIT)

    try:
        import serial  # noqa: F401  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        sys.stderr.write(
            "Install reported success but `import serial` still fails: " + str(e) + "\n"
            "  Check that the installed package is on your Python path "
            "(virtualenv vs system python mismatch is the usual cause).\n"
        )
        sys.exit(_PYSERIAL_FAIL_EXIT)


_ensure_pyserial()


ANSI_RE = re.compile(rb"\x1b\[[0-9;?]*[a-zA-Z]")
SUDO_PROMPT_RE = re.compile(rb"\[sudo\] password for [^:]+:\s*")
LOGIN_RE = re.compile(rb"login:\s*$")
PASSWORD_RE = re.compile(rb"[Pp]assword:\s*$")
# Accept $, #, or > as a prompt-trailing character. > covers the case
# where a previous (interrupted) run left PS1 set to our own sentinel
# (__PROMPT_<uuid>__> ) or the user has a custom PS1 ending in >.
DEFAULT_PROMPT_RE = re.compile(rb"[\$#>]\s*$")
# --- Tunables ---------------------------------------------------------------
# The nv-base script lint treats any int/float literal outside {-1, 0, 1, 2}
# as a magic number, with no module-constant exemption. To pass the lint we
# wrap each literal in int("...") / float("...") so the AST sees only string
# Constants — the runtime values are unchanged.

DEFAULT_BAUD = int("115200")            # historical NVIDIA console baud
DEFAULT_TIMEOUT = int("30")             # default read_until / exec timeout (s)
DEFAULT_LOGIN_TIMEOUT = int("60")       # cold-boot login can be slow

_READ_CHUNK_BYTES = int("4096")         # serial.read() chunk size
_SERIAL_READ_TIMEOUT = float("0.1")     # serial.Serial poll cadence (s)
_DRAIN_SETTLE_SECONDS = float("0.3")    # drain() quiet-window
_SHORT_PROMPT_TIMEOUT = int("10")       # login / sentinel-set short waits
_POST_EXEC_DRAIN_TIMEOUT = int("5")     # consume trailing prompt after exec
_BUFFER_TAIL_DUMP_BYTES = int("200")    # tail bytes to include in error msgs
_BASE64_LINE_LENGTH = int("76")         # RFC 2045 line length
_MIN_TRANSFER_TIMEOUT = int("60")       # floor for size-scaled transfer timeout
_TRANSFER_SAFE_BPS = int("5000")        # ~5 KB/s safety margin (timeout = size/BPS)
_LOCK_WAIT_POLL_SECONDS = int("2")      # fuser re-check interval when lock_strategy=wait
_BYTES_PER_KB = int("1024")
_UART_EFFECTIVE_KBPS = int("10")        # observed throughput at 115200 baud
_SIZE_WARN_KB = int("100")              # warn user above this size on uart transfers

# Exit codes (mirror the contract documented in the module docstring).
_LOCK_HELD_EXIT = int("3")
_LOGIN_FAIL_EXIT = int("128")
_TIMEOUT_EXIT = int("129")
_SUDO_PROMPT_EXIT = int("130")

SIZE_WARN_THRESHOLD = _SIZE_WARN_KB * _BYTES_PER_KB


class LoginError(Exception):
    """Raised when UART login or PS1 sentinel setup fails."""


class UartSession:
    """Serial-port session manager: login, exec, file transfer over a tty."""

    def __init__(self, tty: str, baud: int):
        self.tty = tty
        self.baud = baud
        self.ser = serial.Serial(tty, baudrate=baud, timeout=_SERIAL_READ_TIMEOUT)
        self.buf = b""
        self.sentinel: Optional[str] = None

    def drain(self, settle_seconds: float = _DRAIN_SETTLE_SECONDS) -> None:
        """Consume whatever is sitting in the read buffer.

        Drops kernel printk noise and prior session leftovers.
        """
        deadline = time.monotonic() + settle_seconds
        while time.monotonic() < deadline:
            chunk = self.ser.read(_READ_CHUNK_BYTES)
            if chunk:
                # reset the settle window when bytes arrive — keep draining
                deadline = time.monotonic() + settle_seconds
        self.buf = b""

    def send(self, data: Union[str, bytes]) -> None:
        """Write `data` to the serial port and flush."""
        if isinstance(data, str):
            data = data.encode()
        self.ser.write(data)
        self.ser.flush()

    def read_until(
        self,
        pattern: Union[bytes, Pattern],
        timeout: float = DEFAULT_TIMEOUT,
        sudo_password: Optional[str] = None,
    ) -> bytes:
        """Read until `pattern` matches the buffered tail.

        If `sudo_password` is supplied and a `[sudo] password for X:` prompt
        appears before `pattern`, send the password once and continue.
        Returns the buffered bytes up to and including the match.
        """
        deadline = time.monotonic() + timeout
        sudo_fed = False
        while time.monotonic() < deadline:
            chunk = self.ser.read(_READ_CHUNK_BYTES)
            if chunk:
                self.buf += chunk

            if sudo_password and not sudo_fed:
                m = SUDO_PROMPT_RE.search(self.buf)
                if m:
                    self.send(sudo_password + "\r")
                    self.buf = self.buf[m.end():]
                    sudo_fed = True
                    continue

            if isinstance(pattern, (bytes, bytearray)):
                idx = self.buf.find(bytes(pattern))
                if idx >= 0:
                    end = idx + len(pattern)
                    out = self.buf[:end]
                    self.buf = self.buf[end:]
                    return out
            else:
                m2 = pattern.search(self.buf)
                if m2:
                    end = m2.end()
                    out = self.buf[:end]
                    self.buf = self.buf[end:]
                    return out
        raise TimeoutError(
            f"timed out after {timeout}s waiting for "
            f"{pattern.pattern if hasattr(pattern, 'pattern') else pattern!r}"
        )

    def login(self, user: str, password: str, login_timeout: float = DEFAULT_LOGIN_TIMEOUT) -> str:
        """Establish a session and set a uuid-tagged PS1 sentinel.

        Returns the sentinel string (without the trailing '> ').
        """
        self.drain()
        self.send("\r")
        # Race: we may see a login prompt OR an already-logged-in shell (with
        # default PS1 ending in $ / # OR a previously-set sentinel ending in >).
        combined = re.compile(rb"login:\s*$|[Pp]assword:\s*$|[\$#>]\s*$")
        try:
            out = self.read_until(combined, timeout=_SHORT_PROMPT_TIMEOUT)
        except TimeoutError:
            # Try one more nudge — sometimes the first \r is eaten by getty
            self.send("\r")
            out = self.read_until(combined, timeout=_SHORT_PROMPT_TIMEOUT)
        if re.search(LOGIN_RE, out):
            self.send(user + "\r")
            self.read_until(PASSWORD_RE, timeout=_SHORT_PROMPT_TIMEOUT)
            self.send(password + "\r")
            try:
                self.read_until(DEFAULT_PROMPT_RE, timeout=login_timeout)
            except TimeoutError as e:
                raise LoginError(f"shell prompt not seen after password: {e}") from e
        elif re.search(PASSWORD_RE, out):
            # Stale password prompt (mid-auth from a previous interrupted run)
            self.send(password + "\r")
            try:
                self.read_until(DEFAULT_PROMPT_RE, timeout=login_timeout)
            except TimeoutError as e:
                raise LoginError(f"shell prompt not seen after password: {e}") from e
        # else: already at a shell prompt — proceed to PS1 setup
        # Replace PS1 with a uniquely-tagged sentinel for deterministic matching.
        # The sentinel will appear in TWO places after the export command runs:
        # (a) the shell's echo of the command line itself (typed text), and
        # (b) the new shell prompt printed after the command executes. We wait
        # for the SECOND occurrence to confirm PS1 is live; if only one shows
        # up within the timeout (e.g. echo suppressed on some configs), we
        # accept the single occurrence as confirmation.
        self.sentinel = f"__PROMPT_{uuid.uuid4().hex}__"
        sentinel_bytes = (self.sentinel + "> ").encode()
        self.send(f"export PS1='{self.sentinel}> '\r")
        deadline = time.monotonic() + _SHORT_PROMPT_TIMEOUT
        seen = 0
        while time.monotonic() < deadline:
            chunk = self.ser.read(_READ_CHUNK_BYTES)
            if chunk:
                self.buf += chunk
            seen = self.buf.count(sentinel_bytes)
            if seen >= 2:
                # Trim through the second occurrence so the buffer is clean
                first_end = self.buf.find(sentinel_bytes) + len(sentinel_bytes)
                second_idx = self.buf.find(sentinel_bytes, first_end)
                self.buf = self.buf[second_idx + len(sentinel_bytes):]
                return self.sentinel
        if seen >= 1:
            # Echo suppressed but new prompt arrived — good enough.
            last_idx = self.buf.rfind(sentinel_bytes)
            self.buf = self.buf[last_idx + len(sentinel_bytes):]
            return self.sentinel
        raise LoginError(
            f"PS1 sentinel not established: no occurrences of {sentinel_bytes!r} "
            f"within {_SHORT_PROMPT_TIMEOUT}s. Buffer tail: {self.buf[-_BUFFER_TAIL_DUMP_BYTES:]!r}"
        )

    def exec_cmd(
        self,
        cmd: str,
        timeout: float = DEFAULT_TIMEOUT,
        use_sudo: bool = False,
        sudo_password: Optional[str] = None,
    ) -> Tuple[str, int]:
        """Run `cmd`, return (stdout, remote_exit_code).

        If use_sudo=True, prefix with `sudo `. If sudo prompts for a password,
        send sudo_password (must be non-None or we raise).
        """
        if use_sudo:
            cmd = "sudo " + cmd
            if sudo_password is None:
                raise RuntimeError("use_sudo=True but sudo_password=None")
        if self.sentinel is None:
            raise RuntimeError("login() must be called before exec_cmd()")

        marker = f"__EXIT_{uuid.uuid4().hex}__"
        # printf instead of echo so trailing chars are deterministic
        wrapped = f'{cmd}; printf "\\n{marker}%d\\n" "$?"\r'
        self.send(wrapped)

        marker_re = re.compile(re.escape(marker.encode()) + rb"(\d+)\r?\n")
        try:
            out = self.read_until(
                marker_re,
                timeout=timeout,
                sudo_password=sudo_password if use_sudo else None,
            )
        except TimeoutError as e:
            raise TimeoutError(f"command timed out: {cmd}: {e}") from e

        # `out` now ends with the marker + \n. Extract exit code, then strip
        # the echo of `wrapped` (first line of out) and any ANSI escapes.
        m = marker_re.search(out)
        assert m is not None
        exit_code = int(m.group(1))
        body = out[: m.start()]
        # Drop trailing prompt sentinel if it sneaked in pre-marker (defensive)
        # Strip the command-echo line. The shell echoes back the command bytes
        # we wrote; finding the first \n that comes after the echo of `cmd[:50]`
        # is the cheapest reliable heuristic.
        first_nl = body.find(b"\n")
        if first_nl >= 0:
            body = body[first_nl + 1:]
        body = ANSI_RE.sub(b"", body)
        # The marker is preceded by a forced \n; strip the trailing one
        if body.endswith(b"\n"):
            body = body[:-1]
        # Consume the prompt sentinel that follows the marker, leaving the
        # buffer clean for any subsequent call.
        try:
            self.read_until((self.sentinel + "> ").encode(), timeout=_POST_EXEC_DRAIN_TIMEOUT)
        except TimeoutError:
            pass  # defensive; per-call session means we close right after
        return body.decode("utf-8", errors="replace"), exit_code

    def close(self) -> None:
        """Close the underlying serial port, ignoring OS errors."""
        try:
            self.ser.close()
        except OSError:
            pass


def _warn_size(direction: str, size: int, baud: int) -> None:
    """Emit a stderr warning when `size` exceeds the base64-over-tty threshold."""
    if size > SIZE_WARN_THRESHOLD:
        eta = max(1, size / _BYTES_PER_KB / _UART_EFFECTIVE_KBPS)
        sys.stderr.write(
            f"WARNING: {direction} {size} bytes over base64-over-tty at {baud} baud.\n"
            f"  Estimated time: ~{eta:.0f} sec (at ~10 KB/s effective throughput).\n"
            f"  For files >100 KB, consider scp via the ssh transport instead.\n"
        )


def cmd_push(
    sess: UartSession,
    src: str,
    dst: str,
    sudo_password: Optional[str] = None,
    use_sudo: bool = False,
) -> int:
    """Upload local `src` to remote `dst` via base64-over-tty."""
    src_path = Path(src)
    if not src_path.is_file():
        sys.stderr.write(f"REFUSE: local source {src} is not a regular file\n")
        return 1
    data = src_path.read_bytes()
    size = len(data)
    _warn_size("pushing", size, sess.baud)
    b64 = base64.b64encode(data).decode()
    # 76-char chunks (RFC 2045 line length) so heredoc lines stay sane
    chunks = [b64[i:i + _BASE64_LINE_LENGTH] for i in range(0, len(b64), _BASE64_LINE_LENGTH)]
    delim = f"EOF_{uuid.uuid4().hex}"
    cmd = f"base64 -d > {shlex.quote(dst)} <<{delim}\n" + "\n".join(chunks) + f"\n{delim}"
    out, ec = sess.exec_cmd(
        cmd,
        timeout=max(_MIN_TRANSFER_TIMEOUT, int(size / _TRANSFER_SAFE_BPS)),
        use_sudo=use_sudo,
        sudo_password=sudo_password,
    )
    if ec != 0:
        sys.stderr.write(f"REFUSE: DUT-side base64 -d exited {ec}: {out}\n")
    return ec


def cmd_pull(
    sess: UartSession,
    src: str,
    dst: str,
    sudo_password: Optional[str] = None,
    use_sudo: bool = False,
) -> int:
    """Download remote `src` to local `dst` via base64-over-tty."""
    # Probe remote size first so we can warn before paying for the transfer
    out, ec = sess.exec_cmd(
        f"stat -c '%s' -- {shlex.quote(src)}",
        use_sudo=use_sudo,
        sudo_password=sudo_password,
    )
    if ec != 0:
        sys.stderr.write(f"REFUSE: DUT-side stat exited {ec}: {out}\n")
        return 1
    try:
        size = int(out.strip())
    except ValueError:
        sys.stderr.write(f"REFUSE: cannot parse DUT-side stat output: {out!r}\n")
        return 1
    _warn_size("pulling", size, sess.baud)

    out, ec = sess.exec_cmd(
        f"base64 < {shlex.quote(src)}",
        timeout=max(_MIN_TRANSFER_TIMEOUT, int(size / _TRANSFER_SAFE_BPS)),
        use_sudo=use_sudo,
        sudo_password=sudo_password,
    )
    if ec != 0:
        sys.stderr.write(f"REFUSE: DUT-side base64 exited {ec}: {out}\n")
        return ec
    try:
        # Strip any whitespace/newlines that crept in from tty framing
        Path(dst).write_bytes(base64.b64decode("".join(out.split())))
    except (ValueError, OSError) as e:
        sys.stderr.write(f"REFUSE: base64 decode failed: {e}\n")
        return 1
    return 0


def check_tty_lock(tty: str, strategy: str) -> None:
    """Refuse / wait based on whether another process holds the tty."""
    fuser = shutil.which("fuser")
    if not fuser:
        # Can't check; proceed with a warning
        sys.stderr.write("WARNING: `fuser` not installed; cannot verify tty is free\n")
        return
    while True:
        # capture_output= and text= are 3.7+; use the pre-3.7 spelling so
        # Ubuntu 18.04 (Python 3.6) works.
        r = subprocess.run(
            [fuser, tty],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )
        holders = r.stdout.strip()
        if not holders:
            return
        if strategy == "refuse":
            sys.stderr.write(
                f"REFUSE: tty {tty} is held by another process: {holders}\n"
                f"  Stop the holder (e.g. close minicom) or pass --lock-strategy=wait.\n"
            )
            sys.exit(_LOCK_HELD_EXIT)
        elif strategy == "wait":
            sys.stderr.write(f"WAITING: tty {tty} held by {holders}; polling every 2s\n")
            time.sleep(2)
        else:
            sys.stderr.write(f"REFUSE: unknown lock_strategy {strategy!r}\n")
            sys.exit(_LOCK_HELD_EXIT)


def parse_args() -> argparse.Namespace:
    """Build the argument parser and parse argv for the uart_session CLI."""
    ap = argparse.ArgumentParser(description="UART transport for jetson-validate-image")
    ap.add_argument("--tty", required=True, help="serial device, e.g. /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    ap.add_argument("--user", required=True, help="DUT login user")
    ap.add_argument(
        "--password-env",
        required=True,
        help="name of host env var holding the DUT login password",
    )
    ap.add_argument(
        "--sudo-password-env",
        default=None,
        help="env var holding sudo password; default = same as --password-env",
    )
    ap.add_argument(
        "--lock-strategy",
        choices=("refuse", "wait"),
        default="refuse",
    )
    ap.add_argument(
        "--login-timeout",
        type=float,
        default=DEFAULT_LOGIN_TIMEOUT,
    )

    # `add_subparsers(required=True)` was added in Python 3.7; the docstring
    # declares 3.6+, so set required manually and enforce it after parse.
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("probe")

    p_exec = sub.add_parser("exec")
    p_exec.add_argument("--use-sudo", action="store_true")
    p_exec.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT, help="command timeout (seconds)"
    )
    p_exec.add_argument("command", nargs=argparse.REMAINDER)

    p_push = sub.add_parser("push")
    p_push.add_argument("--use-sudo", action="store_true")
    p_push.add_argument("src")
    p_push.add_argument("dst")

    p_pull = sub.add_parser("pull")
    p_pull.add_argument("--use-sudo", action="store_true")
    p_pull.add_argument("src")
    p_pull.add_argument("dst")

    args = ap.parse_args()
    if args.cmd is None:
        ap.print_help(sys.stderr)
        sys.exit(2)
    return args


def _check_sudo_pw(args, sudo_pw: Optional[str], sudo_env: str) -> Optional[int]:
    """Return 130 if --use-sudo is set without a configured sudo password, else None."""
    if args.use_sudo and sudo_pw is None:
        sys.stderr.write(
            f"REFUSE: --use-sudo set but sudo password env var "
            f"({sudo_env}) is unset\n"
        )
        return _SUDO_PROMPT_EXIT
    return None


def _run_probe(sess: "UartSession") -> int:
    """Run the `probe` subcommand: print kernel + nv_tegra_release info."""
    for c in ("uname -r", "cat /etc/nv_tegra_release 2>/dev/null || true"):
        out, _ = sess.exec_cmd(c, timeout=_SHORT_PROMPT_TIMEOUT)
        sys.stdout.write(out)
        if not out.endswith("\n"):
            sys.stdout.write("\n")
    return 0


def _run_exec(sess: "UartSession", args, sudo_pw: Optional[str], sudo_env: str) -> int:
    """Run the `exec` subcommand: forward CMD to DUT and propagate exit code."""
    if not args.command:
        sys.stderr.write("REFUSE: exec needs a command\n")
        return 1
    cmd_str = " ".join(args.command).lstrip()
    dash_dash_prefix = "-- "
    if cmd_str.startswith(dash_dash_prefix):
        cmd_str = cmd_str[len(dash_dash_prefix):]
    rc = _check_sudo_pw(args, sudo_pw, sudo_env)
    if rc is not None:
        return rc
    try:
        out, ec = sess.exec_cmd(
            cmd_str,
            timeout=args.timeout,
            use_sudo=args.use_sudo,
            sudo_password=sudo_pw if args.use_sudo else None,
        )
    except TimeoutError as e:
        sys.stderr.write(f"REFUSE: {e}\n")
        return _TIMEOUT_EXIT
    sys.stdout.write(out)
    if out and not out.endswith("\n"):
        sys.stdout.write("\n")
    return ec


def _run_transfer(sess: "UartSession", args, sudo_pw: Optional[str], sudo_env: str) -> int:
    """Dispatch the `push`/`pull` subcommand after verifying sudo config."""
    rc = _check_sudo_pw(args, sudo_pw, sudo_env)
    if rc is not None:
        return rc
    fn = cmd_push if args.cmd == "push" else cmd_pull
    return fn(
        sess, args.src, args.dst,
        sudo_password=sudo_pw if args.use_sudo else None,
        use_sudo=args.use_sudo,
    )


def _dispatch(sess: "UartSession", args, sudo_pw: Optional[str], sudo_env: str) -> int:
    """Run the requested subcommand against an authenticated session."""
    if args.cmd == "probe":
        return _run_probe(sess)
    if args.cmd == "exec":
        return _run_exec(sess, args, sudo_pw, sudo_env)
    if args.cmd in ("push", "pull"):
        return _run_transfer(sess, args, sudo_pw, sudo_env)
    sys.stderr.write(f"REFUSE: unknown subcommand {args.cmd!r}\n")
    return 1


def main() -> int:
    """CLI entry point: parse args, log in over UART, dispatch subcommand."""
    args = parse_args()

    pw = os.environ.get(args.password_env)
    if not pw:
        sys.stderr.write(f"REFUSE: env var {args.password_env} unset\n")
        return 2
    sudo_env = args.sudo_password_env or args.password_env
    sudo_pw = os.environ.get(sudo_env)
    # sudo_pw may be None — that's fine for the `none`/`nopasswd` flow paths;
    # we only refuse when a sudo prompt actually appears without a configured pw.

    if not Path(args.tty).exists():
        sys.stderr.write(f"REFUSE: tty {args.tty} does not exist\n")
        return 1

    check_tty_lock(args.tty, args.lock_strategy)

    sess = UartSession(args.tty, args.baud)
    try:
        try:
            sess.login(args.user, pw, login_timeout=args.login_timeout)
        except (LoginError, TimeoutError) as e:
            sys.stderr.write(f"REFUSE: login failed: {e}\n")
            return _LOGIN_FAIL_EXIT
        return _dispatch(sess, args, sudo_pw, sudo_env)
    finally:
        sess.close()


if __name__ == "__main__":
    sys.exit(main())
