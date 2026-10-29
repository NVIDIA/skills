#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a redacted, owner-only copy of a UTF-8 support text file."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from typing import Pattern


MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_LINE_BYTES = 64 * 1024
MAX_KEY_BYTES = 1024 * 1024
MAX_SECONDS = 30
KEY_MARKER = re.compile(r"-----(BEGIN|END) ([^-\r\n]*PRIVATE KEY)-----", re.I)
KEY_LABELS = {
    "PRIVATE KEY", "ENCRYPTED PRIVATE KEY", "RSA PRIVATE KEY",
    "EC PRIVATE KEY", "DSA PRIVATE KEY", "OPENSSH PRIVATE KEY",
}


SENSITIVE_NAME = (
    r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"auth(?:orization)?(?:[_-]?token)?|session[_-]?token|bearer[_-]?token|"
    r"csrf[_-]?token|token|password|passwd|passphrase|secret(?:[_-]?key)?|"
    r"cookie|private[_-]?key|ssh[_-]?key|client[_-]?secret)"
)

SENSITIVE_IDENTIFIER = (
    rf"(?<![A-Za-z0-9_-]){SENSITIVE_NAME}(?![A-Za-z0-9_-])"
)


PATTERNS: tuple[tuple[str, Pattern[str]], ...] = (
    (
        "private_key_block",
        re.compile(
            r"-----BEGIN [^-\n]*PRIVATE KEY-----.*?"
            r"-----END [^-\n]*PRIVATE KEY-----",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "sensitive_header",
        re.compile(
            r"(?im)^([ \t]*(?:authorization|proxy-authorization|cookie|set-cookie)"
            r"[ \t]*(?:=|:))(?![ \t]*<REDACTED>[ \t]*$)[ \t]*.+$"
        ),
    ),
    (
        "sensitive_url_parameter",
        re.compile(
            rf"(?i)([?&]{SENSITIVE_NAME}=)(?!<REDACTED>)[^&#\s]+"
        ),
    ),
    (
        "sensitive_assignment",
        re.compile(
            rf"(?im)([\"']?{SENSITIVE_IDENTIFIER}[\"']?[ \t]*(?:=|:)[ \t]*)"
            r"(?![\"']?<REDACTED>[\"']?)"
            r"(?:\"[^\"\n]*\"|'[^'\n]*'|[^\s,;\]\}}]+)"
        ),
    ),
    (
        "sensitive_cli_flag",
        re.compile(
            rf"(?i)((?:--|(?<![A-Za-z0-9_-])){SENSITIVE_NAME}"
            r"(?![A-Za-z0-9_-])(?:[ \t]*=[ \t]*|[ \t]+(?![ \t]*=)))"
            r"(?![\"']?<REDACTED>[\"']?)"
            r"(?:\"[^\"\n]*\"|'[^'\n]*'|\S+)"
        ),
    ),
    (
        "jwt_like_token",
        re.compile(r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"),
    ),
    (
        "known_credential_prefix",
        re.compile(
            r"(?<![A-Za-z0-9])(?:AKIA[0-9A-Z]{16}|"
            r"gh[pousr]_[A-Za-z0-9]{20,}|nvapi-[A-Za-z0-9_-]{16,}|"
            r"sk-[A-Za-z0-9_-]{20,})(?![A-Za-z0-9])"
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    """Read input and select count-only checking or redacted-file output."""
    parser = argparse.ArgumentParser(
        description=(
            "Redact common credentials from a UTF-8 text file without printing "
            "matched values. Automated redaction still requires manual review."
        )
    )
    parser.add_argument("input", type=Path, help="UTF-8 text file to inspect")
    parser.add_argument(
        "--output",
        type=Path,
        help="new redacted output path; required unless --check is used",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report match counts only; exit 1 when suspected secrets remain",
    )
    return parser.parse_args()


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Replace recognized secret patterns and return per-pattern match counts."""
    counts: dict[str, int] = {}
    redacted = text

    for name, pattern in PATTERNS:
        if name in ("sensitive_assignment", "sensitive_cli_flag"):
            def replacement(match: re.Match[str]) -> str:
                """Preserve the assignment prefix and matching quotes around redacted values."""
                prefix = match.group(1)
                value = match.group(0)[len(prefix):]
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    return f"{prefix}{value[0]}<REDACTED>{value[0]}"
                return f"{prefix}<REDACTED>"
        else:
            replacement = r"\1<REDACTED>" if pattern.groups else "<REDACTED>"
        redacted, count = pattern.subn(replacement, redacted)
        counts[name] = count

    return redacted, counts


def redact_stream(source, output=None) -> dict[str, int]:
    """Scan bounded UTF-8 lines, retaining private-key state across lines."""
    counts = {name: 0 for name, _ in PATTERNS}
    total = 0
    label = None
    key_bytes = 0
    pending = ""
    while True:
        payload = source.readline(min(MAX_LINE_BYTES, MAX_INPUT_BYTES - total) + 1)
        if not payload:
            break
        total += len(payload)
        if total > MAX_INPUT_BYTES:
            raise ValueError("input exceeds 64 MiB; select a smaller relevant excerpt")
        if len(payload) > MAX_LINE_BYTES:
            raise ValueError("line exceeds 64 KiB; select a smaller complete record")
        if b"\x00" in payload:
            raise ValueError("binary input is not supported")
        try:
            line = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("input must be UTF-8 text; binary redaction is unsafe") from exc

        offset = 0
        for marker in KEY_MARKER.finditer(line):
            action, found = marker.group(1).upper(), marker.group(2).upper()
            if label is not None:
                key_bytes += len(line[offset:marker.end()].encode("utf-8"))
                if key_bytes > MAX_KEY_BYTES:
                    raise ValueError("private-key block exceeds 1 MiB")
                if action != "END" or found != label:
                    raise ValueError("nested or mismatched private-key marker")
                label = None
                counts["private_key_block"] += 1
            else:
                if action != "BEGIN":
                    raise ValueError("unmatched private-key closing marker")
                if found not in KEY_LABELS:
                    raise ValueError("unsupported private-key label")
                pending += line[offset:marker.start()] + "<REDACTED>"
                label = found
                key_bytes = len(marker.group().encode("utf-8"))
            offset = marker.end()
        if label is not None:
            key_bytes += len(line[offset:].encode("utf-8"))
            if key_bytes > MAX_KEY_BYTES:
                raise ValueError("private-key block exceeds 1 MiB")
        else:
            pending += line[offset:]
        if len(pending.encode("utf-8")) > MAX_LINE_BYTES:
            raise ValueError("redacted record exceeds 64 KiB")
        if label is None:
            # Match the retained prefix and suffix together, never raw read chunks.
            ending = "\r\n" if pending.endswith("\r\n") else "\n" if pending.endswith("\n") else ""
            body = pending[:-len(ending)] if ending else pending
            clean, matches = redact(body)
            clean += ending
            for name, count in matches.items():
                counts[name] += count
            if output is not None:
                output.write(clean)
            pending = ""
    if label is not None:
        raise ValueError("unfinished private-key block")
    return counts


def restrict_windows_access(path: Path) -> None:
    """Set a protected owner-only DACL before writing sensitive Windows output."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    security = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    convert = security.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
                        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    convert.restype = wintypes.BOOL
    apply = security.SetFileSecurityW
    apply.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p]
    apply.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    descriptor = ctypes.c_void_p()
    # OW = owner rights; OI/CI propagate to new files and directories.
    inheritance = "OICI" if path.is_dir() else ""
    if not convert(f"D:P(A;{inheritance};FA;;;OW)", 1, ctypes.byref(descriptor), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        # DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION
        if not apply(str(path), 0x80000004, descriptor):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.LocalFree(descriptor)


def process_file(source_path: Path, output_path: Path | None) -> dict[str, int]:
    """Scan a regular file and atomically publish a private, non-overwriting copy."""
    temporary = None
    try:
        # O_NONBLOCK avoids hanging when an input path names a FIFO.
        if not stat.S_ISREG(source_path.stat().st_mode):
            raise ValueError("input must be a regular file")
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        descriptor = os.open(source_path, flags)
        with os.fdopen(descriptor, "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise ValueError("input must be a regular file")
            if output_path is None:
                return redact_stream(source)
            if source_path.resolve() == output_path.resolve():
                raise ValueError("output must differ from input")
            output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix=".redaction-", dir=output_path.parent)
            temporary = Path(name)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
                restrict_windows_access(temporary)
                counts = redact_stream(source, output)
            # Linking a complete file is atomic and refuses existing destinations.
            os.link(temporary, output_path)
            return counts
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def redaction_worker(source_path: Path, output_path: Path | None, connection) -> None:
    """Return counts or a handled error to the supervising process."""
    try:
        connection.send((True, process_file(source_path, output_path)))
    except (OSError, ValueError, KeyboardInterrupt) as exc:
        connection.send((False, str(exc)))
    finally:
        connection.close()


def run_bounded(source_path: Path, output_path: Path | None) -> dict[str, int]:
    """Enforce a process timeout and publish only completed, private output."""
    staging = None
    process = None
    receiver, sender = multiprocessing.get_context("spawn").Pipe(duplex=False)
    try:
        staged_output = None
        if output_path is not None:
            if source_path.resolve() == output_path.resolve():
                raise ValueError("output must differ from input")
            output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=".redaction-", dir=output_path.parent))
            restrict_windows_access(staging)
            staged_output = staging / "output"
        process = multiprocessing.get_context("spawn").Process(
            target=redaction_worker, args=(source_path, staged_output, sender))
        process.start()
        sender.close()
        process.join(MAX_SECONDS)
        if process.is_alive():
            raise ValueError(f"redaction exceeded {MAX_SECONDS} seconds; select a smaller relevant excerpt")
        if process.exitcode != 0 or not receiver.poll():
            raise ValueError("redaction worker failed; no output published")
        try:
            success, result = receiver.recv()
        except EOFError as exc:
            raise ValueError("redaction worker returned no result") from exc
        if not success:
            raise ValueError(result)
        if output_path is not None:
            os.link(staged_output, output_path)
        return result
    finally:
        if process is not None and process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join()
            process.close()
        receiver.close()
        sender.close()
        if staging is not None:
            shutil.rmtree(staging)


def main() -> int:
    """Check or redact input without printing matched secrets; report counts only."""
    args = parse_args()

    if args.check and args.output is not None:
        print("error: --check cannot be combined with --output", file=sys.stderr)
        return 2
    if not args.check and args.output is None:
        print("error: --output is required unless --check is used", file=sys.stderr)
        return 2

    try:
        counts = run_bounded(args.input, None if args.check else args.output)
        if args.check:
            print(json.dumps({"suspected_matches": counts}, sort_keys=True))
            return 1 if any(counts.values()) else 0
    except (OSError, ValueError, KeyboardInterrupt) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({"redactions": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
