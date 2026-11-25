#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block until a long stage finishes, by watching what it produces.

Polls for whichever condition is given:

The first condition that holds ends the wait. So an ``--artifact`` the stage creates
while starting up — an output directory, a labels directory — ends it immediately and
the stage reads as finished when it has barely begun. Name only artifacts that appear
on success; when the outputs appear before completion, wait on ``--status-json`` alone.

* ``--artifact`` — a path the stage writes on success. A checkpoint symlink counts
  only once it resolves, so a dangling link is not mistaken for completion.
* ``--status-json`` with ``--status-contains`` — TAO writes one JSON object per
  line to ``status.json``; the stage is done when a line's ``message`` matches.
  Use this when the artifact path is not known up front.

Pass ``--newer-than`` with a timestamp taken just before launching: without it a
retry is satisfied by whatever the failed attempt left on disk, and a restarted job
"finishes" instantly against a stale checkpoint or an old success line.

Exits 0 when a condition is met, 2 on timeout, so a caller can distinguish
"finished" from "still running".

Wait on artifacts, never on a process name: a ``pgrep -f`` pattern matches the
waiting shell's own command line, so the wait never ends.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def artifact_ready(path: Path, newer_than: float | None = None) -> bool:
    """True once the path exists, is non-empty, and is newer than ``newer_than``.

    ``exists()`` follows symlinks, so a checkpoint link pointing at a file that has
    not been written yet is correctly reported as not ready. Size is checked because
    TAO creates some files before filling them, and a zero-byte checkpoint would
    otherwise read as success.
    """
    try:
        if path.is_dir():
            entries = list(path.iterdir())
            if not entries:
                return False
            if newer_than is None:
                return True
            return any(e.stat().st_mtime > newer_than for e in entries)
        if not (path.exists() and path.stat().st_size > 0):
            return False
        return newer_than is None or path.stat().st_mtime > newer_than
    except OSError:
        return False


def status_matches(path: Path, needle: str) -> str | None:
    """Return the matching status message, or None.

    status.json is JSON-lines and is appended to while the stage runs, so it is read
    whole each poll rather than tailed. A malformed trailing line is normal — the
    file may be mid-write — and is skipped rather than raising.
    """
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = str(record.get("message", ""))
        if needle.lower() in message.lower():
            return message.strip()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", action="append", default=[],
                        help="Path the stage writes ON SUCCESS -- not one it creates while "
                             "starting. Any one ending the wait is the point, so a directory "
                             "the stage opens up front (an output dir, a labels dir) ends it "
                             "immediately and the stage reads as finished when it has barely "
                             "begun. When outputs appear before completion, wait on "
                             "--status-json alone. Repeatable; any one ends "
                             "the wait.")
    parser.add_argument("--status-json", default=None,
                        help="TAO status.json to watch (JSON-lines).")
    parser.add_argument("--status-contains", default=None,
                        help="Substring of a status message that means the stage finished, "
                             "e.g. 'finished successfully'. Case-insensitive.")
    parser.add_argument("--timeout-sec", type=int, default=14400,
                        help="Give up after this long. Default 4h.")
    parser.add_argument("--poll-sec", type=int, default=20)
    parser.add_argument("--newer-than", default=None, metavar="PATH_OR_EPOCH",
                        help="Only accept an artifact modified after this: a unix timestamp, "
                             "or a file whose mtime is used. Take it immediately before "
                             "launching the stage. Without it a retry of a failed job is "
                             "satisfied by the artifact the previous attempt left behind.")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        artifacts = [Path(a).expanduser().resolve() for a in args.artifact]
        status = Path(args.status_json).expanduser().resolve() if args.status_json else None

        if not artifacts and not (status and args.status_contains):
            print("ERROR: give --artifact, or --status-json with --status-contains.",
                  file=sys.stderr)
            return 2
        if bool(status) != bool(args.status_contains):
            print("ERROR: --status-json and --status-contains go together.", file=sys.stderr)
            return 2

        newer_than: float | None = None
        if args.newer_than:
            try:
                newer_than = float(args.newer_than)
            except ValueError:
                marker = Path(args.newer_than).expanduser().resolve()
                if not marker.exists():
                    print(f"ERROR: --newer-than is neither a timestamp nor an existing path: "
                          f"{args.newer_than}", file=sys.stderr)
                    return 2
                newer_than = marker.stat().st_mtime

        deadline = time.monotonic() + args.timeout_sec
        while True:
            for path in artifacts:
                if artifact_ready(path, newer_than):
                    print(f"ready: {path}")
                    return 0
            if status is not None and (
                newer_than is None
                or (status.exists() and status.stat().st_mtime > newer_than)
            ):
                message = status_matches(status, args.status_contains)
                if message is not None:
                    print(f"ready: {status} -> {message}")
                    return 0
            if time.monotonic() >= deadline:
                print(f"TIMEOUT after {args.timeout_sec}s; nothing satisfied the wait.",
                      file=sys.stderr)
                for path in artifacts:
                    print(f"  missing: {path}", file=sys.stderr)
                if status is not None:
                    print(f"  no status line matching {args.status_contains!r} in {status}",
                          file=sys.stderr)
                return 2
            time.sleep(args.poll_sec)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
