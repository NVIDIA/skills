#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ONLY writer of TAO job-records (.tao/jobs/<id>.json).

Implements the atomic record-then-launch invariant of the SDK-free execution
architecture: ``open`` mints the job id, binds the resolved ``results_dir``,
and writes the initial PENDING record — and the id it prints is the ONLY
handle a platform skill may launch with. A submit that skipped the verify gate
has no id, so it cannot launch. ``mark`` appends status transitions
(append-only; terminal states are immutable). Every write is atomic
(temp + fsync + rename), guarded by a per-record flock (the in-turn agent and
a detached poller may both write), and passed through the redact_secrets
redactor so no credential material is ever persisted.

State root: $TAO_STATE_DIR if set, else ~/.tao — deliberately OUTSIDE any
synced results tree. Records conform to
skills/tao-artifacts/references/job_record.schema.json.

Subcommands:
  open   --platform P --image I --network-arch N --action A --storage-tier T
         (--results-dir DIR | --results-root ROOT) [--upload-exclude X ...]
         [--parent-job ID] [--retry-of ID]            -> prints the new id
  mark   ID --state S [--message M] [--source agent|poller|backend-hook]
         [--backend-ref REF] [--err-class ERR_INFRA|ERR_PROGRAM]
  show   ID                                           -> redacted record JSON
  list                                                -> id, platform, state, results_dir

"What is running" is NEVER answered from these records — poll the backend
(docker ps / kubectl get / squeue) via backend_ref. Records answer "what was
submitted, where do results live, and how did it end". No tao_sdk imports.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if __name__ == "__main__":
    sys.dont_write_bytecode = True  # no __pycache__ beside the sibling helper
import redact_secrets  # noqa: E402  (sibling helper; the redaction backstop)

STATES = ("PENDING", "RUNNING", "COMPLETE", "ERROR", "CANCELED", "UNKNOWN")
TERMINAL_STATES = ("COMPLETE", "ERROR", "CANCELED")
SOURCES = ("agent", "poller", "backend-hook")
# The platform field is an OPEN set — NOT a registry. Any installed platform
# skill (in-bank or external, e.g. kratos) records under its short name.
# EXAMPLE_PLATFORMS feeds help/error text ONLY — never matched against.
# Values are NORMALIZED into canonical kebab-case (trim, lowercase,
# spaces/underscores -> hyphens); only irreducible garbage is rejected —
# characters that cannot be mapped to a name without inventing one.
EXAMPLE_PLATFORMS = ("docker", "slurm", "kubernetes", "brev", "virtualenv")
_PLATFORM_CANONICAL = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")


def platform_slug(value: str) -> str:
    norm = re.sub(r"[\s_]+", "-", value.strip().lower())
    norm = re.sub(r"-{2,}", "-", norm).strip("-")
    if not _PLATFORM_CANONICAL.fullmatch(norm):
        raise argparse.ArgumentTypeError(
            f"cannot normalize platform {value!r} to a name "
            f"(canonical form {_PLATFORM_CANONICAL.pattern}; "
            f"e.g. {', '.join(EXAMPLE_PLATFORMS)}, kratos)"
        )
    return norm
STORAGE_TIERS = ("A", "B", "C")
ERR_CLASSES = ("ERR_INFRA", "ERR_PROGRAM")

# Mirrors job_record.schema.json's id pattern; also the path-traversal guard
# (no '/', no leading '.').
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

SCHEMA_VERSION = 1


def state_root() -> Path:
    raw = os.environ.get("TAO_STATE_DIR")
    if raw is not None:
        if not raw.strip():
            raise SystemExit("TAO_STATE_DIR is set but empty; unset it or give an absolute path")
        p = Path(raw)
        if not p.is_absolute():
            raise SystemExit(f"TAO_STATE_DIR must be an absolute path (got {raw!r})")
        return p
    return Path.home() / ".tao"


def jobs_dir() -> Path:
    d = state_root() / "jobs"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except NotADirectoryError:
        raise SystemExit(f"TAO_STATE_DIR is not a directory: {state_root()}")
    except PermissionError:
        raise SystemExit(f"cannot create state dir (permission denied): {d}")
    return d


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Path/URI-typed fields get URI-safe redaction (a legit path segment like
# /lustre/exp/token=v1/run must NOT be mangled); all other strings (esp. the
# free-text `message`) get whole-value field redaction.
_URI_FIELDS = frozenset({"results_dir", "image", "backend_ref"})


def _redact_record(obj, key=None):
    """Redact a job-record tree, field-typed so paths are never corrupted."""
    if isinstance(obj, str):
        return redact_secrets.redact_uri(obj) if key in _URI_FIELDS else redact_secrets.redact_field(obj)
    if isinstance(obj, list):
        return [_redact_record(v, key) for v in obj]
    if isinstance(obj, dict):
        return {k: _redact_record(v, k) for k, v in obj.items()}
    return obj


def _atomic_write(path: Path, record: dict) -> None:
    record = _redact_record(record)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(record, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        # An ENOSPC/fsync failure must leave the previous record intact without
        # accumulating misleading zero-byte job-record fragments.
        tmp.unlink(missing_ok=True)
    # fsync the directory so the rename itself survives a power loss
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _record_path(job_id: str) -> Path:
    if not ID_RE.match(job_id):
        raise SystemExit(f"invalid job id: {job_id!r}")
    return jobs_dir() / f"{job_id}.json"


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"no job-record at {path}")
    return json.loads(path.read_text())


class _Locked:
    """Per-record exclusive flock held across a read-modify-write."""

    def __init__(self, path: Path):
        self._lock_path = path.parent / f".{path.name}.lock"

    def __enter__(self):
        self._fh = open(self._lock_path, "w")
        fcntl.flock(self._fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self._fh, fcntl.LOCK_UN)
        self._fh.close()


def _sanitize_component(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    out = re.sub(r"^[^A-Za-z0-9]+", "", out).strip(".-")  # id must start alphanumeric
    return out or "job"


def cmd_open(args) -> int:
    if bool(args.results_dir) == bool(args.results_root):
        raise SystemExit("open: pass exactly one of --results-dir or --results-root")
    for flag, val in (("--image", args.image), ("--network-arch", args.network_arch), ("--action", args.action)):
        if not val or not val.strip():
            raise SystemExit(f"open: {flag} must be non-empty")

    job_id = f"{_sanitize_component(args.network_arch)}-{_sanitize_component(args.action)}-{uuid.uuid4().hex[:6]}"
    results_dir = args.results_dir or os.path.join(args.results_root, job_id)

    record = {
        "schema_version": SCHEMA_VERSION,
        "id": job_id,
        "platform": args.platform,
        "backend_ref": None,
        "image": args.image,
        "network_arch": args.network_arch,
        "action": args.action,
        "results_dir": results_dir,
        "storage_tier": args.storage_tier,
        "upload_excludes": args.upload_exclude or [],
        "submitted_at": _now(),
        "transitions": [
            {"ts": _now(), "state": "PENDING", "message": "record opened before launch", "source": "agent"}
        ],
        "terminal_state": None,
        "redacted": True,
    }
    if args.parent_job:
        record["parent_job"] = args.parent_job
    if args.retry_of:
        record["retry_of"] = args.retry_of

    path = _record_path(job_id)
    # Reserve the id atomically (O_EXCL) so two concurrent opens of a colliding
    # id can't both succeed — a losing racer gets EEXIST, not a silent clobber.
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise SystemExit(f"record already exists: {path}")
    os.close(fd)
    _atomic_write(path, record)  # temp+fsync+rename over the reserved file
    print(job_id)
    return 0


def cmd_mark(args) -> int:
    path = _record_path(args.id)
    with _Locked(path):
        record = _load(path)

        if record.get("terminal_state"):
            if args.state != record["terminal_state"]:
                raise SystemExit(
                    f"record {args.id} is terminal ({record['terminal_state']}); "
                    f"refusing transition to {args.state}"
                )
            # Same terminal state: not a state change, but allow post-hoc
            # ENRICHMENT — the agent classifies err_class AFTER a poller
            # has already marked ERROR, so this is the only way to attach it.
            changed = False
            if args.err_class and not record.get("err_class"):
                record["err_class"] = args.err_class
                changed = True
            if args.backend_ref and not record.get("backend_ref"):
                record["backend_ref"] = args.backend_ref
                changed = True
            if args.message:
                print(f"note: {args.id} already terminal; message not appended", file=sys.stderr)
            if changed:
                _atomic_write(path, record)
            return 0

        record["transitions"].append(
            {"ts": _now(), "state": args.state, "message": args.message or "", "source": args.source}
        )
        if args.backend_ref:
            record["backend_ref"] = args.backend_ref
        if args.err_class:
            record["err_class"] = args.err_class
        if args.state in TERMINAL_STATES:
            record["terminal_state"] = args.state
            record["terminal_write_by"] = args.source

        _atomic_write(path, record)
    return 0


def cmd_show(args) -> int:
    record = _load(_record_path(args.id))
    # records are redacted on write; re-redact on display as defense in depth
    print(json.dumps(_redact_record(record), indent=2))
    return 0


def cmd_list(_args) -> int:
    for path in sorted(jobs_dir().glob("*.json")):
        try:
            r = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"{path.stem}\t<corrupt>", file=sys.stderr)
            continue
        state = r["transitions"][-1]["state"] if r.get("transitions") else "?"
        print(f"{r.get('id', path.stem)}\t{r.get('platform', '?')}\t{state}\t{r.get('results_dir', '?')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_open = sub.add_parser("open", help="mint id + bind results_dir + write PENDING (record-then-launch)")
    p_open.add_argument(
        "--platform", required=True, type=platform_slug,
        help=f"platform slug (open set; e.g. {', '.join(EXAMPLE_PLATFORMS)}, or an "
             "external platform skill's short name)",
    )
    p_open.add_argument("--image", required=True)
    p_open.add_argument("--network-arch", required=True)
    p_open.add_argument("--action", required=True)
    p_open.add_argument("--storage-tier", required=True, choices=STORAGE_TIERS)
    p_open.add_argument("--results-dir", help="use this exact results_dir")
    p_open.add_argument("--results-root", help="derive results_dir=<root>/<job_id>")
    p_open.add_argument("--upload-exclude", action="append")
    p_open.add_argument("--parent-job")
    p_open.add_argument("--retry-of")
    p_open.set_defaults(fn=cmd_open)

    p_mark = sub.add_parser("mark", help="append a status transition (append-only; terminal is immutable)")
    p_mark.add_argument("id")
    p_mark.add_argument("--state", required=True, choices=STATES)
    p_mark.add_argument("--message", default="")
    p_mark.add_argument("--source", default="agent", choices=SOURCES)
    p_mark.add_argument("--backend-ref")
    p_mark.add_argument("--err-class", choices=ERR_CLASSES)
    p_mark.set_defaults(fn=cmd_mark)

    p_show = sub.add_parser("show", help="print the redacted record JSON")
    p_show.add_argument("id")
    p_show.set_defaults(fn=cmd_show)

    p_list = sub.add_parser("list", help="one line per record: id, platform, last state, results_dir")
    p_list.set_defaults(fn=cmd_list)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
