#!/usr/bin/env python3
"""Emit a private run_manifest.json for a Proteina-Complexa run.

This is the R8/R9 replayable-artifact emitter shared by all `complexa-*` skills.
It captures:

  * Timestamp (UTC, ISO-8601)
  * The skill that produced the run, and the resolved `complexa …` command
  * The Hydra-resolved config (copied inline from <output_dir>/.hydra/config.yaml)
  * Git SHA of the repo (resolved by walking up from this file to pyproject.toml)
  * Per-checkpoint SHA-256 over the first 4096 bytes + size  (4K truncation is
    intentional — model files are multi-GB and full hashes are slow; the first
    4K still detects accidental re-downloads or path mix-ups)
  * Optional invocation metadata: cwd, user, and host (only when explicitly requested)
  * Pointers to result CSVs found under <output_dir> (capped at 50)

Stdlib-only on purpose: this often runs before any venv is active.

Usage:
    python3 write_manifest.py \\
        --output-dir /path/to/inference/<run_name> \\
        --command   "complexa design <config> ++run_name=foo …" \\
        --skill     complexa-design \\
        [--out ./run_manifest.json] [--include-system-metadata]

Arguments:
    --output-dir: Run output directory containing an optional Hydra config.
    --command: Complexa invocation to record; common credential values are redacted.
    --skill: Skill name that produced the run.
    --out: Destination manifest path.
    --include-system-metadata: Opt in to recording cwd, username, and hostname.

Output:
    A JSON manifest created with owner-only permissions (mode 0o600).

Exit codes:
    0 on success. argparse uses 2 for invalid arguments; unexpected errors use 1.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CKPT_KEYS = ("ckpt_path", "ckpt_name", "autoencoder_ckpt_path")
HASH_FIRST_N_BYTES = 4096
MAX_CSV_POINTERS = 50
GIT_TIMEOUT_SECONDS = 5
PRIVATE_FILE_MODE = 0o600
SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)(?P<prefix>(?:--)?[A-Za-z0-9_-]*(?:api[_-]?key|token|password|passwd|secret|credential)[A-Za-z0-9_-]*\s*(?:=|\s)\s*)(?P<value>[^\s]+)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True, help="Run output directory (contains .hydra/config.yaml)")
    p.add_argument("--command", required=True, help="The complexa invocation that produced the run")
    p.add_argument("--skill", required=True, help="Skill name (e.g. complexa-design)")
    p.add_argument("--out", default="./run_manifest.json", help="Manifest path (default: ./run_manifest.json)")
    p.add_argument(
        "--include-system-metadata",
        action="store_true",
        help="Record cwd, OS username, and hostname (omitted by default for privacy)",
    )
    return p.parse_args()


def redact_sensitive_values(text: str | None) -> str | None:
    """Redact common credential assignments in command or configuration text."""
    if text is None:
        return None
    return SENSITIVE_ASSIGNMENT_RE.sub(r"\g<prefix><redacted>", text)


def find_repo_root(start: Path) -> Path | None:
    """Walk up from `start` looking for a pyproject.toml. Returns None if not found."""
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def git_sha(repo_root: Path | None) -> str:
    """Best-effort `git rev-parse HEAD`. Returns 'unknown' on any failure."""
    if repo_root is None:
        return "unknown"
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return "unknown"


def read_hydra_config(output_dir: Path) -> tuple[str | None, str | None]:
    """Return (config_text, config_path) or (None, None) if .hydra/config.yaml is absent."""
    candidate = output_dir / ".hydra" / "config.yaml"
    if not candidate.is_file():
        return None, None
    try:
        return candidate.read_text(encoding="utf-8", errors="replace"), ".hydra/config.yaml"
    except OSError:
        return None, ".hydra/config.yaml"


def scan_ckpt_keys(config_text: str) -> dict[str, str]:
    """Pull scalar values for any of CKPT_KEYS out of a Hydra YAML dump.

    Intentionally a tiny line-based parser — stdlib has no YAML module and we
    only need three well-known keys at any depth. Treats the first scalar after
    `<key>:` as the value. Hydra resolvers like ``${oc.env:CKPT_PATH}`` are
    captured verbatim; the caller can decide whether to hash them.
    """
    found: dict[str, str] = {}
    for raw in config_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for key in CKPT_KEYS:
            prefix = f"{key}:"
            if line.startswith(prefix):
                val = line[len(prefix):].strip().strip("'").strip('"')
                # Skip block-scalar/list markers
                if val and not val.startswith(("|", ">", "-", "[", "{")):
                    found.setdefault(key, val)
                break
    return found


def hash_first_4k(path: Path) -> dict[str, Any]:
    """Return {path, sha256_first_4k, size_bytes, exists} for `path`."""
    info: dict[str, Any] = {"path": str(path), "exists": False, "sha256_first_4k": None, "size_bytes": None}
    try:
        if not path.is_file():
            return info
        info["exists"] = True
        info["size_bytes"] = path.stat().st_size
        with path.open("rb") as fh:
            info["sha256_first_4k"] = hashlib.sha256(fh.read(HASH_FIRST_N_BYTES)).hexdigest()
    except (OSError, ValueError):
        pass
    return info


def resolve_ckpts(ckpt_fields: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Build manifest.checkpoints from the scanned ckpt fields.

    Joins `ckpt_path` + `ckpt_name` if both are scalars (no ``${…}`` interpolation
    left). Always also records `autoencoder_ckpt_path` if present.
    """
    out: dict[str, dict[str, Any]] = {}
    ckpt_path = ckpt_fields.get("ckpt_path", "")
    ckpt_name = ckpt_fields.get("ckpt_name", "")
    ae_path = ckpt_fields.get("autoencoder_ckpt_path", "")

    if ckpt_path and ckpt_name and "${" not in ckpt_path and "${" not in ckpt_name:
        out["ckpt"] = hash_first_4k(Path(ckpt_path) / ckpt_name)
    elif ckpt_path or ckpt_name:
        out["ckpt"] = {"path": f"{ckpt_path}/{ckpt_name}".strip("/"), "exists": False,
                       "sha256_first_4k": None, "size_bytes": None,
                       "note": "unresolved Hydra interpolation; cannot hash"}

    if ae_path:
        if "${" not in ae_path:
            out["autoencoder_ckpt"] = hash_first_4k(Path(ae_path))
        else:
            out["autoencoder_ckpt"] = {"path": ae_path, "exists": False,
                                       "sha256_first_4k": None, "size_bytes": None,
                                       "note": "unresolved Hydra interpolation; cannot hash"}
    return out


def collect_csv_pointers(output_dir: Path, cap: int = MAX_CSV_POINTERS) -> list[str]:
    """Walk output_dir for *.csv files, cap at `cap`, return relative paths."""
    if not output_dir.is_dir():
        return []
    found: list[str] = []
    for p in sorted(output_dir.rglob("*.csv")):
        try:
            found.append(str(p.relative_to(output_dir)))
        except ValueError:
            found.append(str(p))
        if len(found) >= cap:
            break
    return found


def write_private_text(path: Path, text: str) -> None:
    """Write text with owner-only permissions, including when replacing a file."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(path, flags, PRIVATE_FILE_MODE)
    try:
        os.fchmod(fd, PRIVATE_FILE_MODE)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(text)
    finally:
        if fd >= 0:
            os.close(fd)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    repo_root = find_repo_root(Path(__file__).resolve())

    config_text, config_path = read_hydra_config(output_dir)
    ckpt_fields = scan_ckpt_keys(config_text) if config_text else {}

    manifest: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "skill": args.skill,
        "command": redact_sensitive_values(args.command),
        "output_dir": str(Path(args.output_dir)),
        "git_sha": git_sha(repo_root),
        "config_path": config_path,
        "config": redact_sensitive_values(config_text),
        "checkpoints": resolve_ckpts(ckpt_fields),
        "pointers": {"csv_files": collect_csv_pointers(output_dir)},
        "notes": {
            "checkpoint_hash": f"sha256 over first {HASH_FIRST_N_BYTES} bytes (truncated for speed)",
            "csv_pointers_cap": MAX_CSV_POINTERS,
        },
    }
    if args.include_system_metadata:
        manifest["invocation"] = {
            "cwd": os.getcwd(),
            "user": getpass.getuser(),
            "host": socket.gethostname(),
        }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_private_text(out_path, json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    print(f"Wrote manifest: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
