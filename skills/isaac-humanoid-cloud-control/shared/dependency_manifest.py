#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest contract between dependency resolution and reference adapters.

Adapters require pinned dependencies and never discover substitute checkouts.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

MANIFEST_ENV = "HUMANOID_CLOUD_CONTROL_DEPS_MANIFEST"
HOME_ENV = "HUMANOID_CLOUD_CONTROL_HOME"
ROOT_ENV = "HUMANOID_CLOUD_CONTROL_UPSTREAM_ROOT"
MANIFEST_NAME = "humanoid-deps.json"
MANIFEST_SCHEMA = "humanoid-cloud-control/deps-manifest@1"

EXIT_BLOCKED = 3


class Blocked(RuntimeError):
    """A dependency is missing, stale, unpinned, or incomplete."""


def humanoid_home() -> Path:
    return Path(os.environ.get(HOME_ENV) or Path.home() / ".humanoid-cloud-control")


def state_dir() -> Path:
    return humanoid_home() / "state"


def default_manifest_path() -> Path:
    explicit = os.environ.get(MANIFEST_ENV)
    return Path(explicit).expanduser() if explicit else state_dir() / MANIFEST_NAME


def preflight_command() -> str:
    here = Path(__file__).resolve().parents[1]
    return f"python3 {here}/scripts/doctor.py dependencies --prepare"


def load(path=None) -> dict:
    """Load the manifest, or raise Blocked with an actionable message."""
    target = Path(path).expanduser() if path else default_manifest_path()
    if not target.is_file():
        raise Blocked(
            f"no dependency manifest at {target}. Run:\n  {preflight_command()}"
        )
    try:
        manifest = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Blocked(f"dependency manifest at {target} is unreadable: {exc}") from exc
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise Blocked(
            f"dependency manifest at {target} has schema "
            f"{manifest.get('schema')!r}, expected {MANIFEST_SCHEMA!r}. "
            f"Regenerate it:\n  {preflight_command()}"
        )
    manifest["_path"] = str(target)
    return manifest


def component(manifest: dict, name: str) -> dict:
    """Return one skill entry, blocking when it is not usable."""
    entry = (manifest.get("skills") or {}).get(name)
    if entry is None:
        raise Blocked(
            f"{name} is absent from the dependency manifest "
            f"({manifest.get('_path')}). Regenerate it:\n  {preflight_command()}"
        )
    if entry.get("status") != "ready":
        detail = "; ".join(
            b for b in manifest.get("blockers", []) if b.startswith(f"{name}:")
        )
        raise Blocked(
            f"{name} is not ready (status {entry.get('status')!r})."
            + (f"\n  {detail}" if detail else "")
            + f"\nResolve it:\n  {preflight_command()}"
        )
    return entry


def entrypoint(manifest: dict, name: str, key: str) -> Path:
    """Resolve one upstream entrypoint, verifying it still exists."""
    entry = component(manifest, name)
    raw = (entry.get("entrypoints") or {}).get(key)
    if not raw:
        raise Blocked(
            f"{name} declares no {key!r} entrypoint in the manifest. "
            f"Regenerate it:\n  {preflight_command()}"
        )
    path = Path(raw)
    if not path.is_file():
        raise Blocked(
            f"{name}:{key} was resolved to {path}, which no longer exists. "
            f"The checkout moved or changed since preflight ran. Re-run:\n"
            f"  {preflight_command()}"
        )
    return path


def skill_md(manifest: dict, name: str) -> Path:
    """Path to the upstream's own runtime SKILL.md, the authority for its behavior."""
    entry = component(manifest, name)
    raw = entry.get("skill_md")
    if not raw or not Path(raw).is_file():
        raise Blocked(
            f"{name} has no readable runtime SKILL.md. Its own contract cannot be "
            f"read, so it must not be invoked. Re-run:\n  {preflight_command()}"
        )
    return Path(raw)


def skill_dir(manifest: dict, name: str) -> Path:
    return Path(component(manifest, name)["skill_dir"])


def require(manifest: dict, name: str) -> dict:
    """Require a ready, pinned manifest and recheck the current upstream files."""
    entry = component(manifest, name)
    if manifest.get("drift_allowed"):
        raise Blocked(
            f"the dependency manifest was generated with --allow-commit-drift, "
            f"so its checkouts are not verified against the pins. That is a "
            "diagnostic mode, which runtime invocation refuses. "
            f"Re-resolve without the flag:\n  {preflight_command()}"
        )
    if not manifest.get("ready", False):
        raise Blocked(
            "the dependency manifest is not ready:\n  "
            + "\n  ".join(manifest.get("blockers") or ["(no detail recorded)"])
        )
    if not entry.get("pinned"):
        raise Blocked(
            f"{name} resolves to an unpinned checkout ({entry.get('unpinned_reason')}). "
            "Runtime invocation refuses unpinned dependencies."
        )
    # A persisted manifest must not authorize code changed after resolution.
    try:
        lock = json.loads((Path(__file__).resolve().parents[1] /
                           "upstream-versions.lock.json").read_text())
        spec = lock["skills"][name]
        expected = lock["upstreams"][spec["upstream"]]["commit"]
        cwd = skill_dir(manifest, name)
        git = shutil.which("git")
        if not git:
            raise Blocked("git is required for dependency verification")
        head = subprocess.run(
            [git, "rev-parse", "HEAD"], cwd=cwd, check=True,
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        dirty = subprocess.run(
            [git, "status", "--porcelain", "--untracked-files=no"], cwd=cwd,
            check=True, capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        if head != expected or dirty:
            raise Blocked(f"{name}: checkout changed since resolution; re-run {preflight_command()}")
        for key, relative in spec["entrypoints"].items():
            if entrypoint(manifest, name, key).resolve() != (cwd / relative).resolve():
                raise Blocked(f"{name}: manifest entrypoint differs from the lock")
        for relative in spec.get("resources", []):
            if not (cwd / relative).exists():
                raise Blocked(f"{name}: missing resource {relative}")
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        raise Blocked(f"{name}: cannot verify current dependency: {exc}") from exc
    return entry
