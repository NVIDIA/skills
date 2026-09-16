#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The interface between preflight, the reference adapters, and the runner.

The persisted manifest is the only channel through which a reference learns
where an upstream skill lives. When MISSION_CONTROL_SHOWCASE_REQUIRE_PREFLIGHT
is set, a reference that cannot load a ready manifest blocks: it never falls
back to discovery and never substitutes another checkout found elsewhere.
"""

import json
import os
from pathlib import Path

MANIFEST_ENV = "MISSION_CONTROL_SHOWCASE_DEPS_MANIFEST"
REQUIRE_ENV = "MISSION_CONTROL_SHOWCASE_REQUIRE_PREFLIGHT"
HOME_ENV = "MISSION_CONTROL_SHOWCASE_HOME"
ROOT_ENV = "MISSION_CONTROL_SHOWCASE_UPSTREAM_ROOT"
MANIFEST_NAME = "showcase-deps.json"
MANIFEST_SCHEMA = "mission-control-showcase/deps-manifest@1"

EXIT_BLOCKED = 3


class Blocked(RuntimeError):
    """A dependency is missing, stale, unpinned, or incomplete."""


def showcase_home() -> Path:
    return Path(os.environ.get(HOME_ENV) or Path.home() / ".mission-control-showcase")


def state_dir() -> Path:
    return showcase_home() / "state"


def default_manifest_path() -> Path:
    explicit = os.environ.get(MANIFEST_ENV)
    return Path(explicit).expanduser() if explicit else state_dir() / MANIFEST_NAME


def preflight_required() -> bool:
    return os.environ.get(REQUIRE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


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
    """Assert a component is usable. Honors REQUIRE_ENV strictness.

    --allow-commit-drift is a diagnostic escape hatch at resolution time only.
    It marks the manifest `drift_allowed` and unpinned; runtime refuses such a
    manifest whenever REQUIRE_PREFLIGHT is set, so drift can never silently
    reach a real showcase run.
    """
    entry = component(manifest, name)
    if preflight_required() and manifest.get("drift_allowed"):
        raise Blocked(
            f"the dependency manifest was generated with --allow-commit-drift, "
            f"so its checkouts are not verified against the pins. That is a "
            f"diagnostic mode; {REQUIRE_ENV} is set, so it is refused. "
            f"Re-resolve without the flag:\n  {preflight_command()}"
        )
    if preflight_required() and not manifest.get("ready", False):
        raise Blocked(
            f"{REQUIRE_ENV} is set and the dependency manifest is not ready:\n  "
            + "\n  ".join(manifest.get("blockers") or ["(no detail recorded)"])
        )
    if entry.get("pinned") is False and preflight_required():
        raise Blocked(
            f"{name} resolves to an unpinned checkout ({entry.get('unpinned_reason')}). "
            f"{REQUIRE_ENV} is set, so an unpinned dependency is refused."
        )
    return entry
