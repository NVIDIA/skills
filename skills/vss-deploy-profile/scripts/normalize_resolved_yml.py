#!/usr/bin/env -S uv run --quiet --script
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""Strip dangling optional depends_on entries from a resolved compose file.

Why this exists
---------------
`docker compose --env-file .env config > resolved.yml` filters out services
that don't match the active COMPOSE_PROFILES, but leaves depends_on: entries
pointing at those filtered-out services. Compose's schema validator rejects
any depends_on target that isn't a defined service in the file — even when
the entry is `required: false` — so `docker compose -f resolved.yml up -d`
aborts with:

    service "X" depends on undefined service "Y": invalid compose project

before any container starts. This script normalizes the generated artifact
by dropping only the dangling depends_on entries; required active deps
(kafka, redis, rtvi-vlm, sensor-ms, streamprocessing-ms, etc.) are preserved.

The script edits ONLY the generated resolved.yml — never the source compose
files. The dependencies are correctly marked optional in the source; profile
filtering is what creates the dangling references in the resolved artifact.

This MUST run after `docker compose ... config > resolved.yml` and before
`docker compose -f resolved.yml up -d`. The vss-deploy-profile skill (SKILL.md Step 3c)
calls this as part of every deploy.

Usage
-----
    uv run skills/vss-deploy-profile/scripts/normalize_resolved_yml.py [path/to/resolved.yml]
        # default path: ./resolved.yml in CWD
        # PEP 723 inline metadata declares pyyaml; uv pulls it into an
        # ephemeral env on demand, so no `pip install` on the host is needed.

Exit codes
----------
    0   normalized successfully (or already clean)
    1   file not found / parse error
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def normalize(path: Path) -> int:
    """Strip dangling depends_on entries in resolved compose at *path*.

    Returns the number of entries removed.
    """
    try:
        with path.open() as f:
            doc = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"ERROR: failed to parse {path}: {e}", file=sys.stderr)
        sys.exit(1)

    services = doc.get("services") or {}
    defined = set(services.keys())

    removed: list[tuple[str, str]] = []

    for name, svc in services.items():
        if not isinstance(svc, dict):
            continue
        deps = svc.get("depends_on")
        if not deps:
            continue

        if isinstance(deps, dict):
            kept = {k: v for k, v in deps.items() if k in defined}
            for k in deps:
                if k not in defined:
                    removed.append((name, k))
            if kept:
                svc["depends_on"] = kept
            else:
                svc.pop("depends_on", None)
        elif isinstance(deps, list):
            kept_list = [k for k in deps if k in defined]
            for k in deps:
                if k not in defined:
                    removed.append((name, k))
            if kept_list:
                svc["depends_on"] = kept_list
            else:
                svc.pop("depends_on", None)

    if removed:
        with path.open("w") as f:
            yaml.safe_dump(doc, f, sort_keys=False)
        print(f"Normalized {path}: dropped {len(removed)} dangling depends_on entries:")
        for svc_name, target in removed:
            print(f"  - {svc_name} -> {target}")
    else:
        print(f"{path} already clean (0 dangling depends_on entries)")

    return len(removed)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("resolved.yml")
    normalize(path)


if __name__ == "__main__":
    main()
