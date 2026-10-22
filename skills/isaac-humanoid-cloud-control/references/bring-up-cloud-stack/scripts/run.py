#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parent-owned adapter for the bring-up-cloud-stack stage.

This is not bring-up-cloud-stack. It resolves that upstream skill through the dependency
manifest, verifies it is ready, and invokes its 'up' entrypoint with the
arguments given. The upstream owns the behavior; this owns only the boundary.
"""

import subprocess  # nosec B404
import sys
from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parents[1]
SKILL_DIR = REFERENCE_DIR.parents[1]
sys.path.insert(0, str(SKILL_DIR / "shared"))

import dependency_manifest as dm  # noqa: E402

COMPONENT = "bring-up-cloud-stack"
ENTRYPOINT = "up"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        manifest = dm.load()
        dm.require(manifest, COMPONENT)
        target = dm.entrypoint(manifest, COMPONENT, ENTRYPOINT)
        dm.skill_md(manifest, COMPONENT)
        cwd = dm.skill_dir(manifest, COMPONENT)
    except dm.Blocked as exc:
        print(f"BLOCKED [{COMPONENT}]: {exc}", file=sys.stderr)
        return dm.EXIT_BLOCKED

    if target.suffix == ".py":
        command = [sys.executable, str(target), *argv]
    else:
        command = [str(target), *argv]

    # Relative resources inside the upstream skill must resolve from its own
    # directory, so the child runs there rather than in the caller's cwd.
    completed = subprocess.run(command, cwd=str(cwd))  # nosec B603
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
