#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export only the portable NV-MONAI-Autoresearch skill, excluding local data and archives."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_FILES = ("SKILL.md", "LICENSE", "skill_manifest.yaml")
PACKAGE_DIRECTORIES = (
    "assets",
    "references",
    "scripts",
    "validators",
    "fixtures",
    "evals",
    "tests",
)
EXCLUDED_NAMES = {"__pycache__", ".pytest_cache", ".DS_Store"}


def export(output_dir: Path) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("output directory already exists; choose a fresh path")
    destination = output_dir.resolve()
    if destination.name != "nv-monai-autoresearch":
        raise ValueError("installed skill directory must be named nv-monai-autoresearch")
    if destination == SKILL_ROOT or SKILL_ROOT in destination.parents:
        raise ValueError("export outside the source checkout")
    sources = [SKILL_ROOT / name for name in PACKAGE_FILES]
    for name in PACKAGE_DIRECTORIES:
        directory = SKILL_ROOT / name
        sources.extend(
            p
            for p in directory.rglob("*")
            if p.is_file() and not EXCLUDED_NAMES.intersection(p.parts)
        )
    for source in sources:
        if source.is_symlink() or SKILL_ROOT not in source.resolve().parents:
            raise ValueError(f"package source must be a regular in-tree file: {source.name}")
        if not source.is_file():
            raise ValueError(f"missing package source: {source.name}")
    destination.mkdir(parents=True)
    for source in sources:
        target = destination / source.relative_to(SKILL_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return {
        "skill": "nv-monai-autoresearch",
        "operation": "export",
        "destination": str(destination),
        "files": len(sources),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = export(args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
