#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a dotted string key from the installed skill bank's versions.yaml."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Any

import yaml


def _find_skill_bank(explicit: pathlib.Path | None) -> pathlib.Path:
    candidates: list[pathlib.Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    if os.environ.get("TAO_SKILL_BANK_PATH"):
        candidates.append(
            pathlib.Path(os.environ["TAO_SKILL_BANK_PATH"]).expanduser()
        )
    candidates.extend(pathlib.Path(__file__).resolve().parents)
    candidates.append(pathlib.Path.home() / "tao-skill-bank")

    seen: set[pathlib.Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "versions.yaml").is_file():
            return resolved
    raise FileNotFoundError(
        "versions.yaml not found; set TAO_SKILL_BANK_PATH or pass --skill-bank"
    )


def resolve(versions_path: pathlib.Path, dotted_key: str) -> str:
    with versions_path.open(encoding="utf-8") as handle:
        data: Any = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{versions_path} must contain a YAML object")

    cursor: Any = data
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise KeyError(f"key {dotted_key!r} not found in {versions_path}")
        cursor = cursor[part]
    if not isinstance(cursor, str) or not cursor.strip():
        raise ValueError(f"key {dotted_key!r} did not resolve to a non-empty string")
    return cursor.strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key", help="Dotted key, for example images.tao_toolkit.pyt")
    parser.add_argument("--skill-bank", type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        skill_bank = _find_skill_bank(args.skill_bank)
        print(resolve(skill_bank / "versions.yaml", args.key))
    except (FileNotFoundError, KeyError, ValueError, yaml.YAMLError) as exc:
        print(f"resolve_versions_key: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
