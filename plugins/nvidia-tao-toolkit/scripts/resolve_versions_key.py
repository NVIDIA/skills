#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a dotted key in versions.yaml to a string value.

Single-purpose companion to ``resolve_tao_image.py``: that tool resolves an
image via ``skills/<name>/references/skill_info.yaml``, this one resolves a
direct key path in ``versions.yaml``. Use it in shell scripts and skill
docs so the YAML schema is known in exactly one place.

Examples
--------
    resolve_versions_key.py images.tao_toolkit.pyt
    resolve_versions_key.py images.metropolis_sdg.paidf_anomalygen
    resolve_versions_key.py wheels.tao_sdk
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    # One line and an exit code of its own instead of a traceback. "python3 -I" and
    # "python3 -s" leave out the user site, where pip installs for a user who is not
    # root, so the line names that case before it names the install command.
    _no_user_site = (
        " It runs without the user site (python3 -I or -s, or PYTHONNOUSERSITE), which hides "
        "a PyYAML that pip installed for the user: start the script without that option "
        "first. If PyYAML is still missing, install it with:"
    )
    print(f"error: this Python ({sys.executable}) cannot import PyYAML, which this helper "
          "script needs." + (_no_user_site if sys.flags.no_user_site else " Install it with:")
          + " python3 -m pip install pyyaml", file=sys.stderr)
    raise SystemExit(4)

# The bank this script is part of, so resolution works from a git clone, a plugin
# install, or anywhere else the tree is unpacked. $TAO_SKILL_BANK_PATH overrides it
# for the case where the caller means a different bank than the one it lives in.
_BUNDLED_SKILL_BANK = Path(__file__).resolve().parents[1]
DEFAULT_SKILL_BANK = Path(
    os.environ.get("TAO_SKILL_BANK_PATH") or _BUNDLED_SKILL_BANK
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "key",
        help="Dotted key path, e.g. 'images.tao_toolkit.pyt'.",
    )
    parser.add_argument(
        "--skill-bank",
        type=Path,
        default=DEFAULT_SKILL_BANK,
        help="Path to the packaged TAO skill bank (defaults to $TAO_SKILL_BANK_PATH).",
    )
    return parser.parse_args()


def resolve(versions_path: Path, dotted_key: str) -> str:
    """Walk ``dotted_key`` through the YAML and return the leaf string."""
    with versions_path.open("r", encoding="utf-8") as handle:
        data: Any = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{versions_path} must contain a YAML object at the top level")

    cursor: Any = data
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise KeyError(f"key '{dotted_key}' not found in {versions_path}")
        cursor = cursor[part]

    if not isinstance(cursor, str) or not cursor.strip():
        raise ValueError(
            f"key '{dotted_key}' did not resolve to a non-empty string in {versions_path}"
        )
    return cursor.strip()


def main() -> int:
    """Print the resolved value or exit non-zero with a diagnostic."""
    args = parse_args()
    versions_path = args.skill_bank.expanduser() / "versions.yaml"
    if not versions_path.exists():
        print(f"versions.yaml not found at {versions_path}", file=sys.stderr)
        return 2
    try:
        print(resolve(versions_path, args.key))
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
