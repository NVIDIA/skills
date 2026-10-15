#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate an ethos.md file against schema version 1.

Checks front matter, schema version, required fields, and the fourteen required
``##`` sections, and rejects duplicate sections. Extra sections and extra
front-matter keys are allowed. Standard library only.

Usage: python validate_ethos.py path/to/ethos.md
Exit status 0 means valid; warnings print to stdout. Errors print to stderr.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = 1
REQUIRED_SECTIONS = (
    "Role",
    "Purpose & Outcomes",
    "Scope",
    "Tools",
    "Harness",
    "Behavior",
    "Principles",
    "Success Criteria",
    "Trade-offs",
    "Constraints",
    "Evaluation Setup",
    "Metric Semantics",
    "Vision",
    "Open Questions",
)

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SECTION = re.compile(r"^## +(.+?)\s*$", re.MULTILINE)
_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)$")


class EthosError(ValueError):
    pass


def _front_matter(block: str) -> dict[str, str]:
    """Read top-level ``key: value`` pairs; nested or comment lines are skipped."""
    front: dict[str, str] = {}
    for line in block.splitlines():
        match = _KEY.match(line)
        if match:
            front[match.group(1)] = match.group(2).strip().strip("'\"")
    return front


def _timestamp(front: dict[str, str], key: str, *, required: bool) -> None:
    value = front.get(key, "")
    if not value:
        if required:
            raise EthosError(f"front matter field {key!r} is required")
        return
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EthosError(f"front matter field {key!r} must be an ISO 8601 datetime") from exc


def validate(text: str) -> list[str]:
    """Return warnings; raise EthosError when the file is invalid."""
    match = _FRONT_MATTER.match(text)
    if match is None:
        raise EthosError("missing YAML front matter")
    front = _front_matter(match.group(1))

    warnings: list[str] = []
    raw_version = front.get("schema_version", "")
    if not raw_version:
        warnings.append(f"front matter has no 'schema_version'; add 'schema_version: {SCHEMA_VERSION}'")
    elif raw_version != str(SCHEMA_VERSION):
        raise EthosError(f"unsupported schema_version {raw_version!r}; expected {SCHEMA_VERSION}")

    for key in ("name", "author"):
        if not front.get(key):
            raise EthosError(f"front matter field {key!r} is required")
    _timestamp(front, "created_timestamp", required=True)
    _timestamp(front, "updated_timestamp", required=False)
    if "owner" in front and not front["owner"]:
        raise EthosError("front matter field 'owner' must be non-empty when present")

    seen: set[str] = set()
    for heading in _SECTION.findall(text[match.end() :]):
        if heading in seen:
            raise EthosError(f"duplicate section: ## {heading}")
        seen.add(heading)
    for heading in REQUIRED_SECTIONS:
        if heading not in seen:
            raise EthosError(f"missing section: ## {heading}")
    return warnings


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-2], file=sys.stderr)
        return 2
    path = Path(argv[1])
    try:
        warnings = validate(path.read_text(encoding="utf-8"))
    except (OSError, EthosError) as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1
    for warning in warnings:
        print(f"warning: {warning}")
    print(f"valid: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
