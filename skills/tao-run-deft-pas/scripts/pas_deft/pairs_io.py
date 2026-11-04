# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared readers for PAS ``*_pairs.json`` files.

PAS datasets use both conventional JSON arrays and line-delimited arrays with
one object per line.  Keeping format detection in one reader lets dataset
inspection describe either layout without loading line-delimited exports into
memory.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any


def _is_record_line(line: str) -> bool:
    """Return whether ``line`` is one compact JSON object."""
    stripped = line.strip()
    return stripped.startswith("{") and stripped.rstrip(",").endswith("}")


def iter_json_records(path: str | os.PathLike[str]) -> Iterator[Any]:
    """Yield records from a JSON array, line-delimited array, or JSON object."""
    with open(path, encoding="utf-8") as handle:
        first = handle.readline()
        second = handle.readline()
    line_delimited = _is_record_line(first) or (
        first.strip() == "[" and _is_record_line(second)
    )

    if line_delimited:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                value = line.strip()
                if not value or value in {"[", "]"}:
                    continue
                if value.endswith(","):
                    value = value[:-1]
                if value:
                    yield json.loads(value)
        return

    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array or object in {path}")
    yield from rows
