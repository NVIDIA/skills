#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check task artifacts without executing agent-authored code.

Harbor mounts the reference checks in /tests/entry.json after the agent runs.
The checks and this grader stay outside both agent-visible skill arms. In
aces_plus_custom mode these scores supplement the standard evaluator metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

MAX_ARTIFACT_BYTES = 1_000_000


def contained(root: Path, relative: str) -> Path:
    """Resolve a verifier-owned path strictly below its allowed root.

    Args:
        root: Allowed directory for the resolved path.
        relative: Path below the allowed directory.
    """
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Artifact path must be relative without parent traversal")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact escapes its allowed root")
    return path


def read_object(path: Path) -> dict:
    """Read a bounded JSON object; linked outputs cannot stand in for artifacts.

    Args:
        path: Regular JSON artifact to read.
    """
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("Expected a regular JSON artifact smaller than 1 MB")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    return data


def value_at(data: object, keys: list[str]) -> object:
    """Look up an explicit sequence of object keys, including keys with dots.

    Args:
        data: Parsed artifact containing nested objects.
        keys: Object keys in lookup order.
    """
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            raise ValueError(f"Missing output field: {key}")
        data = data[key]
    return data


def check_artifact(check: dict, output_root: Path, input_root: Path) -> bool:
    """Evaluate one domain check against files or structured output values.

    Args:
        check: Verifier-owned operation and expected value.
        output_root: Directory containing agent-produced artifacts.
        input_root: Directory containing the reference corpus.
    """
    op = check["op"]
    if op == "symlink_to":
        # These links deliberately point from the generated staging tree into
        # the read-only input corpus. Do not read their target as agent code.
        relative = Path(check["artifact"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Staging link must be below the output directory")
        path = output_root / relative
        if not path.parent.resolve().is_relative_to(output_root.resolve()):
            raise ValueError("Staging link parent escapes the output directory")
        return path.is_symlink() and path.resolve(strict=True) == contained(input_root, check["expected"]).resolve()

    path = contained(output_root, check["artifact"])
    if op == "exists":
        return path.is_file() and not path.is_symlink() and path.stat().st_size > 0
    actual = value_at(read_object(path), check.get("path", []))
    expected = check["expected"]
    if op == "equals":
        return type(actual) is type(expected) and actual == expected
    if op == "close":
        if not isinstance(actual, int | float) or isinstance(actual, bool):
            return False
        try:
            return math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8)
        except OverflowError:
            return False
    if op in {"set_equals", "contains"}:
        if not isinstance(actual, list) or any(not isinstance(item, str) for item in actual):
            return False
        if op == "set_equals":
            return len(actual) == len(set(actual)) and set(actual) == set(expected)
        return set(expected).issubset(actual)
    raise ValueError(f"Unknown verifier operation: {op}")


def assess(entry: dict, output_root: Path, input_root: Path) -> dict:
    """Return conservative artifact scores, including input immutability checks.

    Args:
        entry: Verifier-owned domain checks and input hashes.
        output_root: Directory containing agent-produced artifacts.
        input_root: Directory containing the reference corpus.
    """
    checks = entry.get("domain_checks", [])
    details = {}
    for check in checks:
        try:
            passed = check_artifact(check, output_root, input_root)
            reason = "passed" if passed else "artifact did not satisfy the reference check"
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            passed, reason = False, str(exc)
        details[check["label"]] = {"passed": passed, "reason": reason}
    for relative, expected in entry.get("input_sha256", {}).items():
        try:
            path = contained(input_root, relative)
            passed = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
        except (OSError, ValueError, RuntimeError):
            passed = False
        details[f"unchanged:{relative}"] = {"passed": passed, "reason": "input checksum must remain unchanged"}
    fraction = sum(item["passed"] for item in details.values()) / len(details) if details else 0.0
    score = float(bool(checks) and fraction == 1.0)
    return {
        "overall": score,
        "custom_metrics": {"domain_correctness": score, "domain_check_fraction": fraction},
        "details": details,
    }


def main() -> None:
    """Read Harbor metadata and write the standard custom-grader reward files."""
    entry_path = Path(os.environ.get("HARBOR_ENTRY_JSON", "/tests/entry.json"))
    output_root = Path("/workspace/output")
    input_root = Path("/workspace/input")
    reward = assess(read_object(entry_path), output_root, input_root)
    reward_json = Path(os.environ.get("HARBOR_REWARD_JSON", "/logs/verifier/reward.json"))
    reward_txt = Path(os.environ.get("HARBOR_REWARD_TXT", "/logs/verifier/reward.txt"))
    reward_json.parent.mkdir(parents=True, exist_ok=True)
    reward_json.write_text(json.dumps(reward, indent=2) + "\n")
    reward_txt.parent.mkdir(parents=True, exist_ok=True)
    reward_txt.write_text(str(reward["overall"]) + "\n")


if __name__ == "__main__":
    main()
