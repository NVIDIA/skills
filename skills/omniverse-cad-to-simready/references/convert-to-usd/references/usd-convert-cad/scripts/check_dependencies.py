#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "shared"))

from preflight_manifest import preflight_required, preflight_status_check
from script_utils import check_result as _check, emit_json_report

from run import (
    CONVERTER_PYTHON_ENV,
    EXPECTED_WHEEL_VERSION,
    INSTALL_HINT,
    SKILL,
    UPSTREAM_REPO_URL,
    resolve_converter_python,
    wheel_version,
)


def _write_report(payload: dict[str, Any], report_path: Path | None) -> None:
    emit_json_report(payload, report_path)


def check_dependencies(
    converter_python: Path | None = None,
) -> dict[str, Any]:
    if preflight_required() and converter_python is None:
        preflight_check = preflight_status_check("usd-convert-cad", "usd_convert_cad")
        if not preflight_check["passed"]:
            return {
                "skill": SKILL,
                "passed": False,
                "checks": [preflight_check],
                "errors": [preflight_check["message"]],
                "install_hint": preflight_check["message"],
            }

    python = resolve_converter_python(converter_python)
    version, version_error = wheel_version(python)
    version_matches = version == EXPECTED_WHEEL_VERSION
    checks = [
        _check("orchestrator_python_available", True, f"Orchestrator Python: {sys.executable}"),
        _check(
            "usd_convert_cad_python_resolved",
            python.exists(),
            f"usd-convert-cad interpreter: {python}"
            + ("" if python.exists() else " (path does not exist)"),
        ),
        _check(
            "usd_convert_cad_wheel_importable",
            version is not None,
            f"usd-convert-cad wheel version: {version}"
            if version is not None
            else f"usd-convert-cad wheel is not importable with {python}: {version_error}",
        ),
        _check(
            "usd_convert_cad_wheel_version",
            version_matches,
            f"usd-convert-cad requires version {EXPECTED_WHEEL_VERSION}, found {version or 'not installed'}",
        ),
    ]
    errors = [check["message"] for check in checks if not check["passed"]]
    payload = {
        "skill": SKILL,
        "passed": not errors,
        "converter_python": str(python),
        "converter_python_env": CONVERTER_PYTHON_ENV,
        "upstream_repo": UPSTREAM_REPO_URL,
        "wheel_version": version,
        "checks": checks,
        "errors": errors,
    }
    if errors:
        payload["install_hint"] = INSTALL_HINT
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check usd-convert-cad wheel readiness.")
    parser.add_argument(
        "--usd-convert-cad-python",
        type=Path,
        help=(
            "Python interpreter that has the usd-convert-cad wheel installed. "
            f"Defaults to {CONVERTER_PYTHON_ENV}, the preflight manifest, then this interpreter."
        ),
    )
    parser.add_argument("--usd-convert-cad-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--report", type=Path, help="Write dependency check JSON to this path.")
    args = parser.parse_args(argv)

    payload = check_dependencies(args.usd_convert_cad_python)
    _write_report(payload, args.report)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
