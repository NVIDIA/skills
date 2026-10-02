#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect NVA generic defaults and optionally run a compatibility preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nva_generic_defaults import (
    check_generic_compatibility,
    format_compatibility,
    format_inspection,
    inspect_generic_defaults,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nva-root", required=True, type=Path)
    parser.add_argument(
        "--check-compatibility",
        action="store_true",
        help=(
            "Also verify the generic example files, default registry entries, "
            "service catalogs, and patch insertion points this skill needs."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=("appointment-making", "patient-intake", "custom"),
        help="Also check patch capabilities required by a selected scenario.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON."
    )
    args = parser.parse_args()

    try:
        if args.check_compatibility:
            report = check_generic_compatibility(args.nva_root, scenario=args.scenario)
            output = format_compatibility(report)
            exit_code = 2 if report.get("errors") else 0
        else:
            report = inspect_generic_defaults(args.nva_root)
            output = format_inspection(report)
            exit_code = 0
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
