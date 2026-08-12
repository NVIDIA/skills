#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))

from preflight_manifest import preflight_required, preflight_status_check


SKILL = "assemble-package-source"


def main() -> int:
    checks: list[dict[str, object]] = []
    if preflight_required():
        preflight_check = preflight_status_check(SKILL, "openusd_python")
        if not preflight_check["passed"]:
            payload = {
                "skill": SKILL,
                "passed": False,
                "checks": [preflight_check],
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 1
    try:
        from pxr import Sdf, Usd, UsdUtils  # noqa: F401
    except Exception as exc:
        checks.append(
            {
                "name": "openusd_python",
                "passed": False,
                "message": f"OpenUSD Python APIs are unavailable: {exc}",
            }
        )
    else:
        checks.append(
            {
                "name": "openusd_python",
                "passed": True,
                "message": "OpenUSD Python APIs are available",
            }
        )

    try:
        from PIL import Image, ImageStat  # noqa: F401
    except Exception as exc:
        checks.append(
            {
                "name": "pillow_available",
                "passed": False,
                "message": f"Pillow is unavailable: {exc}. Thumbnail pixel inspection cannot run, so scripts/run.py will fail assembly closed.",
            }
        )
    else:
        checks.append(
            {
                "name": "pillow_available",
                "passed": True,
                "message": "Pillow is available for thumbnail pixel inspection",
            }
        )

    payload = {
        "skill": "assemble-package-source",
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
