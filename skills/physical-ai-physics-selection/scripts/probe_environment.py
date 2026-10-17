#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from neelkm/skills (c820715) for ovnewton.
"""Read environment metadata without importing native simulation libraries.

Run this with the application's interpreter. Results are an inventory, not a
runtime-readiness verdict. In particular, this script never initializes CUDA.
"""

import argparse
import json
import platform
import sys
from importlib import metadata

PACKAGES = (
    "ovnewton",
    "newton",
    "ovstage",
    "warp-lang",
    "ovrtx",
    "newton-usd-schemas",
    "physx-usd-schemas",
    "numpy",
    "scipy",
)


def collect(version_reader=metadata.version, requirements_reader=metadata.requires):
    """Keep readers injectable so missing-package behavior is testable."""
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version_reader(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    requirements = []
    if packages["ovnewton"] is not None:
        requirements = requirements_reader("ovnewton") or []
    return {
        "schema_version": 1,
        "status": "metadata_only",
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "host": {"os": platform.system(), "architecture": platform.machine()},
        "packages": packages,
        "ovnewton_declared_requirements": requirements,
        "runtime_tested": False,
        "gpu_tested": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        result = collect()
    except Exception as exc:
        print(json.dumps({"status": "probe_failed", "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
