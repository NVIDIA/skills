#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report the selected interpreter and visible tools without importing GPU packages.

Usage:
    python /path/to/cupotential-install/scripts/inspect_environment.py
Arguments:
    None. Run with the interpreter whose environment should be inspected.
Output:
    JSON on stdout with package discovery and NVIDIA tool results. GPU execution
    remains untested; absent packages or tools are recorded as observations.
Exit codes:
    0: inventory completed, including when packages or tools are absent.
    2: invalid command-line arguments (argparse).
"""

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import subprocess
import sys


def inspect_package(distribution, module):
    result = {"distribution": distribution, "module": module,
              "distribution_version": None, "module_origin": None,
              "import_attempted": False}
    try:
        result["distribution_version"] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        pass
    except (ValueError, OSError) as error:
        result["metadata_error"] = str(error)
    try:
        spec = importlib.util.find_spec(module)
        result["module_discoverable"] = spec is not None
        if spec is not None:
            result["module_origin"] = spec.origin
    except (ImportError, ValueError, OSError) as error:
        result["module_discoverable"] = None
        result["discovery_error"] = str(error)
    return result


def inspect_tool(arguments):
    executable = shutil.which(arguments[0])
    if executable is None:
        return {"status": "not_available", "executable": None}
    try:
        process = subprocess.run([executable, *arguments[1:]], capture_output=True,
                                 text=True, timeout=15, check=False, shell=False)
        return {"status": "reported" if process.returncode == 0 else "error",
                "executable": executable, "returncode": process.returncode,
                "stdout": process.stdout.strip(), "stderr": process.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "error", "executable": executable, "error": str(error)}


def inventory():
    return {
        "schema_version": 1,
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "packages": [inspect_package("nv-openmm-plugin", "nv_openmm_plugin"),
                     inspect_package("openmm", "openmm")],
        "tools": {
            "nvidia-smi": inspect_tool([
                "nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
            "nvcc": inspect_tool(["nvcc", "--version"]),
        },
        "cuda_execution": {"status": "not_tested",
                           "reason": "Inventory does not import the plugin or create a Context."},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(inventory(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
