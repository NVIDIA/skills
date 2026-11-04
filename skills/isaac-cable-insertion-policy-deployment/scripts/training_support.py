#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use the companion training skill's environment and provenance checks.

Usage: training_support.py TRAINING_ROOT
Arguments: TRAINING_ROOT is the selected isaaclab_training package.
Output: JSON with verified installation, dependency and export-provider details.
Exit codes: 0 when checks pass; 1 for missing inputs or failed runtime checks.
Run with the selected package's .venv/bin/python and sanitized environment.
"""
import importlib.util
from pathlib import Path
import sys

TRAINING_SCRIPTS = Path(__file__).resolve().parents[2] / "isaac-cable-insertion-policy-training/scripts"


def load_training_module(name):
    """Load a helper from the companion training skill or fail if it is absent."""
    path = TRAINING_SCRIPTS / f"{name}.py"
    if not path.is_file():
        raise RuntimeError("Install the companion isaac-cable-insertion-policy-training skill alongside deployment")
    spec = importlib.util.spec_from_file_location(f"cable_training_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch_environment = load_training_module("launch_environment")


def check_export_environment(root):
    import importlib.metadata
    checks = load_training_module("runtime_checks")
    result = {"installation": checks.installation(root), "dependencies": checks.dependencies(root)}
    version = importlib.metadata.version("leapp")
    if version != checks.SUPPORTED_LEAPP_VERSION:
        raise RuntimeError(f"Checkpoint exporter requires locked LEAPP {checks.SUPPORTED_LEAPP_VERSION}; "
                           f"found {version}")
    import leapp
    import onnxruntime
    result.update(leapp_version=version, leapp_source=str(Path(leapp.__file__).resolve()),
                  onnxruntime_version=onnxruntime.__version__,
                  onnxruntime_providers=onnxruntime.get_available_providers())
    return result


if __name__ == "__main__":
    import json
    print(json.dumps(check_export_environment(Path(sys.argv[1])), indent=2))
