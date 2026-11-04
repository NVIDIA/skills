#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch a CLI-supplied command in the selected training environment.

Usage: launch_environment.py --plan PLAN [--print] -- COMMAND [ARG ...]
Arguments: PLAN selects training_root and explicit environment settings; COMMAND
comes from the invoking user, never from the plan or model-generated output.
Output: command output is inherited; --print emits the sanitized environment JSON.
Exit codes: child's exit status, 128 + signal for a terminated child, 2 for invalid
configuration. Termination signals are forwarded to the child process group.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from typing import Mapping

# Snapshot approved context at plan creation, before tmux/systemd can replace it.
INHERITED = {
    "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR",
    "DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "XDG_RUNTIME_DIR",
    "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME",
    "DBUS_SESSION_BUS_ADDRESS", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
    "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
    "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_PYTHON_PREFERENCE", "UV_PYTHON_DOWNLOADS",
    "PIP_CACHE_DIR", "CUDA_CACHE_PATH", "TORCH_HOME", "NUMBA_CACHE_DIR",
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
}
EXPLICIT_ONLY = {
    "LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "CUDA_HOME", "CUDA_PATH",
    "ISAACSIM_ASSET_REGION_PROFILE", "ISAACSIM_ASSET_ROOT", "PIP_CONFIG_FILE",
    "GIT_CONFIG_GLOBAL", "GIT_CONFIG_NOSYSTEM",
}
SYSTEM_PATH = ["/usr/local/bin", "/usr/bin", "/bin"]


def absolute_path(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value or not Path(value).is_absolute():
        raise ValueError(f"{name} must be a nonempty absolute path")
    return value


def normalize_environment(raw: dict | None, *, parent: Mapping[str, str] | None = None) -> dict:
    """Select uv and snapshot context; loader/import overrides require explicit config."""
    parent = os.environ if parent is None else parent
    raw = {} if raw is None else raw
    if not isinstance(raw, dict) or set(raw) - {"uv_executable", "variables", "tool_directories"}:
        raise ValueError("environment accepts only uv_executable, variables and tool_directories")
    uv = raw.get("uv_executable") or shutil.which("uv", path=parent.get("PATH", os.defpath))
    if uv is None:
        raise ValueError("uv is unavailable; install it and set environment.uv_executable")
    uv = absolute_path(uv, "environment.uv_executable")
    directories = raw.get("tool_directories", [])
    if not isinstance(directories, list):
        raise ValueError("environment.tool_directories must be a list")
    directories = [absolute_path(path, "tool directory") for path in directories]
    explicit = raw.get("variables", {})
    if not isinstance(explicit, dict):
        raise ValueError("environment.variables must be an object")
    unknown = set(explicit) - INHERITED - EXPLICIT_ONLY
    if unknown:
        raise ValueError(f"unsupported environment variables: {sorted(unknown)}")
    variables = {name: parent[name] for name in INHERITED if name in parent}
    for name, value in explicit.items():
        if not isinstance(value, str) or "\0" in value:
            raise ValueError(f"environment variable {name} must be a string without NUL")
        if name in {"LD_LIBRARY_PATH", "PYTHONPATH", "LD_PRELOAD"}:
            paths = value.replace(" ", ":").split(":") if name == "LD_PRELOAD" else value.split(":")
            for path in paths:
                absolute_path(path, name)
        variables[name] = value
    variables.setdefault("HOME", str(Path.home()))
    return {"uv_executable": uv, "tool_directories": directories, "variables": variables}


def build_environment(root: Path, configuration: dict, *, eula_accepted: bool) -> dict[str, str]:
    """Use a normalized snapshot without consulting the current shell again."""
    environment = dict(configuration["variables"])
    paths = [str(root / ".venv/bin"), str(Path(configuration["uv_executable"]).parent),
             *configuration["tool_directories"], *SYSTEM_PATH]
    environment["PATH"] = os.pathsep.join(dict.fromkeys(paths))
    environment.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", HEADLESS="1")
    if eula_accepted:
        environment["OMNI_KIT_ACCEPT_EULA"] = "Yes"
    return environment


def run_command(command: list[str], root: Path, environment: dict[str, str]) -> int:
    """Run explicit argv without a shell and keep cancellation semantics."""
    if not command or any(not isinstance(arg, str) or "\0" in arg for arg in command):
        raise ValueError("provide command arguments without NUL bytes")
    executable = command[0]
    if os.path.dirname(executable):
        executable = str((root / executable).absolute())
    else:
        executable = shutil.which(executable, path=environment["PATH"])
    if not executable or not os.path.isfile(executable) or not os.access(executable, os.X_OK):
        raise ValueError("command executable is unavailable in the selected environment")
    # Do not resolve a venv interpreter symlink: its invocation path selects sys.prefix.
    argv = [executable, *command[1:]]
    print(f"Launching {executable} from {root} with the selected environment", file=sys.stderr, flush=True)
    previous = {}
    pending = []
    child = None

    def forward(signum, _frame):
        if child is None:
            pending.append(signum)
        else:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, forward)
        child = subprocess.Popen(argv, cwd=root, env=environment, shell=False, start_new_session=True)
        for signum in pending:
            forward(signum, None)
        code = child.wait()
        return code if code >= 0 else 128 - code
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--print", dest="print_environment", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text())
        root = Path(absolute_path(plan["training_root"], "training_root"))
        config = normalize_environment(plan.get("environment"))
        from supervisor import require_eula_record
        require_eula_record()
        environment = build_environment(root, config, eula_accepted=True)
        if args.print_environment:
            print(json.dumps(environment, indent=2, sort_keys=True))
            return 0
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            raise ValueError("provide a command after --")
        if command[0] == "uv":
            command[0] = config["uv_executable"]
        return run_command(command, root, environment)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
