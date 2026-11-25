#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report PAS control-Python readiness without changing the host."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys


MINIMUM_PYTHON = (3, 9)


@dataclass(frozen=True)
class PrerequisiteCheck:
    name: str
    status: str
    detail: str


def _run(command: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _version(executable: str) -> tuple[int, int, int] | None:
    completed = _run(
        [
            executable,
            "-c",
            "import sys; print('.'.join(str(v) for v in sys.version_info[:3]))",
        ]
    )
    if completed is None or completed.returncode != 0:
        return None
    try:
        parts = tuple(int(part) for part in completed.stdout.strip().split("."))
    except ValueError:
        return None
    return parts if len(parts) == 3 else None


def _probe(executable: str, arguments: list[str]) -> bool:
    completed = _run([executable, *arguments])
    return completed is not None and completed.returncode == 0


def _debian_family() -> bool:
    try:
        release = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        # platform.freedesktop_os_release was added after the minimum Python
        # supported by this control-plane check.
        try:
            lines = (
                pathlib.Path("/etc/os-release")
                .read_text(encoding="utf-8")
                .splitlines()
            )
        except OSError:
            return False
        release = {}
        for line in lines:
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            release[key] = value.strip().strip("\"'")
    identifiers = {
        release.get("ID", "").lower(),
        *release.get("ID_LIKE", "").lower().split(),
    }
    return bool(identifiers & {"debian", "ubuntu"})


def inspect(python_command: str) -> dict:
    executable = shutil.which(python_command)
    checks: list[PrerequisiteCheck] = []
    missing_packages: list[str] = []

    if executable is None:
        checks.append(
            PrerequisiteCheck(
                "python", "missing", f"{python_command!r} is not executable"
            )
        )
        version = None
        venv_ready = False
        pip_ready = False
        # None of the remaining probes can run. Ask for the complete known
        # control-plane dependency set so the next preflight does not reveal
        # one missing package at a time.
        missing_packages.extend(("python3", "python3-venv", "python3-pip"))
    else:
        version = _version(executable)
        version_ready = version is not None and version[:2] >= MINIMUM_PYTHON
        version_text = ".".join(map(str, version)) if version else "unknown"
        checks.append(
            PrerequisiteCheck(
                "python",
                "pass" if version_ready else "missing",
                f"version={version_text}; required>={MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]}",
            )
        )
        if not version_ready:
            missing_packages.append("python3")

        venv_ready = _probe(executable, ["-c", "import ensurepip, venv"])
        checks.append(
            PrerequisiteCheck(
                "venv_ensurepip",
                "pass" if venv_ready else "missing",
                "stdlib modules ensurepip and venv",
            )
        )
        if not venv_ready:
            missing_packages.append("python3-venv")

        pip_ready = _probe(executable, ["-m", "pip", "--version"])
        checks.append(
            PrerequisiteCheck(
                "pip",
                "pass" if pip_ready else "missing",
                "python3 -m pip --version",
            )
        )
        if not pip_ready:
            missing_packages.append("python3-pip")

    missing_packages = list(dict.fromkeys(missing_packages))
    ready = not missing_packages
    install_command = None
    if missing_packages and _debian_family():
        install_command = "sudo apt install -y " + " ".join(missing_packages)

    if ready:
        required_user_prompt = None
        summary = "ready"
    else:
        package_text = " ".join(missing_packages)
        remedy = install_command or f"install these host packages: {package_text}"
        required_user_prompt = (
            "PAS control-Python prerequisites are missing: "
            f"{package_text}. Run `{remedy}`, then ask me to rerun read-only preflight."
        )
        summary = f"blocked; missing={package_text}; remediation={remedy}"

    return {
        "schema_version": 1,
        "python_command": python_command,
        "python_executable": str(pathlib.Path(executable).resolve()) if executable else None,
        "checks": [asdict(check) for check in checks],
        "ready": ready,
        "missing_packages": missing_packages,
        "remediation_command": install_command,
        "approval_summary": summary,
        "next_action": "continue-read-only-preflight" if ready else "request-host-prerequisites",
        "required_user_prompt": required_user_prompt,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="python3", dest="python_command")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = inspect(args.python_command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
