#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify Docker access and that Docker Compose can start a disposable container."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default="busybox:1.36",
        help="Small image used for the Docker Compose smoke test.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    if not _run("docker daemon access", ["docker", "info"], args.timeout):
        return _blocked()

    if not _run("docker compose availability", ["docker", "compose", "version"], args.timeout):
        return _blocked()

    with tempfile.TemporaryDirectory(prefix="nva-compose-smoke-") as tmpdir:
        compose_file = Path(tmpdir) / "compose.yaml"
        compose_file.write_text(_compose_yaml(args.image), encoding="utf-8")
        command = [
            "docker",
            "compose",
            "-f",
            str(compose_file),
            "up",
            "--abort-on-container-exit",
            "--exit-code-from",
            "nva-compose-smoke",
        ]
        ok = _run("docker compose container startup", command, args.timeout)
        _cleanup_compose(compose_file)
        if not ok:
            return _blocked()

    print("Docker Compose access PASS: docker info, docker compose version, and disposable container startup succeeded.")
    return 0


def _compose_yaml(image: str) -> str:
    quoted_image = json.dumps(image)
    return (
        "services:\n"
        "  nva-compose-smoke:\n"
        f"    image: {quoted_image}\n"
        '    command: ["sh", "-c", "echo nva compose smoke ok"]\n'
    )


def _run(label: str, command: list[str], timeout: float) -> bool:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        print(f"Docker Compose access FAIL: {label} failed because `docker` was not found.", file=sys.stderr)
        return False
    except subprocess.TimeoutExpired as exc:
        print(f"Docker Compose access FAIL: {label} timed out after {timeout:g}s.", file=sys.stderr)
        _print_output(exc.stdout, exc.stderr)
        return False

    if completed.returncode != 0:
        print(f"Docker Compose access FAIL: {label} failed with exit code {completed.returncode}.", file=sys.stderr)
        _print_output(completed.stdout, completed.stderr)
        return False

    return True


def _print_output(stdout: str | bytes | None, stderr: str | bytes | None) -> None:
    for name, output in (("stdout", stdout), ("stderr", stderr)):
        if isinstance(output, bytes):
            text = output.decode("utf-8", errors="replace")
        else:
            text = output or ""
        text = text.strip()
        if text:
            print(f"{name}:\n{text[-2000:]}", file=sys.stderr)


def _cleanup_compose(compose_file: Path) -> None:
    try:
        subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "down", "--remove-orphans", "--volumes"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _blocked() -> int:
    print(
        "Stop and ask the user to change coding agent permissions for Docker daemon, "
        "Docker network, image pull, or filesystem access before customization. "
        "Do not use workarounds to bypass Docker access or permission issues.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
