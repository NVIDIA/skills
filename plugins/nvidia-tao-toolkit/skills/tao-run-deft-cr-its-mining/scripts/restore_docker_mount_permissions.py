#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restore host write access to a Docker-written result directory."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def absolute_path(path: str | Path) -> Path:
    """Expand a user path to an absolute path without resolving symlinks."""
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def write_failures(root: Path) -> list[str]:
    """Return paths under root that the current host user cannot write."""
    failures: list[str] = []
    if not root.exists():
        return [f"missing path: {root}"]
    if not root.is_dir():
        return [f"not a directory: {root}"]

    for current_root, dirs, files in os.walk(root):
        current = Path(current_root)
        probe = current / f".deft_write_probe_{os.getpid()}"
        try:
            probe.write_text("ok\n", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            failures.append(f"directory not writable: {current} ({exc})")
            dirs[:] = []
            continue

        for filename in files:
            path = current / filename
            if path.is_symlink():
                continue
            if not os.access(path, os.W_OK):
                failures.append(f"file not writable: {path}")

    return failures


def restore_with_docker(path: Path, docker_image: str, uid: int, gid: int) -> None:
    """Run a short root container that chowns/chmods the mounted result path."""
    if shutil.which("docker") is None:
        raise RuntimeError("docker executable was not found; cannot restore Docker mount permissions")
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{path}:/target",
        docker_image,
        "bash",
        "-lc",
        f"chown -R {uid}:{gid} /target && chmod -R u+rwX /target",
    ]
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, type=Path, help="Host result directory to check/fix")
    parser.add_argument(
        "--docker-image",
        default="",
        help="Container image to use for chown cleanup when the path is not host-writable",
    )
    parser.add_argument("--uid", type=int, default=os.getuid(), help="Host UID that should own outputs")
    parser.add_argument("--gid", type=int, default=os.getgid(), help="Host GID that should own outputs")
    return parser.parse_args()


def main() -> int:
    """Ensure a Docker-written result directory is writable by the host user."""
    args = parse_args()
    path = absolute_path(args.path)
    failures = write_failures(path)
    if not failures:
        print(f"result directory is writable: {path}")
        return 0

    if not args.docker_image:
        print(
            "ERROR: result directory is not writable by the host user. "
            "Provide --docker-image so this script can run a Docker chown cleanup.",
            file=sys.stderr,
        )
        for failure in failures[:10]:
            print(f"  {failure}", file=sys.stderr)
        if len(failures) > 10:
            print(f"  ... {len(failures) - 10} more", file=sys.stderr)
        return 2

    print(f"restoring host write access with Docker: {path}")
    restore_with_docker(path, args.docker_image, args.uid, args.gid)
    failures = write_failures(path)
    if failures:
        print(f"ERROR: path is still not writable after Docker cleanup: {path}", file=sys.stderr)
        for failure in failures[:10]:
            print(f"  {failure}", file=sys.stderr)
        if len(failures) > 10:
            print(f"  ... {len(failures) - 10} more", file=sys.stderr)
        return 2
    print(f"result directory is writable: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
