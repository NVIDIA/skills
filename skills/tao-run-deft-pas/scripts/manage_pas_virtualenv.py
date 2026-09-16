#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan, install, or verify one hash-locked PAS execution virtualenv."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

from virtualenv_runtime import lock_status, validate_tao_virtualenv


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="verb", required=True)
    for verb in ("plan", "install", "verify"):
        command = subparsers.add_parser(verb)
        command.add_argument("--profile", required=True, choices=("pyt", "ds"))
        command.add_argument("--virtualenv", required=True, type=pathlib.Path)
        if verb == "install":
            command.add_argument("--approve-install", action="store_true")
        if verb == "verify":
            command.add_argument("--min-gpus", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        status = lock_status(args.profile)
        if args.verb == "plan":
            status["virtualenv"] = str(args.virtualenv.expanduser().resolve())
            print(json.dumps(status, indent=2, sort_keys=True))
            return 0
        if args.verb == "verify":
            resolved = validate_tao_virtualenv(
                args.virtualenv,
                profile=args.profile,
                probe_imports=True,
                minimum_gpus=args.min_gpus,
            )
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "profile": args.profile,
                        "virtualenv": str(resolved),
                    }
                )
            )
            return 0
        if not args.approve_install:
            raise ValueError("install requires explicit --approve-install after the launch review")
        if not status["ready_to_install"]:
            raise ValueError(str(status["blocker"]))
        target = args.virtualenv.expanduser().resolve()
        if target.exists():
            raise ValueError(f"refusing to overwrite existing virtualenv: {target}")
        python312 = shutil.which("python3.12")
        virtualenv = shutil.which("virtualenv")
        if python312 is None or virtualenv is None:
            raise ValueError("installation requires python3.12 and virtualenv on PATH")
        lock = pathlib.Path(status["lock_file"])
        target.parent.mkdir(parents=True, exist_ok=True)
        # Reserve the final path atomically. Virtualenv console-script shebangs
        # embed this path, so constructing elsewhere and renaming would create
        # a subtly broken, non-relocatable environment.
        target.mkdir()
        installed = False
        try:
            subprocess.run([virtualenv, "--python", python312, str(target)], check=True)
            subprocess.run(
                [
                    str(target / "bin" / "python"),
                    "-m",
                    "pip",
                    "install",
                    "--require-hashes",
                    "--only-binary=:all:",
                    "-r",
                    str(lock),
                ],
                check=True,
            )
            validate_tao_virtualenv(
                target, profile=args.profile, probe_imports=True
            )
            installed = True
        finally:
            if not installed and target.exists():
                shutil.rmtree(target)
        resolved = target
        print(
            json.dumps(
                {
                    "status": "INSTALLED",
                    "profile": args.profile,
                    "virtualenv": str(resolved),
                }
            )
        )
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"manage_pas_virtualenv: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
