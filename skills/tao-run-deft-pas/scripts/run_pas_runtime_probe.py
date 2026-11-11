#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the approved PAS runtime probe in Docker with exact mounts and digest binding."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

from path_contract import safe_absolute_path


def _repository(image: str) -> str:
    value = image.split("@", 1)[0]
    last_slash = value.rfind("/")
    last_colon = value.rfind(":")
    return value[:last_colon] if last_colon > last_slash else value


def resolve_local_digest(image: str) -> tuple[str, str]:
    completed = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", image],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ValueError(f"approved image is not present on the Docker daemon: {detail}")
    try:
        values = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("Docker returned invalid RepoDigests JSON") from exc
    repository = _repository(image)
    matches = [
        value
        for value in values or []
        if isinstance(value, str) and value.startswith(repository + "@sha256:")
    ]
    if len(matches) != 1:
        raise ValueError(
            "approved image must resolve to exactly one matching immutable RepoDigest"
        )
    digest_ref = matches[0]
    digest = digest_ref.rsplit("@", 1)[1]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("Docker RepoDigest is not a sha256 digest")
    return digest_ref, digest


def _gpu_ids(raw: str, minimum: int) -> list[str]:
    ids = [item.strip() for item in raw.split(",") if item.strip()]
    if (
        len(ids) != minimum
        or len(set(ids)) != len(ids)
        or any(not item.isdigit() for item in ids)
    ):
        raise ValueError(
            "--gpu-ids must contain exactly --min-gpus distinct nonnegative IDs"
        )
    return ids


def build_command(args: argparse.Namespace, digest_ref: str, digest: str) -> list[str]:
    probe = pathlib.Path(__file__).resolve().with_name("check_pas_cuda_runtime.py")
    output = safe_absolute_path(args.output.expanduser().absolute(), "--output")
    if output.name in {"", ".", ".."}:
        raise ValueError("--output must name a JSON file")
    gpu_ids = _gpu_ids(args.gpu_ids, args.min_gpus)
    command = [
        "docker",
        "run",
        "--rm",
        "--runtime=nvidia",
        "--gpus",
        '"device=' + ",".join(gpu_ids) + '"',
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--env",
        "USER=tao",
        "--env",
        "LOGNAME=tao",
        "--volume",
        f"{probe}:/probe/check_pas_cuda_runtime.py:ro",
        "--volume",
        f"{output.parent}:/attestation:rw",
        "--workdir",
        "/tmp",  # nosec B108 - workdir inside the ephemeral probe container
        digest_ref,
        "python3",
        "/probe/check_pas_cuda_runtime.py",
        "--min-gpus",
        str(args.min_gpus),
        "--image-ref",
        args.image,
        "--image-digest",
        digest,
        "--output",
        f"/attestation/{output.name}",
    ]
    required_clis = ["clip"] if args.image_kind == "pyt" else ["embedding", "tmm"]
    for cli in required_clis:
        command.extend(["--require-cli", cli])
    if args.image_kind == "pyt" and args.finetuning_method == "lora":
        command.append("--require-clip-lora")
    return command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approved",
        action="store_true",
        help="Confirm the workflow launch review approved this container-starting probe.",
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-kind", required=True, choices=("pyt", "ds"))
    parser.add_argument("--finetuning-method", choices=("lora", "sft"), default="lora")
    parser.add_argument("--min-gpus", required=True, type=int)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if not args.approved:
            raise ValueError(
                "container runtime probes require the explicit --approved gate"
            )
        if args.min_gpus < 1:
            raise ValueError("--min-gpus must be at least 1")
        output = safe_absolute_path(args.output.expanduser().absolute(), "--output")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            if output.is_symlink() or not output.is_file():
                raise ValueError("existing runtime attestation output is unsafe")
            output.unlink()
        digest_ref, digest = resolve_local_digest(args.image)
        command = build_command(args, digest_ref, digest)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode
        if not output.is_file() or output.stat().st_size == 0:
            raise ValueError("runtime probe did not produce its attestation")
    except (OSError, ValueError) as exc:
        print(f"run_pas_runtime_probe: {exc}", file=sys.stderr)
        return 2
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
