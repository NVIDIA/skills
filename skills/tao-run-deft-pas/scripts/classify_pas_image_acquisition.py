#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify whether the selected PAS platform must acquire runtime images."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from tao_slurm_ssh import ssh_command  # noqa: E402


PLATFORMS = ("docker", "slurm", "kubernetes", "brev", "virtualenv")


@dataclass(frozen=True)
class AvailabilityEvidence:
    resource: str
    status: str


def _present(command: list[str]) -> bool:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _evidence(resource: str, present: bool) -> AvailabilityEvidence:
    return AvailabilityEvidence(resource, "present" if present else "missing")


def _probe(
    command: list[str], *, missing_codes: tuple[int, ...] = (),
) -> subprocess.CompletedProcess[str]:
    """Do not convert transport/RBAC/tool errors into credential requirements."""
    try:
        result = subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"{command[0]} probe failed; check platform access") from exc
    if result.returncode not in (0, *missing_codes):
        raise ValueError(
            f"{command[0]} probe failed (exit {result.returncode}); "
            "check platform access before classifying resources or credentials"
        )
    return result


def classify(args: argparse.Namespace) -> dict:
    evidence: list[AvailabilityEvidence]
    authentication_evidence: list[AvailabilityEvidence] = []
    if args.platform == "virtualenv":
        if args.image or args.platform_image:
            raise ValueError("virtualenv does not consume container images")
        evidence = []
    elif args.platform == "docker":
        if not args.image:
            raise ValueError("docker requires at least one --image")
        evidence = [
            _evidence(image, _present(["docker", "image", "inspect", image]))
            for image in args.image
        ]
        authentication_evidence = [
            _evidence(
                f"registry-access/{item.resource}",
                _present(["docker", "manifest", "inspect", item.resource]),
            )
            for item in evidence
            if item.status != "present"
        ]
    elif args.platform == "brev":
        if not args.brev_instance or not args.image:
            raise ValueError("brev requires --brev-instance and at least one --image")
        evidence = []
        for image in args.image:
            remote_command = f"docker image inspect {shlex.quote(image)}"
            evidence.append(
                _evidence(
                    image,
                    _present(["brev", "exec", args.brev_instance, remote_command]),
                )
            )
        authentication_evidence = [
            _evidence(
                f"registry-access/{item.resource}",
                _present(
                    [
                        "brev",
                        "exec",
                        args.brev_instance,
                        f"docker manifest inspect {shlex.quote(item.resource)}",
                    ]
                ),
            )
            for item in evidence
            if item.status != "present"
        ]
    elif args.platform == "slurm":
        if not args.slurm_host or not args.platform_image:
            raise ValueError(
                "slurm requires --slurm-host and the planned .sqsh paths as "
                "--platform-image"
            )
        evidence = [
            _evidence(
                image,
                _probe(
                    ssh_command(args.slurm_host, f"test -r {shlex.quote(image)}"),
                    missing_codes=(1,),
                ).returncode == 0,
            )
            for image in args.platform_image
        ]
        if any(item.status != "present" for item in evidence):
            authentication_evidence = [
                _evidence(
                    "enroot-credentials",
                    _probe(
                        ssh_command(
                            args.slurm_host,
                            "test -s ~/.config/enroot/.credentials",
                        ),
                        missing_codes=(1,),
                    ).returncode == 0,
                )
            ]
    else:
        if not args.kubernetes_image_pull_secret:
            evidence = [AvailabilityEvidence("imagePullSecret", "missing")]
        else:
            command = ["kubectl"]
            if args.kubernetes_context:
                command.extend(["--context", args.kubernetes_context])
            command.extend(
                [
                    "--namespace",
                    args.kubernetes_namespace,
                    "get",
                    "secret",
                    args.kubernetes_image_pull_secret,
                    "--ignore-not-found",
                    "-o",
                    "jsonpath={.type}",
                ]
            )
            secret_type = _probe(command).stdout.strip()
            status = "missing"
            if secret_type:
                status = (
                    "present" if secret_type in {
                        "kubernetes.io/dockerconfigjson", "kubernetes.io/dockercfg",
                    } else "unsupported-type"
                )
            evidence = [
                AvailabilityEvidence(
                    f"imagePullSecret/{args.kubernetes_image_pull_secret}", status,
                )
            ]
        authentication_evidence = list(evidence)

    acquisition = (
        "required"
        if args.platform == "kubernetes"
        or any(item.status != "present" for item in evidence)
        else "not-required"
    )
    ngc_key_requirement = "not-required"
    if acquisition == "required" and (
        not authentication_evidence
        or any(item.status != "present" for item in authentication_evidence)
    ):
        ngc_key_requirement = "required"
    return {
        "schema_version": "1",
        "platform": args.platform,
        "image_acquisition": acquisition,
        "ngc_key_requirement": ngc_key_requirement,
        "evidence": [asdict(item) for item in evidence],
        "authentication_evidence": [
            asdict(item) for item in authentication_evidence
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=PLATFORMS)
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--platform-image", action="append", default=[])
    parser.add_argument("--brev-instance")
    parser.add_argument("--slurm-host")
    parser.add_argument("--kubernetes-image-pull-secret")
    parser.add_argument("--kubernetes-namespace", default="default")
    parser.add_argument("--kubernetes-context")
    parser.add_argument(
        "--format", choices=("json", "ngc-key-requirement"), default="json"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = classify(args)
    except ValueError as exc:
        print(f"classify_pas_image_acquisition: {exc}", file=sys.stderr)
        return 2
    if args.format == "ngc-key-requirement":
        print(report["ngc_key_requirement"])
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
