#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Audit and prepare a provider-neutral Kubernetes GPU compute target."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


class SetupError(RuntimeError):
    """A safe-to-display remote setup error."""


@dataclass
class Runner:
    kubectl_command: str = "kubectl"
    context: str | None = None
    ssh_target: str | None = None
    ssh_options: tuple[str, ...] = ()

    def command(self, args: list[str]) -> list[str]:
        command = shlex.split(self.kubectl_command)
        if not command:
            raise SetupError("--kubectl-command cannot be empty")
        if self.context:
            command.extend(["--context", self.context])
        command.extend(args)
        if not self.ssh_target:
            return command
        return ["ssh", *self.ssh_options, self.ssh_target, shlex.join(command)]

    def run(
        self,
        args: list[str],
        *,
        input_text: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                self.command(args),
                input=input_text,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SetupError(f"Could not execute Kubernetes command: {exc}") from exc
        if check and completed.returncode != 0:
            message = (
                completed.stderr.strip() or completed.stdout.strip() or "command failed"
            )
            raise SetupError(message[:2000])
        return completed

    def json(self, args: list[str]) -> dict[str, Any]:
        completed = self.run(args)
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise SetupError("kubectl returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise SetupError("kubectl returned a non-object JSON value")
        return value


def resource_exists(
    runner: Runner, kind: str, name: str, namespace: str | None = None
) -> bool:
    args = ["get", kind, name]
    if namespace:
        args.extend(["-n", namespace])
    args.extend(["-o", "json"])
    return runner.run(args, check=False).returncode == 0


def summarize_nodes(payload: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") or {}
        status = item.get("status") or {}
        allocatable = status.get("allocatable") or {}
        conditions = status.get("conditions") or []
        ready = any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in conditions
            if isinstance(condition, dict)
        )
        raw_gpu = allocatable.get("nvidia.com/gpu", "0")
        try:
            gpu_count = int(raw_gpu)
        except (TypeError, ValueError):
            gpu_count = 0
        nodes.append(
            {
                "name": str(metadata.get("name", "unknown")),
                "ready": ready,
                "nvidia_gpus": gpu_count,
            }
        )
    return {
        "nodes": nodes,
        "ready_nodes": sum(1 for node in nodes if node["ready"]),
        "ready_gpu_nodes": sum(
            1 for node in nodes if node["ready"] and node["nvidia_gpus"]
        ),
        "ready_gpus": sum(node["nvidia_gpus"] for node in nodes if node["ready"]),
    }


def evaluate_readiness(
    facts: dict[str, Any], service_mode: str
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    required_gpus = 4 if service_mode == "internal" else 1
    if not facts.get("cluster_reachable"):
        reasons.append("Kubernetes API is not reachable")
    if facts.get("ready_nodes", 0) < 1:
        reasons.append("no Ready Kubernetes node was found")
    if facts.get("ready_gpus", 0) < required_gpus:
        reasons.append(
            f"{service_mode} mode needs at least {required_gpus} allocatable NVIDIA GPU(s); "
            f"found {facts.get('ready_gpus', 0)}"
        )
    if not facts.get("can_create_pods"):
        reasons.append(
            "current Kubernetes identity cannot create pods in the workflow namespace"
        )
    if not facts.get("namespace_exists"):
        reasons.append("workflow namespace is missing")
    if not facts.get("registry_secret_exists"):
        reasons.append("NGC image-pull secret is missing")
    if service_mode == "internal" and not facts.get("model_cache_pvc_exists"):
        reasons.append("internal mode requires the model-cache PVC")
    return not reasons, reasons


def audit(runner: Runner, namespace: str, service_mode: str) -> dict[str, Any]:
    version = runner.json(["version", "-o", "json"])
    node_summary = summarize_nodes(runner.json(["get", "nodes", "-o", "json"]))
    can_create = (
        runner.run(["auth", "can-i", "create", "pods", "-n", namespace], check=False)
        .stdout.strip()
        .lower()
        == "yes"
    )
    storage = runner.run(["get", "storageclass", "-o", "json"], check=False)
    storage_classes: list[str] = []
    if storage.returncode == 0:
        try:
            storage_payload = json.loads(storage.stdout)
            storage_classes = [
                str(item.get("metadata", {}).get("name"))
                for item in storage_payload.get("items", [])
                if isinstance(item, dict) and item.get("metadata", {}).get("name")
            ]
        except json.JSONDecodeError:
            pass
    facts: dict[str, Any] = {
        "cluster_reachable": True,
        "server_version": (version.get("serverVersion") or {}).get("gitVersion"),
        **node_summary,
        "can_create_pods": can_create,
        "namespace_exists": resource_exists(runner, "namespace", namespace),
        "registry_secret_exists": resource_exists(
            runner, "secret", "ngc-docker-registry-secret", namespace
        ),
        "model_cache_pvc_exists": resource_exists(
            runner, "pvc", "ngc-model-cache", namespace
        ),
        "storage_classes": storage_classes,
        "service_mode": service_mode,
    }
    ready, blockers = evaluate_readiness(facts, service_mode)
    return {"ready": ready, "blockers": blockers, "facts": facts}


def apply_json(runner: Runner, manifest: dict[str, Any]) -> None:
    runner.run(["apply", "-f", "-"], input_text=json.dumps(manifest))


def registry_secret_manifest(namespace: str, api_key: str) -> dict[str, Any]:
    auth = base64.b64encode(f"$oauthtoken:{api_key}".encode()).decode()
    docker_config = {"auths": {"nvcr.io": {"username": "$oauthtoken", "auth": auth}}}
    encoded_config = base64.b64encode(json.dumps(docker_config).encode()).decode()
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": "ngc-docker-registry-secret", "namespace": namespace},
        "type": "kubernetes.io/dockerconfigjson",
        "data": {".dockerconfigjson": encoded_config},
    }


def pvc_manifest(
    namespace: str, size: str, access_mode: str, storage_class: str | None
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "accessModes": [access_mode],
        "resources": {"requests": {"storage": size}},
    }
    if storage_class:
        spec["storageClassName"] = storage_class
    return {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": "ngc-model-cache", "namespace": namespace},
        "spec": spec,
    }


def prepare(runner: Runner, args: argparse.Namespace) -> None:
    apply_json(
        runner,
        {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": args.namespace}},
    )
    if args.create_registry_secret:
        api_key = os.environ.get("NGC_API_KEY", "").strip()
        if not api_key:
            raise SetupError("NGC_API_KEY is required to create the registry secret")
        apply_json(runner, registry_secret_manifest(args.namespace, api_key))
    if args.create_model_cache_pvc:
        apply_json(
            runner,
            pvc_manifest(
                args.namespace, args.pvc_size, args.pvc_access_mode, args.storage_class
            ),
        )


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--kubectl-command",
        default="kubectl",
        help="For example: kubectl, microk8s kubectl, or k3s kubectl.",
    )
    parser.add_argument("--context")
    parser.add_argument("--ssh-target", help="Optional SSH target such as ubuntu@host.")
    parser.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        help="One ssh option per occurrence, for example -i or a key path.",
    )
    parser.add_argument("--namespace", default="sdg-workflow")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser(
        "audit", help="Run read-only Kubernetes readiness checks."
    )
    common_arguments(audit_parser)
    audit_parser.add_argument(
        "--service-mode", choices=("external", "internal"), default="external"
    )
    audit_parser.add_argument("--json", action="store_true")
    prepare_parser = commands.add_parser(
        "prepare", help="Create explicitly requested prerequisites."
    )
    common_arguments(prepare_parser)
    prepare_parser.add_argument("--create-registry-secret", action="store_true")
    prepare_parser.add_argument("--create-model-cache-pvc", action="store_true")
    prepare_parser.add_argument("--pvc-size", default="500Gi")
    prepare_parser.add_argument(
        "--pvc-access-mode",
        choices=("ReadWriteOnce", "ReadWriteMany"),
        default="ReadWriteMany",
    )
    prepare_parser.add_argument("--storage-class")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    runner = Runner(
        kubectl_command=args.kubectl_command,
        context=args.context,
        ssh_target=args.ssh_target,
        ssh_options=tuple(args.ssh_option),
    )
    try:
        if args.command == "prepare":
            prepare(runner, args)
            print("Requested Kubernetes prerequisites applied. Run audit next.")
            return 0
        report = audit(runner, args.namespace, args.service_mode)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            facts = report["facts"]
            print(f"Kubernetes server: {facts.get('server_version') or 'unknown'}")
            print(f"Ready GPUs: {facts['ready_gpus']}")
            print(f"Service mode: {facts['service_mode']}")
            print(f"Readiness: {'READY' if report['ready'] else 'BLOCKED'}")
            for blocker in report["blockers"]:
                print(f"- {blocker}")
        return 0 if report["ready"] else 2
    except SetupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
