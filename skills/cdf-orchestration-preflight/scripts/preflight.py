#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Read-only Cosmos Data Factory Orchestration installation and workflow preflight."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

READY = "READY"
PARTIAL = "PARTIAL"
BLOCKED = "BLOCKED"
UNVERIFIED = "UNVERIFIED"
NOT_PACKAGED = "NOT_PACKAGED"
WARNING = "WARNING"

NVIDIA_TOOLKIT_CONTAINER = "nvidia-container-toolkit-ctr"
KUBERNETES_API_PROBE_SCRIPT = r"""
command -v /busybox/nc >/dev/null 2>&1 || {
    printf 'MISSING_TOOL\n'
    exit 0
}
help=$(/busybox/nc -h 2>&1 || :)
case "$help" in
    *"-z"*) ;;
    *)
        printf 'UNSUPPORTED\n'
        exit 0
        ;;
esac
case "$help" in
    *"-w"*) ;;
    *)
        printf 'UNSUPPORTED\n'
        exit 0
        ;;
esac
if /busybox/nc -z -w 5 "$1" "$2" >/dev/null 2>&1; then
    printf 'REACHABLE\n'
else
    status=$?
    if [ "$status" -eq 1 ]; then
        printf 'UNREACHABLE\n'
    else
        printf 'ERROR\n'
    fi
fi
""".strip()
NVCUVID_PROBE_ROOT_SPECS = (
    "/driver-root:/driver-root/usr/lib",
    "/driver-root:/driver-root/usr/lib64",
    "/driver-root:/driver-root/lib",
    "/driver-root:/driver-root/lib64",
    "/host:/host/usr/lib",
    "/host:/host/usr/lib64",
    "/host:/host/lib",
    "/host:/host/lib64",
)
NVCUVID_PROBE_SCRIPT = r"""
probe_error() {
    printf 'ERROR\n'
    exit 0
}
command -v find >/dev/null 2>&1 || probe_error
command -v readlink >/dev/null 2>&1 || probe_error
for root_spec in "$@"; do
    mount_root=${root_spec%%:*}
    root=${root_spec#*:}
    [ -n "$mount_root" ] || probe_error
    [ "$root" != "$root_spec" ] || probe_error
    case "$root" in
        "$mount_root"/*) ;;
        *) probe_error ;;
    esac
    [ -d "$root" ] || continue
    candidates=$(find "$root" -name libnvcuvid.so.1 -print 2>/dev/null) || probe_error
    for candidate in $candidates; do
        current=$candidate
        links=0
        while [ -L "$current" ]; do
            links=$((links + 1))
            [ "$links" -le 32 ] || probe_error
            target=$(readlink "$current") || probe_error
            case "$target" in
                /*) current="$mount_root$target" ;;
                *) current="${current%/*}/$target" ;;
            esac
        done
        if [ -f "$current" ]; then
            printf 'PRESENT\n'
            exit 0
        fi
    done
done
printf 'MISSING\n'
""".strip()


@dataclass
class Check:
    id: str
    category: str
    status: str
    summary: str
    required_for_install: bool = False


MAX_COMMAND_STDOUT = 64_000_000


@dataclass
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass
class KubernetesFacts:
    reachable: bool = False
    supported: bool = False
    namespace_exists: bool = False
    total_gpus: int | None = None
    requested_gpus: int | None = None
    free_gpus: int | None = None
    nvcuvid_ready: bool | None = None
    storage_classes: dict[str, str] = field(default_factory=dict)
    repo_nfs_provisioner_ready: bool | None = None
    install_permissions: bool | None = None
    cleanup_permissions: bool | None = None


@dataclass
class ControllerFacts:
    installed: bool = False
    healthy: bool = False
    api_verified: bool = False
    dags: dict[str, dict[str, Any]] = field(default_factory=dict)
    pools: dict[str, dict[str, Any]] = field(default_factory=dict)
    import_error_count: int | None = None
    pvc_phases: dict[str, str] | None = None


class Runner:
    """Subprocess wrapper with bounded output and no shell evaluation."""

    def run(self, command: list[str], timeout: int = 20) -> CommandResult:
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return CommandResult(127, "", type(exc).__name__)
        if len(completed.stdout) > MAX_COMMAND_STDOUT:
            # Truncated JSON is unparseable; report the overflow instead of partial data.
            return CommandResult(1, "", f"stdout exceeded {MAX_COMMAND_STDOUT} characters")
        return CommandResult(
            completed.returncode,
            completed.stdout,
            completed.stderr[:2000],
        )


def json_object(result: CommandResult) -> dict[str, Any] | None:
    if result.returncode != 0:
        return None
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse_version(text: str) -> tuple[int, ...] | None:
    match = re.search(r"(?<!\d)(\d+)\.(\d+)(?:\.(\d+))?", text)
    if not match:
        return None
    return tuple(int(part) for part in match.groups(default="0"))


def version_at_least(version: tuple[int, ...] | None, minimum: tuple[int, ...]) -> bool:
    if version is None:
        return False
    width = max(len(version), len(minimum))
    return version + (0,) * (width - len(version)) >= minimum + (0,) * (width - len(minimum))


SECRET_PRESENCE_NAMES = frozenset(
    {
        "NGC_API_KEY",
        "HF_TOKEN",
        "OSMO_TOKEN",
        "VLM_API_KEY",
        "LLM_API_KEY",
        "AWS_S3_BUCKET",
        "AWS_S3_REGION",
        "AWS_S3_ACCESS_KEY_ID",
        "AWS_S3_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    }
)


def parse_env_presence(path: Path, environ: dict[str, str]) -> dict[str, bool]:
    """Return only variable presence; never evaluate or retain secret values."""
    names = SECRET_PRESENCE_NAMES
    present = {name: bool(environ.get(name, "").strip()) for name in names}
    if not path.is_file():
        return present
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$", raw_line)
        if not match or match.group(1) not in names:
            continue
        raw_value = match.group(2).strip()
        if raw_value in {"", "''", '""'}:
            continue
        present[match.group(1)] = True
    return present


def extract_storage_classes(values_path: Path) -> set[str]:
    if not values_path.is_file():
        return set()
    values = values_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"^\s*storageClass(?:Name)?:\s*[\"']?([^\s#\"']+)",
        values,
        flags=re.MULTILINE,
    )
    return {value for value in matches if value and value not in {"null", "~"}}


def extract_make_value(path: Path, variable: str) -> str | None:
    if not path.is_file():
        return None
    pattern = re.compile(rf"^{re.escape(variable)}\s*\?=\s*(\S+)\s*$", re.MULTILINE)
    match = pattern.search(path.read_text(encoding="utf-8", errors="replace"))
    return match.group(1) if match else None


def parse_prebuilt_controller_image(value: str) -> tuple[str, str] | None:
    """Split a registry-qualified image reference into Helm repository and tag."""
    if not value or value != value.strip() or re.search(r"\s", value):
        return None
    if "://" in value or "@" in value:
        return None
    repository, separator, tag = value.rpartition(":")
    if not separator or not repository or not tag or "/" not in repository:
        return None
    registry, *path_parts = repository.split("/")
    if not path_parts or not all(path_parts):
        return None
    if registry != "localhost" and "." not in registry and ":" not in registry:
        return None
    if not re.fullmatch(r"[a-z0-9][a-z0-9.:-]*", registry):
        return None
    if not all(re.fullmatch(r"[a-z0-9][a-z0-9._-]*", part) for part in path_parts):
        return None
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag):
        return None
    return repository, tag


def quantity_to_int(value: Any) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def effective_gpu_requests(pod: dict[str, Any]) -> int:
    spec = pod.get("spec") or {}

    def container_gpus(container: dict[str, Any]) -> int:
        resources = container.get("resources") or {}
        resource_requests = resources.get("requests") or {}
        limits = resources.get("limits") or {}
        return quantity_to_int(
            resource_requests.get("nvidia.com/gpu", limits.get("nvidia.com/gpu", 0))
        )

    regular = sum(container_gpus(item) for item in spec.get("containers", []))
    init = max(
        (container_gpus(item) for item in spec.get("initContainers", [])),
        default=0,
    )
    return max(regular, init)


def kubernetes_node_inventory(
    nodes_payload: dict[str, Any],
) -> tuple[int, int, set[str], set[str]]:
    """Return Ready count, Ready GPU capacity, all GPU nodes, and Ready GPU nodes."""
    ready_nodes = 0
    total_gpus = 0
    gpu_nodes: set[str] = set()
    ready_gpu_nodes: set[str] = set()
    for node in nodes_payload.get("items", []):
        metadata = node.get("metadata") or {}
        node_name = str(metadata.get("name") or "")
        status = node.get("status") or {}
        conditions = status.get("conditions") or []
        ready = any(
            item.get("type") == "Ready" and item.get("status") == "True" for item in conditions
        )
        allocatable_gpus = quantity_to_int(
            (status.get("allocatable") or {}).get("nvidia.com/gpu", 0)
        )
        capacity_gpus = quantity_to_int((status.get("capacity") or {}).get("nvidia.com/gpu", 0))
        if ready:
            ready_nodes += 1
            total_gpus += allocatable_gpus
        if node_name and max(allocatable_gpus, capacity_gpus):
            gpu_nodes.add(node_name)
            if ready:
                ready_gpu_nodes.add(node_name)
    return ready_nodes, total_gpus, gpu_nodes, ready_gpu_nodes


def nvidia_toolkit_probe_targets(
    pods_payload: dict[str, Any] | None, ready_gpu_nodes: set[str]
) -> dict[str, list[tuple[str, str]]]:
    """Map GPU nodes to Running GPU Operator toolkit pods with host driver mounts."""
    candidates: list[tuple[str, str, str]] = []
    for pod in (pods_payload or {}).get("items", []):
        if str((pod.get("status") or {}).get("phase") or "") != "Running":
            continue
        metadata = pod.get("metadata") or {}
        if metadata.get("deletionTimestamp"):
            continue
        if (metadata.get("labels") or {}).get("app") != ("nvidia-container-toolkit-daemonset"):
            continue
        if not any(
            owner.get("kind") == "DaemonSet"
            and owner.get("name") == "nvidia-container-toolkit-daemonset"
            for owner in metadata.get("ownerReferences") or []
        ):
            continue
        spec = pod.get("spec") or {}
        node_name = str(spec.get("nodeName") or "")
        if node_name not in ready_gpu_nodes:
            continue
        ready_containers = {
            str(status.get("name") or "")
            for status in (pod.get("status") or {}).get("containerStatuses") or []
            if status.get("ready") is True
        }
        if NVIDIA_TOOLKIT_CONTAINER not in ready_containers:
            continue
        host_paths_by_volume = {
            str(volume.get("name") or ""): str((volume.get("hostPath") or {}).get("path") or "")
            for volume in spec.get("volumes") or []
            if volume.get("hostPath")
        }
        for container in spec.get("containers") or []:
            if container.get("name") != NVIDIA_TOOLKIT_CONTAINER:
                continue
            has_read_only_host_root = any(
                mount.get("mountPath") == "/host"
                and mount.get("readOnly") is True
                and host_paths_by_volume.get(str(mount.get("name") or "")) == "/"
                for mount in container.get("volumeMounts") or []
            )
            has_driver_root = any(
                mount.get("mountPath") == "/driver-root"
                and host_paths_by_volume.get(str(mount.get("name") or "")).startswith("/")
                for mount in container.get("volumeMounts") or []
            )
            if not has_read_only_host_root or not has_driver_root:
                continue
            namespace = str(metadata.get("namespace") or "default")
            pod_name = str(metadata.get("name") or "")
            if pod_name:
                candidates.append((node_name, namespace, pod_name))

    targets: dict[str, list[tuple[str, str]]] = {}
    for node_name, namespace, pod_name in sorted(candidates):
        targets.setdefault(node_name, []).append((namespace, pod_name))
    return targets


# Fixed probes: no shell interpolation or credential output from selected containers.
PYTHON_API_PROBE = """
import socket, sys
try:
    with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=5):
        print("REACHABLE")
except OSError:
    print("UNREACHABLE")
"""
PYTHON_NVCUVID_PROBE = """
import ctypes
try:
    ctypes.CDLL("libnvcuvid.so.1")
    print("PRESENT")
except OSError:
    print("MISSING")
"""


def selected_probe_targets(
    pods_payload: dict[str, Any] | None, probe_pods: list[str]
) -> list[tuple[str, str, str, str, bool]]:
    """Validate explicit namespace/pod/container selections; never pick arbitrary workloads."""
    targets = []
    for reference in dict.fromkeys(probe_pods):
        matches = []
        for pod in (pods_payload or {}).get("items", []):
            metadata, spec, status = (pod.get(key) or {} for key in ("metadata", "spec", "status"))
            if metadata.get("deletionTimestamp") or spec.get("hostNetwork"):
                continue
            if status.get("phase") != "Running" or not spec.get("nodeName"):
                continue
            namespace, name = metadata.get("namespace", "default"), metadata.get("name", "")
            for container in spec.get("containers") or []:
                container_name = container.get("name", "")
                if reference != f"{namespace}/{name}/{container_name}":
                    continue
                if not any(
                    item.get("name") == container_name and item.get("ready") is True
                    for item in status.get("containerStatuses") or []
                ):
                    continue
                resources = container.get("resources") or {}
                gpu = quantity_to_int((resources.get("limits") or {}).get("nvidia.com/gpu", 0)) > 0
                matches.append((spec["nodeName"], namespace, name, container_name, gpu))
        if len(matches) != 1:
            return []  # Invalid explicit evidence must not silently fall back to another pod.
        targets.extend(matches)
    return targets


def kubernetes_api_endpoint_check(
    runner: Runner,
    base: list[str],
    server: str,
    ready_gpu_nodes: set[str],
    pods_payload: dict[str, Any] | None,
    probe_pods: list[str] | None = None,
) -> Check:
    """Probe the API endpoint from a selected container or existing toolkit pod."""
    if not server:
        return Check(
            "kubernetes.api_endpoint",
            "kubernetes",
            BLOCKED,
            "Could not determine the kubeconfig API endpoint",
            True,
        )

    try:
        parsed = urllib.parse.urlparse(server)
        hostname = parsed.hostname
        port = parsed.port
        scheme = parsed.scheme
    except ValueError:
        hostname = None
        port = None
        scheme = ""
    if port is None:
        port = 443 if scheme == "https" else 80 if scheme == "http" else None
    if not hostname or port is None:
        return Check(
            "kubernetes.api_endpoint",
            "kubernetes",
            BLOCKED,
            "Kubeconfig API endpoint does not contain a valid host and port",
            True,
        )

    normalized_hostname = hostname.lower().rstrip(".")
    try:
        loopback = ipaddress.ip_address(normalized_hostname).is_loopback
    except ValueError:
        loopback = normalized_hostname == "localhost"
    if loopback:
        return Check(
            "kubernetes.api_endpoint",
            "kubernetes",
            BLOCKED,
            "Kubeconfig API endpoint is loopback and is not a valid pod-reachable cluster endpoint",
            True,
        )

    if probe_pods:
        selected = selected_probe_targets(pods_payload, probe_pods)
        if not selected:
            return Check(
                "kubernetes.api_endpoint",
                "kubernetes",
                UNVERIFIED,
                "Each --probe-pod must identify a Running, ready, non-hostNetwork container",
                True,
            )
        _, namespace, pod_name, container, _ = selected[0]
        result = runner.run(
            [
                *base,
                "exec",
                "-n",
                namespace,
                pod_name,
                "-c",
                container,
                "--",
                "python3",
                "-c",
                PYTHON_API_PROBE,
                hostname,
                str(port),
            ],
            timeout=10,
        )
        marker = result.stdout.strip() if result.returncode == 0 else ""
        status = (
            READY if marker == "REACHABLE" else BLOCKED if marker == "UNREACHABLE" else UNVERIFIED
        )
        return Check(
            "kubernetes.api_endpoint",
            "kubernetes",
            status,
            f"Kubernetes API TCP probe from {namespace}/{pod_name}/{container}: "
            + (
                marker
                if marker in {"REACHABLE", "UNREACHABLE"}
                else "inconclusive; Python 3 and pods/exec access required"
            ),
            True,
        )

    targets = nvidia_toolkit_probe_targets(pods_payload, ready_gpu_nodes)
    candidates = sorted(
        (node_name, namespace, pod_name)
        for node_name, node_targets in targets.items()
        for namespace, pod_name in node_targets
    )
    if not candidates:
        return Check(
            "kubernetes.api_endpoint",
            "kubernetes",
            UNVERIFIED,
            "No ready NVIDIA toolkit pod is available; supply --probe-pod NAMESPACE/POD/CONTAINER with Python 3",
            True,
        )

    _, namespace, pod_name = candidates[0]
    result = runner.run(
        [
            *base,
            "exec",
            "-n",
            namespace,
            pod_name,
            "-c",
            NVIDIA_TOOLKIT_CONTAINER,
            "--",
            "/busybox/sh",
            "-c",
            KUBERNETES_API_PROBE_SCRIPT,
            "api-endpoint-probe",
            hostname,
            str(port),
        ],
        timeout=10,
    )
    marker = result.stdout.strip() if result.returncode == 0 else ""
    if marker == "REACHABLE":
        return Check(
            "kubernetes.api_endpoint",
            "kubernetes",
            READY,
            "Kubernetes API endpoint is reachable from the selected NVIDIA toolkit pod",
            True,
        )
    if marker == "UNREACHABLE":
        return Check(
            "kubernetes.api_endpoint",
            "kubernetes",
            BLOCKED,
            "Kubernetes API endpoint is not reachable from the selected NVIDIA toolkit pod",
            True,
        )
    return Check(
        "kubernetes.api_endpoint",
        "kubernetes",
        UNVERIFIED,
        "Could not execute a conclusive Kubernetes API reachability probe "
        "from the selected cluster pod",
        True,
    )


def nvcuvid_library_check(
    runner: Runner,
    base: list[str],
    gpu_nodes: set[str],
    ready_gpu_nodes: set[str],
    pods_payload: dict[str, Any] | None,
    probe_pods: list[str] | None = None,
) -> tuple[Check, bool | None]:
    """Check libnvcuvid through selected GPU containers or toolkit pods without mutation."""
    if not gpu_nodes:
        return (
            Check(
                "kubernetes.gpu_nvcuvid",
                "compute",
                UNVERIFIED,
                "No nodes advertise NVIDIA GPU capacity",
            ),
            None,
        )

    selected = selected_probe_targets(pods_payload, probe_pods or [])
    if probe_pods and not selected:
        return Check(
            "kubernetes.gpu_nvcuvid",
            "compute",
            UNVERIFIED,
            "Invalid --probe-pod selection; use Running, ready, non-hostNetwork containers",
        ), None
    targets = nvidia_toolkit_probe_targets(pods_payload, ready_gpu_nodes)
    missing: list[str] = []
    unverified = set(gpu_nodes - ready_gpu_nodes)
    for node_name in sorted(ready_gpu_nodes):
        selected_on_node = [
            (ns, pod, container)
            for node, ns, pod, container, gpu in selected
            if node == node_name and gpu
        ]
        node_targets = selected_on_node or [
            (ns, pod, NVIDIA_TOOLKIT_CONTAINER) for ns, pod in targets.get(node_name, [])
        ]
        if not node_targets:
            unverified.add(node_name)
            continue
        present = False
        conclusive_missing = False
        for namespace, pod_name, container in node_targets:
            probe_command = (
                ["python3", "-c", PYTHON_NVCUVID_PROBE]
                if selected_on_node
                else ["sh", "-c", NVCUVID_PROBE_SCRIPT, "nvcuvid-probe", *NVCUVID_PROBE_ROOT_SPECS]
            )
            result = runner.run(
                [
                    *base,
                    "exec",
                    "-n",
                    namespace,
                    pod_name,
                    "-c",
                    container,
                    "--",
                    *probe_command,
                ]
            )
            marker = result.stdout.strip() if result.returncode == 0 else ""
            if marker == "PRESENT":
                present = True
                break
            if marker == "MISSING":
                conclusive_missing = True
        if present:
            continue
        if conclusive_missing:
            missing.append(node_name)
        else:
            unverified.add(node_name)

    if missing:
        summary = "libnvcuvid.so.1 is missing or could not be loaded on GPU node(s): " + ", ".join(
            missing
        )
        if unverified:
            summary += "; could not inspect node(s): " + ", ".join(sorted(unverified))
        return Check("kubernetes.gpu_nvcuvid", "compute", BLOCKED, summary), False
    if unverified:
        return (
            Check(
                "kubernetes.gpu_nvcuvid",
                "compute",
                UNVERIFIED,
                "Could not verify libnvcuvid.so.1 on GPU node(s): " + ", ".join(sorted(unverified)),
            ),
            None,
        )
    return (
        Check(
            "kubernetes.gpu_nvcuvid",
            "compute",
            READY,
            f"libnvcuvid.so.1 is present on all {len(gpu_nodes)} GPU node(s)",
        ),
        True,
    )


def tool_checks(runner: Runner, *, controller_build_required: bool = True) -> list[Check]:
    checks: list[Check] = []
    specs = [
        (
            "python",
            "python3",
            ["python3", "--version"],
            (3, 11, 0),
            (3, 12, 0),
            "Python 3.11",
        ),
        (
            "kubectl",
            "kubectl",
            ["kubectl", "version", "--client", "-o", "json"],
            (1, 26, 0),
            None,
            "kubectl 1.26+",
        ),
        (
            "helm",
            "helm",
            ["helm", "version", "--short"],
            (3, 0, 0),
            (4, 0, 0),
            "Helm 3",
        ),
        ("aws", "aws", ["aws", "--version"], (2, 0, 0), (3, 0, 0), "AWS CLI v2"),
    ]
    if controller_build_required:
        specs.insert(
            3,
            (
                "docker",
                "docker",
                ["docker", "--version"],
                (29, 6, 0),
                None,
                "Docker 29.6.0+",
            ),
        )
    for check_id, executable, command, minimum, maximum, label in specs:
        if not shutil.which(executable):
            checks.append(
                Check(
                    f"tool.{check_id}",
                    "tools",
                    BLOCKED,
                    f"{label} is not on PATH",
                    True,
                )
            )
            continue
        result = runner.run(command)
        combined = f"{result.stdout} {result.stderr}"
        version = parse_version(combined)
        valid = result.returncode == 0 and version_at_least(version, minimum)
        if maximum is not None and version is not None:
            valid = valid and version < maximum
        found = ".".join(str(part) for part in version) if version else "unknown"
        if valid:
            status, summary = READY, f"{label}: found {found}"
        elif result.returncode == 0:
            # The tool runs, but this version is outside the tested configuration.
            status = WARNING
            summary = f"{label} recommended (tested configuration); found {found}, untested, proceed at your own risk"
        else:
            status, summary = BLOCKED, f"{label} required; found {found}"
        checks.append(Check(f"tool.{check_id}", "tools", status, summary, True))
    if not controller_build_required:
        docker_cli = shutil.which("docker") is not None
        checks.append(
            Check(
                "tool.docker",
                "tools",
                READY if docker_cli else BLOCKED,
                "Docker CLI is available for prebuilt image manifest inspection"
                if docker_cli
                else "Docker CLI is required to inspect the prebuilt image manifest",
                True,
            )
        )
    envsubst = shutil.which("envsubst") is not None
    checks.append(
        Check(
            "tool.envsubst",
            "tools",
            READY if envsubst else WARNING,
            "envsubst is available"
            if envsubst
            else "envsubst is missing; it is required only for make install nfs",
        )
    )
    if shutil.which("docker") and controller_build_required:
        daemon = runner.run(["docker", "info", "--format", "{{.ServerVersion}}"])
        checks.append(
            Check(
                "docker.daemon",
                "tools",
                READY if daemon.returncode == 0 else BLOCKED,
                "Docker daemon is reachable"
                if daemon.returncode == 0
                else "Docker is installed but its daemon is not reachable",
                True,
            )
        )
    elif shutil.which("docker"):
        checks.append(
            Check(
                "docker.daemon",
                "tools",
                WARNING,
                "Docker daemon readiness is not required for an explicitly selected prebuilt controller image",
            )
        )
    return checks


def repository_checks(
    repo_root: Path,
    secret_presence: dict[str, bool],
    runner: Runner,
    *,
    prebuilt_controller_image: str | None = None,
    confirmed_push_repository: str | None = None,
) -> list[Check]:
    checks: list[Check] = []
    required_paths = [
        "Makefile",
        "make/setup.mk",
        "make/install.mk",
        "deploy/values.yaml",
        "secrets.env.example",
        "scripts/install_nfs.sh",
    ]
    missing = [path for path in required_paths if not (repo_root / path).is_file()]
    checks.append(
        Check(
            "repo.layout",
            "repository",
            BLOCKED if missing else READY,
            f"Missing required repository files: {', '.join(missing)}"
            if missing
            else "Required repository files are present",
            True,
        )
    )
    secrets_file = repo_root / "secrets.env"
    checks.append(
        Check(
            "repo.secrets_file",
            "credentials",
            READY if secrets_file.is_file() else BLOCKED,
            "secrets.env exists"
            if secrets_file.is_file()
            else "secrets.env is missing; copy secrets.env.example and fill it in",
            True,
        )
    )
    required_secrets = [
        "NGC_API_KEY",
        "HF_TOKEN",
        "AWS_S3_BUCKET",
        "AWS_S3_REGION",
        "AWS_S3_ACCESS_KEY_ID",
        "AWS_S3_SECRET_ACCESS_KEY",
    ]
    missing_secrets = [name for name in required_secrets if not secret_presence.get(name)]
    checks.append(
        Check(
            "credentials.required",
            "credentials",
            BLOCKED if missing_secrets else READY,
            f"Missing required secret/config names: {', '.join(missing_secrets)}"
            if missing_secrets
            else "Required secret/config names are populated",
            True,
        )
    )
    if missing_secrets:
        s3_status = BLOCKED
        s3_summary = (
            "S3 access cannot be tested until all required S3 configuration names are populated"
        )
    else:
        s3_status = UNVERIFIED
        s3_summary = (
            "S3 credential names are populated; bucket read/write access was not "
            "tested by the base audit"
        )
    checks.append(Check("credentials.s3_access", "credentials", s3_status, s3_summary))
    if secret_presence.get("AWS_SESSION_TOKEN"):
        checks.append(
            Check(
                "credentials.aws_session",
                "credentials",
                BLOCKED,
                "AWS_SESSION_TOKEN is present, but this deployment supports long-lived IAM user credentials only",
                True,
            )
        )
    else:
        checks.append(
            Check(
                "credentials.aws_session",
                "credentials",
                READY,
                "No unsupported AWS session token was detected",
                True,
            )
        )

    if prebuilt_controller_image is not None:
        parsed_image = parse_prebuilt_controller_image(prebuilt_controller_image)
        if parsed_image is None:
            checks.append(
                Check(
                    "image.repository",
                    "repository",
                    BLOCKED,
                    "Prebuilt controller image must be a registry-qualified repository:tag reference",
                    True,
                )
            )
            checks.append(
                Check(
                    "image.tag",
                    "repository",
                    BLOCKED,
                    "Prebuilt controller image tag is invalid or missing",
                    True,
                )
            )
            return checks

        repository, tag = parsed_image
        checks.append(
            Check(
                "image.repository",
                "repository",
                READY,
                f"Explicit prebuilt controller image repository is selected ({repository}); push access is not required",
                True,
            )
        )
        checks.append(
            Check(
                "image.tag",
                "repository",
                WARNING if tag == "latest" else READY,
                "Prebuilt controller image uses shared tag 'latest'"
                if tag == "latest"
                else f"Prebuilt controller image tag is configured ({tag})",
            )
        )
        manifest = runner.run(
            ["docker", "manifest", "inspect", prebuilt_controller_image], timeout=60
        )
        manifest_payload = json_object(manifest)
        manifest_verified = bool(
            manifest_payload
            and (
                "schemaVersion" in manifest_payload
                or "manifests" in manifest_payload
                or "config" in manifest_payload
            )
        )
        checks.append(
            Check(
                "image.pull_access",
                "repository",
                READY if manifest_verified else UNVERIFIED,
                f"Prebuilt controller image manifest is accessible with current Docker authentication ({prebuilt_controller_image})"
                if manifest_verified
                else "Could not verify the prebuilt controller image manifest with current Docker authentication",
                True,
            )
        )
        return checks

    common_mk = repo_root / "make/common.mk"
    repository = os.environ.get(
        "AIRFLOW_IMAGE_REPOSITORY", extract_make_value(common_mk, "AIRFLOW_IMAGE_REPOSITORY")
    )
    tag = os.environ.get("AIRFLOW_IMAGE_TAG", extract_make_value(common_mk, "AIRFLOW_IMAGE_TAG"))
    default_repository = "nvcr.io/<your-org>/<your-team>/sdg-workflow-airflow"
    valid_image = parse_prebuilt_controller_image(f"{repository or ''}:{tag or ''}")
    push_confirmed = bool(repository and confirmed_push_repository == repository)
    if not valid_image or (
        repository == default_repository
        and not push_confirmed
        and "AIRFLOW_IMAGE_REPOSITORY" not in os.environ
    ):
        checks.append(
            Check(
                "image.repository",
                "repository",
                BLOCKED,
                "Set AIRFLOW_IMAGE_REPOSITORY and AIRFLOW_IMAGE_TAG in make/common.mk (or export them) "
                "to a registry-qualified path the user can push; the repository default is a placeholder",
                True,
            )
        )
    else:
        checks.append(
            Check(
                "image.repository",
                "repository",
                READY if push_confirmed else UNVERIFIED,
                f"User-confirmed push access for {repository}; no push was performed by the audit"
                if push_confirmed
                else f"Push permission for {repository} is unverified; after confirming access, pass --confirmed-push-repository with this exact repository",
                True,
            )
        )
    checks.append(
        Check(
            "image.tag",
            "repository",
            WARNING if tag == "latest" else READY,
            "AIRFLOW_IMAGE_TAG uses shared tag 'latest'"
            if tag == "latest"
            else f"Airflow image tag is configured ({tag or 'missing'})",
        )
    )
    return checks


def kubectl_base(kubeconfig: Path) -> list[str]:
    return ["kubectl", "--kubeconfig", str(kubeconfig)]


def kubernetes_audit(
    runner: Runner, kubeconfig: Path | None, probe_pods: list[str] | None = None
) -> tuple[list[Check], KubernetesFacts]:
    checks: list[Check] = []
    facts = KubernetesFacts()
    if kubeconfig is None:
        checks.append(
            Check(
                "kubernetes.kubeconfig",
                "kubernetes",
                UNVERIFIED,
                "No kubeconfig path was supplied",
                True,
            )
        )
        return checks, facts
    if os.pathsep in str(kubeconfig):
        checks.append(
            Check(
                "kubernetes.kubeconfig",
                "kubernetes",
                BLOCKED,
                "KUBECONFIG must name one file; path lists are not supported by make setup",
                True,
            )
        )
        return checks, facts
    if not kubeconfig.is_file():
        checks.append(
            Check(
                "kubernetes.kubeconfig",
                "kubernetes",
                BLOCKED,
                f"Kubeconfig file does not exist: {kubeconfig}",
                True,
            )
        )
        return checks, facts
    checks.append(
        Check(
            "kubernetes.kubeconfig",
            "kubernetes",
            READY,
            f"Using user-supplied kubeconfig: {kubeconfig}",
            True,
        )
    )
    base = kubectl_base(kubeconfig)
    version_payload = json_object(runner.run([*base, "version", "-o", "json"]))
    if not version_payload:
        checks.append(
            Check(
                "kubernetes.api",
                "kubernetes",
                BLOCKED,
                "Kubernetes API is not reachable with the supplied kubeconfig",
                True,
            )
        )
        return checks, facts
    facts.reachable = True
    server_version = str(
        (version_payload.get("serverVersion") or {}).get("gitVersion") or "unknown"
    )
    facts.supported = "k3s" not in server_version.lower()
    checks.append(
        Check(
            "kubernetes.api",
            "kubernetes",
            READY,
            f"Kubernetes API is reachable ({server_version})",
            True,
        )
    )
    checks.append(
        Check(
            "kubernetes.distribution",
            "kubernetes",
            READY if facts.supported else BLOCKED,
            "Kubernetes distribution is supported"
            if facts.supported
            else "K3s is not supported by this repository",
            True,
        )
    )

    config_payload = json_object(runner.run([*base, "config", "view", "--minify", "-o", "json"]))
    server = ""
    if config_payload:
        clusters = config_payload.get("clusters") or []
        if clusters:
            server = str((clusters[0].get("cluster") or {}).get("server") or "")

    nodes_payload = json_object(runner.run([*base, "get", "nodes", "-o", "json"]))
    ready_nodes = 0
    total_gpus = 0
    gpu_nodes: set[str] = set()
    ready_gpu_nodes: set[str] = set()
    if nodes_payload is not None:
        ready_nodes, total_gpus, gpu_nodes, ready_gpu_nodes = kubernetes_node_inventory(
            nodes_payload
        )
    facts.total_gpus = total_gpus if nodes_payload is not None else None
    checks.append(
        Check(
            "kubernetes.nodes",
            "kubernetes",
            READY if ready_nodes else BLOCKED,
            f"Ready nodes: {ready_nodes}",
            True,
        )
    )
    checks.append(
        Check(
            "kubernetes.gpu_resource",
            "compute",
            READY if total_gpus else BLOCKED,
            f"Allocatable NVIDIA GPUs on Ready nodes: {total_gpus}"
            if nodes_payload is not None
            else "Could not inspect NVIDIA GPU resources",
        )
    )

    pods_payload = json_object(runner.run([*base, "get", "pods", "-A", "-o", "json"]))
    if pods_payload is not None:
        requested = 0
        for pod in pods_payload.get("items", []):
            phase = str((pod.get("status") or {}).get("phase") or "")
            if phase in {"Pending", "Running"}:
                requested += effective_gpu_requests(pod)
        facts.requested_gpus = requested
        facts.free_gpus = max(total_gpus - requested, 0)
        checks.append(
            Check(
                "kubernetes.gpu_capacity",
                "compute",
                READY,
                f"GPU capacity by requests: {facts.free_gpus} free of {total_gpus} ({requested} requested)",
            )
        )
    else:
        checks.append(
            Check(
                "kubernetes.gpu_capacity",
                "compute",
                UNVERIFIED,
                "Could not list active pods; free GPU capacity is unverified",
            )
        )

    checks.append(
        kubernetes_api_endpoint_check(
            runner, base, server, ready_gpu_nodes, pods_payload, probe_pods
        )
    )

    if nodes_payload is None:
        nvcuvid_check = Check(
            "kubernetes.gpu_nvcuvid",
            "compute",
            UNVERIFIED,
            "Could not inspect GPU nodes for libnvcuvid.so.1",
        )
    else:
        nvcuvid_check, facts.nvcuvid_ready = nvcuvid_library_check(
            runner, base, gpu_nodes, ready_gpu_nodes, pods_payload, probe_pods
        )
    checks.append(nvcuvid_check)

    permission_queries = [
        ["create", "namespaces"],
        ["create", "secrets", "-n", "sdg-workflow"],
        ["create", "persistentvolumeclaims", "-n", "sdg-workflow"],
        ["create", "deployments.apps", "-n", "sdg-workflow"],
        ["create", "clusterroles.rbac.authorization.k8s.io"],
    ]
    permissions = []
    for query in permission_queries:
        result = runner.run([*base, "auth", "can-i", *query])
        permissions.append(result.returncode == 0 and result.stdout.strip().lower() == "yes")
    facts.install_permissions = all(permissions)
    checks.append(
        Check(
            "kubernetes.install_permissions",
            "kubernetes",
            READY if facts.install_permissions else BLOCKED,
            "Current identity has the sampled namespace and cluster-scoped install permissions"
            if facts.install_permissions
            else "Current identity lacks one or more sampled install permissions",
            True,
        )
    )

    cleanup_queries = [
        ["delete", "pods", "-n", "sdg-workflow"],
        ["delete", "deployments.apps", "-n", "sdg-workflow"],
        ["delete", "services", "-n", "sdg-workflow"],
    ]
    cleanup_permissions = []
    for query in cleanup_queries:
        result = runner.run([*base, "auth", "can-i", *query])
        cleanup_permissions.append(
            result.returncode == 0 and result.stdout.strip().lower() == "yes"
        )
    facts.cleanup_permissions = all(cleanup_permissions)

    storage_payload = json_object(
        runner.run([*base, "get", "storageclasses.storage.k8s.io", "-o", "json"])
    )
    if storage_payload is not None:
        for item in storage_payload.get("items", []):
            metadata = item.get("metadata") or {}
            name = str(metadata.get("name") or "")
            if name:
                facts.storage_classes[name] = str(item.get("provisioner") or "")

    repo_nfs_provisioner = "cluster.local/nfs-subdir-external-provisioner"
    if repo_nfs_provisioner in facts.storage_classes.values():
        deployments = json_object(
            runner.run(
                [
                    *base,
                    "get",
                    "deployments.apps",
                    "-A",
                    "-l",
                    "app=nfs-subdir-external-provisioner",
                    "-o",
                    "json",
                ]
            )
        )
        if deployments is not None:
            facts.repo_nfs_provisioner_ready = any(
                quantity_to_int((item.get("status") or {}).get("availableReplicas")) > 0
                for item in deployments.get("items", [])
            )

    namespace = runner.run([*base, "get", "namespace", "sdg-workflow", "-o", "name"])
    facts.namespace_exists = namespace.returncode == 0
    return checks, facts


def storage_checks(repo_root: Path, facts: KubernetesFacts) -> list[Check]:
    required = extract_storage_classes(repo_root / "deploy/values.yaml")
    if not required:
        return [
            Check(
                "storage.values",
                "storage",
                BLOCKED,
                "No configured StorageClass could be read from deploy/values.yaml",
                True,
            )
        ]
    if not facts.reachable:
        return [
            Check(
                "storage.classes",
                "storage",
                UNVERIFIED,
                f"Configured StorageClasses: {', '.join(sorted(required))}; cluster state is unavailable",
                True,
            )
        ]
    missing = sorted(required - set(facts.storage_classes))
    checks = [
        Check(
            "storage.classes",
            "storage",
            BLOCKED if missing else READY,
            f"Missing configured StorageClasses: {', '.join(missing)}"
            if missing
            else f"Configured StorageClasses exist: {', '.join(sorted(required))}",
            True,
        )
    ]
    for name in sorted(required & set(facts.storage_classes)):
        provisioner = facts.storage_classes[name]
        status = UNVERIFIED
        summary = f"StorageClass {name} uses {provisioner}; RWX mount behavior is unverified"
        if provisioner == "cluster.local/nfs-subdir-external-provisioner":
            if facts.repo_nfs_provisioner_ready:
                status = READY
                summary = (
                    f"StorageClass {name} and the repository NFS provisioner "
                    "Deployment are available; a live mount test was not performed"
                )
            else:
                status = BLOCKED
                summary = (
                    f"StorageClass {name} exists, but the repository NFS provisioner "
                    "Deployment is not Available"
                )
        elif "nfs" in provisioner.lower():
            summary = (
                f"StorageClass {name} uses NFS provisioner {provisioner}; provisioner "
                "health and a live mount test remain unverified"
            )
        checks.append(Check(f"storage.class.{name}", "storage", status, summary, True))
    return checks


def controller_audit(
    runner: Runner, kubeconfig: Path | None, facts: KubernetesFacts
) -> tuple[list[Check], ControllerFacts]:
    checks: list[Check] = []
    controller = ControllerFacts()
    if not facts.reachable or kubeconfig is None:
        checks.append(
            Check(
                "controller.state",
                "controller",
                UNVERIFIED,
                "Controller state cannot be checked without Kubernetes access",
            )
        )
        return checks, controller
    base = kubectl_base(kubeconfig)
    if not facts.namespace_exists:
        checks.append(
            Check(
                "controller.state",
                "controller",
                BLOCKED,
                "Namespace sdg-workflow is absent; the controller is not installed",
            )
        )
        return checks, controller

    helm_result = runner.run(
        [
            "helm",
            "--kubeconfig",
            str(kubeconfig),
            "list",
            "-n",
            "sdg-workflow",
            "-o",
            "json",
        ]
    )
    releases: list[dict[str, Any]] = []
    if helm_result.returncode == 0:
        try:
            parsed = json.loads(helm_result.stdout)
            if isinstance(parsed, list):
                releases = [item for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass
    matching = [item for item in releases if item.get("name") == "sdg-workflow-controller"]
    controller.installed = bool(matching)
    deployed = bool(matching and str(matching[0].get("status", "")).lower() == "deployed")
    checks.append(
        Check(
            "controller.helm",
            "controller",
            READY if deployed else BLOCKED,
            "Helm release sdg-workflow-controller is deployed"
            if deployed
            else "Helm release sdg-workflow-controller is not deployed",
        )
    )

    pods_payload = json_object(
        runner.run(
            [
                *base,
                "get",
                "pods",
                "-n",
                "sdg-workflow",
                "-l",
                "release=sdg-workflow-controller",
                "-o",
                "json",
            ]
        )
    )
    pod_items = (pods_payload or {}).get("items", [])
    unhealthy = []
    for pod in pod_items:
        name = str((pod.get("metadata") or {}).get("name") or "unknown")
        phase = str((pod.get("status") or {}).get("phase") or "Unknown")
        if phase not in {"Running", "Succeeded"}:
            unhealthy.append(f"{name}:{phase}")
    pods_healthy = bool(pod_items) and not unhealthy
    checks.append(
        Check(
            "controller.pods",
            "controller",
            READY if pods_healthy else BLOCKED,
            f"Controller pods are healthy ({len(pod_items)})"
            if pods_healthy
            else f"Controller pods are absent or unhealthy: {', '.join(unhealthy) or 'none found'}",
        )
    )

    pvc_payload = json_object(runner.run([*base, "get", "pvc", "-n", "sdg-workflow", "-o", "json"]))
    pvc_items = (pvc_payload or {}).get("items", [])
    if pvc_payload is not None:
        controller.pvc_phases = {
            str((item.get("metadata") or {}).get("name") or "unknown"): str(
                (item.get("status") or {}).get("phase") or "Unknown"
            )
            for item in pvc_items
        }
    pending = [
        str((item.get("metadata") or {}).get("name") or "unknown")
        for item in pvc_items
        if str((item.get("status") or {}).get("phase") or "") != "Bound"
    ]
    pvc_healthy = bool(pvc_items) and not pending
    checks.append(
        Check(
            "controller.pvcs",
            "controller",
            READY if pvc_healthy else BLOCKED,
            f"Controller PVCs are Bound ({len(pvc_items)})"
            if pvc_healthy
            else f"Controller PVCs are absent or not Bound: {', '.join(pending) or 'none found'}",
        )
    )
    controller.healthy = deployed and pods_healthy and pvc_healthy
    return checks, controller


def airflow_get(base_url: str, token: str, path: str) -> dict[str, Any] | None:
    url = f"{base_url.rstrip('/')}{path}"
    if urllib.parse.urlparse(url).scheme not in {"http", "https"}:
        return None
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:  # nosec B310
            payload = json.loads(response.read().decode("utf-8"))
    except (
        OSError,
        urllib.error.HTTPError,
        urllib.error.URLError,
        json.JSONDecodeError,
    ):
        return None
    return payload if isinstance(payload, dict) else None


def airflow_audit(controller: ControllerFacts) -> list[Check]:
    checks: list[Check] = []
    base_url = os.environ.get("AIRFLOW_URL", "").strip()
    token = os.environ.get("AIRFLOW_TOKEN", "").strip()
    if not base_url or not token:
        checks.append(
            Check(
                "controller.airflow_api",
                "controller",
                UNVERIFIED,
                "Set AIRFLOW_URL and AIRFLOW_TOKEN to verify DAGs, pools, and import errors",
            )
        )
        return checks
    dags_payload = airflow_get(base_url, token, "/api/v2/dags?limit=1000")
    pools_payload = airflow_get(base_url, token, "/api/v2/pools?limit=1000")
    if dags_payload is None or pools_payload is None:
        checks.append(
            Check(
                "controller.airflow_api",
                "controller",
                BLOCKED,
                "Airflow API authentication or connectivity failed",
            )
        )
        return checks
    controller.api_verified = True
    controller.dags = {
        str(item.get("dag_id")): item
        for item in dags_payload.get("dags", [])
        if isinstance(item, dict) and item.get("dag_id")
    }
    controller.pools = {
        str(item.get("name")): item
        for item in pools_payload.get("pools", [])
        if isinstance(item, dict) and item.get("name")
    }
    imports_payload = airflow_get(base_url, token, "/api/v2/importErrors?limit=1000")
    if imports_payload is not None:
        controller.import_error_count = len(imports_payload.get("import_errors", []))
    checks.append(
        Check(
            "controller.airflow_api",
            "controller",
            READY,
            f"Airflow API verified {len(controller.dags)} DAGs and {len(controller.pools)} pools",
        )
    )
    if controller.import_error_count:
        checks.append(
            Check(
                "controller.import_errors",
                "controller",
                WARNING,
                f"Airflow reports {controller.import_error_count} import error(s); inspect them before running workflows",
            )
        )
    elif controller.import_error_count == 0:
        checks.append(
            Check(
                "controller.import_errors",
                "controller",
                READY,
                "Airflow reports no import errors",
            )
        )
    else:
        checks.append(
            Check(
                "controller.import_errors",
                "controller",
                UNVERIFIED,
                "Airflow import errors could not be queried",
            )
        )
    return checks


def mode_status(
    profile: dict[str, Any],
    mode: dict[str, Any],
    packaged: bool,
    secret_presence: dict[str, bool],
    kube: KubernetesFacts,
    controller: ControllerFacts,
) -> dict[str, Any]:
    blockers: list[str] = []
    unverified: list[str] = []
    warnings: list[str] = []
    if not packaged:
        return {
            "name": mode["name"],
            "status": NOT_PACKAGED,
            "blockers": [],
            "unverified": [],
            "warnings": [],
        }
    if not kube.reachable:
        blockers.append("Kubernetes API is not reachable")
    elif not kube.supported:
        blockers.append("Kubernetes distribution is unsupported")
    if not controller.installed:
        blockers.append("CDF controller is not installed")
    elif not controller.healthy:
        blockers.append("CDF controller is not healthy")

    required_secrets = mode.get("secrets", [])
    missing_secrets = [name for name in required_secrets if not secret_presence.get(name)]
    if missing_secrets:
        blockers.append(f"missing secret/config names: {', '.join(missing_secrets)}")

    floor = int(mode.get("gpu_floor", 0))
    practical = int(mode.get("gpu_practical", floor))
    if mode.get("requires_nvcuvid"):
        if kube.nvcuvid_ready is False:
            blockers.append("libnvcuvid.so.1 is missing on one or more GPU nodes")
        elif kube.nvcuvid_ready is None:
            unverified.append("libnvcuvid.so.1 was not verified on every GPU node")
    for claim in mode.get("required_pvcs", []):
        name = claim["name"]
        hint = f"; {claim['remediation']}" if claim.get("remediation") else ""
        if controller.pvc_phases is None:
            unverified.append(f"PVC sdg-workflow/{name} was not inspected")
        elif name not in controller.pvc_phases:
            blockers.append(f"PVC sdg-workflow/{name} is missing{hint}")
        elif controller.pvc_phases[name] != "Bound":
            blockers.append(
                f"PVC sdg-workflow/{name} is {controller.pvc_phases[name]}, not Bound{hint}"
            )
    if floor:
        if kube.free_gpus is None:
            unverified.append(f"free GPU capacity for floor {floor} is unknown")
        elif kube.free_gpus < floor:
            blockers.append(f"needs at least {floor} free GPU(s); found {kube.free_gpus}")
        elif kube.free_gpus < practical:
            warnings.append(f"meets floor {floor} but is below practical target {practical}")

    if profile.get("platform") == "operational" and kube.cleanup_permissions is False:
        blockers.append("current Kubernetes identity lacks cleanup delete permissions")
    if mode.get("external_endpoints"):
        unverified.append(
            "payload-specific external endpoints were not tested from the task network"
        )
    if mode.get("osmo_access"):
        unverified.append("OSMO account, pool access, and job-slot capacity were not tested")

    if controller.api_verified:
        dag = controller.dags.get(profile["dag_id"])
        if dag is None:
            blockers.append("DAG is not registered in Airflow")
        elif bool(dag.get("is_paused")):
            blockers.append("DAG is paused")
        for pool_name in mode.get("pools", []):
            pool = controller.pools.get(pool_name)
            if pool is None:
                blockers.append(f"Airflow pool {pool_name} is missing")
                continue
            slots = quantity_to_int(pool.get("slots"))
            open_slots = pool.get("open_slots")
            if slots < 1:
                blockers.append(f"Airflow pool {pool_name} has no slots")
            elif open_slots is not None and quantity_to_int(open_slots) < 1:
                blockers.append(f"Airflow pool {pool_name} has no open slots")
    else:
        unverified.append(
            "Airflow DAG registration, paused state, and pool capacity were not verified"
        )

    status = BLOCKED if blockers else UNVERIFIED if unverified else READY
    return {
        "name": mode["name"],
        "status": status,
        "blockers": blockers,
        "unverified": unverified,
        "warnings": warnings,
    }


def workflow_report(
    repo_root: Path,
    profiles: list[dict[str, Any]],
    secret_presence: dict[str, bool],
    kube: KubernetesFacts,
    controller: ControllerFacts,
) -> tuple[list[dict[str, Any]], str]:
    workflows: list[dict[str, Any]] = []
    expected_ids: set[str] = set()
    for profile in profiles:
        expected_ids.add(profile["dag_id"])
        package_paths = [profile["package_path"], *profile.get("legacy_package_paths", [])]
        packaged = any((repo_root / path).is_file() for path in package_paths)
        modes = [
            mode_status(profile, mode, packaged, secret_presence, kube, controller)
            for mode in profile["modes"]
        ]
        statuses = {mode["status"] for mode in modes}
        if not packaged:
            status = NOT_PACKAGED
        elif READY in statuses:
            status = READY
        elif statuses == {BLOCKED}:
            status = BLOCKED
        else:
            status = UNVERIFIED
        workflows.append(
            {
                "dag_id": profile["dag_id"],
                "platform": profile["platform"],
                "status": status,
                "modes": modes,
            }
        )

    if controller.api_verified:
        for dag_id in sorted(set(controller.dags) - expected_ids):
            workflows.append(
                {
                    "dag_id": dag_id,
                    "platform": "unprofiled",
                    "status": UNVERIFIED,
                    "modes": [
                        {
                            "name": "unprofiled",
                            "status": UNVERIFIED,
                            "blockers": [],
                            "unverified": ["live DAG has no readiness profile"],
                            "warnings": [],
                        }
                    ],
                }
            )

    packaged_workflows = [item for item in workflows if item["status"] != NOT_PACKAGED]
    ready_count = sum(item["status"] == READY for item in packaged_workflows)
    if packaged_workflows and ready_count == len(packaged_workflows):
        overall = READY
    elif ready_count:
        overall = PARTIAL
    elif any(item["status"] == BLOCKED for item in packaged_workflows):
        overall = BLOCKED
    else:
        overall = UNVERIFIED
    return workflows, overall


def install_status(checks: list[Check]) -> str:
    relevant = [check for check in checks if check.required_for_install]
    if any(check.status == BLOCKED for check in relevant):
        return BLOCKED
    if any(check.status == UNVERIFIED for check in relevant):
        return UNVERIFIED
    return READY


def render_text(report: dict[str, Any]) -> str:
    lines = [
        "Cosmos Data Factory Orchestration preflight",
        f"Install readiness: {report['install_readiness']}",
        f"Run readiness: {report['run_readiness']}",
        "",
        "Checks:",
    ]
    for check in report["checks"]:
        lines.append(f"- [{check['status']}] {check['id']}: {check['summary']}")
    lines.extend(["", "Workflows:"])
    for workflow in report["workflows"]:
        lines.append(f"- [{workflow['status']}] {workflow['dag_id']} ({workflow['platform']})")
        for mode in workflow["modes"]:
            lines.append(f"  - [{mode['status']}] {mode['name']}")
            for blocker in mode["blockers"]:
                lines.append(f"    blocker: {blocker}")
            for unknown in mode["unverified"]:
                lines.append(f"    unverified: {unknown}")
            for warning in mode["warnings"]:
                lines.append(f"    warning: {warning}")
    lines.extend(["", "Quick start after install readiness is READY:"])
    prebuilt_supplied = report.get("prebuilt_controller_image") is not None
    prebuilt_image = report.get("prebuilt_controller_image")
    parsed_image = parse_prebuilt_controller_image(prebuilt_image) if prebuilt_image else None
    if prebuilt_supplied and not parsed_image:
        lines.extend(
            [
                "1. Correct --prebuilt-controller-image to a registry-qualified repository:tag.",
                "2. Rerun preflight; do not build, push, or install while the selected image reference is invalid.",
            ]
        )
    elif parsed_image:
        repository, tag = parsed_image
        lines.extend(
            [
                "1. Export the user-supplied KUBECONFIG and resolve storage blockers.",
                "2. Run: make setup",
                "3. Create a temporary copy of deploy/values.yaml for VALUES_DEV and set "
                f"airflow.defaultAirflowRepository={repository} and "
                f"airflow.defaultAirflowTag={tag}.",
                "4. Run: make install sdg-controller VALUES_DEV=/path/to/the/temporary/values.yaml",
                "5. Run: make port-forward, then rerun preflight with AIRFLOW_URL and AIRFLOW_TOKEN.",
            ]
        )
    else:
        lines.extend(
            [
                "1. Export the user-supplied KUBECONFIG and resolve storage blockers.",
                "2. Set a writable AIRFLOW_IMAGE_REPOSITORY and unique AIRFLOW_IMAGE_TAG in make/common.mk.",
                "3. Run: make setup",
                "4. Run: make build sdg-controller",
                "5. Run: make install sdg-controller",
                "6. Run: make port-forward, then rerun preflight with AIRFLOW_URL and AIRFLOW_TOKEN.",
            ]
        )
    return "\n".join(lines)


def resolve_kubeconfig(argument: str | None) -> Path | None:
    raw = argument or os.environ.get("KUBECONFIG", "").strip()
    if not raw:
        return None
    if os.pathsep in raw:
        return Path(raw)  # The audit will report this as a nonexistent single path.
    return Path(raw).expanduser()


def build_report(
    repo_root: Path,
    kubeconfig: Path | None,
    runner: Runner,
    prebuilt_controller_image: str | None = None,
    probe_pods: list[str] | None = None,
    confirmed_push_repository: str | None = None,
) -> dict[str, Any]:
    skill_root = Path(__file__).resolve().parents[1]
    profile_payload = json.loads(
        (skill_root / "config/readiness-profiles.json").read_text(encoding="utf-8")
    )
    profiles = profile_payload["profiles"]
    secrets = parse_env_presence(
        repo_root / "secrets.env",
        {name: os.environ.get(name, "") for name in SECRET_PRESENCE_NAMES},
    )
    checks = tool_checks(runner, controller_build_required=prebuilt_controller_image is None)
    checks.extend(
        repository_checks(
            repo_root,
            secrets,
            runner,
            prebuilt_controller_image=prebuilt_controller_image,
            confirmed_push_repository=confirmed_push_repository,
        )
    )
    kube_checks, kube = kubernetes_audit(runner, kubeconfig, probe_pods)
    checks.extend(kube_checks)
    checks.extend(storage_checks(repo_root, kube))
    controller_checks, controller = controller_audit(runner, kubeconfig, kube)
    checks.extend(controller_checks)
    checks.extend(airflow_audit(controller))
    workflows, run_readiness = workflow_report(repo_root, profiles, secrets, kube, controller)
    return {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "prebuilt_controller_image": prebuilt_controller_image,
        "install_readiness": install_status(checks),
        "run_readiness": run_readiness,
        "checks": [asdict(check) for check in checks],
        "workflows": workflows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="Run a read-only environment audit")
    default_repo = Path(__file__).resolve().parents[3]
    audit.add_argument("--repo-root", type=Path, default=default_repo)
    audit.add_argument(
        "--kubeconfig", help="Single kubeconfig path; KUBECONFIG is used when omitted"
    )
    audit.add_argument("--format", choices=("text", "json"), default="text")
    audit.add_argument(
        "--prebuilt-controller-image",
        help=(
            "Registry-qualified repository:tag for an already-published controller "
            "image; verifies manifest pull access and skips image-build/push prerequisites"
        ),
    )
    audit.add_argument(
        "--probe-pod",
        action="append",
        default=[],
        metavar="NAMESPACE/POD/CONTAINER",
        help="Existing ready non-hostNetwork Python 3 container; first probes API TCP, "
        "GPU containers also probe libnvcuvid. Repeat for each GPU node. Creates no pods.",
    )
    audit.add_argument(
        "--confirmed-push-repository",
        metavar="REGISTRY/REPOSITORY",
        help="User confirmation of push access with the current build credentials to this "
        "exact repository; no push is performed. Does not replace prebuilt manifest checks.",
    )
    audit.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Exit non-zero when either readiness gate is not READY",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    report = build_report(
        repo_root,
        resolve_kubeconfig(args.kubeconfig),
        Runner(),
        prebuilt_controller_image=args.prebuilt_controller_image,
        probe_pods=args.probe_pod,
        confirmed_push_repository=args.confirmed_push_repository,
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))
    if args.fail_on_blocked and (
        report["install_readiness"] != READY or report["run_readiness"] != READY
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
