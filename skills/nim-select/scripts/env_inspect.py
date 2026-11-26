#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""env_inspect: read-only inspection of a GPU host or a Kubernetes cluster for nim-select.

Builds the Environment object that `fit` reads: GPU groups (PCI id, memory, compute capability,
count, idle GPUs and, on a host, their indices and the GPUs whose memory use is unknown), driver,
CPU, memory, disk, Docker GPU access and the namespace GPU quota. A GPU that shares its host's
memory (the GB10 of DGX Spark, Jetson) gets that memory as its own: the host's total memory, or on
Kubernetes (GB10) the node's allocatable memory.
It only runs `nvidia-smi --query-gpu`, `docker context inspect`, `docker info`,
`kubectl config get-contexts -o name` (it reads the local kubeconfig only), `kubectl get nodes` and
`kubectl get resourcequota`, each with a timeout. It never pulls, runs or applies anything. The
three tools are found in the absolute folders of PATH (an empty or relative entry, such as `.`, is
skipped), never through another environment variable, and the host memory is read from
/proc/meminfo. Docker facts come only from a local daemon: DOCKER_HOST, DOCKER_CONTEXT
or the Docker config can name a remote one, whose GPUs are not this host's (docker_remote).

Prints one JSON object: {"ok", "state", "next", "facts", "errors": [{"code", "message", "fix"}]}.
Exit codes: 0 ok, 2 bad input, 3 precondition failed, 4 external tool failed, 5 refused.
Never prompts. Logs go to stderr. Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ns_common as ns

SCHEMA = "nim-select/environment/v1"
Runner = Callable[[list[str], float], tuple[int, str, str]]

SMI_TIMEOUT = 10.0
DOCKER_TIMEOUT = 10.0
KUBECTL_TIMEOUT = 20.0
IDLE_USED_FRACTION = 0.05  # NIM treats a GPU with at most 5% of its memory in use as free
# A GPU built into the system shares the host's memory, so nvidia-smi reports no memory size for
# it. A GPU with no memory size shares it only on evidence: its PCI id (the GB10 of DGX Spark), or
# a compute capability that only such GPUs have (8.7 Jetson Orin, 11.0 Jetson Thor, 12.1 GB10).
# Any other GPU with no memory size, such as an H100 whose nvidia-smi shows [N/A], has unknown
# memory, not the host's. On Kubernetes the PCI id alone says it (inspect_k8s): the GB10's
# nvidia.com/gpu.memory label is missing (GPU Feature Discovery leaves it out when NVML reports no
# memory), 0, or at most the host's total memory. The Grace superchips (GH200, GB200, GB300) have
# memory of their own and report it.
UNIFIED_MEMORY_GPUS = frozenset({"2e12:10de"})
UNIFIED_MEMORY_CAPABILITIES = frozenset({"8.7", "11.0", "12.1"})
SMI_FIELDS = (
    "index,pci.device_id,name,memory.total,memory.used,compute_cap,driver_version,mig.mode.current"
)
MEMINFO = Path("/proc/meminfo")
_MISSING = {"", "n/a", "[n/a]", "not supported", "[not supported]", "unknown", "[unknown]"}
_PCI_TABLE_PATH = Path(__file__).resolve().parent / "gpu_pci.json"
_GPU = "nvidia.com/gpu"
_SHARED = "nvidia.com/gpu.shared"
_MIG_PREFIX = "nvidia.com/mig-"
_PRODUCT = "nvidia.com/gpu.product"
_GPU_MEMORY = "nvidia.com/gpu.memory"
_HOSTNAME = "kubernetes.io/hostname"
# Labels that change over a node's life (health state, timestamps, kernel, driver and CUDA
# versions, GPU Feature Discovery facts): a pod pinned to one could not reschedule after it moves.
_VOLATILE_LABEL = re.compile(
    r"timestamp|nvsentinel|[-.]state$|^feature\.node\.kubernetes\.io/kernel-"
    r"|^nvidia\.com/(cuda\.|gpu\.driver\.|gfd\.)",
    re.I,
)
_POOL_LABEL = re.compile(r"(?:^|[/.-])(?:instance-type|node-?pool|pool)$", re.I)
_QUOTA_GPU = "requests.nvidia.com/gpu"


def default_runner(argv: list[str], timeout: float) -> tuple[int, str, str]:
    """Run a read-only command with no stdin; a timeout is exit code 124."""
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, "", f"{Path(argv[0]).name} timed out after {timeout:g} s"
    except OSError as exc:
        return 127, "", f"cannot run {argv[0]}: {exc.strerror}"
    return proc.returncode, proc.stdout, proc.stderr


def resolve_tool(name: str, environ: Mapping[str, str] | None = None) -> str | None:
    """The executable `name` found in an absolute folder of the PATH of `environ` (default:
    os.environ); None if absent. An empty or relative PATH entry (`::`, a trailing `:`, `.`)
    names the current folder, often a project someone else wrote, so it is skipped. No other
    variable can pick the program."""
    env = os.environ if environ is None else environ
    entries = env.get("PATH", os.defpath).split(os.pathsep)
    absolute = os.pathsep.join(entry for entry in entries if os.path.isabs(entry))
    return shutil.which(name, path=absolute) if absolute else None


def _first_line(*texts: str) -> str:
    for text in texts:
        for line in text.splitlines():
            if line.strip():
                return line.strip()[:300]
    return "no output"


def condition(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def gb(mib: int | None) -> float | None:
    """Decimal GB from MiB, cut (not rounded) to 2 decimals so it never overstates memory: the
    Metadata API's GB floors are decimal GB. 95830 MiB -> 100.48."""
    return None if mib is None else _gb_from_bytes(mib * 1048576)


def _gb_from_bytes(size: int) -> float:
    return (size // 10**7) / 100


def normalise_arch(machine: str | None) -> str | None:
    if not machine:
        return None
    value = machine.strip().lower()
    if value in {"x86_64", "amd64", "x64"}:
        return "amd64"
    if value in {"aarch64", "arm64", "armv8"}:
        return "arm64"
    return value


def normalise_product(name: str) -> str:
    """'NVIDIA-H100-NVL' and 'H100 NVL' both become 'h100nvl'."""
    value = name.strip().lower()
    if value.startswith("nvidia"):
        value = value[len("nvidia") :]
    return re.sub(r"[^a-z0-9]", "", value)


def load_pci_table(path: Path = _PCI_TABLE_PATH) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): [str(item) for item in value] for key, value in data.items()}


def pci_ids_for_product(product: str, table: Mapping[str, list[str]] | None = None) -> list[str]:
    lookup = load_pci_table() if table is None else table
    return list(lookup.get(normalise_product(product), []))


def captured_at() -> str:
    now = datetime.now(timezone.utc)  # noqa: UP017 (datetime.UTC needs Python 3.11)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


# --- host --------------------------------------------------------------------------------------


def _value(raw: str) -> str | None:
    text = raw.strip()
    return None if text.lower() in _MISSING else text


def _mib(raw: str) -> int | None:
    text = _value(raw)
    if text is None:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:MiB)?", text, flags=re.IGNORECASE)
    return int(float(match.group(1))) if match else None


def parse_pci_device_id(raw: str) -> str | None:
    """'0x232110DE' -> '2321:10de' (high 16 bits device, low 16 bits vendor)."""
    text = _value(raw)
    if text is None:
        return None
    try:
        number = int(text, 16)
    except ValueError:
        return None
    if not 0 < number <= 0xFFFFFFFF:
        return None
    return f"{number >> 16:04x}:{number & 0xFFFF:04x}"


def parse_smi_csv(text: str) -> list[dict[str, Any]]:
    """Rows of the nvidia-smi query (noheader; with or without units)."""
    rows: list[dict[str, Any]] = []
    for record in csv.reader(text.splitlines(), skipinitialspace=True):
        if not record or not "".join(record).strip():
            continue
        if len(record) < 8:
            raise ns.ScriptError(
                "nvidia_smi_failed",
                f"unexpected nvidia-smi output: {','.join(record)[:200]}",
                "Run `nvidia-smi` to check the driver, then inspect again",
                ns.EXIT_EXTERNAL,
            )
        index_text = _value(record[0])
        if index_text is None or not index_text.isdigit():
            continue
        mig = (_value(record[7]) or "").lower()
        rows.append(
            {
                "index": int(index_text),
                "pci_device_id": parse_pci_device_id(record[1]),
                "name": _value(record[2]),
                "memory_total_mib": _mib(record[3]),
                "memory_used_mib": _mib(record[4]),
                "compute_capability": _value(record[5]),
                "driver_version": _value(record[6]),
                "mig_enabled": mig == "enabled",
            }
        )
    return rows


def meminfo_total_kb(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^MemTotal:\s+(\d+)\s*kB", text, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def parse_gpu_selection(gpus: str) -> list[int] | None:
    """'all' -> None; '0,1' -> [0, 1]."""
    text = gpus.strip().lower()
    if text in {"", "all"}:
        return None
    try:
        picked = sorted({int(part) for part in text.split(",") if part.strip()})
    except ValueError:
        picked = []
    if not picked or any(index < 0 for index in picked):
        raise ns.ScriptError(
            "gpus_invalid",
            f"--gpus must be 'all' or a comma list of GPU indices, got {gpus!r}",
            "Pass --gpus all or --gpus 0,1 (indices from nvidia-smi)",
        )
    return picked


def _disk_free_gb(cache_dir: str | None) -> float | None:
    folder = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".cache" / "nim"
    probe = folder.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return _gb_from_bytes(shutil.disk_usage(probe).free)
    except OSError:
        return None


def _is_remote(endpoint: str) -> bool:
    return bool(endpoint) and not endpoint.startswith(("unix://", "npipe://"))


def _remote_docker(endpoint: str, where: str) -> dict[str, str]:
    scheme = endpoint.split("://", 1)[0] if "://" in endpoint else endpoint
    return condition(
        "docker_remote",
        f"{where} points to a remote daemon ({scheme}://); its GPUs are not the ones inspected "
        "here, so Docker facts are not reported",
    )


def docker_endpoint(path: str, runner: Runner) -> str | None:
    """The daemon endpoint the docker CLI uses, from `docker context inspect` of the current
    context, which DOCKER_HOST, DOCKER_CONTEXT and the currentContext of the Docker config pick;
    None when that command fails (a docker without contexts honours DOCKER_HOST alone)."""
    argv = [path, "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"]
    code, out, _ = runner(argv, DOCKER_TIMEOUT)
    lines = out.strip().splitlines()
    return lines[0].strip() if code == 0 and lines else None


def _docker_facts(
    path: str | None, runner: Runner, environ: Mapping[str, str], conditions: list[dict[str, str]]
) -> dict[str, bool] | None:
    host = environ.get("DOCKER_HOST", "").strip()
    if _is_remote(host):
        conditions.append(_remote_docker(host, "DOCKER_HOST"))
        return None
    if path is None:
        conditions.append(
            condition("docker_missing", "Docker is not installed or not on PATH on this host")
        )
        return {"present": False, "nvidia_runtime": False, "cdi": False}
    endpoint = docker_endpoint(path, runner)
    if endpoint is not None and _is_remote(endpoint):
        conditions.append(_remote_docker(endpoint, "The Docker context"))
        return None
    code, out, err = runner([path, "info", "--format", "{{json .}}"], DOCKER_TIMEOUT)
    info: Any = None
    if code == 0:
        try:
            info = json.loads(out)
        except json.JSONDecodeError:
            info = None
    if not isinstance(info, dict):
        conditions.append(
            condition(
                "docker_unreachable",
                f"`docker info` failed, so GPU access through Docker is unknown: "
                f"{_first_line(err, out)}",
            )
        )
        return {"present": True, "nvidia_runtime": False, "cdi": False}
    runtimes = info.get("Runtimes")
    nvidia_runtime = isinstance(runtimes, dict) and "nvidia" in runtimes
    devices = info.get("DiscoveredDevices")
    cdi = isinstance(devices, list) and any(
        isinstance(item, dict)
        and item.get("Source") == "cdi"
        and str(item.get("ID", "")).startswith("nvidia.com/gpu=")
        for item in devices
    )
    if not nvidia_runtime and not cdi:
        conditions.append(
            condition(
                "docker_no_gpu_access",
                "The Docker daemon reports no nvidia runtime and no nvidia.com/gpu CDI device",
            )
        )
    return {"present": True, "nvidia_runtime": nvidia_runtime, "cdi": cdi}


def _host_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["pci_device_id"],
            row["name"],
            row["memory_mib"],
            row["compute_capability"],
            row["unified_memory"],
        )
        group = groups.get(key)
        if group is None:
            pci = row["pci_device_id"]
            group = {
                "name": row["name"],
                "pci_device_id": pci,
                "pci_device_ids": [pci] if pci else [],
                "pci_source": "nvidia-smi" if pci else None,
                "memory_mib": row["memory_mib"],
                "memory_gb": gb(row["memory_mib"]),
                "unified_memory": row["unified_memory"],
                "compute_capability": row["compute_capability"],
                "count": 0,
                "idle": 0,
                # GPUs whose memory use nvidia-smi does not give: neither idle nor known busy
                "unknown_use": 0,
                "indices": [],
                "idle_indices": [],
                "node_count": 1,
                "driver_version": row["driver_version"],
            }
            groups[key] = group
        group["count"] += 1
        group["indices"].append(row["index"])
        if row["idle"]:
            group["idle"] += 1
            group["idle_indices"].append(row["index"])
        elif row["idle"] is None:
            group["unknown_use"] += 1
    return list(groups.values())


def inspect_host(
    gpus: str = "all",
    cache_dir: str | None = None,
    runner: Runner = default_runner,
    *,
    tools: Mapping[str, str | None] | None = None,
    meminfo: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Inspect this host. `tools` maps a tool name to its path (None: missing); unset tools are
    found on PATH. `meminfo` replaces MEMINFO (tests)."""
    env = os.environ if environ is None else environ

    def tool(name: str) -> str | None:
        if tools is not None and name in tools:
            return tools[name]
        return resolve_tool(name, env)

    meminfo_path = meminfo or MEMINFO
    selection = parse_gpu_selection(gpus)
    smi = tool("nvidia-smi")
    if smi is None:
        raise ns.ScriptError(
            "nvidia_smi_missing",
            "nvidia-smi was not found, so the GPUs of this host cannot be read",
            "Install the NVIDIA driver (it ships nvidia-smi), or run on the GPU host",
            ns.EXIT_PRECONDITION,
        )
    code, out, err = runner(
        [smi, f"--query-gpu={SMI_FIELDS}", "--format=csv,noheader,nounits"], SMI_TIMEOUT
    )
    if "no devices were found" in (out + err).lower():
        rows: list[dict[str, Any]] = []
    elif code != 0:
        raise ns.ScriptError(
            "nvidia_smi_failed",
            f"nvidia-smi failed (exit {code}): {_first_line(err, out)}",
            "Run `nvidia-smi` to check the driver, then inspect again",
            ns.EXIT_EXTERNAL,
        )
    else:
        rows = parse_smi_csv(out)
    if selection is not None:
        present = {row["index"] for row in rows}
        absent = [index for index in selection if index not in present]
        if absent:
            raise ns.ScriptError(
                "gpus_invalid",
                f"GPU index {absent} not found; this host has {sorted(present)}",
                "Pass --gpus all or indices that nvidia-smi lists",
            )
        rows = [row for row in rows if row["index"] in selection]
    if not rows:
        raise ns.ScriptError(
            "no_gpus",
            "nvidia-smi lists no NVIDIA GPU on this host",
            "Run on a host with an NVIDIA GPU, or inspect a Kubernetes cluster with --target k8s",
            ns.EXIT_PRECONDITION,
        )

    conditions: list[dict[str, str]] = []
    mem_kb = meminfo_total_kb(meminfo_path)
    unified: list[int] = []
    for row in rows:
        total = row["memory_total_mib"]
        row["unified_memory"] = total is None and (
            row["pci_device_id"] in UNIFIED_MEMORY_GPUS
            or row["compute_capability"] in UNIFIED_MEMORY_CAPABILITIES
        )
        if row["unified_memory"]:
            unified.append(row["index"])
            total = None if mem_kb is None else mem_kb // 1024
        row["memory_mib"] = total
        used = row["memory_used_mib"]
        if used is not None and total:
            row["idle"] = used / total <= IDLE_USED_FRACTION
        else:
            # No memory use to judge. A unified-memory GPU never reports one and counts as idle;
            # on any other GPU the use is unknown (None), so it does not count as free.
            row["idle"] = True if row["unified_memory"] else None
    if unified:
        conditions.append(
            condition(
                "unified_memory",
                "GPU memory is shared with the host; the VRAM check uses total system memory",
            )
        )
    gpus_known = True
    mig = [row["index"] for row in rows if row["mig_enabled"]]
    if mig:
        gpus_known = False
        conditions.append(
            condition(
                "mig_enabled",
                f"MIG is enabled on GPU {mig}; whole-GPU capacity is unknown. Pick GPUs without "
                "MIG with --gpus, or disable MIG",
            )
        )
    busy = [row["index"] for row in rows if row["idle"] is False]
    unmeasured = [row["index"] for row in rows if row["idle"] is None]
    if busy or unmeasured:
        why = [f"GPU {busy} already use more than 5% of their memory"] if busy else []
        if unmeasured:
            why.append(f"the memory use of GPU {unmeasured} is unknown")
        conditions.append(condition("gpus_busy", "; ".join(why) + "; only idle GPUs count as free"))
    groups = _host_groups(rows)
    if len(groups) > 1:
        conditions.append(
            condition(
                "mixed_gpus",
                f"This host has {len(groups)} GPU types; a NIM uses GPUs of one type, so each "
                "type is evaluated on its own",
            )
        )
    docker = _docker_facts(tool("docker"), runner, env, conditions)
    return {
        "schema": SCHEMA,
        "target": "host",
        "captured_at": captured_at(),
        "context": None,
        "namespace": None,
        "groups": groups,
        "gpus_known": gpus_known,
        "driver_version": rows[0]["driver_version"],
        "cpu_count": os.cpu_count(),
        "memory_gb": None if mem_kb is None else _gb_from_bytes(mem_kb * 1024),
        "arch": normalise_arch(platform.machine()),
        "disk_free_gb": _disk_free_gb(cache_dir),
        "gpu_quota": None,
        "docker": docker,
        "conditions": conditions,
    }


# --- Kubernetes --------------------------------------------------------------------------------

_SUFFIX: dict[str, Decimal] = {
    "Ki": Decimal(2**10),
    "Mi": Decimal(2**20),
    "Gi": Decimal(2**30),
    "Ti": Decimal(2**40),
    "Pi": Decimal(2**50),
    "Ei": Decimal(2**60),
    "n": Decimal("1e-9"),
    "u": Decimal("1e-6"),
    "m": Decimal("1e-3"),
    "k": Decimal(10**3),
    "M": Decimal(10**6),
    "G": Decimal(10**9),
    "T": Decimal(10**12),
    "P": Decimal(10**15),
    "E": Decimal(10**18),
}
_QUANTITY = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)([KMGTPE]i|[numkMGTPE])?$")


def parse_quantity(value: object) -> Decimal | None:
    """A Kubernetes quantity ('7900m', '2377633344Ki', '8') as a number, or None."""
    if isinstance(value, int | float):
        return Decimal(str(value))
    if not isinstance(value, str):
        return None
    match = _QUANTITY.fullmatch(value.strip())
    if match is None:
        return None
    try:
        number = Decimal(match.group(1))
    except InvalidOperation:
        return None
    suffix = match.group(2)
    return number * _SUFFIX[suffix] if suffix else number


def _whole(value: object) -> int | None:
    quantity = parse_quantity(value)
    if quantity is None or quantity != quantity.to_integral_value():
        return None
    return int(quantity)


def _positive(value: object) -> bool:
    quantity = parse_quantity(value)
    return quantity is not None and quantity > 0


def _mig_resource(allocatable: Mapping[str, Any]) -> bool:
    return any(key.startswith(_MIG_PREFIX) and _positive(v) for key, v in allocatable.items())


def _gpu_signal(labels: Mapping[str, str], allocatable: Mapping[str, Any]) -> bool:
    return (
        _positive(allocatable.get(_GPU))
        or _positive(allocatable.get(_SHARED))
        or _mig_resource(allocatable)
        or labels.get("nvidia.com/gpu.present") == "true"
    )


def is_partitioned(labels: Mapping[str, str], allocatable: Mapping[str, Any]) -> bool:
    """MIG or GPU sharing hides whole-GPU capacity. `nvidia.com/mig.strategy=single` alone does
    not: the GPU Operator sets it on nodes whose MIG configuration is all-disabled. Nor do the
    MIG slices of the mixed strategy (`nvidia.com/mig-<profile>`) on a node that also advertises
    whole GPUs as `nvidia.com/gpu`: those whole GPUs count, the slices do not."""
    product = labels.get(_PRODUCT, "")
    replicas = labels.get("nvidia.com/gpu.replicas", "").strip()
    sharing = labels.get("nvidia.com/gpu.sharing-strategy")
    return (
        (_mig_resource(allocatable) and not _positive(allocatable.get(_GPU)))
        or "-MIG-" in product.upper()
        or _positive(allocatable.get(_SHARED))
        or (replicas.isdigit() and int(replicas) > 1)
        or (sharing is not None and sharing != "none")
        or product.upper().endswith("-SHARED")
    )


def _ready(node: Mapping[str, Any]) -> str:
    for item in node.get("status", {}).get("conditions") or []:
        if isinstance(item, dict) and item.get("type") == "Ready":
            return str(item.get("status", "Unknown"))
    return "Unknown"


def _int_label(labels: Mapping[str, str], key: str) -> int | None:
    text = labels.get(key, "").strip()
    return int(text) if text.isdigit() else None


def _node_driver(labels: Mapping[str, str]) -> str | None:
    full = labels.get("nvidia.com/cuda.driver-version.full", "").strip()
    if full:
        return full
    parts = [labels.get(f"nvidia.com/cuda.driver.{p}", "").strip() for p in ("major", "minor")]
    rev = labels.get("nvidia.com/cuda.driver.rev", "").strip()
    if all(parts):
        return ".".join([*parts, rev] if rev else parts)
    return None


def _kubectl_json(path: str, args: list[str], runner: Runner) -> dict[str, Any]:
    code, out, err = runner([path, *args], KUBECTL_TIMEOUT)
    if code != 0:
        raise ns.ScriptError(
            "kubectl_failed",
            f"kubectl {' '.join(args[2:])} failed (exit {code}): {_first_line(err, out)}",
            "Show the error and ask the user to check the cluster and credentials of that context; "
            "never switch to another context",
            ns.EXIT_EXTERNAL,
        )
    try:
        data = json.loads(out)
    except json.JSONDecodeError as exc:
        raise ns.ScriptError(
            "kubectl_failed",
            f"kubectl {' '.join(args[2:])} did not return JSON: {exc.msg}",
            "Show the error and ask the user to check kubectl and that context (a kubectl plugin "
            "or wrapper can print other text); never switch to another context",
            ns.EXIT_EXTERNAL,
        ) from exc
    if not isinstance(data, dict):
        data = {}
    return data


NAMED_CONTEXTS = 10  # context names an error message names; facts.contexts has them all


def kube_contexts(path: str, runner: Runner) -> list[str]:
    """The context names of the kubeconfig, from `kubectl config get-contexts -o name`, which
    reads only the local kubeconfig and asks no cluster. A name is kept as kubectl prints it,
    spaces included (a context can be named ` spaced `); only line ends and empty lines go."""
    code, out, err = runner([path, "config", "get-contexts", "-o", "name"], KUBECTL_TIMEOUT)
    if code != 0:
        raise ns.ScriptError(
            "kubectl_failed",
            f"kubectl config get-contexts failed (exit {code}): {_first_line(err, out)}",
            "Show the error to the user: the kubeconfig could not be read",
            ns.EXIT_EXTERNAL,
        )
    names = (line.rstrip("\r") for line in out.split("\n"))
    return list(dict.fromkeys(name for name in names if name.strip()))


NO_KUBECONFIG_FIX = (
    "Ask the user which kubeconfig to use (the KUBECONFIG variable, else ~/.kube/config) and "
    "which context; never pick one"
)


def check_context(context: str, contexts: list[str]) -> None:
    """Raise context_not_found (exit 3) when the kubeconfig has no context `context`; its
    contexts go to facts.contexts, for the user to choose from. No context at all means kubectl
    read no kubeconfig with one, which naming a context cannot fix: the user names the
    kubeconfig."""
    if context in contexts:
        return
    if not contexts:
        raise ns.ScriptError(
            "context_not_found",
            f"kubectl found no context, so not {context!r} either: it read no kubeconfig, or one "
            "that lists no context (facts.contexts is empty)",
            NO_KUBECONFIG_FIX,
            ns.EXIT_PRECONDITION,
            {"contexts": contexts},
        )
    named = ", ".join(contexts[:NAMED_CONTEXTS])
    left = len(contexts) - NAMED_CONTEXTS
    raise ns.ScriptError(
        "context_not_found",
        f"the kubeconfig has no context {context!r}; it has {named}"
        f"{f' and {left} more' if left > 0 else ''} (facts.contexts)",
        "Ask the user which context to inspect; never pick one, not even the current context",
        ns.EXIT_PRECONDITION,
        {"contexts": contexts},
    )


def parse_quota(document: Mapping[str, Any]) -> tuple[dict[str, int] | None, bool]:
    """The most constraining unscoped GPU quota as {'hard', 'used'}, and whether a scoped quota
    limits GPUs too (it binds only the pods its scopes match)."""
    best: dict[str, int] | None = None
    scoped = False
    for item in document.get("items") or []:
        if not isinstance(item, dict):
            continue
        spec = item.get("spec") or {}
        status = item.get("status") or {}
        hard_map = status.get("hard") or spec.get("hard") or {}
        if _QUOTA_GPU not in hard_map:
            continue
        if spec.get("scopes") or spec.get("scopeSelector"):
            scoped = True
            continue
        hard = _whole(hard_map[_QUOTA_GPU])
        used = _whole((status.get("used") or {}).get(_QUOTA_GPU, "0"))
        if hard is None or used is None:
            continue
        if best is None or hard - used < best["hard"] - best["used"]:
            best = {"hard": hard, "used": used}
    return best, scoped


def _node_labels(node: Mapping[str, Any]) -> dict[str, str]:
    return {str(k): str(v) for k, v in (node.get("metadata", {}).get("labels") or {}).items()}


def find_placement(
    members: list[tuple[str, dict[str, str]]],
    others: list[tuple[str, dict[str, str]]],
    cluster: list[tuple[str, dict[str, str]]],
) -> dict[str, Any]:
    """How to keep a pod on the nodes of one GPU type. MEMBERS are the nodes of that type, OTHERS
    every GPU node of another or unknown type, including ineligible nodes; CLUSTER every node.

    Automatic keys have a placement contract: gpu.product, instance-type, node-pool or pool.
    A key must have one common value and match no other GPU type. Arbitrary labels can change
    during rollouts and are not inferred as placement. Without a suitable key, use the actual
    kubernetes.io/hostname label values, never assume they equal metadata.name. Missing or
    ambiguous hostname labels leave placement unavailable. Replacing nodes can break the pin.
    """
    common = dict(members[0][1])
    for _, labels in members[1:]:
        common = {k: v for k, v in common.items() if labels.get(k) == v}
    common.pop(_HOSTNAME, None)
    unique = sorted(
        (
            (k, v)
            for k, v in common.items()
            if (k == _PRODUCT or _POOL_LABEL.search(k))
            and not _VOLATILE_LABEL.search(k) and not any(o.get(k) == v for _, o in others)
        ),
        key=lambda kv: (
            kv[0] != _PRODUCT,
            kv[0] != "node.kubernetes.io/instance-type",
            not _POOL_LABEL.search(kv[0]),
            kv,
        ),
    )
    if unique:
        key, value = unique[0]
        matched = sorted(name for name, labels in cluster if labels.get(key) == value)
        return {"selector": {key: value}, "nodes_matched": matched, "source": "node-labels"}
    hostnames = sorted({labels.get(_HOSTNAME, "") for _, labels in members})
    if not all(hostnames) or any(labels.get(_HOSTNAME) in hostnames for _, labels in others):
        return {"selector": None, "nodes_matched": [], "source": "unavailable"}
    matched = sorted({
        name for name, labels in [*cluster, *members]
        if labels.get(_HOSTNAME) in hostnames
    })
    return {
        "selector": None,
        "hostnames": hostnames,
        "nodes_matched": matched,
        "source": "hostname-fallback",
    }


def inspect_k8s(
    context: str | None,
    namespace: str | None,
    runner: Runner = default_runner,
    *,
    tools: Mapping[str, str | None] | None = None,
    environ: Mapping[str, str] | None = None,
    pci_table: Mapping[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Inspect the GPU nodes of a Kubernetes context and the GPU quota of one namespace. The
    context must be one of the kubeconfig's (context_not_found), checked before any cluster
    call and before a missing namespace is reported, so the user is asked for both at once.

    GPU memory comes from the nvidia.com/gpu.memory label, except on a unified-memory GPU
    (UNIFIED_MEMORY_GPUS), which has no memory of its own. As a host gives such a GPU the
    host's total memory (nvidia-smi reports none), a node gives it its allocatable memory, the
    memory a pod can have, whatever the label says: missing, 0 or the node's total memory
    (condition unified_memory)."""
    if not context:
        raise ns.ScriptError(
            "context_required",
            "--context is required for --target k8s; the current kubectl context is never used",
            "Ask the user which context to inspect; never pick one, not even the current context",
        )
    env = os.environ if environ is None else environ
    if tools is not None and "kubectl" in tools:
        kubectl = tools["kubectl"]
    else:
        kubectl = resolve_tool("kubectl", env)
    if kubectl is None:
        raise ns.ScriptError(
            "kubectl_missing",
            "kubectl was not found",
            "Install kubectl and make sure it is on PATH",
            ns.EXIT_PRECONDITION,
        )
    check_context(context, kube_contexts(kubectl, runner))
    if not namespace:
        raise ns.ScriptError(
            "context_required",
            "--namespace is required for --target k8s",
            "Ask the user which namespace the NIM will run in",
        )
    nodes = _kubectl_json(kubectl, ["--context", context, "get", "nodes", "-o", "json"], runner)
    quotas = _kubectl_json(
        kubectl,
        ["--context", context, "-n", namespace, "get", "resourcequota", "-o", "json"],
        runner,
    )
    table = load_pci_table() if pci_table is None else pci_table

    counts = {
        "unschedulable": 0,
        "not_ready": 0,
        "partitioned": 0,
        "not_advertised": 0,
        "mig_mixed": 0,
    }
    no_labels = 0
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    cpus: list[int | None] = []
    memories: list[float | None] = []
    arches: set[str | None] = set()
    drivers: list[str | None] = []
    # Per GPU group: its own nodes' CPU counts, memory (GB, and MiB for a unified-memory GPU) and
    # CPU architectures. A cluster can mix Grace (arm64) and x86 GPU nodes, so the cluster-wide
    # values say nothing about one group.
    group_nodes: dict[tuple[Any, ...], tuple[list[Any], list[Any], list[Any], set[str | None]]] = {}
    cluster: list[tuple[str, dict[str, str]]] = []
    typed: dict[tuple[Any, Any], list[tuple[str, dict[str, str]]]] = {}
    all_typed: dict[tuple[Any, Any], list[tuple[str, dict[str, str]]]] = {}
    for node in nodes.get("items") or []:
        if not isinstance(node, dict):
            continue
        labels = _node_labels(node)
        name = str(node.get("metadata", {}).get("name") or "")
        if name:
            cluster.append((name, labels))
        allocatable = node.get("status", {}).get("allocatable") or {}
        product = labels.get(_PRODUCT, "").strip() or None
        ids = sorted(table.get(normalise_product(product), [])) if product else []
        unified = bool(ids) and set(ids) <= UNIFIED_MEMORY_GPUS
        # A unified-memory GPU gets its node's memory below; its label neither counts nor splits
        # its nodes into groups.
        memory = _int_label(labels, _GPU_MEMORY) if product and not unified else None
        gpu_signal = _gpu_signal(labels, allocatable)
        if name and (product or gpu_signal):
            # Eligibility can change. Exclude other/unknown GPU types even while cordoned,
            # unhealthy, partitioned or missing an advertised whole-GPU count.
            all_typed.setdefault((product, memory), []).append((name, labels))
        if not gpu_signal:
            continue
        ready = _ready(node)
        if (node.get("spec") or {}).get("unschedulable") or ready == "False":
            counts["unschedulable"] += 1
            continue
        if ready != "True":
            counts["not_ready"] += 1
            continue
        if is_partitioned(labels, allocatable):
            counts["partitioned"] += 1
            continue
        count = _whole(allocatable.get(_GPU))
        if count is None or count < 1:
            counts["not_advertised"] += 1
            continue
        if _mig_resource(allocatable):
            counts["mig_mixed"] += 1  # whole GPUs next to MIG slices: the slices are not counted
        major = _int_label(labels, "nvidia.com/gpu.compute.major")
        minor = _int_label(labels, "nvidia.com/gpu.compute.minor")
        cc = f"{major}.{minor}" if major is not None and minor is not None else None
        driver = _node_driver(labels)
        arch = normalise_arch(labels.get("kubernetes.io/arch"))
        cpu_q = parse_quantity(allocatable.get("cpu"))
        mem_q = parse_quantity(allocatable.get("memory"))
        cpus.append(None if cpu_q is None else int(cpu_q))
        memories.append(None if mem_q is None else _gb_from_bytes(int(mem_q)))
        node_mib = int(mem_q) // 1048576 if mem_q is not None and mem_q > 0 else None
        arches.add(arch)
        drivers.append(driver)
        if product is None:
            no_labels += 1
        elif name:
            typed.setdefault((product, memory), []).append((name, labels))
        # Nodes that differ only in allocatable CPUs or memory (often by a few MB) are one group;
        # it records the largest of each (below).
        key = (product, memory, cc, count, driver)
        group = groups.get(key)
        if group is None:
            group = {
                "name": product,
                "pci_device_id": ids[0] if len(ids) == 1 else None,
                "pci_device_ids": ids,
                "pci_source": "product-table" if ids else None,
                "memory_mib": memory,
                "memory_gb": gb(memory),
                "unified_memory": unified,
                "compute_capability": cc,
                "count": count,
                "idle": count,
                # The scheduler picks the GPUs of a pod: no device indices on Kubernetes.
                "indices": [],
                "idle_indices": [],
                "node_count": 0,
                "driver_version": driver,
            }
            groups[key] = group
        group["node_count"] += 1
        own_cpus, own_memories, own_mib, own_arches = group_nodes.setdefault(
            key, ([], [], [], set())
        )
        own_cpus.append(cpus[-1])
        own_memories.append(memories[-1])
        own_mib.append(node_mib)
        own_arches.add(arch)

    conditions: list[dict[str, str]] = []
    texts = {
        "unschedulable": (
            "nodes_unschedulable",
            "{n} GPU node(s) are cordoned or not Ready and are left out",
        ),
        "not_ready": (
            "nodes_not_ready",
            "{n} GPU node(s) report an unknown Ready state; their GPUs are unknown",
        ),
        "partitioned": (
            "gpus_partitioned",
            "{n} GPU node(s) use MIG or GPU sharing; their whole-GPU capacity is not counted",
        ),
        "not_advertised": (
            "gpus_not_advertised",
            "{n} GPU node(s) advertise no allocatable nvidia.com/gpu; check the device plugin",
        ),
        "mig_mixed": (
            "mig_mixed",
            "{n} GPU node(s) also have MIG slices; only their whole GPUs are counted",
        ),
    }
    for kind, number in counts.items():
        if number:
            reason, message = texts[kind]
            conditions.append(condition(reason, message.format(n=number)))
    if no_labels:
        conditions.append(
            condition(
                "gpu_labels_missing",
                f"{no_labels} GPU node(s) have no nvidia.com/gpu.product label (GPU Feature "
                "Discovery); their GPU model and memory are unknown",
            )
        )
    for key, (own_cpus, own_memories, own_mib, own_arches) in group_nodes.items():
        # A NIM pod needs one node, and one with room is enough: the group records
        # the largest CPU count and memory of its nodes, so a smaller node never hides a larger one,
        # and each node's two values as a pair, since the two largest can be on different nodes.
        # The smallest of each say whether every node has room: a pod that requests only GPUs can
        # be placed on any node with enough of them, of this group or another (the hand-off's
        # `placement`).
        group = groups[key]
        group.update(
            arch=next(iter(own_arches)) if len(own_arches) == 1 else None,
            cpu_count=_largest(own_cpus),
            host_memory_gb=_largest(own_memories),
            node_sizes=_node_sizes(own_cpus, own_memories),
            smallest_cpu_count=_minimum(own_cpus),
            smallest_host_memory_gb=_minimum(own_memories),
        )
        if group["unified_memory"]:
            # the smallest of its nodes, as the scheduler can place the pod on any of them
            pool = _minimum(own_mib)
            group.update(memory_mib=pool, memory_gb=gb(pool))
    group_list = list(groups.values())
    for group in group_list:
        type_key = (group["name"], None if group["unified_memory"] else group["memory_mib"])
        members = typed.get(type_key)
        if not members:
            continue
        others = [n for key, nodes_ in all_typed.items() if key != type_key for n in nodes_]
        group["placement"] = find_placement(members, others, cluster)
    inferred = sorted({g["name"] for g in group_list if g["pci_source"] == "product-table"})
    if inferred:
        conditions.append(
            condition(
                "pci_inferred",
                f"PCI ids were inferred from the product label of {inferred}; GPU Feature "
                "Discovery does not publish them",
            )
        )
    unknown = sorted({g["name"] for g in group_list if g["name"] and not g["pci_device_ids"]})
    if unknown:
        conditions.append(condition("pci_unknown", f"No PCI id is known for the product {unknown}"))
    shared = sorted({g["name"] for g in group_list if g["unified_memory"]})
    if shared:
        conditions.append(
            condition(
                "unified_memory",
                f"GPU memory of {shared} is shared with the node; the VRAM check uses the node's "
                "allocatable memory, not the nvidia.com/gpu.memory label",
            )
        )
    # A unified-memory GPU's memory is its node's, not part of the GPU type.
    kinds = {(g["name"], None if g["unified_memory"] else g["memory_mib"]) for g in group_list}
    if len(kinds) > 1:
        conditions.append(
            condition(
                "mixed_gpus",
                f"The cluster has {len(group_list)} GPU node groups; each is evaluated on its own",
            )
        )
    hidden = counts["not_ready"] + counts["partitioned"] + counts["not_advertised"]
    if not group_list and not hidden:
        # As on a host without GPUs: nothing to fit. Hidden GPUs (MIG, not Ready) ask the user.
        cordoned = counts["unschedulable"]
        raise ns.ScriptError(
            "no_gpus",
            f"context {context} has no schedulable GPU node"
            + (f" ({cordoned} GPU node(s) are cordoned or not Ready)" if cordoned else ""),
            "Ask the user whether this is the right context, or whether its cordoned or not-Ready "
            "GPU nodes can be brought back; inspect again with the context the user names",
            ns.EXIT_PRECONDITION,
        )
    quota, scoped = parse_quota(quotas)
    if scoped:
        conditions.append(
            condition(
                "gpu_quota_scoped",
                f"A scoped ResourceQuota in {namespace} limits GPUs for some pods only; it is "
                "not counted",
            )
        )
    known_drivers = [d for d in drivers if d]
    return {
        "schema": SCHEMA,
        "target": "k8s",
        "captured_at": captured_at(),
        "context": context,
        "namespace": namespace,
        "groups": group_list,
        "gpus_known": bool(group_list) or not hidden,
        "driver_version": (
            min(known_drivers, key=lambda d: ns.parse_version(d) or ()) if known_drivers else None
        ),
        "cpu_count": _minimum(cpus),
        "memory_gb": _minimum(memories),
        "arch": next(iter(arches)) if len(arches) == 1 else None,
        "disk_free_gb": None,
        "gpu_quota": quota,
        "docker": None,
        "conditions": conditions,
    }


def _minimum(values: list[Any]) -> Any:
    """The smallest value when every node reports one, else None."""
    if not values or any(value is None for value in values):
        return None
    return min(values)


def _largest(values: list[Any]) -> Any:
    """The largest value when every node reports one, else None."""
    if not values or any(value is None for value in values):
        return None
    return max(values)


def _node_sizes(cpus: list[Any], memories: list[Any]) -> list[dict[str, Any]] | None:
    """Each node's CPU count and memory as a pair, when every node reports both, else None. A node
    that another node matches or beats on both is left out: it adds no room."""
    if not cpus or any(value is None for value in [*cpus, *memories]):
        return None
    pairs = set(zip(cpus, memories))
    kept = [p for p in pairs if not any(q != p and q[0] >= p[0] and q[1] >= p[1] for q in pairs)]
    kept.sort(reverse=True)
    return [{"cpu_count": cpu, "host_memory_gb": memory} for cpu, memory in kept]


# --- CLI ---------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a GPU host or a Kubernetes cluster (read-only) for nim-select.",
        epilog=(
            'Example: python3 -I "<SKILL_DIR>/scripts/nimselect.py" inspect --target k8s '
            "--context prod --namespace nim --out ~/.cache/nim-select/env.json"
        ),
    )
    parser.add_argument("--target", choices=["host", "k8s"], default="host")
    parser.add_argument("--context", help="kubectl context (required for --target k8s)")
    parser.add_argument("--namespace", help="namespace (required for --target k8s)")
    parser.add_argument("--gpus", help="'all' (default) or GPU indices such as 0,1 (host only)")
    parser.add_argument("--cache-dir", help="NIM cache folder for the free-disk check (host)")
    parser.add_argument("--out", type=Path, help="also write the Environment object here")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    # The same checks, result and `next` as `nimselect.py inspect`, which this command runs.
    import nimselect

    nimselect.cmd_inspect(args)


if __name__ == "__main__":
    main()
