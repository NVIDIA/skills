# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/env_inspect.py and scripts/gpu_pci.json (no real GPUs, Docker or cluster)."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import env_inspect as ei
import fit
import ns_common as ns
import pytest
from ns_testkit import (
    LIGHTNING_ID,
    LOCAL_DOCKER,
    SCRIPTS,
    fake_docker,
    fixture,
    lightning_routes,
    on_path,
    run_cli,
    serve,
)

NO_DOCKER: dict[str, str | None] = {"nvidia-smi": "nvidia-smi", "docker": None}
WITH_DOCKER: dict[str, str | None] = {"nvidia-smi": "nvidia-smi", "docker": "docker"}
K8S: dict[str, str | None] = {"kubectl": "kubectl"}
EMPTY_QUOTA = '{"apiVersion": "v1", "items": [], "kind": "List"}'
# `kubectl config get-contexts -o name`: the contexts these tests name
CONTEXTS = "c\nctx\ndefault\n"


def text(name: str) -> str:
    return fixture("env", name).read_text(encoding="utf-8")


class FakeRunner:
    """Answers by tool and subcommand; records every argv. `docker context inspect` names the
    daemon `endpoint` (key "docker-context"); any other docker command answers `docker`.
    `kubectl config get-contexts` answers `contexts`, `kubectl get nodes` `nodes`, and any other
    kubectl command `quota`."""

    def __init__(
        self,
        smi: str = "",
        docker: str = "",
        nodes: str = "",
        quota: str = EMPTY_QUOTA,
        fail: dict[str, tuple[int, str, str]] | None = None,
        endpoint: str = LOCAL_DOCKER,
        contexts: str = CONTEXTS,
    ) -> None:
        self.answers = {
            "nvidia-smi": smi,
            "docker": docker,
            "docker-context": endpoint + "\n",
            "contexts": contexts,
            "nodes": nodes,
            "quota": quota,
        }
        self.fail = fail or {}
        self.calls: list[tuple[list[str], float]] = []

    def __call__(self, argv: list[str], timeout: float) -> tuple[int, str, str]:
        self.calls.append((argv, timeout))
        tool = Path(argv[0]).name
        key = tool
        if tool == "kubectl":
            key = "contexts" if argv[1:2] == ["config"] else "nodes" if "nodes" in argv else "quota"
        if tool == "docker" and argv[1:2] == ["context"]:
            key = "docker-context"
        if key in self.fail:
            return self.fail[key]
        return 0, self.answers[key], ""


def host(
    smi_file: str,
    *,
    meminfo: str = "meminfo-h100nvl.txt",
    tools: dict[str, str | None] = NO_DOCKER,
    gpus: str = "all",
    docker: str = "",
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    runner = FakeRunner(smi=text(smi_file), docker=docker)
    return ei.inspect_host(
        gpus,
        None,
        runner,
        tools=tools,
        meminfo=fixture("env", meminfo),
        environ=environ or {},
    )


def codes(env: dict[str, Any]) -> set[str]:
    return {item["code"] for item in env["conditions"]}


def h100_profile(**fields: Any) -> dict[str, Any]:
    """A profile built for the H100 NVL (one GPU, 80 GB), with `fields` added."""
    return {
        "profile_id": "h100-fp8",
        "precision": "fp8",
        "tensor_parallel_size": 1,
        "min_vram_per_device_gb": 80.0,
        "gpu_device": "2321:10de",
        **fields,
    }


def only_record(result: dict[str, Any]) -> dict[str, Any]:
    """The record of the one profile fit evaluated, whatever its verdict."""
    [record] = result["shortlist"] + result["unconfirmed"] + result["rejected"]["examples"]
    return record


# --- gpu_pci.json ------------------------------------------------------------------------------


def test_pci_table_is_sorted_lowercase_and_has_known_ids() -> None:
    raw = (SCRIPTS / "gpu_pci.json").read_text(encoding="utf-8")
    table = json.loads(raw)
    assert list(table) == sorted(table)
    for key, ids in table.items():
        assert key == ei.normalise_product(key)
        assert ids == sorted(ids)
        assert all(len(i) == 9 and i.endswith(":10de") and i == i.lower() for i in ids)
    assert ei.pci_ids_for_product("NVIDIA-H100-NVL") == ["2321:10de"]
    assert "2330:10de" in ei.pci_ids_for_product("NVIDIA-H100-80GB-HBM3")
    assert ei.pci_ids_for_product("NVIDIA-L40S") == ["26b9:10de"]
    assert ei.pci_ids_for_product("NVIDIA-NOT-A-GPU") == []


@pytest.mark.parametrize(
    ("product", "ids"),
    [
        # pci.ids (2026.11.09): 2bb3 "RTX PRO 5000 Blackwell / RTX PRO 5000 72GB Blackwell";
        # 2bb4 is the RTX PRO 6000 Blackwell Max-Q Workstation Edition
        ("NVIDIA-RTX-PRO-5000", ["2bb3:10de"]),
        ("NVIDIA-RTX-PRO-6000-Blackwell-Max-Q-Workstation-Edition", ["2bb4:10de"]),
        # NVIDIA's supported-gpus.json (driver 580): 1fb2 is the "NVIDIA T400", 1ff2 the "NVIDIA
        # T400 4GB" and "NVIDIA T400E" (pci.ids names 1fb2 only as the Quadro T400 Mobile)
        ("NVIDIA-T400", ["1fb2:10de"]),
        ("NVIDIA-T400-4GB", ["1ff2:10de"]),
        ("NVIDIA-T400E", ["1ff2:10de"]),
    ],
)
def test_pci_table_names_the_device_of_each_product(product: str, ids: list[str]) -> None:
    assert ei.pci_ids_for_product(product) == ids


@pytest.mark.parametrize(
    ("label", "key", "ids"),
    [
        # pci.ids (2026.11.09): 2e12 "GB20B [GB10]", the GPU of DGX Spark (also keyed dgxspark)
        ("NVIDIA-GB10", "gb10", ["2e12:10de"]),
        # 2c02 "GB203 [GeForce RTX 5080]"; 2c19 and 2c59 are the RTX 5080 Max-Q / Mobile
        ("NVIDIA-GeForce-RTX-5080", "geforcertx5080", ["2c02:10de"]),
        # 2c3a "GB203GL [RTX PRO 4500 Blackwell Server Edition]"
        (
            "NVIDIA-RTX-PRO-4500-Blackwell-Server-Edition",
            "rtxpro4500blackwellserveredition",
            ["2c3a:10de"],
        ),
        # 2c31 "GB203GL [RTX PRO 4500 Blackwell]"
        ("NVIDIA-RTX-PRO-4500-Blackwell", "rtxpro4500blackwell", ["2c31:10de"]),
        # 230c and 230e are both "GH100 [H20 NVL16]"
        ("NVIDIA-H20-NVL16", "h20nvl16", ["230c:10de", "230e:10de"]),
    ],
)
def test_kubernetes_product_labels_map_to_pci_ids(label: str, key: str, ids: list[str]) -> None:
    """The nvidia.com/gpu.product label of each GPU reaches its table key through
    normalise_product. A real cluster had the RTX 5080, RTX PRO 4500 Blackwell Server Edition and
    H20 NVL16 labels with no PCI id (pci_unknown); NVIDIA-GB10 is the label of a DGX Spark node."""
    assert ei.normalise_product(label) == key
    assert ei.load_pci_table()[key] == ids
    assert ei.pci_ids_for_product(label) == ids


def test_normalise_product() -> None:
    assert ei.normalise_product("NVIDIA-H100-NVL") == "h100nvl"
    assert ei.normalise_product("H100 NVL") == "h100nvl"
    assert ei.normalise_product("NVIDIA A100-SXM4-80GB") == "a100sxm480gb"


# --- host --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["nvidia-smi-h100nvl-8.csv", "nvidia-smi-h100nvl-8-units.csv"])
def test_real_h100_nvl_host(name: str) -> None:
    env = host(name, tools=WITH_DOCKER, docker=text("docker-info-h100nvl.json"))
    assert env["schema"] == "nim-select/environment/v1"
    assert env["target"] == "host" and env["gpus_known"] is True
    [group] = env["groups"]
    assert group["pci_device_id"] == "2321:10de"
    assert group["pci_device_ids"] == ["2321:10de"]
    assert group["pci_source"] == "nvidia-smi"
    assert group["name"] == "NVIDIA H100 NVL"
    assert group["memory_mib"] == 95830 and group["memory_gb"] == 100.48
    assert group["compute_capability"] == "9.0"
    assert (group["count"], group["idle"], group["indices"]) == (8, 8, list(range(8)))
    assert group["idle_indices"] == list(range(8))
    assert group["unified_memory"] is False
    assert env["driver_version"] == "580.126.09"
    assert env["memory_gb"] == 2434.69
    assert env["docker"] == {"present": True, "nvidia_runtime": True, "cdi": False}
    assert env["conditions"] == []
    json.dumps(env)


def test_smi_command_and_timeouts() -> None:
    runner = FakeRunner(smi=text("nvidia-smi-h100nvl-8.csv"), docker="{}")
    ei.inspect_host("all", None, runner, tools=WITH_DOCKER, meminfo=Path("/nonexistent"))
    (smi_argv, smi_timeout), (context_argv, context_timeout), (docker_argv, docker_timeout) = (
        runner.calls
    )
    assert smi_argv == [
        "nvidia-smi",
        "--query-gpu=index,pci.device_id,name,memory.total,memory.used,compute_cap,"
        "driver_version,mig.mode.current",
        "--format=csv,noheader,nounits",
    ]
    assert context_argv == [
        "docker",
        "context",
        "inspect",
        "--format",
        "{{.Endpoints.docker.Host}}",
    ]
    assert docker_argv == ["docker", "info", "--format", "{{json .}}"]
    assert smi_timeout == 10 and context_timeout == 10 and docker_timeout == 10


def test_pci_device_id_parsing() -> None:
    assert ei.parse_pci_device_id("0x232110DE") == "2321:10de"
    assert ei.parse_pci_device_id(" 0x2E1210DE ") == "2e12:10de"
    assert ei.parse_pci_device_id("[N/A]") is None
    assert ei.parse_pci_device_id("garbage") is None


def test_mig_enabled_hides_capacity() -> None:
    env = host("nvidia-smi-mig.csv")
    assert env["gpus_known"] is False
    assert "mig_enabled" in codes(env)
    only_whole = host("nvidia-smi-mig.csv", gpus="1")
    assert only_whole["gpus_known"] is True
    assert only_whole["groups"][0]["indices"] == [1]


def test_mixed_gpus_make_two_groups() -> None:
    env = host("nvidia-smi-mixed.csv")
    names = [(g["name"], g["pci_device_id"], g["count"]) for g in env["groups"]]
    assert names == [("NVIDIA H100 NVL", "2321:10de", 2), ("NVIDIA L40S", "26b9:10de", 1)]
    assert [g["idle_indices"] for g in env["groups"]] == [[0, 1], [2]]
    assert env["groups"][1]["memory_gb"] == 48.3
    assert "mixed_gpus" in codes(env)
    assert env["gpus_known"] is True  # [N/A] MIG mode is "not MIG capable", not enabled


def test_gb10_unified_memory_uses_meminfo() -> None:
    env = host("nvidia-smi-gb10.csv", meminfo="meminfo-gb10.txt")
    [group] = env["groups"]
    assert group["unified_memory"] is True
    assert group["memory_mib"] == 127817572 // 1024
    assert group["memory_gb"] == 130.88
    assert group["compute_capability"] == "12.1"
    assert group["pci_device_id"] == "2e12:10de"
    assert group["idle"] == 1
    assert "unified_memory" in codes(env)


def test_busy_gpu_is_not_idle() -> None:
    env = host("nvidia-smi-busy.csv")
    [group] = env["groups"]
    assert (group["count"], group["idle"]) == (8, 7)
    assert group["indices"] == list(range(8))
    assert group["idle_indices"] == [0, 1, 2, 4, 5, 6, 7]  # GPU 3 has 47 GB in use
    assert "gpus_busy" in codes(env)


def test_a_gpu_without_a_memory_size_is_not_taken_for_unified_memory(tmp_path: Path) -> None:
    """Only a GPU known to share the host's memory takes the host's memory as its VRAM. An H100
    NVL that nvidia-smi gives no memory size or use has unknown memory, so a VRAM floor stays
    unknown instead of passing on the host's 2.4 TB. Whether it is free is unknown too, so the
    profile is unknown, not rejected for want of GPUs."""
    smi = "0, 0x232110DE, NVIDIA H100 NVL, [N/A], [N/A], 9.0, 580.126.09, Disabled\n"
    meminfo = fixture("env", "meminfo-h100nvl.txt")
    runner = FakeRunner(smi=smi)
    env = ei.inspect_host(
        "all", str(tmp_path), runner, tools=NO_DOCKER, meminfo=meminfo, environ={}
    )
    [group] = env["groups"]
    assert group["unified_memory"] is False
    assert group["memory_mib"] is None and group["memory_gb"] is None
    assert (group["count"], group["idle"], group["unknown_use"]) == (1, 0, 1)
    assert "unified_memory" not in codes(env)
    result = fit.evaluate({"profiles": [h100_profile()]}, env, fit.FitOptions())
    assert result["verdict"] == "unknown"
    assert only_record(result)["reasons"] == ["vram_unknown", "gpu_use_unknown"]


def test_a_gpu_whose_memory_use_is_unknown_does_not_count_as_idle(tmp_path: Path) -> None:
    """nvidia-smi gives no memory use for GPU 1, so it is not counted as idle and the condition
    says why; GPU 0, measured at 0 MiB, is idle. A profile that needs GPU 1 too is unknown, not
    rejected; one that needs more GPUs than the host has is rejected. A GB10, which never reports
    its memory use, still counts as idle (test_gb10_unified_memory_uses_meminfo)."""
    smi = (
        "0, 0x232110DE, NVIDIA H100 NVL, 95830, 0, 9.0, 580.126.09, Disabled\n"
        "1, 0x232110DE, NVIDIA H100 NVL, 95830, [N/A], 9.0, 580.126.09, Disabled\n"
    )
    meminfo = fixture("env", "meminfo-h100nvl.txt")
    runner = FakeRunner(smi=smi)
    env = ei.inspect_host(
        "all", str(tmp_path), runner, tools=NO_DOCKER, meminfo=meminfo, environ={}
    )
    [group] = env["groups"]
    assert (group["count"], group["idle"], group["idle_indices"]) == (2, 1, [0])
    assert group["unknown_use"] == 1
    [busy] = [item for item in env["conditions"] if item["code"] == "gpus_busy"]
    assert busy["message"] == "the memory use of GPU [1] is unknown; only idle GPUs count as free"
    two_gpus = h100_profile(tensor_parallel_size=2)
    result = fit.evaluate({"profiles": [two_gpus]}, env, fit.FitOptions())
    assert (result["verdict"], result["group"]["available_gpus"]) == ("unknown", 1)
    record = only_record(result)
    assert record["reasons"] == ["gpu_use_unknown"]
    why = "Needs 2 GPUs on one node; 1 is available, and the memory use of 1 more is unknown"
    assert why in record["why"]
    three_gpus = h100_profile(tensor_parallel_size=3)
    result = fit.evaluate({"profiles": [three_gpus]}, env, fit.FitOptions())
    assert result["verdict"] == "does_not_fit"
    assert result["rejected"]["counts"] == {"not_enough_gpus": 1}


@pytest.mark.parametrize(
    "smi",
    [
        "0, [N/A], Orin (nvgpu), [N/A], [N/A], 8.7, 540.4.0, [N/A]\n",
        "0, [N/A], NVIDIA Thor, [N/A], [N/A], 11.0, 580.00, Disabled\n",
        "0, [N/A], NVIDIA GB10, [N/A], [N/A], 12.1, 580.95.05, [N/A]\n",
    ],
    ids=["jetson-orin", "jetson-thor", "gb10-without-pci-id"],
)
def test_a_gpu_built_into_the_system_shares_the_host_memory(smi: str, tmp_path: Path) -> None:
    """Jetson Orin and Thor give no memory size, as a GB10 does, and no PCI id to know them by.
    Their compute capability (8.7, 11.0, 12.1) is that of a GPU built into the system, so the
    host's memory is their VRAM and they count as idle."""
    meminfo = fixture("env", "meminfo-gb10.txt")
    runner = FakeRunner(smi=smi)
    env = ei.inspect_host(
        "all", str(tmp_path), runner, tools=NO_DOCKER, meminfo=meminfo, environ={}
    )
    [group] = env["groups"]
    assert group["unified_memory"] is True and group["memory_gb"] == 130.88
    assert (group["idle"], group["unknown_use"]) == (1, 0)
    assert codes(env) == {"unified_memory", "docker_missing"}


def test_gpu_selection() -> None:
    env = host("nvidia-smi-mixed.csv", gpus="2")
    assert [g["name"] for g in env["groups"]] == ["NVIDIA L40S"]
    with pytest.raises(ns.ScriptError) as bad:
        host("nvidia-smi-mixed.csv", gpus="0,9")
    assert bad.value.code == "gpus_invalid" and bad.value.exit_code == ns.EXIT_INPUT
    with pytest.raises(ns.ScriptError):
        ei.parse_gpu_selection("first")


def test_nvidia_smi_missing_failing_and_empty() -> None:
    with pytest.raises(ns.ScriptError) as missing:
        ei.inspect_host("all", None, FakeRunner(), tools={"nvidia-smi": None})
    assert (missing.value.code, missing.value.exit_code) == ("nvidia_smi_missing", 3)
    broken = FakeRunner(fail={"nvidia-smi": (9, "NVIDIA-SMI has failed because it couldn't", "")})
    with pytest.raises(ns.ScriptError) as failed:
        ei.inspect_host("all", None, broken, tools=NO_DOCKER)
    assert (failed.value.code, failed.value.exit_code) == ("nvidia_smi_failed", 4)
    assert "NVIDIA-SMI has failed" in failed.value.message
    timeout = FakeRunner(fail={"nvidia-smi": (124, "", "nvidia-smi timed out after 10 s")})
    with pytest.raises(ns.ScriptError) as slow:
        ei.inspect_host("all", None, timeout, tools=NO_DOCKER)
    assert slow.value.code == "nvidia_smi_failed"
    none = FakeRunner(fail={"nvidia-smi": (6, "No devices were found\n", "")})
    with pytest.raises(ns.ScriptError) as empty:
        ei.inspect_host("all", None, none, tools=NO_DOCKER)
    assert (empty.value.code, empty.value.exit_code) == ("no_gpus", 3)


def test_docker_missing_remote_and_unreachable() -> None:
    assert "docker_missing" in codes(host("nvidia-smi-h100nvl-8.csv"))
    remote = host(
        "nvidia-smi-h100nvl-8.csv",
        tools=WITH_DOCKER,
        environ={"DOCKER_HOST": "ssh://user@elsewhere"},
    )
    assert remote["docker"] is None and "docker_remote" in codes(remote)
    runner = FakeRunner(
        smi=text("nvidia-smi-h100nvl-8.csv"),
        fail={"docker": (1, "", "Cannot connect to the Docker daemon\n")},
    )
    env = ei.inspect_host("all", None, runner, tools=WITH_DOCKER, environ={})
    assert env["docker"] == {"present": True, "nvidia_runtime": False, "cdi": False}
    assert "docker_unreachable" in codes(env)


@pytest.mark.parametrize("endpoint", ["tcp://10.0.0.5:2376", "ssh://user@elsewhere"])
def test_a_remote_docker_context_is_not_inspected(endpoint: str) -> None:
    """DOCKER_CONTEXT, or the currentContext of the Docker config, names a remote daemon: its GPUs
    are not this host's, so `docker info` is not run and no Docker fact is reported."""
    runner = FakeRunner(smi=text("nvidia-smi-h100nvl-8.csv"), docker="{}", endpoint=endpoint)
    env = ei.inspect_host("all", None, runner, tools=WITH_DOCKER, environ={})
    assert env["docker"] is None and "docker_remote" in codes(env)
    assert [argv[1] for argv, _ in runner.calls if argv[0] == "docker"] == ["context"]


def test_a_docker_without_contexts_is_read_as_before() -> None:
    """`docker context inspect` fails (a docker older than contexts, or another CLI named docker):
    DOCKER_HOST alone picks the daemon, so a local one is read."""
    runner = FakeRunner(
        smi=text("nvidia-smi-h100nvl-8.csv"),
        docker=text("docker-info-h100nvl.json"),
        fail={"docker-context": (1, "", "docker: 'context' is not a docker command.\n")},
    )
    env = ei.inspect_host("all", None, runner, tools=WITH_DOCKER, environ={})
    assert env["docker"] == {"present": True, "nvidia_runtime": True, "cdi": False}
    assert "docker_remote" not in codes(env)


@pytest.mark.skipif(
    os.environ.get("NIM_SELECT_LIVE") != "1",
    reason="runs the real docker CLI; set NIM_SELECT_LIVE=1",
)
@pytest.mark.skipif(ei.resolve_tool("docker") is None, reason="needs the docker CLI on PATH")
def test_the_real_docker_cli_names_the_context_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The real docker CLI answers docker_endpoint's command for DOCKER_HOST, for DOCKER_CONTEXT
    and for the default context. It reads only the Docker config of tmp_path; no daemon is
    contacted. Opt-in, as the live tests are: the offline tests use fake runners and a fake
    docker only (test_a_remote_docker_context_is_not_inspected and the tests around it)."""
    docker = ei.resolve_tool("docker")
    assert docker is not None
    for name in ("DOCKER_HOST", "DOCKER_CONTEXT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
    created = subprocess.run(
        [docker, "context", "create", "remote", "--docker", "host=tcp://127.0.0.1:9"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    if created.returncode != 0:  # another CLI named docker (podman's), or one without contexts
        pytest.skip(f"this docker has no contexts: {created.stderr.strip()[:200]}")
    local = ei.docker_endpoint(docker, ei.default_runner)
    assert local is not None and local.startswith(("unix://", "npipe://")), local
    monkeypatch.setenv("DOCKER_CONTEXT", "remote")
    assert ei.docker_endpoint(docker, ei.default_runner) == "tcp://127.0.0.1:9"
    monkeypatch.setenv("DOCKER_HOST", "ssh://user@elsewhere")  # DOCKER_HOST wins over a context
    assert ei.docker_endpoint(docker, ei.default_runner) == "ssh://user@elsewhere"


def test_docker_cdi_device() -> None:
    info = {
        "Runtimes": {"runc": {}},
        "DiscoveredDevices": [{"Source": "cdi", "ID": "nvidia.com/gpu=0"}],
    }
    env = host("nvidia-smi-h100nvl-8.csv", tools=WITH_DOCKER, docker=json.dumps(info))
    assert env["docker"] == {"present": True, "nvidia_runtime": False, "cdi": True}
    no_gpu = host("nvidia-smi-h100nvl-8.csv", tools=WITH_DOCKER, docker='{"Runtimes": {}}')
    assert "docker_no_gpu_access" in codes(no_gpu)


def test_disk_free_uses_nearest_existing_parent(tmp_path: Path) -> None:
    runner = FakeRunner(smi=text("nvidia-smi-h100nvl-8.csv"))
    env = ei.inspect_host("all", str(tmp_path / "not" / "yet"), runner, tools=NO_DOCKER, environ={})
    assert isinstance(env["disk_free_gb"], float) and env["disk_free_gb"] >= 0


def test_arch_normalisation() -> None:
    assert ei.normalise_arch("x86_64") == "amd64"
    assert ei.normalise_arch("aarch64") == "arm64"
    assert ei.normalise_arch("ppc64le") == "ppc64le"
    assert ei.normalise_arch(None) is None


# --- Kubernetes --------------------------------------------------------------------------------


def k8s(nodes_file: str, quota: str = EMPTY_QUOTA) -> dict[str, Any]:
    runner = FakeRunner(nodes=text(nodes_file), quota=quota)
    return ei.inspect_k8s("ctx", "ns1", runner, tools=K8S)


def test_real_k3s_node_with_single_strategy_is_whole() -> None:
    runner = FakeRunner(nodes=text("kubectl-nodes-h100nvl.json"))
    env = ei.inspect_k8s("default", "default", runner, tools=K8S)
    assert [call[0] for call in runner.calls] == [
        ["kubectl", "config", "get-contexts", "-o", "name"],
        ["kubectl", "--context", "default", "get", "nodes", "-o", "json"],
        ["kubectl", "--context", "default", "-n", "default", "get", "resourcequota", "-o", "json"],
    ]
    assert all(call[1] == 20 for call in runner.calls)
    assert env["target"] == "k8s" and env["gpus_known"] is True
    [group] = env["groups"]
    assert group["name"] == "NVIDIA-H100-NVL"
    assert group["pci_device_id"] == "2321:10de" and group["pci_device_ids"] == ["2321:10de"]
    assert group["pci_source"] == "product-table"
    assert (group["count"], group["node_count"], group["indices"]) == (8, 1, [])
    assert group["idle_indices"] == []  # the scheduler picks a pod's GPUs
    assert group["memory_gb"] == 100.48 and group["compute_capability"] == "9.0"
    assert group["driver_version"] == "580.126.09" == env["driver_version"]
    assert env["cpu_count"] == 256 and env["arch"] == "amd64"
    assert env["memory_gb"] == 2434.69
    assert env["gpu_quota"] is None and env["docker"] is None
    assert codes(env) == {"pci_inferred"}


def node_like(
    base: dict[str, Any],
    name: str,
    product: str,
    arch: str,
    cpus: str,
    compute: tuple[str, str] = ("9", "0"),
    memory_mib: str = "95830",
    ram: str = "2377633344Ki",
) -> dict[str, Any]:
    """A copy of the k3s fixture node as another GPU node."""
    node: dict[str, Any] = json.loads(json.dumps(base))
    node["metadata"]["name"] = name
    node["metadata"]["labels"].update(
        {
            "kubernetes.io/hostname": name,
            "kubernetes.io/arch": arch,
            "beta.kubernetes.io/arch": arch,
            "nvidia.com/gpu.product": product,
            "nvidia.com/gpu.memory": memory_mib,
            "nvidia.com/gpu.compute.major": compute[0],
            "nvidia.com/gpu.compute.minor": compute[1],
        }
    )
    node["status"]["allocatable"].update({"cpu": cpus, "memory": ram, "nvidia.com/gpu": "4"})
    return node


def mixed_cluster() -> str:
    """x86 H100 NVL nodes next to Grace (arm64) GB300 nodes, and an L40S type whose own nodes mix
    both architectures."""
    base = json.loads(text("kubectl-nodes-h100nvl.json"))["items"][0]
    gb300 = {"compute": ("10", "3"), "memory_mib": "284208", "ram": "943718400Ki"}
    l40s = {"compute": ("8", "9"), "memory_mib": "46068"}
    items = [
        node_like(base, "x86-1", "NVIDIA-H100-NVL", "amd64", "256"),
        node_like(base, "x86-2", "NVIDIA-H100-NVL", "amd64", "128"),
        node_like(base, "grace-1", "NVIDIA-GB300", "arm64", "144", **gb300),
        node_like(base, "grace-2", "NVIDIA-GB300", "arm64", "144", **gb300),
        node_like(base, "odd-1", "NVIDIA-L40S", "amd64", "64", **l40s),
        node_like(base, "odd-2", "NVIDIA-L40S", "arm64", "64", **l40s),
    ]
    return json.dumps({"apiVersion": "v1", "kind": "List", "items": items})


def test_each_gpu_group_records_its_own_nodes_architecture_cpus_and_memory() -> None:
    env = ei.inspect_k8s("ctx", "ns1", FakeRunner(nodes=mixed_cluster()), tools=K8S)
    # cluster-wide: more than one architecture, and the smallest CPU count and memory
    assert (env["arch"], env["cpu_count"], env["memory_gb"]) == (None, 64, 966.36)
    groups = {g["name"]: g for g in env["groups"]}
    assert len(groups) == len(env["groups"]) == 3
    h100, gb300, l40s = groups["NVIDIA-H100-NVL"], groups["NVIDIA-GB300"], groups["NVIDIA-L40S"]
    # the two H100 NVL nodes have 256 and 128 CPUs: one group, with its largest node's
    assert (h100["arch"], h100["cpu_count"], h100["host_memory_gb"]) == ("amd64", 256, 2434.69)
    assert h100["node_count"] == 2
    assert (gb300["arch"], gb300["cpu_count"], gb300["host_memory_gb"]) == ("arm64", 144, 966.36)
    assert gb300["node_count"] == 2
    assert l40s["arch"] is None  # its own nodes mix amd64 and arm64


def test_a_gb300_profile_fits_on_a_cluster_that_also_has_x86_nodes() -> None:
    """The reported case: a GB300 profile on a cluster with x86 and Grace GPU nodes was always
    unknown (arch_unknown), as the cluster had no single CPU architecture."""
    env = ei.inspect_k8s("ctx", "ns1", FakeRunner(nodes=mixed_cluster()), tools=K8S)
    gb300 = {
        "profile_id": "gb300-nvfp4",
        "precision": "nvfp4",
        "inference_backend": "vllm",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "min_vram_per_device_gb": 85.0,
        "gpu_device": "31c2:10de",
        "tested_gpu_devices": ["31c2:10de"],
    }
    meta = {"release_details": {"architecture": ["AMD 64", "ARM 64"]}, "profiles": [gb300]}
    options = fit.FitOptions(image_architectures=["amd64", "arm64"], arch_required=True)
    result = fit.evaluate(meta, env, options)
    assert result["verdict"] == "fits", result
    [record] = result["shortlist"]
    assert record["profile_id"] == "gb300-nvfp4" and "arch_unknown" not in record["reasons"]


@pytest.mark.parametrize(
    ("cpus", "ram", "need"),
    [
        ("32", "2377633344Ki", {"min_cpu_count": 64}),
        ("128", "263555392Ki", {"system_memory_per_device_gb": 512.0}),
    ],
    ids=["cpus", "memory"],
)
def test_a_larger_node_is_not_hidden_by_a_smaller_one_of_the_same_gpu(
    cpus: str, ram: str, need: dict[str, Any]
) -> None:
    """The reported case: an H100 NVL node with fewer CPUs or less memory than the others gave
    their group its smallest values, so a profile the larger nodes can run was rejected. The group
    now records its largest node's: a NIM pod needs one node, and the scheduler places it on one
    with room."""
    base = json.loads(text("kubectl-nodes-h100nvl.json"))["items"][0]
    items = [
        node_like(base, "small", "NVIDIA-H100-NVL", "amd64", cpus, ram=ram),
        node_like(base, "large-1", "NVIDIA-H100-NVL", "amd64", "128"),
        node_like(base, "large-2", "NVIDIA-H100-NVL", "amd64", "128"),
    ]
    env = ei.inspect_k8s("ctx", "ns1", FakeRunner(nodes=json.dumps({"items": items})), tools=K8S)
    [group] = env["groups"]
    assert (group["node_count"], group["cpu_count"], group["host_memory_gb"]) == (3, 128, 2434.69)
    assert "mixed_gpus" not in codes(env)
    result = fit.evaluate({"profiles": [h100_profile(**need)]}, env, fit.FitOptions())
    assert result["verdict"] == "fits", result


def test_a_profile_needs_its_cpus_and_memory_on_one_node() -> None:
    """The reported case: one H100 NVL node has 64 CPUs and 137.43 GB, another 16 CPUs and
    549.75 GB. The group's largest CPU count and memory come from different nodes, so a profile
    that needs 32 CPUs and 256 GB passed, though no node has both. Each node's pair is kept now:
    the profile does not fit, and it fits once one node has both."""
    base = json.loads(text("kubectl-nodes-h100nvl.json"))["items"][0]
    items = [
        node_like(base, "cpus", "NVIDIA-H100-NVL", "amd64", "64", ram="134217728Ki"),
        node_like(base, "memory", "NVIDIA-H100-NVL", "amd64", "16", ram="536870912Ki"),
    ]
    needs = {"profiles": [h100_profile(min_cpu_count=32, system_memory_per_device_gb=256.0)]}
    env = ei.inspect_k8s("ctx", "ns1", FakeRunner(nodes=json.dumps({"items": items})), tools=K8S)
    [group] = env["groups"]
    assert (group["cpu_count"], group["host_memory_gb"]) == (64, 549.75)
    assert group["node_sizes"] == [
        {"cpu_count": 64, "host_memory_gb": 137.43},
        {"cpu_count": 16, "host_memory_gb": 549.75},
    ]
    result = fit.evaluate(needs, env, fit.FitOptions())
    assert result["verdict"] == "does_not_fit"
    assert result["rejected"]["counts"] == {"node_too_small": 1}
    assert (
        "Needs 32 CPU cores and 256 GB of host memory on one node; no NVIDIA-H100-NVL node has "
        "both (64 cores with 137.43 GB, 16 cores with 549.75 GB)"
    ) in only_record(result)["why"]
    items.append(node_like(base, "both", "NVIDIA-H100-NVL", "amd64", "64", ram="536870912Ki"))
    env = ei.inspect_k8s("ctx", "ns1", FakeRunner(nodes=json.dumps({"items": items})), tools=K8S)
    assert env["groups"][0]["node_sizes"] == [{"cpu_count": 64, "host_memory_gb": 549.75}]
    assert fit.evaluate(needs, env, fit.FitOptions())["verdict"] == "fits"


def test_a_group_also_records_its_smallest_cpus_and_memory() -> None:
    """The reported case: two H100 NVL nodes, one with 16 CPUs and 128 GB, one with 4 CPUs and
    32 GB. A profile that needs 8 CPUs and 64 GB fits on the larger, and node_sizes leaves the
    smaller out (it adds no room), but a pod that requests only GPUs can be placed on it. The
    group records the smallest CPU count and memory too, for the hand-off's placement."""
    base = json.loads(text("kubectl-nodes-h100nvl.json"))["items"][0]
    items = [
        node_like(base, "large", "NVIDIA-H100-NVL", "amd64", "16", ram="125000000Ki"),
        node_like(base, "small", "NVIDIA-H100-NVL", "amd64", "4", ram="31250000Ki"),
    ]
    env = ei.inspect_k8s("ctx", "ns1", FakeRunner(nodes=json.dumps({"items": items})), tools=K8S)
    [group] = env["groups"]
    assert (group["cpu_count"], group["host_memory_gb"]) == (16, 128.0)
    assert group["node_sizes"] == [{"cpu_count": 16, "host_memory_gb": 128.0}]
    assert (group["smallest_cpu_count"], group["smallest_host_memory_gb"]) == (4, 32.0)
    needs = {"profiles": [h100_profile(min_cpu_count=8, system_memory_per_device_gb=64.0)]}
    assert fit.evaluate(needs, env, fit.FitOptions())["verdict"] == "fits"


def test_gpu_nodes_that_differ_by_a_few_megabytes_are_one_gpu_type() -> None:
    """Nodes of one GPU model whose allocatable CPUs and memory differ a little (system
    reservations) are one group, so fit does not report them as several GPU types."""
    base = json.loads(text("kubectl-nodes-h100nvl.json"))["items"][0]
    sizes = (("256", "2377633344Ki"), ("255500m", "2377629248Ki"), ("256", "2377637440Ki"))
    items = [
        node_like(base, f"h100-{n}", "NVIDIA-H100-NVL", "amd64", cpus, ram=ram)
        for n, (cpus, ram) in enumerate(sizes)
    ]
    env = ei.inspect_k8s("ctx", "ns1", FakeRunner(nodes=json.dumps({"items": items})), tools=K8S)
    [group] = env["groups"]
    assert (group["node_count"], group["cpu_count"], group["host_memory_gb"]) == (3, 256, 2434.7)
    result = fit.evaluate({"profiles": [h100_profile()]}, env, fit.FitOptions())
    assert result["verdict"] == "fits"
    assert not any("GPU types" in note for note in result["notes"])


def test_partitioned_node_is_unknown() -> None:
    env = k8s("kubectl-nodes-mig.json")
    assert env["groups"] == [] and env["gpus_known"] is False
    assert "gpus_partitioned" in codes(env)


def test_a_mixed_mig_node_keeps_its_whole_gpus() -> None:
    """The mixed MIG strategy advertises the GPUs without MIG as nvidia.com/gpu next to the MIG
    slices (nvidia.com/mig-1g.10gb): the 4 whole GPUs count, the slices do not, and a condition
    says so. A node with slices only stays partitioned (test_partitioned_node_is_unknown)."""
    doc = json.loads(text("kubectl-nodes-mig.json"))
    [node] = doc["items"]
    node["metadata"]["labels"].pop("nvidia.com/mig.config")  # all-1g.10gb would put all in MIG
    node["status"]["allocatable"].update({"nvidia.com/gpu": "4", "nvidia.com/mig-1g.10gb": "28"})
    env = ei.inspect_k8s("ctx", "ns1", FakeRunner(nodes=json.dumps(doc)), tools=K8S)
    [group] = env["groups"]
    assert (group["name"], group["count"], group["node_count"]) == ("NVIDIA-A100-SXM4-80GB", 4, 1)
    assert group["pci_device_id"] == "20b2:10de" and group["memory_mib"] == 81920
    assert env["gpus_known"] is True and "gpus_partitioned" not in codes(env)
    [mixed] = [item for item in env["conditions"] if item["code"] == "mig_mixed"]
    assert mixed["message"] == (
        "1 GPU node(s) also have MIG slices; only their whole GPUs are counted"
    )


def test_partition_rules() -> None:
    whole = {"nvidia.com/gpu.product": "NVIDIA-H100-NVL", "nvidia.com/mig.strategy": "single"}
    assert ei.is_partitioned(whole, {"nvidia.com/gpu": "8"}) is False
    assert ei.is_partitioned(whole, {"nvidia.com/mig-1g.10gb": "7"}) is True
    mixed = {"nvidia.com/gpu.product": "NVIDIA-A100-SXM4-80GB", "nvidia.com/mig.strategy": "mixed"}
    assert (
        ei.is_partitioned(mixed, {"nvidia.com/gpu": "4", "nvidia.com/mig-1g.10gb": "28"}) is False
    )
    assert ei.is_partitioned(mixed, {"nvidia.com/gpu": "0", "nvidia.com/mig-1g.10gb": "56"}) is True
    assert ei.is_partitioned({"nvidia.com/gpu.product": "NVIDIA-A100-SXM4-40GB-MIG-1g.5gb"}, {})
    assert ei.is_partitioned({"nvidia.com/gpu.product": "NVIDIA-L4-SHARED"}, {})
    assert ei.is_partitioned({"nvidia.com/gpu.replicas": "4"}, {})
    assert ei.is_partitioned({"nvidia.com/gpu.sharing-strategy": "time-slicing"}, {})
    assert ei.is_partitioned({}, {"nvidia.com/gpu.shared": "16"})
    assert not ei.is_partitioned({"nvidia.com/gpu.replicas": "1"}, {})


def test_not_ready_nodes() -> None:
    env = k8s("kubectl-nodes-notready.json")
    assert env["groups"] == [] and env["gpus_known"] is False
    assert {"nodes_unschedulable", "nodes_not_ready"} <= codes(env)


def test_a_cluster_without_a_schedulable_gpu_node_is_no_gpus() -> None:
    """As on a host without GPUs, inspect stops: there is nothing to fit."""
    doc = json.loads(text("kubectl-nodes-cordoned.json"))
    cordoned = [item for item in doc["items"] if (item.get("spec") or {}).get("unschedulable")]
    assert cordoned, "the fixture has a cordoned GPU node"
    cpu_node = {
        "metadata": {"labels": {"kubernetes.io/arch": "amd64"}},
        "status": {
            "allocatable": {"cpu": "8"},
            "conditions": [{"type": "Ready", "status": "True"}],
        },
    }
    for items, extra in (([cpu_node], ""), ([cpu_node, *cordoned], "cordoned or not Ready")):
        runner = FakeRunner(nodes=json.dumps({"items": items}))
        with pytest.raises(ns.ScriptError) as info:
            ei.inspect_k8s("ctx", "ns1", runner, tools=K8S)
        assert (info.value.code, info.value.exit_code) == ("no_gpus", ns.EXIT_PRECONDITION)
        assert "ctx" in info.value.message and extra in info.value.message


def test_cordoned_node_is_left_out() -> None:
    env = k8s("kubectl-nodes-cordoned.json")
    [group] = env["groups"]
    assert (group["name"], group["count"], group["node_count"]) == ("NVIDIA-L40S", 4, 2)
    assert group["pci_device_id"] == "26b9:10de"
    assert group["driver_version"] == "570.124.06"
    assert env["cpu_count"] == 63 and env["gpus_known"] is True
    assert "nodes_unschedulable" in codes(env)


def test_node_without_gfd_labels() -> None:
    env = k8s("kubectl-nodes-no-gfd.json")
    [group] = env["groups"]
    assert group["name"] is None and group["count"] == 2
    assert group["pci_device_ids"] == [] and group["memory_gb"] is None
    assert "gpu_labels_missing" in codes(env)


def test_unknown_product_has_no_pci_ids() -> None:
    doc = json.loads(text("kubectl-nodes-cordoned.json"))
    for item in doc["items"]:
        labels = item["metadata"]["labels"]
        if "nvidia.com/gpu.product" in labels:
            labels["nvidia.com/gpu.product"] = "NVIDIA-FUTURE-GPU"
    env = ei.inspect_k8s("c", "n", FakeRunner(nodes=json.dumps(doc)), tools=K8S)
    [group] = env["groups"]
    assert group["pci_device_ids"] == [] and group["pci_device_id"] is None
    assert group["pci_source"] is None and "pci_unknown" in codes(env)


def test_ambiguous_product_keeps_all_ids() -> None:
    env = ei.inspect_k8s(
        "c",
        "n",
        FakeRunner(nodes=text("kubectl-nodes-cordoned.json")),
        tools=K8S,
        pci_table={"l40s": ["26b9:10de", "26ff:10de"]},
    )
    [group] = env["groups"]
    assert group["pci_device_ids"] == ["26b9:10de", "26ff:10de"]
    assert group["pci_device_id"] is None and group["pci_source"] == "product-table"


def spark_nodes(memory_label: str | None = None) -> dict[str, Any]:
    """kubectl-nodes-gb10.json: one DGX Spark node of a k3s cluster, labelled as GPU Feature
    Discovery labels a GB10, with the Spark's own figures (driver 580.95.05, 20 cores, and the
    MemTotal of meminfo-gb10.txt, less the kubelet's default 100Mi eviction threshold, as
    allocatable memory). NVML reports no GB10 memory (nvidia-smi's [N/A]), so GPU Feature
    Discovery publishes no nvidia.com/gpu.memory label; `memory_label` adds one with that value."""
    doc: dict[str, Any] = json.loads(text("kubectl-nodes-gb10.json"))
    if memory_label is not None:
        doc["items"][0]["metadata"]["labels"]["nvidia.com/gpu.memory"] = memory_label
    return doc


SPARK_MIB = 127715172 // 1024  # the node's allocatable memory, 127715172Ki
SPARK_NOTE = (
    "GPU memory of ['NVIDIA-GB10'] is shared with the node; the VRAM check uses the node's "
    "allocatable memory, not the nvidia.com/gpu.memory label"
)


def unified_note(env: dict[str, Any]) -> str:
    [message] = [item["message"] for item in env["conditions"] if item["code"] == "unified_memory"]
    return message


@pytest.mark.parametrize("memory_label", [None, "0", "4096"])
def test_shared_memory_placement_uses_the_gpu_type_key(
    memory_label: str | None,
) -> None:
    env = ei.inspect_k8s(
        "ctx", "ns1", FakeRunner(nodes=json.dumps(spark_nodes(memory_label))), tools=K8S
    )
    [group] = env["groups"]
    assert group["memory_mib"] == SPARK_MIB
    assert group.get("placement", {}).get("selector") == {
        "nvidia.com/gpu.product": "NVIDIA-GB10"
    }


def test_a_dgx_spark_node_gets_the_pci_id_of_its_gb10() -> None:
    """A DGX Spark node is labelled NVIDIA-GB10. The table gives it 2e12:10de, the id nvidia-smi
    reports on the Spark itself."""
    env = k8s("kubectl-nodes-gb10.json")
    [group] = env["groups"]
    assert (group["name"], group["compute_capability"]) == ("NVIDIA-GB10", "12.1")
    assert group["pci_device_id"] == "2e12:10de" and group["pci_device_ids"] == ["2e12:10de"]
    assert group["pci_source"] == "product-table" and "pci_inferred" in codes(env)
    [spark] = host("nvidia-smi-gb10.csv", meminfo="meminfo-gb10.txt")["groups"]
    assert (spark["pci_source"], spark["pci_device_ids"]) == ("nvidia-smi", group["pci_device_ids"])


@pytest.mark.parametrize(
    "memory_label",
    [None, "0", "4096", "124821"],
    ids=["no-label", "zero", "not-the-shared-memory", "total-memory"],
)
def test_a_gb10_on_kubernetes_has_its_node_allocatable_memory(memory_label: str | None) -> None:
    """The GB10 has no memory of its own. On the Spark, inspect gives it the host's total memory
    (nvidia-smi reports none); on Kubernetes its nvidia.com/gpu.memory label is missing, 0 or a
    value that is not the memory a pod can have, so it gets its node's allocatable memory,
    whatever the label says, and the condition unified_memory says so. A label of the whole
    memory (124821 MiB, MemTotal) is more than a pod can have. Before, a missing label made the
    GB10's memory unknown (every profile vram_unknown), and 0 made it 0 GB (vram_too_small)."""
    env = ei.inspect_k8s(
        "ctx", "ns1", FakeRunner(nodes=json.dumps(spark_nodes(memory_label))), tools=K8S
    )
    [group] = env["groups"]
    assert (group["name"], group["pci_device_id"]) == ("NVIDIA-GB10", "2e12:10de")
    assert group["unified_memory"] is True
    assert (group["memory_mib"], group["memory_gb"]) == (SPARK_MIB, 130.77)
    assert (group["host_memory_gb"], group["arch"], group["cpu_count"]) == (130.78, "arm64", 20)
    assert (group["count"], group["idle"], group["node_count"]) == (1, 1, 1)
    assert group["driver_version"] == "580.95.05" == env["driver_version"]
    assert codes(env) == {"pci_inferred", "unified_memory"} and env["gpus_known"] is True
    assert unified_note(env) == SPARK_NOTE
    # inspect on the Spark itself: the same GPU, unified, with the host's total memory
    spark = host("nvidia-smi-gb10.csv", meminfo="meminfo-gb10.txt")
    [own] = spark["groups"]
    assert (own["pci_device_id"], own["unified_memory"]) == ("2e12:10de", True)
    assert own["memory_mib"] == 127817572 // 1024 > group["memory_mib"]
    assert "unified_memory" in codes(spark)


def test_dgx_spark_nodes_are_one_gpu_type_whatever_their_memory() -> None:
    """Two Spark nodes with other labels (none, 0) and other allocatable memory make one group,
    with the smaller node's memory, as the scheduler can place the pod on either. A third with a
    newer driver is a group of its own, as for any GPU, but no other GPU type (mixed_gpus): a
    unified-memory GPU's memory is its node's."""
    first = spark_nodes()["items"][0]
    second: dict[str, Any] = json.loads(json.dumps(first))
    second["metadata"]["name"] = "spark-2"
    second["metadata"]["labels"].update(
        {"kubernetes.io/hostname": "spark-2", "nvidia.com/gpu.memory": "0"}
    )
    second["status"]["allocatable"]["memory"] = "120000000Ki"
    third: dict[str, Any] = json.loads(json.dumps(first))
    third["metadata"]["name"] = "spark-3"
    third["metadata"]["labels"].update(
        {"kubernetes.io/hostname": "spark-3", "nvidia.com/cuda.driver-version.full": "580.126.09"}
    )
    nodes = json.dumps({"apiVersion": "v1", "kind": "List", "items": [first, second, third]})
    env = ei.inspect_k8s("ctx", "ns1", FakeRunner(nodes=nodes), tools=K8S)
    pair, newer = env["groups"]
    assert (pair["node_count"], pair["driver_version"]) == (2, "580.95.05")
    # 120000000Ki: 117187 MiB, 122.87 GB as GPU memory (cut from MiB), 122.88 GB as host memory.
    # The group's host memory is its larger node's, and the smaller node's is its smallest.
    assert (pair["memory_mib"], pair["memory_gb"]) == (117187, 122.87)
    assert (pair["host_memory_gb"], pair["smallest_host_memory_gb"]) == (130.78, 122.88)
    assert (newer["node_count"], newer["driver_version"]) == (1, "580.126.09")
    assert (newer["memory_mib"], newer["memory_gb"]) == (SPARK_MIB, 130.77)
    assert pair["unified_memory"] is newer["unified_memory"] is True
    assert codes(env) == {"pci_inferred", "unified_memory"}
    assert unified_note(env) == SPARK_NOTE  # named once


def test_a_spark_node_without_allocatable_memory_leaves_the_gb10_memory_unknown() -> None:
    """A node that reports no allocatable memory gives the GB10 no memory: unknown (fit says
    vram_unknown), never the 0 of its label."""
    doc = spark_nodes("0")
    del doc["items"][0]["status"]["allocatable"]["memory"]
    env = ei.inspect_k8s("ctx", "ns1", FakeRunner(nodes=json.dumps(doc)), tools=K8S)
    [group] = env["groups"]
    assert group["unified_memory"] is True
    assert (group["memory_mib"], group["memory_gb"], group["host_memory_gb"]) == (None, None, None)


def test_only_the_gb10_takes_its_node_memory() -> None:
    """Beside a GB300 (a Grace superchip with memory of its own, on arm64 too) and an H100 NVL,
    only the GB10 gets its node's memory; the others keep their nvidia.com/gpu.memory label."""
    base = json.loads(text("kubectl-nodes-h100nvl.json"))["items"][0]
    gb300 = {"compute": ("10", "3"), "memory_mib": "284208", "ram": "943718400Ki"}
    items = [
        node_like(base, "x86-1", "NVIDIA-H100-NVL", "amd64", "256"),
        node_like(base, "grace-1", "NVIDIA-GB300", "arm64", "144", **gb300),
        spark_nodes()["items"][0],
    ]
    nodes = json.dumps({"apiVersion": "v1", "kind": "List", "items": items})
    env = ei.inspect_k8s("ctx", "ns1", FakeRunner(nodes=nodes), tools=K8S)
    memory = {
        g["name"]: (g["unified_memory"], g["memory_mib"], g["memory_gb"]) for g in env["groups"]
    }
    assert memory == {
        "NVIDIA-H100-NVL": (False, 95830, 100.48),
        "NVIDIA-GB300": (False, 284208, 298.01),
        "NVIDIA-GB10": (True, SPARK_MIB, 130.77),
    }
    assert codes(env) == {"pci_inferred", "unified_memory", "mixed_gpus"}
    assert unified_note(env) == SPARK_NOTE


def reason_codes(value: Any) -> set[str]:
    """Every reason code in a select result: those of each profile record, and those that count
    rejected profiles (rejected.counts, and rejected_counts of each image considered)."""
    if isinstance(value, list):
        return set().union(*(reason_codes(item) for item in value))
    if not isinstance(value, dict):
        return set()
    found: set[str] = set()
    if isinstance(value.get("reasons"), list):
        found.update(str(item) for item in value["reasons"])
    for key in ("counts", "rejected_counts"):
        if isinstance(value.get(key), dict):
            found.update(str(item) for item in value[key])
    return found.union(*(reason_codes(item) for item in value.values()))


@pytest.mark.parametrize("memory_label", [None, "0"], ids=["no-label", "zero"])
def test_select_on_a_dgx_spark_node_passes_the_vram_check(
    tmp_path: Path, memory_label: str | None
) -> None:
    """select nemotron-3.5-lightning-30b-a3b with the environment inspect wrote for the Spark
    node, and the metadata NGC published (served locally): its 1-GPU profiles (bf16 needs 66 GB,
    nvfp4 30 GB) pass the VRAM check on the node's 130.77 GB, and the image fits. Before, no
    profile fit: all were unknown (no label) or did not fit (0)."""
    nodes = json.dumps(spark_nodes(memory_label))
    env = ei.inspect_k8s("ctx", "ns1", FakeRunner(nodes=nodes), tools=K8S)
    env_file = tmp_path / "env.json"
    env_file.write_text(json.dumps(env), encoding="utf-8")
    with serve(lightning_routes("2.1.6", "2.1.5")) as server:
        code, payload = run_cli(
            "nimselect.py",
            "select",
            LIGHTNING_ID,
            "--env",
            str(env_file),
            env={"TEST_NGC_BASE": server.base, "NIM_SELECT_RETRY_DELAY": "0"},
        )
    assert code == 0, payload
    facts = payload["facts"]
    assert payload["state"] == "fits", payload
    assert not {reason for reason in reason_codes(facts) if reason.startswith("vram")}, facts
    assert facts["environment_summary"]["groups"][0]["memory_gb"] == 130.77
    recommended = facts["recommended"]
    assert (recommended["verdict"], recommended["gpus_required"]) == ("fits", 1)
    need = f"{recommended['min_vram_per_device_gb']:g}"
    assert f"Needs {need} GB per GPU; each GPU has 130.77 GB" in recommended["why"]
    assert facts["handoff"]["verdict"] == "fits"


def test_the_gpus_a_real_cluster_had_no_pci_id_for_have_one() -> None:
    """A real cluster left three GPU types with no PCI id (pci_unknown). Their labels, compute
    capability, GPU memory and GPU count are as that cluster reported them. H20 NVL16 has two
    ids in pci.ids, so its group keeps both and names no single id."""
    base = json.loads(text("kubectl-nodes-h100nvl.json"))["items"][0]
    seen = [  # product label, compute capability, GPU memory (MiB), GPUs per node
        ("NVIDIA-GeForce-RTX-5080", ("12", "0"), "16303", "1"),
        ("NVIDIA-RTX-PRO-4500-Blackwell-Server-Edition", ("12", "0"), "32623", "8"),
        ("NVIDIA-H20-NVL16", ("9", "0"), "122359", "16"),
    ]
    items = []
    for number, (product, compute, memory_mib, gpus) in enumerate(seen):
        node = node_like(base, f"node-{number}", product, "amd64", "64", compute, memory_mib)
        node["status"]["allocatable"]["nvidia.com/gpu"] = gpus
        items.append(node)
    nodes = json.dumps({"apiVersion": "v1", "kind": "List", "items": items})
    env = ei.inspect_k8s("ctx", "ns1", FakeRunner(nodes=nodes), tools=K8S)
    groups = {g["name"]: g for g in env["groups"]}
    assert {name: g["pci_device_ids"] for name, g in groups.items()} == {
        "NVIDIA-GeForce-RTX-5080": ["2c02:10de"],
        "NVIDIA-RTX-PRO-4500-Blackwell-Server-Edition": ["2c3a:10de"],
        "NVIDIA-H20-NVL16": ["230c:10de", "230e:10de"],
    }
    assert groups["NVIDIA-GeForce-RTX-5080"]["pci_device_id"] == "2c02:10de"
    assert groups["NVIDIA-RTX-PRO-4500-Blackwell-Server-Edition"]["pci_device_id"] == "2c3a:10de"
    assert groups["NVIDIA-H20-NVL16"]["pci_device_id"] is None
    assert {g["pci_source"] for g in groups.values()} == {"product-table"}
    assert codes(env) == {"pci_inferred", "mixed_gpus"}


def test_quota_most_constraining_unscoped() -> None:
    env = k8s("kubectl-nodes-h100nvl.json", quota=text("kubectl-resourcequota.json"))
    assert env["gpu_quota"] == {"hard": 4, "used": 1}
    assert "gpu_quota_scoped" in codes(env)
    assert ei.parse_quota({"items": []}) == (None, False)


def test_quantities() -> None:
    assert ei.parse_quantity("7900m") == Decimal("7.9")
    assert ei.parse_quantity("2Ki") == 2048
    assert ei.parse_quantity("1G") == 10**9
    assert ei.parse_quantity("bad") is None


def test_k8s_requires_context_and_namespace() -> None:
    for context, namespace in ((None, "ns"), ("ctx", None)):
        with pytest.raises(ns.ScriptError) as err:
            ei.inspect_k8s(context, namespace, FakeRunner(), tools=K8S)
        assert err.value.exit_code == ns.EXIT_INPUT


def test_a_named_context_is_checked_before_the_namespace() -> None:
    """A context named with no namespace is checked against the kubeconfig first: one it lacks
    stops with context_not_found and the contexts it has, so the user is asked for the context and
    the namespace at once; one it has asks for the namespace, after only the kubeconfig was read."""
    lacking = FakeRunner()
    with pytest.raises(ns.ScriptError) as not_found:
        ei.inspect_k8s("prod-gpu", None, lacking, tools=K8S)
    assert (not_found.value.code, not_found.value.exit_code, not_found.value.facts) == (
        "context_not_found",
        ns.EXIT_PRECONDITION,
        {"contexts": ["c", "ctx", "default"]},
    )
    found = FakeRunner()
    with pytest.raises(ns.ScriptError) as no_namespace:
        ei.inspect_k8s("ctx", None, found, tools=K8S)
    assert (no_namespace.value.code, no_namespace.value.exit_code) == (
        "context_required",
        ns.EXIT_INPUT,
    )
    for runner in (lacking, found):
        assert [call[0] for call in runner.calls] == [
            ["kubectl", "config", "get-contexts", "-o", "name"]
        ]


def test_kubectl_missing_and_failing() -> None:
    with pytest.raises(ns.ScriptError) as missing:
        ei.inspect_k8s("c", "n", FakeRunner(), tools={"kubectl": None})
    assert (missing.value.code, missing.value.exit_code) == ("kubectl_missing", 3)
    runner = FakeRunner(fail={"nodes": (1, "", 'error: context "c" does not exist\nmore\n')})
    with pytest.raises(ns.ScriptError) as failed:
        ei.inspect_k8s("c", "n", runner, tools=K8S)
    assert (failed.value.code, failed.value.exit_code) == ("kubectl_failed", 4)
    assert 'context "c" does not exist' in failed.value.message
    assert "more" not in failed.value.message
    unreadable = FakeRunner(fail={"contexts": (1, "", "error: error loading config file\n")})
    with pytest.raises(ns.ScriptError) as config:
        ei.inspect_k8s("c", "n", unreadable, tools=K8S)
    assert (config.value.code, config.value.exit_code) == ("kubectl_failed", 4)
    assert "config get-contexts" in config.value.message and len(unreadable.calls) == 1


@pytest.mark.parametrize(
    ("contexts", "named"),
    [
        ("default\n", "; it has default (facts.contexts)"),
        ("".join(f"ctx-{n:02d}\n" for n in range(12)), "ctx-09 and 2 more (facts.contexts)"),
    ],
    ids=["one", "many"],
)
def test_a_context_the_kubeconfig_lacks_asks_the_user(contexts: str, named: str) -> None:
    """A context the user names that the kubeconfig does not have stops inspect before any
    cluster call, with the contexts it has: the user picks one, never the agent, and never the
    current context (k3s has only `default`)."""
    runner = FakeRunner(nodes=text("kubectl-nodes-h100nvl.json"), contexts=contexts)
    with pytest.raises(ns.ScriptError) as info:
        ei.inspect_k8s("prod-gpu", "nim", runner, tools=K8S)
    error = info.value
    assert (error.code, error.exit_code) == ("context_not_found", ns.EXIT_PRECONDITION)
    assert error.message.startswith("the kubeconfig has no context 'prod-gpu'")
    assert error.message.endswith(named)
    assert error.facts == {"contexts": contexts.split()}
    assert error.fix == (
        "Ask the user which context to inspect; never pick one, not even the current context"
    )
    assert [call[0] for call in runner.calls] == [
        ["kubectl", "config", "get-contexts", "-o", "name"]
    ]


def test_no_context_at_all_asks_for_the_kubeconfig() -> None:
    """kubectl exits 0 and prints no context when it reads no kubeconfig (KUBECONFIG unset or
    naming a missing file, or k3s's root-only file): naming a context cannot fix that, so the user
    is asked which kubeconfig to use."""
    runner = FakeRunner(nodes=text("kubectl-nodes-h100nvl.json"), contexts="")
    with pytest.raises(ns.ScriptError) as info:
        ei.inspect_k8s("default", "nim", runner, tools=K8S)
    error = info.value
    assert (error.code, error.exit_code, error.facts) == (
        "context_not_found",
        ns.EXIT_PRECONDITION,
        {"contexts": []},
    )
    assert "it read no kubeconfig" in error.message and "'default'" in error.message
    assert error.fix.startswith("Ask the user which kubeconfig to use (the KUBECONFIG variable")
    assert len(runner.calls) == 1


def test_a_context_name_is_kept_as_kubectl_prints_it() -> None:
    """A context named with spaces around it (` spaced `) matches the name the user gives, spaces
    included; only line ends and empty lines are dropped."""
    contexts = " spaced \r\nprod-gpu\n\n"
    runner = FakeRunner(nodes=text("kubectl-nodes-h100nvl.json"), contexts=contexts)
    env = ei.inspect_k8s(" spaced ", "nim", runner, tools=K8S)
    assert env["context"] == " spaced "
    assert runner.calls[1][0][:3] == ["kubectl", "--context", " spaced "]
    with pytest.raises(ns.ScriptError) as info:
        ei.inspect_k8s("spaced", "nim", FakeRunner(contexts=contexts), tools=K8S)
    assert info.value.facts == {"contexts": [" spaced ", "prod-gpu"]}


# --- CLI ---------------------------------------------------------------------------------------


def fake_tool(folder: Path, name: str, body: str) -> str:
    path = folder / name
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


@pytest.mark.parametrize("direct", [True, False], ids=["direct", "runner"])
def test_cli_host(tmp_path: Path, direct: bool) -> None:
    """Started as SKILL.md starts it (direct) and through the test runner, which also applies the
    TEST_MEMINFO hook."""
    fake_tool(tmp_path, "nvidia-smi", f"cat '{fixture('env', 'nvidia-smi-h100nvl-8.csv')}'")
    fake_tool(tmp_path, "docker", fake_docker(fixture("env", "docker-info-h100nvl.json")))
    env = {"PATH": on_path(tmp_path), "DOCKER_HOST": ""}
    if not direct:
        env["TEST_MEMINFO"] = str(fixture("env", "meminfo-h100nvl.txt"))
    out = tmp_path / "env.json"
    code, payload = run_cli("env_inspect.py", "--out", str(out), env=env, direct=direct)
    assert code == 0 and payload["ok"] and payload["next"] == "run_fit"
    assert payload["facts"]["groups"][0]["pci_device_id"] == "2321:10de"
    assert payload["facts"]["docker"] == {"present": True, "nvidia_runtime": True, "cdi": False}
    if not direct:
        assert payload["facts"]["memory_gb"] == 2434.69  # the hook's file, not /proc/meminfo
    assert json.loads(out.read_text(encoding="utf-8")) == payload["facts"]


def test_cli_k8s(tmp_path: Path) -> None:
    nodes = fixture("env", "kubectl-nodes-mig.json")
    quota = fixture("env", "kubectl-resourcequota.json")
    body = (
        f"case \"$*\" in *get-contexts*) printf 'default\\nc\\n';; *nodes*) cat '{nodes}';; "
        f"*) cat '{quota}';; esac"
    )
    fake_tool(tmp_path, "kubectl", body)
    args = ("--target", "k8s", "--context", "c", "--namespace", "n")
    code, payload = run_cli("env_inspect.py", *args, env={"PATH": on_path(tmp_path)}, direct=True)
    assert code == 0 and payload["next"] == "ask_user"
    assert payload["facts"]["gpus_known"] is False
    assert payload["facts"]["gpu_quota"] == {"hard": 4, "used": 1}
    # a context the kubeconfig does not have: exit 3, ask the user, with the contexts it has
    args = ("--target", "k8s", "--context", "prod-gpu", "--namespace", "nim")
    code, payload = run_cli("env_inspect.py", *args, env={"PATH": on_path(tmp_path)}, direct=True)
    assert (code, payload["ok"], payload["next"]) == (3, False, "ask_user")
    assert payload["errors"][0]["code"] == "context_not_found"
    assert payload["facts"] == {"contexts": ["default", "c"]}


def test_cli_k8s_rejects_gpus(tmp_path: Path) -> None:
    args = ("--target", "k8s", "--context", "c", "--namespace", "n", "--gpus", "0,1")
    code, payload = run_cli("env_inspect.py", *args)
    assert code == 2 and payload["errors"][0]["code"] == "bad_input"


def test_cli_missing_context_exits_2(tmp_path: Path) -> None:
    code, payload = run_cli("env_inspect.py", "--target", "k8s", "--namespace", "n")
    assert code == 2 and payload["ok"] is False
    assert payload["errors"][0]["code"] == "context_required"


def test_cli_missing_nvidia_smi_exits_3(tmp_path: Path) -> None:
    code, payload = run_cli("env_inspect.py", env={"PATH": str(tmp_path)})
    assert code == 3 and payload["errors"][0]["code"] == "nvidia_smi_missing"


def test_cli_help_runs_on_any_python() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "env_inspect.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0 and "Example:" in proc.stdout


# --- placement: which labels select one GPU type --------------------------------------------


def node(name: str, **labels: str) -> tuple[str, dict[str, str]]:
    return name, {"kubernetes.io/hostname": name, **labels}


def test_any_label_only_the_gpu_type_carries_is_the_selector() -> None:
    members = [node("a1", pool="a", zone="x", gen="h"), node("a2", pool="a", zone="y", gen="h")]
    others = [node("b1", pool="b", zone="x", gen="h")]
    cluster = [*members, *others, node("cpu1", pool="a")]
    placement = ei.find_placement(members, others, cluster)
    assert placement == {
        "selector": {"pool": "a"},
        "nodes_matched": ["a1", "a2", "cpu1"],
        "source": "node-labels",
    }


def test_the_gpu_product_label_is_preferred_when_it_qualifies() -> None:
    members = [node("a1", pool="a", **{"nvidia.com/gpu.product": "L40S"})]
    others = [node("b1", pool="b", **{"nvidia.com/gpu.product": "H100"})]
    placement = ei.find_placement(members, others, [*members, *others])
    assert placement["selector"] == {"nvidia.com/gpu.product": "L40S"}


def test_a_label_the_other_gpu_type_shares_is_not_used() -> None:
    members = [node("a1", gen="h")]
    others = [node("b1", gen="h")]
    placement = ei.find_placement(members, others, [*members, *others])
    assert placement["source"] == "hostname-fallback"
    assert placement["hostnames"] == ["a1"] and placement["selector"] is None


def test_the_hostname_label_is_never_a_selector_even_for_one_node() -> None:
    placement = ei.find_placement([node("only")], [node("b1", gen="h")], [])
    assert placement["source"] == "hostname-fallback"


def test_inspected_groups_carry_their_placement() -> None:
    env = k8s("kubectl-nodes-cordoned.json")
    [group] = env["groups"]
    assert group["placement"]["selector"] == {"nvidia.com/gpu.product": "NVIDIA-L40S"}
    assert group["placement"]["source"] == "node-labels"
    assert group["placement"]["nodes_matched"]


def test_labels_that_change_over_a_nodes_life_are_never_the_selector() -> None:
    volatile = {
        "dgxc.nvidia.com/nvsentinel-state": "remediation-failed",
        "feature.node.kubernetes.io/kernel-version.minor": "14",
        "nvidia.com/gfd.timestamp": "1700000000",
        "nvidia.com/cuda.driver.major": "570",
        "nvidia.com/gpu.driver.version": "570.1",
        "nvsentinel.nvidia.com/uncordon-timestamp": "1700000001",
    }
    members = [node("a1", **volatile)]
    others = [node("b1", **{k: "other" for k in volatile})]
    placement = ei.find_placement(members, others, [*members, *others])
    assert placement["source"] == "hostname-fallback" and placement["hostnames"] == ["a1"]


def test_stable_pool_labels_beat_other_labels_and_the_product_label_beats_both() -> None:
    labels = {
        "aaa.example/zone": "z1",
        "nvca.nvcf.nvidia.io/instance-type": "gb300-x4",
        "feature.node.kubernetes.io/kernel-version.minor": "14",
    }
    members = [node("a1", **labels)]
    others = [node("b1", aaa_other="x")]
    placement = ei.find_placement(members, others, [*members, *others])
    assert placement["selector"] == {"nvca.nvcf.nvidia.io/instance-type": "gb300-x4"}
    with_product = [node("a1", **labels, **{"nvidia.com/gpu.product": "GB300"})]
    placement = ei.find_placement(with_product, others, [*with_product, *others])
    assert placement["selector"] == {"nvidia.com/gpu.product": "GB300"}


def test_review_rollout_labels_are_not_placement_contracts() -> None:
    members = [node("a1", **{"gpu.example.com/rollout": "r1"})]
    placement = ei.find_placement(members, [], members)
    assert placement["source"] == "hostname-fallback"


def test_review_hostname_fallback_uses_label_values_not_node_names() -> None:
    members = [("node-a1", {"kubernetes.io/hostname": "worker-a"})]
    placement = ei.find_placement(members, [], members)
    assert placement["hostnames"] == ["worker-a"]
    assert placement["nodes_matched"] == ["node-a1"]


@pytest.mark.parametrize(
    "excluded",
    ["cordoned", "not-ready", "partitioned", "unknown-type", "zero-capacity"],
)
def test_review_excluded_or_unknown_gpu_nodes_cannot_share_a_selector(
    excluded: str,
) -> None:
    base = json.loads(text("kubectl-nodes-h100nvl.json"))["items"][0]
    a = node_like(base, "a", "NVIDIA-H100-NVL", "amd64", "256", memory_mib="80000")
    b = node_like(base, "b", "NVIDIA-H100-NVL", "amd64", "256", memory_mib="40000")
    c = node_like(base, "c", "OTHER-GPU", "amd64", "256")
    for n, pool in ((a, "p"), (b, "q"), (c, "p")):
        n["metadata"]["labels"] = {
            k: v
            for k, v in n["metadata"]["labels"].items()
            if k.startswith("nvidia.com/") or k == "kubernetes.io/hostname"
        }
        n["metadata"]["labels"]["pool"] = pool
    if excluded == "cordoned":
        c["spec"] = {"unschedulable": True}
    elif excluded == "not-ready":
        c["status"]["conditions"] = [{"type": "Ready", "status": "False"}]
    elif excluded == "partitioned":
        c["metadata"]["labels"]["nvidia.com/gpu.sharing-strategy"] = "time-slicing"
    elif excluded == "unknown-type":
        c["metadata"]["labels"].pop("nvidia.com/gpu.product")
    else:
        c["metadata"]["labels"] = {
            k: v
            for k, v in c["metadata"]["labels"].items()
            if k
            in (
                "nvidia.com/gpu.product",
                "nvidia.com/gpu.memory",
                "pool",
                "kubernetes.io/hostname",
            )
        }
        c["status"]["allocatable"]["nvidia.com/gpu"] = "0"
    env = ei.inspect_k8s(
        "ctx", "ns1", FakeRunner(nodes=json.dumps({"items": [a, b, c]})), tools=K8S
    )
    group = next(g for g in env["groups"] if g["memory_mib"] == 80000)
    assert group["placement"]["source"] == "hostname-fallback"
    assert group["placement"]["nodes_matched"] == ["a"]
