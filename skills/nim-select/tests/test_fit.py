# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/fit.py: the fit evaluation and ranking on real Metadata API documents."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import fit
import ns_common as ns
import pytest
from ns_testkit import (
    LIGHTNING_BF16,
    LIGHTNING_NVFP4,
    LIGHTNING_NVFP4_LORA,
    SCRIPTS,
    fixture_json,
    run_cli,
)

H100_NVL = "2321:10de"
H100_SXM = "2330:10de"
GB10 = "2e12:10de"
LLAMA_8B = "metadata-llama-3.1-8b-instruct-2.1.4.json"
LLAMA_8B_OLD = "metadata-llama-3.1-8b-instruct-1.15.5.json"
LLAMA_70B = "metadata-llama-3.3-70b-instruct-latest.json"
NEMOTRON = "metadata-llama-3.3-nemotron-super-49b-v1-1.10.1.json"
# nemotron-3.5-lightning-30b-a3b:2.1.6 and 2.1.5 as NGC published them on 2026-11-18
LIGHTNING = "metadata-nemotron-3.5-lightning-30b-a3b-2.1.6.json"
LIGHTNING_215 = "metadata-nemotron-3.5-lightning-30b-a3b-2.1.5.json"
LIGHTNING_BF16_TP4 = "cd20534b2bb51e6aebdc98f62d3b16fa7addf820454a1ce04b3ba53e2ad6d710"
LIGHTNING_BF16_TP8 = "c4360e990fe29363cddc4588704948a60153060e878159bca67b247206613a5a"


def group(**overrides: Any) -> dict[str, Any]:
    """One GPU group in the Environment shape: 8 idle H100 NVL by default."""
    base: dict[str, Any] = {
        "name": "NVIDIA H100 NVL",
        "pci_device_id": H100_NVL,
        "pci_device_ids": [H100_NVL],
        "pci_source": "nvidia-smi",
        "memory_mib": 95830,
        "memory_gb": 100.48,
        "unified_memory": False,
        "compute_capability": "9.0",
        "count": 8,
        "idle": 8,
        "indices": list(range(8)),
        "node_count": 1,
        "driver_version": "580.126.09",
    }
    base.update(overrides)
    return base


def environment(*groups: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """An Environment object (section 4.3 shape) for a host."""
    env: dict[str, Any] = {
        "schema": "nim-select/environment/v1",
        "target": "host",
        "captured_at": "2026-10-23T00:00:00Z",
        "context": None,
        "namespace": None,
        "groups": list(groups) or [group()],
        "gpus_known": True,
        "driver_version": "580.126.09",
        "cpu_count": 64,
        "memory_gb": 1081.0,
        "arch": "amd64",
        "disk_free_gb": 500.0,
        "gpu_quota": None,
        "docker": {"present": True, "nvidia_runtime": True, "cdi": False},
        "conditions": [],
    }
    env.update(overrides)
    return env


def metadata(name: str) -> dict[str, Any]:
    data: dict[str, Any] = fixture_json("ngc", name)
    return data


def profile(profile_id: str = "p1", **fields: Any) -> dict[str, Any]:
    """A synthetic profile that fits an H100 NVL unless fields say otherwise."""
    base: dict[str, Any] = {
        "profile_id": profile_id,
        "precision": "bf16",
        "inference_backend": "vllm",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "min_vram_per_device_gb": 20.0,
        "feat_lora": False,
        "tested_gpu_devices": [H100_NVL],
    }
    base.update(fields)
    return {key: value for key, value in base.items() if value is not None}


def run(meta: dict[str, Any], env: dict[str, Any] | None = None, **options: Any) -> dict[str, Any]:
    return fit.evaluate(meta, env or environment(), fit.FitOptions(**options))


def by_id(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every profile record in a result (shortlist, unconfirmed, rejected examples)."""
    records = result["shortlist"] + result["unconfirmed"] + result["rejected"]["examples"]
    return {record["profile_id"]: record for record in records}


def all_records(meta: dict[str, Any], env: dict[str, Any], **options: Any) -> dict[str, Any]:
    """All records, with top and the example cap lifted."""
    result = run(meta, env, top=1000, **options)
    records = {r["profile_id"]: r for r in result["shortlist"] + result["unconfirmed"]}
    # Rejected examples are capped at five; evaluate each rejected profile alone to see it.
    for prof in meta["profiles"]:
        if prof["profile_id"] not in records:
            single = run({"profiles": [prof]}, env, **options)
            records[prof["profile_id"]] = single["rejected"]["examples"][0]
    return records


# --- the real documents -------------------------------------------------------------------------


def test_llama_8b_on_h100_nvl_recommends_tested_fp8() -> None:
    result = run(metadata(LLAMA_8B))
    assert result["verdict"] == "fits"
    rec = result["recommended"]
    assert rec["profile_id"].startswith("c4789f7a")
    assert (rec["precision"], rec["tier"], rec["feat_lora"], rec["gpus_required"]) == (
        "fp8",
        "tested",
        False,
        1,
    )
    assert "Tested on this GPU (2321:10de)" in rec["why"]
    assert "Needs 27 GB per GPU; each GPU has 100.48 GB" in rec["why"]
    assert "Needs 1 GPU; 8 are available" in rec["why"]
    assert [r["profile_id"][:8] for r in result["shortlist"]] == [
        "c4789f7a",
        "4d9fed9c",
        "092ed421",
        "82b90feb",
    ]
    assert result["rejected"]["counts"] == {"gpu_name_only": 2, "precision_unsupported": 2}
    assert result["profiles_total"] == 8
    assert "min_driver_version" in result["not_recorded"]
    assert "min_vram_per_device_gb" not in result["not_recorded"]
    assert result["group"]["available_gpus"] == 8
    assert result["groups"] == [
        {
            "name": "NVIDIA H100 NVL",
            "pci_device_id": H100_NVL,
            "pci_device_ids": [H100_NVL],
            "count": 8,
            "available_gpus": 8,
            "fits": 4,
            "unknown": 0,
        }
    ]
    assert result["best_unconfirmed"] is None


def test_nvfp4_agnostic_rejected_on_hopper_kept_on_blackwell() -> None:
    meta = metadata(LLAMA_8B)
    hopper = all_records(meta, environment())
    nvfp4 = [r for r in hopper.values() if r["precision"] == "nvfp4" and r["gpu"] is None]
    assert len(nvfp4) == 2
    for record in nvfp4:
        assert record["tier"] == "agnostic"
        assert record["reasons"] == ["precision_unsupported", "gpu_not_tested"]
        assert any("NVFP4 needs compute capability 10.0" in why for why in record["why"])
    assert any("NVFP4" in note for note in run(meta)["notes"])

    # A cc 10.0 GPU that is not in any tested list: the agnostic NVFP4 profiles stay, unconfirmed.
    blackwell = environment(group(pci_device_id="ffff:10de", pci_device_ids=["ffff:10de"],
                                  compute_capability="10.0", memory_gb=191.5))  # fmt: skip
    result = run(meta, blackwell)
    assert result["verdict"] == "unknown"
    assert result["recommended"] is None
    best = result["best_unconfirmed"]
    assert (best["precision"], best["tier"], best["reasons"]) == (
        "nvfp4",
        "agnostic",
        ["gpu_not_tested"],
    )
    assert "precision_unsupported" not in result["rejected"]["counts"]
    assert fit.UNTESTED_NOTE in result["notes"]


def test_h200_only_profile_rejected_on_h100() -> None:
    records = all_records(metadata(LLAMA_8B), environment())
    h200 = next(r for r in records.values() if r["gpu"] == "NVIDIA-H200")
    assert h200["verdict"] == "rejected"
    assert h200["reasons"] == ["gpu_name_only"]


def test_llama_70b_two_gpus() -> None:
    meta = metadata(LLAMA_70B)
    records = all_records(meta, environment(), max_gpus=2)
    result = run(meta, max_gpus=2)
    rec = result["recommended"]
    assert (rec["precision"], rec["tp"], rec["tier"]) == ("fp8", 2, "tested")
    assert rec["profile_id"].startswith("74dc099a")
    assert result["group"]["available_gpus"] == 2
    for record in records.values():
        if record["precision"] == "fp8" and record["tp"] in (4, 8):
            assert record["reasons"] == ["not_enough_gpus"]
            assert any("only 2 are available" in why for why in record["why"])
        if record["precision"] == "fp8" and record["tp"] == 1:
            # these two were tested on H200 and newer GPUs only
            assert record["reasons"] == ["vram_too_small", "gpu_not_tested"]
            assert "Needs 111 GB per GPU; each GPU has only 100.48 GB" in record["why"]


def test_llama_70b_eight_gpus() -> None:
    result = run(metadata(LLAMA_70B), max_gpus=8)
    rec = result["recommended"]
    assert (rec["precision"], rec["tp"], rec["tier"], rec["feat_lora"]) == (
        "fp8",
        8,
        "tested",
        False,
    )
    assert rec["profile_id"].startswith("a8ef77d8")
    assert [r["tp"] for r in result["shortlist"][:3]] == [8, 4, 2]


def test_nemotron_direct_match_without_vram_is_unconfirmed() -> None:
    result = run(metadata(NEMOTRON))
    assert result["verdict"] == "unknown"
    assert result["recommended"] is None
    best = result["best_unconfirmed"]
    assert best["tier"] == "direct"
    assert best["gpu"] == "h100_nvl"
    assert best["precision"] == "fp8"
    assert best["profile_id"].startswith("a00ce1e7")  # fp8, tp2 before tp1
    assert "vram_not_recorded" in best["reasons"]
    assert "Built for this GPU (2321:10de)" in best["why"]
    assert "min_vram_per_device_gb" in result["not_recorded"]


def test_gpu_device_style_document_on_l40s() -> None:
    l40s = group(name="NVIDIA L40S", pci_device_id="26b9:10de", pci_device_ids=["26b9:10de"],
                 memory_mib=46068, memory_gb=48.3, compute_capability="8.9", count=4, idle=4,
                 indices=[0, 1, 2, 3])  # fmt: skip
    result = run(metadata(LLAMA_8B_OLD), environment(l40s))
    assert result["verdict"] == "unknown"
    best = result["best_unconfirmed"]
    assert (best["tier"], best["gpu"], best["precision"], best["backend"]) == (
        "direct",
        "l40s",
        "fp8",
        "tensorrt_llm",
    )
    assert best["profile_id"].startswith("a4783dad")
    assert all(r["tier"] == "direct" for r in result["unconfirmed"][:4])
    assert result["rejected"]["counts"]["gpu_mismatch"] > 0


# --- targeting --------------------------------------------------------------------------------


def test_gpu_device_and_gpu_as_lists() -> None:
    meta = {
        "profiles": [
            profile("direct", gpu=["h100", "h100_nvl"], gpu_device=["2330:10de", "2321:10DE"],
                    tested_gpu_devices=[]),
            profile("by-name", gpu=["H100 NVL", "H100"], tested_gpu_devices=[]),
            profile("other", gpu="h200", gpu_device=["2335:10de"], tested_gpu_devices=[]),
        ]
    }  # fmt: skip
    records = all_records(meta, environment())
    assert records["direct"]["tier"] == "direct"
    assert records["direct"]["verdict"] == "fits"
    assert records["direct"]["gpu"] == ["h100", "h100_nvl"]
    assert records["by-name"]["reasons"] == ["gpu_name_only"]
    assert records["other"]["reasons"] == ["gpu_mismatch"]


def test_tested_string_and_agnostic_tiers() -> None:
    meta = {
        "profiles": [
            profile("tested", tested_gpu_devices=H100_NVL),
            profile("agnostic", tested_gpu_devices=[]),
            profile("elsewhere", tested_gpu_devices=["2901:10de"]),
        ]
    }
    records = all_records(meta, environment())
    assert records["tested"]["tier"] == "tested"
    assert records["agnostic"]["tier"] == "agnostic"
    assert records["elsewhere"]["tier"] == "agnostic"
    assert run(meta)["recommended"]["profile_id"] == "tested"


def test_profile_not_built_for_or_tested_on_the_gpu_is_unknown() -> None:
    meta = {
        "profiles": [
            profile("agnostic", tested_gpu_devices=[]),
            profile("elsewhere", tested_gpu_devices=["2901:10de", "2330:10de"]),
        ]
    }
    result = run(meta)
    assert (result["verdict"], result["recommended"]) == ("unknown", None)
    records = by_id(result)
    for pid in ("agnostic", "elsewhere"):
        assert (records[pid]["verdict"], records[pid]["reasons"]) == ("unknown", ["gpu_not_tested"])
    assert records["agnostic"]["why"][0] == (
        "GPU-agnostic profile not tested on this GPU (2321:10de); no tested GPU is recorded"
    )
    assert records["elsewhere"]["why"][0] == (
        "GPU-agnostic profile not tested on this GPU (2321:10de); tested on 2901:10de, 2330:10de"
    )
    assert fit.UNTESTED_NOTE in result["notes"]
    # The note explains an unknown verdict only: with a tested profile that fits, it is left out.
    tested = run({"profiles": [*meta["profiles"], profile("tested")]})
    assert tested["recommended"]["profile_id"] == "tested"
    assert fit.UNTESTED_NOTE not in tested["notes"]


def test_product_table_match_adds_note_on_kubernetes() -> None:
    node = group(name="NVIDIA-H100-NVL", pci_source="product-table", idle=None, indices=[])
    env = environment(node, target="k8s", context="default", namespace="nim")
    result = run(metadata(LLAMA_8B), env)
    assert result["recommended"]["tier"] == "tested"
    assert any("inferred from the product name" in why for why in result["recommended"]["why"])
    assert "pci_inferred" in result["recommended"]["reasons"]
    assert any("inferred from the product name" in note for note in result["notes"])
    assert any("Kubernetes" in note for note in result["notes"])
    assert result["group"]["available_gpus"] == 8


def test_unknown_pci_id_on_kubernetes() -> None:
    node = group(name="NVIDIA-FUTURE-GPU", pci_device_id=None, pci_device_ids=[],
                 pci_source=None, idle=None)  # fmt: skip
    env = environment(node, target="k8s", context="default", namespace="nim")
    meta = {
        "profiles": [
            profile("named", gpu="h100_nvl", gpu_device=H100_NVL, tested_gpu_devices=[]),
            profile("agnostic", tested_gpu_devices=[H100_NVL]),
        ]
    }
    records = all_records(meta, env)
    assert records["named"]["verdict"] == "unknown"
    assert records["named"]["reasons"] == ["pci_unknown"]
    # Without a PCI id, the tested list cannot be checked either.
    assert records["agnostic"]["verdict"] == "unknown"
    assert records["agnostic"]["tier"] == "agnostic"
    assert records["agnostic"]["reasons"] == ["pci_unknown"]


# --- precision ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("precision", "cc", "passes"),
    [
        ("fp8", "8.0", False),
        ("fp8", "8.9", True),
        ("nvfp4", "9.0", False),
        ("nvfp4", "10.0", True),
        ("int4_awq", "10.0", False),
        ("int4_awq", "12.0", True),
        ("int4_awq", "9.0", True),
        ("bf16", "7.0", True),
    ],
)
def test_precision_rules_for_agnostic_profiles(precision: str, cc: str, passes: bool) -> None:
    meta = {"profiles": [profile(precision=precision, tested_gpu_devices=[])]}
    record = all_records(meta, environment(group(compute_capability=cc)))["p1"]
    # Untested on this GPU, an agnostic profile is unknown at best; the precision rule rejects.
    if passes:
        assert (record["verdict"], record["reasons"]) == ("unknown", ["gpu_not_tested"])
    else:
        assert record["verdict"] == "rejected"
        assert record["reasons"] == ["precision_unsupported", "gpu_not_tested"]


def test_precision_not_gated_for_tested_profile_and_unknown_capability_is_unknown() -> None:
    tested = {"profiles": [profile(precision="fp8")]}
    assert run(tested, environment(group(compute_capability="8.0")))["verdict"] == "fits"
    agnostic = {"profiles": [profile(precision="fp8", tested_gpu_devices=[])]}
    result = run(agnostic, environment(group(compute_capability=None)))
    assert (result["verdict"], result["recommended"]) == ("unknown", None)
    assert result["best_unconfirmed"]["reasons"] == ["gpu_not_tested", "capability_unknown"]
    assert any("Compute capability is unknown" in note for note in result["notes"])


# --- requirements ------------------------------------------------------------------------------


def test_absent_fields_never_pass() -> None:
    bare = {"profiles": [{"profile_id": "bare", "tested_gpu_devices": []}]}
    result = run(bare, need_lora=True, backend="vllm")
    assert result["verdict"] == "unknown"
    assert result["recommended"] is None
    reasons = result["best_unconfirmed"]["reasons"]
    assert {"vram_not_recorded", "lora_not_recorded", "backend_not_recorded"} <= set(reasons)
    assert result["not_recorded"] == sorted(fit.REQUIREMENT_FIELDS)

    unknown_env = environment(group(memory_gb=None), driver_version=None, cpu_count=None,
                              memory_gb=None, disk_free_gb=None, arch=None)  # fmt: skip
    meta = {
        "profiles": [
            profile(min_driver_version="550.0", min_cpu_count=8, system_memory_per_device_gb=16,
                    disk_space_per_device_gb=40, release_details={"architecture": ["AMD 64"]})
        ]
    }  # fmt: skip
    record = run(meta, unknown_env)["best_unconfirmed"]
    assert record["verdict"] == "unknown"
    assert set(record["reasons"]) == {
        "vram_unknown",
        "driver_unknown",
        "cpu_unknown",
        "memory_unknown",
        "disk_unknown",
        "arch_unknown",
    }


def test_driver_too_old_and_new_enough() -> None:
    old = {"profiles": [profile(min_driver_version="590.10")]}
    record = run(old)["rejected"]["examples"][0]
    assert record["reasons"] == ["driver_too_old"]
    assert "Needs driver 590.10 or newer; this host has 580.126.09" in record["why"]
    ok = {"profiles": [profile(min_driver_version="580.65.06")]}
    assert run(ok)["verdict"] == "fits"
    assert "min_driver_version" not in run(ok)["not_recorded"]


def test_max_gpu_count() -> None:
    meta = {"profiles": [profile(tensor_parallel_size=4, max_gpu_count=2)]}
    record = run(meta)["rejected"]["examples"][0]
    assert record["reasons"] == ["max_gpu_count"]
    allowed = {"profiles": [profile(tensor_parallel_size=4, max_gpu_count=8)]}
    assert run(allowed)["verdict"] == "fits"


def test_pipeline_parallel_multiplies_gpus() -> None:
    meta = {"profiles": [profile(tensor_parallel_size=4, pipeline_parallel_size=2)]}
    assert run(meta)["recommended"]["gpus_required"] == 8
    assert run(meta, max_gpus=4)["rejected"]["counts"] == {"not_enough_gpus": 1}


def test_busy_gpus_and_quota_limit_available() -> None:
    meta = {"profiles": [profile(tensor_parallel_size=4)]}
    assert run(meta, environment(group(idle=3)))["verdict"] == "does_not_fit"
    quota = environment(gpu_quota={"hard": 4, "used": 2})
    result = run(meta, quota)
    assert result["group"]["available_gpus"] == 2
    assert result["rejected"]["counts"] == {"not_enough_gpus": 1}


def test_a_profile_that_needs_gpus_of_unknown_use_is_unknown() -> None:
    """6 GPUs are idle and the memory use of 2 is unknown: a profile that needs 8 is unknown, not
    rejected. One that needs more than that, or more than --max-gpus allows, is rejected."""
    eight = {"profiles": [profile(tensor_parallel_size=8)]}
    some_unknown = environment(group(idle=6, unknown_use=2))
    result = run(eight, some_unknown)
    assert (result["verdict"], result["group"]["available_gpus"]) == ("unknown", 6)
    assert result["best_unconfirmed"]["reasons"] == ["gpu_use_unknown"]
    assert run(eight, environment(group(idle=5, unknown_use=2)))["verdict"] == "does_not_fit"
    assert run(eight, some_unknown, max_gpus=6)["rejected"]["counts"] == {"not_enough_gpus": 1}
    # The list counts every GPU idle (listing.all_gpus): those of unknown use are not added again.
    nine = {"profiles": [profile(tensor_parallel_size=9)]}
    all_idle = environment(group(idle=8, unknown_use=2))
    assert run(nine, all_idle)["rejected"]["counts"] == {"not_enough_gpus": 1}


@pytest.mark.parametrize(
    ("fields", "env_overrides", "reason"),
    [
        ({"min_cpu_count": 128}, {}, "cpu_count_too_low"),
        ({"system_memory_per_device_gb": 200}, {"memory_gb": 300.0}, "memory_too_small"),
        ({"disk_space_per_device_gb": 60}, {"disk_free_gb": 100.0}, "disk_too_small"),
    ],
)
def test_host_requirements(
    fields: dict[str, Any], env_overrides: dict[str, Any], reason: str
) -> None:
    meta = {"profiles": [profile(tensor_parallel_size=2, **fields)]}
    record = run(meta, environment(**env_overrides))["rejected"]["examples"][0]
    assert record["reasons"] == [reason]
    fits = {"profiles": [profile(tensor_parallel_size=1, **fields)]}
    if reason != "cpu_count_too_low":  # per-device amounts scale with the GPU count
        assert run(fits, environment(**env_overrides))["verdict"] == "fits"


def test_release_architecture_and_image_architectures() -> None:
    meta = {
        "release_details": {"architecture": ["AMD 64", "ARM 64"]},
        "profiles": [profile()],
    }
    assert run(meta)["verdict"] == "fits"
    assert "release_details.architecture" not in run(meta)["not_recorded"]
    arm_only = {"release_details": {"architecture": ["ARM 64"]}, "profiles": [profile()]}
    record = run(arm_only)["rejected"]["examples"][0]
    assert record["reasons"] == ["cpu_arch_unsupported"]
    on_arm = environment(arch="aarch64")
    assert run(arm_only, on_arm)["verdict"] == "fits"
    image = run({"profiles": [profile()]}, image_architectures=["arm64"])
    assert image["rejected"]["counts"] == {"cpu_arch_unsupported": 1}
    assert run({"profiles": [profile()]}, image_architectures=["amd64"])["verdict"] == "fits"


def k8s_environment(*groups: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """A Kubernetes Environment whose GPU nodes mix CPU architectures: no cluster-wide arch."""
    fields: dict[str, Any] = {"target": "k8s", "context": "ctx", "namespace": "ns1", "arch": None,
                              "disk_free_gb": None, "docker": None}  # fmt: skip
    return environment(*groups, **{**fields, **overrides})


def test_on_kubernetes_each_gpu_group_has_its_own_cpu_architecture() -> None:
    """A cluster of x86 and Grace GPU nodes has no single CPU architecture; each GPU group is
    judged with its own nodes' architecture, so a group whose nodes are all arm64 can fit."""
    both = {"release_details": {"architecture": ["AMD 64", "ARM 64"]}, "profiles": [profile()]}
    grace = group(name="NVIDIA-GB300", arch="arm64")
    assert run(both, k8s_environment(grace), image_architectures=["amd64", "arm64"])["verdict"] == (
        "fits"
    )
    # only a group whose own nodes mix architectures (or carry no label) is unknown
    mixed = run(both, k8s_environment(group(name="NVIDIA-L40S", arch=None)))["best_unconfirmed"]
    assert mixed["reasons"] == ["arch_unknown"]
    assert (
        "The profile supports amd64, arm64; the CPU architecture of the NVIDIA-L40S nodes is "
        "unknown (more than one, or not labelled)"
    ) in mixed["why"]
    arm_only = {"release_details": {"architecture": ["ARM 64"]}, "profiles": [profile()]}
    x86 = run(arm_only, k8s_environment(group(name="NVIDIA-H100-NVL", arch="amd64")))
    record = x86["rejected"]["examples"][0]
    assert record["reasons"] == ["cpu_arch_unsupported"]
    assert "The profile supports arm64; the NVIDIA-H100-NVL nodes are amd64" in record["why"]
    # an environment file from before groups had an arch: the cluster-wide value, as before
    older = k8s_environment(group(name="NVIDIA-H100-NVL"), arch="amd64")
    assert run(arm_only, older)["rejected"]["counts"] == {"cpu_arch_unsupported": 1}


def test_on_kubernetes_each_gpu_group_has_its_own_driver_cpus_and_memory() -> None:
    """The cluster-wide driver, CPU count and memory are the smallest over every GPU node; each
    group is judged with its own nodes' values instead."""
    needs = {"profiles": [profile(min_driver_version="575.0", min_cpu_count=100,
                                  system_memory_per_device_gb=100)]}  # fmt: skip
    newer = group(name="NVIDIA-GB300", arch="arm64", driver_version="580.126.09", cpu_count=144,
                  host_memory_gb=900.0)  # fmt: skip
    cluster = k8s_environment(newer, driver_version="570.10", cpu_count=64, memory_gb=50.0)
    assert run(needs, cluster)["verdict"] == "fits"
    # on a host the machine's own values apply to every group, as before
    host = environment(newer, driver_version="570.10", cpu_count=64, memory_gb=50.0)
    assert set(run(needs, host)["rejected"]["counts"]) == {
        "driver_too_old",
        "cpu_count_too_low",
        "memory_too_small",
    }


def test_unified_memory_group_uses_its_memory_gb() -> None:
    gb10 = group(name="NVIDIA GB10", pci_device_id="2e12:10de", pci_device_ids=["2e12:10de"],
                 memory_mib=None, memory_gb=128.5, unified_memory=True,
                 compute_capability="12.1", count=1, idle=1, indices=[0])  # fmt: skip
    result = run(metadata(LLAMA_8B), environment(gb10))
    assert result["verdict"] == "fits"
    assert result["recommended"]["precision"] == "nvfp4"
    assert result["recommended"]["tier"] == "tested"


# --- capabilities ------------------------------------------------------------------------------


def test_lora_filter() -> None:
    result = run(metadata(LLAMA_8B), need_lora=True)
    assert result["recommended"]["profile_id"].startswith("4d9fed9c")
    assert result["recommended"]["feat_lora"] is True
    assert result["rejected"]["counts"]["lora_unsupported"] >= 2


def test_backend_filter() -> None:
    meta = metadata(LLAMA_8B)
    assert run(meta, backend="VLLM")["verdict"] == "fits"
    result = run(meta, backend="sglang")
    assert result["verdict"] == "does_not_fit"
    assert result["recommended"] is None
    assert result["rejected"]["counts"]["backend_mismatch"] == 8
    mixed = {
        "profiles": [
            profile("a-vllm", inference_backend="vllm"),
            profile("b-trt", inference_backend="tensorrt_llm"),
        ]
    }
    assert run(mixed)["recommended"]["profile_id"] == "b-trt"
    assert run(mixed, backend="vllm")["recommended"]["profile_id"] == "a-vllm"


# --- ranking -----------------------------------------------------------------------------------


def test_prefer_accuracy_ordering() -> None:
    result = run(metadata(LLAMA_8B), prefer="accuracy")
    assert result["recommended"]["profile_id"].startswith("092ed421")
    assert [r["precision"] for r in result["shortlist"]] == ["bf16", "bf16", "fp8", "fp8"]
    meta = {
        "profiles": [
            profile(f"p-{precision}", precision=precision, tested_gpu_devices=[H100_NVL])
            for precision in ("fp8", "fp32", "int4_awq", "custom", "fp16")
        ]
    }
    accuracy = [r["precision"] for r in run(meta, prefer="accuracy")["shortlist"]]
    assert accuracy == ["fp32", "fp16", "fp8", "int4_awq", "custom"]
    throughput = [r["precision"] for r in run(meta)["shortlist"]]
    assert throughput == ["int4_awq", "fp8", "fp16", "fp32", "custom"]


def test_ties_broken_by_profile_id_and_top() -> None:
    meta = {"profiles": [profile(pid) for pid in ("ccc", "aaa", "bbb")]}
    result = run(meta, top=2)
    assert [r["profile_id"] for r in result["shortlist"]] == ["aaa", "bbb"]
    assert result["recommended"]["profile_id"] == "aaa"


def test_tier_beats_precision_and_lora_last() -> None:
    meta = {
        "profiles": [
            profile("tested-fp8", precision="fp8"),
            profile("direct-bf16-lora", gpu_device=H100_NVL, tested_gpu_devices=[], feat_lora=True),
            profile("direct-bf16", gpu_device=H100_NVL, tested_gpu_devices=[], feat_lora=False),
        ]
    }
    order = [r["profile_id"] for r in run(meta)["shortlist"]]
    assert order == ["direct-bf16", "direct-bf16-lora", "tested-fp8"]
    # Among unknown profiles too: tested on this GPU before not tested on it.
    unknown = {
        "profiles": [
            profile("agnostic-fp8", precision="fp8", tested_gpu_devices=[]),
            profile("tested-bf16-no-vram", min_vram_per_device_gb=None),
        ]
    }
    order = [r["profile_id"] for r in run(unknown)["unconfirmed"]]
    assert order == ["tested-bf16-no-vram", "agnostic-fp8"]


def test_best_group_among_mixed_gpus() -> None:
    small = group(name="NVIDIA L4", pci_device_id="27b8:10de", pci_device_ids=["27b8:10de"],
                  memory_gb=24.15, compute_capability="8.9", count=2, idle=2)  # fmt: skip
    result = run(metadata(LLAMA_8B), environment(small, group()))
    assert result["group"]["name"] == "NVIDIA H100 NVL"
    assert [g["name"] for g in result["groups"]] == ["NVIDIA L4", "NVIDIA H100 NVL"]
    assert result["groups"][0]["fits"] == 0
    assert any("GPU types" in note for note in result["notes"])


# --- the alternative ---------------------------------------------------------------------------


def gb10() -> dict[str, Any]:
    """A DGX Spark's one GB10 as inspect saw it: unified memory, compute capability 12.1."""
    return group(name="NVIDIA GB10", pci_device_id=GB10, pci_device_ids=[GB10], memory_mib=None,
                 memory_gb=128.5, unified_memory=True, compute_capability="12.1", count=1, idle=1,
                 indices=[0])  # fmt: skip


def on_spark(name: str = LIGHTNING, **options: Any) -> dict[str, Any]:
    """nemotron-3.5-lightning-30b-a3b (2.1.6 unless `name` says) on a DGX Spark (arm64), checked as
    select checks it."""
    spark = environment(gb10(), arch="arm64", cpu_count=20, memory_gb=128.5)
    archs = ["arm64", "amd64"]
    return run(metadata(name), spark, image_architectures=archs, arch_required=True, **options)


def test_dgx_spark_keeps_the_tested_bf16_and_offers_the_untested_nvfp4() -> None:
    """In 2.1.6, GB10 is in the tested list of one bf16 profile only; the nvfp4 profiles that fit
    its memory are GPU-agnostic, tested on other GPUs or on none. The tested bf16 profile stays
    the recommendation, and the nvfp4 profile on 1 GPU without LoRA is offered beside it."""
    result = on_spark()
    assert result["verdict"] == "fits"
    rec = result["recommended"]
    assert (rec["profile_id"], rec["precision"], rec["tier"]) == (LIGHTNING_BF16, "bf16", "tested")
    assert [r["profile_id"] for r in result["shortlist"]] == [LIGHTNING_BF16]
    alternative = result["alternative"]
    assert alternative == {
        "confirmed": False,
        **by_id(result)[LIGHTNING_NVFP4],
        "offer": "faster (nvfp4) than the recommended bf16, but this image's NGC metadata records "
        "no test of it on this GPU",
    }
    assert (alternative["verdict"], alternative["tier"], alternative["reasons"]) == (
        "unknown",
        "agnostic",
        ["gpu_not_tested"],
    )
    assert (alternative["precision"], alternative["backend"], alternative["gpus_required"]) == (
        "nvfp4",
        "vllm",
        1,
    )
    assert alternative["why"] == [
        "GPU-agnostic profile not tested on this GPU (2e12:10de); no tested GPU is recorded",
        "Needs 30 GB per GPU; each GPU has 128.5 GB",
        "Needs 1 GPU; 1 is available",
    ]
    # the nvfp4 profile tested on B200 alone ranks after it: it has LoRA, which was not asked for
    unconfirmed = [r["profile_id"] for r in result["unconfirmed"]]
    assert unconfirmed[:2] == [LIGHTNING_NVFP4, LIGHTNING_NVFP4_LORA]


def test_dgx_spark_offers_nothing_when_bf16_ranks_first_or_nothing_fits() -> None:
    accuracy = on_spark(prefer="accuracy")  # bf16 ranks before every other precision of 2.1.6
    assert accuracy["recommended"]["profile_id"] == LIGHTNING_BF16
    assert accuracy["alternative"] is None
    # with --need-lora the tested bf16 profile fails: nothing fits, so nothing is offered beside
    lora = on_spark(need_lora=True)
    assert (lora["verdict"], lora["recommended"], lora["alternative"]) == ("unknown", None, None)
    assert lora["best_unconfirmed"]["profile_id"] == LIGHTNING_NVFP4_LORA
    assert on_spark(backend="sglang")["alternative"] is None


def untested(profile_id: str, **fields: Any) -> dict[str, Any]:
    """A GPU-agnostic profile tested on a B200 only: gpu_not_tested on the H100 NVL."""
    return profile(profile_id, tested_gpu_devices=["2901:10de"], **fields)


def alternative_of(*profiles: dict[str, Any], **options: Any) -> str | None:
    """The id of the profile offered beside a tested bf16 profile ("tested") on the H100 NVL."""
    found = run({"profiles": [profile("tested"), *profiles]}, **options)["alternative"]
    return None if found is None else str(found["profile_id"])


def test_the_alternative_lacks_only_a_test_on_this_gpu() -> None:
    assert alternative_of(untested("fp8", precision="fp8")) == "fp8"
    # any other open check rules it out
    assert alternative_of(untested("fp8", precision="fp8", min_vram_per_device_gb=None)) is None
    no_cc = environment(group(compute_capability=None))  # FP8 needs a known compute capability
    assert alternative_of(untested("fp8", precision="fp8"), env=no_cc) is None
    # and so does any failure, --max-gpus too
    assert alternative_of(untested("fp8", precision="fp8", min_vram_per_device_gb=200)) is None
    two = untested("fp8", precision="fp8", tensor_parallel_size=2)
    assert alternative_of(two) == "fp8"
    assert alternative_of(two, max_gpus=1) is None
    # a profile built for another GPU is never offered
    other = profile("other", precision="fp8", gpu_device="2901:10de", tested_gpu_devices=[])
    assert alternative_of(other) is None


@pytest.mark.parametrize(
    ("fields", "options", "offered"),
    [
        ({"feat_lora": True}, {"need_lora": True}, True),
        ({"feat_lora": False}, {"need_lora": True}, False),
        ({"feat_lora": None}, {"need_lora": True}, False),
        ({"inference_backend": "vllm"}, {"backend": "vllm"}, True),
        ({"inference_backend": "sglang"}, {"backend": "vllm"}, False),
        ({"inference_backend": None}, {"backend": "vllm"}, False),
    ],
)
def test_the_alternative_passes_lora_and_backend_as_asked(
    fields: dict[str, Any], options: dict[str, Any], offered: bool
) -> None:
    tested = profile("tested", feat_lora=True)
    meta = {"profiles": [tested, untested("fp8", precision="fp8", **fields)]}
    assert run(meta, **options)["recommended"]["profile_id"] == "tested"
    assert (run(meta, **options)["alternative"] is not None) == offered


def test_the_alternative_has_fewer_bits_than_every_profile_that_fits() -> None:
    for order in fit.PRECISION_ORDER.values():  # every precision ranked has its width
        assert set(order) == set(fit.PRECISION_BITS)
    assert alternative_of(untested("same", precision="bf16")) is None
    assert alternative_of(untested("worse", precision="fp16")) is None
    # the best-ranked one is offered, with the GPU counts when they differ
    awq = untested("awq", precision="int4_awq", tensor_parallel_size=2)
    result = run({"profiles": [profile("tested"), untested("fp8", precision="fp8"), awq]})
    assert result["alternative"]["profile_id"] == "awq"
    assert result["alternative"]["offer"] == (
        "faster (int4_awq, on 2 GPUs) than the recommended bf16 (on 1 GPU), but this image's NGC "
        "metadata records no test of it on this GPU"
    )
    # under --prefer accuracy a higher precision is offered beside a tested FP8 profile
    tested_fp8 = profile("tested-fp8", precision="fp8")
    accuracy = run({"profiles": [tested_fp8, untested("bf16")]}, prefer="accuracy")
    assert accuracy["alternative"]["offer"] == (
        "more accurate (bf16) than the recommended fp8, but this image's NGC metadata records no "
        "test of it on this GPU"
    )
    assert alternative_of(untested("fp8", precision="fp8"), prefer="accuracy") is None
    # a tested profile that fits beside the recommended one already has that precision: the
    # untested one is no faster choice
    tested_fp8_too = profile("tested-fp8", precision="fp8", feat_lora=True)
    assert alternative_of(tested_fp8_too, untested("fp8", precision="fp8")) is None
    assert alternative_of(tested_fp8_too, untested("awq", precision="int4_awq")) == "awq"
    # a precision outside the order ranks last, which says nothing of speed: no claim is made
    assert alternative_of(untested("int4", precision="int4")) is None
    tested_int4 = profile("tested-int4", precision="int4")
    assert run({"profiles": [tested_int4, untested("fp8", precision="fp8")]})["alternative"] is None


@pytest.mark.parametrize(
    ("prefer", "recommended", "offered"),
    [
        ("throughput", "fp16", "bf16"),
        ("throughput", "int8", "fp8"),
        ("throughput", "mxfp4", "nvfp4"),
        ("throughput", "fp4", "mxfp4"),
        ("accuracy", "fp16", "bf16"),
        ("accuracy", "int8", "fp8"),
        ("accuracy", "nvfp4", "mxfp4"),
    ],
)
def test_no_alternative_of_the_same_width(prefer: str, recommended: str, offered: str) -> None:
    """The order ranks one before the other, but neither is faster or more accurate: no offer
    would be true. On a B200, which runs all of these precisions, the other profile lacks only a
    test on this GPU, so the width alone keeps it from being offered."""
    b200 = group(name="NVIDIA B200", pci_device_id="2901:10de", pci_device_ids=["2901:10de"],
                 memory_mib=183359, memory_gb=192.26, compute_capability="10.0", count=1, idle=1,
                 indices=[0])  # fmt: skip
    tested = profile("tested", precision=recommended, tested_gpu_devices=["2901:10de"])
    other = profile("x", precision=offered)  # tested on the H100 NVL only
    result = run({"profiles": [tested, other]}, environment(b200), prefer=prefer)
    assert result["recommended"]["profile_id"] == "tested"
    assert (by_id(result)["x"]["verdict"], by_id(result)["x"]["reasons"]) == (
        "unknown",
        ["gpu_not_tested"],
    )
    assert result["alternative"] is None


def test_no_alternative_on_a_b200_where_a_tested_nvfp4_profile_fits() -> None:
    """On one B200, 2.1.6 recommends its bf16 profile built for the B200, and the shortlist has an
    nvfp4 profile tested on it: no untested nvfp4 profile is offered as the faster choice."""
    b200 = group(name="NVIDIA B200", pci_device_id="2901:10de", pci_device_ids=["2901:10de"],
                 memory_mib=183359, memory_gb=192.26, compute_capability="10.0", count=1, idle=1,
                 indices=[0])  # fmt: skip
    archs = ["arm64", "amd64"]
    meta = metadata(LIGHTNING)
    result = run(meta, environment(b200), image_architectures=archs, arch_required=True)
    assert (result["recommended"]["precision"], result["recommended"]["tier"]) == ("bf16", "direct")
    tested = [
        (r["profile_id"], r["tier"]) for r in result["shortlist"] if r["precision"] == "nvfp4"
    ]
    assert tested == [(LIGHTNING_NVFP4_LORA, "tested")]
    assert LIGHTNING_NVFP4 in [r["profile_id"] for r in result["unconfirmed"]]
    assert result["alternative"] is None


def test_a_pinned_profile_stays_on_the_gpu_type_chosen_on_every_profile() -> None:
    """select --profile: GB10 (1 GPU) and RTX PRO 4500 (2 GPUs) leave the nvfp4 profile unknown
    alike, and alone the GPUs with more of them would win the tie. The GB10, where the tested bf16
    profile made select offer it, stays the GPU type."""
    rtx = group(name="NVIDIA RTX PRO 4500 Blackwell Server Edition", pci_device_id="2c3a:10de",
                pci_device_ids=["2c3a:10de"], memory_mib=32623, memory_gb=34.2,
                compute_capability="12.0", count=2, idle=2, indices=[1, 2])  # fmt: skip
    both = environment(gb10(), rtx, arch="arm64")
    archs = ["arm64", "amd64"]
    first = run(metadata(LIGHTNING), both, image_architectures=archs, arch_required=True)
    assert (first["group"]["name"], first["alternative"]["profile_id"]) == (
        "NVIDIA GB10",
        LIGHTNING_NVFP4,
    )
    pinned = run(metadata(LIGHTNING), both, image_architectures=archs, arch_required=True,
                 profile=LIGHTNING_NVFP4)  # fmt: skip
    assert (pinned["verdict"], pinned["group"]["name"]) == ("unknown", "NVIDIA GB10")
    assert pinned["best_unconfirmed"]["profile_id"] == LIGHTNING_NVFP4
    assert (pinned["profiles_total"], pinned["recommended"], pinned["alternative"]) == (
        1,
        None,
        None,
    )
    assert [(g["name"], g["unknown"]) for g in pinned["groups"]] == [
        ("NVIDIA GB10", 1),
        ("NVIDIA RTX PRO 4500 Blackwell Server Edition", 1),
    ]
    # the tested bf16 profile pinned fits there
    bf16 = run(metadata(LIGHTNING), both, image_architectures=archs, arch_required=True,
               profile=LIGHTNING_BF16)  # fmt: skip
    assert (bf16["verdict"], bf16["recommended"]["profile_id"]) == ("fits", LIGHTNING_BF16)


def test_the_older_image_that_tested_the_nvfp4_profile_on_gb10_gives_the_faster_fit() -> None:
    """2.1.5 lists the GB10 as tested for the 1-GPU nvfp4 profiles that 2.1.6 tested on no GPU or
    on the B200 only: faster_fit names the one without LoRA, which fits there."""
    newest, older = on_spark(top=1000), on_spark(LIGHTNING_215, top=1000)
    assert [r["profile_id"] for r in newest["shortlist"]] == [LIGHTNING_BF16]
    found = fit.faster_fit(newest["shortlist"], older["shortlist"], "throughput")
    assert found == by_id(older)[LIGHTNING_NVFP4]
    assert (found["verdict"], found["tier"], found["precision"], found["reasons"]) == (
        "fits",
        "tested",
        "nvfp4",
        [],
    )
    assert found["why"][0] == f"Tested on this GPU ({GB10})"
    offer = fit.faster_fit_offer(
        found, newest["recommended"], "throughput", "the older image 2.1.5"
    )
    assert offer == "faster (nvfp4) and tested on this GPU, but in the older image 2.1.5"
    # nothing of 2.1.5 is more accurate than the bf16 profile, nor is 2.1.6 faster than 2.1.5
    accurate = on_spark(prefer="accuracy", top=1000)["shortlist"]
    older_accurate = on_spark(LIGHTNING_215, prefer="accuracy", top=1000)["shortlist"]
    assert fit.faster_fit(accurate, older_accurate, "accuracy") is None
    assert fit.faster_fit(older["shortlist"], newest["shortlist"], "throughput") is None


# --- the records of other images (FitOptions.other_records) ---------------------------------------

CARRIED_WHY = (
    f"Tested on this GPU ({GB10}) in 2.1.5 (same profile id); this image's metadata records no test"
)
FILES = "aaa"  # a workspace_hash: a record counts only for a profile with the same one


def records_of(*names: tuple[str, str]) -> dict[str, list[fit.OtherRecord]]:
    """FitOptions.other_records from the fixtures named, as (image tag, file name)."""
    return fit.other_records((tag, metadata(name)) for tag, name in names)


def test_a_test_another_image_records_for_the_same_profile_id_counts_here() -> None:
    """2.1.6 on a DGX Spark, with what 2.1.5 records: its nvfp4 profile without LoRA was tested on
    the GB10 there, so it counts as tested in 2.1.6 too and is recommended, with a why line naming
    2.1.5. The bf16 profile 2.1.6 itself tested on the GB10 stays a fit below it, and nothing is
    left to offer as the faster choice."""
    result = on_spark(top=1000, other_records=records_of(("2.1.5", LIGHTNING_215)))
    assert result["verdict"] == "fits"
    rec = result["recommended"]
    assert (rec["profile_id"], rec["precision"], rec["tier"], rec["reasons"]) == (
        LIGHTNING_NVFP4,
        "nvfp4",
        "tested",
        [fit.TESTED_ELSEWHERE],
    )
    assert rec["why"] == [
        CARRIED_WHY,
        "Needs 30 GB per GPU; each GPU has 128.5 GB",
        "Needs 1 GPU; 1 is available",
    ]
    shortlist = [(r["profile_id"], r["precision"], r["reasons"]) for r in result["shortlist"]]
    assert shortlist[:3] == [
        (LIGHTNING_NVFP4, "nvfp4", [fit.TESTED_ELSEWHERE]),
        (LIGHTNING_NVFP4_LORA, "nvfp4", [fit.TESTED_ELSEWHERE]),  # LoRA, not asked for: after
        (LIGHTNING_BF16, "bf16", []),  # tested on the GB10 by 2.1.6's own metadata
    ]
    assert by_id(result)[LIGHTNING_BF16]["why"][0] == f"Tested on this GPU ({GB10})"
    assert result["alternative"] is None
    assert fit.TESTED_ELSEWHERE_NOTE in result["notes"]
    assert fit.UNTESTED_NOTE not in result["notes"]
    # --prefer accuracy: the bf16 profile tested here ranks first, as without the records
    accuracy = on_spark(prefer="accuracy", other_records=records_of(("2.1.5", LIGHTNING_215)))
    assert (accuracy["recommended"]["profile_id"], accuracy["alternative"]) == (
        LIGHTNING_BF16,
        None,
    )
    # without the records, 2.1.6 alone: the bf16 profile, and the nvfp4 one offered unconfirmed
    alone = on_spark()
    assert alone["recommended"]["profile_id"] == LIGHTNING_BF16
    assert alone["alternative"]["profile_id"] == LIGHTNING_NVFP4
    assert fit.TESTED_ELSEWHERE_NOTE not in alone["notes"]


def test_only_a_record_of_the_same_profile_id_that_names_this_gpu_counts() -> None:
    """A record counts only for the exact profile id, only when it names one of this GPU's PCI
    ids, and only when the profile's own record names none of them: its own comes first."""
    untested_fp8 = untested("fp8", precision="fp8", workspace_hash=FILES)
    tested_here = fit.OtherRecord("2.1.5", (), (H100_NVL,), FILES)
    elsewhere = fit.OtherRecord("2.1.5", (), ("2901:10de",), FILES)

    def record_of(profile_: dict[str, Any], records: dict[str, list[fit.OtherRecord]]) -> Any:
        return by_id(run({"profiles": [profile_]}, other_records=records))[profile_["profile_id"]]

    carried = record_of(untested_fp8, {"fp8": [tested_here]})
    assert (carried["verdict"], carried["tier"], carried["reasons"]) == (
        "fits",
        "tested",
        [fit.TESTED_ELSEWHERE],
    )
    assert carried["why"][0] == (
        f"Tested on this GPU ({H100_NVL}) in 2.1.5 (same profile id); this image's metadata "
        "records no test"
    )
    for records in (
        {"FP8": [tested_here]},  # ids are matched exactly
        {"fp8-other": [tested_here]},  # another profile id
        {"fp8": [elsewhere]},  # another GPU
        {},
    ):
        kept = record_of(untested_fp8, records)
        assert (kept["verdict"], kept["reasons"]) == ("unknown", ["gpu_not_tested"]), records
    # a build for this GPU in another image is named as such; the first record that names this
    # GPU is the one named (the caller gives them newest version first)
    built_here = fit.OtherRecord("2.1.4", (H100_NVL,), (), FILES)
    built = record_of(untested_fp8, {"fp8": [elsewhere, built_here]})
    assert built["why"][0].startswith(f"Built for this GPU ({H100_NVL}) in 2.1.4 (same profile id)")
    assert built["tier"] == "tested"
    newest = record_of(
        untested_fp8, {"fp8": [tested_here, fit.OtherRecord("2.1.4", (), (H100_NVL,), FILES)]}
    )
    assert newest["why"][0].startswith(f"Tested on this GPU ({H100_NVL}) in 2.1.5")
    # the profile's own record comes first: no note, and its own why line
    own = record_of(profile("fp8", precision="fp8", workspace_hash=FILES), {"fp8": [tested_here]})
    assert (own["tier"], own["reasons"], own["why"][0]) == (
        "tested",
        [],
        f"Tested on this GPU ({H100_NVL})",
    )
    direct_fp8 = profile("fp8", gpu_device=H100_NVL, workspace_hash=FILES)
    direct = record_of(direct_fp8, {"fp8": [tested_here]})
    assert (direct["tier"], direct["reasons"]) == ("direct", [])
    # this GPU's PCI id unknown (a Kubernetes product the table lacks): nothing to match on
    no_pci = environment(group(pci_device_id=None, pci_device_ids=[]))
    unknown = by_id(run({"profiles": [untested_fp8]}, no_pci, other_records={"fp8": [tested_here]}))
    assert unknown["fp8"]["reasons"] == ["pci_unknown"]


def test_a_record_elsewhere_is_evidence_as_a_record_here_would_be() -> None:
    """As for a profile tested on this GPU by its own metadata: the precision limits of
    GPU-agnostic profiles are not applied (FP8 on an A100, compute capability 8.0), a build for
    another GPU is still tested here, a match on a PCI id inferred from the product name says so,
    and every other check still applies."""
    a100 = environment(group(name="NVIDIA A100-SXM4-80GB", pci_device_id="20b2:10de",
                             pci_device_ids=["20b2:10de"], compute_capability="8.0"))  # fmt: skip
    on_a100 = {"fp8": [fit.OtherRecord("2.0.13", (), ("20b2:10de",), FILES)]}
    fp8 = untested("fp8", precision="fp8", workspace_hash=FILES)
    alone = by_id(run({"profiles": [fp8]}, a100))["fp8"]
    assert alone["reasons"] == ["precision_unsupported", "gpu_not_tested"]
    carried = by_id(run({"profiles": [fp8]}, a100, other_records=on_a100))["fp8"]
    assert (carried["verdict"], carried["reasons"]) == ("fits", [fit.TESTED_ELSEWHERE])
    tested_here = {"fp8": [fit.OtherRecord("2.1.5", (), (H100_NVL,), FILES)]}
    b200_build = profile(
        "fp8", precision="fp8", gpu_device="2901:10de", tested_gpu_devices=[], workspace_hash=FILES
    )
    assert run({"profiles": [b200_build]})["rejected"]["counts"] == {"gpu_mismatch": 1}
    assert run({"profiles": [b200_build]}, other_records=tested_here)["verdict"] == "fits"
    inferred = environment(group(pci_source="product-table"))
    result = run({"profiles": [fp8]}, inferred, other_records=tested_here)
    assert result["recommended"]["reasons"] == [fit.TESTED_ELSEWHERE, "pci_inferred"]
    assert result["notes"][:2] == [
        f"Some profiles {fit.PCI_INFERRED_NOTE}.",
        fit.TESTED_ELSEWHERE_NOTE,
    ]
    too_big = untested("fp8", precision="fp8", min_vram_per_device_gb=200, workspace_hash=FILES)
    rejected = run({"profiles": [too_big]}, other_records=tested_here)["rejected"]
    assert rejected["counts"] == {"vram_too_small": 1}
    assert rejected["examples"][0]["tier"] == "tested"


def test_a_record_here_outranks_a_record_elsewhere_that_uses_more_gpus() -> None:
    """2.1.6 on 8 H100 SXM, with what 2.1.5 records: 2.1.6 records its 4-GPU bf16 profile as tested
    on the H100 and no tested GPU for its 8-GPU bf16 profile, which 2.1.5 records as tested on it.
    Equal on tier, precision, backend and LoRA, the profile this image's own metadata tested is
    recommended, as without the records, and the 8-GPU one stays a fit below it. With a record of
    its own, the 8-GPU profile would win on its GPU count."""
    h100 = group(name="NVIDIA H100 80GB HBM3", pci_device_id=H100_SXM, pci_device_ids=[H100_SXM],
                 memory_mib=81559, memory_gb=85.52)  # fmt: skip
    sxm = environment(h100)
    records = records_of(("2.1.5", LIGHTNING_215))
    result = run(metadata(LIGHTNING), sxm, top=1000, other_records=records)
    rec = result["recommended"]
    assert (rec["profile_id"], rec["gpus_required"], rec["reasons"]) == (LIGHTNING_BF16_TP4, 4, [])
    assert rec["why"][0] == f"Tested on this GPU ({H100_SXM})"
    carried = by_id(result)[LIGHTNING_BF16_TP8]
    assert (carried["verdict"], carried["tier"], carried["reasons"]) == (
        "fits",
        "tested",
        [fit.TESTED_ELSEWHERE],
    )
    # this image's own tests first, then the records of 2.1.5, each by GPU count
    shortlist = [(r["precision"], r["gpus_required"], r["reasons"]) for r in result["shortlist"]]
    assert shortlist[:4] == [
        ("bf16", 4, []),
        ("bf16", 1, []),
        ("bf16", 8, [fit.TESTED_ELSEWHERE]),
        ("bf16", 2, [fit.TESTED_ELSEWHERE]),
    ]
    # as 2.1.6 alone, without the records
    assert run(metadata(LIGHTNING), sxm)["recommended"]["profile_id"] == LIGHTNING_BF16_TP4
    # with a record of its own, the 8-GPU profile would rank first, on its GPU count
    own = metadata(LIGHTNING)
    for prof in own["profiles"]:
        if prof["profile_id"] == LIGHTNING_BF16_TP8:
            prof["tested_gpu_devices"] = [H100_SXM]
    assert run(own, sxm, other_records=records)["recommended"]["profile_id"] == LIGHTNING_BF16_TP8


def test_a_record_counts_only_with_the_same_workspace_hash_in_both_images() -> None:
    """A profile id hashes the profile's configuration, not its files: on 2026-11-19 the 1.8.0-RTX
    tags of three repositories list the same five ids with other workspace hashes. A record counts
    only when the profile and the record carry the same workspace_hash: one of other files, or
    with no hash (or an empty one) on either side, never does, and the first record that counts
    is the one named."""
    with_files = untested("fp8", precision="fp8", workspace_hash=FILES)

    def reasons(*records: fit.OtherRecord, profile_: dict[str, Any] = with_files) -> Any:
        result = run({"profiles": [profile_]}, other_records={"fp8": list(records)})
        return by_id(result)["fp8"]["reasons"]

    # the same hash in both images: the record counts, built for or tested on this GPU
    assert reasons(fit.OtherRecord("2.0.13", (), (H100_NVL,), FILES)) == [fit.TESTED_ELSEWHERE]
    assert reasons(fit.OtherRecord("2.0.13", (H100_NVL,), (), FILES)) == [fit.TESTED_ELSEWHERE]
    # other files, or no hash in the other image: never
    for files in ("bbb", None, ""):
        for built, tested in (((), (H100_NVL,)), ((H100_NVL,), ())):
            record = fit.OtherRecord("2.0.13", built, tested, files)
            assert reasons(record) == ["gpu_not_tested"], (files, built)
    # no hash in this image: never, whatever the other image records
    for own in (None, ""):
        no_files = untested("fp8", precision="fp8", workspace_hash=own)
        for files in (FILES, None, ""):
            record = fit.OtherRecord("2.0.13", (), (H100_NVL,), files)
            assert reasons(record, profile_=no_files) == ["gpu_not_tested"], (own, files)
    records = {
        "fp8": [
            fit.OtherRecord("2.1.5", (), (H100_NVL,), "bbb"),
            fit.OtherRecord("2.1.4", (), (H100_NVL,), None),
            fit.OtherRecord("2.0.13", (), (H100_NVL,), FILES),
        ]
    }
    named = by_id(run({"profiles": [with_files]}, other_records=records))["fp8"]
    assert named["why"][0].startswith(f"Tested on this GPU ({H100_NVL}) in 2.0.13 (same profile")


def test_other_records_keep_each_record_that_names_a_gpu_by_profile_id() -> None:
    """By profile id, in the order the documents are given, with the profile's workspace_hash
    there: 2.1.6 records no tested GPU for the nvfp4 profile without LoRA, so only 2.1.5's record
    of it is kept."""
    records = records_of(("2.1.5", LIGHTNING_215), ("2.1.6", LIGHTNING))
    tested_215 = ("2901:10de", "2941:10de", "2bb5:10de", GB10, "3182:10de", "31c2:10de")
    files = {p["profile_id"]: p["workspace_hash"] for p in metadata(LIGHTNING_215)["profiles"]}
    nvfp4 = files[LIGHTNING_NVFP4]
    assert nvfp4.startswith("5c2a8a92774e")  # the nvfp4 files of 2.1.5 and 2.1.6
    assert records[LIGHTNING_NVFP4] == [fit.OtherRecord("2.1.5", (), tested_215, nvfp4)]
    assert records[LIGHTNING_NVFP4_LORA] == [
        fit.OtherRecord("2.1.5", (), tested_215, nvfp4),
        fit.OtherRecord("2.1.6", (), ("2901:10de",), nvfp4),
    ]
    b200_build = "0307da6933369f40342ccb0880d07c7eb00ab0e7c6a0f8a7ea2a39a9b4b8f575"
    assert records[b200_build] == [
        fit.OtherRecord("2.1.5", ("2901:10de",), ("2901:10de",), files[b200_build])
    ]
    # a profile with no id, or with no GPU recorded, gives no record
    loose = {"profiles": [{"precision": "bf16", "tested_gpu_devices": [H100_NVL]},
                          {"profile_id": "none", "tested_gpu_devices": []}, "x"]}  # fmt: skip
    assert fit.other_records([("1.0.0", loose)]) == {}


def test_recorded_everywhere_gives_the_most_records_can_change() -> None:
    """The list fits an image as if every profile id were recorded, with its workspace_hash, as
    tested on these GPUs: on a DGX Spark 2.1.6's nvfp4 profile then ranks first, as the records of
    2.1.5 make it; on one B200 nothing changes, as its bf16 profile built for the B200 ranks first
    whatever the records. A profile with no workspace_hash gets no record: none can count."""
    spark = environment(gb10(), arch="arm64", cpu_count=20, memory_gb=128.5)
    everywhere = fit.recorded_everywhere(metadata(LIGHTNING), spark)
    assert set(everywhere) == {p["profile_id"] for p in metadata(LIGHTNING)["profiles"]}
    files = {p["profile_id"]: p["workspace_hash"] for p in metadata(LIGHTNING)["profiles"]}
    assert everywhere[LIGHTNING_NVFP4] == [
        fit.OtherRecord("another image", (), (GB10,), files[LIGHTNING_NVFP4])
    ]
    most = on_spark(other_records=everywhere)
    assert most["recommended"]["profile_id"] == LIGHTNING_NVFP4
    loose = {"profiles": [untested("fp8", precision="fp8"), untested("bf16", workspace_hash="")]}
    assert fit.recorded_everywhere(loose, environment()) == {}
    b200 = group(name="NVIDIA B200", pci_device_id="2901:10de", pci_device_ids=["2901:10de"],
                 memory_mib=183359, memory_gb=192.26, compute_capability="10.0", count=1, idle=1,
                 indices=[0])  # fmt: skip
    one_b200 = environment(b200)
    alone = run(metadata(LIGHTNING), one_b200)
    most = run(metadata(LIGHTNING), one_b200, other_records=fit.recorded_everywhere(
        metadata(LIGHTNING), one_b200))  # fmt: skip
    assert alone["recommended"] == most["recommended"]
    assert alone["recommended"]["tier"] == "direct"


def fits_of(*profiles: dict[str, Any], **options: Any) -> list[dict[str, Any]]:
    """The records of every profile that fits the H100 NVL, in rank order."""
    shortlist: list[dict[str, Any]] = run({"profiles": list(profiles)}, top=1000, **options)[
        "shortlist"
    ]
    return shortlist


def faster_id(
    chosen: list[dict[str, Any]], fits: list[dict[str, Any]], prefer: fit.Prefer = "throughput"
) -> str | None:
    found = fit.faster_fit(chosen, fits, prefer)
    return None if found is None else str(found["profile_id"])


def test_faster_fit_beats_every_profile_that_fits_in_the_chosen_image() -> None:
    chosen = fits_of(profile("bf16"))
    fp8, nvfp4 = profile("fp8", precision="fp8"), profile("nvfp4", precision="nvfp4")
    assert faster_id(chosen, fits_of(profile("old-bf16"), fp8)) == "fp8"
    # the first in rank order: nvfp4 before fp8 (or, with a tier between them, the direct one)
    assert faster_id(chosen, fits_of(fp8, nvfp4)) == "nvfp4"
    direct_fp8 = profile("direct-fp8", precision="fp8", gpu_device=H100_NVL)
    assert faster_id(chosen, fits_of(nvfp4, direct_fp8)) == "direct-fp8"
    # the chosen image has a profile that fits with as few bits: that is the faster choice
    direct_bf16 = profile("direct-bf16", gpu_device=H100_NVL)
    chosen_fp8_too = fits_of(direct_bf16, profile("tested-fp8", precision="fp8"))
    assert chosen_fp8_too[0]["profile_id"] == "direct-bf16"  # the recommended one
    assert faster_id(chosen_fp8_too, fits_of(fp8)) is None
    assert faster_id(chosen_fp8_too, fits_of(fp8, nvfp4)) == "nvfp4"
    # the same width is no faster, a precision outside the order says nothing, nor does a
    # recommended profile with one
    assert faster_id(chosen, fits_of(profile("fp16", precision="fp16"))) is None
    assert faster_id(chosen, fits_of(profile("int4", precision="int4"))) is None
    assert faster_id(fits_of(profile("int4", precision="int4")), fits_of(fp8)) is None
    # another profile of the chosen image with such a precision is left out of the comparison:
    # fp8 is still faster than the recommended bf16
    chosen_int4_too = fits_of(profile("bf16"), profile("int4", precision="int4"))
    assert [record["profile_id"] for record in chosen_int4_too] == ["bf16", "int4"]
    assert faster_id(chosen_int4_too, fits_of(fp8)) == "fp8"
    # --prefer accuracy: more bits
    chosen_fp8 = fits_of(fp8, prefer="accuracy")
    assert faster_id(chosen_fp8, fits_of(profile("bf16"), prefer="accuracy"), "accuracy") == "bf16"
    assert faster_id(chosen_fp8, fits_of(nvfp4, prefer="accuracy"), "accuracy") is None
    # only a profile that fits, and only beside a chosen image with one
    unknown = run({"profiles": [untested("fp8", precision="fp8")]})["unconfirmed"]
    assert faster_id(chosen, unknown) is None
    assert faster_id([], fits_of(fp8)) is None
    # nor one that fits there only by what a third image records: the select of its note checks
    # that image alone
    third = {"fp8": [fit.OtherRecord("2.0.13", (), (H100_NVL,), FILES)]}
    carried = fits_of(untested("fp8", precision="fp8", workspace_hash=FILES), other_records=third)
    assert carried[0]["reasons"] == [fit.TESTED_ELSEWHERE]
    assert faster_id(chosen, carried) is None
    assert faster_id(chosen, [*carried, *fits_of(nvfp4)]) == "nvfp4"


def test_faster_fit_offer_names_the_evidence_and_the_gpu_counts() -> None:
    recommended = fits_of(profile("bf16"))[0]
    direct = fits_of(
        profile("nvfp4", precision="nvfp4", gpu_device=H100_NVL, tensor_parallel_size=2)
    )
    where = "the hardware variant llama-3.1-8b-instruct-dgx-spark:1.0.0-variant"
    assert fit.faster_fit_offer(direct[0], recommended, "throughput", where) == (
        "faster (nvfp4, on 2 GPUs instead of 1) and built for this GPU, but in the hardware "
        "variant llama-3.1-8b-instruct-dgx-spark:1.0.0-variant"
    )
    fp8 = fits_of(profile("fp8", precision="fp8"))[0]
    accurate = fits_of(profile("bf16"), prefer="accuracy")[0]
    assert fit.faster_fit_offer(accurate, fp8, "accuracy", "the older image 2.0.13") == (
        "more accurate (bf16) and tested on this GPU, but in the older image 2.0.13"
    )


def test_no_alternative_without_a_recommended_profile() -> None:
    result = run({"profiles": [untested("fp8", precision="fp8"), untested("bf16")]})
    assert (result["verdict"], result["recommended"]) == ("unknown", None)
    assert result["best_unconfirmed"]["profile_id"] == "fp8"
    assert result["alternative"] is None
    # on the H100 NVL the agnostic nvfp4 profiles of llama-3.1-8b-instruct fail: nothing to offer
    assert run(metadata(LLAMA_8B))["alternative"] is None


def test_usable_profiles_drops_anything_that_is_not_a_profile() -> None:
    assert fit.usable_profiles({"profiles": [profile(), "x", None, 1]}) == [profile()]
    assert fit.usable_profiles({"profiles": None}) == []
    assert fit.usable_profiles({}) == []


# --- errors and CLI ----------------------------------------------------------------------------


def test_gpus_unknown_raises() -> None:
    env = environment(gpus_known=False)
    with pytest.raises(ns.ScriptError) as info:
        run(metadata(LLAMA_8B), env)
    assert info.value.code == "gpus_unknown"
    assert info.value.exit_code == ns.EXIT_PRECONDITION


def test_no_groups_raises() -> None:
    env = environment()
    env["groups"] = []
    with pytest.raises(ns.ScriptError) as info:
        run(metadata(LLAMA_8B), env)
    assert info.value.code == "no_gpus"


def test_does_not_fit_when_everything_rejected() -> None:
    result = run(metadata(LLAMA_70B), max_gpus=1)
    assert result["verdict"] == "does_not_fit"
    assert result["recommended"] is None
    assert result["best_unconfirmed"] is None
    assert len(result["rejected"]["examples"]) == 5
    assert sum(result["rejected"]["counts"].values()) >= result["profiles_total"]


def test_cli_library_notice_and_help() -> None:
    code, payload = run_cli("fit.py")
    assert code == 0
    assert payload["state"] == "library"
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "fit.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    assert proc.returncode == 0
    assert "Example:" in proc.stdout
