#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""fit: decide which NIM profiles fit an inspected environment, and rank them.

Pure functions, no I/O: `evaluate(metadata, env, options)` takes a NIM Metadata API document and an
Environment object (what `nimselect.py inspect --out` saves) and returns the verdict, the
recommended profile, a shortlist, why every other profile was rejected or left unconfirmed, and
an alternative to offer beside the recommended profile: one with a faster precision (more accurate
with --prefer accuracy) that was only not tested on this GPU. `faster_fit` finds the profile that
select offers instead when another image it checked has one that fits with such a precision.

The checks follow the Metadata API "Choose a Profile for Your Hardware" procedure, plus the
precision limits the NIM runtime applies to GPU-agnostic profiles. An absent VRAM need is unknown,
never a pass, and so is a profile that was neither built for nor tested on the GPU: the metadata
screens profiles before the pull, and only the image's `list-model-profiles`, after it, can confirm
one. An absent host requirement (driver, CPUs, memory, disk) is not checked; `not_recorded` lists
those that no candidate records. select and fit also give `evaluate` what the other images of the
same repository that a select checks record for the same profile ids (`other_records`): a profile
whose own record does not name this GPU, but whose exact id, with the same workspace_hash recorded
in both images, another of those images records as built for or tested on it, counts as tested
here (note `tested_in_other_image`, and a why line naming that image). Run directly, this module
prints a notice: `nimselect.py fit` is the command.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_common as ns

Prefer = Literal["throughput", "accuracy"]

TIER_RANK = {"direct": 0, "tested": 1, "agnostic": 2, "unmatched": 3}
PRECISION_ORDER: dict[str, tuple[str, ...]] = {
    "throughput": (
        "nvfp4", "mxfp4", "fp4", "int4_awq", "fp8", "int8", "bf16", "fp16", "fp32",
    ),
    "accuracy": (
        "fp32", "bf16", "fp16", "fp8", "int8", "int4_awq", "mxfp4", "fp4", "nvfp4",
    ),
}  # fmt: skip
BACKEND_ORDER = ("tensorrt_llm", "trtllm", "vllm", "sglang")
# Requirement fields reported in `not_recorded` when no candidate profile carries them.
REQUIREMENT_FIELDS = (
    "min_vram_per_device_gb",
    "min_driver_version",
    "min_cpu_count",
    "system_memory_per_device_gb",
    "disk_space_per_device_gb",
    "max_gpu_count",
    "release_details.architecture",
)
NVFP4_NOTE = (
    "NVFP4 profiles need compute capability 10.0 or newer. The NIM runtime can emulate NVFP4 on "
    "8.0-9.x with vLLM or SGLang only when a profile opts in, and the metadata does not record "
    "that, so they are treated as unsupported here."
)
PCI_INFERRED_NOTE = "matched on a PCI id inferred from the product name"
# The note code of a profile that counts as tested on this GPU because another image of the same
# repository records its exact id and workspace_hash as built for or tested on it
# (FitOptions.other_records).
TESTED_ELSEWHERE = "tested_in_other_image"
TESTED_ELSEWHERE_NOTE = (
    "Some profiles count as tested on this GPU (tested_in_other_image): this image's metadata "
    "records no test of them here, but another image of the same repository that select checks "
    "records the same profile id and workspace hash as built for or tested on this GPU; their why "
    "lines name that image."
)
UNTESTED_NOTE = (
    "A GPU-agnostic profile that was not tested on this GPU is unknown, not a fit: the metadata "
    "screens profiles before the pull, and only the image's list-model-profiles, after the pull, "
    "can confirm it."
)
# What a precision that ranks first in each --prefer order gives (the alternative's `offer`).
BETTER = {"throughput": "faster", "accuracy": "more accurate"}
# Bits per weight of each precision in the --prefer orders: fewer are faster, more are more
# accurate. The alternative must differ in this from every profile that fits, not only rank ahead
# in the order: bf16 is no faster than fp16, nor nvfp4 than mxfp4, nor fp8 than int8.
PRECISION_BITS = {
    "nvfp4": 4, "mxfp4": 4, "fp4": 4, "int4_awq": 4, "fp8": 8, "int8": 8, "bf16": 16, "fp16": 16,
    "fp32": 32,
}  # fmt: skip


@dataclass(frozen=True)
class OtherRecord:
    """What another image of the same repository records for one profile id: that image (its tag,
    as a why line names it), the PCI ids its metadata records the profile as built for
    (`gpu_device`) and tested on (`tested_gpu_devices`), and the profile's `workspace_hash` there
    (None when not recorded: the record then never counts)."""

    image: str
    built_for: tuple[str, ...]
    tested_on: tuple[str, ...]
    workspace_hash: str | None = None


@dataclass
class FitOptions:
    """What the user asked for, beyond the hardware."""

    max_gpus: int | None = None
    need_lora: bool = False
    backend: str | None = None
    prefer: Prefer = "throughput"
    top: int = 5
    image_architectures: list[str] | None = None
    # True: a profile whose image and metadata record no CPU architecture is `unknown`
    # (`arch_not_recorded`) instead of passing. `fit` and `select` both set it: the image's
    # architectures come from NGC's image list, which a NIM that is not public does not show,
    # or from the repository record, which describes the image tagged latestTag only.
    arch_required: bool = False
    # select --profile: the GPU type is chosen on every profile, as without it, and then only this
    # profile is reported, so a profile taken from a select is checked on the GPUs it named.
    profile: str | None = None
    # select and fit: by profile id, what the other images of this image's repository that a
    # select checks record (other_records), in the order a why line names them. A profile whose
    # own record names neither this GPU's id as built for nor as tested on counts as tested when
    # one of them does (TESTED_ELSEWHERE). Only a record with the profile's own workspace_hash
    # counts, and only when both images record one: never records of another repository, of
    # another profile id, or of other files.
    other_records: Mapping[str, Sequence[OtherRecord]] | None = None


@dataclass
class _Check:
    """One profile's evaluation while it is built."""

    profile: dict[str, Any]
    tier: str = "agnostic"
    failures: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    why: list[str] = field(default_factory=list)
    required: int = 1

    def fail(self, code: str, why: str) -> None:
        if code not in self.failures:
            self.failures.append(code)
        self.why.append(why)

    def unknown(self, code: str, why: str) -> None:
        if code not in self.unknowns:
            self.unknowns.append(code)
        self.why.append(why)

    @property
    def verdict(self) -> str:
        if self.failures:
            return "rejected"
        return "unknown" if self.unknowns else "fits"


# --- small parsers -----------------------------------------------------------------------------


def _num(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _int(value: object) -> int | None:
    number = _num(value)
    return int(number) if number is not None else None


def _gb(value: float) -> str:
    return f"{value:g}"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _ids(value: object) -> list[str]:
    return [item.lower() for item in ns.as_list(value)]


def _capability(text: object) -> tuple[int, int] | None:
    version = ns.parse_version(text if isinstance(text, str) else None)
    if not version:
        return None
    return (version[0], version[1] if len(version) > 1 else 0)


def normalise_arch(text: str) -> str:
    """'AMD 64', 'x86_64', 'amd64' -> 'amd64'; 'ARM 64', 'aarch64' -> 'arm64'; else lowercased."""
    key = "".join(ch for ch in text.lower() if ch.isalnum())
    if key in {"amd64", "x8664", "x64"}:
        return "amd64"
    if key in {"arm64", "aarch64"}:
        return "arm64"
    return text.strip().lower()


def _version_text(version: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version)


def _version_ge(have: tuple[int, ...], need: tuple[int, ...]) -> bool:
    width = max(len(have), len(need))
    return have + (0,) * (width - len(have)) >= need + (0,) * (width - len(need))


# --- the per-profile checks -------------------------------------------------------------------


def _group_ids(group: dict[str, Any]) -> list[str]:
    ids = _ids(group.get("pci_device_ids"))
    single = _ids(group.get("pci_device_id"))
    return ids or single


def _workspace(profile: dict[str, Any]) -> str | None:
    value = profile.get("workspace_hash")
    return value if isinstance(value, str) and value else None


def _recorded_elsewhere(
    profile: dict[str, Any],
    ours: list[str],
    records: Mapping[str, Sequence[OtherRecord]] | None,
) -> tuple[OtherRecord, str, bool] | None:
    """The first record of another image (FitOptions.other_records) that names one of our PCI ids
    for this profile's exact id and workspace_hash: (that record, the id, whether it was built
    for it rather than only tested on it). The id identifies the profile's configuration and the
    workspace_hash its files: a record counts only when both images record the same hash, so one
    of other files under the same id, or with no hash on either side, never does. None when none
    does."""
    pid = profile.get("profile_id")
    workspace = _workspace(profile)
    if not records or not isinstance(pid, str) or not pid or workspace is None:
        return None
    for record in records.get(pid, ()):
        if record.workspace_hash != workspace:
            continue
        for gpu_id in ours:
            if gpu_id in record.built_for or gpu_id in record.tested_on:
                return record, gpu_id, gpu_id in record.built_for
    return None


def _check_target(
    check: _Check,
    group: dict[str, Any],
    ours: list[str],
    records: Mapping[str, Sequence[OtherRecord]] | None = None,
) -> None:
    profile = check.profile
    gpu_device = _ids(profile.get("gpu_device"))
    tested = _ids(profile.get("tested_gpu_devices"))
    gpu_names = ns.as_list(profile.get("gpu"))
    inferred = group.get("pci_source") == "product-table"

    if not ours:
        if gpu_device or gpu_names:
            check.tier = "unmatched"
            check.unknown(
                "pci_unknown",
                "Built for a specific GPU, and this GPU's PCI id is unknown, so the match "
                "cannot be checked",
            )
        else:
            check.tier = "agnostic"
            check.unknown(
                "pci_unknown",
                "GPU-agnostic profile, and this GPU's PCI id is unknown, so whether it was "
                "tested on this GPU cannot be checked",
            )
        return

    direct = [gpu_id for gpu_id in ours if gpu_id in gpu_device]
    matched = [gpu_id for gpu_id in ours if gpu_id in tested]
    if direct:
        check.tier = "direct"
        check.why.append(f"Built for this GPU ({direct[0]})")
    elif matched:
        check.tier = "tested"
        check.why.append(f"Tested on this GPU ({matched[0]})")
    elif (elsewhere := _recorded_elsewhere(profile, ours, records)) is not None:
        # Only when its own record names none of our ids. The same profile id and workspace_hash
        # are the same configuration and files: another image's record of them is evidence here
        # too, as its own would be (the precision limits are skipped, as for `tested`).
        record, gpu_id, built = elsewhere
        check.tier = "tested"
        check.notes.append(TESTED_ELSEWHERE)
        check.why.append(
            f"{'Built for' if built else 'Tested on'} this GPU ({gpu_id}) in {record.image} "
            "(same profile id); this image's metadata records no test"
        )
    elif not gpu_device and not gpu_names:
        check.tier = "agnostic"
        where = f"tested on {', '.join(tested)}" if tested else "no tested GPU is recorded"
        check.unknown(
            "gpu_not_tested",
            f"GPU-agnostic profile not tested on this GPU ({', '.join(ours)}); {where}",
        )
        return
    elif gpu_device:
        check.tier = "agnostic"
        check.fail(
            "gpu_mismatch",
            f"Built for another GPU ({', '.join(gpu_device)}); this GPU is {', '.join(ours)}",
        )
        return
    else:
        check.tier = "agnostic"
        check.fail(
            "gpu_name_only",
            f"Targets {', '.join(gpu_names)} by name only and was not tested on this GPU "
            f"({', '.join(ours)})",
        )
        return
    if inferred:
        check.notes.append("pci_inferred")
        check.why.append(f"Note: {PCI_INFERRED_NOTE}")


def _check_precision(check: _Check, group: dict[str, Any]) -> None:
    if check.tier not in ("agnostic", "unmatched") or check.failures:
        return
    precision = str(check.profile.get("precision") or "").lower()
    if not precision:
        return
    cc = _capability(group.get("compute_capability"))
    gated = precision in ("fp8", "nvfp4", "int4_awq")
    if cc is None:
        if gated:
            check.unknown(
                "capability_unknown",
                f"{precision.upper()} depends on compute capability, which is unknown here",
            )
        return
    shown = f"{cc[0]}.{cc[1]}"
    if precision == "fp8" and cc < (8, 9):
        check.fail(
            "precision_unsupported",
            f"FP8 needs compute capability 8.9 or newer; this GPU is {shown}",
        )
    elif precision == "nvfp4" and cc < (10, 0):
        check.fail(
            "precision_unsupported",
            f"NVFP4 needs compute capability 10.0 or newer; this GPU is {shown} "
            "(emulation is not recorded in the metadata)",
        )
    elif precision == "int4_awq" and (10, 0) <= cc < (12, 0):
        check.fail(
            "precision_unsupported",
            f"INT4 AWQ is not supported on compute capability 10.x-11.x; this GPU is {shown}",
        )


def _check_vram(check: _Check, group: dict[str, Any]) -> None:
    need = _num(check.profile.get("min_vram_per_device_gb"))
    have = _num(group.get("memory_gb"))
    if need is None:
        check.unknown("vram_not_recorded", "The metadata does not record the VRAM it needs")
    elif have is None:
        check.unknown("vram_unknown", f"Needs {_gb(need)} GB per GPU; this GPU's memory is unknown")
    elif have >= need:
        check.why.append(f"Needs {_gb(need)} GB per GPU; each GPU has {_gb(have)} GB")
    else:
        check.fail(
            "vram_too_small", f"Needs {_gb(need)} GB per GPU; each GPU has only {_gb(have)} GB"
        )


def _check_gpus(check: _Check, available: int, possible: int) -> None:
    """`available`: the GPUs known to be free; `possible`: those plus the host GPUs whose memory
    use is unknown, which may be free or busy."""
    profile = check.profile
    tp = _int(profile.get("tensor_parallel_size")) or 1
    pp = _int(profile.get("pipeline_parallel_size")) or 1
    check.required = tp * pp
    need = _plural(check.required, "GPU")
    verb = "is" if available == 1 else "are"
    if check.required > possible:
        check.fail(
            "not_enough_gpus", f"Needs {need} on one node; only {available} {verb} available"
        )
    elif check.required > available:
        check.unknown(
            "gpu_use_unknown",
            f"Needs {need} on one node; {available} {verb} available, and the memory use of "
            f"{possible - available} more is unknown",
        )
    else:
        check.why.append(f"Needs {need}; {available} {verb} available")
    limit = _int(profile.get("max_gpu_count"))
    if limit is not None and limit < check.required:
        check.fail(
            "max_gpu_count", f"Uses {need} but the profile allows at most {_plural(limit, 'GPU')}"
        )


def _host_value(group: dict[str, Any], env: dict[str, Any], key: str, env_key: str = "") -> Any:
    """A host value for one GPU group. On Kubernetes, inspect records each group's driver and CPU
    architecture from its own nodes, since one cluster can mix Grace (arm64) and x86 GPU nodes, and
    the largest CPU count and memory (`host_memory_gb`) among them, since a pod needs one node with
    room; a host, or an older file without them, gives the environment's value."""
    if env.get("target") == "k8s" and key in group:
        return group[key]
    return env.get(env_key or key)


def _check_host(
    check: _Check,
    metadata: dict[str, Any],
    env: dict[str, Any],
    group: dict[str, Any],
    options: FitOptions,
) -> None:
    profile = check.profile
    required = check.required
    driver = _host_value(group, env, "driver_version")
    on_nodes = env.get("target") == "k8s" and "arch" in group
    where = f"the {group.get('name') or 'GPU'} nodes" if on_nodes else "this machine"

    need_driver = profile.get("min_driver_version")
    if need_driver is not None:
        need_v = ns.parse_version(str(need_driver))
        have_v = ns.parse_version(driver)
        if need_v is None:
            pass
        elif have_v is None:
            check.unknown(
                "driver_unknown", f"Needs driver {need_driver}; the driver version is unknown"
            )
        elif _version_ge(have_v, need_v):
            check.why.append(f"Needs driver {need_driver}; this host has {_version_text(have_v)}")
        else:
            check.fail(
                "driver_too_old",
                f"Needs driver {need_driver} or newer; this host has {driver}",
            )

    need_cpu = _int(profile.get("min_cpu_count"))
    have_cpu = _int(_host_value(group, env, "cpu_count"))
    if need_cpu is not None:
        if have_cpu is None:
            check.unknown("cpu_unknown", f"Needs {need_cpu} CPU cores; the core count is unknown")
        elif have_cpu < need_cpu:
            check.fail("cpu_count_too_low", f"Needs {need_cpu} CPU cores; there are {have_cpu}")

    per_device = _num(profile.get("system_memory_per_device_gb"))
    have_memory = _num(_host_value(group, env, "host_memory_gb", "memory_gb"))
    _check_amount(
        check,
        per_device,
        required,
        have_memory,
        ("memory_unknown", "memory_too_small", "host memory"),
    )
    if need_cpu is not None and per_device is not None and have_cpu is not None:
        need_memory = per_device * required
        if have_memory is not None and have_cpu >= need_cpu and have_memory >= need_memory:
            _check_one_node(check, group, env, need_cpu, need_memory)
    _check_amount(
        check,
        _num(profile.get("disk_space_per_device_gb")),
        required,
        _num(env.get("disk_free_gb")),
        ("disk_unknown", "disk_too_small", "free disk"),
    )

    arch = _host_value(group, env, "arch")
    have_arch = normalise_arch(arch) if isinstance(arch, str) and arch else None
    is_arch = "are" if on_nodes else "is"
    sources = (
        ("profile", _architectures(profile, metadata)),
        ("image", [normalise_arch(item) for item in options.image_architectures or []]),
    )
    if options.arch_required and not any(supported for _, supported in sources):
        check.unknown(
            "arch_not_recorded",
            f"The CPU architectures of this image are not recorded; {where} {is_arch} "
            + (have_arch or "of unknown architecture"),
        )
    for label, supported in sources:
        if not supported:
            continue
        if have_arch is None:
            unknown = (
                f"the CPU architecture of {where} is unknown (more than one, or not labelled)"
                if on_nodes
                else "the CPU architecture is unknown"
            )
            check.unknown("arch_unknown", f"The {label} supports {', '.join(supported)}; {unknown}")
        elif have_arch not in supported:
            check.fail(
                "cpu_arch_unsupported",
                f"The {label} supports {', '.join(supported)}; {where} {is_arch} {have_arch}",
            )


def _check_amount(
    check: _Check,
    per_device: float | None,
    required: int,
    have: float | None,
    codes: tuple[str, str, str],
) -> None:
    if per_device is None:
        return
    unknown_code, fail_code, label = codes
    need = per_device * required
    if have is None:
        check.unknown(unknown_code, f"Needs {_gb(need)} GB of {label}; the amount is unknown")
    elif have < need:
        check.fail(fail_code, f"Needs {_gb(need)} GB of {label}; there is {_gb(have)} GB")


def _check_one_node(
    check: _Check, group: dict[str, Any], env: dict[str, Any], need_cpu: int, need_memory: float
) -> None:
    """On Kubernetes a group's `cpu_count` and `host_memory_gb` are the largest of its nodes, which
    can be two nodes, and a NIM pod needs its CPU cores and memory on one. Called when both are
    enough: one node of `node_sizes` must have both. A host is one machine, and a file without
    `node_sizes` is judged on the largest values alone."""
    sizes = group.get("node_sizes") if env.get("target") == "k8s" else None
    pairs = [
        (_int(size.get("cpu_count")), _num(size.get("host_memory_gb")))
        for size in (sizes if isinstance(sizes, list) else [])
        if isinstance(size, dict)
    ]
    nodes = [(cpu, memory) for cpu, memory in pairs if cpu is not None and memory is not None]
    if not nodes or any(cpu >= need_cpu and memory >= need_memory for cpu, memory in nodes):
        return
    shown = ", ".join(f"{cpu} cores with {_gb(memory)} GB" for cpu, memory in nodes)
    check.fail(
        "node_too_small",
        f"Needs {need_cpu} CPU cores and {_gb(need_memory)} GB of host memory on one node; no "
        f"{group.get('name') or 'GPU'} node has both ({shown})",
    )


def _architectures(profile: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    for source in (profile, metadata):
        details = source.get("release_details")
        if isinstance(details, dict):
            values = ns.as_list(details.get("architecture"))
            if values:
                return [normalise_arch(item) for item in values]
    return []


def _check_features(check: _Check, options: FitOptions) -> None:
    profile = check.profile
    if options.need_lora:
        lora = profile.get("feat_lora")
        if lora is None:
            check.unknown("lora_not_recorded", "The metadata does not record LoRA support")
        elif lora is True:
            check.why.append("Supports LoRA adapters")
        else:
            check.fail("lora_unsupported", "Does not support LoRA adapters")
    if options.backend:
        backend = profile.get("inference_backend")
        if not isinstance(backend, str) or not backend:
            check.unknown("backend_not_recorded", "The metadata does not record the backend")
        elif backend.lower() == options.backend.lower():
            check.why.append(f"Uses the {backend} backend")
        else:
            check.fail("backend_mismatch", f"Uses the {backend} backend, not {options.backend}")


# --- ranking and records -----------------------------------------------------------------------


def _rank(ordering: Iterable[str], value: object) -> int:
    order = list(ordering)
    text = value.lower() if isinstance(value, str) else ""
    return order.index(text) if text in order else len(order)


def _sort_key(check: _Check, options: FitOptions) -> tuple[Any, ...]:
    profile = check.profile
    lora = 0
    if not options.need_lora and profile.get("feat_lora") is True:
        lora = 1
    return (
        TIER_RANK.get(check.tier, len(TIER_RANK)),
        _rank(PRECISION_ORDER[options.prefer], profile.get("precision")),
        _rank(BACKEND_ORDER, profile.get("inference_backend")),
        lora,
        # Among otherwise equal profiles, the image's own test outranks another image's record.
        TESTED_ELSEWHERE in check.notes,
        -check.required,
        str(profile.get("profile_id", "")),
    )


def _record(check: _Check) -> dict[str, Any]:
    profile = check.profile
    gpu = ns.as_list(profile.get("gpu"))
    return {
        "profile_id": profile.get("profile_id"),
        "verdict": check.verdict,
        "tier": check.tier,
        "precision": profile.get("precision"),
        "backend": profile.get("inference_backend"),
        "tp": _int(profile.get("tensor_parallel_size")),
        "pp": _int(profile.get("pipeline_parallel_size")),
        "gpus_required": check.required,
        "min_vram_per_device_gb": _num(profile.get("min_vram_per_device_gb")),
        # host needs: a Kubernetes hand-off carries them as the pod's requests
        "min_cpu_count": _int(profile.get("min_cpu_count")),
        "system_memory_per_device_gb": _num(profile.get("system_memory_per_device_gb")),
        "feat_lora": profile.get("feat_lora"),
        "gpu": gpu[0] if len(gpu) == 1 else (gpu or None),
        "reasons": check.failures + check.unknowns + check.notes,
        "why": check.why,
    }


def _available(group: dict[str, Any], env: dict[str, Any], options: FitOptions) -> tuple[int, int]:
    """The GPUs a profile can use on one node: those known to be free, and those that may be free
    (on a host, the idle ones and those whose memory use is unknown, at most the group's count)."""
    count = _int(group.get("count")) or 0
    if env.get("target") == "k8s":
        free = maybe = count
    else:
        idle = _int(group.get("idle"))
        free = idle if idle is not None else count
        unknown_use = _int(group.get("unknown_use")) or 0
        maybe = max(free, min(free + unknown_use, count))
    limits: list[int] = []
    if options.max_gpus is not None:
        limits.append(options.max_gpus)
    quota = env.get("gpu_quota")
    if isinstance(quota, dict):
        hard, used = _int(quota.get("hard")), _int(quota.get("used"))
        if hard is not None:
            limits.append(max(hard - (used or 0), 0))
    return max(min([free, *limits]), 0), max(min([maybe, *limits]), 0)


@dataclass
class _GroupResult:
    group: dict[str, Any]
    available: int
    fits: list[_Check]
    unknown: list[_Check]
    rejected: list[_Check]
    candidates: list[_Check]


def usable_profiles(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """The profiles of a Metadata API document that can be checked. None at all means nothing is
    known before the pull: callers treat it as no metadata, never as `does_not_fit`."""
    return [p for p in metadata.get("profiles") or [] if isinstance(p, dict)]


def other_records(
    documents: Iterable[tuple[str, dict[str, Any]]],
) -> dict[str, list[OtherRecord]]:
    """FitOptions.other_records from the Metadata API documents of other images of one repository,
    given as (image tag, document) in the order a why line should name them: by profile id, each
    record that names a GPU the profile was built for or tested on, with its workspace_hash."""
    found: dict[str, list[OtherRecord]] = {}
    for image, metadata in documents:
        for profile in usable_profiles(metadata):
            pid = profile.get("profile_id")
            built = tuple(_ids(profile.get("gpu_device")))
            tested = tuple(_ids(profile.get("tested_gpu_devices")))
            if isinstance(pid, str) and pid and (built or tested):
                record = OtherRecord(image, built, tested, _workspace(profile))
                found.setdefault(pid, []).append(record)
    return found


def recorded_everywhere(
    metadata: dict[str, Any], env: dict[str, Any]
) -> dict[str, list[OtherRecord]]:
    """FitOptions.other_records as if another image recorded every profile of `metadata`, by its
    id and workspace_hash, as tested on every GPU of `env`: the most that the records of other
    images can change in what `evaluate` gives (a record only makes a profile `tested`, and never
    one with no workspace_hash). The list compares the two to know whether it must read the other
    images select reads."""
    ours = tuple(
        dict.fromkeys(
            gpu_id
            for group in env.get("groups") or []
            if isinstance(group, dict)
            for gpu_id in _group_ids(group)
        )
    )
    found: dict[str, list[OtherRecord]] = {}
    for profile in usable_profiles(metadata):
        pid, workspace = profile.get("profile_id"), _workspace(profile)
        if isinstance(pid, str) and pid and workspace:
            found.setdefault(pid, []).append(OtherRecord("another image", (), ours, workspace))
    return found


def _evaluate_group(
    metadata: dict[str, Any], env: dict[str, Any], group: dict[str, Any], options: FitOptions
) -> _GroupResult:
    profiles = usable_profiles(metadata)
    ours = _group_ids(group)
    available, possible = _available(group, env, options)
    checks: list[_Check] = []
    for profile in profiles:
        check = _Check(profile=profile)
        _check_target(check, group, ours, options.other_records)
        _check_precision(check, group)
        _check_vram(check, group)
        _check_gpus(check, available, possible)
        _check_host(check, metadata, env, group, options)
        _check_features(check, options)
        checks.append(check)
    checks.sort(key=lambda c: _sort_key(c, options))
    targeted = [c for c in checks if not ({"gpu_mismatch", "gpu_name_only"} & set(c.failures))]
    return _GroupResult(
        group=group,
        available=available,
        fits=[c for c in checks if c.verdict == "fits"],
        unknown=[c for c in checks if c.verdict == "unknown"],
        rejected=[c for c in checks if c.verdict == "rejected"],
        candidates=targeted or checks,
    )


def _group_key(result: _GroupResult, options: FitOptions) -> tuple[Any, ...]:
    if result.fits:
        status, best = 0, result.fits[0]
    elif result.unknown:
        status, best = 1, result.unknown[0]
    else:
        return (2, (), -result.available)
    return (status, _sort_key(best, options)[:-1], -result.available)


def _bits(precision: object) -> int | None:
    """Bits per weight of a precision of the --prefer orders; None for one they do not list."""
    return PRECISION_BITS.get(precision.lower()) if isinstance(precision, str) else None


def _beats(bits: int | None, fitting: Sequence[int | None], prefer: Prefer) -> bool:
    """Whether a precision of `bits` is faster (fewer bits; with --prefer accuracy, more accurate:
    more bits) than every one of `fitting`, the precisions of the profiles that fit. Every
    precision compared must be in the --prefer order: one it does not list ranks last, but says
    nothing of speed or accuracy. False when nothing fits."""
    if bits is None or not fitting or None in fitting:
        return False
    sign = 1 if prefer == "throughput" else -1  # the better precision has the lower key
    return all(sign * bits < sign * other for other in fitting if other is not None)


def _alternative(result: _GroupResult, options: FitOptions) -> _Check | None:
    """The profile to offer beside the recommended one, as an explicit choice that stays
    unconfirmed: the best-ranked profile of the same GPU group whose only open check is
    gpu_not_tested (no failure and no other unknown; notes such as pci_inferred are fine), and
    whose precision has fewer bits (--prefer throughput; more with accuracy) than that of every
    profile that fits (_beats). --max-gpus, --need-lora and --backend apply as to any profile: one
    that fails them, or leaves them unknown, is never offered. None when nothing fits or no
    profile qualifies."""
    fitting = [_bits(check.profile.get("precision")) for check in result.fits]
    for check in result.unknown:  # in rank order: the first that qualifies is the best
        bits = _bits(check.profile.get("precision"))
        if check.unknowns == ["gpu_not_tested"] and _beats(bits, fitting, options.prefer):
            return check
    return None


def _alternative_record(check: _Check, recommended: _Check, options: FitOptions) -> dict[str, Any]:
    """The record of the alternative, `confirmed` false, with `offer`: what it gives over the
    recommended profile and what is not known, in words the agent can say (`faster (nvfp4) than the
    recommended bf16, but this image's NGC metadata records no test of it on this GPU`). Both
    precisions are in the --prefer order."""
    order = PRECISION_ORDER[options.prefer]
    precision = order[_rank(order, check.profile.get("precision"))]
    theirs = order[_rank(order, recommended.profile.get("precision"))]
    if check.required == recommended.required:
        better = f"({precision}) than the recommended {theirs}"
    else:
        ours, gpus = _plural(check.required, "GPU"), _plural(recommended.required, "GPU")
        better = f"({precision}, on {ours}) than the recommended {theirs} (on {gpus})"
    offer = (
        f"{BETTER[options.prefer]} {better}, but this image's NGC metadata records no test of it "
        "on this GPU"
    )
    return {"confirmed": False, **_record(check), "offer": offer}


def faster_fit(
    chosen: list[dict[str, Any]], fits: list[dict[str, Any]], prefer: Prefer
) -> dict[str, Any] | None:
    """For select, the alternative found in another image it checked, on the same GPU type: the
    first record of `fits` (that image's profiles that fit, in rank order, every one of them) whose
    precision is faster (more accurate with --prefer accuracy) than that of the recommended profile
    (chosen[0]) and of every other profile that fits in the image select chose (`chosen`, every one
    of them too, in rank order), as `_beats` judges. A precision the --prefer order does not list
    (`int4`) says nothing of speed or accuracy: such a profile of the chosen image is left out of
    the comparison, unless it is the recommended one, which then gets no alternative. A profile
    that fits was built for or tested on this GPU, so it needs no other test; but not one that
    fits there only by the record of a third image (TESTED_ELSEWHERE): the offer says it was
    built for or tested on this GPU in that image, which only that image's own record shows.
    None when none is."""
    if not chosen or _bits(chosen[0].get("precision")) is None:
        return None
    fitting = [bits for record in chosen if (bits := _bits(record.get("precision"))) is not None]
    for record in fits:
        bits = _bits(record.get("precision"))
        own = TESTED_ELSEWHERE not in (record.get("reasons") or ())
        if record.get("verdict") == "fits" and own and _beats(bits, fitting, prefer):
            return record
    return None


def faster_fit_offer(
    record: dict[str, Any], recommended: dict[str, Any], prefer: Prefer, where: str
) -> str:
    """The `offer` of a profile faster_fit found, in words the agent can say: `faster (nvfp4) and
    tested on this GPU, but in the older image 2.1.5`, with the GPU counts when they differ.
    `where` names the image (`the older image 2.1.5`)."""
    order = PRECISION_ORDER[prefer]
    precision = order[_rank(order, record.get("precision"))]
    ours, theirs = int(record["gpus_required"]), int(recommended["gpus_required"])
    gpus = f", on {_plural(ours, 'GPU')} instead of {theirs}" if ours != theirs else ""
    evidence = "built for" if record.get("tier") == "direct" else "tested on"
    return f"{BETTER[prefer]} ({precision}{gpus}) and {evidence} this GPU, but in {where}"


def evaluate(metadata: dict[str, Any], env: dict[str, Any], options: FitOptions) -> dict[str, Any]:
    """Evaluate every profile of `metadata` against every GPU group of `env`; see the module doc."""
    if env.get("gpus_known") is False:
        if env.get("target") == "k8s":
            raise ns.ScriptError(
                "gpus_unknown",
                "no whole GPU can be counted on the cluster's GPU nodes (MIG, GPU sharing, nodes "
                "not Ready or no allocatable nvidia.com/gpu: see the environment's conditions), "
                "so NIM profiles cannot be matched",
                "Ask the user for a context with whole GPUs, or to disable MIG or sharing (or fix "
                "the nodes), then inspect again",
                ns.EXIT_PRECONDITION,
            )
        raise ns.ScriptError(
            "gpus_unknown",
            "MIG or GPU sharing hides whole-GPU capacity, so NIM profiles cannot be matched",
            "Ask the user which whole GPUs to use (inspect with --gpus), or disable MIG/sharing "
            "and inspect again",
            ns.EXIT_PRECONDITION,
        )
    groups = [g for g in env.get("groups") or [] if isinstance(g, dict)]
    if not groups:
        raise ns.ScriptError(
            "no_gpus",
            "the environment lists no GPU group",
            "Run nimselect.py inspect on a machine or cluster with NVIDIA GPUs",
            ns.EXIT_PRECONDITION,
        )

    results = [_evaluate_group(metadata, env, group, options) for group in groups]
    order = sorted(range(len(results)), key=lambda i: (_group_key(results[i], options), i))
    if options.profile is not None:
        # The GPU type stays the one chosen on every profile: chosen on this profile alone, a tie
        # would go to the type with the most GPUs, which the user never confirmed.
        only = [p for p in usable_profiles(metadata) if p.get("profile_id") == options.profile]
        pinned = {**metadata, "profiles": only}
        results = [_evaluate_group(pinned, env, group, options) for group in groups]
    best = results[order[0]]
    top = max(options.top, 0)

    notes: list[str] = []
    all_best = best.fits + best.unknown + best.rejected
    if any("pci_inferred" in c.notes for c in best.fits + best.unknown):
        notes.append(f"Some profiles {PCI_INFERRED_NOTE}.")
    if any(TESTED_ELSEWHERE in c.notes for c in best.fits + best.unknown):
        notes.append(TESTED_ELSEWHERE_NOTE)
    if env.get("target") == "k8s":
        notes.append(
            "GPU usage is unknown on Kubernetes; available GPUs are the allocatable GPUs of "
            "one node."
        )
    if any(
        "precision_unsupported" in c.failures and str(c.profile.get("precision")).lower() == "nvfp4"
        for c in all_best
    ):
        notes.append(NVFP4_NOTE)
    if any("capability_unknown" in c.unknowns for c in all_best):
        notes.append(
            "Compute capability is unknown, so FP8/NVFP4/INT4 AWQ limits were not checked."
        )
    if not best.fits and best.unknown and "gpu_not_tested" in best.unknown[0].unknowns:
        notes.append(UNTESTED_NOTE)
    if len(groups) > 1:
        notes.append(
            f"The environment has {len(groups)} GPU types; a NIM uses GPUs of one type, so each "
            "type was evaluated on its own."
        )

    counts: dict[str, int] = {}
    for check in best.rejected:
        for code in check.failures:
            counts[code] = counts.get(code, 0) + 1

    not_recorded = sorted(
        name
        for name in REQUIREMENT_FIELDS
        if not any(_has(c, metadata, name) for c in best.candidates)
    )

    if best.fits:
        verdict = "fits"
    elif best.unknown:
        verdict = "unknown"
    else:
        verdict = "does_not_fit"
    alternative = _alternative(best, options)

    return {
        "verdict": verdict,
        "group": {**best.group, "available_gpus": best.available},
        "groups": [
            {
                "name": r.group.get("name"),
                "pci_device_id": r.group.get("pci_device_id"),
                # more than one id (a product label two ids share): pci_device_id is None
                "pci_device_ids": _group_ids(r.group),
                "count": r.group.get("count"),
                "available_gpus": r.available,
                "fits": len(r.fits),
                "unknown": len(r.unknown),
            }
            for r in results
        ],
        "recommended": _record(best.fits[0]) if best.fits else None,
        "shortlist": [_record(c) for c in best.fits[:top]],
        "best_unconfirmed": _record(best.unknown[0]) if best.unknown and not best.fits else None,
        "unconfirmed": [_record(c) for c in best.unknown[:top]],
        "alternative": (
            _alternative_record(alternative, best.fits[0], options) if alternative else None
        ),
        "rejected": {
            "total": len(best.rejected),
            "counts": dict(sorted(counts.items())),
            "examples": [_record(c) for c in best.rejected[:5]],
        },
        "not_recorded": not_recorded,
        "profiles_total": len(all_best),
        "notes": notes,
    }


def _has(check: _Check, metadata: dict[str, Any], name: str) -> bool:
    if name == "release_details.architecture":
        return bool(_architectures(check.profile, metadata))
    return check.profile.get(name) is not None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fit evaluation for nim-select (a library); `nimselect.py fit` runs it.",
        epilog='Example: python3 -I "<SKILL_DIR>/scripts/nimselect.py" fit '
        "meta/llama-3.1-8b-instruct --env ~/.cache/nim-select/env.json",
    )
    parser.parse_args(argv)
    ns.emit(True, "library", "done", {"note": "import this module; run nimselect.py fit instead"})


if __name__ == "__main__":
    main()
