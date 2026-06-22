#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
recommend.py

Pick a runtime + memory-related launch flags for an LLM/VLM workload on Jetson.
Reads a jetson-memory-audit JSON to know the SKU, variant, and current memory
state.

Inputs:
    --audit PATH     Path to audit JSON (from jetson-memory-audit). '-' for stdin.
    --runtime NAME   One of: auto | vllm | sglang | llama-cpp | tensorrt-edge.
                     'auto' picks based on SKU + memory headroom.
    --workload NAME  One of: llm-server | vlm-server | embedding | rag.
    --target-mb N    Optional. Desired free DRAM (MB) headroom after the model
                     is loaded. Used to back into gpu-memory-utilization etc.
    --human          Pretty-print JSON.

Outputs: JSON to stdout. See SKILL.md "Output contract" for shape.

Exit codes:
    0  ok
    2  audit JSON malformed
    3  unsupported runtime / workload
"""

from __future__ import annotations

import argparse
import json
import sys

# pylint: disable=missing-function-docstring,too-many-locals

RUNTIMES = {"auto", "vllm", "sglang", "llama-cpp", "tensorrt-edge"}
WORKLOADS = {"llm-server", "vlm-server", "embedding", "rag"}
MIN_GPU_UTIL = 0.30
MAX_GPU_UTIL = 0.90
MIN_FREE_FRACTION = 0.0
FULL_MEMORY_FRACTION = 1.0
MB_PER_GB = 1024
GPU_UTIL_DIGITS = 2

# Per-SKU defaults. Tuned to leave the system responsive while letting the
# server claim most of the GPU pool. Conservative on Nano, more aggressive on
# AGX / Thor.
SKU_DEFAULTS = {
    "orin-nano":  {"gpu_util": 0.50, "max_seqs": 4,  "ngl": 28, "ctx": 4096},
    "orin-nx":    {"gpu_util": 0.60, "max_seqs": 8,  "ngl": 32, "ctx": 4096},
    "orin-agx":   {"gpu_util": 0.70, "max_seqs": 16, "ngl": 80, "ctx": 8192},
    "thor":       {"gpu_util": 0.75, "max_seqs": 32, "ngl": 80, "ctx": 8192},
    "unknown":    {"gpu_util": 0.55, "max_seqs": 4,  "ngl": 24, "ctx": 4096},
}


def pick_runtime(sku: str, mem_gb: int, workload: str) -> tuple[str, str]:
    if workload == "rag":
        return "sglang", "RAG / tool-use flows benefit from SGLang's programmable runtime."
    if sku == "orin-nano" or mem_gb <= 8:
        return (
            "llama-cpp",
            "Tightest memory floor on Orin Nano-class devices using GGUF + 4-bit quantization.",
        )
    if workload in ("llm-server", "vlm-server"):
        return (
            "vllm",
            "Highest throughput at this memory budget given continuous batching + paged attention.",
        )
    if workload == "embedding":
        return (
            "tensorrt-edge",
            "Static-shape embedding workloads benefit from NVIDIA-tuned kernels.",
        )
    return (
        "vllm",
        "Default high-throughput choice when no specific feature pulls another runtime.",
    )


def vllm_flags(d: dict, target_mb: int | None, mem_total_gb: int) -> list[str]:
    util = d["gpu_util"]
    if target_mb:
        # Scale gpu-memory-utilization down so we leave roughly target_mb free.
        free_frac = max(
            MIN_FREE_FRACTION,
            min(MAX_GPU_UTIL, target_mb / max(mem_total_gb * MB_PER_GB, 1)),
        )
        util = round(
            max(MIN_GPU_UTIL, min(MAX_GPU_UTIL, FULL_MEMORY_FRACTION - free_frac)),
            GPU_UTIL_DIGITS,
        )
    return [
        f"--gpu-memory-utilization={util}",
        f"--max-model-len={d['ctx']}",
        f"--max-num-seqs={d['max_seqs']}",
        "--enable-prefix-caching",
    ]


def sglang_flags(d: dict) -> list[str]:
    return [
        f"--mem-fraction-static={d['gpu_util']}",
        f"--max-running-requests={d['max_seqs']}",
    ]


def llama_cpp_flags(d: dict) -> list[str]:
    return [
        f"-ngl {d['ngl']}",
        f"-c {d['ctx']}",
        "--no-mmap",
    ]


def tensorrt_edge_flags(d: dict) -> list[str]:
    return [
        f"--max-batch-size={d['max_seqs']}",
        f"--max-input-len={d['ctx']}",
        "--paged-kv-cache=enable",
    ]


def build_flags(
    runtime: str,
    d: dict,
    target_mb: int | None,
    mem_total_gb: int,
) -> list[str]:
    if runtime == "vllm":
        return vllm_flags(d, target_mb, mem_total_gb)
    if runtime == "sglang":
        return sglang_flags(d)
    if runtime == "llama-cpp":
        return llama_cpp_flags(d)
    if runtime == "tensorrt-edge":
        return tensorrt_edge_flags(d)
    return []


def normalize_sku(audit: dict) -> str:
    raw_sku = str(audit.get("sku") or audit.get("product_line") or "unknown")
    if raw_sku == "thor-agx":
        return "thor"
    if raw_sku in SKU_DEFAULTS:
        return raw_sku
    return "unknown"


def audit_int(audit: dict, key: str) -> int:
    try:
        return int(audit.get(key, 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid audit field {key!r}") from exc


def load_audit(path: str) -> dict:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_runtime(args: argparse.Namespace, sku: str, mem_total_gb: int) -> tuple[str, str]:
    if args.runtime == "auto":
        return pick_runtime(sku, mem_total_gb, args.workload)
    return args.runtime, f"User-selected runtime '{args.runtime}'."


def build_alternatives(defaults: dict, runtime: str, target_mb: int | None,
                       mem_total_gb: int) -> list[dict]:
    rationale = {
        "llama-cpp": "Lowest memory floor (GGUF + 4-bit).",
        "vllm": "Higher throughput with continuous batching.",
        "sglang": "Programmable flows: RAG, tool use, structured output.",
        "tensorrt-edge": "NVIDIA-tuned static-shape kernels.",
    }
    alternatives = []
    for alt in ("llama-cpp", "vllm", "sglang", "tensorrt-edge"):
        if alt != runtime:
            alternatives.append({
                "runtime": alt,
                "rationale": rationale[alt],
                "launch_flags": build_flags(alt, defaults, target_mb, mem_total_gb),
            })
    return alternatives


def quantization_notes(runtime: str, sku: str) -> list[str]:
    notes: list[str] = []
    if runtime in ("vllm", "sglang") and sku == "thor":
        notes.append("Thor: prefer NVFP4 when supported; use W4A16 as fallback.")
    elif runtime in ("vllm", "sglang") and sku in ("orin-nano", "orin-nx"):
        notes.append("Orin Nano/NX: prefer W4A16; use AWQ/GPTQ 4-bit as fallback.")
    elif runtime in ("vllm", "sglang") and sku == "orin-agx":
        notes.append("AGX Orin: prefer W4A16; use AWQ/GPTQ 4-bit as fallback.")
    elif runtime == "llama-cpp":
        notes.append(
            "llama.cpp/Ollama: prefer GGUF INT4/Q4_K_M on Orin and Thor; "
            "choose a smaller INT4 GGUF model if memory is tight."
        )
    return notes


def l4t_major(l4t: str) -> int:
    try:
        return int(str(l4t).split(".", maxsplit=1)[0])
    except (TypeError, ValueError):
        return 0


def runtime_path_notes(runtime: str, sku: str, l4t: str) -> list[str]:
    if sku == "thor" and runtime == "vllm":
        return [
            "Thor vLLM path: use upstream vLLM 0.20+ via vllm/vllm-openai or a "
            "validated native vLLM 0.20+ install; pre-0.20 vLLM is not a valid "
            "Thor test. Do not use the Orin GHCR image."
        ]
    if (sku == "orin" or sku.startswith("orin-")) and runtime == "vllm":
        if l4t_major(l4t) >= 39:
            return [
                "Orin JetPack 7.2 / L4T r39+ vLLM path: use upstream "
                "vLLM 0.20+ via vllm/vllm-openai."
            ]
        return [
            "Orin vLLM path before JetPack 7.2 / L4T r39: prefer "
            "ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin."
        ]
    if sku == "thor" and runtime == "sglang":
        return [
            "Thor SGLang path: use NVIDIA SGLang 26.01 "
            "(nvcr.io/nvidia/sglang:26.01-py3), which includes SGLang "
            "0.5.5.post2 and lists Jetson Thor support. Do not judge Thor "
            "support from older prerelease SGLang results."
        ]
    return []


def build_notes(args: argparse.Namespace, runtime: str, sku: str, l4t: str) -> list[str]:
    notes = quantization_notes(runtime, sku)
    if args.target_mb and runtime == "vllm":
        notes.append(f"--gpu-memory-utilization scaled to leave ~{args.target_mb} MB free.")
    notes.extend(runtime_path_notes(runtime, sku, l4t))
    return notes


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--audit", required=True)
    p.add_argument("--runtime", default="auto", choices=sorted(RUNTIMES))
    p.add_argument("--workload", default="llm-server", choices=sorted(WORKLOADS))
    p.add_argument("--target-mb", type=int, default=None)
    p.add_argument("--human", action="store_true")
    args = p.parse_args()

    try:
        audit = load_audit(args.audit)
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: failed to read audit JSON: {e}", file=sys.stderr)
        return 2

    try:
        sku = normalize_sku(audit)
        variant = str(audit.get("variant", "unknown"))
        l4t = str(audit.get("l4t_version", "unknown"))
        mem_total_gb = audit_int(audit, "mem_total_gb")
    except ValueError as e:
        print(f"ERROR: malformed audit JSON: {e}", file=sys.stderr)
        return 2

    defaults = SKU_DEFAULTS.get(sku, SKU_DEFAULTS["unknown"])
    runtime, rationale = resolve_runtime(args, sku, mem_total_gb)
    flags = build_flags(runtime, defaults, args.target_mb, mem_total_gb)
    alternatives = build_alternatives(defaults, runtime, args.target_mb, mem_total_gb)
    notes = build_notes(args, runtime, sku, l4t)

    out = {
        "sku": sku,
        "variant": variant,
        "l4t_version": l4t,
        "mem_total_gb": mem_total_gb,
        "runtime": runtime,
        "rationale": rationale,
        "launch_flags": flags,
        "alternatives": alternatives,
        "notes": notes,
    }
    json.dump(out, sys.stdout, indent=2 if args.human else None)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
