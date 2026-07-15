---
name: jetson-diagnostic
description: Read-only Jetson health snapshot for identity, memory, GPU, thermal, power, storage, services, and top processes.
version: 0.0.1
license: "Apache-2.0"
metadata:
  author: "Jetson Team"
  tags: [jetson, diagnostic, telemetry]
  languages: [bash]
  data-classification: public
---

# Jetson Diagnostic

A unified, agent-friendly view of a running Jetson device. Replaces the need to remember which of `tegrastats`, `jtop`, `procrank`, `/sys/kernel/debug/nvmap`, `nvpmodel`, `free`, `swapon`, `df`, and `systemctl list-units` produces which slice of the truth.

## Purpose

Capture a read-only health snapshot from the Jetson host so agents can answer device identity, memory, GPU, thermal, power, storage, and service-state questions using live data instead of guesses.

## When to use

Activate when the user asks:

- "What is this Jetson? What SKU? How much memory?"
- "What's running on this Jetson right now?"
- "Why is my Jetson slow / hot / out of memory?"
- "Give me a snapshot of GPU / CPU / power usage."
- "What does my tegrastats output mean?"
- "Which services are running that I could turn off?"
- The user has installed `jetson-memory-audit`, `jetson-headless-mode`, `jetson-inference-mem-tune`, `jetson-llm-benchmark`, `jetson-llm-serve`, or `jetson-package` and needs a baseline measurement before running them.

Do not use this skill to change power modes, drop caches, stop services, install packages, serve models, or tune inference flags. Report the observed state, then hand off to the action-oriented skill.

## Prerequisites

- Run on the Jetson host, or in a sandbox/container with host-visible Jetson system paths and process data.

## Available Scripts

| Script | Purpose | Arguments |
|--------|---------|-----------|
| `scripts/snapshot.sh` | Emits the all-in-one JSON snapshot for identity, memory, GPU, thermal, power, disk, top processes, and candidate services. | `--human`, `--tegra-secs N`, `--top-procs N`. |
| `scripts/mem_summary.sh` | Emits a compact human-readable RAM/GPU/swap summary. | `--short`, `--watch`, `--interval N`. |
| `scripts/detect_jetson.sh` | Exports or prints canonical Jetson SKU/generation/product-line fields for this repo. | No arguments. |

If your agent runtime supports `run_script`, use it to run `scripts/snapshot.sh` or `scripts/mem_summary.sh` and summarize the returned output. Otherwise run the scripts with `bash` from the repository root.

## Instructions

1. Run `scripts/snapshot.sh` for the all-in-one JSON view (preferred default).
2. For a quick human-readable memory line, run `scripts/mem_summary.sh`.
3. To explain a single tegrastats line the user has pasted, see `references/tegrastats-fields.md`.
4. To explain the NvMap clients output, see `references/nvmap-clients.md`.

## Reporting guidance

Run the matching helper script before summarizing device state, and report only fields returned by that script. If direct execution is blocked by the runtime, run it with `bash {baseDir}/scripts/<script-name>` rather than trying to chmod files.

- For "what is this Jetson" questions, quote `product_model` or `sku`, `variant`, `l4t_version`, and `mem_total_gb`.
- For "slow and hot" questions, run `snapshot.sh` and summarize both sides of the symptom: `thermal_c` for heat, plus `top_processes`, `gpu_processes`, `nvmap.top_clients`, or `gpu_source` for load. End with a concrete handoff such as `jetson-memory-audit`, `jetson-headless-mode`, or `jetson-inference-mem-tune`.
- For "which process is using memory" questions, run `snapshot.sh` and name the leading process as `pid <number>`, `cmd`, and its `pss_kb` / MiB value. If NvMap GPU memory is the relevant signal, also quote `gpu_source` and the top `nvmap.top_clients` or `gpu_processes` entry.

If your agent runtime does not automatically execute helper scripts relative to this skill directory, resolve script paths with the AgentSkills `{baseDir}` placeholder:

```bash
{baseDir}/scripts/snapshot.sh
{baseDir}/scripts/mem_summary.sh
```

Do not call `jetson-diagnostic` as a tool name unless the runtime explicitly registers skills as callable tools; Agent Skills are normally instructions plus files, not direct tool functions.

All scripts source the canonical platform detector at `skills/jetson-diagnostic/scripts/detect_jetson.sh` (exports `JETSON_SKU`, `JETSON_GENERATION`, `JETSON_PRODUCT_LINE`, `JETSON_VARIANT`, `JETSON_MEM_GB`, `JETSON_L4T_VERSION`, `JETSON_PRODUCT_MODEL`). Other skills may source this detector rather than duplicating Jetson identification logic. Exits 2 with a remediation message off-platform.

## Limitations

- Seeing this skill file does not guarantee access to Jetson host hardware. If `/proc/device-tree/model`, `/etc/nv_tegra_release`, `tegrastats`, `nvpmodel`, `nvidia-smi`, or `/sys/kernel/debug/nvmap` are missing inside a NemoClaw/OpenClaw sandbox, say the sandbox lacks Jetson host visibility and ask the user to run on the Jetson host or relaunch with a host-visible sandbox profile.
- NvMap debugfs often requires root, so unprivileged runs may report `gpu_source: "none"` or incomplete `nvmap` fields.
- This skill reports observed state only. Do not fabricate memory, GPU, thermal, service, or reclamation data when a tool is missing or inaccessible.

## Error handling

- If a helper exits off-platform, report that the current environment is not a Jetson host or lacks host visibility; do not substitute generic Linux values.
- If `tegrastats`, `nvpmodel`, `nvidia-smi`, or NvMap debugfs are unavailable, preserve the corresponding `null`, `false`, or empty fields from the JSON and explain which signal is limited.
- If `snapshot.sh` emits malformed JSON, report the raw failure and rerun after fixing the helper output; do not hand-edit a synthetic device snapshot.

## Output contract for `snapshot.sh`

```json
{
  "sku": "orin-nano",
  "generation": "orin",
  "product_line": "orin-nano",
  "variant": "orin-nano-8gb",
  "mem_total_gb": 8,
  "l4t_version": "36.4.0",
  "product_model": "nvidia jetson orin nano developer kit",
  "memory_kb": { "total": 8123456, "available": 4123456, "swap_total": 0, "swap_free": 0, "cached": 1234567 },
  "tegrastats_sample": "RAM 4011/8138MB (lfb 8x4MB) ...",
  "thermal_c": { "CPU": 52.3, "GPU": 49.0, "AO": 47.0 },
  "power": { "nvpmodel_id": 0, "nvpmodel_name": "MAXN" },
  "disk": [ { "mount": "/", "used_pct": 41 } ],
  "gpu_source": "nvmap:iovmm-clients",
  "gpu_devices": [],
  "gpu_processes": [],
  "nvmap": {
    "readable": true,
    "total_kb": 654321,
    "stats_total_bytes": 669985280,
    "top_clients": [ { "pid": 1234, "cmd": "vlm-server", "kb": 524288 } ]
  },
  "top_processes": [ { "pid": 4321, "cmd": "vllm", "pss_kb": 4000000 } ],
  "candidate_services": { "gdm3": { "active": "inactive", "enabled": "disabled" } }
}
```

`gpu_source` names the *specific* datum the skill used to attribute per-process GPU memory, so the caller can tell exactly what the numbers represent:

- `"nvidia-smi:compute-apps"` — per-process `used_memory` values from `nvidia-smi --query-compute-apps`. Used on the unified `nvidia.ko` stack (Thor family today). Note: on this stack `nvidia-smi`'s *device-level* `memory.used` query returns `[N/A]` on some BSPs, which is why the skill sums the per-process list rather than reading a top-level total. The summed total appears in `gpu_processes[*].used_mib`.
- `"nvmap:iovmm-clients"` — per-process sizes from `/sys/kernel/debug/nvmap/iovmm/clients`. Used on the `nvgpu` stack (Orin family today), where `nvidia-smi` is a stub that returns `[N/A]` for every compute/memory query. Per-process entries appear in `nvmap.top_clients`; the kernel-side total is in `nvmap.total_kb` and (when readable) `nvmap.stats_total_bytes`.
- `"none"` — no authoritative source reachable. Typical when running unprivileged on an `nvgpu`-stack Jetson (debugfs under `/sys/kernel/debug/nvmap` needs `sudo`); rerun with `sudo` to populate the `nvmap` fields.

The agent should present the salient parts back to the user (SKU, available memory, top GPU consumer per `gpu_source`, hottest zone, power mode) and offer to drill into specifics (`top_processes`, `gpu_processes` / `nvmap`, `services`).

## Safety

This skill is **read-only**. It does not change `nvpmodel`, does not run `jetson_clocks`, does not modify services. To act on findings, hand off to:

- `jetson-memory-audit` — focused memory snapshot + drop_caches verify loop
- `jetson-headless-mode` — disable GUI + auxiliary daemons (safe, reversible)
- `jetson-inference-mem-tune` — pick runtime + memory flags (vLLM / SGLang / llama.cpp / TensorRT Edge-LLM)
- `jetson-llm-serve` — vLLM and related GHCR images with Jetson defaults
- `jetson-llm-benchmark` — reproducible latency / throughput benchmarks
- `jetson-package` — GHCR + Jetson AI Lab PyPI indexes vs generic ARM wheels

## Cross-platform behavior

| Family                | Variants the skill recognises                              | `tegrastats` | `nvidia-smi`          | `nvpmodel` | NvMap debugfs |
|-----------------------|------------------------------------------------------------|--------------|-----------------------|------------|----------------|
| Jetson Thor           | `thor-t5000`, `thor-t4000`                                 | yes          | yes (full)            | yes        | yes (root)     |
| Jetson AGX Orin       | `orin-agx-64gb`, `orin-agx-32gb`, `orin-agx-industrial`    | yes          | yes (stub, `nvgpu`)*  | yes        | yes (root)     |
| Jetson Orin NX        | `orin-nx-16gb`, `orin-nx-8gb`                              | yes          | yes (stub, `nvgpu`)*  | yes        | yes (root)     |
| Jetson Orin Nano      | `orin-nano-8gb`, `orin-nano-4gb`                           | yes          | yes (stub, `nvgpu`)*  | yes        | yes (root)     |

\* On Jetsons whose GPU is driven by the in-tree `nvgpu` kernel driver, the `nvidia-smi` binary is present but most fields (`Memory-Usage`, power, utilisation, compute-process table) report `Not Supported` / `N/A`. To decide which source to trust at runtime, the script does a capability probe — `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` — and only uses `nvidia-smi` for per-process GPU memory when that query returns a real integer. When it doesn't, the script falls back to `/sys/kernel/debug/nvmap/iovmm/clients`, which on `nvgpu`-stack Jetsons is the authoritative per-process GPU-memory source.

The script handles each tool's presence gracefully and reports `null` / `false` for tools it cannot reach (typical when the agent isn't running with the privilege needed for `/sys/kernel/debug`). Variant detection uses the `/proc/device-tree/model` string first (recognising names like `T5000` / `T4000`) and falls back to memory-size heuristics when the model string is generic.
