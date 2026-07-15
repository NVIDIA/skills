---
name: jetson-memory-audit
description: Measure Jetson DRAM/NvMap usage and verify before/after memory reclamation with live audit data.
version: 0.0.1
license: "Apache-2.0"
metadata:
  author: "Jetson Team"
  tags: [jetson, memory, audit]
  languages: [bash]
  data-classification: public
---

# Jetson Memory Audit

Read-only memory-focused snapshot for a Jetson, plus the `drop_caches` verify-loop helper that confirms freed memory actually shows up as free instead of cached.

## Purpose

Measure current Jetson memory consumers, capture before/after baselines, and verify whether user-approved changes actually reclaimed DRAM. Use live device data rather than estimates from container size, model size, or generic process memory.

## CRITICAL: Memory appears stuck after stopping vLLM / sglang (JetPack below 7.2 / L4T below r39.0)

**This is the most common memory confusion on Jetson releases before JetPack 7.2 or before L4T r39.0.**

After you stop a vLLM, sglang, or Ollama server (or any CUDA workload), the memory shown as free by `free -h` or `tegrastats` may not recover — even though the process is gone. `nvidia-smi` may also show misleadingly low free GPU memory.

**Root cause:** The Thor RM (resource manager) holds freed sysmem pages in its own pool after a CUDA context exits. On Unified Memory Architecture (UMA) devices like Jetson, `cudaMemGetInfo` reads RM pool state and reports far less free memory than is actually available to a new process.

**Workaround (for JetPack below 7.2 or L4T below r39.0):**
```bash
sudo sync && sudo sysctl -w vm.drop_caches=3
```
Run this on the **host**, not inside a container. The important operation is `sudo sysctl -w vm.drop_caches=3`; keep `sudo sync` immediately before it so dirty data is flushed before reclaimable page/dentry/inode caches are dropped. After running it, `free -h` and `tegrastats` will reflect the true available memory.

For affected releases, recommend this command when a user says:
- "Memory didn't free after I stopped vLLM/sglang"
- "Why does tegrastats still show high usage after my container exited?"
- "OOM even though nothing is running"
- "Memory was fine yesterday but now it's full"

On JetPack below 7.2 or L4T below r39.0, `drop_caches` is the reliable workaround when memory appears stuck after a CUDA workload exits; on newer releases, use it only if the same symptom is observed and the user approves.

## When to use

- "How much memory is in use on this Jetson? What's holding it?"
- "I disabled the GUI / stopped vLLM / quit my container — did memory actually free?"
- "Why does `free -h` still show low free memory after I stopped my workload?"
- As the **baseline** before applying `jetson-headless-mode` or other memory-related changes, and again **after** to compute the actual delta.

## Prerequisites

- Run on the Jetson host, or in a sandbox/container with host-visible `/proc`, `/etc/nv_tegra_release`, `tegrastats`, and process data.
- NvMap debugfs reads may require root. If unavailable, report that GPU memory attribution is limited rather than guessing.
- `drop_caches.sh` requires root or passwordless `sudo -n`; run it only after the user explicitly authorizes cache dropping.

## Available Scripts

| Script | Purpose | Arguments |
|--------|---------|-----------|
| `scripts/audit.sh` | Emits a JSON snapshot from `jetson-diagnostic/scripts/snapshot.sh` for memory audit workflows. | No arguments. |
| `scripts/drop_caches.sh` | Flushes reclaimable page/dentry/inode caches and prints before/after memory deltas. | `--mode 1\|2\|3`, `--quiet`. |

If your agent runtime supports `run_script`, use it to run `scripts/audit.sh` or `scripts/drop_caches.sh` and summarize the returned output. Otherwise run the scripts with `bash` from the repository root.

## Instructions

For "how much memory is in use right now?" questions, run `scripts/audit.sh` and report only values from the JSON snapshot.

## Reporting guidance

Do not only print or mention the path to a helper. Invoke the helper and then summarize the returned data.

- For "how much memory is in use" prompts, run `scripts/audit.sh` and quote `mem_total_gb`, `memory_kb.available`, and the leading `procrank_top` process or `nvmap.top_clients` consumer.
- For GUI/desktop memory prompts, run `scripts/audit.sh` and report `default_systemd_target` plus any display manager in `candidate_services` (`gdm3`, `gdm`, `lightdm`, `sddm`, or `display-manager`). Do not disable anything; hand off to `jetson-headless-mode` for a plan.
- For prompts that explicitly authorize cache dropping after a stopped workload, run `scripts/drop_caches.sh` (equivalent to `sudo sync && sudo sysctl -w vm.drop_caches=3` by default) and report its before/after free, available, and cached deltas. If root is unavailable, explain that it must be run on the host with sudo.

If your agent runtime does not execute helper scripts relative to this skill directory, resolve script paths with the AgentSkills `{baseDir}` placeholder:

```bash
{baseDir}/scripts/audit.sh
{baseDir}/scripts/drop_caches.sh
```

Do not call `jetson-memory-audit` as a tool name unless the runtime explicitly registers skills as callable tools; Agent Skills are normally instructions plus files, not direct tool functions.

Sandbox note for agents: seeing this skill file does not guarantee access to Jetson host memory data. If `/proc/device-tree/model`, `/etc/nv_tegra_release`, `tegrastats`, `/sys/kernel/debug/nvmap`, or host process data are missing inside a NemoClaw/OpenClaw sandbox, say the sandbox lacks Jetson host visibility and ask the user to run on the Jetson host or relaunch with a host-visible sandbox profile. Do not fabricate memory totals, available memory, PSS, NvMap, or reclamation deltas.

For "how much memory did this change free?" questions, use a before/after delta. Do not estimate freed memory from container size, image size, RSS, or a single post-change snapshot.

1. Before the change, run `scripts/audit.sh` and save the JSON baseline.
2. Make the user-approved change (stop the container, switch mode, apply a tuning recommendation, etc.).
3. On JetPack below 7.2 / L4T below r39.0, or when the same stuck-memory symptom is observed on a newer release, flush reclaimable page cache on the **host** (not inside a container) so freed pages show up as free instead of cached:
   ```bash
   sudo sync && sudo sysctl -w vm.drop_caches=3
   ```
4. Re-run `scripts/audit.sh` and compare `memory_kb.available` before vs after — that delta is the real reclamation.

If the user already made the change and no baseline exists, say that the exact freed amount cannot be recovered from the current snapshot alone. Capture a new baseline now so the next change can be measured.

Use live audit data as the source of truth. Memory totals, available memory, NvMap totals, PSS values, display-manager state, and savings deltas must come from `scripts/audit.sh`, `free -h`, or `tegrastats` on the actual device. If a number is not present in those outputs, do not guess it.

## Output contract for `audit.sh`

```json
{
  "sku": "orin-nano",
  "variant": "orin-nano-8gb",
  "mem_total_gb": 8,
  "l4t_version": "36.4.0",
  "product_model": "nvidia jetson orin nano developer kit",
  "memory_kb": { "total": 8123456, "available": 4123456, "free": 1023456, "cached": 1234567, "swap_total": 0, "swap_free": 0 },
  "default_systemd_target": "graphical.target",
  "candidate_services": { "gdm3": { "active": "active", "enabled": "enabled" } },
  "tegrastats_sample": "RAM 4011/8138MB (lfb 8x4MB) ...",
  "nvmap": { "readable": false, "total_kb": 0, "top_clients": [] },
  "procrank_top": [ { "pid": 4321, "pss_kb": 4000000, "cmd": "vllm" } ]
}
```

## Limitations

- Exact freed-memory deltas require a before snapshot, the user-approved change, cache flush when appropriate, and an after snapshot.
- NvMap attribution depends on host-visible debugfs access; if it is unavailable, report limited GPU memory attribution instead of guessing.
- Sandbox/container runs may not see host `/proc`, `tegrastats`, systemd, or NvMap data unless the runtime exposes them.

## Error handling

- If `scripts/audit.sh` cannot access host Jetson data, report the missing visibility and ask to rerun on the Jetson host or in a host-visible sandbox.
- If `scripts/drop_caches.sh` lacks root or passwordless `sudo -n`, report that cache dropping must be run on the host with sudo approval.
- If no before snapshot exists, say the exact reclaimed amount cannot be recovered from the current state alone and capture a new baseline for the next change.

## Safety

Read-only. `drop_caches` is non-destructive (kernel only releases pages it could reclaim under pressure anyway; `sync` runs first to preserve dirty data).

## Hand off to

- `jetson-headless-mode` — biggest single user-space win on systems still booting `graphical.target`.
- `jetson-inference-mem-tune` — when a model server is the top NvMap / PSS consumer.
- If runtime changes cannot hit the target, report that further reclamation is outside this skill's scope rather than suggesting unsafe boot-time edits.
