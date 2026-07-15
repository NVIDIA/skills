# Design notes — jetson-memory-audit

## Why audit.sh delegates to snapshot.sh

`audit.sh` was originally written as a standalone memory snapshot. It duplicated
logic from `jetson-diagnostic/scripts/snapshot.sh` and diverged in two important
ways that caused it to produce wrong or incomplete results on some Jetsons.

### 1. No nvidia-smi capability probe (Thor and later)

`audit.sh` always read NvMap from `/sys/kernel/debug/nvmap/iovmm/clients` as the
GPU memory source. On Jetsons using the unified `nvidia.ko` driver (Thor T5000/T4000),
NvMap debugfs is present but empty — the authoritative source is
`nvidia-smi --query-compute-apps`. Without the capability probe, `audit.sh`
silently reported 0 KB of GPU memory on Thor, giving a misleading picture of
what was actually holding memory.

`snapshot.sh` introduced a probe — `nvsmi_useful()` — that distinguishes the two
driver stacks at runtime:

```bash
nvsmi_useful() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    local name
    name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
    case "$name" in
        ''|*'(nvgpu)'*) return 1 ;;   # nvgpu stub — use NvMap
        *)              return 0 ;;   # unified nvidia.ko — use nvidia-smi
    esac
}
```

This correctly handles:
- **Thor** (T5000/T4000): `nvidia-smi` returns real GPU names → use compute-apps
- **AGX/NX/Nano Orin**: `nvidia-smi` returns `"<model> (nvgpu)"` → fall back to NvMap

### 2. NvMap parser missed the column layout used by newer L4T

The original `audit.sh` NvMap parser keyed on `$0 ~ /^[ ]*[0-9]+[ ]+/` (leading
PID). The actual `/sys/kernel/debug/nvmap/iovmm/clients` layout on L4T 36+ is:

```
CLIENT                        PROCESS      PID        SIZE
user                  VLLM::EngineCor  1792595   14617792K
```

The `CLIENT` column (`user`) comes before `PROCESS`, so the PID is in column 3,
not column 1. The original parser extracted wrong fields, resulting in garbage
PIDs and sizes. `snapshot.sh` parses by column position (`$3 ~ /^[0-9]+$/`) and
handles the `K`/`M`/`G` suffixes correctly across all L4T versions.

### 3. smaps_rollup fallback missing

`audit.sh` used `procrank` only. `snapshot.sh` adds a Python-based
`/proc/[pid]/smaps_rollup` fallback when `procrank` is not installed, meaning
top-process data is available on any Jetson regardless of whether `procrank` is
in the PATH.

## Current approach

`audit.sh` now calls `snapshot.sh` and extracts the memory-relevant subset via
`jq`. The unique value of this skill is `drop_caches.sh` and the before/after
comparison workflow — not the snapshot itself.
