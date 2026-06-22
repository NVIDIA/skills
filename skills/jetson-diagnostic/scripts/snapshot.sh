#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# snapshot.sh
#
# All-in-one Jetson health snapshot. Read-only with respect to system state;
# uses `sudo -n` only to query privileged status when available.
#
# Usage:
#   snapshot.sh [--human] [--tegra-secs N] [--top-procs N]
#
# Outputs: JSON to stdout (default) or pretty-printed JSON with --human.
#
# Exit codes:
#   0  ok
#   2  not on a Jetson

set -uo pipefail

HUMAN=0
TEGRA_SECS=3
TOP_PROCS=10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
# shellcheck disable=SC1091
. "$SCRIPT_DIR/common.sh"

while [ $# -gt 0 ]; do
    case "$1" in
        --human)         HUMAN=1; shift ;;
        --tegra-secs)    need_value "$@"; TEGRA_SECS="$2"; shift 2 ;;
        --top-procs)     need_value "$@"; TOP_PROCS="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 64 ;;
    esac
done

# shellcheck source=detect_jetson.sh
# shellcheck disable=SC1091
. "$SCRIPT_DIR/detect_jetson.sh" || exit $?
# shellcheck source=runtime_probes.sh
# shellcheck disable=SC1091
. "$SCRIPT_DIR/runtime_probes.sh"

: "${JETSON_SKU:=unknown}"
: "${JETSON_GENERATION:=unknown}"
: "${JETSON_PRODUCT_LINE:=unknown}"
: "${JETSON_VARIANT:=unknown}"
: "${JETSON_MEM_GB:=null}"
: "${JETSON_L4T_VERSION:=unknown}"
: "${JETSON_PRODUCT_MODEL:=unknown}"

json_escape() {
    python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().rstrip("\n")))' 2>/dev/null \
        || sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e ':a;N;$!ba;s/\n/\\n/g'
}

# --- Memory ---
MEM_TOTAL_KB=$(read_meminfo_kb MemTotal)
MEM_AVAIL_KB=$(read_meminfo_kb MemAvailable)
MEM_FREE_KB=$(read_meminfo_kb MemFree)
SWAP_TOTAL_KB=$(read_meminfo_kb SwapTotal)
SWAP_FREE_KB=$(read_meminfo_kb SwapFree)
CACHED_KB=$(read_meminfo_kb Cached)

# --- Tegrastats sample ---
TEGRA_LINE=""
if command -v tegrastats >/dev/null 2>&1; then
    TEGRA_LINE=$(timeout "$TEGRA_SECS" tegrastats --interval 1000 2>/dev/null | tail -n 1)
fi
TEGRA_LINE_JSON=$(printf '%s' "$TEGRA_LINE" | json_escape)

# --- Thermal zones ---
THERMAL_PARTS=()
if [ -d /sys/class/thermal ]; then
    for zone in /sys/class/thermal/thermal_zone*; do
        [ -r "$zone/type" ] || continue
        [ -r "$zone/temp" ] || continue
        type=$(cat "$zone/type" 2>/dev/null)
        temp=$(cat "$zone/temp" 2>/dev/null)
        # Most Jetson zones report milli-degrees.
        c=$(awk -v t="$temp" 'BEGIN { printf "%.1f", t/1000.0 }')
        # Skip negative or absurd readings.
        case "$c" in -*|0.0) continue ;; esac
        THERMAL_PARTS+=("\"$type\":$c")
    done
fi
THERMAL_JSON="{$(IFS=,; echo "${THERMAL_PARTS[*]:-}")}"

# --- Power model ---
NVPMODEL_ID="null"
NVPMODEL_NAME="\"unknown\""
if command -v nvpmodel >/dev/null 2>&1; then
    out=$(sudo -n nvpmodel -q 2>/dev/null || nvpmodel -q 2>/dev/null || true)
    if [ -n "$out" ]; then
        id=$(printf '%s\n' "$out" | awk '/NV Power Mode/ {getline; print $1; exit}')
        name=$(printf '%s\n' "$out" | awk -F': ' '/NV Power Mode/ {print $2; exit}')
        [ -n "$id" ] && NVPMODEL_ID="$id"
        [ -n "$name" ] && NVPMODEL_NAME="\"$name\""
    fi
fi

# --- Disk ---
# df -P (POSIX): Filesystem 1024-blocks Used Available Capacity% Mounted-on
# $1=source $2=blocks $3=used $4=avail $5=capacity% $6=mount
DISK_PARTS=()
while IFS= read -r line; do
    mount=$(echo "$line" | awk '{print $6}')
    used=$(echo "$line" | awk '{print $5}' | tr -d '%')
    case "$mount" in
        ""|/dev|/dev/*|/proc|/sys|/run|/run/*|/snap/*|/sys/*|/boot/efi) continue ;;
        /var/lib/docker/*) continue ;;
    esac
    case "$used" in ''|*[!0-9]*) continue ;; esac
    DISK_PARTS+=("{\"mount\":\"$mount\",\"used_pct\":$used}")
done < <(df -P 2>/dev/null | tail -n +2)
DISK_JSON="[$(IFS=,; echo "${DISK_PARTS[*]:-}")]"

# --- GPU memory source selection ---
# On Jetsons using the `nvgpu` in-tree driver for CUDA (Orin family today), the
# `nvidia-smi` binary is present but most fields (Memory-Usage, memory.used,
# compute-apps) return [N/A] / "Not Supported" because the SMI CLI is bound to
# the display-only nvidia.ko module and cannot query the compute path.
# On Jetsons using the unified `nvidia.ko` compute driver (Thor family today),
# `nvidia-smi --query-compute-apps=used_memory` enumerates CUDA processes with
# real per-process sizes (though `--query-gpu=memory.used` still reports
# "Not Supported" on some Thor BSPs).
# `nvsmi_useful` lives in jetson-diagnostic/scripts/runtime_probes.sh: it returns true
# only when nvidia-smi reports a non-stub GPU name (no "(nvgpu)" suffix), the
# single capability probe that distinguishes the unified nvidia.ko stack from
# the legacy nvgpu stub-over-nvgpu stack across every L4T we have data for.

# Coerce a possibly non-numeric nvidia-smi field to either an integer or "null".
_nvsmi_int_or_null() {
    local v; v=$(printf '%s' "$1" | tr -d ' ')
    case "$v" in
        ''|*[!0-9]*) printf 'null' ;;
        *)           printf '%s' "$v" ;;
    esac
}

GPU_PROCESSES_JSON="[]"
GPU_DEVICES_JSON="[]"
# `gpu_source` is the *specific* datum used to attribute GPU memory:
#   "nvidia-smi:compute-apps"   per-process `used_memory` from
#                               `nvidia-smi --query-compute-apps` (unified
#                               nvidia.ko stack, e.g. Thor).
#   "nvmap:iovmm-clients"       per-process sizes from
#                               /sys/kernel/debug/nvmap/iovmm/clients
#                               (nvgpu stack, e.g. Orin).
#   "none"                      no source reachable (typically unprivileged
#                               invocation without sudo, and no nvidia-smi).
GPU_SOURCE="none"
if nvsmi_useful; then
    GPU_SOURCE="nvidia-smi:compute-apps"
    if NVIDIA_SMI_JSON=$(nvidia-smi \
        --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', *' '
            BEGIN { printf "["; first=1 }
            NF>=3 {
                pid=$1; mib=$NF;
                # Re-join any middle commas that the process name might contain.
                cmd="";
                for (i=2; i<NF; i++) cmd=(cmd=="" ? $i : cmd ", " $i);
                n=split(cmd, a, "/"); if (n>1) cmd=a[n];
                gsub(/"/, "\\\"", cmd);
                if (!first) printf ",";
                printf "{\"pid\":%d,\"cmd\":\"%s\",\"used_mib\":%d}", pid, cmd, mib;
                first=0;
            }
            END { printf "]" }
        ' 2>/dev/null); then
        GPU_PROCESSES_JSON="${NVIDIA_SMI_JSON:-[]}"
    else
        GPU_PROCESSES_JSON="[]"
    fi
    # memory.{total,used,free} report "Not Supported" / [N/A] on some BSPs even
    # when compute-apps work; emit null for any non-numeric field rather than
    # forcing a fake zero.
    if NVIDIA_SMI_JSON=$(nvidia-smi \
        --query-gpu=index,name,memory.total,memory.used,memory.free \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', *' '
            function intornull(s,   v) {
                v=s; gsub(/ /, "", v);
                if (v ~ /^[0-9]+$/) return v;
                return "null";
            }
            BEGIN { printf "["; first=1 }
            NF>=5 {
                gsub(/"/, "\\\"", $2);
                if (!first) printf ",";
                printf "%s%s%s%s%s%s%s%s%s%s%s",
                    "{\"index\":", $1, ",\"name\":\"", $2,
                    "\",\"memory_total_mib\":", intornull($3),
                    ",\"memory_used_mib\":", intornull($4),
                    ",\"memory_free_mib\":", intornull($5), "}";
                first=0;
            }
            END { printf "]" }
        ' 2>/dev/null); then
        GPU_DEVICES_JSON="${NVIDIA_SMI_JSON:-[]}"
    else
        GPU_DEVICES_JSON="[]"
    fi
fi

# --- NvMap (authoritative per-process GPU-memory source on nvgpu-stack Jetsons;
# harmless but near-empty on unified-driver Jetsons where nvidia-smi is used) ---
NVMAP_TOTAL_KB=0
NVMAP_TOP_JSON="[]"
NVMAP_STATS_TOTAL_BYTES="null"
NVMAP_PATH=/sys/kernel/debug/nvmap/iovmm/clients
NVMAP_STATS_PATH=/sys/kernel/debug/nvmap/stats/total_memory
NVMAP_READABLE=false
if [ -r "$NVMAP_PATH" ]; then
    NVMAP_READABLE=true
    # Only promote nvmap to the primary gpu_source if nvidia-smi didn't claim it.
    [ "$GPU_SOURCE" = "none" ] && GPU_SOURCE="nvmap:iovmm-clients"
    # iovmm/clients layout (all Jetsons today):
    #   CLIENT                        PROCESS      PID        SIZE
    #   user                  VLLM::EngineCor  1792595   14617792K
    #   ...
    #   total                                              15319584K
    # The SIZE column is documented as bytes with a K/M (and historically G)
    # suffix. Today's kernels emit K, but we normalize to KB defensively so
    # an M/G value cannot cause a 1024x under-count.
    NVMAP_TOTAL_KB=$(awk '$1=="total" {
            s=$NF;
            if      (s ~ /G$/) { gsub("G","",s); s=s*1024*1024 }
            else if (s ~ /M$/) { gsub("M","",s); s=s*1024 }
            else               { gsub("K","",s) }
            print s; exit
        }' "$NVMAP_PATH" 2>/dev/null)
    case "$NVMAP_TOTAL_KB" in ''|*[!0-9]*) NVMAP_TOTAL_KB=0 ;; esac
    NVMAP_TOP_JSON=$(awk '
            NR==1 { next }
            $1=="total" { exit }
            NF>=4 && $3 ~ /^[0-9]+$/ {
                pid=$3; cmd=$2; size=$NF;
                if      (size ~ /G$/) { gsub("G","",size); size=size*1024*1024 }
                else if (size ~ /M$/) { gsub("M","",size); size=size*1024 }
                else                  { gsub("K","",size) }
                if (size+0 == 0) next;
                printf "%d\t%s\t%s\n", size, pid, cmd;
            }
        ' "$NVMAP_PATH" 2>/dev/null \
        | sort -k1 -nr \
        | head -n "$TOP_PROCS" \
        | awk '
            BEGIN { printf "["; first=1 }
            {
                gsub(/"/, "\\\"", $3);
                if (!first) printf ",";
                printf "{\"pid\":%s,\"cmd\":\"%s\",\"kb\":%s}", $2, $3, $1;
                first=0;
            }
            END { printf "]" }
        ' 2>/dev/null || echo "[]")
    [ -z "$NVMAP_TOP_JSON" ] && NVMAP_TOP_JSON="[]"
fi
if [ -r "$NVMAP_STATS_PATH" ]; then
    v=$(cat "$NVMAP_STATS_PATH" 2>/dev/null)
    case "$v" in [0-9]*) NVMAP_STATS_TOTAL_BYTES="$v" ;; esac
fi

# --- Top PSS processes ---
TOP_PROC_JSON="[]"
if command -v procrank >/dev/null 2>&1; then
    # Capture sudo output separately so pipefail doesn't confuse sudo's exit
    # code (1 = password required) with awk's, which would cause "|| echo"
    # to fire and produce the invalid "[][]" double-array.
    _procrank_out=$(sudo -n procrank 2>/dev/null) || true
    TOP_PROC_JSON=$(printf '%s\n' "$_procrank_out" | awk -v n="$TOP_PROCS" '
        BEGIN {first=1; c=0; printf "["}
        NR>1 && NF>=6 && c<n {
            pid=$1; pss=$4; cmd=$NF;
            gsub("K","",pss);
            if (!first) printf ",";
            printf "{\"pid\":%s,\"pss_kb\":%s,\"cmd\":\"%s\"}", pid, pss, cmd;
            first=0; c++;
        }
        END {printf "]"}
    ')
else
    # Fallback: smaps_rollup walk. Slower but no extra binary needed.
    TOP_PROC_JSON=$(python3 - "$TOP_PROCS" <<'PY' 2>/dev/null || echo "[]"
import json, os, sys
n = int(sys.argv[1])
rows = []
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            pss_kb = 0
            for line in f:
                if line.startswith("Pss:"):
                    pss_kb = int(line.split()[1])
                    break
        with open(f"/proc/{pid}/comm") as f:
            comm = f.read().strip()
        rows.append({"pid": int(pid), "pss_kb": pss_kb, "cmd": comm})
    except (OSError, ValueError):
        continue
rows.sort(key=lambda r: r["pss_kb"], reverse=True)
print(json.dumps(rows[:n]))
PY
)
fi

# --- Service inventory (subset commonly relevant) ---
# Note: `systemctl is-active`/`is-enabled` for unit aliases or not-found units can
# emit two tokens ("not-found\ndisabled"); keep only the first line so the JSON
# values stay single-token.
SERVICES=(gdm3 gdm lightdm sddm display-manager pulseaudio bluetooth ModemManager \
          cups cups-browsed snapd whoopsie kerneloops avahi-daemon \
          unattended-upgrades packagekit nvargus-daemon containerd docker)
SVC_PARTS=()
for svc in "${SERVICES[@]}"; do
    state=$(systemctl is-active "$svc" 2>/dev/null) || true; state="${state%%$'\n'*}"
    enabled=$(systemctl is-enabled "$svc" 2>/dev/null) || true; enabled="${enabled%%$'\n'*}"
    : "${state:=inactive}"; : "${enabled:=disabled}"
    SVC_PARTS+=("\"$svc\":{\"active\":\"$state\",\"enabled\":\"$enabled\"}")
done
SVC_JSON="{$(IFS=,; echo "${SVC_PARTS[*]}")}"

DEFAULT_TARGET=$(systemctl get-default 2>/dev/null || echo unknown)

# --- Emit JSON ---
PRODUCT_MODEL_JSON=$(printf '%s' "$JETSON_PRODUCT_MODEL" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')

PAYLOAD=$(cat <<EOF
{
  "sku": "${JETSON_SKU}",
  "generation": "${JETSON_GENERATION}",
  "product_line": "${JETSON_PRODUCT_LINE}",
  "variant": "${JETSON_VARIANT}",
  "mem_total_gb": ${JETSON_MEM_GB},
  "l4t_version": "${JETSON_L4T_VERSION}",
  "product_model": "${PRODUCT_MODEL_JSON}",
  "default_systemd_target": "${DEFAULT_TARGET}",
  "memory_kb": {
    "total": ${MEM_TOTAL_KB:-0},
    "available": ${MEM_AVAIL_KB:-0},
    "free": ${MEM_FREE_KB:-0},
    "cached": ${CACHED_KB:-0},
    "swap_total": ${SWAP_TOTAL_KB:-0},
    "swap_free": ${SWAP_FREE_KB:-0}
  },
  "tegrastats_sample": ${TEGRA_LINE_JSON:-"\"\""},
  "thermal_c": ${THERMAL_JSON},
  "power": { "nvpmodel_id": ${NVPMODEL_ID}, "nvpmodel_name": ${NVPMODEL_NAME} },
  "disk": ${DISK_JSON},
  "gpu_source": "${GPU_SOURCE}",
  "gpu_devices": ${GPU_DEVICES_JSON},
  "gpu_processes": ${GPU_PROCESSES_JSON},
  "nvmap": {
    "readable": ${NVMAP_READABLE},
    "total_kb": ${NVMAP_TOTAL_KB:-0},
    "stats_total_bytes": ${NVMAP_STATS_TOTAL_BYTES},
    "top_clients": ${NVMAP_TOP_JSON}
  },
  "top_processes": ${TOP_PROC_JSON},
  "candidate_services": ${SVC_JSON}
}
EOF
)

if [ "$HUMAN" = "1" ] && command -v python3 >/dev/null 2>&1; then
    printf '%s' "$PAYLOAD" | python3 -m json.tool
else
    printf '%s\n' "$PAYLOAD"
fi
