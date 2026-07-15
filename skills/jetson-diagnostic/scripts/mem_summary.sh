#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# mem_summary.sh
#
# One-line human-readable "RAM used + GPU used + swap" summary for a Jetson.
# Fast, read-only, no JSON. Auto-selects the GPU-memory source via the same
# capability probe as snapshot.sh:
#   - unified `nvidia.ko` driver (e.g. Thor): `nvidia-smi --query-gpu=memory.used`
#   - `nvgpu` driver (e.g. Orin): /sys/kernel/debug/nvmap/stats/total_memory
#                                or the "total" line of iovmm/clients
#
# Usage:
#   mem_summary.sh           # one-line summary with [source] tag
#   mem_summary.sh --short   # drop the [source] tag for a more compact line
#   mem_summary.sh --watch   # refresh every 2 seconds (combines with --short)
#
# Exit codes:
#   0  ok
#   2  /proc/meminfo unreadable (should not happen on Linux)
#
# Notes:
#   - `nvidia-smi` does not need root.
#   - Reading `/sys/kernel/debug/nvmap/...` typically requires root. If run
#     unprivileged on an Orin-family box, the GPU column will read
#     "(needs sudo)".

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
# shellcheck disable=SC1091
. "$SCRIPT_DIR/common.sh"
# shellcheck source=runtime_probes.sh
# shellcheck disable=SC1091
. "$SCRIPT_DIR/runtime_probes.sh"

WATCH=0
SHORT=0
INTERVAL=2

while [ $# -gt 0 ]; do
    case "$1" in
        --watch)        WATCH=1; shift ;;
        --short)        SHORT=1; shift ;;
        --interval)     need_value "$@"; INTERVAL="$2"; shift 2 ;;
        -h|--help)      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 64 ;;
    esac
done

case "$INTERVAL" in ''|*[!0-9.]*) echo "--interval must be numeric" >&2; exit 64 ;; esac

gib() { awk -v k="$1" 'BEGIN { printf "%.1f", k/1024/1024 }'; }
pct() { awk -v a="$1" -v b="$2" 'BEGIN { if (b+0>0) printf "%.1f", a*100/b; else printf "0.0" }'; }

render_line() {
    local mem_total_kb mem_avail_kb mem_used_kb
    mem_total_kb=$(read_meminfo_kb MemTotal)
    mem_avail_kb=$(read_meminfo_kb MemAvailable)
    if [ -z "$mem_total_kb" ] || [ -z "$mem_avail_kb" ]; then
        echo "mem_summary: /proc/meminfo unreadable" >&2
        return 2
    fi
    mem_used_kb=$((mem_total_kb - mem_avail_kb))

    local swap_total_kb swap_free_kb swap_used_kb
    swap_total_kb=$(read_meminfo_kb SwapTotal)
    swap_free_kb=$(read_meminfo_kb SwapFree)
    swap_used_kb=$(( ${swap_total_kb:-0} - ${swap_free_kb:-0} ))

    # The tag in brackets names the *specific* datum we read, not just the
    # tool, so the reader can tell what's being summed.
    #   sum(nvidia-smi compute-apps)  Thor / unified nvidia.ko: sum of
    #                                 per-process `used_memory` from
    #                                 `nvidia-smi --query-compute-apps`.
    #                                 (nvidia-smi's device-level memory.used
    #                                 query returns [N/A] on this BSP, so we
    #                                 sum the per-process list instead.)
    #   nvmap stats_total_memory      Orin / nvgpu: bytes from
    #                                 /sys/kernel/debug/nvmap/stats/total_memory
    #                                 (kernel-side allocator total).
    #   sum(nvmap iovmm/clients)      Orin fallback: sum of per-process KB
    #                                 from /sys/kernel/debug/nvmap/iovmm/clients.
    local gpu_used_str="(unknown)" gpu_source=""
    if nvsmi_useful; then
        local mib
        mib=$(nvidia-smi --query-compute-apps=used_memory \
                         --format=csv,noheader,nounits 2>/dev/null \
              | awk 'BEGIN{s=0} /^[0-9]+$/ {s+=$1} END{print s+0}')
        case "$mib" in
            ''|*[!0-9]*) : ;;
            *)
                gpu_used_str="$(gib $((mib * 1024))) GiB"
                gpu_source="sum(nvidia-smi compute-apps)"
                ;;
        esac
    fi
    if [ -z "$gpu_source" ] && [ -r /sys/kernel/debug/nvmap/stats/total_memory ]; then
        local b
        b=$(cat /sys/kernel/debug/nvmap/stats/total_memory 2>/dev/null)
        case "$b" in
            ''|*[!0-9]*) : ;;
            *)
                gpu_used_str="$(gib $((b / 1024))) GiB"
                gpu_source="nvmap stats_total_memory"
                ;;
        esac
    fi
    if [ -z "$gpu_source" ] && [ -r /sys/kernel/debug/nvmap/iovmm/clients ]; then
        # SIZE may end with K/M (and historically G). Normalize to KB before
        # feeding gib(), which expects KB.
        local t
        t=$(awk '$1=="total" {
                s=$NF;
                if      (s ~ /G$/) { gsub("G","",s); s=s*1024*1024 }
                else if (s ~ /M$/) { gsub("M","",s); s=s*1024 }
                else               { gsub("K","",s) }
                print s; exit
            }' /sys/kernel/debug/nvmap/iovmm/clients 2>/dev/null)
        case "$t" in
            ''|*[!0-9]*) : ;;
            *)
                gpu_used_str="$(gib "$t") GiB"
                gpu_source="sum(nvmap iovmm/clients)"
                ;;
        esac
    fi

    # If neither path worked, distinguish "no privilege" from "no source".
    #
    # Subtlety: on Jetson /sys/kernel/debug itself is typically mode 700 root,
    # so a non-root invocation can't even stat /sys/kernel/debug/nvmap. That
    # means we can't use "-d /sys/kernel/debug/nvmap" as the privilege probe.
    # If we got here as non-root and nvidia-smi didn't give us a number, the
    # answer is almost always "rerun with sudo" — on an nvgpu-stack Jetson the
    # nvmap debugfs path is the only remaining source, and it's root-only.
    if [ -z "$gpu_source" ]; then
        if [ "$(id -u)" -ne 0 ]; then
            gpu_used_str="(needs sudo)"
        else
            gpu_used_str="(no source)"
        fi
    fi

    local ram_str swap_str gpu_tag=""
    ram_str="$(gib "$mem_used_kb") / $(gib "$mem_total_kb") GiB ($(pct "$mem_used_kb" "$mem_total_kb")%)"
    swap_str="$(gib "$swap_used_kb") / $(gib "${swap_total_kb:-0}") GiB"
    if [ "$SHORT" != "1" ] && [ -n "$gpu_source" ]; then
        gpu_tag=" [$gpu_source]"
    fi

    printf 'RAM  used %s  |  GPU  %s%s  |  swap  %s\n' \
        "$ram_str" "$gpu_used_str" "$gpu_tag" "$swap_str"
}

if [ "$WATCH" = "1" ]; then
    # Simple watch loop. Clear-screen kept minimal to stay portable.
    while true; do
        line=$(render_line)
        rc=$?
        printf '\r\033[K%s' "$line"
        [ "$rc" -ne 0 ] && { echo; exit "$rc"; }
        sleep "$INTERVAL"
    done
else
    render_line
fi
