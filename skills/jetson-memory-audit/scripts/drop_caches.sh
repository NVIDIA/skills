#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# drop_caches.sh
#
# Flush filesystem buffers and drop the kernel page/dentry/inode caches so
# freed memory shows up as "free" instead of "cached" in `free -h` and
# `tegrastats`. Run this on the *host* (not inside a container) after:
#
#   - disabling services / stopping the GUI stack,
#   - stopping a vLLM / llama.cpp / SGLang container,
#   - swapping a model on or off the device.
#
# Non-destructive: drop_caches only releases reclaimable pages; the leading
# `sync` preserves dirty data. See Documentation/admin-guide/sysctl/vm.rst.
#
# Usage:
#   drop_caches.sh [--mode 1|2|3] [--quiet]
#
#   --mode N     Cache class to drop (default 3).
#                  1 = pagecache only
#                  2 = dentries + inodes
#                  3 = pagecache + dentries + inodes
#   --quiet      Suppress the before/after summary.
#
# Exit codes:
#   0  ok
#   2  not on a Jetson
#   5  no root privileges available

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../../jetson-diagnostic/scripts/common.sh
# shellcheck disable=SC1091
. "$SKILLS_ROOT/jetson-diagnostic/scripts/common.sh"

MODE=3
QUIET=0
while [ $# -gt 0 ]; do
    case "$1" in
        --mode)  need_value "$@"; MODE="$2"; shift 2 ;;
        --quiet) QUIET=1; shift ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 64 ;;
    esac
done
case "$MODE" in 1|2|3) ;; *) echo "ERROR: --mode must be 1, 2, or 3" >&2; exit 64 ;; esac

# shellcheck source=../../jetson-diagnostic/scripts/detect_jetson.sh
# shellcheck disable=SC1091
. "$SKILLS_ROOT/jetson-diagnostic/scripts/detect_jetson.sh"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    if sudo -n true 2>/dev/null; then SUDO="sudo -n"
    else echo "ERROR: this script needs root. Re-run with sudo." >&2; exit 5
    fi
fi

read_kb() { awk -v k="$1" '$1==k":" {print $2}' /proc/meminfo 2>/dev/null; }

BEFORE_FREE=$(read_kb MemFree)
BEFORE_AVAIL=$(read_kb MemAvailable)
BEFORE_CACHED=$(read_kb Cached)

if [ -n "$SUDO" ]; then
    sudo -n sync || { echo "ERROR: sync failed" >&2; exit 5; }
    sudo -n sysctl -w "vm.drop_caches=$MODE" >/dev/null || { echo "ERROR: failed to set vm.drop_caches=$MODE" >&2; exit 5; }
else
    sync || { echo "ERROR: sync failed" >&2; exit 5; }
    sysctl -w "vm.drop_caches=$MODE" >/dev/null || { echo "ERROR: failed to set vm.drop_caches=$MODE" >&2; exit 5; }
fi

AFTER_FREE=$(read_kb MemFree)
AFTER_AVAIL=$(read_kb MemAvailable)
AFTER_CACHED=$(read_kb Cached)

if [ "$QUIET" -ne 1 ]; then
    awk -v bf="$BEFORE_FREE" -v ba="$BEFORE_AVAIL" -v bc="$BEFORE_CACHED" \
        -v af="$AFTER_FREE"  -v aa="$AFTER_AVAIL"  -v ac="$AFTER_CACHED"  \
        -v mode="$MODE" '
        BEGIN {
            printf "drop_caches mode=%d on Jetson (%s / %s, %d GiB)\n",
                mode, ENVIRON["JETSON_PRODUCT_LINE"], ENVIRON["JETSON_VARIANT"], ENVIRON["JETSON_MEM_GB"]+0;
            printf "  before:  free=%d MiB  avail=%d MiB  cached=%d MiB\n",
                bf/1024, ba/1024, bc/1024;
            printf "  after:   free=%d MiB  avail=%d MiB  cached=%d MiB\n",
                af/1024, aa/1024, ac/1024;
            printf "  delta:   free=%+d MiB  avail=%+d MiB  cached=%+d MiB\n",
                (af-bf)/1024, (aa-ba)/1024, (ac-bc)/1024;
        }'
fi
