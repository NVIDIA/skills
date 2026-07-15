#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# runtime_probes.sh
#
# Small, stable helpers for diagnostic/runtime scripts. Source this file; do
# not exec it.
#
# Usage from a skill script under skills/<skill-name>/scripts:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   SKILLS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
#   . "$SKILLS_ROOT/jetson-diagnostic/scripts/runtime_probes.sh"
#
# Functions:
#   read_meminfo_kb KEY
#       Print the integer KB value for KEY in /proc/meminfo (e.g. MemTotal,
#       MemAvailable, SwapTotal). Empty output if the key is missing.
#
#   nvsmi_useful
#       Returns 0 (true) when `nvidia-smi` is present AND the GPU name does
#       NOT contain "(nvgpu)". On the unified `nvidia.ko` compute stack
#       (e.g. Thor), nvidia-smi returns real CUDA/process data and the name
#       is e.g. "NVIDIA Thor". On the legacy `nvgpu` stack (Orin today),
#       nvidia-smi is present but stubbed and reports "<model> (nvgpu)";
#       in that case this helper returns 1 so callers know to fall back to
#       /sys/kernel/debug/nvmap/ for per-process GPU memory.

# Guard so multiple sourcings are cheap.
if [ "${__JETSON_RUNTIME_PROBES_SOURCED:-0}" = "1" ]; then
    return 0
fi
__JETSON_RUNTIME_PROBES_SOURCED=1

read_meminfo_kb() {
    awk -v k="$1" '$1==k":" {print $2}' /proc/meminfo 2>/dev/null
}

nvsmi_useful() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    local name
    name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
    case "$name" in
        ''|*'(nvgpu)'*) return 1 ;;
        *)              return 0 ;;
    esac
}
