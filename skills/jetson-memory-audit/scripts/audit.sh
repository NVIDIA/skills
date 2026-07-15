#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# audit.sh
#
# Memory-focused snapshot of a Jetson. Thin wrapper around
# jetson-diagnostic/scripts/snapshot.sh — extracts the memory-relevant fields
# so callers get a smaller, focused document without duplicating collection
# logic. See ../references/DESIGN.md for why this delegation approach was chosen.
#
# Flags:
#   --human          Pretty-print the JSON.
#   --tegra-secs N   Sample tegrastats for N seconds (default 3).
#
# Exit codes:
#   0  ok
#   2  not on a Jetson

set -uo pipefail

HUMAN=0
TEGRA_SECS=3
while [ $# -gt 0 ]; do
    case "$1" in
        --human)        HUMAN=1; shift ;;
        --tegra-secs)   TEGRA_SECS="$2"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 64 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAPSHOT="$SCRIPT_DIR/../../jetson-diagnostic/scripts/snapshot.sh"

if [ ! -f "$SNAPSHOT" ]; then
    echo "ERROR: jetson-diagnostic/scripts/snapshot.sh not found at $SNAPSHOT" >&2
    exit 1
fi

SNAP=$(bash "$SNAPSHOT" --tegra-secs "$TEGRA_SECS") || exit $?

# Extract the memory-relevant subset. top_processes is renamed procrank_top to
# match the audit output contract; all GPU/thermal/disk/power fields are dropped
# since the caller only needs memory for before/after comparison.
if command -v jq >/dev/null 2>&1; then
PAYLOAD=$(printf '%s' "$SNAP" | jq '{
    sku, generation, product_line, variant,
    mem_total_gb, l4t_version, product_model,
    memory_kb,
    default_systemd_target,
    candidate_services,
    tegrastats_sample,
    nvmap,
    procrank_top: .top_processes
}')
else
PAYLOAD=$(SNAP_JSON="$SNAP" python3 - <<'PY'
import json
import os
import sys

doc = json.loads(os.environ["SNAP_JSON"])
out = {
    "sku": doc.get("sku"),
    "generation": doc.get("generation"),
    "product_line": doc.get("product_line"),
    "variant": doc.get("variant"),
    "mem_total_gb": doc.get("mem_total_gb"),
    "l4t_version": doc.get("l4t_version"),
    "product_model": doc.get("product_model"),
    "memory_kb": doc.get("memory_kb"),
    "default_systemd_target": doc.get("default_systemd_target"),
    "candidate_services": doc.get("candidate_services"),
    "tegrastats_sample": doc.get("tegrastats_sample"),
    "nvmap": doc.get("nvmap"),
    "procrank_top": doc.get("top_processes"),
}
print(json.dumps(out, separators=(",", ":")))
PY
)
fi

if [ "$HUMAN" = "1" ]; then
    if command -v jq >/dev/null 2>&1; then
        printf '%s\n' "$PAYLOAD" | jq .
    else
        printf '%s\n' "$PAYLOAD" | python3 -m json.tool
    fi
else
    printf '%s\n' "$PAYLOAD"
fi
