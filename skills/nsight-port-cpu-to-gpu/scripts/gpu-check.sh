#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

# GPU environment check for Nsight AI Porting Agent.
# Usage: gpu-check.sh [flavor-id]
#   flavor-id defaults to fortran-openacc (manifest.id)
#   e.g. fortran-openacc, cpp-cuda, fortran-openmp, cpp-kokkos
#   nested path also works: fortran/openacc

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FLAVOR="${1:-fortran-openacc}"

resolve_manifest() {
    local flavor="$1" m
    if [ -f "$SKILL_ROOT/flavors/$flavor/manifest.json" ]; then
        printf '%s\n' "$SKILL_ROOT/flavors/$flavor/manifest.json"
        return 0
    fi
    shopt -s nullglob
    for m in "$SKILL_ROOT"/flavors/*/*/manifest.json; do
        if grep -q "\"id\"[[:space:]]*:[[:space:]]*\"$flavor\"" "$m"
        then
            printf '%s\n' "$m"
            return 0
        fi
    done
    return 1
}

MANIFEST="$(resolve_manifest "$FLAVOR" || true)"

PASS=0
FAIL=0

check() {
    local label="$1"
    shift
    if "$@" > /dev/null 2>&1; then
        echo "  [OK]   $label"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $label"
        FAIL=$((FAIL + 1))
    fi
}

if [ ! -f "$MANIFEST" ]; then
    echo "Unknown flavor: $FLAVOR (no nested pack manifest)"
    exit 1
fi

# Read a string key from the manifest JSON. Prefers python3, falls back
# to a grep/sed parse. Never aborts under `set -e` (missing python3,
# bad JSON, or absent key all yield an empty string).
read_manifest_key() {
    local key="$1" val=""
    if command -v python3 > /dev/null 2>&1; then
        val=$(python3 -c \
            "import json; print(json.load(open('$MANIFEST')).get('$key',''))" \
            2>/dev/null || true)
    fi
    if [ -z "$val" ]; then
        val=$(grep -o "\"$key\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" "$MANIFEST" \
            2>/dev/null | head -1 \
            | sed 's/.*:[[:space:]]*"\([^"]*\)".*/\1/' || true)
    fi
    printf '%s' "$val"
}

COMPILER=$(read_manifest_key compiler)
DISPLAY=$(read_manifest_key display_name)
STATUS=$(read_manifest_key status)
COMPILER=${COMPILER:-nvfortran}
DISPLAY=${DISPLAY:-$FLAVOR}
STATUS=${STATUS:-unknown}

echo "=== GPU Environment Check ($DISPLAY) ==="
echo "Flavor: $FLAVOR (status: $STATUS)"
echo ""

check "nvidia-smi found" command -v nvidia-smi
check "$COMPILER found" command -v "$COMPILER"
check "git found"        command -v git

if command -v nvidia-smi > /dev/null 2>&1; then
    echo ""
    GPU_OUTPUT=$(nvidia-smi --query-gpu=name,compute_cap,driver_version \
        --format=csv,noheader 2>&1) || GPU_RC=$?
    GPU_RC=${GPU_RC:-0}
    if [ "$GPU_RC" -ne 0 ] \
        || echo "$GPU_OUTPUT" | grep -qi "blocked\|NVML\|fail"; then
        echo "  [FAIL] GPU accessible via nvidia-smi (NVML query failed)"
        echo "$GPU_OUTPUT" | sed 's/^/    /'
        FAIL=$((FAIL + 1))
    else
        echo "  [OK]   GPU accessible via nvidia-smi"
        PASS=$((PASS + 1))
        echo ""
        echo "GPU info:"
        echo "$GPU_OUTPUT" | sed 's/^/  /'
    fi
fi

if command -v "$COMPILER" > /dev/null 2>&1; then
    echo ""
    echo "Compiler ($COMPILER):"
    "$COMPILER" --version 2>/dev/null | head -3 | sed 's/^/  /' || true
fi

if [ "$STATUS" = "stub" ]; then
    echo ""
    echo "  [WARN] Flavor $FLAVOR is a stub — full porting not supported yet."
fi

echo ""
echo "Result: $PASS passed, $FAIL failed"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "Missing requirements (see $MANIFEST for compiler details):"
    command -v nvidia-smi > /dev/null 2>&1 || \
        echo "  - nvidia-smi: install the NVIDIA driver / check GPU access"
    command -v "$COMPILER" > /dev/null 2>&1 || \
        echo "  - $COMPILER: install the toolchain for $DISPLAY (e.g. NVIDIA HPC SDK)"
    command -v git > /dev/null 2>&1 || \
        echo "  - git: install git"
    echo "  If nvidia-smi is present but NVML failed, check driver/container GPU access."
    exit 1
fi

exit 0
