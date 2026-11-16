#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

# Profile environment check.
# Usage: profile-check.sh [flavor-id]
#   flavor-id defaults to fortran-openacc (manifest.id)
#   e.g. fortran-openacc, cpp-cuda, fortran-openmp, cpp-kokkos
#   nested path also works: fortran/openacc
# Exits 0 if basic tools exist; warns on CPU sampling / NVTX issues.
# python3 + sqlite3 is a hard requirement (nsys Round 0).

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
WARN=0

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

warn() {
    echo "  [WARN] $1"
    WARN=$((WARN + 1))
}

echo "=== Profile Environment Check ==="
echo "Flavor: $FLAVOR"
echo ""

check "nvidia-smi found" command -v nvidia-smi
check "nsys found"       command -v nsys
check "git found"        command -v git
check "python3 + sqlite3" python3 -c "import sqlite3"

if [ -f "$MANIFEST" ]; then
    # Parse the manifest safely — never abort under `set -e` if python3
    # is missing or the JSON is malformed; fall back to grep then nvfortran.
    COMPILER=""
    if command -v python3 > /dev/null 2>&1; then
        COMPILER=$(python3 -c \
            "import json; print(json.load(open('$MANIFEST')).get('compiler',''))" \
            2>/dev/null || true)
    fi
    if [ -z "$COMPILER" ]; then
        COMPILER=$(grep -o '"compiler"[[:space:]]*:[[:space:]]*"[^"]*"' "$MANIFEST" \
            2>/dev/null | head -1 \
            | sed 's/.*:[[:space:]]*"\([^"]*\)".*/\1/' || true)
    fi
    [ -n "$COMPILER" ] || COMPILER=nvfortran
    check "$COMPILER found" command -v "$COMPILER"
else
    check "nvfortran found" command -v nvfortran
    COMPILER=nvfortran
fi

if command -v nvidia-smi > /dev/null 2>&1; then
    echo ""
    GPU_OUTPUT=$(nvidia-smi --query-gpu=name,compute_cap,driver_version \
        --format=csv,noheader 2>&1) || GPU_RC=$?
    GPU_RC=${GPU_RC:-0}
    if [ "$GPU_RC" -ne 0 ] \
        || echo "$GPU_OUTPUT" | grep -qi "blocked\|NVML\|fail"; then
        echo "  [FAIL] GPU accessible via nvidia-smi"
        FAIL=$((FAIL + 1))
    else
        echo "  [OK]   GPU accessible via nvidia-smi"
        PASS=$((PASS + 1))
    fi
fi

echo ""
echo "--- Nsight Systems environment ---"
if command -v nsys > /dev/null 2>&1; then
    nsys --version 2>/dev/null | head -2 | sed 's/^/  /' || true
    echo ""
    if nsys status --environment 2>&1 | sed 's/^/  /'; then
        :
    else
        warn "nsys status --environment failed"
    fi
else
    warn "nsys not installed — skip environment status"
fi

echo ""
echo "--- CPU sampling prerequisites ---"
PARANOID_FILE=/proc/sys/kernel/perf_event_paranoid
if [ -r "$PARANOID_FILE" ]; then
    PARANOID=$(tr -d '[:space:]' < "$PARANOID_FILE")
    echo "  perf_event_paranoid: $PARANOID"
    if ! [[ "$PARANOID" =~ ^-?[0-9]+$ ]]; then
        warn "perf_event_paranoid not numeric ('$PARANOID') — cannot assess CPU sampling"
    elif [ "$PARANOID" -gt 2 ]; then
        warn "perf_event_paranoid > 2 — nsys CPU IP/backtrace sampling blocked"
        echo "        Fix (until reboot): sudo sh -c 'echo 2 > $PARANOID_FILE'"
        echo "        See: https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html"
        echo "             #requirements-for-x86-64-and-arm-sbsa-targets-on-linux"
    else
        echo "  [OK]   perf_event_paranoid <= 2 for process-tree CPU sampling"
        PASS=$((PASS + 1))
    fi
else
    warn "Cannot read $PARANOID_FILE"
fi

echo ""
echo "--- NVTX link hints ---"
if command -v "$COMPILER" > /dev/null 2>&1; then
    NVTX3=$("$COMPILER" -print-file-name=libnvtx3interop.so 2>/dev/null || true)
    if [ -z "$NVTX3" ] || [ ! -f "$NVTX3" ]; then
        SDK_BIN=$(dirname "$(command -v "$COMPILER")")
        NVTX3=$(find "$SDK_BIN"/.. -name libnvtx3interop.so -type f 2>/dev/null | head -1)
    fi
    LEGACY=$("$COMPILER" -print-file-name=libnvToolsExt.so 2>/dev/null || true)
    if [ -z "$LEGACY" ] || [ ! -f "$LEGACY" ]; then
        SDK_BIN=$(dirname "$(command -v "$COMPILER")")
        LEGACY=$(find "$SDK_BIN"/.. -name libnvToolsExt.so -type f 2>/dev/null | head -1)
    fi
    if [ -n "$NVTX3" ] && [ -f "$NVTX3" ]; then
        echo "  [OK]   libnvtx3interop: $NVTX3"
        echo "        Link with: -L$(dirname "$NVTX3") -lnvtx3interop"
        PASS=$((PASS + 1))
    elif [ -n "$LEGACY" ] && [ -f "$LEGACY" ]; then
        echo "  [OK]   libnvToolsExt (legacy): $LEGACY"
        echo "        Link with: -L$(dirname "$LEGACY") -lnvToolsExt"
        PASS=$((PASS + 1))
    else
        warn "No NVTX library found via $COMPILER -print-file-name"
        echo "        nvtx3: $NVTX3"
        echo "        legacy: $LEGACY"
    fi
fi

echo ""
echo "Result: $PASS passed, $FAIL failed, $WARN warnings"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi

# Warnings alone do not fail — agent must interpret before Round 0
exit 0
