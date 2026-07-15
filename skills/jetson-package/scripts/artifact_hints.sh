#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# artifact_hints.sh
#
# Emits JSON hints for Jetson-compatible vLLM images and Jetson AI Lab PyPI.
# Sources the canonical Jetson detector.
#
# Usage:
#   scripts/artifact_hints.sh [--human]
#
# Exit codes:
#   0  success
#   2  not on a Jetson (from detect_jetson.sh)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../../jetson-diagnostic/scripts/detect_jetson.sh
# shellcheck disable=SC1091
. "$SKILLS_ROOT/jetson-diagnostic/scripts/detect_jetson.sh"

HUMAN=0
if [ "${1:-}" = "--human" ]; then
    HUMAN=1
fi

l4t_major() {
    raw="${JETSON_L4T_VERSION:-}"
    raw="${raw#r}"
    raw="${raw#R}"
    major="${raw%%.*}"
    case "$major" in
        ''|*[!0-9]*) echo 0 ;;
        *) echo "$major" ;;
    esac
}

case "${JETSON_GENERATION:-}" in
    thor)
        cuda_sm="11.0"
        vllm_image="vllm/vllm-openai:latest"
        ;;
    orin)
        cuda_sm="8.7"
        if [ "$(l4t_major)" -ge 39 ]; then
            vllm_image="vllm/vllm-openai:latest"
        else
            vllm_image="ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin"
        fi
        ;;
    *)
        echo "ERROR: unknown or unset JETSON_GENERATION='${JETSON_GENERATION:-}'" >&2
        echo "ERROR: expected one of: thor, orin; refusing to guess a container image tag" >&2
        exit 1
        ;;
esac

if [ "$HUMAN" -eq 1 ]; then
    printf 'generation=%s product_line=%s sku=%s variant=%s L4T=%s\n' \
        "$JETSON_GENERATION" "$JETSON_PRODUCT_LINE" "$JETSON_SKU" "$JETSON_VARIANT" "$JETSON_L4T_VERSION"
    printf 'CUDA SM hint=%s (Orin 8.7, Thor 11.0)\n' "$cuda_sm"
    printf 'Preferred vLLM image: %s\n' "$vllm_image"
    printf 'GHCR org packages: https://github.com/orgs/NVIDIA-AI-IOT/packages\n'
    printf 'Jetson AI Lab PyPI: https://pypi.jetson-ai-lab.io/\n'
    exit 0
fi

python3 - "$JETSON_SKU" "$JETSON_GENERATION" "$JETSON_PRODUCT_LINE" "$JETSON_VARIANT" "${JETSON_VARIANT_SOURCE:-}" "$JETSON_MEM_GB" "$JETSON_L4T_VERSION" "$JETSON_PRODUCT_MODEL" "$cuda_sm" "$vllm_image" <<'PY'
import json
import sys

sku, generation, product_line, variant, variant_src, mem_gb, l4t, product_model, cuda_sm, vllm_image = sys.argv[1:11]
mem_total_gb = int(mem_gb) if mem_gb.isdigit() else mem_gb
doc = {
    "skill": "jetson-package",
    "sku": sku,
    "generation": generation,
    "product_line": product_line,
    "variant": variant,
    "variant_source": variant_src or None,
    "mem_total_gb": mem_total_gb,
    "l4t_version": l4t,
    "product_model": product_model,
    "cuda_sm_hint": cuda_sm,
    "preferred_vllm_image": vllm_image,
    "ghcr_org_packages_url": "https://github.com/orgs/NVIDIA-AI-IOT/packages",
    "jetson_ai_lab_pypi_url": "https://pypi.jetson-ai-lab.io/",
    "notes": [
        "For vLLM, use upstream vllm/vllm-openai on Thor and Orin L4T r39+; use NVIDIA-AI-IOT GHCR on older Orin.",
        "Orin GPUs use SM 8.7; Thor uses SM 11.0 - many third-party wheels omit these targets.",
        "Thor T5000 vs T4000: variant_source explains inference; nv_boot_control (TNSPEC/COMPATIBLE_SPEC) is used when present; else EEPROM/flash board ID is the hardware ground truth.",
    ],
}
print(json.dumps(doc, indent=2))
PY
