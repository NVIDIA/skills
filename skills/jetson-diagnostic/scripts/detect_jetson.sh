#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# detect_jetson.sh
#
# Canonical Jetson detector owned by jetson-diagnostic. Other Jetson skills
# source this file instead of duplicating platform detection logic.
#
# Usage from a skill script under skills/<skill-name>/scripts:
#
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   SKILLS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
#   . "$SKILLS_ROOT/jetson-diagnostic/scripts/detect_jetson.sh"
#
# Exports:
#   JETSON_SKU               legacy model bucket: thor | orin-agx | orin-nx | orin-nano | unknown
#   JETSON_GENERATION        silicon era: thor | orin | unknown
#   JETSON_PRODUCT_LINE      module line: thor-agx | orin-agx | orin-nx | orin-nano | unknown
#                            (today all Thor kits in-tree are AGX-class -> thor-agx)
#   JETSON_VARIANT           e.g. thor-t5000 | orin-nx-16gb | orin-nano-8gb
#   JETSON_VARIANT_SOURCE    optional; for Thor only, how T5000 vs T4000 was inferred
#                            (see __assign_variant thor notes)
#   JETSON_MEM_GB            total system memory in GiB (integer)
#   JETSON_L4T_VERSION       e.g. 36.4.0; "unknown" if /etc/nv_tegra_release missing
#   JETSON_PRODUCT_MODEL     raw lowercased model string from /proc/device-tree/model
#
# Run standalone to print the detected fields as JSON.
#
# Exit codes:
#   0  detection succeeded (or sourcing succeeded)
#   2  not running on a Jetson

set -u

__jetson_present() {
    [ -r /proc/device-tree/model ] || [ -r /etc/nv_tegra_release ]
}

if ! __jetson_present; then
    echo "ERROR: not running on a Jetson device." >&2
    echo "       SSH to your Jetson and re-run this skill from there." >&2
    # shellcheck disable=SC2317  # exit 2 is reachable when run directly, not sourced
    return 2 2>/dev/null || exit 2
fi

__read_model() {
    if [ -r /proc/device-tree/model ]; then
        tr -d '\0' < /proc/device-tree/model | tr '[:upper:]' '[:lower:]'
    fi
}

__detect_sku() {
    local model="$1"
    case "$model" in
        *thor*)                                          echo "thor" ;;
        *orin*nano*)                                     echo "orin-nano" ;;
        *"orin nx"*|*orin-nx*)                           echo "orin-nx" ;;
        *"agx orin"*|*"orin agx"*)                       echo "orin-agx" ;;
        *orin*)                                          echo "orin-agx" ;;
        *)                                               echo "unknown" ;;
    esac
}

__detect_mem_gb() {
    local kb
    kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
    awk -v k="$kb" 'BEGIN { printf "%d\n", (k/1024/1024) + 0.5 }'
}

# Sets globals JETSON_VARIANT and (for Thor only) JETSON_VARIANT_SOURCE.
#
# Thor T5000 vs T4000 — /proc/device-tree/model is often a generic marketing string
# (e.g. "NVIDIA Jetson Thor Developer Kit") shared across SKUs, so it cannot
# distinguish T4000 vs T5000 unless the DT literally contains t4000/t5000 (uncommon).
#
# Order of checks (first match wins):
#   1) /etc/nv_boot_control.conf TNSPEC + COMPATIBLE_SPEC — primary when present:
#        jetson-agx-thor-t4000, jetson-agx-thor-devkit, optional literal t5000/t4000
#      Extend when new BSP flash profiles add distinct strings (grep Linux_for_Tegra).
#   2) /proc/device-tree/model *t5000* / *t4000* -> device_tree_model (rare boards)
#   3) nvidia-smi GPU name *t5000* / *t4000* (often absent; e.g. just "NVIDIA Thor")
#   4) MemTotal heuristic (RAM line card; not hardware fuse read).
# EEPROM / flash board ID remains ground truth; this script does not parse EEPROM.
__assign_variant() {
    local sku="$1" mem_gb="$2" model="$3"
    JETSON_VARIANT_SOURCE=""
    case "$sku" in
        thor)
            local boot_blob=""
            if [ -r /etc/nv_boot_control.conf ]; then
                boot_blob="$(
                    grep -E '^(TNSPEC|COMPATIBLE_SPEC)' /etc/nv_boot_control.conf 2>/dev/null |
                        tr '[:upper:]' '[:lower:]'
                )"
            fi
            case "$boot_blob" in
                *t5000*)
                    JETSON_VARIANT="thor-t5000"
                    JETSON_VARIANT_SOURCE="nv_boot_control"
                    ;;
                *t4000*)
                    JETSON_VARIANT="thor-t4000"
                    JETSON_VARIANT_SOURCE="nv_boot_control"
                    ;;
                *jetson-agx-thor-devkit*)
                    JETSON_VARIANT="thor-t5000"
                    JETSON_VARIANT_SOURCE="nv_boot_control"
                    ;;
                *)
                    case "$model" in
                        *t5000*)
                            JETSON_VARIANT="thor-t5000"
                            JETSON_VARIANT_SOURCE="device_tree_model"
                            ;;
                        *t4000*)
                            JETSON_VARIANT="thor-t4000"
                            JETSON_VARIANT_SOURCE="device_tree_model"
                            ;;
                        *)
                            local gpu_name
                            gpu_name="$(
                                nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null |
                                    head -1 | tr '[:upper:]' '[:lower:]'
                            )"
                            case "$gpu_name" in
                                *t5000*)
                                    JETSON_VARIANT="thor-t5000"
                                    JETSON_VARIANT_SOURCE="nvidia_smi"
                                    ;;
                                *t4000*)
                                    JETSON_VARIANT="thor-t4000"
                                    JETSON_VARIANT_SOURCE="nvidia_smi"
                                    ;;
                                *)
                                    if   [ "$mem_gb" -ge 96 ]; then
                                        JETSON_VARIANT="thor-t5000"
                                        JETSON_VARIANT_SOURCE="mem_total_heuristic"
                                    elif [ "$mem_gb" -ge 32 ]; then
                                        JETSON_VARIANT="thor-t4000"
                                        JETSON_VARIANT_SOURCE="mem_total_heuristic"
                                    else
                                        JETSON_VARIANT="thor"
                                        JETSON_VARIANT_SOURCE="mem_total_heuristic_incomplete"
                                    fi
                                    ;;
                            esac
                            ;;
                    esac
                    ;;
            esac
            ;;
        orin-agx)
            case "$model" in
                *industrial*)
                    JETSON_VARIANT="orin-agx-industrial"
                    ;;
                *)
                    if   [ "$mem_gb" -ge 56 ]; then JETSON_VARIANT="orin-agx-64gb"
                    elif [ "$mem_gb" -ge 28 ]; then JETSON_VARIANT="orin-agx-32gb"
                    else JETSON_VARIANT="orin-agx"; fi
                    ;;
            esac
            ;;
        orin-nx)
            if   [ "$mem_gb" -ge 14 ]; then JETSON_VARIANT="orin-nx-16gb"
            elif [ "$mem_gb" -ge 7 ];  then JETSON_VARIANT="orin-nx-8gb"
            else JETSON_VARIANT="orin-nx"; fi
            ;;
        orin-nano)
            if   [ "$mem_gb" -ge 7 ]; then JETSON_VARIANT="orin-nano-8gb"
            elif [ "$mem_gb" -ge 3 ]; then JETSON_VARIANT="orin-nano-4gb"
            else JETSON_VARIANT="orin-nano"; fi
            ;;
        *)
            JETSON_VARIANT="unknown"
            ;;
    esac
}

# Maps JETSON_SKU (legacy) -> JETSON_GENERATION + JETSON_PRODUCT_LINE.
__derive_generation_and_product_line() {
    case "$JETSON_SKU" in
        thor)
            JETSON_GENERATION="thor"
            JETSON_PRODUCT_LINE="thor-agx"
            ;;
        orin-agx|orin-nx|orin-nano)
            JETSON_GENERATION="orin"
            JETSON_PRODUCT_LINE="$JETSON_SKU"
            ;;
        *)
            JETSON_GENERATION="unknown"
            JETSON_PRODUCT_LINE="unknown"
            ;;
    esac
}

__detect_l4t() {
    if [ ! -r /etc/nv_tegra_release ]; then
        echo "unknown"
        return
    fi
    awk '
        /^# R[0-9]+/ {
            major=$2; sub("R","",major)
        }
        /REVISION:/ {
            for (i=1; i<=NF; i++) if ($i == "REVISION:") { rev=$(i+1); sub(",","",rev) }
        }
        END {
            if (major != "" && rev != "") printf "%s.%s\n", major, rev;
            else print "unknown";
        }
    ' /etc/nv_tegra_release
}

JETSON_PRODUCT_MODEL="$(__read_model)"
JETSON_SKU="$(__detect_sku "$JETSON_PRODUCT_MODEL")"
__derive_generation_and_product_line
JETSON_MEM_GB="$(__detect_mem_gb)"
__assign_variant "$JETSON_SKU" "$JETSON_MEM_GB" "$JETSON_PRODUCT_MODEL"
JETSON_L4T_VERSION="$(__detect_l4t)"

export JETSON_SKU JETSON_GENERATION JETSON_PRODUCT_LINE \
    JETSON_VARIANT JETSON_VARIANT_SOURCE JETSON_MEM_GB JETSON_L4T_VERSION JETSON_PRODUCT_MODEL

if [ "${BASH_SOURCE[0]:-$0}" = "$0" ]; then
    model_escaped=$(printf '%s' "$JETSON_PRODUCT_MODEL" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')
    src_escaped=$(printf '%s' "$JETSON_VARIANT_SOURCE" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')
    printf '{"sku":"%s","generation":"%s","product_line":"%s","variant":"%s","variant_source":"%s","mem_total_gb":%s,"l4t_version":"%s","product_model":"%s"}\n' \
        "$JETSON_SKU" "$JETSON_GENERATION" "$JETSON_PRODUCT_LINE" \
        "$JETSON_VARIANT" "$src_escaped" "$JETSON_MEM_GB" "$JETSON_L4T_VERSION" "$model_escaped"
fi
