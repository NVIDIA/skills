#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Common shell helpers for Jetson skill scripts. Source this file; do not exec it.

if [ "${__JETSON_COMMON_SOURCED:-0}" = "1" ]; then
    return 0
fi
__JETSON_COMMON_SOURCED=1

need_value() {
    [ $# -ge 2 ] || { echo "ERROR: $1 requires a value" >&2; exit 64; }
}
