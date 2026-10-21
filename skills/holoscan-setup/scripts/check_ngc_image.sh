#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Inspect a full 5.x image tag locally; do not pull or query a registry.
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^v5\.[0-9]+\.[0-9]+([.-][[:alnum:].-]+)?-cuda13$ ]]; then
    echo "Usage: $0 <full-5.x-tag> (example: v5.0.0-cuda13)" >&2
    exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is unavailable; local image presence is unknown." >&2
    exit 2
fi
if ! docker info >/dev/null 2>&1; then
    echo "Docker daemon is inaccessible; local image presence is unknown." >&2
    exit 2
fi
image="nvcr.io/nvidia/clara-holoscan/holoscan:$1"
if docker image inspect --format '{{.Id}}' "$image"; then
    echo "Local image present: $image"
else
    echo "Local image absent: $image" >&2
    exit 1
fi
