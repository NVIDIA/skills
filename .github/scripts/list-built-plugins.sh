#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Print the names of the plugins built by build-plugins.py, one per line.
#
# A built plugin's identity is the `name:` field inside plugins.d/<file>.yml,
# not the filename: build-plugins.py writes plugins/<name>/ from that field.
# The plugin sync, orphan pruning, and README regeneration all need this same
# set, so they share this script rather than each deriving it.
#
# Files starting with `_` (such as _defaults.yml) are includes, not plugins.
# A file too broken to yield a name falls back to its basename, so a caller
# errs toward keeping or protecting a directory rather than touching it.

set -euo pipefail

dir="${1:-plugins.d}"

for f in "$dir"/*.yml; do
  [ -f "$f" ] || continue
  b=$(basename "$f" .yml)
  case "$b" in _*) continue ;; esac
  n=$(yq -r '.name // ""' "$f" 2>/dev/null || true)
  echo "${n:-$b}"
done | sort -u
