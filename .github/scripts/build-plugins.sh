#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Wrapper around build-plugins.py that ensures PyYAML is importable.
# Forwards all arguments to the Python script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! python3 -c "import yaml" 2>/dev/null; then
  echo "PyYAML not found; installing for the current user..." >&2
  if pip3 install --user pyyaml >/dev/null 2>&1; then
    :
  elif pip3 install --user --break-system-packages pyyaml >/dev/null 2>&1; then
    :
  else
    echo "error: failed to install PyYAML; install it manually (pip install pyyaml) and re-run." >&2
    exit 1
  fi
fi

exec python3 "$SCRIPT_DIR/build-plugins.py" "$@"
