#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Wrapper around version-plugins.py that ensures PyYAML + ruamel.yaml
# are importable. Forwards all arguments to the Python script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ensure_pkg() {
  local mod="$1" pkg="$2"
  if python3 -c "import ${mod}" 2>/dev/null; then
    return 0
  fi
  echo "${pkg} not found; installing for the current user..." >&2
  if pip3 install --user "${pkg}" >/dev/null 2>&1; then
    return 0
  fi
  if pip3 install --user --break-system-packages "${pkg}" >/dev/null 2>&1; then
    return 0
  fi
  echo "error: failed to install ${pkg}; install it manually (pip install ${pkg}) and re-run." >&2
  exit 1
}

ensure_pkg yaml pyyaml
ensure_pkg ruamel.yaml ruamel.yaml

exec python3 "$SCRIPT_DIR/version-plugins.py" "$@"
