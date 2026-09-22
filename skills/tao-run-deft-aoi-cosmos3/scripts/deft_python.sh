#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Select one already-provisioned host Python and execute it. This wrapper is
# intentionally install-free so invoking a bundled script can never trigger
# network access in an air-gapped run.

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_parent=$(CDPATH= cd -- "$script_dir/../../../../.." && pwd)

probe='import pyarrow,yaml'
candidates=(
  "${DEFT_PYTHON:-}"
  "${WORKSPACE_DIR:-}/.venv/bin/python"
  "${WORKSPACE_DIR:-}/.venv/bin/python3"
  "${WORKSPACE:-}/.venv/bin/python"
  "${WORKSPACE:-}/.venv/bin/python3"
  "$script_dir/../.venv/bin/python3"
  /usr/bin/python3
  "$repo_parent/.venv/bin/python3"
  "$HOME/.venvs/deft/bin/python3"
  "$(command -v python3 2>/dev/null || true)"
)

selected=
for candidate in "${candidates[@]}"; do
  [ -n "$candidate" ] || continue
  [ -x "$candidate" ] || continue
  [ "$candidate" != "$script_dir/deft_python.sh" ] || continue
  if "$candidate" -c "$probe" >/dev/null 2>&1; then
    selected=$candidate
    break
  fi
done

if [ -z "$selected" ]; then
  echo "deft_python: no installed Python provides pyarrow,yaml" >&2
  echo "deft_python: provision dependencies outside this workflow; no installer will be run" >&2
  exit 2
fi

if [ "$#" -eq 0 ]; then
  printf '%s\n' "$selected"
  exit 0
fi

exec "$selected" "$@"
