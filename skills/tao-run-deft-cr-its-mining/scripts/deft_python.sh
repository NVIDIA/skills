#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Select an already-provisioned Python for the workflow's host-side helpers.
# Installation remains an explicit, user-approved preflight action.

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
skill_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
workspace=${WORKSPACE_DIR:-${WORKSPACE:-}}
probe='import sys,numpy,pandas,pyarrow,yaml,huggingface_hub; assert sys.version_info >= (3, 11)'

candidates=(
  "${DEFT_PYTHON:-}"
  "${workspace:+$workspace/.venv/bin/python}"
  "${workspace:+$workspace/.venv/bin/python3}"
  "$skill_dir/.venv/bin/python"
  "$skill_dir/.venv/bin/python3"
  "$(command -v python3 2>/dev/null || true)"
)

for candidate in "${candidates[@]}"; do
  [ -n "$candidate" ] || continue
  [ -x "$candidate" ] || continue
  if "$candidate" -c "$probe" >/dev/null 2>&1; then
    if [ "$#" -eq 0 ]; then
      printf '%s\n' "$candidate"
    else
      exec "$candidate" "$@"
    fi
    exit 0
  fi
done

workspace_display=${workspace:-/absolute/path/to/deft_workspace}
cat >&2 <<EOF
deft_python: no Python 3.11+ interpreter provides numpy, pandas, pyarrow, yaml, and huggingface_hub
deft_python: after user approval, provision the workspace environment with:
  BOOTSTRAP_PYTHON="\$(command -v python3.11 || command -v python3)"
  "\$BOOTSTRAP_PYTHON" -c 'import sys; assert sys.version_info >= (3, 11)'
  "\$BOOTSTRAP_PYTHON" -m venv "$workspace_display/.venv"
  VENV_PYTHON="$workspace_display/.venv/bin/python"
  "\$VENV_PYTHON" -m pip --version >/dev/null 2>&1 || "\$VENV_PYTHON" -m ensurepip --upgrade
  "\$VENV_PYTHON" -m pip install numpy pandas pyarrow pyyaml huggingface_hub
deft_python: a global pip executable is not required
deft_python: if venv creation or ensurepip fails, ask the user to provide Python 3.11 with venv/ensurepip; do not make a privileged system change
deft_python: rerun preflight after installation; this selector never installs packages itself
EOF
exit 2
