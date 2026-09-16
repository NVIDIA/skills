#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Select one already-provisioned host Python and execute it. This wrapper is
# intentionally install-free and consumes credentials only from its inherited
# process environment. Use ``--runtime`` for bundled PAS/data stages; control
# scripts need Python 3.9+.

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_parent=$(CDPATH= cd -- "$script_dir/../../../../.." && pwd)
export PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}"

runtime=false
workspace_arg=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --runtime)
      runtime=true
      shift
      ;;
    --workspace)
      [ "$#" -ge 2 ] || {
        echo "deft_python: --workspace requires a path" >&2
        exit 2
      }
      workspace_arg=$2
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

# Cap the BLAS/OpenMP thread pools before any Python runs (probe or exec).
# WHY: hosts with >128 cores crash the numpy/sklearn OpenBLAS (built with a
# 128-thread cap) during t-SNE, killing the whole process. min(64, nproc)
# keeps every host safe; values already set by the caller are respected.
cores=$(nproc 2>/dev/null || echo 64)
if [ "$cores" -gt 64 ]; then
  cores=64
fi
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$cores}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$cores}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$cores}"

control_probe='import sys; assert sys.version_info >= (3, 9)'
if [ "$#" -gt 0 ] && [ "$(basename -- "$1")" = run_deft_action.py ]; then
  # The platform seam validates tao-artifacts JSON Schemas before submit and
  # finalize. Do not select an otherwise valid workspace interpreter that
  # lacks the schema validator and then fail only after mutating action state.
  control_probe='import sys; assert sys.version_info >= (3, 9); import jsonschema'
fi
if [ "$runtime" = true ] && [ ! -f "$script_dir/pas_deft/__init__.py" ]; then
  echo "deft_python: bundled PAS runtime is missing from the installed skill" >&2
  exit 2
fi

runtime_probe='import sys; assert sys.version_info >= (3, 9); import pandas,numpy,pyarrow,PIL,yaml,matplotlib,sklearn,torch'
candidates=(
  "${DEFT_PYTHON:-}"
  "${workspace_arg:+$workspace_arg/.venv/bin/python}"
  "${workspace_arg:+$workspace_arg/.venv/bin/python3}"
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
  probe=$control_probe
  if [ "$runtime" = true ]; then
    probe=$runtime_probe
  fi
  if "$candidate" -c "$probe" >/dev/null 2>&1; then
    selected=$candidate
    break
  fi
done

if [ -z "$selected" ]; then
  if [ "$runtime" = true ]; then
    echo "deft_python: no installed Python provides the bundled PAS runtime dependencies (pandas,numpy,pyarrow,PIL,yaml,matplotlib,sklearn,torch)" >&2
    echo "deft_python: provision the approved workspace venv, then retry" >&2
  else
    echo "deft_python: Python 3.9+ not found" >&2
  fi
  exit 2
fi

if [ "$#" -eq 0 ]; then
  printf '%s\n' "$selected"
  exit 0
fi

exec "$selected" "$@"
