#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Select one already-provisioned host Python and execute it. This wrapper is
# intentionally install-free so invoking a bundled script can never trigger
# network access in an air-gapped run.

set -eu

env_file=${DEFT_ENV_FILE:-}
if [ -z "$env_file" ] && [ -n "${WORKSPACE:-}" ]; then
  env_file="$WORKSPACE/.env"
elif [ -z "$env_file" ] && [ -n "${WORKSPACE_DIR:-}" ]; then
  env_file="$WORKSPACE_DIR/.env"
fi
if [ -n "$env_file" ] && [ -f "$env_file" ]; then
  set -a
  . "$env_file"
  set +a
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_parent=$(CDPATH= cd -- "$script_dir/../../../../.." && pwd)

modules='pandas numpy matplotlib pyarrow PIL yaml'
probe="import ${modules// /,}"
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
  echo "deft_python: no installed Python provides $modules" >&2
  # Which modules each interpreter is short of, rather than the whole list every
  # time: a report naming all six sends the reader after five that are present.
  reported=
  for candidate in "${candidates[@]}"; do
    [ -n "$candidate" ] || continue
    [ -x "$candidate" ] || continue
    [ "$candidate" != "$script_dir/deft_python.sh" ] || continue
    case " $reported " in *" $candidate "*) continue ;; esac
    reported="$reported $candidate"
    missing=$("$candidate" -c 'import importlib.util, sys
print(" ".join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))' \
      $modules 2>/dev/null) || continue
    [ -n "$missing" ] && echo "deft_python:   $candidate is missing: $missing" >&2
  done
  echo "deft_python: provision dependencies outside this workflow; no installer will be run" >&2
  exit 2
fi

if [ "$#" -eq 0 ]; then
  printf '%s\n' "$selected"
  exit 0
fi

exec "$selected" "$@"
