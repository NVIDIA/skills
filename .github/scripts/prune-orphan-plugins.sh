#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Prune catalog plugin dirs whose components.d registration was removed.
#
# Plugin counterpart of prune-orphans.sh. The plugin sync only writes to
# plugins/<catalog_dir>/ for dirs declared under `plugins:` in
# components.d/*.yml — it never deletes a dir whose registration
# disappeared. This step makes deregistration remove the catalog copy,
# as part of the sync commit where the deletion is visible in the PR diff.
#
# Safety rails (same as skills):
#   - If any components.d file or catalog-exceptions.yml fails to parse,
#     pruning is skipped for the whole run.
#   - If more than PRUNE_CAP dirs would be pruned in one run, nothing is
#     deleted; the list is written to the overflow file and surfaced as a
#     workflow warning for human triage. The cap is separate from the
#     skills cap — each sync has its own budget.
#
# Outputs:
#   /tmp/pruned-plugins.txt          one pruned dir per line
#   /tmp/pruned-plugins-overflow.txt cap exceeded — dirs NOT deleted

set -euo pipefail

PRUNE_CAP="${PRUNE_CAP:-5}"
EXCEPTIONS_FILE="catalog-exceptions.yml"
pruned="${PRUNED_OUT:-/tmp/pruned-plugins.txt}"
overflow="${PRUNED_OVERFLOW_OUT:-/tmp/pruned-plugins-overflow.txt}"
: > "$pruned"
rm -f "$overflow"

expected=$(mktemp "${PRUNE_TMPDIR:-/tmp}/prune-plugins-expected.XXXXXX")

# 1. Declared set — every plugins[].catalog_dir across components.d/*.yml.
for f in components.d/*.yml; do
  if ! yq e 'true' "$f" > /dev/null 2>&1; then
    echo "::warning::${f} failed to parse — skipping plugin orphan pruning this run"
    exit 0
  fi
  yq -r '.plugins[]?.catalog_dir // ""' "$f" | grep -v '^$' >> "$expected" || true
done

# 1b. Catalog-built plugins — plugins.d/<name>.yml means build-plugins.py
#     owns plugins/<name>/ (e.g. nvidia-skills). The plugin sync never
#     mirrors, prunes or lists these; they stay on the existing approach.
for f in plugins.d/*.yml; do
  [ -f "$f" ] || continue
  b=$(basename "$f" .yml)
  case "$b" in _*) continue ;; esac
  echo "$b" >> "$expected"
done

# 2. Exceptions — plugin dirs allowed to exist without a registration
#    and without a plugins.d spec.
if [ -f "$EXCEPTIONS_FILE" ]; then
  if ! yq e 'true' "$EXCEPTIONS_FILE" > /dev/null 2>&1; then
    echo "::warning::${EXCEPTIONS_FILE} failed to parse — skipping plugin orphan pruning this run"
    exit 0
  fi
  yq -r '.plugin_exceptions[]?.dir // ""' "$EXCEPTIONS_FILE" | grep -v '^$' >> "$expected" || true
fi

if [ ! -s "$expected" ]; then
  echo "::warning::declared plugin set is empty — skipping plugin orphan pruning this run"
  exit 0
fi

# 3. Collect orphans: top-level plugins/ dirs not in the expected set.
orphans=$(mktemp "${PRUNE_TMPDIR:-/tmp}/prune-plugins-orphans.XXXXXX")
for d in plugins/*/; do
  [ -d "$d" ] || continue
  dir=$(basename "$d")
  if ! grep -qxF "$dir" "$expected"; then
    echo "$dir" >> "$orphans"
  fi
done

count=$(wc -l < "$orphans" | tr -d ' ')
if [ "$count" -eq 0 ]; then
  echo "No orphaned plugin dirs."
  exit 0
fi

# 4. Cap check.
if [ "$count" -gt "$PRUNE_CAP" ]; then
  cp "$orphans" "$overflow"
  echo "::warning::${count} orphaned plugin dirs exceed the prune cap (${PRUNE_CAP}) — nothing deleted. Dirs:"
  cat "$orphans"
  exit 0
fi

# 5. Prune.
while read -r dir; do
  git rm -rq "plugins/$dir"
  echo "$dir" >> "$pruned"
  echo "  ✂ pruned plugins/$dir (no components.d plugins: registration)"
done < "$orphans"
