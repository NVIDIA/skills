#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Prune catalog skill dirs whose components.d registration was removed.
#
# The sync only writes to dirs declared in components.d/*.yml — it never
# deletes a dir whose registration disappeared, so deregistered skills
# linger in the catalog (and downstream surfaces) indefinitely. Teams
# consistently assume deregistration removes the catalog copy (Dynamo,
# TAO/tao-run-on-lepton, and cuOpt all hit this), so this step makes that
# assumption true: any top-level skills/ dir that is neither declared in
# components.d nor listed in catalog-exceptions.yml is
# removed as part of the sync commit, where the deletion is visible in
# the sync PR diff.
#
# Safety rails:
#   - If any components.d file fails to parse, pruning is skipped for
#     the whole run — a parse error would make that component's skills
#     look unregistered and mass-delete them.
#   - If more than PRUNE_CAP dirs would be pruned in one run, nothing is
#     deleted; the list is written to /tmp/pruned-orphans-overflow.txt
#     and surfaced as a workflow warning for human triage.
#
# Outputs:
#   /tmp/pruned-orphans.txt          one pruned dir per line
#   /tmp/pruned-orphans-overflow.txt cap exceeded — dirs NOT deleted

set -euo pipefail

PRUNE_CAP="${PRUNE_CAP:-5}"
EXCEPTIONS_FILE="catalog-exceptions.yml"
pruned="${PRUNED_OUT:-/tmp/pruned-orphans.txt}"
overflow="${PRUNED_OVERFLOW_OUT:-/tmp/pruned-orphans-overflow.txt}"
: > "$pruned"
rm -f "$overflow"

expected=$(mktemp "${PRUNE_TMPDIR:-/tmp}/prune-expected.XXXXXX")

# 1. Declared set — every catalog_dir across components.d/*.yml.
for f in components.d/*.yml; do
  if ! yq e 'true' "$f" > /dev/null 2>&1; then
    echo "::warning::${f} failed to parse — skipping orphan pruning this run"
    exit 0
  fi
  yq -r '.skills[]?.catalog_dir // ""' "$f" | grep -v '^$' >> "$expected" || true
done

# 2. Exceptions — dirs allowed to exist without a registration.
if [ -f "$EXCEPTIONS_FILE" ]; then
  if ! yq e 'true' "$EXCEPTIONS_FILE" > /dev/null 2>&1; then
    echo "::warning::${EXCEPTIONS_FILE} failed to parse — skipping orphan pruning this run"
    exit 0
  fi
  yq -r '.exceptions[]?.dir // ""' "$EXCEPTIONS_FILE" | grep -v '^$' >> "$expected" || true
fi

if [ ! -s "$expected" ]; then
  echo "::warning::declared skill set is empty — skipping orphan pruning this run"
  exit 0
fi

# 3. Collect orphans: top-level skills/ dirs not in the expected set.
orphans=$(mktemp "${PRUNE_TMPDIR:-/tmp}/prune-orphans.XXXXXX")
for d in skills/*/; do
  dir=$(basename "$d")
  if ! grep -qxF "$dir" "$expected"; then
    echo "$dir" >> "$orphans"
  fi
done

count=$(wc -l < "$orphans" | tr -d ' ')
if [ "$count" -eq 0 ]; then
  echo "No orphaned skill dirs."
  exit 0
fi

# 4. Cap check.
if [ "$count" -gt "$PRUNE_CAP" ]; then
  cp "$orphans" "$overflow"
  echo "::warning::${count} orphaned skill dirs exceed the prune cap (${PRUNE_CAP}) — nothing deleted. Dirs:"
  cat "$orphans"
  exit 0
fi

# 5. Prune.
while read -r dir; do
  git rm -rq "skills/$dir"
  echo "$dir" >> "$pruned"
  echo "  ✂ pruned skills/$dir (no components.d registration)"
done < "$orphans"
