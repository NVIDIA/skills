#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-shot Codex installer for the TAO skill bank.
#
# What it does:
#   1. Registers the tao-skill-bank marketplace with the Codex CLI.
#   2. Installs the `tao-skill-bank` plugin (skills surface).
#   3. Copies AGENTS.md to ~/.codex/AGENTS.md so the TAO identity loads in
#      every Codex session, not only when codex is launched from a clone.
#      (Codex's AGENTS.md discovery walks the project tree from the git root —
#      plugin-bundled SessionStart hooks do not yet install identity globally;
#      see https://github.com/openai/codex/issues/16430.)
#
# By default it installs the latest release build (DEFAULT_MARKETPLACE_REF
# below) rather than whatever `main` holds. Override the source or the ref via
# env var if you need a fork, a different release, or unreleased work:
#   TAO_SKILL_BANK_MARKETPLACE=ssh://git@host/path/repo.git \
#   TAO_SKILL_BANK_REF=main \
#       scripts/install-codex-agents.sh
# `TAO_SKILL_BANK_REF=` (empty) uses the source's default branch.

set -euo pipefail

# Latest release tag — keep in sync with the pins in README.md on every release.
DEFAULT_MARKETPLACE_REF="7.2.0"

MARKETPLACE_SOURCE="${TAO_SKILL_BANK_MARKETPLACE:-https://github.com/NVIDIA-TAO/tao-skill-bank.git}"
MARKETPLACE_REF="${TAO_SKILL_BANK_REF-$DEFAULT_MARKETPLACE_REF}"
MARKETPLACE_NAME="tao-local-plugins"   # `name` in .agents/plugins/marketplace.json
PLUGIN_NAME="tao-skill-bank"

# A local path marketplace (clone or unzipped release) has no ref to check out.
if [[ -d "$MARKETPLACE_SOURCE" ]]; then
  MARKETPLACE_REF=""
fi

log() { printf '[install-codex-agents] %s\n' "$*"; }
die() { printf '[install-codex-agents] ERROR: %s\n' "$*" >&2; exit 1; }

# 0. Preflight
command -v codex >/dev/null 2>&1 \
  || die "'codex' CLI not found. Install it first: https://developers.openai.com/codex"

# 1. Marketplace
# `codex plugin marketplace add` refuses to overwrite a registration that points
# at a different source or ref, and `upgrade` only re-fetches the ref already
# recorded — so an existing registration is dropped and re-added. That is what
# moves a previously unpinned (or older-release) install onto this ref.
if codex plugin marketplace list 2>/dev/null | grep -qw "$MARKETPLACE_NAME"; then
  log "Marketplace '$MARKETPLACE_NAME' already registered — re-pointing it."
  codex plugin marketplace remove "$MARKETPLACE_NAME"
fi

log "Adding marketplace from ${MARKETPLACE_SOURCE}${MARKETPLACE_REF:+ (ref: ${MARKETPLACE_REF})}"
if [[ -n "$MARKETPLACE_REF" ]]; then
  codex plugin marketplace add "$MARKETPLACE_SOURCE" --ref "$MARKETPLACE_REF"
else
  codex plugin marketplace add "$MARKETPLACE_SOURCE"
fi

# 2. Plugin
# `codex plugin add` is idempotent and re-resolves the version from the
# marketplace snapshot, so it is run unconditionally — an already-installed
# plugin from an older ref is replaced by the one this ref ships.
log "Installing plugin ${PLUGIN_NAME}@${MARKETPLACE_NAME}"
codex plugin add "${PLUGIN_NAME}@${MARKETPLACE_NAME}"

# 3. Global AGENTS.md identity
# Prefer the marketplace snapshot: it is checked out at exactly the ref that was
# just installed, so the identity matches the release the plugin came from. The
# plugin cache keeps every version ever installed, so picking its highest
# version number would hand back a newer identity than a pinned older release
# ships. Fall back to the cache, then to the repo root when running from a clone.
CODEX_ROOT="${CODEX_HOME:-${HOME}/.codex}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_AGENTS="${SCRIPT_DIR}/../AGENTS.md"
CACHE_ROOT="${CODEX_ROOT}/plugins/cache/${MARKETPLACE_NAME}/${PLUGIN_NAME}"

SRC_AGENTS=""
MARKETPLACE_ROOT="$(codex plugin marketplace list 2>/dev/null \
  | awk -v name="$MARKETPLACE_NAME" '$1 == name { print $2; exit }' || true)"
if [[ -n "$MARKETPLACE_ROOT" && -f "${MARKETPLACE_ROOT}/AGENTS.md" ]]; then
  SRC_AGENTS="${MARKETPLACE_ROOT}/AGENTS.md"
fi
if [[ -z "$SRC_AGENTS" && -d "$CACHE_ROOT" ]]; then
  LATEST_VERSION="$(ls -1 "$CACHE_ROOT" 2>/dev/null | sort -V | tail -n1 || true)"
  if [[ -n "$LATEST_VERSION" && -f "${CACHE_ROOT}/${LATEST_VERSION}/AGENTS.md" ]]; then
    SRC_AGENTS="${CACHE_ROOT}/${LATEST_VERSION}/AGENTS.md"
  fi
fi
if [[ -z "$SRC_AGENTS" && -f "$REPO_AGENTS" ]]; then
  SRC_AGENTS="$(cd "$(dirname "$REPO_AGENTS")" && pwd)/AGENTS.md"
fi
if [[ -z "$SRC_AGENTS" ]]; then
  log "WARN: could not locate AGENTS.md; skipping global identity install."
  log "      Verify with: codex plugin list"
  exit 0
fi

DEST_AGENTS="${CODEX_ROOT}/AGENTS.md"
mkdir -p "$CODEX_ROOT"
if [[ -f "$DEST_AGENTS" ]] && ! cmp -s "$SRC_AGENTS" "$DEST_AGENTS"; then
  BACKUP="${DEST_AGENTS}.bak.$(date +%Y%m%d%H%M%S)"
  log "Backing up existing $DEST_AGENTS -> $BACKUP"
  cp "$DEST_AGENTS" "$BACKUP"
fi
cp "$SRC_AGENTS" "$DEST_AGENTS"
log "Installed TAO agent identity -> $DEST_AGENTS"

log "Done. Launch 'codex' from any directory to use the TAO skill bank."
