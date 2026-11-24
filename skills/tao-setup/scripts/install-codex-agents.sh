#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-shot Codex installer for the TAO skill bank.
#
# What it does:
#   1. Registers the tao-skill-bank marketplace with the Codex CLI.
#   2. Installs the `nvidia-tao-toolkit` plugin (skills surface). Releases through 7.3.0
#      ship it as `tao-skill-bank`; the script installs whichever name the
#      chosen release offers and removes the other one if it is installed.
#      Releases after 7.3.0 need Codex CLI 0.142.0 or newer.
#   3. Says how to add the TAO agent identity (AGENTS.md) to a project, and
#      adds it when --project names one. Codex reads AGENTS.md from the project
#      a session works in and from its own home folder, not from an installed
#      plugin. scripts/tao_agent_identity.py writes the identity into the
#      project's AGENTS.md as a marked block that it can update and remove, and
#      leaves the rest of that file alone; Cursor reads the same file.
#      Nothing is written into the Codex home. Earlier versions of this script
#      copied the identity over ~/.codex/AGENTS.md, which replaced the user's
#      own instructions and applied the identity to every Codex session. When
#      such a copy or its backups are still there, the script says so and how
#      to restore the user's file, and changes nothing. A backup that is itself
#      an older copy of the identity is not offered as the user's file.
#
# Usage:
#   scripts/install-codex-agents.sh [--project DIR]
#   When the script is piped into bash from its release URL, as the README
#   shows, pass the options after `bash -s --`: bash -s -- --project DIR
#
#   --project DIR   after the install, add the identity block to DIR/AGENTS.md
#                   (needs python3, and a release that ships the helper script
#                   or a clone of the repository to run this script from)
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
DEFAULT_MARKETPLACE_REF="7.3.1"

MARKETPLACE_SOURCE="${TAO_SKILL_BANK_MARKETPLACE:-https://github.com/NVIDIA-TAO/tao-skill-bank.git}"
MARKETPLACE_REF="${TAO_SKILL_BANK_REF-$DEFAULT_MARKETPLACE_REF}"

# Plugin and marketplace names, newest first. This script is fetched from `main`
# but installs a release tag, so it cannot assume which names that tag ships:
#   after 7.3.0      nvidia-tao-toolkit  in marketplace tao-skill-bank
#   through 7.3.0    tao-skill-bank      in marketplace tao-local-plugins
# (`name` in .codex-plugin/plugin.json and .agents/plugins/marketplace.json.)
PLUGIN_NAMES=("nvidia-tao-toolkit" "tao-skill-bank")
MARKETPLACE_NAMES=("tao-skill-bank" "tao-local-plugins")

# Releases after 7.3.0 use the repository root as the plugin source. Codex CLIs
# older than this register the marketplace and then report that the plugin was
# not found in it.
MIN_CODEX_VERSION="0.142.0"

# A local path marketplace (clone or unzipped release) has no ref to check out.
if [[ -d "$MARKETPLACE_SOURCE" ]]; then
  MARKETPLACE_REF=""
fi

log() { printf '[install-codex-agents] %s\n' "$*"; }
die() { printf '[install-codex-agents] ERROR: %s\n' "$*" >&2; exit 1; }

# Options. They are checked before anything is installed, so that a mistyped
# project folder does not leave a half-finished run behind.
IDENTITY_PROJECT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)
      [[ $# -ge 2 ]] || die "--project needs a directory"
      IDENTITY_PROJECT="$2"
      [[ -n "$IDENTITY_PROJECT" ]] || die "--project needs a directory"
      shift 2
      ;;
    --project=*)
      IDENTITY_PROJECT="${1#--project=}"
      [[ -n "$IDENTITY_PROJECT" ]] || die "--project needs a directory"
      shift
      ;;
    -h|--help)
      printf '%s\n' \
        "Usage: install-codex-agents.sh [--project DIR]" \
        "" \
        "Registers the TAO skill bank marketplace with Codex and installs the" \
        "nvidia-tao-toolkit plugin. Nothing is written into the Codex home besides that." \
        "" \
        "  --project DIR   also add the TAO agent identity to DIR/AGENTS.md as a marked" \
        "                  block that scripts/tao_agent_identity.py can update and remove" \
        "" \
        "TAO_SKILL_BANK_MARKETPLACE and TAO_SKILL_BANK_REF choose another source or ref."
      exit 0
      ;;
    *)
      die "unknown argument: $1 (see --help)"
      ;;
  esac
done
if [[ -n "$IDENTITY_PROJECT" ]]; then
  [[ -d "$IDENTITY_PROJECT" ]] || die "--project: '$IDENTITY_PROJECT' is not an existing directory"
  IDENTITY_PROJECT="$(cd "$IDENTITY_PROJECT" && pwd)"
  command -v python3 >/dev/null 2>&1 \
    || die "--project needs python3 (3.9 or later) to add the identity block"
fi

# 0. Preflight
command -v codex >/dev/null 2>&1 \
  || die "'codex' CLI not found. Install it first: https://developers.openai.com/codex"

codex_older_than() {
  # True when the installed CLI reports a version lower than $1. A version that
  # cannot be read counts as new enough: the caller only uses this for a hint.
  local have
  have="$(codex --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1 || true)"
  [[ -n "$have" && "$have" != "$1" ]] \
    && [[ "$(printf '%s\n%s\n' "$have" "$1" | sort -V | head -n1)" == "$have" ]]
}

# 1. Marketplace
# `codex plugin marketplace add` refuses to overwrite a registration that points
# at a different source or ref, and `upgrade` only re-fetches the ref already
# recorded — so an existing registration is dropped and re-added. That is what
# moves a previously unpinned (or older-release) install onto this ref.
#
# Both generations of names are cleared first. Leaving an older plugin enabled
# next to the new one lists every skill twice, which makes Codex cut each skill
# description to a few words.
marketplace_registered() {
  # Exact match on the name column: marketplace roots are paths that can
  # contain these names too.
  codex plugin marketplace list 2>/dev/null \
    | awk -v name="$1" '$1 == name { found = 1 } END { exit !found }'
}
for i in "${!PLUGIN_NAMES[@]}"; do
  codex plugin remove "${PLUGIN_NAMES[$i]}@${MARKETPLACE_NAMES[$i]}" >/dev/null 2>&1 || true
done
#
# A clone registered by path is listed under the name its manifest carries now,
# but Codex removes a marketplace by the name it was registered under. A clone
# that was updated across the rename is listed as one generation and configured
# as the other, so try the listed name first and then the other generation.
for name in "${MARKETPLACE_NAMES[@]}"; do
  if marketplace_registered "$name"; then
    log "Marketplace '$name' already registered — re-pointing it."
    removed=""
    for candidate in "$name" "${MARKETPLACE_NAMES[@]}"; do
      if codex plugin marketplace remove "$candidate" 2>/dev/null; then
        removed="$candidate"
        break
      fi
    done
    [[ -n "$removed" ]] \
      || die "could not remove the registered marketplace '$name'. Remove it with 'codex plugin marketplace remove <name>' and run this script again."
  fi
done

log "Adding marketplace from ${MARKETPLACE_SOURCE}${MARKETPLACE_REF:+ (ref: ${MARKETPLACE_REF})}"
if [[ -n "$MARKETPLACE_REF" ]]; then
  codex plugin marketplace add "$MARKETPLACE_SOURCE" --ref "$MARKETPLACE_REF"
else
  codex plugin marketplace add "$MARKETPLACE_SOURCE"
fi

# 2. Plugin
# `codex plugin add` is idempotent and re-resolves the version from the
# marketplace snapshot. Try the current name first and fall back to the name
# older releases ship, so one script serves every release tag.
PLUGIN_NAME=""
MARKETPLACE_NAME=""
ADD_ERRORS=""
for i in "${!PLUGIN_NAMES[@]}"; do
  candidate="${PLUGIN_NAMES[$i]}@${MARKETPLACE_NAMES[$i]}"
  if err="$(codex plugin add "$candidate" 2>&1 >/dev/null)"; then
    PLUGIN_NAME="${PLUGIN_NAMES[$i]}"
    MARKETPLACE_NAME="${MARKETPLACE_NAMES[$i]}"
    log "Installed plugin ${candidate}"
    break
  fi
  ADD_ERRORS="${ADD_ERRORS}  ${candidate}: ${err}"$'\n'
done
if [[ -z "$PLUGIN_NAME" ]]; then
  printf '%s' "$ADD_ERRORS" >&2
  if codex_older_than "$MIN_CODEX_VERSION"; then
    printf '[install-codex-agents] This Codex CLI is older than %s, which releases after 7.3.0 need. Upgrade Codex and run this script again.\n' \
      "$MIN_CODEX_VERSION" >&2
  fi
  die "could not install the TAO plugin from ${MARKETPLACE_SOURCE}${MARKETPLACE_REF:+ (ref: ${MARKETPLACE_REF})}"
fi

# 3. Agent identity
# Codex loads AGENTS.md from the project a session works in and from its own
# home folder, never from an installed plugin. This step writes nothing into
# the Codex home. It finds the bank that was just installed and the helper that
# adds its identity to a project's AGENTS.md as a marked block, then prints the
# command or, with --project, runs it.
#
# Prefer the marketplace snapshot: it is checked out at exactly the ref that was
# just installed, so the identity matches the release the plugin came from. The
# plugin cache keeps every version ever installed, so picking its highest
# version number would hand back a newer identity than a pinned older release
# ships. Fall back to the cache, then to the clone this script runs from.
CODEX_ROOT="${CODEX_HOME:-${HOME}/.codex}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CACHE_ROOT="${CODEX_ROOT}/plugins/cache/${MARKETPLACE_NAME}/${PLUGIN_NAME}"
HELPER_NAME="scripts/tao_agent_identity.py"

BANK_ROOT=""
MARKETPLACE_ROOT="$(codex plugin marketplace list 2>/dev/null \
  | awk -v name="$MARKETPLACE_NAME" '$1 == name { sub(/^[^ \t]+[ \t]+/, ""); print; exit }' || true)"
if [[ -n "$MARKETPLACE_ROOT" && -f "${MARKETPLACE_ROOT}/AGENTS.md" ]]; then
  BANK_ROOT="$MARKETPLACE_ROOT"
fi
if [[ -z "$BANK_ROOT" && -d "$CACHE_ROOT" ]]; then
  LATEST_VERSION="$(ls -1 "$CACHE_ROOT" 2>/dev/null | sort -V | tail -n1 || true)"
  if [[ -n "$LATEST_VERSION" && -f "${CACHE_ROOT}/${LATEST_VERSION}/AGENTS.md" ]]; then
    BANK_ROOT="${CACHE_ROOT}/${LATEST_VERSION}"
  fi
fi
# The clone this script runs from counts only when the helper sits beside the
# script: piped into bash, the script has no folder of its own, and the folder
# above the working directory may hold somebody's own AGENTS.md.
SCRIPT_BANK=""
if [[ -f "${SCRIPT_DIR}/tao_agent_identity.py" && -f "${SCRIPT_DIR}/../AGENTS.md" ]]; then
  SCRIPT_BANK="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
# The identity comes from the installed release. The helper does too when that
# release ships it; a release that predates the helper is served by the one in
# the clone, which is then told which bank to read.
IDENTITY_BANK="${BANK_ROOT:-$SCRIPT_BANK}"
HELPER=""
for candidate in "$BANK_ROOT" "$SCRIPT_BANK"; do
  if [[ -n "$candidate" && -f "${candidate}/${HELPER_NAME}" ]]; then
    HELPER="${candidate}/${HELPER_NAME}"
    break
  fi
done

# What earlier versions of this script left in the Codex home: the identity
# copied over the global AGENTS.md, and a copy of the file it replaced, named
# AGENTS.md.bak.<YYYYmmddHHMMSS> (14 digits). Those versions backed up every
# file that differed from the identity they installed, and that includes the
# identity of the release before. A backup is therefore the user's own file
# only when it does not start with the heading of the identity, which is the
# test applied to the global file as well. Nothing is changed there; one line
# says what is left and what to do about it. The backups sort by time, so the
# last one that passes the test is the newest file of the user's own.
GLOBAL_AGENTS="${CODEX_ROOT}/AGENTS.md"
IDENTITY_HEADING=""
if [[ -n "$IDENTITY_BANK" ]]; then
  IDENTITY_HEADING="$(head -n 1 "${IDENTITY_BANK}/AGENTS.md")"
fi
BACKUP_COUNT=0
OWN_COUNT=0
OWN_BACKUP=""
for backup in "$GLOBAL_AGENTS".bak.[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]; do
  [[ -f "$backup" ]] || continue
  BACKUP_COUNT=$((BACKUP_COUNT + 1))
  if [[ -n "$IDENTITY_HEADING" && "$(head -n 1 "$backup" 2>/dev/null || true)" != "$IDENTITY_HEADING" ]]; then
    OWN_COUNT=$((OWN_COUNT + 1))
    OWN_BACKUP="$backup"
  fi
done
OWN_BACKUPS="do not start like the identity: $OWN_COUNT of $BACKUP_COUNT. The newest of them, $OWN_BACKUP, is most likely"
GLOBAL_COPY=""
if [[ -f "$GLOBAL_AGENTS" && -n "$IDENTITY_HEADING" ]]; then
  if cmp -s "${IDENTITY_BANK}/AGENTS.md" "$GLOBAL_AGENTS"; then
    GLOBAL_COPY="is a copy of the TAO agent identity, as earlier versions of this script installed it"
  elif [[ "$(head -n 1 "$GLOBAL_AGENTS" 2>/dev/null || true)" == "$IDENTITY_HEADING" ]]; then
    GLOBAL_COPY="starts like an older copy of the TAO agent identity, as earlier versions of this script installed it"
  fi
fi
NOTE=""
if [[ -n "$GLOBAL_COPY" ]]; then
  NOTE="$GLOBAL_AGENTS $GLOBAL_COPY, so it applies to every Codex session. It was not changed."
  if [[ -n "$OWN_BACKUP" ]]; then
    NOTE="$NOTE Backups beside it that $OWN_BACKUPS your own file from before. Look at it, then restore it with: mv \"$OWN_BACKUP\" \"$GLOBAL_AGENTS\""
  else
    if [[ "$BACKUP_COUNT" -gt 0 ]]; then
      NOTE="$NOTE Every backup beside it ($BACKUP_COUNT) starts like the identity too: these are older copies of it, not a file of your own, and can be deleted (keep any text of your own that you added to one)."
    else
      NOTE="$NOTE No backup of an earlier file is there."
    fi
    NOTE="$NOTE To have no global instructions again, delete the copy with: rm \"$GLOBAL_AGENTS\" (if you added text of your own to it, keep that part)"
  fi
elif [[ "$BACKUP_COUNT" -gt 0 ]]; then
  NOTE="$BACKUP_COUNT backup(s) of $GLOBAL_AGENTS that an earlier version of this script made are still there, and nothing was changed. Deleting them (${GLOBAL_AGENTS}.bak.<timestamp>) ends this note."
  if [[ -z "$IDENTITY_HEADING" ]]; then
    NOTE="$NOTE This run found no identity to compare them with, so it cannot tell a file of your own from an older copy of the identity; look at a backup before you restore or delete it."
  elif [[ -z "$OWN_BACKUP" ]]; then
    NOTE="$NOTE Every one of them starts like the TAO agent identity: these are older copies of it, not files of your own, and can be deleted (keep any text of your own that you added to one)."
  elif [[ -f "$GLOBAL_AGENTS" ]]; then
    NOTE="$NOTE $GLOBAL_AGENTS itself is not a copy of the identity. Backups that $OWN_BACKUPS an earlier file of your own; keep what you still need."
  else
    NOTE="$NOTE Backups that $OWN_BACKUPS the file you had before. If you want it back, restore it before you delete the rest: mv \"$OWN_BACKUP\" \"$GLOBAL_AGENTS\""
  fi
fi
if [[ -n "$NOTE" ]]; then
  log "NOTE: $NOTE"
fi

IDENTITY_COMMAND="python3 \"$HELPER\""
if [[ -n "$HELPER" && "$HELPER" != "${IDENTITY_BANK}/${HELPER_NAME}" ]]; then
  IDENTITY_COMMAND="TAO_SKILL_BANK_PATH=\"$IDENTITY_BANK\" $IDENTITY_COMMAND"
fi
NO_HELPER="this release does not ship ${HELPER_NAME}, which adds the identity to a project's AGENTS.md as a marked block. Run that script from a clone of the repository's main branch"
if [[ -n "$IDENTITY_PROJECT" ]]; then
  [[ -n "$HELPER" ]] \
    || die "the plugin is installed, but the TAO agent identity was not added to $IDENTITY_PROJECT: $NO_HELPER: python3 ${HELPER_NAME} add --project \"$IDENTITY_PROJECT\""
  TAO_SKILL_BANK_PATH="$IDENTITY_BANK" python3 "$HELPER" add --project "$IDENTITY_PROJECT" \
    || die "the plugin is installed, but the TAO agent identity was not added to $IDENTITY_PROJECT; the message above says why."
  log "To update the block later: $IDENTITY_COMMAND add --project \"$IDENTITY_PROJECT\" ('remove' in place of 'add' deletes it)."
elif [[ -n "$HELPER" ]]; then
  log "The TAO agent identity was not written anywhere. Codex and Cursor read AGENTS.md of the project a session works in. To add the identity there as a marked block, and nothing else:"
  log "  $IDENTITY_COMMAND add --project /path/to/your/project"
  log "Run it again to update the block; 'remove' in place of 'add' deletes it."
else
  log "The TAO agent identity was not written anywhere: $NO_HELPER."
fi

log "Done. Launch 'codex' from any directory to use the TAO skill bank."
