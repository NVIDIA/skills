#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SessionStart hook for the TAO skill bank.
#
# Prints one JSON object on standard output and nothing else:
#
#   {"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"<text>"}}
#
# The host adds <text> to the agent's context at session start. Keep it tight:
# every line lands in context for every session. Why JSON and why this shape:
#   Claude Code  takes plain text or this object from a SessionStart hook
#                (https://code.claude.com/docs/en/hooks, "JSON output" and
#                "SessionStart decision control").
#   Cursor       takes JSON only and drops plain text. The field it documents
#                for this event is additional_context
#                (https://cursor.com/docs/hooks, "sessionStart"). Its page on
#                Claude Code hooks says in general that it supports Claude
#                Code's nested hookSpecificOutput responses
#                (https://cursor.com/docs/reference/third-party-hooks,
#                "Response Format Compatibility") and gives the mapping for
#                other events only. That the additionalContext of the object
#                above reaches a session is observed, not documented: the
#                Cursor CLI 2026.11.19 added both parts of this hook. Check it
#                again when Cursor changes its hooks.
#   Codex        documents the same object (https://learn.chatgpt.com/docs/hooks,
#                "SessionStart"), but lists no hook for a plugin folder whose root
#                plugin.json declares the Agent Plugins schema, as the generated
#                plugin package does, and skips a plugin hook the user has not
#                trusted. Codex users get the identity through tao-setup instead.
#
# Usage: session_start.sh [identity|context|all]   (default: all)
#
# Claude Code caps the additionalContext of one hook command at 10,000
# characters and replaces anything longer with a file path and a short preview
# (same page, "JSON output"), so hooks/hooks.json registers this script twice
# and each run prints one part:
#   identity  AGENTS.md up to the split heading (identity + discovery flow)
#   context   the rest of AGENTS.md, then the plugin folder, credentials,
#             card packs and the docker check
# `all` prints both parts in one object, for anything that runs the script
# directly. scripts/tests/test_session_start_hook.py keeps the text and the
# whole output of each part under the cap.
#
# Responsibilities:
#   1. Emit TAO orchestration guidance (the agent's identity + discovery flow).
#   2. Say where the plugin folder is, and export it where the host has a way.
#   3. Report which credential env vars are present in the session (names only).
#      Credentials arrive from the user's own shell or from a user-approved env
#      file the agent sources; this hook never reads or prints values.
#   4. Surface clear setup hints if docker is missing.
#
# This hook does NOT install Python packages. The bank runs SDK-free over native
# platform CLIs; the one wheel-dependent skill, tao-run-automl, installs
# nvidia-tao-automl lazily via its Preflight block.

set -u

TAO_PART="${1:-all}"
case "$TAO_PART" in
  identity|context|all) ;;
  *) echo "usage: session_start.sh [identity|context|all]" >&2; exit 2 ;;
esac

# The split and the JSON encoding below need awk. Without it the hook adds
# nothing rather than print something a host cannot parse.
command -v awk >/dev/null 2>&1 || { echo '{}'; exit 0; }

# Plugin root. Claude Code sets CLAUDE_PLUGIN_ROOT; other harnesses that run this
# hook may set their own variable or none, so fall back to this script's parent.
TAO_ROOT="${CLAUDE_PLUGIN_ROOT:-${CURSOR_PLUGIN_ROOT:-${PLUGIN_ROOT:-}}}"
if [[ -z "$TAO_ROOT" || ! -f "$TAO_ROOT/AGENTS.md" ]]; then
  TAO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

# Line number of the level-2 heading where AGENTS.md is split between the two
# parts: the first "## " heading at or past the midpoint, outside fenced code
# (a "## " line inside a fence is a comment, not a heading). 0 = do not split.
tao_split_line() {
  awk -v total="$(wc -c < "$1")" '
    { if (/^```/) fence = !fence
      if (!found && !fence && /^## / && off >= total / 2) { print NR; found = 1 }
      off += length($0) + 1 }
    END { if (!found) print 0 }' "$1"
}

# $1 as one single-quoted shell word. Single quotes copy every byte as it is,
# in any locale. `printf %q` does not: bash 3.2 in a UTF-8 locale escapes some
# bytes of a multi-byte character and leaves the others, which is not UTF-8.
tao_quote() {
  local _tao_apostrophe="'\\''"
  printf "'%s'" "${1//\'/$_tao_apostrophe}"
}

# Standard input as the one JSON object described at the top of this file.
# Bytes are copied as they are (LC_ALL=C), so UTF-8 text stays UTF-8; a quote, a
# backslash and every control character are escaped, and a line end becomes \n.
# Characters are looked up one by one because awk implementations disagree
# about backslashes in the replacement text of gsub().
tao_json() {
  LC_ALL=C awk '
    BEGIN {
      for (i = 0; i < 32; i++) escaped[sprintf("%c", i)] = sprintf("\\u%04x", i)
      escaped["\t"] = "\\t"; escaped["\r"] = "\\r"
      escaped["\""] = "\\\""; escaped["\\"] = "\\\\"
      printf "{\"hookSpecificOutput\":{\"hookEventName\":\"SessionStart\",\"additionalContext\":\""
    }
    { line = ""; n = length($0)
      for (i = 1; i <= n; i++) {
        c = substr($0, i, 1)
        if (c in escaped) line = line escaped[c]; else line = line c
      }
      printf "%s\\n", line }
    END { print "\"}}" }'
}

# The text of the requested part, as Markdown.
tao_text() {
  # ─── 1. Agent guidance ──────────────────────────────────────────────────
  # Single source of truth: AGENTS.md at the plugin root (cross-runtime spec —
  # https://agents.md/). Edit there to update Claude + Codex + any future
  # runtime in one place. Do not duplicate the prompt inline here or in other
  # hooks.
  if [[ -f "${TAO_ROOT}/AGENTS.md" ]]; then
    _tao_split="$(tao_split_line "${TAO_ROOT}/AGENTS.md")"
    if [[ "$TAO_PART" == "all" || "$_tao_split" -le 1 ]]; then
      # No split point: the identity part carries the whole file.
      [[ "$TAO_PART" != "context" ]] && { cat "${TAO_ROOT}/AGENTS.md"; echo; }
    elif [[ "$TAO_PART" == "identity" ]]; then
      head -n "$((_tao_split - 1))" "${TAO_ROOT}/AGENTS.md"
    else
      echo "<!-- TAO agent guidance, part 2 of 2 (part 1: identity and discovery flow) -->"
      tail -n "+${_tao_split}" "${TAO_ROOT}/AGENTS.md"
      echo
    fi
  fi
  [[ "$TAO_PART" == "identity" ]] && return 0

  # ─── 1b. Where the plugin folder is ─────────────────────────────────────
  # The bank's scripts (resolve_tao_image.py, tao_job_record.py) and templates/
  # read $TAO_SKILL_BANK_PATH to find versions.yaml, and skills name files by
  # it. Claude Code persists it for the session's shell commands through
  # CLAUDE_ENV_FILE (https://code.claude.com/docs/en/hooks, "Persist
  # environment variables"); no other host gives a hook a way to do that. The
  # folder is stated in one line either way, because an export alone tells
  # the agent nothing, and the helper scripts fall back to the bank they
  # ship in.
  _tao_export="export TAO_SKILL_BANK_PATH=$(tao_quote "$TAO_ROOT")"
  if [[ -n "${CLAUDE_ENV_FILE:-}" ]] && { grep -qxF "$_tao_export" "$CLAUDE_ENV_FILE" \
      || echo "$_tao_export" >> "$CLAUDE_ENV_FILE"; } 2>/dev/null; then
    _tao_exported="It is exported as \`TAO_SKILL_BANK_PATH\` to the shell commands of this session."
  else
    _tao_exported="It is not exported here: set \`TAO_SKILL_BANK_PATH\` to it in a shell command that needs it."
  fi
  echo "TAO plugin folder: \`${TAO_ROOT}\`. ${_tao_exported} Skills are under \`${TAO_ROOT}/skills\`, helper scripts under \`${TAO_ROOT}/scripts\`."
  echo

  # ─── 2. Credentials ─────────────────────────────────────────────────────
  # Presence only: names, never values. The credential policy itself lives in
  # AGENTS.md, which this hook already emits above.
  echo "## Credentials"
  echo
  # Known credential vars across the platform/model skills. Names only.
  _tao_cred_vars="NGC_KEY BREV_API_TOKEN \
ACCESS_KEY SECRET_KEY S3_BUCKET_NAME S3_ENDPOINT_URL HF_TOKEN WANDB_API_KEY"
  _tao_present=""
  for _v in $_tao_cred_vars; do
    [[ -n "${!_v:-}" ]] && _tao_present="${_tao_present} ${_v}"
  done
  if [[ -n "${_tao_present// /}" ]]; then
    echo "Detected in this session's environment (names only):"
    for _v in $_tao_present; do echo "- $_v"; done
  else
    echo "No TAO credential vars detected in this session's environment."
  fi
  echo
  echo "Credentials are read from the environment — export what you need in your"
  echo "shell **before launching**, or source a user-approved env file, e.g.:"
  echo "\`\`\`bash"
  echo "export NGC_KEY=...            # nvcr.io image pulls"
  echo "export HF_TOKEN=...           # gated HuggingFace models"
  echo "# platform-specific: BREV_API_TOKEN, ACCESS_KEY/SECRET_KEY/S3_*"
  echo "\`\`\`"
  echo "Have an env file (~/.tao/secrets.env, ~/.config/tao/.env, or one you"
  echo "point the agent at)? Load it with \`set -a; source /path/to/.env; set +a\`."
  # Only a README that is there is named: the repository has one at its root,
  # and the generated plugin package ships its own.
  [[ -f "${TAO_ROOT}/README.md" ]] \
    && echo "The Credentials section of \`${TAO_ROOT}/README.md\` has the full var list."
  echo

  # ─── 2b. Token-efficient execution — card packs, when they can run ──────
  # A card pack is the cards/ folder of a skill; its driver runs the workflow
  # as fresh per-stage sessions of the Pi harness and stops without `pi`. So
  # the packs are named only where `pi` is installed, and only as a route the
  # user asks for: a session without `pi` cannot use them. Skills sit two
  # levels below skills/ in the repository and one level below it in the
  # generated plugin package, so both depths are scanned and only skill
  # folders count.
  if [[ -d "${TAO_ROOT}/skills" ]] && { command -v pi >/dev/null 2>&1 \
      || [[ -x "${HOME:-}/.local/share/pi-node/current/bin/pi" ]]; }; then
    _tao_packs=""
    for _p in "${TAO_ROOT}"/skills/*/cards "${TAO_ROOT}"/skills/*/*/cards; do
      [[ -d "$_p" && -f "$(dirname "$_p")/SKILL.md" ]] \
        && _tao_packs="${_tao_packs} $(basename "$(dirname "$_p")")"
    done
    if [[ -n "${_tao_packs// /}" ]]; then
      echo "## Card packs (optional)"
      echo
      echo "The \`pi\` harness is installed here, and card packs exist for:${_tao_packs}."
      echo "A pack's \`cards/driver.sh\` runs that workflow as fresh per-stage \`pi\`"
      echo "sessions. Use a pack only when the user asks for a card-pack run; read the"
      echo "pack's README first for the workflow, data layout and environment it"
      echo "expects. Otherwise run the workflow in this conversation. The"
      echo "\`tao-launch-workflow\` gate applies in full either way — the driver launch"
      echo "is the side-effecting action it reviews. Framework: the"
      echo "\`tao-token-efficient-execution\` skill."
      echo
    fi
  fi

  # ─── 3. Docker preflight ────────────────────────────────────────────────
  if ! command -v docker >/dev/null 2>&1; then
    # The check script of the tao-setup-nvidia-gpu-host skill, in either layout.
    _tao_setup=""
    for _s in "${TAO_ROOT}"/skills/tao-setup-nvidia-gpu-host/scripts/setup-nvidia-gpu-host.sh \
        "${TAO_ROOT}"/skills/*/tao-setup-nvidia-gpu-host/scripts/setup-nvidia-gpu-host.sh; do
      [[ -f "$_s" ]] && { _tao_setup="$_s"; break; }
    done
    echo "## ⚠ Docker missing"
    echo
    echo "\`docker\` is not on PATH in this session. Most TAO skills need Docker plus"
    echo "the TAO GPU host runtime, by default at least:"
    echo "- NVIDIA driver branch 580"
    echo "- CUDA Toolkit 13.0"
    echo "- NVIDIA Container Toolkit 1.19.0"
    echo
    if [[ -n "$_tao_setup" ]]; then
      echo "To see what this host has, run the read-only check (it installs nothing):"
      echo "\`\`\`bash"
      echo "bash $(tao_quote "$_tao_setup") --backend docker --check-only"
      echo "\`\`\`"
    fi
    echo "Installing Docker or the NVIDIA runtime changes the system and needs the"
    echo "user's approval first; the \`tao-setup-nvidia-gpu-host\` skill has the steps."
    echo "Manual install reference: https://docs.docker.com/engine/install/"
    echo
  fi
}

tao_text | tao_json
