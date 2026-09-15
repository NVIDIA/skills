#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fallback setup for `nemo-speech-asr-finetune` (see workflow.md §4b).
#
# `nemo-speech-asr-finetune` is a project-local Claude Code skill that lives
# inside the NeMo/Speech repo itself (.claude/skills/nemo-speech-asr-finetune),
# not the nvidia-skills catalog, so it is not installable with the standard
# `npx skills add nvidia/skills --skill ...` flow. Try that installer against
# the owning repo first:
#
#   npx skills add NVIDIA-NeMo/Speech --skill nemo-speech-asr-finetune \
#       --agent claude-code --global --yes
#
# Only fall back to this script if that does not work. This does NOT make the
# sub-skill's commands runnable by itself -- its commands are paths relative to
# the NeMo repo root, so the orchestrator must still change its working
# context into that checkout before Stage 5 (see workflow.md §4b). This script
# only makes the skill discoverable and records where the real repo root is.
#
# Usage: link-nemo-subskill.sh <path-to-NeMo-Speech-checkout>

set -euo pipefail

NEMO_ROOT="${1:?Usage: $0 <path-to-NeMo-Speech-checkout>}"
SKILL_SRC="$NEMO_ROOT/.claude/skills/nemo-speech-asr-finetune"
SKILL_DST="$HOME/.claude/skills/nemo-speech-asr-finetune"

# 1. Validate the source before touching anything -- never create a
#    dangling/garbage symlink.
if [[ ! -f "$SKILL_SRC/SKILL.md" ]]; then
  echo "ERROR: $SKILL_SRC/SKILL.md not found." >&2
  echo "Not a valid checkout, or nemo-speech-asr-finetune isn't present at this NeMo version." >&2
  exit 1
fi

mkdir -p "$HOME/.claude/skills"

# 2. Refuse to clobber a real (non-symlink) directory someone may have put
#    there on purpose.
if [[ -e "$SKILL_DST" && ! -L "$SKILL_DST" ]]; then
  echo "ERROR: $SKILL_DST exists and is a real directory, not a symlink." >&2
  echo "Refusing to overwrite -- remove or rename it manually first." >&2
  exit 1
fi

# 3. If already a symlink, confirm before silently repointing to a different
#    NeMo checkout (avoids silently swapping NeMo versions on a prior setup).
if [[ -L "$SKILL_DST" ]]; then
  CURRENT="$(readlink -f "$SKILL_DST")"
  NEW="$(readlink -f "$SKILL_SRC")"
  if [[ "$CURRENT" != "$NEW" ]]; then
    echo "WARNING: $SKILL_DST currently points to:" >&2
    echo "  $CURRENT" >&2
    echo "Would repoint it to:" >&2
    echo "  $NEW" >&2
    read -r -p "Repoint it? [y/N] " ans
    [[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "Aborted."; exit 1; }
  fi
fi

# 4. Link. Idempotent -- guarded above, so -f is now safe.
ln -sfn "$SKILL_SRC" "$SKILL_DST"

# 5. Record the real repo root explicitly. This is what actually closes the
#    execution-time gap: one canonical lookup instead of re-deriving it from
#    the symlink (readlink + path math) every time the sub-skill is invoked.
echo "$NEMO_ROOT" > "$HOME/.claude/skills/.nemo-speech-asr-finetune.repo-root"

echo "Linked. nemo-speech-asr-finetune is now discoverable globally."
echo "Its commands are relative to: $NEMO_ROOT"
echo "cd there (or read $HOME/.claude/skills/.nemo-speech-asr-finetune.repo-root) before running any of its commands."
