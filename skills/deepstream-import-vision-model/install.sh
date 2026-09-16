#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# deepstream-import-vision-model — Install script
# Installs the skills and runtime scripts into a target project.
# Supports Claude Code (.claude/), Codex (.codex/), and Cursor (.cursor/) out of the box.
#
# Usage:
#   bash install.sh --target <project-path> [--dry-run] [--no-cursor]
#
# Examples:
#   bash install.sh --target ~/work/my-project --dry-run
#   bash install.sh --target ~/work/my-project
#   bash install.sh --target ~/work/my-project --no-cursor
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"

TARGET=""
DRY_RUN=false
NO_CURSOR=false

usage() {
    local rc="${1:-0}"
    cat <<EOF
Usage: $0 --target <project-path> [--dry-run] [--no-cursor]

  --target <path>   Project directory to install into (required)
  --dry-run         Show what would be done without making any changes
  --no-cursor       Skip Cursor (.cursor/skills/) installation
  -h, --help        Show this help

Example:
  bash install.sh --target ~/work/my-deepstream-project
EOF
    exit "$rc"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            if [[ $# -lt 2 || -z "${2:-}" || "$2" == -* ]]; then
                echo "Error: --target requires a path argument" >&2
                usage 1
            fi
            TARGET="$2"
            shift 2
            ;;
        --dry-run)          DRY_RUN=true; shift ;;
        --no-cursor)        NO_CURSOR=true; shift ;;
        -h|--help)          usage 0 ;;
        *)                  echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

# --- TARGET validation (hardened) ----------------------------------------------
if [[ -z "$TARGET" ]]; then
    echo "Error: --target is required" >&2
    usage 1
fi

case "$TARGET" in
    ""|"/"|*..*)
        echo "Error: invalid --target value: $TARGET" >&2
        exit 1
        ;;
esac

if [[ ! -d "$TARGET" ]]; then
    echo "Error: target directory not found: $TARGET" >&2
    exit 1
fi

TARGET="$(cd "$TARGET" && pwd -P)"

if [[ "$TARGET" == "/" ]] || [[ ${#TARGET} -lt 3 ]]; then
    echo "Error: refusing to install into '$TARGET' (path too short / is root)" >&2
    exit 1
fi
# -------------------------------------------------------------------------------

do_copy() {
    local src="$1"
    local dest="$2"
    if $DRY_RUN; then
        echo "  [dry-run] cp -r $src -> $dest"
    else
        mkdir -p "$(dirname "$dest")"
        cp -r "$src" "$dest"
        echo "  Copied: $(basename "$dest")"
    fi
}

# Scoped cleanup helper: only removes a path that is a subdirectory of $TARGET.
safe_rm_under_target() {
    local path="$1"
    local resolved
    resolved="$(cd "$(dirname "$path")" 2>/dev/null && pwd -P)/$(basename "$path")" || return 1
    case "$resolved" in
        "$TARGET"/*) ;;
        *) echo "  Refusing to remove $resolved (not under $TARGET)" >&2; return 1 ;;
    esac
    if $DRY_RUN; then
        echo "  [dry-run] rm -rf $resolved"
    else
        echo "  Removing: $resolved"
        rm -rf "$resolved"
    fi
}

# Install the WHOLE self-contained skill (SKILL.md + references/ + scripts/ + setup.sh + .gitattributes)
# into a skills directory, so it runs through Docker as one mounted tree (-v <root>:/work). Works for
# .claude/skills/, .codex/skills/, and .cursor/skills/.
install_skills_to_dir() {
    local skills_dir="$1"
    local skill_dest="$skills_dir/deepstream-import-vision-model"
    if [[ -d "$skill_dest" ]]; then
        local dest_real
        dest_real="$(cd "$skill_dest" && pwd -P)"
        if [[ "$dest_real" == "$SKILL_DIR" ]]; then
            echo "  Already installed at $skill_dest; source and destination are identical"
            return
        fi
        safe_rm_under_target "$skill_dest"
    fi
    if $DRY_RUN; then
        echo "  [dry-run] cp -r $SKILL_DIR -> $skill_dest  (whole skill; minus __pycache__/*.pyc/built parser .so)"
        return
    fi
    mkdir -p "$(dirname "$skill_dest")"
    cp -r "$SKILL_DIR" "$skill_dest"
    # strip machine-local build artifacts (the venv + parser .so are rebuilt in-container by setup.sh)
    find "$skill_dest" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
    find "$skill_dest" \( -name '*.pyc' -o -name '*.so' -o -name '*.o' \) -delete 2>/dev/null || true
    echo "  Copied self-contained skill -> $skill_dest"
}

echo "=== deepstream-import-vision-model Install ==="
echo "Skill dir: $SKILL_DIR"
echo "Target:    $TARGET"
echo "Cursor:    $($NO_CURSOR && echo "disabled (--no-cursor)" || echo "enabled")"
echo ""

# Step 1: Claude Code and Codex skills
echo "Claude Code skills -> $TARGET/.claude/skills/"
install_skills_to_dir "$TARGET/.claude/skills"

echo ""
echo "Codex skills -> $TARGET/.codex/skills/"
install_skills_to_dir "$TARGET/.codex/skills"

echo ""
# Step 2: Cursor — skills only (Cursor does not support agents)
if ! $NO_CURSOR; then
    echo "Cursor skills -> $TARGET/.cursor/skills/"
    install_skills_to_dir "$TARGET/.cursor/skills"
    echo ""
fi

echo ""
echo "=== Done ==="
echo ""
echo "Next — bootstrap the environment IN the container (nothing installs on the host):"
echo "  docker run --rm -it --gpus all --shm-size=16g -v \"\$PWD\":/work -w /work \\"
echo "    --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch \\"
echo "    .claude/skills/deepstream-import-vision-model/setup.sh"
echo "  (PowerShell: use -v \"\${PWD}:/work\"  ·  see references/windows.md)"
echo ""
echo "Claude Code — invoke the skill:"
echo "  Use deepstream-import-vision-model to run this model: https://huggingface.co/onnx-community/yolov8n"
if ! $NO_CURSOR; then
    echo ""
    echo "Cursor — invoke the skill:"
    echo "  @deepstream-import-vision-model run this model: https://huggingface.co/onnx-community/yolov8n"
fi
