#!/usr/bin/env bash
# Install the NVIDIA omniverse-realtime-viewer skill into Claude Code.
# Run once on your local machine before opening this project in Claude Code.
#
# What this does:
#   Adds the skill to ~/.claude/skills/ so Claude Code loads it automatically
#   when you work on viewer projects. The skill provides implementation
#   contracts for ovrtx, ovstream, camera controls, and WebRTC streaming.
set -euo pipefail

echo "Installing NVIDIA omniverse-realtime-viewer skill for Claude Code..."

# Install the skill from the NVIDIA catalog.
npx skills add nvidia/skills \
  --skill omniverse-realtime-viewer \
  --agent claude-code

echo ""
echo "Skill installed. Claude Code will load it automatically."
echo "Verify with:  claude skills list"
