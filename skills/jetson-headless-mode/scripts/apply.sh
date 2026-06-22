#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# apply.sh
#
# Apply safe, reversible user-space memory-reclamation knobs from a plan JSON.
# Defaults to dry-run; --apply is required to mutate the system.
#
# Usage:
#   apply.sh --plan plan.json [--apply] [--reboot] [--drop-caches]
#
#   --plan PATH     JSON output from plan.sh (or '-' for stdin).
#   --apply         Actually run the commands. Without this flag, only print them.
#   --reboot        After applying, reboot the system. Off by default.
#   --drop-caches   After applying (and only if not rebooting), flush the kernel
#                   page cache so freed memory shows up in `free -h` immediately.
#                   Off by default.
#
# This script is limited to safe user-space changes and does not:
#   - Touch /boot/extlinux/extlinux.conf
#   - Modify the device tree or carveouts
#   - Trigger a re-flash
#   - Apply recommendations not filtered as "safety": "safe"
#
# Anything in those classes is out of scope for this device-runtime script.
#
# Exit codes:
#   0  ok
#   2  not on a Jetson, or plan JSON malformed

set -uo pipefail

PLAN=""
APPLY=0
DO_REBOOT=0
DROP_CACHES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../../jetson-diagnostic/scripts/common.sh
# shellcheck disable=SC1091
. "$SKILLS_ROOT/jetson-diagnostic/scripts/common.sh"

while [ $# -gt 0 ]; do
    case "$1" in
        --plan)         need_value "$@"; PLAN="$2"; shift 2 ;;
        --apply)        APPLY=1; shift ;;
        --reboot)       DO_REBOOT=1; shift ;;
        --drop-caches)  DROP_CACHES=1; shift ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 64 ;;
    esac
done

if [ -z "$PLAN" ]; then
    echo "ERROR: --plan is required (path to plan JSON, or '-')" >&2
    exit 64
fi

# shellcheck source=../../jetson-diagnostic/scripts/detect_jetson.sh
# shellcheck disable=SC1091
. "$SKILLS_ROOT/jetson-diagnostic/scripts/detect_jetson.sh"

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is required to parse the plan." >&2
    exit 2
fi

if [ "$PLAN" = "-" ]; then
    PLAN_JSON=$(cat) || { echo "ERROR: failed to read plan JSON from stdin" >&2; exit 2; }
else
    [ -r "$PLAN" ] || { echo "ERROR: cannot read plan file: $PLAN" >&2; exit 64; }
    PLAN_JSON=$(cat "$PLAN") || { echo "ERROR: failed to read plan file: $PLAN" >&2; exit 2; }
fi

SAFE_LINES=$(printf '%s' "$PLAN_JSON" | python3 -c '
import json, sys
try:
    plan = json.load(sys.stdin)
except json.JSONDecodeError as e:
    print(f"ERROR: malformed plan JSON: {e}", file=sys.stderr)
    sys.exit(2)
if not isinstance(plan, dict):
    print("ERROR: malformed plan JSON: top-level value must be an object", file=sys.stderr)
    sys.exit(2)
recommendations = plan.get("recommendations", [])
if not isinstance(recommendations, list):
    print("ERROR: malformed plan JSON: recommendations must be an array", file=sys.stderr)
    sys.exit(2)
for r in recommendations:
    if not isinstance(r, dict): continue
    if r.get("safety") != "safe": continue
    cmd = r.get("command")
    if not cmd: continue
    knob = r.get("knob", "?")
    savings = r.get("estimated_savings_mb", 0)
    print(f"{cmd}\t{knob}\t{savings}\tsafe")
') || exit 2

run_allowed_cmd() {
    local arg
    case "$1" in
        sudo\ systemctl\ set-default\ *)
            arg="${1#sudo systemctl set-default }"
            case "$arg" in ''|*[!A-Za-z0-9_.@-]*) return 64 ;; esac
            sudo systemctl set-default "$arg"
            ;;
        sudo\ systemctl\ disable\ --now\ *)
            arg="${1#sudo systemctl disable --now }"
            case "$arg" in ''|*[!A-Za-z0-9_.@-]*) return 64 ;; esac
            sudo systemctl disable --now "$arg"
            ;;
        sudo\ systemctl\ enable\ --now\ *)
            arg="${1#sudo systemctl enable --now }"
            case "$arg" in ''|*[!A-Za-z0-9_.@-]*) return 64 ;; esac
            sudo systemctl enable --now "$arg"
            ;;
        *)
            return 64
            ;;
    esac
}

if [ -z "$SAFE_LINES" ]; then
    echo "Nothing to apply: plan contains no knobs with safety=safe."
    exit 0
fi

echo "Plan contains the following safe knobs:"
echo "$SAFE_LINES" | awk -F'\t' '{ printf "  - %-32s ~%s MB    %s\n", $2, $3, $1 }'
echo

if [ "$APPLY" -ne 1 ]; then
    echo "DRY RUN. Re-run with --apply to execute the commands above."
    exit 0
fi

echo "About to apply $(echo "$SAFE_LINES" | wc -l) commands as root."
echo "Press Ctrl-C within 5 seconds to abort."
sleep 5

echo "$SAFE_LINES" | while IFS=$'\t' read -r cmd knob savings safety; do
    if [ "$safety" != "safe" ]; then
        echo "WARN: skipping recommendation without safety=safe for knob '$knob'." >&2
        continue
    fi
    echo "+ $cmd   # $knob (~${savings} MB)"
    if run_allowed_cmd "$cmd"; then
        continue
    else
        rc=$?
    fi
    if [ "$rc" -eq 64 ]; then
        echo "WARN: skipping unsupported command for knob '$knob': $cmd" >&2
    else
        echo "WARN: command failed for knob '$knob'. Continuing." >&2
    fi
done

echo
if [ "$DO_REBOOT" -eq 1 ]; then
    echo "Rebooting in 5 seconds..."
    sleep 5
    sudo systemctl reboot
else
    echo "Done. Some changes (e.g. default systemd target) only take effect after reboot."
    if [ "$DROP_CACHES" -eq 1 ]; then
        echo
        echo "Flushing page cache so freed memory shows up immediately..."
        sudo sync && sudo sysctl -w vm.drop_caches=3 >/dev/null
        echo "Done. Run jetson-memory-audit/scripts/audit.sh to verify the delta."
    else
        echo "Tip: pass --drop-caches to flush reclaimable page cache so 'free -h' reflects the new baseline."
    fi
fi
