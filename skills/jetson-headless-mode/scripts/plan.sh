#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# plan.sh
#
# Build a "headless mode" plan from a jetson-memory-audit JSON. Emits the same
# JSON shape as recommend.py: an array of {layer, knob, estimated_savings_mb,
# safety, command, reversible_command, rationale, reference, notes}.
#
# Only safety="safe" entries are produced; apply.sh refuses to run anything else.
#
# Usage:
#   plan.sh --audit PATH        # PATH to audit JSON, or '-' for stdin
#   plan.sh --audit - --human   # pretty-print
#
# Exit codes:
#   0  ok
#   2  audit JSON missing or malformed

set -uo pipefail

AUDIT=""
HUMAN=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../../jetson-diagnostic/scripts/common.sh
# shellcheck disable=SC1091
. "$SKILLS_ROOT/jetson-diagnostic/scripts/common.sh"

while [ $# -gt 0 ]; do
    case "$1" in
        --audit) need_value "$@"; AUDIT="$2"; shift 2 ;;
        --human) HUMAN=1; shift ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 64 ;;
    esac
done

if [ -z "$AUDIT" ]; then
    echo "ERROR: --audit is required (path to audit JSON, or '-')" >&2
    exit 64
fi

# shellcheck source=../../jetson-diagnostic/scripts/detect_jetson.sh
# shellcheck disable=SC1091
. "$SKILLS_ROOT/jetson-diagnostic/scripts/detect_jetson.sh"

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
if [ "$AUDIT" = "-" ]; then
    cat > "$TMP" || { echo "ERROR: failed to read audit JSON from stdin" >&2; exit 2; }
else
    [ -r "$AUDIT" ] || { echo "ERROR: cannot read audit file: $AUDIT" >&2; exit 2; }
    cp "$AUDIT" "$TMP" || { echo "ERROR: failed to read audit file: $AUDIT" >&2; exit 2; }
fi

PRETTY="$HUMAN" AUDIT_PATH="$TMP" python3 <<'PY'
import json, os, sys

try:
    with open(os.environ["AUDIT_PATH"], "r", encoding="utf-8") as f:
        audit = json.load(f)
except (OSError, json.JSONDecodeError) as e:
    print(f"ERROR: failed to read audit JSON: {e}", file=sys.stderr)
    sys.exit(2)

target = audit.get("default_systemd_target", "unknown")
services = audit.get("candidate_services", {})

def is_active(svc):
    return services.get(svc, {}).get("active") == "active"

REF = "SKILL.md"
recs = []

if target == "graphical.target":
    recs.append({
        "layer": 1,
        "knob": "disable-graphical-target",
        "estimated_savings_mb": 865,
        "safety": "safe",
        "rationale": "Default boot target is graphical.target. Switching to multi-user.target stops the display manager and Xorg/Wayland stack on next boot.",
        "reference": REF,
        "command": "sudo systemctl set-default multi-user.target",
        "reversible_command": "sudo systemctl set-default graphical.target",
        "notes": ["Reboot required to fully reclaim memory."],
    })

for svc in ("gdm3", "gdm", "lightdm", "sddm", "display-manager"):
    if is_active(svc):
        recs.append({
            "layer": 1,
            "knob": f"stop-{svc}",
            "estimated_savings_mb": 200,
            "safety": "safe",
            "rationale": f"Display manager {svc} is active; not needed in headless deployments.",
            "reference": REF,
            "command": f"sudo systemctl disable --now {svc}",
            "reversible_command": f"sudo systemctl enable --now {svc}",
            "notes": [],
        })

aux = {
    "pulseaudio": ("Audio daemon; disable if no audio I/O is needed.", 8),
    "bluetooth":  ("Bluetooth stack; disable on wired-only deployments.", 6),
    "ModemManager": ("Cellular modem manager; disable if no WWAN.", 4),
    "cups":       ("Print server; almost never needed on edge.", 5),
    "cups-browsed": ("CUPS browse helper; same as above.", 3),
    "snapd":      ("Snap package daemon; disable if you don't ship snaps.", 30),
    "whoopsie":   ("Ubuntu crash reporter; disable on production.", 4),
    "kerneloops": ("Kernel oops reporter; disable on production.", 2),
    "avahi-daemon": ("mDNS responder; disable if not used.", 3),
    "unattended-upgrades": ("Auto-update daemon; disable on locked production builds.", 6),
    "packagekit": ("Background package indexer; disable on production.", 8),
}
for svc, (reason, mb) in aux.items():
    if is_active(svc):
        recs.append({
            "layer": 2,
            "knob": f"stop-{svc}",
            "estimated_savings_mb": mb,
            "safety": "safe",
            "rationale": reason,
            "reference": REF,
            "command": f"sudo systemctl disable --now {svc}",
            "reversible_command": f"sudo systemctl enable --now {svc}",
            "notes": [],
        })

out = {
    "sku": audit.get("sku", "unknown"),
    "variant": audit.get("variant", "unknown"),
    "mem_total_gb": audit.get("mem_total_gb", 0),
    "l4t_version": audit.get("l4t_version", "unknown"),
    "product_model": audit.get("product_model", ""),
    "use_case": "headless",
    "estimated_total_savings_mb": sum(max(r["estimated_savings_mb"], 0) for r in recs),
    "recommendations": recs,
}
indent = 2 if os.environ.get("PRETTY") == "1" else None
json.dump(out, sys.stdout, indent=indent)
sys.stdout.write("\n")
PY
