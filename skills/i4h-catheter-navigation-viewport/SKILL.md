---
name: i4h-catheter-navigation-viewport
version: "0.7.0"
description: Launch the interactive Slang fluoroscopy viewport with XPBD catheter physics. Use when asked to open the viewport, teleop a catheter, or demo fluoro navigation.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - catheter-navigation
    - viewport
    - interactive
---

# i4h Catheter Navigation - Interactive Viewport

## Purpose

Launch the interactive fluoroscopy viewport with real-time XPBD catheter physics, DSA, C-arm presets, and centerline-guided insertion. Requires a digital twin cache from [[i4h-catheter-navigation-digital-twin]].

## Base Code

```bash
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/workflows/catheter_navigation" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/i4h-workflows}"
  [ -d "$ROOT/workflows/catheter_navigation" ] || git clone https://github.com/isaac-for-healthcare/i4h-workflows "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"; cd "$ROOT"
```

## Basics

- Entry mode: `./i4h run catheter_navigation interactive_viewport` (preferred)
- Requires display (X11); remote/VNC may need `QT_QPA_PLATFORM=xcb`.
- Recommended flags: `--vessel-source real --insertion-axis centerline --dsa --key-hold-ttl 0.20`

## Run

Run the steps below in order. Each step is a separate bash call; variables persist in the local agent's tmux session.

### Step 1 - resolve cache and run dir

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/catheter_navigation" ] || REPO_ROOT="$HOME/i4h-workflows"
WF_ROOT="${REPO_ROOT}/workflows/catheter_navigation"
RUN_DIR="${WF_ROOT}/runs/viewport_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${WF_ROOT}/runs/.latest"

CACHE="${CACHE:-/tmp/ct_cache}"
if [ ! -f "${CACHE}/mu_volume.npy" ]; then
  echo "viewport: set CACHE to a digital-twin dir (missing mu_volume.npy). Run i4h-catheter-navigation-digital-twin first." >&2
  exit 1
fi
```

### Step 2 - launch viewport

```bash
QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}" \
"${REPO_ROOT}/i4h" run catheter_navigation interactive_viewport --local \
  --run-args="--ct-dir ${CACHE} --vessel-source real --insertion-axis centerline --det-size 1024 --pixel-spacing-mm 0.6 --dsa" \
  2>&1 | tee "${RUN_DIR}/logs/viewport.log"
```

## Controls (summary)

| Input | Action |
|---|---|
| W / S | Advance / retract catheter along centerline |
| A / D | Rotate catheter CCW / CW |
| F / C | Toggle vessel / centerline overlay |
| X | Toggle DSA contrast bolus |
| 1-4 | C-arm presets (AP, LAO-45, Lateral, RAO-30) |
| Space | Pause / resume |
| R | Reset catheter |
| Q / Esc | Quit |

Full table: `workflows/catheter_navigation/README.md`.

## Prerequisites

- [[i4h-catheter-navigation-setup]] and [[i4h-catheter-navigation-digital-twin]] completed.
- Display server (local X11 or remote desktop with keyboard focus).
- GPU with >= 8 GB VRAM (16 GB recommended).

## Limitations

- Interactive GUI - not suitable for headless CI.
- Keyboard focus on remote desktops may require pointer over the fluoro window.

## Troubleshooting

- **Error:** controls appear dead on remote X11 - Fix: `QT_QPA_PLATFORM=xcb` (Step 2 sets this by default).
- **Error:** missing cache - Fix: run [[i4h-catheter-navigation-digital-twin]].
- **Error:** Vulkan / slang init failure - Fix: verify GPU driver; try Docker without `--local`.

## Final Response

Report `CACHE` used, note any display/input workarounds applied, and point to README controls if the user needs the full key map.
