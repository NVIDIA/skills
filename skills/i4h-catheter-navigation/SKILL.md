---
name: i4h-catheter-navigation
version: "0.7.0"
description: Overview of `workflows/catheter_navigation/` (fluorosim DRR, XPBD physics, vasculature digital twin). Use when the user asks what the catheter navigation workflow is, what's supported, or where to start.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - catheter-navigation
    - fluoroscopy
    - overview
---

# i4h Catheter Navigation Workflow

## Purpose

Orient on the endovascular catheter navigation workflow before touching a specific stage: which CLI modes exist, how fluorosim / physics / digital-twin pieces fit together, and which per-stage skill to invoke next. This skill routes - it runs no pipeline stage itself.

## Base Code

These steps drive the i4h-workflows base code (the `workflows/catheter_navigation/` tree). To reuse an existing checkout, set `I4H_WORKFLOWS` to its path (no clone happens). Otherwise this resolves the current repo, or clones to `~/i4h-workflows` - pick that default without prompting. Run every command below from the resolved root:

```bash
# Resolve the i4h-workflows base code (provides workflows/catheter_navigation/).
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/workflows/catheter_navigation" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/i4h-workflows}"
  [ -d "$ROOT/workflows/catheter_navigation" ] || git clone https://github.com/isaac-for-healthcare/i4h-workflows "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"; cd "$ROOT"
```

## Basics

- **Mode registry (source of truth):** `workflows/catheter_navigation/metadata.json` - lists `./i4h` modes and their runtime commands.
- **Human docs:** `workflows/catheter_navigation/README.md` - data download, digital-twin build, viewport controls, troubleshooting.
- Each pipeline stage has its own skill. Compose them or use [[i4h-catheter-navigation-e2e]] for a full smoke run.
- No CT data is shipped; users bring their own NIfTI/DICOM (e.g. TotalSegmentator subset).

## Supported Modes

Fallback when `./i4h modes catheter_navigation` fails (setup incomplete):

| Mode | Purpose |
|---|---|
| `preprocess_ct` | CT -> attenuation volume (`mu_volume.npy`) + optional HU cache |
| `segment_vessels` | Arterial tree mask + centerline (vasculature digital twin) |
| `render_drr` | Single DRR frame (synthetic phantom if no cache) |
| `interactive_viewport` | Live fluoroscopy viewport with XPBD catheter physics |

## Subprojects

| Directory | Purpose |
|---|---|
| `workflows/catheter_navigation/scripts/simulation/fluorosim/examples/interactive_catheter_slang_viewport.py` | Interactive viewport glue script (invoked by `interactive_viewport` mode) |
| `workflows/catheter_navigation/scripts/simulation/fluorosim/rendering/` | Local DRR rendering helpers used by viewport fallback path |
| `workflows/catheter_navigation/metadata.json` | `./i4h` mode registry and command wiring |
| Installed packages | `fluorosim`, `vasculature_digital_twin`, `catheter_vasculature_solver` provide renderer + digital twin + solver runtime |
| `workflows/catheter_navigation/tests/` | CPU smoke tests (no GPU required) |

## Run

Run the step below before answering. Each step is a separate bash call; variables persist in the local agent's tmux session.

### Step 1 - list supported modes

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/catheter_navigation" ] || REPO_ROOT="$HOME/i4h-workflows"
"${REPO_ROOT}/i4h" modes catheter_navigation
```

If the command fails, use the **Supported Modes** fallback table above.

## Skill Index

| Skill | Purpose |
|---|---|
| `i4h-catheter-navigation-setup` | Host/GPU preflight, PYTHONPATH, smoke-test verify |
| `i4h-catheter-navigation-digital-twin` | Download CT data, preprocess, segment vessels |
| `i4h-catheter-navigation-render-drr` | Render a single DRR frame |
| `i4h-catheter-navigation-viewport` | Launch interactive fluoroscopy + catheter sim |
| `i4h-catheter-navigation-smoke` | Run CPU unittest smoke suite |
| `i4h-catheter-navigation-e2e` | End-to-end smoke (setup -> digital twin -> render -> tests) |

## Prerequisites

- NVIDIA GPU with CUDA >= 12.8 and Vulkan-capable driver for GPU modes (render, viewport).
- For any hands-on stage, run [[i4h-catheter-navigation-setup]] first.
- For patient-specific runs, run [[i4h-catheter-navigation-digital-twin]] before viewport or cache-based DRR.

## Limitations

- Overview/routing only - each stage has its own skill; this one performs no preprocessing, rendering, or simulation.
- RL/IL training integration with Isaac Lab is not yet a first-class skill stage in v0.7; use the workflow README and fluorosim API for custom data generation.

## Troubleshooting

- **Error:** mode not found - Cause: typo or `./i4h` not run from repo root. Fix: `cd` to repo root and re-run Step 1; inspect `workflows/catheter_navigation/metadata.json`.
- **Error:** import errors for `fluorosim` / `vasculature_digital_twin` / `catheter_vasculature_solver` - Cause: dependencies not installed in active env. Fix: run [[i4h-catheter-navigation-setup]] and install README requirements.

## Final Response

The response is incomplete unless it includes all three parts below, each separated by a blank line.

1. **Orientation** - one short paragraph: fluorosim stack, digital twin, `./i4h` modes.

2. **Supported modes** - heading plus markdown table from Step 1 or the **Supported Modes** fallback table.

3. **Available skills** - heading plus the full **Skill Index** table copied into the response.

Then recommend [[i4h-catheter-navigation-setup]] if setup is not done, and name one next stage skill matched to the user's goal.
