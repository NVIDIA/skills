---
name: i4h-workflow
version: "0.6.0"
description: Overview of `workflows/agentic/` (IsaacLab-Arena + GR00T/openpi). Use when the user asks what i4h workflow is, what's supported, or where to start.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - agentic-workflow
    - robotics
    - overview
---

# i4h Agentic Workflow

## Purpose

Orient on the agentic workflow before touching a specific stage: which envs/robots/policies are supported, how the `workflows/agentic/` subprojects fit together, and which per-stage skill to invoke next. This skill routes — it runs no pipeline stage itself.

## Base Code

These steps drive the i4h-workflows base code (the `workflows/agentic/` tree). To reuse an existing checkout, set `I4H_WORKFLOWS` to its path (no clone happens). Otherwise this resolves the current repo, or clones to `~/i4h-workflows` — pick that default without prompting. Run every command below from the resolved root:

```bash
# Resolve the i4h-workflows base code (provides workflows/agentic/).
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/workflows/agentic" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/i4h-workflows}"
  [ -d "$ROOT/workflows/agentic" ] || git clone https://github.com/isaac-for-healthcare/i4h-workflows "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"; cd "$ROOT"
```

## Basics

- Env YAMLs at `workflows/agentic/config/environments/<env>.yaml` are the source of truth.
- Each pipeline stage has its own skill. Compose them or use [[i4h-workflow-e2e]] for full runs.
- For env authoring or scene-edit source changes, load `skills/i4h-workflow/references/repo-map.md` before generating code.

## Supported Envs

Fallback when `--list-envs` fails (setup incomplete):

| Env | Robot | Policy |
|---|---|---|
| `scissor_pick_and_place` | SO-ARM 101 | GR00T N1.5 (N1.7 alternative) |
| `locomanip_tray_pick_and_place` | Unitree G1 | GR00T N1.6 (shared `policy.locomanip.*`) |
| `locomanip_push_cart` | Unitree G1 | GR00T N1.6 (shared `policy.locomanip.*`) |
| `assemble_trocar` | Unitree G1 + Dex hands | GR00T N1.5 (inference-only) |
| `ultrasound_liver_scan` | Franka-style arm | openpi PI0 |
| `surgical_reach_psm` | dVRK PSM | GR00T N1.5 or scripted state machine |
| `surgical_reach_dual_psm` | dVRK dual PSM | GR00T N1.5 or scripted state machine |
| `surgical_reach_star` | STAR | GR00T N1.5 or scripted state machine |
| `surgical_lift_block` | dVRK PSM | GR00T N1.5 or scripted state machine |
| `surgical_lift_needle` | dVRK PSM | GR00T N1.5 or scripted state machine |
| `surgical_lift_needle_organs` | dVRK PSM | GR00T N1.5 or scripted state machine |

## Run

Run the step below before answering. Each step is a separate bash call; variables persist in the local agent's tmux session.

### Step 1 — list supported envs

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
"${REPO_ROOT}/workflows/agentic/policy/run.sh" --list-envs
```

If the command fails, use the **Supported Envs** fallback table above.

## Subprojects

| Directory | Purpose |
|---|---|
| `workflows/agentic/arena/` | IsaacLab-Arena envs, scenes, tasks, teleop, record, replay |
| `workflows/agentic/policy/` | Policy daemons and train dispatchers |
| `workflows/agentic/dataset/` | HDF5 → LeRobot conversion and visualization |
| `workflows/agentic/mimic/` | HDF5 trajectory expansion |
| `workflows/agentic/annotator/` | VLM success labels and filtering |
| `workflows/agentic/cosmos/` | Optional Cosmos Transfer video augmentation |
| `workflows/agentic/common/` | Shared config, messaging, robot constants |

## Skill Index

| Skill | Purpose |
|---|---|
| `i4h-workflow-setup` | Install / sync / check third-party deps |
| `i4h-workflow-create` | Add a new env |
| `i4h-workflow-scene-edit` | Edit an existing scene / task / camera |
| `i4h-workflow-dataset-teleop` | Record human demos |
| `i4h-workflow-dataset-replay` | Replay HDF5 episodes |
| `i4h-workflow-dataset-mimic` | Expand HDF5 demos with noise |
| `i4h-workflow-dataset-annotate` | VLM label / filter episodes |
| `i4h-workflow-dataset-convert` | Convert HDF5 to LeRobot |
| `i4h-workflow-finetune` | Train supported envs |
| `i4h-workflow-validate` | Roll out / evaluate policy checkpoints |
| `i4h-workflow-e2e` | Run the full pipeline |
| `i4h-lerobot-viz` | Open the LeRobot HTML viewer |

## Prerequisites

- For any hands-on stage, set up the workflow first — see [[i4h-workflow-setup]] (component venvs and third-party checkouts present).
- Env YAMLs at `workflows/agentic/config/environments/<env>.yaml` are the source of truth.

## Limitations

- Overview/routing only — each pipeline stage has its own skill; this one performs no recording, training, or rollout.

## Troubleshooting

- **Error:** env not found / unsupported — Cause: typo, unregistered env, or setup incomplete. Fix: run `workflows/agentic/policy/run.sh --list-envs`; if that fails, run [[i4h-workflow-setup]] first; otherwise check the **Supported Envs** table.

## Final Response

The response is incomplete unless it includes all three parts below, each separated by a blank line. Do not reply with prose only or run sections together without spacing.

1. **Orientation** — one short paragraph: env YAMLs, subprojects, sim-to-policy pipeline.

2. **Supported envs** — heading plus markdown table from Step 1 (`--list-envs` output: env, stack, description) or the **Supported Envs** fallback table.

3. **Available skills** — heading plus the full **Skill Index** table copied into the response.

Then (after another blank line) recommend [[i4h-workflow-setup]] if setup is not done, and name one next stage skill matched to the user's goal.
