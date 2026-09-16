---
name: i4h-workflow
description: Orient users to the i4h workflow runtime and route them to the correct stage skill. Use for architecture, support, or where-to-start questions; do not execute a known stage.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  tags:
    - isaac-for-healthcare
    - i4h
    - robotics
    - onboarding
---

# i4h Workflows

## Purpose

Orient the user from live repository facts, then hand execution to the narrowest stage skill.

## Instructions

1. Run the base-checkout resolver.
2. Read live support and `DESIGN.md`.
3. Use only current architecture facts in the answer.
4. Use the narrowest stage skill for execution.

## Resolve the checkout

```bash
export I4H_WORKFLOWS_REPO_URL="${I4H_WORKFLOWS_REPO_URL:-https://github.com/isaac-for-healthcare/i4h-workflows}"
I4H_REPO_DIR_NAME="${I4H_WORKFLOWS_REPO_URL%/}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME##*/}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME##*:}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME%.git}"
[ -n "$I4H_REPO_DIR_NAME" ] || { echo "Cannot derive a checkout name from I4H_WORKFLOWS_REPO_URL" >&2; exit 2; }
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/workflows/i4h_workflows" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/$I4H_REPO_DIR_NAME}"
  [ -d "$ROOT/workflows/i4h_workflows" ] || git clone "$I4H_WORKFLOWS_REPO_URL" "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"
cd "$ROOT"
```

Treat this resolver as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `workflows/i4h_workflows`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

## Inspect before answering

Read `./DESIGN.md` for architecture and `skills/i4h-workflow/references/repo-map.md` for ownership. Discover current support instead of copying a static table:

```bash
./run.sh list
```

If discovery fails because setup is incomplete, report that limitation and route to `i4h-workflow-setup`.

## Explain the design

Keep the summary precise:

- A Scene owns the simulated world, assets, embodiment, cameras, randomization, adapters, and reset hooks.
- A Task owns one reusable capability. It reads `ctx.scene`, writes `ctx.act`, and never advances the simulator.
- A Workflow selects one Scene, exposes run-mode-specific `TaskGraph` builders, and owns goal semantics. A run mode answers how that workflow should run; code and CLI use the shorter term `mode`.
- The Engine schedules graph nodes; the shared `SimulationRunner` alone resets, steps, renders, records, retries whole episodes, and prints run summaries.
- Online RL is a separate training lifecycle: its trainer owns vectorized stepping and returns a checkpoint to the normal policy Task and `SimulationRunner` validation path.
- Simulator-compatible exported RSL-RL actors may run as in-process Tasks; incompatible foundation-model policy stacks remain remote.
- Remote policy stacks run out of process and communicate over Zenoh; offline dataset tools remain independent of the simulator.
- Python owns behavior. Manifests carry facts across dependency boundaries.

Do not describe retired environment YAMLs, per-mode runners, or separate policy/Arena launchers.

## Route the next action

| Goal | Skill |
|---|---|
| Install, sync, or repair dependencies | `i4h-workflow-setup` |
| Create a new workflow/environment | `i4h-workflow-create` |
| Edit an existing scene, camera, task, or success rule | `i4h-workflow-scene-edit` |
| Record demonstrations | `i4h-workflow-dataset-teleop` |
| Replay HDF5 | `i4h-workflow-dataset-replay` |
| Augment HDF5 | `i4h-workflow-dataset-mimic` |
| Grade/filter HDF5 with a VLM | `i4h-workflow-dataset-annotate` |
| Convert HDF5 to LeRobot | `i4h-workflow-dataset-convert` |
| Inspect LeRobot in a browser | `i4h-lerobot-viz` |
| Fine-tune a manifest-backed policy task | `i4h-workflow-finetune` |
| RL post-train a supported policy in simulation | `i4h-workflow-train-rl` |
| Run policy or rule-based rollouts | `i4h-workflow-validate` |
| Run the maintained complete pipeline | `i4h-workflow-e2e` |

For `Stop all`, do not load a stage skill. Run `./stop.sh all` from the repository root and report the stopped process count.

## Troubleshooting

If discovery fails, verify the resolved checkout and run setup. If a mode is absent, report it as unsupported.

## Prerequisites

Require a readable base checkout or network access to clone it.

## Limitations

This router does not install, author, simulate, process data, train, or evaluate.

## Examples

- `What does the i4h workflow include, and where should I start?` → inspect live support, summarize `DESIGN.md`, and recommend one stage skill.

## Completion gate

Answer with the live workflow/mode list, a short architecture summary, and one concrete next skill. If the requested workflow or mode is absent from `run.sh list`, say it is unsupported instead of inventing a command.
