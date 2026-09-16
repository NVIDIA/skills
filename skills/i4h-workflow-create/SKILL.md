---
name: i4h-workflow-create
description: Create a minimal blank Workflow scaffold with a Scene containing ground and light plus an idle run mode. Use for fast new Workflow scaffolding.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  tags:
    - isaac-for-healthcare
    - i4h
    - isaac-sim
    - workflow-authoring
---

# Create a Blank i4h Workflow

## Purpose

Create a complete runnable blank Workflow without searching for an existing Workflow or booting Isaac Sim.

## Instructions

1. Resolve the checkout using `I4H_WORKFLOWS` when set; otherwise use the current git root. If neither contains `workflows/i4h_workflows`, use the checkout resolver from the repository `AGENTS.md`.
2. Choose a lowercase snake_case Workflow id.
3. Select exactly one approved product specialty: `laparoscopic-robotics`, `ultrasound-robotics`, `endoluminal-robotics`, or `hospital-automation-robotics`.
4. Generate the blank Workflow and run the static validation below.

If the request does not identify a specialty, ask the user to choose one; do not infer a product category from the workflow name alone.

Run:

```bash
./scripts/create_blank_environment.py <workflow_id> --specialty <specialty>
```

This is the only authoring utility that generates workflow-specific source. It rejects an existing public workflow id in any specialty before dry-run output or file creation. It writes the fixed empty templates that establish the design boundary; later scene and task code is written by the coding agent. Use `--dry-run` to preview its output, `--description TEXT` to customize the scene-manifest description, and `--validate` to run the focused static checks after creation. Do not combine `--dry-run` and `--validate`.

The script refuses to overwrite files and creates only:

```text
./
├── arena/i4h_arena/assets/<workflow_id>.py
├── arena/i4h_arena/scenes/<workflow_id>.py
├── arena/i4h_arena/scenes/manifest/<workflow_id>.yaml
├── workflows/i4h_workflows/<specialty>/<workflow_id>.py
└── workflows/tests/test_<workflow_id>_contract.py
```

The result contains a ground plane, dome light, no embodiment or declared robots, no task-specific objects or cameras, and one `idle` mode. Its contract test verifies durable Workflow and manifest invariants rather than asserting that the Scene stays blank, so later Scene, Task, and run-mode authoring does not require rewriting the test. Do not inspect robot assets, choose a policy, create Task manifests, load upstream Isaac Sim skills, or launch the simulator during blank creation.

## Utility

| Command | Purpose | Arguments |
|---|---|---|
| `./scripts/create_blank_environment.py` | Create the complete overwrite-safe blank Workflow | `<workflow_id> --specialty <specialty> [--description TEXT] [--dry-run] [--validate]` |

## Validate

Run only fast static checks:

```bash
./scripts/create_blank_environment.py <workflow_id> --specialty <specialty> --validate
```

When the Workflow already exists because it was created without `--validate`, run the equivalent checks directly:

```bash
cd "$(git rev-parse --show-toplevel)"
./run.sh show <workflow_id> --mode idle
./run.sh lint <workflow_id> --mode idle
./run.sh lint --all
workflows/.venv/bin/python -m pytest workflows/tests/test_<workflow_id>_contract.py -q
arena/.venv/bin/python -m pytest arena/tests/test_scene.py -q
```

`--validate` requires the workflow and Arena environments; run `./setup.sh` first if either is missing. It verifies the generated contract and the shared zero-DOF blank-scene runtime adapter without launching Isaac Sim. If a component virtual environment is elsewhere, use that component's Python for its direct test command. Do not run visible or dynamic validation unless the user explicitly requests it.

## Troubleshooting

- If the checkout cannot be resolved, set `I4H_WORKFLOWS` to the existing repository root or use the resolver in `AGENTS.md`.
- If generation reports that the workflow id already exists, report its specialty source and ask whether to edit it or use a different id; never overwrite, relocate, or duplicate it.
- If `--validate` reports a missing component environment, run `./setup.sh` and repeat the same validation.

## Prerequisites

Require a writable root-level i4h-workflows checkout and Python 3.11 or newer. Static `--validate` also requires the workflow and Arena component environments.

## Limitations

This skill creates only the five-file idle scaffold. It does not add an embodiment, task behavior, policy, cameras, or task-specific assets, and it does not visibly validate the Scene.

## Continue

Treat creation as the first stage of an incremental workflow:

1. Create the blank Workflow scaffold with this skill: blank Scene, scene manifest, and idle-only Workflow.
2. Use `i4h-workflow-scene-edit` to add and visibly verify assets, robots, cameras, layout, and physics, then bake the confirmed Scene.
3. Define behavior only after the user provides a concrete goal and success condition. Reuse compatible Tasks; create a focused Task only when the behavior is new. Do not create an empty placeholder Task.
4. Add rule-based, teleop, replay, or policy modes only when requested and only when their Task and action contracts are runnable.
5. Validate every newly enabled mode with the owning stage skill.

The blank Workflow is intentionally useful before behavior exists: its shared idle Task opens the Scene for authoring, so a blank task manifest or implementation would add no capability.

## Examples

- `Create a blank hospital-automation-robotics workflow named my_workflow.` → generate `my_workflow` in the hospital automation specialty and run static validation.
- `Make a new laparoscopic-robotics workflow called training_sandbox as fast as possible. Start blank.` → generate the same five-file idle-only scaffold under `training_sandbox` in the laparoscopic specialty.

## Completion gate

Report the created id, five files, idle-only status, and static validation results. Do not commit unless explicitly asked.
