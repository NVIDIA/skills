---
name: i4h-workflow-create
version: "0.7.0"
description: Create a new agentic env by forking an existing env. Use for new env/task scaffolding, not scene edits or baking.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - agentic-workflow
    - environment
    - scaffolding
---

# i4h Workflow - Create Env

## Purpose

Create the first runnable version of a new `workflows/agentic` environment by forking the closest existing env. Keep this skill focused on env scaffolding: YAML, assets, task, env class, runtime, and validation. Do not use this skill to polish an existing scene, add optional props/cameras, or bake bridge edits; use [[i4h-workflow-scene-edit]] for that.

## Base Code

Resolve and work from the i4h-workflows root:

```bash
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/workflows/agentic" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/i4h-workflows}"
  [ -d "$ROOT/workflows/agentic" ] || git clone https://github.com/isaac-for-healthcare/i4h-workflows "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"; cd "$ROOT"
```

## What to Load

Before editing, load only the references that match the request:

- Always load `skills/i4h-workflow/references/repo-map.md`.
- Always load `skills/i4h-workflow-create/references/create-contract.md`.
- Load `skills/i4h-workflow-create/references/env-authoring-patterns.md` to choose the source env, robot owner, policy stack, and YAML pattern.
- Load `skills/i4h-workflow-create/references/hybrid-layout-rules.md` only for hybrid scene+robot envs, G1 footprint/table-height work, catalog USD scale, or support-surface layout.
- Load a recipe file only when the prompt matches it exactly enough to avoid re-deciding components.

Recipe routing:

| Prompt shape | Reference |
|---|---|
| Surgical tool sorting using G1 based on `scissor_pick_and_place` | `skills/i4h-workflow-create/references/g1-surgical-tool-sort.md` |

## Create Contract

Create the env shell first. A normal create task should produce exactly these surfaces unless the chosen existing pattern requires an explicit exception:

| Surface | Path |
|---|---|
| Env YAML | `workflows/agentic/config/environments/<env>.yaml` |
| Assets | `workflows/agentic/arena/arena/assets/<env>.py` |
| Task | `workflows/agentic/arena/arena/tasks/<env>.py` |
| Env class | `workflows/agentic/arena/arena/environments/<env>_environment.py` |
| Runtime | `workflows/agentic/arena/arena/runtimes/<env>.py` |

Keep all paths repo-root relative and keep the `workflows/agentic/` prefix. Do not create a new policy package, README, docs, or shared-module edits unless the user explicitly asks and the policy stack truly needs it.

## Workflow

1. Resolve components: env id, source env, scene/assets source, robot owner, policy stack, model/checkpoint, cameras, objects/destinations, success rule.
2. Inspect the selected source env YAML, env class, assets, task, runtime, and policy stack files before writing code.
3. Fork the closest working pattern. Preserve inline-scene vs registry-asset style; do not invent a new architecture.
4. Keep the first version minimal and runnable. Optional visual polish, extra props, new cameras, and layout changes happen through scene-edit after the env exists.
5. Run validation and fix source until the env builds and the rendered scene is visually sane.

For an exact recipe match, do not re-derive the architecture after loading the recipe. Read only the source files the recipe names, skip large source runtime files when the recipe says the runtime is a re-export, then write the five contract files immediately. Do not inspect third-party framework internals unless a validation error requires it.

Plan shape:

```text
Env id:
Source env / recipe:
Scene/assets source:
Robot owner:
Policy stack + model/checkpoint:
Objects/destinations:
Success rule:
Files to create:
Validation:
```

## Hard Rules

- Env YAML is the source of truth for robot, policy, cameras, task text, dataset mapping, and train defaults.
- Fork from the nearest existing implementation. For hybrid envs, robot integration comes from the robot owner and scene construction comes from the scene source.
- Sorting tasks need at least two object types, at least two destinations, and a success rule that fails swapped placements.
- Static/kinematic destination props may not expose `.data.root_pos_w`; success checks must fall back to `entity.get_world_poses()`.
- Functions referenced by `func=` inside config classes must be defined above those classes.
- G1 WBC envs need footprint clearance and matched ground/base-height values; load `hybrid-layout-rules.md` before choosing those numbers.
- Additional fixed cameras are normally scene-edit/bake work. If a create prompt explicitly requires a policy/dataset camera, load the scene-edit camera and bake references and wire every surface in one pass.

## Validation

Run these static checks:

```bash
python -m py_compile <changed-python-files>
workflows/agentic/policy/run.sh --list-envs
workflows/agentic/arena/run.sh --env <env> --dry-run
workflows/agentic/policy/run.sh --env <env> --dry-run
```

Then run the real build/visual gate from `skills/i4h-workflow-create/references/create-validation.md`. `--dry-run` is shallow: it does not prove the task/assets instantiate or the scene is visually usable. Do not report a new env as ready until the bridge reaches ready, key objects are valid, a viewport capture has been inspected, and the bridge is stopped with:

```bash
workflows/agentic/arena/stop.sh --env <env>
```

If Isaac Sim cannot launch on the host, report that as a blocker and include the static check results; do not present static-only as success.

## Hand Off to Scene Edit

After the env shell passes create validation, use [[i4h-workflow-scene-edit]] for:

- Adding, moving, resizing, or replacing objects/assets.
- Adding fixed room/overhead/wrist cameras.
- Live bridge edits.
- Baking live changes into source.
- Running `local-agent/validate-bake.sh`.

The scene-edit workflow owns object snippets, camera snippets, bridge endpoint details, and bake checklists so this create workflow stays small.

## Prerequisites

- Workflow setup has completed via [[i4h-workflow-setup]]; `.venv`, third-party checkouts, and Isaac Sim launch support are present.
- The source env and target env id are known, or the prompt provides enough information to choose them from existing patterns.
- Bridge validation runs on a GPU host that can launch Isaac Sim.

## Limitations

- Create one env per invocation.
- Fork existing patterns; do not invent a new policy stack or shared framework.
- Use scene-edit for post-create polish, optional props, camera baking, and source persistence of live edits.
- `--dry-run` is not a substitute for the bridge build and visual validation gate.

## Troubleshooting

- **Missing setup**: if `.venv`, third-party checkouts, or `run.sh` entrypoints are missing, run [[i4h-workflow-setup]] first.
- **Env not listed**: check that the YAML exists at `workflows/agentic/config/environments/<env>.yaml` and all files use the full `workflows/agentic/` repo-root prefix.
- **Build fails after static checks pass**: inspect the first Isaac/arena stack trace; common causes are bad cfg kwargs, a helper defined below a config class, or an observation pointing at a missing sensor.
- **Scene looks wrong**: fix source or use scene-edit to live-adjust and bake; do not report success from static checks alone.

## Final Response

Report the env id, source choices, files created, validation commands and results, capture paths, and any blocker. Do not commit changes unless the user explicitly asks.
