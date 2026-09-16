---
name: i4h-workflow-scene-edit
description: Edit an existing workflow Scene or task contract. Use for assets, layout, cameras, randomization, task text, or success rules; do not use to create a new workflow.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  tags:
    - isaac-for-healthcare
    - i4h
    - isaac-sim
    - scene-authoring
---

# Edit an i4h Workflow Scene

## Purpose

Iterate on an existing Scene in one live simulator session, and persist the confirmed state only when explicitly asked to bake, save, or persist.

## Instructions

1. Run the checkout resolver and inspect target ownership.
2. Start or reuse one bridge-backed live-authoring session by default and capture a visible baseline.
3. Read the minimal upstream guidance and apply each requested edit as its own observable live-stage transaction without changing source.
4. On an explicit bake/save/persist instruction, export the accumulated live state, inspect its resolved authoring facts, and have the coding agent edit the smallest owning sources.
5. Keep the session running for further prompts until the user explicitly says stop or exit.
6. Run static, persisted-visible, and affected dynamic validation when baking.

## Resolve and inspect

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
./run.sh list
./run.sh show <workflow>
```

Treat the resolver above as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `workflows/i4h_workflows`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

Read `DESIGN.md`, the target workflow, Scene Python, scene manifest, embodiment manifest, and relevant task manifests. Use `$ROOT/skills/i4h-workflow/references/repo-map.md` to resolve ownership.

For a G1 face, walk, reach, or collision-sensitive success contract, read `references/g1-reach-and-contact.md`. Reuse its Tasks and Scene-owned footprint interface; do not generate workflow-specific locomotion helpers or hardcode object extents in the Workflow.

An open/run request without an edit means `./run.sh <workflow> --idle`; do not invent source changes. An edit-scene request always means start or reuse `./run.sh <workflow> --live` unless the user explicitly requests offline/no-live operation. Live authoring is the default and does not require confirmation. “Save,” “persist,” or “bake” means serialize the accumulated live-stage edits into their owning sources; without one of those words, leave source untouched. “Stop” or “exit” closes the session; if persistence was not requested, discard the live-only edits. Do not stop merely because one edit prompt completed.

Before adding or resizing an asset that already appears in a maintained scene, read `references/existing-scene-assets.md`. Reuse its known USD identity, authored scale, support height, physics role, and embodiment convention instead of rediscovering them from the raw USD. Treat owning Python as source of truth and use visual-language inspection only for bounded scene-specific refinement after the known baseline is visible.

## Establish a visible baseline

Open the existing Scene as one persistent live-authoring session. Use the bridge only when `I4H_LOCAL_AGENT=1`: Local Agent commands share one serialized shell, so a foreground `--live` command would block every later edit transaction.

```bash
I4H_RUN_DIR="$(pwd)/runs/<workflow>/$(date +%Y%m%d_%H%M%S)"
if [ "${I4H_LOCAL_AGENT:-0}" = 1 ] && [ -x ./local-agent/bridge.sh ]; then
  ./local-agent/bridge.sh start <workflow> "$I4H_RUN_DIR"
else
  ./run.sh <workflow> --live --run-dir "$I4H_RUN_DIR"
fi
```

The fallback `./run.sh <workflow> --live` must run through the host agent's persistent/yieldable foreground-session mechanism. Never launch that fallback as an ordinary blocking shell call and wait for it to exit before editing.

`--live` resolves the workflow's declared `idle` mode, enables `isaacsim.code_editor.python_server` on port 8226, and keeps the simulator open until explicitly stopped. Wait for port 8226, then use the pinned upstream `isaac-sim-remote` client to inspect and modify the running stage. Keep every ordinary edit only in that live stage; do not change owning source yet. Accumulate later edit prompts in the same session. Only “bake,” “save,” or “persist” authorizes writing the confirmed live values into source. After baking, restart through `run.sh`, verify the persisted result matches the live stage, and stop when requested. An offline source edit followed by a reopen is not live authoring.

## Preserve the live interpreter experience

Treat a compound prompt as an ordered stream of edits, not as one batch script. Run one bridge transaction for one user-visible operation, wait for its viewport update, inspect its result, and only then apply the next operation. For example, adding a table, two tools, two trays, and a robot is six live transactions. Never hide all requested edits inside one remote Python file or patch owning source while the user is waiting for the stage to change.

Use the one-operation helper from the workflow root for common edits:

```bash
arena/.venv/bin/python scripts/live_scene_edit.py add-known-asset \
  --asset surgical_table \
  --prim-path /World/envs/env_0/Table \
  --position 0,0,0

arena/.venv/bin/python scripts/live_scene_edit.py add-cube \
  --prim-path /World/envs/env_0/RedCube \
  --position 0,0,0.3 \
  --size 0.1 \
  --color 1,0,0

arena/.venv/bin/python scripts/live_scene_edit.py scale-by \
  --prim-path /World/envs/env_0/RedCube \
  --factor 2

arena/.venv/bin/python scripts/live_scene_edit.py set-transform \
  --prim-path /World/envs/env_0/Robot \
  --position=-4.64,0,0.8 \
  --rotation 0,0,0

arena/.venv/bin/python scripts/live_scene_edit.py set-view \
  --eye 2.6,-7,3.4 \
  --target=-1.8,0,0.75

arena/.venv/bin/python scripts/live_scene_edit.py camera-from-view \
  --prim-path /World/envs/env_0/RoomCamera

arena/.venv/bin/python scripts/live_scene_edit.py capture-camera \
  --prim-path /World/envs/env_0/RoomCamera \
  --output-path "$I4H_RUN_DIR/room-camera.png"
```

When a comma-separated vector begins with a negative number, bind it with `=` (for example, `--position=-0.5,0.5,0.1`) so the argument parser does not treat the value as another option.

<!-- markdownlint-disable-next-line MD013 -->
The helper intentionally accepts one operation per invocation, selects the affected prim, advances visible render updates, and prints the resulting world bounds. Prefer `add-known-asset` for catalogued Healthcare assets: it reuses canonical USD, scale, orientation, physics metadata, attached-camera metadata, embodiment metadata, and expected metric bounds, then rejects a result whose size differs by more than 20%. The `g1` preset must create the standard `head` camera below the live robot preview; treat a missing camera prim as a failed robot edit, activate it, and verify its view before continuing. If no executable preset exists, warm-start from `references/existing-scene-assets.md` and its owning source before using generic `add-usd`. Both asset-add commands place the reference below a transform wrapper so a referenced asset's authored root transform cannot discard the requested live position, rotation, or scale. `add-known-asset`, `add-usd`, `add-cube`, and `camera-from-view` tag their prims for deterministic export; pass `--name` or `--alias` when the source/manifest name cannot be derived generically from the prim path. Inspect bounds before continuing. Use `capture-camera` for fast visible camera checks: it activates the requested camera, schedules a synchronous `FileCapture`, advances the renderer, and rejects an absent, empty, or stale output. Do not use the upstream asynchronous viewport screenshot helper in the persistent bridge session. Use `activate-camera` when capture is unnecessary and `inspect` for one-prim verification. Use raw `isaacsim_send.py` only for an operation the helper does not support, and still send one observable edit per call; record any untagged prim explicitly when baking.

Infer ordinary support relationships from the requested workspace and measured bounds. A robot or object intended for a table, cart, tray, pad, or floor must have its lower support bound aligned with that surface and its footprint plausibly contained by it; do not accept a floor-mounted, floating, or visibly interpenetrating placement merely because every named asset is present. Include those support relationships in the bounded visual rubric before baking.

Send a short progress update while the scene visibly changes. Do not spend extended time designing the eventual source representation before the first requested live edit. Inspect ownership and prepare baking after the live result exists.

## Session lifecycle

- On the first “edit scene” prompt, use `local-agent/bridge.sh` only when `I4H_LOCAL_AGENT=1`; otherwise launch `./run.sh <workflow> --live` through a persistent/yieldable host session. Wait for port 8226.
- On every later scene prompt, detect and reuse the open bridge session; do not reset or relaunch the Scene unless the requested change requires it.
- Apply each add/move/rotate/scale/material/camera operation separately to the same live stage, select the affected prim, advance the viewport, and verify it before continuing.
- After adding a robot, verify every camera declared by its authoring preset. For G1, require the live `Robot/Asset/head_link/RobotHeadCam` preview and bake with a registered G1 embodiment whose `robot_head_cam` sensor is exposed through the `head` alias.
- Return control to the user after each prompt while leaving the simulator and bridge running.
- Bake only on explicit authorization. Baking does not imply stop unless the user also says stop/exit.
- On stop/exit without bake, close the session and leave source unchanged. A Local Agent session closes with `./local-agent/bridge.sh stop <workflow>`.
- Use offline/source-first editing only when the user explicitly disables live mode or the bridge cannot operate. Report a bridge blocker before using that fallback; never silently substitute it.

## Use upstream Isaac Sim skills

Read `references/isaacsim-skill-routing.md`, then load only the upstream skills required by the request. State the selection before editing. Use current upstream semantics for generic physics, cameras, sensors, USD, rendering, and spatial reasoning; integrate them through the closest current i4h Scene pattern.

## Iterate live, then let the coding agent bake

Apply requested asset, layout, physics, camera, and transform changes through port 8226 first. For a compound prompt, preserve its order and inspect the live stage after each individual operation. Do not preemptively patch files merely because their eventual owner is known.

When explicitly asked to bake/save/persist, export the confirmed live values and resolve the reusable catalog facts without launching another simulator:

```bash
I4H_RUN_DIR="runs/<workflow>/<YYYYMMDD_HHMMSS>"
mkdir -p "$I4H_RUN_DIR"

arena/.venv/bin/python scripts/live_scene_edit.py export-scene \
  --workflow <workflow> \
  --root-path /World/envs/env_0 \
  --output-path "$I4H_RUN_DIR/live_scene.json"

arena/.venv/bin/python scripts/authoring_info.py snapshot \
  <workflow> "$I4H_RUN_DIR/live_scene.json"
```

`export-scene` records every helper-managed asset, primitive, robot, and camera with its confirmed transform and camera optics. Keep that snapshot in the run directory as authoring evidence. Pass a previous run snapshot through `--baseline` only when a later live export needs to merge it: existing prims are re-read from the current stage, newly tagged prims are added, and removed prims are omitted. `authoring_info.py` is read-only; it validates the snapshot and returns code-ready catalog metadata and derived manifest capabilities immediately. It never generates or edits workflow code.

The coding agent then patches the existing asset, Scene, and manifest templates using the closest maintained source pattern. Commit only those owning sources; do not commit the exported authoring snapshot or treat it as a second Scene contract. Never copy a reusable USD path, canonical scale, mass, embodiment registry name, action contract, attached camera, or camera alias from memory: query `authoring_info.py asset <preset>` or the complete snapshot report, then use the catalog from owning source. Scene-specific names, placement, camera optics, and explicit overrides come from the snapshot.

Write only to the owning layer:

- Bake assets, layout, physics, cameras, randomization, view aliases, actuation mapping, or reset hooks into Scene/asset/envcfg source.
- Bake cross-boundary camera/object names, control rate, cap, or mode overrides into the scene manifest.
- Bake mode composition and goal semantics into the workflow.
- Bake reusable behavior into a Task.
- Bake cross-process robot labels, calibration, or teleop devices into the embodiment manifest.
- Bake a policy prompt, camera, observation, model, or training contract into the remote-task manifest.

Preserve the quaternion convention at each concrete API boundary. Do not add compatibility conversions or duplicate catalog facts across Python and YAML.

For a camera based on the current perspective, treat the viewport pose as an initial estimate. Compute and validate a stable live look-at from task-relevant bounds. On bake, add the confirmed env-local camera through the closest Scene pattern, declare it in the manifest, and verify every recording/policy consumer that should receive it.

## Validate

```bash
./run.sh show <workflow> --mode <affected-mode>
./run.sh lint <workflow> --mode <affected-mode>
./run.sh lint --all
uvx ruff check --config pyproject.toml <changed-python-files...>
```

Run focused tests through each affected component's uv project. After the coding agent's static validation, reopen once with `./run.sh <workflow> --live`, compare every requested visual change and declared camera against the exported snapshot, then stop when requested. Do not add extra restarts between export, source editing, and this persisted-visible check. Run affected dynamic modes when physics, reset, actuation, policy observations, task behavior, or success changed. Idle is insufficient for those changes.

When success excludes collision, dynamic validation must include a forced-contact negative case and a fresh-reset recovery case from `references/g1-reach-and-contact.md`. Do not call the success rule validated if its configured contact signal has only ever returned false.

Stop leftovers with `./stop.sh all`.

## Troubleshooting

Fix manifest/workflow lint before launch. If the rendered result differs, compare the baseline, authored prims, bounds, cameras, and owning source before retrying.

## Prerequisites

Require a supported existing workflow, complete simulator setup, and the relevant upstream Isaac Sim skills for generic scene semantics.

## Limitations

Use `i4h-workflow-create` for a new workflow. Live idle validates stationary layout but not physics, actuation, policy observations, or success. A live session is process-local; export before stopping because unexported edits are lost if it dies. The live helper covers common assets, raw USDs, cubes, transforms, cameras, inspection, capture, and export; the coding agent handles source authoring and any semantics outside those utilities.

## Examples

- `Add a red cube, move G1, add a room camera, bake all changes, and stop.` → keep one bridge-backed simulator session open, apply the three edits live in order, export and inspect one snapshot at “bake,” patch and statically validate the owning source, reopen once for persisted-visible validation, and stop.

## Completion gate

Report upstream skills used, live session/bridge status, edits applied live, whether persistence was authorized, owning sources changed only during bake, static tests, live and persisted visible observations, camera checks, dynamic rollout results including collision-negative evidence when applicable, clean stop status, and any unresolved mismatch.
