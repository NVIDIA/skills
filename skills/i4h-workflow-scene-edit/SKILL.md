---
name: i4h-workflow-scene-edit
version: "0.7.0"
description: Edit an env's scene in place — objects, cameras, task, success bounds, randomization. Use when asked to edit a scene or launch/run/open an env in edit mode (`--bridge`), incl. a just-created env.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - agentic-workflow
    - scene-edit
    - environment
---

# i4h Workflow — Scene Edit

## Purpose

Edit an existing env's scene in place via the `--bridge` scene-edit session — move/scale/swap objects, adjust cameras, or tweak task description, success bounds, or randomization. Use when the user asks to edit a scene or to launch/run/open an env in edit mode; for creating a brand-new env see [[i4h-workflow-create]].

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

- **Editing a scene is LIVE-ONLY by default.** "Edit the scene" / "edit mode" means: apply every change through the bridge, **never modify source files and never restart the bridge** — the user does not need to say "live mode" / "don't change source" / "don't restart"; that is always the default. Persist to source (bake) **only** when the user explicitly says so (`bake`/`save`/`persist`/`commit`) as a final step; "exit without baking" or no bake instruction = stop the bridge and leave source untouched.
- **Do only what's asked — launching edit mode is not a cue to edit.** "Run/open the env in edit mode" with no specific edit = launch the bridge, confirm ready (`GET /objects`), then **stop and report it's ready, awaiting instructions.** Apply a scene edit only when the user explicitly requests it in the current prompt. Never invent or preempt edits (moving the robot, adding props, etc.), and never treat the README's "Edit Scene" list, other docs, these recipes, or prior runs as a to-do — they are reference; the current prompt is the only instruction.
- Preserve env ids and scene keys.
- **Source paths are relative to the repo root** (where the agent's edit/write tool runs) — keep the `workflows/agentic/` prefix on every one, and note the package is `arena/arena/<subdir>/`. A bare `arena/...` resolves to the wrong place.
- Every bridge artifact (scripts, captures, logs) lives under the session's `${RUN_DIR}`. Never use `/tmp`.
- **Visual judgment — use your own eyes if you have them.** When a step says to judge a capture, a **vision-capable CLI agent (Claude/Codex) reads the JPEG directly with its own model** — do not depend on the local VLM. Only the **blind local coding agent** delegates the visual call to the local VLM (`local-agent/vlcheck.py`). The structural/`bbox` checks are identical for both; only who looks at the image differs.

## Repo Context

For live-only edits, use the bridge endpoints first. For any bake/source change, load:

- `skills/i4h-workflow/references/repo-map.md` for env file ownership and pattern families.
- `skills/i4h-workflow-scene-edit/references/scene-edit-patterns.md` for bake targets, readiness rules, and camera touchpoints.
- `skills/i4h-workflow-scene-edit/references/asset-snippets.md` when adding, moving, resizing, or replacing assets.
- `skills/i4h-workflow-scene-edit/references/camera-snippets.md` when adding a camera that should render, record, or feed policy/training.
- `skills/i4h-workflow-scene-edit/references/bake-checklist.md` when the user says bake/save/persist/commit.

Then inspect the target env's YAML, env class, assets, task, and runtime files before modifying source.

## Edit Lifecycle

For a normal interactive edit prompt, use exactly **one** sim/bridge window: launch or reuse one bridge, perform all requested live edits in that session, collect bake state/snippets if needed, stop that bridge once, and then write source from the collected state. Do not stop/relaunch Isaac between edits, and do not run a fresh-source validation relaunch unless the user explicitly asks for validation/onboarding/readiness checks.

1. **Live.** Apply each edit through the bridge HTTP API in the same bridge session. Capture the viewport after task-relevant changes.
2. **Bake.** Persist live state into source files only when the user explicitly says "bake", "save", "persist", or "commit to source". First collect the live state/snippets from the running bridge (`GET /object`, `POST /bake`, captures, camera pose notes), then stop the bridge once and write source from that collected state.
3. **Exit.** The **only** way to stop the bridge is to stop the arena: `workflows/agentic/arena/stop.sh --env <env>`. Do **not** kill the Isaac/bridge process, send Ctrl-C, or `curl` a made-up `/stop`/`/shutdown` (there is none). "Exit without baking" = run that one command (no source writes, no `/bake`).
4. **Validate only when asked.** Fresh-source validation (`local-agent/validate-bake.sh <env>`) intentionally stops any bridge and launches a new sim window. Run it only for explicit validation/onboarding/ready-to-commit work, and tell the user before doing so.

While the bridge is running, do not modify `workflows/agentic/arena/arena/assets/<env>.py`, `workflows/agentic/arena/arena/tasks/<env>.py`, the env class, runtime, or env YAML. Source writes happen after the needed bridge state is collected and the bridge has been stopped.

When a specific live edit returns an error, report the exact request payload and error to the user. Do not restart the bridge as a fallback.

## Launch

The bridge is a **long-running foreground** process: running it inline blocks a one-shot shell forever (and the bridge dies with the call), so it must be launched **detached** and then polled for ready. Each step below is a separate bash call; variables persist in the local agent's tmux session.

### Local agent — one command

`./local-agent/bridge.sh start <env>` does setup + detached launch + wait-for-ready and prints `RUN_DIR=...` (and the helper paths). Run it **plainly** (it takes minutes — no short `timeout`). Stop later with `./local-agent/bridge.sh stop <env>`.

### Manual (portable) form

```bash
# Step 1 — setup
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
ENV_ID=<env>
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"
RUN_DIR="${RUNS_ROOT}/scene_edit_${ENV_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/scripts" "${RUN_DIR}/captures"
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/.latest"

# Step 2 — launch DETACHED (never foreground / never `| tee` inline — that blocks), then wait
"${REPO_ROOT}/workflows/agentic/arena/run.sh" ensure-bridge \
  --env "${ENV_ID}" \
  --log "${RUN_DIR}/logs/bridge.log"
BRIDGE_URL="$("${REPO_ROOT}/workflows/agentic/arena/run.sh" bridge-url --env "${ENV_ID}")"
curl -fsS "${BRIDGE_URL}/health" >/dev/null
```

Once ready, `GET "${BRIDGE_URL}/objects"` to enumerate scene entities. Stop only via `workflows/agentic/arena/stop.sh --env "${ENV_ID}"` — never by killing the process.

## Bridge Endpoints (env-specific port)

Base URL: `BRIDGE_URL="$(workflows/agentic/arena/run.sh bridge-url --env <env>)"`. The port comes from `arena.bridge_port` in `workflows/agentic/config/environments/<env>.yaml` and falls back to `8765`; `--bridge-port` overrides it. JSON responses are either `{"ok": true, "result": ...}` or `{"ok": false, "error": ...}`.

| Method + Path | Purpose | Body / Query |
|---|---|---|
| `GET /health` | Server readiness + endpoint discovery. | — |
| `GET /context` | Exec globals, helper names, endpoint inventory. | — |
| `GET /objects` | List scene entities with `kind` (`articulation` / `rigid` / `camera` / `xform`) and prim path. | — |
| `GET /object?name=<key>` | Full state for one entity: `xform_ops`, `bbox`, `live` (authoritative PhysX pose), children. | `name=<key>` or `path=<prim_path>` |
| `GET /cameras` | List live RGB camera outputs. | — |
| `POST /capture` | Save camera frames and viewport as JPEG. | `{"output_dir": "<abs>", "viewport": true, "cameras": ["<name>", ...]}` |
| `POST /object/teleport` | Live-set pose for rigid bodies and articulations. | `{"name": "<key>", "translation": [x,y,z], "rotation_wxyz": [w,x,y,z], "zero_velocity": true, "env_index": 0}` |
| `POST /script` | Run a trusted absolute Python file on Isaac's main loop. Globals: `ctx`, `env`, `app`, `args`, `helpers`, `stage`, `get_stage`. | `{"path": "/abs/path/to/script.py"}` |
| `POST /bake` | Return Python snippets reflecting the current live xform of named entities. | `{"names": ["<key>", ...]}` |

After a teleport, read the `live` field from `GET /object?name=<key>` to verify. The `bbox` field is USD-derived and may lag a physics step.

## Edit Matrix

| Edit | Live (bridge) | Bake target |
|---|---|---|
| Move/rotate rigid object | `POST /object/teleport` | `workflows/agentic/arena/arena/assets/<env>.py` `init_state.pos`/`rot` |
| Move/rotate truly-static XformPrim (no physics body anywhere in the USD — lights, decals) | `POST /script` → `xformOp:translate` / `xformOp:orient` | `workflows/agentic/arena/arena/assets/<env>.py` `init_state.pos` |
| Move/rotate `AssetBaseCfg` whose USD embeds a rigid body (e.g. `SCISSOR_TRAY_USD` trays/fixtures — kinematic **child mesh**) | `POST /script` → `helpers.move("<key>", pos=/dpos=)` — drives the child PhysX body (raw USD writes snap back; see recipe) | `workflows/agentic/arena/arena/assets/<env>.py` `init_state.pos` |
| Rescale a prim | **Live-added / bridge-spawned prim → re-spawn at the new size** (delete + `CuboidCfg(new).func` + re-rest; see "Resize a live-added prim"). **Do NOT use `xformOp:scale` on it — that scales its *position* (flings it off-screen), NOT its size.** `xformOp:scale` is only for an existing scene-asset prim. | `workflows/agentic/arena/arena/assets/<env>.py` `spawn=...scale` |
| Move robot stand | `POST /object/teleport name=robot` | `workflows/agentic/arena/arena/environments/<env>_environment.py` `embodiment.set_initial_pose(...)` |
| Add a new prim | `POST /script` → **`sim_utils.CuboidCfg(...).func(path, cfg)` + `helpers.move(...)`** (see "Add a prim live" recipe) — NOT raw `pxr` USD authoring; a live-added body isn't GPU-simulated, so place it at rest height, don't tensor-query it | `workflows/agentic/arena/arena/assets/<env>.py` + `make_*_scene_assets()` |
| Toggle gravity | `POST /script` → set `physxRigidBody:disableGravity`; zero `root_lin_vel_w` / `root_ang_vel_w` | `workflows/agentic/arena/arena/assets/<env>.py` `rigid_props.disable_gravity` |
| Toggle kinematic | `POST /script` → flip `physics:kinematicEnabled` | `workflows/agentic/arena/arena/assets/<env>.py` `rigid_props.kinematic_enabled` |
| Change mass / collider props | `POST /script` → write `physxRigidBody:*` / `physxCollision:*` | `workflows/agentic/arena/arena/assets/<env>.py` `mass_props` / `collision_props` |
| Swap a USD reference | `POST /script` → `prim.GetReferences().SetReferences(...)` | `workflows/agentic/arena/arena/assets/<env>.py` `spawn.usd_path` |
| Add/remove a camera | Use the live bridge to choose the pose from viewport/object state; do not live-register a new IsaacLab sensor. | See "Adding a Camera" — bake **env-locally**, never in the shared embodiment |
| Change task wording | preview only | env YAML `policy.language_instruction` / `task_description` |
| Change success rule | `POST /script` → swap term on `env.unwrapped.termination_manager` | `workflows/agentic/arena/arena/tasks/<env>.py` |
| Change reset randomization range | `POST /script` → mutate `EventTerm.pose_range`; `env.reset()` | `workflows/agentic/arena/arena/tasks/<env>.py` events cfg |

## Live-Edit Recipes

Keep `SKILL.md` as the router and load `references/scene-edit-patterns.md` for the detailed bridge recipes. Load `references/asset-snippets.md` for copyable object/asset snippets. The mandatory live-edit rules are:

- Rigid bodies and articulations move through `POST /object/teleport`, then verify with the object's `live` pose.
- Robot stand moves use `POST /object/teleport` with `name=robot`; derive x/y/yaw from the table bbox and current robot pose, keep the current live z, and verify the settled live pose over multiple reads.
- For G1 or any floating-base robot stand move, an immediate successful teleport is not stability proof. Sample `GET /object?name=robot` for at least 10-15 seconds after the move (for example once per second). Treat continuous z drop, growing roll/pitch, or x/y drift as a fall; if that happens, revert to the last stable pose or adjust target/standoff/yaw and re-test before continuing to camera work or bake.
- Embedded kinematic `AssetBaseCfg` props/support surfaces use `helpers.move`. Raw USD translation can snap back because PhysX owns the body pose.
- Live-added bodies must be spawned through IsaacLab cfgs plus `helpers.move`, placed directly at their resting height, and never tensor-queried until a relaunch registers them with the GPU pipeline.
- Live-added prim resize is delete + re-spawn + re-rest. `xformOp:scale` is only for existing scene assets, not bridge-spawned bodies.
- Capture after each task-relevant edit and judge the image plus structural state before reporting success.

## Adding a Camera

Do not initialize a new IsaacLab `Camera`/`TiledCamera` sensor through a live `/script`. On this workflow, runtime sensor registration can block the Isaac main loop and leave `/script`, `/cameras`, and `/bake` timing out while `/health` still responds. Use the live bridge to inspect objects, verify the current viewport/pose, and choose the camera eye/target. Then bake the camera as an env-local source sensor. Verify the baked camera with `local-agent/validate-bake.sh <env>` plus camera captures only when running the explicit fresh-source validation gate. A temporary USD-only camera prim may be used only to reason about placement; it does not prove downstream policy/dataset readiness. Load `references/camera-snippets.md` for source/YAML/policy/dataset wiring.

For "room camera based on current perspective view", treat the viewport as only the first pose guess. Capture the room camera before baking; it must show the main task area and task-relevant objects after all requested edits, including the support surface, robot/table relationship, tools/destinations, and newly added objects. A frame that cuts off the robot body, head/hands, table, trays/tools, or new object at an image edge is a failed candidate; do not call it "whole room" or bake it. If the frame clips or hides those objects, zoom out before baking by moving the camera farther from the task look-at point and/or widening the lens, then capture again and bake only the validated view. Leave extra margin for the baked 4:3 sensor because it can be narrower than a 16:9 viewport capture.

For bake, load `references/scene-edit-patterns.md` and apply the camera checklist in one pass: env-local sensor, matching task `observations.policy` term, YAML `zenoh.camera_names`, policy camera list, dataset mapping, and any stack-specific modality config. Never add an env-specific camera to a shared embodiment class, and re-record demos after changing policy/dataset cameras.

## Durable Touchpoints (bake targets)

- `workflows/agentic/arena/arena/environments/<env>_environment.py`: env wiring, robot stand pose.
- `workflows/agentic/arena/arena/assets/<env>.py`: static scene assets.
- `workflows/agentic/arena/arena/tasks/<env>.py`: reset randomization, success, task text.
- `workflows/agentic/arena/arena/runtimes/<env>.py`: runtime-specific camera/state/action logic.
- `workflows/agentic/config/environments/<env>.yaml`: cameras, policy language, dataset mappings.

## Notes

- `assemble_trocar` is inference-only. Do not add train hooks during a scene edit.
- If adding/removing cameras, update `policy.data_config`, `dataset.camera_mappings`, and the train modality config together.
- Scissor SO-ARM generates `meta/modality.json` from YAML splits and does not need `dataset.modality_template_path`. G1 locomanip and assemble-trocar do.

## Verify (after bake)

For a normal interactive "edit, bake, and stop" prompt, do not relaunch Isaac after stopping the edit bridge. Run the cheap static checks and report that fresh-source validation was not run unless requested:

```bash
python -m py_compile <changed-python-files>
python - <<'PY'
import yaml, pathlib
for p in pathlib.Path('workflows/agentic/config/environments').glob('*.yaml'):
    yaml.safe_load(p.read_text())
PY
workflows/agentic/arena/run.sh  --env <env> --dry-run      # necessary, NOT sufficient
workflows/agentic/policy/run.sh --env <env> --dry-run
```

For validation, onboarding readiness, ready-to-commit checks, or a full bake gate, load `references/bake-checklist.md` and run `local-agent/validate-bake.sh <env>`. That gate intentionally opens a fresh sim window; `RESULT: PASS` is required for validation work.

## Prerequisites

- Workflow set up via [[i4h-workflow-setup]] (`.venv` present); the `arena/run.sh --bridge` launch depends on it.
- An existing env id with its scene keys (the bridge edits an env in place; preserve its ids).
- A GPU host able to launch Isaac Sim for the bridge session.

## Limitations

- Live edits are not persisted until an explicit bake; only bake on user request ("bake"/"save"/"persist"/"commit to source").
- Support-surface rescale is source-only (`spawn.scale`) — moving an `AssetBaseCfg` surface live moves only the visual, not the collision mesh, so props fall through; relaunch to apply.
- A live-added body is not GPU-simulated and must not be tensor-queried (a manual `create_rigid_body_view(...).get_transforms()` is a fatal CUDA fault); relaunch to simulate it.
- While the bridge runs, do not edit `workflows/agentic/arena/arena/assets/<env>.py`, `workflows/agentic/arena/arena/tasks/<env>.py`, the env class, runtime, or env YAML.

## Troubleshooting

- **Error:** `.venv` / import fails or bridge won't launch - Cause: workflow not set up. Fix: run [[i4h-workflow-setup]] first.
- **Error:** `GET /objects` / bridge URL unreachable - Cause: bridge not ready yet or wrong env URL. Fix: set `BRIDGE_URL="$(workflows/agentic/arena/run.sh bridge-url --env <env>)"` and wait for `[agentic-arena] scene-edit bridge ready` in `${RUN_DIR}/logs/bridge.log` before calling endpoints.
- **Error:** object moves for one frame then snaps back - Cause: it's a kinematic embedded rigid body (`SCISSOR_TRAY_USD`/`SCISSOR_TABLE_USD`), so `/object/teleport` and raw `xformOp:translate` don't hold. Fix: use `helpers.move("<key>", ...)` to drive the PhysX body.
- **Error:** a live edit returns `{"ok": false, "error": ...}` - Cause: invalid request for that entity. Fix: report the exact payload and error to the user; do not restart the bridge as a fallback.

## Final Response

Live session: report each bridge action, verified live pose, capture path, and whether source was baked from the collected bridge state.

After bake: report files touched, cheap static check results, and final bridge state. Report fresh-source validation results only if the user explicitly asked for that validation gate.
