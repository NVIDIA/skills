# Scene Edit Context

Load this with `skills/i4h-workflow/references/repo-map.md` when editing a scene live or baking bridge changes to source.

## Context to Inspect

For the target env, inspect:

- `workflows/agentic/config/environments/<env>.yaml`
- `workflows/agentic/arena/arena/environments/<env>_environment.py`
- `workflows/agentic/arena/arena/assets/<env>.py` or the shared assets file from the repo map
- `workflows/agentic/arena/arena/tasks/<env>.py` or the shared task file from the repo map
- `workflows/agentic/arena/arena/runtimes/<env>.py` when cameras, state/action IO, or policy publishing changes

Then use the bridge to inspect live truth: `/objects`, `/object?name=...`, `/cameras`, `/capture`, and `/context`.
Resolve the base URL for the target env with `BRIDGE_URL="$(workflows/agentic/arena/run.sh bridge-url --env <env>)"`.

## Readiness Rules

- **Launch detached, never foreground.** The bridge is a long-running foreground
  process — running `arena/run.sh --env <env> --bridge` (or piping it to `tee`)
  directly blocks your shell, floods stdout with Isaac logs, and the bridge dies
  with the call. Use `workflows/agentic/arena/run.sh ensure-bridge --env <env>
  --log <RUN_DIR>/logs/bridge.log`, which launches a detached `--keep-open`
  bridge and waits for readiness. The local agent helper
  `local-agent/bridge.sh start <env>` is also acceptable when available.
- Wait for `[agentic-arena] scene-edit bridge ready` in the bridge log (or a healthy
  `GET /health`) before issuing edits — do not `tee` the long log into the shell.
- Stop Arena to stop the bridge: `workflows/agentic/arena/stop.sh --env <env>`
  (or `bridge.sh stop <env>`), not `run.sh`. There is no `/stop` endpoint and you
  must not kill the process or send Ctrl-C.

## Live Edit Rules

- Keep every script, capture, and log under the session `${RUN_DIR}`. Do not use `/tmp`.
- Verify a movement from `GET /object?name=<key>` live pose, not only `bbox`.
- Capture the viewport after task-relevant edits; camera frames verify policy view, viewport verifies whole-scene layout.
- If an endpoint returns `ok: false`, report the payload and error. Do not restart the bridge as a fallback.
- Do not edit source files while the bridge is running unless the user explicitly asked to bake/persist source changes.

## Bridge Live-Edit API (globals + helpers)

Run `GET /context` once the bridge is ready — it returns the **live, authoritative**
globals + helper names + endpoint list for this build (never drifts from the code).

A `POST /script` runs a Python **file** (`{"path": "<abs path>"}`, written under
`${RUN_DIR}/scripts/`) on Isaac's main loop; set a `result` variable to return JSON.
Globals in the script: `env`, `app`, `args`, `ctx`, `stage`, `get_stage()`, and
**`helpers`**. Useful `helpers` methods:

- `helpers.env_live_path("<Name>")` → prim path for a NEW live object (`/World/envs/env_0/LiveEdit/<Name>`).
- `helpers.world_bbox("<key|path>")` → `{"min":[x,y,z],"max":...,"size":...}`; `helpers.surface_top_z("<key>")` → tabletop world z.
- `helpers.scene_prim_path` / `resolve_path` / `get_prim` / `env_origin`.
- `helpers.move("<key|path>", pos=/dpos=/rot_wxyz=, zero_velocity=True)` → drives a **registered** rigid body via PhysX (pose holds); for a **live-added** or non-physics prim it moves USD-only and returns `{"mode":"usd","warning":...}` (`/context` may not list `move`, but it exists).

### Verified recipe — live-add a prim resting on the table

A live-added body is **NOT GPU-simulated** (it wasn't present at bridge init), so it
won't fall/settle — **place it at its resting height explicitly**, and never
tensor-query it (that's a fatal CUDA fault). Write the script under
`${RUN_DIR}/scripts/`, then `POST /script {"path": "<abs>"}`:

```python
import isaaclab.sim as sim_utils
bb = helpers.world_bbox("table")                      # support surface, world coords
top = bb["max"][2]                                    # tabletop z
cx, cy = (bb["min"][0]+bb["max"][0])/2, (bb["min"][1]+bb["max"][1])/2
CUBE_SIZE_M = 5 / 100
CUBE_MASS_KG = 5 / 100
TABLE_CLEARANCE_X_M = 18 / 100
RED = (9 / 10, 5 / 100, 5 / 100)
path = helpers.env_live_path("RedCube")
cfg = sim_utils.CuboidCfg(
    size=(CUBE_SIZE_M, CUBE_SIZE_M, CUBE_SIZE_M),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=RED, roughness=1 / 2, metallic=0),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=5 / 1000, rest_offset=1 / 1000),
    mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS_KG),
)
cfg.func(path, cfg)                                   # spawn USD prim + physics schemas
helpers.move(path, pos=(cx + TABLE_CLEARANCE_X_M, cy, top + CUBE_SIZE_M / 2), zero_velocity=True)  # rest on table, clear of props
result = {"path": path, "tabletop_z": top}
```

Verify: `GET /object?path=/World/envs/env_0/LiveEdit/RedCube` — its bbox `min.z`
should ≈ the table's bbox `max.z` (resting, not sunk); then `POST /capture` and judge it
(read it yourself if vision-capable — Claude/Codex; only the blind local agent runs
`local-agent/vlcheck.py`). This is **live only** — the prim
vanishes on relaunch and becomes permanent only via an explicit bake (a `CuboidCfg`
field in the assets file + its name in `make_*_scene_assets()`).

## Bake Targets

| Live change | Source file |
|---|---|
| Asset move/rotation/scale | `workflows/agentic/arena/arena/assets/<env>.py` |
| Robot stand pose | `workflows/agentic/arena/arena/environments/<env>_environment.py` |
| Reset randomization or success rule | `workflows/agentic/arena/arena/tasks/<env>.py` |
| Camera observation and recording | assets file plus task `policy` observations plus env YAML camera/dataset fields |
| Policy task text | env YAML `policy.task_description` or `policy.language_instruction` |
| Runtime camera/state/action publishing | `workflows/agentic/arena/arena/runtimes/<env>.py` |

## Camera Bake Checklist

A camera intended for dataset/policy use needs all of these checked:

- It exists as a rendered sensor in the scene assets.
- It is added to task `observations.policy`; recorder serializes the `policy` group.
- YAML `zenoh.camera_names` includes the camera label.
- GR00T YAML camera fields such as `pov_cam_names_sim` use the list form when present.
- YAML `dataset.camera_mappings` maps the simulation camera to the LeRobot video key.
- Stack-specific modality files/configs match the resulting video keys.

### Env-local camera sensor pattern

Register the camera as a resolvable sensor, not just as a camera prim. The most
reliable source pattern is a field on a small `@configclass` returned by the task's
`get_scene_cfg()`, plus a matching `observations.policy` term in `modify_env_cfg`.
The field name, obs-term name, and `dataset.camera_mappings` key should match; the
`_rgb` suffix belongs only on `policy.pov_cam_names_sim.obs_key`.

Never add an env-specific camera to the shared G1 embodiment class. If the camera
must ride the embodiment, create a per-env camera config in that env's `get_env`
and assign it to that embodiment instance only.

Use the list form for policy camera YAML:

```yaml
policy:
  pov_cam_names_sim:
    - {obs_key: robot_head_cam_rgb, video_key: ego_view}
    - {obs_key: robot_room_cam_rgb, video_key: room}
```

Do not use the singular `pov_cam_name_sim` key. Single-camera and dual-camera
checkpoints are not interchangeable.

## Polishing a scene (distractors, fixed cameras)

These are scene-edit additions to an existing env, not base-create requirements.
Bake into source only after validating the live view in the bridge.

### Adding a distractor prop

A distractor is just another scene asset in the env's `make_*_scene_assets()`
builder (or scene cfg). Use a `RigidObjectCfg` for a physics prop (a primitive via
`sim_utils.CuboidCfg`, or a USD) with the same `mass_props`/`collision_props`
pattern the other tabletop props use, a `PreviewSurfaceCfg` color that stands out,
and an initial pose a few mm above the tabletop. Pick its position so it does not
overlap the task props or block the manipulation zone. Include its scene key in
the assets builder, then verify after relaunch:

```bash
curl -fsS "${BRIDGE_URL}/object?name=<key>" -o "${RUN_DIR}/object.json"
python -m json.tool "${RUN_DIR}/object.json"
```

### Baking a fixed scene camera

Validate the desired view live first, then bake it. The asset is a `TiledCameraCfg`
(`data_types=["rgb"]`, the resolution and `update_period` your stack expects) with
a `PinholeCameraCfg` spawn and the offset pose you confirmed in the bridge; add its
scene key to the assets builder. To make it a **policy/dataset** camera, wire every
surface in one pass (see the camera checklist above):

- Task `observations.policy`: add the obs term with `SceneEntityCfg("<key>")`. The
  observation term name is the HDF5 key — do **not** give it a `_rgb` suffix.
- YAML: add the camera to `zenoh.camera_names`; add `{obs_key: <key>_rgb, video_key: <video>}`
  to `policy.pov_cam_names_sim`; add `<key>: observation.images.<video>` to `dataset.camera_mappings`.
- Switch to the stack's multi-camera modality/config files when adding a second policy camera.
- Re-record data — old HDF5 without the camera can't be back-filled by YAML edits.

Verify after relaunch:

```bash
curl -fsS "${BRIDGE_URL}/cameras" -o "${RUN_DIR}/cameras.json"
python -m json.tool "${RUN_DIR}/cameras.json"
curl -fsS -X POST --json '{"output_dir":"workflows/agentic/runs/<run>/captures","viewport":true}' \
  "${BRIDGE_URL}/capture" -o "${RUN_DIR}/capture.json"
python -m json.tool "${RUN_DIR}/capture.json"
```

## Anti-Patterns

- Raw USD xform writes for embedded kinematic rigid bodies that snap back; use bridge helpers when available.
- Live-rescaling support surfaces and assuming collision changed; source `spawn.scale` plus relaunch is required.
- Tensor-querying live-added rigid bodies; they are not GPU-simulated until relaunch.
- Baking camera visuals but forgetting task observations or dataset mappings.
- Trusting a single viewport frame for G1 stability; read robot live pose over a few steps.
