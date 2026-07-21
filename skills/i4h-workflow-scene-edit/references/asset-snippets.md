# Asset Edit Snippets

Load this when adding, moving, resizing, or replacing scene assets through the bridge. Keep scripts under the session `${RUN_DIR}/scripts`; keep captures under `${RUN_DIR}/captures`.

## Live-Add a Cube on a Table

Use IsaacLab spawners, not raw USD authoring. Place live-added bodies directly at rest height; they are not GPU-simulated until relaunch.

```python
import isaaclab.sim as sim_utils

red = tuple(channel / 255 for channel in (230, 5, 5))
table = helpers.world_bbox("table")
top = table["max"][2]
cx = (table["min"][0] + table["max"][0]) / 2
cy = (table["min"][1] + table["max"][1]) / 2
size = 0.10
path = helpers.env_live_path("RedCube")

cfg = sim_utils.CuboidCfg(
    size=(size, size, size),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=red, roughness=0.45),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.001),
    mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
)
cfg.func(path, cfg)
move = helpers.move(path, pos=(cx, cy, top + size / 2), zero_velocity=True)
result = {"path": path, "size": size, "move": move}
```

Verify with `GET /object?path=<path>`:

- bbox size matches the requested size
- bbox `min.z` is approximately the table `max.z`
- viewport capture shows the cube resting on the surface

## Resize a Live-Added Prim

For live-added bodies, delete and respawn. Do not set `xformOp:scale`; it can scale the prim's position about the world origin.

```python
import isaaclab.sim as sim_utils
from pxr import Sdf

red = tuple(channel / 255 for channel in (230, 5, 5))
path = "/World/envs/env_0/LiveEdit/RedCube"
old_bbox = helpers.world_bbox(path)
new_size = old_bbox["size"][0] * 2
cx = (old_bbox["min"][0] + old_bbox["max"][0]) / 2
cy = (old_bbox["min"][1] + old_bbox["max"][1]) / 2
top = helpers.world_bbox("table")["max"][2]

helpers.get_stage().RemovePrim(Sdf.Path(path))
cfg = sim_utils.CuboidCfg(
    size=(new_size, new_size, new_size),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=red, roughness=0.45),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.001),
    mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
)
cfg.func(path, cfg)
move = helpers.move(path, pos=(cx, cy, top + new_size / 2), zero_velocity=True)
result = {"path": path, "new_size": new_size, "move": move}
```

## Move a Rigid Body or Articulation

Use `/object/teleport` for registered rigid bodies and articulations:

```bash
curl -fsS -X POST --json '{"name":"robot","translation":[0,0,1],"rotation_wxyz":[1,0,0,0],"zero_velocity":true}' \
  "${BRIDGE_URL}/object/teleport" -o "${RUN_DIR}/teleport.json"
python -m json.tool "${RUN_DIR}/teleport.json"
```

Read `GET /object?name=<key>` after the move and trust the `live` pose over the USD bbox.

For G1 or another floating-base robot, keep sampling `GET /object?name=robot` for at least 10-15 seconds after a stand move. A stable pose has bounded z plus no growing roll/pitch and no continuing x/y drift. If z keeps dropping or attitude/drift grows, the robot is falling; stop, revert or adjust the target/standoff/yaw, and re-test before baking.

## Move Embedded Kinematic Props

For USDs whose rigid body is embedded under an `AssetBaseCfg` root, use `helpers.move`. Raw USD xform writes can snap back.

```python
target_x = measured_x
target_y = measured_y
result = helpers.move("tray_a", pos=(target_x, target_y, None), zero_velocity=True)
```

`mode: "physx"` means the body was registered and moved through PhysX.
`mode: "usd"` means the prim was not registered at bridge init; relaunch is required for simulation.

## Bake Asset Changes

After visual validation, persist only requested changes:

| Live change | Source target |
|---|---|
| new rigid primitive | `arena/arena/assets/<env>.py` as `RigidObjectCfg` |
| moved rigid object | asset `init_state.pos` / `rot` |
| moved robot stand | env class `embodiment.set_initial_pose(...)` |
| resized support surface | source `spawn.scale` plus relaunch |
| moved kinematic tray/fixture | asset `init_state.pos` |

For a live-added primitive that lacks a `/bake` snippet, bake from the measured live bbox, not an approximate clearance. Set `init_state.pos.z` so the source bbox bottom reproduces the validated live bbox bottom (`table_top + size / 2` for a cube resting on the table). Do not add contact offset/rest offset to the visual center; that creates a visible floating prop.

Run `python -m py_compile`, dry-runs, then `local-agent/validate-bake.sh <env>`.
