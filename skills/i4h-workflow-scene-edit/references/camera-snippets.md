# Camera Edit Snippets

Load this when adding a fixed scene camera or making a camera usable by policy, recording, conversion, or training.

## Rule

A camera prim is not enough. A downstream camera must be a rendered sensor and must be wired through task observations, YAML, dataset mapping, and modality config in one pass.

## Source Sensor Pattern

Add a `TiledCameraCfg` as an env-local scene sensor. Prefer naming the scene key, recorded obs key, and dataset mapping key the same.

```python
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils

camera_pos = (x, y, z)  # choose from a live viewport pose or measured scene pose
camera_rot = (qw, qx, qy, qz)
clip_near = 0.05
clip_far = 100000.0

robot_room_cam = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/RoomCamera",
    offset=TiledCameraCfg.OffsetCfg(
        pos=camera_pos,
        rot=camera_rot,
        convention="ros",
    ),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(clip_near, clip_far),
    ),
    width=640,
    height=480,
    update_period=1 / 30.0,
)
```

For inline scene assets, add the camera key to the asset list returned by `make_<env>_scene_assets()`.

## Task Observation Pattern

Record the camera through `observations.policy`:

```python
import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

env_cfg.observations.policy.robot_room_cam = ObsTerm(
    func=mdp.image,
    params={"sensor_cfg": SceneEntityCfg("robot_room_cam"), "data_type": "rgb", "normalize": False},
)
```

The observation term name is the HDF5 `obs/<key>` name. Do not add `_rgb` to the recorded key; `_rgb` belongs to policy runtime camera observations.

## YAML Pattern for G1 GR00T Locomanip

```yaml
zenoh:
  camera_names: [head, room]
policy:
  pov_cam_names_sim:
    - {obs_key: robot_head_cam_rgb, video_key: ego_view}
    - {obs_key: robot_room_cam_rgb, video_key: room}
  train:
    modality_config_path: policy/gr00t_n16/policy/locomanip/config_dualcam.py
dataset:
  camera_mappings:
    robot_head_cam: observation.images.ego_view
    robot_room_cam: observation.images.room
  modality_template_path: policy/gr00t_n16/policy/locomanip/modality_dualcam.json
```

For other stacks, inspect the closest working env and policy train config before choosing modality files.

## Live Preview Pattern

Do not create or initialize an IsaacLab `Camera`/`TiledCamera` sensor in a live `/script`, and do not inspect `cam.data` from a bridge script. Those calls can force replicator/RTX sensor initialization on the Isaac main loop, causing the sim window to freeze and subsequent `/script`, `/cameras`, or `/bake` requests to time out even though `/health` still responds.

For a quick preview, a `/script` may create a USD-only camera prim or compute a camera pose from the viewport/scene geometry. That proves placement only, not downstream readiness. Before reporting a baked camera as usable, freshly relaunch and verify:

```bash
curl -fsS "${BRIDGE_URL}/cameras" -o "${RUN_DIR}/cameras.json"
python -m json.tool "${RUN_DIR}/cameras.json"
curl -fsS -X POST --json '{"output_dir":"<run>/captures","viewport":true}' \
  "${BRIDGE_URL}/capture" -o "${RUN_DIR}/capture.json"
python -m json.tool "${RUN_DIR}/capture.json"
```

Expected:

- `/cameras` lists the new sensor key
- `local-agent/validate-bake.sh <env>` reports each camera obs key recorded in the policy group
- the capture image from the new camera is not blank and frames the intended area

For a room camera derived from the current perspective view, validate the frame against the final edited scene, not the pre-edit viewport. The capture must include the room/task context and the main objects: robot/table relationship, support surface, task tools/destinations, and any newly added object. Reject any candidate where the robot body/head/hands, table, trays/tools, or newly added object touches or is cut off by an image edge. If the current perspective is too tight, zoom out before baking by backing the camera away from the look-at point and/or widening the lens; re-capture until the main objects are all visible with margin. When baking a 640x480 fixed camera from a 16:9 viewport, use extra horizontal margin or a wider lens because the final sensor may crop differently.
