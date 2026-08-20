# Camera Auto-Select

## Purpose

When a user says "here's my stage" the viewer should open to a meaningful
camera on the very first frame — not to an arbitrary default position. This
skill inspects the stage for authored cameras and picks the best one, or
computes a fit-all fallback.

## Triggers

Use when:
- Loading a user-provided USD stage for the first time.
- Building a viewer application that must "just work" with arbitrary stages.
- The default camera position is wrong or unhelpful for a given scene.
- A build pipeline needs to determine the hero camera before deployment.

## Priority Heuristic

Evaluate cameras in this order. Stop at the first match.

| Priority | Condition | Rationale |
|----------|-----------|-----------|
| 1 | Stage metadata has `defaultCamera` | Author explicitly chose one. |
| 2 | Camera prim named `*Main*`, `*Hero*`, `*Default*`, `*Persp*` (case-insensitive) | Common naming convention. |
| 3 | Exactly one camera in the stage | No ambiguity — use it. |
| 4 | Camera with widest FOV (lowest `focalLength`) that is NOT top-down (X-rotation ≈ 90°) | Likely the overview/hero shot. |
| 5 | First camera in scene traversal order | Deterministic fallback. |
| 6 | Compute bbox-fit camera | Stage has no authored cameras at all. |

## Runtime Boundary

For an attached OVStage viewer, obtain camera metadata and bounds through the
application runtime adapter. The adapter owns the active stage, returns copied
camera DTOs, and applies the selected initial camera through the OVStage data
plane before publishing its ordinal. Do not open a second `Usd.Stage`, retain a
USD prim outside that boundary, or use renderer attribute APIs as an alternate
source of camera state.

The `pxr` helpers below are for an explicitly separate offline/preflight path:
for example, inspecting a known USD file before an application creates its
runtime. They are not part of an attached viewer live load, query, or write
path.

## Offline / Preflight Implementation (Python / pxr)

```python
import math
from pxr import Usd, UsdGeom, Gf

HERO_NAME_PATTERNS = ["main", "hero", "default", "persp", "perspective"]


def find_best_camera(stage: Usd.Stage) -> str | None:
    """Return the prim path of the best camera, or None if bbox-fit needed."""

    # Priority 1: explicit defaultCamera in layer metadata
    root_layer = stage.GetRootLayer()
    default_cam = root_layer.customLayerData.get("defaultCamera")
    if default_cam:
        prim = stage.GetPrimAtPath(default_cam)
        if prim and prim.IsA(UsdGeom.Camera):
            return str(prim.GetPath())

    # Collect all cameras
    cameras = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            cameras.append(prim)

    if not cameras:
        return None  # caller should use bbox-fit

    # Priority 2: name matching
    for cam in cameras:
        name_lower = cam.GetName().lower()
        for pattern in HERO_NAME_PATTERNS:
            if pattern in name_lower:
                return str(cam.GetPath())

    # Priority 3: single camera
    if len(cameras) == 1:
        return str(cameras[0].GetPath())

    # Priority 4: widest FOV, skip top-down
    best_cam = None
    lowest_focal = float("inf")
    for cam in cameras:
        focal = cam.GetAttribute("focalLength").Get() or 50.0
        if _is_top_down(stage, cam):
            continue
        if focal < lowest_focal:
            lowest_focal = focal
            best_cam = cam

    if best_cam:
        return str(best_cam.GetPath())

    # Priority 5: first in traversal
    return str(cameras[0].GetPath())


def _is_top_down(stage: Usd.Stage, cam_prim) -> bool:
    """Heuristic: camera looking straight down (X rotation ~90°)."""
    xformable = UsdGeom.Xformable(cam_prim)
    xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # Extract the forward vector (negative Z in camera space)
    forward = xform.TransformDir(Gf.Vec3d(0, 0, -1))
    up_axis = UsdGeom.GetStageUpAxis(stage)
    if up_axis == UsdGeom.Tokens.z:
        world_down = Gf.Vec3d(0, 0, -1)
    else:
        world_down = Gf.Vec3d(0, -1, 0)
    # If forward is within ~10° of straight down, it's top-down
    dot = Gf.Dot(forward.GetNormalized(), world_down)
    return dot > math.cos(math.radians(10))
```

### Bbox-Fit Fallback

When no authored camera exists, compute a fit-all orbit:

```python
def compute_bbox_fit_camera(stage: Usd.Stage, fov_deg: float = 60.0):
    """Return (target, distance, elevation, azimuth) for an OrbitCamera."""
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    world_bbox = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot())
    bbox_range = world_bbox.ComputeAlignedBox()

    center = (bbox_range.GetMin() + bbox_range.GetMax()) / 2.0
    size = bbox_range.GetMax() - bbox_range.GetMin()
    max_dim = max(size[0], size[1], size[2])

    # Distance to fit the bounding sphere in view
    half_fov = math.radians(fov_deg / 2.0)
    distance = (max_dim / 2.0) / math.tan(half_fov) * 1.2  # 20% padding

    # Default orbit angles: slight elevation, 3/4 azimuth
    elevation = math.radians(25.0)
    azimuth = math.radians(-45.0)

    return {
        "target": [center[0], center[1], center[2]],
        "distance": distance,
        "elevation": elevation,
        "azimuth": azimuth,
    }
```

### Emitting camera_config.json

During offline preflight, write a config the frontend can consume:

```python
import json

def emit_camera_config(stage: Usd.Stage, output_path: str = "camera_config.json"):
    """Write camera config for the frontend."""
    cameras = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            cam = UsdGeom.Camera(prim)
            cameras.append({
                "path": str(prim.GetPath()),
                "name": prim.GetName(),
                "focalLength": cam.GetFocalLengthAttr().Get() or 50.0,
            })

    best = find_best_camera(stage)
    config = {
        "cameras": cameras,
        "defaultCamera": best,
        "hasBboxFallback": best is None,
    }

    if best is None:
        config["bboxFit"] = compute_bbox_fit_camera(stage)

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    return config
```

## Integration Points

### Attached OVStage Viewer (stage-loading)

After the runtime owner has populated the composed scene, ask its camera query
adapter for copied camera metadata, default-camera metadata, and a bounds
summary. Apply the selected authored camera or bbox-fit pose through the camera
command adapter. That command owns the OVStage transform/lens writes, waits for
them, advances the write floor, and returns the committed ordinal for the next
render.

```python
camera_state = runtime.camera_queries.initial_view()

if camera_state.authored_camera_path:
    committed = runtime.camera_commands.use_authored_camera(
        camera_state.authored_camera_path
    )
else:
    committed = runtime.camera_commands.apply_bbox_fit(camera_state.bbox_fit)

renderer.step(render_products, delta_time, ordinal=committed.ordinal)
```

The method names are an application adapter contract, not OVStage API names.
Implement them from the current OVStage population/query/data-plane examples and
the current OVRTX attached-stage guidance. Keep all USD and OVStage handles
inside the runtime owner.

### Offline / Preflight Tool

An offline tool may call `Usd.Stage.Open()` and the helpers above to emit a
static `camera_config.json`. Treat that output as a hint: the attached viewer
must re-query its active OVStage generation after population or scene switching.

### Frontend (streaming-client)

On stage load response, read the camera list and set the initial view. If the
app includes a camera picker (see `camera-picker` skill), populate it from the
same data.

### Build Pipeline

For pre-built/deployed apps where the stage is known at build time, run
`emit_camera_config()` during the build step and bundle the JSON with the app
assets. The frontend reads it at startup without needing a round-trip to the
server.

## Gotchas

- `defaultCamera` in layer metadata is a custom field — not all stages set it.
  The heuristic handles this gracefully.
- Some stages define cameras inside referenced assets (props with internal
  cameras). Filter to cameras under `/World/Cameras` or at the root level to
  avoid picking internal asset cameras.
- Top-down cameras are useful for plan views but make poor defaults for first
  impressions. The heuristic deprioritizes them.
- For multi-GPU or multi-viewport setups, each viewport can have its own
  camera. This skill picks the *initial default* only.
- In an attached viewer, invalidate camera DTOs on a stage-generation change;
  never reuse an offline `pxr` prim or a previous runtime path handle.

## See Also

- `camera-controls` — orbit, pan, zoom, and fly input handling.
- `camera-picker` — UI dropdown for switching between stage cameras.
- `stage-loading` — stage open and session setup.
- `stage-hierarchy` — traversal and bbox computation.
