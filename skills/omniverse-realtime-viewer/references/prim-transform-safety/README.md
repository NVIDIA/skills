# Prim Transform Safety

## Triggers

Use this skill for OVStage transform writes, row-major `omni:xform` matrices, write-floor transform publication, standalone `bind_attribute` `PrimMode.CREATE_NEW` safety, zero-scale discovery, selection animation, transform restore, or prims jump to the origin.

Use this whenever a viewer writes live `omni:xform` attributes for scene prims. The risky operations are selection animation, hide/show discovery, zero-scale isolation, and any runtime transform manipulation.

For OVStage transform data-plane or ovrtx standalone compatibility behavior not
covered here, read `references/dependencies` for acquisition guidance and
supplemental dependency documentation.

## Core Rule

For new ovrtx viewers, write runtime transforms through OVStage, not direct
renderer bindings. Snapshot authored or current runtime transforms before
creating or replacing any live `omni:xform` column, write the intended matrix at
a new ordinal, wait, advance the write floor, then render that committed
ordinal. If the app writes identity as a fallback or renders above the write
floor, prims can jump to the origin or appear stale.

Safe sequence:

1. Query world transforms from USD before binding.
2. Create or reuse an OVStage query for the prim paths that will move.
3. Write each saved world transform to `omni:xform` at a new ordinal before any render step that can show the new runtime column.
4. Perform temporary edits, such as zero-scale isolation or selection animation.
5. Restore from the saved world transform or the app's live-transform cache, not from an uninitialized runtime value.

For standalone compatibility, `renderer.bind_attribute(...,
prim_mode=PrimMode.CREATE_NEW)` creates a Fabric attribute if one does not
already exist. For `omni:xform`, that new attribute initializes to identity. If
the app renders before writing the real transform, the prim can jump to the
origin and lose its authored placement in the rendered stage.

## Query World Transforms First

Use `pxr` directly or through a worker process, depending on the viewer's import isolation.

```python
from pxr import Usd, UsdGeom
import numpy as np

def get_world_transforms(stage: Usd.Stage, prim_paths: list[str]) -> dict[str, np.ndarray]:
    result = {}
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
            continue
        mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        xform = np.array(mat, dtype=np.float64).reshape(4, 4)
        if np.isfinite(xform).all():
            result[path] = xform
    return result
```

Do this before creating a live OVStage `omni:xform` value or standalone
`bind_attribute`. Reading a newly-created runtime value is not a safe
substitute; it may already be identity.

## OVStage Write And Publish

Use an OVStage query for the target prims. Matrix values are row-major
`float64`, with translation in row `[3][0..2]`. In OVStage DLPack form,
`omni:xform` is one 16-lane element per prim (`shape=[N]`, lanes `16`), not an
`N x 4 x 4` tensor of one-lane elements.

```python
import numpy as np
from ovstage import (
    AttributeSemantic,
    DLDataType,
    DLDataTypeCode,
    PrimMode,
    make_dltensor,
)

def publish_xforms(stage, query, matrices_by_path, ordered_paths, ordinal):
    matrices = np.stack([matrices_by_path[path] for path in ordered_paths])
    xform_tensor = make_dltensor(
        matrices.reshape(len(ordered_paths), 16).astype(np.float64),
        dtype=DLDataType(code=DLDataTypeCode.kDLFloat, bits=64, lanes=16),
        shape=[len(ordered_paths)],
    )
    stage.write_attribute(
        query,
        "omni:xform",
        ordinal=ordinal,
        tensors=xform_tensor,
        is_array=False,
        semantic=AttributeSemantic.MATRIX,
        prim_mode=PrimMode.UPSERT,
    ).wait()
    stage.advance_write_floor(ordinal).wait()
```

After publishing, render with `renderer.step(..., ordinal=ordinal)`. Python
`step(..., ordinal=...)` updates the attached renderer from OVStage before
rendering; call `renderer.update_from_stage(ordinal)` separately only when the
app needs renderer state updated before stepping.

## Standalone Binding Compatibility

```python
import warp as wp
from ovrtx import Device, PrimMode

bindings = {}
base_xforms = get_world_transforms(stage, prim_paths)

for path in prim_paths:
    if path not in base_xforms:
        continue
    bindings[path] = renderer.bind_attribute(
        prim_paths=[path],
        attribute_name="omni:xform",
        dtype="float64",
        shape=(4, 4),
        prim_mode=PrimMode.CREATE_NEW,
    )

for path, bind in bindings.items():
    with bind.map(device=Device.CPU) as mapped:
        wp.from_dlpack(mapped.tensor).numpy().reshape(1, 4, 4)[0] = base_xforms[path]
```

In standalone compatibility mode, the second loop must run before the next
render step. This turns the new live attribute into a faithful copy of the
authored world transform before later code manipulates it.

## Safe Temporary Manipulation

For isolation-based ID discovery, hide one prim by writing a zero matrix, render
the committed ordinal, then restore its saved transform. Prefer renderer-native
picking for new viewers; use this pattern only when a selected capability still
needs render-diff discovery.

```python
zero_xform = np.zeros((4, 4), dtype=np.float64)

baseline_ids = render_and_read_instance_ids(renderer)

for path in ordered_paths:
    ordinal = runtime.next_ordinal()
    publish_xforms(stage, queries[path], {path: zero_xform}, [path], ordinal)
    hidden_ids = render_and_read_instance_ids(renderer, ordinal=ordinal)
    missing_ids = baseline_ids - hidden_ids

    ordinal = runtime.next_ordinal()
    publish_xforms(stage, queries[path], {path: base_xforms[path]}, [path], ordinal)
    render_and_read_instance_ids(renderer, ordinal=ordinal)

    for instance_id in missing_ids:
        id_to_path[instance_id] = path
```

The same pattern applies to animation: compose offsets with `base_xforms[path]`, and restore that base transform when animation ends.

```python
def write_offset(path: str, offset_xyz: np.ndarray) -> None:
    offset_mat = np.eye(4, dtype=np.float64)
    offset_mat[3, 0:3] = offset_xyz
    ordinal = runtime.next_ordinal()
    publish_xforms(stage, queries[path], {path: base_xforms[path] @ offset_mat}, [path], ordinal)
```

## Standalone Batch Write Alternative

If a standalone compatibility helper does not need long-lived bindings, use
`renderer.write_attribute`, but still query transforms first and write the real
values immediately.

```python
from ovrtx import DataAccess, PrimMode, Semantic

paths = [path for path in prim_paths if path in base_xforms]
xforms = np.stack([base_xforms[path] for path in paths]).astype(np.float64)

renderer.write_attribute(
    prim_paths=paths,
    attribute_name="omni:xform",
    tensor=xforms,
    semantic=Semantic.XFORM_MAT4x4,
    prim_mode=PrimMode.CREATE_NEW,
    data_access=DataAccess.SYNC,
)
```

Use standalone bound attributes for per-frame compatibility updates. Use batch
writes for one-shot standalone initialization or reset. In attached ovrtx
apps, use the OVStage write/publish pattern above instead.

## Interactive Gizmo Drag Pattern

For selected-prim transform gizmos, treat the gizmo as a UI input source and
keep transform authority in one runtime model. Do not stop at rendering the
handle; every drag path must call a live `omni:xform` write.

Safe drag lifecycle:

1. On selection, keep the selected path list separate from the mesh paths used
   for highlight outlines.
2. On drag start, snapshot each selected prim's current transform. Prefer the
   app's live-transform cache when the prim has already moved; otherwise query
   USD world transform before creating a live `omni:xform`.
3. On drag move, compose the drag delta from the drag-start snapshot rather
   than incrementally reading back a newly-created `omni:xform`.
4. Write the composed transform through OVStage at a new ordinal, wait, advance
   the write floor, and render or update the renderer with that ordinal.
5. In standalone compatibility code only, use `Semantic.XFORM_MAT4x4`,
   `PrimMode.CREATE_NEW`, and `DataAccess.SYNC`.
6. On drag end, clear the snapshot and refresh selected-prim telemetry from the
   live-transform cache.

```python
class TransformDragModel:
    def __init__(self, runtime):
        self.runtime = runtime
        self.selected_paths = []
        self.start_xforms = {}

    def on_drag_start(self):
        self.start_xforms = {}
        for path in self.selected_paths:
            xform = self.runtime.get_live_or_usd_world_transform(path)
            if xform is not None:
                self.start_xforms[path] = xform

    def on_drag_moved(self, delta_matrix):
        for path, base in self.start_xforms.items():
            self.runtime.write_live_xform(path, base @ delta_matrix)

    def on_drag_ended(self):
        self.start_xforms.clear()
```

Validation must assert a numeric transform delta for a known prim. A screenshot
showing a visible gizmo is not enough; the selected prim must move and the
highlight/inspector must follow the live transform.

## Scene Lifecycle

- Recompute world transforms after every scene load, reload, variant change, or selectable-set rebuild.
- Recreate OVStage queries/maps and standalone bindings after stage replacement,
  OVStage reset, or standalone renderer reset.
- Do not keep transform queries, maps, path IDs, or bindings across scenes.
- Do not call `renderer.step()` concurrently with transform discovery,
  OVStage population/write-floor work, or scene reset.
- If a prim is missing a valid world transform, skip writing it instead of falling back to identity.

## Anti-Patterns

```python
# Wrong: CREATE_NEW may read back identity, not the authored transform.
bind = renderer.bind_attribute(..., attribute_name="omni:xform", prim_mode=PrimMode.CREATE_NEW)
with bind.map(device=Device.CPU) as mapped:
    original = wp.from_dlpack(mapped.tensor).numpy().reshape(1, 4, 4)[0].copy()

# Wrong: identity fallback silently moves a prim to the origin.
original_xforms[path] = np.eye(4, dtype=np.float64)

# Wrong: authored USD xform ops are not the live ovrtx update path.
renderer.write_attribute(..., attribute_name="xformOp:transform")

# Wrong: direct renderer writes become the scene truth while an OVStage is attached.
renderer.write_attribute(..., attribute_name="omni:xform")
```

## Gotchas

- Standalone `PrimMode.EXISTING_ONLY` can skip missing live attributes; use it only when inline session data already created them. OVStage examples use OVStage `PrimMode.UPSERT` for app-owned runtime transforms.
- `omni:xform` matrices are `float64`, row-major, with translation in row 3 in the viewer patterns used here. OVStage DLPack writes use one 16-lane element per prim; standalone ovrtx Python compatibility writes may use `(N, 4, 4)` NumPy matrices.
- A zero-scale or zero matrix hide operation must always have a known restore matrix.
- Multi-mesh or instanceable assets can produce several segmentation IDs for one selected path; preserve path-to-many-ID behavior when needed.
- Transform-safe discovery should use the baseline, hide, diff, restore pattern
  above; never discover IDs by reading a newly-created `omni:xform` value or
  leaving a prim hidden across frames.

See also: `ovstage-data-plane`, `object-selection`, `selection-animation`, `ovrtx-rendering`, `stage-hierarchy`, `stage-management`.
