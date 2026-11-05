# Prim Transform Safety

Use this reference for live object transforms, gizmo drags, animation, and
restoring temporary edits without changing the user's USD asset.

## Transform ownership and coordinate space

Keep one application-owned transform model. Read the current OVStage value for
an already-edited prim, or capture its authored local transform before creating
a runtime override. Write `omni:xform` through OVStage, publish the ordinal, then
render it. Standalone renderer bindings are for renderer-owned applications.

`omni:xform` is a **local transform**. Matrices are row-major `float64`, with
translation in row 3. Writing a world transform as local applies the parent
transform again and moves nested objects incorrectly.

For row-vector matrices, world `W = L @ P`, where `L` is local and `P` is the
current parent-to-world transform. Convert a desired world transform back with
`L = W @ inverse(P)`. Account for USD reset-xform-stack behavior when resolving
`P`; a reset stack has no inherited parent transform. Reject singular parents
for world-space manipulation rather than substituting identity.

## Resolve current world transforms

Use the same runtime transform adapter for the inspector, gizmo, and animation.
Read copied `omni:xform` and `omni:resetXformStack` values for the prim and its
ancestors at one published floor. Compose locals from the prim upward, stopping
after the first reset-stack prim; its descendants still inherit from it.
For a reset-stack target, its effective parent matrix is identity.

Batch the required ancestry reads and reuse the snapshot within the owner's
tick. Invalidate descendant world caches when an ancestor changes. Authored
`pxr` transforms cannot reflect later OVStage-only edits, and a cached world
matrix for only the edited path leaves its descendants stale. If using runtime
derived `omni:fabric:worldMatrix` outputs instead, finish direct local/reset-stack
writes at `N`, then call `stage.compute_hierarchy(input_ordinal=N,
output_ordinal=N, model=model)` before advancing the floor. Select a supported
model using `stage.get_hierarchy_computation_models()` and ensure the derived
output rows exist through population/setup. Wait for async computation if used.
See the upstream [transform contract](https://github.com/NVIDIA-Omniverse/ovstage/blob/main/docs/scene/transforms.rst).
This explicit step is for consumers requiring computed world outputs after direct
input writes; composing copied locals yourself does not require it. USD changes
processed through `RENDERING` population (including `ALL`) already compute their
derived transforms; population without the rendering domain needs explicit
hierarchy computation before consumers read derived world outputs.

## Capture a baseline

If metadata comes from `pxr`, run this in the USD worker and return copied
arrays. Use the selected playback time when working with animated assets:

```python
from pxr import Usd, UsdGeom
import numpy as np

def get_local_transforms(stage, prim_paths, time=Usd.TimeCode.Default()):
    result = {}
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsA(UsdGeom.Xformable):
            continue
        local = UsdGeom.Xformable(prim).GetLocalTransformation(time)
        matrix = np.array(local, dtype=np.float64).reshape(4, 4)
        if np.isfinite(matrix).all():
            result[path] = matrix
    return result
```

Keep baseline values separate from animation output. Starting a drag snapshots
the current live transform; ending a temporary animation restores its captured
baseline. If a valid baseline is unavailable, leave the prim unchanged.

## Batch writes and publication

OVStage transform write inputs use one 16-lane element per prim (`shape=[N]`,
lanes `16`). Read/map outputs use scalar lanes and shape `[N,4,4]` with explicit
strides; map-creation dtypes remain lane-based. Match the matrix rows to
the query's actual prim order; a sorted UI selection is not necessarily that
order.

```python
import numpy as np
from ovstage import AttributeSemantic, DLDataType, DLDataTypeCode, PrimMode, make_dltensor

def write_local_transforms(stage, query, ordered_paths, matrices_by_path, ordinal):
    matrices = np.stack([matrices_by_path[path] for path in ordered_paths])
    tensor = make_dltensor(
        matrices.reshape(len(ordered_paths), 16).astype(np.float64),
        dtype=DLDataType(code=DLDataTypeCode.kDLFloat, bits=64, lanes=16),
        shape=[len(ordered_paths)],
    )
    stage.write_attribute(
        query, "omni:xform", ordinal=ordinal, tensors=tensor,
        is_array=False, semantic=AttributeSemantic.MATRIX,
        prim_mode=PrimMode.UPSERT,
    ).wait()

# Group camera, animation, and interaction writes in one runtime tick.
write_local_transforms(stage, query, ordered_paths, local_matrices, ordinal)
stage.advance_write_floor(ordinal).wait()
products = renderer.step(render_products={product_path}, delta_time=dt, ordinal=ordinal)
```

The blocking write keeps the tensor alive until consumption completes. For
asynchronous writes, retain the source tensor until its operation completes.
Publish once after all writes contributing to the tick have finished.

## Gizmos and animation

The UI owns pointer interaction; the runtime owns transforms:

1. Snapshot selected prims' current local transforms at drag start. Keep this
   selection separate from descendant meshes used for outlines.
2. Compose each update from the snapshot, not the previous frame, to avoid
   drift. Interpret the delta in the gizmo's chosen coordinate space.
3. Convert world-space results back to local and enqueue the resulting matrices
   for the runtime's next publication. Preserve valid parent transforms when
   both a parent and descendant are selected.
4. Refresh inspector values from the live transform model. Commit the final
   value on release or restore the snapshot on cancel.

For a translation expressed in the prim's parent coordinates:

```python
offset = np.eye(4, dtype=np.float64)
offset[3, :3] = offset_in_parent_coordinates
local_matrices[path] = drag_start_local[path] @ offset
```

Coordinate gizmos and animations through the same model. Pause an animation
while the user manipulates its prim, or explicitly compose its offset with the
drag result; competing writers otherwise overwrite one another.

Use renderer-native picking to discover selected paths. Visibility changes
belong to the visibility attribute, rather than zeroing transforms to infer
segmentation IDs.

## Saving runtime edits

For saving selected runtime transforms or an edited subtree, read OVStage's
current [exporting-to-USD skill](https://github.com/NVIDIA-Omniverse/ovstage/blob/main/skills/exporting-to-usd/SKILL.md).
It owns export selection/projection APIs, destination lifetimes, completion,
and saving. Choose explicitly between a sparse edit overlay and a typed subtree;
neither implies a complete scene/asset round trip.

Capture the intended selection and committed runtime snapshot on the owner.
Choose an application-controlled destination or validated user save workflow,
with explicit overwrite policy. Keep source assets unchanged unless editing them
was requested. Reopen the result through a supported USD boundary and verify the
intended world pose, parent/reset-stack semantics, and required asset references.
An accepted export operation alone does not prove a usable saved composition.
Do not prescribe a release-specific repair without verifying the selected
exporter's current contract and a focused round-trip result.

## Standalone bindings

In a renderer-owned app, `bind_attribute(..., prim_mode=PrimMode.CREATE_NEW)`
can create a missing `omni:xform` initialized to identity. Capture the local
baseline before creating the binding and initialize it before the next step.
Use `Semantic.XFORM_MAT4x4` for standalone matrix writes. Keep the binding and
its mapped data inside the renderer's lifetime; attached apps use the OVStage
path above instead.

## Lifecycle and verification

Recreate queries and bindings after scene replacement. Refresh baselines when
composition or playback changes the authored transform, and discard late
commands from a previous stage generation.

Test a prim under a translated and rotated parent, then move the parent too.
Check numeric local/world results, cancel/restore behavior, and synchronized
highlight/inspector updates. A visible gizmo alone does not prove the write path.

See [OVStage data plane](../ovstage-data-plane/README.md),
[selection-animation](../selection-animation/README.md), and
[stage-management](../stage-management/README.md).

For XR, let OVXR consume the committed edit through its normal frame path;
do not add an application-owned `renderer.step()` to apply the transform.
