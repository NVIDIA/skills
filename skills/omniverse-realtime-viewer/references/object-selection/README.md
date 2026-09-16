# Object Selection

## Triggers

Use this skill for pick objects, click to select, object selection, native picking, pick queries, marquee selection, pickable prims, select prim, or wrong prim selected requests.

Use this for viewport picking and selected-prim state. Selection visuals live in
`selection-feedback`; selected-prim properties live in `prim-info-display`.

For ovrtx, the recommended path is the native picking API documented by
`native-picking-selection`.

For ovrtx selection or picking behavior not covered here, read
`references/dependencies` for acquisition guidance and supplemental dependency
documentation.

Read `viewer-input-routing` first when selection is driven by viewport clicks,
marquee gestures, WebRTC input, or click-vs-drag discrimination.

Do not create new `GpuPicker`, `cpu_picking.py`, `seg_outline.py`, Warp outline,
`learn_mapping`, isolation discovery, ID drift repair, or CPU ray/AABB fallback
systems for generated apps.

## Native Picking Facts

- Queue pick work with `Renderer.enqueue_pick_query()` in Python or
  `ovrtx_enqueue_pick_query()` in C/C++.
- Queue the pick query before the `renderer.step()` that should produce the
  result.
- Convert RenderProduct pixels to top-left normalized coordinates before
  enqueueing. `left` and `top` are inclusive; use one pixel past the selected
  bounds for `right` and `bottom`.
- The consumed step exposes the synthetic render var
  `ovrtx.OVRTX_RENDER_VAR_PICK_HIT` / `OVRTX_RENDER_VAR_PICK_HIT`.
- Pick hits store prim path IDs. Resolve them with
  `renderer.resolve_prim_path_id()` in Python or the C path dictionary utilities
  before printing names or updating selection.
- Multiple pick queries for the same RenderProduct before a single step are not
  queued independently; treat the last query as authoritative.
- The synthetic pick-hit output is automatic for the consuming step; do not add
  `ovrtx_pick_hit` to USD `orderedVars`.

## Pickability

Mark selectable prims with the native pickable helper. Pickability is renderer
interaction state, so the native OVRTX helper remains the right surface here.
Pass renderer path IDs, such as decoded pick-hit IDs; do not write a legacy
pickable attribute through `renderer.write_attribute()`. New apps should not build
or maintain a separate segmentation ID mapping just to decide what can be
selected.

Python:

```python
def set_pickable(renderer, prim_path_ids: list[int], enabled: bool) -> None:
    if prim_path_ids:
        renderer.set_pickable(prim_path_ids, enabled)
```

C/C++:

```c
// Prefer the native helper when writing C/C++ integration code.
ovrtx_set_pickable(renderer, prim_path_ids, prim_path_count, true);
```

When a frontend sends `makePrimsSelectable` or `makePrimsPickable`, update the
server's canonical pickable set and write pickability for the changed prims.
Do not recompute CPU bbox maps or segmentation ID maps as part of the normal
ovrtx selection path.

## Click Pick Flow

Only run selection on a completed click gesture. Camera orbit/pan/zoom drags
must not also select.

```python
def pick_at_render_pixel(renderer, render_product_path: str, x: int, y: int, width: int, height: int) -> list[str]:
    renderer.enqueue_pick_query(
        render_product_path=render_product_path,
        left_ndc=x / width,
        top_ndc=y / height,
        right_ndc=(x + 1) / width,
        bottom_ndc=(y + 1) / height,
    )

    products = renderer.step(
        render_products={render_product_path},
        delta_time=1.0 / 60.0,
    )

    with products as ctx:
        frame = ctx[render_product_path].frames[0]
        return resolve_pick_hit_paths(renderer, frame)
```

Keep UI coordinate conversion outside the picker:

1. Measure the displayed video/image rectangle.
2. Reject clicks in letterboxed areas.
3. Convert to RenderProduct pixel coordinates.
4. Clamp to `[0, width - 1]` and `[0, height - 1]`.
5. Call the native pick query.

## Marquee Selection

Drag selection uses the same native API with a larger rectangle:

```python
def marquee_pick(renderer, render_product_path: str, x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> list[str]:
    left = min(x0, x1)
    top = min(y0, y1)
    right = max(x0, x1) + 1
    bottom = max(y0, y1) + 1

    renderer.enqueue_pick_query(
        render_product_path=render_product_path,
        left_ndc=left / width,
        top_ndc=top / height,
        right_ndc=right / width,
        bottom_ndc=bottom / height,
    )

    products = renderer.step(
        render_products={render_product_path},
        delta_time=1.0 / 60.0,
    )

    with products as ctx:
        frame = ctx[render_product_path].frames[0]
        return resolve_pick_hit_paths(renderer, frame)
```

For additive or subtractive marquee modes, apply modifier keys after resolving
the hit paths and before broadcasting the canonical selection state.

## Decode Pick Hits

```python
import numpy as np
import ovrtx


def resolve_pick_hit_paths(renderer: ovrtx.Renderer, frame) -> list[str]:
    if ovrtx.OVRTX_RENDER_VAR_PICK_HIT not in frame.render_vars:
        return []
    pick_var = frame.render_vars[ovrtx.OVRTX_RENDER_VAR_PICK_HIT]

    mapping = pick_var.map(device=ovrtx.Device.CPU)
    try:
        magic = int(np.from_dlpack(mapping.params["magic"]).reshape(-1)[0])
        version = int(np.from_dlpack(mapping.params["version"]).reshape(-1)[0])
        hit_count = int(np.from_dlpack(mapping.params["hitCount"]).reshape(-1)[0])
        prim_path_ids = np.from_dlpack(mapping["primPath"]).copy().reshape(-1)
    finally:
        mapping.unmap()

    if magic != ovrtx.OVRTX_PICK_HIT_MAGIC or version != ovrtx.OVRTX_PICK_HIT_VERSION:
        raise RuntimeError("Unexpected ovrtx pick-hit schema")

    paths: list[str] = []
    seen: set[str] = set()
    for prim_path_id in prim_path_ids[:hit_count]:
        path = renderer.resolve_prim_path_id(int(prim_path_id))
        if path and path not in seen:
            paths.append(path)
            seen.add(path)
    return paths
```

If the app needs world-space hit data for gizmos or labels, also read
`worldPositionM`, `worldNormal`, `objectType`, and `geometryInstanceId` from the
same pick-hit render var after validating the schema.

## Selection State

Keep selection state centralized on the renderer/server side:

```python
def select_paths(paths: list[str], mode: str = "replace") -> None:
    previous = set(selection.paths)

    if mode == "add":
        current = previous | set(paths)
    elif mode == "subtract":
        current = previous - set(paths)
    else:
        current = set(paths)

    selection.paths = sorted(current)
    selection.mesh_paths = expand_to_descendant_meshes(selection.paths)
    selection_feedback.update(selection.mesh_paths)
    message_handler.send_message("stageSelectionChanged", {"prims": selection.paths})
```

Tree/sidebar selection and viewport selection must both call the same state
transition. The frontend mirrors `stageSelectionChanged`; it should not maintain
an independent authoritative selection.

For Xform or Scope selection, use `stage-hierarchy` to expand to descendant mesh
paths only for visual feedback. Preserve the user's selected tree path for the
stage tree and info panel.

## Runtime Feature Handoff

Selection is not a blanket OVStage write. Keep canonical selected paths in the
app/runtime controller, use native OVRTX selection groups for outlines, and only
handoff to OVStage when selection drives runtime data such as transforms,
material/effect attributes, visibility, physics poses, or reversible
viewer-owned state.

```text
native pick result
  -> resolve prim path IDs for the active stage generation
  -> update canonical selected paths
  -> update native OVRTX selection groups/outlines
  -> dispatch feature commands to the OVStage runtime owner
  -> feature writes publish at ordinal N
  -> attached renderer consumes N on its normal render step
```

Do not let the picker callback perform direct OVRTX material, visibility, or
`omni:xform` writes. The picker should enqueue intent, and the single runtime
owner should serialize OVStage writes with the same ordinal publication protocol
used for camera, animation, transform tools, and physics handoff.

## Scene Lifecycle

On scene switch or renderer reset:

- Do not let pick queries run while the renderer is loading or resetting.
- Clear selected paths, hover state, info panels, and pending pick requests.
- Reapply pickability after the new stage has loaded.
- Clear previous native selection outline groups through `selection-feedback`.
- Do not preserve pick-hit records across scenes.

## Deprecated Segmentation Fallback

Segmentation-buffer picking from `InstanceSegmentationSD` is a deprecated
compatibility path for older OVRTX builds or custom post-process tools. It is
not the recommended ovrtx path.

Only use the deprecated path when native pick queries are unavailable and the
user explicitly accepts the limitations: per-frame ID buffers, scene-local IDs,
ID-to-path mapping, ID drift after reloads, and possible mismatch with UI
selection. Do not scaffold it in new ovrtx apps.

## Gotchas

- Selection bugs are usually coordinate bugs. Convert through the visible
  image/video rectangle before calling the native pick query.
- Pick queries are consumed by a renderer step; enqueue before stepping.
- Native picking returns path IDs, not path strings.
- Write pickability before expecting a prim to appear in pick results.
- Do not call `renderer.step()` concurrently with scene reset/load or from
  input callbacks. Enqueue selection work for the render loop.
- A selection outline requires `selection-feedback`; picking only decides what
  is selected.

See also: `viewer-input-routing`, `native-picking-selection`,
`selection-feedback`, `prim-info-display`, `camera-controls`, `local-viewer`,
`streaming-server`, `streaming-messages`, `stage-hierarchy`,
`stage-management`.

## Generated Module Checklist - selection_controller.py

- [ ] Converts UI coordinates to RenderProduct pixels before picking.
- [ ] Queues click picks with `Renderer.enqueue_pick_query()` using normalized
      RenderProduct coordinates.
- [ ] Queues marquee picks with a native pick rectangle.
- [ ] Decodes `OVRTX_RENDER_VAR_PICK_HIT`.
- [ ] Resolves prim path IDs before broadcasting selection.
- [ ] Writes pickability with `Renderer.set_pickable()` or
      `ovrtx_set_pickable()` using renderer path IDs.
- [ ] Keeps selected paths server/runtime authoritative.
- [ ] Expands selected Xforms/Scopes to descendant mesh paths only for feedback.
- [ ] Dispatches transform, effect, visibility, and physics side effects through
      the OVStage runtime owner instead of direct picker writes.
- [ ] Clears selection and pending pick state on scene switch.
- [ ] Does not create `GpuPicker`, `cpu_picking.py`, `seg_outline.py`, or Warp
      outline systems.
