<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# OVStage Render And Camera (CAE CFD Visualization)

## Triggers

Use this reference when a CAE/CFD viewer must publish a generated USD scene into
OVStage, render it with `ovrtx`, and move an orbit camera interactively without a
startup or rebuild "jump." It covers the full path: `population.open_usd_from_string`
publication, `renderer.attach_ovstage` + `renderer.step` frame extraction of
`LdrColor`, the `omni:xform` data-plane camera commit, write-floor ordering, the
camera-jump fix, command coalescing, and the recommended geometry-swap architecture.

This is the applied recipe. For the general contracts it builds on, read
`references/ovstage-runtime/README.md`, `references/ovstage-data-plane/README.md`,
`references/ovrtx-rendering/README.md`, and `references/prim-transform-safety/README.md`.
Follow `references/conventions.md` for camera matrix layout, input mapping, and
renderer ownership.

## What This Doc Delivers

Publish the composed USD to OVStage, render it with `ovrtx`, and commit live camera
poses through the data plane so orbit/pan/dolly update the view without a flash. One
serialized loop owns all population, writes, and `step()` calls.

## Publish The Stage

Create an OVStage `Stage`, populate the composed USDA string at ordinal 1, wait for
the write floor, then attach the renderer. Author the live camera pose into the USDA
(see the camera-jump fix below) so the authored ordinal-1 pose already matches the
pose the next tick will commit.

```python
import ovstage
from ovstage import PopulationDomain, population

ordinal = 1
usda = build_viewer_usda(surface, width, height, colormap, camera_xform=pose)  # live pose in

stage = ovstage.Stage("viewer")
population.open_usd_from_string(
    stage, usda, ordinal=ordinal, domains=PopulationDomain.RENDERING
)
stage.advance_write_floor(ordinal).wait()

renderer.attach_ovstage(stage)   # attach only after the first publication is committed
```

Keep the `Stage` alive until `renderer.detach_ovstage()` or renderer destruction.
Detach the prior stage before swapping to a new one, so a `step()` never runs against
a stale stage.

## Construct The Renderer And Extract Frames

Build the renderer once, in sync mode, pinned to GPU 0, and kept alive across stage
swaps (long-running viewers replace stages repeatedly).

```python
import os
os.environ.setdefault("OVRTX_SKIP_USD_CHECK", "1")   # BEFORE importing ovrtx

from ovrtx import Renderer, RendererConfig, Device

renderer = Renderer(config=RendererConfig(
    sync_mode=True, active_cuda_gpus="0", keep_system_alive=True,
))
```

`renderer.step()` returns a `RenderProductSetOutputs`, not a `dict`: it supports `[]`
and `in` but not `.get()`. Some builds also make it a context manager, so branch on
`hasattr(products, "__enter__")`. Extract `LdrColor` (RGBA8) and **copy inside the
`.map()` context** — the DLPack view is invalid after the context exits.

```python
def step_extract(renderer, render_product_path, render_var, ordinal, delta_time=1/60):
    products = renderer.step(
        render_products={render_product_path},
        delta_time=float(delta_time),
        ordinal=int(ordinal),
    )
    if hasattr(products, "__enter__"):
        with products as ctx:
            return _extract(ctx[render_product_path], render_var)
    return _extract(products[render_product_path], render_var)

def _extract(product, render_var):
    for frame in product.frames:
        render_vars = frame.render_vars           # supports `in`/[], NOT .get()
        if render_var in render_vars:
            with render_vars[render_var].map(device=Device.CPU) as rv:
                return np.from_dlpack(rv).copy()   # COPY inside the map ctx
    return None
```

`LdrColor` is a tonemapped `uint8 [H, W, 4]` RGBA image (channel-last). For an
ovstream/BGRA display path, swap R/B before streaming; a local RGBA viewport can use
the copy directly.

### First cold step compiles RTX shaders

The first `renderer.step()` on a cold machine or empty cache can take minutes while
RTX pipelines and shaders compile. This is CPU/driver-bound, so `nvidia-smi` can show
~0% GPU during compilation — the process is not hung. Use a first-frame timeout of at
least 300s and poll for the first non-`None` `LdrColor` frame before failing.

## Commit Camera Pose Through The Data Plane (Fast Path)

Moving the camera is a single `omni:xform` write to the camera prim at a new ordinal —
no stage rebuild. Reference: `ovstage_bridge.commit_camera_pose` /
`_write_camera_dataplane`.

```python
from ovstage import (
    AttributeSemantic, DLDataType, DLDataTypeCode, PrimMode, make_dltensor,
)

def commit_camera_pose(stage, camera_path, xform_rowmajor_f64, ordinal):
    m = np.asarray(xform_rowmajor_f64, dtype=np.float64).reshape(4, 4)
    # ONE 16-lane float64 element for the single camera prim: shape=[1], lanes=16.
    data = np.ascontiguousarray(m.reshape(1, 16), dtype=np.float64)
    tensor = make_dltensor(
        data,
        dtype=DLDataType(code=DLDataTypeCode.kDLFloat, bits=64, lanes=16),
        shape=[1],
    )

    pd = ovstage.PathDictionary(stage)
    with pd:
        path_list = pd.create_path_list_from_strings([camera_path])
        try:
            query = stage.query_from_path_list(path_list)
            try:
                op = stage.write_attribute(
                    query, "omni:xform", ordinal, tensor,
                    is_array=False,
                    semantic=AttributeSemantic.MATRIX,
                    prim_mode=PrimMode.UPSERT,
                )
                op.wait()
                if not op.ok:
                    raise RuntimeError(f"write_attribute failed: {op.error_message()}")
            finally:
                stage.release_query(query)          # release query handle
        finally:
            pd.destroy_path_list(path_list)         # release path list
    del data                                        # keep buffer alive until wait() returns

    stage.advance_write_floor(ordinal).wait()       # advance floor before stepping @ordinal
    return ordinal
```

Key facts:

- `omni:xform` is a **row-major `float64`** matrix (USD row-vector convention:
  row 0 = right, row 1 = up, row 2 = `-forward`, row 3 = eye/translation).
- OVStage DLPack storage is **one 16-lane element per prim** (`shape=[1]`, `lanes=16`,
  `AttributeSemantic.MATRIX`) — **NOT** an `N x 4 x 4` tensor. For the single camera,
  `shape=[1]`. Older standalone ovrtx Python compatibility snippets show `(N, 4, 4)`;
  do not use that shape on the OVStage data plane.
- Use `PrimMode.UPSERT` for this app-owned runtime attribute.
- Keep the backing NumPy buffer (`data`) alive until `op.wait()` returns; release the
  query and path list in `finally` blocks; then `advance_write_floor(ordinal).wait()`.

## Write-Floor Ordering Rules

- **Never render above the write floor.** The order is: write @ N →
  `advance_write_floor(N).wait()` → `renderer.step(..., ordinal=N)`.
- **One serialized loop owns everything** — all population, all `step()` calls, and all
  attribute writes run on the same lane. UI callbacks mutate local input state or
  enqueue a command; they never touch OVStage or `ovrtx` directly.
- Because the loop drains all queued commands (including stage rebuilds) *before* it
  ever calls `step()`, a step can never run while the stage is being rebuilt. **Never
  `step()` while (re)building the stage.**

Per-tick shape (see `runtime.tick` / `_render_current`):

```
drain commands            # may rebuild the stage at most once
if camera dirty:
    pose    = camera.get_camera_xform()
    ordinal = commit_camera_pose(stage, camera_path, pose, ordinal + 1)  # write, wait, advance floor
    rgba    = step_extract(renderer, render_product_path, "LdrColor", ordinal)
    viewport.set_frame(rgba)
```

## Camera-Jump Fix (Headline Gotcha)

**Symptom:** a flash/jump on startup and on every geometry rebuild.

**Cause:** the stage is authored with a *default* camera pose at ordinal 1, but one
ordinal later the live `omni:xform` commit writes a *different* fitted pose. The single
frame rendered between the two shows the default pose, so the view visibly snaps.

**Fix:** fit the camera **before** authoring, and thread the live pose into the USDA
build so the authored pose already equals the pose that will be committed:

```python
# On a new dataset or the first build, fit; on a preserved-camera rebuild, keep the
# current pose. Either way ordinal 1 (authored xform) == ordinal 2 (omni:xform commit).
if refit or self._build is None:
    self.camera.fit(surface.bbox_min, surface.bbox_max)
pose = self.camera.get_camera_xform()
build = build_stage(surface, width, height, colormap, camera_xform=pose)
```

Do the same on colormap-only rebuilds, but with `refit=False`: pass the *current* live
pose so the material change does not reframe the view.

**Related jump — identity UPSERT:** query the prim's world transform *before* creating a
live `omni:xform` column. A freshly UPSERTed column initializes to identity, so if a
render happens before the real matrix is written, the prim jumps to the origin. See
`references/prim-transform-safety/README.md` (snapshot the authored/world transform
first; never fall back to identity).

## Command Coalescing

A dragged slider (iso value, slice position, colormap) can enqueue dozens of commands
per frame. Drain the whole queue **last-wins per kind** and rebuild the stage **at most
once per frame**, so a single drag does not queue dozens of stage rebuilds. See
`runtime._drain_commands`:

```python
latest = {}
while True:
    try:
        kind, payload = self._commands.get_nowait()
    except queue.Empty:
        break
    latest[kind] = payload          # last value per kind wins
# apply latest -> set flags -> rebuild ONCE if anything changed
```

## Recommended Next Architecture (Note, Not Required)

Rebuilding the entire `Stage` on every edit forces a renderer detach/attach and
recreates the camera prim (and its live `omni:xform`) each time — which is exactly what
reopens the jump risk and the identity-UPSERT hazard.

Prefer keeping **one persistent `Stage`** and swapping only the geometry prim, using
`population.add_usd_reference_from_string` / `population.remove_usd` /
`population.apply_usd_changes` instead of re-publishing the whole stage. The camera prim
and its live `omni:xform` stay alive across edits, eliminating per-edit detach/attach and
the associated jump. Adopt this when the CFD viewer needs frequent geometry edits
(field/representation/iso/slice changes) with a stable camera.

## See Also

- `references/ovstage-runtime/README.md` — ordinals, write floor, runtime ownership.
- `references/ovstage-data-plane/README.md` — queries, `omni:xform`, tensor lifetime.
- `references/ovrtx-rendering/README.md` — `Renderer`, `step`, `LdrColor`, first-run compile.
- `references/prim-transform-safety/README.md` — snapshot before live `omni:xform`; no identity fallback.
- `references/camera-controls/README.md` — orbit/pan/dolly and camera fit.
