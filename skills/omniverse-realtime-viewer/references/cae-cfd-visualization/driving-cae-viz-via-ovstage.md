<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Driving CAE Visualization Through OVStage (Control → Correct Writes)

## Triggers

Use this reference when a UI control on a CAE/CFD viewer — a `setStreamlineSeed`
message, an iso-value or slice slider, a field/representation/dataset picker, a
colormap choice, an emitter drag, or an orbit-camera move — must be turned into
the **correct** OVStage writes that update the rendered visualization. It answers
the judgment question: *given this control changed, do I do a cheap live
attribute write, or must I re-author geometry?* — and it draws a hard line
between stable API contracts and application-specific optimization. The persistent stage, the emitter-marker
`omni:xform` write, and in-place data-plane array writes of live geometry are
patterns to validate against the current OVStage API before relying on them in a
generated app.

This is the write-classification recipe. For the mechanics it builds on, read
`ovstage-render-and-camera.md` (the full camera data-plane block, frame
extraction, camera-jump fix), `references/ovstage-runtime/README.md` (ordinals,
write floor, single owner), `references/ovstage-data-plane/README.md` (queries,
tensor lifetime), `references/ovstage-population/README.md` (publication /
reference-swap APIs), and `references/prim-transform-safety/README.md` (snapshot
before a live `omni:xform`; never fall back to identity). Follow
`references/conventions.md` for camera-matrix layout and renderer ownership.
The examples below are self-contained patterns. Adapt them to the current
upstream OVStage contracts rather than importing unpublished local projects.

## What This Doc Delivers

A single decision — *how does a control become the right OVStage write* — plus
the loop that carries it. Every control follows one path and every write is
classified by cost. The UI callback never touches OVStage or the renderer.

Although the examples here are CAE/CFD (iso-value, slice, streamline emitter,
colormap, dataset), **the decision and the loop are general**: any realtime viewer
that edits live geometry or attributes from a control faces the same
data-plane-write-vs-structural-rebuild choice. The generic local and streaming
viewer recipes (`references/ovui-local-viewer-recipe/README.md`,
`references/streaming-viewer-recipe/README.md`) point here for that pattern — read
it as the reusable "control → correct write" recipe, with CAE operators as the
worked example.

## The Loop: Control → Correct Writes (One Serialized Owner)

Exactly one thread ever populates the stage, writes an attribute, or calls
`renderer.step`. A control (an ovui callback, an ovstream `on_message`, a raw
input event) does **one** of two things and nothing else:

- mutate local, numpy-only input/camera state (through an `InputController`), or
- **enqueue** a typed command on a thread-safe queue.

The owning loop drains that queue, decides what kind of write each change needs,
performs it, advances the write floor, and only then steps the renderer.

```
control message  ─┐
slider drag      ─┤─►  validate + ENQUEUE(kind, payload)      # UI thread, no OV
raw input event  ─┘

owning loop tick:                                             # the ONLY OV lane
    drain queue        # last-wins coalesce per kind (see below)
    if a rebuild-affecting kind changed:
        recompute the warp-simdata operator (surface/iso/slice/streamlines)
        author the new geometry into OVStage at a NEW ordinal
    ordinal = commit_camera_pose(...)   # omni:xform write @N (data-plane)
    advance_write_floor(N).wait()       # never render above the floor
    rgba = renderer.step(ordinal=N)     # frame now reflects the control
    viewport.set_frame(rgba)
```

The invariant is simple: UI callbacks enqueue; the loop owns every write and
every `step`.

## Classify Every Write By Cost (The Key Judgment)

The central decision. When a control changes, ask *what actually changed in the
scene* — a transform/scalar, or the geometry topology — and pick the cheapest
correct write.

| Control that changed | What actually changes | Write class | Proven? |
| --- | --- | --- | --- |
| Orbit/pan/dolly camera | camera `omni:xform` | **Data plane** (matrix) | **Yes** — use the standard matrix-write helper |
| Emitter seed-sphere **position** (marker prim) | that prim's `omni:xform` | **Data plane** (matrix) | **Yes** — persistent unit sphere moved and scaled by an `omni:xform` write |
| Emitter drag (streamlines it emits) | same curves prim, new `points`/`curveVertexCounts`/`displayColor` arrays | **Data plane** (variable-length array) | **Yes** — in-place array writes to a persistent `BasisCurves` |
| Colormap / scalar-range tweak | material LUT / `displayColor` primvar values | **Data plane** (array/scalar), in principle | Use a structural rebuild until the selected OVStage version supports the attribute update you need |
| Representation swap (surface↔iso↔slice↔streamlines↔points) | new geometry **prim/topology class** | **Structural swap** (persistent stage) | **Yes** — incremental geometry-reference swap (~2.9× vs full rebuild) |
| Field swap / iso-value / slice-position (mesh/points reps) | operator recompute → new points/counts arrays | **Structural swap** (today) / geometry-array write (goal) | Ref-swap proven; array write for these reps is the remaining frontier |
| Dataset / preset load | whole scene + bbox + camera refit | **Structural rebuild** | Yes (rebuild path) |

Rule of thumb: **a transform or a scalar/array value on an existing prim is a
data-plane write; a change in topology (vertex/curve counts, connectivity, prim
set) is a structural change.** Reach for the cheap path first; only rebuild when
the topology genuinely changed.

### Data-plane writes (cheap, live, no rebuild)

The validated in the selected application data-plane write is the camera `omni:xform` commit: one 16-lane
`float64` matrix, `AttributeSemantic.MATRIX`, `PrimMode.UPSERT`, at a new
ordinal. The full lifecycle (path-dictionary → query → write → release handles →
keep the buffer alive until `op.wait()`) lives in `ovstage-render-and-camera.md`;
the load-bearing call is:

```python
# Application pattern — adapt to the current OVStage API
data   = np.ascontiguousarray(m.reshape(1, 16), dtype=np.float64)   # one prim, 16 lanes
tensor = make_dltensor(data,
                       dtype=DLDataType(code=DLDataTypeCode.kDLFloat, bits=64, lanes=16),
                       shape=[1])
pd = ovstage.PathDictionary(stage)
with pd:
    path_list = pd.create_path_list_from_strings([camera_path])
    try:
        query = stage.query_from_path_list(path_list)
        try:
            op = stage.write_attribute(query, "omni:xform", ordinal, tensor,
                                       is_array=False,
                                       semantic=AttributeSemantic.MATRIX,
                                       prim_mode=PrimMode.UPSERT)
            op.wait()
            if not op.ok:
                raise RuntimeError(op.error_message())
        finally:
            stage.release_query(query)
    finally:
        pd.destroy_path_list(path_list)
del data                                    # backing buffer alive through the write
stage.advance_write_floor(ordinal).wait()   # then render @ordinal
```

**Any prim's `omni:xform` uses this exact pattern** — so the emitter *seed-sphere
marker* is a persistent **unit** icosphere authored once at the origin and moved
live by writing its `omni:xform` at a new ordinal (a scale-by-radius +
translate-to-seed matrix; `_marker_xform` / `_sync_emitter_marker`). This is
validated in the selected application — the marker never gets re-authored on a drag; only its transform
column changes. Snapshot the prim's authored/world transform before the first
live column (a fresh UPSERT column initializes to identity — a prim rendered
before the real matrix lands jumps to the origin; see `prim-transform-safety`).

Colormap and scalar-range tweaks are *conceptually* data-plane (write the LUT or
`displayColor` primvar values on the standing material/geometry prim). The
generated app should start with the conservative rebuild path when the selected data plane
cannot update its colormap representation, so a colormap change currently takes the rebuild path
with `refit=False` (rebuild, but preserve the camera — a colormap change must not
reframe the view). Migrating colormap to a data-plane attribute write is a
straightforward, high-value evolution.

### Structural rebuilds (justified when topology changes)

A representation, field, iso-value, slice-position, dataset, or preset change
produces geometry with **different topology** (new vertex/curve counts, new
connectivity). That is a genuine re-author, not a transform tweak. Recompute the
operator (`representation` / `emitter_streamlines`) and republish the composed USDA:

```python
# Application-owned rebuild function
surface = self._compute_surface()                 # operator recompute (new topology)
if refit or self._build is None:                  # refit only on a NEW dataset
    self.camera.fit(surface.bbox_min, surface.bbox_max)
pose  = self.camera.get_camera_xform()            # thread the LIVE pose in (jump fix)
build = scene.build_stage(surface, w, h, colormap, camera_xform=pose)
#   → population.open_usd_from_string(stage, usda, ordinal=1, ...); advance_write_floor(1)
```

**Persistent stage + incremental geometry swap (proven).** Republishing the whole
USDA forces a renderer detach/attach and recreates the camera prim (and its live
`omni:xform`) every edit — which is exactly what reopens the jump risk and the
identity-UPSERT hazard. Instead, create one **persistent `Stage`** with the
"chrome" (camera, render product/settings, lights, material, and an empty
`/Session/Geometry` container that carries the material binding), attach the
renderer **once**, and swap only the geometry subtree via
`population.add_usd_reference_from_string` / `population.remove_usd` /
`population.apply_usd_changes(stage, ordinal=N)`, leaving the camera, lights, and
material prims (and their live `omni:xform`) untouched. This keeps live camera state stable and avoids a renderer detach/attach for every edit. Use an
incremental geometry-reference swap for any change that alters the prim class or topology
(representation swap, mesh/points field/iso/slice edits), after validating its behavior against
the current OVStage APIs.

## Live Geometry As Array Writes (Proven — The Realtime Emitter Path)

The highest-value pattern for a CFD viewer — treating **live geometry itself as a
data-plane array write** — is appropriate when supported by the installed OVStage release. A dragged control updates the
streamline arrays *in place on the same prim* with no USDA text re-author, reparse, or
population reference swap. Validate this path with the generated app before relying on it.

Keep the streamlines as a **persistent `UsdGeomBasisCurves`** prim, authored once.
On each emitter move, recompute the operator and overwrite its arrays at a new
ordinal — `points` (POINT), `curveVertexCounts` (int), and
`primvars:displayColor` (COLOR). The **curve and point count may differ between
writes: variable-length array writes are supported.**

```python
# Application pattern: in-place writes to the same persistent BasisCurves.
# Query is built exactly like the camera commit:
#   PathDictionary.create_path_list_from_strings → query_from_path_list.
def _dp_write_array(stage, path, name, arr, *, lanes, code, bits, semantic, ordinal):
    data   = np.ascontiguousarray(arr)                 # keep alive through op.wait()
    tensor = make_dltensor(data, dtype=DLDataType(code=code, bits=bits, lanes=lanes),
                           shape=[int(data.shape[0])])
    pd = ovstage.PathDictionary(stage)
    with pd:
        path_list = pd.create_path_list_from_strings([path])
        try:
            query = stage.query_from_path_list(path_list)
            try:
                op = stage.write_attribute(query, name, ordinal, tensor,
                                           is_array=True, semantic=semantic,
                                           prim_mode=PrimMode.UPSERT)
                op.wait()
                if not op.ok:
                    raise RuntimeError(op.error_message())
            finally:
                stage.release_query(query)
        finally:
            pd.destroy_path_list(path_list)
    del data

# Per emitter move (variable length — point/curve count may change each write):
_dp_write_array(stage, curves_path, "points", pts_f32,   # (P, 3) float32
                lanes=3, code=DLDataTypeCode.kDLFloat, bits=32,
                semantic=AttributeSemantic.POINT, ordinal=N)
_dp_write_array(stage, curves_path, "curveVertexCounts", cvc_i32,
                lanes=1, code=DLDataTypeCode.kDLInt,   bits=32,
                semantic=AttributeSemantic.NONE,  ordinal=N)
_dp_write_array(stage, curves_path, "primvars:displayColor", colors_f32,  # (P, 3)
                lanes=3, code=DLDataTypeCode.kDLFloat, bits=32,
                semantic=AttributeSemantic.COLOR, ordinal=N)
stage.advance_write_floor(N).wait()   # then render @N
```

Mechanics (query lifetime, tensor buffer kept alive until `op.wait()`, handle
release) are identical to the camera commit — see `references/ovstage-data-plane/README.md`
and `ovstage-render-and-camera.md`; don't duplicate them, reuse the same helper.

> **CAVEAT — a per-vertex primvar whose length must track a changing point count
> goes STALE if you don't also rewrite it.** If the point count changes on a drag
> and you leave, say, a per-vertex `widths` array at its old length, RTX rejects
> the prim (*"widths primvar is not valid / not enough data"*) and the curves
> vanish. Two ways to stay valid: rewrite *every* per-vertex array on each move,
> **or** author length-sensitive attributes as **`constant`** so they never need
> resizing. The reference authors `widths` as `float[] widths = [ w ]
> (interpolation = "constant")` — one value for the whole prim — so a changing
> point count never invalidates it. `displayColor` is fine per-vertex here
> *because* the data-plane write updates it (to the new length) on every move.
> See `usd-authoring-and-materials.md` for the widths/interpolation rules.

### Interactive LOD — cheap during the drag, full quality when it settles

To keep a burst of emitter edits smooth, recompute the operator **cheaply** while
the user is actively dragging and re-run at full quality once the interaction
settles (~200 ms idle). "Cheap" means fewer seeds and fewer integration steps, and
skipping the field probe entirely — **color by the free integration-time field**
instead of probing the velocity field per vertex. Choose the seed count, integration steps,
and settle timeout for the generated app's responsiveness target. The cheap and full paths
both flow through the same in-place array write above; only the point count
differs, which the variable-length write handles for free.

### Two proven speed tiers, and when to use each

| Tier | Mechanism | Use when | Cost |
| --- | --- | --- | --- |
| **Data-plane array write** (fastest) | overwrite `points`/`curveVertexCounts`/`displayColor` on the **same** persistent prim | same-prim geometry update — the emitter drag | Measure in the generated app |
| **Incremental geometry swap** | `add_usd_reference_from_string` / `remove_usd` / `apply_usd_changes` on the persistent stage | structural / topology / prim-class change — representation swap, mesh-rep field/iso/slice | Measure against a full rebuild in the generated app |

Measure emitter latency from command drain through a geometry frame in the generated app.
Keep a data-plane capability flag and fall back to re-authoring only the small `BasisCurves`
layer (not the whole scene) if a write is rejected.

### Proven vs. still-frontier — be honest

| Capability | Status |
| --- | --- |
| Camera / any-prim `omni:xform` matrix write (16-lane `float64`, MATRIX, UPSERT) | Use the standard matrix-write helper |
| Emitter-marker `omni:xform` write (persistent unit sphere, scale+translate) | Use the same persistent-prim matrix-write pattern |
| Operator recompute → geometry read-back (surface/iso/slice/streamlines/emitter) | Implement with the selected CAE operator library |
| Persistent stage + incremental geometry-reference swap | Validate against the current OVStage publication APIs |
| Live curves as variable-length `is_array=True` `points`/`curveVertexCounts`/`displayColor` writes | Validate variable-length writes against the installed OVStage release |
| Structural rebuild via `open_usd_from_string` republication | Baseline-compatible fallback |
| Colormap/scalar as a data-plane attribute (no rebuild) | Use structural rebuild until the selected data plane supports the update |
| Same array-write path for **mesh/points** reps (field/iso/slice), not just curves | Frontier — curves proven; mesh/points still take the ref-swap |

When generating, reach for the proven cheap path first: `omni:xform` for
transforms, the in-place array write for same-prim geometry, the incremental
reference swap for topology/prim-class changes, and a full rebuild only for a new
dataset. Propose only the last two rows above as explicit next steps, not shipped
facts.

## Ordering Rules (Non-Negotiable)

- **Never render above the write floor.** For every render-visible change:
  write @ N → `advance_write_floor(N).wait()` → `renderer.step(..., ordinal=N)`.
- **One serialized owner.** All population, all attribute writes, and all `step`
  calls run on the same lane. Controls enqueue; they never write directly.
- **Never `step` while (re)building.** The loop drains commands (including
  rebuilds) *before* it steps, so a step can never observe a half-built stage.
  Detach the prior stage before swapping so `step` never runs against a stale one.
- **Coalesce dragged controls to one rebuild/tick.** Drain the queue **last-wins
  per kind** so a slider drag that enqueues dozens of values triggers at most one
  operator recompute and one publication per tick:

  ```python
  latest = {}
  while True:
      try:
          kind, payload = self._commands.get_nowait()
      except queue.Empty:
          break
      latest[kind] = payload          # last value per kind wins
  # apply latest → set rebuild flags → recompute + publish ONCE
  ```

- **Avoid the camera jump on every rebuild.** A rebuild authors ordinal 1 and the
  next tick commits `omni:xform` at ordinal 2; if those poses differ, the one
  frame between them snaps. Fit *before* authoring and thread the **live** pose
  into the USDA build so `ordinal 1 == ordinal 2`. Use `refit=True` only on a new
  dataset; pass the current pose (`refit=False`) on colormap/field/iso/slice edits
  so the view is preserved. Full analysis in `ovstage-render-and-camera.md`.

## See Also

- `ovstage-render-and-camera.md` — full camera data-plane block, frame
  extraction, first-cold-step compile, and the camera-jump fix.
- `ovui-controls.md` — the ovui shell and the callback→queue control surface.
- `data-and-operators.md` — warp-simdata operators and geometry read-back.
- `usd-authoring-and-materials.md` — `widths`/interpolation rules and the
  constant-vs-vertex primvar caveat for data-plane array writes.
- `references/ovstage-runtime/README.md` — ordinals, write floor, single owner.
- `references/ovstage-data-plane/README.md` — queries, `omni:xform`, tensor lifetime.
- `references/ovstage-population/README.md` — publication and reference-swap APIs.
- `references/prim-transform-safety/README.md` — snapshot before a live
  `omni:xform`; never fall back to identity.
