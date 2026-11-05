<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Transient / Temporal Playback (time-varying CAE)

## Triggers

Use this reference to play back a **time-varying (transient) CAE dataset** — a timeline
with play / scrub, per-frame field interpolation, and a locked-or-auto colormap domain.
Trigger terms: transient, temporal, time-varying, timeline, playback, scrub, play/pause,
`Space` to play, per-frame update, field interpolation, "Enable Field Interpolation",
keyframe lerp, deforming mesh over time, bumper-beam crash, 51 steps, timeSamples,
`update_from_usd_time`, domain lock across timesteps, "animate the strain field".

Scope: driving a persistent mesh through a series of frames. Reading the native
`.case`/`.cgns` series into per-step arrays is in
[cae-data-ingestion.md](cae-data-ingestion.md); the data-plane write mechanics are in
[driving-cae-viz-via-ovstage.md](driving-cae-viz-via-ovstage.md). This doc is the
**playback engine** that sits between them.

Implement two small application-owned components: a `TransientClip` containing shared
topology plus per-frame arrays, and a `PlaybackController` that maps play/pause, rate, and
scrub input to a frame pair and interpolation factor. Validate with a synthetic two-frame
fixture whose midpoint values are known, including both locked and auto colormap domains.

## Two engines — pick by how the frames were produced

| Engine | When to use | Publication |
|---|---|---|
| **Per-frame data-plane ARRAY writes** (recommended, live) | Live server driving frames on demand; stable topology; frames produced at runtime | Changed arrays at one committed ordinal |
| **USD `timeSamples`** (`population.update_from_usd_time`) | All frames baked into a USDA up front; exporting a self-contained clip; not a live server | Whole clip baked into the stage before playback |

The examples below use fixed topology and vary points + color per frame. The data-plane path is
the live engine; the timeSamples path is for **baked-clip export**, not the interactive loop.

## Per-frame array writes for a fixed-topology clip

For this example, verify that connectivity and node ordering stay fixed across
the clip. Author a persistent `UsdGeomMesh` once, then update `points` and
`primvars:displayColor` for each sample. Its unchanged `faceVertexCounts` and
`faceVertexIndices` remain cached.

```
author persistent UsdGeomMesh ONCE:
    faceVertexCounts / faceVertexIndices   (topology, from frame 0)
    points          (frame 0)              (interpolation implicit)
    primvars:displayColor (frame 0)        (vertex OR uniform — see below)

per timeline tick at fractional t:
    prepare points and colors for the requested sample
    owner writes both arrays at ordinal N and waits for completion
    advance_write_floor(N).wait()  ->  renderer.step(ordinal=N)
```

Use the typed-write and tensor-lifetime mechanics in
`references/ovstage-data-plane/README.md`. The publication choice for meshes,
curves and changing topology is defined in
[driving-cae-viz-via-ovstage.md](driving-cae-viz-via-ovstage.md).

### Stable topology and changing geometry

Cache face counts/indices when connectivity and node ordering are unchanged.
Equal point counts alone do not establish that condition. For changing mesh
topology, update face counts/indices with points and every affected primvar on
the existing Mesh, in one committed publication. Replace structure when prim
type or composition changes; follow
[control-to-publication routing](driving-cae-viz-via-ovstage.md).

Interpolate samples only when topology, ordering and field association match.
Otherwise select a complete sample instead of blending unrelated vertices.

Reusing an OVStage display prim does not establish that scientific input caches
remain valid across samples. Before rerunning operators on changed solver data,
follow the selected release's ownership/cache contract, reached through
`references/dependencies/README.md` → Warp SimData → datasets and fields.

## Field interpolation = numpy lerp between keyframes

Field interpolation is a plain linear
interpolation of **both points and field** between the two bracketing keyframes at a
fractional `t`, then recolor. No GPU, no operator recompute:

```python
i0 = int(np.floor(t)); i1 = min(i0 + 1, n_frames - 1); a = t - i0
pts = (1 - a) * load_points(i0) + a * load_points(i1)   # deformed geometry
fld = (1 - a) * load_field(i0)  + a * load_field(i1)    # interpolated scalar
colors = apply_lut(fld, vmin, vmax, lut)                # recolor
```

With interpolation **off**, snap to `round(t)` and use that frame's arrays verbatim. This is
a pure CPU coloring/geometry choice made just before the data-plane write.

## Domain LOCK is a pure CPU coloring choice (not a stage-wide setting)

Domain locking freezes the colormap range so it does not auto-rescale per frame.
In this engine it is simply **which (vmin, vmax) you pass to the LUT** before the write:

- **Locked:** use the clip's **global** field range (min/max across all frames). The color of
  a given strain value is stable through the whole animation — the correct default for
  reading how a field grows.
- **Auto:** use the **current frame's** min/max. Each frame fills the colormap but values are
  not comparable across time.

```python
if lock_domain:
    vmin, vmax = clip.field_global_range()     # computed once, cached
else:
    vmin, vmax = float(np.nanmin(fld)), float(np.nanmax(fld))
```

Compute the global range once and cache it (a full sweep of the clip). It is not a per-attribute USD metadata flag; in a bake-per-vertex-color viewer it is just the range argument.

## Architecture: `TransientClip` + `PlaybackController`

- **`TransientClip`** — random access over a stable topology: `n_frames`, `node_count`,
  the authored-once `face_counts` / `face_indices`, lazy `load_points(i)` / `load_field(i)`
  (so a long clip is never fully resident), the `field_kind` (`"node"` → per-vertex color,
  `"element"` → per-face color), and `field_global_range()` for the lock. Build one from the
  native readers in [cae-data-ingestion.md](cae-data-ingestion.md), loading geometry
  and fields at the same source time code and verifying topology before reuse.
- **`PlaybackController`** — turns a fractional `t` into `(points, displayColor, (vmin,vmax))`
  ready for the write, applying interpolation and the locked-or-auto domain. Config is cheap:
  `set_colormap` / `set_lock` / `set_interpolate`.

Playback state and the render loop:

- New command kinds on the serialized owner: `timeline{t}` (scrub), `play{playing,fps,speed}`,
  `domain_lock{locked}`, `interpolate{enabled}` — all enqueued like any other control (see
  [driving-cae-viz-via-ovstage.md](driving-cae-viz-via-ovstage.md)).
- While playing, advance `t += dt * fps * speed` each tick, looped; scrub sets `t` directly.
- Everything runs under **one ordinal / one `advance_write_floor`** on the render thread, so
  transient playback composes with an emitter or camera move in the same tick.

## The USD `timeSamples` alternative (baked clips only)

For a self-contained exported clip, author every frame as USD `timeSamples` and drive it with
`population.update_from_usd_time(stage, ordinal=N, time=seconds)` — `time` is in
**seconds** (via the stage `timeCodesPerSecond`), then `advance_write_floor(N)` + step. Use
this only when all frames are baked up front (clip export), **not** for the live server —
baking a long transient into a USDA is wasteful compared to streaming two arrays per tick.

> **ovstage inline-USDA parser is stricter than pxr:** no semicolon one-line blocks — **one
> attribute per line**. A `timeSamples` block that `pxr` accepts on one line will fail to
> parse through `open_usd_from_string`. Author each sample on its own line.

## Gotchas

- **Cache unchanged topology.** Publish changed connectivity and all dependent
  arrays together when a sample changes mesh topology; count changes alone do
  not require a new prim.
- **`field_kind` picks the primvar interpolation** — `element` fields color per **face**
  (`uniform`, length `F`), node fields per **vertex** (`vertex`, length `N`). Author the
  matching interpolation once; the per-frame color array length must match it.
- **Lock = global range; auto = per-frame range.** Default to lock so a value's color is
  comparable across time; compute the global range once and cache it.
- **Don't bake a live transient into timeSamples.** Data-plane array writes are
  the live engine; timeSamples is for baked-clip export, and its inline USDA must be one
  attribute per line.

## See also

- [cae-data-ingestion.md](cae-data-ingestion.md) — reading the `.case`/`.cgns` series into
  the per-step arrays a `TransientClip` loads.
- [driving-cae-viz-via-ovstage.md](driving-cae-viz-via-ovstage.md) — the data-plane
  array-publication flow, dependent primvar lengths, and the
  serialized command loop.
- [usd-authoring-and-materials.md](usd-authoring-and-materials.md) — primvar interpolation
  (`vertex` vs `uniform`) and colormap authoring.
- [streaming-cae-viewer.md](streaming-cae-viewer.md) — the timeline UI + `sceneState`
  timeline contract that surfaces this to the browser.
- `references/ovstage-population/README.md` — `update_from_usd_time` for the baked-clip path.
