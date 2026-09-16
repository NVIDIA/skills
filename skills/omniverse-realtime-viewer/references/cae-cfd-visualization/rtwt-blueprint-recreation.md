<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Recreating the RTWT "Digital Twins for Fluid Simulation" Blueprint

## Triggers

Use this reference to rebuild NVIDIA's **"Digital Twins for Fluid Simulation"**
blueprint — a.k.a. **RTWT / Real-Time Wind Tunnel**, the AI-surrogate automotive
external-aero "virtual wind tunnel" — built on this repo's `ovrtx` / `ovstage`
/ `ovstream` / `ovui` stack. Trigger terms: `digital-twins-for-fluid-simulation`,
RTWT, real-time wind tunnel, virtual wind tunnel, AeroNIM, DoMINO, DoMINO-Automotive-Aero,
PhysicsNeMo, Trame, `aero_auto_inference.npz`, `aero_auto_low.stl`, Unit-Sphere seeds,
"recreate the fluid-simulation blueprint", "car aero streamlines", movable emitter,
AI surrogate car.

**Critical scope: this doc uses PRE-BAKED data only — no NIM, no live inference.** We
consume the blueprint's shipped `.npz` point cloud + `.stl` car directly and render an
interactive streamable digital twin from it. Live DoMINO/AeroNIM inference is an explicit, later, out-of-scope gap (see Parity / Gaps).

## Scope and target architecture

RTWT is a containerized digital-twin reference application published at `github.com/NVIDIA-Omniverse-blueprints/digital-twins-for-fluid-simulation`. This skill recreates its pre-baked-data interaction model with the library stack directly:

- `warp-simdata` turns solver arrays into an in-memory dataset and derived geometry.
- The application authors `UsdGeomMesh`, `UsdGeomPoints`, or `UsdGeomBasisCurves` and publishes through OVStage.
- `ovrtx` renders frames; `ovstream` delivers them to a browser video element with DOM controls.
- A serialized controller separates visualization-only edits from dataset and configuration changes.

The supplied `.npz` represents the flow field. Do not add a surrogate inference service or direct-volume renderer to this recipe.

## The library-native recreation recipe (pre-baked, no inference)

The blueprint's "lite mode" already proves you can drive the whole twin from cached
fields. We take that one step further: treat `aero_auto_inference.npz` as the
flow field and render + stream supported geometry representations with our stack.

### Pipeline

1. **Load pre-baked data.** `np.load("data/aero_auto_inference.npz")` → `coordinates`
   (128k Z-up points) + `velocity` (vec3 per point). No Triton or volume ingest — the
   `.npz` is the surrogate output already.
2. **Point cloud → interpolable velocity volume.** A bare point cloud has no cells, so
   voxelizing it yields an all-zero field and streamlines never advect. Instead **bin
   the scattered velocities onto a regular hex lattice** (box-average per node +
   dilate-fill of empty nodes) and build a SIDS **HEXA_8** (`element_type=17`)
   `warp_simdata` `Dataset` with a node-associated `velocity` field. Trilinear
   interpolation now works and streamlines flow. This is the technique documented in
   [emitter-and-seed-sources.md](emitter-and-seed-sources.md); implement this as an application-owned binning function. Choose cell size from the data bounds
   and desired streamline fidelity, then validate that probes return nonzero values.
3. **Emitter → streamlines colored by speed.** Seed an upstream Y-Z emitter plane at low
   X (`SEED_X_FRAC = 0.06`, ahead of the car) and trace `Direction.FORWARD` with
   `streamlines.compute(ds, "velocity", seeds_ds, ...)`. **Probe** the velocity at each
   streamline vertex (`probe.compute`) to get a real speed gradient, then color per
   vertex through a colormap LUT (`coolwarm`: fast = red, slow = blue). Clamp the LUT
   domain to the 3rd–97th percentile so the near-body slowdown reads instead of the
   freestream (|v|≈30) collapsing to one color. In the interactive app this emitter
   plane/sphere becomes **movable** (`setStreamlineSeed{center,radius}` →
   `emitter_streamlines` backend), the direct analog of RTWT's Unit-Sphere seed source.
4. **STL car surface → lit `UsdGeomMesh`.** Parse `aero_auto_low.stl` (already Z-up),
   vertex-cluster decimate (`DECIMATE_CELL = 0.035`, snap-weld-drop-degenerate), and
   author a `UsdGeomMesh` with a metallic `UsdPreviewSurface`.
5. **Compose the viewer USDA.** Build a **Z-up multi-prim USDA text string** — camera,
   `RenderProduct` + `RenderVar LdrColor` + `RenderSettings`, dome + distant lights
   (a path tracer renders unlit geometry **black**), a `FieldMat` primvar-reader
   material for the streamlines, the `CarMat`, and the geometry scope (streamlines
   `BasisCurves` + car `Mesh` + faint context point cloud). Per-vertex speed color goes
   into `primvars:displayColor`. This is exactly the authoring contract in
   [usd-authoring-and-materials.md](usd-authoring-and-materials.md).
6. **Publish → render.** `ovstage.population.open_usd_from_string(stage, usda,
   ordinal=1, domains=PopulationDomain.RENDERING)` → `stage.advance_write_floor(1).wait()`
   → attach the `RendererRuntime` → `step_extract` the `LdrColor` frame. See
   [ovstage-render-and-camera.md](ovstage-render-and-camera.md).

The six steps above are sufficient for a headless hero-frame implementation. To reach the
interactive, streamable twin, compose the same application-owned CAE modules with the streaming
loop.

### From hero frame to interactive twin

The interactive recreation needs a server plus React frontend. Add an **`emitter`** representation and a
`setStreamlineSeed` message so the user drags the seed sphere and watches streamlines
recompute — the RTWT interaction, minus inference. To build it:

- **Server** (`ovrtx` + `ovstream`): build the CAE stage with the ingestion, operator, USDA
  authoring, and OVStage contracts in this family; warm up, convert each `LdrColor` RGBA8 frame
  to BGRA on CUDA, and `stream_video`. One
  render thread owns `step()`, all OVStage writes, and picking. See
  [streaming-cae-viewer.md](streaming-cae-viewer.md) and
  `references/streaming-server/README.md`.
- **Controller** (the `AppStateOperator` analog): React control →
  `{event_type,payload}` over the JSON data channel → server validates → **enqueues** a
  command on the render-loop queue → loop **drains with last-wins coalescing** (a dragged
  slider queues at most one rebuild) → OVStage write / geometry rebuild at a new ordinal
  → advance write floor → next `renderer.step(ordinal=N)` → stream the next frame. The
  `emitter` command rebuilds the seed point cloud and re-runs `streamlines.compute`.
  Split state into **viz-only** (colormap / slice / camera) vs. **would-be
  inference-triggering** (dataset / config) exactly as RTWT's controller does — here the
  latter just loads a different pre-baked `.npz` instead of calling Triton. See
  [driving-cae-viz-via-ovstage.md](driving-cae-viz-via-ovstage.md) and
  `references/streaming-messages/README.md`.
- **React frontend**: `@nvidia/ov-web-rtc` `AppStreamer.connect` over WebRTC, a `<video>`
  element, and DOM panels for representation / field / colormap / iso / slice / emitter.
  Mouse/keyboard forward natively (never as JSON); only app state travels the data
  channel. See [streaming-cae-viewer.md](streaming-cae-viewer.md) and
  `references/streaming-client/README.md`.

### What maps to which of our skills

Build the whole twin from skills alone in this order:

| Blueprint step | Skill reference |
|---|---|
| Ingest `.npz` → SIDS `Dataset`; hex-lattice velocity volume; seed sources | [data-and-operators.md](data-and-operators.md), [emitter-and-seed-sources.md](emitter-and-seed-sources.md) |
| Emitter streamlines + probe-color; STL car mesh; colormap LUT | [emitter-and-seed-sources.md](emitter-and-seed-sources.md), [usd-authoring-and-materials.md](usd-authoring-and-materials.md) |
| Compose viewer USDA (lights, materials, render product) | [usd-authoring-and-materials.md](usd-authoring-and-materials.md) |
| Publish to OVStage, render with `ovrtx`, live camera | [ovstage-render-and-camera.md](ovstage-render-and-camera.md), `references/ovstage-runtime/README.md`, `references/ovstage-data-plane/README.md`, `references/ovrtx-rendering/README.md` |
| Server-side streaming (RTX → WebRTC), BGRA convert, frame loop | [streaming-cae-viewer.md](streaming-cae-viewer.md), `references/streaming-server/README.md`, `references/streaming-lifecycle/README.md` |
| App-state fan-out controller (viz-only vs. inference-triggering) | [driving-cae-viz-via-ovstage.md](driving-cae-viz-via-ovstage.md), `references/streaming-messages/README.md` |
| React browser client + DOM controls | [streaming-cae-viewer.md](streaming-cae-viewer.md), `references/streaming-client/README.md`, `references/viewer-input-routing/README.md` |
| Local (non-streamed) window variant | [ovui-controls.md](ovui-controls.md), `references/ovui-local-viewer-recipe/README.md` |

Validate the generated app with a rendered hero frame, a connected streaming client,
and a controlled representation change. Keep those artifacts with the generated app;
do not cite or import unpublished local projects.

## Parity / gaps to reach the full blueprint

The recreation above delivers the **flagship look** (car surface + speed-colored
streamlines from a movable emitter, streamed to a browser) without a legacy application runtime or NIM. To reach
full RTWT parity, in ascending effort:

1. **Colormap-domain lock** — expose a fixed-vs-auto range toggle so the baked
   `displayColor` range does not rescale across frames/datasets (a locked range; RTWT locks the transfer-function domain so a config change is comparable). We already
   set a fixed percentile range at build; add the per-frame lock UI.
2. **Glyphs** — author a `UsdGeomPointInstancer` (arrow/sphere prototype, `positions`
   from cell points, `scales`/`orientations` from a vector field). Mirrors
   `CaeVizGlyphsAPI`; pure USD, no new render path.
3. **Biplane / triplane slices** — our `slice` is one plane per operator. Add a
   multi-plane widget (2–3 synchronized planes) to match RTWT's slice cross-sections.
4. **App-state fan-out controller (viz-only vs. inference-triggering)** — formalize the
   split the streaming controller: a controller that routes viz-only commands to a
   cheap OVStage rebuild and reserves a separate "recompute fields" path. With pre-baked
   data the inference path just swaps the cached `.npz`; the routing is the parity item.
5. **Temporal playback** — cache per-frame surfaces and lerp fields on time change for
   transient datasets (interpolation and timeline); RTWT plays flow
   evolution.
6. **NVIDIA IndeX volume rendering** — RTWT's dense volumetric look requires a
   capability this skill package does not support. Do not create `.nvdb`, `.vdb`,
   `.vti`, or `UsdVol` assets as a substitute. Offer slices, iso-surfaces, points,
   glyphs, or streamlines when they meet the visualization goal; otherwise state
   that true volume rendering requires a different documented runtime.
7. **Live inference (only if wanted later)** — reintroduce the **DoMINO / AeroNIM**
   surrogate as-is: a **Triton client** that sends car config + inlet velocity, receives
   a result in the surrogate's own data format, and a **result cache** keyed by the 16 config variants. This is
   reused as-is from the blueprint and is **not the viewer's job** — the viewer only
   consumes whatever field arrays the cache/inference hands it, identical to how it
   consumes the pre-baked `.npz` today. Wiring it turns "replay cached fields" back into
   "recompute on config change" with no change to the render/stream/controller path.

## See Also

- [README.md](README.md) — the CAE/CFD visualization family index and pipeline.
- [emitter-and-seed-sources.md](emitter-and-seed-sources.md) — hex-lattice velocity
  volume + movable-emitter streamline seeding.
- [streaming-cae-viewer.md](streaming-cae-viewer.md) — server-side RTX → `ovstream`
  WebRTC → React browser client.
- [driving-cae-viz-via-ovstage.md](driving-cae-viz-via-ovstage.md) — the app-state fan-out
  controller and OVStage data-plane commit path.
- Blueprint mirror:
  `github.com/NVIDIA-Omniverse-blueprints/digital-twins-for-fluid-simulation`.
