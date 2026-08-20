<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CAE/CFD Visualization

## Triggers

Use this reference family when the app must **visualize CAE/CFD simulation data** — CGNS,
OpenFOAM, VTK/VTU, EnSight, FLASH, CGNS-coded `.npz`, raw numpy arrays, in-memory solver
output, or a streamed field — inside an Omniverse Realtime Viewer.

**Cold, from-scratch phrasings this family answers:** "help me visualize some CAE/CFD data in realtime", "build a realtime CAE/CFD viewer",
"visualize my simulation / solver data live", "show my CFD results interactively", "render a
CFD field in a viewer", "make a live viewer for my flow/thermal/structural data".

Trigger terms: `warp-simdata`, `warp_simdata`, `cae-openusd-plugins`, CFD, CAE, simulation data, solver output, iso-surface / isosurface, planar slice,
streamlines, external surface, point cloud, glyphs, volume-like CFD visualization, colormap on a field,
Mach/Pressure/Temperature/Velocity field, "color the mesh by
a field", SIDS/CGNS mesh.

Not this family for: plain USD asset viewing (use `usd-viewer-app`), streaming transport
(`streaming-*`), or generic camera/selection (`camera-controls`, `object-selection`) —
those compose *under* this family.

## This is a LAYER, not a standalone app — start with the delivery path

CAE/CFD visualization **bolts onto** the standard viewer recipes; it is not its own app
silo. A from-scratch request follows four steps, in order:

1. **Pick the viewer type (delivery path).** Streaming to a browser, or a local desktop
   window? Decide with `references/streaming-vs-local/README.md`. This choice is made
   *before* any CAE work.
2. **Build the base viewer with the existing recipe.** For a local window, follow
   `references/ovui-local-viewer-recipe/README.md` (+ `references/local-viewer/README.md`).
   For a browser stream, follow `references/streaming-viewer-recipe/README.md` (+
   `references/streaming-server/README.md`, `references/streaming-client/README.md`,
   `references/streaming-messages/README.md`). You get a working RTX viewer with a camera and
   a render loop **before** you touch CAE data.
3. **Add the CAE data + scene layer on top** — ingestion → operators → USD authoring →
   OVStage scene, from the docs in this family (read order below).
4. **Wire the controls** — map each UI control to the correct scene write (the wiring pattern
   below).

Everything in this family assumes that base viewer already exists and slots the CAE layer
into its single serialized render loop. Cross-reference the generic skills (camera,
streaming transport, ovstage population, controls) — this family does not re-document them.

## The crux: data/scene-edit → UI wiring pattern

The one pattern to internalize for a *live* CAE viewer: **a UI control (slider, dropdown,
message) maps either to a cheap data-plane array/attribute write OR to a structural rebuild —
and you must decide which.** A camera move, an emitter drag, a timeline scrub, or a colormap
tweak is a cheap in-place write of `points` / `displayColor` / `omni:xform` / timeline on an
existing prim; a representation swap or a new dataset changes topology and needs a rebuild.
The full decision table and the end-to-end control → message → server op → OVStage write →
render path live in **[driving-cae-viz-via-ovstage.md](driving-cae-viz-via-ovstage.md)** —
read it before wiring any interactive control. It is the generalized version of the same
"control → correct write" pattern any live-editing viewer needs; the generic viewer recipes
point here for the CAE-style live data/scene editing case.

## What this builds

An RTX-rendered viewer — **local `ovui` window or browser stream** — that loads CAE solver
data, runs **GPU visualization operators** on it, and presents the result with interactive
controls. The operator set covers external surfaces, iso-surfaces, slices, streamlines, and points, each colored by any solution field. Any data source that yields node coordinates, connectivity, and fields feeds the same operators. It reuses this repo's `ovrtx`/`ovstage`/`ovui`(+`ovstream`) stack — the CAE layer
slots into a base viewer built from the standard recipes.

This family is self-contained: build the CAE scene layer in the generated app from the
contracts below. Keep ingestion, operator execution, USD authoring, and the serialized
render loop as application modules; do not depend on an unshipped sample app or a local
script. For the browser delivery composition, see
[streaming-cae-viewer.md](streaming-cae-viewer.md).

## Pipeline

```
CAE file (CGNS/OpenFOAM/VTK/EnSight)          .npz (CGNS-coded arrays)
        │  cae-openusd-plugins                        │
        │  (read-only SdfFileFormat + OmniSci schemas)│  warp_simdata SIDS
        └──────────────► USD scientific prims ◄───────┘  create_dataset
                                   │
                    warp-simdata GPU operators
     element_faces │ iso_surface │ slice │ streamlines │ points
                                   │  (surface_mesh / curves + field)
                                   ▼
        author UsdGeomMesh / UsdGeomPoints / UsdGeomBasisCurves
        + colormap LUT → displayColor primvar + UsdPreviewSurface + lights
                                   │
                     OVStage (open_usd_from_string, write floor)
                                   │  attach_ovstage
                                   ▼
                     ovrtx  ── LdrColor RGBA8 ──►  ovui window
             (data-plane omni:xform camera commit; controls: rep/field/iso/slice/colormap)
```

Key constraint (inherited from the repo): all USD is managed through **OVStage**, never
authored directly against `ovrtx`; one serialized loop owns every OVStage write and every
`renderer.step`.

## Read in this order

1. **[data-and-operators.md](data-and-operators.md)** — get CAE data into a
   `warp_simdata.Dataset` (SIDS from `.npz`/CGNS arrays, field attach) and run the
   operators; read back geometry + fields.
2. **[cae-data-ingestion.md](cae-data-ingestion.md)** — read **native** CAE files
   library-only: EnSight Gold `.case`/`.geo` (C-Binary parser) and CGNS `.cgns`
   (`warp_simdata.io.cgns` + the h5py transient workaround), NODE vs ELEMENT association.
   Prefer the library-direct readers — `cae-openusd-plugins` may be absent.
3. **[usd-authoring-and-materials.md](usd-authoring-and-materials.md)** — turn operator
   output into a colored, lit USD prim (Mesh/Points/BasisCurves + colormap material). A
   path tracer renders unlit geometry **black** — author lights. Covers MDL beauty materials
   + HDRI lighting for full fidelity.
4. **[coordinate-systems-and-up-axis.md](coordinate-systems-and-up-axis.md)** — foundational:
   CAE solver data is often **Z-up** while the viewer scene is **Y-up**; rotate the display
   geometry to the scene up while keeping physics native. Also define an explicit application unit policy and expose a user-facing Scale/Up-Axis/Offset transform control.
5. **[ovstage-render-and-camera.md](ovstage-render-and-camera.md)** — publish to OVStage,
   render with `ovrtx`, and commit camera moves via the `omni:xform` data plane; write-floor
   ordering and the camera-jump fix.
6. **[ovui-controls.md](ovui-controls.md)** — the `ovui` window shell and the
   representation / field / iso / slice / colormap / sample-preset controls, driven through
   one serialized command loop.
7. **[glyphs.md](glyphs.md)** — render a vector field as oriented arrows
   (`UsdGeomPointInstancer` + cone prototype, orient/scale/color by the field). Pure USD,
   no new renderer path.
8. **[volume-rendering.md](volume-rendering.md)** — current volume limitations and the
   supported fallback for volume-like CAE visualization.
### Interaction, streaming, and recreating the blueprint

10. **[driving-cae-viz-via-ovstage.md](driving-cae-viz-via-ovstage.md)** — the core pattern:
    turning a control (slider/message) into the **correct OVStage writes** — data-plane array
    writes (proven) vs. structural rebuild, write-floor ordering, the realtime emitter.
11. **[temporal-playback.md](temporal-playback.md)** — transient/time-varying playback: a
    persistent mesh with per-frame data-plane `points`/`displayColor` writes (~1 ms/frame),
    numpy-lerp field interpolation, and locked-vs-auto colormap domain; timeSamples alternative.
12. **[emitter-and-seed-sources.md](emitter-and-seed-sources.md)** — the movable emitter sphere
    → live streamlines , and the critical point-cloud-needs-a-
    real-volume gotcha (voxelizing a bare point cloud yields a zero field; bin to a hex lattice).
13. **[streaming-cae-viewer.md](streaming-cae-viewer.md)** — the server (ovrtx→ovstream) + React
    version; **composes** the `streaming-*` skills and documents the CAE-specific runtime,
    the ovstream↔ov-web-rtc message framing, the example-preset "uber-UI" pattern, and
    control→server-op flow.
14. **[rtwt-blueprint-recreation.md](rtwt-blueprint-recreation.md)** — recreate NVIDIA's Digital
    Twins for Fluid Simulation (RTWT) blueprint from pre-baked data, with the
    component→our-library map and the AI-surrogate car recipe.

## Depends on

- Acquisition of `ovrtx`/`ovstage`/`ovui` and the two CAE libraries —
  **the `warp-simdata` GPU operator library** and **the `cae-openusd-plugins` USD
  reader/schema library** — is centralized in `references/dependencies/README.md`
  (see its **CAE/CFD Visualization Libraries** section). Both are **public on
  GitHub** (`NVIDIA/warp-simdata`, `NVIDIA-Omniverse/cae-openusd-plugins`); the
  dependencies doc is the single source for their URLs. Family docs refer to them
  by role, never by inline URL.
- Rendering/stage substrate: `references/ovstage-runtime/README.md`,
  `references/ovstage-data-plane/README.md`, `references/ovrtx-rendering/README.md`,
  `references/stage-loading/README.md`, `references/prim-transform-safety/README.md`.
- Local window shell: `references/ovui-local-viewer-recipe/README.md`,
  `references/local-viewer/README.md`, `references/camera-controls/README.md`.
