<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CAE/CFD Coordinate Systems And Up-Axis Handling

## Triggers

Use this reference when CAE/CFD solver data uses a **different up-axis than the
viewer scene** and geometry renders "sideways", lies on its side, wheels-up, or
the camera frames an empty region. Trigger terms: Z-up, Y-up, `upAxis`, up axis,
"model is rotated 90°", "car is on its side", "streamlines go vertical",
CGNS/EnSight/automotive Z-up data, "camera frames the wrong box", "shading is
wrong after rotating", coordinate system mismatch, handedness.

The concrete case this solves: CAE solvers commonly emit **Z-up** data
(CGNS/SIDS, EnSight, most automotive/aero pipelines), while USD/`ovrtx` viewer
scenes are typically authored **Y-up** (see the `upAxis = "Y"` header in
`usd-authoring-and-materials.md`). The mismatch renders the model rotated 90°
about X and mis-frames the camera.

Read `usd-authoring-and-materials.md` first for how geometry USDA is authored,
and `ovstage-render-and-camera.md` for the camera-fit / `omni:xform` data plane
this document rotates into. This document does **not** re-cover camera math or
USDA prim authoring — it covers only the up-axis transform applied to that output.

## The problem

A single scene has two frames that must agree:

- **Data frame** — the solver's native coordinates. CAE data is frequently
  **Z-up**: gravity along `-Z`, the ground plane at `Z = 0`, the model's roof at
  high `Z`. CGNS/SIDS meshes, EnSight cases, and automotive/aero datasets almost
  always land here.
- **Scene frame** — the composed viewer stage. The `ovrtx` chrome in this family
  authors `upAxis = "Y"`, and the camera, orbit controls, and lights all assume
  Y-up.

Author Z-up points into a Y-up stage unchanged and the model renders tipped onto
its side (rotated 90° about X); the camera-fit bbox then frames the wrong volume,
so the object can be off-screen or tiny. This is a data problem, not a render bug
— fix it in the authoring/orientation step, not by nudging the camera.

## Recommended approach: rotate DISPLAY geometry, keep PHYSICS native

Run every operator (streamlines, velocity, iso-surface, slice, external faces) in
the data's **native frame** — the solver, the GPU operators, and any field probe
all expect native coordinates, and rotating the volume before integration is both
wasteful and error-prone. Then rotate **only the authored display output** to the
scene up-axis, right before it becomes a USD prim.

The Z-up → Y-up map is a single **−90° rotation about +X** (det = +1, a proper
rotation), so it maps points AND direction vectors (normals, displayed velocity
glyph directions) with the *same* call:

```
(x, y, z)  ->  (x, z, -y)
```

```python
def _zup_to_yup(arr):
    """Rotate Z-up coords/vectors to Y-up via (x,y,z)->(x, z, -y). Returns (N,3) f64.
    Proper rotation (det=+1) about +X by -90 deg -> maps points AND direction
    vectors (e.g. mesh normals) with the same call."""
    a = np.asarray(arr, dtype=np.float64).reshape(-1, 3)
    out = np.empty_like(a)
    out[:, 0] = a[:, 0]
    out[:, 1] = a[:, 2]
    out[:, 2] = -a[:, 1]
    return out
```

Apply it **consistently to every authored piece**. Miss any one and the scene
desyncs. Apply the same map to:

- **Surface / streamline points** — rotate the computed points and refresh their bbox after
  the operator runs on the native volume.
- **Mesh points AND normals** — because this is a pure rotation, normals transform with the
  identical map (no inverse-transpose needed).
- **The emitter / seed marker** — rotate the emitter center before placing the marker, and
  rotate generated sphere vertices too when the marker is baked geometry.
- **The bbox used for camera-fit** — rotate the **8 corners** of the axis-aligned Z-up box and
  take the min/max of the rotated corners. Rotating only `min`/`max` is wrong.

```python
def _zup_to_yup_bbox(self, bmin, bmax):
    corners = np.array([[x, y, z]
                        for x in (bmin[0], bmax[0])
                        for y in (bmin[1], bmax[1])
                        for z in (bmin[2], bmax[2])], dtype=np.float64)
    r = self._zup_to_yup(corners)          # rotate all 8 corners
    return (r.min(axis=0), r.max(axis=0))  # re-bound the rotated box
```

Physics stays native: run the streamline operator on the Z-up volume and rotate only the
resulting display curve points. Keep the seed/emitter center in native coordinates for the
operator, and rotate its copy only when authoring the marker.

Genericize the map for other up-axis pairs by using the appropriate signed axis
permutation (the inverse Y-up → Z-up is `(x, y, z) -> (x, -z, y)`). Handedness
matters — use a proper rotation (det = +1) so normals and velocity vectors keep
their sense; a reflection would flip winding and invert lighting.

## Alternative: author the whole scene in the data's up-axis

Instead of rotating output, set the stage `upAxis` to match the data (e.g.
`upAxis = "Z"` in the composed chrome) and give the camera/orbit a matching up
vector. Geometry then needs no per-piece rotation.

Tradeoff: simpler geometry authoring, but **everything else must agree** — the
camera up vector, orbit/pan/dolly math, camera-fit, the HDRI dome orientation,
and any lights all have to be consistent with the chosen up-axis, or you have
merely moved the mismatch. This is also brittle when a single persistent scene
mixes datasets of **different** up-axes (e.g. a Z-up automotive case beside a
Y-up mesh): one stage has one `upAxis`, so at most one dataset gets the free
ride and the rest still need rotation. Prefer the rotate-output approach when the
viewer must load heterogeneous datasets into one stage; the fixed-`upAxis`
approach is fine for a single-dataset viewer with full control of the camera.

## Units: define an explicit application policy

Up-axis is not the only alignment convention that bites. The viewer must choose and document its stage unit policy; it must not silently infer or convert source units. Data authored in meters, millimeters, or inches is mapped according to the application's explicit transform. Consequences:

- **There is no hidden geometric offset** in the pipeline — geometry sits exactly where the
  source places it. A "wrong size" is a *units* question, not a bug to hunt in the authoring.
- **Temporal sampling is not geometric scale.** Keep frame spacing, playback speed, and frame interpolation separate from the model's geometric Scale control.

| Control | Effect |
|---|---|
| Playback speed / frame spacing | **Temporal** — timing only |
| Viewer **Transform > Scale** | **Geometric** — uniform size multiplier on the model |

Because nothing converts units for you, expose a **user-facing geometric Scale** so a user
whose data is in meters can align it against cm context geometry themselves.

## Make alignment explicit and user-controllable (Scale / Up-Axis / Offset)

Rather than baking a silent orientation and hoping it is right, expose alignment as an
**explicit control** applied via a single OVStage `omni:xform` **data-plane matrix write** on
a persistent **display-root Xform** (e.g. `/Session/Geometry`) that every geometry prim lives
under. One matrix scales, re-orients, and offsets the whole model at once — **no geometry
recompute, no stage rebuild** (the same fast path the camera and emitter marker use). This
makes alignment visible and controllable, never a silent surprise.

Build the matrix in USD **row-vector** convention (a point transforms as `p' = p * M`;
translation in row 3, matching the camera/marker writes):

```
M[:3,:3] = scale * R          # R = identity for upAxis="Y";
                              # R = [[1,0,0],[0,0,-1],[0,1,0]]  for upAxis="Z"  (= -90 about +X)
M[3,:3]  = offset             # world-space translation (viewer units)
```

Defaults `scale=1, upAxis="Y", offset=(0,0,0)` give the **identity matrix** — a pure no-op, so
nothing moves until the user acts. This is an override *on top of* each dataset's own
up-mapping (the rotate-output approach above), not a replacement, so existing renders are
unchanged. Write it once per tick after any geometry swap, only when a control changed, and
re-arm it after a scene-mode change re-authors the chrome. Keep this matrix and its
`omni:xform` write in the app's scene runtime. The `omni:xform` on the display root **survives
geometry reference swaps**, so
the alignment persists across representation edits.

End-to-end wiring mirrors any other control (a `setTransform {scale?, upAxis?, offset?}`
message → validate → enqueue → one data-plane matrix write → echo back in `sceneState`); see
[driving-cae-viz-via-ovstage.md](driving-cae-viz-via-ovstage.md) for the write classification
and [streaming-cae-viewer.md](streaming-cae-viewer.md) for the message/state contract. Give the
user an in-UI hint that the viewer does not guess units — e.g. *"Data uses its native units;
align it here with the declared scale."*

## Tie-ins

- **HDRI DomeLight must be oriented to the scene up-axis too.** A latlong
  environment dome wraps around the scene up; in a Y-up scene a Z-up-authored
  horizon needs a rotation or the reflected horizon sits vertical. Author `double3 xformOp:rotateXYZ = (-90, 0, 0)` on the `DomeLight` for this Z-up-to-Y-up convention. See `usd-authoring-and-materials.md` for the dome/light
  block this rotation applies to.
- **Camera-fit and orbit assume the scene up-axis.** The camera frames the
  (already-rotated) bbox and orbits about the scene up. See
  `ovstage-render-and-camera.md` for the camera-fit and `omni:xform` commit, and
  `references/camera-controls/README.md` for orbit/pan/dolly — do not add an
  ad-hoc camera tilt to compensate for un-rotated geometry; fix the geometry.

## Gotchas

- **Forgot to rotate the bbox** → geometry looks right but the camera frames the
  wrong (un-rotated) box: object off-screen, tiny, or mis-centered. Rotate the
  8 corners, then re-bound.
- **Rotated points but not normals** → wrong shading (facets lit from the wrong
  side, dark where they should be bright). Apply the same map to `normals` when
  authoring the prim.
- **Rotated geometry but not velocity vectors** → if you also display velocity as
  arrow/glyph directions, un-rotated vectors point the wrong way over correctly
  placed geometry. Direction vectors use the identical `_zup_to_yup` map.
- **Rotated the seed marker's position but not consistently with the streamlines**
  → the emitter sphere floats away from where the streamlines actually originate.
  Rotate the marker center with the same map used for the curve points.
- **Used a reflection instead of a rotation** (e.g. just swapping Y/Z without the
  sign) → det = −1 flips winding order and inverts lighting. Keep det = +1.
- **Rotated the volume before the operator** → slow and defeats native-frame
  physics; rotate only the authored output.

See also: `usd-authoring-and-materials.md`, `ovstage-render-and-camera.md`,
`emitter-and-seed-sources.md`, `references/camera-controls/README.md`,
`references/conventions.md`.
