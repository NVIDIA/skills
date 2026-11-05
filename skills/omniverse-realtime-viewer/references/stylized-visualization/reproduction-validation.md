# Stylized Reproduction Validation Profile

Use this profile before claiming that an app recreates the canonical presets or
runs them at real-time rates. It deliberately separates image fidelity,
temporal stability, renderer throughput, browser decoding, presentation, and
capture cadence.

## Contents

- [Canonical Fixture](#canonical-fixture)
- [Canonical Camera Evidence](#canonical-camera-evidence)
- [Fidelity Gates](#fidelity-gates)
- [Temporal Gates](#temporal-gates)
- [Real-Time Counters](#real-time-counters)
- [Evidence Packet](#evidence-packet)

## Canonical Fixture

Use two neutral, nonproprietary fixtures:

1. **Smooth fixture:** sphere, cylinder, torus segment, and concave blend in one
   view. It exposes banding, rim stability, normal orientation, and hatch swim.
2. **Hard-surface fixture:** a hollow tube connecting two broad circular
   flanges, a central bore, shoulders, and twelve evenly spaced through-holes
   per flange. Tessellate to roughly 5k-10k triangles and retain at least 2k
   explicit feature edges. It exposes hidden lines, repeated dash phase,
   silhouettes, occlusion, and topology density.

The fixture may be authored directly in USD or produced by any geometry system.
The visualizer must consume ordinary render meshes and optional generic feature
metadata; the validation asset must not require a modeling-kernel dependency.

Normalize the object to a world-space bounds diagonal of `4.0`, center its
bounds at the origin, and use Y-up. Target the origin from distance `6.0`. Use a
1920x1080 RenderProduct, focal length `24.0 mm`, horizontal aperture
`20.955 mm`, vertical aperture `20.955/(1920/1080) mm`, near `0.01`, far
`500000`, and the canonical studio rig from
`style-recipes.md`. The capture fixture has no authored lights; canonical
capture automatically enables only that viewer-owned rig for every preset.
Interactive viewers remain stage-lit unless the separate lighting control is
set to `studio`.

This repository does not ship a fixture USD or golden images. Therefore use
`recipe_conformant` for results that pass the numerical recipes and tolerances
below. Reserve `reference_image_identical` for a downstream project that pins a
public runtime and publishes its exact fixture, camera matrices, and goldens.

## Canonical Camera Evidence

Capture fixed views at azimuth/elevation degrees:

- front three-quarter: `(28, 20)`;
- rear three-quarter: `(208, 20)`;
- high technical: `(42, 38)`;
- grazing close-up: `(82, 8)` at distance `3.333333`.

Capture a ten-second monotonic orbit from azimuth `28` through `388` degrees,
elevation `20 + 3*sin(phase)`. Warm all lazy shaders/kernels and settle scene
updates before timing.

## Regular-Mesh X-Ray Extension

When an X-Ray result is claimed for ordinary `UsdGeomMesh` geometry, also
follow `mesh-xray.md`. Use a small support fixture with at least one broad
occluding slab plus curved and hard-surface objects; verify that the passive
support is included in the line adapter. Capture a fitted above-support orbit,
then a below/behind-occluder proof. The normal-shaped shell must keep surfaces
legible in the above-support frame; a pure-outline result does not satisfy this
extension.

Exercise all mesh X-Ray tiers: surface without `DepthSD`, enhanced with a
validated depth tensor, and the selected direct-GPU continuation or accurate
private-depth line profile. Confirm that the tier and its degradation reason are
visible to the reviewer. For any direct GPU continuation path, include one
long-edge/near-plane stress case and verify its bounded-work policy.

## Fidelity Gates

For each of the 14 preset IDs:

- requested and effective preset IDs match and fidelity is `canonical` for the
  main matrix; capture fallbacks separately and label them degraded;
- output is nonblank, correct resolution, and preserves alpha policy;
- compare source equations and SPG image output on identical same-step raw
  inputs; record explicit byte tolerance, rounding and alpha, separately from
  a replacement rasterizer comparison;
- sampled palette anchors are within `0.035` normalized RGB of the canonical
  values in `style-recipes.md` before video compression;
- feature/silhouette width, pattern, and opacity pass the tolerances in
  `line-compositor-delivery.md`;
- guide-driven presets pass with valid guides and remain live with each guide
  independently removed;
- X-Ray is labeled presentation X-ray and shows quiet rear features distinct
  from visible features and the solid silhouette.
- mesh X-Ray also shows current-camera support/object through-lines without
  disconnected guide artifacts or a stale overlay during the orbit.

Keep one lossless fixed-camera matrix as the numerical baseline. A compressed
turntable is review evidence, not the source of pixel-golden assertions.

## Temporal Gates

- Adding one source color code value to a flat tonal region changes coverage gradually;
  it must not toggle an entire hatch, rim, or halftone layer.
- With a fixed camera and lighting, no more than `0.25%` of surface pixels may
  change between identical post-process inputs.
- During orbit, deterministic screen patterns may move relative to the surface,
  but their spacing and orientation cannot jitter from frame to frame.
- Animated Blueprint dash velocity remains within `10%` of `1.6 px/s` regardless
  of missed frames.
- Hidden/visible classification cannot flicker on more than `2%` of evaluated
  feature samples outside a two-pixel occlusion-boundary band.

## Real-Time Counters

Measure at least these independent quantities over the warmed ten-second orbit:

| Counter | Measurement |
|---|---|
| Server render | successful `renderer.step()` outputs / wall time, plus p50/p95 frame time |
| Stream submission | frames handed to ovstream / wall time |
| Browser decode | delta of `totalVideoFrames`, plus `droppedVideoFrames` |
| Browser presentation | `requestVideoFrameCallback` callbacks / wall time |
| Line refresh | completed current-camera SPG geometry/composites / wall time; tag pose/revision rather than counting repeated output |
| Interaction | camera poses actually applied / wall time |
| Capture composite | unique source frames written to recorder / wall time |
| Recorder target | requested track/container rate; never substitute for unique frames |

Report one of these claim levels:

| Claim | Meaning |
|---|---|
| `recipe_conformant` | fixed frames satisfy recipe, guide, and line tolerances; no real-time claim |
| `realtime_full_style` | base image and a current-camera line overlay satisfy all floors below |
| `realtime_base_deferred_lines` | base stream satisfies its floors but line cadence or latency does not; report both |

For a 60 fps target, the `realtime_full_style` acceptance floor is:

- server output and stream submission: at least `57 fps` average after warmup;
- browser decode: at least `57 fps`, with dropped frames below `1%`;
- browser presentation and unique capture composite: at least `48 fps`;
- current-camera line refresh during orbit: at least `45 fps`, p95 applied-pose
  to matching-overlay latency below `50 ms`, and no line overlay more than two
  applied camera poses old;
- no resource growth after the first two seconds of the measured interval.

Missing the line floor while retaining the base floors qualifies only as
`realtime_base_deferred_lines`, never full-style real-time. These floors are not
a universal meaning of real-time. Publish the target GPU, driver, OVRTX/ovstream versions,
resolution, fixture triangle/edge counts, codec, and every measured counter.

## Evidence Packet

Retain:

- runtime-observed AOV keys, dtypes, shapes, and capability decisions;
- lossless fixed-camera matrix for all 14 styles;
- ten-second hard-surface turntable with controls and effective style visible;
- for a regular-mesh X-Ray claim, the mesh extension's above-support orbit and
  below/behind-occluder proof;
- close-ups for line alignment, dash continuity, and hidden-edge distinction;
- one fallback matrix with normal, ID, and SPG geometry capabilities independently
  disabled;
- frame-time and browser counter report;
- cleanup/reconnect result and warnings.

Do not claim faithful recreation when only a genre-appropriate result exists.
Do not claim 60 fps from encoder metadata or decoder totals when presentation,
interaction, or unique composited motion is materially lower.

## SPG stage and migration evidence

Validate the graph incrementally: image identity/formula, independent triangle
and edge projection, complete bin counts including the last row and overflow,
private depth, and visible/hidden stroke samples. A correctly shaped final
image can conceal a geometry node that never covered its full launch domain.
Interpret published buffer bytes by declared layout and test every vector or
matrix lane before relying on it.

Compare the old GL compositor on the same camera, geometry and raw base image.
Record maximum/mean errors and inspect localized pattern/occlusion differences;
exact image-equation parity does not imply exact analytic-versus-GL rasterization.
Record release-specific launch/upload/cache findings as observed evidence, not
as restrictions on every later SDK. Profile cold compile and warmed operation
separately. Static descriptors, cached geometry uploads and skipped GPU work
are independent claims.

Add regressions for welded tip faces that used to create partial spoke fans,
smooth duplicate-index seams, true sharp creases and passive occluders. An
explicit wireframe extension must include all accepted triangle edges, and a
no-fill variant must leave a uniform background with lines disabled. Test actual
RenderProduct resize separately from browser resize.

## Optional actual-BREP validation

When CAD provenance is part of the requested claim, generate real solids with
bores, bolt holes, rear contours and multiple parts through the chosen public
CAD provider. Record worker/package revision, parameters, tessellation settings
and closed-solid/volume checks. Keep its kernel isolated when interpreter or
OVRTX coexistence is unverified.

Distinguish direct tessellation of valid in-memory BREP handles from exact-USD
export/import. Round-trip fidelity needs its own topology, volume and geometry
checks with a declared healing policy. A successful SPG image or live browser
test cannot validate that file path. Report failures independently and identify
which geometry the renderer actually consumed.
