# Telemetry contract

Version boundary: INSET refers to an older binding. Inspect the installed
`LocalFoveationMode` enum; current OVXR uses AUTO/DISABLED/WARPED. Never reuse old
integer values or treat an INSET→DISABLED compatibility alias as foveated proof.
Custom compositor exports are application APIs, not shipped OVXR entry points.


Telemetry must distinguish application intent, native acceptance, per-eye
geometry, GPU submission, and observed output.

## Required state

- Panel: LOCAL pose/revision/source, document and uploaded revisions, mode,
  active/world-locked state, injected frames/fallbacks, image domain and
  rejections.
- Per eye: pose, FOV tangents, full image extent, subimage offset/extent,
  viewport/scissor, projected bounds, clip counts, draw/scissor deltas.
- PQW: per-eye raw dimensions/parameters, frame/domain, submitted domain, and
  maximum forward/inverse residual.
- Depth: format, min/max depth, near/far, compare convention, representative
  written-depth residual, matching color/depth subimages.
- Pointer: chosen hand, focused/tracked/aim-valid state, pose space/time,
  ray availability, LOCAL origin/direction, hit, UV, hit distance, phase,
  captured/hovered node, press value, and ownership.
- Auxiliaries: accepted enabled/revision plus per-eye composed and clipped mesh
  parts, projected bounds, rejection reason, union-scissor inclusion, draw index
  span, depth-contract validity, and nullable depth-passing fragment evidence
  for controller tooltip and prim gizmo.
- Performance: texture uploads, unchanged uploads, primary copies/transfers,
  same-frame waits, per-eye draws, GPU/compose/queue timing.
- Identity: staged layer DLL hash, manifest, build source, and process start
  time for live runs.

## Invariants

- Active projection UI requires valid LOCAL conversion, source depth, and
  proven submission domain.
- Tooltip enabled requires focused controller ownership and a tracked valid
  LOCAL aim. Loss disables it within one accepted frame.
- Per-eye draw/scissor deltas match injected-frame deltas.
- Stable document revisions cause no texture upload.
- Primary full-eye copy/transfer and same-frame-wait counters remain zero.
- `INSET` is not a domain proof.

Log transitions, not frame-rate pose spam. Retain pointer hit fields in polled
status; transient response-only data is insufficient for headset diagnosis.
Treat missing measurement as unknown, never passed.

## Auxiliary proof ladder

Do not infer a later boundary from an earlier one:

1. `accepted`: application state and revision passed native validation.
2. `meshComposed`: finite post-clip/PQW vertices and indices were emitted.
3. `drawIncluded`: the submitted combined draw includes those indices and its
   effective scissor intersects the projected bounds.
4. `depthContractValid`: derived depth and compare convention are valid. This
   is not proof that a fragment passed.
5. `fragmentsVisible`: measured nonzero-coverage fragments survived scissor and
   depth and produced nonzero output; report `unknown` without measurement.
6. `outputObserved`: correlated primary, decoded, or headset pixels show it.

For bounded diagnosis, a per-surface fragment counter may be read only after an
existing frame-slot fence signals. It must not add a second production draw,
full-eye copy, same-frame wait, or permanent hot-path readback.

## Minimal live triage

For the first headset pass, require only: input epoch/frame/time; controller
owner/focus; separate grip and aim activity/valid flags; exact LOCAL reference
space and `xrLocateSpace` result/flags; LOCAL ray origin/direction; panel LOCAL
pose/revision; hit/rejection/UV/distance/phase/owner; tooltip enabled/revision
and per-eye composed/clipped counts; projection display time and PQW domain.
Expand to the full contract only when this subset crosses its boundary.
