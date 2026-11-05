# Foveation and edge math

Read upstream foveation-and-display through [the SDK route](../dependencies/ovxr.md)
for current modes, feature negotiation and output domains. This reference owns
the custom compositor's domain acceptance and geometric validation.

## Capability boundary

For the complete rectilinear equations, clip planes and actual producer depth
encoding, read [projection and depth](projection-and-depth.md).

The rectilinear path is reconstructable from standard OpenXR projection/depth
headers. A warped path additionally needs the runtime's actual vendor PQW
extension definition and normative forward/inverse mapping, matched to its
binary. Core OpenXR headers or borrowed session handles do not provide it.
Obtain that contract from the supplied runtime/SDK owner through the
[dependency route](../dependencies/nvidia-runtime.md#ovxr); never guess a
structure type, memory layout, or warp from image dimensions.

Until the selected runtime supplies a public supported definition and mapping,
report warped primary-depth composition unavailable. An explicitly requested
and negotiated nonfoveated baseline can implement the panel/tool path described
here; do not silently disable requested foveation or count that baseline as
warped validation.


An advertised local mode does not establish the submitted projection's domain. Use only frame-correlated per-eye PQW metadata chained to the
actual projection:

- two unwarped views: direct rectilinear projection;
- two warped views: map the projected surface through the exact per-eye PQW
  transform using tessellated geometry;
- all PQW nodes absent: accept standard rectilinear only with explicit negotiated
  `local_foveation_mode=DISABLED`, `local_foveated_rendering_enabled=false`,
  `foveated_submission_enabled=false`, and valid positive per-eye extents;
- otherwise missing, mixed, stale, or inconsistent metadata: omit rather than guess.

Diagnose edge distortion in this order:

1. Record each eye's pose, four FOV angles/tangents, full image extent, subimage
   offset/extent, viewport, and scissor.
2. Verify LOCAL-to-eye corner projection and homogeneous clipping before PQW.
3. Verify PQW submitted/unwarped/warped dimensions and the selected domain for
   the same frame.
4. Measure forward and inverse PQW residuals at center, boundaries, corners,
   and the outer 15 percent of each eye.
5. Verify the controller/panel pose time and the projection display time. A
   static offset is a space error; motion-only divergence is usually timing.
6. Verify depth convention and CloudXR receives matching primary color/depth
   without an extra client unwarp or an assumed rectilinear input.

Do not average stereo FOVs, reuse left-eye data, normalize against full image
dimensions when rendering to a subimage, or discard subimage offsets. Do not
infer the domain from the local foveation mode or resolution ratio.

Required math tests include asymmetric FOVs, nonzero subimage offsets, reduced
and unequal per-eye extents, near/edge clipping, normal and reversed depth,
PQW identity, PQW round-trip, and maximum edge residual.

The explicit disabled/nonfoveated case may legitimately omit vendor PQW nodes
entirely. Report rectilinear/unwarped with zero PQW views observed; do not invent
identity vendor metadata. Mode alone, an unknown native domain, or stream
activity alone is insufficient. All other cases retain actual consistent
frame-correlated per-eye PQW requirements. Composition telemetry still needs
visible panel pixels before it becomes a UI visibility claim.
