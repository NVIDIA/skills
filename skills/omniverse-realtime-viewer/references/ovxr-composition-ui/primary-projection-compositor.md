# Primary projection compositor

## Ownership and frame shape

OVXR owns instance, session, frame timing, and submission. The API layer may
defer release of the matching primary color/depth images, compose once those
images are waited, release them exactly once, and forward the same projection.
It must not begin/end frames or create a competing projection owner.

For every eye, use the projection view's own pose, asymmetric FOV, subimage
offset/extent, array layer, color image, and chained depth subimage. Treat left
and right as independent; never derive one eye from the other or assume equal
extents. Validate color/depth subimages, sample counts, formats, protected
content, acquisition state, and depth parameters before drawing.

## Geometry and clipping

Transform LOCAL surface corners into each eye view, project with that eye's
FOV tangents, then clip in homogeneous space before division. Map NDC into the
submitted subimage, including nonzero offsets. Vulkan viewport and scissor are
in full image coordinates and must match the actual target subimage.

The one effective scissor per eye is the union of every surviving panel and
auxiliary bound. A tooltip or gizmo outside a panel-only scissor is a compositor
bug even when its mesh was otherwise valid. Batch compatible surfaces.

Keep base panel and auxiliary validity independent. A clipped tooltip or gizmo
must not disable the panel; a clipped panel must not prevent an otherwise valid
auxiliary from being diagnosed. Count per-eye composed and clipped mesh parts.
Fail only when no drawable surface remains or a required resource contract is
invalid.

## Depth and alpha

Use the chained `XrCompositionLayerDepthInfoKHR` for attachment/range metadata,
and verify its producer's actual pixel encoding. The supplied OVXR XRDepth is
reversed even with physical near < far; use the equations and encoding gate in
[projection/depth](projection-and-depth.md), not plane ordering alone. Preserve
scene color/depth with LOAD semantics. Blend covered UI fragments and leave
uncovered pixels unchanged.

## Synchronization and performance

Implement [the worked pass/barrier sequence](compositor-passes-and-sync.md)
for private gizmo depth, UI priority, retained texture updates and retirement.

- Reuse long-lived Vulkan resources and bounded fence-polled slots.
- Upload the retained UI texture only when its document revision changes.
- Do not copy or transfer full eye images.
- Do not wait on a fence submitted by the same frame.
- If every slot is busy, omit the UI for that frame and retain honest counters.
- Budget one scissored draw per compatible batch per eye. Explicitly account
  for additional foreground/depth-policy batches; distinguish draw commands,
  render passes and scissor changes in telemetry.

Track full-eye copies/bytes, transfer commands/bytes, texture uploads and
unchanged-revision uploads, per-eye draw/scissor deltas, slot exhaustion,
same-frame waits, GPU time, compose CPU time, and queue-submit CPU time.

### Destruction is an ownership transaction

Before asking the SDK to destroy its context/device, retire application GPU
work while those borrowed handles are still valid. A void SDK destroy wrapper
may hide lower-level OpenXR teardown errors: do not rely on that wrapper to
report a timeout. Provide an application-owned preflight, not an invented OVXR
SDK method. On timeout/device loss, retain owners, stop further submissions and
captures, surface failure, and permit an explicit teardown retry. Never clear
the last context reference and continue destroying its renderer after failure.
