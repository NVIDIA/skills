# XR frame scheduling and capture handoff

Read [library frame-loop, stage-publication and frame-readback](../dependencies/ovxr.md) for signatures, plan ownership and pointer lifetime. The viewer implements [input scheduling](../ovxr-library-runtime/frame-input-order.md), with one frame owner.

Default bind performs rendering and owns its result. Route application render products and explicit application-stepped mode to the upstream frame-loop skill; the latter requires creation-time opt-in, explicit bind, and caller-owned result cleanup. Do not insert another renderer step into the default loop.

A mirror consumes capture instead of rendering another eye. [Mirror lifetime](../ovxr-library-runtime/mirror-lifetime.md) owns application leases, bounded copies and staging; [desktop delivery](../xr-desktop-mirror/README.md) owns cadence, resize and handoff. Record source sequence/product/domain. Missing capture means stale or unavailable preview.

Do not alter headset resolution/foveation for a mirror preference. [Preview policy](../ovxr-foveation-and-display/README.md) requires measured capture support. Capture success alone does not prove overlays, alpha, depth or decoded headset pixels.
