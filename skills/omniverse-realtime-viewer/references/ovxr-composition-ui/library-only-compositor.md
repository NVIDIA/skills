# Constructing a compositor with libraries and skills

This is an implementation plan for code your application owns. OVXR supplies
neither a panel renderer nor the custom native layer described here. Required
inputs are the installed OVXR/OVStage/OVRTX contracts, OpenXR loader/API-layer
headers, Vulkan headers/loader, and a shader compiler. Do not depend on a sample
checkout, Kit extension, private controller asset, or an undocumented OVXR
callback. If these native build inputs are unavailable, report that dependency;
a successful OVXR context alone cannot provide projection-depth UI.

When replacing the high-level render call with explicit SDK frames, use the
[SDK backend recipe](sdk-frame-backend.md). This preserves the native compositor
and retained UI while keeping exactly one begin/bind/submit owner.

## Components to implement

| Component | Input → output and lifetime |
| --- | --- |
| Retained document | Stable node IDs, text/layout and document revision → RGBA atlas plus hit rectangles; upload only changed revisions. |
| Interaction router | One immutable, owned controller snapshot and LOCAL rays → hover/capture/action; UI ownership runs before scene tools. |
| Placement | Fixed LOCAL pose, or world pose + active anchor + measured reference-space mapping → LOCAL surface pose/extents. |
| Application native ABI | Revisioned document/placement updates → copied native state; getter returns measured mapping and telemetry. |
| OpenXR API layer | Borrowed projection/swapchains → in-place color/depth draw and unchanged forwarded projection. |
| Vulkan compositor | Retained texture, per-eye vertices and source attachments → bounded scissored batches per eye, with an explicit budget for depth-policy splits; bounded reusable slots. |

Keep the ABI explicit: use fixed-width fields, size/version tags, documented
quaternion order and units, and copy caller arrays before returning. Resolve
optional exports individually and reject incompatible required fields. Do not
pretend an absent composition-mode or auxiliary-pose field was accepted. A
mapping record needs valid flag, source-space identity, session generation,
frame counter, display time, quaternion XYZW and translation XYZ in metres.
Support explicit session selection if the layer can host multiple sessions.

## Native layer wiring

Use [native build/discovery](native-build-and-discovery.md) for compilation
units, exports, manifest and process-local activation. Use
[projection math](projection-and-depth.md) and
[pass synchronization](compositor-passes-and-sync.md) for the implementation
behind the ownership outline below.

Build a shared library and an OpenXR API-layer manifest naming that library and
the negotiated layer name. Implement loader negotiation using
`openxr_loader_negotiation.h`, `xrCreateApiLayerInstance`, and your
`xrGetInstanceProcAddr`. Advance the loader-provided next-info chain when
creating the downstream instance; retain downstream dispatch per instance.
Forward unhandled functions without substituting global handles.

Intercept session creation/destruction, reference spaces, swapchain
creation/destruction/enumeration/acquire/wait/release, and `xrEndFrame`.

At each intercepted `xrCreateSwapchain`, retain the actual format, usage flags,
sample count, dimensions and array size. In-place composition requires
`XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT` on the color image and
`XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT` on the depth image; transfer
usage alone is insufficient. Configure the producer to request these usages or,
where permitted, copy and augment the create info before downstream creation.
Confirm runtime acceptance and record the resulting contract. Creating a new
Vulkan image view cannot add missing image usage. If attachment creation or the
required depth extension/submission policy is unsupported, report depth-aware
UI unavailable before claiming that feature works; do not silently replace it
with always-on-top layers. A non-UI pass-through mode may still submit the
original unmodified scene.

Obtain the Vulkan device and queue from the session graphics binding; do not
create a second device or OpenXR session. Track each acquired image by
swapchain and index. Defer only eligible, already-waited primary image releases
until the corresponding end-frame composition, then forward each release
exactly once, including skipped/error paths. Unsupported image formats,
protected content, missing depth, or unavailable resources skip UI and retain
normal application submission. Preserve all original layer chains and views.

Create an identity LOCAL reference space during intercepted session setup.
At end frame, inspect the actual projection `space`, display time, per-eye
pose/FOV, subimage and depth chain, then locate that source relative to LOCAL
at the submitted display time even when drawing is gated. Making mapping
availability depend on visible UI creates a readiness deadlock.

If placement is converted before bind, implement an application-owned getter
that actively calls downstream `xrLocateSpace` for the borrowed source and
owned LOCAL at `plan.predicted_display_time`. Return a copied mapping with
source handle, session/reference-space generation, target time and validity.
Do not label the previous frame's end-frame cache as a current-frame mapping.
Alternatively retain world-space placement until the layer converts it at
end frame using that frame's measured mapping. Cache reuse requires matching
source/session/time provenance; invalidate on reference-space change, recenter
or session replacement. The getter is not an OVXR export. Match its source to
the runtime's borrowed `xr_space` before applying it to OVXR tracking input.

The core OpenXR operation is ordinary documented API use:

```cpp
// session and source are borrowed; local was created for this same session.
XrSpaceLocation location{XR_TYPE_SPACE_LOCATION};
XrResult result = nextLocateSpace(source, local, displayTime, &location);
const XrSpaceLocationFlags required =
    XR_SPACE_LOCATION_POSITION_VALID_BIT |
    XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
bool valid = XR_SUCCEEDED(result) &&
    (location.locationFlags & required) == required;
// On valid: copy location.pose, source, displayTime, session generation.
// On invalid: invalidate the cache; never substitute an estimated eye height.
```

Render using the application's Vulkan queue and correct attachment layouts,
access barriers and queue ordering. LOAD both attachments, blend covered color,
and resolve the producer depth encoding alongside the chained range. Never CPU-wait for your
same-frame submission. Reuse a slot only after its prior fence signals; if none
is ready, skip UI. Flush deferred releases before forwarding downstream
`xrEndFrame`; do not invoke a second OVXR render or OpenXR frame loop. See
[the detailed compositor contract](primary-projection-compositor.md).

## Placement and reference-space math

Use one explicit row-vector convention below (transpose consistently if your
math library uses columns). Let B convert OpenXR metres/Y-up to stage units/up
axis, A be the active SDK managed anchor, and M the measured tracking-to-LOCAL
rigid transform in metres:

```
world = trackingMetres * B * A
localMetres = world * inverse(B * A) * M
```

For an affine world surface, transform its origin with homogeneous w=1 and its
tangent axes with w=0. Axis lengths give physical LOCAL extents; normalize the
axes for orientation. This preserves positive uniform user scale exactly once.
Reject singular/nonfinite matrices and unsupported shear instead of discarding
scale through a rotation-only inverse. Normalize transformed ray directions;
transform their origins as points. Transform controller aim, grip, head-facing
basis and any head position into the same LOCAL space. Native poses already
located in LOCAL bypass this conversion.

A fixed LOCAL default panel bypasses world conversion entirely. For an explicit
world anchor, recompute placement when the *effective active matrix* or mapping
changes, including a changed spawn base with unchanged navigation revision.
Pending managed-anchor acceptance is not frame-latched placement. With the
convenience path, read active state after a successful frame for subsequent
placement. With split frames, use the frame mapping latched at begin and
the intended display-time mapping, following
[frame/input scheduling](../ovxr-library-runtime/frame-input-order.md).
Update placement once per changed value without rerasterizing the document. Invalidate on missing source prim,
tracking loss or context replacement.

## Image domain and lifecycle

For required depth-aware UI, create the context with
`depth_submission_policy=ovxr.FeaturePolicy.REQUIRED`, and verify the negotiated
depth extension and per-view depth chains before enabling the compositor.
`AUTO` may omit unsupported depth and cannot establish the required feature.
The attachment usage gate above is separate from depth negotiation.
For the convenience
schedule, SDK frame ownership remains intact: poll input/control work on the owner thread, perform
serialized live anchor/resolution updates, call OVXR render, then consume its
successful-frame state. With after-frame input sampling, the next projection
consumes that snapshot; carry its time/epoch instead of claiming same-frame
sampling. With split frames, begin supplies predicted timing and the README-defined
frame mapping; convert input after begin and publish eligible placement before
bind renders. Preserve sample and mapping provenance. Do not call OVRTX step again for UI.

No-PQW acceptance requires explicit DISABLED local foveation, negotiated local
foveated rendering false, negotiated foveated submission false, and valid
per-eye extents. Unknown booleans are not false. Other modes require consistent
actual per-eye metadata from the submitted projection. A mirror's unwarped
readback flag says nothing about the submitted projection domain. See
[foveation math](foveation-and-edge-math.md).

Live managed-anchor changes and documented resolution updates do not recreate
OVXR. Create-time descriptor changes refresh its context: invalidate borrowed
handles, mapping caches, captures and input epochs, then bind the new session
and republish retained document/placement. CloudXR service restart is a separate
transport lifetime and requires reconnect. Stop callbacks/consumers before
native teardown. Destroy only spaces/resources your layer created; OVXR owns
its borrowed session and frame state. Retire outstanding GPU resources safely
at teardown, rather than carrying their handles into a new device/session.
