# Worked compositor passes and synchronization

This application design implements scene-depth tools, foreground gizmos and
UI above gizmos without copying the primary eye images. It is not an OVXR
callback API. First satisfy [image ownership](primary-projection.md) and
[projection/depth encoding](projection-and-depth.md).

## Preconditions and per-eye passes

Use the borrowed session's graphics device/queue, waited color image C and
matching scene depth D. Verify attachment usage, format, sample count, array
layer, subimage bounds, layout and producer synchronization. Do not sample D
unless it also has sampled-image usage and a verified layout contract. This
baseline uses attachment depth tests, not depth sampling.

Allocate private gizmo depth G for the selected extent/sample count, with a
bounded per-slot lifetime. Keep pipeline keys explicit: formats, samples,
encoding/compare, blending, depth-write policy and pass. For each eye:

| Pass | Attachments/load | Draw/test/write | Result |
| --- | --- | --- | --- |
| Scene tools | C LOAD, D LOAD | Rays/arcs against D; choose depth-write off for this baseline | Tools respect scene occlusion without altering the depth used by later UI |
| Foreground gizmos | C LOAD, G CLEAR | Gizmos test/write G; clear far (0 reversed, 1 normal) | Gizmos overlay scene/tools but self-occlude; D is untouched |
| Panels/chrome | C LOAD, D LOAD | Covered UI tests D with the declared panel policy; write covered opaque/masked depth | Visible UI is drawn after gizmos and wins their overlap; transparent margins reveal prior content |

STORE C/D when used; G is private and never attached to the exported projection
depth chain. Do not draw the gizmo a second time in the final UI batch. When no
foreground gizmos exist, compatible scene/UI batches may share a pass. Count
actual draws and passes, not merely composed eyes.

Within the UI pass, actual panel depth determines overlap for opaque/masked
windows; deterministic surface priority breaks coincident-depth ties and must
also govern pointer hit arbitration. Fractional AA fringes need a declared
depth-write rule. One practical policy separates opaque interior depth writes
from color-only edge blending; measure extra draws before adopting it. This
does not solve independently reprojected translucent glass, which needs more
than one depth per pixel. Do not claim that limitation is fixed by draw order.

## Queue and barrier recipe

For one graphics queue, serialize queue access with its owner. Submission order
is necessary but not a substitute for memory dependencies. The following is a
conservative baseline outside render passes, using the corresponding legacy
`vkCmdPipelineBarrier` masks (or synchronization2 only if enabled):

| Boundary | Source | Destination |
| --- | --- | --- |
| Producer → color LOAD/blend | ALL_COMMANDS / MEMORY_WRITE | COLOR_ATTACHMENT_OUTPUT / COLOR_ATTACHMENT_READ + WRITE |
| Producer → scene depth | ALL_COMMANDS / MEMORY_WRITE | EARLY + LATE_FRAGMENT_TESTS / DEPTH_STENCIL_ATTACHMENT_READ + permitted WRITE |
| Previous pass → next color pass | COLOR_ATTACHMENT_OUTPUT / COLOR_ATTACHMENT_WRITE | COLOR_ATTACHMENT_OUTPUT / COLOR_ATTACHMENT_READ + WRITE |
| Previous depth write → later depth test | EARLY + LATE_FRAGMENT_TESTS / DEPTH_STENCIL_ATTACHMENT_WRITE | EARLY + LATE_FRAGMENT_TESTS / DEPTH_STENCIL_ATTACHMENT_READ + permitted WRITE |
| Retained texture upload → sampling | TRANSFER / TRANSFER_WRITE | FRAGMENT_SHADER / SHADER_READ |

Scope image barriers to the used aspect, mip and array layers. Preserve known
color/depth layouts or transition explicitly; never use UNDEFINED for scene C
or D because that discards content. UNDEFINED is suitable only for newly
allocated or intentionally discarded private G/texture contents, with ordering
against any prior uses still respected. Restore the graphics binding's required
release layouts before forwarding image release; do not use window-system
PRESENT_SRC merely because these are called swapchains.

Memory hazards and stage/access pairing follow the
[Khronos synchronization examples](https://docs.vulkan.org/guide/latest/synchronization_examples.html).
Release/queue requirements come from the selected
[OpenXR Vulkan graphics binding](https://registry.khronos.org/OpenXR/specs/1.1-khr/html/xrspec.html#XR_KHR_vulkan_enable).
If rendering finishes on another queue, require the producer's semaphore and
queue-family ownership contract. A same-queue barrier cannot repair an unknown
cross-queue dependency; omit composition rather than inventing a CPU wait.

## Slots, textures and retirement

Before recording, poll a bounded slot fence. A busy slot keeps its command
buffer, staging data, descriptors, textures and G alive. If all slots are busy,
skip this frame's UI, still flush eligible deferred releases exactly once and
forward the original frame. Never reset an unsignaled fence or reuse its memory.

On a new raster revision, copy bytes to owned staging, flush noncoherent host
memory over aligned ranges when required, transition the retained texture from
its tracked layout to TRANSFER_DST, copy, then transition to SHADER_READ_ONLY.
Geometry host writes need the corresponding visibility to vertex/index reads.
For unchanged dimensions, queue-order the texture write after earlier sampling;
for replacement textures, allocate a new resource and retire the old one only
after every referencing slot signals. Do not update a descriptor still in use;
use per-slot descriptors or another explicitly enabled lifetime-safe design.

Native acceptance, upload queued, GPU completed and frame submitted are distinct
revisions. Keep the matching raster/hit map through that chain as described in
[async publication](async-raster-pipeline.md). A failed queue submission cannot
advance uploaded/presented state. Device loss enters explicit recovery instead
of pretending fence completion. Preflight retirement before destroying OVXR's
borrowed device; a teardown timeout must retain owners, not free in-flight data.

## Required validation

Use Vulkan validation/synchronization validation in a diagnostic run when
available. Exercise worker/upload failure, texture resize during queued work,
slot exhaustion, per-eye unequal extents, source swapchain destruction and
context refresh. Capture scene-through-UI, gizmo-through-handle and transparent
margin crossings. Missing image usage or producer synchronization is a blocked
capability, not permission to change the scene-depth presentation policy.
