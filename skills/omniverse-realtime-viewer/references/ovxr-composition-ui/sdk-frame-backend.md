# SDK frames with retained spatial UI

Read [library frame-loop, stage-publication and composition-integration](../dependencies/ovxr.md). Default bind renders and owns its result. The release supports additional layers, borrowed Vulkan handles, application render products, and explicit application-stepped mode. Delegate their exact signatures and resource lifetime to upstream rather than duplicating SDK instructions here.

## Application boundary

The viewer owns layout, raster/hit maps, gestures, textures, shaders and its primary-depth compositor. OVXR owns eyes, frame and submission. [Scheduling](../ovxr-library-runtime/frame-input-order.md) defines coordinator/latency; do not add another step or submission owner.

A custom API layer may implement [native construction](library-only-compositor.md) and [primary projection](primary-projection.md), establishing graphics binding, attachment lifetime and dispatch interception. Borrowed session handles do not supply primary images or a Vulkan callback.

Additional quads and depth-aware primary UI remain different policies. Prove the chosen supported seam; do not describe absent SDK layer methods as available or silently substitute always-on-top content.

Keep raster/pose revisions separate, preserve selection/capture ownership and retire app GPU work before context replacement.

## Proof

Test one completion, no-render/errors, head motion with UI on/off, both-eye depth/alpha, cancellation and replacement. Measure compositor cost separately from runtime waits. Historical prerelease captures do not validate this package.
