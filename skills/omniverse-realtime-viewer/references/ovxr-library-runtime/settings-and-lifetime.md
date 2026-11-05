# Viewer settings and recovery lanes

Use upstream context-creation, tracking, display and runtime-adapter guidance
through [the library route](../dependencies/ovxr.md) to classify supported live
operations and construction-only settings. The viewer maps an intent to a live
update, context refresh, stage replacement or runtime-service restart. Keep the
selected lane and its completion visible to the user; do not infer live support
from a successful arbitrary attribute write.

[AR compositing](ar-compositing.md) defines renderer-side fixtures; the optional
[AR control policy](live-ar-controls.md) exposes only public supported apply
paths and leaves other controls staged or unavailable.

## Apply a live intent

Validate the full candidate and capability before queuing it to the runtime
owner. Track requested, accepted and frame-applied revisions independently.
Do not mark an anchor applied merely because its setter accepted the request;
use upstream's active-state boundary, then require new submitted/decoded pixels
for visual acceptance. Reject completions superseded by a newer request.

Use actual configured/discovered products. Measure rendered, submitted and
mirror extents separately; a successful resolution setter cannot establish
client renegotiation. Keep canonical settings across context replacement and
show unavailable or failed state when the requested feature cannot be applied.

## Refresh a context without losing the application

Use upstream's safe context release/create sequence. Add this viewer transaction:

1. Validate the complete replacement request; mark refresh pending and quiesce
   old input, UI captures and image consumers on their owner boundaries.
2. Retain the source-safe scene, canonical anchor, accepted settings, renderer
   and stage where compatible. Complete old consumer reads before release.
3. Recreate through the shared publication coordinator, reapply canonical state,
   and reacquire capabilities using the new context. SDK handles and buffers
   cannot survive through an application cache.
4. Advance context generation and input epoch; reset sequences, hit/capture
   state and visual publication. Preserve stage generation unless the scene
   changed. Reopen consumer channels only after their new owners are ready.
5. Report applied versus rendering-ready separately. Failure leaves an explicit
   detached/error state or a deliberate reconstruction of the prior request.
   Never report success against the released context.

Record renderer, stage, service and context identities to prove which lane ran.
Client reattachment remains a target-runtime test. Runtime-service changes and
native-loader/process changes follow vendor guidance; service recreation is
not proof that process-cached configuration changed.

## Correlated lifecycle controls

An application may expose a context-refresh request carrying a supported
settings patch and a separate service-recovery request. These are application
messages, not SDK exports. Keep service configuration distinct from reconnect
recovery. Acknowledging the queued request is not completion or delivery proof:
the channel may close during refresh. Recover canonical status with the new
generation after reconnect rather than blindly resending a timed-out mutation.

If the application hosts a service, preserve orderly shutdown ownership across
service restart using the vendor's process/signal contract. Test shutdown after
recovery as well as the first reconnected frame; do not copy a particular
runtime release's signal-handler workaround into every viewer.

## Consumer and performance boundary

Upstream frame-readback and application-flow own SDK concurrency, pointer expiry
and shutdown rules. [Mirror lifetime](mirror-lifetime.md) owns this viewer's
copy/resize/transport transaction. Throttle before image work, reuse owned
allocations, and keep extraction, profile discovery, BVH building and shader
warmup outside steady-state frame scheduling. Measure owner-thread, transport
and presentation costs separately.
