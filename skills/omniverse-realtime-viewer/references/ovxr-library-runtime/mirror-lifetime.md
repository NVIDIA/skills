# Independent mirror publication and borrowed GPU lifetime

A mirror is an application-owned consumer of rendered images. It must not change
headset RenderProduct resolution when its window is resized. Keep source size,
requested physical display bound, actual publication size and headset submission
size as separate facts. Preserve aspect ratio, avoid unwanted upscaling, and
retain newest-frame-wins behavior in the producer, SHM transport and browser.

## Copy before releasing the source

Use upstream frame-readback and application-flow through
[the SDK route](../dependencies/ovxr.md) for pointer validity, source-reading
leases, concurrency and teardown. Do not extend that lease through the encoder
or assume a Python guard makes an asynchronous reader safe.

The viewer adds GPU resize and transport lifecycle to that source-copy boundary:
copy/downscale into persistent application-owned staging while source reads are
protected, complete those reads, release the source, then offer the owned image
to transport. Acquisition, expiry and SDK synchronization mechanics remain
upstream; the destination's longer ownership follows the transport contract.

Do not hold the source lease through slow SHM/network/encoder calls, including
transport stop/close/start on resize or resize-error recovery. Plan transport
recreation outside the lease; release after source reads complete before
entering transport lifecycle work. Do not
replace the copy with a raw borrowed pointer on failure. Explicitly complete or
cancel source-reading work before releasing on error. A private-stream wait is
bounded source-lifetime synchronization, not license for a device-wide barrier.
Retain the transport's destination until its documented last consumer is done;
verify the installed OVStream buffer contract before reusing staging.
When staging must grow, retain the old destination until its transport consumer
has finished; allocating a replacement does not release that ownership. On a
resize error, finish any already-enqueued source reads before fallback or lease
release. Test startup, shrink/grow, partial enqueue failure and concurrent stop,
asserting that source release precedes every transport lifecycle operation.

## Demand, resize and reconnect

Use actual presentation demand, not connection state alone. A hidden/minimized
window can stay connected. Send visibility and physical pixel bounds on changes
and replay retained demand after each new connection epoch. Throttle before
acquiring readback and again at the publisher if several producers share it.
An idle heartbeat is an application policy; report it rather than claiming no
work. Restoration can clear the cadence deadline for an immediate next frame.

When SHM dimensions require server recreation, consumers must close old handles
and reconnect. Serialize wait/copy/release/send/close with bounded waits. Advance
a connection epoch only on successful attachment; clear old pixels, dimensions,
sequence, pixel order and foveation. Wait for real new-producer metadata instead
of inferring RGBA/BGRA from a previous connection. Reject stale callbacks, replay
viewport demand, and release borrowed client frames before close.

Source/capture metadata must be atomically associated or exactly joined by a
producer-provided key. Unknown correlation stays unknown. Never substitute the
current write floor for the ordinal of a captured frame or match nearest times.

## Verification boundary

Use a changing real source. Prove increasing capture/publication sequences,
new dimensions, aspect ratio, decoded pixel order, and nonblank changing images
before claiming a mirror works. Resize and reconnect while head tracking remains
active. Compare active versus hidden publication/copy counts and check restore.
Keep headset source resolution stable during desktop-only size changes.

A browser adapter exercising the real React UI/SHM is not native Tauri evidence.
Control-only mode proves command routing and persisted state, not desktop frames.
Fake-client tests establish ordering/resource discipline, not GPU lifetime proof.
Record source-copy timing separately from transport and draw, and retain these
limitations with the evidence.
