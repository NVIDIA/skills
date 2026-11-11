# Bounded retained raster pipeline

This is an application protocol, not an OVXR export. Use it when adding dense
panels or fixing interaction-time stalls. [Skia/Yoga](skia-yoga.md) owns paint
and layout; [raster ABI](raster-abi.md) owns the native copy/upload boundary.

## State and ownership

| Owner | State | Allowed work |
| --- | --- | --- |
| XR/application owner | Canonical model, selected IDs, desired document revision, displayed image/hit-map pair | Apply commands; offer immutable snapshots; poll ready results without waiting |
| Raster supervisor worker | Child generation, one in-flight request, one replaceable pending request, one completed result | Bounded serialization, child I/O, deadline/restart handling |
| Raster child | Fonts, theme, Yoga layout, Skia canvas | Render immutable request; return RGBA and its exact hit regions |
| Native compositor | Accepted/uploaded image revisions and retained GPU texture | Copy caller bytes, upload changed revisions, retire GPU resources |

One in-flight/one pending/one result is a bounded policy, not permission to
allocate unbounded snapshots. Bound request bytes, result bytes, dimensions,
node count, diagnostic stderr and total surfaces. For multiple windows use a
bounded dirty-surface set with fair scheduling so a busy toolbar cannot starve
the stage tree. A shared atlas is optional, not required for state ownership.

Each request/result carries:

```text
surfaceId, stageGeneration, surfaceGeneration, workerGeneration,
documentRevision, width, height, theme/font identity
result additionally: format, alphaConvention, pitchBytes, pixels, hitRegions
```

Pose revision is separate. Never send a new raster merely because the hand,
anchor or panel pose moved. Reject old stage/surface/worker generations even
if their document number is large. Within a generation, revisions increase.

## Owner-side algorithm

```text
on semantic model change:
    build bounded immutable snapshot outside expensive stage extraction
    increment desired document revision
    replace pending snapshot under a short lock; return immediately

each owner update:
    take at most one complete result without waiting
    validate identity, bounds, lengths, finite hit rectangles and revision
    reject stale generation or revision <= currently displayed
    retain newest eligible complete image/hit-map pair
    publish it through the copied native image ABI
    keep pose-only updates independent
```

A valid completed revision may precede the latest desired revision. Accepting
it avoids starvation during continuous hover changes; continue toward the
latest pending request. Never regress the displayed revision or combine newer
hit rectangles with older pixels. If native upload/presentation lags, either
promote hit testing only with its acknowledged displayed revision or disable
activation until the pair is coherent. Native copy acceptance alone is not
proof the headset has shown those pixels.

Route actions by stable surface/node ID plus document/stage generation, never
row number alone. Resolve against current authoritative state before mutation.
Deleted or disabled targets reject cleanly; stale visual state cannot authorize
a destructive action. For captured press/release, retain that node's identity
and cancellation policy across later raster revisions.

## Child transport and failure

Use framed binary RGBA plus bounded metadata, not large base64 images in the
XR loop. Define magic/version, metadata length, pixel length, width/height and
pitch; check overflow and `pitch >= width*4`, and require the exact declared
payload. Read exact byte counts on the worker. Reserve stdout for the protocol;
drain stderr into a bounded diagnostic ring so a full pipe cannot deadlock it.

Start a deadline **before writing** the request. A reply timeout does not bound
a blocked stdin write. The supervisor must be able to terminate its own stuck
child and close that generation's pipes, then join/reap it within a declared
shutdown budget. Reader threads capture their own process/pipe generation;
late output must never enter a replacement child's response queue.

Keep the last complete displayed pair on transient failure, label it stale and
disable unsafe actions. Retry the newest still-relevant request only within a
bounded attempt/time policy. On exhaustion expose unavailable/error while XR
continues; do not restart the viewer or spin-retry every frame. On shutdown stop
new offers, cancel pending work, and clean up the owned child off the XR frame
path. A stage replacement clears obsolete action captures immediately even if
old pixels are retained briefly for presentation.

## Tests and counters

Test blocked request writes, truncated/oversized results, child death, late
old-generation responses, continuous hover bursts and multi-window fairness.
Assert bounded memory, nonblocking owner updates, coherent image/hit maps and
no pose-only raster uploads. Record requested/completed/displayed revisions,
queue replacements, timeouts/retries, raster/IPC/upload time and bytes separately.
