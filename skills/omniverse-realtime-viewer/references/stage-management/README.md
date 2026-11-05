# Stage Management

Use this reference for opening, switching, reloading, and composing USD scenes
while preserving viewer settings and keeping the interface responsive.

## Scene owner and UI state

One runtime owner coordinates the stage, renderer attachment, current source,
stage generation, and committed ordinal. UI and transport send commands and
receive copied state; they do not own stage handles.

Keep the active scene interactive while downloading assets or gathering
metadata. Track a requested load ID separately from the active stage generation;
only the latest scene-replacement request may proceed to the renderer handoff.
Additive edits remain ordered commands. At handoff, disable scene interactions
and retain the last displayed frame while replacing the runtime stage. Tag
picks and queries with the active generation so late results cannot update the
new scene's inspector or selection.

The server's current scene is authoritative on startup and reconnect. Send it
with initial state (for example, `openStageResult`) and let the client update its
selector. Deduplicate requests for the already-loaded source; an explicit reload
bypasses that check. This avoids loading the dropdown default on every reconnect.

## Compose the source

Keep display labels separate from source paths. Preserve relative asset
resolution in local files or cloud caches; see [stage-loading](../stage-loading/README.md)
and [cloud-assets](../cloud-assets/README.md).

Populate viewer cameras, RenderProducts, RenderVars, and non-destructive settings
from a wrapper/session source. Live edits go through the application-owned
OVStage. [ovstage-population](../ovstage-population/README.md) owns the operation
choices: a fresh Stage for full replacement, reference handles for additive
content, and publication after each composition change.

## Replace a scene

Run the transition through the runtime owner, between render steps:

1. Assign a load request ID and show progress without changing the active scene.
2. Resolve the source and gather required metadata, including source up-axis and
   units for wrapper composition through [stage-loading](../stage-loading/README.md).
   Keep `pxr` metadata work in
   its worker process; return copied bounds, paths, and hierarchy data. A failed
   download or metadata request leaves the current scene available.
3. Confirm this is still the latest request, then begin the renderer handoff:
   snapshot persistent camera/settings state, invalidate the old generation,
   and disable scene interactions. Finish pending renderer/consumer work and
   release old queries, mappings, bindings, picks, and selection state. Detach
   consumers before destroying the old Stage to bound peak scene memory.
4. Create a fresh Stage, populate the composed source, wait for completion, and
   advance its write floor. Attach it to the renderer using
   [ovstage-ovrtx-integration](../ovstage-ovrtx-integration/README.md).
5. Restore validated viewer settings and the selected camera policy. Publish
   these writes and rebuild stage-specific dictionaries and feature state.
6. Render the first valid frame, then publish scene success, hierarchy, and
   selection state to the UI. On failure, report the load error and allow retry;
   do not step an unattached or incompletely initialized renderer.

This sequence preserves the displayed image during runtime population, after
download and metadata work have finished. Continuing to interact with the old
scene during that final population phase requires an extra live Stage and an
explicit memory budget.

Population and renderer attachment are serialized with rendering. In particular,
do not mutate the attached stage while `step()` or `step_async()` is in flight.
`renderer.step(..., ordinal=N)` consumes the committed publication; a separate
`update_from_stage(N)` is needed only for pre-step renderer queries.

## Keep loading responsive

Use asynchronous population and return control to the application's event loop
between polls. Start the app-owned load job once, with a fresh Stage for scene
replacement:

```python
op = population.open_usd_async(stage, composite_path, ordinal=ordinal)
```

Poll on later runtime ticks; keep displaying the last owned frame meanwhile:

```python
# Population operations have their own wait API; do not use stage.wait_op().
# wait(timeout=0) returns None while pending and raises OvstageError on failure.
if op is not None and op.wait(timeout=0) is not None:
    stage.advance_write_floor(ordinal).wait()
    op = None  # Stop polling/publication after this job completes.
    # Continue attachment and first-frame work on the runtime owner.
```

Route an enqueue, population, or publication exception to the load-error path
and stop ticking that job. Do not attach or render the incomplete stage.

Do not report success merely because an operation was enqueued. Complete the
publication and first-frame handoff. A superseded load may finish in the
background, but its generation must not replace the latest requested scene.

## State across scenes

| State | Scene-switch policy |
| --- | --- |
| Render settings | Keep validated app preferences; reapply them through the new session/runtime stage. |
| Camera | Choose `fit`, `preserve`, or an authored stage camera with fit fallback. |
| Selection and inspector | Clear old paths and native selection groups immediately. |
| Hierarchy and bounds | Rebuild from the new scene; use its detected root/default prim rather than assuming `/World`. |
| Queries, path dictionaries, pick IDs, animation bindings | Recreate for the new Stage. |
| Optional pick effects | Drop old bookkeeping; capture new baselines without overwriting authored material values. |
| Available AOVs | Rediscover valid outputs for the new scene and preserve the selection only if available. |

Store viewer preferences outside the user USD asset unless saving edits is part
of the requested product. Sanitize restored camera values and fit when the new
scene cannot use the previous target or distance.

Reload uses the same full replacement path, including generation invalidation.
Removing one additive asset uses its reference handle and an apply/publish
cycle, retaining the rest of the scene.

## Verify the transition

Switch between scenes with different roots, sizes, and materials; reload the
current scene; then try an invalid source. Check first-frame readiness, camera
fit, settings persistence, cleared selection, and recovery from failure. For
streamed viewers, reconnect and confirm the client adopts the active scene
without starting another load.

See [render-settings](../render-settings/README.md),
[camera-controls](../camera-controls/README.md), and
[selection-feedback](../selection-feedback/README.md) for feature behavior.
