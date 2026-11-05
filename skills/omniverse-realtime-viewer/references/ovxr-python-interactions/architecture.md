# Python interaction architecture

## Ownership

- OVXR owns OpenXR session/frame lifecycle, predicted eye poses, projection
  products, rendering coordination, and pixel submission.
- One application owner (called `OvxrRuntime` in the reference sample) owns
  renderer/OVStage; implement it locally rather than importing a sample helper.
- Browser, SHM, and message callbacks enqueue immutable intent.
- Python owns profile normalization, semantic edges, picking policy, tool
  capture, gestures, navigation policy, UI documents, and physics orchestration.
- OVStage is the visible scene-write plane. OVRTX consumes committed ordinals.
- An optional OVPhysX child owns only its own fixture USD. The parent renderer
  never imports OVPhysX and the child never mutates the renderer's stage.

## Render-loop order

Use one convenience `Context.render_frame()` path or one SDK split-frame path,
never both. [Frame/input scheduling](../ovxr-library-runtime/frame-input-order.md)
defines timing, anchor latching and publication for each. Drain a
bounded queue of validated immutable intents on its owner thread. Publish anchor
movement through the documented setters and wait for its next frame latch.
Publish application scene changes through documented OVStage transactions,
using one publication coordinator and the SDK's application ordinal mode.
Never add a second frame owner or mutate the stage concurrently.

Native action startup and scene-transform SDK calls follow upstream input and
stage-publication guidance through [the library route](../dependencies/ovxr.md).
Apply this viewer's normalized inputs and transforms through the shared owner;
use [transform safety](../prim-transform-safety/README.md) for local/world
composition and derived-world freshness.

## Coordinate contract

Use the current public anchor and `transform_openxr_pose` methods as specified in
[navigation](../xr-spatial-input/navigation.md). Stage metadata,
spawn anchor, canonical navigation and raw tracking poses remain separate.
No eye-camera alignment inference or inverse `/World` transform is needed for
poses already in the runtime-selected tracking space. An independently LOCAL
compositor still needs the actual tracking-to-LOCAL `xrLocateSpace` mapping;
managed anchor conversion does not erase that reference-space distinction. Preserve legacy fallback only behind an
explicit missing-capability branch and document its unverified limits.

## Message contract

Use `{event_type, payload, request_id}`. Mutating streams carry `schemaVersion`,
stable client-session ID, input epoch, strictly monotonic per-stream sequence,
strictly monotonic source timestamp, focus/tracking state, application-owned
stage generation, and absolute or revisioned state. Preview the envelope before
the family parser and commit freshness only after the semantic payload is
accepted; a malformed future epoch must not poison a valid current source.
Reject duplicates and stale revisions without replaying deltas. Begin, update,
end, and cancel for one gesture are one stream and therefore use distinct
increasing sequences. Reconnects fetch canonical state; a stage change advances
generation and invalidates packets captured against the prior stage.

The controlled client may supply stage-space poses when it explicitly applies
the shared application conversion. Do not claim this is native OpenXR action
registration or OVXR pose conversion.

Keep lifecycle terms precise: CloudXR/OpenXR/OVXR-context reconnect is allowed;
the Python process, OVRTX renderer, OVStage owner, and SHM producer remain
long-lived. Do not design feature recovery around fast full-stack restart.

## Minimal owner implementation

The example below is the convenience path with explicitly subsequent-frame
input. For same-plan predicted-time input, implement the split schedule linked
above instead; do not splice a separate begin/bind/submit into this example.

Keep application state separate from SDK handles: stage generation, accepted
input epoch/sequence, canonical navigation revision, pending anchor matrix,
last frame-latched anchor, source snapshot generation, and current capture owner.
The following names are pseudocode, not methods provided by OVXR:

```text
while running:
    commands = queue.take_at_most(command_budget)
    validate_and_apply_commands_on_owner(commands)
    # Scene writes: allocate through the shared publication coordinator;
    # resolve all lower writes before publishing its owned ordinal.
    outcome = drive_selected_sdk_frame_through_coordinator(dt)
    reconcile_sdk_publication_and_plan_outcome(outcome)
    if outcome.submitted:
        if pending_anchor_revision:
            compare_active_anchor_to_pending_and_record_applied()
        sample_or_accept_tracking_for_the_next_update()
        run_semantic_edges_and_publish_app_scene_changes()
        publish_preview_only_if_a_new_capture_sequence_exists()
```

Use [the shared data plane](../ovstage-data-plane/README.md) for scene writes
and upstream stage-publication for co-writing OVXR calls. Do not duplicate SDK
operation signatures or their cleanup rules in the semantic interaction layer.

Refresh detaches/destroys only the OVXR context, reattaches the same renderer and
stage, reinstalls canonical user placement, clears borrowed handles/reference
mappings and input captures, then accepts a new transport epoch. Stage reload
also invalidates hierarchy/raycast caches and advances stage generation. A
CloudXR restart is a separate service operation; do not hide it inside live
movement or an ordinary render-product setting update.
