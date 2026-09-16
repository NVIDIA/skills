# ovstage Runtime

## Purpose

`ovstage` is the application-owned runtime scene substrate for viewer apps. It
holds the post-composition scene representation shared by rendering, physics,
sensors, and application code. It is not a renderer and it does not replace USD
as the authored composition format.

Use this reference for stage lifetime, ordinals, write floors, shared path
identity, operation completion, attached renderer scheduling, retention recovery,
and consumer teardown.

## Runtime owner

Give one application component ownership of the stage. That component creates
the stage, owns a monotonically increasing ordinal clock, populates composed
USD, serializes structural changes, waits for operations, advances the write
floor, and exposes a small command/query boundary to the rest of the app.

| Owner | Responsibilities |
| --- | --- |
| Runtime owner | stage lifetime, ordinals, publication, recovery, teardown |
| ovstage | runtime hierarchy, attributes, queries, cloning, path dictionary, change membership |
| ovrtx | rendering, outputs, renderer-native picks, selection visuals |
| UI/transport | commands and DTOs; never stage handles or DLPack buffers |

New ovrtx viewer apps should use this split by default. Standalone
renderer-owned stage APIs remain compatibility behavior; do not use them as the
main scene data plane when an OVStage is attached.

## Publication protocol

Every mutation belongs to a new ordinal `N`:

1. Enqueue population, clone, write, map/unmap, or USD-change work at `N`.
2. Wait for the operations that contribute to `N`.
3. Advance the write floor to `N` and wait.
4. Let consumers observe/render the committed publication.

The ordinal is a publication gate, not a historical snapshot selector. Retain
an observed ordinal per consumer. When it is older than the stage retention
frontier, discard stale derived state and resynchronize from current data.

Do not render above the write floor. After population or data-plane writes, wait
for the operation, advance the floor, and then let ovrtx consume that committed
ordinal through `Renderer.step(..., ordinal=N)` or an explicit
`Renderer.update_from_stage(N)`.

## Safety rules

- Keep producer DLPack tensors alive until their asynchronous write completes.
- Do not pass path tokens, query handles, mapped buffers, or borrowed result
  groups across UI, transport, reload, or process boundaries.
- Release results and unmap buffers before conflicting operations.
- Treat a stage as one serialized runtime lane unless the pinned runtime
  documents a stronger concurrency contract.
- Detach every consumer before destroying the stage.
- Keep viewer state out of user USD files. Compose viewer cameras, render
  products, render vars, and non-destructive overrides in wrapper/session USD;
  keep live transforms, material/effect state, and app-owned visibility in
  OVStage runtime state.

See `ovstage-population`, `ovstage-data-plane`, and
`ovstage-ovrtx-integration` for the focused workflows.
