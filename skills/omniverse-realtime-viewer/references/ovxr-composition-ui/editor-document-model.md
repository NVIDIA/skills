# Editor panels, toolbar and settings model

Use this application model to build the complete editor UI without importing
a sample's widgets. It is independent of Skia, ImGui or a particular shell.
Read [spatial surfaces](spatial-surfaces-and-interaction.md) for pose/capture and
[async raster](async-raster-pipeline.md) for publication.

## Authoritative records

Keep these distinct, with stable IDs and generation-scoped references:

| Record | Contents | Invalidation |
| --- | --- | --- |
| Scene snapshot | Stage generation, prim paths, hierarchy, type/visibility, selectable leaf mapping | Stage replacement; affected metadata change |
| Selection | None, scene prim, or panel handle; selected target and revision | Replacement, target removal, explicit clear |
| Surface | ID, visible/enabled, LOCAL or world anchor, physical size, pose revision | Hide, anchor loss, context replacement |
| Document | Surface ID, rows/widgets/actions, theme, raster dimensions, document revision | Semantic/layout/style change, not pose movement |
| Capture | Source/hand, target/node, begin transform, input epoch and stage generation | Release/cancel, owner or lifecycle loss |
| Render policy | Canonical requested settings, revision, staged/applied/error state | Acknowledged owner mutation; applied state resets on new products |

Persist deliberate user settings/placement, not borrowed handles, captures or
hover. Load preferences through the same validation as live commands. Keep the
source scene immutable and do not silently resurrect a missing saved prim.

## Stage tree and properties

Extract the composed hierarchy once per stage generation using supported
OVStage queries or an isolated USD worker. Record provenance: a source snapshot
does not automatically observe later runtime edits. Overlay app-owned changes
or invalidate affected entries. Do not traverse USD or read Fabric every frame.

Flatten only expanded nodes and virtualize the visible row range. Preserve
stable IDs such as `(stageGeneration, absolutePrimPath)` when scrolling or
reordering; row index is layout, not identity. Yoga produces both paint and hit
bounds. Keep clipping/scroll offsets identical for each. Properties requests
carry selected prim and generation; discard responses for a replaced selection.

A row press or scene-ray activation replaces the single selected prim directly,
clearing selected panel/drag state. The same accepted selection drives outline,
properties and gizmo placement. Clicking a different prim must not first toggle
the old one off and require another click. Same-target toggle is optional policy.

## Permanent grab bars and toolbelt

Each movable surface owns a persistent grab bar below its content, with its own
stable hit ID. A bar press selects the surface and changes its highlight; the
normal fist/controller transform then captures it without snapping. Selecting
a scene prim clears the selected bar. Keep invisible hit enlargement clear of
content controls and use the same pose for the raster, chrome and hit geometry.

The toolbelt is a set of semantic actions with enabled/selected/pending states,
not a second selection service. Use real vector/glyph coverage for labels and
analytic AA for procedural edges; see [depth tools](depth-tools.md). Maintain
one action dispatcher across desktop controls and headset widgets. A tooltip
or highlight can be an analytic pose/state update instead of a full text repaint.

For competing left-pinch meanings, follow the input arbitration contract before
dispatching actions; drawing a handle does not itself grant it input ownership.

## Settings controls

Group AR controls into Passthrough, Materials & shadows, and Exposure using
[AR policy and support state](../ovxr-library-runtime/live-ar-controls.md). Expose
pending, staged, applied and error honestly. A UI click enqueues a command; only
the render owner writes products. Keep one mutation in flight or explicitly
coalesce, merge partial requests against the canonical revision, and display
the acknowledged state. Old status responses must not overwrite a newer ack.

Expose gesture tuning separately: pinch start/release thresholds, bounded hold
grace, wrist-facing gain/range, positional/directional filter parameters, and
panel reach gain. Validate units and relational constraints (release is more
permissive than start; gain transition end exceeds start). Apply changed capture
parameters on the next capture, or explicitly cancel/reset the current one;
do not change the reference basis mid-drag without a visible lifecycle choice.

## Closure

Test scrolling/reordering during capture, selection replacement through both
desktop and headset, removed targets, rejected settings, stage reload and
worker delay. Pose-only movement leaves document and texture revisions stable.
One screenshot does not prove the selection/command/renderer round trip.
