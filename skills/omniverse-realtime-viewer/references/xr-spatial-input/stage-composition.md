# Source-safe stage composition

## Ownership

The source USD is immutable application input. Cameras, render products,
controllers, rays, teleport markers, hand joints, and navigation overrides are
viewer-owned session data. Never write generated wrappers beside the source or
author changes back to its root layer.

Create content-addressed USDA wrappers in an OS temporary session-layer cache.
Use an absolute sublayer identifier for the source; references inside the source
still resolve relative to their authored layer. Write wrappers atomically and
reuse identical content.

OVStage must open the final composite through `population.open_usd`. Adding a
USD reference after population can lose RenderProduct relationships in Fabric.
Do not author a kitless `RenderSettings` prim.

## Generated prim layout

Keep viewer-owned roots distinct from source scene roots, for example:

- `/DesktopCamera` and `/Render/DesktopView` for the desktop product;
- `/AppXR/Controllers` for flattened controller/ray/teleport leaves;
- `/ViewerHands` for joint and fixture leaves;
- OVXR-owned eye products under OVXR's documented paths.

Generated renderable leaves begin hidden. A zero-scale display matrix is a
possible visibility convention for application geometry, never a valid user
anchor. Flattening generated renderable leaves with identity parents lets an
application write complete world matrices without ambiguous inherited transforms.
Publish geometry through the README-documented `update_prim_transforms`
operation with a coordinated ordinal, or the installed OVStage scene-write API; OVXR's
`update_camera_transforms` documents cameras and is not a general object API.
Do not prescribe it for controller bodies, hands or selected objects.

## Navigation and arbitrary stages

Use the current OVXR managed anchor for user movement/yaw/scale; see
[navigation](navigation.md). Source roots, their baseline transforms, and sibling
geometry stay unchanged, so arbitrary multi-root scenes move consistently from
the user's perspective without root discovery or scene-write permission.

The former inverse scene-root navigation strategy is a legacy fallback only
when the anchor API is absent. Do not combine it with managed-anchor movement.
Application object editing is separate: validate target ownership and preserve
source baselines in session state without saving the input USD.

Stage reload is a lifecycle boundary: detach the OVXR context, populate the new
composite, rediscover roots/metadata, reset input generations and generated-leaf
publication state, rebuild scene-query geometry once, then reattach. Preserve
the long-lived renderer when the library contract permits it.

## Validation

Hash the source file before and after composition and interaction tests. Verify:

- no new files appear in the source directory;
- the generated wrapper sublayers the absolute source path;
- source default prim, metres-per-unit, up-axis, and base root matrix are
  preserved;
- controller/hand leaves contain valid geometry and start hidden;
- user navigation changes the managed anchor without modifying source roots;
- a reload clears stale action/capture/BVH state.

For stage01, test a non-identity `/World` baseline and a non-origin viewer pivot
so identity-only or origin-only implementations cannot pass accidentally.

## Building hierarchy and world-pose lookup without sample code

Use [stage hierarchy](../stage-hierarchy/README.md),
[the shared data plane](../ovstage-data-plane/README.md) and
[transform safety](../prim-transform-safety/README.md) for current native
queries and derived-world computation; their dependency routes own exact
signatures. Read the intended committed state and validate row count, finite
matrices and path correspondence. A hierarchy relation result does not itself
author world transforms, and a missing computed attribute does not authorize a
fabricated identity.

A portable source-only fallback uses an isolated USD worker:

```python
# Run in the worker interpreter, never in an ABI-conflicting render process.
from pxr import Usd, UsdGeom
stage = Usd.Stage.Open(final_composite_path)
cache = UsdGeom.XformCache(Usd.TimeCode.Default())
prim = stage.GetPrimAtPath(requested_absolute_path)
if not prim or not prim.IsActive():
    raise ValueError("missing or inactive spawn prim")
world = cache.GetLocalToWorldTransform(prim)
matrix16 = [float(world[r][c]) for r in range(4) for c in range(4)]
```

Return this matrix with the exact composite identity, stage generation, path,
metres-per-unit, up-axis and provenance. The parent validates all of them. Cache
source hierarchy per generation; for app-owned edits retain authoritative
committed transforms or invalidate affected cached ancestors/descendants.
Animated or externally edited source poses require current native data or a new
snapshot, not stale worker values. Default-anchor selection reads this complete
world matrix and copies it into a managed anchor; it never writes it onto the
source prim's local xform.
