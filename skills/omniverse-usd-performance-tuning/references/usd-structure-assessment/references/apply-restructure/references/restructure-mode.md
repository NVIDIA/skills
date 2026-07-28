# Restructure Mode

Use this reference for `apply-restructure` mode=`restructure`, invoked when
`restructure-decision` selects the `extract-as-assets` or
`decompose-for-selective-loading` branch.

## Internal-Reference Scan

Before finalizing boundaries, scan for internal `Sdf.Reference` objects with an
empty `assetPath` whose `primPath` escapes the candidate boundary. CAD/BIM
exports often place instance prims under a level or discipline and canonical
meshes/materials under sibling scopes such as `/A/Prototypes` or `/A/Looks`.

If an internal reference escapes the boundary, choose one branch and record it
in the dry-run plan:

- Promote the shared dependency to its own layer and sublayer it where needed.
- Inline the dependency into every boundary that needs it.
- Abort and recommend `optimize-as-is` when the dependency graph is too tangled
  to split cleanly.

## Input Validation

Confirm:

- `input_stage` exists and opens.
- `output_dir` exists and is not the input stage directory.
- Every boundary prim path exists.
- `dry_run=true` emits a report and writes no USD files.

## Dedupe Plan

When the plan includes `dedupe`, the mid-level instanceable-reference collapse is
authored **per descent region by the native `deduplicateHierarchies` op**, driven
by the identity-first frontier — not by a single global cut and not by a
hand-authored rewrite of the whole candidate report. The frontier decision core
(`usd-hierarchy-dedupe-candidates/scripts/select_frontier.py`) selects, at each
level, the `paths` set and a **small per-region `maxDepth`**; the per-level driver
invokes `deduplicateHierarchies(paths=[node], maxDepth=<small, per-region>)` on
each descent node and recurses (the native form of Phase 2g's "re-run boundary
inference on each extracted asset"). The op authors the internal instanceable
references; the driver owns the identity gate and emits the restructure-role
manifest (`kept_inline_for_merge` tagging, `descent_converged`,
`final_rescan_new_groups_above_floor`). Do **not** set one stage-wide `maxDepth`
and run once — named units sit at varying depths, so no single global cut lands on
the ragged frontier (see the `deduplicateHierarchies` catalog entry in
`operations/operations.json` for the measured depth→grain evidence). The
hand-authored `Sdf` path stays permitted for what the native op does not cover:
external-prototype materialization into payload files (below) and any boundary the
op cannot author cleanly.

While materializing boundaries:

- Use the candidate report from `usd-hierarchy-dedupe-candidates`; the frontier
  (`select_frontier.py`) chooses the grain identity-first (`kind` → naming →
  semantics; the hash only confirms reuse).
- Keep only user-approved, non-overlapping candidate groups.
- **Author a bottom-up nested library by default** (the "author nested, not flat"
  rule in `workflow.md` Phase 2g): parent prototypes *reference* child prototypes
  rather than inlining them. Flat / outermost-only sharing is insufficient for
  multi-file disk recovery. Honor the MINP inclusion floor, keep sub-floor leaves
  inline for later merge, and on a mostly-identical-with-outliers group share the
  majority and recurse only into the variant's differing branches.
- **Mesh merge is a WITHIN-prototype prim-count reduction, not a sharing move.**
  It is a first-class Phase-4 step that EXECUTES the manifest `merge` disposition
  (not a "someday" option): run `merge` *inside* a prototype (merge once, benefit
  N instances), never across an instance boundary. The payoff is a
  scene-graph win — cheaper stage-open + composition/traversal + per-prim memory,
  plus fewer draw calls — NOT a disk win (bytes ~= sum; the crate already
  byte-dedups). It is intent-gated (it destroys per-part addressability) and gated
  on bounds coherence and weak/none identity, with a conditional vertex-weld tail
  whose reclaimed bytes are credited to the disk tier via the weld source. Full op-chain
  + eligibility: `mesh-merge-rewrite-spec.md` §9 (per-prototype op chain
  + merge-eligibility guard).
- **Read existing composition first**: when the input arrives already instanced or
  BIM/CAD-exported, treat existing prototypes as the candidate set at that level
  (collapsing byte-identical-but-separately-authored prototypes) and resume the
  descent there rather than restarting from the top (`workflow.md` Phase 2g,
  "resume the descent from the level the asset is already at").
- Prefer `external_prototype` unless the user explicitly chooses
  `internal_reference`.
- Inline local material bindings and UsdShade networks that cross the boundary
  unless the user asks to preserve shared material-library dependencies.
- Set `instanceable=true` only for sites that passed instanceability checks.
- Record skipped groups and reasons in the manifest.

## Instanced Asset Extraction

When the boundary plan records `goal: extract_as_assets`, apply
the dedupe rules above (shared prototype, `instanceable=true` for passing
sites) AND structure each site using the
[reference-payload pattern](https://docs.omniverse.nvidia.com/usd/latest/learn-openusd/independent/asset-structure-principles.html#structuring-an-asset-interface):
the site's interface prim is referenced into the assembly, and heavy content is
behind a payload arc internal to that asset.

**Required structure per duplicate site:**

Each site becomes a self-contained asset with interface/payload separation:

```
site_N.inter.usd       (interface layer — kind, assetInfo, extent hints)
  └─ payloads = [@./site_N.pay.usd@]   (payload arc to heavy content)

site_N.pay.usd         (payload layer — reference to shared prototype)
  └─ references = [@./shared_prototype.usd@]
      instanceable = true   (when instancing-readiness gate passes for this group)
```

On the assembly root, reference each site's interface layer. The assembly
consumer can then selectively load/unload each site via standard payload
controls without affecting other sites or the shared prototype.

See the [VFI guide: Factory-Level Structuring](https://docs.omniverse.nvidia.com/vfi/latest/guide/factory-level-structuring.html)
for the broader factory/facility assembly pattern this follows.

Execution order:
1. Write shared prototypes first (one per dedupe group).
2. For each duplicate site on the assembly root:
   a. Create the interface + payload layers following the reference-payload
      pattern above.
   b. Set `instanceable=true` on the payload root prim only when
      `instancing-readiness` (see `skills/omniverse-usd-performance-tuning/references/usd-structure-assessment/references/restructure-decision/README.md`
      §"instancing-readiness gate") passes for that site's dedupe group.
   c. Reference the site's interface layer from the assembly root.
3. For unique (non-duplicate) boundary candidates, extract as independent
   payloads (standard decompose behavior — same interface/payload pattern,
   without instancing).
4. Validate all outputs per §"Authoring Requirements" below.

## Boundary Materialization

For each boundary, copy the subtree into a new prototype layer and replace the
original subtree on the assembly root with a reference to that prototype. When
dedupe selected duplicate hierarchy groups, write one prototype per approved
group and rewrite every duplicate site to reference it.

### Cross-Boundary Material Bindings

Before extracting a sub-hierarchy as a standalone payload, scan prims inside
the extraction boundary for material bindings that reference prims OUTSIDE the
boundary (e.g. `/Root/Materials/Metal_01` while the payload only contains
`/Root/Floor_1/Cabinet_01/...`).

When the payload is opened standalone (for validation per "Post-Restructure
Validation Strategy" or for SO per-payload ops), cross-boundary bindings become
unresolvable dangling references. This silently breaks `optimizeMaterials`,
material-binding validators, and `deduplicateGeometry` material-index grouping.

Apply the boundary plan's `material_policy` (top-level field, not just inside
`dedupe`):

- `inline_local_external` (default): copy the bound material scope into the
  payload if it's defined in the same layer stack. The payload becomes
  self-contained.
- `preserve_external`: leave the binding as-is. Document that standalone open
  will have dangling refs — material validators must run on the composed stage,
  not per-payload standalone.
- `block_on_external`: halt and ask the user when cross-boundary materials are
  detected.

Use:

- `Sdf.Layer.CreateNew(path)`
- `Sdf.CopySpec(srcLayer, srcPath, dstLayer, dstPath)`
- `Usd.Stage.Open(layer)` and `prim.GetReferences().AddReference(asset_path)`
- `prim.SetActive(False)` only when deactivation is the chosen reversible
  alternative to deletion.

### Material-Network Closure at a Prototype Boundary

The native `deduplicateHierarchies` op collapses structurally-identical subtrees
and preserves their bindings, but it does **not** judge what counts as a *valid*
prototype boundary when a material network crosses that boundary. That decision
stays here: whenever a subtree is externalized as a prototype (native-op internal
reference, hand-authored external prototype, or point-instancer prototype), decide
material-network closure before the boundary is final.

Cross-boundary material relationships are common in CAD and digital twin assets:
duplicate equipment, furniture, or HVAC assemblies often bind geometry inside the
candidate subtree to materials under a shared `/Looks`, `/Materials`, or similar
scope outside that subtree. If those relationships are left pointing at the source
stage, the prototype is harder to validate, version, move, and optimize
independently.

When `material_policy=inline_local_external` (the default), inline local material
dependencies into each prototype:

1. For the canonical source subtree, collect authored material bindings and
   UsdShade connections whose targets are outside the selected subtree.
2. Treat material targets as inlineable when the target prim is part of the input
   stage or package and is not an explicit external material-library dependency.
3. Build the material-network closure for each inlineable material: the Material
   prim, Shader and NodeGraph descendants, and connected shader or nodegraph prims
   required by that network.
4. Copy each material network into the prototype, preferably under a stable child
   scope such as `/<PrototypeRoot>/Looks`.
5. Rewrite copied geometry bindings and copied shader connections so they target
   the inlined material-network paths.
6. Preserve texture and other asset-valued inputs, but validate that they still
   resolve from the prototype layer. If a relative asset path would stop resolving,
   rewrite it relative to the prototype layer or mark the group `blocked` until the
   dependency move is explicit.

Do not decide material equivalence by material prim name alone. If different copies
bind to different material paths, compare the material-network closure or split the
candidate group. If the material networks differ, skip the group or leave the
affected sites uninstanceable; do not silently collapse distinct looks.

When `material_policy=preserve_external`, keep external material targets and record
them in the manifest. When `material_policy=block_on_external`, block any selected
group with material bindings or shader connections that cross the prototype
boundary.

## Point-Instance Route

Use this for the reduction frontier's **`reduction_route = point_instance`**
decision (a scene-graph / draw-call win): replace many **anonymous, high-count
repeated** prims with a single `UsdGeomPointInstancer` — one (or a few)
prototype(s) plus `protoIndices`, `positions`, `orientations`, and `scales` arrays.

**It is identity-losing and intent-gated.** Geometry is preserved, but per-prim
path addressability collapses into instance indices. It is reserved for the
disposition matrix's weak-identity row only:

- **Eligible:** anonymous / `structural_fallback` units, in **very high counts**,
  with **no path-level addressability need** (bolts, fasteners, vegetation,
  repeated fixtures) — the `instancing-tradeoffs.md` "point instancer" row.
- **Never:** an addressable / `kind` / named / semantic subcomponent, anything
  articulated / physics-bearing / variant-bearing, or any unit a maintenance or
  service twin must select per-instance. The manifest contract (`identity_signal`
  in {kind, naming, semantic} with `reduction_route` point_instance) **fails** this
  gate — enforced in `usd-hierarchy-dedupe-candidates/scripts/select_frontier.py`
  and `validate_report.py` `validate_manifest_structure`.

It is **intent-gated for all archetypes** — no fidelity tolerance can bound an
identity loss (`operation-safety.md` § Apply authority). It is surfaced via the
Phase-7 iteration-2 opt-in menu (the identity-losing batch), never run
automatically. A `point_instance` candidate that has not been confirmed stays
`kept_inline_for_merge` instead (preserving within-prototype merge-ability).

**Mechanism.** For the confirmed, intent-gated route, the native
`deduplicateGeometry(duplicateMethod=4)` op authors the `UsdGeomPointInstancer`
directly (usd-optimize 1.0.4+; issue #169). Where the native op does not fit —
notably external-prototype materialization into payload files — the hand-authored
path stays permitted: author the `UsdGeomPointInstancer` directly via the USD API.
That hand-authored path is a **USD authoring route**, **not a Usd Optimize
operation**, so it does **not** require a `usdOptimize.operationsAvailable` check;
its only precondition is an importable `pxr` (USD Python) runtime. Either way:
open the target as its **own root layer** (edit-target invariant), partition
occurrences into one value-variant per prototype, author the instancer at a stable
parent path with the prototype under its `prototypes` rel, decompose each
occurrence's transform into `positions` / `orientations` / `scales` with its
`protoIndices` id, remove the now-redundant per-occurrence prims, recompute
extents, and persist with the compacting `Sdf.Layer.Export` + atomic replace (not
`Save()`).

**Material boundary.** This route inherits the same material-boundary problem as
hierarchy dedupe: bindings that cross the prototype boundary are silently dropped
when geometry moves into a Point Instancer prototype. Run the
[Material-Network Closure](#material-network-closure-at-a-prototype-boundary) step
above before rewriting. A PI rewrite that has not run that collection step is not
safe to apply.

**Preserve the pre-instanced source.** Keep the pre-instanced source layer so an
alternate merge-first (draw-call-bound) deliverable can still be produced.
Point-instancing and mesh-merge optimize different axes; pick per target intent
(memory-bound vs draw-call-bound). Record in the manifest `phase4_targets[]`:
`reduction_route: point_instance`, `identity_signal` (must be weak —
`none`/`structural_fallback`), `copy_count`, and the `arc_estimate` contrast. The
scene-graph win is `unverified-at-render` until a Kit/omniperf profile exists.

Do not use this route to convert addressable subcomponents (use references / the
nested library) or to animate populations (clip-driven Point Instancers are a
modeling choice, not this reduction route).

## Post-Restructure Placement Check (mechanism-agnostic guardrail)

Run the per-prim placement-drift gate after every restructure, **regardless of
which mechanism authored the references** — native `deduplicateHierarchies`,
hand-authored `Sdf`, or the point-instance route. It compares each mesh's
placement before and after and fails the restructure if any prim moved beyond
tolerance.

Keep it for two honest reasons, and **not** because the native op drifts:

- **The native op is transform-correct.** A dedicated drill-down measured it
  placement-correct to sub-micron — max per-mesh world-centroid displacement
  **0.00028 mm** across an **85,871-mesh automotive assembly**, by a path-based
  1:1 comparison of every mesh; rendered mesh count preserved at every depth. The
  op excludes the root `xformOp:*` from identity and preserves each duplicate's
  local transform by construction, so no hand-rolled transform-correct placement
  is needed.
- **Cheap insurance over a still-permitted path.** The gate is cheap, and the
  hand-authored `Sdf` paths remain permitted for what the op does not cover
  (external-prototype materialization, material-network closure, direct
  point-instancer authoring). A mechanism-agnostic check covers those paths too,
  so the guardrail earns its keep without overstating native-op risk.

## Edit-Target Invariant (never optimize through a reference)

Usd Optimize authors into the stage's **current edit-target (root) layer**.
If a prototype / library arrives via a reference and you run SO on the *composed
assembly*, the edits land as **overrides on the assembly layer** while the
referenced library keeps its heavy geometry — that is override bloat, not
reduction. Therefore:

- **Each library / sub-asset is opened as its OWN root layer** so SO's edit
  target *is* that file's bytes. "Optimize the assembly in one pass" is wrong.
  This is exactly the per-sub-asset parallel model the batch scheduler batches —
  one target = one own-layer file = one job.
- **De-class abstract `class` prototype namespaces (`Class → Def`) before the
  chain, and restore after.** Optimizing an abstract `class` namespace silently
  no-ops: a default-predicate stage walk returns 0 of its meshes (see the
  zero-work diagnostic below).
- **Every library file must resolve standalone** — its own material bindings and
  nested children reachable via explicit asset paths — to be a valid independent
  edit target.

## Authoring Requirements (Critical for Phase 4 Compatibility)

- `Sdf.CopySpec` preserves the source specifier. If copying from an over-only
  layer, the destination spec will also be Over — fix it after copy.
- Fresh specs from `Sdf.CreatePrimInLayer` default to `Sdf.SpecifierOver`.
  **You MUST set `Sdf.SpecifierDef` on every ancestor prim in the payload that
  is not brought in by composition (reference/sublayer).**
- Bare `Sdf.Reference(assetPath=...)` resolves to the target layer
  `defaultPrim`; set `defaultPrim` or pass `primPath`.
- Every extracted payload/prototype MUST have `defaultPrim` set to the root
  prim of the extracted sub-hierarchy.

### Why Specifier Correctness Is Critical

Usd Optimize operations that use USD's default-predicate prim traversal
(including `decimateMeshes`, `meshCleanup`, `fitPrimitives`, `removeSmallGeometry`)
will **silently skip** all meshes under Over-spec ancestors. The operation returns
`success=True` with zero work done, and the tool itself surfaces no error or warning
to signal the failure. Because these operations mutate or delete geometry, warn the
user and get explicit confirmation before running them, and proactively raise this
silent-skip risk yourself — do not proceed on the tool's silent `success=True` alone.

Operations that enumerate via material bindings or instance indices
(`deduplicateGeometry`, `removeUnusedUVs`, `optimizeMaterials`) may still work,
creating a confusing partial-success state.

### Verification (On Unexpected Zero-Work Results)

If a Phase 4 operation returns `success=True` with zero work on a target known
to contain geometry, check for Over-spec ancestors:

```python
from pxr import Usd, UsdGeom, Sdf

stage = Usd.Stage.Open(payload_path)
mesh_count = sum(
    1 for p in Usd.PrimRange.Stage(stage, Usd.PrimDefaultPredicate)
    if p.IsA(UsdGeom.Mesh)
)
if mesh_count == 0:
    # Promote Over specs to Def on all ancestors
    layer = stage.GetRootLayer()
    for prim in stage.Traverse():
        if prim.GetSpecifier() == Sdf.SpecifierOver:
            layer.GetPrimAtPath(prim.GetPath()).specifier = Sdf.SpecifierDef
    layer.Save()
```

This is NOT a routine post-write check — it is a diagnostic for the red-flag
pattern described in `operation-safety.md` §"SO Operation Returns Success With
Zero Work".

## Output Validation

Run the runner's minimum-openability check on every written USD. Record
`pass | fail | skipped` in the manifest and never delete failed outputs.

## Datasmith/Revit Shape

Typical monolithic exports have level scopes that internally reference shared
prototype and material scopes:

```text
/A
  /A/Level1
  /A/Level2
  /A/Prototypes
  /A/Looks
```

When every level depends on `/A/Prototypes` and `/A/Looks`, prefer promoting
those shared scopes to shared layers rather than inlining them into every
level. The shared layers are valid Phase 4 targets because optimizing them
propagates to every instance site.
