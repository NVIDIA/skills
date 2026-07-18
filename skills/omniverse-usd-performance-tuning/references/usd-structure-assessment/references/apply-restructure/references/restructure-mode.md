<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

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

When the plan includes `dedupe`, follow
`hierarchy-dedupe-rewrite-tool-spec.md` while materializing boundaries:

- Use the candidate report from `usd-hierarchy-dedupe-candidates`.
- Keep only user-approved, non-overlapping candidate groups.
- **Author a bottom-up nested library by default** (`hierarchy-dedupe-rewrite-tool-spec.md`
  §5a): parent prototypes *reference* child prototypes rather than inlining them.
  Flat / outermost-only sharing is insufficient for multi-file disk recovery.
  Honor the MINP inclusion floor, keep sub-floor leaves inline for later merge,
  and on a mostly-identical-with-outliers group share the majority and recurse
  only into the variant's differing branches.
- **Mesh merge is a WITHIN-prototype prim-count reduction, not a sharing move.**
  It is a first-class Phase-4 step that EXECUTES the manifest `merge` disposition
  (not a "someday" option): run `merge` *inside* a prototype (merge once, benefit
  N instances), never across an instance boundary. The payoff is a
  scene-graph win — cheaper stage-open + composition/traversal + per-prim memory,
  plus fewer draw calls — NOT a disk win (bytes ~= sum; the crate already
  byte-dedups). It is intent-gated (it destroys per-part addressability) and gated
  on bounds coherence and weak/none identity, with a conditional vertex-weld tail
  whose reclaimed bytes are credited to the disk tier via the weld source. Full op-chain
  + eligibility: `hierarchy-dedupe-rewrite-tool-spec.md` §9 (per-prototype op chain
  + merge-eligibility guard).
- **Read existing composition first** (§5b): when the input arrives already
  instanced or BIM/CAD-exported, treat existing prototypes as the candidate set
  at that level (collapsing byte-identical-but-separately-authored prototypes) and
  resume the descent there rather than restarting from the top.
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
`success=True` with zero work done — no error, no warning, no indication of failure.

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
