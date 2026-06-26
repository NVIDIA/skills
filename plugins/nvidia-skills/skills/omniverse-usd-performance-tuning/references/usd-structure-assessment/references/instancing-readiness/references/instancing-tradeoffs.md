<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# USD Instancing and Dedupe - Tradeoffs and Decision Tree

> The actionable decisions in the agent flow are split between `usd-hierarchy-dedupe-candidates` (find candidate subtrees, read-only) and `instancing-readiness` (per-candidate safety gate). This reference holds the deeper trade-off framing, merge safety policy, and findings taxonomy that both skills cite.

---

## Purpose

Guide decisions between scenegraph instancing, point instancing, hierarchy dedupe, mesh-level dedupe, and merge operations for repeated digital twin content.

This reference is consulted by:

- `usd-hierarchy-dedupe-candidates` when choosing between hierarchy dedupe vs unscoped mesh dedupe.
- `instancing-readiness` when explaining merge safety to the user before authoring `instanceable=true`.
- `apply-restructure` when planning Phase 2f restructure orchestration.
- `usd-optimize-interpret-validators` when recommending merge or dedupe ops based on validator findings.

## Prerequisites

- Composition or structure context for repeated assets, payloads, variants, and edit boundaries.
- Current performance signals such as prim count, mesh count, draw-call pressure, or validator findings.
- User constraints for editability, semantic part identity, streaming, and visual-review tolerance.

## Limitations

- This is decision guidance only; it does not run Usd Optimize operations or rewrite the stage.
- Mesh-level dedupe does not collapse copied hierarchies or create shared asset boundaries by itself.
- Point instancing and mesh merge reduce editability, so they need explicit fit with the user's workflow.

## Troubleshooting

- If `instanceable=true` gives no benefit on copied local hierarchies, rewrite duplicates as references or payloads first.
- If unscoped mesh dedupe would touch very large mesh counts, prefer hierarchy candidates, explicit prototypes, or scoped mesh paths.
- If merge crosses composition boundaries or semantic parts, keep it out of the recommendation unless the user explicitly accepts that tradeoff.

## Examples

- "Decide whether repeated racks should use references, point instancers, or mesh dedupe."
- "Review merge risk before running deduplicateGeometry on a factory stage."

## Decision tree

Repeated full assets:

- Prefer references or payloads to one prototype asset.
- Mark referenced or payloaded prims `instanceable=true` when the prototype is identical and read-only instance behavior is acceptable.
- Do not expect `instanceable=true` to help copied local hierarchies that duplicate mesh data.

Large numbers of small repeated objects:

- Prefer `UsdGeomPointInstancer` for bolts, fasteners, vegetation, repeated fixtures, and similar small objects.
- Keep per-instance variation constraints explicit; point instancers reduce editability.

Duplicated hierarchies:

- Detect repeated subtrees by source names, asset metadata, or subtree hashes.
- Rewrite duplicates as references to one prototype before relying on mesh dedupe.
- Run mesh-level dedupe after hierarchy reuse has been established.
- Use `usd-hierarchy-dedupe-candidates` for a read-only candidate pass when the stage is monolithic, has copied assemblies, or has high mesh count with little or no instancing.

Duplicate mesh data:

- Usd Optimize dedupe can help at the mesh-data level.
- It does not collapse entire repeated hierarchies by itself.
- Avoid whole-stage mesh dedupe on very large mesh counts unless the user explicitly accepts a long run. Prefer explicit prototypes, authored sub-assets, or scoped `meshPrimPaths`.
- If a stage has ~50K+ meshes and no instancing, treat unscoped `deduplicateGeometry` as high-risk for customer friction.

## The boundary / disposition matrix (identity × reuse)

Once boundaries are recovered **identity-first** (authored `kind` → meaningful
name → semantic recognizability; the hash only *confirms* reuse — see
`usd-structure-assessment/README.md` §2.5 and `workflow.md` Phase 2g), each
candidate unit's disposition follows from two questions: *does it have identity?*
and *does it repeat, and how?*

| Identity (kind / name / semantics) | Reuse | Disposition |
|---|---|---|
| Strong — named component / subcomponent | repeats ≥2, single variant | **Externalize once, reference** (`inherits` + `instanceable`) — the default; preserves name, selectability, override, serviceability |
| Strong | repeats, with a few real variants | **One prototype per genuine variant**; recurse into the differing branches only |
| Strong | unique, but contains shared children | **Keep as an addressable container; reference its shared children** (nested library) |
| Strong | unique, no shared interior | **Keep local; clean in place** — no sharing |
| Weak — anonymous, identity not wanted | repeats in very high counts | **Point instancer** — compact, but *only because* identity isn't wanted here |
| Weak — identity-free, adjacent | n/a | **Merge** — draw-call win; only when per-part identity is genuinely irrelevant |
| Weak — below the inclusion floor | any | **Leave inline** for a later merge pass |

The two "weak identity" rows are the **only** places identity may be destroyed,
and both are deliberate, intent-gated choices — never the default, and never
applied to anything identity signals marked as a real part.

### What each disposition costs (OpenUSD content-reuse guidance)

The choice between reference, point instancer, and merge is a choice about what you
keep and what you spend. Citing the OpenUSD content-reuse guidance:

- **Reference (scenegraph instance).** Keeps **name, selectability, override, and
  serviceability** — the part stays a part you can point at, override per-instance,
  and swap. Costs roughly *(one prototype per variant) + (one reference per
  occurrence)* in composition arcs. This is the default for anything with identity.
- **Point instancer.** Compact and cheap to compose for very high counts, but pays
  in **"flexibility, addressability and legibility"** — instances are array
  elements, not named prims, so you lose per-part selection and override. Reserve it
  for anonymous high-count repeats (fasteners, vegetation, bolts) where identity is
  genuinely unwanted. Never for an addressable subcomponent. This is the
  `reduction_route = point_instance` landing; it is authored via the
  point-instancer authoring route
  (`apply-restructure/references/point-instancer-rewrite-spec.md`)
  and is intent-gated.
- **Merge.** Buys **draw calls** by fusing meshes, but **destroys per-part
  identity** and crosses boundaries. Only for identity-free static fragments.

Two constraints govern how these are applied:

- **Share at the coarsest unit that captures the reuse — the named subcomponent,
  never the individual mesh.** Mesh-level sharing explodes anonymous arcs and throws
  away identity; the same reuse at the subcomponent level needs orders of magnitude
  fewer arcs and leaves every part addressable. (Reuse recovery may then justify
  descending to finer *named* subcomponents — never to meshes.)
- **Instancing and merging fight at the instance boundary.** Once you instance a
  tiny part you cannot merge it into its parent without un-instancing first, so
  instancing finely *bakes in* a granularity a later draw-call pass must tear down.
  **Instance coarsely; do any mesh merging *inside* the shared prototype** (merge
  once, benefit on every instance); keep the pre-instanced source for a possible
  merge-first variant.

Sharing (the matrix) is orthogonal to **data reduction**: any unit you keep or
share can additionally have its geometry reduced within its fidelity band without
touching identity — drop unused primvars, index primvars, decimate, or **refit a
mesh that is really a primitive to that primitive** (box / cylinder / cone). On
prismatic CAD/BIM geometry, primitive-fitting and unused-primvar removal are
frequently the dominant *disk* levers even after sharing is done.

## Merge safety

Do not recommend mesh merge when:

- The stage is already heavily scenegraph-instanced.
- The repeated content should become point instanced instead.
- Geometry streaming is in use.
- Editability or semantic part identity must be preserved.
- The merge target crosses payload, reference, or variant boundaries without explicit approval.

Consider merge when:

- The bottleneck is draw-call or prim-count overhead.
- The content is static.
- Materials and spatial clustering make the merge coherent.
- Before/after validation and visual review are part of the plan.

**Group the fan by `(scope × material)`.** You can only fuse meshes that share a
material, or fuse into one mesh carrying a per-material `UsdGeomSubset`. So within
the merge boundary (the nearest named/`kind` ancestor, preserved) gather the
same-material fan and fuse it to one `Mesh` per material; when a few materials must
coexist in a single prim, fuse into one `Mesh` + a `UsdGeomSubset` per material so
bindings survive — and stop when the per-subset overhead approaches the per-mesh
overhead it replaced. The grouping/execution mechanic and the archetype-gated
merge depth are specified in
`usd-structure-assessment/references/apply-restructure/references/mesh-merge-rewrite-spec.md`,
surfaced by `usd-structure-assessment/references/usd-mesh-fragmentation-candidates/`.

## Findings taxonomy

When emitting findings (e.g. from `usd-hierarchy-dedupe-candidates` or `usd-optimize-interpret-validators`), use these tags so downstream references can route consistently:

- `copied-hierarchy-candidate`
- `reference-instancing-candidate`
- `point-instancer-candidate`
- `mesh-dedupe-candidate`
- `mesh-fragmentation-candidate` — a converter face-explosion fan to merge by `(scope × material)`
- `merge-risk-instanced-content`
- `merge-risk-geometry-streaming`

## Handoff to Usd Optimize

Only hand off dedupe or merge operations after:

- Composition audit identifies repeated content boundaries.
- Hierarchy-level duplication has been assessed or ruled out.
- Edit target planner chooses output isolation.
- Validation has no structural blockers.
- The operation package includes whether the target is mesh-level or hierarchy-level.

## References

Before assessing instancing opportunities, read:

- `skills/omniverse-usd-performance-tuning/references/usd-structure-assessment/references/asset-structure-principles.md` - instancing granularity, variant/primvar compatibility, the reference-payload pattern.
- `skills/omniverse-usd-performance-tuning/references/usd-structure-assessment/references/factory-level-structuring.md` - instance at rigid-body level, deduplication informs granularity.

If you have network access, prefer the live URLs (noted in each reference file) for the most current version.
