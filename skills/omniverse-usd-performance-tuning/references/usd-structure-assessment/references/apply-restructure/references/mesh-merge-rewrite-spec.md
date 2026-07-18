<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Mesh-Merge Rewrite — Behavioral Specification

Status: draft (rev 1)
Audience: a coding agent authoring the `reduction_route = merge` landing.
Style: behavior-only.

## 1. Purpose and when it applies

This is the authoring route for the reduction frontier's **`reduction_route =
merge`** decision (a scene-graph / draw-call win). It fuses a **fragmented fan of small sibling meshes**
into far fewer `Mesh` prims, cutting the rendered (unique) mesh-prim count and the
scene-graph weight (per-prim composition, traversal, stage-open, and per-mesh
renderer overhead). It is the counterpart to
`point-instancer-rewrite-spec.md`: that route collapses **repeated** subtrees by
reference; this route collapses **distinct, same-material fragments** that recur
only because a CAD/BIM/EDA converter authored one prim per modeled face.

The win is the **count of meshes, not bytes**. A merge concatenates geometry
(stored bytes ≈ the sum of the members, and the crate already byte-deduplicates
identical arrays within a layer), so merge makes **no disk claim** — it is scored
on the scene-graph axis, flagged `unverified-at-render` until a runtime
profile confirms it. Only the optional coincident-seam **vertex weld** the merge
enables is a small, legitimate disk credit (`disk_win_source = vertex_weld`),
never attributed to the merge itself. See `tools/oracle/score_run.py` and §9 of
`hierarchy-dedupe-rewrite-tool-spec.md`.

**It is geometrically lossless but identity-destroying.** The rendered result is
unchanged (the same triangles, re-packaged), but a merged mesh can no longer be
selected, overridden, or serviced as its constituent parts. **Identity is the
cost** — exactly the disposition-matrix `merge` row — so it is reserved for the
weak/none-identity row and is intent-gated (§2).

This is a **USD authoring outcome**, realized either by the cataloged Scene
Optimizer `merge` op (run scoped, §4) or by direct `Sdf`/`UsdGeom` authoring when
the op does not fit. Like the point-instancer route, the precondition is an
importable `pxr` runtime; the Usd Optimize path additionally requires `merge` in
`usdOptimize.operationsAvailable`.

## 2. Gating — archetype-gated merge depth

Merge is **intent- and archetype-gated**; how deep it may go is a **product
decision**, not a technical one (how much component-level identity may be
dissolved without asking). The default is **conservative**:

- **Default (all archetypes):** merge only **clearly identity-free leaf
  fragments** — the anonymous per-face/per-feature shards (`Mesh_N`-named
  tessellation pieces, pads, shells) that nobody addresses individually. Named or
  `kind`-tagged components are **preserved**.
- **`render` / `visualization` archetype:** may merge **aggressively up to the
  named-part boundary** (the nearest `kind`/named ancestor), because a
  render-only twin never addresses the parts apart.
- **`service` / `BOM` / `simulation` archetype:** keeps **component identity** and
  merges only **sub-component tessellation shards** below it. Never dissolves a
  reference-designator (`U302`, `R14`) or any unit the consumer selects.

A `merge` candidate that has not been confirmed for its archetype stays
`kept_inline_for_merge` (preserving the option) — it is surfaced via the Phase-7
iteration-2 opt-in menu (the identity-losing batch), never run automatically. No
fidelity tolerance can bound an identity loss (`operation-safety.md` § Apply
authority).

## 3. Inputs

- `input_stage` / target: opened as its **own root layer** (edit-target
  invariant — never the composed assembly; see `restructure-mode.md`). For
  instanced content, merge runs **inside the prototype** (merge once, benefit N
  instances), so the target is the prototype asset, never the composed stage.
- `merge_group`: the approved fan of sibling mesh paths to fuse, already
  partitioned by the **(scope × material) key** (§4) and confirmed weak/none
  identity. The fragmentation suggester
  (`usd-mesh-fragmentation-candidates/`) proposes these; the user/archetype
  confirms.
- `merge_boundary`: the nearest ancestor with real identity (the `kind`-tagged /
  named `component`) — preserved, and the ceiling the merge must not cross.

## 4. The merge unit — (scope × material), with a GeomSubset fallback

You can only fuse meshes that **share a material**, or fuse into one mesh that
carries a per-material `UsdGeomSubset` so bindings survive. The merge group key is
therefore **(bounded scope × material)**:

1. **Bound the scope at the merge boundary.** Within the nearest identity-bearing
   ancestor (the `component`), gather its anonymous descendant `Mesh` fan. Never
   reach across a sibling component, and **preserve the boundary prim and
   everything at or above it** — you destroy only the meaningless per-face
   pseudo-identity below it.
2. **Partition the fan by bound material.** Each material partition fuses to **one
   `Mesh`**. Bindings and bounds are preserved; introduce no dangling.
3. **GeomSubset fallback when materials must coexist in one prim.** If the
   boundary should collapse to a single `Mesh` (to take the prim count to 1) but
   spans several materials, fuse into one `Mesh` and author a `UsdGeomSubset`
   (familyName `materialBind`) per material, re-binding each subset. This keeps
   one prim while preserving every material assignment.
4. **Stop when GeomSubset overhead erodes the win.** If a scope spans many
   materials, the per-subset overhead approaches the per-mesh overhead it
   replaced — leave it one mesh per material (step 2) or do not merge. The merge
   exists to cut prim/mesh count; a one-prim result with dozens of subsets has not.

The skill applies **no numeric threshold** to gate the merge and does **no**
up-front vertex-coincidence test (the weld is realized by the op, not measured to
decide — see `OQ12` and §6). The (scope × material) key and the user's
archetype-gated confirmation are the whole decision.

## 5. Rewrite

Prefer the cataloged Usd Optimize `merge` op where it fits; fall back to direct
`Sdf`/`UsdGeom` authoring.

**SO `merge` path (preferred).** `merge` (Merge Static Meshes) is a real,
registered op (`operations/operations.json`, canonical, intent-gated,
`risk_class: high`).

**Run ONE scoped call PER named-component subtree — NEVER a single global call.**
For each merge boundary, pass `meshPrimPaths` = *only the anonymous fan inside that
one component*, partitioned by material, and iterate over components. Do **NOT**
run one stage-wide `merge`: a global call buckets meshes by material **across the
entire stage**, so each bucket's members span many different parents, and
`mergePoint=Parent` then resolves to their **common ancestor — the stage root**.
The result is anonymous `/<root>/merged*` blobs at the root and the named
`kind`-tagged components **emptied of their geometry** — identity destruction, not
just a size regression (observed: a global call dumped hundreds of `merged*` prims under
the stage root and pulled geometry out of its components). Scoping per component
keeps each fused mesh **nested under its own boundary** and the named parts keep
their geometry. The per-material scoping also stops the **default material-collapse**
from flattening distinct materials into one (an unscoped `merge` collapses all
bound materials, which is lossy). If the prototype namespace is an abstract
`class`, de-class it (Class→Def) so the op can author. Then run the conditional
tail: `merge → vertex-weld-where-contiguous → computeExtents` (§9 of
`hierarchy-dedupe-rewrite-tool-spec.md`).

**Direct-authoring fallback.** When the op does not fit (e.g. a GeomSubset-into-one
result), author the fused `Mesh` directly: concatenate the members' `points`,
`faceVertexCounts`, `faceVertexIndices`, and per-vertex/face primvars with the
correct index offsets; author per-material `GeomSubset`s if needed (§4.3);
re-bind materials; remove the now-redundant member prims.

In both paths:

- **Persist with a compacting `Sdf.Layer.Export(tmp) + atomic replace`, not
  `Save()`** (as for every Phase-4 target) — `Save()` appends without GC'ing the
  arrays the fuse orphaned and silently grows the file.
- **Recompute extents** and verify bounds are preserved.

**Eligibility guard (do not restate — apply it).** Before fusing, apply the
**merge-eligibility guard** in §9 of `hierarchy-dedupe-rewrite-tool-spec.md`:
weak/none identity only, and `merge_bounds_coherence ≤ K` (default 2.0) — a
dispersed merge balloons the AABB, harms the spatial structure, earns no scene-graph
credit, and must not be performed.

## 6. Preserve the pre-merge source

Keep the pre-merge source layer. Merge and point-instancing optimize different
axes and fight at the instance boundary (once you instance a part you must
un-instance it to merge). The reconciliation is **decision order ≠ execution
order**:

- **Decision:** choose instancing granularity FIRST — instance coarsely at the
  named component; never instance below the merge unit. A sub-unit left for a
  later merge is marked `kept_inline_for_merge`.
- **Execution:** merge runs INSIDE the chosen prototype and **before the
  geometry-dedup tail** (`deduplicateGeometry`/value-hash) — deduped faces would
  have to be un-instanced to merge. Merge once in the prototype; the win
  propagates to every instance site.

Keeping the pre-merge source lets an alternate re-addressing / merge-first variant
stay producible.

## 7. Reporting

Record in the manifest `phase4_targets[]`: `reduction_route: merge`,
`identity_signal` (must be weak — `structure`/`none`), `identity_disposition`,
the `merge_boundary` preserved, and the per-(scope × material) grouping.

Emit into the optimization report (`optimization-report.schema.json`):
`steps_applied` including the merge, `preservation.allow_merge = true` with
`rendered_mesh_merged_count` (so `rendered_mesh_count + merged == known`), and —
the **authoritative silent-loss invariant** — `preservation.rendered_triangle_count`.
A merge fuses prims and welds points, so the mesh count and the distinct point
count both legitimately change; **renderable triangles must not** (a re-pack
preserves every face), within the oracle's small `triangle_tolerance_pct` for the
zero-area degenerate faces a weld cleans. Emit `preservation.merge_locality_preserved`
(and `meshes_relocated_outside_boundary`): every fused mesh must stay UNDER its
named-component boundary — a merge that relocated geometry to a shallower ancestor
or the root fails preservation even if triangles balanced. Verify by checking each
merged prim's parent against its boundary; kind-prim *survival* alone is NOT enough
(the `kind` Xforms can persist while emptied of geometry). Also emit
`merge_identity_class` (weak/none), `merge_bounds_coherence`, and
`prototype_rendered_mesh_delta_pct`. The scene-graph win is `unverified-at-render`
until a runtime profile exists.

## 8. Non-goals

- **Identical tiny repeats** (the same bolt ×10,000) → a **PointInstancer**
  (preserves the prototype's identity; `point-instancer-rewrite-spec.md`), not a
  merge.
- **Tiny parts the consumer addresses** → leave them; prefer an identity-
  preserving mesh-count remedy (instancing of identical sub-parts, GeomSubset
  consolidation) over destroying identity.
- **Crossing a composition boundary** (payload / reference / variant) or a sibling
  component boundary without explicit approval.
- A disk-saving claim attributed to the merge (zeroed by the report's sharing-only
  guard — the merge is a scene-graph win; only the separate `vertex_weld` tail
  is a disk win).
