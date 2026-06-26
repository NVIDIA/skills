<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Point-Instancer Rewrite — Behavioral Specification

Status: draft (rev 1)
Audience: a coding agent authoring the `reduction_route = point_instance` landing.
Style: behavior-only.

## 1. Purpose and when it applies

This is the authoring route for the reuse/descent frontier's **`reduction_route =
point_instance`** decision (a scene-graph / draw-call win). It replaces many **anonymous, high-count
repeated** prims with a single `UsdGeomPointInstancer` — one (or a few)
prototype(s) plus `protoIndices`, `positions`, `orientations`, and `scales`
arrays. This is a **USD authoring route** (author the
`UsdGeomPointInstancer` directly via the USD API), not a Usd Optimize
operation — exactly as `hierarchy-dedupe-rewrite-tool-spec.md` owns the direct
nested-library authoring while `deduplicateHierarchies` covers only the
instanceable-reference collapse (nested on 1.0.4; no manifest/identity
contract). Because
it is not an SO op, it does **not** require a `usdOptimize.operationsAvailable`
check; the precondition is an importable `pxr` (USD Python) runtime.

**It is identity-losing and intent-gated.** Geometry is preserved, but per-prim
path addressability collapses into instance indices. Therefore it is reserved
for the matrix's weak-identity row only:

- **Eligible:** anonymous / `structural_fallback` units, in **very high counts**,
  with **no path-level addressability need** (bolts, fasteners, vegetation,
  repeated fixtures) — the `instancing-tradeoffs.md` "point instancer" row.
- **Never:** an addressable / `kind` / named / semantic subcomponent, anything
  articulated / physics-bearing / variant-bearing, or any unit a maintenance or
  service twin must select per-instance. The manifest contract (`identity_signal`
  in {kind, naming, semantic} with `reduction_route` point_instance) **fails**
  this — see `validate_report.py` `validate_manifest_structure`.

## 2. Gating

The point-instancer authoring route is **intent-gated for all archetypes** — no
fidelity tolerance can bound an identity loss (`operation-safety.md` § Apply
authority). It is surfaced via the Phase-7 iteration-2 opt-in menu (the
identity-losing batch), never run automatically. A `point_instance` candidate
that has not been confirmed stays `kept_inline_for_merge` instead (preserving
within-prototype merge-ability; see the instancing-granularity-vs-merge rule).

## 3. Inputs

- `input_stage` / target: opened as its **own root layer** (edit-target
  invariant — never the composed assembly).
- `instance_group`: the approved set of sibling prim paths to collapse (one
  value-variant; partition first, one prototype per genuine variant).
- `prototype_choice`: which member becomes the prototype (default: the first
  approved path).

## 4. Rewrite

1. Author a `UsdGeomPointInstancer` at a stable parent path.
2. Move/author the chosen prototype geometry under the instancer's
   `prototypes` rel (prototypes are children of the PointInstancer so
   de-activation cascades).
3. For each occurrence, append its world/local transform decomposed into
   `positions` / `orientations` / `scales`, and its prototype id to
   `protoIndices`.
4. Remove the now-redundant per-occurrence prims.
5. Recompute extents; persist with the compacting `Sdf.Layer.Export` + atomic
   replace (not `Save()`), as for every Phase-4 target.

## 5. Preserve the pre-instanced source

Keep the pre-instanced source layer so an alternate merge-first
(draw-call-bound) deliverable can still be produced. Point-instancing and
mesh-merge optimize different axes; pick per target intent (memory-bound vs
draw-call-bound).

## 6. Reporting

Record in the manifest `phase4_targets[]`: `reduction_route: point_instance`,
`identity_signal` (must be weak — `none`/`structural_fallback`), `copy_count`,
and the `arc_estimate` contrast. The scene-graph win is `unverified-at-render`
until a Kit/omniperf profile exists.

## 7. Non-goals

- Converting addressable subcomponents (use references / nested library).
- Animating populations (object-handling Point Instancers driven by clips are a
  modeling choice, not this reduction route — see `factory-level-structuring.md`).

## Material bindings

This route inherits the same material-boundary problem as hierarchy dedupe:
bindings that cross the prototype boundary are silently dropped when geometry
moves into a Point Instancer prototype. Before rewriting, collect bindings
exactly as `hierarchy-dedupe-rewrite-tool-spec.md` §6 (material inlining
policy) prescribes, and apply the same inline/preserve/block decision per
prototype. A PI rewrite that has not run the §6 collection step is not safe to
apply.
