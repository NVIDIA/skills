<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Mesh-Fragmentation Suggester — Behavioral Specification

Status: draft (rev 1)
Audience: a coding agent running or extending `mesh_fragmentation_finder.py`.
Style: behavior-only.

## 1. Purpose

Surface parents whose children look like a CAD/BIM/EDA converter
**face-explosion** — a flat fan of anonymous, same-material `Mesh` prims under a
named/`kind`-tagged unit — and **suggest** merging that fan up to the named
boundary, grouped by material. It is the cheap entry point for the
`reduction_route = merge` landing (`mesh-merge-rewrite-spec.md`).

It is **read-only** and **decides nothing**. Merge is intent/archetype-gated, so
the suggester only surfaces and routes; the user (or the archetype default)
confirms. It is **not** a geometric engine: it does **no** vertex-coincidence
computation and applies **no** numeric merge threshold (OQ12 — proving the weld is
an O(all-points) pass and pointless for detection, since the merge op welds
anyway).

## 2. Inputs (cheap or already computed)

- The Phase-2c **`perf_small_mesh`** validator finding — the tininess signal and
  the entry point (free; the validator already ran). Passed as
  `small_mesh_paths`.
- Cheap structural reads per candidate parent (`scan_stage`, pxr): direct child
  count + type uniformity, anonymous child naming (`Mesh_N`), the parent's
  `kind`/name, and the distinct bound materials across the fan (a
  binding-relationship target read — **no geometry, no points**).
- Optionally `instanced_paths` — the subtrees the instance-candidate finder
  already claims (§5).

## 3. The signals (surface, don't gate)

A parent is surfaced when the **qualitative pattern** holds — *meaningless
children under a meaningful unit*:

- a **flat fan** (mesh children dominate the direct children — low nesting);
- **anonymously named** children (`Mesh_N`, numeric/uuid tokens) — reference
  designators (`U302`) and semantic names are NOT anonymous;
- under a parent that **carries identity** (`kind` ∈ {assembly, group, component,
  subcomponent}, or a semantic name); and
- a **high mesh:material ratio** (many meshes, few distinct materials — re-
  stitchable into one mesh per material).

The `SUGGEST_MIN_FAN`, `*_HINT` knobs are **surfacing/ranking heuristics, not
merge gates**: they decide what to *show* and in what order, never whether a shown
fan may merge. The merge decision is the user's archetype-gated confirmation.

## 4. Output

For each surfaced parent, one suggestion: the **merge boundary** (the named/`kind`
ancestor to preserve), the `identity_signal` that keeps it addressable, the
`identity_disposition` (`weak` — the anonymous fan), the per-material grouping
(`merge_groups = distinct_materials`, `geomsubset_fallback` when > 1), and the
human-readable suggestion. Plus `routed_small_geometry`: the **one real decision**
— `perf_small_mesh` members inside a surfaced fan are real faces → **merge
(re-stitch)**; the rest are negligible → **removeSmallGeometry (delete)**. It does
not route a fan to delete and does not merge a scattered tiny mesh.

The confirmed group feeds `mesh-merge-rewrite-spec.md` (which owns the
(scope × material) execution and the eligibility guard).

## 5. Division of labor with the instance-candidate finder

The two target different things and must never claim the same prims twice:

- the **instance-candidate finder** finds **repeated subtrees** to make
  `instanceable` (reference reuse, identity preserved);
- this suggester finds **fragmented same-material fans** to **merge** (re-pack,
  identity destroyed).

**Precedence:** a fan that is *also* a repeated subtree is **instanced at the
component first, then its faces merged INSIDE the prototype** (merge once, benefit
N instances). When `instanced_paths` overlaps a surfaced fan, the suggestion is
**annotated** (`composes_with_instance_candidate: true`) rather than dropped — the
two compose; they do not compete. This mirrors the decision-order-≠-execution-
order rule in `mesh-merge-rewrite-spec.md` §6.

## 6. Non-goals

- No merge execution (that is `mesh-merge-rewrite-spec.md`).
- No vertex-coincidence / weld-ratio computation, no numeric merge threshold.
- No identity dissolution decision — it surfaces and routes; the user confirms.
