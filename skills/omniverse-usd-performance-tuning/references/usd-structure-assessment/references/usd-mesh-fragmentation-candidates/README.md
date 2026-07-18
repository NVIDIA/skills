# USD Mesh-Fragmentation Candidates

## When to Use

Use when a named unit holds a flat fan of anonymous, same-material `Mesh` prims —
the CAD/BIM/EDA converter "face-explosion" pattern — and that fan could be
re-stitched into one mesh per material. This is the cheap entry point for a
within-prototype mesh merge.

## Instructions

See `references/_shared/standard-instructions.md`.

## Output Format

See `references/_shared/standard-output-format.md`.

Use this after `usd-structure-assessment` and before a within-prototype `merge`
when a stage shows converter face-explosion.

## Purpose

Produce a read-only suggestion report for parents whose children look like a
converter face-explosion — a flat fan of anonymous, same-material `Mesh` prims
under a named or `kind`-tagged unit — and suggest merging that fan up to the
named boundary, grouped by material. This is hierarchy-level analysis that
surfaces and routes only; it must not modify the stage.

**Surface, don't gate.** Merge is identity-destroying and intent/archetype-gated,
so this reference only surfaces candidates and routes them; the user (or the
archetype default) confirms. It does no vertex-coincidence computation and
applies no numeric merge threshold — the merge op welds anyway.

## Prerequisites

- Run after `usd-structure-assessment` when possible.
- The Phase-2c `perf_small_mesh` validator finding (the tininess signal and entry
  point); pass its members as the small-mesh paths.
- Know the scan root, or use the composed stage root when the user gives no
  narrower scope.

## Limitations

- Suggestions are advisory; no savings are achieved until a merge rewrite and
  after-profile confirm them.
- `SUGGEST_MIN_FAN` and `*_HINT` knobs are surfacing/ranking heuristics, not merge
  gates — they decide what to show and in what order, never whether a shown fan
  may merge.
- This is not a geometric engine: no weld-ratio computation, no merge threshold.
- It surfaces and routes; it makes no identity-dissolution decision.

## Troubleshooting

- If a fan is likely but nothing is surfaced, confirm `perf_small_mesh` ran and
  that children are genuinely anonymous (`Mesh_N`, numeric/uuid tokens) — reference
  designators (`U302`) and semantic names are NOT anonymous.
- If suggestions are noisy, raise `SUGGEST_MIN_FAN` or tighten the
  mesh:material-ratio hint.
- If a surfaced fan overlaps an instance candidate, do not drop it — annotate it
  (`composes_with_instance_candidate: true`); the two compose (see Handoff).

## Examples

- "This control cabinet's boards exploded into thousands of `Mesh_N` leaves — can
  they be merged per material?"
- "Find converter face-explosion fans before planning a within-prototype merge."

## When To Run

Run when the qualitative pattern holds — *meaningless children under a meaningful
unit*:

- a flat fan (mesh children dominate the direct children, low nesting);
- anonymously named children (`Mesh_N`, numeric/uuid tokens);
- under a parent that carries identity (`kind` ∈ {assembly, group, component,
  subcomponent}, or a semantic name); and
- a high mesh:material ratio (many meshes, few distinct materials —
  re-stitchable into one mesh per material).

Skip when children carry their own identity (reference designators, semantic
names), when the fan spans many distinct materials, or when no `perf_small_mesh`
signal is present.

## Method

1. Open the composed stage read-only.
2. For each candidate parent, read cheap structural signals only: direct child
   count and type uniformity, anonymous child naming, the parent's `kind`/name,
   and the distinct bound materials across the fan (a binding-relationship target
   read — no geometry, no points).
3. Surface a parent when the qualitative pattern above holds; rank by fan size and
   mesh:material ratio.
4. For each surfaced parent, name the merge boundary (the named/`kind` ancestor to
   preserve), the `identity_signal` that keeps it addressable, the
   `identity_disposition` (`weak` — the anonymous fan), and the per-material
   grouping (`merge_groups = distinct_materials`, `geomsubset_fallback` when > 1).
5. Route small geometry: `perf_small_mesh` members inside a surfaced fan are real
   faces → merge (re-stitch); the rest are negligible → `removeSmallGeometry`
   (delete). Do not route a fan to delete and do not merge a scattered tiny mesh.

## Division of Labor with Hierarchy-Dedupe Candidates

The two target different things and must never claim the same prims twice:

- `usd-hierarchy-dedupe-candidates` finds repeated subtrees to make `instanceable`
  (reference reuse, identity preserved);
- this reference finds fragmented same-material fans to merge (re-pack, identity
  destroyed).

**Precedence:** a fan that is also a repeated subtree is instanced at the
component first, then its faces merged INSIDE the prototype (merge once, benefit
N instances). On overlap, the suggestion is annotated
(`composes_with_instance_candidate: true`), not dropped — the two compose.

## Output

Report:

- Root scanned and the `perf_small_mesh` entry signal.
- Surfaced parents with merge boundary, identity signal/disposition, per-material
  grouping, and the human-readable suggestion.
- `routed_small_geometry` dispositions (merge vs `removeSmallGeometry`).
- Any `composes_with_instance_candidate` annotations.
- Caveats that the report is advisory and no stage edits were made.

## Handoff

For confirmed suggestions:

1. Choose an edit target with `usd-edit-target-planner` (merge runs within each
   prototype opened as its own root layer).
2. Use `restructure-decision` to confirm the intent-gated merge before mutation.
3. The confirmed `(scope × material)` group feeds the merge rewrite, which owns
   execution and the eligibility guard.

For the precise analyzer behavior, read
`references/mesh-fragmentation-finder-spec.md` only when implementing or debugging
the suggester. The merge win is reported on the draw-call axis, never as a disk
win; do not claim savings as achieved until a rewrite is performed and
after-profile metrics confirm it.
