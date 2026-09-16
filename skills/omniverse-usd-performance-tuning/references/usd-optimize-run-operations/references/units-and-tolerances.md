# Units and Tolerances

Shared reference for any operation that converts user-specified mm tolerances
to Usd Optimize stage-unit parameters. Referenced by `operation-safety.md`
and consumed by any `parameter_prerequisites` block with a `conversion` field
in `references/operations/operations.json`.

## Source of Truth

The `asset_physical_context` section of the SA report provides:

| Field | Meaning |
|-------|---------|
| `metersPerUnit` | Stage scale factor (1.0 = meters, 0.01 = cm, 0.001 = mm) |
| `upAxis` | Stage orientation (X, Y, or Z) |
| `scale_hint` | Human label: "meters", "centimeters", "millimeters", "other" |

## Conversion Formula

```
tolerance_stage_units = mm_tolerance / (metersPerUnit × 1000)
```

### Worked Examples

| User says | metersPerUnit | Stage units | Result |
|-----------|---------------|-------------|--------|
| "0.5 mm" | 0.01 (cm) | centimeters | 0.5 / (0.01 × 1000) = **0.05** |
| "1.0 mm" | 1.0 (m) | meters | 1.0 / (1.0 × 1000) = **0.001** |
| "2.0 mm" | 0.001 (mm) | millimeters | 2.0 / (0.001 × 1000) = **2.0** |
| "0.1 mm" | 0.01 (cm) | centimeters | 0.1 / (0.01 × 1000) = **0.01** |

## Scale-banded per-target default tolerance

The default bounded-loss tolerance is **not a single number** — it tracks each
*target's* physical scale and differs by op family (`fitPrimitives` runs ~10×
looser than `decimateMeshes`). These are the **conservative** bands that the
`auto-within-tolerance` apply-authority class (see `operation-safety.md`) runs
with a notice rather than a prompt:

| Target scale | `decimateMeshes` | `fitPrimitives` |
|--------------|------------------|-----------------|
| Building / entire-building (`large-spatial` archetype, or max extent ≥ ~10 m — tunable) | 1 mm | 1 cm |
| Component & smaller (default) | 0.1 mm | 1 mm |

**Resolved per target.** Because Phase 4 optimizes per target and the bounded
recursive descent gives each target its own extent + `archetype` tag, the band is resolved
per target: a building shell (`assembly_root` / `large-spatial`) gets the coarse
pair, while an extracted valve gets the fine pair *even though it lives inside
the building*. The user overrides the default globally and, optionally, per
archetype.

**These real-world lengths convert to stage units** via the formula above using
the target's `asset_physical_context` (`metersPerUnit`, `scale_hint`), so the
same declared band yields the right stage-unit value on a centimeter CAD asset
and a meter-scale architecture asset.

**Functional-tolerance gate.** The bands measure *visual* deviation. When a
target carries a functional-precision signal (`articulated` / physics /
sim-ready / metrology / variant-bearing — read from SA semantics + the
`importance` / `articulated` target-tree tags), bounded-loss ops drop from
`auto-within-tolerance` back to `intent-gated` regardless of band, because a
visual band cannot bound functional tolerance. `operation-safety.md` owns this
routing.

## Elicitation Template

Use this structure when the operation has a `parameter_prerequisites` block in
`references/operations/operations.json` with an `elicit_from_user` entry. For an
operation with no block, `operation-safety.md` §"Fallback: confirm-required ops
with no `parameter_prerequisites` block" owns the prompt, and the numbers come
from the scale bands above rather than from a defaults array.

1. **State the asset's physical scale:**
   > "This stage uses {scale_hint} (metersPerUnit = {metersPerUnit})."

2. **Ask the canonical question** from the operation's `parameter_prerequisites`:
   > "{canonical_question}"

3. **Offer defaults** from the prerequisites block:
   > Present the `defaults` array from the operation's `parameter_prerequisites`.
   > The user picks one or provides their own value.

4. **Offer the skip option.**

## Parameter Glossary

| SO Parameter | Unit | Range | Meaning |
|-------------|------|-------|---------|
| `maxMeanError` | stage units | 0.0 = disabled | QEM error budget per vertex. Primary quality knob. |
| `reductionFactor` | float 0.0–100.0 | 100.0 = keep all, 0.0 = disabled | Percentage of the original **vertex** count to KEEP. Secondary stop condition; upstream default is `50.0`. |
| `pinBoundaries` | boolean | default `false` | Preserve mesh boundary edges. Pass `true` explicitly for sub-mesh decimation. |

**Critical:** `reductionFactor` is "keep percent" of the vertex count **after the
operation's internal weld**, not of the authored count. On welded input the two
are the same and the nominal value holds: measured on 1.0.4 and 1.1.0, a
3,721-vertex mesh at `reductionFactor: 90.0` came back with 3,348, exactly 90.0%.

CAD input is never welded, so the nominal value badly overstates what you keep.
Measured on a 1,761-mesh CAD slice (2,827,779 authored vertices, roughly 2.4x
redundant), `maxMeanError` disabled so nothing else could bind first:

| `reductionFactor` | vertices kept | faces kept |
|---|---|---|
| 90.0 | 35.5% | 79.9% |
| 80.0 | 31.5% | 71.0% |
| 50.0 | 19.7% | 44.3% |
| 45.0 | 17.7% | 39.8% |
| 20.0 | 7.9% | 17.6% |

Faces track the nominal value at a consistent ~0.88x; vertices do not track it
at all. An independent run on a different asset reproduced this (45.0 retaining
17.8%, 80.0 retaining 31.9%).

Three consequences. **Compare on faces, not vertices** — a CAD stage's authored
vertex count is an artifact of the exporter, not a measure of its geometry. And
**you cannot predict the output of a stated triangle target through
`reductionFactor`** without first knowing the asset's weld ratio, which is one
more reason a triangle target must drive `maxMeanError` iteratively rather than
switching the op to rate mode. See `operation-safety.md § A stated triangle
target is a CEILING, not a mode switch`.

Third, and the actionable one: **weld first and the parameter becomes meaningful
again.** The unpredictability above is entirely the redundant vertices. Validate
the weld state with the `vertex_weld` concept, run `meshCleanup` if it fires, and
only then reduce — at which point `reductionFactor: 90.0` keeps 90% because the
authored and welded counts finally agree. The workflow states this as a
precondition rather than an ordering preference; see `workflow.md § Operation
ordering invariants`.

`decimateMeshes` has no absolute triangle cap; `reductionFactor` is its only
count-based stop condition, and it is not a reliable one.

## Anti-Patterns

1. **Do NOT ask "reduce by 10%?"** — that's rate-framing.
   The canonical question is fidelity-budget: "what detail to preserve?"
   See `operation-safety.md § Anti-pattern: rate-framing`.

2. **Do NOT use integer `0` for disabled float conditions** — use `0.0`.
   JSON `"maxMeanError": 0` is ambiguous; `"maxMeanError": 0.0` is explicit.

3. **Do NOT omit `pinBoundaries: true`** when decimating sub-meshes or
   meshes that share boundary edges with neighbors.

4. **Do NOT invent percentage options** without the user first providing a
   rate-based constraint. If the user hasn't said "I want N triangles" or
   "keep X%", the tolerance question is the correct entry point.

5. **Do NOT skip the conversion step.** A user saying "1mm tolerance" on a
   centimeter stage means `maxMeanError: 0.1`, not `maxMeanError: 1.0`.

## Operations That Use This Reference

Any operation with tolerance knobs benefits from this formula:

- `decimateMeshes` — `maxMeanError` (primary)
- `deduplicateGeometry` — `tolerance` (coincidence threshold)
- `findCoincidingGeometry` — `tolerance`
- `mergeVertices` — `tolerance`
- `removeSmallGeometry` — `threshold` (min extent in stage units)

## deduplicateGeometry parameter gotchas (field-validated)

These were learned on real large-CAD optimization runs; upstream docs own the
full parameter reference, but these three traps are load-bearing enough to
record locally:

- **`tolerance` is ABSOLUTE (stage units, worldspace) on usd-optimize 1.0.4**
  — verified empirically (2026-06-11): the same 0.01-unit point delta deduped
  at `tolerance: 0.02` and not at `0.005`, identically at coordinates ~1 and
  ~10,000, so the mm-conversion formula above DOES apply on 1.0.4 (matching
  the argument description "stage unit in worldspace"). Over-matching is the dangerous
  direction: when in doubt, tune DOWN first.
- **`considerDeepTransforms` defaults to `true` and can corrupt placement** —
  the standalone run observed instances landing with wrong transforms under the
  default. Pass `considerDeepTransforms: 0` unless placement has been verified
  on a sample after a trial run. Use `0`, not `false`: the CLI parses this
  argument as an integer and rejects the boolean literal with `failed to set
  argument: stoi`, which aborts the run before any work is done.
- **`duplicateMethod` default (Instanceable Reference, 2) makes later
  decimation a no-op** — dedupe output is instances, and `decimateMeshes`
  skips instanced prims. Decimate before dedupe, or use a non-instancing
  method and author `instanceable` afterwards. See the ordering-invariant
  caveat in `workflow.md § Operation ordering invariants`.

## fitPrimitives parameter gotchas (primvar preservation)

`fitPrimitives` replaces tessellated meshes with fitted primitives (cylinder,
cone, cube, sphere). Two boolean parameters govern which meshes it touches. **The
defaults are aggressive (fit more); override only to PRESERVE data, never
reflexively.**

- **`ignoreNonConstPrimvars` (default `true`).** At the default, fitPrimitives
  fits a candidate mesh even when it carries non-constant primvars, **discarding
  any non-`Normal` primvars (UVs, displayColors) on replacement**. Set it to
  `false` — the RESTRICTIVE setting — to SKIP meshes whose validator analysis
  reports non-constant non-`Normal` primvars, preserving those primvars intact.
  Per upstream `Primitive.cpp:321-322`, `Normal` primvars are explicitly
  allow-listed and never block fitting at any setting, so a scene whose only
  non-constant primvar is `normals` fits identically with default args.
- **`ignoreSubsets` (default `true`).** Analogous: the default fits meshes that
  carry `GeomSubset` partitions; set `false` to skip them when subsets must
  survive.

**When to override.** Add `ignoreNonConstPrimvars: false` (and/or
`ignoreSubsets: false`) only when BOTH hold: validator analysis reports
`nonconstPrimvarMeshCount > 0`, AND the user has asked to preserve UVs /
displayColors / subsets. Otherwise use default args.

**Anti-pattern.** Do NOT reflexively set `ignoreNonConstPrimvars: false` on every
CAD/BIM scene. The default already fits; the restrictive setting only narrows
what gets fitted, and is justified only by an explicit preservation intent. This
corrects a historic inverted-doc reading that told agents to set `false`
everywhere.

Where an operation's entry in `references/operations/operations.json` carries a
`parameter_prerequisites` block, that block specifies which fields the operation
needs and what conversion applies. `fitPrimitives` has no such block: its number
comes from the `fitPrimitives` column of the scale-band table above, and its
data-preservation decision from the two boolean parameters described in this
section. This file owns the shared formula and the per-target bands; individual
ops own their specific parameter semantics.
