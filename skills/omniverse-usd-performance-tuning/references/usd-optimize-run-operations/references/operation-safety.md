<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Operation Safety

Use this reference before running any Usd Optimize chain that may delete,
collapse, regenerate, or otherwise irreversibly change authored content.
Usd Optimize operation mechanics are owned by upstream
[usd-optimize](https://github.com/NVIDIA-Omniverse/usd-optimize/) and the
prebuilt Usd Optimize package. Resolve guidance from an extracted package
root via `$USD_OPTIMIZE_ROOT`. If no package
root exists, download/extract the published the prebuilt Usd Optimize release package (current asset name + download: `references/upstreams/usd-optimize.md`)
package (direct archive URLs are in `references/upstreams/usd-optimize.md`) or
use the package path, URL, or extracted root supplied by the user. Do not clone the
source repo just to read SO guidance. This file owns only the digitaltwin
approval gate and confirmation focus.

## Confirmation Prompt

Always prepend the full runtime context block from
`skills/omniverse-usd-performance-tuning/references/setup-usd-performance-tuning/references/runtime-context-header.md`
Format A. A destructive-op approval must name the Kit application, Scene
Optimizer version, and usd-validation-nvidia version that will mutate the stage.

## Parameter Prerequisites Gate

Before composing the confirmation prompt for any destructive or bounded-loss
operation, read its YAML frontmatter `parameter_prerequisites` block (in
`references/operations/<key>.md`).

For each entry:

- **`field:` entries with `required: true`** — verify the named field exists in
  the SA report (`asset_physical_context` section) or `setup-preflight.json`. If
  missing, **BLOCK** with reason: `"asset preflight incomplete: missing {field}"`.
  Do not proceed to the confirmation prompt.
- **`field:` entries with `required: false`** — if present, use the value to
  enrich suggested defaults or context derivation. If absent, proceed normally;
  do not block.
- **`elicit_from_user:` entries** — include the `canonical_question` with its
  `defaults` as options in the single upfront confirmation prompt. Use the
  `conversion` formula to map the user's answer to the Usd Optimize parameter. If a
  `context_derivation` is present and the referenced field is available, use
  it to suggest a default.
- **`skip_option`** — always offer the skip option. If the user selects it,
  remove that operation from the chain.
- **`default_option`** — if present, this is the pre-selected answer when the
  user doesn't express a preference. It does NOT remove the operation (unlike
  `skip_option`).

All `elicit_from_user` questions for a given operation MUST be batched into a
single prompt (the "single upfront prompt" pattern). Do not ask them as
separate mid-run gates.

### Anti-pattern: rate-framing

**Do not frame tolerance questions as "reduce by X%" or "how much to keep?"**
unless the user has explicitly provided a target reduction rate (memory budget,
LOD level target, explicit percentage).

The canonical framing is fidelity-budget: "what detail to preserve?" This maps
to `maxMeanError` which preserves silhouette quality proportional to the
specified tolerance.

Rate-mode (`reductionFactor` as primary stop) bypasses the silhouette-preserving
default and produces decisions the user cannot evaluate without first seeing
rendered output. It is acceptable ONLY when:

1. The user explicitly says "reduce to N triangles" or "keep X%", or
2. The workflow is LOD generation with known level targets.

### Anti-pattern: improvised option sets

Do not present options that don't trace to a `parameter_prerequisites` block
or a user-supplied constraint. If the agent is about to ask "10% or 25%?", the
contract says: "no — tolerance questions go through the `elicit_from_user`
template; rate questions require explicit user-supplied targets."

See also: `references/usd-optimize-run-operations/references/units-and-tolerances.md` for
the shared unit conversion formula and parameter glossary.

List the destructive operations in the proposed chain, explain what each one
does, then ask for confirmation before invoking the runner.

## Destructive Or Bounded-Loss Operations

| Op | Risk | Confirmation focus |
|---|---|---|
| `findOccludedMeshes` → `removePrims` | Deletes internal geometry. | Two-stage, and the stages split on AUTHORITY not cost: (1) the scoped probe on SA containment pairs runs WITHOUT approval — cost is bounded by scope + `timeout_recorded`; (2) the deletion of discovered occluded prims is intent-gated (the agent cannot know whether the twin needs its internals), so present it on the opt-in menu. Exclude transparent enclosures. The scoped probe runs in Phase 4 (no approval); when the deletion is opted into, it runs FIRST among that target's applies. |
| `deduplicateHierarchies` | Replaces subtrees with instanceable references to shared prototypes. | Confirm dedupe-candidate groups (from hierarchy-dedupe-candidates report). Lossless but structural — changes composition topology. |
| `decimateMeshes` | Drops vertices. | mm tolerance (maxMeanError); applied uniformly to all meshes. See upstream `.agents/operations/decimateMeshes.md`. |
| `fitPrimitives` | Replaces mesh geometry with analytic primitives. | Analysis first and data-preservation intent; see upstream `.agents/operations/fitPrimitives.md`. |
| `removeSmallGeometry` | Removes small meshes. | Threshold, visibility, user intent; see upstream `.agents/operations/removeSmallGeometry.md`. |
| `meshCleanup` with `makeManifold: true` | Repairs topology. | Topology repair vs. simpler cleanup; see upstream `.agents/operations/meshCleanup.md`. |
| `optimizeMaterials` with `convertToColor: true` | Replaces material networks with colors. | Only run on explicit flat-color requests; see upstream `.agents/operations/optimizeMaterials.md`. |
| `removePrims` / `deletePrims` / `removeUntypedPrims` / `deleteHiddenPrims` | Deletes prims. | Affected prim list, variant/runtime visibility, reversible alternatives; see the matching operation reference. |
| `boxClip` | Removes or retains geometry by AABB. | Extent and keep-vs-clip mode; see the `boxClip` entry in `references/operations/README.md` and the upstream handoff. |
| `diceMeshes`, `manifoldMeshes`, `remeshMeshes`, `shrinkwrap` | Regenerates or slices topology. | Grid/voxel settings, topology loss, preview scope. |
| `merge` | Collapses multiple meshes into one or more meshes. | Loss of source hierarchy/path identity and instancing risk. |
| `pythonScript` | Executes user-supplied code. | Require a user-supplied or reviewed script. |
| `removeAttributes` | Removes or blocks attributes. | Exact attribute list and downstream consumers. |
| `sparseMeshes` | Analysis that often drives split/dice follow-ups. | Confirm acting on the analysis result. |

## Apply authority: auto vs intent-gated routing

The axis that decides "needs a user decision" is **authority + reversibility, not
compute cost**. A scoped analysis probe is cost-bounded and runs without approval;
*applying* a result that deletes geometry or collapses identity needs the user,
because only they know the digital twin's purpose (a showroom exterior render can
drop an engine; a service/training/CFD twin cannot; a maintenance twin needs
per-instance selection, a viz twin does not). Cost is orthogonal — PointInstancer
conversion is cheap to analyze but identity-losing to apply.

Each op's **base** apply-authority class is machine-readable as the
`apply_authority` field on every entry in
`references/operations/operations.json` (enum `auto` / `auto-within-tolerance` /
`intent-gated`). That catalog field is the single source a data-driven consumer
(status derivation, the scheduler, interpret-validators) reads to DERIVE the
class; this section is the canonical *explanation* of what each class means and
owns the **target-conditional** gating rule the static field cannot express. The
field encodes only the BASE class and is cross-checked against
`requires_confirmation` (`requires_confirmation == (apply_authority != "auto")`):
`auto` never gates; `intent-gated` and `auto-within-tolerance` both carry
`requires_confirmation: true`, because `auto-within-tolerance` keeps the
conservative flag set until a target is confirmed visually-toleranced at the
conservative band (see the downgrade rule below). There are **three**
apply-authority classes:

- **`auto` (lossless — not in the table above):** `removeUnusedUVs`,
  `deduplicateGeometry`, `optimizeMaterials` dedup, `computeExtents`,
  `pruneLeaves`, `optimizeTimeSamples`. Run in **iteration 1** per target, no
  prompt, unattended-friendly. (`meshCleanup` invoked **weld-only** also runs
  here — see the sub-mode note below — but its catalog BASE class is
  `intent-gated`, not `auto`, because the full op bundles topology-repair
  sub-modes that need a decision.)
- **`auto-within-tolerance` (bounded-loss × conservative per-target band ×
  visually-toleranced target):** the bounded-loss ops with a deviation parameter
  (`decimateMeshes`, `fitPrimitives`) run with a **one-line notice, not a
  prompt**, when ALL of these hold: (a) the op runs at the *conservative*
  per-target scale band (resolved per target from its extent — see
  `units-and-tolerances.md`), and (b) the target is **visually-toleranced** (no
  functional-precision signal). This is the deliberate mild bounded-loss default
  that guards against under-optimization (the ludicrously-over-tessellated mesh
  that a pure opt-in menu lets sail through). The notice names the op, the
  per-target band, and that deviation is bounded to the band.
- **intent-gated (in the table above):** never silently dropped; always presented
  for an explicit decision. A bounded-loss op drops from `auto-within-tolerance`
  back to **intent-gated** whenever (a) it would run **above the conservative
  band** (more aggressive deviation), OR (b) the target carries a
  **functional-precision signal** — `articulated` / physics / sim-ready /
  metrology / variant-bearing — because the band measures *visual* deviation, not
  *functional* tolerance (mating faces, collision/airflow surfaces, kinematic
  features); when the signal is ambiguous, fall back to intent-gated. The
  functional-tolerance signal is read from SA semantics + the existing
  `importance` / `articulated` target-tree tags — **not a new closed
  archetype enum**. Routes:
  - **Inline-elicited** (`decimateMeshes`, `fitPrimitives` when above-band or on
    a functional-precision target): offered in-plan through their
    `parameter_prerequisites` (fidelity budget, data-preservation intent). The
    tolerance question carries the authority.
  - **Purpose/identity-gated** (`findOccludedMeshes`→`removePrims`,
    `removeSmallGeometry`, `merge`, `optimizeMaterials`+`convertToColor`,
    PointInstancer-convert): identity-losing — no tolerance can bound them, so
    they stay intent-gated for ALL archetypes. Presented as the **batched
    per-asset opt-in menu in Phase 7 iteration 2**, with win AND loss quantified
    per asset. The scoped detection probe (e.g. `findOccludedMeshes`) runs earlier
    in Phase 4 without approval — its result quantifies the menu; only the
    destructive apply fires on opt-in.

The real authority boundary is **above-band / identity-losing / functional-precision
target**, not lossless-vs-lossy. A bounded-loss op at the conservative band on a
visually-toleranced target is `auto-within-tolerance` (notice); the same op above
the band, or on an articulated/physics/sim-ready target, is `intent-gated`
(prompt). This `auto-within-tolerance` → `intent-gated` downgrade is
**target-conditional, not op-static**, so it is deliberately NOT written into the
per-op `apply_authority` field (which carries the BASE class only): it is applied
at plan time from SA semantics + the `importance` / `articulated` target-tree tags.

**`meshCleanup` is sub-mode-conditional (the same pattern, on a different axis).**
Its catalog BASE `apply_authority` is `intent-gated` because the full op bundles
topology-repair sub-modes (`makeManifold`, isolated/degenerate removal — see the
destructive table) that change geometry and need a decision. But the
**vertex-weld-only** invocation — the default Phase-4 step-1 use, welding
coincident verts within tolerance — is lossless and **effectively `auto`**: it
runs unattended in iteration 1 alongside the other lossless ops. As with the
`auto-within-tolerance` downgrade, this weld-only-is-auto nuance is
**invocation-conditional, not op-static**, so it is deliberately NOT written into
the per-op `apply_authority` field (which carries the conservative BASE class
only); the prose owns it. `requires_confirmation: true` stays set on the catalog
entry because the conservative base holds until a weld-only invocation is the
confirmed scope — the same reason `auto-within-tolerance` keeps its flag set.

### Caveat: `pruneLeaves` on unloaded payloads

`pruneLeaves` removes prims that have no children. A prim whose **payload is
authored but not loaded** presents as a childless leaf, because its real
children live inside the unloaded payload — so `pruneLeaves` will delete it and
silently lose that content. `pruneLeaves` must NOT target prims with unloaded
payloads. Before pruning, either **load the payloads** across the target subtree
so the real children are visible, or **exclude prims that have unloaded payloads**
from the prune target set. Never let an unloaded-payload prim be mistaken for an
empty leaf.

## Conservative Fallback

If the user is uncertain, run only `safe-cleanup` first:

- `computeExtents`
- `pruneLeaves`
- `deduplicateGeometry`
- `optimizeMaterials`
- `optimizeTimeSamples`

Run destructive or bounded-loss operations as a later pass after the user has
reviewed the safe-cleanup result.

## Pipeline Notes

For named pipelines, only `mesh-count-reduction` and `data-quality-baseline`
contain destructive ops today. `safe-cleanup`, `memory-reduction`, and
`load-time-reduction` are lossless. For hierarchy-level dedupe, use
`usd-hierarchy-dedupe-candidates` plus `apply-restructure`; do not substitute
mesh merge for a USD-authored hierarchy rewrite.

`merge` (Merge Static Meshes) is a different, complementary tool: a **draw-call
within-prototype** op — fuse small adjacent meshes inside a prototype so
the win propagates to every instance. It is **not** a disk lever (merge
concatenates geometry; bytes ~= sum, and the crate already byte-dedups within a
layer), and it is only eligible on **spatially-coherent, weak/none-identity**
clusters: merging dispersed meshes balloons the AABB and degrades BVH/raytracing.
See `../../usd-structure-assessment/references/apply-restructure/references/hierarchy-dedupe-rewrite-tool-spec.md`
§9 (op-chain `merge → conditional vertex-weld → computeExtents` and the
bounds-coherence eligibility guard). Any bytes the weld tail reclaims are credited
to the disk tier via the weld source, never attributed to the merge.

### Anti-pattern: silently dropping intent-gated ops

**Do NOT skip, omit, or silently defer an intent-gated op without ever presenting
it.** Every intent-gated op must reach the user as an explicit decision — via
either the iteration-1 inline-elicitation prompt (`decimateMeshes`,
`fitPrimitives`) or the batched per-asset opt-in menu in Phase 7 iteration 2
(occlusion removal, `removeSmallGeometry`, `merge`, `convertToColor`,
PointInstancer). Removal is legitimate ONLY when the user selects `skip_option`
or declines the menu item.

The batched iter-2 menu IS the explicit offer: deferring an identity/purpose-gated
op to it preserves user agency and is NOT the silent-deferral anti-pattern. (This
is also why the Conservative Fallback runs destructive ops as a reviewed later
pass — same principle.)

Acceptable: "decimateMeshes is recommended — what's the smallest detail to
preserve? [0.1 / 0.5 / 1.0 / 2.0 / 5.0 mm / skip decimation]" (inline, iteration 1).

Acceptable: "Iteration-2 options for this prototype: remove 1,240 occluded interior
meshes (−X MB, but you lose the internals); convert 3 fastener families to
PointInstancers (−18K prims, but you lose per-screw selection). Pick per asset."

Not acceptable: running lossless ops and then declaring the run done while
intent-gated wins were never surfaced to the user at all. That removes agency.

---

## Red Flag: SO Operation Returns Success With Zero Work on Known-Heavy Target

| Signal | Meaning |
|--------|---------|
| `elapsed_ms: 0` or < 1ms on a target with known high vertex/mesh count | Operation could not find meshes to process |
| `success: true` but vertex_count delta = 0 on a target SA flagged for optimization | Structural blockage, not "nothing to do" |
| Multiple operations show zero work on same target | Almost certainly a traversal issue (Over-spec ancestors, population mask, wrong root prim) |

**Action:** Do NOT report "operation found nothing to optimize" when SA or manifest
metadata indicates the target should have significant geometry. Instead:

1. Check specifiers on ancestor prims (Over vs Def) — see `restructure-mode.md`
   §"Authoring Requirements" for the diagnostic snippet.
2. Check that the target's `defaultPrim` is set correctly.
3. Check that the stage is not masked or filtered in a way that excludes content.
4. Report the structural issue to the user rather than rationalizing the no-op.
