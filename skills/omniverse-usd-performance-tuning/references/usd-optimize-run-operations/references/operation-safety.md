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
operation, find that operation's entry in
`references/operations/operations.json` and read its optional
`parameter_prerequisites` value. The catalog is the only source for this
contract. The per-op markdown files that once carried it in YAML frontmatter
were retired when the data moved into the catalog, so there is no per-op file
to open and no "no prerequisites file" state to interpret.

The value is optional, and its shape is op-specific: either a flat array of
entries, or an object whose `fields` and `elicit_from_user` keys hold the same
entries alongside op-specific extras such as `ordering`, `scoping`,
`action_chain`, and `apply_approval`. `decimateMeshes` uses the array form and
`findOccludedMeshes` the object form. Read the entries the same way in both.

**These operations carry a `parameter_prerequisites` block today:
`decimateMeshes` and `findOccludedMeshes`.** Every other confirm-required
operation takes the fallback described below. Keep that list in step with the
catalog — `tests/operations/test_parameter_prerequisites_gate.py` pins the
sentence and the catalog together, so authoring a new block fails the suite
until this sentence names the op.

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

### Fallback: confirm-required ops with no `parameter_prerequisites` block

Most confirm-required operations carry no block. A missing block does not mean
the operation needs no approval. The approval requirement comes from
`requires_confirmation` and `apply_authority` in the catalog and from the
destructive-operations table below — never from whether a prerequisites block
happens to exist. Compose the confirmation from these four sources instead:

1. **What the operation does and what it costs.** Use the "Confirmation focus"
   cell for that operation in the destructive-operations table below. Every
   confirm-required operation has a row there. State the loss in the same
   breath as the win, quantified from evidence already gathered (SA report,
   validator findings, the scoped probe result) — "removes 1,240 interior
   meshes, −180 MB, and you lose the internals", not "−180 MB".
2. **Where the decision is presented.** Read the operation's `apply_authority`
   and follow the routing in "Apply authority" below: inline in-plan, or the
   batched per-asset opt-in menu in Phase 7 iteration 2.
3. **Numeric parameters.** Source each value in this order: a constraint the
   user stated; the evidence artifact the Confirmation focus cell names (the
   per-region `maxDepth` and `paths` for `deduplicateHierarchies` come from the
   hierarchy-dedupe-candidates report, not from the user); a per-target scale
   band that `units-and-tolerances.md` declares for the operation. If none of
   those supplies the value and the operation has a usable upstream default, run
   at that default and say in the prompt that the value is upstream's. If the
   parameter has no defensible default — `boxClip`'s extent is the clear case —
   ask for it as an open value and say what it controls. Never close the gap
   with an invented option ladder.
4. **The skip option.** Always offer one in plain words ("skip
   `removeSmallGeometry`"), even though there is no `skip_option` string to
   quote. Selecting it removes the operation from the chain.

A fallback confirmation is a proceed-or-skip decision on a named operation with
its loss stated. It is not permission to improvise parameter values.

### Anti-pattern: rate-framing

**Do not frame tolerance questions as "reduce by X%" or "how much to keep?"**
unless the user has explicitly provided a target reduction rate (memory budget,
LOD level target, explicit percentage).

The canonical framing is fidelity-budget: "what detail to preserve?" This maps
to `maxMeanError` which preserves silhouette quality proportional to the
specified tolerance.

Rate-mode (`reductionFactor` as primary stop) bypasses the silhouette-preserving
default and produces decisions the user cannot evaluate without first seeing
rendered output.

#### A stated triangle target is a CEILING, not a mode switch

A user saying "about 11 million triangles" or "keep 30%" does **not** license
disabling the quality budget. Never set `maxMeanError: 0.0` because a count was
supplied. Instead:

1. Set `maxMeanError` from the tolerance the user gave, or from the archetype
   band in `units-and-tolerances.md` if they gave none.
2. Set `reductionFactor: 0.0` so the quality budget is the only stop condition.
   Leaving it unset applies the upstream default of `50.0` — a silent vertex
   quota nobody asked for.
3. Decimate, then measure the resulting count.
4. If it is still above the stated ceiling, tighten `maxMeanError` and repeat.
   If the ceiling cannot be met within a defensible tolerance, **say so and
   stop** rather than spending fidelity silently.

Why this matters. Two measured runs of the same brief on one CAD station,
differing only in whether it named a triangle target:

| Brief | Decimation | Surface lost |
|---|---|---|
| "roughly 11 million triangles" | `reductionFactor: 25.5`, `maxMeanError: 0.0` | **3.84%**, with 2 m regions losing up to 45% |
| "preserve detail to half a millimetre" | `maxMeanError: 0.5`, `reductionFactor: 0.0` | **0.00%** |

Naming a polygon budget is the most natural request an artist makes, and under
the old reading it silently selected the worst-quality path. The count is a
constraint to satisfy, not an instruction to abandon quality control.

Rate-mode as the *primary* stop remains acceptable only for LOD generation with
known level targets, where the level count is the specification.

### Anti-pattern: improvised option sets

Do not present a **numeric or parametric option set** — tolerances, thresholds,
percentages, triangle counts, depths, extents — unless it traces to one of the
sources the gate names: an `elicit_from_user` entry in the operation's
`parameter_prerequisites`, a constraint the user supplied, the evidence artifact
the operation's Confirmation focus cell names, or a per-target scale band
declared in `units-and-tolerances.md`. If the agent is about to ask "10% or
25%?", the contract says: "no — tolerance questions go through the
`elicit_from_user` template; rate questions require explicit user-supplied
targets."

This rule bars invented numbers. It does not bar the fallback prompt above: a
proceed-or-skip confirmation built from the "Confirmation focus" cell and the
quantified loss offers no numeric option set, and it is the required form for a
confirm-required operation with no block.

See also: `references/usd-optimize-run-operations/references/units-and-tolerances.md` for
the shared unit conversion formula and parameter glossary.

List the destructive operations in the proposed chain, explain what each one
does, then ask for confirmation before invoking the runner.

## Destructive Or Bounded-Loss Operations

| Op | Risk | Confirmation focus |
|---|---|---|
| `findOccludedMeshes` → `removePrims` | Deletes internal geometry. | Two-stage, and the stages split on AUTHORITY not cost: (1) the scoped probe on SA containment pairs runs WITHOUT approval — cost is bounded by scope + `timeout_recorded`; (2) the deletion of discovered occluded prims is intent-gated (the agent cannot know whether the twin needs its internals), so present it on the opt-in menu. Exclude transparent enclosures. The scoped probe runs in Phase 4 (no approval); when the deletion is opted into, it runs FIRST among that target's applies. |
| `deduplicateHierarchies` | Replaces subtrees with instanceable references to shared prototypes. | Confirm dedupe-candidate groups (from hierarchy-dedupe-candidates report). Lossless but structural — changes composition topology. Invoke per frontier region: `paths` scoped to that region plus a per-region `maxDepth` to control grain; never one stage-wide `maxDepth` across branches (see the `deduplicateHierarchies` operations catalog entry). Placement-correct by construction: root `xformOp:*` excluded from identity, duplicate local transform preserved. |
| `decimateMeshes` | Drops vertices. | mm tolerance (maxMeanError); applied uniformly to all meshes. See upstream `docs/operations/decimateMeshes.rst`. |
| `fitPrimitives` | Replaces mesh geometry with analytic primitives. | **Read the target's `archetype` first.** On `piping` and `large-spatial` — pipe runs, conduit, duct, structural runs, architectural shells — a cylinder or box IS the part, so run at default args with a one-line notice and no prompt; this is the expected win and skipping it is the failure. On `encapsulated-product` prompt, because the parts of a serviceable machine carry part numbers a maintenance twin needs addressable. On `generic` lean toward fitting when the geometry is prismatic. When a prompt is warranted, its content is the data-preservation intent (`ignoreNonConstPrimvars` / `ignoreSubsets`) plus the `fitPrimitives` scale band; never an invented tolerance ladder. Bias toward fitting: an unwanted refit is visible and reversible from the source asset, while a fit never attempted is an invisible loss. See upstream `docs/operations/fitPrimitives.rst`. |
| `removeSmallGeometry` | Removes small meshes. | Threshold, visibility, user intent; see upstream `docs/operations/removeSmallGeometry.rst`. |
| `meshCleanup` with `makeManifold: true` | Repairs topology. | Topology repair vs. simpler cleanup; see upstream `docs/operations/meshCleanup.rst`. |
| `optimizeMaterials` with `convertToColor: true` | Replaces material networks with colors. | Only run on explicit flat-color requests; see upstream `docs/operations/optimizeMaterials.rst`. |
| `removePrims` / `deletePrims` / `removeUntypedPrims` / `deleteHiddenPrims` | Deletes prims. | Affected prim list, variant/runtime visibility, reversible alternatives; see the matching operation reference. |
| `boxClip` | Removes or retains geometry by AABB. | Extent and keep-vs-clip mode; see the `boxClip` entry in `references/operations/README.md` and the upstream handoff. |
| `diceMeshes`, `manifoldMeshes`, `remeshMeshes`, `shrinkwrap` | Regenerates or slices topology. | Grid/voxel settings, topology loss, preview scope. |
| `merge` | Collapses multiple meshes into one or more meshes. | Loss of source hierarchy/path identity and instancing risk. **Always pass an absolute `rootPath` under the target prim.** With `rootPath` unset the output goes to `/merged` at the stage root: on an ordinary stage that parks it outside the `defaultPrim`, and on a masked stage that path is outside the population mask, so the merge deletes every source mesh, authors nothing and returns success. Measured on one masked unit: 334 meshes / 292,058 triangles in, 0 / 0 out. With `rootPath` set the same merge keeps every triangle. `meshPrimPaths` does not affect this. |
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
  **`fitPrimitives` on a `piping` or `large-spatial` target belongs here, at
  default args.** On pipe runs, conduit, duct, structural runs, and
  architectural shells a cylinder or box IS the correct representation of the
  part, so fit and notice it. Declining on that content is the failure this
  class exists to prevent, and the notice is the whole of the user's
  involvement.
- **intent-gated (in the table above):** never silently dropped; always presented
  for an explicit decision. A bounded-loss op drops from `auto-within-tolerance`
  back to **intent-gated** whenever (a) it would run **above the conservative
  band** (more aggressive deviation), OR (b) the target carries a
  **functional-precision signal** — `articulated` / physics / sim-ready /
  metrology / variant-bearing — because the band measures *visual* deviation, not
  *functional* tolerance (mating faces, collision/airflow surfaces, kinematic
  features); when the signal is ambiguous, fall back to intent-gated.
  **`fitPrimitives` reads clause (b) through the target's `archetype`.**
  `encapsulated-product` is its prompt case: the parts of a serviceable machine
  carry part numbers a maintenance twin needs addressable, so a fit can erase
  something real. `piping` and `large-spatial` stay on the notice path above;
  the ambiguity fallback does not reach them, because a primitive is already the
  right representation of a pipe or a duct. `generic` uses judgement and leans
  toward fitting on prismatic geometry. The functional-tolerance signal is read
  from SA semantics plus the existing `importance` / `articulated` /
  `archetype` target-tree tags; **no new enum is introduced**. Routes:
  - **Inline-elicited** (`decimateMeshes`; `fitPrimitives` when above-band or on
    an `encapsulated-product` / functional-precision target): offered in-plan as
    a tolerance-and-intent question rather than as a menu item. The tolerance
    question carries the authority. The two ops reach that question by different
    routes: `decimateMeshes` has a `parameter_prerequisites` block and sources
    its fidelity-budget question, defaults, conversion, and skip option from
    there. `fitPrimitives` has **no** block in the catalog and is not getting
    one (issue #207 records the decision — a block is a tolerance ladder, and
    the ladder is what talks an agent out of a win it should be taking), so
    build its prompt from the fallback above: the `fitPrimitives` per-target
    band in `units-and-tolerances.md` supplies the number, its row in the
    destructive-operations table supplies the data-preservation intent
    (`ignoreNonConstPrimvars` / `ignoreSubsets`, whose semantics
    `units-and-tolerances.md` owns), and the prompt ends in a plain-words skip
    option. Do not invent a numeric ladder for `fitPrimitives`. A `piping` or
    `large-spatial` target never arrives on this route at the conservative band;
    it runs at defaults with a notice.
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

`pruneLeaves` removes prims that have no children, and a prim whose **payload is
authored but not loaded** presents as a childless leaf because its real children
live inside the unloaded payload. The operation guards this itself: its
`preserveUnloadedPayloads` argument defaults to `true` and skips such prims.
Verified on usd-optimize 1.0.4 and 1.1.0: with the payload unloaded, the
payload-bearing prim survives at the default and is deleted only at
`preserveUnloadedPayloads: false`, while a genuinely empty sibling is pruned in
both cases.

Do not set `preserveUnloadedPayloads: false` unless you intend to drop
unloaded-payload prims. Loading payloads across the target subtree before
pruning buys nothing and costs memory and time on a large CAD stage.

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

Six named pipelines are wired into the catalog; membership is the `pipelines`
field on each entry in `references/operations/operations.json`. What decides
whether a pipeline runs unattended is its members' `apply_authority`, so read
the classification from that field. `loss_class` describes what an operation
does to authored content and settles nothing about gating on its own:
`deduplicateHierarchies` and `meshCleanup` are both `lossless` and both still
require an explicit decision.

- **Unattended end to end:** `safe-cleanup` and `load-time-reduction`. Every
  member is `auto`.
- **One gated member:** `data-quality-baseline` (`meshCleanup`),
  `memory-reduction` (`deduplicateHierarchies`), and `instancing`
  (`deduplicateHierarchies`, its only member). The remaining members run
  unattended; the gated one is presented on its own terms.
- **Mostly gated:** `mesh-count-reduction`. `deduplicateGeometry` is its
  only `auto` member. `decimateMeshes` is `auto-within-tolerance` (a notice at
  the conservative per-target band, a prompt above it or on a
  functional-precision target), and `deduplicateHierarchies`, `merge`,
  `meshCleanup`, and `removeSmallGeometry` are `intent-gated`.

For hierarchy-level dedupe, use `usd-hierarchy-dedupe-candidates` plus
`apply-restructure`; do not substitute mesh merge for a USD-authored hierarchy
rewrite.

`merge` (Merge Static Meshes) is a different, complementary tool: a **draw-call
within-prototype** op — fuse small adjacent meshes inside a prototype so
the win propagates to every instance. It is **not** a disk lever (merge
concatenates geometry; bytes ~= sum, and the crate already byte-dedups within a
layer), and it is only eligible on **spatially-coherent, weak/none-identity**
clusters: merging dispersed meshes balloons the AABB and degrades BVH/raytracing.
See `../../usd-structure-assessment/references/apply-restructure/references/mesh-merge-rewrite-spec.md`
§9 (op-chain `merge → conditional vertex-weld → computeExtents` and the
bounds-coherence eligibility guard). Any bytes the weld tail reclaims are credited
to the disk tier via the weld source, never attributed to the merge.

### Anti-pattern: silently dropping intent-gated ops

**Do NOT skip, omit, or silently defer an intent-gated op without ever presenting
it.** Every intent-gated op must reach the user as an explicit decision — via
either the iteration-1 inline-elicitation prompt (`decimateMeshes`;
`fitPrimitives` once it has actually dropped to intent-gated for that target)
or the batched per-asset opt-in menu in Phase 7 iteration 2
(occlusion removal, `removeSmallGeometry`, `merge`, `convertToColor`,
PointInstancer). Removal is legitimate ONLY when the user takes the skip path —
the `skip_option` where the op's `parameter_prerequisites` names one, the
plain-words skip the fallback requires otherwise — or declines the menu item.
An op with no `parameter_prerequisites` block still has to be presented; the
absent block removes a data source, not the obligation.

The batched iter-2 menu IS the explicit offer: deferring an identity/purpose-gated
op to it preserves user agency and is NOT the silent-deferral anti-pattern. (This
is also why the Conservative Fallback runs destructive ops as a reviewed later
pass — same principle.)

Acceptable: "decimateMeshes is recommended — what's the smallest detail to
preserve? [0.1 / 0.5 / 1.0 / 2.0 / 5.0 mm / skip decimation]" (inline, iteration 1).

Acceptable: "Iteration-2 options for this prototype: remove 1,240 occluded interior
meshes (−X MB, but you lose the internals); convert 3 fastener families to
PointInstancers (−18K prims, but you lose per-screw selection). Pick per asset."

Acceptable: "Fitting the pipe and duct runs to primitives at default args
(`fitPrimitives`, `piping` archetype, conservative 1 mm component band);
deviation is bounded to that band." (notice, iteration 1, no prompt.)

Not acceptable: running lossless ops and then declaring the run done while
intent-gated wins were never surfaced to the user at all. That removes agency.

Not acceptable: declining `fitPrimitives` on a `piping` or `large-spatial`
target because the op is bounded-loss, or because the twin is serviceable. That
target is `auto-within-tolerance` at defaults, the notice is the offer, and an
undeclared decline removes agency the same way a silent skip does.

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
