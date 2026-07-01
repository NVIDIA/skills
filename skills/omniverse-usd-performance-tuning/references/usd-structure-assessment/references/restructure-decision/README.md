# Restructure Decision

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

## When to Use

Use to decide whether a monolithic USD stage should be restructured (asset-boundary materialization + hierarchy dedupe) before optimization, or optimized as-is. Asks the user; invokes apply-restructure when the user confirms.

## Instructions

1. Confirm the target asset, artifact, or user intent and check the prerequisites listed below.
2. Read only the referenced files needed for the current phase, failure mode, or output contract.
3. Follow the workflow, rules, and safety gates in this reference before invoking downstream references or shell commands.
4. Return the result using the Output Format section and name any blocked prerequisite or unresolved user decision.


## Pre-flight Checklist

Before presenting the restructure gate, re-read and confirm:

- [ ] SA report contract — `phase_recommendation`, `hierarchy_dedupe`,
   `asset_boundary_suggestions` fields.
- [ ] `setup-preflight.json` runtime header — know what runtime is available.
- [ ] Present the restructure choices plus optimize-as-is — do not pre-select on
   the user's behalf. The gate chooses *how* to optimize, never *whether*; there
   is no diagnose-and-exit option.
## Output Format

Return a concise status or report that names the input, selected runtime or evidence source, actions planned or performed, artifacts written, blockers, and the next validation or user-decision step. When a schema or template is referenced below, conform to that contract.

## Purpose

Phase 2e of the canonical optimization flow (see
`skills/omniverse-usd-performance-tuning/references/workflow.md`).
After `usd-structure-assessment` has classified the asset and
`usd-hierarchy-dedupe-candidates` has produced asset-boundary signal, this
skill is the user-confirm gate that decides whether to restructure the stage
before optimization.

This is a small decision-tier skill. It does not perform the rewrite - that's
the execution-tier `apply-restructure`, which uses `pxr`/`Sdf` USD authoring to
materialize boundaries and apply the hierarchy-dedupe rewrite described in
`skills/omniverse-usd-performance-tuning/references/usd-structure-assessment/references/apply-restructure/references/hierarchy-dedupe-rewrite-tool-spec.md`.

**Bounded recursive descent.** This gate fires once per descent level:
after the first restructure, boundary inference re-runs on each extracted asset
(assembly → component → subcomponent) to a bounded depth, so the gate may recur
per node. The descent contract, the `level`/`importance`/`articulated`/
`archetype` target-tree tags, and the stopping rule are defined in `workflow.md`
Phase 2g. Always prefer shared prototypes with `instanceable=true` references
over N unshared per-node payloads; descend to `subcomponent` only for
"important" (articulated / physics / variant-bearing) sub-hierarchies.

**Descent convergence gate (confirm per level; converge before Phase 4).** The
per-node stop conditions (`min_meaningful_unit` / `arc_cost` / `below_floor`) say
why a *single* node stopped; they do NOT establish that the *whole* descent has
bottomed out — and **how deep to decompose is the user's call, not an autonomous
plunge** (this gate is the per-level confirmation point; see "Bounded recursive
descent"). So make the descent a **per-level checkpoint**: after authoring a level,
re-run the reuse analyzer (`usd-hierarchy-dedupe-candidates`, the cheap HASH_LEVEL-2
structural pass) on the result, **present what it finds one level down** (the new
shareable groups above the inclusion floor — MINP, occurrence ≥2 — with the
addressability / layer-count cost), and **ask whether to descend further or stop**.
Keep the asks meaningful: the ones that matter are **crossing a named identity
boundary** (descending into a component's internals changes what stays addressable)
and any **identity-destroying route** (point-instance / merge); a routine
lossless-sharing tail can be offered as "auto-finish the rest" so the user isn't
prompted for every trivial sublevel. The descent is **complete** when the user
stops it or the re-scan is dry above the floor (residue = sub-MINP
`kept_inline_for_merge` leaves, already-split value-variants, or unique content).
**Do NOT continue to Phase 4 geometry ops (decimation, within-prototype merge)
until the descent is complete**, and record `descent_converged: true` plus
`final_rescan_new_groups_above_floor` in the manifest. Structure must settle before
geometry: decimating or merging an unconverged structure wastes work on geometry
that further sharing would have collapsed, and a merge that runs before the descent
reserved its `kept_inline_for_merge` leaves ends up fusing already-shared geometry
(premature-merge inflation; see the merge-eligibility guard in
`hierarchy-dedupe-rewrite-tool-spec.md` §9). Phase 7 `resume-descent`
is the *exception* (reuse that only became visible after a transform, or a level
deliberately deferred), not a license to run geometry ops on an incomplete descent;
a resumed descent re-checkpoints before re-entering geometry ops.

## Prerequisites

- A completed `usd-structure-assessment` report including:
  - `phase_recommendation` (`structuring | optimization | already_optimized`).
  - `hierarchy_dedupe.recommended` and `hierarchy_dedupe.top_candidates` (when present).
  - The §2.5 asset-boundary identification output (when the stage is monolithic).
- Optional: `usd-hierarchy-dedupe-candidates` read-only candidate report when the stage is monolithic.
- Optional: Phase 2c `usd-validation-runner` findings corpus (informs the decision when validators flagged structural-only issues that restructure would help with).

## Examples

- "Should I restructure this CAD stage before running mesh ops?"
- "The factory.usd is monolithic with 12 repeated assemblies - what's next?"

## Inputs

The agent assembles a decision packet from prior phases:

| Input | From | Used to decide |
|---|---|---|
| SA classification | `usd-structure-assessment` Phase 2a | Monolithic vs composed; restructure recommended? |
| Asset-boundary candidates | `usd-structure-assessment` §2.5 + `usd-hierarchy-dedupe-candidates` | Where the cut points are if restructure is chosen |
| Validator findings | Phase 2c `usd-validation-runner` selected probes | Whether structural-only fixes would be wasted on a stage about to be restructured |
| Instancing assessment | Phase 2d (read from SA `instancing` field) | Estimated leverage from restructure |
| User constraints | session context | Time budget, mutation policy, output policy |

## Decision branches

Compute the recommended branch from the inputs, then **always present the choice to the user** - do not auto-proceed.

| SA classification | hierarchy_dedupe.recommended | Recommended | Branches offered |
|---|---|---|---|
| `monolithic-needs-restructure` | true | ask (see below) | deduplicate-internally / extract-as-assets / optimize-as-is |
| `monolithic-needs-restructure` | false | decompose-for-selective-loading | decompose-for-selective-loading / optimize-as-is |
| `monolithic-fine-as-is` | — | optimize-as-is | optimize-as-is (continue) |
| `monolithic-fine-as-is` + `payload_count=0` + clear boundaries | — | ask | decompose-for-selective-loading / optimize-as-is |
| `composed` (already structured) | — | continue (no Phase 2f) | continue (Phase 3) |
| `phase_recommendation = already_optimized` | — | continue (Phase 3-5 find no work) | continue → `no_op` report |

#### When hierarchy_dedupe.recommended=true

Present exactly two restructure strategies (plus optimize-as-is):

1. **Deduplicate hierarchies internally** — run the **same structured descent** as
   the external path (`apply-restructure`, `mode: internal_reference`): the reuse
   analyzer confirms which meaningful (kind/named/semantic) units repeat, one
   prototype is authored per genuine value-variant, and each duplicate site is
   rewritten as a reference marked `instanceable=true`. The only differences from
   the external path are **materialization** — prototypes live in an internal
   namespace (e.g. `/__HierarchyPrototypes`) inside the single stage rather than as
   separate files — and **no parallel per-asset Phase 4** (one file). It still
   produces the frontier and a restructure-role manifest (`identity_disposition:
   internal_share`, sub-MINP leaves tagged `kept_inline_for_merge`). Authoring is
   the **direct value-hash** rewrite, NOT a `deduplicateHierarchies` invocation:
   on 1.0.x that operator authors a strong instanceable-reference collapse
   (nested), but
   without the frontier manifest, identity gating, or `kept_inline_for_merge`
   tagging this branch's contract requires — it is used here only to *suggest*
   candidates. Appropriate when selective
   loading is not needed.

2. **Extract duplicate hierarchies as payloaded, instanced assets** — The
   hierarchy-dedupe rewrite tool runs with `mode: external_prototype`, extracting
   each shared prototype as an external asset file. Each instance site references
   the prototype via a payload arc, making it independently loadable. This is
   the full restructure: the monolith becomes an assembly root + prototype
   assets. Appropriate when selective loading matters (large scenes,
   collaborative workflows, streaming).

Both strategies run the **same descent through `apply-restructure`** and produce
instanced prototypes. The difference is only **materialization**: whether
prototypes live inside the stage (internal namespace, `mode: internal_reference`)
or as separate files (external payloaded assets, `mode: external_prototype`) — and
whether Phase 4 can fan out per-asset in parallel (external only).
`deduplicateHierarchies` is NOT the authoring mechanism for either — it
authors an instanceable-reference collapse (nested on 1.0.4) without the
manifest/identity contract, so it serves
as a candidate source only.

The boundary plan records:
- `goal: deduplicate_internally` → hands off to `apply-restructure` with
  `dedupe.mode: internal_reference` (value-hash nested library authored into an
  internal namespace, `identity_disposition: internal_share`, restructure-role
  manifest with `kept_inline_for_merge` tagging).
- `goal: extract_as_assets` → hands off to `apply-restructure` with `dedupe.mode: external_prototype`

Do NOT offer a "selective loading without instancing" option — extracting N
identical subtrees as N independent files without sharing a prototype is always
wrong when the hash confirms structural identity.

#### Selective loading (no dedupe candidates)

When `hierarchy_dedupe.recommended=false` but `usd-structure-assessment` reports
`payload_count: 0` and clear spatial, discipline, linked-model, category, or
building-wing boundaries, present a selective-loading choice:

- `decompose-for-selective-loading`: materialize the chosen boundary level as
  loadable sub-assets (payloads). Each boundary becomes its own file.
- `optimize-as-is`: keep the monolithic delivery package and proceed to
  validation / SO optimization.

If the user picks `decompose-for-selective-loading`, ask which candidate level
from `asset_boundary_suggestions.candidate_levels` should be used unless the
user already specified it. This path still hands off to `apply-restructure`;
the boundary plan should record `goal: selective_loading` so downstream mesh
ops know the split is for packaging and workflow, not for instancing.

#### Extract-as-assets authoring details

When the user picks `extract-as-assets`, the authoring recipe in
`restructure-mode.md` §"Instanced Asset Extraction" applies:

- Identical subtrees share one prototype file.
- Each instance site gets a lightweight placement prim (`instanceable=true`)
  inside its payload layer.
- Instancing is decided per dedupe group, not globally. Some extracted
  assets may be instanceable (their group passes the `instancing-readiness`
  gate) while others are extracted as unique payloads.
- The boundary plan records the per-group decision.

The `apply-restructure` skill handles the file extraction and assembly-root
rewrite. This skill (`restructure-decision`) only captures the user's choice.

#### User overrides the recommendation

When SA recommends `optimize-as-is` (or `already_optimized`) but the user
picks restructure anyway, confirm the user's goal before authoring. Restructure
does **not** improve geometry-level metrics — those land in Phase 4. What
restructure actually buys:

- **Selective loading via payloads** — split a 1 GB monolithic stage into
  per-floor / per-discipline payloads the user can load on demand.
- **Modular collaboration** — separate sub-assets so multiple authors can
  edit in parallel without conflict.
- **Per-asset Phase 4 targets** — Phase 4 mesh ops can run on shared
  prototypes once, with results propagating to all instance sites.

Ask the user which of those they want and capture it in the decision packet
so Phase 4 knows whether to target prototypes or the monolith. Do not assume
restructure-for-its-own-sake.

### deduplicate-internally

User accepts the dedupe candidates but wants the stage to stay a single file.
**Run the same structured descent as the external path — do NOT skip it.** Invoke
`apply-restructure` with `dedupe.mode: internal_reference`: it authors the
value-hash nested library into an internal namespace, rewrites each duplicate site
as an `instanceable=true` reference, and returns a **restructure-role manifest**
(`identity_disposition: internal_share`) — the same frontier, identity gate, and
`kept_inline_for_merge` tagging as the external path. It does NOT rely on
`deduplicateHierarchies` for the mid-level merge — on 1.0.x that operator
authors a strong instanceable-reference collapse (nested-instancing support
landed in 1.0.4),
but without the manifest/identity contract, so it remains a candidate source
here. The last-mile
`deduplicateGeometry` cleanup still runs inside the authored leaf prototypes
(after any within-prototype merge).

The two paths differ ONLY in materialization (internal namespace + single file vs
extracted files) and parallelism (no per-asset Phase 4 fan-out for one file); the
descent decisions are identical. Skipping the frontier here is what leaves
high-count tiny repeats un-instanced and drops the merge frontier: with no
`kept_inline_for_merge` reservation, a later within-prototype merge runs *after*
the dedup/instancing passes already shared the geometry, has to un-instance it to
fuse, and inflates triangles/disk (the `hierarchy-dedupe-rewrite-tool-spec.md` §9
failure). The `kept_inline_for_merge` tagging reserves sub-MINP merge-candidate
leaves from the dedup/instancing passes so the merge runs **before** the
geometry-dedup tail.

Hand off to `apply-restructure`, then continue to Phase 3 with the restructured
(single-file) stage.

### extract-as-assets

User accepts the boundary candidates and wants external payloaded assets.
Invoke `apply-restructure` with:

- `restructure_plan`: the boundary cut points + dedupe candidates + `dedupe.mode: external_prototype`.
- `output_dir`: where to write prototype USDs and the new assembly root.
- `dry_run`: false (writes are executed).

`apply-restructure` returns a manifest of new prototype paths + the new
assembly stage root path. Continue to Phase 3 with the restructured stage.

### decompose-for-selective-loading

User wants selective loading boundaries without hierarchy dedup (no dedupe
candidates exist in this branch). Invoke `apply-restructure` with:

- `restructure_plan`: the selected boundary level + `goal: selective_loading`.
- `output_dir`: where to write payload USDs and the assembly root.
- `dry_run`: false (writes are executed).

Continue to Phase 3 with the decomposed stage.

### optimize-as-is

User accepts the existing structure. Skip Phase 2f. Continue to Phase 3 (instancing) and Phase 4 (mesh ops) targeting the original stage.

This gate chooses *how* to optimize, never *whether* to. There is no
diagnose-and-exit option: every branch proceeds into the optimization pipeline.

### already_optimized

Used when SA's `phase_recommendation = already_optimized`. Continue through the
pipeline; Phases 3-5 find no work, and Phase 6 produces a report with
`workflow_mode: no_op` and `verdict: neutral`. (No skip-to-verify shortcut.)

## How to ask

The Phase 2e prompt commits the user to a structural decision that downstream
phases cannot easily undo. The user must see exactly which Kit / Scene
Optimizer / usd-validation-nvidia versions authored the assessment and will execute
the restructure. **Prepend the full runtime context block** from
`skills/omniverse-usd-performance-tuning/references/setup-usd-performance-tuning/references/runtime-context-header.md` (Format A) before any of the analysis
or choice text below. Source: the `runtime_context` object in
`<output_path>/setup-preflight.json` (canonical location; see
`skills/omniverse-usd-performance-tuning/references/setup-usd-performance-tuning/references/runtime-context-header.md` *Where artifacts live*). If that
file is missing, invoke `setup-usd-performance-tuning` first.

Present the recommended branch with the evidence behind it, then list the alternatives. Example:

```
─── Runtime context ───────────────────────────────────────────────────────
Kit application:    USD Composer 110.1.0
  path:             D:\build\chk\usd_composer-fat\110.1.0+main.…\kit
  build:            110.1.0+main.10181.f4b28ef2.gl.windows-x86_64.release
Usd Optimize:    omni.scene.optimizer.core 110.0.4
usd-validation-nvidia:    usd-validation-nvidia 1.x.y via pip
───────────────────────────────────────────────────────────────────────────

The asset analysis shows:
  - 1 monolithic root layer, 0 references, 0 prototypes.
  - 4 repeated assembly patterns detected (suggesting 4 candidate prototypes
    saving an estimated 47% of prims).
  - Most duplicate geometry that would need per-mesh cleanup sits inside those
    4 repeated patterns — restructuring replaces it with shared prototypes, so
    per-target mesh fixes on the copies would be wasted (Tier 2/3 validation
    runs later, per prototype, in Phase 4).

Recommended: extract as payloaded, instanced assets. This will:
  - Materialize 4 prototype USDs to <output_dir>/prototypes/
  - Rewrite the root assembly to reference them
  - Run subsequent mesh ops on the prototypes (changes propagate)

Alternatives:
  - optimize-as-is: skip restructure, run mesh ops on the monolith. Faster
    to start but fewer downstream wins.

Which would you like?
```

## Output

Record the user's choice in the optimization plan and emit it for downstream phases:

```json
{
  "phase": "2e",
  "choice": "deduplicate-internally | extract-as-assets | decompose-for-selective-loading | optimize-as-is | already_optimized",
  "recommended": "deduplicate-internally",
  "reasoning": "monolithic with 4 repeated patterns; restructure recommended",
  "boundary_plan_ref": "<path to plan packet for apply-restructure>",
  "user_confirmed_at": "<ISO 8601 timestamp>"
}
```

## Rules

- Always present the choice; do not auto-proceed even when SA's recommendation is high-confidence.
- **Headless / batch / non-interactive contexts:** If the agent cannot ask the
  user (e.g. running in a scripted pipeline or with no interactive session),
  **STOP and write the decision as a blocker** in the preflight or report
  artifact. Do NOT substitute a default choice like "optimize-as-is" on the
  user's behalf. The gate exists because restructure-vs-optimize-as-is has
  irreversible consequences that only the user can weigh. Write a
  `restructure_decision_pending` artifact and halt Phase 2e until a human
  confirms.
- Do not recommend restructure when SA's `phase_recommendation = already_optimized`.
- Always present the selective-loading choice when SA reports `payload_count: 0`
  and clear asset-boundary candidates, even if hierarchy dedupe is not
  recommended and the asset is otherwise ready for mesh optimization.
- If the user picks `deduplicate-internally`, run Phase 2f (`apply-restructure`,
  `mode: internal_reference`) — the same structured descent as the external path,
  authored into an internal namespace in the single stage. It produces the
  restructure-role manifest (`internal_share`, `kept_inline_for_merge`); it does
  NOT skip the frontier, and `deduplicateHierarchies` remains a candidate
  source, not the authoring mechanism (it lacks the manifest/identity
  contract).
- If the user picks `extract-as-assets`, hand off to `apply-restructure` with
  the boundary plan and `goal: extract_as_assets`; do not perform writes from
  this reference.
- If the user picks `decompose-for-selective-loading`, hand off to
  `apply-restructure` with the selected boundary level and
  `goal: selective_loading`; do not perform writes from this reference.

## Limitations

- Decision skill only; does not write USD files.
- Depends on SA's classification quality; if SA's `phase_recommendation` is missing, return to `usd-structure-assessment` rather than guessing.
- Asset-boundary candidates from SA §2.5 are suggestions, not enforcement; the user can override the cut points before invoking `apply-restructure`.

## Troubleshooting

- If SA reports no candidates and the user wants to restructure anyway, ask for explicit cut points (prim paths) before invoking `apply-restructure`.
- If validator findings (Phase 2c) say the asset has structural issues that would block restructure (e.g. unresolved references), surface them to the user before asking for a choice.

## References

Read before deciding:

- `skills/omniverse-usd-performance-tuning/references/workflow.md` - the canonical 7-phase flow context for where this gate sits.
- `skills/omniverse-usd-performance-tuning/references/usd-structure-assessment/references/instancing-readiness/references/instancing-tradeoffs.md` - merge safety and dedupe trade-offs that affect the restructure-vs-optimize-as-is call.
- `usd-structure-assessment/README.md` §2.5 (Asset boundary identification) - the source of boundary candidates.
- `usd-structure-assessment/references/usd-edit-target-planner/README.md` - downstream skill that places the restructure outputs into a coherent edit-target plan.
