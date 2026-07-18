# USD Validation Runner (master router)

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

## When to Use

Use this reference when the performance workflow reaches Phase 2c (Tier 1
whole-stage), Phase 4 (per-target Tier 2/3 + 4d re-verify), Phase 6b, or an
iteration that needs validation evidence. Validators always run as phases of the
one optimize+validate pipeline; there is no validate-and-stop entry.

## Instructions

1. Confirm which validation phase of the optimization workflow this is (Phase
   2c, Phase 4, Phase 6b, or an iteration); validators run inside the pipeline,
   not as a standalone validation-only request.
2. Use structure assessment and profile evidence before selecting validators.
   Do not instantiate a validator engine, import Usd Optimize validators, or
   enumerate/run rules until a selected validation plan exists.
3. Select the smallest validation stack that can change the user-visible
   decision or operation plan.
4. Ask before any full usd-validation-nvidia sweep, full-stage (un-scoped) Tier 3
   probe, or expanded iteration scope. Scoped Tier 3 probes run without approval
   (see the Tier 3 coverage rule and Hard Rule 4).
5. Route execution to the owning validation reference or skill and preserve
   evidence paths for later reporting.

## Ownership Boundary

This runner is the single owner for validation scoping, full-sweep approval,
large-stage thresholds, masked-stage spot-check policy, and selected-probe
planning. Downstream validator references such as
`references/validate-usd-validation-nvidia.md` consume the scope note and own
runtime invocation details only.

## Pre-flight Checklist

Before running validators, confirm:

- [ ] The workflow has SA `summary_counts`, `phase_recommendation`, and
   `validation_scope` (always available, since validation runs as a pipeline
   phase downstream of structural assessment).
- [ ] The stage is classified for validation planning as small or large using
   the thresholds below.
- [ ] The plan names selected rules and probes, why they were selected, why a
   full sweep was skipped or approved, and artifact paths.
- [ ] Expensive checks and full sweeps have explicit user approval when needed.
- [ ] Findings will be routed to `usd-optimize-interpret-validators` for op-chain
   construction; do not map findings to ops yourself.

## Output Format

Return a scoped validation plan or validation summary naming the selected
validator stack, selected rules and probes, skipped expensive checks, approval
gates, artifact paths, and findings that affect the optimization plan.

For Phase 2c (whole-stage Tier 1) and for each Phase 4 target (per-target Tier
2/3), write a compact scope note matching
`scripts/validation-scope-note.schema.json`. Validators are named by **canonical
concept**, not runtime class name:

```json
{
  "scope": "targeted",
  "concepts": ["primvar_indexability", "geom_duplicates"],
  "targets": [
    { "concept": "primvar_indexability", "paths": ["/World/Racks/Rack_A"] },
    { "concept": "geom_duplicates", "mask_paths": ["/World/Racks"] }
  ],
  "tier_assignments": { "primvar_indexability": 2, "geom_duplicates": 3 },
  "selection_reason": "...",
  "full_sweep": { "status": "skipped", "reason": "...", "approved_by_user": false },
  "artifact_paths": ["..."]
}
```

The scope note is the input contract for `scripts/usd_validation_executor.py`.

## Purpose

Use this reference whenever a workflow needs to surface USD validity or
performance validator issues. It picks the smallest validation stack that can
affect the optimization plan, records the evidence contract, and routes concrete
execution to the owning skill or reference.

This reference **does not** execute optimization operations and **does not**
choose fix strategies.

For broad performance diagnosis, slow loading, high memory, low FPS, or "what
should I optimize?", start with `omniverse-usd-performance-tuning` so structure
assessment can scope validation before expensive validator runs.

For `omniverse://` targets, start with `omniverse-authentication` before this
skill attempts runtime probing or stage open.

## Prerequisites

- Target stage or asset paths and resolver context.
- Available validator runtime (Omni usd-validation-nvidia inside Kit, project-managed
  AV install, or installed Usd Optimize APIs).
- Artifact directory for logs, CSV/JSON findings, and provider summaries.
- Baseline, waiver, or failure policy for pre/post processing gates.
- For performance-stack scoping: `usd-structure-assessment` report with
  `summary_counts`, `phase_recommendation`, and `validation_scope`.

## Session-start runtime gate

If this reference is the **entry skill** for the user's request (i.e., the
agent invoked `/usd-validation-runner` directly rather than through
`omniverse-usd-performance-tuning`), run the session-start gate from
`skills/omniverse-usd-performance-tuning/references/setup-usd-performance-tuning/references/runtime-context-header.md`
before any routing. The gate determines `output_path`, checks
`<output_path>/setup-preflight.json`, invokes `setup-usd-performance-tuning`
if the preflight is missing, then surfaces Format A + the 4-option
confirmation. Do not pick a validator stack until the user has confirmed the
runtime.

If invoked downstream of an entry skill that already fired the gate in the same
session, skip the gate and proceed.

---

## Validation placement: 2c (whole-stage) vs Phase 4 (per target)

Validation is split by granularity, not run all at once:

- **Phase 2c — whole-stage, pre-restructure.** Tier 1 cheap whole-stage
  stats/probes plus the minimum-openability/structural checks that classify the
  stage and feed the 2e restructure decision. This is not a default validator
  sweep.
- **Phase 4 — per target, post-restructure.** Tier 2 (per `phase4_targets[]`
  entry) and Tier 3 (per flagged cross-component pair) run inside Phase 4's
  per-target loop, on the actual restructured artifact, scoped to one
  target/pair by construction. This is the primary model for per-target/per-pair
  concepts — see **Post-Restructure / Post-Decompose Validation Strategy**.
  (A pre-restructure Tier 2/3 diagnosis runs only on explicit user request.)

Required order at 2c:

1. Read Phase 1 profile and `usd-structure-assessment` output.
2. Classify the asset as small or large for validation planning.
3. Build the Tier 1 + structural plan from `summary_counts`,
   `phase_recommendation`, and `validation_scope`.
4. Record the scope note/artifact.
5. Only then run the selected Tier 1 rules/probes.

For monolithic `optimize-as-is`, the original stage remains the optimization
target (Phase 4 target N=1), and its Tier 2/3 validation runs in the Phase 4
loop like any other target. A monolithic target does not authorize a full sweep.

## Large for Validation Planning

Treat a stage as **large for validation planning** when any condition is true:

- Resolved stage/root package size is unknown or `>100 MB`.
- Composed prim count is `>10,000`.
- Mesh count or prototype/proxy mesh contribution is high enough that a
  category sweep would traverse substantial geometry.
- The target is customer-scale CAD/BIM/MEP/factory/plant/city content.
- The request is performance optimization rather than formal conformance.

Large-stage behavior:

- Do not run a default full-stage usd-validation-nvidia or Usd Optimize rule sweep.
- Ask before full sweep if the user explicitly wants exhaustive validation.
- Prefer minimum-openability, Tier 1 cheap whole-stage stats/probes, targeted
  rules, Tier 2/3 subprocess runs with timeouts, or masked-stage
  spot checks.
- Record skipped full-sweep rationale in the scope note/artifact.

## Full-Sweep Approval Gate

Trigger before any command or API call that enables the default AV rule set,
all registered rules, all categories, or all SO performance validators over the
whole composed stage when any large-stage condition above holds.

Ask before full sweep and offer:

- **Recommended:** minimum-openability + targeted rule/probe checks.
- **Full sweep:** default rule set with explicit timeout and artifact dir.
- **Defer:** skip full sweep until after mutation or a narrower follow-up.

When approved, record `scope: "approved_full_sweep"`,
`approved_by_user: true`, `timeout_seconds`, and artifact paths. If not
approved, do not launch.

## Validator Tiers

Tiers describe **execution posture**. Which concept is which tier lives only in
`validator-concepts.json` — do not infer or assert a concept's tier here.

### Tier 1: Cheap Whole-Stage Stats/Probes

Tier 1 registry concepts plus pure profiling probes that are not concepts
(`printStats`, `countVertices`). Safe to run in one batch over the SA-selected
target; not a default AV all-rules sweep.

### Tier 2: Targeted Medium Probes

Tier 2 registry concepts, run per `phase4_targets[]` entry (or a bounded sample)
in killable subprocesses, inside the Phase 4 per-target loop.

### Tier 3: Expensive Probes (evidence-gated, mandatory when flagged)

Spatial, pairwise, or high-cardinality analysis. The Tier 3 set is exactly the
concepts `validator-concepts.json` marks `tier: 3` — resolve it from the
registry, do not enumerate it here (see the rule at the top of this section).

**Tier 3 is not optional.** When structure assessment flags a target for a
Tier 3 concept, running the **scoped probe is required** — it carries signal the
later op plan depends on, and skipping it is how runs miss real optimizations.
What is approval-gated is *cost*, not *coverage*:

- **Scoped probe = default, no approval needed.** Restrict to the flagged
  paths/pairs with `paths=` / `Usd.Stage.OpenMasked()` and run in a bounded
  subprocess with a timeout. This is the normal Tier 3 path — it runs in the
  Phase 4 per-target loop, per flagged pair.
- **Full-stage probe = approval-gated.** Only run the un-scoped, whole-stage
  version after the full-sweep approval gate.
- **Timeout is a recorded disposition, not a skip.** If the scoped probe times
  out, record `timeout_recorded` and retry a masked/standalone sample — do not
  drop the target.

**Analyze ≠ apply.** The scoped probe is *analysis* and is never gated on cost —
it produces a candidate list only. That is the runner's own scoping rule, and it
is all this section owns: scoped probes run without approval. Any destructive
*apply* its findings motivate (e.g. `removePrims` of occluded interiors) is gated
**separately** under the apply-authority model owned by
`usd-optimize-run-operations/references/operation-safety.md` "Apply authority" — see that
section for why the apply (not the probe) needs user intent and how it is
surfaced. So "ask the user" attaches to the apply step, not to the probe.

Every flagged Tier 3 target must end in a coverage-ledger disposition (see
**Completion Gate**). "I skipped it because it was expensive" is not a valid
outcome; the valid outcomes are probed (clean or with findings), `user_declined`
after an explicit ask, `timeout_recorded`, or `blocked_validation_runtime`.

## Completion Gate (coverage ledger)

`scripts/usd_validation_executor.py` emits a `coverage_ledger` in every
`validation-report.json`. Each flagged `(target, concept)` from the scope note
must appear with a resolved status:
`probed_with_findings | probed_clean | user_declined | timeout_recorded |
blocked_validation_runtime`. `coverage_ledger.complete` is `true` only when no
flagged target is unresolved, and the report `summary.status` is `BLOCKED`
until it is. **Do not advance to the optimization report or declare the
iteration done while the ledger is incomplete.**

## Tier Decision Inputs

No schema contains a single `tier` field. Tier selection is policy applied to
the structure-assessment and validator reports:

| Source | Fields | How they affect tier/scoping |
|---|---|---|
| `usd-structure-assessment-report.schema.json` | `phase_recommendation` | Selects the default validation posture: `structuring`, `optimization`, or `already_optimized`. |
| `usd-structure-assessment-report.schema.json` | `summary_counts.prim_count`, `summary_counts.mesh_count`, `summary_counts.prototype_count`, `summary_counts.instance_count`, `summary_counts.reference_count`, `summary_counts.payload_count` | Determines large-stage status and whether Tier 2/3 must run per target, sampled, or not at all. |
| `usd-structure-assessment-report.schema.json` | `validation_scope.per_asset`, `validation_scope.cross_component_pairs`, `validation_scope.skip` | Defines the concrete target set for Tier 2 and Tier 3. |
| `usd-structure-assessment-report.schema.json` | `findings`, `hierarchy_dedupe.recommended`, `hierarchy_dedupe.top_candidates` | Supplies reasons to include targeted Tier 2 probes or to ask for Tier 3 probes. |
| `validator-concepts.json` | `tier`, `cost_class`, `gpu_bound`, `scope_policy` per canonical concept | Single source of truth for a concept's tier and scope. Read it; do not restate tiers elsewhere. |
| `rule-reference.md` | Validator signal → canonical concept → backing op | Interpretation map only (signal to concept to fix op). Carries no tier. |
| `validation-report.schema.json` | `validators[].canonical_name`, `validators[].status`, `validators[].issues`, `summary.errorCount`, `coverage_ledger` | The canonical executor's own report — what ran (by canonical concept and resolved `(module, class_name)` identity) and what was found. Use it to narrow later iterations, not to widen scope silently. |

Selected validators are named by **canonical concept name** (e.g.
`primvar_indexability`, `geom_duplicates`), defined in
`references/validator-concepts.json`. The canonical executor resolves each
concept to a unique `(module, class_name)` identity at run time. Do not put
runtime class names (`IndexedPrimvarChecker`), operation names, display labels,
or category names (`Geometry`, `Usd:Performance`) in the plan — class names are
not unique across providers and categories are lookup buckets, not approval
scope. The registry's `preferred_provider` decides Usd Optimize vs Asset
Validator; performance tuning prefers the Usd Optimize implementation.

## Phase-Aware Defaults

| `phase_recommendation` | Default scope |
|---|---|
| `structuring` | Minimum-openability + targeted structural blockers only. Do not validate geometry about to be restructured. |
| `optimization` | **2c:** Minimum-openability + Tier 1 cheap whole-stage stats/probes. **Phase 4 (per target):** Tier 2 per `phase4_targets[]` entry; Tier 3 scoped probe mandatory per flagged cross-component pair. No full-stage Tier 2/3 default; whole-stage Tier 3 only on explicit user request. |
| `already_optimized` | Minimum-openability + Tier 1 cheap whole-stage stats/probes only; ask before expanding. |
| missing | Run structure assessment first. Do not begin with validators. |

## Deterministic Selection

Selection is a **function of structure-assessment evidence, not agent
judgment**. Two runs over the same SA report must select the same concept set,
because disagreement between runs is the variance this runner exists to remove.
Apply the table top-to-bottom; each matched row contributes its concepts. Do not
add concepts that no row selects, and do not drop a concept a row selects. Tier
and scope policy for each concept come from `validator-concepts.json` (the
"Target" column states only the selection granularity, not the tier).

**Execution phase follows the "Target" granularity:** **whole-stage** rows
(the "Always" safety gate + the `optimization` whole-stage row) run at Phase 2c
as Tier 1; **per-target / flagged-pair / flagged-subtree** rows run in the Phase
4 per-target loop on the restructured artifact (target N=1 = the monolith when
no restructure happened). The selection function is identical in both phases;
only the artifact it runs against differs.

| SA signal (condition) | Concepts selected | Target |
|---|---|---|
| Always (any `optimization`/`already_optimized` run) | `composition_missing_ref`, `material_path`, `material_dangling_binding`, `texture_bind`, `texture_normalmap` | whole-stage safety gate |
| `phase_recommendation = optimization` | `material_duplicates`, `structure_empty_leaf`, `structure_invisible`, `structure_flat_hierarchy`, `extents_zero`, `perf_small_mesh`, `perf_sparse_mesh`, `perf_rtx_mesh_count`, `perf_redundant_timesamples`, `perf_high_vertex_count` | whole stage |
| Asset posture is CAD / BIM / MEP / converted (e.g. Revit/HOOPS) | `primitive_fit` | per flagged target — **mandatory**, never dropped |
| Any `optimization` run — cheap, lossless/self-gating primvar cleanup (no flag required) | `primvar_indexability`, `primvar_unused` | per target or sample |
| Any `optimization` run — geometry-intrinsic mesh hygiene (no flag required) | `vertex_weld`, `topology_zero_area_faces`, `normals_winding` | per target, executed at Phase 4 per `phase4_targets[]` |
| `hierarchy_dedupe.recommended` or duplicate-geometry signal | `geom_duplicates` (+ `geom_duplicates_fuzzy` if near-duplicates) | flagged subtree |
| `validation_scope.cross_component_pairs[*]` not explicitly transparent (`enclosure_opaque` true or unset) | `spatial_occluded` | flagged pair — **mandatory** scoped probe |
| `validation_scope.cross_component_pairs[*]` (routing/overlap) | `spatial_overlapping`, `spatial_coinciding` | flagged pair — **mandatory** scoped probe |
| Target is simulation-ready (physics/Boolean/3D-print), not visualization | `topology_manifold`, `normals_validity` | flagged target |

If `validation_scope.skip` lists a target, it is excluded from all rows. With
**zero** flagged assets, the "Always" + whole-stage rows, the flag-independent
per-target rows (cheap primvar cleanup, mesh hygiene), and the CAD/converted
`primitive_fit` row all still fire; only the evidence-gated rows (Tier-3 spatial
pairs, duplicate-geometry, sim-ready) stay empty when their signal is absent. Ask
before adding concepts no row selects.

**Why the primvar/mesh-hygiene rows are not flag-gated.**
`primvar_unused`/`primvar_indexability` (lossless, self-gating UV/primvar cleanup)
and the mesh-hygiene trio (`vertex_weld`, `topology_zero_area_faces`,
`normals_winding`) are geometry-intrinsic, not spatial. SA is structural-only and
reads no geometry arrays, so it cannot pre-flag a primvar/mesh-hygiene issue at
all; gating their *selection* on any SA structural flag would make them
unreachable on extent-only CAD — detection would silently never run. Selection is
therefore
evidence-independent (always, on any `optimization` run); the registry
`cost_class: cheap` / `per_target_or_sample` policy still governs cost and
granularity, and the *apply* step (lossless auto-apply for primvars;
`meshCleanup` confirmation-gated) still governs whether a fix is written. The op
being lossless is irrelevant if the concept is never selected to recommend it.

`validation_scope.cross_component_pairs[*]` items are boundary-ID reference
objects resolved through `asset_boundary_suggestions.boundaries[]`; SA owns how
pairs are nominated (`candidate_source` hash OR semantics) and when `spatial_role`
/ `enclosure_opaque` are set — see `usd-structure-assessment` §2.1. A pair is
probed unless it is *explicitly* transparent (`enclosure_opaque: false`); `true`
or unset (unknown) both schedule the mandatory Tier-3 probe, which resolves
opacity. `bbox_confirmed` is informational confirmation only — it never gates
whether the probe runs.

## Iteration Subtraction

Re-validation in later iterations is **same-or-narrower by construction**:

- Start from the previous iteration's selected concept set.
- **Subtract** every `(target, concept)` whose ledger status was `probed_clean`
  or that a completed operation resolved. Resolved-clean targets are not
  re-probed.
- **Keep** targets that were `probed_with_findings` (re-verify the fix),
  `timeout_recorded` (retry masked/standalone), or regressed.
- **Never widen** to new Tier 3 targets/pairs, new concepts no SA row selects,
  or full-stage scope without explicit user approval.
- Keep the FIRST pass's baseline metrics; do not re-baseline.

This makes each pass cheaper and convergent, and guarantees a later run cannot
silently disagree with an earlier one by re-expanding scope.

## Scoping Rules

1. Structure assessment is the first filter. Use `summary_counts`,
   duplicate-hierarchy candidates, and `validation_scope` to
   decide which validators can change the optimization plan.
2. **Which concepts to run is decided by Deterministic Selection above; tier and
   scope policy come from `validator-concepts.json`.** This section does not
   re-derive selection or tiering.
3. Do not start performance work with a full default AV sweep.
4. Keep SO analysis in the validation workflow. Importing Usd Optimize validators makes
   rules discoverable; it does not authorize running all of them.
5. For cross-component validators, use `Usd.Stage.OpenMasked()` covering only
   the flagged pair and dependency closures, or validate standalone target files.
6. Do not run noisy/slow concepts globally in Phase 2c. Any registry concept
   that is `gpu_bound`, `cost_class: expensive`, or `stage_dependent` is scoped
   to flagged targets/pairs only — never a full-stage default.
7. Category-scoped usd-validation-nvidia is still a scoped whole-stage traversal for that category.
   On large stages, ask before full sweep and prefer masked spot checks or
   bounded parallel subprocesses with timeouts.
8. Prefer summaries over issue dumps. Apply
   `runtime-artifact-token-budget.md` for CSV/log/summary handling.

## Selected-Rule Execution Pattern

Do not use `ValidationEngine()` or
`ValidationEngine(init_rules=True)` unless the user explicitly approved
exhaustive validation. That pattern runs every registered OAV rule plus every SO
validator that auto-registered.

Execution model:

- **Tier 1:** run selected cheap whole-stage stats/probes in one batch for the
  scoped target. This is not a default all-rules sweep.
- **Tier 2:** run selected rules per target in isolated OS subprocesses with an
  explicit wall-clock timeout. Parallelize independent target/rule subprocesses
  within resource budget.
- **Tier 3:** ask first, then use the same subprocess pattern on flagged targets
  only.
- **Timeout fallback:** if a Tier 2 or Tier 3 rule times out, record a timeout
  finding and rerun a masked-stage spot sample or standalone payload/prototype
  sample instead of widening to a full sweep.
- **Do not batch Tier 2/3 rules in one engine** unless the target is small and
  the user explicitly accepted the risk. One slow C++ rule can dominate or hang
  the whole batch, and Python `signal.alarm` or threads may not interrupt it.

Inside Kit, import `omni.asset_validator.core` instead of
`omni.asset_validator`, but keep the same selected-rule posture. Ask before full
sweep before any copyable pattern that enables default/all rules.

### Canonical executor (the only supported runner)

The runner ships a canonical executor at `scripts/usd_validation_executor.py`.
**Call it directly — do not reimplement rule resolution and do not write your
own script.** It resolves each canonical concept to a unique `(module,
class_name)` via `references/validator-concepts.json`, enables exactly those
rule classes (never `init_rules=True`), and opens the stage scoped. It is
fail-closed by contract: unknown concept, ambiguous identity, unregistered rule,
or missing runtime all raise — there is no bare-name lookup and no CLI fallback.
This is what disambiguates the Usd Optimize `IndexedPrimvarChecker` (fast
triage) from the usd-validation-nvidia one (full audit) that share a class name.

```python
from usd_validation_executor import (
    load_registry,
    validate_concepts,
    ValidationRuntimeUnavailable,
)

registry = load_registry()  # references/validator-concepts.json

# Tier 1: one batch over the SA-selected target.
issues = validate_concepts(
    stage_path,
    ["material_duplicates", "structure_empty_leaf"],
    registry=registry,
)

# Tier 2 / Tier 3: one concept + target group per bounded subprocess.
issues = validate_concepts(
    stage_path,
    ["primvar_indexability"],
    registry=registry,
    mask_paths=["/World/Racks/Rack_A"],
)
```

If the validation runtime cannot be imported, `validate_concepts` raises
`ValidationRuntimeUnavailable`; record `blocked_validation_runtime` in the
coverage ledger rather than fabricating a pass.

Call `validate_concepts` once for a Tier 1 batch. For Tier 2 and Tier 3, run the
whole scope note through `run_scope_note` with the **subprocess** runner so each
concept executes in a killable child process and timeouts become a recorded
disposition rather than a hang:

```python
from usd_validation_executor import run_scope_note, subprocess_concept_runner

report = run_scope_note(
    stage_path,
    scope_note,                       # validation-scope-note.schema.json
    registry=registry,
    concept_runner=subprocess_concept_runner(timeout_seconds=120),
    phase="baseline",
)
# report["coverage_ledger"]["complete"] gates "done"; timeouts -> timeout_recorded.
```

`subprocess_concept_runner` invokes this module as a child (`python
usd_validation_executor.py`, JSON job on stdin) — an internal worker protocol,
not a CLI. The default in-process runner is for Tier 1 only, where a hang is not
a risk. If a concept times out, `run_scope_note` records `timeout_recorded`;
retry that target with `mask_paths` from the spot-check policy below.

## Masked-Stage Spot Checks

Use masked-stage spot checks when full-stage or per-target validation is too
expensive but prim-level findings can still change the optimization plan.

Use spot checks when:

- Stage is large for validation planning.
- SA can identify representative candidate subtrees.
- Rule set is mostly prim-local geometry/material/schema checks.
- Tier 2 or Tier 3 subprocess validation times out.
- Result is optimization evidence, not formal conformance.

Sample selection:

1. Build a cheap whole-stage inventory first: top branches by mesh count,
   semantic names, top prototype/fingerprint groups, material-heavy branches,
   and instance-heavy branches.
2. Include all SA-flagged targets that may change the operation plan.
3. Cover at least 25% of mesh-bearing content by mesh count, `rtxMeshCount`, or
   instance-proxy mesh contribution. If impractical, record why and mark the
   result as limited sample evidence.
4. Include high-risk exemplars: largest mesh, deepest hierarchy,
   material-heavy mesh, repeated module, top prototype/fingerprint family, and
   dominant mesh-bearing semantic classes.
5. Add closure paths needed by the sample, such as material/looks scopes or
   shared class/inherit sources.
6. Reject empty samples; if proxy/prototype-aware counts report 0 meshes,
   resample instead of reporting "0 findings."

Label output `scope: "masked_stage_spot_check"` with sampled paths, semantic
tags, mesh coverage percentage, and evidence scope.

## Post-Restructure / Post-Decompose Validation Strategy

This is the **primary model** for Tier 2/3 (per-target/per-pair) validation, not
an exception: it runs inside the Phase 4 per-target loop after restructure.

After `apply-restructure` or `decompose-for-selective-loading` produces an
assembly root plus payload/prototype files, do not open the full composed stage
with all payloads loaded for a blanket validator sweep.

- **Assembly skeleton:** open with `Usd.Stage.OpenMasked()` excluding payload
  prim paths. Run structural validators only: reference resolution, kind
  hierarchy, layer structure, defaultPrim, extent hints, assetInfo.
- **Assembly root as optimization target:** if it retains mesh content after
  extraction, validate and optimize it like any other target using this policy.
- **Each payload/prototype:** open each file independently with
  `Usd.Stage.Open(payload_file)`. Plan validation per target based on that
  target's prim/mesh count.
- **Cross-payload pairs:** open with `Usd.Stage.OpenMasked(root, mask)` covering
  only the relevant payload subtrees. Run Tier 3 only per flagged pair.

Each target re-enters this runner independently; approval gates and spot-check
thresholds apply per target, not to the original composed stage.

## usd-validation-nvidia Load Rules

The usd-validation-nvidia's `ComplianceChecker` opens a new stage from the input's
root layer with default `LoadAll` semantics. Caller `StageLoadRules` such as
`LoadNone` are discarded. `StagePopulationMask` is preserved, so
`Usd.Stage.OpenMasked()` is the reliable scoping mechanism.

Do not rely on `LoadNone` or `stage.Load(specific_path)` for validation
scoping. Use `OpenMasked` or validate standalone payload/prototype files.

For small/medium stages, use the standard selected validation plan via
`usd-optimize-run-validators`, but keep the same tier execution model: Tier 1 may batch;
Tier 2 and Tier 3 use bounded subprocesses.

## Validation Plan Shape

The plan is the scope note defined by `scripts/validation-scope-note.schema.json`
— there is no separate plan format. **Deterministic Selection** decides which
concepts the note contains; the registry supplies each concept's tier and scope
policy; masked spot-check fields are described under **Masked-Stage Spot
Checks**.

## Routing Decision

| Intent | Stacks |
|---|---|
| Validate this USD before mutation | Pre-mutation USD stack: minimum-openability plus targeted usd-validation-nvidia coverage when needed. |
| Broad performance ask | `usd-structure-assessment` first, then selected performance stack per this runner. Add pre-mutation USD stack only when validity affects mutation safety. |
| Run perf validators only | Performance stack only, selected from SA evidence or the user's explicit target list. |
| Validate optimized output | Phase 4 re-verify per target (4d) + the Phase 2c Tier 1 whole-stage re-run in Phase 6b for a fair stage-level comparison; same-or-narrower unless the user approves expansion. |
| Formal conformance/exhaustive validation | Ask before full sweep, then route through the selected usd-validation-nvidia/runtime with explicit timeout and artifacts. |

Doc ownership for each stack (read the owning doc before writing commands):

- Pre-mutation USD stack -> `references/validate-usd-validation-nvidia.md`.
- Performance stack (Usd Optimize validator execution) -> `references/usd-optimize-run-validators/README.md`.
- Turning validator findings into operations -> `references/usd-optimize-interpret-validators/README.md`.
- Concept selection / tiers / scope notes -> this README plus
  `references/validator-concepts.json` (registry; tiers live only there).


## Required Gates

Pre-processing:

- Stage opens.
- Asset paths resolve.
- Minimum-openability and selected checks complete.
- Known blocker findings are either fixed or waived.

Post-processing:

- Stage opens.
- Validation is no worse than baseline unless explicitly accepted.
- Generated outputs are recorded.
- Processor report and validation report are attached to the optimization plan.

## Output

Emit `validation-report.json` matching `scripts/validation-report.schema.json`
when that report is produced. The report must point to provider artifacts such
as `issues.csv`, `provider-summary.json`, and `run.log`, and include the chosen
phase scoping so Phase 6 and Phase 7 can reproduce or narrow it.

## Hard Rules

1. Never run all validators on all assets by default.
2. Never use `ValidationEngine()` or `ValidationEngine(init_rules=True)` after
   Usd Optimize validator registration unless exhaustive validation was approved.
3. Never run Tier 3 without structural evidence from the assessment.
4. When SA flags a Tier 3 target, the **scoped** probe is mandatory and needs no
   approval; ask only before the **full-stage** version. Silent omission of a
   flagged expensive probe is a defect, not a cost saving.
5. Ask before full sweep on any large stage.
6. Never start a performance workflow with a full default AV sweep.
7. Prefer masked-stage spot checks over dropping validation when full-stage
   validation is too expensive.
8. Run Tier 2 and Tier 3 validation through bounded subprocesses; if a rule
   times out, record `timeout_recorded` and retry with a masked or standalone
   sample — never silently drop the target.
9. Always report what was skipped and why; the user may override.
10. Never declare an iteration done while `coverage_ledger.complete` is `false`.
    Every flagged `(target, concept)` must reach a resolved disposition first.

## Troubleshooting

- If `omni_asset_validate` is unavailable, record it as missing rather than
  fabricating a pass.
- If Usd Optimize validator imports fail, do not report SO-specific results.
- If the bundled `validator-venv` is slow or lacks dependencies, prefer a Kit or
  project-managed usd-validation-nvidia environment.
- Named validator unavailable: record the gap and choose the nearest supported
  source only when it answers the same scoped question.

## References

- `references/validate-usd-validation-nvidia.md` - usd-validation-nvidia runtime
  invocation details.
- `skills/omniverse-usd-performance-tuning/references/usd-validation-runner/references/usd-optimize-run-validators/references/infrastructure.md` - Usd Optimize validator infrastructure.
- `skills/omniverse-usd-performance-tuning/references/workflow.md` - canonical
  7-phase flow context for where validation sits.
