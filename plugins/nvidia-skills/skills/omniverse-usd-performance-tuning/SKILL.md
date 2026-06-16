---
name: omniverse-usd-performance-tuning
description: "Top-level workflow skill for USD performance diagnosis and optimization. Handles slow loading, high memory, low FPS, and broad scene-optimization requests; delegates auth/runtime setup to Phase 0 owners."
version: "0.1.0"
license: Apache-2.0
tools:
  - Read
  - Shell
  - Write
compatibility: >
  Orchestrator skill. Downstream phases may require Kit, Usd Optimize, usd-validation-nvidia, USD Python, writable output paths, and omniverse:// authentication selected by setup-usd-performance-tuning.
metadata:
  author: NVIDIA Omniverse
  tags:
    - triage
    - performance
    - usd
    - profiling
  domain: ai-ml
  languages:
    - python
---
# Omniverse USD Performance Tuning

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

## When to Use

Use this workflow for broad performance asks such as slow loading, high memory, low FPS, GPU crashes, conversion-quality triage, or generic requests to optimize a USD scene.
Do not invoke this performance workflow for non-performance build requests such
as viewer or application creation unless the user separately asks for USD
performance diagnosis or optimization.

## Instructions

1. Start from the mandatory runtime context gate before producing tuning output, unless the prompt is only asking for a static classification test.
2. Classify every optimization request as `ready_to_plan` and route it through the one full optimize+validate pipeline; never run a named operation, a validation-only pass, or a structure-only exit as a standalone bypass of structural assessment, validators, and the op chain. Destructive/lossy ops are gated where they execute by the apply-authority class in `operation-safety.md` (`intent-gated` → surface approval at the apply gate), not pre-authorized at plan time.
3. Plan the full canonical chain through `optimization-report`, preserving the structured milestone order and the `profile-stage:baseline` / `profile-stage:after` labels when listing milestones. For broad optimization, default to 3 scoped iterations unless the user opts out, asks for a quick pass, or stop criteria apply.
4. Invoke downstream skill bodies only when their phase is reached, and keep raw runtime artifacts on disk while reading compact summaries.

Frontmatter keeps `version` and `tools` at top level for agentskills.io runtime
compatibility. NVCARPS discoverability fields live under `metadata`.

## Output Format

<!-- The milestone subsequence below is duplicated in references/workflow.md
     (Milestones section). Keep the two lists character-identical. -->
Return a plan or status summary that names the selected entry skill, uses `ready_to_plan` for generic optimization requests, includes the full milestone chain through `optimization-report`, and labels profile phases as `profile-stage:baseline` and `profile-stage:after`. For structured outputs, the broad-optimization milestone subsequence is `omniverse-usd-performance-tuning` -> `profile-stage:baseline` -> `usd-structure-assessment` -> `usd-validation-runner` -> `restructure-decision` -> `apply-restructure` -> `usd-optimize-run-validators` -> `usd-optimize-interpret-validators` -> `usd-optimize-run-operations` -> `profile-stage:after` -> `compare-profiles` -> `optimization-report`. End-to-end execution should produce an optimized stage when mutation runs and a report conforming to the `optimization-report` reference's schema (`scripts/optimization-report.schema.json` within that reference). Broad optimization should plan 3 scoped iterations by default; each iteration writes an interim report/update and later passes reuse prior evidence instead of restarting the full workflow.

## Entry skill rule

This skill is the named entry point for broad performance work whenever the
agent has any verified way to do that work. Runtime probing details live in
`setup-usd-performance-tuning`; this rule only decides which skill owns the
user-facing performance request.

- If the setup probe shows **any** verified runtime path - Kit, standalone, or
  even a partial stack such as usd-validation-nvidia only - enter here. If the
  user's requested tool is missing, return the specific `blocked_code`
  (`blocked_missing_usd_optimize`, `blocked_missing_usd_optimize_operation`, etc.)
  instead of substituting another workflow.
- Enter at `setup-usd-performance-tuning` only when **no** runtime path is
  verified and runtime choice/setup is the first unresolved problem.
- For `omniverse://` assets, enter at `omniverse-authentication` first.
  Authentication precedes setup and triage for remote assets.

The decision is about ownership, not order. Setup, authentication, and triage all run in their normal phase order; this rule only fixes which skill the agent **names as the entry skill** in its response.

## Runtime context — session-start gate (mandatory)

**Before any other tuning output**, follow the mandatory session-start gate in
`references/setup-usd-performance-tuning/references/runtime-context-header.md`.
That reference owns `output_path`, the canonical `setup-preflight.json`
location, Format A/Format B, and the "do not improvise a silent probe"
anti-pattern.

Required outcomes:

- Missing or unreadable preflight: invoke `setup-usd-performance-tuning`.
- Present preflight: print Format A and wait for the user to choose Continue,
  Change Kit, Switch to standalone, or Re-run probe.
- Confirmed runtime in the same session: use compact Format B for follow-up
  status.

```
[Kit: {runtime_context.kit.application} {runtime_context.kit.version}  |  SO: {runtime_context.usdOptimize.version}  |  AV: {runtime_context.assetValidator.version}]
```

## Response robustness

A setup or runtime gate blocks runtime *execution* — it never blanks out the
response. Some models over-treat the gate as a reason to stop; do not.

- **Always return a non-empty, routed response.** If preflight is missing,
  invoke `setup-usd-performance-tuning` as Phase 0 and *still* emit the entry
  skill, the `decision`, and the full planning skeleton through
  `optimization-report` in the same response. Never make "waiting for setup" the
  whole reply.
- The gate blocks runtime *execution* — starting Kit, running Usd Optimize or
  usd-validation-nvidia, profiling, log inspection. It does **not** block planning,
  reading existing reports, detecting zero-work from before/after metrics, or
  issuing approval prompts.

## Runtime artifact token budget

Before reading Kit logs, usd-validation-nvidia CSVs, Usd Optimize logs, Tracy CSVs,
or other runtime output, follow
`references/runtime-artifact-token-budget.md`. Keep raw artifacts on disk, read
summary JSON first, and use bounded log snapshots instead of full dumps or live
streams.

## Plan-time vs execution-time approval

Approval is an **apply-authority concern**, not a plan-time routing branch. Every
optimization request — generic or naming a specific destructive op — is planned
through the one full pipeline (structural assessment → validators → op chain).
A named destructive op becomes an evidence-driven step gated where it executes
(`operation-safety.md` apply-authority class); there is no "request names a
destructive op → run just that op" shortcut, and a request never pre-authorizes
an `intent-gated` operation.

**The `decision` token is DERIVED from the response's own shape — never judged
from whether the request mentioned a destructive op:**

- `blocked` ⟺ a `blocked_code` applies: a runtime/auth obstacle stops every
  committed step.
- `approval_required` ⟺ **this response halts awaiting the user**: the
  committed plan stops at a gate the response is surfacing now
  (`approval_required_reason` names it) and `planned_phases` carries the
  post-approval continuation.
- `ready_to_plan` ⟺ otherwise: nothing in this response awaits the user.
  Future gates — a later `restructure-decision` prompt, `intent-gated` ops
  collected for the Phase 7 opt-in menu — belong in `gates_observed`, never in
  `decision`.

Derivation invariants (enforced by the runtime harness): `ready_to_plan` ⇒
`committed_milestones` equals `planned_phases`; `approval_required` ⇒
`committed_milestones` is a strict prefix of `planned_phases`. If your draft
response violates an invariant, the `decision` value is wrong — recompute it
from the lists, not the other way around.

Before executing a destructive or lossy operation such as flattening,
decimation, deletion, merge, quantization, primitive fitting, or topology edit,
ask for approval. The approval prompt must name the intended output path, state
that the source will not be overwritten unless in-place overwrite was requested,
and summarize baseline assessment, pre-mutation validation, planned
post-mutation validation, and operation-specific guardrails.

For structured runtime-test responses and similar planning summaries:

- Derive `decision` per *Plan-time vs execution-time approval* above; do not restate or re-judge it per scenario.
- Whenever a chain names profile phases, use the exact labels `profile-stage:baseline` and `profile-stage:after`; do not emit the ambiguous bare `profile-stage` token.
- Start structured milestone lists with `omniverse-usd-performance-tuning` as the owning entry skill. Include `setup-usd-performance-tuning` only as additional Phase 0 context, not as a replacement for the entry skill milestone.
- For broad optimization requests, preserve the milestone subsequence from *Output Format* above exactly, with optional extra analysis steps inserted only where they do not reorder it.
- Do not list `usd-optimize-run-validators` or `usd-optimize-interpret-validators` before `restructure-decision` in broad optimization milestone summaries. Phase-aware validator routing still happens through `usd-validation-runner`; the Usd Optimize validator executor/interpreter milestones appear after the restructure decision path in the structured plan contract.

## Output expectation

End-to-end work produces an optimized USD stage (when mutation runs) and a
structured report conforming to the `optimization-report` schema
(`scripts/optimization-report.schema.json`); render the HTML from
`references/report-templates/optimization-report.html.template` via
`render_preview.py` (never hand-write HTML), and report the optimized-stage,
JSON, Markdown, and HTML paths. If schema validation or HTML rendering did not
run, report the run as blocked/incomplete, not complete.

Every request runs the full pipeline — there is no diagnose-and-exit or
validate-and-stop mode. The one legitimate stage-less ending is the
**runtime-forced degraded path** (Scene Optimizer unavailable and the user
declines install/setup): set `workflow_mode: "structural_only"`, a verdict such
as `no_optimized_stage_written`, the blocked/missing-optimizer evidence, and a
`next_profile_capture` note for later runtime profiling.

## Purpose

Route digital twin USD performance requests into the right diagnostic and
optimization workflow while preserving evidence before mutation.

## Prerequisites

- Stage path or enough context to identify the target asset.
- Runtime availability status from `setup-usd-performance-tuning` when not already known (standalone is the sole optimize+validate runtime; Kit→omniperf is an opt-in profiling adjunct).
- Permission status for in-place mutation vs writing a separate optimized output.

## Examples

- "This USD loads slowly; triage what to check first."
- "Route a low-FPS CAD scene through the performance workflow."

## Triage order

0. **Runtime gate.** Follow the mandatory session-start gate above before
   validation, profiling, or optimization. Do not scan, probe, install, or pick
   Kit/standalone runtimes directly in this skill; `setup-usd-performance-tuning`
   owns probe/chooser/install dispatch and writes the preflight consumed here.

1. Identify the target problem:
   - Load time.
   - FPS or interactivity.
   - GPU or system memory.
   - Crash or device lost.
   - CAD conversion quality.
   - Validation failure.

2. Gather minimum context:
   - Stage path and size.
   - Whether the stage is local, mounted, or `omniverse://` remote. For remote
     assets, route through `omniverse-authentication` before first open.
   - Standalone USD runtime (the optimize+validate runtime); note separately if
     the user explicitly wants the opt-in Kit→omniperf profiling adjunct.
   - Whether the workload is CAD, VFI, a data-center digital twin, Isaac, or generic OpenUSD.
   - Whether in-place mutation is allowed.

3. Route:
   - USD composition questions: `usd-structure-assessment` (composition is now part of the SA umbrella; deeper detail in `references/usd-structure-assessment/references/composition-audit.md`).
   - Validation and content issues: `usd-validation-runner` (master router; routes to `validate-*` family or `usd-optimize-run-validators` based on intent).
   - Edit/output decisions: `usd-edit-target-planner` (also owns variant/payload gates).
   - Repeated copied hierarchy or high mesh count with no instancing:
     `usd-hierarchy-dedupe-candidates`.
   - Restructure decision (monolithic stage, asset boundary materialization): `restructure-decision`.
   - CAD converter settings: read `references/cad-conversion/README.md` (niche pre-USD concern; see reference for details).
   - Usd Optimize: `usd-optimize-run-validators`, `usd-optimize-interpret-validators`, `usd-optimize-run-operations`.

## Optimization ordering

Follow the canonical ordering in
[workflow.md § Operation ordering invariants](references/workflow.md#operation-ordering-invariants).
The high-level rule: **prototypes first → per-asset validation → stage-level
operations last.** The workflow reference owns the full invariant list
(meshCleanup before decimateMeshes, deduplication before decimation, never
merge if instanced, etc.) and the analysis-only ops catalogue.

### Large monolithic repeated-CAD pass

For large monolithic CAD-style stages with many repeated meshes and low or no
instancing, when the user asks for the safest useful optimization before
decimation, follow the execution contract in
[references/large-monolithic-cad-pass.md](references/large-monolithic-cad-pass.md):
lossless hierarchy/geometry dedup or prototype/reference restructuring is the
primary win (a repack is only secondary packaging); no decimation or other lossy
op without explicit approval; write a separate optimized stage; record
baseline/after metrics; run targeted (not full-sweep) validation; report all
three repack-normalized footprint sizes and attribute the re-encode vs.
structural split; and state which runtime metrics were not measured.

## Rules

- Always run composition audit before mutation.
- Always validate before and after processor execution.
- Optimize prototypes before per-asset validation.
- Do not run whole-stage mesh deduplication on very large CAD scenes before
  checking for hierarchy-level reuse.
- Do not recommend a fixed optimization stack without bottleneck evidence.
- Do not invent numeric thresholds or expected percentage wins.
- For decimation requests, decimate only eligible high-poly meshes, skip
  already-simple meshes, preserve materials, UVs, and normals where possible,
  record zero-work/no-op cases, and compare before/after mesh and file metrics.
- Treat occlusion checks, cross-component duplicate sweeps, exhaustive
  equivalence checks, and other broad expensive validation as opt-in work. Route
  validation through `usd-validation-runner`, present the default targeted
  validation that can run now, and ask for explicit approval before expensive
  full-sweep or cross-component checks.
- For standalone Usd Optimize or fixture-only runs, do not claim runtime
  performance improved from file size, prim count, load proxy, or operation
  report evidence alone. Runtime improvement is unconfirmed unless Kit,
  Omniperf, or equivalent profiling captured FPS, frame time, VRAM, Hydra, RTX,
  renderer, or draw-call metrics.
- **Prefer canonical Usd Optimize ops over specialty/documentary ones.** The
  `curation` block in `references/operations/operations.json` classifies every op
  as `canonical`, `specialty`, `analysis`, `documentary`, or `deprecated`.
  Recommend the canonical op first: `meshCleanup` (with explicit flags) over the
  legacy standalone `mergeVertices`; `deduplicateGeometry` over analysis-only
  `findCoincidingGeometry` (which only produces a report); and
  `usd-hierarchy-dedupe-candidates` + `apply-restructure` for hierarchy dedupe.
  Never recommend `documentary`-status ops (`boxClip`, `deletePrims`,
  `removeAttributes`, `removeUntypedPrims`, or `merge` outside its narrow
  non-instanced case) without an explicit user request. Specialty ≠ documentary:
  recommend a `specialty` op when its validator fires or its downstream context
  applies — e.g. `sparseMeshes`/`optimizePrimvars` (validator-wired) or
  `primitivesToMeshes`/`utilityFunction`/`pythonScript` (load-bearing escape
  hatches). See `operations.json` and upstream `usd-optimize` for op mechanics
  and local approval policy.

## Limitations

- Does not replace downstream reference instructions; load each required
  reference before executing it.
- Does not install runtimes directly; follow setup or install references when
  requirements are missing.
- Does not authorize mutation when the user has not allowed writes.

## Troubleshooting

- If runtime status is unclear, run `setup-usd-performance-tuning` before profiling or validation.
- If the reported problem is vague, gather stage path, workload type, and whether diagnosis or execution is requested.
- If the workflow suggests mutation before evidence, return to baseline profiling and composition audit first.

## References

Before routing, read:

- `references/usd-structure-assessment/references/optimization-tradeoffs.md` — identify which pipeline phase the scene is in (extraction, structuring, or optimization). The right action depends on the phase.
- `references/usd-structure-assessment/references/factory-level-structuring.md` — understand the three pillars (assets, aggregation, animation) and the seven-step structuring pattern.

If you have network access, prefer the live URLs (noted in each reference file) for the most current version.

## Required execution flow

Read `references/workflow.md` for the canonical Phase 0-7 flow, including
Kit/standalone branches, validator-stack routing, operation ordering,
termination conditions, duration hints, and the default three-pass scoped
iteration pattern.
The compact root map at `references/skill-map.md` only routes agents
into this workflow.

Do not treat downstream phase names as plain checklist labels. Before executing
each step, load that phase's nested `README.md` reference and follow its
instructions. Claude Code only exposes the public catalog skill; it does not
recursively inject `profile-stage`, `usd-structure-assessment`, or other nested
references.

The final deliverable must come from `optimization-report`: save both the structured JSON report and the generated Markdown summary. Do not substitute an ad hoc `SUMMARY.md` or chat-only recap for the optimization report.

For deeper subtopic guidance, `references/workflow.md` and
`references/skill-map.md` route into the nested material:
`usd-structure-assessment/` (composition-audit, layer-health,
instancing-tradeoffs, variants-payloads), `cad-conversion/`,
`upstreams/usd-optimize.md`, `usd-validation-runner/` (validator infrastructure
and the tier 1/2/3 probe plan with large-stage guardrails), and
`optimization-report/` (the data contract every phase populates).

For full Kit runtime profiling (FPS, frame time, Hydra/RTX metrics), refer to the external profiling skills at NVIDIA/omniperf.
