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

## Scope

Use for broad USD performance work: slow loading, low FPS/interactivity, high GPU or system memory, GPU crash/device lost, validation failures, CAD/conversion-quality triage, profiling, or requests to optimize a scene. This skill owns the user-facing workflow; setup, authentication, profiling, validation, mutation, and reporting are executed by phase references when reached.

Frontmatter keeps `version` and `tools` at top level for agentskills.io compatibility; NVCARPS fields live under `metadata`.

## Mandatory session-start gate

Before any tuning output, except a static classification-only answer, follow `skills/omniverse-usd-performance-tuning/references/setup-usd-performance-tuning/references/runtime-context-header.md`. That reference owns `output_path`, `setup-preflight.json`, Format A/Format B, and forbids silent ad hoc probing.

Required behavior:

- Missing or unreadable preflight: invoke `setup-usd-performance-tuning`.
- Present preflight: print Format A and wait for Continue, Change Kit, Switch to standalone, or Re-run probe.
- Runtime already confirmed in this session: use compact Format B:

```text
[Kit: {runtime_context.kit.application} {runtime_context.kit.version}  |  SO: {runtime_context.usdOptimize.version}  |  AV: {runtime_context.assetValidator.version}]
```

For standalone/usd-optimize packages, runtime evidence must include package/sentinel checks plus shared-library or import/load verification, not just the Python executable/version. For `omniverse://` assets, route through `omniverse-authentication` before setup, triage, or first open.

## Entry-skill and decision rules

- Name `omniverse-usd-performance-tuning` as the entry skill whenever any runtime path is verified: Kit, standalone, or a partial stack such as usd-validation-nvidia only. If the requested tool or operation is missing, return the specific blocker code such as `blocked_missing_usd_optimize` or `blocked_missing_usd_optimize_operation`; do not substitute a different workflow.
- Name `setup-usd-performance-tuning` as entry only when no runtime path is verified and runtime choice/setup is the first unresolved problem.
- This is ownership, not phase order: authentication, setup, and triage still run in their normal order.

Planning decision — derive `decision` from the response's shape, never from whether the request named a destructive op (the harness enforces the shape invariants):

- `ready_to_plan` — nothing in this response awaits the user; `committed_milestones` equals `planned_phases`. This is the default for generic optimization: the lossless canonical chain, plus a proactive `auto-within-tolerance` bounded-loss pass (an over-tessellated, visually-toleranced target reduced at its conservative per-target band, applied with a one-line notice rather than a prompt).
- `approval_required` — this response halts at a gate it is surfacing now; `committed_milestones` is a strict prefix of `planned_phases`, and `approval_required_reason` names the gate. The trigger is an unresolved decision the agent must surface before it can plan the op, not the fact that an op is destructive. Default primitive-fitting on a fitting-candidate scene (e.g. BIM/CAD pipe and duct runs with `fitPrimitives` at default args) is the standard, expected win and stays `ready_to_plan`, gated at execution. A bounded-loss op (`decimateMeshes`, `fitPrimitives`) becomes **inline-elicited** — its tolerance or data-preservation parameter must be answered now — only when it would run above the conservative band, on a functional-precision target (`articulated`/physics/sim-ready/metrology/variant-bearing), or under an explicit preservation intent (the user asks to keep UVs/displayColors/subsets). A `decimateMeshes` request with no stated tolerance is inline-elicited via its one upfront `mm_tolerance` question. A `restructure-decision` the response is presenting now is `approval_required` for the same reason. See `usd-optimize-run-operations/references/operation-safety.md` for the apply-authority classes.
- `blocked` — a `blocked_code` applies.
- Future gates that genuinely fire later — a downstream `restructure-decision` not yet reached, identity-gated ops collected for the Phase 7 opt-in menu — belong in `gates_observed`, never in `decision`.

## Canonical plan contract

For broad optimization, structured plans/status summaries must:

- Start milestone lists with `omniverse-usd-performance-tuning`; include `setup-usd-performance-tuning` only as Phase 0 context when relevant.
- Set top-level `decision` to `ready_to_plan` for generic optimization.
- Include the chain through `optimization-report` in both `committed_milestones` and `planned_phases`.
- Use exact profile labels `profile-stage:baseline` and `profile-stage:after`; never emit bare `profile-stage`.
- Preserve this subsequence exactly, inserting optional analysis only where it does not reorder it:

`omniverse-usd-performance-tuning` -> `profile-stage:baseline` -> `usd-structure-assessment` -> `usd-validation-runner` -> `restructure-decision` -> `apply-restructure` -> `usd-optimize-run-validators` -> `usd-optimize-interpret-validators` -> `usd-optimize-run-operations` -> `profile-stage:after` -> `compare-profiles` -> `optimization-report`

Two milestones are **conditionally required** — when the trigger holds they must appear as committed milestones at this position, not merely be routed to:

- `usd-hierarchy-dedupe-candidates` — after `usd-structure-assessment`, before `restructure-decision`, whenever the stage shows repeated copied hierarchy, high mesh count with little or no instancing, or a monolithic root. Do not conclude `hierarchy_dedupe.recommended: false` without it.
- `usd-edit-target-planner` — after `apply-restructure`, before the Usd Optimize validator/operations chain, whenever the stage is composed (references or payloads) and each target must be optimized as its own root layer.

Do not list `usd-optimize-run-validators` or `usd-optimize-interpret-validators` before `restructure-decision` in broad optimization milestone summaries. Phase-aware validator routing still happens inside `usd-validation-runner`.

Default broad optimization to three scoped iterations unless the user opts out, asks for a quick pass, or stop criteria apply. Each iteration writes an interim report/update; later passes reuse prior evidence instead of restarting the full workflow.

## Execution discipline

- Load `references/workflow.md` before end-to-end execution; it owns Phase 0-7 flow, Kit/standalone branches, validator routing, operation ordering, termination criteria, duration hints, and the default three-pass pattern.
- Do not treat nested phase names as checklist labels. Before executing a phase, load that phase's nested `README.md` or reference and follow it. Invoke downstream skill bodies only when their phase is reached.
- If local workspace instructions or helper commands are supplied, inspect them and use them to create, validate, or render required artifacts. If a file/tool is unavailable, report the observed blocker instead of fabricating completion.
- For binary or large assets, do not print raw contents. Use bounded metadata, checksums, sizes, validation/profile summaries, compact facts, or tool reports.
- Before reading Kit logs, usd-validation-nvidia CSVs, Usd Optimize logs, Tracy CSVs, or other runtime output, follow `references/runtime-artifact-token-budget.md`: keep raw artifacts on disk, read summary JSON first, and use bounded snapshots rather than full dumps or live streams.

Minimum context to gather: target stage, problem/goal, local/mounted/remote location, runtime, workload type when known, diagnosis-only vs mutation, and permission/output target for writes. Never overwrite the source unless explicitly allowed; prefer a separate optimized output for mutation. Do not invent thresholds, percentage wins, metrics, or runtime evidence.

## Routing map

- Composition, structure, layer health, instancing readiness: `usd-structure-assessment`.
- Validation/content issues: `usd-validation-runner`, which routes to validate-* or Usd Optimize validators as needed.
- Edit target, variant, payload, and output decisions: `usd-edit-target-planner`.
- Repeated copied hierarchy/high mesh count with no instancing: `usd-hierarchy-dedupe-candidates`.
- Monolithic stage or asset-boundary materialization: `restructure-decision` then `apply-restructure` when approved.
- CAD converter settings: `references/cad-conversion/README.md`.
- Usd Optimize execution: `usd-optimize-run-validators`, `usd-optimize-interpret-validators`, `usd-optimize-run-operations`.
- Full Kit runtime profiling such as FPS, frame time, Hydra/RTX metrics: external NVIDIA/omniperf profiling skills.

Before routing broad work, read the `usd-structure-assessment` tradeoff references for pipeline phase and factory-level structuring when those decisions matter.

## Mutation and operation rules

Follow `references/workflow.md#operation-ordering-invariants`. High-level invariant: prototypes first -> per-asset validation -> stage-level operations last.

Always:

- Run composition audit before mutation.
- Validate before and after processor execution.
- Optimize prototypes before per-asset validation.
- Check hierarchy-level reuse before whole-stage mesh dedupe on very large CAD scenes.
- Base recommendations on bottleneck evidence; do not recommend fixed stacks without findings.
- Do not authorize mutation when writes are not allowed.

Usd Optimize curation:

- Prefer `canonical` operations from `references/operations/operations.json` when multiple ops could address the same finding.
- Vertex welding: prefer canonical `meshCleanup` with explicit flags over standalone `mergeVertices`; follow upstream `usd-optimize` mechanics and local approval policy before mutating.
- Hierarchy dedupe: for the Phase 2 descent, prefer `usd-hierarchy-dedupe-candidates` plus `apply-restructure` (it owns the manifest/identity contract); a standalone approved-chain dedup run drives `deduplicateHierarchies` directly, invoked per frontier region (`paths` + per-region `maxDepth`).
- Per-mesh dedupe: prefer canonical `deduplicateGeometry`; `findCoincidingGeometry` is analysis/report only.
- Do not agent-initiate `documentary` operations such as `boxClip`, `deletePrims`, `removeAttributes`, `removeUntypedPrims`, or broad `merge` except in its narrow non-instanced case, unless explicitly requested.
- `specialty` operations are allowed when validator evidence wires them into `usd-optimize-interpret-validators` or downstream context requires them, such as `sparseMeshes`, `optimizePrimvars`, `primitivesToMeshes`, `utilityFunction`, or `pythonScript` recipes.

## Deliverables and final response

End-to-end optimization must produce an optimized USD stage when mutation runs and an `optimization-report` report. Diagnosis-only work must still end with a report or summary stating that no optimized stage was written.

Report requirements:

- Structured JSON must conform to `optimization-report`'s `scripts/optimization-report.schema.json`.
- Save the generated Markdown summary.
- Render HTML from `references/report-templates/optimization-report.html.template` via `render_preview.py`; never hand-write HTML.
- Do not substitute an ad hoc summary file or chat-only recap for report artifacts.

Final runtime response must explicitly name:

- selected entry skill and selected runtime/preflight state, including standalone package sentinel/load evidence when applicable;
- optimized USD output path when written, or that no mutation ran;
- source-not-overwritten/in-place mutation status;
- exact operation chain executed, especially safe/lossless chains when claimed;
- before/after validation and profile metrics available from evidence;
- validated report JSON, generated Markdown, rendered HTML, schema/validation verdict, score when present, and `workflow_mode`.

If preflight is missing, validation/rendering failed, report artifacts are absent, or no mutation ran, say so plainly and do not replace missing artifacts with a chat-only recap.

## Limitations and references

This skill does not install runtimes, replace downstream reference instructions, authenticate remote assets itself, approve unrequested destructive writes, or guarantee performance gains without evidence. If runtime status is unclear, return to the setup gate; if mutation appears before evidence, return to baseline profiling and composition audit first.

Primary references:

- `references/workflow.md`
- `references/runtime-artifact-token-budget.md`
- `references/skill-map.md`
- `skills/omniverse-usd-performance-tuning/references/setup-usd-performance-tuning/references/runtime-context-header.md`
- `skills/omniverse-usd-performance-tuning/references/usd-structure-assessment/references/optimization-tradeoffs.md`
- `skills/omniverse-usd-performance-tuning/references/usd-structure-assessment/references/factory-level-structuring.md`
- `skills/omniverse-usd-performance-tuning/references/usd-structure-assessment/references/composition-audit.md`
- `skills/omniverse-usd-performance-tuning/references/usd-validation-runner/README.md`
- `skills/omniverse-usd-performance-tuning/references/optimization-report/references/optimization-report-template.md`
- `references/upstreams/usd-optimize.md`

Use live URLs noted in reference files when network access is available and current upstream behavior matters.
