# Setup USD Performance Tuning

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

## When to Use

Minimum supported runtime: usd-optimize 1.0.x (GitHub release packages) with
the usd-validation-nvidia pip package.

Use this reference when standalone runtime availability is unknown or the user explicitly asks to set up, verify, or install the standalone Usd Optimize / usd-validation-nvidia path (or to set up the opt-in Kit→omniperf profiling adjunct).

## Instructions

1. Check for an existing `setup-preflight.json` and verify whether it matches the current target and runtime intent.
2. Probe the standalone Usd Optimize, usd-validation-nvidia, and USD Python paths (plus any opt-in Kit profiling root) without silently choosing between viable alternatives.
3. Ask the user before installing the standalone runtime when no verified path satisfies the request.
4. Write or refresh the preflight artifact with runtime versions, paths, and available Usd Optimize operations.


## Pre-flight Checklist

Before executing setup/preflight, re-read and confirm:

- [ ] `references/runtime-context-header.md` — runtime context block format.
- [ ] `references/runtime-probe.md` — probe sequence and failure handling.
- [ ] Output workspace policy from parent `references/output-workspace.md`.
- [ ] Write `setup-preflight.json` conforming to `scripts/setup-preflight.schema.json`.
## Output Format

Return the selected runtime route, any user decision needed, and the path to `setup-preflight.json`. The preflight artifact records Kit, Usd Optimize, usd-validation-nvidia, USD Python, and `operationsAvailable` evidence.

Use this reference before running validation, profiling, or optimization from this
skill package in a fresh environment. The goal is to choose and verify one
runtime path before invoking the workflow skills.

## When this is the entry skill

This reference is the **named entry skill** in an agent's response only when no
runtime path is verified at all — that is, when the setup probe reports every
candidate (Kit, standalone Usd Optimize, standalone usd-validation-nvidia) as
unavailable, missing, or unverified. In that case there is no way to route
performance work, so resolving the runtime is the agent's first responsibility.

As soon as **any** runtime path is verified — even partial availability such
as `kit_runtime: available, asset_validator: available, scene_optimizer:
unavailable` — the named entry skill is `omniverse-usd-performance-tuning`, not this one.
Triage then routes to the correct outcome, including blocking on a specific
missing component when needed. This reference still runs in its normal Phase 0
position; it just isn't the entry skill the agent names.

For `omniverse://` assets, `omniverse-authentication` is the named entry skill
ahead of both setup and triage. Authentication preflight precedes runtime
probing for remote assets.

This rule is about **which skill the agent names as the entry**, not about
execution order. Setup, authentication, and triage continue to run in their
normal phase order regardless.

## Purpose

Verify the **standalone** runtime — the sole runtime for Usd Optimize
execution and usd-validation-nvidia validation — before downstream references run.
Kit is not an alternate optimization/validation runtime; it is retained only as
an explicit opt-in render-profiling adjunct (Kit→omniperf), set up on request.

Phase 0 **guarantees an importable `pxr` (USD Python)** as a precondition for the
pipeline: every standalone route lands `pxr` (Kit and SO standalone provide it
directly; a validator-only standalone venv installs `usd-core`). After Phase 0
completes on any route, `import pxr` succeeds — downstream phases that author USD
(e.g. `apply-restructure`) can rely on it.

## Prerequisites

- Current shell access to probe local installs.
- Any user-provided Kit, USD Composer, Isaac Sim, or standalone library path.
- Permission to run lightweight Python import probes from candidate runtimes.

## Examples

- "Set up this repo before I run validation."
- "Check whether my Kit path can run Usd Optimize and usd-validation-nvidia."

## Runtime choices

**Prefer standalone SO + AV when available.** The standalone path is lighter
(no Kit overhead), deterministic, and sufficient for all optimization and
validation workflows. The SO package includes
`omni.scene.optimizer.validators` with `@register_rule` decorators that
auto-register 25 SO performance validators into OAV when both packages share
the same Python 3.12 environment. No manual `register_all()` call is needed
for rule discovery — just ensure both are importable. Selected runs go through
`usd-validation-runner/scripts/usd_validation_executor.py`, which uses
`ValidationEngine(init_rules=False)` plus `enable_rule()` after resolving each
scope-note **canonical concept** to a rule class by identity.

> Standalone achieves the same validator coverage as Kit: install
> `omniverse-asset-validator` via pip into the same venv where the Usd Optimize package
> is on PYTHONPATH, and the `@register_rule` decorators register Usd Optimize validators
> at import time.

Standalone is the only runtime for optimization and validation; there is no Kit
optimization fallback. Kit (USD Composer, Isaac Sim, or Kit SDK) is retained
**only as an opt-in render-time profiling adjunct** — the one way to capture the
FPS / Hydra / RTX / VRAM / draw-call metrics required before claiming
runtime wins. That profiling path is delegated to the external `NVIDIA/omniperf`
skills; set it up only when the user explicitly asks for render-time profiling.
Kit is never auto-selected as a silent fallback for SO/AV work.

## Requirement-to-skill map

- Standalone Usd Optimize operations: invoke `install-usd-optimize-standalone` when
  the extracted prebuilt Usd Optimize package is missing
  (asset name + download: `references/upstreams/usd-optimize.md`).
- Standalone Omni usd-validation-nvidia: invoke `install-usd-validation-nvidia-standalone`
  when missing. Usd Optimize validators auto-register when both packages share the same
  Python environment.
- Opt-in render profiling only: an existing/ user-supplied Kit or USD Composer
  runtime is verified for the Kit→omniperf profiling adjunct (`install-kit` /
  `install-usd-optimize-standalone` are profiling-setup helpers, not a default optimization
  fallback). Do not install Kit just to run SO/AV work.

## Output workspace contract

Everything this reference writes goes under the user's `output_path` (see
`references/runtime-context-header.md` *Where artifacts live*):

- `<output_path>/setup-preflight.json` — canonical name + location for
  the runtime config consumed by every downstream skill. **Do not write
  it under any other filename or location** (no `probe_result.json`, no
  `_work/`, no temp dirs). Downstream skills check this exact path; a
  different name leaves the session-start gate broken.
  Include `cuda_available` when probed; the Phase-4 batch scheduler treats `false`
  as a hard signal not to run `gpu_bound` operations through slow CPU fallback,
  and treats `null`/missing as unknown (fail closed to CPU-only).
- `<output_path>/scripts/probe_setup.py` — the generated Python probe
  driven through Step 3.
- `<output_path>/scripts/probe_setup.log` and
  `<output_path>/scripts/probe_setup.stderr.log` — probe stdout / stderr.

Follow `skills/omniverse-usd-performance-tuning/references/runtime-artifact-token-budget.md`
for all probe logs. Parse the JSON object from stdout, keep the full stdout /
stderr files on disk, and surface only bounded tails or targeted error matches
when troubleshooting Kit launch noise.

If `output_path` is not yet known when this reference is invoked, prompt the
user for it before proceeding. Do not pick a default and do not write
to the working directory.

## Step 1 - Determine standalone runtime

The agent performs setup checks directly from the current shell. Do not rely on
repo-local setup scripts or ask the user to run scripts.

Check for standalone Usd Optimize and usd-validation-nvidia packages first —
they are the preferred runtime (lighter, no Kit overhead, deterministic).
Follow `references/standalone-runtime.md` for discovery and verification.

If standalone packages are found and importable, set
`runtime_route: "standalone"` in `<output_path>/setup-preflight.json` and
continue to Step 1.6.

If standalone packages are not found, this is a runtime block: surface the
missing-SO/AV install path (`install-usd-optimize-standalone` /
`install-usd-validation-nvidia-standalone`). Do not fall back to Kit for optimization
or validation — Kit is not an SO/AV runtime here.

## Step 1.5 - Opt-in Kit profiling discovery

Only when the user explicitly asks for render-time profiling, look for a Kit
installation to drive the Kit→omniperf profiling adjunct. Follow
`references/kit-discovery.md` for discovery order, path classification,
auto-enumeration, and candidate records.

Always ask before broad filesystem scanning. Prefer a user-supplied Kit /
USD Composer / Isaac Sim path; if exactly one candidate is discovered, write it
to `runtime_context.kit`. If multiple candidates exist, ask the user to choose;
never silently pick one. This path is for profiling only — it never becomes the
optimization/validation runtime.

Record the chosen candidate and `runtime_context.kit.chosen_by` as described in
`references/kit-discovery.md`.

## Step 1.6 - Probe the runtime for SO and AV versions

Once the standalone runtime is confirmed (or an opt-in Kit profiling root is
set), run the Python probe and write the probe result to
`<output_path>/setup-preflight.json`. Follow `references/runtime-probe.md` for
the launcher, import-mode, version-source, and `operationsAvailable` contract.

The `runtime_context` object is the literal input to the header template in
`references/runtime-context-header.md`. Downstream skills read from this object,
not from the raw probe `kit` / `usdOptimize` / `assetValidator` source
fields.

Downstream skills (`usd-optimize-run-operations`, `omniverse-usd-performance-tuning`, every
`usd-optimize-interpret-validators` recommendation) cross-check `operationsAvailable`
against the op key they intend to invoke and refuse to call any op the
runtime does not register.

## Step 2 - Interpret status

- `ready-standalone`: use standalone Usd Optimize for operations and Omni usd-validation-nvidia from Python. This is the optimization+validation runtime.
- `ready-profiling-kit`: an opt-in Kit→omniperf profiling root is available **in addition to** standalone; it is used only for render-time profiling, never for SO/AV work.
- `needs-runtime-choice`: standalone SO/AV is not importable — stop and ask the user to supply the standalone package path or a pip-installable environment.

When status is `needs-runtime-choice`, ask for the standalone SO / AV package
path or a pip-installable environment. (A Kit path does not resolve this; Kit is
not an SO/AV runtime.)

Do not continue to `usd-optimize-run-validators`, `usd-optimize-run-operations`, or deep validation
until standalone SO/AV is resolved.

## Non-interactive (batch / CI) mode

The "stop and ask" behaviors above — the `output_path` prompt, the multiple-Kit
chooser, and the `needs-runtime-choice` gate — assume an interactive session.
For unattended batch or CI runs the caller can pre-supply those inputs, and the
agent must then proceed without blocking:

- If `output_path`, a runtime preference, and any required candidate paths are
  all supplied, do not prompt.
- When the preference is `auto`, resolve the optimization runtime
  deterministically: standalone Usd Optimize + usd-validation-nvidia, if importable.
  There is no Kit optimization fallback in the policy. A user-supplied Kit path
  is recorded only as the opt-in profiling adjunct, never as the SO/AV runtime.
- Record `runtime_context.chosen_by: standalone_preferred` in
  `setup-preflight.json` so downstream skills and the report can show the runtime
  was selected unattended rather than confirmed by a human.
- If standalone does not resolve under this policy, stop with
  `needs-runtime-choice` and name the missing standalone inputs — do not guess a
  runtime, fall back to Kit, or scan without permission.

## Step 3 - Verify standalone path

If standalone is chosen (Step 1 succeeded), verify each standalone requirement
with its dedicated install reference. Follow `references/standalone-runtime.md` for
the user-facing prompt, Python 3.12 requirement, expected standalone layout,
and handoff rules.

## Step 4 - Verify Kit path (fallback)

For a Kit root (Step 1.5), verify Usd Optimize and Omni usd-validation-nvidia core
both load, and capture the runtime versions that Step 1.6 surfaces to the user.
Use `references/runtime-probe.md` for the exact launcher, import, version, and
log discipline.

Do not pre-check extension folders, `exts/`, `extscache/`, or any other
filesystem layout before running the probe. If the probe fails, ask for a
different Kit path.

## Step 5 - Continue workflow

After setup:

1. `omniverse-usd-performance-tuning` for broad performance requests.
2. `usd-structure-assessment` before choosing optimizations.
3. `usd-validation-runner` for validation; its references own the specific `validate-*` command details.
4. `usd-optimize-run-validators`, `usd-optimize-interpret-validators`, and `usd-optimize-run-operations` only after runtime setup is ready.

Record the chosen runtime path in the response so later commands use the same
Kit or standalone environment.

## Step 6 - Print the runtime context header before continuing

Every downstream user-facing prompt must lead with the runtime context block
defined in `references/runtime-context-header.md`. This reference writes the
canonical `runtime_context` object into
`<output_path>/setup-preflight.json` (see *Output workspace contract*);
downstream references consume it from that exact path.

The header has two formats:

- **Format A (full block)** — required at this reference's runtime-choice prompt,
  at the `restructure-decision` Phase 2e prompt, at the `usd-optimize-run-operations`
  destructive-op confirmation, and at the first user-facing message of any
  session that starts mid-workflow.
- **Format B (compact one-liner)** — used for routine status messages and
  follow-up prompts once the user has already seen Format A in the session.

When `runtime_context.kit` is set (single candidate or user has picked), print
Format A once as the conclusion of this reference's interaction with the user, before the
agent hands off to `omniverse-usd-performance-tuning`. The user must see exactly which Kit
application, Usd Optimize, and usd-validation-nvidia version will be in effect
for the rest of the session.

## Limitations

- Does not install unless a dedicated install reference is invoked.
- Does not choose optimization operations or validator scope.
- Standalone Usd Optimize validators auto-register via `@register_rule` decorators when
  both `omniverse-asset-validator` and the Usd Optimize package are importable in the
  same Python 3.12 environment. Kit auto-registers them via its extension
  session.

## Troubleshooting

- If standalone packages are found but the probe fails (import error, version mismatch), fall through to Kit discovery.
- If multiple valid Kit installs are found, ask the user to choose or record the newest unattended choice.
- If the Kit probe cannot import Usd Optimize or usd-validation-nvidia, try another Kit path.
- If standalone paths are incomplete, invoke the relevant install reference instead of reusing a bundled validator environment.
