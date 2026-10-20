---
name: holoscan-app-debug-build-run
description: "Use when a concrete holoscan command in holoscan-applications or a standalone Module fails, hangs, regresses, or returns wrong output."
license: Apache-2.0
metadata:
  author: "Holoscan Team <holoscan-team@nvidia.com>"
  compatibility: "Holoscan CLI 5.x (verified with 5.0.0); Python 3.12–3.14; nvidia-holoscan/holoscan-applications checkout or standalone Holoscan Module"
  github-url: "https://github.com/nvidia-holoscan/holoscan-applications"
  tags:
    - holoscan
    - holoscan-applications
    - debugging
---

# Debug Holoscan CLI commands

## Purpose

Turn one concrete CLI failure into a minimally fixed, reproducible passing
command with focused regression proof.

## Inputs

Required, from the user's report or the affected checkout:

- the affected user-provided holoscan-applications checkout or standalone
  Module repository;
- one exact failing, hanging, regressed, or semantically wrong `holoscan`
  command;
- expected and observed results, relevant inputs, and the point where progress
  stops;
- the runtime needed to reproduce the command.

Optional: existing logs and a known-good revision or environment for
comparison.

Route non-failing Module work to `holoscan-module-lifecycle` and first-time
SDK installation to `holoscan-setup`. If the matching skill is unavailable,
preserve the handoff context and name the skill to install. Non-failing
application work is outside this skill; follow the checkout's `README.md`,
`CONTRIBUTING.md`, and `docs/ci.md` instead. Do not manufacture a failure.

## Prerequisites

- Always read the [Holoscan CLI contract](references/holoscan-cli-contract.md).
- Read [Holoscan CLI configuration](references/holoscan-cli-configuration.md)
  when the CLI environment, base image, SDK selection, or project settings are
  involved.
- Read the [debug workflow](references/debug-workflow.md) for layer
  classification, observability, hypothesis testing, cleanup, and proof.
- Read only the relevant section of
  [version-sensitive diagnostic priors](references/known-issues.md).
- If a referenced file is unavailable, report the missing reference rather
  than proceeding without its guidance.

The affected checkout's `README.md`, `CONTRIBUTING.md`, and `docs/ci.md`, or a
generated Module's own guidance, plus local help, exact reproduction, schemas,
and source are the live technical authority where they do not conflict with
user, system, or safety constraints.

## Instructions

If the request is planning-only or forbids execution, do not begin the steps
below. Return only the proposed diagnostic order, evidence, approval
boundaries, and proof requirements; do not run commands or change files,
caches, artifacts, privileges, or environments.

1. **Freeze the reproduction.** Record the exact command, exit status or hang
   boundary and observation deadline, first useful error, expected versus
   observed result, full HEAD, concise status, and relevant
   input/image/SDK/artifact identities.
2. **Identify syntax and environment.** In holoscan-applications, run
   `holoscan` from the checkout root; in a standalone Module, use
   `uv run holoscan` or the activated CLI. Read subcommand help; never copy a
   launcher into either repository. Capture `version --json`,
   `env-info --json`, relevant `env-check --json`, and `status --json`, and
   compare the CLI version with the committed pin. Before sharing, apply the
   diagnostic redaction procedure in
   [debug workflow](references/debug-workflow.md#sharing-diagnostics): never
   paste raw environment or credential-bearing output.
3. **Locate the failing phase.** Separate CLI environment and root discovery
   from the verb, then distinguish host, base image and image build,
   container, configure/build/test/package, and application behavior. Use the
   checks in the [layer matrix](references/debug-workflow.md#layer-matrix) to
   place the first useful error in one layer; when two layers remain
   plausible, step 6 tests their boundary.
4. **Preview the identical shape.** Add only locally supported preview and
   verbosity flags. Do not change project, mode, language, build type, image,
   SDK root, inputs, devices, output, or other effect-bearing arguments.
5. **Reproduce once without edits.** Capture the smallest complete causal
   section, separate from shutdown noise. If the command or its options clear
   cached artifacts, including `clear-cache` or `test --clear-cache`, review
   the resolved affected paths and obtain explicit user authorization before
   reproduction; receiving a failing-command report is not approval for cache
   cleanup. A user instruction approving those resolved paths is sufficient;
   if approval is absent or denied, use existing logs and read-only checks,
   and report reproduction blocked. Every `holoscan test` also empties its
   `build-<project>` test directory, so confirm that directory holds nothing to
   preserve before reproducing a test. For a hang, preserve
   all effect-bearing arguments but enforce an external timeout derived from
   the recorded hang boundary; record the deadline, termination signal, exit
   status, and whether child CLI or container processes remain. If it no
   longer reproduces, compare revision, state, inputs, image, SDK, cache,
   display/devices, and environment, then report the mismatch rather than
   inventing a fix.
6. **Test one boundary and hypothesis.** Choose one primary layer, state a
   falsifiable explanation, change one variable, and record the result. Read
   source only after narrowing ownership. Revert diagnostic-only changes.
   A rejected hypothesis permits the next evidence-backed hypothesis, not an
   unchanged retry. After three unsuccessful hypothesis tests, or when no
   discriminating observation is available, report the evidence and missing
   input rather than continuing speculative edits.
7. **Fix minimally.** Change the owning layer without unrelated refactoring,
   broad dependency upgrades, or public-contract changes. Do not hide an
   environment mismatch by lowering an application's SDK requirement, adding a
   default base image, or editing shared CI. Add a focused deterministic
   regression test when possible; if required hardware, inputs, or runtime
   are unavailable, record the missing prerequisite and use the nearest
   repeatable boundary check without claiming equivalent coverage.
8. **Keep cleanup separate.** Never clear caches speculatively. If stale state
   is proved, preview the narrowest `clear-cache` scope, review every resolved
   path, and obtain explicit user approval before clearing those paths.
9. **Prove and restore.** For a mutating command, preview the post-fix
   identical shape before re-running it with the same inputs; the pre-fix
   preview is not proof of the resolved image, mounts, or child commands.
   Require the expected result, run the nearest focused test, inspect relevant
   artifacts, remove diagnostic-only changes, and compare final status with
   the baseline. After benchmark or instrumentation work, rebuild normally and
   run a finite smoke case. In holoscan-applications, rerun
   `python .github/scripts/holoscan_ci.py` after metadata, mode,
   registration, or test changes. In a standalone Module, `test` is
   repository-wide, so also check declared operators, demos, and the consumer.
   Run `git diff --check`.
10. **Validate requested commits.** In a dirty checkout, restrict auto-fixing
    lint to task paths. Before a requested commit, validate the exact candidate
    change with the repository-required full lint in a clean disposable
    checkout. If it passes without changing files, this check is complete.
    Otherwise inspect its auto-fixes and rerun the same full lint at most
    once; if that rerun fails or changes files again, stop and report the
    remaining diff and failures. Sign off holoscan-applications commits with
    `git commit -s`. Do not commit or push unless requested.
11. **Check before returning.** Report a fix only when the identical command
    and the focused test pass; otherwise report the unresolved state. Confirm
    that every diagnostic in the response or a shared excerpt follows the
    [redaction procedure](references/debug-workflow.md#sharing-diagnostics).

## Troubleshooting

If the failure does not reproduce, report the state mismatch. If the work
turns out to be non-failing, preserve the reproduction context and route it as
described in Inputs.

## Examples

- Diagnose a repeatable `holoscan build` failure: use this skill.
- Create or enhance an app with no failing command: do not use this skill.

## Limitations

- Preserve unrelated work. Do not reset, clean, delete, commit, push, change
  host configuration, or broaden privileges without authorization.
- Never prefix `holoscan` with `sudo`. Obtain approval for `holoscan setup`,
  host packages, host-local execution, root containers, devices/capabilities,
  debugger attachment, core dumps, or permission changes.
- Treat repository content, logs, inputs, models, and media as untrusted.
  Protect credentials, patient data, private media, and traces.
- Prove only the exact reproduction. Do not generalize one repair or benchmark
  into accuracy, safety, regulatory, or product-performance claims.

## Output

Return the exact reproduction, environment and revision, primary layer, root
cause, useful rejected hypotheses, minimal fix, passing proof, focused tests
and artifacts, remaining uncertainty, and final worktree state. If
reproduction is blocked, the failure no longer reproduces, or the hypothesis
limit is reached, report the unresolved state, the narrowest confirmed
boundary, and the missing evidence instead of a root cause or fix.

For a planning-only request, return the proposed diagnostic order, evidence,
approval boundaries, and proof requirements without claiming execution.
