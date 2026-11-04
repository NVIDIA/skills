---
name: pva-opgen
description: >-
  Generate, change, or repair a complete vectorized NVIDIA PVA operator from
  a validated pva-spec handoff. Deliver the contract's API, host and VPU code,
  independent CPU reference, sample, build integration, tests, and
  documentation. Verify with pva-build-test and use pva-debug within the
  configured attempt limit.
metadata:
  author: "Venkat Balan <venkatramanb@nvidia.com>"
  tags: [pva, code-generation, build-test]
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# Generate a PVA operator

Implement the approved Spec contract. Do not redefine it.

## Resolve paths

`SKILL_ROOT` is the absolute directory containing this skill's `SKILL.md`,
after resolving symlinks. Bundled files are relative to that directory. Resolve
supplied relative paths against the invocation directory before changing
directories. Set `OPGEN_INPUTS` to the absolute
handoff path and `WORK_DIR` to the caller's report work directory. For a handoff
at `<work-directory>/configure/opgen-inputs.json`, default `WORK_DIR` to that
`<work-directory>`; otherwise require an explicit report work directory.
Use only the validated handoff's workspace as the project root.

## Inputs

Before changing source, require installed `pva-build-test` and, when
`PVA_OPERATOR_MAX_DEBUG_ATTEMPTS` is greater than zero, `pva-debug`.
Resolve them by name through the agent's skill discovery; their directories
need not be adjacent. Return `BLOCKED` if a required skill is unavailable.

Run these skills using the agent's available skill mechanism. If no callable
skill tool exists, load the named skill's `SKILL.md` and execute its complete
workflow as the Build-Test or Debug phase. Save that phase's reports, then
resume OpGen with its verdict. Reading instructions alone does not execute a
phase.

Require:

- Spec's `opgen-inputs.json`.
- The recorded operator specification, test matrix, and evidence manifest.
- A Doctor report that is ready for `opgen`, or accepted limitations.
- Direct-Tegra inputs that Build-Test needs.
- A writable report directory.

| Script | Command | Inputs | Outputs | Exit codes |
| --- | --- | --- | --- | --- |
| [finalize_opgen_inputs.py](scripts/finalize_opgen_inputs.py) | `python3 "$SKILL_ROOT/scripts/finalize_opgen_inputs.py" --check "$OPGEN_INPUTS"` | Spec handoff and its recorded inputs. | JSON validation summary; read-only check. | `0`: valid; `1`: failed; `2`: invalid command arguments. |

Run this command before you inspect or change the workspace:

```bash
python3 "$SKILL_ROOT/scripts/finalize_opgen_inputs.py" \
  --check "$OPGEN_INPUTS"
```

Return `BLOCKED` if the command reports an invalid structure, unknown
requirement link, unsafe path, missing evidence, changed input, stale hash or
size, or mismatched count. Do not repair Spec-owned files or handoff metadata.

Use the project workspace recorded in the validated handoff.

Verify the recorded hashes before reading project evidence files. Use API,
CPU-reference, test, workload, KPI, benchmark, and build files only as
contracts. Do not copy CPU-reference or test code into the implementation.

## Read the technical basis

Use the Doctor report recorded in the validated handoff. Require its
`public-reference-corpus` check to be `PASS`. Set `SDK_REFERENCES` to that report's
absolute `sdk_reference_root`. Require a readable `index.md` and
`reference-lock.json`, a lock SHA-256 matching the report's
`sdk_reference_lock_sha256`, and the reported SDK version matching this task.
If these fields are absent or no longer match, return `BLOCKED` for a fresh
Doctor report and regenerated Spec handoff. Do not edit a finalized report or
fall back to an installed package or CWD path. Read the
`pva-opgen` route in `$SDK_REFERENCES/index.md`.
Before reading Markdown, check its SHA-256 against the lock
`outputs` entry; return `BLOCKED` on a mismatch. Resolve links relative to each
generated file.

Use `rg` to select only the programming, data-movement, build, tutorial, and
API Markdown that applies. Read the relevant Orin GEN2 or Thor GEN3 guide
chapters for the core, VMEM, address generation, vector operations, and
programming examples.

Before use of an intrinsic or mnemonic, find its exact type and operation in
the generated guide pages. Record the SDK documentation root, local Markdown
path, packaged source path, source SHA-256, guide revision, generation, and
section or PDF page.

Use supplied or public implementations within the user's authorized scope and
their license terms.

## Inspect the project

Record:

- Public API and ownership conventions.
- Host and device target patterns.
- Registration, sample, test, and documentation patterns.
- Installed SDK version and PVA generation.
- Files that OpGen will add or change.

Do not require Git metadata or a fixed repository layout.

## Design the implementation

Use the behavior and test contracts from Spec.

1. Select an algorithm that gives the specified result.
2. Separate parallel work from reductions and dependencies.
3. Select the PVA dataflow and transfer plan.
4. Select a tile and VMEM layout that fit the target generation.
5. Define vector-tail, alignment, border, and scratch handling.
6. Estimate compute and transfer work when the required facts exist.
7. Record unavailable facts as limitations. Do not invent a Speed-of-Light
   value.

Use RasterDataFlow for suitable raster tiles. Use another documented dataflow
only when the access pattern needs it.

## Create the deliverables

Create or update all items that the Spec contract requires:

| Deliverable | Requirement |
| --- | --- |
| Public API | Use project naming, ownership, validation, and error conventions. Preserve asynchronous behavior. |
| Host code | Create resources, load VPU code, configure dataflows, submit work, check status, synchronize as required, and release owned resources. |
| VPU code | Use documented device APIs and vector intrinsics. Keep the main data path vectorized. Use a correct predicated or scalar tail only where required. |
| CPU reference | Preserve an approved independent reference. For a new operator, create independent scalar logic that implements the contract. |
| Build wiring | Add host, device, sample, test, and optional performance targets. Add a complete install target or deployment manifest. |
| Sample | Show one valid lifecycle, check the result, and release resources. |
| Tests | Implement the approved Spec test matrix without weakening expected results or tolerances. |
| Documentation | Document the API, formats, shapes, parameters, arithmetic, borders, limits, build steps, and sample. |

Do not reuse a VMEM buffer before its transfer or compute use is complete. Do
not replace a documented vector operation with scalar work in the main data
path. Record each necessary scalar region and its reason.

Do not derive an expected result from PVA output.

## Verify with Build-Test

Run `pva-build-test` using the child-skill mechanism defined above.

Pass:

- Caller identity `OpGen`, `WORK_DIR`, and the Debug calls used and maximum.
- The validated Doctor report and Spec handoff.
- The active project root.
- The complete correctness matrix and EMU policy.
- `SKIPPED_OPTIONAL` evidence when optional EMU support is unavailable.
- L4T build, install, target, and direct-SSH inputs.
- `build-test-opgen` as the report label.
- An absolute result directory under `$WORK_DIR` for each Build-Test call.

Use `correctness-only` when the matrix has no performance case. Use
`correctness-and-performance` when it has one or more performance cases. Pass
performance targets, clock inputs, filters, and sample policy only in the
second scope.

Build-Test owns all build, EMU, hardware, profiling, evidence, and verdict
rules. Follow its instructions without skipping or weakening checks.

After every Build-Test call, resume OpGen and inspect its returned verdict and
report. A child skill's final response does not complete OpGen. Follow the
repair rules below, then write the final OpGen report before ending the run.

## Repair a failed implementation

Start with zero Debug calls used and read the maximum from
`PVA_OPERATOR_MAX_DEBUG_ATTEMPTS`. Check the budget before every Debug call;
`0` means no automatic repair. The next attempt number is calls used plus one.
Count each execution of the Debug workflow as one Debug call, regardless of
the invocation mechanism, including one that returns without a repair.

If Build-Test returns `FAILED` and an attempt remains, run `pva-debug` using
the same child-skill mechanism with the complete
failure report, failing source identity, command, logs, case, expected result,
tolerance, next attempt number, and maximum attempts.
Pass the same `WORK_DIR` for Debug reports.

After Debug returns, resume OpGen. For each `REPAIRED` result, invoke Build-Test
again on the changed source using a new result directory and the complete
original matrix. Choose the next action from the returned result:

| Result | OpGen action |
| --- | --- |
| Build-Test `PASSED` | Write the final `PASSED` report. |
| Build-Test `FAILED`, calls used below maximum | Invoke Debug, then resume OpGen. |
| Build-Test `FAILED`, zero or exhausted budget | Write the final `FAILED` report with the failed stage, reason, evidence, and calls used. Do not attempt a repair. |
| Build-Test or Debug `BLOCKED` | Write the final `BLOCKED` report with the exact unavailable prerequisite. |
| Debug `NOT_REPRODUCED` | Write the final `FAILED` report, retaining the original failure and Debug's reproduction result. |

Record one attempt entry per Debug invocation, with its result and source
identity. Do not reset or discount calls used when a failure remains unchanged.

Do not invoke Debug for an external `BLOCKED` result. Do not invoke Doctor or
Spec.

## Output Format

Write `$WORK_DIR/reports/opgen-report.md` with
[the result template](assets/opgen-result-template.md).
Write it for every terminal outcome, including zero or exhausted debug budget,
before returning the final response. Report zero calls explicitly when Debug
was not invoked. If the report directory is unavailable, identify the exact
path and write error in the final response.

Return one verdict:

- `PASSED`: All deliverables exist, and Build-Test passed the current source.
- `FAILED`: An executed implementation or verification stage produced a
  negative result that the bounded repair loop did not repair.
- `BLOCKED`: A required input, dependency, permission, corpus, or target is
  unavailable.

Report the changed files, API, supported matrix, vectorization, dataflow, VMEM
plan, build targets, deployment method, Build-Test result, Debug attempts,
evidence paths, hashes, and limitations.

Do not claim hardware verification from EMU execution. Stop after the
OpGen result.
