---
name: pva-debug
description: >-
  Diagnose and narrowly repair one reproduced NVIDIA PVA operator failure. Use
  from OpGen or manually for compile, link, cuPVA lifecycle, dataflow, DMA,
  numerical, EMU-versus-hardware, runtime, or performance failures. Preserve
  the original evidence and return the repaired source to the caller for a
  fresh Build-Test.
metadata:
  author: "Venkat Balan <venkatramanb@nvidia.com>"
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# Debug a PVA operator

Find one demonstrated cause. Make the smallest safe repair. Do not declare the
complete operator passed.

## Resolve paths

`SKILL_ROOT` is the absolute directory containing this skill's `SKILL.md`,
after resolving symlinks. Bundled files are relative to that directory. Resolve
supplied relative paths against the invocation directory before changing
directories. Reuse OpGen's `WORK_DIR` for its reports.
For a direct call, use the supplied absolute report path, or resolve the caller's
report work directory as `WORK_DIR`.

## Get the inputs

Require:

- A Doctor report that covers the failed capability.
- Absolute source and artifact paths, plus a report path or work directory.
- Caller identity: OpGen or a user with the same complete failure evidence.
- Failed stage, exact command, return code, full log, and source identity.
- Smallest known failing input, expected result, and tolerance.
- SDK version, PVA generation, compiler backend, build mode, and target
  identity when relevant.
- EMU and hardware results that already exist.
- One-based attempt number and maximum attempts for an OpGen call.

Return `BLOCKED` before a source change when required evidence is absent or the
attempt number is greater than the maximum.

## Load the technical basis

Require Doctor's `public-reference-corpus` check to be `PASS`. Set
`SDK_REFERENCES` to the report's absolute `sdk_reference_root`. Require a readable
`index.md` and `reference-lock.json`, a lock SHA-256 matching the report's
`sdk_reference_lock_sha256`, and the reported SDK version matching this task.
If these fields are absent or no longer match, return `BLOCKED` and request a
fresh Doctor report. Do not fall back to an installed package or CWD path. Read the
`pva-debug` route in `$SDK_REFERENCES/index.md`.
Before reading Markdown, check its SHA-256 against the lock
`outputs` entry; return `BLOCKED` on a mismatch. Resolve links relative to each
generated file. Use `rg`
to find the narrow Markdown that matches the first failed stage.

For each product fact, record the SDK documentation root, local Markdown path,
packaged source path, source SHA-256, version, and section or PDF page.

Use supplied or public implementations within the user's authorized scope and
their license terms.

## Preserve and reproduce the failure

1. Save the original source identity, uncommitted changes, command, return
   code, log, input, actual output, and expected output.
2. Reproduce the failure without a source change.
3. Reduce the input only when the failure signature stays the same.
4. Identify the first failed command, operation, or output.

Return `NOT_REPRODUCED` if the supplied source and command do not reproduce the
failure. Do not make a speculative repair.

## Diagnose one cause

Classify the first failure as:

- Compile.
- Link or device integration.
- cuPVA API or lifecycle.
- Dataflow, DMA, memory, or synchronization.
- Numerical or output correctness.
- EMU-pass but hardware-fail.
- Target runtime.
- Performance regression.

For each hypothesis:

1. State one falsifiable cause.
2. Name the supporting evidence.
3. Run the smallest diagnostic that can prove or reject it.
4. Save the command and result.
5. Reject an unsupported hypothesis before you test another.

Use the independent CPU reference and first wrong element for numerical work.
Use hardware evidence for a hardware-only failure. Use target compiler listings
and hardware profiles for a performance regression. Do not use EMU timing
as PVA performance evidence.

## Repair and check

Change only the code that the proved cause requires. Do not change the public
API, required result, tolerance, or performance test to hide the failure.

Record each changed file and reason. Keep the original evidence unchanged.
Write diagnostics and repair evidence to new files.

Then:

1. Rebuild the failed target.
2. Run the original failing case.
3. Run nearby affected cases.
4. Run the EMU matrix when source changed and EMU support is available or
   required. Missing optional EMU support is not a failure. Record
   `SKIPPED_OPTIONAL` when optional EMU support is absent.
5. Run the original hardware case for a hardware-only or performance failure.
6. Confirm that the original signature is absent.
7. Record commands, return codes, logs, outputs, and repaired source identity.

These checks prove only the narrow repair. Build-Test owns final verification.

## Return control

Write the report with
[the debug report template](assets/debug-report-template.md). For an OpGen
workflow, use `$WORK_DIR/reports/debug-attempt-<number>.md`. For a direct call,
use the supplied absolute report path or `$WORK_DIR/reports/debug-report.md`.

Return one result:

- `REPAIRED`: The original case and affected checks pass.
- `NOT_REPRODUCED`: The supplied source and command do not reproduce the
  failure.
- `BLOCKED`: Required evidence, source, toolchain, target, contract, or attempt
  budget is unavailable.

Return the report to the caller. Direct the caller to invoke
`$pva-build-test` with the complete original matrix and repaired source.
For an OpGen call, resume the existing OpGen workflow: `REPAIRED` requires
fresh Build-Test, and other results require OpGen's final report. Do not
reinvoke OpGen or end the overall run with only the Debug response.

Check public release notes before you identify an SDK limitation. Do not claim
an SDK defect without direct evidence.
