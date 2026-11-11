---
name: pva-build-test
description: >-
  Build and verify an NVIDIA PVA operator with the public PVA SDK. Use alone or
  from OpGen. Run optional EMU correctness, a fresh L4T build with
  the NVVPU or ASIP/Chess backend that Doctor proved, direct-SSH Tegra hardware
  correctness, and optional hardware performance. Return PASSED, FAILED, or
  BLOCKED with exact evidence.
metadata:
  author: "Venkat Balan <venkatramanb@nvidia.com>"
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# Build and test a PVA operator

Build the current source. Test it on a directly connected Tegra. Do not change
source code, expected results, or tolerances.

Build-Test owns verification stages, evidence, and verdicts.

## Resolve paths

`SKILL_ROOT` is the absolute directory containing this skill's `SKILL.md`,
after resolving symlinks. Bundled files are relative to that directory. Resolve
supplied relative paths against the invocation directory before changing
directories. `PROJECT_ROOT` is the active source
workspace; `RESULT_DIR` is the caller's result directory. All build directories,
reports, logs, and manifests below are under `RESULT_DIR`, not the shell's CWD.

## Get the inputs

Require these inputs for all scopes:

- Caller identity: `OpGen` for an OpGen call; otherwise the direct user.
- A Doctor report that is ready for `build-test`, or accepted limitations.
- Absolute project (`PROJECT_ROOT`) and result (`RESULT_DIR`) paths.
- Operator name and PVA generation.
- Test scope: `correctness-only` or `correctness-and-performance`.
- Report label. Use `build-test` when none is supplied.
- SDK version, safety variant, and Doctor's proved compiler backend.
- EMU policy; when EMU execution is available or required,
  Doctor's proved EMU build mode and configure options, plus test commands.
- L4T configure command and hardware system-test target and filter.
- Correctness cases, independent references, and tolerances.
- Complete install target or deployment manifest.
- Tegra address, user, port, and persistent known-hosts file.

For `correctness-and-performance`, also require:

- KPI or profiling target and case filter.
- Profiling method, metric, warm-up count, and at least 1000 samples.
- User-approved clock and clock-verification commands.
- Expected clock values and target power mode.
- Performance target and acceptance rule when the caller requests a comparison.

Do not require a KPI target, profiling artifact, clock input, sample count, or
performance filter for `correctness-only`.

Do not guess a command, target account, filter, expected result, clock setting,
or acceptance rule.

## Verify the inputs and references

Require Doctor's `public-reference-corpus` check to be `PASS`. Set
`SDK_REFERENCES` to the report's absolute `sdk_reference_root`. Require a readable
`index.md` and `reference-lock.json`, a lock SHA-256 matching the report's
`sdk_reference_lock_sha256`, and the reported SDK version matching this task.
If these fields are absent or no longer match, return `BLOCKED` and request a
fresh Doctor report. Do not fall back to an installed package or CWD path. Read the
`pva-build-test` route in `$SDK_REFERENCES/index.md`.
Before reading Markdown, check its SHA-256 against the lock
`outputs` entry; return `BLOCKED` on a mismatch. Resolve links relative to each
generated file. Use `rg` to
select relevant build, compiler, deployment, signing, EMU, and
profiling Markdown.

Read [the direct-SSH procedure](references/direct-ssh-tegra.md) before target
access. Fill [the JSON result template](assets/build-test-result-template.json)
using [the result rules](references/build-test-result-schema.md), then write the
Markdown report from the same values.

Validate:

- Absolute and readable project paths and a writable result path.
- At least one correctness case with an independent expected result.
- Unambiguous build, test, deployment, and target commands.
- At least one matching hardware system test.
- Performance inputs only when performance is in scope.
- PVA generation, SDK runtime, and safety-variant consistency.

Return `BLOCKED` for a missing or ambiguous required input.

For each product fact, record the SDK documentation root, local Markdown path,
packaged source path, source SHA-256, version, and section or PDF page.

## Protect evidence

Before the build:

1. Create new build directories under the result directory.
2. Write `source-manifest-before.sha256` with paths relative to the project
   root. Exclude new build and result directories.
3. Record Git identity and dirty state when Git metadata exists.
4. Record each command, exit code, log, and principal artifact hash.

Do not reuse a build, log, target result, or profile from an earlier source
state.

## Run the stages

Run these stages in order:

| Stage | Required action |
| --- | --- |
| Input | Validate inputs, source identity, references, and scope. |
| EMU build | If EMU is available or required, create a fresh build using Doctor's proved EMU mode and configure options. Otherwise record `SKIPPED_OPTIONAL`. |
| EMU test | Run the approved EMU cases. Stop with `FAILED` on a demonstrated source or result defect. |
| L4T build | Create a fresh release build with `PVA_BUILD_MODE=L4T`, the requested generation, `PVA_DISASSEMBLE=ON`, and Doctor's proved NVVPU or ASIP/Chess backend. |
| Listing | Record and inspect the generated compiler listing. Confirm vectorization in the main data path. |
| Target probe | Use the direct-SSH procedure. Confirm Arm64, BSP, runtime version, safety variant, libraries, and required permissions. |
| Deploy | Stage and transfer an explicit archive. Check dynamic dependencies on the target. |
| Hardware test | List and run the filtered system tests for the complete correctness matrix. |
| Performance | Run only for `correctness-and-performance`, after hardware correctness passes. |
| Freshness | Hash the final source and compare it with the initial manifest. |

### EMU rules

Use `PVA_BUILD_MODE=EMU` with Doctor's proved SDK, generation, compiler, and
other EMU configure options. Keep the L4T compiler selection separate.
Replace the probe's source and build paths with `PROJECT_ROOT` and a new
directory under `RESULT_DIR`;
use the project's supplied test targets. Record the selected mode and options
in `emulator` and the full command in `commands.log`. Return `BLOCKED`
if an available or required EMU backend lacks this evidence or its settings
conflict with the requested SDK or generation. Do not substitute another host
execution tool for EMU.

With policy `optional`, an unavailable EMU component is
`SKIPPED_OPTIONAL`. An EMU stage that starts and shows a source, build, or
result defect is `FAILED`.

With policy `required`, an unavailable EMU component is `BLOCKED`.

EMU execution is an early correctness aid. It is not hardware evidence.

Run the same approved correctness cases, inputs, references, and tolerances on
EMU and hardware when EMU runs. Do not reduce the EMU matrix because it is slow.
Record the case IDs and counts for both environments.

### L4T and compiler rules

Build the hardware system-test executable for both scopes. Build the KPI or
profiling executable only for `correctness-and-performance`.

Accept the NVVPU or ASIP/Chess backend that Doctor proved. Do not require an
ASIP path or Synopsys license when NVVPU is selected. Return `BLOCKED` if the
proved backend is no longer usable. Return `FAILED` when current source causes
the build to fail.

Require the compiler listing after a successful build. Record its path, hash,
generation, entry point, main vector operations, address-generation evidence,
spills, warnings, and scheduling facts that affect correctness. Return
`FAILED` if the main data path is not vectorized.

### Deployment rules

Use a project install target when it creates a complete deployment tree.
Otherwise, use the supplied manifest.

Always include the hardware system-test executable, project libraries, device
data, and test assets. Include the KPI or profiling executable and its assets
only for `correctness-and-performance`.

Do not copy host-architecture libraries. Retain the remote directory unless
the user asks for its removal.

### Hardware correctness rules

Run `--gtest_list_tests` with the approved filter. Reject a zero-match filter.
Then run the same filtered system-test executable.

For each case, record the input shape, format, pitch, parameters, independent
reference, tolerance, exit code, comparison, and maximum error when relevant.

Return `FAILED` if the process, command status, output, case coverage, or
tolerance check fails. Do not use a KPI-binary pass as correctness evidence.

### Performance rules

Skip this stage for `correctness-only`.

For `correctness-and-performance`:

1. Run the approved clock command and verify clocks and power mode.
2. Run warm-ups separately.
3. Collect at least 1000 hardware samples with the approved public profiling
   method.
4. Verify clocks and power mode again.
5. Record the metric, unit, workload, warm-ups, samples, minimum, maximum,
   mean, median, target conditions, and raw data path.
6. Compare with a target only when the caller supplied a target and acceptance
   rule.

Do not use EMU time, host wall time, or total test duration as PVA execution
time. Do not derive a target from the implementation under test.

A missed target is `FAILED` only when the acceptance rule makes it required.

## Decide the verdict

- `PASSED`: Every stage required by the selected scope passed for the current
  source. An optional EMU stage can be `SKIPPED_OPTIONAL`.
- `FAILED`: A stage ran and produced a negative source, build, correctness,
  vectorization, or required performance result.
- `BLOCKED`: A required stage could not run because an external input,
  dependency, permission, corpus, or target was unavailable.

Do not return `PASSED` from evidence for another source state. Do not change
`BLOCKED` to `PASSED` because EMU tests passed.

## Write the outputs

Always write:

- `reports/<report-label>.md`
- `reports/<report-label>.json`
- `source-manifest-before.sha256`
- `source-manifest.sha256`
- `source-identity.txt`
- `commands.log`

Write stage logs and artifact hashes only for stages that run. For
`correctness-only`, do not require clock, profile, KPI, or raw performance
files. If work stops early, mark later stages `NOT_RUN` and name the exact
failed or blocked stage.

Do not remove raw evidence or replace it with a summary.

## Return control

Return the verdict and absolute report paths to the caller. When called by
OpGen, continue the existing OpGen workflow before ending the overall run:
`FAILED` with remaining attempts goes to Debug; zero or exhausted attempts go
to OpGen's final `FAILED` report. `PASSED` and `BLOCKED` also require OpGen's
final report. Do not reinvoke OpGen or end with only a Build-Test response.
