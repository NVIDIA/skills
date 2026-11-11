---
name: pva-doctor
description: >-
  Check whether a host is ready for requested NVIDIA PVA work. Use before Spec,
  OpGen, Build-Test, or Debug. Verify references to the installed SDK
  documentation, inspect only the requested capabilities, and prove that either
  NVVPU or ASIP/Chess can compile target VPU code when L4T compilation is
  required. Report READY, READY_WITH_LIMITATIONS, or NOT_READY. Do not install
  or configure software.
metadata:
  author: "Venkat Balan <venkatramanb@nvidia.com>"
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# Check a PVA development environment

Check only the capabilities that the user requests. Record other capabilities
as notes. Do not let them lower the verdict.

Do not install software. Do not change host or target settings. Do not print a
license value, credential, token, or private key.

## Get the inputs

Resolve these inputs:

- Requested capabilities: one or more of `spec`, `opgen`, `build-test`,
  or `debug`.
- PVA SDK root and requested version; record the installed version separately.
  An absent installation is a check failure, not a missing user input.
- PVA generation: `GEN2`, `GEN3`, or both.
- Project type: PVA SDK application or PVA Solutions source.
- Project path and version, when a project is in scope.
- EMU policy: `optional` by default, or `required` when the user
  requires it.
- Work directory (`WORK_DIR`). Default to `./pva-operator/`.
- Optional features, such as CUDA or NvSci interoperability.

For direct target work, also resolve the target address, user, port,
known-hosts file, and PVA generation. An expected BSP release is optional.
Do not guess a credential, user, proxy, or transport.

Use this scope table:

| Capability | Required checks |
| --- | --- |
| `spec` | References, SDK identity, project read access, and Python. |
| `opgen` | References, writable project, build tools, L4T compiler, target access, and `PVA_OPERATOR_MAX_DEBUG_ATTEMPTS`. |
| `build-test` | References, build tools, L4T compiler, transfer tools, and target access. |
| `debug` | References and only the tools, source access, and target access needed for the supplied failure. |

## Resolve paths

`SKILL_ROOT` is the absolute directory containing this skill's `SKILL.md`,
after resolving symlinks. Bundled files are relative to that directory. Resolve
user-supplied relative paths, including `WORK_DIR` and `PVA_SDK_ROOT`, against
the invocation directory before changing directories.

Set `SDK_REFERENCES` to `$WORK_DIR/references/sdk`. Keep generated references
outside installed skill directories and the SDK documentation tree, including
through symlinks. Reject an overlapping destination before creating directories.

## Verify the local references

First inspect the selected SDK root and packaged documentation. If the SDK is
absent, record required `sdk-installation` and `public-reference-corpus` checks
as `FAIL`, with the inspected path and requested version. If the SDK exists
but required documentation is missing, record `public-reference-corpus` as
`FAIL` and identify the missing files. Do not run materialization, read a
nonexistent generated index, or invent an installed version or lock hash.

Continue independent host, project, and target checks within the requested
scope. Skip probes that need the missing SDK or documentation and explain why
they were not performed. Write the report even when setup is incomplete.
Return `NOT_READY` and recommend that the user invoke `pva-setup` for the
missing installation components; do not invoke Setup or install anything.
An absent optional emulator remains neutral.

Doctor only checks SDK references. Do not create, replace, or repair the corpus,
even when reference writes are authorized. If references are missing, outdated,
or damaged, record the exact path and verification failure, return `NOT_READY`,
and recommend that the user run `pva-setup` for the same SDK and work directory,
then rerun Doctor. Setup runs the materializer; a cache failure alone does not
require reinstalling the SDK. Do not give manual `sync` commands as the next action.

Record this required check as `public-reference-corpus` in the Doctor report.
Set it to `PASS` only after `verify` succeeds for the selected SDK below;
record a failed verification or a missing required source as `FAIL`. Use
`LIMITATION` only when sources exist but verification was not performed.

Check `python3 --version` first; the materializer requires Python 3.9 or newer.
A missing prerequisite is `FAIL`. `verify` needs no PDF converter; mark the
`pdftotext` check `NOT_REQUESTED` in Doctor.

Require:

- `$SKILL_ROOT/scripts/materialize_sdk_references.py`
- `$SKILL_ROOT/references/sdk-doc-sources.json`
- `$PVA_SDK_ROOT/share/doc/VERSION`
- The packaged SDK HTML files and the Orin and Thor Programmer's Guide PDFs.

If `$SDK_REFERENCES/` or its lock is absent, record `public-reference-corpus`
as `FAIL` and leave the missing paths absent. Otherwise run this read-only check:

```bash
python3 "$SKILL_ROOT/scripts/materialize_sdk_references.py" verify \
  --manifest "$SKILL_ROOT/references/sdk-doc-sources.json" \
  --sdk-doc-root "$PVA_SDK_ROOT/share/doc" \
  --output-dir "$SDK_REFERENCES"
```

Record the absolute `sdk_reference_root` and the SHA-256 of
`$SDK_REFERENCES/reference-lock.json` as `sdk_reference_lock_sha256` in the
report, together with the selected SDK version and verification command/result.
Record a hash only after successful verification. Consumers use these report
fields; they do not infer a corpus location from their installation or CWD.

After successful verification, read the `pva-doctor` route in
`$SDK_REFERENCES/index.md`. Resolve links from each generated
Markdown file relative to that file. Use only the local Markdown that applies
to this check.

## Use these check results

- `PASS`: A direct check proves the capability.
- `LIMITATION`: A required operation is not proved, but no check failed.
- `FAIL`: A required item is absent, incompatible, or unusable.
- `NOT_AVAILABLE`: An optional capability is unavailable.
- `NOT_REQUESTED`: The check is outside the requested scope.

Record whether each check is required, plus its command, result, evidence, and
remediation. Required checks use `PASS`, `LIMITATION`, or `FAIL`.
`NOT_AVAILABLE` and `NOT_REQUESTED` are only for non-required checks and do not
lower the verdict. Redact all secrets.

## Check the host and SDK

For the requested scope:

1. Record the host operating system and architecture.
2. Check the matching SDK release notes. If they are unavailable, record a
   `LIMITATION` and continue direct probes.
3. Check each required executable with `command -v` and its version command.
4. Read the CMake requirement from the selected SDK's Build System and
   Installation documentation. Check additional minimums only for requested
   features, such as CUDA interoperability; record the local citation.
5. For L4T, discover compiler names and paths from the selected SDK toolchain
   and its Installation documentation. Probe those executables, including a
   user-supplied toolchain when configured.
6. Check `ssh`, `scp`, and `tar` only for direct target work.
7. Check `sha256sum` or `shasum -a 256` for evidence manifests.
8. Require `PVA_OPERATOR_MAX_DEBUG_ATTEMPTS` only for `opgen`. This environment
   variable caps Debug invocations after failed Build-Test runs; `0` disables
   automatic repair. Require a non-negative integer and use the inherited
   value. Do not load an arbitrary `.env` file.
9. Confirm the SDK root is readable. Discover the files needed for the requested
   capability from its package metadata, CMake configuration, and matching SDK
   documentation instead of requiring a fixed subdirectory list.
10. Check required packages with package metadata when it is available. Do not
    construct a package name from a version.
11. Create a temporary CMake project outside the SDK. Put
    `find_package(pva-sdk REQUIRED)` before `project(...)` when using the SDK's
    toolchains: the SDK's Build System / CMake Scripts section requires this
    order to select the toolchain before CMake enables languages. With an
    explicit custom toolchain, follow the selected SDK's documented setup.
    Configure with the requested SDK root; confirm the selected package and path.
12. Remove the temporary project.

Do not accept a directory as proof that an executable works.
If an SDK requirement cannot be established, record a required `LIMITATION`
and the available probe evidence instead of guessing a version or layout.

## Check the project

When PVA Solutions is selected, use the supplied project path or
`$WORK_DIR/source`. Record a required `pva-solutions-source` check. Verify that
the directory and its documented build files are present and readable, and
check any requested version using local metadata. Record the absolute path,
version, and repository URL and commit when Git metadata is available.
Accept a supplied source archive or another checkout; do not require a specific
remote URL or a private repository layout.

A missing or incomplete required source tree, or a confirmed version mismatch,
is `FAIL`. If its identity or requested version cannot be verified, record
`LIMITATION`. Continue these checks even when the SDK is missing. Recommend
`pva-setup`, which defaults to https://github.com/nvidia/pva-solutions.git;
do not clone, fetch, or restore files in Doctor. For a standalone SDK
application, mark `pva-solutions-source` as non-required `NOT_REQUESTED`.

When a project is in scope:

1. Verify its supplied version from local build or package metadata.
2. Confirm that required build files are readable.
3. Prove write access with a temporary file only for `opgen` or `debug`.
   Remove the file.
4. Read local minimum and default SDK settings. Use a configure or build probe
   to prove compatibility.
5. Check CUDA, NvSci, DLInference, Git, or Git LFS only when the requested work
   needs them. Git metadata is optional for an extracted source archive.
6. When compilation is requested, record `project-lfs-assets-<mode>` for each
   build mode, such as `l4t` or `emu`. Identify needed LFS assets from local
   build rules and LFS metadata, including required submodules. Check that
   each working file exists, is readable, and is not an unresolved LFS pointer
   (first line `version https://git-lfs.github.com/spec/v1`). An installed LFS
   tool or a cached object alone does not prove the working file is present.
   Apply the file check to extracted source archives too.

Record the mode, required status, affected paths, evidence, and remediation.
Missing assets for a required mode are `FAIL`. Missing optional EMU assets
are `NOT_AVAILABLE`, with EMU status `UNAVAILABLE`; required L4T checks
continue. Record `NOT_REQUESTED` when a mode needs no LFS assets. If the needed
asset set cannot be established, record a required `LIMITATION`, or optional
EMU `NOT_AVAILABLE` with status `UNPROVED`, and explain the evidence gap.
This asset check reads files only; it does not build, link, run, or fetch assets.

Do not require a private repository layout.

## Check target compilation

Run this check only when the requested scope needs an L4T build.

Discover NVVPU and ASIP/Chess automatically. Do not ask the user to select a
backend.

For NVVPU:

1. Check installed package metadata or equivalent SDK files.
2. Check the CMake toolchain and compiler executable.
3. Do not accept their presence as final proof.

For ASIP/Chess:

1. Resolve the ASIP Programmer from the CMake option, environment, or
   documented default.
2. Check the release for the selected PVA generation.
3. Check its executable tools.
4. Check `LM_LICENSE_FILE` or `SNPSLMD_LICENSE_FILE`. Do not print either
   value.
5. For a license file, check that it is readable. For a server, report only
   that a server specification exists.

When build probes are permitted, use a supplied project or SDK sample:

1. Try NVVPU first when its required files are present.
2. Configure L4T for the requested generation in a new build directory.
3. Build one VPU target in the foreground.
4. If NVVPU does not succeed, try ASIP/Chess in a separate build directory.
5. Record the backend, executable, compiler model, command, and exit code.
6. Remove each temporary build directory.

Use these rules:

- A successful NVVPU build proves compiler readiness. Set ASIP and Synopsys
  license checks to `NOT_REQUESTED`.
- A successful ASIP/Chess build proves compiler and license readiness. Report
  NVVPU as an unused optional backend.
- Fail the compiler check only when neither backend can complete the probe.
- If no probe is permitted, report a required `LIMITATION`. Do not return
  `READY` from files or environment values alone.

## Check optional EMU support

Check that the selected SDK documents and provides `PVA_BUILD_MODE=EMU` for
host emulation. Do not substitute another host execution tool. Keep EMU
settings separate from the NVVPU or ASIP/Chess compiler used for L4T builds.

Before reporting EMU support as available, configure, build, and run a small
supplied EMU target in a temporary build directory outside the SDK. Record
the proved build mode, SDK selection, generation, compiler and other configure
options, plus the exact commands, exit codes, and logs in the report. Build-Test
reuses these settings with fresh project and build paths. Remove the temporary
build directory after saving the evidence.

For a supplied project whose EMU build needs LFS assets, EMU status
`AVAILABLE` also requires its `project-lfs-assets-<mode>` check to pass. A
successful SDK sample does not override missing or unproved project assets.

When EMU support is `optional`, report missing support as `NOT_AVAILABLE`.
Continue all required L4T checks.

When EMU support is `required`, report a missing dependency or failed probe as
`FAIL`. If no probe is permitted, record `LIMITATION` and EMU status `UNPROVED`;
do not claim a working backend from package presence alone.

Do not report EMU execution as hardware verification.

## Check a direct Tegra

Run this check only when the requested capability needs hardware access.

Read the installed BSP release over SSH and check compatibility with the
selected SDK. Check an expected BSP release only when the user supplies one.
Record the installed release, compatibility source, and result.

Use key or SSH-agent authentication in batch mode. Use the supplied
known-hosts file. Check the Arm64 architecture, BSP, cuPVA runtime version and
safety variant, required libraries, and permissions for the requested commands.
Use the supplied account without elevation where possible. Check administrative
permissions only for a requested operation that needs them, such as clock
configuration or privileged deployment. Accept a user-approved command-scoped
noninteractive method or an already prepared target; do not require unrestricted
passwordless `sudo`. Probe the specific permission without changing target state.

Do not install packages or change target permissions. For a required operation,
record a demonstrated permission denial as `FAIL`, or an unproved permission as
`LIMITATION`, with the exact operation. Mark unneeded privileged operations
`NOT_REQUESTED`. An unavailable optional target capability is `NOT_AVAILABLE`.

## Decide and report

- `READY`: Every required check is `PASS`. When L4T compilation is required,
  either NVVPU or ASIP/Chess passed the target build probe.
- `READY_WITH_LIMITATIONS`: No required check is `FAIL`, and at least one is
  `LIMITATION`.
- `NOT_READY`: At least one required check is `FAIL`.

Only checks marked required determine the verdict. Missing optional EMU
support (`NOT_AVAILABLE`) and out-of-scope checks (`NOT_REQUESTED`) are neutral.
A required capability must not be relabeled optional to avoid a failed check.

Write `$WORK_DIR/reports/doctor-report.md` with
[the report template](assets/doctor-report-template.md).

Return:

```text
Verdict: READY | READY_WITH_LIMITATIONS | NOT_READY
Requested capabilities: <list>
Required mode: L4T
EMU policy: <optional | required>
PVA generation: <generation>
Target compiler: <NVVPU | ASIP/Chess | none>
Failed checks: <count and names>
Limitations: <count and names>
Report: <absolute path>
Next action: <one action or none>
```

For each product fact, cite the SDK documentation root, local Markdown path,
packaged source path, source SHA-256, version, and section or PDF page.
