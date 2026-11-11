---
name: pva-setup
description: >-
  Install and configure the NVIDIA PVA SDK, NVVPU compiler, and L4T target
  runtime from the selected installation guide. Use when
  the user requests PVA environment setup on a development host and board.
metadata:
  author: "Venkat Balan <venkatramanb@nvidia.com>"
---

# Set up a PVA development environment

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

Support regular x86-64 Ubuntu on the development host and Jetson Linux (L4T)
on the Arm64 target. Run the agent and build tools on the development host.
`PVA_BUILD_MODE=L4T` cross-compiles for execution on the board;
`PVA_BUILD_MODE=EMU` builds for emulation on the development host.
Use NVVPU for non-safety development. DriveOS and ASIP setup are outside
this workflow.

Workflow: Doctor → Setup when needed → Doctor → Spec → OpGen.
Doctor checks readiness without installing software. The user invokes
Setup when installation or configuration is needed.

## Resolve the inputs

Use the user request and Doctor report to resolve:

- Doctor report and requested capabilities.
- Installation guide, supplied as a local file or URL; PVA generation.
- Optional SDK release or installation-path override supplied by the user.
- Development host and target SSH connection; expected BSP release, if supplied.
- Work directory (`WORK_DIR`); use Doctor's work directory when supplied.
- Optional EMU, CUDA, NvSci, or PVA Solutions requirements and project path.

When PVA Solutions is selected, use the supplied project path or default to
`$WORK_DIR/source`. Reuse an existing source tree. Otherwise clone
https://github.com/nvidia/pva-solutions.git there, unless the user supplies
another repository or source archive. Honor a requested Solutions tag, branch,
or commit; without one, use the repository's default branch. Record the source
URL, absolute project path, version, and resolved commit when available.
Do not infer a Solutions version from the SDK version.

Resolve the SDK release, installer URL, package names, and installation paths
from the selected guide and package metadata before invoking Doctor. This
includes SDK, NVVPU, CMake package, and target runtime paths. Do not hard-code
package paths or construct names or paths from a version. Do not require the
user to repeat values that these sources provide. Honor an explicit release
or path override; check that the guide and installer support it. If the release
or installer source cannot be established, report `BLOCKED`. Do not guess a
version or substitute another release.

Resolve relative paths against the invocation directory before changing
directories. Execute host installation commands on the development host,
and target commands through its target connection. Keep generated files in
the work directory, outside installed skills and SDK documentation.

Use inherited credentials or configured authentication. Never print secrets.
In headless runs, report `BLOCKED` with the exact missing input or access
requirement instead of asking questions.

## 1. Read Doctor's findings

If no applicable Doctor report exists, invoke `pva-doctor` first using the
agent's available skill mechanism. Keep the same capability and target scope.

Determine readiness from current checks, not a marker that Setup ran.
Confirm the report describes the selected host, target, SDK, and scope.
Recheck package versions and paths before deciding whether changes are needed.

- If requested capabilities are ready and the installation matches the
  request, make no package changes. Rerun Doctor and report the result.
- If required components are missing, identify the installation steps.
- If SDK references are missing, outdated, or damaged, prepare them in step 5.
  Reuse a matching installed SDK; this failure alone needs no installer or sudo.
- If a failure concerns operator code or tests, route it to Build-Test or
  Debug instead of reinstalling the environment.

An already prepared environment does not need an installer or sudo access.
Do not treat `READY_WITH_LIMITATIONS` as proof that every component works;
resolve or report the specific unproved check.

## 2. Check installation prerequisites

For each needed change, read the selected release's installation guide and
compatibility notes. Record the source, release identity, and relevant section.
If the guide or artifact does not match the requested release, report
`BLOCKED` before package changes. Do not substitute the latest release.

Check the affected machine's operating system, architecture, installed
versions, BSP compatibility, installer access, and available disk space.
Inspect installer package metadata before installing it. Use package names
and versions from the guide and repository metadata; do not guess them.

List the necessary changes and reuse components that satisfy the request.
Do not upgrade, downgrade, or remove unrelated packages. If the package
transaction would exceed the requested setup, report the proposed changes
as `BLOCKED` rather than applying them.

Use existing authorized administrative access for the specific operations.
Do not require unrestricted passwordless sudo. If an operation is unavailable,
report `BLOCKED` with the machine, operation, and reason.

## 3. Set up the development host

Follow the guide for the selected release:

1. Obtain and inspect the local installer.
2. Register its package repository and refresh package metadata.
3. Confirm the selected package versions, dependencies, and transaction.
4. Install the SDK, NVVPU compiler, and required build dependencies.
5. Verify versions and actual paths from installed package metadata.

Use the package names listed in the installation guide. Read their actual
versions from the installer or package repository. Confirm they match the
selected release before installation, then check the installed versions.

Provide Python 3.9+ and `pdftotext` when Setup needs them to materialize SDK
references. Install CUDA or NvSci dependencies only when requested work needs
them. Record any necessary environment settings without silently editing
the user's shell startup files or credential files.

EMU is included in the core SDK for releases that document this support.
Use only the packages that the selected guide requires for EMU.

If PVA Solutions is selected, obtain its source as resolved above. Provide Git
for a clone and Git LFS when the source uses it. Restore required submodules
and working LFS files according to the repository's instructions for the
selected build modes. Preserve local changes; do not replace a nonempty
destination, reset a checkout, or pull unrelated updates. If source access or
a requested revision is unavailable, report `BLOCKED` with the reason.
A cached LFS object alone does not prove the working file exists.

Follow the guide's cleanup steps only for temporary installers and repository
registrations created by this run. Preserve preexisting repositories and files.

## 4. Set up the L4T target

Read the board's installed BSP release over SSH and check compatibility with
the selected SDK before installing its runtime. Check an expected BSP release
only when the user supplies one.
Use the supplied SSH account and trusted host key. Transfer the installer from
the development host when the board cannot access the package source directly.

Follow the guide to install the Arm64 L4T runtime package it names. Inspect
the transaction and reuse a compatible installed runtime. Verify its version,
architecture, and non-safety variant against
the selected release's compatibility rules and the user's version request.

Do not flash the board or change SSH trust, security settings, or clocks
as part of package setup. If a command fails, save the failure and inspect
current state before retrying; do not repeat a failed transaction blindly.

## 5. Prepare SDK references

Resolve the installed `pva-doctor` skill directory through the available skill
mechanism. Use its bundled materializer and manifest with the selected SDK's
documentation and `$WORK_DIR/references/sdk`. Follow its
`references/reference-maintenance.md` for path checks, prerequisites, and commands.
Keep the installed skills and SDK documentation unchanged.

Run `verify` first when a corpus exists. Reuse it if verification passes.
For a missing or failed corpus, run `sync`, then `verify`, within the authorized
workspace. Preserve the initial Doctor report and failure evidence before
replacement. Report a materializer failure accurately; do not edit hashes or
claim reference readiness. No SDK reinstall is needed solely to refresh references.

## 6. Run Doctor again

Invoke `pva-doctor` with the same requested capabilities, SDK, generation,
project, target, and emulator policy. Preserve the initial Doctor report before
writing the final report. Doctor owns compiler probes, reference verification,
and readiness checks; package installation alone does not prove readiness.

Keep unavailable optional capabilities neutral. Report each remaining required
failure or limitation and its next action. Do not proceed to Spec or OpGen.

## Report

Write `WORK_DIR/reports/pva-setup-report.md`, including on an early stop:

- Requested release, scope, installation source, and host/target identity.
- Components reused, installed, or configured, with commands and exit codes.
- Verified versions and paths, plus saved evidence locations.
- Remaining failures or limitations and any partial changes.
- Initial and final Doctor report paths and the final readiness verdict.

Print a concise summary and next action. Use Doctor's final verdict when it
ran. Otherwise report `BLOCKED` for missing input or access, or `NOT_READY`
for a demonstrated installation failure. Never claim an unperformed check passed.
