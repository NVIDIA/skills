---
name: nvat-install-and-guide
description: Use when installing or upgrading the NVIDIA Attestation SDK (NVAT or nvattest), checking platform compatibility, or answering questions about its CLI, C API, or Rust bindings from official documentation.
license: Apache-2.0
---

# Install and use NVAT

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Select the evidence format for the user's hardware, release, and consuming
verifier. Prefer CoRIM when the required workflow is supported. For Hopper legacy/TCG
evidence collection, use `collect-evidence`, not `collect-evidence-v2`.
CoRIM interfaces are experimental; confirm support before giving commands.

## Identify the environment

For an installation request with shell access, inspect the target environment:

```sh
uname -s
uname -m
if [ -r /etc/os-release ]; then cat /etc/os-release; fi
command -v nvattest
```

If installed, run `nvattest version` and `nvattest --help`. For CoRIM, also
check `nvattest collect-evidence-v2 --help` and `nvattest attest-v2 --help`.
An unknown command means the installed build lacks that interface.

Establish whether the target is the current machine, a remote host, container,
or confidential VM. Inspect the target, not the agent's host. When access is
unavailable, ask only for the missing details. Distinguish `x86_64`/`amd64` from
`aarch64`/`arm64` when selecting artifacts.

For live collection, run the
[GPU and NVML checks](references/install-and-docs.md#check-gpu-and-nvml-access)
in the target environment.
A GPU on the host does not establish access inside a container or CVM.
File-based appraisal is separate from live hardware collection and can still
need network access for reference manifests and certificate checks.

## Install for the requested task

Read [installation choices](references/install-and-docs.md#installation-choices)
and match the detected OS, OS version, architecture, and desired feature to
the official download matrix. Confirm the package source before installing.
Do not invent repository URLs, download filenames, or support for macOS,
Windows, or an unlisted Linux distribution.

Use the existing task authorization for installation. Keep driver, firmware,
GPU-mode, and unrelated system changes outside the installation scope unless
the user requests them. If a required privilege or artifact is unavailable,
explain the specific blocker and provide the remaining steps.

Verify the installed version, library availability, and required subcommand
help. Package installation success does not demonstrate GPU attestation.
Report what was installed and what remains untested.

## Answer CLI and API questions

For Blackwell, FSP responders currently use CoRIM, while GSP responders use
TCG RIM. This may change depending on the deployment model; confirm support
for the installed SDK release. Establish the responder from the deployment's
configuration or validated device certificate information. The GPU family
alone does not select the verifier; ask when the responder is unknown.

For Hopper, TCG RIM, or pre-CoRIM collection questions, read
[legacy evidence collection](references/install-and-docs.md#legacy-evidence-collection).
Explain which format the command produces and which verifier accepts it.
Legacy evidence files are not interchangeable with CMW.

Read only the relevant entry in the
[documentation map](references/install-and-docs.md#documentation-map).
Use the installed version's help and matching release documentation/source first.
Treat `/latest/` and public `main` as moving references. If they disagree with
the installed version, explain the difference and use a matching release;
do not silently mix APIs or flags.

Give a direct answer, a minimal example when useful, the applicable version,
and links to the official pages that support it. Read signatures and examples
before writing API code. Verify Cargo dependency syntax against the package's
metadata. If documentation is inaccessible, use matching local docs or public
source and state the limitation. Do not fill gaps from memory.

For a failing workflow, collect the exact command, version, and error, then
consult the matching legacy or CoRIM documentation in the documentation map.
For a new CoRIM appraisal, follow the CLI command reference. Neither task
requires another skill to be installed.
