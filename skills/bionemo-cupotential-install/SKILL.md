---
name: cupotential-install
description: Install cuPotential's OpenMM plugin in a selected Python environment, verify CUDA execution, and diagnose driver, package, or shared-library loading problems. Use for installation, upgrade planning, missing nv_openmm_plugin, or a plugin that imports but cannot run.
license: Apache-2.0
metadata:
  author: Ohad Mosafi <omosafi@nvidia.com>
  tags: [cupotential, openmm, cuda, installation]
  status: alpha
  version: "0.1.0"
---

# Install and verify cuPotential

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Deliver a usable installation in the user's chosen environment, or a precise
explanation of the requirement preventing execution. The distribution is
`nv-openmm-plugin`; the Python import is `nv_openmm_plugin`.

The upstream library is [NVIDIA/cuPotential](https://github.com/NVIDIA/cuPotential).
This skill was reviewed against [v0.3.5](https://github.com/NVIDIA/cuPotential/tree/v0.3.5);
match version-specific behavior to the user's selected runtime.

## Instructions

### Choose an installation route

Identify the interpreter and requested release before changing packages. Run
`scripts/inspect_environment.py` with that interpreter for a read-only report of
package metadata, module locations, and visible NVIDIA devices. Resolve the script
relative to this skill's installed location, not the user's working directory.
The report does not import OpenMM or establish working CUDA execution.

Read [environment and smoke checks](references/environment.md) for the selected
route. Prefer a compatible published package for a consumer installation; use
the release's Docker/source workflow when the user needs a build or an isolated
toolchain. A repository clone is optional for package users. Resolve actual
release assets rather than issuing commands containing documentation placeholders.

Use documentation matching the selected release. The reviewed 0.3.5 source uses
CUDA 13.4; the blog's CUDA 13.1 environment is a different build example. A driver's
reported CUDA ceiling does not identify an installed compiler or the libraries
the selected Python process loads.

An explanation or environment assessment can finish without installing software.
When installation is requested, keep changes in the selected environment and
record the resolved versions and route.

### Verify the result

Check imports, OpenMM CUDA platform availability and loader failures, then run
the small energy/force probe in the reference. Successful import and a completed
CUDA evaluation are separate observations. Missing distribution metadata in a
source installation does not itself make a successful import invalid.

For failure, capture the exception and actual module/library locations. Separate
an absent GPU, an unsupported platform, a missing CUDA library, an OpenMM/plugin
version conflict, and a runtime ABI mismatch before choosing a remedy. Use a
matching dependency set instead of globally replacing shared libraries to hide
a loader error.

Return the interpreter, resolved versions, installation command, executed probe
and result, and any remaining limitation. Without CUDA execution, report the
installation or code preparation achieved and leave the GPU probe unverified.

### CPU-only verification fallback

If a CPU fallback is permitted, use a separate stock OpenMM `CPU` or `Reference`
energy/force control to verify the available Python/OpenMM setup. Install stock
OpenMM in a new isolated environment only when allowed. Record its interpreter,
versions, platform and observed numbers separately from the cuPotential install.
Do not replace a requested release or interpreter to make that installation pass.
Report cuPotential installation and CUDA verification independently; a successful
stock CPU control leaves the cuPotential GPU probe unverified. If GPU execution
was required, the original verification remains incomplete.

### Bundled helper

| Script | Inputs and output |
|---|---|
| `scripts/inspect_environment.py` | No arguments; writes JSON describing the selected interpreter, package discovery and visible NVIDIA tools. Exit 0 means the inventory completed, including when dependencies are absent. It does not run CUDA. |

## Examples

Request: "Check whether my selected Python environment can run cuPotential before
installing or changing packages."

Run the inventory with that interpreter, resolving the script from the installed
skill directory:

```bash
python /path/to/cupotential-install/scripts/inspect_environment.py
```

Use the report to select a compatible route. When execution verification is
requested, run the separate [CUDA probe](references/environment.md) and report
its actual result alongside the inventory.
