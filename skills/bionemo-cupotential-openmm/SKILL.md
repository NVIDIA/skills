---
name: cupotential-openmm
description: Adapt an OpenMM simulation to cuPotential, run NVT or NPT, and save or resume checkpoints while preserving the supplied force field and simulation settings. Use for nv_openmm_plugin force replacement, simulation setup, parameter changes, and restart problems.
license: Apache-2.0
metadata:
  author: Ohad Mosafi <omosafi@nvidia.com>
  tags: [cupotential, openmm, molecular-dynamics, checkpoints]
  status: alpha
  version: "0.1.0"
---

# Run OpenMM simulations with cuPotential

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Produce an adapted simulation and evidence for the requested execution or restart.
Use the user's application and selected interpreter; installed-plugin users do
not need a cuPotential source checkout.

The upstream library is [NVIDIA/cuPotential](https://github.com/NVIDIA/cuPotential).
This skill was reviewed against [v0.3.5](https://github.com/NVIDIA/cuPotential/tree/v0.3.5);
match version-specific behavior to the user's selected runtime.

## Instructions

### Establish the simulation contract

Inventory the input System, coordinates, box, forces, exceptions, constraints,
virtual sites, masses, precision, integration settings, and requested ensemble.
Keep an independent stock System for comparisons. Preserve supplied files.
Determine whether the task requests a force replacement alone or also cuPotential's
integrator and barostat. The blog's example settings describe its benchmark system;
retain the user's timestep and hydrogen-mass/constraint policy.

Read [conversion, execution and restart](references/simulation.md) for the chosen
operation. Confirm the installed release's supported source methods and numerical
settings before converting. The reviewed 0.3.5 API accepts standard PME/LJPME
conversion with its fixed numerical policy; it does not accept arbitrary pinned
meshes or accuracy targets. Explain incompatible requirements before changing
them. Geometry restrictions also need checking for the supplied box and cutoff.

### CPU reference fallback

When CUDA/cuPotential is unavailable and a CPU fallback is permitted, run a
separate stock OpenMM control on `CPU` or `Reference`, preserving the input model.
Keep the stock integrator, COM remover and, for NPT, stock barostat in that control.
The [CPU procedure](references/simulation.md#cpu-reference-control) and bundled
`scripts/run_cpu_reference.py` exercise energy/forces and a fresh-process restart.
An isolated stock OpenMM installation is an option when package changes are allowed.

Label its result `CPU_REFERENCE_PASS` or `CPU_REFERENCE_FAILED` and record
cuPotential/CUDA as `NOT_RUN`. A stock checkpoint does not test native cuPotential
serialization, COM transfer or barostat state. Prepare those conversions separately.
If the request requires CUDA, the requested cuPotential execution remains incomplete.
Retain failed attempts; use `python -B` to avoid bytecode cache cleanup.

### Adapt and execute

Construct each replacement before removing its source force, and replace it once.
OpenMM owns and deletes removed forces: extract everything needed from a source
before removal. Preserve untouched forces and particle ordering. Check the final
inventory for omissions and double-counting.

If replacing the integrator, transfer the existing center-of-mass removal policy
explicitly; cuPotential's default frequency can otherwise introduce a change.
If retaining the stock integrator, retain its remover. Use one intended barostat,
preserving pressure, temperature, frequency, and the requested seed policy.

Create the Context only after configuring the System. Observe initial finite
energy/forces, then execute the requested duration with appropriate reporting and
checkpoint output. Global value changes use `Context.setParameter`; supported
particle values use `updateParametersInContext`. Changes to force structure,
expression layout, or certain policies require reinitialization.

### Verify the requested outcome

Check state restoration in a fresh compatible Context when restart is requested.
Distinguish restored positions, velocities, box, time and parameters from identical
future stochastic trajectories, especially for NPT barostat bookkeeping.

Report actual steps and outputs, preserved settings, deliberate changes and
unresolved errors. A running simulation or a restored checkpoint does not establish
numerical accuracy; compare fixed configurations when accuracy is part of the task.

## Examples

Request: "Convert my periodic OpenMM simulation to cuPotential, retain my NPT
settings, run 1,000 steps, and verify a checkpoint restart."

Use the [conversion and restart procedure](references/simulation.md). Return the
adapted script, before/after force inventory, observed step count and checkpoint
restoration checks from a fresh compatible Context. Preserve the supplied
thermostat, barostat and COM policy unless the request changes them.
