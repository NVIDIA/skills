---
name: cupotential-custom-potentials
description: Build and validate cuPotential custom pair potentials through its OpenMM expression API, including WCA, LJ fusion, soft-core interactions and live globals. Review path-specific support for alchemical estimators and analytic energy-parameter derivatives.
license: Apache-2.0
metadata:
  author: Ohad Mosafi <omosafi@nvidia.com>
  tags: [cupotential, openmm, pair-potentials, alchemy]
  status: alpha
  version: "0.1.0"
---

# Implement a custom pair potential

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Deliver a supported expression, its physical conventions, and checks of its
energy, forces and requested parameter changes. This workflow uses
`nv_openmm_plugin`; caller-written CUDA evaluators and experimental C ABI
integration need separate version-specific work.

The upstream library is [NVIDIA/cuPotential](https://github.com/NVIDIA/cuPotential).
This skill was reviewed against [v0.3.5](https://github.com/NVIDIA/cuPotential/tree/v0.3.5);
match version-specific behavior to the user's selected runtime.

## Instructions

### Select the requested deliverable

For an implementation task, use the selected interpreter and available runtime.
Read the supplied model and bundled recipes first. Inspect the installed
constructor/setter signatures once, then build and exercise the requested model.
Resolve installed files from module locations and package metadata; use matching
official documentation for a specific unresolved API question. Limit diagnostic
output to interpreter/module paths, package versions, platform names and the
specific failure. Avoid dumping the full environment, credentials or package
index configuration while diagnosing an import or CUDA failure.
For preinstalled-runtime tasks, preserve the supplied environment instead of
substituting a different package stack. If an import or CUDA probe fails, retain
the exact failure and prepare the remaining code and independent checks. Bound
runtime discovery to the selected environment instead of scanning the filesystem
root; identify which requested execution remains blocked.

For an assessment based on a supplied API contract, use that contract directly.
Write the requested decision artifact with the required observable, the relevant
contract provision, the implementation/protocol decision, and remaining evidence
needs. Importing packages, installing software and running a simulation are
unnecessary for this mode.

### CPU reference fallback

If CPU fallback is permitted, execute independent analytic/finite-difference
checks and, when stock OpenMM is available, a separate stock `CPU` or `Reference`
definition. For WCA, retain unequal-particle mixing, nonzero and outside-support
controls, exclusions and the reversible strength sequence. For a fused expression,
the stock control must retain full LJ, electrostatics and the addition.
An isolated stock OpenMM installation is an option only when package changes are allowed.

Record CPU results separately from the prepared cuPotential definition. Label
cuPotential CUDA evaluation, live native parameter updates and native-versus-stock
comparisons `NOT_RUN` until actually exercised. If CUDA was required, the requested
execution remains incomplete. Keep failed checks and temporary caches rather than
cleaning away evidence; `python -B` avoids creating bytecode caches.

### Define the interaction before implementing it

Establish the complete pair energy, physical units, symmetric combining rules,
cutoff, switching, exclusions, exceptional pairs, and any periodic lattice terms.
Identify which existing interactions are being replaced and which remain.
For alchemical inputs, identify the meanings of the lambda parameters and the
unchanged bonded/exception terms as well as the nonbonded expression.

Read [expression recipes and checks](references/potentials.md) for the appropriate
construction and its limitations. The reviewed expression API uses nanometers
and kJ/mol; parameter arrays contain numbers in the expression's declared units.

### Choose the supported construction

Convert a supported periodic-cutoff `openmm.CustomNonbondedForce` for a standalone
custom interaction. To combine custom LJ with standard periodic electrostatics,
use a standard cuPotential force and its `setCutoffInteraction` API.
That operation replaces the complete ordinary LJ contribution. Include the
ordinary interactions that must remain in the new expression.

Custom cutoff expressions do not acquire a reciprocal sum automatically. A
declared lattice term describes the decomposition of a bare term already present
in the full expression. Check the installed release's supported kernels and
corrections. Apply a switch once, and account for its derivative and tail treatment.
Interaction groups, tabulated functions, and requested energy-parameter derivatives
need explicit compatibility checks rather than automatic conversion assumptions.

### Exercise the definition

Start with a small nonzero interaction having an independent analytic or stock
reference. Check force sign and symmetry, finite differences over several step
sizes, exceptional/excluded pairs, and both sides of the cutoff. Keep the chosen
acceptance checks separate from exploratory diagnostics. The reference contains
an independent WCA formula and a complete LJ-plus-Gaussian construction pattern.

Verify a live global update by changing its value, observing the requested
energy/force response, and returning to the starting value. Rebuild or reinitialize
when the expression/schema changes instead of treating it as a scalar update.

For soft-core work, include endpoint and intermediate-state checks appropriate to
the requested model. The upstream
[TYK2 alchemical sample](https://github.com/NVIDIA/cuPotential/tree/v0.3.5/plugins/openmm/samples/alchemical)
demonstrates parameter updates and short dynamics. It is an optional source
example in a separate cuPotential checkout. A binding free-energy claim
additionally needs endpoint sampling, overlap/convergence analysis, an estimator,
and applicable corrections.

### Reconcile the evidence before reporting completion

For requested execution, account for each applicable item below in the output
artifacts. Record its command, observed result and status: passed, failed or
blocked. Keep earlier outputs and failed checks when revising the implementation;
use a new run subdirectory when an output location already contains results.

- The selected runtime created the cuPotential Context and produced finite
  energy/forces, including a smooth-interior control with an expected nonzero force.
- An independent reference and a finite-difference step sweep exercised the
  requested interaction, with budgets declared before candidate comparisons.
- Pair support, exclusions/exceptions, retained electrostatics and ordinary LJ
  match the requested model; force inventories account for each contribution.
- A requested live parameter change was exercised and restored in the same
  Context at fixed geometry.
- The requested destination contains runnable code, raw results and a concise
  conclusion that identifies any failed or blocked check.

## Examples

Request: "Replace ordinary LJ with WCA, preserve periodic electrostatics and
exceptions, and check that the custom force matches its energy derivative."

Follow the [WCA recipe](references/potentials.md), documenting the combining
rules, cutoff and retained exceptions. Return the expression and construction
code together with a nonzero analytic control and finite-difference checks on
smooth configurations. Report boundary checks separately.

Request: "This installed API supports live lambda updates but lacks analytic
energy-parameter derivatives. Can I use my TI estimator?"

Compare the estimator's required observable with that path's contract. Explain
the incompatibility and a supported alternative or explicit protocol change;
retain the distinction between an API assessment and an executed estimator.
