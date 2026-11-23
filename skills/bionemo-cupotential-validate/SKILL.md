---
name: cupotential-validate
description: Investigate cuPotential energy or force differences, validate a converted simulation, and assess numerical or physical simulation failures. Use for comparison with OpenMM, custom-force finite differences, cutoff artifacts, energy drift, or questionable simulation results.
license: Apache-2.0
metadata:
  author: Ohad Mosafi <omosafi@nvidia.com>
  tags: [cupotential, openmm, numerical-validation, forces]
  status: alpha
  version: "0.1.0"
---

# Validate cuPotential results

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Deliver a reproducible conclusion with the comparison actually performed, its
acceptance criteria, measured errors, and remaining uncertainty.

The upstream library is [NVIDIA/cuPotential](https://github.com/NVIDIA/cuPotential).
This skill was reviewed against [v0.3.5](https://github.com/NVIDIA/cuPotential/tree/v0.3.5);
match version-specific behavior to the user's selected runtime.

## Instructions

Snapshot comparison and analytic checks can run on CPU. If generating a stock
OpenMM reference on `CPU` or `Reference`, record that platform explicitly and
keep it separate from cuPotential CUDA results. Missing candidate GPU results
remain unavailable; CPU reference success cannot establish candidate agreement.

### Choose the question and reference

Separate preservation of the requested physical model, numerical evaluation
accuracy, and physical/statistical validity of a trajectory. Establish the
installed versions and the smallest faithful reproduction before changing a run.
Preserve the failing inputs and original observations.

Read [reference protocols and snapshot format](references/protocols.md). Match
coordinates, particle order, box, units, exceptions, switching and tail conventions.
For electrostatic accuracy, assess stock and cuPotential against an appropriately
converged reference. Default Prolate and stock PME can have different approximation
errors. For truncated LJ, keep the same physical cutoff and correction model.
Higher precision alone does not prove reference convergence.

Select checks and budgets before examining candidate results, according to the
user's requested claim and independently assessed reference/precision error.
Retain a failed required check. A later calibration is a separately documented
protocol change, not a reinterpretation of the original pass/fail outcome.

### Measure and investigate

Use fixed configurations first. Report absolute energy difference, force RMS,
largest force-component difference, and a defined relative-force metric when
appropriate. Avoid relative normalization by zero or a nearly cancelling energy
without an absolute budget. The bundled `scripts/compare_snapshots.py` checks
canonical JSON snapshots without importing OpenMM. With no thresholds it produces
diagnostics; with thresholds it judges only those declared checks.

For custom interactions, compare finite differences across step sizes away from
nonsmooth boundaries, then inspect cutoff behavior separately. For unstable
trajectories, investigate units, overlapping particles, forces, constraints and
integration settings. A deliberate diagnostic change should retain a baseline
and should not silently alter the production model.

Use NVE conservation/timestep diagnostics for a suitable constant-energy test,
and ensemble/kinetic checks for thermostatted or barostatted simulations. A stable
temperature or matching checkpoint state is only evidence for that observation.
Statistical conclusions require adequate equilibrated, effectively independent data.

### State the supported conclusion

Return commands, input/build identities, reference construction, metrics, budgets,
and passed, failed or unavailable checks. Do not score skipped tests or empty test
selections as executed coverage. CPU analysis of saved arrays can establish their
numeric differences; it does not establish that a GPU execution occurred.

### Bundled helper

| Script | Inputs and output |
|---|---|
| `scripts/compare_snapshots.py` | Candidate and reference JSON snapshots plus optional budget flags; writes JSON metrics and declared-check results. Exit 0: diagnostics/pass; 1: failed checks; 2: invalid input or an unevaluable requested check. See the [snapshot contract](references/protocols.md). |

## Examples

Request: "Compare these saved candidate and reference snapshots and explain what
the differences establish. I have not selected acceptance budgets yet."

```bash
python /path/to/cupotential-validate/scripts/compare_snapshots.py candidate.json reference.json
```

This produces diagnostics, not a passing accuracy test. To judge a declared
protocol, supply its preselected budget flags; use `--help` and the
[metric definitions](references/protocols.md) to match their units. Keep reference
convergence, model equivalence and GPU execution evidence separate from this
array comparison.
