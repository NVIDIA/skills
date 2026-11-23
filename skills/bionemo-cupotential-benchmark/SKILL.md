---
name: cupotential-benchmark
description: Benchmark cuPotential against stock OpenMM on a user's molecular system while checking accuracy and matching simulation settings. Use for speedup measurements, startup overhead, performance regressions, or diagnosing unexpectedly slow cuPotential simulations.
license: Apache-2.0
metadata:
  author: Ohad Mosafi <omosafi@nvidia.com>
  tags: [cupotential, openmm, molecular-dynamics, benchmarking]
  status: alpha
  version: "0.1.0"
---

# Measure cuPotential performance

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Deliver measured throughput and timing variation for an explicitly defined
comparison, together with the accuracy evidence needed to interpret it.

The upstream library is [NVIDIA/cuPotential](https://github.com/NVIDIA/cuPotential).
This skill was reviewed against [v0.3.5](https://github.com/NVIDIA/cuPotential/tree/v0.3.5);
match version-specific behavior to the user's selected runtime.

## Instructions

### Define comparable arms

Record the user's system, coordinates/box, force field, precision, timestep,
constraints, hydrogen masses, COM policy, ensemble, output cadence and GPU.
Identify whether the requested comparison concerns force evaluation alone,
complete simulation steps, or end-to-end application runtime.

Use the same physical settings in both arms. Label a cuPotential force plus its
integrator/barostat as an integrated configuration. A force-only speedup needs a
comparison isolating that change. Changing a cutoff, tail model, timestep or
represented spectrum is an accuracy/performance study with additional validation.

Read [measurement procedure](references/measurement.md) for a source-tree runner
or a benchmark in an installed-plugin application. Source checkout and committed
golden results are optional. Verify fixed-configuration accuracy against an
appropriate independent reference before recommending a faster configuration.

### CPU-only fallback

When CPU fallback is permitted, exercise a stock OpenMM `CPU` or `Reference`
control with the same supplied model and report its platform explicitly. Check
finite energy/forces, actual step counts and the timing protocol; an isolated
stock OpenMM environment is an option if installation is allowed. Prepare the
cuPotential arms, but mark their CUDA accuracy checks, timings and speedups
`NOT_RUN`. CPU throughput is a control measurement, not a cuPotential or GPU
benchmark, and cannot establish a GPU speedup or hardware comparison.
Keep an explicitly requested GPU benchmark incomplete until its arms execute.

### Measure the workload

Measure installation/build work, cold startup/Context/JIT, and warm execution
separately. Warm both arms, synchronize timed boundaries, use a declared step
budget, and repeat paired trials. Record absolute times, throughput, variation
and the speedup calculation. Include representative reporter/checkpoint costs
when the question concerns the user's real application.

Check GPU activity and clock/thermal conditions. Use the environment's available
clock policy and label unlocked measurements. If attributing a change to graphs
or a backend, inspect actual counters/selection evidence before making that claim.
Profile the observed bottleneck before trying execution controls.

The reviewed OpenMM API derives its own mesh and has a fixed accuracy policy;
do not invent a public alpha/grid autotuning interface. Recheck the installed
version before using numerical or execution settings.

### Report the comparison

Return arm definitions, input/build/device identities, correctness results,
raw trial times, cold/warm summaries and limitations. A completed local timing
without historical golden data is still a measurement. A syntax check, a skipped
GPU run or the blog's published speedup is not a measurement of this workload.

## Examples

Request: "Measure cuPotential's speedup on my OpenMM system without changing my
timestep or force field. Include startup and checkpoint costs."

Use the [measurement procedure](references/measurement.md) to define the two
configurations and collect paired trials. Return fixed-configuration accuracy,
cold-start costs, warm throughput and timing variation. Identify any unavailable
GPU measurements instead of substituting published performance numbers.
