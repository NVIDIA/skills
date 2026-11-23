# Measuring a comparable workload

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Use the source runner when a matching checkout is available; otherwise benchmark
the consumer application's actual installed runtime. The following notes target
the reviewed 0.3.5 workflow.

## Record conditions before timing

Save system/geometry identifiers, package/runtime versions, device and driver,
precision, timestep, ensemble, constraints, hydrogen masses, COM policy, nonbonded
settings, reporting/checkpoint cadence and active GPU processes. State which
components differ between arms. Decide whether the user's question includes
startup and output costs. Keep raw times and correctness results together.

Evaluate a fixed configuration against a suitable independent reference first.
Check switching/tails and approximation error as well as the requested precision.
Comparing equal physical settings and comparing equal measured accuracy are
different experiments; label the chosen protocol and every parameter changed.

## With a source tree

Use a separate checkout of [NVIDIA/cuPotential](https://github.com/NVIDIA/cuPotential).
For a fresh checkout of the reviewed release, start in the directory where you
want the library source to live:

```bash
git clone --branch v0.3.5 https://github.com/NVIDIA/cuPotential.git
cd cuPotential
```

If a matching checkout already exists, change to its root instead. The following
commands run from that cuPotential checkout, which contains `run_test` and
`run_docker`. Discover stages and test selections in the selected release:

```bash
./run_test --list-stages
./run_test openmm_bench --collect-only -q -k apoa1
./run_test openmm_bench -k apoa1
```

Use the matching `run_docker` wrapper when the configured environment requires
it. Check the collection count: a filter selecting zero tests can appear successful
in a stage wrapper. The shipped runner builds dependencies as needed, so total
command duration includes more than the timing reported by a selected benchmark.

The reviewed benchmark uses 500 warm-up steps and 3,000 timed steps for its
published configuration. These are a reproducible starting procedure, not a
universal duration requirement for other workloads. Clock locking can require
privileges unavailable in a user's environment. Its documented
`CUPO_ALLOW_UNLOCKED_CLOCKS=1` mode records unlocked measurements; keep that
condition in the report. A release checkout may have no historical golden data.
Distinguish a measurement from a comparison with a historical baseline.

## In a consumer application

Construct both Contexts with matched physical settings. Record import/Context
initialization separately, then perform untimed warm-up in both arms. Where a
reproducible paired starting point is needed, restore the same physical state
and parameters for each trial and account for restart limitations.

An explicit synchronized wall-time boundary for complete steps is:

```python
import time

def timed_steps(context, integrator, steps):
    if steps <= 0:
        raise ValueError("The measured step count must be positive")
    context.getState(getEnergy=True)  # Complete previous GPU work.
    start = time.perf_counter()
    integrator.step(steps)
    context.getState(getEnergy=True)  # Include completion in the timed boundary.
    return time.perf_counter() - start
```

This boundary includes an energy query at the end for both arms. Use sufficiently
long trials to assess that overhead. A reporter-driven `Simulation.step` workload
is a different boundary and should include the actual reporters. Repeated force
queries measure force-query throughput, not complete MD simulation throughput.

Repeat paired trials and alternate or randomize arm order where practical.
Report individual times and a summary with variation. For `n` timed steps,
timestep `dt_ps`, and elapsed seconds `t`, throughput in ns/day is
`n * dt_ps * 86.4 / t`. For matched work, speedup is `t_stock / t_cupotential`.
Keep JIT/cold-cache experiments isolated; clearing a shared active cache can
invalidate another process and the measurement.

## Diagnose slow results

Separate Python/reporting overhead, setup/JIT, force work and integration before
changing execution settings. If claiming whole-step graph use, inspect actual
capture/replay counters from the installed integrator. If a backend change alters
the represented spectrum, repeat accuracy checks and report the approximation
change. The reviewed public OpenMM API does not expose arbitrary mesh/alpha
tuning; avoid proposing unsupported methods to optimize it.

Return a table with arm definition, accuracy result, cold time, each warm trial,
throughput and conditions. A slower or inconclusive result is still an outcome.

Sources: [NVIDIA application/benchmark article](https://developer.nvidia.com/blog/accelerate-molecular-dynamics-simulations-with-accurate-flexible-primitives-using-nvidia-cupotential/),
[benchmark documentation](https://docs.nvidia.com/cuda/cupotential#benchmarks), and
[Deserno and Holm's comparison](https://arxiv.org/abs/cond-mat/9807099) for the need
to interpret particle-mesh performance together with approximation error.
