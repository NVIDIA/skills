---
name: newton-performance
description: Measure or improve the runtime, startup time, memory use, or throughput of a Newton simulation that already behaves correctly, or review a Newton step loop for performance problems such as CUDA graph capture mistakes and host synchronization. Not for setting up simulations, general solver or API questions, debugging unstable or incorrect physics without a performance goal, or optimizing Warp code that does not use Newton.
---

# Newton Performance

Make a correct Newton workload faster without changing its results. Newton's
documentation holds the technical guidance; this skill orders the work and
defines what counts as evidence.

## 1. Scope the request

- If the simulation is unstable, produces invalid states, or the user's goal is
  correctness rather than speed, use `newton-usage` instead and stop here.
- Decide the mode. A **plan** request (how should I profile or optimize this?)
  gets a documentation route and one bounded experiment. A **review** request
  (is this loop correct and fast?) gets findings against the documentation.
  In both, reading code and documentation is fine, but do not run the
  simulation, benchmarks, or profilers unless asked. A **measure** or
  **implement** request runs the workflow below.
- Record the workload, the objective, and the timing scope. Keep these scopes
  separate, because they have different causes: startup (import, model build,
  kernel compilation), warm stepping, rendering, host/device transfers, memory,
  and training throughput.
- Record the fidelity checks that must still pass after a change, for example
  finite state, no contact overflow, and a trajectory or force metric against a
  reference run.

## 2. Gather context once

Reuse facts the user already gave. Otherwise run once:

```
python -c "import sys, newton; from importlib.metadata import version; print(sys.executable, version('newton'), version('warp-lang'), newton.__file__)"
```

Check GPU availability only when the experiment depends on it. If the workload,
a required profiler, or required permissions are missing, report exactly what
is missing and stop. If elevated privileges are needed (for example GPU
performance counters), give the exact command for the user to run and stop.

## 3. Route to the documentation

Use the Newton documentation for the installed release,
`https://newton-physics.github.io/newton/<version>/`, and flag a mismatch
before relying on an API or default. Open only the page the task needs; paths
are relative to that root:

- `concepts/simulation_tuning.html`: trading substeps, iterations, and contact
  settings against accuracy.
- `concepts/collisions.html`: broad phase, geometry, contact capacity, and
  collision frequency.
- `solvers/index.html` and the API reference: solver options, capacities,
  `ModelFlags`, and `State.assign` for odd substep counts in a captured frame.
- Warp's Profiling page for instrumentation and timing, under the installed
  Warp version's documentation, `https://nvidia.github.io/warp/v<major>.<minor>/`.
  The page path differs between Warp versions, so find it from that version's
  index. If that version is not published, use `stable` and flag the mismatch.

Cite the page and section each recommendation relies on.

When the user only needs a documentation location, give the URL without
opening it.

## 4. Measure, change, verify

1. Measure a reproducible baseline for the recorded workload and timing scope.
2. Choose the profiling path for that scope: startup (import, model build,
   compilation), eager stepping, or captured-graph replay. Each needs different
   tools and timing boundaries; Warp's profiling guide gives the tools.
   Exclude warm-up, compilation, and the first graph launch from steady-state
   timing, synchronize at the timing boundaries, and add only the
   instrumentation the question needs.
3. Collect the profile and inspect it: identify the dominant kernels, gaps, or
   synchronization points, and state the evidence. Do not choose a change
   before this step, and do not start from universal tuning values.
4. Select a documented lever whose accuracy tradeoff fits the task. Change one
   factor at a time. When settings interact (for example collision
   frequency, contact model, mesh resolution, and gap), run a small bounded
   sweep over the coupled set instead.
5. Repeat before and after measurements under the same conditions and
   instrumentation to estimate variance, and re-run the fidelity checks from
   section 1.
6. Keep a change only if it meets the objective and passes the fidelity
   checks; otherwise revert it.

## 5. Review checks

When reviewing a Newton step loop, check these against the documentation
selected in section 3 and cite the page:

- the whole frame, including collision, is captured in one graph and warmed up
  before capture;
- an odd substep count with a plain state swap inside a captured frame;
- arrays replaced instead of written in place after capture, or Python values
  expected to change between replays;
- host reads (`.numpy()`, contact counts, printing) inside the frame;
- capacities sized from defaults rather than measured peaks, and overflow never
  checked;
- rendering or logging inside the substep loop, or included in simulation
  timings.

Credit any correct fix the documentation supports, not only the first one
listed there.

## 6. Report

Report the Newton and Warp versions, the environment, the documentation used,
the workload and timing scope, the change, the before and after measurements
with variance, the fidelity evidence, and the largest remaining cost or
blocker. Label proposed experiments separately from measurements actually
collected. CPU measurements or source inspection do not establish a GPU
speedup.

Compatibility with specific Newton releases has not been validated.
