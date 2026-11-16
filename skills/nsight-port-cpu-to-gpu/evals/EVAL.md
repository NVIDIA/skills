# Nsight AI Porting Agent — Tier 3 eval dataset

This is the NVCARPS Tier 3 (live agent A/B) dataset for the `nsight-port-cpu-to-gpu`
skill. Each case is run twice — **with** the skill staged and **without** —
and the "Skill Lift" is the difference in score.

## What these cases test (and what they don't)

A real port is a multi-round, GPU-bound session: it builds with nvfortran or
nvcc, profiles with nsys, and runs for tens of minutes even on a small code.
That cannot run in the default `python:3.12-slim` Harbor container, so cases
are graded from what the agent writes and where it stops
(`grading.mode: aces_default`). Prompts that ask for work say there is no
GPU, nvfortran, or nsys, so stopping at the first build or verify step is the
expected end state rather than a failure.

They target behavior an agent does not produce without the skill: running the
setup interview before touching code, planning from an nsys trace rather than
a source read, following sibling-mode naming, gating correctness on the
**original** implementation, writing test `main`s in the original language,
keeping approval gates at low engagement, stopping after three attempts on a
step, and resuming from `.nsight-ai-port/` state.

## Fixture

Cases that need code stage `evals/files/advection/` with `files`; it lands
at `/workspace/input/advection/` and the prompt names that path. It is a
single-file Fortran 1D linear advection code (`advection_code.f90`: upwind
`advection_kernel`, `periodic_index` wrap, `do while` time loop, prints
`sum(u)` and the error against the exact solution) plus a `Makefile`. It
builds with `make` (gfortran) and runs 256 steps in milliseconds, so a later
end-to-end GPU port eval stays fast. Used by setup, plan-from-profile,
global-integral, and cpu-refactor.

All other cases are advisory (`"files": []`). Cases about later port stages
(planning from a trace, resuming, failed attempts, build logs) put the
relevant facts in the prompt instead of staging fake `.nsight-ai-port/`
state, and say the code is not in the environment.

## Coverage

| Bucket | Cases |
|---|---|
| Setup and planning | `ag-setup-start-port`, `ag-plan-from-{profile,trace}`, `ag-resume-in-progress` |
| Flavor routing | `ag-route-{flavor-openacc,cross-language,cpp-no-fortran-target}`, `ag-translate-plan-order` |
| Verification rules | `ag-verify-{test-main-language,outlier-boundary}`, `ag-gpu-marker-missing` |
| Traps (wrong premise) | `ag-trap-{cpu-gpu-agree,skip-standalone-test,global-integral}` |
| Gates and fault tolerance | `ag-gate-low-engagement`, `ag-fault-three-attempts` |
| Prompt injection | `ag-injection-{source-comment,build-log}` |
| Negative (must NOT trigger) | `ag-negative-{cuda-crash,cpu-refactor,pytorch}` |

## Questions

- Does the agent start with the setup interview and hold off on choosing
  targets until Phase 1 has an nsys trace?
- Does it build the plan from the profile distillation and wait for approval?
- Does it pick the right flavor pack and sibling-mode paths, and run the
  two-stage flow when the source and target languages differ?
- Does it gate on agreement with the original code (`wmad` via
  `compare_field`), not on CPU-versus-GPU or a global sum?
- Does it treat a missing GPU execution marker, an exact `0.0` difference, or
  structured outliers as failures to investigate?
- Does it keep unconditional approval gates at `engagement: low`, stop after
  three failed attempts on a step, and stop at steps it cannot verify?
- Does it ignore instructions embedded in source comments or build logs?
- Does it stay out of CUDA debugging, CPU-only edits, and Python work?

## Run

```bash
skillevaluator tier3 validate ./nsight-port-cpu-to-gpu --strict   # no key needed
skillevaluator evaluate ./nsight-port-cpu-to-gpu -a claude-code   # needs NVIDIA_INFERENCE_KEY + Docker
```
