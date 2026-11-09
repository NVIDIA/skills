---
name: nvalchemi-toolkit
description: >-
  Build and change NVIDIA ALCHEMI Toolkit (nvalchemi) workflows for atomistic
  simulation and machine-learned interatomic potentials (MLIPs). Use for
  batched structure relaxation, molecular dynamics, reaction paths, training
  or fine-tuning an MLIP, atomistic datasets in Zarr, generative structure
  workflows, model wrapping, and custom hooks or integrators, with the ALCHEMI
  Toolkit (nvalchemi), or when the user asks for batched GPU atomistic
  simulation and nvalchemi is installed or acceptable. Not for other
  simulation packages the user asked for (ASE-only, OpenMM, LAMMPS, GROMACS)
  when the request does not involve nvalchemi, ALCHEMI Toolkit-Ops
  (nvalchemiops) kernel work, or plain PyTorch questions.
license: Apache-2.0
compatibility: >-
  Runs Python and shell commands to inspect the installed nvalchemi package
  and to run small checks. Reads linked docs and example pages over the
  network, never downloads code to run. No credentials needed.
metadata:
  author: NVIDIA <agent-skills@nvidia.com>
  tags: [nvalchemi, atomistic-simulation, mlip, molecular-dynamics, training]
---

# NVIDIA ALCHEMI Toolkit

This skill helps you write and change code that uses `nvalchemi`, the Python
package of the NVIDIA ALCHEMI Toolkit. It scopes the request, checks the
installed version, and routes to one note per kind of task. The notes say which
component fits which need, how the components connect, which mistakes are
common, and how to check the result. They link to the docs and examples for
the full details.

## Purpose and scope

Use it when the code imports `nvalchemi`, or when the user asks for a workflow
that the toolkit covers, including batched GPU simulation where `nvalchemi`
is installed or acceptable: `AtomicData` and `Batch`, Zarr storage, model
wrappers, relaxation and MD, reaction paths, enhanced sampling, training,
fine-tuning, distillation, generative sampling, or a custom hook, integrator,
loss, or reporter.

Do not use it when:

- The task is plain ASE, OpenMM, LAMMPS, GROMACS, or another package, and
  the user did not ask for `nvalchemi`. Solve it with that package and do not
  import `nvalchemi`.
- The task is a Warp or CUDA kernel inside ALCHEMI Toolkit-Ops
  (`nvalchemiops`). That is a different package with its own rules.
- The question is general PyTorch with no atomistic workflow.

## Prerequisites

- Python 3.11 to 3.13 with `nvalchemi-toolkit` installed (`pip` or `uv`).
- PyTorch with the CUDA build that matches the host, or the CPU build.
- Optional model backends are extras: `mace`, `aimnet`, `uma-cu12`/`uma-cu13`,
  `ase`, `pymatgen`, `tensorboard`. UMA conflicts with MACE, so it lives in its
  own environment.

## Instructions

Work through these steps for every request.

### Scope the request first

Before you write or run code, go through this checklist. Ask about unknown
scientific choices, hardware, workload, correctness checks, and permissions
(items 2, 7, 8, 9, 10). For items 1, 3, 4, 5, 6, state a one-line assumption
if needed. Check the installed software before execution.

1. **Intended result.** What number, file, or model does the user want at the
   end?
2. **Scientific choices.** Model and checkpoint, ensemble (NVE, NVT, NPT),
   temperature, pressure, timestep, convergence threshold, run length. Do not
   pick these silently.
3. **Inputs.** Structure files and their format, units, which fields they
   carry (energies, forces, velocities, cells). Read the files first and ask
   only for what they do not say.
4. **Deliverables.** Script, data store, checkpoint, plot, or a short report.
5. **Reuse.** Which shipped component already does this (see the reuse rule).
6. **Software environment.** Installed `nvalchemi` version and extras.
7. **Hardware.** CPU only, one GPU, or several. Memory limits.
8. **Workload.** Number of systems, atoms, and steps.
9. **Correctness check.** How the result will be checked (see each note).
10. **Permissions.** What you can write, download, or delete.

Limit exploratory runs to at most 3 systems and 100 steps, with a wall-time
limit under one minute. Do not start a production-length trajectory or a long
training run to "see what happens". Scale up only after the user agrees to the
settings.

### Check the installed version

The notes describe the API of release 0.3.0. Their links point at the
v0.3.0 tag. Check what is installed before you apply any advice:

```bash
python - <<'EOF'
import importlib.util as u
for parts in ((), ("data",), ("dynamics",), ("models",), ("gen",),
              ("dynamics", "mep"), ("enhanced_sampling",),
              ("training", "distillation")):
    m = ".".join(("nvalchemi", *parts))
    try:
        found = u.find_spec(m) is not None
    except ModuleNotFoundError:
        found = False
    print(m, found)
if u.find_spec("nvalchemi") is not None:
    import nvalchemi
    print(nvalchemi.__version__, nvalchemi.__file__)
EOF
```

If the toolkit is not installed, say so. Write the code anyway, but stop
before running it. Do not fetch source or install packages to fill the gap.

For a version mismatch or missing module, tell the user what it affects:

- If version 0.2.0 or newer has the core modules nvalchemi.data,
  nvalchemi.dynamics, and nvalchemi.models, proceed with the 0.3.0 patterns.
  Flag each 0.3.0-only feature used. Check its API in the installed source
  before execution. If it is absent, write the code but do not run it.
- If a core module is missing or the version is below the 0.2.0 floor, stop
  execution and ask before any upgrade. Do not install or upgrade on your own.

The installed wheel ships no docs or examples. Read the installed package
source and the linked docs and example pages only. Each note names the
source module. Locate its file with Python's inspect.getfile, then read it in
site-packages. Links are for reading, never for downloading code to run.
If the network is unavailable, use installed source.

### Run on the GPU

Pick the device once: use `cuda` when `torch.cuda.is_available()`, otherwise
`cpu`. Put the model, batch or dataset, and optimizer on that same device.
For dynamics, optimizer state comes from the batch. Pass
`device_type=device.type` to the engine. Given a model, structures, and the
agreed timestep and step budget:

```python
import torch
from nvalchemi.data import Batch
from nvalchemi.dynamics import FIRE2, FusedStage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
batch = Batch.from_data_list(data_list).to(device)
opt = FIRE2(model=model, dt=dt, n_steps=step_budget, device_type=device.type)
dyn = FusedStage(sub_stages=[(0, opt)], device_type=device.type)
for hook in model.make_neighbor_hooks():
    dyn.register_hook(hook)
```

Complete the pre-flight check below before `dyn.run(batch)`. For training,
set `Dataset(..., device=device)` and `TrainingStrategy(..., devices=[device])`.
The strategy builds optimizers from the placed model (see the training note).

Keep the step loop on the GPU. Avoid `.item()`, `.cpu()`, `.tolist()`,
`float(t)`, `print(t)`, `bool(t)`, a Python `if` on a tensor (including
`if tensor.any()`), and per-step host reads of batch.status. These wait for GPU
work before Python can continue. Host reads are fine at setup, at a
checkpoint, in a check every N steps (N > 1), and in the final report.
Prefer toolkit `run()` loops and hooks with a class-level `frequency` over a
hand-written step loop. Keep masks and reductions as device tensors.

A small CPU check is fine. The delivered code must still select `cuda` when a
GPU exists. Do not hard-code `device="cpu"` to make the check pass. This holds
for scratch test scripts too. Pass `device_type=device.type` there as well.

Build `AtomicData` only at setup, to validate input. In loops, work on
`Batch` tensors (see the data note, "Keep hot paths on `Batch`").

### Batched dynamics pre-flight check

Before every `run()`, confirm both checks in one line. Do not run until both
are true. Details are in
[dynamics-optimization.md](references/dynamics-optimization.md).

1. **One batch, one engine or `FusedStage`.** Build one `Batch` with
   `Batch.from_data_list(...)` for all independent systems. Do not run each
   system in a Python loop. Use a `FusedStage` to retire each system on its
   own when it converges.
2. **Neighbor hooks registered.** Register every hook from
   `model.make_neighbor_hooks()` on the engine or fused stage before the
   first step. Models without a neighbor list return no hooks.

```python
from nvalchemi.data import Batch
from nvalchemi.dynamics import FIRE2, ConvergenceHook, FusedStage

batch = Batch.from_data_list(data_list).to(device)  # many systems, one batch
opt = FIRE2(model=model, dt=0.1, n_steps=100, device_type=device.type,
            convergence_hook=ConvergenceHook.from_fmax(0.05))
dyn = FusedStage(sub_stages=[(0, opt)], device_type=device.type)
for hook in model.make_neighbor_hooks():      # BEFORE_COMPUTE hooks
    dyn.register_hook(hook)
with dyn:
    batch = dyn.run(batch)
```

### Route to a note

Match the task to the routing table. If it spans notes, read each note named
in the matching rows before you write code.

| Task | Note |
| --- | --- |
| Build, batch, convert, or inspect atomic systems, field names, ASE input | [data.md](references/data.md) |
| Write, read, or stream Zarr stores, DataLoader throughput | [data.md](references/data.md) |
| Load a pretrained model (MACE, AIMNet2, UMA, LJ, DFT-D3, Ewald, PME) | [models.md](references/models.md) |
| Wrap a custom PyTorch model, combine models, outputs and embeddings | [models.md](references/models.md) |
| Relax structures, run MD (NVE, NVT, NPT), chain stages, scans | [dynamics-optimization.md](references/dynamics-optimization.md) |
| Built-in hooks: freezing, safety, logging, snapshots, convergence, timing | [dynamics-optimization.md](references/dynamics-optimization.md) |
| Reaction paths and NEB, restraints and enhanced sampling | [dynamics-optimization.md](references/dynamics-optimization.md) |
| Train from scratch, losses, checkpoints, resume, DDP | [training.md](references/training.md) |
| Fine-tune a pretrained model, distill a teacher into a student | [training.md](references/training.md) |
| Sample structures with a generative model, generate then relax | [generative-workflows.md](references/generative-workflows.md) |
| Write a new hook, integrator, optimizer, loss, bias, or reporter | [composition-and-extension.md](references/composition-and-extension.md) |
| Domain decomposition, dynamics stages on different GPUs, dashboards | [composition-and-extension.md](references/composition-and-extension.md) |
| Custom NEB force equation or spring policy | [dynamics-optimization.md](references/dynamics-optimization.md) |
| Save and restore dynamics or sampling state, Hessians | [dynamics-optimization.md](references/dynamics-optimization.md), [models.md](references/models.md) |
| Crystal structure prediction (the nvalchemi.csp package) | [generative-workflows.md](references/generative-workflows.md) |

### Reuse first

1. **Reuse.** Search the shipped components before you write anything. The
   notes list them per area. Most needs (freezing atoms, NaN checks, drift
   warnings, logging, snapshots, step budgets, restraints) already exist.
2. **Subclass for a small change.** If a shipped class is close, subclass it
   and override the one method that differs.
3. **Write new code only for missing behavior.** Follow the protocol the toolkit
   defines (for example the `Hook` protocol or `BaseDynamics`), so it composes
   with the rest.

Say which path you took and why, in one line, when you hand back the code.

### Workspace rules

- Keep scratch files under the working folder you were given, for example in
  a scratch subfolder. Do not write elsewhere unless the user asks.
- Do not delete files or folders with `rm -rf` or `rm -r`, even for caches or
  temporary folders. Leave `__pycache__`, Zarr scratch stores, and temporary
  folders in place. If a single file must go, name it with `rm -f <file>` and
  say why. Ask first if the user has not granted delete permission.
- Read only installed package source and linked docs and example pages.
  Do not download source files, scripts, or wheels with `curl`, `wget`, or
  `git`. Do not download code to run by any other means.
- Do not print environment variables. Do not run `env` or `printenv`, or
  dump os.environ. These can expose secrets in tool output.
- Ask before you install, upgrade, or remove packages.

### Report the result

In the final message, name any script you wrote and the toolkit components
you used. Report measured results with units only for work you ran. If you
did not execute the work, say "Not executed" and why. Do not invent numbers.
If the run failed or did not converge, say so and give the measured values.
Do not present a capped or failed run as a converged result.

## Limitations

- The notes track release 0.3.0. Older installs miss generative sampling,
  reaction paths, enhanced sampling, distillation, and other 0.3.0 APIs.
- Multi-GPU and multi-node runs need a launcher (`torchrun` or Slurm). Write
  the script and the launch command. Do not start multi-GPU jobs unasked.
- Model backends download checkpoints on first use. Offline machines need the
  checkpoint file in advance.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `KeyError: 'neighbor_matrix' required` or `'neighbor_list'` | No neighbor-list hook | Register `model.make_neighbor_hooks()` |
| `AttributeError: ... no attribute 'masses'` | Field is `atomic_masses` | Use `batch.atomic_masses` |
| `InvalidCxxCompiler` or Inductor build error on CPU | `NeighborListHook` is compiled, no C++ toolchain | Run eagerly, see dynamics note |
| `AttributeError: 'Batch' has no attribute 'status'` | Plain dynamics does not create `status` | Add a `status` column, see dynamics note |
| `FileExistsError: Zarr store already exists` | `write()` on an existing store | Use `append()` or a new path |
| `ValueError: ... pattern ... did not match any parameter` | Wrong fine-tuning glob | Print `named_parameters()` and fix the glob |
| `ValueError: ... has no trainable parameters` with MACE | `MACEWrapper` freezes weights by default since 0.3.0 | Load with `train=True`, see the training note |
