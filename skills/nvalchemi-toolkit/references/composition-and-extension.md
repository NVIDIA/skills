# Composition and extension

This note covers new code that plugs into the toolkit: hooks, integrators and
optimizers, loss terms, biases, reporters, and domain decomposition across
GPUs. Use it only after the reuse check in `SKILL.md`: most
needs already have a shipped component, listed in the other notes.

## Where to read more

| Topic | Docs page | Example | Installed source |
| --- | --- | --- | --- |
| Hook protocol, contexts, stages | [hooks.md][doc-hooks] | [02_custom_hook.py][ex-hook] | `nvalchemi/hooks/_protocol.py`, `_context.py`, `nvalchemi/dynamics/base.py` (`DynamicsStage`) |
| Custom integrator or optimizer | [dynamics_simulations.md][doc-sims] (writing your own dynamics) | [05_custom_integrator.py][ex-integ] | `nvalchemi/dynamics/base.py`, `nvalchemi/dynamics/integrators/nve.py` |
| Custom convergence criteria | [hooks.md][doc-hooks] | [03_custom_convergence.py][ex-conv] | `ConvergenceHook` in `nvalchemi/dynamics/base.py` |
| Custom loss term | [losses.md][doc-loss] (writing your own loss) | none | `nvalchemi/training/losses/composition.py`, `terms.py` |
| Custom bias | [enhanced_sampling.md][doc-es] (writing your own bias) | [15_rmsd_metadynamics.py][ex-rmsd] | `nvalchemi/enhanced_sampling/_bias.py` |
| Reporting and dashboards | [reporting.md][doc-rep] | [07_rich_training_reporting.py][ex-rich] | `nvalchemi/hooks/reporting/` |
| Domain decomposition | [distributed.md][doc-dd], [distributed_byo.md][doc-byo] | [03_mace_nvt_distributed.py][ex-dd], [04_byo_pytorch_mpnn.py][ex-byo] | `nvalchemi/distributed/` |
| Multi-rank stage pipelines | [dynamics.md][doc-dyn] (distributed pipelines) | [01_distributed_pipeline.py][ex-pipe] | `DistributedPipeline` in `nvalchemi/dynamics/base.py` |

API reference sources: [docs/modules/hooks.rst][api-hooks],
[docs/modules/dynamics/][api-dyn], [docs/modules/distributed.rst][api-dist].

## Custom hooks

A hook is any object with `stage`, `frequency`, and
`__call__(ctx, stage)`. No base class is needed. Dynamics hooks receive a
`DynamicsContext` with `batch`, `step_count`, `model`, `active_graph_mask`,
`converged_mask`, and `workflow` (the engine that owns the hook).

### Checklist

Go through every item. Hooks that skipped one of these were the most common
failure in tested scripts.

1. **Class attributes.** Declare `stage` and `frequency` on the class. For
   "every N steps", set `frequency = N` and let the engine gate the call.
   Do not use `frequency = 1` with your own step counter, and do not count
   steps inside `__call__`.
2. **Vectorized per-graph math.** Reduce per graph with
   `index_add_` or `scatter_reduce_` over `batch.batch_idx`. No Python loop
   over graphs, and no `.item()` or `.tolist()` per graph in the hot path.
3. **Respect the active mask.** Gate every write on `ctx.active_graph_mask`
   (it can be `None` when the batch has no `status`). Graphs that already
   finished must not be touched again.
4. **Stopping a graph.** Write `ctx.workflow.exit_status` into
   `batch.status` for the graphs to stop, with `torch.where`, not a loop.
   A plain engine has no `status` column; add one before `run()`. Register
   the hook on the `FusedStage` (not a sub-stage) when it should end the
   whole pipeline for that graph.
5. **Pick the right stage.** Model outputs change at `AFTER_COMPUTE`;
   observers and stoppers go at `AFTER_STEP`. One-time per-system setup goes
   at `ON_ADMISSION`, which fires once when a batch is admitted, before
   forces are primed. Work on systems that just finished goes at
   `ON_GRADUATE` (read `ctx.graduated_mask`).
6. **Record cheaply.** Store per-step results as detached tensors
   (`.detach().clone()`), and convert to Python or files after the run.

### Preserve state when batch membership changes

`BaseDynamics` and `FusedStage` pass a read-only
`ctx.previous_graph_indices` at `ON_ADMISSION`: each current graph's index at
its last admission, or `-1` for an arrival, regardless of status. This mapping
composes membership changes since that admission. First admissions, new runs,
and rebuilds after complete removal mark every graph new. Fused and sub-stage
hooks receive the same full-batch mapping. Combine
`previous_graph_indices < 0` with `active_graph_mask` to select a sub-stage's
arrivals. Other stages receive `None`. Path hooks can recover previous paths
using this mapping and their cached layout.

### Template

This hook records a per-graph quantity every N steps and stops graphs that
cross a threshold. It was run against `main`:

```python
class PerGraphStopHook:
    stage = DynamicsStage.AFTER_STEP
    frequency = 25                                       # every 25 steps

    def __init__(self, threshold):
        self.threshold = threshold
        self.history = []                                # (step, [B] tensor)

    def __call__(self, ctx, stage):
        b = ctx.batch
        idx = b.batch_idx.long()
        per_atom = 0.5 * b.atomic_masses * (b.velocities ** 2).sum(-1)    # [V]
        value = torch.zeros(b.num_graphs, dtype=per_atom.dtype,
                            device=per_atom.device).index_add_(0, idx, per_atom)
        value = value / b.num_nodes_per_graph                              # [B]
        self.history.append((ctx.step_count, value.detach().clone()))
        active = ctx.active_graph_mask
        if active is None:
            active = torch.ones_like(value, dtype=torch.bool)
        stop = active & (value > self.threshold)
        status = b.status.view(-1)[: b.num_graphs]
        done = torch.full_like(status, ctx.workflow.exit_status)
        status.copy_(torch.where(stop, done, status))
```

Replace the quantity with your own per-graph reduction built the same way:
per-atom values, then `index_add_` (sums, means) or `scatter_reduce_`
(max, min) over `batch_idx`. Before `run()` on a plain engine:
`batch["status"] = torch.zeros(batch.num_graphs, 1, dtype=torch.long, device=batch.device)`.
Register with `dynamics.register_hook(hook)` or `hooks=[...]`.

Hooks that fire at several stages define `_runs_on_stage(stage) -> bool`.
Hooks that hold files implement `__enter__` and `__exit__`; the engine
enters and exits them in `run()`. Restartable training hooks implement
`state_dict` and `load_state_dict` (`CheckpointableHook`). A hook whose state
changes during the run (deposited hills, an adaptive parameter) follows
`StatefulHook` from `nvalchemi.hooks`: set `read_only = False` and add
`commit()`. Engines that evaluate one step twice use `read_only` to avoid a
double update. See [enhanced_sampling.md][doc-es] (Adaptive biases).
`save_checkpoint` in `nvalchemi/_checkpoint.py` saves any such hook.

## Custom integrators and optimizers

Subclass `BaseDynamics` and override only `pre_update(batch)` and
`post_update(batch)`. The base class runs the model, hooks, convergence, and
the step loop. Do not override `step`, `run`, or `compute`. This velocity
Verlet works alone and inside a `FusedStage`; it was run against `main`:

```python
class VelocityVerlet(BaseDynamics):
    __needs_keys__ = {"forces"}                          # model outputs it reads
    __provides_keys__ = {"positions", "velocities"}      # fields it writes

    def __init__(self, model, dt, **kwargs):
        super().__init__(model=model, **kwargs)          # forwards n_steps, hooks, ...
        self.dt = fs_to_internal_time(dt)                # fs -> kernel time unit

    def pre_update(self, batch):
        with torch.no_grad():
            m = batch.atomic_masses.unsqueeze(-1)
            batch.velocities.add_(0.5 * self.dt * batch.forces / m)
            batch.positions.add_(self.dt * batch.velocities)

    def post_update(self, batch):
        with torch.no_grad():
            m = batch.atomic_masses.unsqueeze(-1)
            batch.velocities.add_(0.5 * self.dt * batch.forces / m)
```

Imports: `from nvalchemi.dynamics.base import BaseDynamics` and
`from nvalchemi.dynamics._units import fs_to_internal_time`.

- Update tensors in place inside `torch.no_grad()`. The base step does not
  turn off autograd for you.
- Use `batch.atomic_masses`, not `masses`.
- Convert `dt` from fs with `fs_to_internal_time`, as the shipped
  integrators do, so velocities stay in sqrt(eV/amu) and match
  `initialize_velocities` and the thermostats.
- Inside a `FusedStage`, the base class masks your updates to the graphs
  your stage owns. You do not mask by hand.
- Per-system state (thermostat variables) goes in instance attributes or the
  `_state` mechanism the shipped integrators use; see `nve.py` and
  `nvt_langevin.py`.
- Under domain decomposition, per-atom updates are safe, but global
  reductions (total kinetic energy, temperature, FIRE dot products) need the
  `HookScope.GLOBAL` pattern in [distributed.md][doc-dd].

Check: NVE total energy (potential plus `0.5 * m * v**2`, summed per graph)
stays flat over a few hundred small steps, and `relax + VelocityVerlet(...)`
runs with all graphs reaching `exit_status`.

## Custom loss terms

Subclass `BaseLossFunction` (`nvalchemi.training`) and override
`compute_residual(pred, target, valid)`. Set `target_key` and
`prediction_key` so `ComposedLossFunction` can route tensors. Override
`mask` to drop invalid entries, `normalize` for per-atom scaling, and
`reduce` for graph-balanced means (`per_graph_sum`, `per_graph_mean` in
`nvalchemi/training/losses/reductions.py`). Zero invalid entries in
`compute_residual` with `torch.where(valid, ..., 0)`. Hooks that receive
`**kwargs` must accept them, because composition forwards batch metadata.

## Custom biases

Subclass `ConservativeBias` (`nvalchemi.enhanced_sampling`), call
`super().__init__(name=...)`, and implement `energy(batch) -> [B, 1]`.
Forces and stress come from autograd. Keep data-dependent Python branches
out of `energy()` if you set `compile_biases=True`; put validation in a
`forward()` override that calls `super().forward(data)`. Run it with
`EnhancedSampling(biases={name: bias}, ...)` as in
`references/dynamics-optimization.md`.

## Reporting and observability

- Live dashboards and TensorBoard: register one
  `ReportingOrchestrator([RichReporter(...), TensorBoardReporter(dir)],
  stages={...}, frequency=N)` as a hook (from `nvalchemi.hooks`). Reporters
  are owned by the orchestrator; do not register them directly.
- Dynamics per-system rows: `LoggingHook(backend="csv", ...)` from
  `nvalchemi.dynamics.hooks`. Do not reuse it as a training logger.
- Training: `stages={"AFTER_OPTIMIZER_STEP"}` for loss and learning rate.
- Under DDP, rendering and writing stay on rank 0, but a reporter with
  `rank_reduction="mean"` (for a global loss curve) runs a collective: every
  rank must reach it, with the same scalar keys. Do not skip nonzero ranks
  before the orchestrator. Give each rank its own `LoggingHook` path.
- Find slow stages with `StageTimingHook` (per-stage timing, NVTX ranges);
  use `TorchProfilerHook` for kernel-level traces. Both are in
  `nvalchemi.dynamics.hooks`. See [reporting.md][doc-rep], [hooks.md][doc-hooks].
- Extra metrics: `custom_scalars={"name": callable}`; keep the callable cheap.

## Dynamics stages on different GPUs

`relax + md` fuses stages on one GPU. `relax | md` (dynamics engines) builds a
`DistributedPipeline`: one stage per rank, converged systems sent to the next
rank. Every stage needs a `buffer_config=BufferConfig(...)` (from
`nvalchemi.dynamics.base`), identical across adjacent stages. A plain stage's
`n_steps` does not graduate systems in a pipeline; wrap fixed-length stages in
a `FusedStage`. This `|` differs from the generative `gen | stage` pipeline.
Details: [dynamics.md][doc-dyn], example [01_distributed_pipeline.py][ex-pipe].

## Domain decomposition across GPUs

- `DomainParallel(dynamics=integrator, config=DomainConfig(cutoff=...,
  skin=..., mesh=...))` splits one large system over ranks. Launch one process
  per GPU with `torchrun`.
- Choose the strategy in `DomainConfig.strategy`: halo (default) for models
  with a fixed cutoff that accept a padded view (MACE, LJ, Ewald, PME);
  graph partition for models that build their own graph inside `forward`
  (UMA).
- A custom model declares `distribution_spec()` returning a preset such as
  `SPEC_MPNN_HALO` (autograd-force message passing) or an `MLIPSpec` with
  adapters for opaque custom ops. Check it on one CUDA GPU with
  `trace_and_validate(model_factory, sample_batch, world_size=2)`, where the
  factory returns a fresh wrapper and the small batch is on `cuda`.
- Under halo decomposition a message-passing `forward` must rebind scatter
  results (`ae = ae.scatter_add_(...)`, use the returned tensor), drop
  neighbor indices `>= n_atoms` from padding, and return per-atom energies for
  all local atoms, ghosts included. Missing these gives wrong energies, not an
  error. Reference: [distributed_byo.md][doc-byo] and
  [04_byo_pytorch_mpnn.py][ex-byo].
- Neighbor-list hooks go on the inner integrator; trajectory and logging
  hooks go on the outer `DomainParallel` with `HookScope.RANK_ZERO` or
  `HookScope.GLOBAL`.
- `FIRE2` does not declare its global reductions yet. Use `FIRE` for
  relaxation under `DomainParallel`. `FIRE` is deprecated and warns on
  construction, but this is the one place it is still the right choice.
- Set `reprime_on_entry=True` on the wrapped dynamics, not on
  `DomainParallel`. The wrapper raises `TypeError`.
- Write the scripts and launch commands; do not start multi-GPU runs without
  the user's go-ahead.

## Common mistakes

- Writing a new hook where `FreezeAtomsHook`, `NaNDetectorHook`,
  `EnergyDriftMonitorHook`, `LoggingHook`, or `SnapshotHook` already fits.
- Per-instance `self.frequency` set from a constructor argument while the
  class attribute says something else, or `frequency = 1` with a manual
  step counter.
- Python loops over graphs with `.item()` per graph.
- Writing `status` for graphs that are no longer active.
- Overriding `step()` in a dynamics subclass.
- Integrators that read `masses` or skip `torch.no_grad()`.
- The deprecated `BiasedPotentialHook` for a new bias.

## How to check the result

- Hooks: run 2 to 3 graphs for a few hundred steps with `frequency = N`;
  the recorded step numbers are multiples of `N`, values are per graph, and
  stopped graphs keep their positions afterwards.
- Integrators: energy conservation under NVE, and the fused-stage run above.
- Losses: a hand-computed value on a two-graph batch matches the loss.
- Biases: finite-difference forces of `energy()` match the bias forces.
- Domain decomposition: `trace_and_validate` passes, and a short run matches a
  single-process run of the same system.

[doc-hooks]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/hooks.md
[doc-sims]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/dynamics_simulations.md
[doc-dyn]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/dynamics.md
[doc-loss]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/losses.md
[doc-es]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/enhanced_sampling.md
[doc-rep]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/reporting.md
[doc-dd]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/distributed.md
[doc-byo]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/distributed_byo.md
[api-hooks]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/modules/hooks.rst
[api-dyn]: https://github.com/NVIDIA/nvalchemi-toolkit/tree/v0.3.0/docs/modules/dynamics
[api-dist]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/modules/distributed.rst
[ex-hook]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/advanced/02_custom_hook.py
[ex-conv]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/advanced/03_custom_convergence.py
[ex-integ]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/advanced/05_custom_integrator.py
[ex-rmsd]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/advanced/15_rmsd_metadynamics.py
[ex-rich]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/intermediate/07_rich_training_reporting.py
[ex-pipe]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/distributed/01_distributed_pipeline.py
[ex-dd]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/distributed/03_mace_nvt_distributed.py
[ex-byo]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/distributed/04_byo_pytorch_mpnn.py
