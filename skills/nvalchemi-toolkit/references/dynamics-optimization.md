# Dynamics and optimization

This note covers relaxation, molecular dynamics, chained stages, built-in
hooks, reaction paths (NEB), and biased sampling. Writing a new hook,
integrator, or bias is in
`references/composition-and-extension.md`.

## Where to read more

| Topic | Docs page | Example | Installed source |
| --- | --- | --- | --- |
| Execution loop, `FusedStage`, compile, pipelines | [dynamics.md][doc-dyn] | [01_multistage_pipeline.py][ex-multi] | `nvalchemi/dynamics/base.py` |
| Optimizers and integrators | [dynamics_simulations.md][doc-sims] | [02_geometry_optimization.py][ex-opt], [05_nvt_langevin.py][ex-nvt], [04_nve_energy_conservation.py][ex-nve] | `nvalchemi/dynamics/optimizers/`, `nvalchemi/dynamics/integrators/` |
| NPT | [dynamics_simulations.md][doc-sims] | [03_npt_barostat_validation.py][ex-npt] | `nvalchemi/dynamics/integrators/npt.py` |
| Hooks and convergence | [hooks.md][doc-hooks] | [05_safety_and_monitoring.py][ex-safety], [03_custom_convergence.py][ex-conv] | `nvalchemi/dynamics/hooks/`, `nvalchemi/hooks/` |
| Freezing atoms, ASE input | [hooks.md][doc-hooks] | [03_ase_integration.py][ex-ase] | `nvalchemi/dynamics/hooks/freeze.py` |
| Trajectory sinks | [dynamics_sinks.md][doc-sinks] | [02_trajectory_zarr_io.py][ex-zarr] | `nvalchemi/dynamics/sinks.py` |
| Inflight batching | [dynamics.md][doc-dyn] | [04_inflight_batching.py][ex-inflight] | `nvalchemi/dynamics/sampler.py` |
| Reaction paths, NEB | [dynamics_mep.md][doc-mep] | [11_batched_neb.py][ex-neb] | `nvalchemi/dynamics/mep/` |
| Restraints, umbrella, metadynamics, replica exchange | [enhanced_sampling.md][doc-es] | [12_umbrella_sampling.py][ex-umb], [13_replica_exchange.py][ex-rex], [14_metadynamics.py][ex-metad] | `nvalchemi/enhanced_sampling/` |
| L-BFGS relaxation | [dynamics_simulations.md][doc-sims] (L-BFGS) | [06_geometry_optimization_lbfgs.py][ex-lbfgs] | `nvalchemi/dynamics/optimizers/lbfgs.py` |
| Checkpoint and restore of a run's state | [enhanced_sampling.md][doc-es] (Checkpoint and restore) | none | `nvalchemi/_checkpoint.py`, `nvalchemi/enhanced_sampling/_runner.py` |

API reference sources: [docs/modules/dynamics/][api-dyn].

## The two batched-run rules

**Rule 1: one batch, one pass.** Put independent systems in one `Batch` and
run one optimizer or integrator over it. A Python loop with one run per
system is slower and drops per-system bookkeeping that hooks and stages rely
on. A plain engine checks convergence for the batch as a whole: it stops when
every system meets the criterion, and keeps moving the ones that already did.
To retire each system as it converges, wrap the engine in a `FusedStage`
(`FusedStage(sub_stages=[(0, opt)])` or `relax + md`), which gives every
system a `status`.

**Rule 2: neighbor hooks before `run()`.** Any model with a cutoff needs its
neighbor list rebuilt before each force call. Register the hooks from
`model.make_neighbor_hooks()` (they target `BEFORE_COMPUTE`). For a
`FusedStage`, register them on the fused stage once; for a strategy
(`NEB`, `EnhancedSampling`) pass them as the strategy asks (see below).
Without them the model raises `KeyError: 'neighbor_matrix' required` or the
COO equivalent.

## Which component for which need

| Need | Component (classes ending in `Hook` from `nvalchemi.dynamics.hooks`, others from `nvalchemi.dynamics`, unless noted) |
| --- | --- |
| Fixed-cell relaxation | `FIRE2` (default) or `LBFGS` (fewer steps, costlier per step). `FIRE` is deprecated. |
| Relax positions and cell | `FIRE2VariableCell` or `LBFGSVariableCell` with `AlignCellHook`. The model must return `stress`. A `cell_force_scale` above 1 moves the cell less per step. |
| Relax a group of graphs as one unit | `LBFGS(by_group=True)` after `batch.set_group_layout(...)` |
| NVE | `NVE` |
| NVT | `NVTLangevin` (`temperature`, `friction`, `random_seed`) or `NVTNoseHoover` |
| NPT, NPH | `NPT`, `NPH`; model must return `stress` |
| Relax, then MD, per system | `relax + md` (a `FusedStage`) |
| Stop a system when converged | `ConvergenceHook.from_fmax(threshold)` or `ConvergenceHook(criteria=[...])` (from `nvalchemi.dynamics`); per-system only inside a `FusedStage` |
| Cap steps per system in a stage | `n_steps=N` on a sub-stage of a `FusedStage` (a per-system budget) |
| Stream many structures through a fixed GPU budget | `SizeAwareSampler` plus `refill_frequency` |
| Freeze a subset of atoms | `FreezeAtomsHook` (from `nvalchemi.dynamics.hooks`) |
| Stop on NaN or Inf | `NaNDetectorHook` |
| Clip huge forces | `MaxForceClampHook(max_force=...)` |
| Warn or stop on energy drift | `EnergyDriftMonitorHook(threshold=..., action="warn" or "raise")` |
| Per-system CSV rows | `LoggingHook(backend="csv", log_path=..., frequency=...)` |
| Save frames | `SnapshotHook(sink=ZarrData(path) or HostMemory(capacity), frequency=...)` |
| Save the final converged frame | `ConvergedSnapshotHook(sink=...)` |
| Wrap atoms back into the cell | `WrapPeriodicHook` (from `nvalchemi.hooks`); not with L-BFGS |
| Stability metrics over a run | `StabilityMonitor` (`nvalchemi.dynamics.hooks`) |
| Swap replicas between temperatures | `ReplicaExchange` with `ThermodynamicState` in enhanced sampling |
| Swap per-system state with your own acceptance rule | `PairSwapHook(accept_fn=..., params_fn=..., slot_field=..., n_slots=...)`. The engine must implement `apply_per_system_params` (`NVTLangevin`, `NVTNoseHoover` do) |
| Resume an enhanced-sampling run exactly | `sampling.checkpoint(path)` at an epoch boundary, then `restore(path)` on a rebuilt `EnhancedSampling`. Model weights are not saved. |
| Save engine, hook, and batch state for your own workflow | `save_checkpoint(path, {"md": engine, ...}, batch=batch)` and `load_checkpoint` from `nvalchemi._checkpoint`. Any object with `state_dict`/`load_state_dict` works, including `BaseDynamics` |
| Refresh forces when a graph enters a stage | `reprime_on_entry=True` on that stage's dynamics, not on the `FusedStage`. See [dynamics.md][doc-dyn]. |
| Find slow stages; NVTX ranges | `StageTimingHook`; full PyTorch traces: `TorchProfilerHook` |
| Different stages on different GPUs | `stage_a \| stage_b` builds a `DistributedPipeline`; see `references/composition-and-extension.md` |
| Thermal starting velocities | `initialize_velocities(v, masses, T, batch_idx_int32, ...)` |
| Minimum-energy path | `nvalchemi.dynamics.mep.NEB` |
| Different NEB force equation or spring policy | `method=` (`TorchNEBMethod` callable or Warp `NEBMethod`), `spring=` (`SpringConfig`); see [dynamics_mep.md][doc-mep] |
| Restraint, umbrella window, wall, metadynamics | `nvalchemi.enhanced_sampling` biases run by `EnhancedSampling` |
| Behavior none of these cover | A custom hook, see `references/composition-and-extension.md` |

Units: positions in Å, energies in eV, `dt` in fs, `temperature` in K,
`friction` in 1/fs, pressure in eV/Å³. Velocities are stored in sqrt(eV/amu).

## How the pieces connect

```text
step():  BEFORE_STEP
         BEFORE_PRE_UPDATE  -> pre_update()  -> AFTER_PRE_UPDATE
         BEFORE_COMPUTE (neighbor list) -> model -> AFTER_COMPUTE (freeze, clamp, NaN)
         BEFORE_POST_UPDATE -> post_update() -> AFTER_POST_UPDATE (wrap)
         AFTER_STEP (logging, snapshots, drift)  -> convergence -> ON_CONVERGE
run():   repeats step() until n_steps or every system converged
```

- Each hook declares `stage` and `frequency`; it fires when
  `step_count % frequency == 0`. Hooks at the same stage fire in
  registration order.
- `relax + md` builds a `FusedStage`. Each system carries `status`. Stage `i`
  owns systems with `status == i`. When a system converges in stage `i`, or
  uses up that stage's `n_steps`, it moves to `i + 1`. One model call per step
  covers all stages.
- A plain engine has `exit_status = 1`. A `FusedStage` of `k` stages has
  `exit_status = k`. A system at `status >= exit_status` is frozen.
- Strategies (`NEB`, `EnhancedSampling`) build the engine for you and add
  their own hooks; you pass extra hooks to them.

## Relax then MD for many systems

```python
relax = FIRE2(model=model, dt=0.1, n_steps=200,
              convergence_hook=ConvergenceHook.from_fmax(0.05))
md = NVTLangevin(model=model, dt=1.0, temperature=300.0, friction=0.01,
                 random_seed=0, n_steps=500)      # per-system MD budget
fused = relax + md
for hook in model.make_neighbor_hooks():
    fused.register_hook(hook)
with fused:
    batch = fused.run(batch, n_steps=2000)        # overall cap
```

The `n_steps` on the MD stage is a per-system budget: each system gets that
many MD steps after it leaves relaxation, then graduates. Always give the
last stage either a budget or a convergence criterion, and give `run()` an
overall cap, or a stage with no exit runs forever.

Check `batch.status` after the run. All systems at `exit_status` means each
finished both stages.

## Running without a C++ compiler

`NeighborListHook.__call__` is decorated with `torch.compile`. On a machine
without a working C++ toolchain, the first dynamics step fails with
`InductorError: InvalidCxxCompiler: No working C++ compiler found`. Do not
patch the hook. Run eagerly, and say so in the script:

```python
import torch

# NeighborListHook is torch.compile'd; this machine has no C++ compiler.
torch.compiler.set_stance("force_eager")
```

Put this before the first dynamics step. Setting the environment variable
`TORCHDYNAMO_DISABLE=1` before Python starts has the same effect. Results are
the same; only speed changes. Do not call `fused.compile(...)` in that mode.

## Built-in hooks: reuse before you write

Most per-step needs have a shipped hook. Check this list before you write a
class:

- `FreezeAtomsHook()` freezes atoms with
  `atom_categories == AtomCategory.SPECIAL.value`, or the atoms in a boolean
  node mask named by `mask_key`. It restores positions and zeros their
  forces and velocities. Register the same instance on every stage that must
  keep them frozen.
- `NaNDetectorHook()` raises `RuntimeError` with the bad graph indices.
- `EnergyDriftMonitorHook(threshold=..., metric="per_atom_per_step",
  action="warn")` tracks total-energy drift. `action="raise"` stops the run.
  It does not support status-filtered batches yet.
- `LoggingHook` writes per-system rows; `by_group=True` writes per group.
- `ConvergenceHook` takes `criteria` with `key`, `threshold`, `reduce_op`,
  `reduce_dims`, or a `custom_op` that returns a per-graph boolean.

A hook that runs every N steps declares `frequency = N` as a class
attribute, and it must skip inactive graphs (read `ctx.active_graph_mask`).
The full checklist for new hooks is in
`references/composition-and-extension.md`.

## Stopping individual systems

A plain engine (not a `FusedStage`) does not create a `status` column. A hook
that writes `batch.status` on a plain engine needs one first:

```python
batch["status"] = torch.zeros(batch.num_graphs, 1, dtype=torch.long,
                              device=batch.device)
```

Then set a system's status to `ctx.workflow.exit_status` to stop it. The
engine treats graphs at `status >= exit_status` as no-ops for the rest of the
run. A plain engine still runs to its `n_steps` (it exits early only through
its `ConvergenceHook`), so keep `n_steps` bounded. Inside a `FusedStage`,
`ctx.workflow` is the stage that owns the hook, so register such a hook on
the fused stage to use the fused `exit_status`.

Prefer `ConvergenceHook` for a force threshold. For a missing stop condition,
keep the per-graph decision as a GPU mask. This hook takes an already computed
per-graph Boolean from `stop_mask(ctx)`; that callable must use only tensor
operations and return shape `[B]` on `ctx.batch.device`:

```python
class MaskStopHook:
    stage = DynamicsStage.AFTER_STEP
    frequency = 25

    def __init__(self, stop_mask):
        self.stop_mask = stop_mask

    def __call__(self, ctx, stage):
        stop = self.stop_mask(ctx)
        if ctx.active_graph_mask is not None:    # Python metadata, not a tensor test
            stop = stop & ctx.active_graph_mask
        status = ctx.batch.status.view(-1)[:ctx.batch.num_graphs]
        done = torch.full_like(status, ctx.workflow.exit_status)
        status.copy_(torch.where(stop, done, status))
```

Register it on the engine that owns the exit status. No `if stop.any()`,
`bool(stop.all())`, or per-step conversion of `batch.status` to host values.
For observations, retain detached tensor snapshots and convert after the run,
or use `LoggingHook` / `SnapshotHook` with `frequency > 1`. Only report on the
host at those scheduled checks, not in every step. The complete reduction and
history pattern is in `references/composition-and-extension.md`.

## Chaining a helper stage into a production stage

Some workflows run a cheap helper (an analytic potential, a pre-relaxation, an
interpolation fix-up) before the physical model. Carry only the positions
(and cell, if it changed) into a fresh object for the production run:

```python
prod = template.clone()                    # the batch you built for production
prod.positions.copy_(helper_out.positions)
prod.forces = torch.zeros_like(prod.positions)
prod.energy = prod.positions.new_zeros(prod.num_graphs, 1)
prod.velocities = torch.zeros_like(prod.positions)
```

Do not reuse the helper's output batch directly. Its `status`, neighbor data,
helper-specific fields, and optimizer state would leak into the production
run, and a fresh engine or strategy instance gives the production stage its
own optimizer history.

## Inflight replacement

`refill_check()` runs every `refill_frequency` steps. It writes
finished graphs to sinks, selects the remaining graphs, requests replacements
within the sampler's count, atom, and edge limits, then appends them. Supplied
bookkeeping survives; missing fields use registered defaults such as `status=0`
and, when registered, `fmax=inf`. Reset status and stage counters before reusing
completed inputs. Refill returns a new batch after graduation, the original
batch if none graduate, or `None` once the sampler and active batch are empty.

Use `by_group=True` on both the sampler and engine for whole-group graduation
and replacement. `max_batch_size` counts groups, and atom/edge limits include
all graphs in each group. Grouped refill is supported on a single device;
`DistributedPipeline` rejects grouped stages.

`SizeAwareSampler(field_defaults=...)` fills missing fields before packing.
Map each name to `(level, default)`, for example
`{"fixed_mask": ("node", False)}`. Levels are node, edge, or system. A default
may be a scalar or a callable accepting `(num_rows, device)` and returning a
tensor. Supplied fields are preserved. Loading adds defaults and IDs to dataset
items in place; assembly copies the data for simulation. Register defaults
before loading samples.

## Reaction paths (NEB)

- Build images with `interpolate_paths(initial, final, num_images=...)`. Each
  path is one group in the batch; atoms must match by index.
  `remove_translation_and_rotation=True` aligns the endpoints first.
- If linear interpolation puts atoms on top of each other, refine with IDPP:
  `prepare_idpp_targets(paths.clone())`, run
  `NEB(model=IDPPModel(), ...)`, then carry only the positions into a new
  batch for the real model, as in the chaining pattern above.
- Run `NEB(model=..., spring=..., climbing=..., fmax=..., n_steps=...)`.
  `NEB` builds the neighbor hooks from `model.make_neighbor_hooks()` unless
  you pass `neighbor_hooks=`.
- If the saddle point matters, use climbing-image NEB. Set
  `ClimbingImageConfig(mode="after_regular", regular_fmax=...)`: let the band
  relax to a modest force before climbing, rather than enabling climbing at
  the start or while forces are still large. An early switch can oscillate
  instead of converging. Choose the switching threshold above the final
  `fmax`, then check that the regular band actually reaches it.
- Run the regular and climbing stages in one `NEB` call with
  `mode="after_regular"`, so the switch follows the band's force.
- Bound each stage with `ClimbingImageConfig.max_regular_steps` and
  `max_climbing_steps`. `NEB.n_steps` caps the whole run, not each stage.
  Given agreed thresholds and budgets:

  ```python
  from nvalchemi.dynamics.mep import ClimbingImageConfig, NEB

  neb = NEB(model=model, fmax=final_fmax,
            n_steps=regular_budget + climbing_budget,
            climbing=ClimbingImageConfig(
                mode="after_regular", regular_fmax=switch_fmax,
                max_regular_steps=regular_budget,
                max_climbing_steps=climbing_budget))
  paths = neb.run(paths)
  ```

  Pick a `regular_fmax` that the regular stage can reach within
  `max_regular_steps`. If the regular stage reaches its step cap above
  `regular_fmax`, climbing starts from a poorly relaxed band and often ends
  unconverged. Report the run as unconverged with its final fmax, and offer
  a larger regular budget or a looser switch threshold for the next run. Do
  not present that barrier as the answer.
- `interpolate_paths` drops `forces`, `energy`, and `velocities`. Reattach
  them before any optimizer run.
- Read `batch.status` after the run and the diagnostics CSV
  (`diagnostics_log_path=`): one row per path with `fmax`, `energy_barrier`,
  and `path_length`. Use `diagnostics_frequency > 1` for periodic logging.
  With step budgets, "finished" does not mean converged; check `fmax`.
- Report the forward barrier (highest interior energy minus the initial
  endpoint energy), the final force in eV/Å, and whether each path converged.
  CSV `energy_barrier` uses the lower endpoint, so it is not necessarily the
  forward barrier. Do not report an unconverged estimate as a final barrier.

### Stream prepared paths on one device

Prepare interpolation, IDPP, and constraints first. Each dataset item returns
`(path_batch, metadata)` containing one complete group and optimizer inputs.
Its `get_metadata(index)` reports total atoms and edges across the path.
Use `SizeAwareSampler(prepared_dataset, by_group=True, ...)`, then
`NEB(...).run(sampler=sampler, sinks=[results])`, where `results` can be
`HostMemory(capacity=...)`. `max_batch_size` counts paths, and budgets include
every image. `NEB.refill_frequency` defaults to 10 steps.

`run()` takes either a batch or a sampler. Streaming builds a fresh engine and
preserves resident optimizer history and fixed climbing choices across refills.
It returns `None` on completion or the active batch at the global `n_steps`
limit. Use `build_engine(sampler=..., sinks=...)` for manual stepping. Sink
output is in finish order; per-image `group_id` identifies the dataset path.
Clear incoming status and stage counters before reusing finished paths.

Fix atoms with `batch.neb_fixed_node_mask`, a Boolean node field of shape
`(batch.num_nodes,)` on the batch device, and include it in incoming paths.
NEB creates the mask if absent, adds fixed endpoints in place, and registers
`FreezeAtomsHook` to enforce it even with relaxed endpoints.
`fixed_atom_indices` adds constraints for static runs but is rejected for
sampler and pipeline workflows.

### Custom path hooks

For manual NEB, use a group-aware optimizer and register path hooks on the
`FusedStage` in this order: `PathEnergyStatsHook`, optional
`ClimbingImageSelectionHook`, `NEBForceHook`,
`FreezeAtomsHook(mask_key="neb_fixed_node_mask")`, optional
`PathDiagnosticsHook`, then optional `LoggingHook(by_group=True, ...)`.
Share the same energy-stats hook instance and scope stage-specific hooks with
`status_code`. Sub-stage `AFTER_COMPUTE` hooks run before fused-stage hooks,
so path hooks on a sub-stage would read stale fused energy statistics.

At admission, `ctx.previous_graph_indices` lets path hooks preserve residents'
fixed climbing choices, admission-time springs, and constraints. Arrivals start
fresh; a missing mapping resets all paths. Diagnostics reset at admission and
update at `AFTER_COMPUTE`; `get_diagnostics()` reads the cache.
A custom `SpringConfig.resolve(context)` returns one constant per link;
`refresh` selects `ON_ADMISSION` or `AFTER_COMPUTE`. See
[dynamics_mep.md][doc-mep] for the force-method protocols.

## Restraints and biased sampling

Use `nvalchemi.enhanced_sampling`, not `BiasedPotentialHook`. The hook is
deprecated: its bias adds no stress, so an NPT or NPH barostat does not see
it. `ConservativeBias` derives forces and stress from one energy function.

```python
from nvalchemi.enhanced_sampling import (
    EnhancedSampling, HarmonicUmbrellaBias, pair_distance,
)

bias = HarmonicUmbrellaBias(cv=lambda b: pair_distance(b, torch.tensor([i, j])),
                            centers=r0, stiffness=k, name="restraint")
sampling = EnhancedSampling(
    model=model, engine=NVTLangevin,               # the class, not an instance
    engine_kwargs=dict(dt=0.5, temperature=T, friction=0.05),
    biases={"restraint": bias},
    extra_hooks=list(model.make_neighbor_hooks()),
)
batch = sampling.run(batch, n_steps=n)
```

- `engine` takes any integrator class; pass its arguments in
  `engine_kwargs`. Barostat runs (`NPT`, `NPH`) see the bias through its
  stress.
- `pair_distance` takes atom indices local to each graph and returns `[B, 1]`.
- Periodic biased runs need a `stress` buffer on the batch
  (`stress=torch.zeros(1, 3, 3)` per `AtomicData`) and a model with stress
  active (`model.set_config("active_outputs", {"energy", "forces", "stress"})`).
- Other shipped biases: `UpperWall`, `LowerWall`, `FlatBottomRestraint`,
  `WellTemperedMetaDynamicsBias`, `RMSDMetaDynamicsBias`,
  `AdaptiveBiasingForce`. A custom bias subclasses `ConservativeBias`.

## Common mistakes

- One run per system in a Python loop (rule 1).
- No neighbor hooks, or hooks created but never registered (rule 2).
- `batch.masses` instead of `batch.atomic_masses`.
- Patching toolkit source to get past the compiler error instead of
  running eagerly.
- Writing a new hook for freezing, NaN checks, drift, logging, or snapshots.
- A last stage with neither a budget nor a convergence criterion.
- `BiasedPotentialHook` for new work.
- NPT, NPH, or variable-cell runs with a model that does not return stress.
- `FIRE` or `FIREVariableCell` for new work. They are deprecated. Use `FIRE2`
  or `LBFGS`. The one exception is `DomainParallel` (see the composition
  note).
- `FusedStage.register_fused_hook()`. It is deprecated. Use
  `register_hook()`.
- Passing `reprime_on_entry` to `FusedStage` or `DomainParallel`. Both raise
  `TypeError`. Set it on the wrapped dynamics. It is a bool, not a set of
  status codes.
- `save_checkpoint` drops batch fields the Zarr writer does not know. Name
  custom per-graph fields in `batch_fields=`.
- `WrapPeriodicHook` with L-BFGS; it corrupts the curvature history.
- A long trajectory as a first test. Run tens of steps first.

## How to check the result

- Relaxation: per-system max force norm
  (`forces.norm(dim=-1)` reduced per graph) below the threshold. In a
  `FusedStage`, also check `batch.status` at `exit_status`.
- NVE: total energy drift per atom over the run is small compared with
  `kT`; `EnergyDriftMonitorHook` reports it.
- NVT: mean kinetic temperature
  `2 * KE / (3 * N * KB_EV)` near the target after equilibration
  (`KB_EV` from `nvalchemi.dynamics`).
- NPT: volume settles and mean pressure is near the target.
- NEB: the climbing image's `fmax` below threshold, and a smooth energy
  profile along the path.
- Bias: the collective variable's histogram centers on the restraint target.

[doc-dyn]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/dynamics.md
[doc-sims]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/dynamics_simulations.md
[doc-hooks]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/hooks.md
[doc-sinks]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/dynamics_sinks.md
[doc-mep]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/dynamics_mep.md
[doc-es]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/enhanced_sampling.md
[api-dyn]: https://github.com/NVIDIA/nvalchemi-toolkit/tree/v0.3.0/docs/modules/dynamics
[ex-opt]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/basic/02_geometry_optimization.py
[ex-ase]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/basic/03_ase_integration.py
[ex-nve]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/basic/04_nve_energy_conservation.py
[ex-nvt]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/basic/05_nvt_langevin.py
[ex-multi]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/intermediate/01_multistage_pipeline.py
[ex-zarr]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/intermediate/02_trajectory_zarr_io.py
[ex-npt]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/intermediate/03_npt_barostat_validation.py
[ex-inflight]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/intermediate/04_inflight_batching.py
[ex-safety]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/intermediate/05_safety_and_monitoring.py
[ex-conv]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/advanced/03_custom_convergence.py
[ex-neb]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/advanced/11_batched_neb.py
[ex-umb]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/advanced/12_umbrella_sampling.py
[ex-rex]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/advanced/13_replica_exchange.py
[ex-metad]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/advanced/14_metadynamics.py
[ex-lbfgs]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/basic/06_geometry_optimization_lbfgs.py
