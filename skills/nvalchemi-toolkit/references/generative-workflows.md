# Generative workflows

This note covers sampling structures from a generative model with
`AtomisticGenerator`, filtering them with hooks, and sending them straight
into relaxation or MD. `TrainingStrategy` accepts only `BaseModelMixin`
energy models, so it rejects a `GenerativeModelMixin` network: train the
generative network with your own PyTorch loop (or wrap it in a
`BaseModelMixin` adapter), then hand it to a sampler.

## Where to read more

| Topic | Docs page | Example | Installed source |
| --- | --- | --- | --- |
| Driver, output contract, hooks, pipelines | [generative.md][doc-gen] | [09_generative_samplers.py][ex-gen] | `nvalchemi/gen/generator.py`, `pipeline.py`, `stages.py` |
| Model-side mixin and config | [generative.md][doc-gen] | [09_generative_samplers.py][ex-gen] | `nvalchemi/models/gen/base.py`, `demo.py` |
| Crystal structure prediction (a generator subclass) | [csp.md][doc-csp] | none | `nvalchemi/csp/` |
| Relaxing generated structures | [dynamics_simulations.md][doc-sims] | [02_geometry_optimization.py][ex-opt] | `nvalchemi/dynamics/` |

API reference sources: [docs/modules/gen.rst][api-gen].

## Which component for which need

| Need | Component |
| --- | --- |
| Run any sampler (diffusion, flow, GAN, VAE, random search) | `AtomisticGenerator(generator_func=...)` from `nvalchemi.gen` |
| The sampler itself | A callable `f(inputs=None, *, num_samples=1, rng=None, **kw) -> Batch` |
| A sampler that owns a trained network | A small class with `__call__`, plus `device`, `required_inputs`, `outputs` attributes |
| Declare what a generative network needs and makes | `GenerativeModelMixin` with `GenerativeModelConfig` (`nvalchemi.models.gen`) |
| Test models with no checkpoint | `DemoGANModel`, `DemoDiffusionModel` |
| Turn raw inputs into conditioning | `condition_func=` on the generator, or a `condition` attribute on the sampler |
| Drop bad candidates | A hook at `GenerationStage.AFTER_GENERATE` that replaces `ctx.batch` |
| Generate then relax or run MD | `gen | relax` builds a `GenerationPipeline` |
| Many batches in a loop | `gen.stream(inputs, max_batches=N, num_samples=k)` |
| Reproducible draws | `seed=` on the generator; use the `rng` you receive |
| Molecular crystal starting structures | `CSPGenerator` (`nvalchemi.csp`) with `OverlapReliefPacker` (`nvalchemi.csp.packer`) |
| Drop duplicate crystal candidates | `deduplicate_batch`, `deduplicate_stream` (`nvalchemi.csp.comparison`), or `DeduplicateHook` (`nvalchemi.csp.hooks`), see [csp.md][doc-csp] (Screen similar and duplicate structures) |
| Save a generator or pipeline recipe | `model_dump_json()` / `model_validate_json()`. Weights stay in the model checkpoint |
| PhysicsNeMo diffusion samplers | Call them inside your sampler; see the docs page |

## How the pieces connect

```text
inputs -> [BEFORE_CONDITION] condition() [AFTER_CONDITION]
       -> generator_func(inputs, num_samples=..., rng=...) -> Batch
       -> [AFTER_GENERATE hooks filter ctx.batch] -> Batch
       -> next pipeline stage: dynamics.run(batch) or another callable
```

- The sampler owns the model and the sampling loop. The driver owns seeds,
  device checks, hooks, sessions, and streams.
- Returning a `Batch` is the contract. Any other return type passes through
  raw: no hooks, no device checks, and dynamics stages downstream reject it.
- In a pipeline, a stage with a `run()` method (any dynamics engine) runs to
  completion on the batch; other callables are called with the batch.

## Minimal sampler and pipeline

This pattern was run against `main`:

```python
def sample_fn(inputs=None, *, num_samples=1, rng=None, **kw):
    dev = rng.device if rng is not None else "cpu"
    pos = torch.randn(num_samples, n_atoms, 3, generator=rng, device=dev) * scale
    z = torch.full((n_atoms,), atomic_number, dtype=torch.long, device=dev)
    data = [AtomicData(positions=p, atomic_numbers=z) for p in pos]
    return Batch.from_data_list(data)

gen = AtomisticGenerator(generator_func=sample_fn, seed=0,
                         required_inputs=frozenset(),
                         outputs=frozenset({"positions", "atomic_numbers"}))
relax = FIRE2(model=model, dt=0.1, n_steps=200,
              convergence_hook=ConvergenceHook.from_fmax(0.05))
for hook in model.make_neighbor_hooks():
    relax.register_hook(hook)
with (gen | relax) as pipe:
    relaxed = pipe(stage_kwargs=[{"num_samples": 8}, None])
```

A generator used alone needs no declarations: `gen.sample(num_samples=8)`
returns the batch.

## Points to get right

- **Declare inputs and outputs for pipelines.** `gen | stage` fails at
  construction with "missing declarations: required_inputs, outputs" unless
  the generator or the sampler declares them. Set them on
  `AtomisticGenerator(...)` or as attributes of the sampler object.
- **Use the `rng` argument** for every random draw, so `seed` makes runs
  reproducible. Draw on `rng.device`.
- **Stay on the resolved device.** The returned batch must live on the
  generator's device, or the call raises `ValueError`. A sampler object can
  expose a `device` property.
- **Return a batch of `num_samples` graphs.** Batch first: do not loop and
  call the generator once per structure.
- **Total rejection.** If a filter rejects everything, set
  `ctx.batch = Batch.empty(num_systems=0, num_nodes=0, num_edges=0,
  device=...)`. Downstream stages then skip that batch. Boolean indexing
  that keeps zero graphs is not supported.
- **Conditioning does not enforce physics.** Put physical constraints in the
  sampler (guidance), in filter hooks, or in a relaxation stage.
- **Bounded streams.** `iter(gen)` never ends. Use
  `gen.stream(..., max_batches=N)`.
- **Compile the network, not the driver.** Building a `Batch` breaks
  `torch.compile` graphs. Compile the model inside the sampler and keep the
  sampler eager.
- **Relaxation needs model fields.** Generated batches usually carry only
  positions and numbers (plus cell and pbc for crystals). Add `cell` and
  `pbc` for periodic structures before relaxing, and register the model's
  neighbor hooks (rule 2 in `references/dynamics-optimization.md`).

## Filter hook pattern

A filter hook follows the shared `Hook` protocol (`stage`, `frequency`,
`__call__(ctx, stage)`). Compute per-structure scores vectorized over the
batch with `batch.batch_idx`, then keep a boolean mask:

```python
class KeepCompact:
    stage = GenerationStage.AFTER_GENERATE
    frequency = 1
    def __init__(self, max_radius): self.max_radius = max_radius
    def __call__(self, ctx, stage):
        b = ctx.batch
        r = b.positions.norm(dim=-1)
        rmax = r.new_zeros(b.num_graphs).scatter_reduce_(   # same dtype as r
            0, b.batch_idx.long(), r, reduce="amax")
        keep = rmax <= self.max_radius
        ctx.accepted_mask = keep
        ctx.batch = b[keep] if keep.any() else Batch.empty(
            num_systems=0, num_nodes=0, num_edges=0, device=r.device)
```

## Common mistakes

- Returning a list of `AtomicData` or a raw tensor instead of a `Batch`.
- Drawing noise with the global torch RNG instead of `rng`.
- Building a pipeline without `required_inputs` and `outputs` declarations.
- Unbounded `for batch in gen:` loops.
- Relaxing generated crystals without `cell`, `pbc`, or neighbor hooks.

## How to check the result

- `batch.num_graphs == num_samples` (before filtering), and atom counts per
  graph as expected.
- Same `seed` gives the same positions on two runs; different seeds differ.
- No overlapping atoms: the minimum pair distance per structure is above a
  sensible floor for the elements.
- After relaxation: per-structure max force below the threshold. A plain
  optimizer does not add a `status` column, so check forces, not status.

[doc-gen]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/generative.md
[doc-csp]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/csp.md
[doc-sims]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/dynamics_simulations.md
[api-gen]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/modules/gen.rst
[ex-gen]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/intermediate/09_generative_samplers.py
[ex-opt]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/basic/02_geometry_optimization.py
