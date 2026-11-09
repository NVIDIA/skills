# Models: loading, wrapping, and combining potentials

This note covers the shipped model wrappers, wrapping your own PyTorch model so
dynamics and training can call it, combining models, and reading outputs and
embeddings. Domain-decomposition specs for your own model are in
`references/composition-and-extension.md`.

## Where to read more

| Topic | Docs page | Example | Installed source |
| --- | --- | --- | --- |
| Wrapper contract, `ModelConfig`, outputs | [models.md][doc-models] | [06_ddp_mlp_training.py][ex-mlp] (a small wrapped MLP), [04_byo_pytorch_mpnn.py][ex-byo] | `nvalchemi/models/base.py` |
| MACE | [models.md][doc-models] | [04_mace_nvt.py][ex-mace] | `nvalchemi/models/mace.py` |
| AIMNet2 and electrostatics | [models.md][doc-models] | [08_aimnet2_ewald_pipeline.py][ex-aimnet] | `nvalchemi/models/aimnet2.py`, `ewald.py`, `pme.py` |
| UMA (fairchem) | [models.md][doc-models] | [09_uma_nve.py][ex-uma] | `nvalchemi/models/uma.py` |
| Lennard-Jones, DFT-D3 | [models.md][doc-models] | [02_geometry_optimization.py][ex-opt] | `nvalchemi/models/lj.py`, `dftd3.py` |
| Combining models | [models.md][doc-models] (composition) | [07_composable_model_composition.py][ex-comp] | `nvalchemi/models/pipeline.py` |
| Test model with no extras | [models.md][doc-models] | [01_data_structures.py][ex-data] | `nvalchemi/models/demo.py` |

API reference sources: [docs/modules/models.rst][api-models].

## Which component for which need

| Need | Component |
| --- | --- |
| Quick test model, no extras, no neighbor list | `DemoModelWrapper(DemoModel())` (not physical) |
| Analytic pair potential for small tests | `LennardJonesModelWrapper(epsilon=..., sigma=..., cutoff=...)` |
| Pretrained MACE (MACE-MP, OFF) | `MACEWrapper.from_checkpoint(name_or_path, device=...)` |
| Pretrained AIMNet2 | `AIMNet2Wrapper.from_checkpoint(path, device=...)` |
| Pretrained UMA | `UMAWrapper.from_checkpoint(name_or_path, task_name=...)` |
| Dispersion or long-range electrostatics | `DFTD3ModelWrapper`, `EwaldModelWrapper`, `PMEModelWrapper` |
| Your own PyTorch model | Subclass `nn.Module` and `BaseModelMixin` |
| Sum independent models | `model_a + model_b` |
| One model's output feeds another (charges to Ewald) | `PipelineModelWrapper` with `PipelineGroup(use_autograd=True)` |
| Energy only, or energy plus stress | `model.set_config("active_outputs", {...})` |
| Hessians | `model.hessian_vector_product(batch, v)`, `model.prepare_hessian(batch)` (reusable operator), or `model.compute_hessian(batch)` (dense, writes `batch["hessian"]`). See [models.md][doc-models] (Hessians). |

`MACEWrapper`, `AIMNet2Wrapper`, and `UMAWrapper` import lazily. A missing
extra raises an `OptionalDependencyError` that names the extra to install.
Import the base classes from `nvalchemi.models.base` when you do not need a
backend.

## How the pieces connect

```text
Batch -> wrapper.adapt_input -> your nn.Module.forward -> wrapper.adapt_output
      -> ModelOutputs {"energy": [B,1], "forces": [V,3], "stress": [B,3,3], ...}
dynamics.compute() writes these back into the batch each step
model.model_config.neighbor_config -> model.make_neighbor_hooks() -> dynamics
```

- `ModelConfig` declares capabilities (`outputs`, `autograd_outputs`,
  `autograd_inputs`, `required_inputs`, `supports_pbc`, `neighbor_config`)
  and holds runtime choices (`active_outputs`, `gradient_keys`).
- `neighbor_config=NeighborConfig(cutoff=..., format=NeighborListFormat.COO
  or MATRIX)` tells the toolkit what neighbor data to build. The dynamics
  stage builds it through the hooks from `make_neighbor_hooks()`. The model
  reads `batch.neighbor_list` (COO, `[E, 2]`) or `batch.neighbor_matrix`
  (MATRIX).
- Dynamics check that the model returns every key in the integrator's
  `__needs_keys__` (usually `forces`; NPT and variable-cell optimizers also
  need `stress`).

## Wrapping your own model

Required pieces: `self.model_config` set in `__init__`, the
`embedding_shapes` property, `compute_embeddings`, and `forward`. Call
`super().adapt_input(...)` before you read positions: it turns on
`requires_grad` for `autograd_inputs` when an autograd output is active.
Compute forces from the energy with `torch.autograd.grad`. This shape was
checked against `main`:

```python
class MyWrapper(nn.Module, BaseModelMixin):       # nn.Module first
    def __init__(self, hidden=8, cutoff=5.0):
        super().__init__()
        self.hidden, self.cutoff = hidden, cutoff  # same names as __init__ args
        self.model = PairNet(hidden)               # composition, built here
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            autograd_outputs=frozenset({"forces"}),
            autograd_inputs=frozenset({"positions"}),
            neighbor_config=NeighborConfig(cutoff=cutoff,
                                           format=NeighborListFormat.COO),
        )
    @property
    def embedding_shapes(self): return {}
    def compute_embeddings(self, data, **kw): raise NotImplementedError
    def forward(self, data, **kw):
        self.adapt_input(data, **kw)               # sets requires_grad on positions
        e = self.model(data.positions, data.neighbor_list,
                       data.batch_idx, data.num_graphs)
        out = self.adapt_output({"energy": e}, data)   # energy -> [B, 1]
        if "forces" in self.model_config.active_outputs:
            out["forces"] = -torch.autograd.grad(e.sum(), data.positions,
                                                 create_graph=self.training)[0]
        return out
```

Imports: `from nvalchemi.models.base import BaseModelMixin, ModelConfig,
NeighborConfig, NeighborListFormat`. The full step-by-step guide, with
embeddings and stress through `prepare_strain`, is in [models.md][doc-models].

Points to get right:

- Store every `__init__` argument as an attribute of the same name. Training
  checkpoints rebuild the wrapper from those attributes; a missing one makes
  `CheckpointHook` fail with "model spec generation failed". Build the inner
  network inside `__init__` (as above) rather than passing a live module, or
  give the wrapper a `checkpoint_spec()` as `MACEWrapper` does.
- Per-graph energies must be reduced with `batch.batch_idx`
  (`index_add_` or `scatter_add_`), not a Python loop over graphs.
- `create_graph=self.training` keeps force losses trainable and keeps
  inference cheap.
- A model without a cutoff sets `neighbor_config=None`; then
  `make_neighbor_hooks()` returns an empty list.
- Declare `supports_pbc=True` only if the model handles periodic images.
  With COO lists, periodic shifts are in `batch.shifts` (Cartesian) or
  `batch.neighbor_list_shifts` (integer images).
- Classical potentials with no learned features stub the two embedding
  members as above.

## Loading pretrained checkpoints

Pass the shared device as `from_checkpoint(..., device=device)`; for a custom
wrapper use `model.to(device)` before building optimizer state or running the
batch. Select `cuda` when available, and place data on that device too.

- `MACEWrapper.from_checkpoint("medium-0b2", device="cuda")` accepts a named
  foundation checkpoint or a local file. Prefer it over loading the raw MACE
  module, because it records how to rebuild the model for training
  checkpoints.
- **Local MACE paths.** With some `mace-torch` versions (seen with 0.3.15),
  a plain local path fails with "unknown url type", because the path goes
  through MACE's download helper. Pass a `file://` URL instead:

  ```python
  from pathlib import Path
  ckpt = Path("model.pt").resolve()
  model = MACEWrapper.from_checkpoint(ckpt.as_uri(), device="cuda")
  ```

  Try the plain path first; fall back to `as_uri()` on that error.
- `MACEWrapper` and `MACEWrapper.from_checkpoint` take `train` (since
  0.3.0). The default `train=False` freezes all weights and sets eval mode.
  Forces still work, because they come from autograd on positions. Pass
  `train=True` to train or fine-tune the full model. `compile_model` no
  longer freezes weights. `train` alone decides that.
- Keep `compile_model=False` for fine-tuning. The fine-tuning docs do not
  cover compiled training.
- UMA needs its own environment (`uma-cu12` or `uma-cu13` extra), separate
  from MACE. `task_name` picks the head (`"omol"`, `"omat"`, ...);
  `active_outputs` does not. With OMol, supplying a cell turns periodic
  boundaries on by default, and without a cell no stress is returned even
  though it is declared. Set `pbc` explicitly.
- AIMNet2 runs its kernels in float32.
- `AIMNet2Wrapper` leaves out the Coulomb and D3 dispersion terms on
  purpose. The module docstring in `nvalchemi/models/aimnet2.py` says so.
  Do not wrap the raw `AIMNet2Calculator` in your own `BaseModelMixin`
  adapter to get them back. Compose instead, as in the next section. If
  the user does not ask for long-range terms, use `AIMNet2Wrapper` as is.
  Say in the answer that Coulomb and D3 are not included.

## Combining models

- `lj + dftd3` builds a `PipelineModelWrapper` that sums energies, forces, and
  stress from models that do not depend on each other.
- When one model's output is another's input (AIMNet2 charges into Ewald or
  PME) and forces must flow through both, use
  `PipelineModelWrapper(groups=[PipelineGroup(steps=[a, b], use_autograd=True)])`.
- To add long-range terms to AIMNet2, feed its charges into
  `EwaldModelWrapper` (`nvalchemi.models.ewald`) or `PMEModelWrapper`
  (`nvalchemi.models.pme`) in one autograd group. Add `DFTD3ModelWrapper`
  (`nvalchemi.models.dftd3`) for dispersion. [08_aimnet2_ewald_pipeline.py][ex-aimnet]
  shows the AIMNet2 and Ewald pipeline.
- Always get neighbor hooks from the composed model:
  `composed.make_neighbor_hooks()`. It may build more than one list when the
  cutoffs differ. A hook built by hand from `composed.model_config` covers
  only the largest cutoff.

## Common mistakes

- Missing neighbor-list hook in dynamics, which shows as
  `KeyError: 'neighbor_matrix' required but not found` or the COO
  equivalent. Register `model.make_neighbor_hooks()`.
- `class MyWrapper(BaseModelMixin, nn.Module)`: put `nn.Module` first.
- Forgetting `self.model_config` in `__init__` (raises `TypeError`).
- Returning `energy` with shape `[B]`; the contract is `[B, 1]`.
  `adapt_output` unsqueezes a 1-D energy for you.
- Detaching positions or wrapping the forward pass in `torch.no_grad()`;
  autograd forces then come back as zeros or raise.
- Asking for `stress` from a model that does not list it in `outputs`.
  `set_config` warns and the integrator fails.
- A `MACEWrapper` loaded with the default `train=False` and passed to a
  training strategy. It has no trainable parameters, so the strategy raises.
- Reading `"hessian"` from a MACE model output. `MACEWrapper` no longer
  returns it. Use the Hessian methods in the table above.
- `compute_hessian` keeps the neighbor list of the batch you pass. Build the
  neighbor data on that batch first. The method runs no neighbor hooks.

## How to check the result

Run only the checks the user asked for. For a quick shape check, do one
forward pass, print the shapes, and stop. List the other checks below as next
steps in your answer. Do not run them unasked.

- Call the wrapper on a two-graph `Batch` and check shapes:
  `energy [2, 1]`, `forces [V, 3]`.
- Finite differences: move one atom by ±1e-3 Å and compare
  `-(E+ - E-) / 2e-3` with the force component (use float64 for this check).
- Translation invariance: shifting all positions leaves the energy unchanged
  and the forces summing to about zero per graph.
- Run 10 to 20 NVE steps with a small timestep and check that total energy
  stays flat (see the dynamics note).

[doc-models]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/models.md
[api-models]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/modules/models.rst
[ex-data]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/basic/01_data_structures.py
[ex-opt]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/basic/02_geometry_optimization.py
[ex-mace]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/advanced/04_mace_nvt.py
[ex-mlp]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/intermediate/06_ddp_mlp_training.py
[ex-byo]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/distributed/04_byo_pytorch_mpnn.py
[ex-comp]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/advanced/07_composable_model_composition.py
[ex-aimnet]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/advanced/08_aimnet2_ewald_pipeline.py
[ex-uma]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/advanced/09_uma_nve.py
