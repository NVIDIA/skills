# Training: from scratch, fine-tuning, and distillation

This note covers `TrainingStrategy`, losses, optimizers, validation,
checkpoints and resume, data-parallel training, `FineTuningStrategy`, the
training CLI, and `DistillationStrategy`. Writing a custom loss term is in
`references/composition-and-extension.md`.

## Where to read more

| Topic | Docs page | Example | Installed source |
| --- | --- | --- | --- |
| Strategy lifecycle, optimizers, validation, checkpoints, CLI | [training.md][doc-train] | [06_ddp_mlp_training.py][ex-ddp], [10_mace_training.py][ex-mace] | `nvalchemi/training/strategy.py`, `runtime.py`, `cli.py` |
| Losses, weights, schedules | [losses.md][doc-loss] | none | `nvalchemi/training/losses/` |
| Fine-tuning and its CLI | [finetuning.md][doc-ft] | [08_lora_finetuning.py][ex-lora] | `nvalchemi/training/finetune.py`, `nvalchemi/training/hooks/finetune.py` |
| Multi-GPU training | [distributed_training.md][doc-ddp] | [06_ddp_mlp_training.py][ex-ddp] | `nvalchemi/training/hooks/ddp.py`, `nvalchemi/training/distributed.py` |
| Distillation | [distillation.md][doc-dist] | [09_offline_distillation.py][ex-off], [10_onpolicy_distillation.py][ex-on] | `nvalchemi/training/distillation/` |
| Saving and rebuilding specs | [serialization.md][doc-ser] | none | `nvalchemi/_serialization.py`, `nvalchemi/specs.py` |

API reference sources: [docs/modules/training/][api-train].

## Which component for which need

| Need | Component (from `nvalchemi.training` unless noted) |
| --- | --- |
| Train a model from scratch | `TrainingStrategy(models=..., optimizer_configs=..., training_fn=default_training_fn, loss_fn=..., num_epochs=...)`; `training_fn` is required |
| Adapt a pretrained model | `FineTuningStrategy(models=..., trainable_patterns=...)` |
| Low-rank adapters (LoRA) on a frozen base | `FineTuningStrategy(..., peft_config=LoRAConfig(...))` (`nvalchemi.training.peft.lora`), see [finetuning.md][doc-ft] (LoRA fine-tuning) |
| Load a saved LoRA checkpoint into a base model | `load_peft_checkpoint_into_model(base, dir)` (`nvalchemi.training.peft`) |
| Start a new run from a native checkpoint's weights | `FineTuningStrategy.from_pretrained_checkpoint(dir, ...)` |
| Continue an interrupted run | `TrainingStrategy.load_checkpoint(dir)` or `FineTuningStrategy.load_checkpoint(dir)` |
| Energy, force, stress losses | `EnergyMSELoss() + ForceMSELoss()`, or `ComposedLossFunction([...], weights=[...])`; see "Choosing losses" |
| Loss weight ramps | `LinearWeight`, `CosineWeight`, `PiecewiseWeight` times a loss term |
| Optimizer and scheduler | `OptimizerConfig(optimizer_cls=..., optimizer_kwargs=..., scheduler_cls=...)` |
| Validation | `ValidationConfig(validation_data=loader, every_n_epochs=1)` |
| Periodic restart checkpoints | `CheckpointHook(dir, epoch_interval=... or step_interval=...)` (`nvalchemi.training.hooks`) |
| EMA weights, mixed precision | `EMAHook`, `MixedPrecisionHook` (both from `nvalchemi.training.hooks`) |
| Replace a submodule before training | `module_patches={"main.model.readout": create_model_spec(...)}` |
| Several GPUs | `DistributedManager` plus `DDPHook`, launched with `torchrun` |
| Quick runs from a JSON spec | `nvalchemi-training` CLI (`train init`, `finetune init`, `spec report`, `spec run`, `spec resume`) |
| Small student from a large teacher | `DistillationStrategy` (`nvalchemi.training.distillation`) |
| Custom batch routing or several models | `training_fn=` for predictions, plus `loss_target_assembler=` for targets not on the batch |

## How the pieces connect

```text
DataLoader -> training_fn(model, batch) -> predictions {"predicted_energy", ...}
           -> loss_fn(predictions, targets from batch) -> backward -> optimizer step
hooks fire at TrainingStage points (BEFORE_FORWARD, AFTER_OPTIMIZER_STEP, ...)
CheckpointHook -> checkpoint dir -> load_checkpoint (resume) or
                                    from_pretrained_checkpoint (new run)
```

- Models are keyed by name. A single model becomes `"main"`, so parameter
  names start with `main.` (for example `main.model.readouts.0.linear.weight`
  for MACE).
- Loss terms read `prediction_key` and `target_key`. The default
  `training_fn` produces `predicted_energy`, `predicted_forces`, and so on.
- `ComposedLossFunction` normalizes weights by default. Pass
  `normalize_weights=False` for literal coefficients.
- Built-in losses refuse dtype mismatches (`dtype_policy="strict"`). If labels
  are float64 and the model float32, set
  `loss_fn.dtype_policy = "prediction_to_target"` or `"target_to_prediction"`.
- Models with a cutoff need a neighbor list on every training batch. Add it
  yourself as a hook; the strategy does not add one. Take the hooks from the
  model, so composed models get every list they plan, and retarget them:

  ```python
  from nvalchemi.training import TrainingStage

  def neighbor_hooks(model):
      hooks = model.make_neighbor_hooks()      # BEFORE_COMPUTE by default
      for hook in hooks:
          hook.stage = TrainingStage.BEFORE_FORWARD
      return hooks
  ```

  Pass them in `hooks=[...]`. The same stage also runs before strategy-owned
  validation.
- `training_fn` has no default; pass `default_training_fn` for the stock
  single-model path. It returns predictions only. Targets come from batch
  fields named by each loss's `target_key`. When a target is computed in
  `training_fn` (for example a frozen reference model's output), also pass
  `loss_target_assembler=` with the signature of
  `nvalchemi.training.losses.composition.assemble_loss_targets`. Models
  without an `optimizer_configs` entry stay frozen.
- Checkpoints rebuild each model from a spec read off its constructor: every
  `__init__` argument must be stored as an attribute of the same name (see
  `references/models.md`), or the first checkpoint write fails with "model
  spec generation failed".
- Only `BaseModelMixin` models can be trained here. Generative models
  (`GenerativeModelMixin`) are rejected; train them with your own loop or an
  energy-model adapter.

## Device placement and the step loop

Choose `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
once. Move the model before creating optimizer state. Put the dataset on that
device and pass `devices=[device]` to the strategy; omitting it defaults
training to CPU. The strategy places the models, stages each batch, and builds
the optimizers from those parameters. Use its `run()` instead of rebuilding
forward, backward, and optimizer stepping in Python.

With `reader`, `model`, and the agreed epoch budget already set, use
`Dataset` / `DataLoader` from `nvalchemi.data.datapipes` and the training
components from `nvalchemi.training`:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
dataset = Dataset(reader, device=device)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
strategy = TrainingStrategy(
    models=model, devices=[device],
    optimizer_configs=OptimizerConfig(optimizer_cls=torch.optim.AdamW),
    training_fn=default_training_fn, loss_fn=EnergyMSELoss() + ForceMSELoss(),
    hooks=[*neighbor_hooks(model), LossLogHook()], num_epochs=epochs,
)
strategy.run(loader)
```

For loss reporting, let the strategy gate a hook rather than reading the loss
every step. This hook uses `TrainingStage` from `nvalchemi.training`:

```python
class LossLogHook:
    stage = TrainingStage.AFTER_BATCH
    frequency = 25

    def __call__(self, ctx, stage):
        parts = ctx.losses["per_component_unweighted"]  # name -> 0-dim tensor
        values = torch.stack([ctx.loss.detach(),
                              *(v.detach() for v in parts.values())])
        print(ctx.step_count, dict(zip(["total", *parts], values.tolist())))
```

Keep `frequency` above 1, even for a short run. For a 10-step run, log every
5 steps or once at the end. A hook with `frequency = 1` reads the host on
every step. That is the sync the GPU rule forbids.

Read the host once per firing. Each `float()` or `.item()` call is its own
sync. Three `float()` calls in one hook cost three syncs. Stack every value
into one tensor first, then call `.tolist()` once, as the hook above does.

Define the hook before constructing the strategy. These scheduled reads are
fine; `.item()` on every loss is not. To keep observations entirely on the
GPU, retain bounded detached snapshots and convert after the run. Checkpoints
and validation can also read the host at their configured intervals.

If a custom loop is necessary, keep loss and mask calculations on the device;
log only every N steps (N > 1). Do not call `.item()`, `.cpu()`, `.tolist()`,
`float(loss)`, or `print(loss)` every step. `if torch.isfinite(loss)` and
`bool(tensor.all())` also synchronize: use toolkit safety/update hooks where
they fit, or check at an explicit interval. Do not replace the strategy just
to add a logger. For custom optimizer creation, construct it after
`model.to(device)`; on resume let the strategy restore optimizer placement.

## Choosing losses and schedules

- MSE (`EnergyMSELoss`, `ForceMSELoss`) is the default. MAE or Huber
  (`EnergyMAELoss`, `EnergyHuberLoss`, `ForceHuberLoss`) is more robust to
  noisy labels. `ForceL2NormLoss` scores the per-atom force vector instead of
  each component.
- For structures of different sizes, `EnergyMSELoss(per_atom=True)` fits
  energy per atom; total-energy MSE weights large structures much more.
  `ForceMSELoss(normalize_by_atom_count=True)` balances graphs instead of
  atoms. These are different objectives, so pick one on purpose.
- Add `StressMSELoss` only with reliable stress labels and `stress` active on
  the model.
- Schedules with `per_epoch=True` need `epoch=` when you call the loss
  yourself. Metric schedulers such as `ReduceLROnPlateau` step after
  validation, so they need `validation_config`; set
  `scheduler_metric_adapter` when the metric is not `"total_loss"`.
- Details: [losses.md][doc-loss], [training.md][doc-train].

## Fine-tuning a pretrained model

```python
strategy = FineTuningStrategy(
    devices=[device],                           # same device as model and loaders
    models=pretrained,                          # e.g. MACEWrapper.from_checkpoint(..., train=True)
    trainable_patterns=("main.model.readouts.*",),
    optimizer_configs=OptimizerConfig(optimizer_cls=torch.optim.AdamW,
                                      optimizer_kwargs={"lr": 1e-4}),
    training_fn=default_training_fn,
    loss_fn=EnergyMSELoss() + ForceMSELoss(normalize_by_atom_count=True),
    validation_config=ValidationConfig(validation_data=val_loader, every_n_epochs=1),
    hooks=[*neighbor_hooks(pretrained),
           CheckpointHook("runs/ft/checkpoints", epoch_interval=1)],
    num_epochs=5,
)
strategy.run(train_loader)
```

- `trainable_patterns` is a glob allow-list over full parameter names. A
  pattern that matches nothing raises `ValueError`. Print the names first:
  `[n for n, _ in strategy.models["main"].named_parameters()]` or
  `pretrained.named_parameters()` with a `main.` prefix. Glob names differ
  between models; MACE uses `readouts` (plural).
- `freeze_patterns` excludes a broad set first; `trainable_patterns` then
  re-admits exceptions.
- Use low learning rates for full-model fine-tuning.
- Since 0.3.0, `MACEWrapper.from_checkpoint` freezes all weights unless you
  pass `train=True`. Full-model fine-tuning needs `train=True`. Without it,
  the strategy raises "no trainable parameters" unless `trainable_patterns`,
  a trainable module patch, or `peft_config` selects parameters.
  `compile_model` does not freeze weights. Keep it off for training.
- With `peft_config=LoRAConfig(...)` the base stays frozen and only the
  adapters train. Save small checkpoints with
  `CheckpointHook(..., save_trainable_state_only=True)`. Load them for
  inference with `load_peft_checkpoint_into_model`, on the same base model.
- Loading MACE from a local file can fail with "unknown url type" under some
  `mace-torch` versions. Pass `Path(path).resolve().as_uri()` (a `file://`
  URL) instead of the plain path. See `references/models.md`.

## Resume versus a new run

| Goal | Call | Restores optimizer, scheduler, counters, hook state |
| --- | --- | --- |
| Continue after a crash or preemption | `FineTuningStrategy.load_checkpoint(dir, training_fn=...)` | Yes |
| New run from earlier weights | `FineTuningStrategy.from_pretrained_checkpoint(dir, ...)` | No |
| Training from scratch, continue | `TrainingStrategy.load_checkpoint(dir)` | Yes |

```python
resumed = FineTuningStrategy.load_checkpoint(
    "runs/ft/checkpoints", training_fn=default_training_fn,
    hooks=[*neighbor_hooks(model),               # rebuild every runtime hook
           CheckpointHook("runs/ft/checkpoints", epoch_interval=1)],
)
resumed.validation_config = ValidationConfig(validation_data=val_loader,
                                             every_n_epochs=1)   # not saved
resumed.run(train_loader)                       # picks up the counters
```

- Hooks are not rebuilt from the checkpoint. Pass every hook the run needs
  (neighbor lists, checkpointing, EMA, DDP, reporting); saved hook state is
  loaded only into hooks you pass. Without the neighbor hooks the first batch
  fails with `KeyError: 'neighbor_list' required`, and without
  `CheckpointHook` no further checkpoints are written.
- The directory is the one given to `CheckpointHook`. The latest checkpoint is
  used unless you pass `checkpoint_index`.
- Only native strategy checkpoints resume. A raw `state_dict` or a foreign
  checkpoint is a weight import: load it, then save a fresh strategy
  checkpoint before you rely on resume.
- Write checkpoints early and often enough for preemptible jobs. A checkpoint
  only exists after the first interval completes.

## The training CLI

`nvalchemi-training` scaffolds and runs JSON specs:

```bash
nvalchemi-training finetune init mace small-0b --dataset data/train.zarr \
  --output-dir runs/ft --out ft.json
nvalchemi-training spec report ft.json        # check before running
nvalchemi-training spec run ft.json
nvalchemi-training spec resume runs/ft/checkpoints --spec ft.json --checkpoint-index -1
```

The resume option is `--checkpoint-index` (hyphen). Older copies of the
fine-tuning docs show `--checkpoint_index`, which the CLI rejects. Check
`nvalchemi-training spec resume --help` on the installed version. Run
`spec report` before `spec run`; it warns about high learning rates, missing
validation data, and unsafe output paths.

## Multi-GPU training

```python
from nvalchemi.distributed import DistributedManager
from nvalchemi.training.hooks import DDPHook

DistributedManager.initialize()
strategy = TrainingStrategy(..., distributed_manager=DistributedManager(),
                            hooks=[DDPHook(), CheckpointHook(...)])
```

Launch with `torchrun --standalone --nproc_per_node=N train.py`. `DDPHook`
wraps the trainable models, picks the rank's device, and injects a
distributed sampler. Validation reduces across ranks and `CheckpointHook`
writes from rank 0, so do not gate either on rank. With one process the same
script runs unchanged. Write the launch command; do not start multi-GPU jobs
without the user's go-ahead.

## Distillation

- `DistillationStrategy(models={"student": s, "teacher": t},
  optimizer_configs={"student": [...]}, loss_fn=..., num_steps=...)`. The
  teacher is frozen because it has no optimizer entry.
- Point loss targets at teacher fields:
  `EnergyMSELoss(target_key="teacher_energy")`,
  `ForceMSELoss(target_key="teacher_forces")`. A term left on `energy` trains
  against dataset labels.
- Label once with `label_dataset(dataset, InProcessTeacherScorer(t,
  ("energy", "forces")), "labeled.zarr")`, then train from that store with no
  teacher forward passes.
- On-policy runs (`OnPolicyConfig`) generate frames with the student's own
  dynamics. They need a teacher-labeled `reference_dataset` when
  `replay_ratio < 1`; at `replay_ratio=1` pass none.
- An MD propagator never converges, so without a lifetime it runs the first
  batch of structures for the whole run. Set
  `max_trajectory_steps` (a multiple of `generation_steps`) to retire each
  trajectory and backfill from the initial structures. It works alone or
  with `fmax`.
- If a diverged student geometry crashes the neighbor-list build, set
  `quarantine_diverged=True`. It makes the frozen graph finite until it
  retires. That frame is never stored.
- Beyond energy and forces: `EmbeddingMatchingLoss` needs
  `training_fn=embedding_distillation_fn` (plus a trained `EmbeddingProjector`
  when widths differ); `HessianMatchingLoss` needs
  `hessian_distillation_fn` and the `hessian` signal; `BoltzmannMatchingLoss`
  needs an equilibrium on-policy loop and cannot be a validation loss.
- Resume: checkpoints and specs omit the live on-policy loop. Pass `models=`,
  `on_policy=`, and `reference_dataset=` to `DistillationStrategy.load_checkpoint`
  (plus the hooks, as above). The trajectory and replay buffer start fresh.
- Evaluate with `nvalchemi.training.distillation.evaluation`
  (`evaluate_accuracy`, `StabilityMonitor`, `build_acceptance_report`). They
  score exactly the model you pass: hand over the EMA weights
  (`EMAHook.get_averaged_model()`) if those are what you ship. A bar with no
  matching measurement fails the student; use `measured_bars(...)` to set
  only measured bars.
- Details and the on-policy constraints: [distillation.md][doc-dist].

## Common mistakes

- Resuming with `from_pretrained_checkpoint` (it resets optimizer and
  counters) or starting a new experiment with `load_checkpoint`.
- Glob patterns that do not match the model's parameter names.
- Float64 labels with a float32 model and the default strict dtype policy.
- Forgetting that `ComposedLossFunction` renormalizes weights.
- Force losses on a model whose `active_outputs` lacks `forces`.
- No `NeighborListHook` at `BEFORE_FORWARD` for a model with a cutoff.
- Full fine-tuning of a `MACEWrapper` loaded without `train=True`.
- Gating validation or checkpointing on rank 0 under DDP.
- A full training run as a first test. Run one or two epochs on a few
  batches, check that the loss falls, then scale.

## How to check the result

- The loss on a small subset falls over a few steps (overfit test).
- Only the intended parameters change: compare a frozen parameter and a
  trainable one before and after a step, or read the trainable-parameter
  summary that fine-tuning prints.
- Resume: the resumed run's step counter continues from the checkpoint and
  the first loss matches the last logged loss within noise.
- Validation energy and force MAE against held-out data, in eV/atom and
  eV/Å.

[doc-train]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/training.md
[doc-loss]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/losses.md
[doc-ft]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/finetuning.md
[doc-ddp]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/distributed_training.md
[doc-dist]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/distillation.md
[doc-ser]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/serialization.md
[api-train]: https://github.com/NVIDIA/nvalchemi-toolkit/tree/v0.3.0/docs/modules/training
[ex-ddp]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/intermediate/06_ddp_mlp_training.py
[ex-lora]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/intermediate/08_lora_finetuning.py
[ex-off]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/intermediate/09_offline_distillation.py
[ex-on]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/intermediate/10_onpolicy_distillation.py
[ex-mace]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/advanced/10_mace_training.py
