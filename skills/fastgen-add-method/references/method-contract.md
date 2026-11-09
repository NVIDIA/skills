# Method extension contract

Read before implementing an algorithm or auxiliary objective. The `FastGenModel`
docstrings in [model.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/model.py) own the lifecycle; the network and
run contracts own preparation, precision, supervision, distributed execution and resume.

## Register what the lifecycle manages

- `FastGenModel.__init__` calls the overridden `build_model()` and registers every
  `model_dict` entry before the subclass's remaining constructor code runs, so anything
  `build_model` needs must come from the config or be assigned before `super().__init__`.
  Skip training-only auxiliaries when `config.inference` is set so inference can load.
- Trainable roots belong in `model_dict`; frozen components still go through
  `register_modules` so preparation, casting and validation see them, but nothing is
  optimised merely by being registered. Provide `optimizer_dict` and `scheduler_dict`
  entries for every optimizer and return the active ones per phase from
  `get_optimizers`; DMD2's alternating phases are an example, not a required schedule.
- `build_model` injects `config.net_adapter` into `net` before method-specific
  construction; derive auxiliaries from the config (`teacher_config`) rather than by
  copying `net`, and check each optimizer's parameter set after construction.
- `Trainer.save_checkpoint` saves the `model_dict` roots, optimizers, schedulers, the
  grad scaler and callback state keyed by callback name. Statistics persist only as
  buffers on such a root or as explicit callback state; attributes on the method or on
  other registered modules are not restored.
- Validation calls the method under no-grad; it must not advance training histories,
  queues or carried state unless that is part of the design.
- Add a typed config subclass with `model_class = L(<Model>)(config=None)` under
  `fastgen/configs/methods/` and export the class in
  [methods/__init__.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/__init__.py) without circular imports.

## Return a loss graph the trainer can use

Read `Trainer.train_step` in [trainer.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/trainer.py) and
[common_loss.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/common_loss.py).

- `single_train_step(data, iteration)` returns `(loss_map, outputs)` with a scalar
  `total_loss` connected to the parameters of the active optimizers. The trainer owns
  accumulation scaling, backward, clipping and the optimizer and scheduler steps.
- Extra precomputed model-space batch fields go into the method's `state_fields` so
  `prepare_batch` casts them; raw encoding and rich conditioning stay in `prepare_batch`
  and the network's `build_conditioning`. A loader key is not forwarded to the network by
  itself.
- Match the reference's detach boundaries, signs and reductions: `mse_loss` forbids shape
  broadcasting and promotes precision, and a per-sample `loss_scale` weights the batch
  mean without renormalising by the number of unmasked samples.
- Return diagnostics detached unless the jointly backwarded loss covers them; with
  unused-parameter detection, [DDPWrapper](https://github.com/NVlabs/FastGen/blob/main/fastgen/utils/distributed/ddp.py) traverses
  every attached output graph. Reuse `_get_outputs` for sample logging: it separates the
  deferred fresh generation from the detached training estimate.
- Freezing parameters does not justify `no_grad`: keep autograd through frozen networks
  when a loss needs input derivatives, such as teacher features in a generator GAN loss.
  Detached targets can use `no_grad`; verify each loss term separately.
  JVP objectives need more than ordinary backward. Keep collective participation uniform
  across ranks when branches or masks differ; a masked loss keeps forwards that skipping
  would turn into a hang.

## Objective and student rollout must agree

- The base `generator_fn` is one x0-based rollout on the scheduler's student grid, not a
  universal sampler. A method that changes parameterisation, integration, secondary
  times or cache state implements its own generator and validates the effective grid at
  every supported NFE.
- Use the carry machinery ([carry.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/callbacks/carry.py),
  [backward_simulation.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/backward_simulation.py)) only when its
  semantics match; its state is not checkpointed.
- Compare loss values and parameter/input gradients with an independent tensor reference
  at fixed times and noise, including boundary times and disabled terms. For weighting,
  use unit, zero and mixed weights that distinguish the intended reduction denominator;
  an all-zero loss must stay connected for backward. Cover each changed method variant
  or explicitly leave it unsupported. Then exercise every optimizer phase, sampling and
  save/reload; construction plus a decreasing aggregate loss is not parity.
