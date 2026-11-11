# Source excerpts

From [NVlabs/FastGen commit e0a92619fc56daf4dfa081273c7729c045bf1dcc](https://github.com/NVlabs/FastGen/tree/e0a92619fc56daf4dfa081273c7729c045bf1dcc); line numbers refer to that checkout.

`config_dmd2_wan.txt` is a byte-for-byte copy of `fastgen/configs/experiments/WanT2V/config_dmd2.py`. `initial_config.py` is an authored experiment that composes that real recipe using its actual config and callback classes. The resume/health inventories describe review scenarios, not execution evidence produced by this evaluation.

## fastgen/trainer.py:254

```python
        # ``max_iter`` is inclusive, so the final checkpoint, labelled with the last completed iteration,
        # holds the state after ``max_iter`` optimizer iterations.
        iter_cur = iter_start
        for iter_cur in range(iter_start + 1, self.config.trainer.max_iter + 1):
            self.callbacks.on_training_step_begin(model, iteration=iter_cur)
            for grad_accum_iter in range(self.config.trainer.grad_accum_rounds):
                data = next(dataloader_train_iter)
                data = model.prepare_batch(data, augment_pipe)
                logger.debug(
                    f"iteration: {iter_cur} | grad_accum_iter: {grad_accum_iter} | data: {basic_utils.to_str(data)}"
```

## fastgen/trainer.py:343

```python
    def resolve_sampler_start_idx(self, iteration: int) -> None:
        """
        Resolve unset (``None``) dataloader sampler start indices for a run starting at ``iteration``.
        """
        nimg = (
            iteration * self.config.dataloader_train.batch_size * self.config.trainer.grad_accum_rounds * world_size()
        )
        if getattr(self.config.dataloader_train, "sampler_start_idx", 0) is None:
            logger.info(f"Setting sampler start index to {nimg} images for dataloader_train")
            self.config.dataloader_train.sampler_start_idx = nimg
        # Only the training stream consumed ``nimg`` samples, so the validation loader starts at its beginning.
        if self.config.dataloader_val and getattr(self.config.dataloader_val, "sampler_start_idx", 0) is None:
            self.config.dataloader_val.sampler_start_idx = 0
```

## fastgen/methods/distribution_matching/dmd2.py:830

```python
    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        """Return the optimizers active on this alternating iteration."""
        if iteration % self.config.student_update_freq == 0:
            return [self.net_optimizer]

        optimizers = []
        if self.use_vsd:
            optimizers.append(self.fake_score_optimizer)
        if self.use_gan:
            optimizers.append(self.discriminator_optimizer)
        return optimizers
```

## fastgen/callbacks/wandb.py:581

```python
    def on_training_step_end(
        self,
        model: "FastGenModel",
        data_batch: Dict[str, torch.Tensor],
        output_batch: Dict[str, Any],
        loss_dict: Dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> None:
        self.loss_dict_record.add(loss_dict)
        # Scalars fire on ``logging_iter``; media fires on the (typically
        # slower) ``sample_logging_iter`` so decode + W&B uploads can be
        # throttled without losing scalar telemetry. Iter 1 fires both
        # for an early sanity-check log.
        log_scalars = iteration == 1 or iteration % self.logging_iter == 0
        log_media = iteration == 1 or iteration % self.sample_logging_iter == 0
        if not (log_scalars or log_media):
            return
        time_start = time.perf_counter()
        if log_media:
            self._step(
                model,
                data_batch,
                output_batch,
                iteration,
                "train",
                "",
                0,
                self.loss_dict_record,
            )
        if log_scalars:
            self._log_stats(self.loss_dict_record, iteration, "train")
```

## fastgen/callbacks/grad_clip.py:57

```python
    def nan_to_num(self, module: torch.nn.Module) -> tuple[int, torch.dtype | None]:
        grad_dtype = None
        non_finite_grads_count = 0

        for name, param in module.named_parameters():
            if param.grad is not None:
                grad_dtype = param.grad.dtype

                # Extract local tensor for DTensor (avoids triggering distributed ops on CPU)
                if isinstance(param.grad, DTensor):
                    grad = param.grad._local_tensor
                else:
                    grad = param.grad

                non_finite_grads = grad.numel() - grad.isfinite().sum().item()
                if non_finite_grads:
                    non_finite_grads_count += non_finite_grads
                    logger.debug(
                        f"Gradient of {name} (dtype {grad_dtype}) is not finite: "
                        f"Setting {grad.isnan().sum().item()} NaNs to 0 and {grad.isinf().sum().item()} Infs "
                        f"to {self.posinf} or {self.neginf}."
                    )
                    torch.nan_to_num(grad, nan=0.0, posinf=self.posinf, neginf=self.neginf, out=grad)

        return non_finite_grads_count, grad_dtype
```

## [fastgen/configs/callbacks.py:25](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/configs/callbacks.py#L25)

```python
    ema=L(EMACallback)(
        type="constant", beta=0.9999, gamma=16.97, ema_halflife_kimg=500, ema_rampup_ratio=0.05, start_iter=0
    ),
)

```

## [fastgen/configs/methods/config_dmd2.py:114](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/configs/methods/config_dmd2.py#L114)

```python
    config = Config()
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            "grad_clip_fake_score": L(GradClipCallback)(grad_norm=None, model_key="fake_score"),
            "grad_clip_discriminator": L(GradClipCallback)(grad_norm=None, model_key="discriminator"),
            **BASE_CALLBACKS,
        }
    )

    config.dataloader_train.batch_size = 256
```

## [fastgen/callbacks/carry.py:35](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/callbacks/carry.py#L35)

```python
    """Carry one atomic output dictionary along training trajectories.

    One independent slot per gradient-accumulation step: the callback captures
    ``output_batch[CARRY_OUTPUT_KEY]`` after each accum step and atomically
    replaces the next preprocessed batch in the same slot with it (a
    state-free terminal marker is overlaid instead). Methods include
    ``CARRY_OVERLAY_KEYS`` fields when they must stay paired with the carried
    state. Validation carry is opt-in with ``enable_validation=True``; by
    default, every validation batch remains an independent sample. The optional
    validation slot is reset around every pass, so trajectories never leak
    across validation configurations or checkpoints.

    Every rank still loads and preprocesses a fresh batch each step -- even
    when an active carry then discards it -- keeping sampler resume accounting
    and sharded preprocessing rank-synchronous. Carry state is deliberately
    not checkpointed: resumed slots start empty and methods rebuild the phase
    encoded by the resumed iteration from fresh data, preserving the per-rank
    phase schedule but not the exact in-flight trajectory.
    """

    def __init__(self, enable_validation: bool = False):
        self.enable_validation = enable_validation
        self._slots_dict: dict[int, dict[str, Any]] = {}
        self._validation_slot: dict[str, Any] | None = None
```

## [fastgen/methods/backward_simulation.py:144](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/methods/backward_simulation.py#L144)

```python
    if advance_type == "staggered":
        phase_classes = math.gcd(num_steps, update_period)
        if num_ranks % phase_classes != 0:
            raise ValueError(
                f"{owner} cannot cover every student timestep uniformly with num_steps={num_steps}, "
                f"update_period={update_period}, and world_size={num_ranks}. Student updates preserve "
                f"the trajectory step modulo gcd(num_steps, update_period)={phase_classes}, so "
                f"world_size must be divisible by {phase_classes} to weight every phase class equally. "
                "Use a divisible world size or choose a coprime update period."
            )
        offset = (rank + iteration - 1) % num_steps
    elif advance_type == "synced":
```

## [fastgen/utils/scripts.py:149](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/utils/scripts.py#L149)

```python
    ddp_enabled = config.trainer.ddp
    fsdp_enabled = config.trainer.fsdp
    if fsdp_enabled and config.trainer.checkpointer.save_adapter_only:
        raise NotImplementedError("save_adapter_only is not supported with FSDP (DCP checkpointing)")

    # Check for dryrun
    if getattr(args, "dryrun", False):
        logger.info("Dryrun")
        logger.info(serialize_config(config, return_type="yaml"))
        sys.exit(0)

    # Always join a launcher-provided process group. FSDP and DDP training require one; evaluation never
    # wraps the model in DDP.
    distributed = ddp.init(required=fsdp_enabled or (ddp_enabled and not evaluation))
    if not evaluation and world_size() > 1 and not (ddp_enabled or fsdp_enabled):
        raise ValueError(
            f"{world_size()} ranks are running without trainer.ddp or trainer.fsdp, so each rank would train "
            "an independent replica; enable one of them or launch a single process."
```

## [fastgen/utils/scripts.py:182](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/utils/scripts.py#L182)

```python

    # Global batch size
    if config.trainer.batch_size_global is not None:
        batch_size = config.dataloader_train.batch_size * world_size()
        accum_rounds = max(config.trainer.batch_size_global // batch_size, 1)
        new_batch_size_global = accum_rounds * batch_size
        if new_batch_size_global != config.trainer.batch_size_global:
            logger.critical(
                f"Requested global batch size {config.trainer.batch_size_global} is not divisible by current batch size {batch_size}. New global batch size will be {new_batch_size_global}."
            )

        if accum_rounds != config.trainer.grad_accum_rounds:
            logger.info(
                f"Changing gradient accumulation rounds from {config.trainer.grad_accum_rounds} to {accum_rounds} to match requested global batch size."
            )
            config.trainer.grad_accum_rounds = accum_rounds

    logger.critical(
        f"Global batch size: {get_batch_size_total(config)} (Batch size per GPU: {config.dataloader_train.batch_size}, Gradient accumulation rounds: {config.trainer.grad_accum_rounds}, World size: {world_size()})"
```

## [fastgen/trainer.py:249](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/trainer.py#L249)

```python
        if iter_start == 0 and dataloader_val is not None:
            # validation before first training step
            self.validate(model_ddp, model, dataloader_val, iteration=iter_start)

```

## [fastgen/trainer.py:285](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/trainer.py#L285)

```python

            # validation
            if (is_final_iter or iter_cur % self.config.trainer.validation_iter == 0) and dataloader_val is not None:
                self.validate(model_ddp, model, dataloader_val, iteration=iter_cur)

            # save checkpoint
            just_saved_checkpoint = False
            latest_checkpoint_path = None
            if is_final_iter or iter_cur % self.config.trainer.save_ckpt_iter == 0:
                latest_checkpoint_path = self.save_checkpoint(model, iter_cur)
                just_saved_checkpoint = True
```

## [fastgen/trainer.py:442](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/trainer.py#L442)

```python
        for idx, val_vars in enumerate(self.config.trainer.global_vars_val):
            with set_temp_global_vars(val_vars), set_tmp_random_seed(
                self.config.trainer.val_seed,
                by_rank=True,
                devices=[model.device] if model.device.type == "cuda" else [],
            ):
                self.callbacks.on_validation_begin(model, iteration=iteration, idx=idx)
                logger.info(f"Validation iteration {iteration}")
                batches = synced_iter(dataloader_val, device=model.device)
                for step, data in enumerate(islice(batches, getattr(global_vars, "MAX_VAL_STEPS", None))):
                    data = model.prepare_batch(data)
                    self.callbacks.on_validation_step_begin(model, data, step=step, iteration=iteration, idx=idx)
                    logger.debug(f"iteration: {iteration} | validation step: {step} | data: {basic_utils.to_str(data)}")
                    with model.autocast():
                        loss_map, outputs = model_ddp.single_train_step(data, iteration)
                    self.callbacks.on_validation_step_end(
                        model, data, outputs, loss_map, step=step, iteration=iteration, idx=idx
                    )
                self.callbacks.on_validation_end(model, iteration=iteration, idx=idx)
```

## [fastgen/callbacks/carry.py:135](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/callbacks/carry.py#L135)

```python

    def on_validation_end(self, model: FastGenModel, iteration: int = 0, idx: int = 0) -> None:
        self._validation_slot = None

    def state_dict(self) -> dict[str, Any]:
        # Carry state is transient in-flight sampling state and is intentionally
        # not persisted; the method reconstructs it as needed on resume.
        return {}

```
