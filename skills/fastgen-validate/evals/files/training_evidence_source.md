# FastGen source evidence

Source revision: `e0a92619fc56daf4dfa081273c7729c045bf1dcc` in [NVlabs/FastGen](https://github.com/NVlabs/FastGen/tree/e0a92619fc56daf4dfa081273c7729c045bf1dcc).
These are copied source excerpts, not execution results.

The evidence inventory is an authored review scenario. These production paths establish what its recorded operations can and cannot test.

## [fastgen/methods/distribution_matching/dmd2.py:790](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/methods/distribution_matching/dmd2.py#L790)

```python
    def single_train_step(
        self,
        data: Dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, TensorOrDict | Callable]]:
        """Run one alternating student or auxiliary training step."""
        self._setup_grad_requirements(iteration)

        inputs = self._training_inputs(data, iteration)

        update_step = (
            self._student_update_step
            if iteration % self.config.student_update_freq == 0
            else self._auxiliary_update_step
        )
        return update_step(inputs, data)

```

## [fastgen/methods/distribution_matching/dmd2.py:830](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/methods/distribution_matching/dmd2.py#L830)

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

    def get_lr_schedulers(self, iteration: int) -> list[torch.optim.lr_scheduler.LRScheduler]:
        """Return the learning-rate schedulers active on this iteration."""
        if iteration % self.config.student_update_freq == 0:
            return [self.net_lr_scheduler]

        schedulers = []
        if self.use_vsd:
            schedulers.append(self.fake_score_lr_scheduler)
        if self.use_gan:
            schedulers.append(self.discriminator_lr_scheduler)
        return schedulers
```

## [fastgen/utils/checkpointer.py:578](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/utils/checkpointer.py#L578)

```python
    def load(
        self,
        model_dict: torch.nn.ModuleDict,
        optimizer_dict: Dict[str, torch.optim.Optimizer] | None = None,
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        callbacks: CallbackDict | None = None,
        path: str | None = None,
        device: torch.device | str = "cpu",
        strict: bool = True,
    ) -> int:
        """Load the model checkpoint.

        Loading is not transactional: a failure may leave earlier tensors, models, optimizers,
        or scalar state changed. Discard or restore the entire target collection before reuse.

        Every rank reads the file. Monolithic checkpoints require unsharded model and optimizer
        targets; load pretrained weights before FSDP wrapping and use DCP for full FSDP resume.
        ``None`` entries in the model, optimizer, and scheduler dicts name objects the caller does not
        hold and receive nothing; the checkpoint's state for them is ignored. A model dict whose roots
        are all ``None`` is rejected; an empty one restores scalar state alone.

        Args:
            model_dict (torch.nn.ModuleDict): The model dict to load
            optimizer_dict (Dict[str, torch.optim.Optimizer]): The optimizer dict to load
            scheduler_dict (Dict[str, torch.optim.lr_scheduler]): The scheduler dict to load
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training)
            callbacks (CallbackDict | None): The callbacks to load
            path (str): A monolithic ``.pth`` checkpoint file or DCP completion marker. None loads the latest
                completed checkpoint in ``save_dir``, if any.
            device (torch.device | str): Where the ``.pth`` file's tensors are placed before copying; DCP
                shards load straight into the targets. Targets keep their placement and dtypes and receive values by
                copy, except loaded roots that hold meta tensors on any rank: every rank first places those
                on this device, materializing the meta tensors. Defaults to "cpu".
            strict (bool): Whether state-dict keys must match exactly. In strict mode, an adapter requires
                all adapter tensors, and loading any optional base parameter requires the complete optional
                base state. Optional buffers alone may load without the base parameters.

        Returns:
            int: The saved iteration, or 0 when no checkpoint is found.

        Raises:
            ValueError: If ``path`` does not name a ``.pth`` file, or discovery needs an absolute or
                ``s3://`` ``save_dir``, or monolithic loading is requested for FSDP/DTensor targets, or
                every root of a nonempty ``model_dict`` is ``None``.
            FileNotFoundError: If the checkpoint file is missing, or a DCP checkpoint lacks the shards of a
```

## [scripts/inference/inference_utils.py:452](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/scripts/inference/inference_utils.py#L452)

```python
    ckpt_iter = checkpointer.load(model_dict_infer, path=ckpt_path, strict=strict)
    logger.success(f"Loading successfully checkpoint {ckpt_iter}")
    return ckpt_iter
```

## [fastgen/utils/checkpointer.py:405](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/utils/checkpointer.py#L405)

```python
    def _load_scalar_state(
        state: Dict[str, Any],
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LambdaLR],
        grad_scaler: torch.amp.GradScaler | None,
        callbacks: CallbackDict | None,
    ) -> int:
        """Restore shared scalar state and return the saved iteration."""
        if scheduler_dict:
            logger.info("Loading the scheduler_dict...")
            scheduler_state = state.get("scheduler") or {}
            for k, v in scheduler_dict.items():
                if k in scheduler_state:
                    v.load_state_dict(scheduler_state[k])
                else:
                    logger.warning(f"Scheduler {k} not found in checkpoint.")

        if grad_scaler is not None:
            logger.info("Loading the gradient scaler...")
            grad_scaler_state = state.get("grad_scaler")
            if grad_scaler_state:
                grad_scaler.load_state_dict(grad_scaler_state)
            else:
                logger.warning("Gradient scaler state is empty (likely saved from disabled scaler), skipping load.")

        if callbacks is not None:
            logger.info("Loading the callbacks...")
            callbacks_state = state.get("callbacks")
            if callbacks_state is not None:
                callbacks.load_state_dict(callbacks_state)
            else:
                logger.warning("Callbacks not found in checkpoint.")

        if "iteration" not in state:
            logger.warning("Iteration not found in checkpoint.")
            return 0
        return state["iteration"]
```
