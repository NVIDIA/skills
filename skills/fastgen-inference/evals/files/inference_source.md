# FastGen source evidence

Source revision: `e0a92619fc56daf4dfa081273c7729c045bf1dcc`. Excerpts below are input evidence, not an installed checkout.

## scripts/inference/inference_utils.py:416

```text
def load_checkpoint(
    checkpointer: Checkpointer,
    model: Any,
    ckpt_path: str,
    strict: bool = False,
) -> Optional[int]:
    """Load an explicit checkpoint into ``model.net`` and its EMA networks, if a path is given.

    The checkpoint is read on the CPU, where meta-initialized networks are also materialized;
    ``setup_inference_modules`` places the networks afterwards.

    Args:
        checkpointer: Checkpointer instance
        model: Model whose ``net`` and EMA networks receive the weights
        ckpt_path: Explicit .pth checkpoint file or DCP completion marker
        strict: Whether checkpoint keys must match the model exactly. The default tolerates
            missing or unexpected keys, such as an optional logvar head, as long as at least
            one parameter matches.

    Returns:
        Checkpoint iteration, or None when no checkpoint path was given.

    Raises:
        ValueError: A nonempty checkpoint path does not end with ``.pth``.
        FileNotFoundError: The ``.pth`` file or the DCP shards it marks do not exist.
        RuntimeError: The checkpoint matches no parameter, or holds unexpected keys under ``strict``.
            In a distributed run, each of these failures surfaces as a ``RuntimeError`` on every rank.
    """
    if not ckpt_path:
        logger.warning("No checkpoint path provided; sampling from the pretrained network")
        return None

    # Build model dict for loading
    model_dict_infer = torch.nn.ModuleDict({"net": model.net, **model.ema_dict})
    logger.info(f"ckpt_path: {ckpt_path}")

    ckpt_iter = checkpointer.load(model_dict_infer, path=ckpt_path, strict=strict)
    logger.success(f"Loading successfully checkpoint {ckpt_iter}")
    return ckpt_iter
```

## scripts/inference/inference_utils.py:480

```text
def setup_inference_modules(
    model: Any,
    do_teacher_sampling: bool,
    do_student_sampling: bool,
) -> Tuple[Optional[Any], Optional[Any], Any]:
    """Keep and set up only the model modules selected for inference.

    This must run after checkpoint loading: every network selected for sampling is placed as a
    frozen component, with its state stored in ``model.precision_model`` and its predictions
    returning in ``model.precision``, and the roots inference does not need are discarded before
    the preprocessors are built on a surviving one. Training-only modules (fake_score,
    discriminator, reward) are already skipped at construction time when
    ``config.model.inference=True`` (set by ``load_config(..., evaluation=True)``).

    Discarded modules are removed from ``module_dict`` and their attributes are set to None
    rather than deleted, so that model_dict property accesses (which read e.g.
    ``self.fake_score``) return None instead of raising AttributeError.

    Args:
        model: The model instance
        do_teacher_sampling: Whether to set up teacher for sampling
        do_student_sampling: Whether to set up student for sampling

    Returns:
        Tuple of (teacher, student, net). Teacher and student may be None; ``net``
        is always the surviving network that hosts the preprocessors, which callers
        need for text/image encoding and decoding via ``net.preprocessors["vae"]``.
    """
    if not do_teacher_sampling and not do_student_sampling:
        raise ValueError("At least one of teacher or student sampling must be enabled")

    # A method whose training run had no teacher (e.g. plain SFT or non-CD consistency) never
    # built one. Build and load it on demand here, before any mutation below: build_teacher
    # instantiates the teacher (or base) architecture and loads `pretrained_model_path` on top of
    # whatever weights the network already loaded itself.
    if do_teacher_sampling and getattr(model, "teacher", None) is None:
        logger.info("No teacher was built for this method; building and loading one for teacher sampling.")
        model.build_teacher()

    # The module_dict entries to preserve, starting with the roots that host the sampled
    # networks; each of those is prepared exactly once, as it is selected.
    keep = set()

    student = None
    if do_student_sampling:
        student_name = model.use_ema[0] if model.use_ema else "net"
        student = model.module_dict[student_name]
        keep.add(student_name)
        model.prepare_module(student, name=student_name)
        logger.info(f"Evaluating student with {student.noise_scheduler.default_student_steps} steps.")

    teacher = None
    if do_teacher_sampling:
        # Sampling runs through the teacher when there is one -- including a SharedBaseTeacher,
        # a parameterless view over net -- but only a separate teacher is a root of its own.
        teacher = model.teacher
        teacher_name = "net" if isinstance(teacher, SharedBaseTeacher) else "teacher"
        if teacher_name not in keep:
            keep.add(teacher_name)
            model.prepare_module(model.module_dict[teacher_name], name=teacher_name)
```

## scripts/inference/inference_utils.py:565

```text
def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared between image and video inference.

    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        "--ckpt_path",
        default="",
        type=str,
        help="Path to a .pth checkpoint file or DCP marker (optional, uses pretrained if not provided)",
    )
    parser.add_argument(
        "--do_student_sampling",
        default=True,
        type=str2bool,
        help="Whether to perform student sampling",
    )
    parser.add_argument(
        "--do_teacher_sampling",
        default=False,
        type=str2bool,
        help="Whether to perform teacher sampling (a method trained without a teacher builds one from its config)",
    )
```

## scripts/inference/image_model_inference.py:267

```text
    parser.add_argument(
        "--sample_steps",
        default=None,
        type=int,
        help="Number of teacher denoising steps, i.e. sampling-grid intervals rather than grid points (default: the "
        "network's own default). The student, including an SFT model, samples model.student_grid_cfg.steps steps "
        "instead.",
    )

```

## Attached recipe provenance

`config_dmd2_sd15.py` is copied byte-for-byte from `fastgen/configs/experiments/SD15/config_dmd2.py` at `e0a92619fc56daf4dfa081273c7729c045bf1dcc`. The sampling request explicitly adds `model.use_ema=True`; the base recipe does not enable EMA on its own.

## [fastgen/networks/Flux/network.py:489](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/networks/Flux/network.py#L489)

```python
        # Add logvar linear layer for variance estimation on the time embedding
        inner_dim = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
        self.transformer.logvar_linear = torch.nn.Linear(inner_dim, 1)
```
