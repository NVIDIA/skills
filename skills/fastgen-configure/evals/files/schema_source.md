# FastGen source fixture

Source revision: `e0a92619fc56daf4dfa081273c7729c045bf1dcc`. Excerpts below are input evidence, not an installed checkout.

## fastgen/configs/config_utils.py:82

```text
def config_from_dict(ref_instance: Any, kwargs: Any) -> Any:
    """
    Construct an instance of the same type as ref_instance using the provided dictionary or data or unstructured data

    Args:
        ref_instance: The reference instance to determine the type and fields when needed
        kwargs: A dictionary of keyword arguments to use for constructing the new instance or primitive data or unstructured data

    Returns:
        Any: A new instance of the same type as ref_instance constructed using the provided kwargs or the primitive data or unstructured data

    Raises:
        AssertionError: If the fields do not match or if extra keys are found.
        Exception: If there is an error constructing the new instance.
    """
    is_type = is_attrs_or_dataclass(ref_instance)
    if not is_type:
        return kwargs
    else:
        ref_fields = set(get_fields(ref_instance))
        assert isinstance(kwargs, dict) or isinstance(kwargs, DictConfig), "kwargs must be a dictionary or a DictConfig"
        keys = set(kwargs.keys())

        # ref_fields must include all keys
        extra_keys = keys - ref_fields
        assert keys <= ref_fields, f"Unknown config fields {extra_keys} for {type(ref_instance)}"

        resolved_kwargs: Dict[str, Any] = {}
        for f in keys:
```

## fastgen/utils/scripts.py:149

```text
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
        )

    logger.info(
        f"Distributed process group initialized={distributed}, world size={world_size()}, "
        f"DDP configured={ddp_enabled}, FSDP configured={fsdp_enabled}."
    )

    # Configure credentials and the Hugging Face cache before model/trainer initialization.
    set_env_vars(config.credentials_dir)

    # Meta initialization needs a distributed source rank, but does not require FSDP:
    # eager FastGenModule preparation materializes and synchronizes meta ranks too.
    if config.model.meta_init and not distributed:
        logger.warning("Ignoring meta_init since distributed execution is not active.")
        config.model.meta_init = False

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
```

## fastgen/configs/config.py:300

```text
    # Resume
    resume: bool = True

    # DDP Parallelism
    ddp: bool = False
    # FSDP Parallelism
    fsdp: bool = False
    # FP32 precision of cuBLAS matmuls and cuDNN convolutions: TensorFloat32 if True, IEEE float32 if False.
    # The defaults are PyTorch's, which references that leave these settings alone run with.
    tf32_matmul: bool = False
    tf32_conv: bool = True
    # Allow reduced-precision reductions in cuBLAS fp16 matmuls (PyTorch's default).
    fp16_reduced_precision_reduction: bool = True

    # Number of gradient accumulation rounds
    grad_accum_rounds: int = 1

    # Global batch size (if not None, overrides grad_accum_rounds to match the specified batch size)
    batch_size_global: int | None = None

    # Temporarily offload registered non-decoder components during latent decoding (not supported with FSDP)
    offload_module_in_decoding: bool = False

    # apply cpu offloading in fsdp
    fsdp_cpu_offload: bool = False
    # Fallback minimum number of parameters for FSDP wrapping
    # (10M wraps large models into fairly small shards)
    # The FastGenNetwork should provide a fully_shard method that can be used to shard the network.
    # If we need to shard a different module, we fall back to an auto-sharding policy based on this value.
    fsdp_min_num_params: int = 10_000_000
    # Sharding group size for FSDP. If None, fully shard across all ranks.
    # If set, creates a 2D mesh with (replicate, shard) dimensions.
    fsdp_sharding_group_size: Optional[int] = None

    # global variables
    global_vars: Optional[dict] = None
    global_vars_val: List[dict | None] = attrs.field(factory=lambda: [None])

    # augment config
```

## [fastgen/configs/config.py:284](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/configs/config.py#L284)

```python
    # save checkpoint frequency
    save_ckpt_iter: int = 5000
    # test on validation set frequency
    validation_iter: int = 1000
    # logging frequency
    logging_iter: int = 1000
    # Total optimizer steps, including any restored from a checkpoint; excludes accumulation microbatches.
    # A fresh run executes iterations 1 through max_iter, saving the final state at max_iter.
    max_iter: int = 1000000
```

## fastgen/configs/config_utils.py:121

```python
def override_config_with_opts(config: Any, opts: Optional[List[str]] = None) -> Any:
    """
    Override the config with the opts.

    Args:
        config (Any): The config object.
        opts (Optional[List[str]]): Hydra override strings, starting with the "-" separator
            (e.g. ["-", "trainer.max_iter=10"]).

    Returns:
        Any: The config object.
    """

    # Convert Config object to a DictConfig object for Hydra

    if not isinstance(config, DictConfig):
        config_dict = attrs.asdict(config)
        config_dict = DictConfig(content=config_dict, flags={"allow_objects": True})
    else:
        config_dict = config

    # Use Hydra to handle overrides
    cs = ConfigStore.instance()
    cs.store(name="config", node=config_dict)

    if opts is None:
        opts = []

    if len(opts) > 0 and opts[0] != "-":
        raise ValueError(f"opts must start with '-' to separate from other arguments. Got: {opts}")
    opts = opts[1:]
    with initialize(version_base=None):
        try:
            cfg = compose(config_name="config", overrides=opts)
        except Exception as e:
            raise ValueError(f"Failed to compose config with opts: {e}")

        OmegaConf.resolve(cfg)

    config = config_from_dict(config, cfg)

    return config
```
