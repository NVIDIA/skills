# FastGen source evidence

Source revision: `e0a92619fc56daf4dfa081273c7729c045bf1dcc` in [NVlabs/FastGen](https://github.com/NVlabs/FastGen/tree/e0a92619fc56daf4dfa081273c7729c045bf1dcc).
Excerpts are copied from that checkout; they are evidence for repository-targeted artifacts, not an installed runtime.

The attached `config_dmd2_wan.py` is a byte-for-byte copy of `fastgen/configs/experiments/WanT2V/config_dmd2.py` at this revision. Its validation loader is unset; its training loader geometry is computed inside `create_config()` before CLI overrides.

## [fastgen/utils/scripts.py:100](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/utils/scripts.py#L100)

```python
def load_config(config_file: str, opts: list[str] | None = None, evaluation: bool = False) -> BaseConfig:
    """Load a config file, apply command-line overrides, and prepare it for training or evaluation.

    An unnamed config takes the file's name. The config passes through Hydra even without overrides, so every entry
    point gets the same resolved config. Callers resolve the checkpoint directory with ``resolve_save_dir`` in their
    own working directory.
    """
    config: BaseConfig = config_utils.import_config_from_python_file(config_file)
    if config.log_config.name is None:
        config.log_config.name = os.path.splitext(os.path.basename(config_file))[0]
    config = config_utils.override_config_with_opts(config, opts)

    # Evaluation skips training-only networks and consumes raw prompts/images.
```

## [fastgen/configs/config_utils.py:69](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/configs/config_utils.py#L69)

```python

def model_pixel_shape(model_config: Any) -> Tuple[int, ...]:
    """``(H, W)`` or ``(T, H, W)`` of the pixels that encode to ``model_config.input_shape``.

    Asks the class of ``model_config.net`` through ``pixel_shape_for``, so a config can size its loaders before any
    network exists. A modality dict contributes its ``"video"`` entry.
    """
    input_shape = model_config.input_shape
    if isinstance(input_shape, Mapping):
        input_shape = input_shape["video"]
    return model_config.net._target_.pixel_shape_for(input_shape)
```

## [fastgen/networks/Wan/network.py:1108](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/networks/Wan/network.py#L1108)

```python
    @classmethod
    def pixel_shape_for(cls, latent_shape: Sequence[int]) -> Tuple[int, ...]:
        """``FastGenNetwork.pixel_shape_for`` with the compression of the VAE that the latent channels identify.

        Wan2.2's VAE (TI2V-5B) has 48 latent channels and downsamples 16x spatially, the Wan2.1 VAE 16 and 8x.
        """
        factors = (4, 16, 16) if latent_shape[0] == 48 else (4, 8, 8)
        return pixel_shape_from_factors(latent_shape, factors)

```

## [fastgen/networks/network.py:67](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/networks/network.py#L67)

```python
def pixel_shape_from_factors(latent_shape: Sequence[int], factors: Sequence[int]) -> Tuple[int, ...]:
    """``(H, W)`` or ``(T, H, W)`` of the pixels that encode to a ``[C, H, W]`` or ``[C, T, H, W]`` latent.

    ``factors`` are the matching ``(H, W)`` or ``(T, H, W)`` downsampling factors. The first latent frame covers one
    pixel frame and every later one ``T``.
    """
    if len(factors) == 2 and len(latent_shape) == 3:
        h_factor, w_factor = factors
        return (latent_shape[1] * h_factor, latent_shape[2] * w_factor)
    if len(factors) == 3 and len(latent_shape) == 4:
        t_factor, h_factor, w_factor = factors
        return ((latent_shape[1] - 1) * t_factor + 1, latent_shape[2] * h_factor, latent_shape[3] * w_factor)
    raise ValueError(
        "Expected a [C, H, W] latent with (H, W) factors or a [C, T, H, W] latent with (T, H, W) factors, "
        f"got {tuple(latent_shape)} with {tuple(factors)}"
    )
```

## [fastgen/configs/data.py:89](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/configs/data.py#L89)

```python
    datatags=["WDS:/path/to/videos"],
    batch_size=2,
    key_map={"real_raw": "mp4", "condition_raw": "txt"},
    presets_map={"neg_condition_raw": "neg_prompt_wan"},
    sequence_length=81,
    img_size=(832, 480),
    num_workers=2,
    sampler_start_idx=None,
)

VideoAudioLoaderConfig = L(AudioVideoWDSLoader)(
    datatags=["WDS:/path/to/videos_with_audio"],
    batch_size=1,
    key_map={"real_raw": "mp4", "condition_raw": "txt"},
```

## [tests/test_configs.py:116](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/tests/test_configs.py#L116)

```python
def assert_loaders_match_input_shape(config, config_file: str):
    """Raw-media loaders produce the pixels that encode to ``input_shape``, as ``model_pixel_shape`` derives them."""
    if config.model.net is None or tuple(config.model.net.get("modalities") or ()) == ("audio",):
        return
    for name in ("dataloader_train", "dataloader_val"):
        loader = getattr(config, name)
        if loader is None or all(loader.get(key) is None for key in ("input_res", "img_size", "sequence_length")):
            continue
        *frames, height, width = model_pixel_shape(config.model)
        input_res = loader.get("input_res")
        actual = {
            "input_res": (input_res, input_res) if isinstance(input_res, int) else input_res,
            "img_size": loader.get("img_size"),
            "sequence_length": loader.get("sequence_length"),
        }
        expected = {
            "input_res": (width, height),
            "img_size": (width, height),
            "sequence_length": next(iter(frames), None),
        }
        for key, value in actual.items():
            if value is not None:
                value = value if key == "sequence_length" else tuple(value)
                assert value == expected[key], (
                    f"{config_file}: {name}.{key} is {value}, but input_shape {config.model.input_shape} "
                    f"decodes to {expected[key]}"
                )


```

## [fastgen/datasets/wds_utils.py:544](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/datasets/wds_utils.py#L544)

```python
    def __init__(
        self,
        datatags: List[str],
        batch_size: int,
        num_workers: int | None = None,
        train: bool = True,
        deterministic: bool = False,
        strict: bool = False,
        sampler_start_idx: int | None = None,
        shard_count_file: str | None = None,
        ignore_index_paths: List[str] | None = None,
        shard_start: int = 0,
        shard_end: int | None = None,
        shards_per_worker: int | None = None,
        partial: bool = False,
        drop_last: bool = True,
        shuffle_size: int = 1000,
    ):
        """Initialize common loader attributes.

```

## [fastgen/datasets/wds_utils.py:923](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/datasets/wds_utils.py#L923)

```python
    # Bind rank information before DataLoader creates isolated worker processes.
    rank, size = get_rank(), world_size()

    shard_errors = _ShardReadHandler(strict)
    dataset = wds.WebDataset(
        all_urls,
        resampled=train,
        nodesplitter=None if train else functools.partial(_nodesplitter, rank=rank, size=size),
        shardshuffle=False,
        handler=shard_errors,
        # Finite workers may have no assigned shards or only empty shards.
        empty_check=train,
    )
```

## [fastgen/trainer.py:249](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/trainer.py#L249)

```python
        if iter_start == 0 and dataloader_val is not None:
            # validation before first training step
            self.validate(model_ddp, model, dataloader_val, iteration=iter_start)

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
```
