# FastGen source fixture

Source revision: `e0a92619fc56daf4dfa081273c7729c045bf1dcc`. Read these implementation excerpts; they are not a substitute runtime.

## fastgen/configs/config.py:30

```text
@attrs.define
class LogConfig:
    # Project name
    project: str = "fastgen"
    # Experiment name
    group: str = "cifar10"
    # Run/job name; None takes the name of the config file it is loaded from
    name: Optional[str] = None
    # W&B mode, can be "online", "offline", or "disabled".
    wandb_mode: str = "online"

    # save path
    @property
    def save_path(self) -> str:
        assert self.name is not None, "log_config.name is unset: set a name or load the config with load_config"
        return os.path.join(output_root(), f"{self.project}/{self.group}/{self.name}")
```

## fastgen/callbacks/wandb.py:263

```text
@rank0_only
def init_wandb(config: "BaseConfig") -> None:
    """Initialize the wandb run on rank 0.

    No-op on non-rank-0 ranks (via ``@rank0_only``), so ``wandb.run``
    only ever exists on rank 0. Every chartable user metric flows
    through rank 0; cross-rank artifacts (prompts, log-module rows)
    are gathered explicitly by callers.
    """
    wandb_config = config.log_config

    logger.info(f"wandb_config.save_path: {wandb_config.save_path}")
    os.makedirs(wandb_config.save_path, exist_ok=True)
    wandb_id_path = f"{wandb_config.save_path}/wandb_id.txt"
    if os.path.isfile(wandb_id_path):
        with open(wandb_id_path, encoding="utf-8") as f:
            wandb_id = f.read().strip()
        logger.info(f"Resuming with an existing wandb id: {wandb_id}")
    else:
        wandb_id = _generate_wandb_id()
        with open(wandb_id_path, "w", encoding="utf-8") as f:
            f.write(f"{wandb_id}\n")
        logger.info(f"Generating a wandb id: {wandb_id}")

    config_resolved = serialize_config(config, return_type="dict")

    wandb.init(
        id=wandb_id,
        project=wandb_config.project,
        group=wandb_config.group,
        name=wandb_config.name,
        config=config_resolved,
        dir=wandb_config.save_path,
        resume="allow",
        mode=wandb_config.wandb_mode,
    )

    upload_code = str2bool(os.getenv("WANDB_UPLOAD_CODE", "false"))
    if upload_code:
        logger.info("Uploading code to wandb (this may take a few minutes)...")
        wandb.run.log_code(".")
        logger.info("Code upload to wandb completed")
    else:
        logger.info("Wandb code upload disabled (set WANDB_UPLOAD_CODE=true to enable)")
```

## fastgen/utils/scripts.py:83

```text
def resolve_save_dir(config: BaseConfig) -> str:
    """Return the checkpoint directory, which the checkpointer uses as given.

    An S3 URI or absolute ``save_dir`` is kept, whatever ``use_s3`` is. Any other ``save_dir`` goes
    below the run's directory, so two runs never share a checkpoint prefix and ``resume`` cannot pick
    up another run's checkpoints: below the log path, or with ``use_s3`` below the run's S3 prefix.
    """
    log_config, checkpointer = config.log_config, config.trainer.checkpointer
    if s3.is_uri(checkpointer.save_dir) or os.path.isabs(checkpointer.save_dir):
        return checkpointer.save_dir
    if not checkpointer.use_s3:
        return os.path.join(log_config.save_path, checkpointer.save_dir)
    return os.path.join(
        checkpointer.s3_container, log_config.project, log_config.group, log_config.name, checkpointer.save_dir
    )
```
