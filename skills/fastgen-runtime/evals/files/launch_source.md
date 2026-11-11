# FastGen source fixture

Source revision: `e0a92619fc56daf4dfa081273c7729c045bf1dcc`. Read these implementation excerpts; they are not a substitute runtime.

## fastgen/utils/scripts.py:22

```text
def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument(
        "--config", required=True, help="Path to the config Python file, relative to the working directory"
    )
    parser.add_argument("--log_level", default="INFO", help="Log level (e.g. DEBUG, INFO)")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Config overrides at the end of the command, after a single '-': `- path.key=value ...` (Hydra syntax).",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dry run without training. Useful for debugging the config.",
    )
    args = parser.parse_args()
    return args
```

## fastgen/utils/scripts.py:100

```text
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
    config.model.inference = evaluation
    if evaluation:
        # Training configs may disable these when their conditions are precomputed.
        logger.info("Enabling preprocessors for evaluation")
        config.model.enable_preprocessors = True
    return config
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
