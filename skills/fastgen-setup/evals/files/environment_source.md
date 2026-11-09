# FastGen source fixture

Source revision: `e0a92619fc56daf4dfa081273c7729c045bf1dcc`. Read these implementation excerpts; they are not a substitute runtime.

## fastgen/utils/env.py:24

```text
def output_root() -> str:
    """Return the absolute root for FastGen checkpoints, logs, caches, and datasets."""
    return os.path.abspath(os.getenv("FASTGEN_OUTPUT_ROOT", "").strip() or "FASTGEN_OUTPUT")
```

## fastgen/utils/env.py:103

```text
def _load_token(path: str, env_key: str, override: bool = True) -> None:
    """Set ``env_key`` from the single-line token file ``path`` unless the file is empty.

    With ``override=False`` an already set, non-empty ``env_key`` is kept; the file is still validated.
    """
    with open(path, encoding="utf-8") as credential_file:
        token = credential_file.read().strip()
    if not token:
        return
    if "\n" in token or "\r" in token:
        raise ValueError(f"{path} must hold {env_key} on a single line.")
    if not override and os.environ.get(env_key):
        logger.info(f"Keeping the preset {env_key}; ignoring {path}")
        return
    os.environ[env_key] = token
    logger.info(f"Loading {env_key} from {path}")
```

## fastgen/utils/env.py:121

```text
def set_env_vars(credentials_dir: str | None = "./credentials", override: bool = True) -> None:
    """Load optional credentials from ``credentials_dir`` into the environment and configure the HF cache.

    ``wandb_api.txt``, ``hf_token.txt``, and ``vlm_api_token.txt`` hold ``WANDB_API_KEY``, ``HF_TOKEN``, and
    ``OPENAI_API_KEY``; ``s3.json`` holds the AWS/S3 credentials. A non-empty file overrides the existing value,
    and a missing or empty file leaves the environment alone, so this function stays safe in CI and standalone
    scripts. ``override=False`` inverts the precedence: a variable that is already set and non-empty (e.g. a
    secret injected by the scheduler) is kept, and files only fill in the missing ones.

    Raises:
        ValueError: If a token file spans several lines.
    """
    if credentials_dir:
        s3_path = os.path.join(credentials_dir, "s3.json")
        if os.path.isfile(s3_path):
            _load_aws_credentials(s3_path, override=override)
        for filename, env_key in (
            ("wandb_api.txt", "WANDB_API_KEY"),
            ("hf_token.txt", "HF_TOKEN"),
            ("vlm_api_token.txt", "OPENAI_API_KEY"),
        ):
            path = os.path.join(credentials_dir, filename)
            if os.path.isfile(path):
                _load_token(path, env_key, override=override)

    # A launcher may supply ``HF_HOME`` relative; it is inherited by subprocesses and
    # passed to Hugging Face as ``cache_dir``, so pin it rather than forking the cache.
    os.environ["HF_HOME"] = os.path.abspath(os.getenv("HF_HOME", os.path.join(output_root(), ".cache")))
```

## fastgen/utils/env.py:29

```text
def hf_load(loader: Callable[..., Any], model_id_or_local_path: str, *args: Any, **kwargs: Any) -> Any:
    """Call a Hugging Face ``loader`` such as ``from_pretrained`` or ``hf_hub_download`` with FastGen's defaults.

    ``HF_HOME`` and ``LOCAL_FILES_ONLY`` supply ``cache_dir`` and ``local_files_only`` unless passed explicitly.
    Hub access failures (gated, private, or misspelled repos) are re-raised as a ``RuntimeError`` listing the
    license and authentication steps; diffusers otherwise reports them as a misleading "is not a local folder".
    """
    if "HF_HOME" in os.environ:
        kwargs.setdefault("cache_dir", os.environ["HF_HOME"])
    kwargs.setdefault("local_files_only", str2bool(os.getenv("LOCAL_FILES_ONLY", "false")))
    try:
        return loader(model_id_or_local_path, *args, **kwargs)
    except OSError as e:
        # diffusers and transformers re-raise Hub errors as a plain OSError but keep the original in the chain.
        if not any(s in str(e).lower() for s in ("gated", "401", "is not a local folder")):
            raise
        # Lazy: importing huggingface_hub fixes its HF_HOME-derived paths, which set_env_vars may still change.
        from huggingface_hub.errors import GatedRepoError

        gated_error = e
        while gated_error is not None and not isinstance(gated_error, GatedRepoError):
            gated_error = gated_error.__cause__ or gated_error.__context__
        license_step = f"accept the license at https://huggingface.co/{model_id_or_local_path}, " if gated_error else ""
        raise RuntimeError(
            f"Could not load {model_id_or_local_path!r} from Hugging Face. To fix: {license_step}"
            "authenticate with `hf auth login` or `export HF_TOKEN=...` (visible to all torchrun ranks), "
            f"check the spelling of the model ID or local path, and relaunch.\nOriginal error: {e}"
        ) from e
```

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

## [fastgen/configs/config.py:361](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/configs/config.py#L361)

```python
    # Folder with the optional credential files: wandb_api.txt, hf_token.txt, vlm_api_token.txt, and s3.json.
    credentials_dir: str = "./credentials"
```

## [fastgen/utils/scripts.py:174](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/utils/scripts.py#L174)

```python
    # Configure credentials and the Hugging Face cache before model/trainer initialization.
    set_env_vars(config.credentials_dir)
```
