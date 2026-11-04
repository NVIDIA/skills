"""Generate inference and evaluation config files from a pipeline config.

Given a base pipeline config (e.g., search_binder_pipeline.yaml), this script
applies sweep combinations from a YAML sweeper file and/or CLI overrides, then
writes one inference + evaluation config pair per combination.

Sweep axes (from --sweeper YAML file) are combined via cartesian product.
Overrides (from --override KEY=VAL) are applied to every generated config.
If a key appears in both, the override wins (collapses that sweep axis).

Usage:
    # No sweep, single config
    python generate_inference_configs.py \\
        --config_name search_binder_pipeline --run_name my_run

    # Override target
    python generate_inference_configs.py \\
        --config_name search_binder_pipeline --run_name my_run \\
        --override generation.task_name=22_DerF21

    # Sweep from YAML file
    python generate_inference_configs.py \\
        --config_name search_binder_pipeline --run_name my_run \\
        --sweeper configs/sweeps/beam_width.yaml

    # Sweep + override (override pins a value across all sweep combos)
    python generate_inference_configs.py \\
        --config_name search_binder_pipeline --run_name my_run \\
        --sweeper configs/sweeps/beam_width.yaml \\
        --override generation.task_name=22_DerF21 generation.args.nsteps=400

    # Dry run to preview
    python generate_inference_configs.py \\
        --config_name search_binder_pipeline --run_name my_run \\
        --sweeper configs/sweeps/beam_width.yaml --dryrun
"""

import argparse
import itertools
import logging
import math
import os
from typing import Any

import hydra
import yaml
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


# =============================================================================
# Value Parsing
# =============================================================================


def parse_scalar(value: str) -> Any:
    """Parse a string into a typed Python scalar.

    Supports int, float, bool (true/false), None (null), and falls back to str.
    """
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null" or value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_override(entry: str) -> tuple[str, Any]:
    """Parse a single KEY=VAL override string into (key, typed_value).

    Args:
        entry: String in the form "dotted.key=value".

    Returns:
        Tuple of (key, parsed_value).

    Raises:
        ValueError: If entry has no '=' or empty key/value.
    """
    if "=" not in entry:
        raise ValueError(f"Override must be KEY=VALUE, got: {entry!r}")
    key, _, raw_value = entry.partition("=")
    key = key.strip()
    raw_value = raw_value.strip()
    if not key:
        raise ValueError(f"Empty key in override: {entry!r}")
    if not raw_value:
        raise ValueError(f"Empty value in override: {entry!r}")
    return key, parse_scalar(raw_value)


# =============================================================================
# Sweeper Construction
# =============================================================================


def load_sweeper_file(path: str) -> dict[str, list]:
    """Load a sweeper YAML file and validate that all values are lists.

    Scalar values are automatically wrapped in a single-element list.

    Returns:
        Dictionary mapping dot-notation keys to lists of values.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sweeper file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Sweeper YAML must be a mapping, got {type(raw).__name__}")

    sweep_dict: dict[str, list] = {}
    for key, val in raw.items():
        if not isinstance(key, str):
            raise ValueError(f"Sweeper keys must be strings, got {type(key).__name__}: {key}")
        sweep_dict[key] = val if isinstance(val, list) else [val]
    return sweep_dict


def build_sweeper(
    sweeper_file: str | None = None,
    overrides: list[str] | None = None,
) -> tuple[dict[str, list], dict[str, Any]]:
    """Build sweep dict and override dict from CLI inputs.

    If a key appears in both the sweeper file and overrides, the override wins
    and that key is removed from the sweep (collapsing the axis to one value).

    Args:
        sweeper_file: Path to a YAML sweep file (optional).
        overrides: List of "KEY=VAL" strings (optional).

    Returns:
        (sweep_dict, override_dict) where sweep_dict has list values and
        override_dict has scalar values.
    """
    sweep_dict: dict[str, list] = {}
    if sweeper_file:
        sweep_dict = load_sweeper_file(sweeper_file)

    override_dict: dict[str, Any] = {}
    for entry in overrides or []:
        key, value = parse_override(entry)
        override_dict[key] = value

    # Override wins: remove conflicting keys from sweep
    for key in override_dict:
        sweep_dict.pop(key, None)

    return sweep_dict, override_dict


# =============================================================================
# Config Helpers
# =============================================================================


def dot_key_dict_to_nested_dict(input_dict: dict) -> dict:
    """Convert dictionary with dot-notation keys to a nested dictionary.

    Example::

        {"A.B.x": 1, "A.B.y": 2, "C": 3}
        -> {"A": {"B": {"x": 1, "y": 2}}, "C": 3}
    """
    output_dict: dict = {}
    child_keys: set = set()

    for key, value in input_dict.items():
        if "." not in key:
            output_dict[key] = value
        else:
            child_keys.add(key.split(".")[0])

    for child_key in child_keys:
        child_input = {".".join(k.split(".")[1:]): v for k, v in input_dict.items() if k.startswith(child_key + ".")}
        child_output = dot_key_dict_to_nested_dict(child_input)

        if child_key in output_dict:
            assert isinstance(output_dict[child_key], dict), (
                f"Key {child_key} should map to dict, got {type(output_dict[child_key])}"
            )
            output_dict[child_key].update(child_output)
        else:
            output_dict[child_key] = child_output

    return output_dict


def get_task_name_from_config(cfg) -> str | None:
    """Extract task_name from a pipeline config.

    Checks generation.task_name first, then
    generation.dataloader.dataset.task_name.
    """
    try:
        if "generation" in cfg and "task_name" in cfg.generation:
            return cfg.generation.task_name
    except Exception:
        pass
    try:
        if "generation" in cfg and "dataloader" in cfg.generation:
            ds = cfg.generation.dataloader.dataset
            if "task_name" in ds:
                return ds.task_name
    except Exception:
        pass
    return None


def create_eval_config(infer_cfg) -> OmegaConf:
    """Derive an evaluation config from a pipeline config.

    Deep-copies the pipeline config and sets the evaluation/analysis output
    paths based on the inference root_path.
    """
    eval_cfg = OmegaConf.create(OmegaConf.to_container(infer_cfg, resolve=False))
    OmegaConf.set_struct(eval_cfg, False)

    root_path = infer_cfg.get("root_path", "./inference/default")

    eval_output_dir = root_path.replace("./inference/inf_", "./evaluation_results/eval_")
    if eval_output_dir == root_path:
        eval_output_dir = root_path.replace("inference", "evaluation_results", 1)

    eval_cfg["sample_storage_path"] = root_path
    eval_cfg["output_dir"] = eval_output_dir
    eval_cfg["results_dir"] = eval_output_dir
    eval_cfg["eval_njobs"] = infer_cfg.get("eval_njobs", infer_cfg.get("gen_njobs", 1))

    task_name = get_task_name_from_config(infer_cfg)
    if task_name:
        if "dataset" not in eval_cfg:
            eval_cfg["dataset"] = {}
        eval_cfg["dataset"]["task_name"] = task_name
        if "generation" in infer_cfg and "target_dict_cfg" in infer_cfg.generation:
            eval_cfg["dataset"]["target_dict_cfg"] = infer_cfg.generation.target_dict_cfg

    OmegaConf.set_struct(eval_cfg, True)
    return eval_cfg


# =============================================================================
# Core: Apply Sweep + Overrides and Save Configs
# =============================================================================


def apply_sweeper_and_save_configs(
    base_cfg,
    sweep_dict: dict[str, list],
    override_dict: dict[str, Any],
    infer_output_dir: str,
    eval_output_dir: str,
    run_name: str | None = None,
    start_index: int = 0,
) -> int:
    """Apply cartesian-product sweep and per-config overrides, then save.

    For each combination in the cartesian product of ``sweep_dict`` values,
    merges the combination into ``base_cfg``, applies ``override_dict`` on top,
    sets output paths, derives the eval config, and saves both YAML files.

    Args:
        base_cfg: Base pipeline OmegaConf config.
        sweep_dict: ``{dotted_key: [val1, val2, ...]}`` for sweep axes.
        override_dict: ``{dotted_key: scalar}`` applied to every config.
        infer_output_dir: Directory for generated inference configs.
        eval_output_dir: Directory for generated evaluation configs.
        run_name: Optional run name suffix for config filenames.
        start_index: Starting index for sequential config numbering.

    Returns:
        Number of config pairs generated.
    """
    # Build cartesian product of sweep axes
    keys = list(sweep_dict.keys())
    value_lists = [sweep_dict[k] for k in keys]

    if keys:
        combinations = [dict(zip(keys, combo, strict=False)) for combo in itertools.product(*value_lists)]
    else:
        combinations = [{}]

    os.makedirs(infer_output_dir, exist_ok=True)
    os.makedirs(eval_output_dir, exist_ok=True)

    for i, combo in enumerate(combinations):
        # Deep-copy base config so we never mutate the caller's object
        cfg_copy = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))

        # Merge sweep combination into the copy
        nested_combo = dot_key_dict_to_nested_dict(combo)
        override_cfg = OmegaConf.create(nested_combo)

        OmegaConf.set_struct(cfg_copy, False)
        infer_cfg = OmegaConf.merge(cfg_copy, override_cfg)

        # Apply scalar overrides on top
        for key, value in override_dict.items():
            OmegaConf.update(infer_cfg, key, value, merge=True)

        # Build filename: {idx}[_{run_name}]
        # Task name is intentionally NOT included — the bash launcher builds
        # CONFIG_SUFFIX from its own CLI args and must be able to predict
        # the filename without inspecting config content.
        idx = start_index + i
        name_parts = [str(idx)]
        if run_name:
            name_parts.append(run_name)
        base_name = "_".join(name_parts)

        # Set inference output path
        infer_cfg["root_path"] = f"./inference/inf_{base_name}"
        OmegaConf.set_struct(infer_cfg, True)

        # Derive eval config
        eval_cfg = create_eval_config(infer_cfg)

        # Save both configs
        inf_filename = f"inf_{base_name}.yaml"
        eval_filename = f"eval_{base_name}.yaml"
        OmegaConf.save(config=infer_cfg, f=os.path.join(infer_output_dir, inf_filename))
        OmegaConf.save(config=eval_cfg, f=os.path.join(eval_output_dir, eval_filename))

        logger.info(f"Generated: {inf_filename}, {eval_filename}")

    return len(combinations)


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Separated for testability."""
    parser = argparse.ArgumentParser(
        description="Generate inference and evaluation config files from a pipeline config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Name of the pipeline config file (without .yaml)",
    )
    parser.add_argument(
        "--infer_dir_cfgs",
        type=str,
        default="./configs/inference_configs",
        help="Directory to write generated inference config files",
    )
    parser.add_argument(
        "--eval_dir_cfgs",
        type=str,
        default="./configs/eval_configs",
        help="Directory to write generated evaluation config files",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name suffix embedded in config filenames",
    )
    parser.add_argument(
        "--sweeper",
        type=str,
        default=None,
        help="Path to a YAML file defining sweep axes (cartesian product)",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="One or more KEY=VAL overrides applied to every generated config",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Show what would be generated without creating files",
    )
    return parser


def main(argv: list[str] | None = None):
    """Entry point for config generation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = build_parser()
    args = parser.parse_args(argv)

    # Build sweep + overrides
    sweep_dict, override_dict = build_sweeper(args.sweeper, args.override)

    # Compute expected config count
    if sweep_dict:
        config_count = math.prod(len(v) for v in sweep_dict.values())
    else:
        config_count = 1

    # Log configuration summary
    logger.info(f"Pipeline config: {args.config_name}")
    if sweep_dict:
        logger.info(f"Sweep axes ({config_count} combinations):")
        for key, vals in sweep_dict.items():
            logger.info(f"  {key}: {vals}")
    else:
        logger.info("No sweep axes — generating single config")
    if override_dict:
        logger.info("Overrides (applied to every config):")
        for key, val in override_dict.items():
            logger.info(f"  {key} = {val!r}")

    if args.dryrun:
        logger.info(f"DRY RUN — would generate {config_count} config pair(s)")
        return

    # Load pipeline config via Hydra
    base_config_path = "../configs"
    with hydra.initialize(base_config_path, version_base=hydra.__version__):
        base_cfg = hydra.compose(config_name=args.config_name)
        logger.info(f"Loaded config: {args.config_name}")

    # Generate configs
    num_generated = apply_sweeper_and_save_configs(
        base_cfg=base_cfg,
        sweep_dict=sweep_dict,
        override_dict=override_dict,
        infer_output_dir=args.infer_dir_cfgs,
        eval_output_dir=args.eval_dir_cfgs,
        run_name=args.run_name,
        start_index=0,
    )

    logger.info(f"Generated {num_generated} config pair(s)")
    logger.info(f"Inference configs: {args.infer_dir_cfgs}")
    logger.info(f"Evaluation configs: {args.eval_dir_cfgs}")


if __name__ == "__main__":
    main()
