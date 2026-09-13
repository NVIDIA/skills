# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standard Hugging Face Trainer FedAvg job template.

Copy this file beside ``client.py``, ``server_model.py``, and packaged
project-local modules such as ``model.py``. Adapt the client argument names and
capability-gated recipe options to the source project. Use
``resolve_source_local_path`` only for an inspected source argument that names a
local file or directory, and pass the inspected source-project root explicitly;
do not use it for Hub identifiers, URLs, or site-local paths. Keep the required
Recipe, packaging, and execution structure; do not replace local file names
with parent traversal paths.
"""

import argparse
import math
import os
import tempfile
from contextlib import contextmanager
from numbers import Integral, Real
from pathlib import Path

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, set_per_site_config

DEFAULT_MAX_STEPS = 10
SOURCE_DIR = Path(__file__).resolve().parent


def resolve_source_local_path(value: str | Path, *, source_root: str | Path) -> Path:
    """Resolve a local path against its explicit inspected source-project root."""
    path = Path(value)
    if path.is_absolute():
        return path.resolve()

    root = Path(source_root).expanduser()
    if not root.is_absolute():
        raise ValueError("source_root must be an absolute path")
    return (root / path).resolve()


@contextmanager
def _source_directory():
    """Resolve Recipe resources beside this job without retaining a cwd change."""
    original_cwd = Path.cwd()
    os.chdir(SOURCE_DIR)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def _simulation_workspace(workspace_root):
    """Return an explicit workspace or a private one that preserves run results."""
    if workspace_root is not None:
        return workspace_root
    return Path(tempfile.mkdtemp(prefix="nvflare-hf-trainer-"))


def _token(value, name: str) -> str:
    text = str(value)
    if not text or any(char.isspace() for char in text):
        raise ValueError(f"{name} must be a non-empty whitespace-free value")
    return text


def _positive_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float_arg(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def build_train_args(
    model_name_or_path: str,
    num_clients: int,
    *,
    max_steps: int | None = None,
    num_train_epochs: float | None = None,
    preserve_source_budget: bool = False,
) -> str:
    """Build arguments consumed by the generated client's strict parser."""
    if preserve_source_budget and (max_steps is not None or num_train_epochs is not None):
        raise ValueError("preserve_source_budget cannot be combined with max_steps or num_train_epochs")
    if max_steps is not None and num_train_epochs is not None:
        raise ValueError("specify only one of max_steps or num_train_epochs")
    if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, Integral) or max_steps <= 0):
        raise ValueError("max_steps must be a positive integer")
    if num_train_epochs is not None and (
        isinstance(num_train_epochs, bool)
        or not isinstance(num_train_epochs, Real)
        or not math.isfinite(num_train_epochs)
        or num_train_epochs <= 0
    ):
        raise ValueError("num_train_epochs must be a finite positive number")

    args = [
        "--model_name_or_path",
        _token(model_name_or_path, "model_name_or_path"),
        "--num_clients",
        str(num_clients),
    ]
    if not preserve_source_budget:
        if max_steps is None and num_train_epochs is None:
            max_steps = DEFAULT_MAX_STEPS
        if max_steps is not None:
            args.extend(("--max_steps", str(max_steps)))
        else:
            args.extend(("--num_train_epochs", str(num_train_epochs)))
    return " ".join(args)


def build_recipe(
    *,
    name: str,
    model_name_or_path: str,
    num_clients: int,
    num_rounds: int,
    key_metric: str = "",
    max_steps: int | None = None,
    num_train_epochs: float | None = None,
    preserve_source_budget: bool = False,
    per_site_config: dict[str, dict] | None = None,
    recipe_options: dict | None = None,
):
    """Build FedAvg using only options confirmed by ``recipe show``."""
    train_args = build_train_args(
        model_name_or_path,
        num_clients,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        preserve_source_budget=preserve_source_budget,
    )
    options = dict(recipe_options or {})
    if "per_site_config" in options:
        raise ValueError(
            "pass per_site_config to build_recipe instead of using the deprecated recipe constructor option"
        )
    if isinstance(per_site_config, dict) and any(
        isinstance(site_config, dict) and "train_script" in site_config for site_config in per_site_config.values()
    ):
        raise ValueError(
            "per_site_config must use the shared train_script='client.py'; "
            "site-specific train_script overrides are not supported by this job template"
        )

    with _source_directory():
        recipe = FedAvgRecipe(
            name=name,
            model={
                "class_path": "server_model.ServerModel",
                "args": {"model_name_or_path": model_name_or_path},
            },
            min_clients=num_clients,
            num_rounds=num_rounds,
            train_script="client.py",
            train_args=train_args,
            key_metric=key_metric,
            **options,
        )
        if per_site_config is not None:
            set_per_site_config(recipe, per_site_config)
        recipe.add_server_file(str(SOURCE_DIR / "server_model.py"))
        recipe.add_server_file(str(SOURCE_DIR / "model.py"))
        recipe.add_client_file(str(SOURCE_DIR / "client.py"))
        recipe.add_client_file(str(SOURCE_DIR / "model.py"))
    return recipe


def build_sim_env(
    num_clients: int,
    workspace_root: Path,
    *,
    per_site_config: dict[str, dict] | None = None,
) -> SimEnv:
    """Build a simulator environment with exactly one client-topology owner."""
    if per_site_config is None:
        return SimEnv(
            num_clients=num_clients,
            workspace_root=str(workspace_root),
        )

    clients = list(per_site_config)
    return SimEnv(
        num_clients=num_clients,
        clients=clients,
        workspace_root=str(workspace_root),
    )


def main():
    # Importing the Recipe API before parsing lets it consume --export and
    # --export-dir; this parser owns only the generated job's local options.
    parser = argparse.ArgumentParser(description="Hugging Face Trainer FedAvg job", allow_abbrev=False)
    parser.add_argument("--name", default="hf-trainer-fedavg")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=2)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--max_steps", type=_positive_int_arg)
    budget.add_argument("--num_train_epochs", type=_positive_float_arg)
    budget.add_argument("--preserve_source_budget", action="store_true")
    parser.add_argument(
        "--key_metric",
        default="",
        help="exact higher-is-better server metric key; leave empty to disable best-model selection",
    )
    parser.add_argument(
        "--workspace_root",
        type=Path,
        default=None,
        help="simulation workspace; defaults to a new private temporary directory retained for result inspection",
    )
    args = parser.parse_args()

    recipe = build_recipe(
        name=args.name,
        model_name_or_path=args.model_name_or_path,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        preserve_source_budget=args.preserve_source_budget,
        key_metric=args.key_metric,
    )
    workspace_root = _simulation_workspace(args.workspace_root)
    run = recipe.execute(build_sim_env(args.num_clients, workspace_root))
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result())


if __name__ == "__main__":
    main()
