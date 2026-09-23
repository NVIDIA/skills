#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Initialize the disk-backed Cosmos3 DEFT AOI run state."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import sys
import tempfile
from typing import Any

from metric_contract import render_target, validate_contract
from render_report import render as render_html_report


WORKFLOW = "tao-run-deft-aoi-cosmos3"
STAGES = [
    "train",
    "evaluate_benchmark",
    "benchmark_metrics",
    "evaluate_proxy",
    "proxy_rcca",
    "routing",
    "anomalygen",
    "data_mining",
    "assemble_data",
    "validate_data",
    "loop_stop",
]
STATUSES = ["pending", "in_progress", "complete", "failed"]
# Language-side projections only; the vision tower keeps its pretrained
# weights. The schema also accepts "all-linear", which additionally adapts the
# vision linear layers — use it only when explicitly requested.
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _resolve_image_from_versions_yaml(*path: str) -> str | None:
    skill_bank = os.environ.get("TAO_SKILL_BANK_PATH")
    if not skill_bank:
        return None
    versions = pathlib.Path(skill_bank) / "versions.yaml"
    if not versions.is_file():
        return None
    try:
        import yaml
    except ImportError:
        return None
    try:
        node: Any = yaml.safe_load(versions.read_text())
        for key in path:
            node = node[key]
        return str(node)
    except (KeyError, TypeError, yaml.YAMLError):
        return None


DEFAULT_COSMOS_IMAGE = os.environ.get(
    "COSMOS_RL_IMAGE"
) or _resolve_image_from_versions_yaml("images", "tao_toolkit", "cosmos_rl")
DEFAULT_MINING_IMAGE = os.environ.get(
    "TAO_DS_IMAGE"
) or _resolve_image_from_versions_yaml("images", "tao_toolkit", "data_services")
DEFAULT_ANOMALYGEN_IMAGE = os.environ.get(
    "AG_IMAGE"
) or _resolve_image_from_versions_yaml(
    "images", "metropolis_sdg", "paidf_anomalygen"
)
BASE_MODEL_ALIASES = {
    "nano": "nvidia/Cosmos3-Nano",
    "cosmos3-nano": "nvidia/Cosmos3-Nano",
    "nvidia/cosmos3-nano": "nvidia/Cosmos3-Nano",
    "edge": "nvidia/Cosmos3-Edge",
    "cosmos3-edge": "nvidia/Cosmos3-Edge",
    "nvidia/cosmos3-edge": "nvidia/Cosmos3-Edge",
    "super": "nvidia/Cosmos3-Super",
    "cosmos3-super": "nvidia/Cosmos3-Super",
    "nvidia/cosmos3-super": "nvidia/Cosmos3-Super",
}
SUPPORTED_BASE_MODELS = tuple(dict.fromkeys(BASE_MODEL_ALIASES.values()))


def canonicalize_base_model(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("--base-model must not be empty")
    return BASE_MODEL_ALIASES.get(stripped.lower(), stripped)


def validate_base_model(value: str) -> str:
    canonical = canonicalize_base_model(value)
    if canonical not in SUPPORTED_BASE_MODELS:
        allowed = ", ".join(SUPPORTED_BASE_MODELS)
        raise ValueError(
            f"unsupported --base-model {value!r}; allowed values: {allowed} "
            "(aliases: nano, edge, super)"
        )
    return canonical


def _absolute_executable(path: pathlib.Path | str) -> pathlib.Path:
    """Make an interpreter path absolute without resolving a venv symlink."""
    return pathlib.Path(os.path.abspath(pathlib.Path(path).expanduser()))


def _resolve_specs(
    workspace: pathlib.Path, args: argparse.Namespace
) -> dict[str, pathlib.Path]:
    """Resolve the train/proxy/benchmark spec paths.

    Two layouts are valid. A per-role evaluate spec
    (``evaluate_spec_proxy.toml`` / ``evaluate_spec_benchmark.toml``) is
    preferred: each file is concrete, already carrying its own
    ``annotation_path``, ``results_dir``, and ``save_folder``, so no job has to
    mutate a shared file at launch and a Proxy job cannot pick up the Benchmark
    annotation. A single ``evaluate_spec.toml`` is still accepted and is then
    materialized per stage. Explicit flags override both.
    """
    specs_dir = workspace / "specs"
    shared_evaluate = specs_dir / "evaluate_spec.toml"

    def evaluate_for(role: str, override: pathlib.Path | None) -> pathlib.Path:
        if override is not None:
            return override.expanduser().resolve()
        per_role = specs_dir / f"evaluate_spec_{role}.toml"
        return (per_role if per_role.is_file() else shared_evaluate).resolve()

    train = (
        args.train_spec.expanduser().resolve()
        if args.train_spec is not None
        else (specs_dir / "train_spec.toml").resolve()
    )
    return {
        "train": train,
        "proxy": evaluate_for("proxy", args.proxy_spec),
        "benchmark": evaluate_for("benchmark", args.benchmark_spec),
    }


def _anomalygen_path(
    override: pathlib.Path | None, default: pathlib.Path
) -> pathlib.Path:
    """Prefer an explicit path; resolve symlinks so the record is mountable."""
    return (override or default).expanduser().resolve()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_contract(
    *,
    results_dir: pathlib.Path,
    metric: str,
    threshold: float,
) -> dict[str, Any]:
    display_names = {
        "recall_ng": "NG recall",
        "precision_ng": "NG precision",
        "f1_ng": "NG F1",
        "accuracy": "Accuracy",
    }
    return validate_contract(
        {
            "name": metric,
            "display_name": display_names[metric],
            "operator": ">=",
            "target": threshold,
            "unit": "",
            "evaluator": {
                "type": "artifact",
                "producer": "scripts/analyze_gaps.py",
                "path_template": str(
                    results_dir
                    / "{iter_label}"
                    / "benchmark_metrics"
                    / "metric_result.json"
                ),
            },
            "constraints": [
                {
                    "name": "unknown_predictions",
                    "display_name": "Unknown predictions",
                    "operator": "<=",
                    "target": 0,
                    "unit": "",
                }
            ],
        }
    )


def build_state(args: argparse.Namespace) -> dict[str, Any]:
    workspace = args.workspace.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    annotations = {
        "proxy": (args.proxy_annotations or workspace / "annotations/proxy_kpi.json")
        .expanduser()
        .resolve(),
        "benchmark": (
            args.benchmark_annotations
            or workspace / "annotations/benchmark_kpi.json"
        )
        .expanduser()
        .resolve(),
        "mining": (
            args.mining_annotations
            or workspace / "annotations/mining_pool.json"
        )
        .expanduser()
        .resolve(),
    }
    missing = [f"{role}={path}" for role, path in annotations.items() if not path.is_file()]
    if missing:
        raise ValueError("annotation file(s) missing: " + ", ".join(missing))

    # Specs are staged before state is initialized. Checking them here keeps the
    # failure recoverable: state is written exactly once and must never be
    # hand-edited, so do not persist paths that are already known to be absent.
    specs = _resolve_specs(workspace, args)
    missing_specs = [
        f"{role}={path}" for role, path in specs.items() if not path.is_file()
    ]
    if missing_specs:
        raise ValueError(
            "spec file(s) missing: "
            + ", ".join(missing_specs)
            + ". Build them from the tao-finetune-cosmos-reason templates and "
            "stage them before initializing state."
        )
    contract = _metric_contract(
        results_dir=results_dir,
        metric=args.kpi_metric,
        threshold=args.kpi_threshold,
    )
    benchmark_hash = _sha256(annotations["benchmark"])
    media_root = (args.media_root or workspace).expanduser().resolve()
    base_model = validate_base_model(args.base_model)
    network_mode = getattr(args, "network_mode", None) or (
        "airgap" if os.environ.get("AIR_GAPPED") == "1" else "network-enabled"
    )
    network_source = getattr(args, "network_mode_source", None) or (
        "environment:AIR_GAPPED" if os.environ.get("AIR_GAPPED") == "1" else "default"
    )
    python_executable = _absolute_executable(
        getattr(args, "python_executable", None) or sys.executable
    )
    offline = network_mode == "airgap"

    return {
        "version": 5,
        "workflow": WORKFLOW,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "status": "in_progress",
        "kpi_target": render_target(contract),
        "metric_contract": contract,
        "results_dir": str(results_dir),
        "max_iterations": args.max_iterations,
        "current_iteration": 0,
        "execution_policy": {
            "network_mode": network_mode,
            "activation_source": network_source,
            "allow_package_install": not offline,
            "allow_remote_fetch": not offline,
            "allow_container_pull": not offline,
            "allow_registry_login": not offline,
            "python_launcher": "scripts/deft_python.sh",
            "python_executable": str(python_executable),
            "hf_offline": offline,
        },
        "config": {
            "workspace": str(workspace),
            "platform": args.platform,
            "model_skill": "tao-finetune-cosmos-reason",
            "base_model": base_model,
            "automl_policy": "off",
            "annotation_mode": "bare_okng",
            "media_root": str(media_root),
            "annotations": {role: str(path) for role, path in annotations.items()},
            "evaluation": {
                "proxy": {
                    "annotations": str(annotations["proxy"]),
                    "drives_rcca": True,
                    "drives_loop_stop": False,
                },
                "benchmark": {
                    "annotations": str(annotations["benchmark"]),
                    "sha256": benchmark_hash,
                    "drives_rcca": False,
                    "drives_loop_stop": True,
                },
            },
            "specs": {role: str(path) for role, path in specs.items()},
            "containers": {
                "cosmos_rl": args.cosmos_container,
                "data_services": args.mining_container,
            },
            "training": {
                "annotation_source": "generated_from_mining_and_anomalygen",
                "num_gpus": args.num_gpus,
                "num_nodes": args.num_nodes,
                "gpu_model": args.gpu_model,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "lora": {
                    "r": args.lora_r,
                    "alpha": args.lora_alpha,
                    "target_modules": list(DEFAULT_LORA_TARGET_MODULES),
                },
            },
            "mining": {
                "top_k_per_target": args.top_k_per_target,
                "metric": "cosine",
                "min_similarity": args.min_similarity,
                "history_aware": {
                    "enabled": True,
                    "identity": "filepath",
                    "history_file": str(results_dir / "mining_history.json"),
                },
            },
            "anomalygen": {
                "sub_skill": "tao-generate-anomalies",
                # The DEFT loop only needs Phases 2-3 (AMP routing + SDG
                # diffusion); Phases 4-7 are SDG-quality optimization and
                # contribute no training pairs.
                "mode": "inference_only",
                "num_search_run": 0,
                "nn_threshold": 0,
                "project": args.anomalygen_project,
                "num_SDG": args.num_sdg,
                "container": args.anomalygen_container,
                # Explicit flags win; otherwise derive the workspace
                # convention. Resolve through symlinks so the recorded path is
                # the real one — a symlinked subtree under the workspace
                # dangles inside the container when only $WS is mounted.
                "checkpoint_dir": str(_anomalygen_path(
                    args.anomalygen_checkpoint_dir,
                    workspace
                    / "augmentation/anomalygen/checkpoints"
                    / args.anomalygen_project,
                )),
                "dataset_dir": str(_anomalygen_path(
                    args.anomalygen_dataset_dir,
                    workspace
                    / "augmentation/anomalygen/datasets"
                    / args.anomalygen_project,
                )),
                "defect_spec": str(_anomalygen_path(
                    args.anomalygen_dataset_dir,
                    workspace
                    / "augmentation/anomalygen/datasets"
                    / args.anomalygen_project,
                ) / "defect_spec.jsonl"),
                "cosmos_models_dir": str(_anomalygen_path(
                    args.cosmos_models_dir,
                    workspace / "augmentation/anomalygen/base_checkpoints",
                )),
                "label": "NG",
            },
        },
        "iterations": {},
        "events": [],
        "_completed_step_values": STAGES,
        "_status_values": STATUSES,
    }


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--workspace", required=True, type=pathlib.Path)
    parser.add_argument(
        "--network-mode",
        choices=("airgap", "network-enabled"),
        help="Immutable execution mode; defaults from AIR_GAPPED=1, otherwise network-enabled.",
    )
    parser.add_argument(
        "--network-mode-source",
        help="Human-readable source of the mode decision, such as ci-prompt or operator.",
    )
    parser.add_argument(
        "--python-executable",
        type=pathlib.Path,
        help="Dependency-complete Python selected during preflight; defaults to this interpreter.",
    )
    parser.add_argument("--platform", required=True)
    parser.add_argument("--max-iterations", required=True, type=int)
    parser.add_argument(
        "--kpi-metric",
        choices=("recall_ng", "precision_ng", "f1_ng", "accuracy"),
        default="recall_ng",
    )
    parser.add_argument("--kpi-threshold", type=float, default=1.0)
    parser.add_argument(
        "--base-model",
        default="nvidia/Cosmos3-Nano",
        help="Cosmos3 base model; Nano is default, Edge/Super require explicit selection.",
    )
    parser.add_argument("--media-root", type=pathlib.Path)
    parser.add_argument("--train-spec", type=pathlib.Path)
    parser.add_argument(
        "--proxy-spec",
        type=pathlib.Path,
        help="Proxy evaluate spec. Defaults to specs/evaluate_spec_proxy.toml, "
        "falling back to specs/evaluate_spec.toml.",
    )
    parser.add_argument(
        "--benchmark-spec",
        type=pathlib.Path,
        help="Benchmark evaluate spec. Defaults to "
        "specs/evaluate_spec_benchmark.toml, falling back to "
        "specs/evaluate_spec.toml.",
    )
    parser.add_argument("--proxy-annotations", type=pathlib.Path)
    parser.add_argument("--benchmark-annotations", type=pathlib.Path)
    parser.add_argument("--mining-annotations", type=pathlib.Path)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument(
        "--gpu-model",
        required=True,
        help=(
            "Exact accelerator model recorded by the selected platform's "
            "Preflight, including memory when available"
        ),
    )
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--top-k-per-target", type=int, default=5)
    parser.add_argument("--min-similarity", type=float, default=0.9)
    parser.add_argument("--cosmos-container", default=DEFAULT_COSMOS_IMAGE)
    parser.add_argument("--mining-container", default=DEFAULT_MINING_IMAGE)
    parser.add_argument("--anomalygen-container", default=DEFAULT_ANOMALYGEN_IMAGE)
    parser.add_argument(
        "--anomalygen-project",
        default="nvpcb",
        help="Directory label for this AnomalyGen project's checkpoint + dataset.",
    )
    parser.add_argument(
        "--num-sdg",
        type=int,
        default=20,
        help="Per-iteration synthetic defect budget, allocated across defect types.",
    )
    # Without these, assets outside the workspace convention need either a
    # symlink (which dangles inside the container) or a full copy.
    parser.add_argument(
        "--anomalygen-checkpoint-dir",
        type=pathlib.Path,
        help="Override the derived AnomalyGen checkpoint directory.",
    )
    parser.add_argument(
        "--anomalygen-dataset-dir",
        type=pathlib.Path,
        help="Override the derived AnomalyGen dataset directory.",
    )
    parser.add_argument(
        "--cosmos-models-dir",
        type=pathlib.Path,
        help="Override the derived Cosmos base-checkpoints cache directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.gpu_model = args.gpu_model.strip()
    if not args.gpu_model:
        print("init_deft_state: --gpu-model must not be empty", file=sys.stderr)
        return 2
    if args.network_mode is not None and args.network_mode_source is None:
        args.network_mode_source = "cli:--network-mode"
    if args.network_mode is None and args.network_mode_source is not None:
        print(
            "init_deft_state: --network-mode-source requires --network-mode",
            file=sys.stderr,
        )
        return 2
    if os.environ.get("AIR_GAPPED") == "1" and args.network_mode == "network-enabled":
        print(
            "init_deft_state: AIR_GAPPED=1 cannot be overridden by --network-mode network-enabled",
            file=sys.stderr,
        )
        return 2
    if args.python_executable is not None:
        executable = _absolute_executable(args.python_executable)
        if not executable.is_file() or not os.access(executable, os.X_OK):
            print(
                f"init_deft_state: --python-executable must be executable: {executable}",
                file=sys.stderr,
            )
            return 2
        args.python_executable = executable
    positive = {
        "max_iterations": args.max_iterations,
        "num_gpus": args.num_gpus,
        "num_nodes": args.num_nodes,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "top_k_per_target": args.top_k_per_target,
        "num_sdg": args.num_sdg,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        print(f"init_deft_state: positive values required: {invalid}", file=sys.stderr)
        return 2
    if not 0.0 <= args.kpi_threshold <= 1.0:
        print("init_deft_state: --kpi-threshold must be in [0, 1]", file=sys.stderr)
        return 2
    if not -1.0 <= args.min_similarity <= 1.0:
        print("init_deft_state: --min-similarity must be in [-1, 1]", file=sys.stderr)
        return 2
    if not args.cosmos_container or not args.mining_container:
        print(
            "init_deft_state: both container images are required; resolve "
            "images.tao_toolkit.{cosmos_rl,data_services} from versions.yaml",
            file=sys.stderr,
        )
        return 2
    output = args.results_dir.expanduser().resolve() / "deft_state.json"
    if output.exists():
        print(
            f"init_deft_state: refusing to overwrite existing state: {output}",
            file=sys.stderr,
        )
        return 2
    try:
        state = build_state(args)
        _atomic_json(output, state)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"init_deft_state: {exc}", file=sys.stderr)
        return 2
    print(f"init_deft_state: wrote {output}", file=sys.stderr)
    try:
        render_html_report(args.results_dir)
    except Exception as exc:  # noqa: BLE001 - state initialization remains valid
        print(f"init_deft_state: report hook failed: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
