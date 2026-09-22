# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initialize ${RESULTS_DIR}/deft_state.json with a guaranteed-unique key set.

Why this exists: earlier inline-dict writes drifted from the canonical schema
in `references/deft_state.json` and produced duplicate top-level keys (`kpi_target`, `results_dir`, `max_iterations`, `current_iteration`) — Python
3.12+ now emits a `SyntaxWarning` for these and the loop's resume logic reads
whichever copy parsing keeps, which is not stable across edits.

This script builds the dict with literal-once keys and writes the JSON. Atomic
write (tmp + os.replace). Refuses to overwrite an existing file unless `--force`
is passed — the resume path is supposed to read disk, not regenerate.

CLI:

    python scripts/init_deft_state.py \
        --results-dir ~/workspace/results/run_20260514_143000 \
        --workspace ~/workspace \
        --kpi-target "Quality score >= 0.9" \
        --metric-evaluator ~/workspace/metrics/evaluate_quality.py \
        --max-iterations 2 \
        --num-gpus 4 \
        --gpu-model "NVIDIA RTX PRO 6000 Blackwell (96 GB)" \
        --num-epochs 20 \
        --num-sdg 20 \
        --project nvpcb \
        --step 14000

The output schema mirrors `references/deft_state.json` exactly.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys
import tempfile

from metric_contract import parse_target_expression, render_target, validate_contract
from render_report import render as render_html_report


_COMPLETED_STEP_VALUES = [
    "evaluate",
    "rca",
    "anomalygen_finetune",
    "anomalygen",
    "routing",
    "data_mining",
    "data_merge",
    "train",
    "loop_stop",
]
_STATUS_VALUES = ["pending", "in_progress", "complete", "failed"]


def _absolute_executable(path: pathlib.Path | str) -> pathlib.Path:
    """Make an interpreter path absolute without resolving a venv symlink."""
    return pathlib.Path(os.path.abspath(pathlib.Path(path).expanduser()))


def _resolve_image_from_versions_yaml(*path: str) -> str | None:
    """Return a resolved image URI from versions.yaml at the given key path.

    Looks at TAO_SKILL_BANK_PATH (exported by the plugin's session_start
    hook). Returns None if the env var is unset, the file is missing, the
    key path is absent, or PyYAML is unavailable. In that case the caller
    must pass the corresponding CLI flag explicitly; the script intentionally
    has no hardcoded fallback tag so versions.yaml remains the single source
    of truth.
    """
    sb = os.environ.get("TAO_SKILL_BANK_PATH")
    if not sb:
        return None
    vy = pathlib.Path(sb) / "versions.yaml"
    if not vy.is_file():
        return None
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        data = yaml.safe_load(vy.read_text())
        node = data
        for p in path:
            node = node[p]
        return str(node)
    except (KeyError, TypeError, yaml.YAMLError):
        return None


_DEFAULT_TRAIN_CONTAINER = os.environ.get(
    "TAO_PYT_IMAGE"
) or _resolve_image_from_versions_yaml("images", "tao_toolkit", "pyt")
_DEFAULT_AG_CONTAINER = os.environ.get(
    "AG_IMAGE"
) or _resolve_image_from_versions_yaml(
    "images", "metropolis_sdg", "paidf_anomalygen"
)


def _resolve_anomalygen_checkpoint_dir(ws: pathlib.Path, project: str) -> pathlib.Path:
    base = ws / "augmentation" / "anomalygen" / "checkpoints" / project
    direct_config = base / "ag_config.yaml"
    direct_latest = base / "checkpoints" / "latest_checkpoint.txt"
    if direct_config.is_file() and direct_latest.is_file():
        return base
    search_base = base.resolve() if base.exists() else base
    candidates = sorted(
        path.parent
        for path in search_base.glob("**/ag_config.yaml")
        if (path.parent / "checkpoints" / "latest_checkpoint.txt").is_file()
        or len(
            [
                model
                for model in path.parent.glob("iter_[0-9]*.pt")
                if "_reg_model" not in model.name and model.is_file()
            ]
        )
        == 1
    )
    if len(candidates) == 1:
        return candidates[0]
    return base


def _resolve_workspace_images_dir(ws: pathlib.Path) -> pathlib.Path:
    """Resolve the canonical real-image root with legacy-layout fallback."""
    canonical = ws / "images"
    if canonical.is_dir():
        return canonical.resolve()
    legacy = ws / "kpi" / "images"
    if legacy.is_dir():
        return legacy.resolve()
    # Keep a deterministic canonical path so Pre-Flight reports the missing
    # input instead of silently recording the obsolete legacy location.
    return canonical.resolve()


def build_state(args: argparse.Namespace) -> dict:
    ws = args.workspace.resolve()
    rd = args.results_dir.resolve()
    ag_checkpoint_dir = _resolve_anomalygen_checkpoint_dir(ws, args.project)

    images_dir = _resolve_workspace_images_dir(ws)
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
    state = {
        "version": 4,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "status": "in_progress",
        "kpi_target": args.kpi_target_text,
        "metric_contract": args.metric_contract,
        "results_dir": str(rd),
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
            "specs_file": str(ws / "specs" / "baseline_spec.yaml"),
            "training_csv": str(ws / "train" / "base" / "training_set.csv"),
            "validation_csv": str(ws / "train" / "base" / "validation_set.csv"),
            "kpi_test_csv": str(ws / "kpi" / "testing_set.csv"),
            "images_dir": str(images_dir),
            "mining_pool_csv": str(
                ws / "augmentation" / "mining_pool" / "mining_pool.csv"
            ),
            "mining_images_root": str(
                (getattr(args, "mining_images_root", None) or images_dir)
                .expanduser()
                .resolve()
            ),
            "resolved_mining_pool_csv": str(
                (
                    getattr(args, "resolved_mining_pool_csv", None)
                    or rd / "inputs" / "mining_pool.resolved.csv"
                )
                .expanduser()
                .resolve()
            ),
            "backbone_weight_dir": str(ws / "augmentation" / "backbone"),
            "train_container": args.train_container,
            "num_gpus": args.num_gpus,
            "gpu_model": args.gpu_model,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "anomalygen": {
                "sub_skill": "tao-generate-anomalies",
                "mode": "inference_only",
                "project": args.project,
                # defect_spec lives under `datasets/<project>/` (sibling of
                # `checkpoints/<project>/`), per references/tao-generate-anomalies.md.
                "defect_spec": str(
                    ws
                    / "augmentation"
                    / "anomalygen"
                    / "datasets"
                    / args.project
                    / "defect_spec.jsonl"
                ),
                # ag_checkpoint_dir: the directory holding ag_config.yaml +
                # checkpoints/{latest_checkpoint.txt, model/iter_<step>.pt, ...}.
                # The underlying skill takes this as `ag_checkpoint_dir`.
                "checkpoint_dir": str(ag_checkpoint_dir),
                # dataset_dir: parent-staged pool root; not the raw datasets/.
                # Resolved per-iteration to `${RESULTS_DIR}/iter${N}/pool_anomalygen/inputs/`.
                "dataset_dir_source": str(
                    ws / "augmentation" / "anomalygen" / "datasets" / args.project
                ),
                "step": args.step,
                "num_SDG": args.num_sdg,
                "container": args.ag_container,
            },
            "mining_filter": {
                "sub_skill": "tao-mine-aoi-images",
                "top_k_per_target": args.top_k_per_target,
                "metric": args.knn_metric,
                "min_similarity": args.min_similarity,
                "history_aware": {
                    "enabled": True,
                    "identity": "filepath",
                    "history_file": str(rd / "mining_history.json"),
                },
            },
        },
        "iterations": {},
        "events": [],
        "_completed_step_values": list(_COMPLETED_STEP_VALUES),
        "_status_values": list(_STATUS_VALUES),
    }
    return state


def write_atomic(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize deft_state.json with a guaranteed-unique key set. "
            "Refuses to overwrite an existing file unless --force."
        ),
    )
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
    parser.add_argument("--mining-images-root", type=pathlib.Path)
    parser.add_argument("--resolved-mining-pool-csv", type=pathlib.Path)
    parser.add_argument(
        "--kpi-target",
        help=(
            'Compact target, e.g. "quality_score >= 0.9". For a metric without '
            'a matching bundled evaluator, also pass '
            "--metric-evaluator."
        ),
    )
    parser.add_argument(
        "--metric-name",
        help="Structured metric key, e.g. escape_rate_pct (use with operator/target)",
    )
    parser.add_argument(
        "--metric-display-name",
        help="Human-facing metric label; defaults to --metric-name",
    )
    parser.add_argument(
        "--metric-operator",
        choices=("<", "<=", ">", ">="),
        help="Success comparison operator (use with --metric-name/target)",
    )
    parser.add_argument(
        "--metric-target",
        type=float,
        help="Numeric success target (use with --metric-name/operator)",
    )
    parser.add_argument(
        "--metric-unit",
        default="",
        help="Display unit such as %% or cost/board",
    )
    parser.add_argument(
        "--metric-evaluator",
        help=(
            "builtin:<supported-id>, an absolute customer evaluator executable, "
            "or artifact when an external system supplies metric_result JSON"
        ),
    )
    parser.add_argument(
        "--metric-evaluator-args-json",
        default="[]",
        help="JSON argv list appended when invoking a command evaluator",
    )
    parser.add_argument(
        "--metric-artifact-producer",
        help="System or owner that writes the result for an artifact evaluator",
    )
    parser.add_argument(
        "--metric-artifact-path-template",
        help=(
            "Absolute expected result path containing {iter_label}, used only "
            "with --metric-evaluator artifact"
        ),
    )
    parser.add_argument(
        "--metric-constraints-json",
        default="[]",
        help=(
            "JSON list of secondary gates, each with name/operator/target and "
            "optional display_name/unit"
        ),
    )
    parser.add_argument("--max-iterations", required=True, type=int)
    parser.add_argument("--num-gpus", required=True, type=int)
    parser.add_argument(
        "--gpu-model",
        required=True,
        help=(
            "Exact accelerator model recorded by the selected platform's "
            "Preflight, including memory when available"
        ),
    )
    parser.add_argument("--num-epochs", required=True, type=int)
    parser.add_argument("--num-sdg", required=True, type=int)
    parser.add_argument("--project", required=True, help="AnomalyGen project name (e.g. nvpcb)")
    parser.add_argument("--step", required=True, type=int, help="AnomalyGen checkpoint step")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--top-k-per-target", default=5, type=int)
    parser.add_argument(
        "--knn-metric",
        default="cosine",
        choices=("cosine", "euclidean", "manhattan"),
    )
    parser.add_argument(
        "--min-similarity",
        default=None,
        type=float,
        help="Cosine similarity threshold for mining (e.g. 0.9). Omit for none.",
    )
    parser.add_argument(
        "--train-container",
        default=_DEFAULT_TRAIN_CONTAINER,
        help=(
            "TAO toolkit container URI. Defaults to TAO_PYT_IMAGE, then "
            "versions.yaml::images.tao_toolkit.pyt via TAO_SKILL_BANK_PATH."
        ),
    )
    parser.add_argument(
        "--ag-container",
        default=_DEFAULT_AG_CONTAINER,
        help=(
            "Cosmos AnomalyGen container URI. Defaults to AG_IMAGE, then "
            "versions.yaml::images.metropolis_sdg.paidf_anomalygen via "
            "TAO_SKILL_BANK_PATH."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing deft_state.json. Off by default to protect resume state.",
    )
    return parser


def _build_metric_contract(args: argparse.Namespace) -> tuple[dict, str]:
    structured_values = (args.metric_name, args.metric_operator, args.metric_target)
    if args.kpi_target and any(value is not None for value in structured_values):
        raise ValueError(
            "use either --kpi-target or the structured --metric-name/operator/target flags"
        )
    if args.kpi_target:
        contract = parse_target_expression(args.kpi_target)
        target_text = args.kpi_target
    else:
        if any(value is None for value in structured_values):
            raise ValueError(
                "provide --kpi-target or all of --metric-name, --metric-operator, "
                "and --metric-target"
            )
        contract = {
            "name": args.metric_name,
            "display_name": args.metric_display_name or args.metric_name,
            "operator": args.metric_operator,
            "target": args.metric_target,
            "unit": args.metric_unit,
            "evaluator": {"type": "unconfigured"},
            "constraints": [],
        }
        target_text = ""

    try:
        evaluator_args = json.loads(args.metric_evaluator_args_json)
        constraints = json.loads(args.metric_constraints_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid metric JSON argument: {exc}") from exc
    if not isinstance(evaluator_args, list) or not all(
        isinstance(value, str) for value in evaluator_args
    ):
        raise ValueError("--metric-evaluator-args-json must be a JSON string list")
    if not isinstance(constraints, list):
        raise ValueError("--metric-constraints-json must be a JSON list")
    contract["constraints"] = constraints

    evaluator_value = args.metric_evaluator
    artifact_options = (
        args.metric_artifact_producer,
        args.metric_artifact_path_template,
    )
    if any(artifact_options) and evaluator_value != "artifact":
        raise ValueError(
            "--metric-artifact-producer/path-template require "
            "--metric-evaluator artifact"
        )
    if evaluator_value:
        if evaluator_value.startswith("builtin:"):
            evaluator_id = evaluator_value.split(":", 1)[1]
            if evaluator_id != "far_at_recall":
                raise ValueError(f"unsupported builtin metric evaluator {evaluator_id!r}")
            contract["evaluator"] = {
                "type": "builtin",
                "id": evaluator_id,
                "parameters": contract.get("evaluator", {}).get("parameters", {}),
            }
        elif evaluator_value == "artifact":
            if evaluator_args:
                raise ValueError(
                    "artifact metric evaluators do not accept command argv"
                )
            if not args.metric_artifact_producer:
                raise ValueError(
                    "artifact metric evaluator requires --metric-artifact-producer"
                )
            if not args.metric_artifact_path_template:
                raise ValueError(
                    "artifact metric evaluator requires "
                    "--metric-artifact-path-template"
                )
            contract["evaluator"] = {
                "type": "artifact",
                "producer": args.metric_artifact_producer,
                "path_template": args.metric_artifact_path_template,
            }
        else:
            evaluator_path = pathlib.Path(evaluator_value).expanduser()
            if not evaluator_path.is_absolute():
                raise ValueError("custom metric evaluator path must be absolute")
            if not evaluator_path.is_file():
                raise ValueError(
                    f"custom metric evaluator does not exist: {evaluator_path}"
                )
            if not os.access(evaluator_path, os.X_OK):
                raise ValueError(
                    f"custom metric evaluator is not executable: {evaluator_path}"
                )
            contract["evaluator"] = {
                "type": "command",
                "path": str(evaluator_path.resolve()),
                "args": evaluator_args,
            }

    evaluator = contract.get("evaluator", {})
    if evaluator.get("type") == "unconfigured" and contract.get("name") == "far_pct":
        contract["evaluator"] = {
            "type": "builtin",
            "id": "far_at_recall",
            "parameters": {"recall_target_pct": 100.0},
        }
        evaluator = contract["evaluator"]
    if contract.get("name") == "far_pct":
        contract["unit"] = "%"
        parameters = contract.setdefault("evaluator", {}).setdefault(
            "parameters", {}
        )
        recall_target = float(parameters.get("recall_target_pct", 100.0))
        parameters["recall_target_pct"] = recall_target
        if not contract["constraints"]:
            contract["constraints"] = [
                {
                    "name": "recall_pct",
                    "display_name": "Recall",
                    "operator": ">=",
                    "target": recall_target,
                    "unit": "%",
                }
            ]
    if evaluator.get("type") == "unconfigured":
        raise ValueError(
            "custom metrics require --metric-evaluator with an absolute command "
            "path or the value 'artifact'"
        )
    contract = validate_contract(contract)
    return contract, target_text or render_target(contract)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.gpu_model = args.gpu_model.strip()
    if not args.gpu_model:
        print("init_deft_state: --gpu-model must not be empty", file=sys.stderr)
        return 2
    try:
        args.metric_contract, args.kpi_target_text = _build_metric_contract(args)
    except ValueError as exc:
        print(f"init_deft_state: {exc}", file=sys.stderr)
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
    positive_ints = {
        "max_iterations": args.max_iterations,
        "num_gpus": args.num_gpus,
        "num_epochs": args.num_epochs,
        "num_sdg": args.num_sdg,
        "batch_size": args.batch_size,
        "top_k_per_target": args.top_k_per_target,
    }
    invalid = {name: value for name, value in positive_ints.items() if value <= 0}
    if invalid:
        detail = ", ".join(f"{name}={value}" for name, value in invalid.items())
        print(
            f"init_deft_state: positive integers required ({detail})",
            file=sys.stderr,
        )
        return 2
    if not args.train_container:
        print(
            "init_deft_state: --train-container is required because neither "
            "TAO_PYT_IMAGE nor versions.yaml could be resolved (export "
            "TAO_PYT_IMAGE, set TAO_SKILL_BANK_PATH, or pass --train-container).",
            file=sys.stderr,
        )
        return 2
    if not args.ag_container:
        print(
            "init_deft_state: --ag-container is required because neither AG_IMAGE "
            "nor versions.yaml could be resolved (export AG_IMAGE, set "
            "TAO_SKILL_BANK_PATH, or pass --ag-container).",
            file=sys.stderr,
        )
        return 2
    out = args.results_dir / "deft_state.json"
    if out.exists() and not args.force:
        print(
            f"init_deft_state: refusing to overwrite {out} (use --force).",
            file=sys.stderr,
        )
        return 2
    state = build_state(args)
    write_atomic(out, state)
    print(f"init_deft_state: wrote {out}", file=sys.stderr)
    try:
        render_html_report(args.results_dir)
    except Exception as exc:  # noqa: BLE001 - state initialization remains valid
        print(f"init_deft_state: report hook failed: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
