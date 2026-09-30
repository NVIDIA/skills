# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initialize ${RESULTS_DIR}/deft_state.json for an PAS CLIP DEFT run.

Why this exists: inline-dict writes drift from the canonical schema and can
produce duplicate top-level keys — Python 3.12+ emits a `SyntaxWarning` for
these and the loop's resume logic reads whichever copy parsing keeps, which is
not stable across edits.

This script builds the dict with literal-once keys and writes the JSON. Atomic
write (tmp + os.replace). It refuses to reinitialize an existing
deft_state.json — the resume path is supposed to read disk, not regenerate —
and it appends nothing to loop_log.jsonl.

CLI:

    python scripts/init_deft_state.py \
        --results-dir ~/workspace/results/run_20260805_090000 \
        --workspace ~/workspace \
        --dataset-root ~/workspace/data/pas_v31_tao_ft \
        --max-iterations 3 \
        --metric-target 0.85 \
        --platform docker \
        # unpinned: illustrative CLI placeholder; executable default is release-managed below
        --pyt-image nvcr.io/nvidia/tao/tao-toolkit:pyt \
        # unpinned: illustrative CLI placeholder; executable default is release-managed below
        --ds-image nvcr.io/nvidia/tao/tao-toolkit:ds \
        --deft-config ~/workspace/specs/deft_config.yaml \
        --tao-spec ~/workspace/specs/tao_spec.yaml

On-disk layout recorded in `config.layout` follows the bundled PAS runtime
conventions: baseline (zero-shot) evaluate artifacts live under
${RESULTS_DIR}/zs/, iteration N artifacts under ${RESULTS_DIR}/iter_<N>/, and
run-level shared artifacts under ${RESULTS_DIR}/pas_splits/,
${RESULTS_DIR}/embeddings/source/, and the two selection-history JSON files.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile

import yaml

from deft_action_contract import SUPPORTED_PLATFORMS, safe_absolute_path
from virtualenv_runtime import resolve_virtualenv_profiles

from metric_contract import validate_contract


WORKFLOW = "tao-run-deft-pas"
PINNED_PYT_IMAGE = "nvcr.io/nvidia/tao/tao-toolkit:7.2.0-pyt"  # versions-key: images.tao_toolkit.deft_pas_pyt
PINNED_DS_IMAGE = "nvcr.io/nvidia/tao/tao-toolkit:7.2.0-data-services"  # versions-key: images.tao_toolkit.deft_pas_data_services
RUN_SPEC_NAMES = (
    "deft_config.yaml",
    "tao_spec.yaml",
    "text_embed_spec.yaml",
    "image_embed_spec.yaml",
    "mining_spec.yaml",
    "approval.json",
)

_COMPLETED_STEP_VALUES = [
    "dataset_setup",
    "pool_embed",
    "evaluate",
    "gap_analysis",
    "data_mining",
    "history_select",
    "visualize",
    "train",
    "loop_stop",
]
_STATUS_VALUES = ["pending", "in_progress", "complete", "failed"]


def _parse_gpu_ids(raw: str, num_gpus: int) -> list[int]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError('--gpu-ids must be a non-empty comma list, e.g. "0,1"')
    try:
        gpu_ids = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"--gpu-ids must be a comma list of integers: {raw!r}") from exc
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError(f"--gpu-ids must be non-negative device indices: {raw!r}")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"--gpu-ids contains duplicate device indices: {raw!r}")
    if len(gpu_ids) != num_gpus:
        raise ValueError(
            f"--gpu-ids lists {len(gpu_ids)} device(s) but --num-gpus is {num_gpus}"
        )
    return gpu_ids


def _approval_gpu_ids(value: object, field: str) -> list[int]:
    """Parse one approval GPU namespace without coercing malformed JSON types."""
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise ValueError(f"approval.json {field} must be a non-empty integer list")
    return _parse_gpu_ids(",".join(str(item) for item in value), len(value))


def _resolve_dataset_root(root: pathlib.Path) -> pathlib.Path:
    """Resolve the intended dataset root before dataset_setup materializes it."""
    resolved = safe_absolute_path(root, "--dataset-root")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"--dataset-root exists but is not a directory: {resolved}")
    return resolved


def _workspace_child(path: pathlib.Path, workspace: pathlib.Path, name: str) -> pathlib.Path:
    resolved = safe_absolute_path(path, name)
    try:
        relative = resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{name} must be under --workspace {workspace}: {resolved}") from exc
    if relative == pathlib.Path("."):
        raise ValueError(f"{name} must be a child of --workspace, not the workspace itself")
    return resolved


def _validate_metric_gate(
    name: str, query_type: str, operator: str, target: float | None
) -> dict:
    """Reject malformed metric gates before they reach disk.

    A missing --metric-target is legal: the gate then never passes and the
    loop runs to --max-iterations. A present target must be a finite number,
    and Rank-style metrics are fractions in [0, 1].
    """
    contract = validate_contract(
        {
            "metric_name": name,
            "query_type": query_type,
            "op": operator,
            "target": target,
        }
    )
    if target is not None and name.strip().lower().startswith("rank") and not 0.0 <= target <= 1.0:
        raise ValueError(
            f"--metric-target for {name} must be a fraction in [0, 1], got {target}"
        )
    return contract


def _required_input_file(path: pathlib.Path, name: str) -> pathlib.Path:
    resolved = safe_absolute_path(path, name, require_exists=True)
    if not resolved.is_file():
        raise ValueError(f"{name} must be an existing file: {path}")
    if resolved.stat().st_size == 0:
        raise ValueError(f"{name} must not be empty: {path}")
    return resolved


def _required_input_dir(path: pathlib.Path, name: str) -> pathlib.Path:
    resolved = safe_absolute_path(path, name, require_exists=True)
    if not resolved.is_dir():
        raise ValueError(f"{name} must be an existing directory: {path}")
    return resolved


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_checksum_manifest(
    manifest: pathlib.Path, archives: tuple[pathlib.Path, pathlib.Path]
) -> None:
    referenced: set[pathlib.Path] = set()
    for line_number, raw in enumerate(manifest.read_text().splitlines(), 1):
        line = raw.strip("\n")
        if not line or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"[0-9A-Fa-f]{64} ([ *])(.+)", line)
        if not match:
            raise ValueError(
                f"--checksums-file line {line_number} is not GNU SHA256 format"
            )
        filename = match.group(2)
        if "\x00" in filename or filename.startswith("\\"):
            raise ValueError(
                "--checksums-file uses an unsupported escaped filename entry"
            )
        referenced.add((manifest.parent / filename).resolve())
    missing = [str(path) for path in archives if path.resolve() not in referenced]
    if missing:
        raise ValueError(
            "--checksums-file must bind both approved archives; missing "
            + ", ".join(missing)
        )


def _python_tree_sha256(root: pathlib.Path) -> str:
    files = sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    if not files:
        raise ValueError(f"bundled PAS runtime contains no Python files: {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _mapping(root: dict, key: str, source: str) -> dict:
    value = root.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{source}.{key} must be an object")
    return value


def _load_run_config(args: argparse.Namespace) -> dict:
    config_dir = args.deft_config.parent.resolve()
    expected_config_dir = args.results_dir.resolve() / "config"
    if config_dir != expected_config_dir:
        raise ValueError(
            f"prepared config directory must be {expected_config_dir}, got {config_dir}"
        )
    if args.tao_spec.parent.resolve() != config_dir:
        raise ValueError("--deft-config and --tao-spec must share the run config directory")
    spec_sha256: dict[str, str] = {}
    for name in RUN_SPEC_NAMES:
        spec_path = _required_input_file(config_dir / name, f"run config {name}")
        spec_sha256[name] = _sha256(spec_path)
    approval_path = config_dir / "approval.json"
    approval = json.loads(approval_path.read_text())
    approved_host_gpu_ids = _approval_gpu_ids(
        approval.get("host_gpu_ids"), "host_gpu_ids"
    )
    approved_container_gpu_ids = _approval_gpu_ids(
        approval.get("container_gpu_ids"), "container_gpu_ids"
    )
    if len(approved_container_gpu_ids) != len(approved_host_gpu_ids):
        raise ValueError(
            "approval.json host_gpu_ids and container_gpu_ids must have equal length"
        )
    if approved_container_gpu_ids != list(range(len(approved_host_gpu_ids))):
        raise ValueError(
            "approval.json container_gpu_ids must be the dense zero-based "
            "namespace for the approved host GPU allocation"
        )
    approval_version = approval.get("schema_version")
    expected_approval = {
        "schema_version": approval_version,
        "workflow": WORKFLOW,
        "workspace": str(args.workspace.resolve()),
        "results_dir": str(args.results_dir.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "pas_deft_bundle_sha256": args.pas_deft_bundle_sha256,
        "images_archive": str(args.images_archive.resolve()),
        "metadata_archive": str(args.metadata_archive.resolve()),
        "checksums_file": (
            str(args.checksums_file.resolve())
            if args.checksums_file is not None
            else None
        ),
        "requires_hf_token": args.requires_hf_token,
        "max_iterations": args.max_iterations,
        "host_gpu_ids": approved_host_gpu_ids,
        "container_gpu_ids": approved_container_gpu_ids,
        "metric_contract": args.metric_contract,
        "pyt_image": args.pyt_image,
        "ds_image": args.ds_image,
    }
    if approval_version == "4":
        expected_approval["platform"] = args.platform
        expected_approval["docker_remote"] = args.docker_remote
        expected_approval["virtualenvs"] = (
            {name: str(path) for name, path in args.virtualenvs.items()}
            if args.virtualenvs is not None
            else None
        )
    elif approval_version == "3":
        if (
            args.platform != "docker"
            or args.docker_remote
            or args.virtualenvs is not None
        ):
            raise ValueError(
                "approval.json schema version 3 represents local Docker only; "
                "rerun config preparation for another platform"
            )
    elif approval_version == "2":
        if (
            args.platform != "docker"
            or args.docker_remote
            or args.virtualenvs is not None
        ):
            raise ValueError(
                "approval.json schema version 2 represents local Docker only; "
                "rerun config preparation for another platform"
            )
        expected_approval.pop("pas_deft_bundle_sha256")
    else:
        raise ValueError(
            "approval.json schema_version must be 2, 3, or 4"
        )
    if approval != expected_approval:
        raise ValueError(
            "approval.json does not match the approved initialization inputs; "
            "rerun config preparation in a fresh results directory"
        )
    from pas_deft.config import PasDeftConfig

    typed = PasDeftConfig(str(args.deft_config))
    tao = yaml.safe_load(args.tao_spec.read_text())
    if not isinstance(tao, dict):
        raise ValueError("--tao-spec must have an object root")
    train = _mapping(tao, "train", "tao_spec")
    evaluate = _mapping(tao, "evaluate", "tao_spec")
    dataset = _mapping(tao, "dataset", "tao_spec")
    dataset_train = _mapping(dataset, "train", "tao_spec.dataset")
    dataset_val = _mapping(dataset, "val", "tao_spec.dataset")
    optim = _mapping(train, "optim", "tao_spec.train")
    text_embed = yaml.safe_load((config_dir / "text_embed_spec.yaml").read_text())
    image_embed = yaml.safe_load((config_dir / "image_embed_spec.yaml").read_text())
    if not isinstance(text_embed, dict) or not isinstance(image_embed, dict):
        raise ValueError("prepared embedding spec roots must be objects")
    if typed.iteration.start != 1:
        raise ValueError("prepared deft_config iteration.start must be 1")
    if typed.iteration.end != args.max_iterations:
        raise ValueError(
            "prepared deft_config iteration.end must equal --max-iterations"
        )
    if pathlib.Path(typed.experiment.results_path).resolve() != args.results_dir.resolve():
        raise ValueError("prepared deft_config experiment.results_path must equal --results-dir")
    num_gpus = train.get("num_gpus")
    gpu_ids = train.get("gpu_ids")
    if not isinstance(num_gpus, int) or num_gpus < 1:
        raise ValueError("prepared tao_spec train.num_gpus must be >= 1")
    if not isinstance(gpu_ids, list):
        raise ValueError("prepared tao_spec train.gpu_ids must be a list")
    parsed_gpu_ids = _parse_gpu_ids(",".join(str(item) for item in gpu_ids), num_gpus)
    if parsed_gpu_ids != approved_container_gpu_ids:
        raise ValueError(
            "prepared TAO gpu_ids must match approval.json container_gpu_ids"
        )
    if num_gpus != len(approved_host_gpu_ids):
        raise ValueError(
            "prepared TAO num_gpus must match the approved host GPU allocation"
        )
    if evaluate.get("num_gpus") != num_gpus or evaluate.get("gpu_ids") != gpu_ids:
        raise ValueError("prepared TAO train/evaluate GPU shapes must match")
    training_epochs = train.get("num_epochs")
    if not isinstance(training_epochs, int) or training_epochs < 1:
        raise ValueError("prepared tao_spec train.num_epochs must be >= 1")
    if typed.gap_analysis.metric_name != args.metric_contract["metric_name"]:
        raise ValueError(
            "prepared gap_analysis.metric_name must match the immutable "
            "metric contract"
        )
    replay = typed.mining.history_aware.replay_fraction
    eval_path = pathlib.Path(typed.pas.eval_pairs_source_file).resolve()
    dataset_root = args.dataset_root.resolve()
    if eval_path.parent != dataset_root or eval_path.name not in {
        "val_pairs.json",
        "test_pairs.json",
    }:
        raise ValueError(
            "prepared pas.eval_pairs_source_file must be val_pairs.json or "
            "test_pairs.json directly under --dataset-root"
        )
    eval_split = eval_path.name.removesuffix("_pairs.json")
    queries_per_slice = typed.gap_analysis.queries_per_slice
    gap_query_types = typed.gap_analysis.query_types
    for name, value in (
        ("tao_spec.train.optim.vision_lr", optim.get("vision_lr")),
        ("tao_spec.train.optim.text_lr", optim.get("text_lr")),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or float(value) <= 0.0
        ):
            raise ValueError(f"prepared {name} must be greater than zero")
    batch_sizes = {
        "train_batch_size": dataset_train.get("batch_size"),
        "val_batch_size": dataset_val.get("batch_size"),
        "eval_batch_size": evaluate.get("batch_size"),
    }
    invalid_batches = {
        name: value
        for name, value in batch_sizes.items()
        if not isinstance(value, int) or isinstance(value, bool) or value < 1
    }
    if invalid_batches:
        raise ValueError(
            "prepared batch sizes must be integers >= 1: "
            + ", ".join(f"{name}={value!r}" for name, value in invalid_batches.items())
        )
    text_embed_model = text_embed.get("model")
    if text_embed_model != "SigLIP":
        raise ValueError(
            "prepared text_embed_spec.model must be SigLIP, the adapter token "
            "supported by both TAO 7.2 image and text embedding"
        )
    if image_embed.get("model") != text_embed_model:
        raise ValueError(
            "prepared image_embed_spec.model must match text_embed_spec.model "
            "for the shared embedding checkpoint"
        )
    text_embed_model_path = text_embed.get("model_path")
    if (
        not isinstance(text_embed_model_path, str)
        or not text_embed_model_path.strip()
        or image_embed.get("model_path") != text_embed_model_path
    ):
        raise ValueError(
            "prepared image/text embedding model_path values must be the same "
            "non-empty shared checkpoint"
        )
    for key in ("train_config", "eval_config"):
        if pathlib.Path(getattr(typed.experiment, key)).resolve() != args.tao_spec.resolve():
            raise ValueError(
                f"prepared deft_config experiment.{key} must equal --tao-spec"
            )
    return {
        "config_dir": str(config_dir),
        "approval_manifest": str(approval_path),
        "spec_sha256": spec_sha256,
        "deft_config_sha256": _sha256(args.deft_config),
        "tao_spec_sha256": _sha256(args.tao_spec),
        "training_epochs": training_epochs,
        "num_gpus": num_gpus,
        "gpu_ids": approved_host_gpu_ids,
        "container_gpu_ids": parsed_gpu_ids,
        "history_aware": typed.mining.history_aware.enabled,
        "replay_fraction": replay,
        "mining_topn": typed.mining.topn,
        "knn_metric": typed.mining.knn_metric,
        "target_query_count": typed.gap_analysis.target_query_count,
        "eval_split": eval_split,
        "queries_per_slice": queries_per_slice,
        "gap_query_types": gap_query_types,
        "vision_lr": float(optim.get("vision_lr")),
        "text_lr": float(optim.get("text_lr")),
        **batch_sizes,
        "text_embed_model": text_embed_model,
        "continual_dataset": typed.training.continual_dataset,
        "continual_model": typed.training.continual_model,
        "visualize": typed.visualization.enabled,
        "visualize_embeddings": typed.visualization.embeddings,
    }


def build_state(args: argparse.Namespace) -> dict:
    ws = args.workspace.expanduser().resolve()
    rd = args.results_dir.expanduser().resolve()

    state = {
        "schema_version": "3",
        "workflow": WORKFLOW,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "results_dir": str(rd),
        "max_iterations": args.max_iterations,
        "current_iteration": 0,
        "gate_met": False,
        # The canonical contract shape record_metric_result.py and
        # metric_contract.contract_from_state consume.
        "metric_contract": dict(args.metric_contract),
        "config": {
            "workspace": str(ws),
            "dataset_root": str(args.dataset_root),
            "pas_deft_bundle_sha256": args.pas_deft_bundle_sha256,
            "images_archive": str(args.images_archive),
            "metadata_archive": str(args.metadata_archive),
            "checksums_file": (
                str(args.checksums_file) if args.checksums_file is not None else None
            ),
            "checksums_file_sha256": (
                _sha256(args.checksums_file)
                if args.checksums_file is not None
                else None
            ),
            "requires_hf_token": args.requires_hf_token,
            "platform": args.platform,
            "docker_remote": args.docker_remote,
            "virtualenvs": (
                {name: str(path) for name, path in args.virtualenvs.items()}
                if args.virtualenvs is not None
                else None
            ),
            "pyt_image": args.pyt_image,
            "ds_image": args.ds_image,
            "deft_config": str(args.deft_config),
            "tao_spec": str(args.tao_spec),
            **args.run_config,
            # Bundled PAS runtime conventions; iteration labels map to
            # directories as baseline -> zs/ and iterN -> iter_<N>/.
            "layout": {
                "baseline_dir": str(rd / "zs"),
                "iteration_dir_template": str(rd / "iter_{N}"),
                "pas_splits_dir": str(rd / "pas_splits"),
                "source_embeddings_dir": str(rd / "embeddings" / "source"),
                "caption_selection_history": str(
                    rd / "caption_selection_history.json"
                ),
                "mining_selection_history": str(
                    rd / "mining_selection_history.json"
                ),
            },
        },
        "iterations": {"baseline": {"status": "pending"}},
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
            "Initialize deft_state.json for an PAS CLIP DEFT run with a "
            "guaranteed-unique key set. Refuses to reinitialize an existing "
            "file."
        ),
    )
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--workspace", required=True, type=pathlib.Path)
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=pathlib.Path,
        help=(
            "Intended pas_v31_tao_ft directory. It may not exist until "
            "baseline/dataset_setup materializes it."
        ),
    )
    parser.add_argument("--images-archive", required=True, type=pathlib.Path)
    parser.add_argument("--metadata-archive", required=True, type=pathlib.Path)
    parser.add_argument("--checksums-file", type=pathlib.Path)
    parser.add_argument(
        "--requires-hf-token",
        action="store_true",
        help="Must match approval.json; token values are never persisted.",
    )
    parser.add_argument("--max-iterations", required=True, type=int)
    parser.add_argument(
        "--metric-name",
        default="Rank-1",
        help="Primary metric key gating the loop (default Rank-1).",
    )
    parser.add_argument(
        "--metric-query-type",
        default="medium",
        help="Query split the gate reads the metric from (default medium).",
    )
    parser.add_argument(
        "--metric-op",
        default=">=",
        choices=(">=", ">", "<=", "<"),
        help="Success comparison operator.",
    )
    parser.add_argument(
        "--metric-target",
        type=float,
        help=(
            "Numeric success target. When absent the gate never passes and "
            "the loop runs to --max-iterations."
        ),
    )
    parser.add_argument(
        "--platform",
        default="docker",
        choices=SUPPORTED_PLATFORMS,
        help="TAO execution platform (default: docker for CLI compatibility).",
    )
    parser.add_argument(
        "--docker-remote",
        action="store_true",
        help="Record the approved remote-DOCKER_HOST staging contract.",
    )
    parser.add_argument(
        "--virtualenv",
        type=pathlib.Path,
        help=(
            "Compatibility form: one environment that independently satisfies both "
            "pyt and ds profiles; cannot be combined with profile-specific options."
        ),
    )
    parser.add_argument(
        "--pyt-virtualenv",
        type=pathlib.Path,
        help="Python 3.12 TAO PyTorch execution profile for train/evaluate.",
    )
    parser.add_argument(
        "--ds-virtualenv",
        type=pathlib.Path,
        help="Python 3.12 TAO data-services execution profile for embedding/mining.",
    )
    parser.add_argument(
        "--pyt-image",
        required=True,
        help="TAO PyTorch container URI used for train/evaluate.",
    )
    parser.add_argument(
        "--ds-image",
        required=True,
        help="TAO data-services container URI used for embedding/mining.",
    )
    parser.add_argument(
        "--deft-config",
        required=True,
        type=pathlib.Path,
        help="The run's pas_deft loop config file.",
    )
    parser.add_argument(
        "--tao-spec",
        required=True,
        type=pathlib.Path,
        help="The run's TAO experiment spec file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        args.virtualenvs = resolve_virtualenv_profiles(
            platform=args.platform,
            legacy=args.virtualenv,
            pyt=args.pyt_virtualenv,
            ds=args.ds_virtualenv,
            probe_imports=True,
        )
        if args.docker_remote and args.platform != "docker":
            raise ValueError("--docker-remote is valid only with --platform docker")
        if args.virtualenvs is not None:
            control_environment = (args.workspace.expanduser().resolve() / ".venv").resolve()
            if any(path == control_environment for path in args.virtualenvs.values()):
                raise ValueError(
                    "TAO execution profiles must be separate from the workspace control .venv"
                )
        args.metric_contract = _validate_metric_gate(
            args.metric_name, args.metric_query_type, args.metric_op, args.metric_target
        )
        args.dataset_root = _resolve_dataset_root(args.dataset_root)
        pas_runtime = _required_input_dir(
            pathlib.Path(__file__).resolve().parent / "pas_deft",
            "bundled PAS runtime",
        )
        args.pas_deft_bundle_sha256 = _python_tree_sha256(pas_runtime)
        args.workspace = _required_input_dir(args.workspace, "--workspace")
        if args.workspace == pathlib.Path(args.workspace.anchor):
            raise ValueError("--workspace must not be a filesystem root")
        args.results_dir = _workspace_child(
            args.results_dir, args.workspace, "--results-dir"
        )
        args.dataset_root = _workspace_child(
            args.dataset_root, args.workspace, "--dataset-root"
        )
        if (
            args.results_dir in args.dataset_root.parents
            or args.dataset_root in args.results_dir.parents
        ):
            raise ValueError(
                "--results-dir and --dataset-root must not contain one another"
            )
        if args.dataset_root.parent == args.workspace:
            raise ValueError(
                "--dataset-root must be nested below a workspace data directory"
            )
        args.images_archive = _required_input_file(args.images_archive, "--images-archive")
        args.metadata_archive = _required_input_file(args.metadata_archive, "--metadata-archive")
        if args.checksums_file is not None:
            args.checksums_file = _required_input_file(args.checksums_file, "--checksums-file")
            _validate_checksum_manifest(
                args.checksums_file,
                (args.images_archive, args.metadata_archive),
            )
        args.deft_config = _required_input_file(args.deft_config, "--deft-config")
        args.tao_spec = _required_input_file(args.tao_spec, "--tao-spec")
        args.run_config = _load_run_config(args)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        yaml.YAMLError,
    ) as exc:
        print(f"init_deft_state: {exc}", file=sys.stderr)
        return 2

    positive_ints = {
        "max_iterations": args.max_iterations,
    }
    invalid = {name: value for name, value in positive_ints.items() if value <= 0}
    if invalid:
        detail = ", ".join(f"{name}={value}" for name, value in invalid.items())
        print(
            f"init_deft_state: positive integers required ({detail})",
            file=sys.stderr,
        )
        return 2
    if args.pyt_image != PINNED_PYT_IMAGE or args.ds_image != PINNED_DS_IMAGE:
        print(
            "init_deft_state: --pyt-image and --ds-image must be the pinned "
            "PAS workflow images",
            file=sys.stderr,
        )
        return 2

    out = safe_absolute_path(
        args.results_dir / "deft_state.json", "deft state output"
    )
    if out.exists():
        print(
            f"init_deft_state: refusing to reinitialize {out}; deft_state.json "
            "is written exactly once per run. Resume reads disk state — to "
            "start over, choose a fresh --results-dir.",
            file=sys.stderr,
        )
        return 2
    try:
        state = build_state(args)
        write_atomic(out, state)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"init_deft_state: {exc}", file=sys.stderr)
        return 2
    print(f"init_deft_state: wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
