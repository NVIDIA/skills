# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize one immutable, run-specific PAS DEFT configuration.

The canonical templates remain unchanged under ``specs/``. This command copies
them into ``RESULTS_DIR/config`` and applies only values approved in the
pre-flight summary.  All paths written to the DEFT config are absolute host
paths, so later tool calls do not depend on a previous ``cd`` or ``export``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile
from typing import Any

import yaml

from metric_contract import validate_contract
from deft_action_contract import SUPPORTED_PLATFORMS, safe_absolute_path
from finetuning_contract import expected_lora_record, materialize_peft
from virtualenv_runtime import resolve_virtualenv_profiles


def _bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def _gpu_ids(raw: str, count: int) -> list[int]:
    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--gpu-ids must be a comma-separated integer list") from exc
    if len(values) != count or len(set(values)) != len(values) or any(v < 0 for v in values):
        raise ValueError(
            "--gpu-ids must contain exactly --num-gpus distinct non-negative IDs"
        )
    return values


def _query_types(raw: str) -> str:
    allowed = {
        "easy",
        "medium",
        "hard",
        "natural_caption",
        "original_captions",
    }
    values = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [item for item in values if item not in allowed]
    if not values or invalid or len(set(values)) != len(values):
        detail = f"; unsupported values: {', '.join(invalid)}" if invalid else ""
        raise ValueError(
            "--gap-query-types must be a non-empty, duplicate-free comma-separated "
            f"list drawn from {sorted(allowed)}{detail}"
        )
    return ",".join(values)


def _workspace_child(path: pathlib.Path, workspace: pathlib.Path, name: str) -> pathlib.Path:
    resolved = safe_absolute_path(path, name)
    try:
        relative = resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{name} must be under --workspace {workspace}: {resolved}") from exc
    if relative == pathlib.Path("."):
        raise ValueError(f"{name} must be a child of --workspace, not the workspace itself")
    return resolved


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty template: {path}")
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be an object: {path}")
    return payload


def _existing_path(path: pathlib.Path, name: str, *, directory: bool) -> pathlib.Path:
    resolved = safe_absolute_path(path, name, require_exists=True)
    valid = resolved.is_dir() if directory else resolved.is_file()
    if not valid or (not directory and resolved.stat().st_size == 0):
        kind = "directory" if directory else "non-empty file"
        raise ValueError(f"{name} must be an existing {kind}: {resolved}")
    return resolved


@dataclasses.dataclass(frozen=True)
class RuntimeAttestation:
    """Validated immutable capability evidence for the approved PyTorch image."""

    image_ref: str
    image_digest: str
    finetuning_methods: tuple[str, ...]
    clip_lora: dict[str, Any] | None


def _validate_pyt_runtime_attestation(
    path: pathlib.Path, image: str, finetuning_method: str
) -> RuntimeAttestation:
    attestation_path = _existing_path(
        path, "--pyt-runtime-attestation", directory=False
    )
    attestation = json.loads(attestation_path.read_text())
    if not isinstance(attestation, dict):
        raise ValueError("PyTorch runtime attestation root must be an object")
    if attestation.get("schema_version") != "1":
        raise ValueError("PyTorch runtime attestation schema_version must be 1")
    if attestation.get("status") != "PASS" or attestation.get("image_ref") != image:
        raise ValueError("PyTorch runtime attestation does not pass for --pyt-image")
    digest = attestation.get("image_digest")
    if not isinstance(digest, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", digest
    ):
        raise ValueError(
            "PyTorch runtime attestation lacks an immutable image_digest"
        )
    methods = attestation.get("finetuning_methods")
    if (
        not isinstance(methods, list)
        or any(method not in {"sft", "lora"} for method in methods)
        or len(methods) != len(set(methods))
        or finetuning_method not in methods
    ):
        raise ValueError(
            "PyTorch runtime attestation does not support the approved "
            f"fine-tuning method {finetuning_method}"
        )

    contract = attestation.get("clip_lora")
    if finetuning_method == "sft":
        return RuntimeAttestation(
            image_ref=image,
            image_digest=digest,
            finetuning_methods=tuple(methods),
            clip_lora=contract if isinstance(contract, dict) else None,
        )
    if not isinstance(contract, dict):
        raise ValueError("PyTorch runtime attestation lacks the CLIP LoRA contract")
    if contract.get("checkpoint_behavior") != "register-and-merge":
        raise ValueError("PyTorch runtime attestation lacks LoRA checkpoint behavior")
    required_symbols = {
        "LoRALinear",
        "inject_lora",
        "merge_lora",
        "_register_lora_checkpoint_compatibility",
    }
    runtime_symbols = contract.get("runtime_symbols")
    if not isinstance(runtime_symbols, list) or set(runtime_symbols) != required_symbols:
        raise ValueError("PyTorch runtime attestation lacks LoRA runtime symbols")
    schema = contract.get("schema")
    if (
        not isinstance(schema, dict)
        or schema.get("method") != "lora"
        or not isinstance(schema.get("enabled"), bool)
    ):
        raise ValueError("PyTorch runtime attestation lacks the PEFT schema")
    expected_targets = ["q_proj", "k_proj", "v_proj", "out_proj"]
    required_fields = {
        "mode",
        "target_modules",
        "num_last_blocks",
        "rank",
        "alpha",
        "dropout",
    }
    for tower in ("vision", "text"):
        block = schema.get(tower)
        if (
            not isinstance(block, dict)
            or not required_fields.issubset(block)
            or block.get("target_modules") != expected_targets
        ):
            raise ValueError(
                f"PyTorch runtime attestation lacks the {tower} LoRA adapter contract"
            )
    return RuntimeAttestation(
        image_ref=image,
        image_digest=digest,
        finetuning_methods=tuple(methods),
        clip_lora=contract,
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


def _atomic_yaml(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    skill_root = pathlib.Path(__file__).resolve().parent.parent
    templates = skill_root / "specs"
    runtime_dir = skill_root / "scripts" / "pas_deft"
    runtime_sha256 = _python_tree_sha256(runtime_dir)
    workspace = safe_absolute_path(
        args.workspace, "--workspace", require_exists=True
    )
    if not workspace.is_dir() or workspace == pathlib.Path(workspace.anchor):
        raise ValueError(f"--workspace must be an existing non-root directory: {workspace}")
    results_dir = _workspace_child(args.results_dir, workspace, "--results-dir")
    dataset_root = _workspace_child(args.dataset_root, workspace, "--dataset-root")
    virtualenvs = resolve_virtualenv_profiles(
        platform=args.platform,
        legacy=args.virtualenv,
        pyt=args.pyt_virtualenv,
        ds=args.ds_virtualenv,
        probe_imports=True,
    )
    if args.docker_remote and args.platform != "docker":
        raise ValueError("--docker-remote is valid only with --platform docker")
    if virtualenvs is not None:
        control_environment = (workspace / ".venv").resolve()
        if any(path == control_environment for path in virtualenvs.values()):
            raise ValueError(
                "TAO execution profiles must be separate from the workspace control .venv"
            )
    images_archive = _existing_path(
        args.images_archive, "--images-archive", directory=False
    )
    metadata_archive = _existing_path(
        args.metadata_archive, "--metadata-archive", directory=False
    )
    checksums_file = (
        _existing_path(args.checksums_file, "--checksums-file", directory=False)
        if args.checksums_file is not None
        else None
    )
    if dataset_root.parent == workspace:
        raise ValueError(
            "--dataset-root must be nested below a workspace data directory "
            "(for example <workspace>/data/pas_v31_tao_ft)"
        )
    if results_dir in dataset_root.parents or dataset_root in results_dir.parents:
        raise ValueError("--results-dir and --dataset-root must not contain one another")
    if (results_dir / "deft_state.json").exists():
        raise ValueError(
            "refusing to rewrite config for an initialized run; resume from "
            "deft_state.json or choose a fresh --results-dir"
        )
    config_dir = results_dir / "config"
    safe_absolute_path(config_dir, "run config directory")

    positive = {
        "max_iterations": args.max_iterations,
        "training_epochs": args.training_epochs,
        "num_gpus": args.num_gpus,
        "mining_topn": args.mining_topn,
        "target_query_count": args.target_query_count,
        "queries_per_slice": args.queries_per_slice,
        "train_batch_size": args.train_batch_size,
        "val_batch_size": args.val_batch_size,
        "eval_batch_size": args.eval_batch_size,
    }
    invalid = {key: value for key, value in positive.items() if value < 1}
    if invalid:
        raise ValueError(
            "positive integers required: "
            + ", ".join(f"{key}={value}" for key, value in invalid.items())
        )
    for option, image in (("--pyt-image", args.pyt_image), ("--ds-image", args.ds_image)):
        if not image.strip():
            raise ValueError(f"{option} must be a non-empty image reference")
    if not 0.0 <= args.replay_fraction <= 1.0:
        raise ValueError("--replay-fraction must be in [0, 1]")
    if args.knn_metric not in {"cosine", "euclidean"}:
        raise ValueError("--knn-metric must be cosine or euclidean")
    pyt_attestation = None
    if args.platform != "virtualenv":
        if args.pyt_runtime_attestation is None:
            raise ValueError(
                "container platforms require --pyt-runtime-attestation for the "
                "approved fine-tuning method"
            )
        pyt_attestation = _validate_pyt_runtime_attestation(
            args.pyt_runtime_attestation,
            args.pyt_image,
            args.finetuning_method,
        )
    # ``gpu_ids`` is an allocation in the launcher/host namespace. Container
    # runtimes expose that allocation as a dense zero-based CUDA namespace, so
    # TAO must never receive the host ordinals directly.
    host_gpu_ids = _gpu_ids(args.gpu_ids, args.num_gpus)
    container_gpu_ids = list(range(args.num_gpus))
    for name, value in (
        ("vision_lr", args.vision_lr),
        ("text_lr", args.text_lr),
    ):
        if value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than zero")
    gap_query_types = _query_types(args.gap_query_types)
    metric_contract = validate_contract(
        {
            "metric_name": args.metric_name,
            "query_type": args.metric_query_type,
            "op": args.metric_op,
            "target": args.metric_target,
        }
    )

    deft = _load_yaml(templates / "deft_config.yaml")
    tao = _load_yaml(templates / "tao_spec.yaml")
    text_embed = _load_yaml(templates / "text_embed_spec.yaml")
    image_embed = _load_yaml(templates / "image_embed_spec.yaml")
    mining_spec = _load_yaml(templates / "mining_spec.yaml")

    deft["experiment"].update(
        {
            "results_path": str(results_dir),
            "train_config": str(config_dir / "tao_spec.yaml"),
            "eval_config": str(config_dir / "tao_spec.yaml"),
            "visualize": args.visualize,
            "visualize_embeddings": args.visualize_embeddings,
        }
    )
    deft["iteration"].update({"start": 1, "end": args.max_iterations})
    deft["training"].update(
        {
            "continual_model": args.continual_model,
            "continual_dataset": args.continual_dataset,
        }
    )
    deft["mining"].setdefault("history_aware", {}).update(
        {"enabled": args.history_aware, "replay_fraction": args.replay_fraction}
    )
    deft["gap_analysis"].update(
        {
            "metric_name": args.metric_name,
            "target_query_count": args.target_query_count,
            "queries_per_slice": args.queries_per_slice,
            "query_types": gap_query_types,
        }
    )
    pas = deft["pas"]
    pas.update(
        {
            "pool_pairs_source_file": str(dataset_root / "train_pairs.json"),
            "eval_pairs_source_file": str(
                dataset_root / f"{args.eval_split}_pairs.json"
            ),
            "train_image_dir": str(dataset_root / "images"),
            "train_caption_dir": str(dataset_root / "captions"),
            "source_image_dir": str(dataset_root / "images"),
            "source_caption_dir": str(dataset_root / "captions"),
            "eval_image_dir": str(dataset_root / "images"),
            "eval_caption_dir": str(dataset_root / "captions"),
        }
    )
    if pas.get("train_pairs_source_file"):
        pas["train_pairs_source_file"] = str(dataset_root / "train_pairs.json")

    tao["train"].update(
        {
            "num_epochs": args.training_epochs,
            "num_gpus": args.num_gpus,
            "gpu_ids": container_gpu_ids,
        }
    )
    tao["train"].setdefault("optim", {}).update(
        {"vision_lr": args.vision_lr, "text_lr": args.text_lr}
    )
    tao.setdefault("model", {}).update(
        {"freeze_vision_encoder": False, "freeze_text_encoder": False}
    )
    tao["peft"] = materialize_peft(args.finetuning_method)
    tao["dataset"].setdefault("train", {}).update(
        {"batch_size": args.train_batch_size}
    )
    tao["dataset"].setdefault("val", {}).update(
        {"batch_size": args.val_batch_size}
    )
    tao["evaluate"].update(
        {
            "num_gpus": args.num_gpus,
            "gpu_ids": container_gpu_ids,
            "batch_size": args.eval_batch_size,
        }
    )
    tao["inference"].update(
        {"num_gpus": args.num_gpus, "gpu_ids": container_gpu_ids}
    )

    # These values are consumed from separate TAO data-services specs. Those
    # specs are the single materialized authority; do not duplicate the same
    # controls in deft_config.yaml where they can drift independently.
    # Image and text embeddings are compared in the same vector space, so both
    # specs use the common adapter token supported by TAO 7.2's image and text
    # entry points and the same checkpoint. Updating only one side silently
    # changes mining; exposing a text-only token that image embedding rejects
    # makes visualization fail late in the loop.
    text_embed["model"] = args.text_embed_model
    image_embed["model"] = args.text_embed_model
    mining_spec.update({"topn": args.mining_topn, "knn_metric": args.knn_metric})

    _atomic_yaml(config_dir / "deft_config.yaml", deft)
    _atomic_yaml(config_dir / "tao_spec.yaml", tao)
    _atomic_yaml(config_dir / "text_embed_spec.yaml", text_embed)
    _atomic_yaml(config_dir / "image_embed_spec.yaml", image_embed)
    _atomic_yaml(config_dir / "mining_spec.yaml", mining_spec)

    # Validate the complete materialized bundle through the same typed schema
    # that every host-side stage consumes. This prevents a generated config
    # from crossing the approval boundary with an unknown, missing, mistyped,
    # out-of-range, or cross-spec-conflicting field.
    from pas_deft.config import PasDeftConfig

    PasDeftConfig(str(config_dir / "deft_config.yaml"))
    lora_rank, lora_alpha = expected_lora_record(args.finetuning_method)
    _atomic_json(
        config_dir / "approval.json",
        {
            "schema_version": "6",
            "workflow": "tao-run-deft-pas",
            "platform": args.platform,
            "docker_remote": args.docker_remote,
            "virtualenvs": (
                {name: str(path) for name, path in virtualenvs.items()}
                if virtualenvs is not None
                else None
            ),
            "workspace": str(workspace),
            "results_dir": str(results_dir),
            "dataset_root": str(dataset_root),
            "pas_deft_bundle_sha256": runtime_sha256,
            "images_archive": str(images_archive),
            "metadata_archive": str(metadata_archive),
            "checksums_file": str(checksums_file) if checksums_file else None,
            "requires_hf_token": args.requires_hf_token,
            "finetuning_method": args.finetuning_method,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "max_iterations": args.max_iterations,
            "host_gpu_ids": host_gpu_ids,
            "container_gpu_ids": container_gpu_ids,
            "metric_contract": metric_contract,
            "pyt_image": args.pyt_image,
            "pyt_image_digest": (
                pyt_attestation.image_digest if pyt_attestation is not None else None
            ),
            "ds_image": args.ds_image,
        },
    )

    return {
        "config_dir": str(config_dir),
        "deft_config": str(config_dir / "deft_config.yaml"),
        "tao_spec": str(config_dir / "tao_spec.yaml"),
        "max_iterations": args.max_iterations,
        "training_epochs": args.training_epochs,
        "num_gpus": args.num_gpus,
        "gpu_ids": host_gpu_ids,
        "container_gpu_ids": container_gpu_ids,
        "platform": args.platform,
        "docker_remote": args.docker_remote,
        "virtualenvs": (
            {name: str(path) for name, path in virtualenvs.items()}
            if virtualenvs is not None
            else None
        ),
        "requires_hf_token": args.requires_hf_token,
        "finetuning_method": args.finetuning_method,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "pyt_image_digest": (
            pyt_attestation.image_digest if pyt_attestation is not None else None
        ),
        "pas_deft_bundle_sha256": runtime_sha256,
        "eval_split": args.eval_split,
        "queries_per_slice": args.queries_per_slice,
        "gap_query_types": gap_query_types,
        "vision_lr": args.vision_lr,
        "text_lr": args.text_lr,
        "train_batch_size": args.train_batch_size,
        "val_batch_size": args.val_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "text_embed_model": args.text_embed_model,
        "approval_manifest": str(config_dir / "approval.json"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=pathlib.Path)
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--dataset-root", required=True, type=pathlib.Path)
    parser.add_argument("--images-archive", required=True, type=pathlib.Path)
    parser.add_argument("--metadata-archive", required=True, type=pathlib.Path)
    parser.add_argument("--checksums-file", type=pathlib.Path)
    parser.add_argument(
        "--platform",
        default="docker",
        choices=SUPPORTED_PLATFORMS,
        help="TAO execution platform (default: docker for CLI compatibility).",
    )
    parser.add_argument(
        "--docker-remote",
        action="store_true",
        help=(
            "Approve Docker execution through a remote DOCKER_HOST; requires "
            "remote staging/absence attestation instead of launcher-local mounts."
        ),
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
        "--requires-hf-token",
        action="store_true",
        help="Approve forwarding the existing HF_TOKEN by environment name to model stages.",
    )
    parser.add_argument("--max-iterations", required=True, type=int)
    parser.add_argument("--training-epochs", default=1, type=int)
    parser.add_argument(
        "--finetuning-method",
        choices=("lora", "sft"),
        default="lora",
        help="Fine-tuning method (default: lora). SFT explicitly disables PEFT.",
    )
    parser.add_argument(
        "--pyt-runtime-attestation",
        "--lora-capability-attestation",
        dest="pyt_runtime_attestation",
        type=pathlib.Path,
        help=(
            "Digest-bound PyTorch runtime/capability attestation. The former "
            "--lora-capability-attestation spelling remains as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--pyt-image",
        required=True,
        help="PyTorch image resolved from the workflow versions key.",
    )
    parser.add_argument(
        "--ds-image",
        required=True,
        help="Data-services image resolved from the workflow versions key.",
    )
    parser.add_argument("--num-gpus", default=1, type=int)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--mining-topn", default=25, type=int)
    parser.add_argument("--knn-metric", default="cosine")
    parser.add_argument("--target-query-count", default=10000, type=int)
    parser.add_argument("--queries-per-slice", default=256, type=int)
    parser.add_argument("--gap-query-types", default="easy,medium")
    parser.add_argument("--eval-split", default="val", choices=("val", "test"))
    parser.add_argument("--vision-lr", default=1.0e-7, type=float)
    parser.add_argument("--text-lr", default=1.0e-7, type=float)
    parser.add_argument("--train-batch-size", default=32, type=int)
    parser.add_argument("--val-batch-size", default=64, type=int)
    parser.add_argument("--eval-batch-size", default=32, type=int)
    parser.add_argument(
        "--text-embed-model", default="SigLIP", choices=("SigLIP",)
    )
    parser.add_argument("--history-aware", default=True, type=_bool)
    parser.add_argument("--replay-fraction", default=0.20, type=float)
    parser.add_argument("--continual-dataset", default=True, type=_bool)
    parser.add_argument("--continual-model", default=False, type=_bool)
    parser.add_argument("--visualize", default=True, type=_bool)
    parser.add_argument("--visualize-embeddings", default=True, type=_bool)
    parser.add_argument("--metric-name", default="Rank-1")
    parser.add_argument("--metric-query-type", default="medium")
    parser.add_argument("--metric-op", default=">=", choices=(">=", ">", "<=", "<"))
    parser.add_argument("--metric-target", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = materialize(args)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(f"prepare_deft_config: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
