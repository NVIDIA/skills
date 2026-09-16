#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a dataset-neutral Cosmos evaluation from sealed training artifacts.

This helper never searches historical runs and never treats the packaged
template as an experiment profile.  Fingerprint-locked evaluator profiles
packaged below are allowed to supply verified dataset protocol semantics.  It
records the source of every semantic field, returns a bounded list of
genuinely missing user inputs, and writes a runtime TOML only after the
evaluation request is complete.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import tomllib
from cosmos_common import WorkflowError, sha256_file, stable_hash
from cosmos_workflow import dump_toml

SUCCESS = {"SUCCESS", "COMPLETE", "COMPLETED"}
SCRIPT_ROOT = Path(__file__).resolve().parent

VERIFIED_EVALUATOR_PROFILES: dict[str, dict[str, Any]] = {
    # Full-validation protocols verified on 2026-08-16. Every entry is keyed
    # by annotation bytes, never by a mutable path or dataset nickname.
    "c33afc26f979cbdb488b8f1aefdc65604992cd7552d5e75ea782e4565fdc21e1": {
        "name": "VALIDATION_C33AFC26",
        "protocol_fingerprint": "3585a053ec6665b8aa81e95f48c45d86887a388eb2e63e38b84f54aa1ded5a47",
        "answer_type": "letter",
        "batch_size": 1,
        "seed": 42,
        "generation": {
            "max_tokens": 1024,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
    },
    "6a30babb1921af59155dfe45cf766465597b57cafa1e0e83663a159d89289b6a": {
        "name": "VALIDATION_6A30BABB",
        "protocol_fingerprint": "92f3c918fc1f14e49251a603eb303f95672a580ff69b8860c0a131d99b24c267",
        "answer_type": "freeform",
        "batch_size": 1,
        "seed": 42,
        "generation": {
            "max_tokens": 1024,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
    },
    "f828a63f1bbdd45197e1f3393fb94f76ebfdfc785402617aa8c1397b0b47c555": {
        "name": "VALIDATION_F828A63F",
        "protocol_fingerprint": "69a99bd671fecaf0156975916e93e74955614a5d4eda739101fb4ca48918bb70",
        "answer_type": "letter",
        "batch_size": 1,
        "seed": 42,
        "generation": {
            "max_tokens": 1024,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
    },
    # Verified PEFT HPO-validation protocol.  This is intentionally keyed by
    # annotation bytes, not by a path or a development dataset name.
    "f120ca66f28e3e5b5a01a3ace93d16c856cf13098faf61b44263a4afc449c709": {
        "name": "PEFT_HPO_VALIDATION_F120CA66",
        "protocol_fingerprint": "9872bf5de29f78f76b4ba39a79a69d57f35ebe2d9080b339cb58ef9233dc33fa",
        "answer_type": "freeform",
        "batch_size": 8,
        "seed": 1,
        "generation": {
            "max_tokens": 1024,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WorkflowError(f"expected a JSON object: {path}")
    return value


def _atomic_write(path: Path, text: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_training_plan(path: Path) -> dict[str, Any]:
    plan = _load_json(path)
    artifact = plan.get("plan_artifact")
    if not isinstance(artifact, Mapping):
        raise WorkflowError("training_plan must be a sealed plan artifact")
    if artifact.get("schema_version") != 1:
        raise WorkflowError("training_plan has an unsupported plan artifact schema")
    expected = str(artifact.get("sha256") or "")
    payload = copy.deepcopy(plan)
    payload.pop("plan_artifact", None)
    actual = stable_hash(payload)
    if not expected or expected != actual:
        raise WorkflowError(
            f"sealed training plan checksum mismatch: expected {expected or '<missing>'}, found {actual}"
        )
    if plan.get("action") != "train" or plan.get("backend") not in {"cosmos-rl", "cosmos-framework"}:
        raise WorkflowError("training_plan must be a Cosmos train plan with an explicit backend")
    required = ("training", "datasets", "model", "compute")
    missing = [key for key in required if not isinstance(plan.get(key), Mapping)]
    if missing:
        raise WorkflowError(f"training plan is missing required sections: {missing}")
    return plan


def _status_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        records = value.get("records", [value])
    elif isinstance(value, list):
        records = value
    else:
        records = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WorkflowError(f"invalid structured status JSON at line {number}: {exc}") from exc
            if isinstance(item, dict):
                records.append(item)
    if not records or not all(isinstance(item, dict) for item in records):
        raise WorkflowError("structured training status contains no object records")
    terminal = str(records[-1].get("status", "")).upper()
    if terminal not in SUCCESS:
        raise WorkflowError(
            f"training status is not terminal-success ({terminal or 'missing'}); checkpoint evaluation is blocked"
        )
    return records


def _checkpoint_events(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        values: dict[str, Any] = {}
        for key in ("kpi", "metrics", "data"):
            if isinstance(record.get(key), Mapping):
                values.update(record[key])
        path = record.get("checkpoint_path", values.get("checkpoint_path", values.get("checkpoint/path")))
        if not path:
            continue
        epoch = values.get("epoch", record.get("epoch"))
        if epoch is None:
            match = re.search(r"(?:^|/)epoch_(\d+)(?:/|$)", str(path))
            if match:
                epoch = int(match.group(1))
        key = (str(path), str(epoch))
        if key in seen:
            continue
        seen.add(key)
        events.append(
            {
                "path": str(path),
                "epoch": epoch,
                "phase": str(record.get("phase", values.get("phase", ""))),
            }
        )
    return events


def _identity_originals(dataset: Mapping[str, Any], key: str) -> list[str]:
    values = dataset.get(key, [])
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        if isinstance(value, Mapping):
            original = value.get("original")
            if isinstance(original, str) and original:
                result.append(original)
    return result


def _prepared_base_model(plan: Mapping[str, Any]) -> str:
    preparation = plan.get("model_preparation", {})
    output = preparation.get("output", {}) if isinstance(preparation, Mapping) else {}
    if isinstance(output, Mapping):
        for key in ("original", "resolved"):
            value = output.get(key)
            if isinstance(value, str) and value:
                return value
    value = plan.get("prepared_model_container_path")
    if isinstance(value, str) and value:
        return value
    supplied = plan.get("model", {}).get("supplied", {})
    if isinstance(supplied, Mapping):
        return str(supplied.get("original") or supplied.get("resolved") or "")
    return ""


def _choose_checkpoint(
    *,
    explicit: str | None,
    epoch: int | None,
    evaluation_contract: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> tuple[str | None, str | None, list[dict[str, Any]]]:
    if explicit and epoch is not None:
        raise WorkflowError("supply either checkpoint or checkpoint_epoch, not both")
    if explicit:
        return explicit, "user", list(events)
    recorded = evaluation_contract.get("checkpoint_selection")
    if isinstance(recorded, Mapping) and isinstance(recorded.get("path"), str) and recorded["path"]:
        return recorded["path"], "sealed_training_plan.evaluation_contract", list(events)
    if epoch is not None:
        matches = [event for event in events if str(event.get("epoch")) == str(epoch)]
        if len(matches) != 1:
            raise WorkflowError(
                f"checkpoint_epoch={epoch} matched {len(matches)} structured checkpoint events; supply the exact checkpoint"
            )
        return str(matches[0]["path"]), "training_status.epoch", list(events)
    if len(events) == 1:
        return str(events[0]["path"]), "training_status.single_checkpoint", list(events)
    return None, None, list(events)


def _source(value: Any, origin: str) -> dict[str, Any]:
    return {"value": value, "source": origin}


def _supporting_files(*names: str) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for name in names:
        path = SCRIPT_ROOT / name
        if not path.is_file():
            raise WorkflowError(f"declared action helper is missing: {path}")
        files.append(
            {
                "source": f"scripts/{name}",
                "destination": name,
                "sha256": sha256_file(path),
            }
        )
    return files


def _selected_image(plan: Mapping[str, Any]) -> str:
    image = plan.get("image")
    if not isinstance(image, Mapping):
        return ""
    for key in ("tag", "container_image", "selected"):
        value = image.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _framework_runtime_preflight(config: Mapping[str, Any]) -> str:
    vision = config.get("vision") if isinstance(config.get("vision"), Mapping) else {}
    expected_runtime = {
        "video_decoder": "torchcodec-cuda-on-demand",
        "process_threads": 8,
        "decoder_device": "cuda",
        "dataloader_multiprocessing_context": "spawn",
    }
    mismatches = {
        key: {"expected": value, "actual": vision.get(key)}
        for key, value in expected_runtime.items()
        if vision.get(key) != value
    }
    worker_profile = (
        int(vision.get("dataloader_num_workers", -1)),
        int(vision.get("dataloader_prefetch_factor", -1)),
        bool(vision.get("dataloader_persistent_workers")),
    )
    if (
        mismatches
        or worker_profile not in {(1, 2, True), (0, 0, False)}
        or int(vision.get("decoder_threads") or 0) <= 0
        or int(vision.get("video_cache_size") or 0) <= 0
        or int(vision.get("max_pixels") or 0) <= 0
        or int(vision.get("min_pixels") or 0) != int(vision.get("max_pixels") or 0)
    ):
        raise WorkflowError(
            "Framework evaluation config does not preserve its sealed TorchCodec runtime: "
            f"runtime={json.dumps(mismatches, sort_keys=True)} worker_profile={worker_profile}"
        )
    worker_probe = (
        "assert 'if self.dataloader_num_workers == 0' in source; "
        "assert ('yield self.prepare_tasks' in source or "
        "'TAO_FRAMEWORK_DIRECT_PREFETCH_ATTESTATION' in source); "
        if worker_profile == (0, 0, False)
        else "assert 'persistent_workers=self.dataloader_persistent_workers' in source; "
        "assert 'multiprocessing_context=self.dataloader_multiprocessing_context' in source; "
    )
    probe = (
        "import inspect; "
        "from cosmos_rl.evaluation.base import BaseEvaluator; "
        "from cosmos_rl.framework.runtime import CosmosFrameworkRuntime; "
        "from cosmos_rl.utils.framework_torchcodec_video import FrameworkTorchCodecVideoPreprocessor; "
        "assert 'torchcodec-cuda-on-demand' in inspect.getsource(BaseEvaluator.load_model); "
        "assert '_framework_decoded_media' in inspect.getsource(CosmosFrameworkRuntime._task_conversation); "
        "source = inspect.getsource(FrameworkTorchCodecVideoPreprocessor); "
        "assert 'iter_prepared_batches' in source; "
        + worker_probe
        + "assert FrameworkTorchCodecVideoPreprocessor.__name__ == 'FrameworkTorchCodecVideoPreprocessor'"
    )
    return "/workspace/.venv/bin/python -c " + shlex.quote(probe)


def _evaluation_spec_bundle(
    plan: Mapping[str, Any], backend: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    image = _selected_image(plan)
    if not image:
        raise WorkflowError("sealed training plan does not record the selected backend image")
    total_gpus = int(config.get("num_gpus") or 0)
    compute = plan.get("compute") if isinstance(plan.get("compute"), Mapping) else {}
    gpus_per_node = int(compute.get("gpus_per_node") or total_gpus)
    if total_gpus <= 0 or gpus_per_node <= 0 or total_gpus % gpus_per_node:
        raise WorkflowError(
            f"evaluation GPU topology is invalid: total={total_gpus}, per_node={gpus_per_node}"
        )
    nodes = total_gpus // gpus_per_node
    model = config.get("model") if isinstance(config.get("model"), Mapping) else {}
    dataset = config.get("dataset") if isinstance(config.get("dataset"), Mapping) else {}
    results_dir = str(config.get("results_dir") or "")
    command_name = (
        "cosmos-framework-evaluate" if backend == "cosmos-framework" else "cosmos-rl-evaluate"
    )
    environment = {
        "NCCL_DEBUG": "WARN",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,video",
        "PYTHONHASHSEED": str(config.get("evaluation", {}).get("seed", 0)),
        "PYTHONUNBUFFERED": "1",
        "TAO_API_JOB_ID": "{job_id}",
        "TAO_API_RESULTS_DIR": "{results_dir}",
        "TAO_JOB_ID": "{job_id}",
        "TAO_RESULTS_ROOT": "{results_dir}",
        "TAO_STATUS_FILE": "{results_dir}/status.json",
        "TAO_STATUS_PATH": "{results_dir}/status.json",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    }
    vision = config.get("vision") if isinstance(config.get("vision"), Mapping) else {}
    if backend == "cosmos-rl":
        environment.update(
            {
                "FORCE_QWENVL_VIDEO_READER": "pynvvideocodec",
                "TAO_PYNV_DECODER_CACHE_SIZE": str(vision.get("decoder_cache_size", 4)),
                "TAO_PYNV_FRAME_TRANSFER": str(vision.get("frame_transfer", "host_rgb")),
                "TAO_PYNV_VIDEO_CACHE_SIZE": str(vision.get("video_cache_size", 0)),
            }
        )
    pre_commands = (
        [_framework_runtime_preflight(config)]
        if backend == "cosmos-framework"
        else []
    )
    runtime_spec = copy.deepcopy(dict(config))
    runtime_spec["results_dir"] = "{results_dir}"
    return {
        "network_arch": "cosmos-reason",
        "action": "evaluate",
        "image": image,
        "mode": "config",
        "command": f"{command_name} --config {{config_path}}",
        "config_format": "toml",
        "spec": runtime_spec,
        "declared_inputs": [
            {"spec_key": "dataset.annotation_path", "type": "file", "uri": str(dataset.get("annotation_path") or "")},
            {"spec_key": "dataset.media_dir", "type": "folder", "uri": str(dataset.get("media_dir") or "")},
            {"spec_key": "model.model_name", "type": "folder", "uri": str(model.get("model_name") or "")},
        ],
        "declared_outputs": [{"spec_key": "results_dir", "type": "folder"}],
        "upload_excludes": ["inputs/"],
        "compute_shape": {"gpus": gpus_per_node, "nodes": nodes},
        "gpu_spec_key": "num_gpus",
        "execution": {
            "environment": environment,
            "pre_commands": pre_commands,
            "distributed": {
                "launcher": "torchrun",
                "processes_per_node": gpus_per_node,
                "tasks_per_node": 1,
            },
            "supporting_files": [],
            "completion": {
                "child_exit_code_path": "{results_dir}/child_exit_code",
                "structured_status_path": "{results_dir}/status.json",
                "success_states": ["SUCCESS"],
            },
        },
    }


def _verify_cosmos_rl_checkpoint_manifest(
    manifest_path: Path,
    *,
    source_checkpoint: str,
    action_model_path: str,
    training_mode: str,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path.expanduser().resolve())
    if manifest.get("schema_version") != 1 or manifest.get("status") != "VERIFIED":
        raise WorkflowError(
            "Cosmos-RL checkpoint manifest is not terminal VERIFIED schema version 1"
        )
    if manifest.get("backend") != "cosmos-rl":
        raise WorkflowError("Cosmos-RL checkpoint manifest has the wrong backend")
    expected = {
        "source_checkpoint": source_checkpoint,
        "action_model_path": action_model_path,
        "training_mode": training_mode,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise WorkflowError(
            "Cosmos-RL checkpoint manifest does not bind the selected training artifact: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise WorkflowError(
            "Cosmos-RL checkpoint manifest has no verified file inventory"
        )
    for item in files:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("path"), str)
            or not item["path"]
            or Path(item["path"]).is_absolute()
            or ".." in Path(item["path"]).parts
            or not isinstance(item.get("size"), int)
            or item["size"] <= 0
            or not re.fullmatch(r"[a-f0-9]{64}", str(item.get("sha256") or ""))
        ):
            raise WorkflowError("Cosmos-RL checkpoint manifest has an invalid file inventory")
    return manifest


def _verify_framework_checkpoint_manifest(
    manifest_path: Path,
    *,
    source_checkpoint: str,
    action_model_path: str,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path.expanduser().resolve())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "VERIFIED"
        or manifest.get("backend") != "cosmos-framework"
    ):
        raise WorkflowError(
            "Framework checkpoint action manifest is not terminal VERIFIED schema version 1"
        )
    expected = {
        "source_checkpoint": source_checkpoint,
        "action_model_path": action_model_path,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise WorkflowError(
            "Framework checkpoint action manifest does not bind the selected DCP/export: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    verification = manifest.get("verification")
    if not isinstance(verification, Mapping) or verification.get("ok") is not True:
        raise WorkflowError("Framework checkpoint action manifest lacks export verification")
    if verification.get("action_model_path") != action_model_path:
        raise WorkflowError("Framework export verification path does not match action_model_path")
    weights = verification.get("weight_files")
    if not isinstance(weights, list) or not weights:
        raise WorkflowError("Framework checkpoint action manifest has no verified weight inventory")
    return manifest


def _verified_evaluator_profile(validation: Mapping[str, Any]) -> dict[str, Any] | None:
    manifests = validation.get("annotation_manifest", [])
    if not isinstance(manifests, list) or len(manifests) != 1:
        return None
    manifest = manifests[0]
    if not isinstance(manifest, Mapping):
        return None
    fingerprint = str(manifest.get("sha256") or "").removeprefix("sha256:")
    selected = VERIFIED_EVALUATOR_PROFILES.get(fingerprint)
    return copy.deepcopy(selected) if selected is not None else None


def resolve(args: argparse.Namespace) -> dict[str, Any]:
    training_plan_path = args.training_plan.expanduser().resolve()
    plan = _verify_training_plan(training_plan_path)
    backend = str(plan["backend"])
    training = plan["training"]
    validation = plan["datasets"]["validation"]
    evaluation_contract = plan.get("evaluation_contract", {})
    if not isinstance(evaluation_contract, Mapping):
        evaluation_contract = {}
    profile = evaluation_contract.get("task_profile", validation.get("evaluation_profile", {}))
    if not isinstance(profile, Mapping):
        profile = {}
    verified_profile = _verified_evaluator_profile(validation)
    verified_profile_source = ""
    if verified_profile is not None:
        verified_profile_source = (
            f"verified_evaluator_profile.{verified_profile['name']}:"
            f"sha256:{verified_profile['protocol_fingerprint']}"
        )

    required_user_inputs: list[dict[str, Any]] = []
    automated_actions: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}

    annotations = list(args.validation_annotation)
    if annotations:
        provenance["dataset.annotation_path"] = _source(annotations, "user")
    else:
        annotations = list(evaluation_contract.get("validation_annotations", []))
        if not annotations:
            annotations = _identity_originals(validation, "annotations")
        provenance["dataset.annotation_path"] = _source(annotations, "sealed_training_plan.validation")
    resolved_annotation = args.action_validation_annotation
    if len(annotations) > 1 and not resolved_annotation:
        automated_actions.append(
            {
                "action": "materialize_exact_validation_manifest",
                "owner": "scripts/cosmos_common.py materialize-dataset",
                "input_annotations": annotations,
                "validation_dataset_fingerprint": validation.get("dataset_fingerprint"),
                "required_output": "action_validation_annotation",
                "user_input": False,
            }
        )
    elif len(annotations) == 1 and not resolved_annotation:
        resolved_annotation = annotations[0]
    elif not annotations:
        required_user_inputs.append(
            {
                "field": "dataset.annotation_path",
                "reason": "no validation annotation was recorded",
                "recorded_candidates": annotations,
            }
        )
    if args.action_validation_annotation:
        provenance["dataset.annotation_path"] = _source(
            args.action_validation_annotation, "materialize_exact_validation_manifest"
        )

    media_roots = list(args.validation_media_root)
    if media_roots:
        provenance["dataset.media_dir"] = _source(media_roots, "user")
    else:
        media_roots = list(evaluation_contract.get("validation_media_roots", []))
        if not media_roots:
            media_roots = _identity_originals(validation, "media_roots")
        provenance["dataset.media_dir"] = _source(media_roots, "sealed_training_plan.validation")
    unique_media_roots = list(dict.fromkeys(media_roots))
    resolved_media_root = args.action_validation_media_root
    if len(unique_media_roots) == 1 and not resolved_media_root:
        resolved_media_root = unique_media_roots[0]
    elif len(unique_media_roots) > 1 and not resolved_media_root:
        automated_actions.append(
            {
                "action": "materialize_validation_manifest_with_absolute_media",
                "owner": "scripts/cosmos_common.py",
                "input_media_roots": unique_media_roots,
                "validation_dataset_fingerprint": validation.get("dataset_fingerprint"),
                "required_output": "action_validation_media_root",
                "user_input": False,
            }
        )
    elif not unique_media_roots:
        required_user_inputs.append(
            {
                "field": "dataset.media_dir",
                "reason": "no validation media root was recorded",
                "recorded_candidates": media_roots,
            }
        )
    if args.action_validation_media_root:
        provenance["dataset.media_dir"] = _source(
            args.action_validation_media_root, "materialize_validation_manifest_with_absolute_media"
        )

    if args.system_prompt is not None:
        system_prompt = args.system_prompt
        provenance["dataset.system_prompt"] = _source(system_prompt, "user")
    elif "system_prompt" in evaluation_contract:
        system_prompt = evaluation_contract["system_prompt"]
        provenance["dataset.system_prompt"] = _source(system_prompt, "sealed_training_plan.evaluation_contract")
    elif "system_prompt" in training:
        system_prompt = training["system_prompt"]
        provenance["dataset.system_prompt"] = _source(system_prompt, "sealed_training_plan.training")
    else:
        system_prompt = None
        required_user_inputs.append(
            {"field": "dataset.system_prompt", "reason": "training artifacts do not record it; an explicit empty string is valid"}
        )

    task_type = args.task_type
    if task_type is not None:
        provenance["task.type"] = _source(task_type, "user")
    elif "inferred_task_type" in profile and not profile.get("unresolved_accuracy_tasks"):
        task_type = str(profile.get("inferred_task_type", ""))
        provenance["task.type"] = _source(task_type, "sealed_training_plan.validation.evaluation_profile")
    else:
        task_type = None
        required_user_inputs.append(
            {
                "field": "task.type",
                "reason": "validation answer semantics are not unambiguous in the sealed training plan",
                "recorded_profile": profile,
            }
        )

    answer_type = args.answer_type
    if answer_type is not None:
        provenance["evaluation.answer_type"] = _source(answer_type, "user")
    elif verified_profile is not None:
        answer_type = str(verified_profile["answer_type"])
        provenance["evaluation.answer_type"] = _source(answer_type, verified_profile_source)
    elif profile.get("answer_type"):
        answer_type = str(profile["answer_type"])
        provenance["evaluation.answer_type"] = _source(answer_type, "sealed_training_plan.validation.evaluation_profile")
    else:
        answer_type = None
        required_user_inputs.append(
            {"field": "evaluation.answer_type", "reason": "not inferable from validation task semantics"}
        )

    metric_names = list(args.metric)
    if metric_names:
        provenance["metrics.names"] = _source(metric_names, "user")
    elif (
        isinstance(profile.get("metric_names"), list)
        and "metrics.names" not in profile.get("requires_user_input", [])
    ):
        metric_names = list(profile["metric_names"])
        provenance["metrics.names"] = _source(metric_names, "sealed_training_plan.validation.evaluation_profile")
    else:
        required_user_inputs.append(
            {"field": "metrics.names", "reason": "no evaluation metric semantics were recorded"}
        )

    generation_contract = evaluation_contract.get("generation", {})
    if not isinstance(generation_contract, Mapping):
        generation_contract = {}
    max_tokens = args.generation_max_tokens
    if max_tokens is not None:
        provenance["generation.max_tokens"] = _source(max_tokens, "user")
    elif verified_profile is not None:
        max_tokens = int(verified_profile["generation"]["max_tokens"])
        provenance["generation.max_tokens"] = _source(max_tokens, verified_profile_source)
    elif generation_contract.get("max_tokens") is not None:
        max_tokens = int(generation_contract["max_tokens"])
        provenance["generation.max_tokens"] = _source(max_tokens, "sealed_training_plan.evaluation_contract")
    elif backend == "cosmos-framework" and (
        answer_type == "letter" or task_type in {"binary", "mcq"}
    ):
        # The Framework evaluator extracts a bounded classification label and
        # already clamps letter generation to ten tokens at runtime. Resolve
        # that same bound during planning instead of asking the user for an
        # irrelevant free-form generation length. Keep Cosmos-RL unchanged.
        max_tokens = 10
        provenance["generation.max_tokens"] = _source(
            max_tokens, "framework_bounded_classification_protocol"
        )
    else:
        required_user_inputs.append(
            {"field": "generation.max_tokens", "reason": "generation length is not a fine-tuning parameter"}
        )

    status_events: list[dict[str, Any]] = []
    if args.training_status:
        status_events = _checkpoint_events(_status_records(args.training_status.expanduser().resolve()))
    checkpoint, checkpoint_source, status_events = _choose_checkpoint(
        explicit=args.checkpoint,
        epoch=args.checkpoint_epoch,
        evaluation_contract=evaluation_contract,
        events=status_events,
    )
    if checkpoint:
        provenance["checkpoint"] = _source(checkpoint, checkpoint_source or "unknown")
    else:
        required_user_inputs.append(
            {
                "field": "checkpoint_selection",
                "reason": "no single exact checkpoint is selected by the training artifacts",
                "recorded_candidates": status_events,
            }
        )

    if not args.results_dir:
        required_user_inputs.append(
            {"field": "results_dir", "reason": "evaluation outputs require a new user-owned path"}
        )
    else:
        provenance["results_dir"] = _source(args.results_dir, "user")

    training_mode = str(training.get("training_mode", ""))
    action_model_path = args.action_model_path
    action_checkpoint_manifest: dict[str, Any] | None = None
    if backend == "cosmos-framework" and checkpoint and not action_model_path:
        automated_actions.append(
            {
                "action": "framework_checkpoint_pre_action",
                "owner": "scripts/framework_checkpoint_action.py",
                "supporting_files": _supporting_files(
                    "framework_checkpoint_action.py", "cosmos_common.py"
                ),
                "input_checkpoint": checkpoint,
                "required_output": "action_model_path",
                "user_input": False,
            }
        )
    if backend == "cosmos-rl" and checkpoint and not action_model_path:
        automated_actions.append(
            {
                "action": "cosmos_rl_checkpoint_pre_action",
                "owner": "scripts/cosmos_rl_checkpoint_action.py",
                "supporting_files": _supporting_files(
                    "cosmos_rl_checkpoint_action.py", "cosmos_common.py"
                ),
                "input_checkpoint": checkpoint,
                "checkpoint_epoch": next(
                    (
                        event.get("epoch")
                        for event in status_events
                        if event.get("path") == checkpoint
                    ),
                    None,
                ),
                "training_mode": training_mode,
                "required_outputs": ["action_model_path", "action_model_manifest"],
                "user_input": False,
            }
        )
    if backend == "cosmos-rl" and action_model_path and checkpoint:
        action_model_manifest_path = getattr(args, "action_model_manifest", None)
        if not action_model_manifest_path:
            raise WorkflowError(
                "Cosmos-RL action_model_path requires --action-model-manifest from "
                "cosmos_rl_checkpoint_action.py"
            )
        action_checkpoint_manifest = _verify_cosmos_rl_checkpoint_manifest(
            action_model_manifest_path,
            source_checkpoint=checkpoint,
            action_model_path=action_model_path,
            training_mode=training_mode,
        )
    if backend == "cosmos-framework" and action_model_path and checkpoint:
        action_model_manifest_path = getattr(args, "action_model_manifest", None)
        if not action_model_manifest_path:
            raise WorkflowError(
                "Framework action_model_path requires --action-model-manifest from "
                "framework_checkpoint_action.py prepare/verify"
            )
        action_checkpoint_manifest = _verify_framework_checkpoint_manifest(
            action_model_manifest_path,
            source_checkpoint=checkpoint,
            action_model_path=action_model_path,
        )
    model_name = action_model_path if action_model_path and checkpoint else (
        checkpoint if backend == "cosmos-framework" else None
    )
    if backend == "cosmos-framework" and action_model_path:
        provenance["model.model_name"] = _source(action_model_path, "framework_checkpoint_pre_action")
    elif backend == "cosmos-rl" and action_model_path and checkpoint:
        provenance["model.model_name"] = _source(action_model_path, "cosmos_rl_checkpoint_pre_action")
    elif model_name:
        provenance["model.model_name"] = _source(model_name, "selected_checkpoint")

    enable_lora = backend == "cosmos-rl" and training_mode == "peft"
    base_model_path = _prepared_base_model(plan) if enable_lora else ""
    if enable_lora and not base_model_path:
        automated_actions.append(
            {
                "action": "recover_prepared_base_model_from_training_provenance",
                "required_output": "model.base_model_path",
                "user_input": False,
            }
        )
    provenance["model.enable_lora"] = _source(enable_lora, "sealed_training_plan.training_mode_and_backend")
    provenance["model.base_model_path"] = _source(base_model_path, "sealed_training_plan.model_preparation")

    inherited_vision = evaluation_contract.get("vision")
    if not isinstance(inherited_vision, Mapping):
        inherited_vision = {}
    inherited_vision = dict(inherited_vision)
    frames = int(
        inherited_vision.get("nframes")
        or evaluation_contract.get("frames")
        or training.get("frames")
        or 0
    )
    fps = inherited_vision.get("fps")
    if args.max_video_pixels is not None:
        max_video_pixels = args.max_video_pixels
        provenance["vision.max_pixels"] = _source(max_video_pixels, "user")
    else:
        recorded_max_video_pixels = (
            evaluation_contract.get("max_video_pixels")
            or plan.get("processor_profile", {}).get("max_video_pixels")
        )
        max_video_pixels = int(recorded_max_video_pixels) if recorded_max_video_pixels else None
    precision = str(evaluation_contract.get("precision") or training.get("precision") or "")
    requested_model_max_length = getattr(args, "model_max_length", None)
    max_length = int(
        requested_model_max_length
        if requested_model_max_length is not None
        else training.get("sequence_length") or 0
    )
    if args.evaluation_seed is not None:
        seed = args.evaluation_seed
        seed_source = "user"
    elif verified_profile is not None:
        seed = int(verified_profile["seed"])
        seed_source = verified_profile_source
    else:
        seed = int(evaluation_contract.get("seed", training.get("seed", 0)))
        seed_source = "sealed_training_plan"
    if args.evaluation_batch_size is not None:
        batch_size = args.evaluation_batch_size
        batch_size_source = "user"
    elif verified_profile is not None:
        batch_size = int(verified_profile["batch_size"])
        batch_size_source = verified_profile_source
    else:
        batch_size = int(
            evaluation_contract.get("batch_size")
            or plan.get("planner_request", {}).get("validation_batch_size")
            or 0
        )
        batch_size_source = "sealed_training_plan"
    requested_progress_interval = getattr(
        args, "evaluation_progress_interval_batches", None
    )
    progress_interval_batches = (
        int(requested_progress_interval)
        if requested_progress_interval is not None
        else 16
    )
    num_gpus = args.num_gpus or int(plan["compute"].get("total_gpus") or 0)
    inherited_values = {
        "model.dtype": precision,
        "model.max_length": max_length,
        "model.tp_size": 1,
        "evaluation.seed": seed,
        "evaluation.batch_size": batch_size,
        "num_gpus": num_gpus,
    }
    if fps is not None:
        inherited_values["vision.fps"] = fps
    else:
        inherited_values["vision.num_frames"] = frames
    for field, value in inherited_values.items():
        provenance[field] = _source(value, "sealed_training_plan")
        if value in {"", None} or (value == 0 and field != "evaluation.seed"):
            required_user_inputs.append(
                {"field": field, "reason": "the sealed training plan did not record a usable value"}
            )
    provenance["evaluation.seed"] = _source(seed, seed_source)
    provenance["evaluation.batch_size"] = _source(batch_size, batch_size_source)
    provenance["evaluation.progress_interval_batches"] = _source(
        progress_interval_batches,
        "user"
        if requested_progress_interval is not None
        else "status_io_throttle_default",
    )
    provenance["model.max_length"] = _source(
        max_length,
        "user" if requested_model_max_length is not None else "sealed_training_plan",
    )
    requested_shard_strategy = getattr(args, "evaluation_shard_strategy", None)
    validation_profile = validation.get("profile", {})
    validation_family = (
        str(validation_profile.get("family", ""))
        if isinstance(validation_profile, Mapping)
        else ""
    )
    shard_strategy = requested_shard_strategy or (
        "media_balanced"
        if validation_family in {"video_conversation", "task_aware_video_reasoning"}
        else "stride"
    )
    optimized_framework_media_profile = (
        backend == "cosmos-framework"
        and validation_family == "video_conversation"
        and shard_strategy == "media_balanced"
    )
    provenance["evaluation.shard_strategy"] = _source(
        shard_strategy,
        "user" if requested_shard_strategy is not None else "dataset_profile",
    )
    if batch_size <= 0:
        required_user_inputs.append(
            {"field": "evaluation.batch_size", "reason": "no usable evaluation batch size was resolved"}
        )
    if args.max_video_pixels is None:
        provenance["vision.max_pixels"] = _source(max_video_pixels, "sealed_training_plan")
    model_tier = str(plan.get("processor_profile", {}).get("model_tier") or "")
    if max_video_pixels is None and model_tier != "nano":
        required_user_inputs.append(
            {
                "field": "vision.max_pixels",
                "reason": "the sealed training plan did not record a usable value",
            }
        )

    if backend == "cosmos-framework":
        framework_runtime = plan.get("framework_video_runtime")
        if not isinstance(framework_runtime, Mapping) or framework_runtime.get(
            "selected_profile"
        ) != "torchcodec-cuda-on-demand":
            raise WorkflowError(
                "Framework evaluation requires the sealed torchcodec-cuda-on-demand runtime profile"
            )
        if framework_runtime.get("decoder_device_binding") != "explicit_local_rank":
            raise WorkflowError(
                "Framework evaluation requires explicit local-rank CUDA decoder binding"
            )
        framework_decoder = "torchcodec-cuda-on-demand"
        decoder_threads_override = getattr(args, "framework_decoder_threads", None)
        framework_decoder_threads = (
            int(framework_runtime["decoder_threads"])
            if decoder_threads_override is None
            else int(decoder_threads_override)
        )
        if framework_decoder_threads <= 0:
            raise WorkflowError("Framework evaluation decoder_threads must be positive")
        framework_cache_override = getattr(args, "framework_video_cache_size", None)
        framework_cache_size = (
            1
            if framework_cache_override is None and optimized_framework_media_profile
            else int(framework_runtime["video_cache_size"])
            if framework_cache_override is None
            else int(framework_cache_override)
        )
        if framework_cache_size < 0:
            raise WorkflowError(
                "Framework evaluation video_cache_size must be non-negative"
            )
        worker_override = getattr(args, "framework_dataloader_num_workers", None)
        framework_workers = (
            0
            if worker_override is None and optimized_framework_media_profile
            else int(framework_runtime["dataloader_num_workers"])
            if worker_override is None
            else int(worker_override)
        )
        if framework_workers not in (0, 1):
            raise WorkflowError(
                "Framework evaluation dataloader_num_workers must be zero or one"
            )
        prefetch_override = getattr(args, "framework_dataloader_prefetch_factor", None)
        framework_prefetch = (
            (0 if framework_workers == 0 else int(framework_runtime["dataloader_prefetch_factor"]))
            if prefetch_override is None
            else int(prefetch_override)
        )
        if framework_workers == 0 and framework_prefetch != 0:
            raise WorkflowError(
                "Framework evaluation prefetch factor must be zero when workers are zero"
            )
        if framework_workers == 1 and framework_prefetch != 2:
            raise WorkflowError(
                "Framework evaluation prefetch factor must be two when one worker is selected"
            )
        vision: dict[str, Any] = {
            "num_frames": frames,
            "video_decoder": framework_decoder,
            "video_cache_size": framework_cache_size,
            "process_threads": int(framework_runtime["sft_process_threads"]),
            "decoder_threads": framework_decoder_threads,
            "decoder_device": str(framework_runtime["decoder_device"]),
            "dataloader_num_workers": framework_workers,
            "dataloader_prefetch_factor": framework_prefetch,
            "dataloader_multiprocessing_context": str(
                framework_runtime["dataloader_multiprocessing_context"]
            ),
            "dataloader_persistent_workers": (
                framework_workers > 0
                and bool(framework_runtime["dataloader_persistent_workers"])
            ),
        }
        provenance["vision.video_decoder"] = _source(
            vision["video_decoder"],
            "sealed_training_plan.framework_video_runtime",
        )
        provenance["vision.video_cache_size"] = _source(
            vision["video_cache_size"],
            (
                "media_balanced_on_demand_cache"
                if framework_cache_override is None
                and optimized_framework_media_profile
                else
                "sealed_training_plan.framework_video_runtime"
                if framework_cache_override is None
                else "user"
            ),
        )
        provenance["vision.decoder_threads"] = _source(
            vision["decoder_threads"],
            "sealed_training_plan.framework_video_runtime"
            if decoder_threads_override is None
            else "user",
        )
        provenance["vision.dataloader_num_workers"] = _source(
            vision["dataloader_num_workers"],
            "media_balanced_direct_loader"
            if worker_override is None and optimized_framework_media_profile
            else "sealed_training_plan.framework_video_runtime"
            if worker_override is None
            else "user",
        )
        provenance["vision.dataloader_prefetch_factor"] = _source(
            vision["dataloader_prefetch_factor"],
            "sealed_training_plan.framework_video_runtime"
            if prefetch_override is None and framework_workers > 0
            else "derived_zero_worker_profile"
            if prefetch_override is None
            else "user",
        )
    else:
        # Cosmos-RL evaluation owns a distinct PyNvVideoCodec runtime.  Reuse
        # the sealed training runtime when present so a fresh evaluation does
        # not silently fall back to host-RGB transfer or disable the rank-local
        # on-demand processed-video cache.  Explicit evaluation overrides are
        # useful for bounded throughput experiments and remain recorded in the
        # plan provenance.
        rl_runtime = plan.get("rl_video_runtime")
        if not isinstance(rl_runtime, Mapping):
            rl_runtime = {}
        recorded_cache_size = int(rl_runtime.get("video_cache_size", 0))
        recorded_decoder_cache_size = int(rl_runtime.get("decoder_cache_size", 4))
        recorded_frame_transfer = str(rl_runtime.get("frame_transfer", "host_rgb"))
        cache_override = getattr(args, "rl_video_cache_size", None)
        decoder_cache_override = getattr(args, "rl_video_decoder_cache_size", None)
        frame_transfer_override = getattr(args, "rl_video_frame_transfer", None)
        if cache_override is not None:
            cache_size = int(cache_override)
            cache_source = "user"
        elif shard_strategy == "media_balanced" and recorded_cache_size > 0:
            # Media-balanced sharding restores source order after assigning a
            # complete media group to one rank. A single on-demand processed
            # entry therefore captures every adjacent reuse without retaining
            # the full training-era working set in evaluation memory.
            cache_size = 1
            cache_source = "media_balanced_on_demand_cache"
        else:
            cache_size = recorded_cache_size
            cache_source = "sealed_training_plan.rl_video_runtime"
        decoder_cache_size = (
            min(recorded_decoder_cache_size, 4)
            if decoder_cache_override is None
            else int(decoder_cache_override)
        )
        frame_transfer = (
            recorded_frame_transfer
            if frame_transfer_override is None
            else str(frame_transfer_override)
        )
        if cache_size < 0:
            raise WorkflowError("Cosmos-RL evaluation video_cache_size must be non-negative")
        if decoder_cache_size <= 0:
            raise WorkflowError("Cosmos-RL evaluation decoder_cache_size must be positive")
        if frame_transfer not in {"host_rgb", "device_rgbp"}:
            raise WorkflowError(
                "Cosmos-RL evaluation frame_transfer must be host_rgb or device_rgbp"
            )
        vision: dict[str, Any] = {
            "video_decoder": "pynvvideocodec",
            "video_cache_size": cache_size,
            "decoder_cache_size": decoder_cache_size,
            "frame_transfer": frame_transfer,
        }
        if fps is not None:
            vision["fps"] = fps
        else:
            vision["num_frames"] = frames
        runtime_source = (
            "sealed_training_plan.rl_video_runtime" if rl_runtime else "evaluator_default"
        )
        provenance["vision.video_cache_size"] = _source(
            cache_size, cache_source
        )
        provenance["vision.decoder_cache_size"] = _source(
            decoder_cache_size,
            "user" if decoder_cache_override is not None else runtime_source,
        )
        provenance["vision.frame_transfer"] = _source(
            frame_transfer,
            "user" if frame_transfer_override is not None else runtime_source,
        )
    feature_cache_override = getattr(args, "video_feature_cache_size", None)
    processor_cache_override = getattr(args, "video_processor_cache_size", None)
    feature_cache_size = (
        1
        if feature_cache_override is None and optimized_framework_media_profile
        else feature_cache_override
    )
    processor_cache_size = (
        1
        if processor_cache_override is None and optimized_framework_media_profile
        else processor_cache_override
    )
    if feature_cache_size is not None:
        vision["video_feature_cache_size"] = int(feature_cache_size)
        provenance["vision.video_feature_cache_size"] = _source(
            int(feature_cache_size),
            "media_balanced_on_demand_cache"
            if feature_cache_override is None
            else "user",
        )
    if processor_cache_size is not None:
        vision["video_processor_cache_size"] = int(processor_cache_size)
        provenance["vision.video_processor_cache_size"] = _source(
            int(processor_cache_size),
            "media_balanced_on_demand_cache"
            if processor_cache_override is None
            else "user",
        )
    for field in (
        "min_frames",
        "max_frames",
        "video_start",
        "video_end",
        "resized_height",
        "resized_width",
        "min_pixels",
        "total_pixels",
    ):
        value = inherited_vision.get(field)
        if value is not None:
            vision[field] = value
            provenance[f"vision.{field}"] = _source(value, "sealed_training_plan")
    if max_video_pixels is not None:
        vision["max_pixels"] = max_video_pixels
        if backend == "cosmos-framework":
            # Framework evaluation passes already-decoded PIL frames back
            # through qwen-vl-utils.  Qwen otherwise supplies its larger
            # runtime default minimum and rejects a deliberately lower sealed
            # maximum before resizing.  Preserve the requested maximum by
            # making the compatible lower bound explicit.  Keep this scoped
            # to the Framework branch: Cosmos-RL owns the equivalent
            # normalization in its registered video reader.
            vision["min_pixels"] = max_video_pixels
            provenance["vision.min_pixels"] = _source(
                max_video_pixels,
                "framework_preserve_explicit_max_pixels",
            )
    decoder_artifact = plan.get("decoder_artifact", {})
    if backend == "cosmos-rl" and isinstance(decoder_artifact, Mapping) and decoder_artifact.get("enabled"):
        vision["video_override_map"] = decoder_artifact.get("path")
        provenance["vision.video_override_map"] = _source(
            decoder_artifact.get("path"), "sealed_training_plan.decoder_artifact"
        )

    resolved_generation = generation_contract
    if verified_profile is not None:
        resolved_generation = verified_profile["generation"]

    config = {
        "results_dir": args.results_dir or "",
        "task": {"type": task_type if task_type is not None else ""},
        "dataset": {
            "annotation_path": resolved_annotation or "",
            "media_dir": resolved_media_root or "",
            "system_prompt": system_prompt if system_prompt is not None else "",
        },
        "model": {
            "model_name": model_name or "",
            "dtype": precision,
            "max_length": max_length,
            "tp_size": 1,
            "enable_lora": enable_lora,
            "base_model_path": base_model_path,
            "config_file": "",
            "export_dir": "",
            "vit_checkpoint_path": "",
        },
        "evaluation": {
            "answer_type": answer_type or "",
            "num_processes": 1,
            "skip_saved": False,
            "seed": seed,
            "limit": -1,
            "shard_id": 0,
            "batch_size": batch_size,
            "progress_interval_batches": progress_interval_batches,
            "shard_strategy": shard_strategy,
            "barrier_timeout_seconds": 14400,
            "soft_accuracy": {"enabled": True, "f1_threshold": 0.8},
        },
        "vision": vision,
        "generation": {
            "max_retries": 10,
            "max_tokens": max_tokens or 0,
            "temperature": float(resolved_generation.get("temperature", 0.0)),
            "repetition_penalty": float(resolved_generation.get("repetition_penalty", 1.0)),
            "presence_penalty": float(resolved_generation.get("presence_penalty", 0.0)),
            "frequency_penalty": float(resolved_generation.get("frequency_penalty", 0.0)),
        },
        "metrics": {"names": metric_names},
        "results": {
            "save_individual_results": True,
            "save_confusion_matrix": True,
            "save_metrics_summary": True,
        },
        "num_gpus": num_gpus,
    }
    requested_attention = getattr(args, "attention_implementation", None)
    if requested_attention:
        config["model"]["attn_implementation"] = str(requested_attention)
        provenance["model.attn_implementation"] = _source(
            str(requested_attention), "optimization_candidate"
        )

    blockers = list(required_user_inputs)
    if backend == "cosmos-framework" and checkpoint and not action_model_path:
        blockers.append(
            {
                "field": "model.model_name",
                "reason": "awaiting the mandatory automatic Framework checkpoint pre-action",
                "user_input": False,
            }
        )
    if backend == "cosmos-rl" and checkpoint and not action_model_path:
        blockers.append(
            {
                "field": "model.model_name",
                "reason": "awaiting mandatory Cosmos-RL HF safetensors checkpoint verification",
                "user_input": False,
            }
        )
    if enable_lora and not base_model_path:
        blockers.append(
            {
                "field": "model.base_model_path",
                "reason": "awaiting deterministic recovery from training provenance",
                "user_input": False,
            }
        )
    for action in automated_actions:
        output = str(action.get("required_output") or "")
        if output in {"action_validation_annotation", "action_validation_media_root"}:
            blockers.append(
                {
                    "field": output,
                    "reason": f"awaiting automated action {action['action']}",
                    "user_input": False,
                }
            )
    ready = not blockers
    spec_bundle = _evaluation_spec_bundle(plan, backend, config) if ready else None
    result = {
        "schema_version": 1,
        "ready": ready,
        "backend": backend,
        "training_plan": {
            "path": str(training_plan_path),
            "sha256": sha256_file(training_plan_path),
            "experiment_id": plan.get("experiment_id"),
            "dataset_fingerprint": validation.get("dataset_fingerprint"),
            "model_fingerprint": plan.get("model", {}).get("fingerprint"),
        },
        "checkpoint": {
            "selected": checkpoint,
            "source": checkpoint_source,
            "events": status_events,
            "action_model_path": action_model_path,
            "action_model_manifest": action_checkpoint_manifest,
        },
        "verified_evaluator_profile": verified_profile,
        "required_user_inputs": required_user_inputs,
        "automated_actions": automated_actions,
        "blockers": blockers,
        "provenance": provenance,
        "config": config,
        "spec_bundle": spec_bundle,
        "spec_bundle_sha256": stable_hash(spec_bundle) if spec_bundle else None,
        "config_sha256": hashlib.sha256(dump_toml(config).encode()).hexdigest() if ready else None,
    }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--training-status", type=Path)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-epoch", type=int)
    parser.add_argument("--action-model-path")
    parser.add_argument("--action-model-manifest", type=Path)
    parser.add_argument("--action-validation-annotation")
    parser.add_argument("--action-validation-media-root")
    parser.add_argument("--validation-annotation", action="append", default=[])
    parser.add_argument("--validation-media-root", action="append", default=[])
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument(
        "--task-type",
        choices=("", "binary", "mcq", "text", "its_directionality", "metropolis_sgd"),
        default=None,
    )
    parser.add_argument("--answer-type", choices=("letter", "reasoning", "freeform", "naive"))
    parser.add_argument("--evaluation-batch-size", type=int)
    parser.add_argument("--evaluation-progress-interval-batches", type=int)
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument(
        "--evaluation-shard-strategy", choices=("stride", "media_balanced")
    )
    parser.add_argument("--rl-video-cache-size", type=int)
    parser.add_argument("--rl-video-decoder-cache-size", type=int)
    parser.add_argument(
        "--rl-video-frame-transfer", choices=("host_rgb", "device_rgbp")
    )
    parser.add_argument("--framework-video-cache-size", type=int)
    parser.add_argument("--framework-decoder-threads", type=int)
    parser.add_argument("--framework-dataloader-num-workers", type=int)
    parser.add_argument("--framework-dataloader-prefetch-factor", type=int)
    parser.add_argument("--video-feature-cache-size", type=int)
    parser.add_argument("--video-processor-cache-size", type=int)
    parser.add_argument("--generation-max-tokens", type=int)
    parser.add_argument("--model-max-length", type=int)
    parser.add_argument(
        "--attention-implementation",
        choices=("flash_attention_2", "sdpa", "eager", "cosmos"),
    )
    parser.add_argument("--max-video-pixels", type=int)
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--config-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.generation_max_tokens is not None and args.generation_max_tokens <= 0:
            raise WorkflowError("generation_max_tokens must be positive")
        if args.evaluation_batch_size is not None and args.evaluation_batch_size <= 0:
            raise WorkflowError("evaluation_batch_size must be positive")
        if (
            args.evaluation_progress_interval_batches is not None
            and args.evaluation_progress_interval_batches <= 0
        ):
            raise WorkflowError(
                "evaluation_progress_interval_batches must be positive"
            )
        if args.evaluation_seed is not None and args.evaluation_seed < 0:
            raise WorkflowError("evaluation_seed must be non-negative")
        if args.rl_video_cache_size is not None and args.rl_video_cache_size < 0:
            raise WorkflowError("rl_video_cache_size must be non-negative")
        if (
            args.framework_video_cache_size is not None
            and args.framework_video_cache_size < 0
        ):
            raise WorkflowError("framework_video_cache_size must be non-negative")
        if (
            args.framework_dataloader_num_workers is not None
            and args.framework_dataloader_num_workers not in (0, 1)
        ):
            raise WorkflowError(
                "framework_dataloader_num_workers must be zero or one"
            )
        if (
            args.framework_dataloader_prefetch_factor is not None
            and args.framework_dataloader_prefetch_factor < 0
        ):
            raise WorkflowError(
                "framework_dataloader_prefetch_factor must be non-negative"
            )
        if args.video_feature_cache_size is not None and args.video_feature_cache_size < 0:
            raise WorkflowError("video_feature_cache_size must be non-negative")
        if args.video_processor_cache_size is not None and args.video_processor_cache_size < 0:
            raise WorkflowError("video_processor_cache_size must be non-negative")
        if (
            (args.video_feature_cache_size or args.video_processor_cache_size)
            and args.evaluation_batch_size not in (None, 1)
        ):
            raise WorkflowError(
                "video feature/processor caches preserve singleton evaluation; "
                "evaluation_batch_size must be 1"
            )
        if (
            args.rl_video_decoder_cache_size is not None
            and args.rl_video_decoder_cache_size <= 0
        ):
            raise WorkflowError("rl_video_decoder_cache_size must be positive")
        if args.max_video_pixels is not None and args.max_video_pixels <= 0:
            raise WorkflowError("max_video_pixels must be positive")
        if args.num_gpus is not None and args.num_gpus <= 0:
            raise WorkflowError("num_gpus must be positive")
        result = resolve(args)
        _atomic_write(args.plan_output, json.dumps(result, indent=2, sort_keys=True) + "\n")
        if result["ready"]:
            if not args.config_output:
                raise WorkflowError("config_output is required when the evaluation request is ready")
            encoded = dump_toml(result["config"])
            tomllib.loads(encoded)
            _atomic_write(args.config_output, encoded)
            result["config_path"] = str(args.config_output.expanduser().resolve())
            result["config_sha256"] = sha256_file(args.config_output.expanduser().resolve())
            _atomic_write(args.plan_output, json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ready"] else 3
    except (OSError, json.JSONDecodeError, tomllib.TOMLDecodeError, WorkflowError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
