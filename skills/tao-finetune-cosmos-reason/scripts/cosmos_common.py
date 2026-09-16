#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure validation and provenance primitives for Cosmos TAO workflows.

This module deliberately has no machine- or user-specific defaults.  Every
filesystem location in its output originates in a runtime request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


URI_RE = re.compile(r"^(?:hf_model://|https?://|s3://|ngc://|hf://)")
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
ACCURACY_TASKS = {"bcq", "binary", "mcq", "binary_choice", "multiple_choice"}
DATASET_FAMILIES = {"auto", "video_conversation", "task_aware_video_reasoning"}
MEDIA_FIELDS = ("video", "video_id", "image", "image_id", "media", "media_path")
TAO_VL_MEDIA_FIELDS = ("video_id", "image_id")
CLASSIFICATION_LABEL_SETS = (
    frozenset({"a", "b", "c", "d"}),
    frozenset({"yes", "no"}),
)
EVALUATOR_TASK_ALIASES = {
    "bcq": "binary",
    "binary": "binary",
    "binary_choice": "binary",
    "mcq": "mcq",
    "multiple_choice": "mcq",
}


class WorkflowError(ValueError):
    """A deterministic, actionable request or parity failure."""


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def path_identity(value: str, *, required: bool = True) -> dict[str, Any]:
    """Preserve a supplied path and add non-destructive normalization details."""
    if not value:
        if required:
            raise WorkflowError("required runtime path is missing")
        return {"original": "", "expanded": "", "resolved": None, "exists": False}
    expanded = str(Path(value).expanduser())
    path = Path(expanded)
    exists = path.exists()
    return {
        "original": value,
        "expanded": expanded,
        "resolved": str(path.resolve()) if exists else None,
        "exists": exists,
        "kind": "directory" if path.is_dir() else "file" if path.is_file() else "missing",
    }


def is_model_uri(value: str) -> bool:
    return bool(URI_RE.match(value) or MODEL_ID_RE.fullmatch(value))


def _file_inventory(
    root: Path, names: Iterable[str], *, hash_content: bool = True
) -> list[dict[str, Any]]:
    inventory = []
    for name in names:
        path = root / name
        if path.is_file():
            entry = {"path": name, "size": path.stat().st_size}
            if hash_content:
                entry["sha256"] = sha256_file(path)
            inventory.append(entry)
    return inventory


def inspect_model(
    value: str,
    revision: str = "",
    prepared: str = "",
    *,
    fast_weight_fingerprint: bool = False,
) -> dict[str, Any]:
    if not value:
        raise WorkflowError("base_model_path_or_uri is required for every Cosmos training request")
    supplied = path_identity(value)
    result: dict[str, Any] = {
        "supplied": supplied,
        "revision": revision or None,
        "prepared_checkpoint": path_identity(prepared, required=False),
    }
    if supplied["exists"]:
        root = Path(supplied["resolved"])
        if not root.is_dir():
            raise WorkflowError(f"base model must be a directory or supported URI: {value}")
        config_path = root / "config.json"
        if not config_path.is_file():
            raise WorkflowError(f"base model is missing config.json: {value}")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise WorkflowError(f"base model config.json is invalid: {value}: {exc}") from exc
        weight_files = sorted(p.name for p in root.glob("*.safetensors"))
        index = root / "model.safetensors.index.json"
        if not weight_files and not index.is_file():
            raise WorkflowError(f"base model contains no safetensors weights or index: {value}")
        indexed_weight_files: list[str] = []
        if index.is_file():
            try:
                index_payload = json.loads(index.read_text(encoding="utf-8"))
                weight_map = index_payload.get("weight_map", {})
            except json.JSONDecodeError as exc:
                raise WorkflowError(f"model safetensors index is invalid: {index}: {exc}") from exc
            if not isinstance(weight_map, dict) or not weight_map:
                raise WorkflowError(f"model safetensors index has no weight_map: {index}")
            indexed_weight_files = sorted(set(weight_map.values()))
            for relative in indexed_weight_files:
                if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise WorkflowError(f"model safetensors index contains an unsafe weight path: {relative!r}")
                if not (root / relative).is_file():
                    raise WorkflowError(f"model safetensors index references a missing weight file: {relative}")
        metadata_files = [
            "config.json", "generation_config.json", "model.safetensors.index.json",
            "tokenizer.json", "tokenizer_config.json", "processor_config.json",
            "preprocessor_config.json", "chat_template.json",
        ]
        weight_names = list(dict.fromkeys([*weight_files, *indexed_weight_files]))
        inventory = _file_inventory(root, metadata_files) + _file_inventory(
            root, weight_names, hash_content=not fast_weight_fingerprint
        )
        result.update(
            {
                "source_type": "local",
                "format": config.get("model_type", "unknown"),
                "config": {"model_type": config.get("model_type"), "architectures": config.get("architectures")},
                "files": inventory,
                "fingerprint": stable_hash(inventory),
                "fingerprint_mode": (
                    "metadata_content_and_weight_sizes"
                    if fast_weight_fingerprint
                    else "full_content"
                ),
            }
        )
    elif is_model_uri(value):
        if not revision:
            raise WorkflowError(
                "base_model_revision is required for a model URI/identifier so a clean run is immutable"
            )
        result.update(
            {
                "source_type": "uri",
                "format": "unresolved",
                "fingerprint": stable_hash({"uri": value, "revision": revision}),
            }
        )
    else:
        raise WorkflowError(f"base model path is inaccessible and is not a supported URI: {value}")

    if prepared:
        prepared_id = result["prepared_checkpoint"]
        if not prepared_id["exists"] or prepared_id["kind"] != "directory":
            raise WorkflowError(f"prepared checkpoint is inaccessible: {prepared}")
        prepared_result = inspect_model(
            prepared, fast_weight_fingerprint=fast_weight_fingerprint
        )
        result["prepared_checkpoint"].update(
            {
                "format": prepared_result["format"],
                "fingerprint": prepared_result["fingerprint"],
                "files": prepared_result["files"],
            }
        )
    return result


def load_annotation(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"cannot read annotation {path}: {exc}") from exc
    if isinstance(payload, list):
        records, metadata = payload, {}
    elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
        records, metadata = payload["items"], payload.get("metadata", {}) or {}
    else:
        raise WorkflowError(f"annotation must be a JSON array or an object containing an items array: {path}")
    if not records:
        raise WorkflowError(f"annotation contains zero records: {path}")
    if not all(isinstance(record, dict) for record in records):
        raise WorkflowError(f"annotation contains a non-object record: {path}")
    return records, metadata if isinstance(metadata, dict) else {}


def _record_media(record: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for field in MEDIA_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value:
            values.append(value)
        elif isinstance(value, list):
            values.extend(item for item in value if isinstance(item, str) and item)
    return values


def decode_media(path: Path) -> None:
    """Decode-probe one frame so corrupt video inputs fail before launch."""
    if path.suffix.casefold() not in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}:
        return
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise WorkflowError(
            "media decode validation requires ffprobe on the inspection host; "
            f"cannot validate {path}"
        )
    completed = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#1",
         "-show_entries", "frame=media_type", "-of", "csv=p=0", str(path)],
        text=True, capture_output=True, check=False,
    )
    if completed.returncode or "video" not in completed.stdout.split():
        detail = completed.stderr.strip().splitlines()
        reason = detail[-1] if detail else "no decodable video frame"
        raise WorkflowError(f"media decode validation failed for {path}: {reason}")


def _record_key(record: Mapping[str, Any]) -> str:
    identity = {
        "id": record.get("id") or record.get("sample_id") or record.get("video_id"),
        "media": _record_media(record),
        "question": record.get("question"),
        "conversations": record.get("conversations"),
    }
    return stable_hash(identity)


def _record_task(record: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    value = record.get("task") or record.get("task_type") or metadata.get("task") or ""
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def _conversation_target(record: Mapping[str, Any]) -> str | None:
    conversations = record.get("conversations") or record.get("messages")
    if not isinstance(conversations, list) or not conversations:
        return None
    final = conversations[-1]
    if not isinstance(final, Mapping):
        return None
    value = final.get("value") if "value" in final else final.get("content")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _detect_dataset_family(records: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> str:
    if metadata.get("task") or metadata.get("tasks") or any(_record_task(record, metadata) for record in records):
        return "task_aware_video_reasoning"
    if all(isinstance(record.get("conversations") or record.get("messages"), list) for record in records):
        return "video_conversation"
    raise WorkflowError("cannot infer dataset family from annotations; specify a supported structural family")


def _numeric_metadata(record: Mapping[str, Any], metadata: Mapping[str, Any]) -> tuple[float | None, float | None, float | None, float | None]:
    combined = {**metadata, **record}
    resolution = combined.get("resolution")
    width = combined.get("width") or combined.get("video_width")
    height = combined.get("height") or combined.get("video_height")
    if isinstance(resolution, Mapping):
        width = width or resolution.get("width")
        height = height or resolution.get("height")
    elif isinstance(resolution, Sequence) and not isinstance(resolution, (str, bytes)) and len(resolution) >= 2:
        width, height = width or resolution[0], height or resolution[1]
    fps = combined.get("fps") or combined.get("video_fps")
    duration = combined.get("duration") or combined.get("duration_seconds")

    def number(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    return number(width), number(height), number(fps), number(duration)


def inspect_dataset(
    *,
    dataset_family: str,
    annotations: Sequence[str],
    media_roots: Sequence[str],
    selected_tasks: Sequence[str] = (),
    verify_media_content: bool = True,
) -> dict[str, Any]:
    if not annotations:
        raise WorkflowError("at least one runtime annotation path is required")
    if not media_roots:
        raise WorkflowError("at least one runtime media root is required")
    if len(media_roots) not in {1, len(annotations)}:
        raise WorkflowError("supply one shared media root or one media root per annotation")
    dataset_family = dataset_family.casefold()
    if dataset_family not in DATASET_FAMILIES:
        raise WorkflowError(f"unsupported dataset family: {dataset_family}")
    requested_tasks = {item.casefold().replace("-", "_") for item in selected_tasks}

    annotation_ids = [path_identity(value) for value in annotations]
    media_ids = [path_identity(value) for value in media_roots]
    for item in annotation_ids:
        if not item["exists"] or item["kind"] != "file":
            raise WorkflowError(f"annotation path is inaccessible: {item['original']}")
    for item in media_ids:
        if not item["exists"] or item["kind"] != "directory":
            raise WorkflowError(f"media root is inaccessible: {item['original']}")

    record_keys: list[str] = []
    task_counts: dict[str, int] = {}
    manifest_entries: list[dict[str, Any]] = []
    media_entries: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    schema_errors: list[str] = []
    observed_families: set[str] = set()
    widths: list[float] = []
    heights: list[float] = []
    frame_rates: list[float] = []
    durations: list[float] = []
    task_metrics: dict[str, str] = {}
    answer_targets: dict[str, list[str]] = {}
    for annotation_index, annotation_id in enumerate(annotation_ids):
        # Keep dataset runtime paths in the caller's lexical namespace. A
        # shared-filesystem alias where the submitted and canonical roots differ
        # may resolve differently on the remote inspection host than on the
        # submission host.  Returning the remote realpath would then make an
        # otherwise valid explicit container mount impossible to match.
        annotation_path = Path(annotation_id["expanded"]).absolute()
        root_id = media_ids[0 if len(media_ids) == 1 else annotation_index]
        root = Path(root_id["expanded"]).absolute()
        records, metadata = load_annotation(annotation_path)
        observed_family = _detect_dataset_family(records, metadata)
        observed_families.add(observed_family)
        if dataset_family != "auto" and observed_family != dataset_family:
            schema_errors.append(
                f"{annotation_path}: detected {observed_family}, requested {dataset_family}"
            )
        manifest_entries.append(
            {
                "original": annotation_id["original"],
                "resolved": annotation_id["resolved"],
                "sha256": sha256_file(annotation_path),
                "count": len(records),
                "metadata": metadata,
            }
        )
        for index, record in enumerate(records):
            task = _record_task(record, metadata)
            active_family = observed_family if dataset_family == "auto" else dataset_family
            if active_family == "video_conversation":
                conversations = record.get("conversations") or record.get("messages")
                if not isinstance(conversations, list) or len(conversations) < 2:
                    schema_errors.append(f"{annotation_path}:{index}: conversation record needs >=2 turns")
                if not _record_media(record):
                    schema_errors.append(f"{annotation_path}:{index}: conversation record has no media field")
                if requested_tasks:
                    schema_errors.append("task selection is only valid for task-aware datasets")
            elif active_family == "task_aware_video_reasoning":
                if not task:
                    schema_errors.append(f"{annotation_path}:{index}: task-aware record has no task")
                if requested_tasks and task not in requested_tasks:
                    continue
                canonical_media = [
                    field for field in TAO_VL_MEDIA_FIELDS
                    if isinstance(record.get(field), str) and record.get(field)
                ]
                if len(canonical_media) != 1:
                    aliases = [field for field in MEDIA_FIELDS if record.get(field)]
                    schema_errors.append(
                        f"{annotation_path}:{index}: task-aware tao-vl-reason-v1.0 "
                        "record requires exactly one canonical video_id or image_id; "
                        f"found {aliases or 'none'}"
                    )
                conversations = record.get("conversations") or record.get("messages")
                has_conversation_target = isinstance(conversations, list) and len(conversations) >= 2
                has_question_answer = (
                    isinstance(record.get("question"), str)
                    and bool(record.get("question", "").strip())
                    and isinstance(record.get("answer"), str)
                    and bool(record.get("answer", "").strip())
                )
                if not has_conversation_target and not has_question_answer:
                    schema_errors.append(
                        f"{annotation_path}:{index}: task-aware record needs either >=2 conversation turns or non-empty question/answer"
                    )
            else:
                raise WorkflowError(f"unsupported dataset family: {active_family}")
            width, height, fps, duration = _numeric_metadata(record, metadata)
            if width is not None:
                widths.append(width)
            if height is not None:
                heights.append(height)
            if fps is not None:
                frame_rates.append(fps)
            if duration is not None:
                durations.append(duration)
            metric = record.get("metric") or metadata.get("metric")
            if task and isinstance(metric, str):
                task_metrics[task] = metric.casefold().replace("-", "_")
            record_keys.append(_record_key(record))
            task_key = task or "default"
            task_counts[task_key] = task_counts.get(task_key, 0) + 1
            target = _conversation_target(record)
            if target is None:
                raw_target = record.get("answer", record.get("references"))
                if isinstance(raw_target, str) and raw_target.strip():
                    target = raw_target.strip()
                elif isinstance(raw_target, list):
                    values = [str(value).strip() for value in raw_target if str(value).strip()]
                    if len(values) == 1:
                        target = values[0]
            if target is not None:
                answer_targets.setdefault(task_key, []).append(target)
            for relative in _record_media(record):
                candidate = Path(relative)
                media_path = candidate if candidate.is_absolute() else root / candidate
                runtime_path = str(media_path.absolute())
                if not media_path.is_file():
                    missing.append({"annotation": str(annotation_path), "record": index, "media": relative})
                    continue
                if runtime_path not in media_entries:
                    entry = {"path": runtime_path, "size": media_path.stat().st_size}
                    if verify_media_content:
                        entry["sha256"] = sha256_file(media_path)
                    media_entries[runtime_path] = entry
    if schema_errors:
        raise WorkflowError("dataset schema validation failed: " + "; ".join(schema_errors[:20]))
    if len(observed_families) != 1:
        raise WorkflowError(f"annotation files mix incompatible dataset families: {sorted(observed_families)}")
    resolved_family = next(iter(observed_families))
    if missing:
        raise WorkflowError("referenced media is missing: " + json.dumps(missing[:20], sort_keys=True))
    if not record_keys:
        raise WorkflowError("task selection produced zero records")
    duplicate_count = len(record_keys) - len(set(record_keys))
    if duplicate_count:
        raise WorkflowError(f"dataset contains {duplicate_count} duplicate logical records")
    media_manifest = sorted(media_entries.values(), key=lambda item: item["path"])
    media_sizes = [entry["size"] for entry in media_manifest]
    inferred_metrics: dict[str, str] = {}
    for task, values in answer_targets.items():
        normalized = {value.casefold() for value in values}
        if len(values) == task_counts[task] and any(
            normalized <= labels for labels in CLASSIFICATION_LABEL_SETS
        ):
            task_metrics.setdefault(task, "accuracy")
            inferred_metrics[task] = (
                "all conversation targets are deterministic classification labels"
            )
    accuracy_tasks = sorted(
        task for task in task_counts
        if task in ACCURACY_TASKS or task_metrics.get(task) in {"accuracy", "exact_match_accuracy"}
    )
    task_semantics: dict[str, dict[str, Any]] = {}
    for task in sorted(task_counts):
        labels = {
            value.casefold().strip()
            for value in answer_targets.get(task, [])
            if value.strip()
        }
        task_type = EVALUATOR_TASK_ALIASES.get(task)
        source = "task_metadata" if task_type else None
        if task_type is None:
            if labels and labels <= {"yes", "no"}:
                task_type, source = "binary", "complete_label_vocabulary"
            elif labels and labels <= {"a", "b"}:
                task_type, source = None, "ambiguous_a_b_label_vocabulary"
            elif labels and labels <= {"a", "b", "c", "d"}:
                task_type, source = "mcq", "complete_label_vocabulary"
            elif task in accuracy_tasks:
                task_type, source = None, "accuracy_declared_but_answer_semantics_unknown"
            else:
                task_type, source = "text", "non_accuracy_task"
        task_semantics[task] = {
            "task_type": task_type,
            "source": source,
            "target_count": len(answer_targets.get(task, [])),
            "complete_target_coverage": len(answer_targets.get(task, [])) == task_counts[task],
            "classification_vocabulary": (
                sorted(labels)
                if labels <= {"yes", "no", "a", "b", "c", "d"}
                else []
            ),
            "declared_metric": task_metrics.get(task),
        }
    known_task_types = {
        value["task_type"] for value in task_semantics.values() if value["task_type"]
    }
    unresolved_accuracy_tasks = sorted(
        task for task, value in task_semantics.items()
        if task in accuracy_tasks and value["task_type"] is None
    )
    inferred_task_type = next(iter(known_task_types)) if len(known_task_types) == 1 else ""
    metric_names = sorted({
        metric for metric in task_metrics.values()
        if metric not in {"accuracy", "exact_match_accuracy"}
    })
    evaluation_profile = {
        "inferred_task_type": inferred_task_type,
        "answer_type": "letter" if known_task_types == {"mcq"} else "freeform",
        "task_semantics": task_semantics,
        "metric_names": metric_names,
        "normalization": "tao-cosmos-shared-v2",
        "requires_user_input": [
            *(["task.type"] if unresolved_accuracy_tasks else []),
            *(["metrics.names"] if set(task_counts) - set(accuracy_tasks) and not metric_names else []),
        ],
        "unresolved_accuracy_tasks": unresolved_accuracy_tasks,
    }
    profile = {
        "family": resolved_family,
        "record_count": len(record_keys),
        "quantity_class": "small" if len(record_keys) < 10_000 else "medium" if len(record_keys) < 100_000 else "large",
        "unique_media_count": len(media_manifest),
        "records_per_media": len(record_keys) / max(len(media_manifest), 1),
        "media_reuse_class": "reused" if len(record_keys) > len(media_manifest) else "mostly_unique",
        "media_extensions": sorted({Path(item["path"]).suffix.casefold() for item in media_manifest}),
        "media_bytes": {
            "total": sum(media_sizes),
            "min": min(media_sizes),
            "median": statistics.median(media_sizes),
            "max": max(media_sizes),
        },
        "resolution": {
            "sample_count": min(len(widths), len(heights)),
            "median_width": statistics.median(widths) if widths else None,
            "median_height": statistics.median(heights) if heights else None,
            "max_width": max(widths) if widths else None,
            "max_height": max(heights) if heights else None,
            "class": (
                "unknown" if not widths or not heights
                else "up_to_720p" if statistics.median(widths) * statistics.median(heights) <= 1280 * 720
                else "up_to_1080p" if statistics.median(widths) * statistics.median(heights) <= 1920 * 1080
                else "above_1080p"
            ),
        },
        "video": {
            "fps_sample_count": len(frame_rates),
            "median_fps": statistics.median(frame_rates) if frame_rates else None,
            "duration_sample_count": len(durations),
            "median_duration_seconds": statistics.median(durations) if durations else None,
        },
        "annotation_metadata": [entry["metadata"] for entry in manifest_entries],
    }
    return {
        "dataset_family": resolved_family,
        "profile": profile,
        "annotations": annotation_ids,
        "media_roots": media_ids,
        "annotation_manifest": manifest_entries,
        "record_count": len(record_keys),
        "record_keys_sha256": stable_hash(sorted(record_keys)),
        "record_key_set": sorted(record_keys),
        "duplicate_records": duplicate_count,
        "missing_media": 0,
        "media_count": len(media_manifest),
        "media_manifest": media_manifest,
        "media_fingerprint": stable_hash(media_manifest),
        "dataset_fingerprint": stable_hash(
            {"records": sorted(record_keys), "media": media_manifest, "tasks": task_counts}
        ),
        "tasks": task_counts,
        "metric_coverage": {
            "accuracy_tasks": accuracy_tasks,
            "excluded_tasks": sorted(set(task_counts) - set(accuracy_tasks)),
            "task_metrics": task_metrics,
            "inferred_metrics": inferred_metrics,
            "aggregate": "example_weighted_over_accuracy_defined_tasks",
        },
        "evaluation_profile": evaluation_profile,
    }


def assert_no_overlap(train: Mapping[str, Any], validation: Mapping[str, Any]) -> None:
    overlap = set(train["record_key_set"]) & set(validation["record_key_set"])
    if overlap:
        raise WorkflowError(f"train/validation overlap contains {len(overlap)} logical records")


def model_parity(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    same = left.get("fingerprint") == right.get("fingerprint")
    return {"status": "equivalent" if same else "invalid_mismatch", "left": left.get("fingerprint"), "right": right.get("fingerprint")}


def dataset_parity(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    same = left.get("dataset_fingerprint") == right.get("dataset_fingerprint")
    return {"status": "equivalent" if same else "invalid_mismatch", "left": left.get("dataset_fingerprint"), "right": right.get("dataset_fingerprint")}


def optimization_parity(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "training_mode", "epochs", "effective_global_batch", "optimizer", "learning_rate",
        "scheduler", "warmup", "weight_decay", "gradient_clip", "precision", "seed",
        "sequence_length", "frames", "vision", "system_prompt", "validation_frequency_epochs",
        "checkpoint_frequency_epochs", "lora",
    )
    differences = {key: {"left": left.get(key), "right": right.get(key)} for key in keys if left.get(key) != right.get(key)}
    return {"status": "equivalent" if not differences else "invalid_mismatch", "differences": differences}


def validate_provenance(
    provenance: Mapping[str, Any], expected_commits: Mapping[str, str],
    expected_trees: Mapping[str, str] | None = None,
) -> None:
    if not provenance:
        raise WorkflowError("image provenance is missing")
    repositories = provenance.get("repositories")
    if not isinstance(repositories, Mapping):
        raise WorkflowError("image provenance has no repository manifest")
    for name, commit in expected_commits.items():
        actual = repositories.get(name, {})
        actual_commit = actual.get("commit") if isinstance(actual, Mapping) else None
        if actual_commit != commit:
            raise WorkflowError(f"image source mismatch for {name}: expected {commit}, found {actual_commit}")
        if expected_trees is not None and actual.get("tree") != expected_trees.get(name):
            raise WorkflowError(
                f"image tree mismatch for {name}: expected {expected_trees.get(name)}, found {actual.get('tree')}"
            )
        if actual.get("dirty"):
            raise WorkflowError(f"image provenance reports dirty source for {name}")


def validate_metadata(metadata: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "experiment_id", "dataset", "training_mode", "backend", "tao_job_id",
        "slurm", "image", "repositories", "config", "paths", "dataset_fingerprints", "model",
        "launch_command", "stdout", "stderr", "results_dir", "checkpoint_dir", "timestamps",
        "scheduler", "child_process", "terminal_tao_status", "metrics", "artifacts",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise WorkflowError(f"SLURM metadata is incomplete; missing: {missing}")
    slurm_required = {
        "job_id", "submission_host", "cluster", "partition", "account", "qos", "reservation",
        "requested_resources", "allocated_resources", "node_list", "master_address", "master_port",
        "requeue", "exclusive", "time_limit", "timeout",
    }
    slurm = metadata.get("slurm")
    if not isinstance(slurm, Mapping) or slurm_required - set(slurm):
        raise WorkflowError(f"SLURM metadata is incomplete; missing slurm fields: {sorted(slurm_required - set(slurm or {}))}")
    if metadata.get("child_process", {}).get("exit_code") not in {None, 0} and metadata.get("terminal_tao_status") == "SUCCESS":
        raise WorkflowError("nonzero child-process exit code cannot have terminal TAO SUCCESS")
    if metadata.get("slurm", {}).get("requeue") and metadata.get("child_process", {}).get("exit_code") not in {None, 0}:
        raise WorkflowError("requeue cannot hide a child-process failure")
    if metadata.get("scheduler", {}).get("state") == "COMPLETED" and metadata.get("child_process", {}).get("exit_code") is None:
        raise WorkflowError("scheduler COMPLETED is invalid without a captured child-process exit code")
    if metadata.get("terminal_tao_status") == "SUCCESS":
        if metadata.get("scheduler", {}).get("state") != "COMPLETED" or metadata.get("child_process", {}).get("exit_code") != 0:
            raise WorkflowError("TAO SUCCESS requires scheduler COMPLETED and child-process exit code zero")


def selected_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Return only reproducibility settings; credentials are never persisted."""
    allow = {
        "PYTHONUNBUFFERED", "PYTHONHASHSEED", "NCCL_DEBUG", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "PYTORCH_CUDA_ALLOC_CONF", "NVIDIA_DRIVER_CAPABILITIES", "CUDA_FORWARD_COMPAT",
        "FORCE_QWENVL_VIDEO_READER", "TAO_SFT_BATCH_THREADS",
        "TAO_PYNV_FRAME_TRANSFER", "TAO_PYNV_VIDEO_CACHE_SIZE",
        "TAO_PYNV_DECODER_CACHE_SIZE", "COSMOS_CACHE",
        "TAO_VIDEO_CACHE_SIZE", "TAO_FRAMEWORK_SFT_PROCESS_THREADS",
        "TAO_VIDEO_DECODER_DEVICE", "TAO_VIDEO_DECODER_THREADS",
    }
    return {key: environment[key] for key in sorted(allow & set(environment))}


def planned_path_identity(value: str) -> dict[str, Any]:
    """Describe a requested output path without creating it.

    Launch preflight runs before result/checkpoint/cache directories are created.
    Preserve the requested value and prove that the closest existing parent is
    writable instead of mutating the target merely to test it.
    """
    identity = path_identity(value)
    if identity["exists"]:
        identity.update(
            {
                "nearest_existing_parent": identity["resolved"],
                "parent_writable": os.access(identity["resolved"], os.W_OK | os.X_OK),
            }
        )
        return identity

    expanded = Path(identity["expanded"])
    parent = expanded.parent
    while parent != parent.parent and not parent.exists():
        parent = parent.parent
    parent_exists = parent.exists()
    identity.update(
        {
            "nearest_existing_parent": str(parent.resolve()) if parent_exists else None,
            "parent_writable": bool(parent_exists and os.access(parent, os.W_OK | os.X_OK)),
        }
    )
    return identity


def _split_labeled_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise WorkflowError("runtime paths must use LABEL=PATH syntax")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise WorkflowError("runtime paths must include a non-empty label and path")
    return label.strip(), path.strip()


def _inspect_inputs_cli(argv: Sequence[str]) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog="cosmos_common.py inspect-inputs",
        description="Inspect Cosmos model and video-dataset inputs from their compute frame.",
    )
    parser.add_argument("--base-model-path-or-uri", required=True)
    parser.add_argument("--base-model-revision", default="")
    parser.add_argument("--prepared-checkpoint-path", default="")
    parser.add_argument(
        "--dataset-family",
        choices=sorted(DATASET_FAMILIES),
        default="auto",
    )
    parser.add_argument("--train-annotation", action="append", default=[])
    parser.add_argument("--train-media-root", action="append", default=[])
    parser.add_argument("--validation-annotation", action="append", default=[])
    parser.add_argument("--validation-media-root", action="append", default=[])
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--runtime-path", action="append", default=[])
    parser.add_argument("--fast-media-fingerprint", action="store_true")
    parser.add_argument("--fast-model-fingerprint", action="store_true")
    args = parser.parse_args(list(argv))

    model = inspect_model(
        args.base_model_path_or_uri,
        args.base_model_revision,
        args.prepared_checkpoint_path,
        fast_weight_fingerprint=args.fast_model_fingerprint,
    )
    train = inspect_dataset(
        dataset_family=args.dataset_family,
        annotations=args.train_annotation,
        media_roots=args.train_media_root,
        selected_tasks=args.task,
        verify_media_content=not args.fast_media_fingerprint,
    )
    validation = inspect_dataset(
        dataset_family=args.dataset_family,
        annotations=args.validation_annotation,
        media_roots=args.validation_media_root,
        selected_tasks=args.task,
        verify_media_content=not args.fast_media_fingerprint,
    )
    if train["dataset_family"] != validation["dataset_family"]:
        raise WorkflowError("training and validation annotations resolve to different dataset families")
    assert_no_overlap(train, validation)
    runtime_paths = {
        label: planned_path_identity(path)
        for label, path in map(_split_labeled_path, args.runtime_path)
    }
    return {
        "schema_version": 1,
        "frame": "target_compute",
        "model": model,
        "datasets": {"train": train, "validation": validation},
        "runtime_paths": runtime_paths,
    }


def materialize_dataset(
    *,
    dataset_family: str,
    annotations: Sequence[str],
    output_path: str,
    selected_tasks: Sequence[str] = (),
    sample_limit: int = 0,
) -> dict[str, Any]:
    """Merge/filter annotations into one deterministic, atomic runtime manifest."""
    if not annotations:
        raise WorkflowError("at least one annotation is required for materialization")
    if sample_limit < 0:
        raise WorkflowError("sample_limit must be nonnegative")
    normalized_tasks = {
        item.casefold().replace("-", "_").replace(" ", "_")
        for item in selected_tasks
    }
    output = Path(output_path).expanduser()
    resolved_inputs = {Path(item).expanduser().resolve() for item in annotations}
    if output.resolve() in resolved_inputs:
        raise WorkflowError("materialized output must not overwrite a source annotation")

    records: list[dict[str, Any]] = []
    observed_families: set[str] = set()
    for value in annotations:
        annotation = Path(value).expanduser()
        if not annotation.is_file():
            raise WorkflowError(f"annotation path is inaccessible: {value}")
        items, metadata = load_annotation(annotation)
        observed = _detect_dataset_family(items, metadata)
        observed_families.add(observed)
        if dataset_family != "auto" and observed != dataset_family:
            raise WorkflowError(
                f"{annotation}: detected {observed}, requested {dataset_family}"
            )
        for item in items:
            task = _record_task(item, metadata)
            if normalized_tasks and task not in normalized_tasks:
                continue
            copied = dict(item)
            if observed == "task_aware_video_reasoning" and task and not copied.get("task"):
                copied["task"] = task
            records.append(copied)
    if len(observed_families) != 1:
        raise WorkflowError(
            f"annotation files mix incompatible dataset families: {sorted(observed_families)}"
        )
    if not records:
        raise WorkflowError("materialization selected zero records")
    if sample_limit:
        records = records[:sample_limit]

    resolved_family = next(iter(observed_families))
    payload: Any
    if resolved_family == "task_aware_video_reasoning":
        payload = {
            "format": "tao-vl-reason-v1.0",
            "metadata": {"task": "mixed"},
            "items": records,
        }
    else:
        payload = records
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not os.access(output.parent, os.W_OK | os.X_OK):
        raise WorkflowError(f"materialization output parent is not writable: {output.parent}")
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{output.name}.", suffix=".tmp",
            dir=output.parent, delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()
    return {
        "schema_version": 1,
        "dataset_family": resolved_family,
        "sample_limit": sample_limit,
        "selected_tasks": sorted(normalized_tasks),
        "record_count": len(records),
        "output": path_identity(str(output)),
        "sha256": sha256_file(output),
    }


def _materialize_dataset_cli(argv: Sequence[str]) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog="cosmos_common.py materialize-dataset",
        description="Create a deterministic merged/smoke manifest in the target compute frame.",
    )
    parser.add_argument("--dataset-family", choices=sorted(DATASET_FAMILIES), default="auto")
    parser.add_argument("--annotation", action="append", default=[])
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--sample-limit", type=int, default=0)
    args = parser.parse_args(list(argv))
    return materialize_dataset(
        dataset_family=args.dataset_family,
        annotations=args.annotation,
        output_path=args.output_path,
        selected_tasks=args.task,
        sample_limit=args.sample_limit,
    )


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] not in {"inspect-inputs", "materialize-dataset"}:
        print(
            "usage: cosmos_common.py {inspect-inputs,materialize-dataset} [options]",
            file=sys.stderr,
        )
        return 2
    try:
        payload = (
            _inspect_inputs_cli(values[1:])
            if values[0] == "inspect-inputs"
            else _materialize_dataset_cli(values[1:])
        )
    except WorkflowError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
