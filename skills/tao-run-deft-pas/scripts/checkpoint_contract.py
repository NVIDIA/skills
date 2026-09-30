# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate canonical PAS best-checkpoint publication evidence."""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
from typing import Any


BEST_RELPATH = pathlib.Path("best/clip_best_val_t2i_mAP.pth")
METADATA_RELPATH = pathlib.Path("best/clip_best_val_t2i_mAP.json")


def checkpoint_lineage_started_ns(
    status: dict[str, Any], run_started_at: str
) -> int:
    """Resolve a retry lineage lower bound anchored to the immutable run.

    New statuses carry ``lineage_started_ns`` explicitly. Legacy attempt-1
    statuses use their own start, while legacy retries—whose first-attempt
    timestamp was never persisted—fall back to the run start rather than the
    retry start that caused NVBug 6603466. The run anchor also prevents a
    status copied from an earlier occupant of the results path from widening
    the checkpoint window before this run existed.
    """
    started_ns = status.get("started_ns")
    attempt = status.get("attempt", 1)
    if (
        not isinstance(started_ns, int)
        or isinstance(started_ns, bool)
        or started_ns < 1
    ):
        raise ValueError("train command status started_ns must be a positive integer")
    if (
        not isinstance(attempt, int)
        or isinstance(attempt, bool)
        or not 1 <= attempt <= 2
    ):
        raise ValueError("train command status attempt must be 1 or 2")
    if not isinstance(run_started_at, str) or not run_started_at.strip():
        raise ValueError("run started_at must be a non-empty timestamp")
    try:
        run_started = dt.datetime.fromisoformat(run_started_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("run started_at must be an ISO-8601 timestamp") from exc
    if run_started.tzinfo is None:
        raise ValueError("run started_at must include a timezone")
    run_started_ns = int(run_started.timestamp() * 1_000_000_000)
    if run_started_ns < 1 or run_started_ns > started_ns:
        raise ValueError("run started_at must be no later than the train attempt")

    if "lineage_started_ns" in status:
        lineage_started_ns = status["lineage_started_ns"]
    elif attempt == 1:
        lineage_started_ns = started_ns
    else:
        lineage_started_ns = run_started_ns
    if (
        not isinstance(lineage_started_ns, int)
        or isinstance(lineage_started_ns, bool)
        or not run_started_ns <= lineage_started_ns <= started_ns
    ):
        raise ValueError(
            "train command status lineage_started_ns must fall between the "
            "immutable run start and current attempt start"
        )
    return lineage_started_ns


def _absolute_lexical(path: pathlib.Path | str, name: str) -> pathlib.Path:
    raw = pathlib.Path(path).expanduser()
    if not raw.is_absolute():
        raise ValueError(f"{name} must be absolute: {raw}")
    absolute = pathlib.Path(os.path.abspath(raw))
    if absolute != raw:
        raise ValueError(f"{name} must be a normalized absolute path: {raw}")
    return absolute


def _raw_checkpoint(path: pathlib.Path, train_root: pathlib.Path, name: str) -> pathlib.Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{name} must be an existing non-empty file: {path}")
    if path.is_symlink() or path.resolve() != path:
        raise ValueError(f"{name} must not be or traverse a symlink: {path}")
    try:
        relative = path.relative_to(train_root)
    except ValueError as exc:
        raise ValueError(f"{name} must be strictly within {train_root}: {path}") from exc
    if not relative.parts or relative.parts[0] == "best":
        raise ValueError(f"{name} must be a raw checkpoint outside best/: {path}")
    if path.suffix.lower() not in {".pth", ".ckpt", ".safetensors"}:
        raise ValueError(f"{name} has an unsupported checkpoint suffix: {path}")
    return path


def validate_best_checkpoint(
    best_path: pathlib.Path | str,
    train_root: pathlib.Path | str,
    *,
    started_ns: int,
) -> dict[str, Any]:
    """Validate the canonical publication and return normalized provenance.

    The PAS runtime normally creates a relative symlink at the canonical best
    path, with hardlink/copy fallbacks. Only that one lexical symlink is
    allowed; its target must be a direct, regular raw checkpoint in the same
    iteration's train directory and must belong to the current Docker attempt.
    """
    if not isinstance(started_ns, int) or isinstance(started_ns, bool) or started_ns < 1:
        raise ValueError("train started_ns must be a positive integer")
    train = _absolute_lexical(train_root, "train root")
    if not train.is_dir() or train.resolve() != train:
        raise ValueError(f"train root must be an existing non-symlink directory: {train}")
    best = _absolute_lexical(best_path, "best checkpoint")
    expected = train / BEST_RELPATH
    if best != expected:
        raise ValueError(f"best checkpoint must be {expected}, got {best}")
    if best.parent.resolve() != best.parent:
        raise ValueError(f"best checkpoint parent must not traverse a symlink: {best.parent}")
    if not os.path.lexists(best) or not best.is_file() or best.stat().st_size == 0:
        raise ValueError(f"best checkpoint is missing, empty, or dangling: {best}")

    metadata_path = train / METADATA_RELPATH
    if (
        not metadata_path.is_file()
        or metadata_path.stat().st_size == 0
        or metadata_path.is_symlink()
        or metadata_path.resolve() != metadata_path
    ):
        raise ValueError(f"best checkpoint metadata must be a non-symlink file: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"best checkpoint metadata is invalid JSON: {metadata_path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"best checkpoint metadata root must be an object: {metadata_path}")

    source_value = metadata.get("selected_checkpoint")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("best checkpoint metadata.selected_checkpoint is required")
    source = _raw_checkpoint(
        _absolute_lexical(source_value, "selected checkpoint"),
        train,
        "selected checkpoint",
    )
    if source.stat().st_mtime_ns < started_ns:
        raise ValueError(
            f"selected checkpoint predates the train attempt lineage: {source}"
        )
    if metadata.get("published_checkpoint") != str(best):
        raise ValueError(
            f"best checkpoint metadata.published_checkpoint must be {best}"
        )
    if metadata.get("metric_name") != "val/t2i_mAP":
        raise ValueError("best checkpoint metadata.metric_name must be 'val/t2i_mAP'")
    selection_strategy = metadata.get("selection_strategy")
    if selection_strategy not in {"metric", "newest_fallback"}:
        raise ValueError("best checkpoint metadata.selection_strategy is invalid")
    if selection_strategy == "metric" and not isinstance(metadata.get("metric"), dict):
        raise ValueError("metric checkpoint selection requires metric evidence")
    if selection_strategy == "newest_fallback" and metadata.get("metric") is not None:
        raise ValueError("newest fallback selection must record metric=null")
    mode = metadata.get("publish_mode")
    if mode not in {"symlink", "hardlink", "copy"}:
        raise ValueError("best checkpoint metadata.publish_mode is invalid")

    if best.is_symlink():
        if mode != "symlink":
            raise ValueError("symlink checkpoint must record publish_mode='symlink'")
        raw_target = pathlib.Path(os.readlink(best))
        lexical_target = (
            raw_target
            if raw_target.is_absolute()
            else pathlib.Path(os.path.abspath(best.parent / raw_target))
        )
        if lexical_target != source:
            raise ValueError(
                f"best checkpoint symlink must directly name selected checkpoint {source}"
            )
        if lexical_target.is_symlink() or lexical_target.resolve() != lexical_target:
            raise ValueError("best checkpoint symlink target must not be a symlink chain")
        if best.resolve() != source:
            raise ValueError("best checkpoint symlink resolves to the wrong source")
    else:
        if best.resolve() != best:
            raise ValueError(f"best checkpoint must not traverse a symlink: {best}")
        if mode == "symlink":
            raise ValueError("regular checkpoint cannot record publish_mode='symlink'")
        if mode == "hardlink" and best.stat().st_ino != source.stat().st_ino:
            raise ValueError("hardlink checkpoint does not share the selected source inode")
        if best.stat().st_size != source.stat().st_size:
            raise ValueError("published checkpoint size differs from selected source")

    return {
        "best_ckpt_path": str(best),
        "best_ckpt_metadata": str(metadata_path),
        "best_ckpt_source": str(source),
        "publish_mode": mode,
        "checkpoint_selection_strategy": selection_strategy,
    }
