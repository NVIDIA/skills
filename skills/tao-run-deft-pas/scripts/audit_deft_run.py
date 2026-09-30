# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit PAS DEFT disk state and report the only safe resume/completion status.

This is deliberately read-only. It cross-checks ``deft_state.json``,
``loop_log.jsonl``, every artifact path recorded in iteration state, the
mined-list / eval-split leakage invariant, and ``mining_selection_history.json``
coherence. Agents run it on startup, after compaction, before each stage, and
before claiming that the loop or an iteration completed. ``commit_stage.py``
runs it after every commit and rolls back on INVALID.

Exit codes:
  0: structurally valid (IN_PROGRESS, FAILED, or COMPLETE)
  1: inconsistent/invalid, non-terminal with --require-terminal, or
     unsuccessful with --require-complete
  2: input file could not be loaded
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import sys
from typing import Any

import yaml

from checkpoint_contract import (
    checkpoint_lineage_started_ns,
    validate_best_checkpoint,
)
from command_contract import (
    command_sha256,
    expected_container_command,
    expected_hf_forwarding,
    expected_image_kind,
    validate_content_bound_outputs,
)
from deft_action_contract import (
    SUPPORTED_PLATFORMS,
    platform_evidence_error,
    remote_freshness_attested,
    validate_tao_virtualenv,
)
from metric_contract import (
    compare,
    pick_best,
    validate_contract,
)
from pas_deft.pas_artifacts import PAS_METRICS_AGGREGATE_FILENAME
from parse_pas_metrics import build_result


WORKFLOW = "tao-run-deft-pas"
SCHEMA_VERSION = "3"
VALID_ITERATION_STATUSES = {"pending", "in_progress", "complete", "failed"}
VALID_LOG_STATUSES = {"ok", "error", "skip"}
SKIPPABLE_STAGES = {"visualize"}
VALID_STAGES = {
    "dataset_setup",
    "pool_embed",
    "evaluate",
    "gap_analysis",
    "data_mining",
    "history_select",
    "visualize",
    "train",
    "loop_stop",
}
VALID_COMPLETED_STAGES = VALID_STAGES - {"loop_stop"}
LOOP_STOP_REASONS = ("kpi_met", "max_iterations", "hard_stop")
COMPLETION_REASONS = ("kpi_met", "max_iterations")
VERIFY_MARKER = "VERIFY: PASS"
CHECKSUM_MARKER = "CHECKSUM_VERIFY: PASS"
EVAL_SUCCESS_MARKER = "Evaluate finished successfully"
TRAIN_SUCCESS_MARKER = "Train finished successfully."
RUN_SPEC_NAMES = (
    "deft_config.yaml",
    "tao_spec.yaml",
    "text_embed_spec.yaml",
    "image_embed_spec.yaml",
    "mining_spec.yaml",
    "approval.json",
)
PINNED_PYT_IMAGE = "nvcr.io/nvidia/tao/tao-toolkit:7.2.0-pyt"  # versions-key: images.tao_toolkit.deft_pas_pyt
PINNED_DS_IMAGE = "nvcr.io/nvidia/tao/tao-toolkit:7.2.0-data-services"  # versions-key: images.tao_toolkit.deft_pas_data_services

# Artifact fields recorded by commit_stage._apply_success, grouped by the
# containment scope commit_stage enforced at commit time.
RESULTS_SCOPED_FILE_FIELDS = {
    "source_pool_parquet",
    "pool_embeddings_parquet",
    "gaps_parquet",
    "target_embeddings_parquet",
    "mined_parquet",
    "candidate_pairs",
    "mined_image_list",
    "mined_pairs",
    "mined_manifest",
    "cumulative_names",
    "tsne_plot",
    "mining_history",
    "caption_history",
    "pool_embed_command_status",
    "verify_log",
    "checksum_verify_log",
    "dataset_materialize_status",
    "gap_analysis_status",
}
PHASE_SCOPED_FILE_FIELDS = {
    "metrics_aggregate_csv",
    "eval_status_json",
    "best_ckpt_path",
    "best_ckpt_metadata",
    "best_ckpt_source",
    "iteration_summary",
    "target_embed_command_status",
    "knn_command_status",
    "eval_command_status",
    "train_command_status",
    "train_tao_status_json",
    "visualize_prepare_status",
    "visualize_finish_status",
    "pretrained_state",
    "train_config",
    "eval_config",
    "eval_config_status",
    "iteration_summary_status",
    "mining_postprocess_status",
    "history_select_status",
    "train_config_status",
    "publish_checkpoint_status",
}
UNSCOPED_FILE_FIELDS: set[str] = set()
RESULTS_SCOPED_DIR_FIELDS = {"pas_splits_dir", "samples_dir"}
PATH_FIELDS = (
    RESULTS_SCOPED_FILE_FIELDS
    | PHASE_SCOPED_FILE_FIELDS
    | UNSCOPED_FILE_FIELDS
    | RESULTS_SCOPED_DIR_FIELDS
)
MARKER_FIELDS = {
    "verify_log": VERIFY_MARKER,
    "checksum_verify_log": CHECKSUM_MARKER,
    "eval_status_json": EVAL_SUCCESS_MARKER,
    "train_tao_status_json": TRAIN_SUCCESS_MARKER,
}
FIELD_STAGE = {
    "pas_splits_dir": "dataset_setup",
    "source_pool_parquet": "dataset_setup",
    "verify_log": "dataset_setup",
    "checksum_verify_log": "dataset_setup",
    "dataset_materialize_status": "dataset_setup",
    "pool_embeddings_parquet": "pool_embed",
    "pool_embed_command_status": "pool_embed",
    "metrics_aggregate_csv": "evaluate",
    "eval_status_json": "evaluate",
    "metric_result": "evaluate",
    "eval_command_status": "evaluate",
    "iteration_summary": "evaluate",
    "eval_config": "evaluate",
    "eval_config_status": "evaluate",
    "iteration_summary_status": "evaluate",
    "gaps_parquet": "gap_analysis",
    "caption_history": "gap_analysis",
    "gap_analysis_status": "gap_analysis",
    "target_embeddings_parquet": "data_mining",
    "mined_parquet": "data_mining",
    "candidate_pairs": "data_mining",
    "target_embed_command_status": "data_mining",
    "knn_command_status": "data_mining",
    "mining_postprocess_status": "data_mining",
    "mined_image_list": "history_select",
    "mined_pairs": "history_select",
    "mined_manifest": "history_select",
    "cumulative_names": "history_select",
    "mining_history": "history_select",
    "history_select_status": "history_select",
    "samples_dir": "visualize",
    "tsne_plot": "visualize",
    "visualize_skipped": "visualize",
    "visualize_prepare_status": "visualize",
    "visualize_finish_status": "visualize",
    "best_ckpt_path": "train",
    "best_ckpt_metadata": "train",
    "best_ckpt_source": "train",
    "publish_mode": "train",
    "pretrained_state": "train",
    "train_config": "train",
    "train_command_status": "train",
    "train_tao_status_json": "train",
    "train_config_status": "train",
    "publish_checkpoint_status": "train",
}
# Proof fields required in state for a successful (status=ok) log event. A
# status=skip visualize event requires visualize_skipped=true instead.
STAGE_REQUIRED_FIELDS = {
    "dataset_setup": (
        "pas_splits_dir",
        "source_pool_parquet",
        "verify_log",
        "dataset_materialize_status",
    ),
    "pool_embed": ("pool_embeddings_parquet", "pool_embed_command_status"),
    "evaluate": (
        "metrics_aggregate_csv",
        "eval_status_json",
        "metric_result",
        "eval_command_status",
        "eval_config",
        "eval_config_status",
    ),
    "gap_analysis": ("gaps_parquet", "caption_history", "gap_analysis_status"),
    "data_mining": (
        "target_embeddings_parquet",
        "mined_parquet",
        "candidate_pairs",
        "target_embed_command_status",
        "knn_command_status",
        "mining_postprocess_status",
    ),
    "history_select": (
        "mined_image_list",
        "mined_pairs",
        "mined_manifest",
        "cumulative_names",
        "history_select_status",
    ),
    "visualize": (),
    "train": (
        "best_ckpt_path",
        "best_ckpt_metadata",
        "best_ckpt_source",
        "publish_mode",
        "pretrained_state",
        "train_config",
        "train_command_status",
        "train_tao_status_json",
        "train_config_status",
        "publish_checkpoint_status",
    ),
}
# "Read first" column of the Stage Reference Modules table in
# references/scripts-and-agents.md.
STAGE_REFERENCES = {
    "dataset_setup": ("references/data-layout.md",),
    "pool_embed": ("references/mining.md",),
    "evaluate": ("references/clip-train-eval.md", "references/metric-contract.md"),
    "gap_analysis": ("references/gap-analysis.md",),
    "data_mining": ("references/mining.md",),
    "history_select": ("references/mining.md",),
    "visualize": ("references/visualization.md",),
    "train": ("references/clip-train-eval.md",),
    "loop_stop": ("references/pipeline-and-state.md",),
}
PIPELINE_REFERENCE = ("references/pipeline-and-state.md",)
RECOVERY_REFERENCE = "references/scripts-and-agents.md"


def _skill_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def _render_references(names: tuple[str, ...]) -> str:
    root = _skill_root()
    return ", ".join(str(root / name) for name in names)


def _load_state(path: pathlib.Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("deft_state.json root must be an object")
    return data


def _load_log(path: pathlib.Path, errors: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"loop_log.jsonl:{lineno}: invalid JSON: {exc}")
            continue
        if not isinstance(entry, dict):
            errors.append(f"loop_log.jsonl:{lineno}: entry must be an object")
            continue
        entries.append(entry)
    return entries


def _iteration_sort_key(label: str) -> tuple[int, int]:
    if label == "baseline":
        return (0, 0)
    match = re.fullmatch(r"iter([1-9][0-9]*)", label)
    return (1, int(match.group(1))) if match else (2, 0)


def _phase_dir(results_dir: pathlib.Path, label: str) -> pathlib.Path | None:
    """Map an iteration label to its pas_deft results directory.

    baseline artifacts live under zs/ (zero-shot); iterN artifacts live under
    iter_<N>/ — note the underscore. Mirrors commit_stage._phase_dir.
    """
    if label == "baseline":
        return results_dir / "zs"
    match = re.fullmatch(r"iter([1-9][0-9]*)", label)
    return results_dir / f"iter_{match.group(1)}" if match else None


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _gate_from_state(
    state: dict[str, Any], errors: list[str]
) -> dict[str, Any] | None:
    """Return the one canonical metric contract or record why it is invalid."""
    contract = state.get("metric_contract")
    if contract is None:
        errors.append("state.metric_contract is required")
        return None
    try:
        normalized = validate_contract(contract)
    except ValueError as exc:
        errors.append(f"invalid state.metric_contract: {exc}")
        return None
    return {
        key: normalized[key]
        for key in ("metric_name", "query_type", "op", "target")
    }


def _gate_passes(gate: dict[str, Any], value: Any) -> bool:
    """A null-target gate never passes: the loop runs to max_iterations."""
    if gate.get("target") is None:
        return False
    return compare(float(value), gate["op"], float(gate["target"]))


def _render_target(gate: dict[str, Any]) -> str:
    if gate.get("target") is None:
        return "none (run to max_iterations)"
    return f"{gate['op']} {gate['target']:g}"


def _expected_next(
    entry: dict[str, Any], state: dict[str, Any]
) -> set[tuple[str, str]]:
    """Legal successor events, including KPI and iteration branch predicates."""
    label = str(entry.get("iteration"))
    stage = str(entry.get("stage"))
    if entry.get("status") == "error":
        return {(label, "loop_stop")}
    if stage == "loop_stop":
        return set()
    if label == "baseline":
        if stage == "dataset_setup":
            return {("baseline", "pool_embed")}
        if stage == "pool_embed":
            return {("baseline", "evaluate")}
        if stage == "evaluate":
            info = state.get("iterations", {}).get("baseline", {})
            result = info.get("metric_result") if isinstance(info, dict) else None
            return (
                {("baseline", "loop_stop")}
                if isinstance(result, dict) and result.get("passed") is True
                else {("baseline", "gap_analysis")}
            )
        if stage == "gap_analysis":
            return {("iter1", "data_mining")}
        return set()
    match = re.fullmatch(r"iter([1-9][0-9]*)", label)
    if not match:
        return set()
    number = int(match.group(1))
    if stage == "data_mining":
        return {(label, "history_select")}
    if stage == "history_select":
        return {(label, "visualize")}
    if stage == "visualize":
        return {(label, "train")}
    if stage == "train":
        return {(label, "evaluate")}
    if stage == "evaluate":
        info = state.get("iterations", {}).get(label, {})
        result = info.get("metric_result") if isinstance(info, dict) else None
        passed = isinstance(result, dict) and result.get("passed") is True
        raw_maximum = state.get("max_iterations")
        if not isinstance(raw_maximum, int) or isinstance(raw_maximum, bool):
            return set()
        maximum = raw_maximum
        return (
            {(label, "loop_stop")}
            if passed or number >= maximum
            else {(label, "gap_analysis")}
        )
    if stage == "gap_analysis":
        raw_maximum = state.get("max_iterations")
        if not isinstance(raw_maximum, int) or isinstance(raw_maximum, bool):
            return set()
        maximum = raw_maximum
        return (
            {(f"iter{number + 1}", "data_mining")}
            if number < maximum
            else set()
        )
    return set()


def _basenames(path: pathlib.Path, field: str, errors: list[str]) -> set[str]:
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        errors.append(f"{field} cannot be read: {exc}")
        return set()
    return {
        pathlib.Path(line.strip()).name
        for line in text.splitlines()
        if line.strip()
    }


def _expected_artifact_path(
    field: str,
    label: str,
    results_dir: pathlib.Path,
    config: dict[str, Any],
) -> pathlib.Path | None:
    phase = _phase_dir(results_dir, label)
    if phase is None:
        return None
    fixed = {
        "pas_splits_dir": results_dir / "pas_splits",
        "source_pool_parquet": results_dir / "embeddings" / "source" / "source_pool.parquet",
        "verify_log": results_dir / "dataset_setup" / "rebuild_verify.log",
        "checksum_verify_log": results_dir / "dataset_setup" / "checksum_verify.log",
        "dataset_materialize_status": results_dir / "dataset_setup" / "dataset-materialize.host.status.json",
        "pool_embeddings_parquet": results_dir / "embeddings" / "source" / "embeddings.parquet",
        "pool_embed_command_status": results_dir / "embeddings" / "source" / "pool_embed.status.json",
        "metrics_aggregate_csv": phase / "evaluate" / PAS_METRICS_AGGREGATE_FILENAME,
        "eval_status_json": phase / "evaluate" / "status.json",
        "eval_command_status": phase / "evaluate" / "evaluate.status.json",
        "iteration_summary": phase / "iteration_summary.json",
        "eval_config": phase / "specs" / "eval_config.yaml",
        "eval_config_status": phase / "specs" / "eval-config.host.status.json",
        "iteration_summary_status": phase / "iteration-summary.host.status.json",
        "target_embeddings_parquet": phase / "embeddings" / "target" / "embeddings.parquet",
        "target_embed_command_status": phase / "embeddings" / "target" / "target_embed.status.json",
        "mined_parquet": phase / "mining" / "mined_samples.parquet",
        "knn_command_status": phase / "mining" / "knn.status.json",
        "mined_image_list": phase / "mining" / "mined_image_list.txt",
        "mined_pairs": phase / "mining" / "mined_pairs.json",
        "mined_manifest": phase / "mining" / "mined_dataset.json",
        "cumulative_names": phase / "mining" / "cumulative_mined_unique_names.json",
        "mining_history": results_dir / "mining_selection_history.json",
        "caption_history": results_dir / "caption_selection_history.json",
        "samples_dir": phase / "visualization" / "samples",
        "tsne_plot": phase / "visualization" / "tsne_plot.png",
        "visualize_prepare_status": phase / "visualization" / "visualize-prepare.host.status.json",
        "visualize_finish_status": phase / "visualization" / "visualize-finish.host.status.json",
        "best_ckpt_path": phase / "train" / "best" / "clip_best_val_t2i_mAP.pth",
        "best_ckpt_metadata": phase / "train" / "best" / "clip_best_val_t2i_mAP.json",
        "train_tao_status_json": phase / "train" / "status.json",
        "train_command_status": phase / "train" / "train.status.json",
        "pretrained_state": phase / "pretrained" / "model_state.pth",
        "train_config": phase / "specs" / "train_config.yaml",
        "mining_postprocess_status": phase / "mining" / "mining-postprocess.host.status.json",
        "history_select_status": phase / "mining" / "history-select.host.status.json",
        "train_config_status": phase / "specs" / "train-config.host.status.json",
        "publish_checkpoint_status": phase / "train" / "publish-checkpoint.host.status.json",
    }
    if field == "gaps_parquet":
        feed = 1 if label == "baseline" else int(label[4:]) + 1
        return results_dir / f"iter_{feed}" / "gaps" / "kpi_gaps.parquet"
    if field == "gap_analysis_status":
        feed = 1 if label == "baseline" else int(label[4:]) + 1
        return results_dir / f"iter_{feed}" / "gaps" / "gap-analysis.host.status.json"
    if field == "candidate_pairs":
        candidate = phase / "mining"
        if config.get("history_aware"):
            candidate = candidate / "history_candidates"
        return candidate / "mined_pairs.json"
    return fixed.get(field)


def _validate_json_shape(
    path: pathlib.Path,
    field: str,
    errors: list[str],
    expected_type: type,
) -> Any | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{field} must contain valid JSON: {path}: {exc}")
        return None
    if not isinstance(payload, expected_type) or not payload:
        errors.append(
            f"{field} JSON must be a non-empty {expected_type.__name__}: {path}"
        )
        return None
    return payload


def _validate_parquet(
    path: pathlib.Path,
    field: str,
    errors: list[str],
    required_columns: set[str],
) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        errors.append(
            f"{field} validation requires pyarrow in the selected interpreter"
        )
        return
    try:
        parquet = pq.ParquetFile(path)
        rows = parquet.metadata.num_rows
        columns = set(parquet.schema_arrow.names)
    except Exception as exc:
        errors.append(f"{field} is not a readable parquet: {path}: {exc}")
        return
    if rows < 1:
        errors.append(f"{field} parquet has zero rows: {path}")
    missing = sorted(required_columns - columns)
    if missing:
        errors.append(f"{field} parquet is missing columns {missing}: {path}")


def _validate_png(path: pathlib.Path, field: str, errors: list[str]) -> None:
    try:
        with path.open("rb") as handle:
            header = handle.read(8)
    except OSError as exc:
        errors.append(f"{field} cannot be read: {path}: {exc}")
        return
    if header != b"\x89PNG\r\n\x1a\n":
        errors.append(f"{field} must be a PNG image: {path}")


def _validate_image_dir(path: pathlib.Path, field: str, errors: list[str]) -> None:
    try:
        images = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
    except OSError as exc:
        errors.append(f"{field} cannot be listed: {path}: {exc}")
        return
    links = [item for item in path.rglob("*") if item.is_symlink()]
    if links:
        errors.append(f"{field} must not contain symlinks: {links[0]}")
        return
    for image in images:
        try:
            with image.open("rb") as handle:
                header = handle.read(8)
        except OSError:
            continue
        if header == b"\x89PNG\r\n\x1a\n" or header[:3] == b"\xff\xd8\xff":
            return
    errors.append(
        f"{field} must contain at least one non-empty PNG or JPEG image: {path}"
    )


def _config_section(
    root: dict[str, Any], key: str, field: str, errors: list[str]
) -> dict[str, Any]:
    value = root.get(key)
    if not isinstance(value, dict):
        errors.append(f"{field}.{key} must be an object")
        return {}
    return value


def _python_tree_sha256(root: pathlib.Path) -> str:
    files = sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    if not files:
        raise ValueError(f"no Python files under {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _results_root_for_scope(scope: pathlib.Path) -> pathlib.Path:
    """Find the canonical run root while retaining phase-scoped status checks."""
    resolved = scope.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "deft_state.json").is_file():
            return candidate
    return resolved


def _validate_command_status(
    path: pathlib.Path,
    field: str,
    errors: list[str],
    scope: pathlib.Path,
    required_name: str | None = None,
    required_command: list[str] | None = None,
    required_image_kind: str | None = None,
    required_image: str | None = None,
    required_hf_forwarding: bool | None = None,
    required_platform: str | None = None,
) -> dict[str, Any] | None:
    payload = _validate_json_shape(path, field, errors, dict)
    if payload is None:
        return None
    if payload.get("schema_version") not in {"1", "2"}:
        errors.append(f"{field}.schema_version must be '1' or '2'")
    if not isinstance(payload.get("name"), str) or not payload.get("name", "").strip():
        errors.append(f"{field}.name must be a non-empty string")
    elif required_name is not None and payload.get("name") != required_name:
        errors.append(
            f"{field}.name must be {required_name!r}, got {payload.get('name')!r}"
        )
    if required_command is not None:
        if required_platform is None:
            errors.append(f"{field} lacks an initialized workflow platform contract")
            evidence_error = "expected platform was not supplied"
        else:
            evidence_error = platform_evidence_error(payload, required_platform)
        if evidence_error is not None:
            errors.append(f"{field} does not prove native action success: {evidence_error}")
        if payload.get("command") != required_command:
            errors.append(f"{field} does not record the approved command argv")
        if payload.get("command_sha256") != command_sha256(required_command):
            errors.append(f"{field}.command_sha256 does not match approved argv")
        if payload.get("image_kind") != required_image_kind:
            errors.append(
                f"{field}.image_kind must be {required_image_kind!r}, "
                f"got {payload.get('image_kind')!r}"
            )
        if payload.get("image") != required_image:
            errors.append(
                f"{field}.image must be {required_image!r}, "
                f"got {payload.get('image')!r}"
            )
        if payload.get("passed_hf_token") is not required_hf_forwarding:
            errors.append(
                f"{field}.passed_hf_token must be {required_hf_forwarding!r}"
            )
    attempt = payload.get("attempt")
    if (
        not isinstance(attempt, int)
        or isinstance(attempt, bool)
        or not 1 <= attempt <= 2
    ):
        errors.append(f"{field}.attempt must be an integer in [1, 2]")
    if (
        payload.get("workflow") != WORKFLOW
        or payload.get("status") != "ok"
        or payload.get("exit_code") != 0
    ):
        errors.append(f"{field} does not record a successful PAS DEFT command: {path}")
    started_ns = payload.get("started_ns")
    if not isinstance(started_ns, int) or isinstance(started_ns, bool) or started_ns < 1:
        errors.append(f"{field}.started_ns must be a positive integer")
    if not isinstance(payload.get("finished_at"), str) or not payload.get("finished_at", "").strip():
        errors.append(f"{field}.finished_at must be a non-empty timestamp string")
    fresh_outputs = payload.get("fresh_outputs")
    results_root = _results_root_for_scope(scope)
    if not isinstance(fresh_outputs, list) or not fresh_outputs:
        errors.append(f"{field}.fresh_outputs must be a non-empty list")
    else:
        for item in fresh_outputs:
            if not isinstance(item, str):
                errors.append(f"{field}.fresh_outputs entries must be absolute paths")
                continue
            raw_output = pathlib.Path(item).expanduser()
            absolute_output = pathlib.Path(os.path.abspath(raw_output))
            if not raw_output.is_absolute() or raw_output != absolute_output:
                errors.append(
                    f"{field}.fresh_outputs entry must be a normalized "
                    f"absolute path: {item}"
                )
                continue
            try:
                absolute_output.resolve().relative_to(results_root)
            except ValueError:
                errors.append(
                    f"{field}.fresh_outputs entry must resolve under "
                    f"{results_root}: {item}"
                )
    log_path = pathlib.Path(str(payload.get("log_path", ""))).expanduser()
    if (
        not log_path.is_absolute()
        or not log_path.is_file()
        or log_path.stat().st_size == 0
        or log_path.is_symlink()
        or log_path.resolve() != log_path
    ):
        errors.append(
            f"{field}.log_path does not exist, is empty, or is unsafe: {log_path}"
        )
    try:
        path.resolve().relative_to(scope.resolve())
        log_path.resolve().relative_to(scope.resolve())
    except ValueError:
        errors.append(f"{field} and its log must be under {scope}")
    return payload


def _validate_derived_specs_on_disk(
    results_dir: pathlib.Path, max_iterations: int, errors: list[str]
) -> None:
    candidates = [
        (
            results_dir / "zs" / "specs" / "eval_config.yaml",
            results_dir / "zs" / "specs" / "eval-config.host.status.json",
            "eval-config",
            results_dir / "zs",
        )
    ]
    for number in range(1, max_iterations + 1):
        phase = results_dir / f"iter_{number}"
        candidates.extend(
            [
                (
                    phase / "specs" / "train_config.yaml",
                    phase / "specs" / "train-config.host.status.json",
                    "train-config",
                    phase,
                ),
                (
                    phase / "specs" / "eval_config.yaml",
                    phase / "specs" / "eval-config.host.status.json",
                    "eval-config",
                    phase,
                ),
            ]
        )
    for spec_path, status_path, producer, scope in candidates:
        if not spec_path.exists() and not status_path.exists():
            continue
        if not status_path.is_file():
            errors.append(f"derived spec lacks producer evidence: {spec_path}")
            continue
        try:
            raw_payload = json.loads(status_path.read_text())
        except (OSError, json.JSONDecodeError):
            raw_payload = None
        # A failed/running producer is retry evidence, not a trusted spec. The
        # container gate rejects it; audit content-checks only successful output.
        if not isinstance(raw_payload, dict) or raw_payload.get("status") != "ok":
            continue
        payload = _validate_command_status(
            status_path,
            f"derived {producer} status",
            errors,
            scope,
            required_name=producer,
        )
        if payload is not None:
            try:
                validate_content_bound_outputs(
                    payload, [spec_path], f"derived {producer} status"
                )
            except (OSError, ValueError) as exc:
                errors.append(str(exc))


def _history_iteration_numbers(
    payload: dict[str, Any], errors: list[str]
) -> list[int]:
    iterations = payload.get("iterations")
    if isinstance(iterations, dict):
        raw_items = list(iterations.keys())
    elif isinstance(iterations, list):
        raw_items = [
            item.get("iteration") if isinstance(item, dict) else item
            for item in iterations
        ]
    else:
        errors.append(
            "mining_selection_history.json must contain an 'iterations' "
            "object or list"
        )
        return []
    numbers: list[int] = []
    for raw in raw_items:
        try:
            number = int(raw)
        except (TypeError, ValueError):
            errors.append(
                f"mining_selection_history.json has a non-integer iteration "
                f"entry: {raw!r}"
            )
            continue
        if number < 1:
            errors.append(
                f"mining_selection_history.json has an out-of-range iteration "
                f"entry: {number}"
            )
            continue
        numbers.append(number)
    return numbers


def _next_action(
    state: dict[str, Any],
    entries: list[dict[str, Any]],
    iterations: dict[str, Any],
    status: str,
    terminal: bool,
) -> tuple[str, tuple[str, ...] | None]:
    if status == "INVALID":
        return "repair disk-state inconsistencies before running another stage", None
    if status == "COMPLETE":
        return (
            "render the deterministic loop-end report, then present the "
            "best-iteration evidence",
            PIPELINE_REFERENCE,
        )
    if status == "FAILED":
        if terminal:
            return (
                "surface the logged hard stop; do not retry automatically",
                PIPELINE_REFERENCE,
            )
        label = str(entries[-1].get("iteration"))
        return f"{label}/loop_stop", STAGE_REFERENCES["loop_stop"]
    if not entries:
        return "baseline/dataset_setup", STAGE_REFERENCES["dataset_setup"]

    last = entries[-1]
    label = str(last.get("iteration"))
    stage = str(last.get("stage"))
    max_iterations = state.get("max_iterations")
    valid_max = isinstance(max_iterations, int) and not isinstance(
        max_iterations, bool
    )
    if stage == "dataset_setup":
        nxt = ("baseline", "pool_embed")
    elif stage == "pool_embed":
        nxt = ("baseline", "evaluate")
    elif stage == "evaluate":
        info = iterations.get(label, {})
        result = info.get("metric_result") if isinstance(info, dict) else None
        passed = isinstance(result, dict) and result.get("passed") is True
        match = re.fullmatch(r"iter([1-9][0-9]*)", label)
        reached_max = bool(
            match and valid_max and int(match.group(1)) >= max_iterations
        )
        nxt = (
            (label, "loop_stop")
            if passed or reached_max
            else (label, "gap_analysis")
        )
    elif stage == "gap_analysis":
        if label == "baseline":
            nxt = ("iter1", "data_mining")
        else:
            number = int(label[4:])
            if valid_max and number >= max_iterations:
                nxt = (label, "loop_stop")
            else:
                nxt = (f"iter{number + 1}", "data_mining")
    elif stage == "data_mining":
        nxt = (label, "history_select")
    elif stage == "history_select":
        nxt = (label, "visualize")
    elif stage == "visualize":
        nxt = (label, "train")
    elif stage == "train":
        nxt = (label, "evaluate")
    else:
        return (
            "inspect references/pipeline-and-state.md before continuing",
            PIPELINE_REFERENCE,
        )
    return f"{nxt[0]}/{nxt[1]}", STAGE_REFERENCES[nxt[1]]


def audit(results_dir: pathlib.Path, require_complete: bool = False) -> dict[str, Any]:
    results_dir = results_dir.expanduser().resolve()
    state_path = results_dir / "deft_state.json"
    log_path = results_dir / "loop_log.jsonl"
    state = _load_state(state_path)
    errors: list[str] = []
    warnings: list[str] = []
    entries = _load_log(log_path, errors)
    transaction_path = results_dir / ".deft_commit_transaction.json"
    if transaction_path.exists():
        errors.append(
            f"interrupted stage transaction found: {transaction_path}; run "
            f"{pathlib.Path(__file__).resolve().parent / 'recover_commit.py'} "
            f"--results-dir {results_dir} before any other mutation"
        )

    if state.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"state.schema_version must be {SCHEMA_VERSION!r}, "
            f"got {state.get('schema_version')!r}"
        )
    if state.get("workflow") != WORKFLOW:
        errors.append(
            f"state.workflow must be {WORKFLOW!r}, got {state.get('workflow')!r}"
        )
    started_at = state.get("started_at")
    if not isinstance(started_at, str) or not started_at.strip():
        errors.append("state.started_at must be a non-empty timestamp string")

    recorded_results = pathlib.Path(str(state.get("results_dir", ""))).expanduser()
    if not recorded_results.is_absolute():
        errors.append("state.results_dir must be an absolute path")
    elif recorded_results.resolve() != results_dir:
        errors.append(
            f"state.results_dir={recorded_results.resolve()} does not match "
            f"{results_dir}"
        )

    for name in ("max_iterations", "current_iteration"):
        value = state.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"state.{name} must be an integer")
    max_iterations = state.get("max_iterations")
    valid_max = isinstance(max_iterations, int) and not isinstance(
        max_iterations, bool
    )
    current_iteration = state.get("current_iteration")
    valid_current = isinstance(current_iteration, int) and not isinstance(
        current_iteration, bool
    )
    if valid_max and max_iterations < 1:
        errors.append("state.max_iterations must be >= 1")
    if valid_current and current_iteration < 0:
        errors.append("state.current_iteration must be >= 0")
    if valid_max and valid_current and current_iteration > max_iterations:
        errors.append("state.current_iteration must not exceed max_iterations")
    if valid_max and max_iterations >= 1:
        _validate_derived_specs_on_disk(results_dir, max_iterations, errors)

    gate = _gate_from_state(state, errors)
    gate_met = state.get("gate_met")
    if not isinstance(gate_met, bool):
        errors.append("state.gate_met must be a boolean")

    config = state.get("config")
    layout: dict[str, Any] = {}
    dataset_committed = any(
        entry.get("iteration") == "baseline"
        and entry.get("stage") == "dataset_setup"
        and entry.get("status") == "ok"
        for entry in entries
    )
    if not isinstance(config, dict):
        errors.append("state.config must be an object")
        config = {}
    else:
        for field in ("workspace",):
            value = config.get(field)
            if not value:
                errors.append(f"state.config.{field} is required")
                continue
            path = pathlib.Path(str(value)).expanduser()
            if not path.is_absolute():
                errors.append(f"state.config.{field} must be absolute: {value}")
            elif not path.is_dir():
                errors.append(
                    f"state.config.{field} is not an existing directory: {value}"
                )
        workspace_path = pathlib.Path(str(config.get("workspace", ""))).expanduser()
        pas_runtime_path = pathlib.Path(__file__).resolve().parent / "pas_deft"
        try:
            runtime_digest = _python_tree_sha256(pas_runtime_path)
        except (OSError, ValueError) as exc:
            errors.append(f"bundled PAS runtime cannot be hashed: {exc}")
        else:
            if config.get("pas_deft_bundle_sha256") != runtime_digest:
                errors.append("bundled PAS runtime changed after initialization")
        if workspace_path.is_absolute() and workspace_path.is_dir():
            workspace_path = workspace_path.resolve()
            if workspace_path == pathlib.Path(workspace_path.anchor):
                errors.append("state.config.workspace must not be a filesystem root")
            else:
                try:
                    relative_results = results_dir.resolve().relative_to(workspace_path)
                except ValueError:
                    errors.append("state.results_dir must be under state.config.workspace")
                else:
                    if relative_results == pathlib.Path("."):
                        errors.append("state.results_dir must not equal state.config.workspace")
        dataset_value = config.get("dataset_root")
        if not dataset_value:
            errors.append("state.config.dataset_root is required")
        else:
            dataset_path = pathlib.Path(str(dataset_value)).expanduser()
            if not dataset_path.is_absolute():
                errors.append(
                    f"state.config.dataset_root must be absolute: {dataset_value}"
                )
            elif dataset_path.exists() and not dataset_path.is_dir():
                errors.append(
                    f"state.config.dataset_root is not a directory: {dataset_value}"
                )
            elif dataset_committed:
                for name in ("images", "captions"):
                    if not (dataset_path / name).is_dir():
                        errors.append(
                            f"committed dataset root is missing {name}/: {dataset_path}"
                        )
                for name in ("train_pairs.json", "val_pairs.json"):
                    candidate = dataset_path / name
                    if not candidate.is_file() or candidate.stat().st_size == 0:
                        errors.append(
                            f"committed dataset root is missing non-empty {name}: "
                            f"{dataset_path}"
                        )
            if workspace_path.is_absolute() and workspace_path.is_dir():
                try:
                    relative_dataset = dataset_path.resolve().relative_to(
                        workspace_path.resolve()
                    )
                except ValueError:
                    errors.append(
                        "state.config.dataset_root must be under state.config.workspace"
                    )
                else:
                    if relative_dataset == pathlib.Path("."):
                        errors.append(
                            "state.config.dataset_root must not equal state.config.workspace"
                        )
                if (
                    results_dir.resolve() in dataset_path.resolve().parents
                    or dataset_path.resolve() in results_dir.resolve().parents
                ):
                    errors.append(
                        "state.results_dir and state.config.dataset_root must not "
                        "contain one another"
                    )
                if dataset_path.resolve().parent == workspace_path.resolve():
                    errors.append(
                        "state.config.dataset_root must be nested below a "
                        "workspace data directory"
                    )
        for field in ("images_archive", "metadata_archive"):
            value = config.get(field)
            path = pathlib.Path(str(value or "")).expanduser()
            if not value or not path.is_absolute() or not path.is_file() or path.stat().st_size == 0:
                errors.append(
                    f"state.config.{field} must be an existing absolute non-empty file: {value}"
                )
        checksum_value = config.get("checksums_file")
        if checksum_value is not None:
            checksum_path = pathlib.Path(str(checksum_value)).expanduser()
            if (
                not checksum_path.is_absolute()
                or not checksum_path.is_file()
                or checksum_path.stat().st_size == 0
            ):
                errors.append(
                    "state.config.checksums_file must be null or an existing "
                    f"absolute non-empty file: {checksum_value}"
                )
            else:
                checksum_digest = hashlib.sha256(checksum_path.read_bytes()).hexdigest()
                if config.get("checksums_file_sha256") != checksum_digest:
                    errors.append(
                        "state.config.checksums_file changed after initialization"
                    )
        elif config.get("checksums_file_sha256") is not None:
            errors.append(
                "state.config.checksums_file_sha256 must be null when no manifest is approved"
            )
        config_dir_value = config.get("config_dir")
        config_dir_path: pathlib.Path | None = None
        expected_config_dir = pathlib.Path(os.path.abspath(results_dir / "config"))
        if not config_dir_value or not pathlib.Path(str(config_dir_value)).is_absolute():
            errors.append("state.config.config_dir must be an absolute path")
        else:
            raw_config_dir = pathlib.Path(str(config_dir_value)).expanduser()
            config_dir_path = pathlib.Path(os.path.abspath(raw_config_dir))
            if config_dir_path != raw_config_dir:
                errors.append("state.config.config_dir must be a normalized absolute path")
            if config_dir_path.resolve() != config_dir_path:
                errors.append("state.config.config_dir must not traverse a symlink")
            if config_dir_path != expected_config_dir:
                errors.append(
                    f"state.config.config_dir must be {expected_config_dir}, "
                    f"got {config_dir_path}"
                )
            if not config_dir_path.is_dir():
                errors.append(f"state.config.config_dir does not exist: {config_dir_path}")

        expected_hashes = config.get("spec_sha256")
        if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(RUN_SPEC_NAMES):
            errors.append(
                "state.config.spec_sha256 must bind all immutable run config files"
            )
            expected_hashes = {}
        if config_dir_path is not None:
            for name in RUN_SPEC_NAMES:
                path = config_dir_path / name
                if not path.is_file() or path.stat().st_size == 0:
                    errors.append(f"approved run spec is missing or empty: {path}")
                    continue
                if path.resolve() != path:
                    errors.append(f"approved run spec must not be a symlink: {path}")
                    continue
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if expected_hashes.get(name) != digest:
                    errors.append(
                        f"state.config spec changed after approval: {path}"
                    )
            for field, name in (
                ("deft_config", "deft_config.yaml"),
                ("tao_spec", "tao_spec.yaml"),
            ):
                value = config.get(field)
                expected_path = config_dir_path / name
                if not value or not pathlib.Path(str(value)).is_absolute():
                    errors.append(f"state.config.{field} must be an absolute path")
                elif pathlib.Path(os.path.abspath(str(value))) != expected_path:
                    errors.append(
                        f"state.config.{field} must be {expected_path}"
                    )
                legacy_digest = config.get(f"{field}_sha256")
                if legacy_digest != expected_hashes.get(name):
                    errors.append(
                        f"state.config.{field}_sha256 disagrees with spec_sha256"
                    )

            approval_path = config_dir_path / "approval.json"
            approval_value = config.get("approval_manifest")
            if (
                not approval_value
                or pathlib.Path(os.path.abspath(str(approval_value))) != approval_path
            ):
                errors.append(
                    f"state.config.approval_manifest must be {approval_path}"
                )
            if approval_path.is_file():
                try:
                    approval = json.loads(approval_path.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append(f"approval manifest is invalid JSON: {exc}")
                else:
                    approval_version = approval.get("schema_version")
                    expected_approval = {
                        "schema_version": approval_version,
                        "workflow": WORKFLOW,
                        "workspace": str(pathlib.Path(str(config.get("workspace", ""))).resolve()),
                        "results_dir": str(results_dir.resolve()),
                        "dataset_root": str(pathlib.Path(str(config.get("dataset_root", ""))).resolve()),
                        "pas_deft_bundle_sha256": config.get("pas_deft_bundle_sha256"),
                        "images_archive": str(pathlib.Path(str(config.get("images_archive", ""))).resolve()),
                        "metadata_archive": str(pathlib.Path(str(config.get("metadata_archive", ""))).resolve()),
                        "checksums_file": (
                            str(pathlib.Path(str(config.get("checksums_file"))).resolve())
                            if config.get("checksums_file") is not None
                            else None
                        ),
                        "requires_hf_token": config.get("requires_hf_token"),
                        "max_iterations": max_iterations,
                        "host_gpu_ids": config.get("gpu_ids"),
                        "container_gpu_ids": config.get("container_gpu_ids"),
                        "metric_contract": gate,
                        "pyt_image": config.get("pyt_image"),
                        "ds_image": config.get("ds_image"),
                    }
                    if approval_version == "4":
                        expected_approval["platform"] = config.get("platform")
                        expected_approval["docker_remote"] = config.get(
                            "docker_remote", False
                        )
                        if "virtualenvs" in approval:
                            expected_approval["virtualenvs"] = config.get("virtualenvs")
                        else:
                            legacy_virtualenv = config.get("virtualenv")
                            profiles = config.get("virtualenvs")
                            if (
                                legacy_virtualenv is None
                                and isinstance(profiles, dict)
                                and profiles.get("pyt") == profiles.get("ds")
                            ):
                                legacy_virtualenv = profiles.get("pyt")
                            expected_approval["virtualenv"] = legacy_virtualenv
                    elif approval_version == "3":
                        if (
                            config.get("platform") != "docker"
                            or config.get("docker_remote", False)
                            or config.get("virtualenvs") is not None
                        ):
                            errors.append(
                                "approval manifest schema version 3 represents "
                                "local Docker only"
                            )
                    elif not (
                        approval_version == "2"
                        and config.get("platform") == "docker"
                        and config.get("virtualenv") is None
                    ):
                        errors.append(
                            "approval manifest schema must be version 4; versions 2 "
                            "and 3 are accepted only for legacy local Docker runs"
                        )
                    else:
                        expected_approval.pop("pas_deft_bundle_sha256")
                    if approval != expected_approval:
                        errors.append(
                            "state immutable approval fields disagree with approval.json"
                        )

            deft_path = config_dir_path / "deft_config.yaml"
            tao_path = config_dir_path / "tao_spec.yaml"
            text_embed_path = config_dir_path / "text_embed_spec.yaml"
            image_embed_path = config_dir_path / "image_embed_spec.yaml"
            mining_spec_path = config_dir_path / "mining_spec.yaml"
            if all(
                path.is_file()
                for path in (
                    deft_path,
                    tao_path,
                    text_embed_path,
                    image_embed_path,
                    mining_spec_path,
                )
            ):
                typed_config = None
                try:
                    from pas_deft.config import PasDeftConfig

                    typed_config = PasDeftConfig(str(deft_path))
                except Exception as exc:
                    errors.append(f"approved DEFT config fails typed schema validation: {exc}")
                try:
                    deft_payload = yaml.safe_load(deft_path.read_text())
                    tao_payload = yaml.safe_load(tao_path.read_text())
                    text_embed_payload = yaml.safe_load(text_embed_path.read_text())
                    image_embed_payload = yaml.safe_load(image_embed_path.read_text())
                    mining_spec_payload = yaml.safe_load(mining_spec_path.read_text())
                except (OSError, yaml.YAMLError) as exc:
                    errors.append(f"approved run config is not readable YAML: {exc}")
                else:
                    if not all(
                        isinstance(payload, dict)
                        for payload in (
                            deft_payload,
                            tao_payload,
                            text_embed_payload,
                            image_embed_payload,
                            mining_spec_payload,
                        )
                    ):
                        errors.append("approved run config roots must be objects")
                    else:
                        experiment = _config_section(
                            deft_payload, "experiment", "deft_config", errors
                        )
                        iteration = _config_section(
                            deft_payload, "iteration", "deft_config", errors
                        )
                        mining = _config_section(
                            deft_payload, "mining", "deft_config", errors
                        )
                        gap_config = _config_section(
                            deft_payload, "gap_analysis", "deft_config", errors
                        )
                        tao_train = _config_section(
                            tao_payload, "train", "tao_spec", errors
                        )
                        tao_evaluate = _config_section(
                            tao_payload, "evaluate", "tao_spec", errors
                        )
                        tao_dataset = _config_section(
                            tao_payload, "dataset", "tao_spec", errors
                        )
                        tao_dataset_train = _config_section(
                            tao_dataset, "train", "tao_spec.dataset", errors
                        )
                        tao_dataset_val = _config_section(
                            tao_dataset, "val", "tao_spec.dataset", errors
                        )
                        tao_optim = _config_section(
                            tao_train, "optim", "tao_spec.train", errors
                        )
                        eval_pairs_path = pathlib.Path(
                            typed_config.pas.eval_pairs_source_file
                            if typed_config is not None
                            else ""
                        ).expanduser()
                        eval_split = (
                            eval_pairs_path.name.removesuffix("_pairs.json")
                            if eval_pairs_path.name in {
                                "val_pairs.json",
                                "test_pairs.json",
                            }
                            else None
                        )
                        if typed_config is not None:
                            derived = {
                                "training_epochs": tao_train.get("num_epochs"),
                                "num_gpus": tao_train.get("num_gpus"),
                                "container_gpu_ids": tao_train.get("gpu_ids"),
                                "history_aware": typed_config.mining.history_aware.enabled,
                                "replay_fraction": typed_config.mining.history_aware.replay_fraction,
                                "mining_topn": typed_config.mining.topn,
                                "knn_metric": typed_config.mining.knn_metric,
                                "target_query_count": typed_config.gap_analysis.target_query_count,
                                "eval_split": eval_split,
                                "queries_per_slice": typed_config.gap_analysis.queries_per_slice,
                                "gap_query_types": typed_config.gap_analysis.query_types,
                                "vision_lr": tao_optim.get("vision_lr"),
                                "text_lr": tao_optim.get("text_lr"),
                                "train_batch_size": tao_dataset_train.get("batch_size"),
                                "val_batch_size": tao_dataset_val.get("batch_size"),
                                "eval_batch_size": tao_evaluate.get("batch_size"),
                                "text_embed_model": text_embed_payload.get("model"),
                                "continual_dataset": typed_config.training.continual_dataset,
                                "continual_model": typed_config.training.continual_model,
                                "visualize": typed_config.visualization.enabled,
                                "visualize_embeddings": typed_config.visualization.embeddings,
                            }
                            for field, expected_value in derived.items():
                                if config.get(field) != expected_value:
                                    errors.append(
                                        f"state.config.{field}={config.get(field)!r} "
                                        f"disagrees with approved config {expected_value!r}"
                                    )
                        if iteration.get("start") != 1 or iteration.get("end") != max_iterations:
                            errors.append(
                                "approved deft_config iteration range must be 1..max_iterations"
                            )
                        results_value = pathlib.Path(
                            str(experiment.get("results_path", ""))
                        ).expanduser()
                        if results_value.resolve() != results_dir.resolve():
                            errors.append(
                                "approved deft_config experiment.results_path "
                                "does not match state results_dir"
                            )
                        for key in ("train_config", "eval_config"):
                            value = pathlib.Path(str(experiment.get(key, ""))).expanduser()
                            if value.resolve() != tao_path.resolve():
                                errors.append(
                                    f"approved deft_config experiment.{key} must be {tao_path}"
                                )
                        if (
                            tao_evaluate.get("num_gpus") != tao_train.get("num_gpus")
                            or tao_evaluate.get("gpu_ids") != tao_train.get("gpu_ids")
                        ):
                            errors.append("approved TAO train/evaluate GPU shapes must match")
                        if image_embed_payload.get("model") != text_embed_payload.get(
                            "model"
                        ):
                            errors.append(
                                "approved image/text embedding model names must match"
                            )
                        text_model_path = text_embed_payload.get("model_path")
                        if (
                            not isinstance(text_model_path, str)
                            or not text_model_path.strip()
                            or image_embed_payload.get("model_path") != text_model_path
                        ):
                            errors.append(
                                "approved image/text embedding model paths must match"
                            )
                        if any(
                            key in mining
                            and mining_spec_payload.get(key) != mining.get(key)
                            for key in ("topn", "knn_metric")
                        ):
                            errors.append(
                                "legacy deft_config.mining topn/knn_metric duplicates "
                                "must match the authoritative mining_spec"
                            )
                        if gate is not None and gap_config.get("metric_name") != gate["metric_name"]:
                            errors.append(
                                "approved gap metric does not match the immutable metric contract"
                            )
                        dataset_base = pathlib.Path(str(config.get("dataset_root", ""))).resolve()
                        expected_pas_paths = {
                            "pool_pairs_source_file": dataset_base / "train_pairs.json",
                            "eval_pairs_source_file": dataset_base
                            / f"{config.get('eval_split')}_pairs.json",
                            "train_image_dir": dataset_base / "images",
                            "train_caption_dir": dataset_base / "captions",
                            "source_image_dir": dataset_base / "images",
                            "source_caption_dir": dataset_base / "captions",
                            "eval_image_dir": dataset_base / "images",
                            "eval_caption_dir": dataset_base / "captions",
                        }
                        if typed_config is not None:
                            for field, expected_path in expected_pas_paths.items():
                                actual_path = pathlib.Path(
                                    str(getattr(typed_config.pas, field))
                                ).expanduser()
                                if actual_path.resolve() != expected_path.resolve():
                                    errors.append(
                                        f"approved deft_config pas.{field} must be {expected_path}"
                                    )
        for field in ("platform", "pyt_image", "ds_image"):
            value = config.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"state.config.{field} must be a non-empty string")
        if config.get("platform") not in SUPPORTED_PLATFORMS:
            errors.append(
                "state.config.platform must be one of "
                + ", ".join(SUPPORTED_PLATFORMS)
            )
        docker_remote = config.get("docker_remote", False)
        if not isinstance(docker_remote, bool):
            errors.append("state.config.docker_remote must be boolean")
        elif docker_remote and config.get("platform") != "docker":
            errors.append(
                "state.config.docker_remote may be true only for platform=docker"
            )
        virtualenv = config.get("virtualenv")
        virtualenvs = config.get("virtualenvs")
        if config.get("platform") == "virtualenv":
            if isinstance(virtualenvs, dict) and set(virtualenvs) == {"pyt", "ds"}:
                selected_profiles = virtualenvs
            elif isinstance(virtualenv, str) and virtualenv.strip():
                selected_profiles = {"pyt": virtualenv, "ds": virtualenv}
            else:
                selected_profiles = {}
                errors.append(
                    "state.config.virtualenvs must bind the pyt and ds profiles"
                )
            for profile, selected in selected_profiles.items():
                if pathlib.Path(str(selected)).expanduser().resolve() == (
                    pathlib.Path(str(config.get("workspace"))).expanduser().resolve()
                    / ".venv"
                ).resolve():
                    errors.append(
                        f"state.config.virtualenvs.{profile} must be separate from "
                        "the workspace control .venv"
                    )
                    continue
                try:
                    venv = validate_tao_virtualenv(
                        pathlib.Path(str(selected)),
                        profile=profile,
                        probe_imports=False,
                    )
                except (OSError, ValueError) as exc:
                    errors.append(
                        f"state.config.virtualenvs.{profile} is not TAO-capable: {exc}"
                    )
                else:
                    if not venv.is_absolute():
                        errors.append(
                            f"state.config.virtualenvs.{profile} must be absolute"
                        )
        elif virtualenv is not None or virtualenvs is not None:
            errors.append(
                "state virtualenv configuration must be null unless platform is virtualenv"
            )
        if config.get("pyt_image") != PINNED_PYT_IMAGE:
            errors.append("state.config.pyt_image must be the pinned PAS PyTorch image")
        if config.get("ds_image") != PINNED_DS_IMAGE:
            errors.append("state.config.ds_image must be the pinned PAS data-services image")
        for field in (
            "training_epochs",
            "num_gpus",
            "mining_topn",
            "target_query_count",
            "queries_per_slice",
            "train_batch_size",
            "val_batch_size",
            "eval_batch_size",
        ):
            value = config.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                errors.append(f"state.config.{field} must be an integer >= 1")
        if config.get("eval_split") not in {"val", "test"}:
            errors.append("state.config.eval_split must be 'val' or 'test'")
        if not isinstance(config.get("gap_query_types"), str) or not config.get(
            "gap_query_types"
        ).strip():
            errors.append("state.config.gap_query_types must be a non-empty string")
        if config.get("text_embed_model") != "SigLIP":
            errors.append(
                "state.config.text_embed_model must be the TAO 7.2 shared "
                "image/text adapter token SigLIP"
            )
        for field in ("vision_lr", "text_lr"):
            value = config.get(field)
            if not _is_finite_number(value) or value <= 0.0:
                errors.append(f"state.config.{field} must be greater than zero")
        gpu_ids = config.get("gpu_ids")
        if not isinstance(gpu_ids, list) or len(gpu_ids) != config.get("num_gpus"):
            errors.append("state.config.gpu_ids must match state.config.num_gpus")
        container_gpu_ids = config.get("container_gpu_ids")
        num_gpus = config.get("num_gpus")
        expected_container_gpu_ids = (
            list(range(num_gpus))
            if isinstance(num_gpus, int)
            and not isinstance(num_gpus, bool)
            and num_gpus >= 1
            else None
        )
        if (
            expected_container_gpu_ids is None
            or container_gpu_ids != expected_container_gpu_ids
        ):
            errors.append(
                "state.config.container_gpu_ids must be a dense zero-based list "
                "matching state.config.num_gpus"
            )
        for field in (
            "history_aware",
            "continual_dataset",
            "continual_model",
            "visualize",
            "visualize_embeddings",
        ):
            if not isinstance(config.get(field), bool):
                errors.append(f"state.config.{field} must be a boolean")
        if not isinstance(config.get("requires_hf_token"), bool):
            errors.append("state.config.requires_hf_token must be a boolean")
        replay = config.get("replay_fraction")
        if not _is_finite_number(replay) or not 0.0 <= replay <= 1.0:
            errors.append("state.config.replay_fraction must be in [0, 1]")
        if config.get("knn_metric") not in {"cosine", "euclidean"}:
            errors.append("state.config.knn_metric must be cosine or euclidean")
        raw_layout = config.get("layout")
        if not isinstance(raw_layout, dict):
            errors.append("state.config.layout must be an object")
        else:
            layout = raw_layout
            expected_layout = {
                "baseline_dir": results_dir / "zs",
                "iteration_dir_template": results_dir / "iter_{N}",
                "pas_splits_dir": results_dir / "pas_splits",
                "source_embeddings_dir": results_dir / "embeddings" / "source",
                "caption_selection_history": results_dir / "caption_selection_history.json",
                "mining_selection_history": results_dir / "mining_selection_history.json",
            }
            for field, expected_path in expected_layout.items():
                value = layout.get(field)
                if not value:
                    errors.append(f"state.config.layout.{field} is required")
                elif not pathlib.Path(str(value)).expanduser().is_absolute():
                    errors.append(
                        f"state.config.layout.{field} must be absolute: {value}"
                    )
                else:
                    actual_path = pathlib.Path(str(value)).expanduser()
                    actual_absolute = pathlib.Path(os.path.abspath(actual_path))
                    expected_absolute = pathlib.Path(os.path.abspath(expected_path))
                    if actual_path != actual_absolute or actual_absolute != expected_absolute:
                        errors.append(
                            f"state.config.layout.{field} must be {expected_absolute}"
                        )
                    elif actual_absolute.resolve() != actual_absolute:
                        errors.append(
                            f"state.config.layout.{field} must not traverse a symlink"
                        )

    iterations = state.get("iterations")
    if not isinstance(iterations, dict):
        errors.append("state.iterations must be an object")
        iterations = {}

    # ---- loop_log.jsonl structural validation ------------------------------
    seen_keys: set[tuple[str, str]] = set()
    successful_keys: set[tuple[str, str]] = set()
    last_successful_stage: dict[str, str] = {}
    error_labels: set[str] = set()
    labels_with_events: set[str] = set()
    expected_seq = 0
    for index, entry in enumerate(entries, 1):
        seq = entry.get("seq")
        if seq != expected_seq:
            errors.append(
                f"loop_log entry {index} has seq={seq!r}; expected {expected_seq}"
            )
            expected_seq = (
                seq + 1
                if isinstance(seq, int) and not isinstance(seq, bool)
                else expected_seq + 1
            )
        else:
            expected_seq += 1
        label = entry.get("iteration")
        stage = entry.get("stage")
        log_status = entry.get("status")
        if label != "baseline" and not (
            isinstance(label, str) and re.fullmatch(r"iter[1-9][0-9]*", label)
        ):
            errors.append(f"loop_log seq={seq}: invalid iteration label {label!r}")
        if stage not in VALID_STAGES:
            errors.append(f"loop_log seq={seq}: invalid stage {stage!r}")
        if log_status not in VALID_LOG_STATUSES:
            errors.append(f"loop_log seq={seq}: invalid status {log_status!r}")
        elif log_status == "skip" and stage not in SKIPPABLE_STAGES:
            errors.append(
                f"loop_log seq={seq}: status 'skip' is valid only for "
                f"{sorted(SKIPPABLE_STAGES)}, got stage {stage!r}"
            )
        ts = entry.get("ts")
        if not isinstance(ts, str) or not ts.strip():
            errors.append(f"loop_log seq={seq}: ts must be a non-empty string")
        if not isinstance(entry.get("summary"), str) or not entry["summary"].strip():
            errors.append(f"loop_log seq={seq}: summary must be non-empty")
        duration = entry.get("duration_s")
        if duration is not None and (
            not _is_finite_number(duration) or duration < 0
        ):
            errors.append(
                f"loop_log seq={seq}: duration_s must be null or a "
                f"non-negative finite number"
            )
        tokens = entry.get("tokens")
        if tokens is not None and (
            not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0
        ):
            errors.append(
                f"loop_log seq={seq}: tokens must be null or a non-negative integer"
            )

        key = (str(label), str(stage))
        if key in seen_keys:
            errors.append(
                f"loop_log contains duplicate stage event {key}; one event per "
                f"stage is required"
            )
        seen_keys.add(key)
        labels_with_events.add(str(label))
        if log_status in ("ok", "skip") and stage != "loop_stop":
            successful_keys.add(key)
            last_successful_stage[str(label)] = str(stage)
        if log_status == "error":
            error_labels.add(str(label))
        if index == 1 and key != ("baseline", "dataset_setup"):
            errors.append(
                f"loop_log seq={seq}: first stage must be baseline/dataset_setup, "
                f"got {label}/{stage}"
            )
        elif index > 1:
            previous = entries[index - 2]
            allowed = _expected_next(previous, state)
            if key not in allowed:
                rendered = ", ".join(f"{i}/{s}" for i, s in sorted(allowed))
                errors.append(
                    f"loop_log seq={seq}: illegal transition "
                    f"{previous.get('iteration')}/{previous.get('stage')} -> "
                    f"{label}/{stage}; expected one of "
                    f"[{rendered or 'end-of-log'}]"
                )

    iter_numbers_in_log = sorted(
        {
            int(match.group(1))
            for entry in entries
            if (
                match := re.fullmatch(
                    r"iter([1-9][0-9]*)", str(entry.get("iteration"))
                )
            )
        }
    )
    expected_current = max(iter_numbers_in_log, default=0)
    if valid_current and current_iteration != expected_current:
        errors.append(
            f"state.current_iteration={current_iteration} does not match the "
            f"highest iteration committed in loop_log ({expected_current})"
        )
    if valid_max:
        for number in iter_numbers_in_log:
            if number > max_iterations:
                errors.append(
                    f"loop_log contains iter{number} events beyond "
                    f"max_iterations={max_iterations}"
                )

    # ---- per-iteration state validation ------------------------------------
    passed_labels: set[str] = set()
    metric_candidates: list[tuple[str, dict[str, Any]]] = []
    for label in sorted(iterations, key=_iteration_sort_key):
        info = iterations[label]
        if label != "baseline" and not re.fullmatch(r"iter[1-9][0-9]*", label):
            errors.append(f"state.iterations has invalid key {label!r}")
        if not isinstance(info, dict):
            errors.append(f"state.iterations.{label} must be an object")
            continue
        iter_status = info.get("status")
        if iter_status not in VALID_ITERATION_STATUSES:
            errors.append(
                f"state.iterations.{label}.status={iter_status!r} is invalid"
            )
        elif iter_status == "failed" and label not in error_labels:
            errors.append(
                f"state.iterations.{label}.status is 'failed' but loop_log has "
                f"no error event for it"
            )
        elif iter_status == "pending" and label in labels_with_events:
            errors.append(
                f"state.iterations.{label}.status is 'pending' despite "
                f"committed log events"
            )
        if label not in labels_with_events and not (
            label == "baseline" and iter_status == "pending"
        ):
            errors.append(
                f"state.iterations.{label} exists but loop_log has no events "
                f"for it"
            )

        completed = info.get("stage_completed")
        if completed is not None and completed not in VALID_COMPLETED_STAGES:
            errors.append(
                f"state.iterations.{label}.stage_completed={completed!r} is "
                f"invalid; use one of {sorted(VALID_COMPLETED_STAGES)}"
            )
        elif completed is not None and (label, str(completed)) not in successful_keys:
            errors.append(
                f"state says {label}/{completed} completed but loop_log has no "
                f"matching successful event"
            )
        expected_completed = last_successful_stage.get(label)
        if expected_completed is not None and completed != expected_completed:
            errors.append(
                f"state.iterations.{label}.stage_completed={completed!r} is "
                f"stale; last successful log stage is {expected_completed!r}"
            )

        phase_root = _phase_dir(results_dir, label)
        for field, value in info.items():
            if field in PATH_FIELDS and value:
                path = pathlib.Path(str(value)).expanduser()
                if not path.is_absolute():
                    errors.append(
                        f"state.iterations.{label}.{field} must be absolute: "
                        f"{value}"
                    )
                else:
                    absolute = pathlib.Path(os.path.abspath(path))
                    if absolute != path:
                        errors.append(
                            f"state.iterations.{label}.{field} must be a normalized "
                            f"absolute path: {value}"
                        )
                    elif field != "best_ckpt_path" and path.resolve() != absolute:
                        errors.append(
                            f"state.iterations.{label}.{field} must not traverse "
                            f"a symlink: {value}"
                        )
                if path.is_absolute() and field in RESULTS_SCOPED_DIR_FIELDS:
                    if not path.is_dir():
                        errors.append(
                            f"state.iterations.{label}.{field} does not exist "
                            f"or is not a directory: {value}"
                        )
                    else:
                        try:
                            populated = any(path.iterdir())
                        except OSError as exc:
                            populated = False
                            errors.append(
                                f"state.iterations.{label}.{field} cannot be "
                                f"listed: {exc}"
                            )
                        if not populated:
                            errors.append(
                                f"state.iterations.{label}.{field} is empty: "
                                f"{value}"
                            )
                elif path.is_absolute() and not path.exists():
                    errors.append(
                        f"state.iterations.{label}.{field} does not exist: "
                        f"{value}"
                    )
                elif path.is_absolute() and not path.is_file():
                    errors.append(
                        f"state.iterations.{label}.{field} is not a file: "
                        f"{value}"
                    )
                elif path.is_absolute() and path.stat().st_size == 0:
                    errors.append(
                        f"state.iterations.{label}.{field} is empty: {value}"
                    )
                elif path.is_absolute():
                    marker = MARKER_FIELDS.get(field)
                    if marker:
                        try:
                            text = path.read_text(errors="replace")
                        except OSError as exc:
                            errors.append(
                                f"state.iterations.{label}.{field} cannot be "
                                f"read: {exc}"
                            )
                        else:
                            if marker not in text:
                                errors.append(
                                    f"state.iterations.{label}.{field} must "
                                    f"contain {marker!r}: {value}"
                                )
                if path.is_absolute():
                    scope: pathlib.Path | None = None
                    if field in RESULTS_SCOPED_FILE_FIELDS or (
                        field in RESULTS_SCOPED_DIR_FIELDS
                    ):
                        scope = results_dir
                    elif field in PHASE_SCOPED_FILE_FIELDS:
                        scope = phase_root
                    if scope is not None:
                        try:
                            path.resolve().relative_to(scope.resolve())
                        except ValueError:
                            errors.append(
                                f"state.iterations.{label}.{field} must be "
                                f"under {scope}: {value}"
                            )
            stage = FIELD_STAGE.get(field)
            if (
                stage
                and value is not None
                and (label, stage) not in successful_keys
            ):
                errors.append(
                    f"state.iterations.{label}.{field} is set but loop_log "
                    f"lacks a successful {label}/{stage}"
                )

        for field in PATH_FIELDS:
            value = info.get(field)
            if not value:
                continue
            expected_path = _expected_artifact_path(
                field, label, results_dir, config
            )
            if expected_path is not None:
                actual_path = pathlib.Path(str(value)).expanduser()
                actual_absolute = pathlib.Path(os.path.abspath(actual_path))
                expected_absolute = pathlib.Path(os.path.abspath(expected_path))
                if actual_absolute != expected_absolute:
                    errors.append(
                        f"state.iterations.{label}.{field} must be "
                        f"{expected_absolute}, got {actual_absolute}"
                    )

        parquet_contracts = {
            "source_pool_parquet": {"filepath", "text"},
            "pool_embeddings_parquet": {"filepath", "embedding"},
            "gaps_parquet": {"filepath", "text", "weak_attribute"},
            "target_embeddings_parquet": {"filepath", "embedding"},
            "mined_parquet": {"filepath"},
        }
        for field, columns in parquet_contracts.items():
            value = info.get(field)
            if value and pathlib.Path(str(value)).is_file():
                _validate_parquet(
                    pathlib.Path(str(value)),
                    f"state.iterations.{label}.{field}",
                    errors,
                    columns,
                )

        json_contracts = {
            "caption_history": dict,
            "candidate_pairs": list,
            "mined_pairs": list,
            "mined_manifest": dict,
            "cumulative_names": list,
            "iteration_summary": dict,
        }
        for field, expected_type in json_contracts.items():
            value = info.get(field)
            if value and pathlib.Path(str(value)).is_file():
                _validate_json_shape(
                    pathlib.Path(str(value)),
                    f"state.iterations.{label}.{field}",
                    errors,
                    expected_type,
                )

        samples_value = info.get("samples_dir")
        if samples_value and pathlib.Path(str(samples_value)).is_dir():
            _validate_image_dir(
                pathlib.Path(str(samples_value)),
                f"state.iterations.{label}.samples_dir",
                errors,
            )
        tsne_value = info.get("tsne_plot")
        if tsne_value and pathlib.Path(str(tsne_value)).is_file():
            _validate_png(
                pathlib.Path(str(tsne_value)),
                f"state.iterations.{label}.tsne_plot",
                errors,
            )

        if (label, "dataset_setup") in successful_keys:
            splits = pathlib.Path(str(info.get("pas_splits_dir", "")))
            for name in ("eval_list.txt", "val_list.txt", "aug_pool_list.txt"):
                candidate = splits / name
                if not candidate.is_file() or candidate.stat().st_size == 0:
                    errors.append(f"committed PAS split is missing or empty: {candidate}")
            for name in ("eval_pairs.json", "aug_pool_pairs.json"):
                candidate = splits / name
                if candidate.is_file():
                    _validate_json_shape(
                        candidate, f"PAS split {name}", errors, list
                    )
                else:
                    errors.append(f"committed PAS split is missing: {candidate}")

        status_scopes = {
            "pool_embed_command_status": results_dir,
            "dataset_materialize_status": results_dir,
            "gap_analysis_status": results_dir,
        }
        if phase_root is not None:
            status_scopes.update(
                {
                    "target_embed_command_status": phase_root,
                    "knn_command_status": phase_root,
                    "eval_command_status": phase_root,
                    "train_command_status": phase_root,
                    "eval_config_status": phase_root,
                    "iteration_summary_status": phase_root,
                    "mining_postprocess_status": phase_root,
                    "history_select_status": phase_root,
                    "train_config_status": phase_root,
                    "publish_checkpoint_status": phase_root,
                }
            )
        status_outputs = {
            "pool_embed_command_status": ("pool_embeddings_parquet",),
            "dataset_materialize_status": ("source_pool_parquet",),
            "gap_analysis_status": ("gaps_parquet",),
            "target_embed_command_status": ("target_embeddings_parquet",),
            "knn_command_status": ("mined_parquet",),
            "eval_command_status": ("metrics_aggregate_csv", "eval_status_json"),
            "train_command_status": ("train_tao_status_json",),
            "eval_config_status": ("eval_config",),
            "iteration_summary_status": ("iteration_summary",),
            "mining_postprocess_status": ("candidate_pairs",),
            "history_select_status": (
                "mined_image_list",
                "mined_pairs",
                "mined_manifest",
                "cumulative_names",
                "mining_history",
            ),
            "train_config_status": ("train_config",),
            "publish_checkpoint_status": (
                "pretrained_state",
                "best_ckpt_metadata",
            ),
        }
        status_names = {
            "pool_embed_command_status": "pool_embed",
            "dataset_materialize_status": "dataset-materialize",
            "gap_analysis_status": "gap-analysis",
            "target_embed_command_status": "target_embed",
            "knn_command_status": "knn",
            "eval_command_status": "evaluate",
            "train_command_status": "train",
            "eval_config_status": "eval-config",
            "iteration_summary_status": "iteration-summary",
            "mining_postprocess_status": "mining-postprocess",
            "history_select_status": "history-select",
            "train_config_status": "train-config",
            "publish_checkpoint_status": "publish-checkpoint",
        }
        for field, scope in status_scopes.items():
            value = info.get(field)
            if not value or not pathlib.Path(str(value)).is_file():
                continue
            required_command = None
            required_kind = None
            required_image = None
            required_hf = None
            if field in {
                "pool_embed_command_status",
                "target_embed_command_status",
                "knn_command_status",
                "eval_command_status",
                "train_command_status",
            }:
                try:
                    required_command = expected_container_command(
                        status_names[field], label, config
                    )
                    required_kind = expected_image_kind(status_names[field])
                    required_image = config.get(f"{required_kind}_image")
                    required_hf = expected_hf_forwarding(status_names[field], config)
                except ValueError as exc:
                    errors.append(
                        f"state.iterations.{label}.{field} command contract: {exc}"
                    )
            payload = _validate_command_status(
                pathlib.Path(str(value)),
                f"state.iterations.{label}.{field}",
                errors,
                scope,
                required_name=status_names[field],
                required_command=required_command,
                required_image_kind=required_kind,
                required_image=required_image,
                required_hf_forwarding=required_hf,
                required_platform=config.get("platform"),
            )
            output_fields = status_outputs[field]
            if payload is not None:
                fresh = {
                    str(pathlib.Path(str(item)).resolve())
                    for item in payload.get("fresh_outputs", [])
                }
                started_ns = payload.get("started_ns")
                output_paths: list[tuple[str, pathlib.Path]] = []
                for output_field in output_fields:
                    output_value = info.get(output_field)
                    if not output_value:
                        continue
                    output_paths.append(
                        (output_field, pathlib.Path(str(output_value)).resolve())
                    )
                if field == "dataset_materialize_status":
                    split_root = pathlib.Path(str(info.get("pas_splits_dir", "")))
                    output_paths.extend(
                        (f"pas_splits_dir/{name}", (split_root / name).resolve())
                        for name in (
                            "eval_list.txt",
                            "eval_pairs.json",
                            "val_list.txt",
                            "aug_pool_list.txt",
                            "aug_pool_pairs.json",
                        )
                    )
                if field == "history_select_status" and not isinstance(
                    payload.get("resume"), bool
                ):
                    errors.append(
                        f"state.iterations.{label}.{field}.resume must be a boolean"
                    )
                for output_field, output_path in output_paths:
                    if str(output_path) not in fresh:
                        errors.append(
                            f"state.iterations.{label}.{field} does not bind fresh "
                            f"output {output_path}"
                        )
                    if (
                        isinstance(started_ns, int)
                        and output_path.is_file()
                        and output_path.stat().st_mtime_ns < started_ns
                        and not remote_freshness_attested(payload)
                        and not (
                            field == "history_select_status"
                            and payload.get("resume") is True
                        )
                    ):
                        errors.append(
                            f"state.iterations.{label}.{output_field} predates {field}"
                        )

        if (label, "train") in successful_keys and phase_root is not None:
            train_command_value = info.get("train_command_status")
            try:
                train_payload = json.loads(
                    pathlib.Path(str(train_command_value)).read_text()
                )
                if not isinstance(train_payload, dict):
                    raise ValueError("train command status root must be an object")
                lineage_started_ns = checkpoint_lineage_started_ns(
                    train_payload, state.get("started_at")
                )
                provenance = validate_best_checkpoint(
                    pathlib.Path(str(info.get("best_ckpt_path", ""))),
                    phase_root / "train",
                    started_ns=lineage_started_ns,
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                errors.append(
                    f"state.iterations.{label} best checkpoint provenance is invalid: {exc}"
                )
            else:
                for field in (
                    "best_ckpt_path",
                    "best_ckpt_metadata",
                    "best_ckpt_source",
                    "publish_mode",
                ):
                    if info.get(field) != provenance[field]:
                        errors.append(
                            f"state.iterations.{label}.{field} disagrees with "
                            "checkpoint publication evidence"
                        )
                publish_value = info.get("publish_checkpoint_status")
                try:
                    publish_payload = json.loads(
                        pathlib.Path(str(publish_value)).read_text()
                    )
                    if not isinstance(publish_payload, dict):
                        raise ValueError(
                            "publish checkpoint status root must be an object"
                        )
                except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
                    errors.append(
                        f"state.iterations.{label}.publish_checkpoint_status "
                        f"cannot be checked against best checkpoint: {exc}"
                    )
                else:
                    publish_outputs = publish_payload.get("fresh_outputs")
                    if not isinstance(publish_outputs, list):
                        publish_outputs = []
                    publish_fresh = {
                        str(
                            pathlib.Path(
                                os.path.abspath(pathlib.Path(str(item)).expanduser())
                            )
                        )
                        for item in publish_outputs
                        if isinstance(item, str)
                    }
                    if provenance["best_ckpt_path"] not in publish_fresh:
                        errors.append(
                            f"state.iterations.{label}.publish_checkpoint_status "
                            "does not bind the canonical best checkpoint"
                        )

        visual_statuses = info.get("visualize_command_statuses")
        if visual_statuses is not None and phase_root is not None:
            if not isinstance(visual_statuses, list) or not visual_statuses:
                errors.append(
                    f"state.iterations.{label}.visualize_command_statuses must "
                    "be a non-empty list"
                )
            else:
                expected_visual_outputs = [
                    phase_root / "embeddings" / "viz_weak" / "embeddings.parquet",
                    phase_root / "embeddings" / "augmented" / "mined_embeddings.parquet",
                ]
                previous_input = (
                    phase_root / "embeddings" / "previous" / "prev_pool.parquet"
                )
                if previous_input.is_file() and previous_input.stat().st_size > 0:
                    expected_visual_outputs.append(
                        phase_root / "embeddings" / "previous" / "embeddings.parquet"
                    )
                if len(visual_statuses) != len(expected_visual_outputs):
                    errors.append(
                        f"state.iterations.{label}.visualize_command_statuses "
                        f"has {len(visual_statuses)} item(s), expected "
                        f"{len(expected_visual_outputs)}"
                    )
                for index, value in enumerate(visual_statuses):
                    path = pathlib.Path(str(value))
                    if not path.is_file():
                        errors.append(
                            f"state.iterations.{label}.visualize_command_statuses"
                            f"[{index}] does not exist: {path}"
                        )
                    else:
                        if index >= len(expected_visual_outputs):
                            continue
                        command_name = (
                            "viz_weak_embed"
                            if index == 0
                            else "viz_mined_embed"
                            if index == 1
                            else "viz_previous_embed"
                        )
                        expected_status = (
                            expected_visual_outputs[index].parent
                            / f"{command_name}.status.json"
                        )
                        if pathlib.Path(os.path.abspath(path)) != expected_status:
                            errors.append(
                                f"state.iterations.{label}.visualize_command_statuses"
                                f"[{index}] must be {expected_status}, got {path}"
                            )
                        payload = _validate_command_status(
                            path,
                            f"state.iterations.{label}.visualize_command_statuses[{index}]",
                            errors,
                            phase_root,
                            required_name=command_name,
                            required_command=expected_container_command(
                                command_name, label, config
                            ),
                            required_image_kind=expected_image_kind(command_name),
                            required_image=config["ds_image"],
                            required_hf_forwarding=expected_hf_forwarding(
                                command_name, config
                            ),
                            required_platform=config.get("platform"),
                        )
                        output = expected_visual_outputs[index]
                        _validate_parquet(
                            output,
                            f"state.iterations.{label}.visualization_embedding[{index}]",
                            errors,
                            {"filepath", "embedding"},
                        )
                        if payload is not None:
                            fresh = {
                                str(pathlib.Path(str(item)).resolve())
                                for item in payload.get("fresh_outputs", [])
                            }
                            if str(output.resolve()) not in fresh:
                                errors.append(
                                    f"state.iterations.{label}.visualize_command_statuses"
                                    f"[{index}] does not bind fresh output {output.resolve()}"
                                )
                            started_ns = payload.get("started_ns")
                            if (
                                isinstance(started_ns, int)
                                and output.is_file()
                                and output.stat().st_mtime_ns < started_ns
                                and not remote_freshness_attested(payload)
                            ):
                                errors.append(
                                    f"state.iterations.{label}.visualization_embedding"
                                    f"[{index}] predates its command status"
                                )

        if phase_root is not None and info.get("visualize_prepare_status"):
            prepare_path = pathlib.Path(str(info["visualize_prepare_status"]))
            prepare_payload = _validate_command_status(
                prepare_path,
                f"state.iterations.{label}.visualize_prepare_status",
                errors,
                phase_root,
                required_name="visualize-prepare",
            )
            prepare_outputs: list[pathlib.Path] = []
            if config.get("visualize") and info.get("samples_dir"):
                prepare_outputs.append(pathlib.Path(str(info["samples_dir"])))
            if config.get("visualize_embeddings"):
                weak_input = phase_root / "embeddings" / "viz_weak" / "input.parquet"
                mined_input = phase_root / "mining" / "mined_unique_images.parquet"
                _validate_parquet(
                    weak_input,
                    f"state.iterations.{label}.visualization_weak_input",
                    errors,
                    {"filepath"},
                )
                _validate_parquet(
                    mined_input,
                    f"state.iterations.{label}.visualization_mined_input",
                    errors,
                    {"filepath"},
                )
                prepare_outputs.extend([weak_input, mined_input])
                previous_input = (
                    phase_root / "embeddings" / "previous" / "prev_pool.parquet"
                )
                if previous_input.is_file() and previous_input.stat().st_size > 0:
                    _validate_parquet(
                        previous_input,
                        f"state.iterations.{label}.visualization_previous_input",
                        errors,
                        {"filepath"},
                    )
                    prepare_outputs.append(previous_input)
            if prepare_payload is not None:
                fresh = {
                    str(pathlib.Path(str(item)).resolve())
                    for item in prepare_payload.get("fresh_outputs", [])
                }
                started_ns = prepare_payload.get("started_ns")
                for output in prepare_outputs:
                    if str(output.resolve()) not in fresh:
                        errors.append(
                            f"state.iterations.{label}.visualize_prepare_status "
                            f"does not bind fresh output {output.resolve()}"
                        )
                    if (
                        isinstance(started_ns, int)
                        and output.exists()
                        and output.stat().st_mtime_ns < started_ns
                        and not remote_freshness_attested(prepare_payload)
                    ):
                        errors.append(
                            f"state.iterations.{label} visualization input/output "
                            f"predates visualize_prepare_status: {output}"
                        )

        if phase_root is not None and info.get("visualize_finish_status"):
            finish_path = pathlib.Path(str(info["visualize_finish_status"]))
            finish_payload = _validate_command_status(
                finish_path,
                f"state.iterations.{label}.visualize_finish_status",
                errors,
                phase_root,
                required_name="visualize-finish",
            )
            tsne = pathlib.Path(str(info.get("tsne_plot", ""))).expanduser()
            if finish_payload is not None and tsne.is_file():
                fresh = {
                    str(pathlib.Path(str(item)).resolve())
                    for item in finish_payload.get("fresh_outputs", [])
                }
                if str(tsne.resolve()) not in fresh:
                    errors.append(
                        f"state.iterations.{label}.visualize_finish_status does "
                        f"not bind fresh output {tsne.resolve()}"
                    )
                started_ns = finish_payload.get("started_ns")
                if (
                    isinstance(started_ns, int)
                    and tsne.stat().st_mtime_ns < started_ns
                    and not remote_freshness_attested(finish_payload)
                ):
                    errors.append(
                        f"state.iterations.{label}.tsne_plot predates "
                        "visualize_finish_status"
                    )

        raw_result = info.get("metric_result")
        if raw_result is not None:
            if not isinstance(raw_result, dict):
                errors.append(
                    f"state.iterations.{label}.metric_result must be an object"
                )
            else:
                value = raw_result.get("value")
                valid_value = _is_finite_number(value)
                if not valid_value:
                    errors.append(
                        f"state.iterations.{label}.metric_result.value must be "
                        f"a finite number"
                    )
                if gate is not None:
                    if (
                        str(raw_result.get("metric_name", "")).strip().lower()
                        != gate["metric_name"].lower()
                    ):
                        errors.append(
                            f"state.iterations.{label}.metric_result."
                            f"metric_name={raw_result.get('metric_name')!r} "
                            f"does not match the gate metric "
                            f"{gate['metric_name']!r}"
                        )
                    if (
                        str(raw_result.get("query_type", "")).strip().lower()
                        != gate["query_type"].lower()
                    ):
                        errors.append(
                            f"state.iterations.{label}.metric_result."
                            f"query_type={raw_result.get('query_type')!r} does "
                            f"not match the gate query type "
                            f"{gate['query_type']!r}"
                        )
                passed = raw_result.get("passed")
                if not isinstance(passed, bool):
                    errors.append(
                        f"state.iterations.{label}.metric_result.passed must "
                        f"be a boolean"
                    )
                elif gate is not None and valid_value:
                    recomputed = _gate_passes(gate, value)
                    if passed != recomputed:
                        errors.append(
                            f"state.iterations.{label}.metric_result.passed="
                            f"{passed} disagrees with the metric gate "
                            f"comparison ({recomputed})"
                        )
                if passed is True:
                    passed_labels.add(label)
                evidence = raw_result.get("evidence_path")
                if not evidence:
                    errors.append(
                        f"state.iterations.{label}.metric_result.evidence_path "
                        f"is required"
                    )
                else:
                    evidence_path = pathlib.Path(str(evidence)).expanduser()
                    if not evidence_path.is_absolute():
                        errors.append(
                            f"state.iterations.{label}.metric_result."
                            f"evidence_path must be absolute"
                        )
                    elif pathlib.Path(os.path.abspath(evidence_path)) != evidence_path:
                        errors.append(
                            f"state.iterations.{label}.metric_result.evidence_path "
                            "must be a normalized absolute path"
                        )
                    elif evidence_path.resolve() != evidence_path:
                        errors.append(
                            f"state.iterations.{label}.metric_result.evidence_path "
                            "must not traverse a symlink"
                        )
                    elif not evidence_path.is_file():
                        errors.append(
                            f"state.iterations.{label}.metric_result."
                            f"evidence_path does not exist: {evidence}"
                        )
                    elif evidence_path.stat().st_size == 0:
                        errors.append(
                            f"state.iterations.{label}.metric_result."
                            f"evidence_path is empty: {evidence}"
                        )
                    elif gate is not None and phase_root is not None:
                        expected_evidence = phase_root / "evaluate" / "metric_result.json"
                        if evidence_path != pathlib.Path(os.path.abspath(expected_evidence)):
                            errors.append(
                                f"state.iterations.{label}.metric_result.evidence_path "
                                f"must be {expected_evidence}"
                            )
                        try:
                            raw_evidence = json.loads(evidence_path.read_text())
                        except (OSError, json.JSONDecodeError) as exc:
                            errors.append(
                                f"state.iterations.{label}.metric_result evidence "
                                f"is invalid JSON: {exc}"
                            )
                        else:
                            if not isinstance(raw_evidence, dict):
                                errors.append(
                                    f"state.iterations.{label}.metric_result "
                                    "evidence root must be an object"
                                )
                                raw_evidence = {}
                            metrics_path = (
                                phase_root
                                / "evaluate"
                                / PAS_METRICS_AGGREGATE_FILENAME
                            )
                            if raw_evidence.get("iter_label") != label:
                                errors.append(
                                    f"metric evidence iter_label={raw_evidence.get('iter_label')!r} "
                                    f"does not match {label!r}"
                                )
                            source = pathlib.Path(
                                str(raw_evidence.get("source_csv", ""))
                            ).expanduser()
                            if source.resolve() != metrics_path.resolve():
                                errors.append(
                                    f"metric evidence source_csv must be {metrics_path}"
                                )
                            elif metrics_path.is_file():
                                try:
                                    recomputed = build_result(
                                        argparse.Namespace(
                                            metrics_csv=metrics_path,
                                            metric_name=gate["metric_name"],
                                            query_type=gate["query_type"],
                                            op=gate["op"],
                                            target=gate["target"],
                                            iter_label=label,
                                        )
                                    )
                                except (OSError, ValueError) as exc:
                                    errors.append(
                                        f"metric evidence cannot be recomputed for {label}: {exc}"
                                    )
                                else:
                                    if not math.isclose(
                                        float(value),
                                        float(recomputed["value"]),
                                        rel_tol=1e-12,
                                        abs_tol=1e-12,
                                    ):
                                        errors.append(
                                            f"state.iterations.{label}.metric_result.value "
                                            "disagrees with its source CSV"
                                        )
                                    for source_name, payload in (
                                        ("metric evidence", raw_evidence),
                                        (f"state.iterations.{label}.metric_result", raw_result),
                                    ):
                                        for field in (
                                            "schema_version",
                                            "workflow",
                                            "iter_label",
                                            "metric_name",
                                            "query_type",
                                            "op",
                                            "passed",
                                            "row",
                                        ):
                                            if payload.get(field) != recomputed.get(field):
                                                errors.append(
                                                    f"{source_name}.{field} disagrees with "
                                                    "the canonical CSV-derived result"
                                                )
                                        payload_target = payload.get("target")
                                        expected_target = recomputed.get("target")
                                        if (payload_target is None) != (expected_target is None):
                                            errors.append(
                                                f"{source_name}.target disagrees with the "
                                                "metric contract"
                                            )
                                        elif payload_target is not None and (
                                            not _is_finite_number(payload_target)
                                            or not math.isclose(
                                                float(payload_target),
                                                float(expected_target),
                                                rel_tol=1e-12,
                                                abs_tol=1e-12,
                                            )
                                        ):
                                            errors.append(
                                                f"{source_name}.target disagrees with the "
                                                "metric contract"
                                            )
                                        payload_value = payload.get("value")
                                        if (
                                            not _is_finite_number(payload_value)
                                            or not math.isclose(
                                                float(payload_value),
                                                float(recomputed["value"]),
                                                rel_tol=1e-12,
                                                abs_tol=1e-12,
                                            )
                                        ):
                                            errors.append(
                                                f"{source_name}.value disagrees with the "
                                                "canonical CSV-derived result"
                                            )
                if valid_value:
                    metric_candidates.append((label, raw_result))

        if iter_status == "complete":
            if not isinstance(raw_result, dict):
                errors.append(
                    f"complete iteration {label} is missing metric_result for "
                    f"the configured gate metric"
                )
            if (label, "evaluate") not in successful_keys:
                errors.append(
                    f"complete iteration {label} has no committed evaluate event"
                )
            if completed not in {"evaluate", "gap_analysis"}:
                errors.append(
                    f"complete iteration {label} must end at evaluate or "
                    f"gap_analysis, got {completed!r}"
                )

    # ---- log -> state proof validation --------------------------------------
    for entry in entries:
        label = str(entry.get("iteration"))
        stage = str(entry.get("stage"))
        log_status = entry.get("status")
        if stage == "loop_stop":
            continue
        info = iterations.get(label)
        if not isinstance(info, dict):
            errors.append(
                f"loop_log commits {label}/{stage} but "
                f"state.iterations.{label} is missing"
            )
            continue
        if log_status == "error":
            if info.get("status") != "failed":
                errors.append(
                    f"loop_log records {label}/{stage} error but "
                    f"state.iterations.{label}.status is not 'failed'"
                )
            continue
        if log_status == "skip":
            if info.get("visualize_skipped") is not True:
                errors.append(
                    f"loop_log records {label}/visualize skip but state lacks "
                    f"visualize_skipped=true"
                )
            if config.get("visualize") or config.get("visualize_embeddings"):
                errors.append(
                    f"loop_log skips {label}/visualize although an approved "
                    "visualization flag is enabled"
                )
            continue
        if log_status != "ok":
            continue
        required = STAGE_REQUIRED_FIELDS.get(stage)
        if required:
            missing = [field for field in required if not info.get(field)]
            if missing:
                errors.append(
                    f"loop_log commits {label}/{stage} but state lacks "
                    f"required proof: {'+'.join(missing)}"
                )
        if stage == "evaluate" and label != "baseline" and not info.get(
            "iteration_summary"
        ):
            errors.append(
                f"loop_log commits {label}/evaluate but state lacks "
                "iteration_summary"
            )
        if stage == "evaluate" and label != "baseline" and not info.get(
            "iteration_summary_status"
        ):
            errors.append(
                f"loop_log commits {label}/evaluate but state lacks "
                "iteration_summary_status"
            )
        if (
            stage == "dataset_setup"
            and config.get("checksums_file")
            and not info.get("checksum_verify_log")
        ):
            errors.append(
                "loop_log commits baseline/dataset_setup with an approved checksum "
                "manifest but state lacks checksum_verify_log"
            )
        if stage == "history_select" and config.get("history_aware") and not info.get(
            "mining_history"
        ):
            errors.append(
                f"loop_log commits {label}/history_select but state lacks "
                "mining_history"
            )
        if stage == "visualize":
            visual_missing: list[str] = ["visualize_prepare_status"] if not info.get(
                "visualize_prepare_status"
            ) else []
            if config.get("visualize") and not info.get("samples_dir"):
                visual_missing.append("samples_dir")
            if config.get("visualize_embeddings"):
                if not info.get("tsne_plot"):
                    visual_missing.append("tsne_plot")
                if not info.get("visualize_command_statuses"):
                    visual_missing.append("visualize_command_statuses")
                if not info.get("visualize_finish_status"):
                    visual_missing.append("visualize_finish_status")
            if visual_missing:
                errors.append(
                    f"loop_log commits {label}/visualize but state lacks "
                    + "+".join(visual_missing)
                )
        if stage == "evaluate" and label not in error_labels:
            if info.get("status") != "complete":
                errors.append(
                    f"loop_log commits {label}/evaluate but "
                    f"state.iterations.{label}.status is not 'complete'"
                )

    # ---- gate_met coherence --------------------------------------------------
    if isinstance(gate_met, bool):
        if gate_met and not passed_labels:
            errors.append(
                "state.gate_met is true but no recorded metric_result has "
                "passed=true"
            )
        elif not gate_met and passed_labels:
            errors.append(
                f"state.gate_met is false but {sorted(passed_labels)} recorded "
                f"passed=true"
            )

    # ---- loop_stop coherence --------------------------------------------------
    ok_stop_events = [
        entry
        for entry in entries
        if str(entry.get("stage")) == "loop_stop" and entry.get("status") == "ok"
    ]
    reason = state.get("loop_stop_reason")
    if ok_stop_events:
        if reason not in LOOP_STOP_REASONS:
            errors.append(
                f"loop_log has a successful loop_stop but "
                f"state.loop_stop_reason={reason!r} is not one of "
                f"{list(LOOP_STOP_REASONS)}"
            )
        elif reason == "kpi_met" and gate_met is not True:
            errors.append(
                "loop_stop reason kpi_met requires gate_met=true in state"
            )
        elif reason == "max_iterations" and valid_max and valid_current:
            if current_iteration < max_iterations:
                errors.append(
                    f"loop_stop reason max_iterations requires "
                    f"current_iteration >= max_iterations "
                    f"({current_iteration} < {max_iterations})"
                )
    elif reason is not None:
        errors.append(
            "state.loop_stop_reason is set but loop_log has no successful "
            "loop_stop event"
        )

    # ---- gaps parquet feed for started iterations ----------------------------
    for number in iter_numbers_in_log:
        label = f"iter{number}"
        feeder = "baseline" if number == 1 else f"iter{number - 1}"
        if (feeder, "gap_analysis") not in successful_keys:
            errors.append(
                f"{label} has started but loop_log lacks a committed "
                f"{feeder}/gap_analysis to feed it"
            )
        feeder_info = iterations.get(feeder)
        gaps = (
            feeder_info.get("gaps_parquet")
            if isinstance(feeder_info, dict)
            else None
        )
        if not gaps:
            errors.append(
                f"{label} has started but state.iterations.{feeder}."
                f"gaps_parquet is not recorded"
            )
        else:
            gaps_path = pathlib.Path(str(gaps)).expanduser()
            if not gaps_path.is_file() or gaps_path.stat().st_size == 0:
                errors.append(
                    f"{label} depends on a missing or empty gaps parquet: {gaps}"
                )

    # ---- leakage guard: mined lists must not intersect the eval split --------
    baseline_info = iterations.get("baseline")
    splits_value = (
        baseline_info.get("pas_splits_dir")
        if isinstance(baseline_info, dict)
        else None
    ) or layout.get("pas_splits_dir")
    eval_list_path: pathlib.Path | None = None
    if splits_value:
        candidate = pathlib.Path(str(splits_value)).expanduser() / "eval_list.txt"
        if candidate.is_file():
            eval_list_path = candidate
    if eval_list_path is not None:
        eval_names = _basenames(eval_list_path, str(eval_list_path), errors)
        for label in sorted(iterations, key=_iteration_sort_key):
            info = iterations[label]
            if not isinstance(info, dict):
                continue
            if (label, "history_select") not in successful_keys:
                continue
            mined_value = info.get("mined_image_list")
            if not mined_value:
                continue  # missing proof reported above
            mined_path = pathlib.Path(str(mined_value)).expanduser()
            if not mined_path.is_file():
                continue  # missing artifact reported above
            overlap = sorted(
                _basenames(
                    mined_path,
                    f"state.iterations.{label}.mined_image_list",
                    errors,
                )
                & eval_names
            )
            if overlap:
                sample = ", ".join(overlap[:3])
                errors.append(
                    f"state.iterations.{label}.mined_image_list leaks "
                    f"{len(overlap)} eval-split image(s) into training "
                    f"(e.g. {sample}); mined lists must have zero basename "
                    f"overlap with {eval_list_path}"
                )
    elif any(key[1] == "history_select" for key in successful_keys):
        errors.append(
            "history_select is committed but pas_splits/eval_list.txt is "
            "missing; leakage cannot be validated"
        )

    # ---- caption_selection_history.json coherence ----------------------------
    caption_path = results_dir / "caption_selection_history.json"
    committed_gap_feeds: set[int] = set()
    for gap_label, gap_stage in successful_keys:
        if gap_stage != "gap_analysis":
            continue
        if gap_label == "baseline":
            committed_gap_feeds.add(1)
            continue
        gap_match = re.fullmatch(r"iter([1-9][0-9]*)", gap_label)
        if gap_match:
            committed_gap_feeds.add(int(gap_match.group(1)) + 1)
    if caption_path.is_file():
        try:
            caption_payload = json.loads(caption_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"caption_selection_history.json is not readable JSON: {exc}")
        else:
            if not isinstance(caption_payload, dict):
                errors.append("caption_selection_history.json root must be an object")
            else:
                raw_entries = caption_payload.get("entries")
                if not isinstance(raw_entries, list):
                    errors.append("caption_selection_history.json entries must be a list")
                    caption_numbers: set[int] = set()
                else:
                    caption_numbers = {
                        item.get("iter")
                        for item in raw_entries
                        if isinstance(item, dict)
                        and isinstance(item.get("iter"), int)
                        and not isinstance(item.get("iter"), bool)
                        and item.get("iter") >= 1
                    }
                    malformed = [
                        item for item in raw_entries
                        if not isinstance(item, dict)
                        or not isinstance(item.get("iter"), int)
                        or isinstance(item.get("iter"), bool)
                        or item.get("iter") < 1
                    ]
                    if malformed:
                        errors.append(
                            "caption_selection_history.json has entries with invalid iter values"
                        )
                for number in sorted(committed_gap_feeds - caption_numbers):
                    errors.append(
                        f"gap analysis feeding iter{number} is committed but caption "
                        "history has no matching entry"
                    )
                extra_caption = caption_numbers - committed_gap_feeds
                expected_in_flight: set[int] = set()
                if entries:
                    for next_label, next_stage in _expected_next(entries[-1], state):
                        if next_stage == "gap_analysis":
                            expected_in_flight.add(
                                1 if next_label == "baseline" else int(next_label[4:]) + 1
                            )
                for number in sorted(extra_caption):
                    message = (
                        f"caption_selection_history.json records feed iteration {number} "
                        "before its gap_analysis commit; rerun the idempotent gap adapter "
                        "once, then commit"
                    )
                    if number in expected_in_flight and not require_complete:
                        warnings.append(message)
                    else:
                        errors.append(message)
                last_iter = caption_payload.get("last_iter")
                if caption_numbers and last_iter != max(caption_numbers):
                    errors.append(
                        "caption_selection_history.json last_iter does not match its entries"
                    )
    elif committed_gap_feeds:
        errors.append(
            "gap_analysis is committed but caption_selection_history.json is missing"
        )

    # ---- mining_selection_history.json coherence -----------------------------
    history_value = layout.get("mining_selection_history") or str(
        results_dir / "mining_selection_history.json"
    )
    history_path = pathlib.Path(str(history_value)).expanduser()
    committed_select = {
        int(match.group(1))
        for key in successful_keys
        if key[1] == "history_select"
        and (match := re.fullmatch(r"iter([1-9][0-9]*)", key[0]))
    }
    if history_path.is_file():
        try:
            history_payload = json.loads(history_path.read_text())
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(
                f"mining_selection_history.json is not readable JSON: {exc}"
            )
        else:
            if not isinstance(history_payload, dict):
                errors.append("mining_selection_history.json root must be an object")
            else:
                history_numbers = set(
                    _history_iteration_numbers(history_payload, errors)
                )
                started = set(iter_numbers_in_log)
                in_flight = max(started, default=0)
                for number in sorted(history_numbers):
                    if number in committed_select:
                        continue
                    if number in started and number == in_flight:
                        message = (
                            f"mining_selection_history.json records iteration "
                            f"{number} but iter{number}/history_select is not "
                            f"committed; a prior attempt crashed mid-stage — "
                            f"follow the sanctioned recovery in "
                            f"{RECOVERY_REFERENCE} before rerunning "
                            f"history_select"
                        )
                        if require_complete:
                            errors.append(message)
                        else:
                            warnings.append(message)
                    else:
                        errors.append(
                            f"mining_selection_history.json records iteration "
                            f"{number} with no committed "
                            f"iter{number}/history_select and it is not the "
                            f"in-flight iteration"
                        )
                for number in sorted(committed_select - history_numbers):
                    errors.append(
                        f"iter{number}/history_select is committed but "
                        f"mining_selection_history.json has no iteration "
                        f"{number} entry"
                    )
    elif config.get("history_aware") and committed_select:
        errors.append(
            f"history-aware selections are committed but history ledger is "
            f"missing: {history_path}"
        )

    # ---- terminal / status classification -------------------------------------
    last = entries[-1] if entries else None
    terminal = bool(last and str(last.get("stage")) == "loop_stop")
    error_entries = [entry for entry in entries if entry.get("status") == "error"]
    if (
        terminal
        and last.get("status") == "ok"
        and reason in COMPLETION_REASONS
        and not error_entries
    ):
        incomplete = sorted(
            label
            for label, info in iterations.items()
            if isinstance(info, dict) and info.get("status") != "complete"
        )
        if incomplete:
            errors.append(
                f"loop_stop reason {reason} requires every iteration to be "
                f"complete; not complete: {incomplete}"
            )
    if errors:
        status = "INVALID"
    elif terminal and (
        error_entries or last.get("status") != "ok" or reason == "hard_stop"
    ):
        status = "FAILED"
        warnings.append("run ended after a hard stop; do not claim KPI completion")
    elif terminal:
        status = "COMPLETE"
    elif last and last.get("status") == "error":
        status = "FAILED"
        warnings.append(
            "last committed stage is a hard stop; commit loop_stop and do not "
            "auto-retry"
        )
    else:
        status = "IN_PROGRESS"

    best_label: str | None = None
    best_result: dict[str, Any] | None = None
    if gate is not None and metric_candidates:
        candidates = [
            (label, iterations[label], result)
            for label, result in metric_candidates
            if isinstance(iterations.get(label), dict)
        ]
        best_label, _, best_result = pick_best(candidates, gate)

    next_action, reference_names = _next_action(
        state, entries, iterations, status, terminal
    )
    required_reference = (
        _render_references(reference_names) if reference_names else None
    )
    if reference_names:
        for name in reference_names:
            if not (_skill_root() / name).is_file():
                warnings.append(
                    f"stage reference {name} is missing under {_skill_root()}; "
                    f"stop rather than substituting generic commands"
                )

    return {
        "status": status,
        "terminal": terminal,
        "results_dir": str(results_dir),
        "workflow": state.get("workflow"),
        "max_iterations": state.get("max_iterations"),
        "current_iteration": state.get("current_iteration"),
        "gate_met": state.get("gate_met"),
        "loop_stop_reason": state.get("loop_stop_reason"),
        "metric_gate": gate,
        "log_entries": len(entries),
        "last_committed": last,
        "best_iteration": best_label,
        "best_metric_result": best_result,
        "next_action": next_action,
        "required_reference": required_reference,
        "errors": errors,
        "warnings": warnings,
    }


def _print_text(report: dict[str, Any]) -> None:
    print(f"DEFT_RUN_STATUS={report['status']}")
    print(f"results_dir={report['results_dir']}")
    print(
        f"iteration={report['current_iteration']}/{report['max_iterations']} "
        f"log_entries={report['log_entries']} gate_met={report['gate_met']}"
    )
    if report["last_committed"]:
        last = report["last_committed"]
        print(
            "last_committed="
            f"seq:{last.get('seq')} {last.get('iteration')}/{last.get('stage')} "
            f"status:{last.get('status')}"
        )
    else:
        print("last_committed=none")
    gate = report["metric_gate"]
    if report["best_iteration"] is not None and gate is not None:
        result = report["best_metric_result"]
        print(
            f"best={report['best_iteration']} "
            f"metric={gate['metric_name']}({gate['query_type']}) "
            f"value={float(result['value']):.6g} target={_render_target(gate)}"
        )
    print(f"next_action={report['next_action']}")
    if report["required_reference"]:
        print(f"read_before_action={report['required_reference']}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    for error in report["errors"]:
        print(f"ERROR: {error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--require-terminal",
        action="store_true",
        help="fail unless the last committed event is loop_stop (failed runs allowed)",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "fail unless a successful loop_stop with reason kpi_met or "
            "max_iterations exists and every iteration is complete"
        ),
    )
    args = parser.parse_args(argv)
    try:
        report = audit(args.results_dir, require_complete=args.require_complete)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        ArithmeticError,
        json.JSONDecodeError,
        yaml.YAMLError,
    ) as exc:
        print(f"audit_deft_run: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_text(report)
    if report["status"] == "INVALID":
        return 1
    if args.require_terminal and not report["terminal"]:
        return 1
    if args.require_complete and report["status"] != "COMPLETE":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
