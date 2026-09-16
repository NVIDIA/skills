# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic host-side adapters for the PAS DEFT stages.

TAO work is prepared/finalized by ``run_deft_action.py`` and executed by the
selected platform skill. This command exposes the
canonical Python calls as bounded subcommands so an agent never has to invent
inline Python or rediscover function signatures.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time
from typing import Any

from checkpoint_contract import (
    METADATA_RELPATH,
    checkpoint_lineage_started_ns,
    validate_best_checkpoint,
)
from command_contract import (
    fresh_output_sha256,
    command_sha256,
    expected_container_command,
    expected_hf_forwarding,
    expected_image_kind,
)
from deft_action_contract import platform_evidence_error, remote_freshness_attested


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _state(results: pathlib.Path) -> dict[str, Any]:
    state_path = results / "deft_state.json"
    if not state_path.is_file():
        raise ValueError(f"state file not found: {state_path}")
    payload = json.loads(state_path.read_text())
    if (
        not isinstance(payload, dict)
        or payload.get("workflow") != "tao-run-deft-pas"
        or payload.get("schema_version") != "3"
    ):
        raise ValueError(f"invalid PAS DEFT state: {state_path}")
    if pathlib.Path(str(payload.get("results_dir", ""))).resolve() != results:
        raise ValueError("state.results_dir does not match --results-dir")
    return payload


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


def _config(path: pathlib.Path, results: pathlib.Path):
    import pas_deft
    from pas_deft.config import PasDeftConfig

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"DEFT config not found: {resolved}")
    state = _state(results)
    state_config = state.get("config")
    if not isinstance(state_config, dict):
        raise ValueError("state.config must be an object")
    source = pathlib.Path(__file__).resolve().parent / "pas_deft"
    origin = pathlib.Path(str(getattr(pas_deft, "__file__", ""))).resolve()
    try:
        origin.relative_to(source)
    except ValueError as exc:
        raise ValueError(
            f"imported pas_deft at {origin} is outside bundled runtime {source}"
        ) from exc
    if _python_tree_sha256(source) != state_config.get("pas_deft_bundle_sha256"):
        raise ValueError("bundled PAS runtime changed after initialization")
    expected = pathlib.Path(str(state_config.get("deft_config", ""))).resolve()
    if resolved != expected or expected != results / "config" / "deft_config.yaml":
        raise ValueError(
            f"--deft-config must be the immutable state config {expected}, got {resolved}"
        )
    hashes = state_config.get("spec_sha256")
    expected_digest = hashes.get("deft_config.yaml") if isinstance(hashes, dict) else None
    actual_digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if not expected_digest or actual_digest != expected_digest:
        raise ValueError("approved DEFT config hash does not match state")
    return PasDeftConfig(str(resolved))


def _results(path: pathlib.Path) -> pathlib.Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"results directory not found: {resolved}")
    return resolved


def _iter_dir(results: pathlib.Path, number: int) -> pathlib.Path:
    if number < 1:
        raise ValueError("iteration number must be >= 1")
    maximum = _state(results).get("max_iterations")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or number > maximum:
        raise ValueError(
            f"iteration number {number} is outside the approved range 1..{maximum}"
        )
    return results / f"iter_{number}"


def _require(paths: list[pathlib.Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError("stage did not produce required file(s): " + ", ".join(missing))


def _training_checkpoint(cfg, number: int) -> str:
    return cfg.training_checkpoint_for_iter(number)


def dataset_materialize(args: argparse.Namespace) -> dict[str, Any]:
    from pas_deft.data_mining import (
        convert_clip_image_list_to_parquet,
        materialize_pas_eval_split,
        materialize_pas_pool_split,
    )

    results = _results(args.results_dir)
    cfg = _config(args.deft_config, results)
    splits = results / "pas_splits"
    eval_list = splits / "eval_list.txt"
    eval_pairs = splits / "eval_pairs.json"
    val_list = splits / "val_list.txt"
    pool_list = splits / "aug_pool_list.txt"
    pool_pairs = splits / "aug_pool_pairs.json"
    source_pool = results / "embeddings" / "source" / "source_pool.parquet"

    materialize_pas_eval_split(
        eval_pairs_source_file=cfg.pas.eval_pairs_source_file,
        eval_image_list_file=str(eval_list),
        eval_pairs_file=str(eval_pairs),
        query_types=cfg.pas.query_types,
        val_image_list_file=str(val_list),
        val_sample_size=cfg.pas.val_sample_size,
    )
    materialize_pas_pool_split(
        pool_pairs_source_file=cfg.pas.pool_pairs_source_file,
        aug_pool_image_list_file=str(pool_list),
        aug_pool_pairs_file=str(pool_pairs),
        augmented_suffix=cfg.pas.augmented_suffix,
        query_types=cfg.pas.query_types,
        max_aug_pool_rows=cfg.pas.max_aug_pool_rows,
        mining_pool_mode=cfg.pas.mining_pool_mode,
    )
    convert_clip_image_list_to_parquet(
        image_list_file=str(pool_list),
        image_dir=cfg.pas.source_image_dir,
        output_parquet=str(source_pool),
        caption_dir=cfg.pas.source_caption_dir,
        caption_file_suffix=".txt",
        pairs_file=str(pool_pairs),
    )
    _require([eval_list, eval_pairs, val_list, pool_list, pool_pairs, source_pool])
    return {
        "pas_splits_dir": str(splits),
        "eval_list": str(eval_list),
        "eval_pairs": str(eval_pairs),
        "val_list": str(val_list),
        "pool_list": str(pool_list),
        "pool_pairs": str(pool_pairs),
        "source_pool_parquet": str(source_pool),
    }


def gap_analysis(args: argparse.Namespace) -> dict[str, Any]:
    from pas_deft.analyze_gaps import analyze_clip_inference_gaps
    from pas_deft.utils import resolve_prev_eval_dir

    results = _results(args.results_dir)
    cfg = _config(args.deft_config, results)
    current = _iter_dir(results, args.iter_num)
    gaps = current / "gaps" / "kpi_gaps.parquet"
    previous_eval = resolve_prev_eval_dir(
        base_experiment_path=str(results),
        iter_num=args.iter_num,
        train_ann_path="",
        eval_subdir="evaluate",
    )
    analyze_clip_inference_gaps(
        results_dir=previous_eval,
        gaps_parquet=str(gaps),
        kpi_image_dir=cfg.pas.eval_image_dir,
        logs_dir=str(current / "logs"),
        kpi_caption_dir=cfg.pas.eval_caption_dir,
        caption_file_suffix=".txt",
        kpi_pairs_file=str(results / "pas_splits" / "eval_pairs.json"),
        metric_name=cfg.gap_analysis.metric_name,
        queries_per_slice=cfg.gap_analysis.queries_per_slice,
        min_num_queries=cfg.gap_analysis.min_num_queries,
        query_types=cfg.gap_analysis.query_types,
        weak_attribute_topk=cfg.gap_analysis.weak_attribute_topk,
        target_query_count=cfg.gap_analysis.target_query_count,
        caption_diversity_enabled=(
            "true" if cfg.gap_analysis.caption_diversity.enabled else "false"
        ),
        caption_history_file=cfg.caption_history_file,
        iter_num=args.iter_num,
        total_iters=_state(results)["max_iterations"],
        continual_dataset="true" if cfg.training.continual_dataset else "false",
        caption_history_policy=cfg.gap_analysis.caption_diversity.history_policy,
        caption_coverage_target=cfg.gap_analysis.caption_diversity.coverage_target,
        min_unique_texts_per_attribute=(
            cfg.gap_analysis.caption_diversity.min_unique_texts_per_attribute
        ),
        max_unique_texts_per_attribute=(
            cfg.gap_analysis.caption_diversity.max_unique_texts_per_attribute
        ),
        max_rows_per_unique_text=(
            cfg.gap_analysis.caption_diversity.max_rows_per_unique_text
        ),
        max_rows_per_image_path=(
            cfg.gap_analysis.caption_diversity.max_rows_per_image_path
        ),
        recent_exclude_iters=cfg.gap_analysis.caption_diversity.recent_exclude_iters,
        replay_fraction_when_noncontinual=(
            cfg.gap_analysis.caption_diversity.replay_fraction_when_noncontinual
        ),
    )
    _require([gaps])
    return {"gaps_parquet": str(gaps), "previous_eval_dir": str(previous_eval)}


def mining_postprocess(args: argparse.Namespace) -> dict[str, Any]:
    from pas_deft.data_mining import (
        convert_mined_parquet_to_clip_image_list,
        summarize_knn_mining,
    )

    results = _results(args.results_dir)
    cfg = _config(args.deft_config, results)
    current = _iter_dir(results, args.iter_num)
    target = current / "embeddings" / "target" / "embeddings.parquet"
    mined = current / "mining" / "mined_samples.parquet"
    mining_dir = current / "mining"
    history_enabled = cfg.mining.history_aware.enabled
    candidates = mining_dir / "history_candidates" if history_enabled else mining_dir
    candidate_list = candidates / "mined_image_list.txt"
    candidate_pairs = candidates / "mined_pairs.json"
    candidate_manifest = candidates / "mined_dataset.json"
    source_embeddings = results / "embeddings" / "source" / "embeddings.parquet"

    summarize_knn_mining(
        mined_parquet=str(mined),
        target_parquet=str(target),
        output_dir=str(mining_dir),
        topn=cfg.mining.topn,
    )
    convert_mined_parquet_to_clip_image_list(
        mined_parquet=str(mined),
        image_dir=cfg.pas.source_image_dir,
        caption_dir=cfg.pas.source_caption_dir,
        caption_file_suffix=".txt",
        output_image_list_file=str(candidate_list),
        manifest_path=str(candidate_manifest),
        source_pairs_file=str(results / "pas_splits" / "aug_pool_pairs.json"),
        output_pairs_file=str(candidate_pairs),
        target_query_count=(
            0 if history_enabled else cfg.gap_analysis.target_query_count
        ),
        caption_expansion_enabled=(
            "true" if cfg.mining.caption_expansion.enabled else "false"
        ),
        caption_expansion_mode=cfg.mining.caption_expansion.mode,
        caption_expansion_max_pairs_per_image_path=(
            cfg.mining.caption_expansion.max_pairs_per_image_path
        ),
        caption_expansion_max_expanded_pair_fraction=(
            cfg.mining.caption_expansion.max_expanded_pair_fraction
        ),
        caption_expansion_dedupe_normalized_caption=(
            "true" if cfg.mining.caption_expansion.dedupe_normalized_caption else "false"
        ),
        caption_expansion_count_expanded_pairs_toward_target=(
            cfg.mining.caption_expansion.count_expanded_pairs_toward_target
        ),
        source_embedding_shards_dir=str(results / "embeddings" / "source"),
        write_detailed_csv="false" if history_enabled else "true",
        source_embeddings_parquet=str(source_embeddings),
        target_embeddings_parquet=str(target),
    )
    _require([target, mined, candidate_list, candidate_pairs, candidate_manifest])
    return {
        "target_embeddings_parquet": str(target),
        "mined_parquet": str(mined),
        "candidate_pairs": str(candidate_pairs),
    }


def history_select(args: argparse.Namespace) -> dict[str, Any]:
    from pas_deft.data_mining import track_cumulative_mined_unique_names
    from pas_deft.history_aware_mining import select_history_aware_mined_pairs

    results = _results(args.results_dir)
    cfg = _config(args.deft_config, results)
    mining = _iter_dir(results, args.iter_num) / "mining"
    output_list = mining / "mined_image_list.txt"
    output_pairs = mining / "mined_pairs.json"
    manifest = mining / "mined_dataset.json"
    cumulative = mining / "cumulative_mined_unique_names.json"
    if cfg.mining.history_aware.enabled:
        candidates = mining / "history_candidates"
        select_history_aware_mined_pairs(
            candidate_pairs_file=str(candidates / "mined_pairs.json"),
            candidate_manifest_file=str(candidates / "mined_dataset.json"),
            output_image_list_file=str(output_list),
            output_pairs_file=str(output_pairs),
            manifest_path=str(manifest),
            history_file=str(results / "mining_selection_history.json"),
            source_pool_image_list_file=str(results / "pas_splits" / "aug_pool_list.txt"),
            iter_num=args.iter_num,
            target_query_count=cfg.gap_analysis.target_query_count,
            continual_dataset=(
                "true" if cfg.training.continual_dataset else "false"
            ),
            replay_fraction=cfg.mining.history_aware.replay_fraction,
            resume="true" if args.resume else "false",
        )
    track_cumulative_mined_unique_names(
        mined_pairs_file=str(output_pairs),
        base_experiment_path=str(results),
        iter_num=args.iter_num,
        output_file=str(cumulative),
    )
    _require([output_list, output_pairs, manifest, cumulative])

    eval_names = {
        pathlib.Path(line.strip()).name
        for line in (results / "pas_splits" / "eval_list.txt").read_text().splitlines()
        if line.strip()
    }
    mined_names = {
        pathlib.Path(line.strip()).name
        for line in output_list.read_text().splitlines()
        if line.strip()
    }
    overlap = sorted(eval_names & mined_names)
    if overlap:
        raise ValueError(
            f"eval leakage: {len(overlap)} mined basename(s) overlap eval split; "
            f"first={overlap[0]}"
        )
    return {
        "mined_image_list": str(output_list),
        "mined_pairs": str(output_pairs),
        "mined_manifest": str(manifest),
        "cumulative_names": str(cumulative),
        "history_file": str(results / "mining_selection_history.json"),
    }


def visualize_prepare(args: argparse.Namespace) -> dict[str, Any]:
    from pas_deft.utils import resolve_prev_clip_train_config
    from pas_deft.visualization import (
        export_clip_sample_contact_sheets,
        prepare_clip_images_for_embedding,
        prepare_prev_clip_data_for_embedding,
    )

    results = _results(args.results_dir)
    cfg = _config(args.deft_config, results)
    current = _iter_dir(results, args.iter_num)
    gaps = current / "gaps" / "kpi_gaps.parquet"
    mined = current / "mining" / "mined_samples.parquet"
    samples = current / "visualization" / "samples"
    outputs: dict[str, Any] = {}
    if cfg.visualization.enabled:
        if samples.exists():
            shutil.rmtree(samples)
        export_clip_sample_contact_sheets(
            weak_parquet=str(gaps),
            mined_parquet=str(mined),
            output_dir=str(samples),
            source_pairs_file=str(current / "mining" / "mined_pairs.json"),
            max_samples_per_group=cfg.visualization.max_samples_per_group,
            max_total_samples=cfg.visualization.max_total_samples,
            tile_size=cfg.visualization.tile_size,
            host_path_map={
                "/results": str(results),
                "/data": str(pathlib.Path(cfg.pas.source_image_dir).parent.parent),
            },
        )
        if not samples.is_dir() or not any(samples.iterdir()):
            raise ValueError(f"contact-sheet output is empty: {samples}")
        outputs["samples_dir"] = str(samples)
    if cfg.visualization.embeddings:
        weak_input = current / "embeddings" / "viz_weak" / "input.parquet"
        mined_input = current / "mining" / "mined_unique_images.parquet"
        prepare_clip_images_for_embedding(
            input_parquet=str(gaps),
            output_parquet_path=str(weak_input),
            image_dir=cfg.pas.eval_image_dir,
        )
        prepare_clip_images_for_embedding(
            input_parquet=str(mined),
            output_parquet_path=str(mined_input),
            image_dir=cfg.pas.source_image_dir,
        )
        _require([weak_input, mined_input])
        outputs.update(
            {"weak_input_parquet": str(weak_input), "mined_input_parquet": str(mined_input)}
        )
        has_previous = args.iter_num > 1 or bool(cfg.pas.train_pairs_source_file)
        if cfg.training.continual_dataset and has_previous:
            previous_config = resolve_prev_clip_train_config(
                base_experiment_path=str(results),
                iter_num=args.iter_num,
                continual_dataset=cfg.training.continual_dataset,
                base_template=cfg.experiment.train_config,
                train_image_list_file="",
            )
            previous_pool = current / "embeddings" / "previous" / "prev_pool.parquet"
            prepare_prev_clip_data_for_embedding(
                prev_train_config_yaml=previous_config,
                output_parquet_path=str(previous_pool),
                host_path_map={
                    "/results": str(results),
                    "/data": str(pathlib.Path(cfg.pas.source_image_dir).parent.parent),
                },
            )
            _require([previous_pool])
            outputs["previous_input_parquet"] = str(previous_pool)
    return outputs


def visualize_finish(args: argparse.Namespace) -> dict[str, Any]:
    from pas_deft.visualization import create_tsne_visualization

    results = _results(args.results_dir)
    cfg = _config(args.deft_config, results)
    current = _iter_dir(results, args.iter_num)
    if not cfg.visualization.embeddings:
        return {"visualize_embeddings": False}
    output = current / "visualization" / "tsne_plot.png"
    try:
        output.unlink()
    except FileNotFoundError:
        pass
    create_tsne_visualization(
        weak_embeddings_dir=str(current / "embeddings" / "viz_weak"),
        augmented_embeddings_dir=str(current / "embeddings" / "augmented"),
        previous_embeddings_dir=str(current / "embeddings" / "previous"),
        output_plot_path=str(output),
    )
    _require([output])
    return {"tsne_plot": str(output)}


def train_config(args: argparse.Namespace) -> dict[str, Any]:
    from pas_deft.utils import create_clip_train_config, resolve_prev_clip_train_config

    results = _results(args.results_dir)
    cfg = _config(args.deft_config, results)
    current = _iter_dir(results, args.iter_num)
    previous = resolve_prev_clip_train_config(
        base_experiment_path=str(results),
        iter_num=args.iter_num,
        continual_dataset=cfg.training.continual_dataset,
        base_template=cfg.experiment.train_config,
        train_image_list_file="",
    )
    output = current / "specs" / "train_config.yaml"
    create_clip_train_config(
        base_config_yaml=previous,
        new_config_yaml=str(output),
        output_dir=f"/results/iter_{args.iter_num}",
        checkpoint_path=_training_checkpoint(cfg, args.iter_num),
        sweep_args=cfg.sweep_args_str,
        mined_image_dir=cfg.pas.source_image_dir,
        mined_caption_dir=cfg.pas.source_caption_dir,
        mined_image_list_file=f"/results/iter_{args.iter_num}/mining/mined_image_list.txt",
        caption_file_suffix=".txt",
        train_image_dir=cfg.pas.train_image_dir,
        train_caption_dir=cfg.pas.train_caption_dir,
        train_image_list_file="",
        train_pairs_file="",
        mined_pairs_file=f"/results/iter_{args.iter_num}/mining/mined_pairs.json",
        val_image_list_file="/results/pas_splits/val_list.txt",
        val_image_dir=cfg.pas.eval_image_dir,
        val_caption_dir=cfg.pas.eval_caption_dir,
        continual_dataset=cfg.training.continual_dataset,
    )
    _require([output])
    return {"train_config": str(output), "checkpoint_input": _training_checkpoint(cfg, args.iter_num)}


def publish_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    from pas_deft.utils import get_current_checkpoint, normalize_clip_pretrained_checkpoint

    results = _results(args.results_dir)
    _config(args.deft_config, results)
    state = _state(results)
    state_config = state.get("config")
    if not isinstance(state_config, dict):
        raise ValueError("state.config must be an object")
    current = _iter_dir(results, args.iter_num)
    train_dir = current / "train"
    command_status = pathlib.Path(os.path.abspath(args.train_command_status.expanduser()))
    expected_status = train_dir / "train.status.json"
    if command_status != expected_status:
        raise ValueError(
            f"--train-command-status must be {expected_status}, got {command_status}"
        )
    if (
        not command_status.is_file()
        or command_status.stat().st_size == 0
        or command_status.is_symlink()
        or command_status.resolve() != command_status
    ):
        raise ValueError(f"train command status is missing or unsafe: {command_status}")
    payload = json.loads(command_status.read_text())
    expected_command = expected_container_command(
        "train", f"iter{args.iter_num}", state_config
    )
    attempt = payload.get("attempt") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") not in {"1", "2"}
        or payload.get("workflow") != "tao-run-deft-pas"
        or platform_evidence_error(payload, str(state_config.get("platform"))) is not None
        or payload.get("name") != "train"
        or payload.get("status") != "ok"
        or payload.get("exit_code") != 0
        or payload.get("image_kind") != expected_image_kind("train")
        or payload.get("image") != state_config.get("pyt_image")
        or payload.get("command") != expected_command
        or payload.get("command_sha256") != command_sha256(expected_command)
        or payload.get("passed_hf_token")
        is not expected_hf_forwarding("train", state_config)
        or not isinstance(attempt, int)
        or isinstance(attempt, bool)
        or not 1 <= attempt <= 2
        or not isinstance(payload.get("finished_at"), str)
        or not payload.get("finished_at", "").strip()
    ):
        raise ValueError("--train-command-status does not prove the approved train command")
    started_ns = payload.get("started_ns")
    lineage_started_ns = checkpoint_lineage_started_ns(
        payload, state.get("started_at")
    )
    log_path = pathlib.Path(str(payload.get("log_path", "")))
    if (
        not log_path.is_absolute()
        or not log_path.is_file()
        or log_path.stat().st_size == 0
        or log_path.is_symlink()
        or log_path.resolve() != log_path
    ):
        raise ValueError("train command status log_path is missing or unsafe")
    try:
        log_path.relative_to(train_dir)
    except ValueError as exc:
        raise ValueError("train command status log_path must be inside train/") from exc
    train_tao_status = train_dir / "status.json"
    fresh = {
        str(pathlib.Path(str(item)).resolve())
        for item in payload.get("fresh_outputs", [])
        if isinstance(item, str)
    }
    if str(train_tao_status.resolve()) not in fresh:
        raise ValueError("train command status does not bind train/status.json")
    if (
        not train_tao_status.is_file()
        or train_tao_status.stat().st_size == 0
        or train_tao_status.is_symlink()
        or train_tao_status.resolve() != train_tao_status
        or (
            train_tao_status.stat().st_mtime_ns < started_ns
            and not remote_freshness_attested(payload)
        )
    ):
        raise ValueError("TAO train status is missing, unsafe, empty, or stale")
    if "Train finished successfully." not in train_tao_status.read_text(errors="replace"):
        raise ValueError("TAO train status lacks 'Train finished successfully.'")

    best = train_dir / "best" / "clip_best_val_t2i_mAP.pth"
    metadata = train_dir / METADATA_RELPATH
    normalized = current / "pretrained" / "model_state.pth"
    for directory in (train_dir, best.parent, normalized.parent):
        if directory.resolve() != directory:
            raise ValueError(
                f"checkpoint output directory must not traverse a symlink: {directory}"
            )
    for output in (best, metadata, normalized):
        try:
            output.unlink()
        except FileNotFoundError:
            pass
    published = pathlib.Path(
        get_current_checkpoint(
            str(train_dir), earliest_mtime_ns=lineage_started_ns
        )
    )
    if pathlib.Path(os.path.abspath(published)) != best:
        raise ValueError(f"checkpoint publisher returned {published}, expected {best}")
    provenance = validate_best_checkpoint(
        best, train_dir, started_ns=lineage_started_ns
    )
    normalize_clip_pretrained_checkpoint(str(best), str(normalized))
    _require([best, metadata, normalized])
    if normalized.is_symlink() or normalized.resolve() != normalized:
        raise ValueError(f"normalized checkpoint must not traverse a symlink: {normalized}")
    return {
        "best_ckpt": str(best),
        "best_ckpt_metadata": provenance["best_ckpt_metadata"],
        "pretrained_state": str(normalized),
    }


def eval_config(args: argparse.Namespace) -> dict[str, Any]:
    from pas_deft.utils import create_clip_eval_config

    results = _results(args.results_dir)
    cfg = _config(args.deft_config, results)
    if args.iter_label == "baseline":
        host_dir = results / "zs"
        container_dir = "/results/zs"
        checkpoint = cfg.training.init_checkpoint
    else:
        number = int(args.iter_label[4:])
        host_dir = _iter_dir(results, number)
        container_dir = f"/results/iter_{number}"
        checkpoint = f"{container_dir}/train/best/clip_best_val_t2i_mAP.pth"
    output = host_dir / "specs" / "eval_config.yaml"
    create_clip_eval_config(
        base_config_yaml=cfg.experiment.eval_config,
        new_config_yaml=str(output),
        output_dir=container_dir,
        checkpoint_path=checkpoint,
        eval_image_dir=cfg.pas.eval_image_dir,
        eval_caption_dir=cfg.pas.eval_caption_dir,
        eval_image_list_file="/results/pas_splits/eval_list.txt",
        caption_file_suffix=".txt",
    )
    _require([output])
    return {"eval_config": str(output), "checkpoint": checkpoint}


def iteration_summary(args: argparse.Namespace) -> dict[str, Any]:
    from pas_deft.data_mining import write_iteration_summary

    results = _results(args.results_dir)
    cfg = _config(args.deft_config, results)
    current = _iter_dir(results, args.iter_num)
    output = pathlib.Path(
        write_iteration_summary(
            experiment_dir=str(current),
            iter_num=args.iter_num,
            gaps_parquet=str(current / "gaps" / "kpi_gaps.parquet"),
            mined_parquet=str(current / "mining" / "mined_samples.parquet"),
            mined_pairs_file=str(current / "mining" / "mined_pairs.json"),
            training_checkpoint=_training_checkpoint(cfg, args.iter_num),
            next_checkpoint_path=f"/results/iter_{args.iter_num}/train/best/clip_best_val_t2i_mAP.pth",
        )
    ).resolve()
    _require([output])
    return {"iteration_summary": str(output)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    def common(name: str, *, iteration: bool = False) -> argparse.ArgumentParser:
        child = sub.add_parser(name)
        child.add_argument("--results-dir", required=True, type=pathlib.Path)
        child.add_argument("--deft-config", required=True, type=pathlib.Path)
        if iteration:
            child.add_argument("--iter-num", required=True, type=int)
        return child

    common("dataset-materialize")
    common("gap-analysis", iteration=True)
    common("mining-postprocess", iteration=True)
    history = common("history-select", iteration=True)
    history.add_argument("--resume", action="store_true")
    common("visualize-prepare", iteration=True)
    common("visualize-finish", iteration=True)
    common("train-config", iteration=True)
    publish = common("publish-checkpoint", iteration=True)
    publish.add_argument(
        "--train-command-status", required=True, type=pathlib.Path
    )
    evaluate = common("eval-config")
    evaluate.add_argument("--iter-label", required=True)
    common("iteration-summary", iteration=True)
    return parser


HANDLERS = {
    "dataset-materialize": dataset_materialize,
    "gap-analysis": gap_analysis,
    "mining-postprocess": mining_postprocess,
    "history-select": history_select,
    "visualize-prepare": visualize_prepare,
    "visualize-finish": visualize_finish,
    "train-config": train_config,
    "publish-checkpoint": publish_checkpoint,
    "eval-config": eval_config,
    "iteration-summary": iteration_summary,
}

_STAGE_OUTPUT_KEYS = {
    "dataset-materialize": {
        "eval_list",
        "eval_pairs",
        "val_list",
        "pool_list",
        "pool_pairs",
        "source_pool_parquet",
    },
    "gap-analysis": {"gaps_parquet"},
    "mining-postprocess": {"candidate_pairs"},
    "history-select": {
        "mined_image_list",
        "mined_pairs",
        "mined_manifest",
        "cumulative_names",
        "history_file",
    },
    "visualize-prepare": {
        "samples_dir",
        "weak_input_parquet",
        "mined_input_parquet",
        "previous_input_parquet",
    },
    "visualize-finish": {"tsne_plot"},
    "train-config": {"train_config"},
    "publish-checkpoint": {
        "best_ckpt",
        "best_ckpt_metadata",
        "pretrained_state",
    },
    "eval-config": {"eval_config"},
    "iteration-summary": {"iteration_summary"},
}


def _status_directory(args: argparse.Namespace, results: pathlib.Path) -> pathlib.Path:
    if args.stage == "dataset-materialize":
        return results / "dataset_setup"
    if args.stage == "eval-config":
        return (
            results / "zs" / "specs"
            if args.iter_label == "baseline"
            else results / f"iter_{int(args.iter_label[4:])}" / "specs"
        )
    number = getattr(args, "iter_num", None)
    if not isinstance(number, int):
        raise ValueError(f"cannot derive host status directory for {args.stage}")
    current = _iter_dir(results, number)
    if args.stage == "gap-analysis":
        return current / "gaps"
    if args.stage in {"mining-postprocess", "history-select"}:
        return current / "mining"
    if args.stage in {"visualize-prepare", "visualize-finish"}:
        return current / "visualization"
    if args.stage == "train-config":
        return current / "specs"
    if args.stage == "publish-checkpoint":
        return current / "train"
    if args.stage == "iteration-summary":
        return current
    raise ValueError(f"cannot derive host status directory for {args.stage}")


def _result_paths(
    report: dict[str, Any], results: pathlib.Path, stage: str
) -> list[str]:
    paths: list[str] = []
    output_keys = _STAGE_OUTPUT_KEYS.get(stage)
    if output_keys is None:
        raise ValueError(f"no output contract for host stage {stage}")
    for key in output_keys:
        value = report.get(key)
        if not isinstance(value, str):
            continue
        candidate = pathlib.Path(value).expanduser()
        if not candidate.is_absolute():
            continue
        absolute = pathlib.Path(os.path.abspath(candidate))
        resolved = absolute.resolve()
        try:
            resolved.relative_to(results)
        except ValueError:
            continue
        if absolute.exists():
            # Preserve the lexical canonical output (not a symlink target) in
            # evidence while using the resolved path only for containment.
            paths.append(str(absolute))
    return sorted(set(paths))


def _execute_stage(
    args: argparse.Namespace, results: pathlib.Path, status_dir: pathlib.Path
) -> int:
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / f"{args.stage}.host.status.json"
    log_path = status_dir / f"{args.stage}.host.log"
    lock_path = status_dir / f"{args.stage}.host.launch.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                f"run_pas_stage[{args.stage}]: another process owns {lock_path}",
                file=sys.stderr,
            )
            return 2

        prior_attempt = 0
        if status_path.exists():
            try:
                existing = json.loads(status_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                print(
                    f"run_pas_stage[{args.stage}]: existing status is unreadable: {exc}",
                    file=sys.stderr,
                )
                return 2
            if not isinstance(existing, dict) or existing.get("name") != args.stage:
                print(
                    f"run_pas_stage[{args.stage}]: existing status has the wrong identity",
                    file=sys.stderr,
                )
                return 2
            raw_attempt = existing.get("attempt", 1)
            if (
                not isinstance(raw_attempt, int)
                or isinstance(raw_attempt, bool)
                or raw_attempt < 1
            ):
                print(
                    f"run_pas_stage[{args.stage}]: existing status has invalid attempt",
                    file=sys.stderr,
                )
                return 2
            prior_attempt = raw_attempt
            # Acquiring the process-held flock proves no earlier adapter still
            # owns this stable stage, even if its last status remained running.
            if prior_attempt >= 2:
                print(
                    f"run_pas_stage[{args.stage}]: attempt budget exhausted "
                    f"(attempt={prior_attempt}); hard-stop instead of retrying",
                    file=sys.stderr,
                )
                return 2

        started_ns = time.time_ns()
        started_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        base_status = {
            "schema_version": "1",
            "workflow": "tao-run-deft-pas",
            "kind": "host",
            "name": args.stage,
            "attempt": prior_attempt + 1,
            "pid": os.getpid(),
            "resume": bool(
                args.stage == "history-select" and getattr(args, "resume", False)
            ),
            "started_at": started_at,
            "started_ns": started_ns,
            "log_path": str(log_path),
            "fresh_outputs": [],
        }
        _atomic_json(
            status_path,
            {
                **base_status,
                "finished_at": None,
                "status": "running",
                "exit_code": None,
            },
        )
        try:
            with (
                log_path.open("w") as log,
                contextlib.redirect_stdout(log),
                contextlib.redirect_stderr(log),
            ):
                report = HANDLERS[args.stage](args)
                log.write(json.dumps(report, sort_keys=True) + "\n")
        except Exception as exc:  # adapters convert operational failures to evidence
            finished_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            with log_path.open("a") as log:
                log.write(f"{type(exc).__name__}: {exc}\n")
            _atomic_json(
                status_path,
                {
                    **base_status,
                    "finished_at": finished_at,
                    "status": "error",
                    "exit_code": 2,
                },
            )
            print(f"run_pas_stage[{args.stage}]: {exc}", file=sys.stderr)
            return 2
        fresh_outputs = _result_paths(report, results, args.stage)
        output_hashes = fresh_output_sha256(fresh_outputs)
        finished_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        _atomic_json(
            status_path,
            {
                **base_status,
                "finished_at": finished_at,
                "status": "ok",
                "exit_code": 0,
                "fresh_outputs": fresh_outputs,
                "fresh_output_sha256": output_hashes,
            },
        )
        report = dict(report)
        report["host_status"] = str(status_path)
        report["host_log"] = str(log_path)
        print(json.dumps(report, sort_keys=True))
        return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stage == "eval-config" and args.iter_label != "baseline":
        if not args.iter_label.startswith("iter") or not args.iter_label[4:].isdigit() or int(args.iter_label[4:]) < 1:
            print("run_pas_stage: --iter-label must be baseline or iterN", file=sys.stderr)
            return 2
    try:
        results = _results(args.results_dir)
        status_dir = _status_directory(args, results)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"run_pas_stage[{args.stage}]: {exc}", file=sys.stderr)
        return 2
    return _execute_stage(args, results, status_dir)


if __name__ == "__main__":
    raise SystemExit(main())
