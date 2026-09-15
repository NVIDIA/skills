#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workstation finetune runner — composes prepare_data + check_checkpoint outputs
into a `main.py finetune` invocation (which calls task/cross_validate.py).

The runner is responsible for:
  - reading the prepare_data.json manifest produced by prepare_data.py --mode finetune
  - validating the input ckpt is a pretrain checkpoint (via check_checkpoint.py
    --mode finetune_init) and extracting its encoder arch
  - merging hyperparameters from defaults_finetune.json with per-flag CLI overrides
  - building the final main.py finetune argv (data + features + arch + training)
  - emitting a reproducible run.json manifest (source-repo commit, image digest,
    cmd_replay, args_applied with per-flag source attribution)
  - executing the finetune (or writing the manifest only, with --dry-run)

Arch params come exclusively from the ckpt's saved_args (via check_checkpoint.py).
User-supplied arch flags are not exposed; the runner refuses to override what
the ckpt dictates so the FFN heads attach to a consistent encoder.

Single-GPU is the default for finetune. `--gpus N` picks a specific device id
(defaults to GPU 0). For data-parallel multi-GPU finetuning, pass `--num-gpus N`
(N>1): the runner sets `WORLD_SIZE=N` and `main.py finetune` spawns one process
per GPU, and `--batch-size` is interpreted per-GPU.

CLI
---
    run_finetune_local.py
        --ckpt <path>                  # input pretrain ckpt (required)
        --prepare-manifest <path>      # prepare_data.json (mode=finetune)
        --out <run-dir>                # output dir
        --dataset-type {regression|classification|multiclass}   # required
        [--ckpt-validator-out <path>]  # cached check_checkpoint.py JSON
        [--gpus 0]                     # single GPU id (default 0)
        [--dry-run]                    # write run.json + print command, do not execute
        [--epochs N] [--batch-size N] [--init-lr F] [--max-lr F] [--final-lr F]
        [--warmup-epochs F] [--weight-decay F] [--dropout F]
        [--metric NAME] [--seed N]
        [--ensemble-size N] [--num-folds N]
        [--split-sizes TRAIN VAL TEST]
        [--ffn-hidden-size N] [--ffn-num-layers N]
        [--ffn-num-task-specific-layers N] [--ffn-task-specific-hidden-size N]
        [--early-stop-epoch N] [--dist-coff F] [--bond-drop-rate F]
        [--show-individual-scores]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# sys.path tweak so `_utils` is importable whether launched via kermt_run
# (PYTHONPATH=/workspace) or bare-Python from the host.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import (  # noqa: E402
    resolve_kermt_repo, assert_prepare_manifest_basics, docker_image_digest, format_cmd_replay,
    git_commit_with_env_override, load_json, merge_default_into_applied,
    resolve_single_gpu, run_checkpoint_validator, runner_environment,
)


REPO_ROOT = resolve_kermt_repo()
SKILL_ROOT = Path(__file__).resolve().parent.parent
DEFAULTS_PATH = SKILL_ROOT / "config" / "defaults_finetune.json"
CHECK_CHECKPOINT_PATH = SKILL_ROOT / "scripts" / "check_checkpoint.py"
MAIN_PY_PATH = REPO_ROOT / "main.py"

# Architecture fields the runner pulls from the ckpt and refuses to let the
# user override. Mirrors the pretrain runner's ARCH_FLAGS_FROM_CKPT but adds
# the self-attention triple since finetune-time FFN attaches above the
# self-attention layer if the encoder was pretrained with it.
ARCH_FLAGS_FROM_CKPT = (
    "hidden_size", "depth", "num_attn_head", "activation", "backbone",
    "embedding_output_type", "self_attention", "attn_hidden", "attn_out",
)

# Hyperparameter flag groups + their JSON path in defaults_finetune.json.
# Flags listed here are *eligible* for default-config lookup; whether a key
# exists in defaults_finetune.json is independent. Anything in this tuple that
# the user passes via CLI ends up in args_applied with source="user"; anything
# not passed but present in defaults ends up source="default-config". Anything
# in this tuple that's neither passed nor in defaults is simply absent from
# args_applied (so the argv-builder skips it and main.py finetune's own
# argparse default takes effect).
TRAINING_FLAGS = (
    "epochs", "batch_size", "init_lr", "max_lr", "final_lr",
    "dropout", "bond_drop_rate", "dist_coff", "seed", "tensorboard",
    "warmup_epochs", "weight_decay", "early_stop_epoch",
)
TASK_FLAGS = (
    "dataset_type", "metric", "split_type", "ensemble_size", "num_folds",
    "no_features_scaling", "show_individual_scores",
)
FFN_FLAGS = (
    "ffn_hidden_size", "ffn_num_layers",
    "ffn_num_task_specific_layers", "ffn_task_specific_hidden_size",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verify_prepare_manifest(manifest: dict[str, Any]) -> None:
    assert_prepare_manifest_basics(manifest, "finetune")
    out = manifest.get("outputs", {})
    method = manifest.get("split_method")
    if method == "deferred_to_runner":
        if "clean_full_csv" not in out:
            raise ValueError(
                "split_method=deferred_to_runner but the manifest doesn't include "
                "clean_full_csv. Was prepare_data run with --skip-features but no clean step?"
            )
    elif method in ("user_provided", "random"):
        missing = [k for k in ("clean_train_csv", "clean_val_csv", "clean_test_csv") if k not in out]
        if missing:
            raise ValueError(
                f"prepare_data manifest (split_method={method}) is missing required outputs: "
                f"{missing}. Rerun prepare_data without --skip-features."
            )
    else:
        raise ValueError(
            f"prepare_data manifest has unrecognized split_method='{method}'. "
            "Expected one of: user_provided, random, deferred_to_runner."
        )


def _arch_from_validator(validator_out: dict[str, Any]) -> dict[str, Any]:
    arch = validator_out.get("arch") or {}
    # Most fields are required; self_attention/attn_hidden/attn_out are only
    # required when self_attention is True. The ckpt's arch dict will have
    # self_attention=False with the attn_* fields as None in the common case.
    required_always = ("hidden_size", "depth", "num_attn_head", "activation",
                       "backbone", "embedding_output_type", "self_attention")
    missing = [k for k in required_always if arch.get(k) is None]
    if missing:
        raise ValueError(
            f"checkpoint validator did not surface required arch fields: {missing}. "
            "The ckpt must have saved_args with these set (the standard "
            "save_model_for_restart format)."
        )
    if arch.get("self_attention"):
        attn_missing = [k for k in ("attn_hidden", "attn_out") if arch.get(k) is None]
        if attn_missing:
            raise ValueError(
                f"ckpt has self_attention=True but arch is missing {attn_missing}; "
                "the ckpt's saved_args must include them."
            )
    return arch


def _apply_defaults(args: argparse.Namespace, defaults: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Merges defaults_finetune.json + CLI overrides into args_applied, keyed by
    flag name (snake_case) with {value, source} entries. Source is 'user' if the
    user supplied the flag on the CLI, else 'default-config'."""
    applied: dict[str, dict[str, Any]] = {}
    training = defaults.get("training", {})
    task_cfg = defaults.get("task", {})
    ffn = defaults.get("ffn_head", {})

    for f in TRAINING_FLAGS:
        merge_default_into_applied(applied, args, f, training)
    for f in TASK_FLAGS:
        merge_default_into_applied(applied, args, f, task_cfg)
    for f in FFN_FLAGS:
        merge_default_into_applied(applied, args, f, ffn)

    return applied


def _validate_mtl_consistency(applied: dict[str, dict[str, Any]]) -> None:
    """If ffn_num_task_specific_layers > 0, ffn_task_specific_hidden_size must be set."""
    n_layers = applied.get("ffn_num_task_specific_layers", {}).get("value", 0) or 0
    hidden = applied.get("ffn_task_specific_hidden_size", {}).get("value")
    if n_layers > 0 and not hidden:
        raise ValueError(
            f"ffn_num_task_specific_layers={n_layers} but ffn_task_specific_hidden_size is "
            "unset. When MTL heads are enabled, the hidden size must be supplied — pass "
            "--ffn-task-specific-hidden-size H (or set it in defaults_finetune.json)."
        )


def _build_argv(
    *, gpu: int, out_dir: Path, manifest: dict[str, Any], ckpt: Path,
    arch: dict[str, Any], applied: dict[str, dict[str, Any]],
    num_gpus: int = 1,
) -> list[str]:
    """Constructs the full finetune argv as a list of strings.

    Single-process and multi-GPU (DDP) finetune share one entrypoint,
    `main.py finetune ...`. DDP is selected at runtime by the WORLD_SIZE env var
    (this runner sets it to num_gpus when num_gpus > 1; main.py then spawns one
    process per GPU). The argv is therefore identical for both; num_gpus is kept
    only to document that --batch_size is per-GPU under DDP.
    """
    outputs = manifest["outputs"]
    method = manifest["split_method"]

    argv: list[str] = [sys.executable, "-u", str(MAIN_PY_PATH), "finetune"]

    # Data + features paths
    if method == "deferred_to_runner":
        argv += ["--data_path", outputs["clean_full_csv"]]
        if "clean_full_npz" in outputs:
            argv += ["--features_path", outputs["clean_full_npz"]]
        # Split is done inside task/train.py via args.split_type/split_sizes/seed.
        split_type = manifest.get("split_type") or applied.get("split_type", {}).get("value", "random")
        argv += ["--split_type", str(split_type)]
        if manifest.get("split_fractions"):
            sf = manifest["split_fractions"]
            argv += ["--split_sizes", str(sf["train"]), str(sf["val"]), str(sf["test"])]
    else:
        # user_provided or random: prep produced ready-made train/val/test CSVs + npz.
        argv += ["--data_path", outputs["clean_train_csv"]]
        argv += ["--separate_val_path",  outputs["clean_val_csv"]]
        argv += ["--separate_test_path", outputs["clean_test_csv"]]
        if "clean_train_npz" in outputs:
            argv += ["--features_path", outputs["clean_train_npz"]]
        if "clean_val_npz" in outputs:
            argv += ["--separate_val_features_path", outputs["clean_val_npz"]]
        if "clean_test_npz" in outputs:
            argv += ["--separate_test_features_path", outputs["clean_test_npz"]]
        # task/train.py with separate_val + separate_test paths won't re-split,
        # but split_type is still required by argparse — pass the manifest-or-default value.
        split_type_val = applied.get("split_type", {}).get("value", "random")
        argv += ["--split_type", str(split_type_val)]

    # Pretrained encoder weights — task/train.py loads them into the model
    # via args.checkpoint_paths and then attaches the new FFN head.
    argv += ["--checkpoint_path", str(ckpt)]

    # Task semantics
    if "dataset_type" in applied:
        argv += ["--dataset_type", str(applied["dataset_type"]["value"])]
    if "metric" in applied:
        argv += ["--metric", str(applied["metric"]["value"])]
    if applied.get("no_features_scaling", {}).get("value"):
        argv += ["--no_features_scaling"]
    if "ensemble_size" in applied:
        argv += ["--ensemble_size", str(applied["ensemble_size"]["value"])]
    if "num_folds" in applied:
        argv += ["--num_folds", str(applied["num_folds"]["value"])]

    # Architecture — sourced from the ckpt's validator output.
    argv += [
        "--hidden_size",  str(arch["hidden_size"]),
        "--depth",        str(arch["depth"]),
        "--num_attn_head", str(arch["num_attn_head"]),
        "--activation",   str(arch["activation"]),
        "--embedding_output_type", str(arch["embedding_output_type"]),
    ]
    if arch.get("self_attention"):
        argv += ["--self_attention",
                 "--attn_hidden", str(arch["attn_hidden"]),
                 "--attn_out",    str(arch["attn_out"])]

    # FFN head
    if "ffn_hidden_size" in applied:
        argv += ["--ffn_hidden_size", str(applied["ffn_hidden_size"]["value"])]
    if "ffn_num_layers" in applied:
        argv += ["--ffn_num_layers", str(applied["ffn_num_layers"]["value"])]
    if applied.get("ffn_num_task_specific_layers", {}).get("value", 0):
        argv += ["--ffn_num_task_specific_layers", str(applied["ffn_num_task_specific_layers"]["value"])]
        argv += ["--ffn_task_specific_hidden_size", str(applied["ffn_task_specific_hidden_size"]["value"])]

    # Training schedule
    for name in ("epochs", "batch_size", "init_lr", "max_lr", "final_lr",
                 "dropout", "bond_drop_rate", "dist_coff", "seed",
                 "weight_decay", "warmup_epochs", "early_stop_epoch"):
        if name in applied:
            argv += [f"--{name}", str(applied[name]["value"])]

    if applied.get("tensorboard", {}).get("value"):
        argv += ["--tensorboard"]
    if applied.get("show_individual_scores", {}).get("value"):
        argv += ["--show_individual_scores"]

    # GPU + save_dir. Under DDP (num_gpus>1) gpu is None: main.py finetune pins
    # each rank to its own device, and run_training ignores args.gpu when
    # distributed, so no --gpu flag is passed.
    if gpu is not None:
        argv += ["--gpu", str(gpu)]
    argv += ["--save_dir", str(out_dir / "ckpt")]

    return argv


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    # 1. Load defaults + prepare manifest.
    defaults = load_json(DEFAULTS_PATH, name="defaults_finetune.json")
    prep_manifest_path = Path(args.prepare_manifest).resolve()
    manifest = load_json(prep_manifest_path, name="prepare_data.json")
    _verify_prepare_manifest(manifest)

    # 2. Validate ckpt + extract arch.
    ckpt = Path(args.ckpt).resolve()
    if args.ckpt_validator_out:
        validator_out = load_json(Path(args.ckpt_validator_out), name="ckpt validator output")
    else:
        validator_out = run_checkpoint_validator(ckpt, mode="finetune_init", script_path=CHECK_CHECKPOINT_PATH)
    if not validator_out.get("ok"):
        raise ValueError(
            f"check_checkpoint.py rejected the input ckpt: {validator_out.get('errors')}"
        )
    arch = _arch_from_validator(validator_out)
    model_type = validator_out.get("model_type")

    # 3. GPU selection.
    num_gpus = max(1, int(getattr(args, "num_gpus", 1) or 1))
    distributed = num_gpus > 1
    if distributed:
        # Data-parallel DDP: main.py finetune uses GPUs 0..num_gpus-1 (one rank
        # each) via WORLD_SIZE; a single-device pin does not apply.
        gpu = None
    else:
        gpu = resolve_single_gpu(args.gpus, workflow="finetune")

    # 4. Apply defaults + collect args_applied.
    applied = _apply_defaults(args, defaults)
    # MTL FFN consistency
    _validate_mtl_consistency(applied)

    # 5. Build the finetune argv (always main.py finetune; DDP selected via WORLD_SIZE).
    argv = _build_argv(
        gpu=gpu, out_dir=out_dir, manifest=manifest, ckpt=ckpt,
        arch=arch, applied=applied, num_gpus=num_gpus,
    )

    # 6. Build the run.json manifest.
    commit, dirty = git_commit_with_env_override(REPO_ROOT)
    image_tag = os.environ.get("KERMT_IMAGE", "kermt:latest")
    image_digest = docker_image_digest(image_tag)
    replay_env = {"WORLD_SIZE": str(num_gpus)} if distributed else {"CUDA_VISIBLE_DEVICES": gpu}
    cmd_replay = format_cmd_replay(argv, env=replay_env)
    targets = manifest.get("targets")
    run_manifest: dict[str, Any] = {
        "workflow": "finetune",
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "container": {"image_tag": image_tag, "image_digest": image_digest},
        "repo": {"commit": commit, "dirty": dirty},
        "inputs": {
            "ckpt": str(ckpt),
            "prepare_data_manifest": str(prep_manifest_path),
            "ckpt_validator_out": (
                str(Path(args.ckpt_validator_out).resolve()) if args.ckpt_validator_out else None
            ),
            "targets": list(targets) if targets else None,
        },
        "model_type": model_type,
        "gpu": gpu,
        "num_gpus": num_gpus,
        "distributed": distributed,
        "args_applied": applied,
        "arch": arch,
        "save_dir": str(out_dir / "ckpt"),
        "logs_dir": str(out_dir / "logs"),
        "tensorboard_dir": str(out_dir / "logs" / "tb"),
        "argv": argv,
        "cmd_replay": cmd_replay,
        "ok_to_replay": (not dirty) and (commit != "unknown"),
        "dry_run": bool(args.dry_run),
    }
    (out_dir / "run.json").write_text(json.dumps(run_manifest, indent=2))

    # 7. Execute (unless --dry-run).
    if args.dry_run:
        run_manifest["status"] = "dry_run"
        return run_manifest

    env = runner_environment(REPO_ROOT)
    if distributed:
        # DDP: main.py finetune reads WORLD_SIZE and spawns one process per GPU.
        # Do not pin CUDA_VISIBLE_DEVICES to a single device.
        env["WORLD_SIZE"] = str(num_gpus)
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # main.py enables strict deterministic algorithms via
    # `torch.use_deterministic_algorithms(True)`; CuBLAS then requires this env var.
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    log_file = out_dir / "logs" / "finetune.log"
    with log_file.open("w") as logf:
        proc = subprocess.run(argv, env=env, stdout=logf, stderr=subprocess.STDOUT)
    run_manifest["exit_code"] = proc.returncode
    run_manifest["status"] = "ok" if proc.returncode == 0 else "failed"
    (out_dir / "run.json").write_text(json.dumps(run_manifest, indent=2))
    return run_manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Workstation finetune runner — wraps main.py finetune via subprocess."
    )
    p.add_argument("--ckpt", required=True,
                   help="Path to the input pretrain checkpoint (grover_base / cmim / hybrid).")
    p.add_argument("--prepare-manifest", required=True,
                   help="Path to a prepare_data.json produced with --mode finetune.")
    p.add_argument("--out", required=True, help="Output run directory.")
    p.add_argument("--ckpt-validator-out", default=None,
                   help="Optional cached check_checkpoint.py JSON; computed if absent.")
    p.add_argument("--gpus", default=None,
                   help="Single GPU id for single-process finetune (default 0). "
                        "Ignored when --num-gpus > 1 (DDP uses ranks 0..N-1).")
    p.add_argument("--num-gpus", type=int, default=1,
                   help="Number of GPUs for data-parallel (DDP) finetune. Default 1 "
                        "(single-process main.py finetune, unchanged). N>1 runs "
                        "main.py finetune with WORLD_SIZE=N (one process per GPU); "
                        "--batch-size is per-GPU.")
    p.add_argument("--dry-run", action="store_true",
                   help="Write run.json + print the command without executing.")

    # Task semantics — dataset_type is required (modify_train_args asserts it).
    p.add_argument("--dataset-type", default=None,
                   choices=["regression", "classification", "multiclass"])
    p.add_argument("--metric", default=None)
    p.add_argument("--ensemble-size", type=int, default=None)
    p.add_argument("--num-folds", type=int, default=None)
    p.add_argument("--show-individual-scores", action="store_true", default=None)

    # Training overrides — defaults None so we can distinguish user vs default-config source.
    for f, t in [("epochs", int), ("batch-size", int), ("init-lr", float), ("max-lr", float),
                 ("final-lr", float), ("warmup-epochs", float), ("weight-decay", float),
                 ("dropout", float), ("bond-drop-rate", float), ("dist-coff", float),
                 ("early-stop-epoch", int), ("seed", int)]:
        p.add_argument(f"--{f}", type=t, default=None)

    # FFN head overrides
    p.add_argument("--ffn-hidden-size", type=int, default=None)
    p.add_argument("--ffn-num-layers", type=int, default=None)
    p.add_argument("--ffn-num-task-specific-layers", type=int, default=None)
    p.add_argument("--ffn-task-specific-hidden-size", type=int, default=None)

    args = p.parse_args(argv)

    try:
        manifest = run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(json.dumps({"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}, indent=2))
        return 1
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        print(json.dumps({"ok": False, "errors": [f"unhandled: {type(exc).__name__}: {exc}"]},
                         indent=2))
        return 1

    print(json.dumps({"ok": True, "manifest": manifest}, indent=2))
    return 0 if manifest.get("status") != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
