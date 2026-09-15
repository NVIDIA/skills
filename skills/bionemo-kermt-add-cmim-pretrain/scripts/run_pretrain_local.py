#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workstation pretrain runner — composes prepare_data + ckpt-validator outputs
into a pretrain_ddp.py invocation.

Continues pretraining from a user-provided checkpoint. The model type
(grover_base / cmim / hybrid) is inferred from the validator's output and
drives the pretrain_ddp.py flag set; arch params come exclusively from the
ckpt; training/loss hyperparameters come from config/defaults_pretrain.json
with per-flag CLI overrides.

How it interacts with pretrain_ddp.py's auto-resume:
    pretrain_ddp.py looks at <save_dir>/last_checkpoint.pt and resumes from it
    if present. The runner sets `--save_dir <out>/ckpt` and symlinks the user's
    input ckpt to <out>/ckpt/last_checkpoint.pt so the resume path picks it up.

Run.json manifest:
    Records source-repo commit + image digest + a copy-pasteable `cmd_replay`
    + per-flag `args_applied` so the artifact is self-contained and replayable.

CLI
---
    run_pretrain_local.py
        --ckpt <path>                  # input pretrain ckpt (required)
        --prepare-manifest <path>      # prepare_data.json from a prior prepare run
        --out <run-dir>                # output dir (typically runs/continue-pretrain_<ts>/)
        [--ckpt-validator-out <path>]  # cached check_checkpoint.py JSON; computed if absent
        [--gpus 0,2]                   # subset of detected GPUs; default = all visible
        [--dry-run]                    # write run.json + print command, do not execute
        [--epochs N] [--batch-size N] [--init-lr F] [--max-lr F] [--final-lr F]
        [--warmup-epochs F] [--weight-decay F] [--dropout F]
        [--save-interval N] [--seed N]
        [--vocab-loss-weight F]            # hybrid only
        [--latent-dim N] [--contrastive-temperature F]   # cmim/hybrid only
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

# Add the scripts/ dir to sys.path so `_utils` is importable whether
# this script is launched via `kermt_run` (PYTHONPATH=/workspace) or as a
# bare `python scripts/run_pretrain_local.py …` from the host.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import (  # noqa: E402
    resolve_kermt_repo, assert_prepare_manifest_basics, count_vocab_entries, docker_image_digest,
    format_cmd_replay, git_commit_with_env_override, load_json, load_checkpoint,
    merge_default_into_applied, run_checkpoint_validator, runner_environment,
)


REPO_ROOT = resolve_kermt_repo()
SKILL_ROOT = Path(__file__).resolve().parent.parent
DEFAULTS_PATH = SKILL_ROOT / "config" / "defaults_pretrain.json"
CHECK_CHECKPOINT_PATH = SKILL_ROOT / "scripts" / "check_checkpoint.py"
PRETRAIN_DDP_PATH = REPO_ROOT / "pretrain_ddp.py"

# Model-type → pretrain_ddp.py `--pretrain_mode` value.
MODEL_TYPE_TO_PRETRAIN_MODE = {
    "grover_base": "vocab",
    "cmim": "cmim",
    "hybrid": "hybrid",
}

# Hyperparameter flags the runner exposes for CLI override + the corresponding
# key path in defaults_pretrain.json. None means the value isn't in defaults
# (e.g. seed has a default but lives at the top of training; lookup is direct).
TRAINING_FLAGS = (
    "batch_size", "dropout", "epochs", "init_lr", "max_lr", "final_lr",
    "warmup_epochs", "weight_decay", "save_interval", "seed", "tensorboard",
    "use_cuikmolmaker_featurization",
)
LOSS_FLAGS = ("contrastive_temperature", "vocab_loss_weight")
DECODER_FLAGS = (
    "latent_dim",
    "decoder_num_layers",
    "decoder_num_attention_heads",
    "decoder_ffn_hidden_size",
    "decoder_dropout",
    "decoder_max_seq_len",
    "decoder_positional_encoding",
    "decoder_gate_self_attn",
    "decoder_gate_cross_attn",
)

ARCH_FLAGS_FROM_CKPT = (
    "hidden_size", "depth", "num_attn_head", "activation", "backbone",
    "embedding_output_type", "self_attention",
)

# cMIM-decoder + latent-distribution arch fields. For continue-pretrain on a
# cmim/hybrid ckpt these MUST come from the ckpt's saved_args (so the model
# being constructed matches the ckpt's weights at load time); the
# defaults_pretrain.json `add_cmim_decoder` block is for add-cmim-pretrain's
# upgrade-time decoder construction only, and is intentionally ignored
# during continue-pretrain.
CMIM_DECODER_FLAGS_FROM_CKPT = (
    "latent_dim",
    "decoder_num_layers",
    "decoder_num_attention_heads",
    "decoder_ffn_hidden_size",
    "decoder_dropout",
    "decoder_max_seq_len",
    "decoder_positional_encoding",
    "decoder_gate_self_attn",
    "decoder_gate_cross_attn",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# JSON loading delegated to the shared _utils.load_json. Alias kept for the
# existing internal callsites that use the leading-underscore convention.
_load_json = load_json


def _detect_gpus(override: str | None) -> tuple[int, str]:
    """Returns (world_size, CUDA_VISIBLE_DEVICES_string)."""
    if override:
        gpu_list = [g.strip() for g in override.split(",") if g.strip()]
        return len(gpu_list), ",".join(gpu_list)
    # Honor an existing CUDA_VISIBLE_DEVICES in the environment.
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env:
        ids = [g for g in env.split(",") if g]
        return len(ids), ",".join(ids)
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    return n, ",".join(str(i) for i in range(n))


def _verify_prepare_manifest(manifest: dict[str, Any]) -> None:
    assert_prepare_manifest_basics(manifest, "pretrain")
    out = manifest.get("outputs", {})
    required_keys = ("train_dir", "val_dir", "atom_vocab", "bond_vocab")
    missing = [k for k in required_keys if k not in out]
    if missing:
        raise ValueError(
            f"prepare_data manifest is missing required outputs: {missing}. "
            "Was prepare_data.py invoked with --skip-vocab or --skip-split?"
        )


def _apply_defaults(args: argparse.Namespace, defaults: dict[str, Any],
                    model_type: str, world_size: int) -> dict[str, dict[str, Any]]:
    """Returns args_applied: dict mapping flag → {value, source}.
    Source is 'user' if the user passed a value on the CLI, else 'default-config'
    (from defaults_pretrain.json) or 'auto-1gpu' / 'auto-multi-gpu' for the
    auto-fallback values. Only includes flags relevant to the model_type."""
    applied: dict[str, dict[str, Any]] = {}

    training_defaults = defaults.get("training", {})
    loss_defaults = defaults.get("loss", {})
    decoder_defaults = defaults.get("add_cmim_decoder", {})

    for f in TRAINING_FLAGS:
        merge_default_into_applied(applied, args, f, training_defaults)

    # Single-GPU fallback: batch_size 32, save_interval 500.
    if world_size <= 1:
        if applied.get("batch_size", {}).get("source") != "user":
            applied["batch_size"] = {"value": 32, "source": "auto-1gpu"}
        if applied.get("save_interval", {}).get("source") != "user":
            applied["save_interval"] = {"value": 500, "source": "auto-1gpu"}

    if model_type in ("cmim", "hybrid"):
        for f in LOSS_FLAGS if model_type == "hybrid" else ("contrastive_temperature",):
            merge_default_into_applied(applied, args, f, loss_defaults)
        for f in DECODER_FLAGS:
            merge_default_into_applied(applied, args, f, decoder_defaults)

    return applied


def _arch_from_validator(validator_out: dict[str, Any]) -> dict[str, Any]:
    arch = validator_out.get("arch") or {}
    missing = [k for k in ARCH_FLAGS_FROM_CKPT if arch.get(k) is None]
    if missing:
        raise ValueError(
            f"checkpoint validator did not surface required arch fields: {missing}. "
            "If the ckpt has no saved_args blob, these can't be inferred from state-dict "
            "shapes alone; please supply a ckpt with args saved (the standard "
            "save_model_for_restart format)."
        )
    return arch


def _build_argv(
    *, world_size: int, gpus_str: str, out_dir: Path, manifest: dict[str, Any],
    model_type: str, pretrain_mode: str, arch: dict[str, Any],
    applied: dict[str, dict[str, Any]],
) -> list[str]:
    """Constructs the full pretrain_ddp.py argument list as a list of strings."""
    outputs = manifest["outputs"]
    argv = [sys.executable, "-u", str(PRETRAIN_DDP_PATH)]

    # Data + vocab paths
    argv += ["--train_data_path", outputs["train_dir"],
             "--val_data_path",   outputs["val_dir"],
             "--atom_vocab_path", outputs["atom_vocab"],
             "--bond_vocab_path", outputs["bond_vocab"]]
    if model_type in ("cmim", "hybrid"):
        argv += ["--smiles_vocab_path", outputs["smiles_vocab"]]

    # Pretrain mode + loss
    argv += ["--pretrain_mode", pretrain_mode]
    if "vocab_loss_weight" in applied and model_type == "hybrid":
        argv += ["--vocab_loss_weight", str(applied["vocab_loss_weight"]["value"])]
    if "contrastive_temperature" in applied and model_type in ("cmim", "hybrid"):
        argv += ["--contrastive_temperature", str(applied["contrastive_temperature"]["value"])]
    # cMIM/decoder arch: emit every applied flag. For continue-pretrain on a
    # cmim/hybrid ckpt, every entry will be source="ckpt_saved_args" (see the
    # overlay loop in run()). For pretrain-from-scratch / add-cmim-pretrain
    # the values come from defaults_pretrain.json's add_cmim_decoder block.
    if model_type in ("cmim", "hybrid"):
        for f in ("latent_dim", "decoder_num_layers", "decoder_num_attention_heads",
                  "decoder_ffn_hidden_size", "decoder_dropout",
                  "decoder_max_seq_len", "decoder_positional_encoding"):
            if f in applied:
                argv += [f"--{f}", str(applied[f]["value"])]
        # Boolean store_true flags: emit the bare flag only when True.
        if applied.get("decoder_gate_self_attn", {}).get("value"):
            argv += ["--decoder_gate_self_attn"]
        if applied.get("decoder_gate_cross_attn", {}).get("value"):
            argv += ["--decoder_gate_cross_attn"]

    # Architecture — sourced from validator's arch block, never from CLI/defaults.
    argv += [
        "--hidden_size",  str(arch["hidden_size"]),
        "--depth",        str(arch["depth"]),
        "--num_attn_head", str(arch["num_attn_head"]),
        "--activation",   str(arch["activation"]),
        "--backbone",     str(arch["backbone"]),
        "--embedding_output_type", str(arch["embedding_output_type"]),
    ]
    if arch.get("self_attention"):
        argv += ["--self_attention"]

    # Training schedule
    for name in ("batch_size", "dropout", "epochs", "init_lr", "max_lr", "final_lr",
                 "warmup_epochs", "weight_decay", "save_interval", "seed"):
        if name in applied:
            argv += [f"--{name}", str(applied[name]["value"])]
    if applied.get("tensorboard", {}).get("value"):
        argv += ["--tensorboard"]
    if applied.get("use_cuikmolmaker_featurization", {}).get("value"):
        argv += ["--use_cuikmolmaker_featurization"]

    # W&B logging (pass-through; pretrain_ddp.py only inits W&B when project is set).
    if "wandb_project" in applied:
        argv += ["--wandb_project", str(applied["wandb_project"]["value"])]
        if "wandb_run_name" in applied:
            argv += ["--wandb_run_name", str(applied["wandb_run_name"]["value"])]

    # Where pretrain_ddp.py auto-resumes from (we'll symlink the user ckpt there).
    argv += ["--save_dir", str(out_dir / "ckpt")]

    return argv


def _symlink_ckpt_into_save_dir(user_ckpt: Path, save_dir: Path) -> Path:
    """--resume path: symlink the user ckpt as <save_dir>/last_checkpoint.pt.
    pretrain_ddp.py's auto-resume then restores everything from the ckpt:
    model weights, optimizer state, scheduler_step, epoch, batch_idx,
    wandb_run_id."""
    save_dir.mkdir(parents=True, exist_ok=True)
    link = save_dir / "last_checkpoint.pt"
    if link.exists() or link.is_symlink():
        link.unlink()
    # Symlink to the absolute user_ckpt so it works regardless of cwd.
    link.symlink_to(user_ckpt.resolve())
    return link


# Schedule fields that --resume inherits from ckpt.saved_args and that default
# (fresh-schedule) mode takes from CLI/defaults_pretrain.json.
SCHEDULE_FLAGS = ("epochs", "warmup_epochs", "init_lr", "max_lr", "final_lr")


def _materialize_ckpt_for_fresh_schedule(user_ckpt: Path, save_dir: Path) -> Path:
    """Default (fresh-schedule) continue-pretrain path: write a CLEANED copy
    of the user ckpt to <save_dir>/last_checkpoint.pt with scheduler_step,
    epoch, batch_idx, and wandb_run_id reset to fresh-start values. Model
    weights AND optimizer state pass through unchanged — so Adam's running
    moments warm-start the new schedule (helpful because the new init_lr is
    usually close to the previous run's final_lr).

    Why a fresh-state copy instead of a symlink: pretrain_ddp.py's
    `trainer.load()` restores EVERYTHING in the ckpt including scheduler_step
    and epoch. We can't selectively load just the model + optimizer through
    that code path. The minimal-invasive workaround is to materialize a
    ckpt that has the unwanted counters zeroed before the loader sees it.
    pretrain_ddp.py then restores everything as normal, but everything it
    restores reads as a fresh-start.

    Cost: one ~700 MB disk write per run. Pretrain is days-long, so it's
    negligible. Done on the host before docker run.
    """
    import torch  # delayed import — keeps the runner light in --dry-run paths
    save_dir.mkdir(parents=True, exist_ok=True)
    target = save_dir / "last_checkpoint.pt"
    if target.exists() or target.is_symlink():
        target.unlink()
    ckpt = load_checkpoint(user_ckpt)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(
            f"ckpt {user_ckpt} is not in the expected save_model_for_restart "
            "dict format (need at least 'state_dict' key)."
        )
    ckpt["scheduler_step"] = 0
    ckpt["epoch"] = 0
    ckpt["batch_idx"] = 0
    ckpt["wandb_run_id"] = None
    torch.save(ckpt, target)
    return target


def _validate_resume_state(user_ckpt: Path) -> dict[str, Any]:
    """--resume mode: confirm the ckpt was saved via the save_model_for_restart
    format and carries the full state pretrain_ddp.py needs to resume mid-run
    (optimizer state, scheduler_step, epoch, batch_idx). Returns a small
    `resume_state` dict for the manifest so users can see what was restored.
    Raises ValueError with a clear redirect if the ckpt is too lean."""
    ckpt = load_checkpoint(user_ckpt)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(
            f"ckpt {user_ckpt} is not in the expected save_model_for_restart "
            "dict format."
        )
    required = ("optimizer", "scheduler_step", "epoch", "batch_idx")
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise ValueError(
            f"--resume requires the ckpt to carry the full mid-run state, but "
            f"these keys are missing: {missing}. The ckpt was probably saved "
            "without enough metadata to pure-resume — use the default "
            "fresh-schedule mode (drop --resume) if you just want to continue "
            "training with a new schedule."
        )
    return {
        "scheduler_step": int(ckpt["scheduler_step"]),
        "epoch": int(ckpt["epoch"]),
        "batch_idx": int(ckpt["batch_idx"]),
        "wandb_run_id": ckpt.get("wandb_run_id"),
    }


# Vocab-entry counting delegated to _utils.count_vocab_entries. Alias kept for
# the existing internal callsites.
_count_vocab_entries = count_vocab_entries


def _verify_vocab_sizes_match_ckpt(
    manifest: dict[str, Any], validator_out: dict[str, Any], model_type: str,
) -> dict[str, Any]:
    """For continue-pretrain only: compare each vocab file's entry count against
    the ckpt's vocab head dimensions. Aborts on mismatch with a helpful error
    pointing the user at the matching vocab. Returns a `vocab_check` block to
    attach to run.json for transparency."""
    ckpt_sizes = validator_out.get("vocab_sizes") or {"atom": None, "bond": None, "smiles": None}
    outputs = manifest.get("outputs", {})
    check: dict[str, Any] = {"vocab_source": manifest.get("vocab_source", "unknown")}
    for which in ("atom", "bond", "smiles"):
        ckpt_size = ckpt_sizes.get(which)
        vocab_path_str = outputs.get(f"{which}_vocab")
        check[which] = {"ckpt_size": ckpt_size, "manifest_vocab": vocab_path_str, "manifest_size": None}
        if ckpt_size is None:
            # ckpt doesn't have this head; nothing to verify.
            continue
        # ckpt has this head — the manifest MUST include the corresponding vocab.
        if not vocab_path_str:
            raise ValueError(
                f"ckpt has a '{which}' vocab head (size {ckpt_size}) but the prepare_data "
                f"manifest doesn't include a {which}_vocab file. Rerun prepare_data with "
                f"--vocab-dir <ckpt's parent dir> (or --{which}-vocab <path>) so the runner "
                f"can pass the matching vocab through."
            )
        manifest_size = _count_vocab_entries(Path(vocab_path_str))
        check[which]["manifest_size"] = manifest_size
        if manifest_size != ckpt_size:
            raise ValueError(
                f"{which} vocab size mismatch — ckpt's head expects {ckpt_size} entries, "
                f"but {vocab_path_str} has {manifest_size}. The released ckpt's vocab is the "
                f"authoritative one for continue-pretrain; pass --vocab-dir <ckpt's parent dir> "
                f"(or --{which}-vocab <path>) to prepare_data so vocab built from the new corpus "
                f"isn't used. If you actually want to pretrain from scratch on a different "
                f"vocab, use the kermt-pretrain-scratch workflow instead."
            )
    return check


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    from_scratch = bool(args.from_scratch)
    resume = bool(args.resume)

    # Mode-conflict validation up-front so the user fails fast.
    if from_scratch and resume:
        raise ValueError("--resume is incompatible with --from-scratch.")
    if resume and not args.ckpt:
        raise ValueError("--resume requires --ckpt; nothing to resume from otherwise.")
    if resume:
        # CLI overrides of schedule args are forbidden in --resume mode — pure
        # resume means the schedule shape from the ckpt is authoritative.
        cli_overrides = [
            f for f in SCHEDULE_FLAGS if getattr(args, f, None) is not None
        ]
        if cli_overrides:
            raise ValueError(
                f"--resume inherits schedule args from the ckpt's saved_args; "
                f"explicit CLI override is forbidden. You passed: {cli_overrides}. "
                "Drop those flags to pure-resume, or use the default fresh-schedule "
                "mode (no --resume) if you want a new schedule."
            )

    workflow = "pretrain-scratch" if from_scratch else "continue-pretrain"
    if resume:
        mode = "continue_pretrain_resume"
    elif from_scratch:
        mode = "pretrain_from_scratch"
    else:
        mode = "continue_pretrain_fresh_schedule"

    # 1. Load defaults + prepare manifest.
    defaults = _load_json(DEFAULTS_PATH, name="defaults_pretrain.json")
    prep_manifest_path = Path(args.prepare_manifest).resolve()
    manifest = _load_json(prep_manifest_path, name="prepare_data.json")
    _verify_prepare_manifest(manifest)

    # 2. Branch: continue-pretrain (load ckpt + validate) vs from-scratch (no ckpt).
    ckpt: Path | None = None
    validator_out: dict[str, Any] | None = None
    link: Path | None = None
    vocab_check: dict[str, Any] | None = None
    resume_state: dict[str, Any] | None = None  # populated only when --resume

    if from_scratch:
        if args.ckpt:
            raise ValueError("--from-scratch is incompatible with --ckpt; pass one or the other.")
        if not args.pretrain_target_mode:
            raise ValueError("--pretrain-target-mode is required when --from-scratch is set "
                             "(choose vocab, cmim, or hybrid).")
        pretrain_mode = args.pretrain_target_mode
        model_type = {"vocab": "grover_base", "cmim": "cmim", "hybrid": "hybrid"}[pretrain_mode]
        # Arch from defaults_pretrain.json's `arch` group (with CLI overrides applied later
        # if we expose any; for now we just use defaults).
        arch_defaults = defaults.get("arch") or {}
        if not arch_defaults:
            raise ValueError("defaults_pretrain.json has no `arch` group; cannot pretrain from scratch.")
        arch = {k: arch_defaults.get(k) for k in ARCH_FLAGS_FROM_CKPT}
        # `latent_dim` lives in the add_cmim_decoder group for from-scratch cmim/hybrid;
        # treat it as part of the arch for argv-building purposes.
        if pretrain_mode in ("cmim", "hybrid"):
            arch["latent_dim"] = (defaults.get("add_cmim_decoder") or {}).get("latent_dim")
        else:
            arch["latent_dim"] = None
    else:
        if not args.ckpt:
            raise ValueError("--ckpt is required for continue-pretrain. "
                             "Use --from-scratch to pretrain a fresh model on the corpus.")
        ckpt = Path(args.ckpt).resolve()
        if args.ckpt_validator_out:
            validator_out = _load_json(Path(args.ckpt_validator_out), name="ckpt validator output")
        else:
            validator_out = run_checkpoint_validator(ckpt, mode="continue_pretrain", script_path=CHECK_CHECKPOINT_PATH)
        if not validator_out.get("ok"):
            raise ValueError(
                f"check_checkpoint.py rejected the input ckpt: {validator_out.get('errors')}"
            )
        model_type = validator_out.get("model_type")
        if model_type not in MODEL_TYPE_TO_PRETRAIN_MODE:
            raise ValueError(
                f"model_type='{model_type}' cannot continue pretrain. "
                f"Supported: {sorted(MODEL_TYPE_TO_PRETRAIN_MODE)}. "
                "For an encoder-only ckpt with no pretrain head, use the "
                "upgrade_to_hybrid workflow."
            )
        if model_type == "grover_base" and not validator_out.get("has_vocab_head"):
            raise ValueError(
                "grover_base ckpt has no vocab head — cannot continue vocab pretrain. "
                "Use the upgrade_to_hybrid workflow to add a cMIM decoder, "
                "or finetune directly from the encoder."
            )
        pretrain_mode = MODEL_TYPE_TO_PRETRAIN_MODE[model_type]
        arch = _arch_from_validator(validator_out)
        # Vocab-size verification — refuse mismatched corpora before launching pretrain_ddp.py.
        vocab_check = _verify_vocab_sizes_match_ckpt(manifest, validator_out, model_type)
        # --resume needs the ckpt to carry the full mid-run state. Validate now;
        # also surface what's being restored in the manifest.
        if resume:
            resume_state = _validate_resume_state(ckpt)

    # 3. GPU selection.
    world_size, gpus_str = _detect_gpus(args.gpus)
    if world_size <= 0:
        raise ValueError(
            "No GPUs detected. pretrain_ddp.py requires at least one CUDA device. "
            "Set CUDA_VISIBLE_DEVICES or pass --gpus <ids>."
        )

    # 4. Apply defaults + collect args_applied.
    applied = _apply_defaults(args, defaults, model_type, world_size)
    # --resume overlays schedule args from the ckpt's saved_args (the only path
    # where source="ckpt_saved_args" can appear in args_applied). Fail loudly if
    # any schedule field is missing from saved_args — pure-resume can't proceed
    # without the original schedule shape.
    if resume:
        saved_args = validator_out.get("saved_args") or {}
        missing = [f for f in SCHEDULE_FLAGS if f not in saved_args]
        if missing:
            raise ValueError(
                f"--resume requires the ckpt's saved_args to include all schedule "
                f"fields, but these are missing: {missing}. The ckpt was saved "
                "without enough metadata to pure-resume — use the default "
                "fresh-schedule mode and specify --epochs / --warmup-epochs / "
                "--init-lr / --max-lr / --final-lr explicitly."
            )
        for f in SCHEDULE_FLAGS:
            applied[f] = {"value": saved_args[f], "source": "ckpt_saved_args"}

    # Continue-pretrain on a cmim/hybrid ckpt: cMIM/decoder arch must come
    # from the ckpt's saved_args, not from defaults or CLI. This is the
    # cmim/decoder analogue of the encoder-arch passthrough already done by
    # `_arch_from_validator` (and matches the README guarantee that
    # `add_cmim_decoder` defaults are ignored during continue-pretrain).
    if not from_scratch and model_type in ("cmim", "hybrid"):
        cli_latent_dim_override = args.latent_dim is not None
        if cli_latent_dim_override:
            raise ValueError(
                "--latent-dim cannot be overridden during continue-pretrain on a "
                "cmim/hybrid ckpt — the value is fixed by the ckpt's saved_args "
                "(passing a different value would mismatch the loaded decoder "
                "weights). Drop --latent-dim, or use kermt-pretrain-scratch if "
                "you intentionally want a different latent dimension."
            )
        saved_args = validator_out.get("saved_args") or {}
        cmim_missing = [f for f in CMIM_DECODER_FLAGS_FROM_CKPT if f not in saved_args]
        if cmim_missing:
            raise ValueError(
                f"continue-pretrain on a {model_type} ckpt requires the ckpt's "
                f"saved_args to include cmim/decoder arch fields, but these are "
                f"missing: {cmim_missing}. The ckpt was saved without enough "
                "metadata to faithfully reconstruct the decoder."
            )
        for f in CMIM_DECODER_FLAGS_FROM_CKPT:
            applied[f] = {"value": saved_args[f], "source": "ckpt_saved_args"}

    # Optional W&B logging: pass-through, no defaults — forwarded only when the
    # user sets --wandb-project (run name is honored only alongside a project).
    for f in ("wandb_project", "wandb_run_name"):
        v = getattr(args, f, None)
        if v is not None:
            applied[f] = {"value": v, "source": "user"}

    # 5. Build the pretrain_ddp.py argv.
    argv = _build_argv(
        world_size=world_size, gpus_str=gpus_str, out_dir=out_dir, manifest=manifest,
        model_type=model_type, pretrain_mode=pretrain_mode, arch=arch, applied=applied,
    )

    # 6. (continue-pretrain only) Stage the ckpt into <save_dir>/last_checkpoint.pt
    #    so pretrain_ddp.py's auto-resume picks it up. Mode-dispatched:
    #    - --resume: symlink to user ckpt. pretrain_ddp.py restores everything
    #      (model + optimizer + scheduler_step + epoch + batch_idx + wandb_run_id).
    #    - default (fresh-schedule): materialize a state-cleaned copy of the
    #      ckpt — model weights + optimizer pass through, but scheduler_step /
    #      epoch / batch_idx / wandb_run_id are reset to 0/None. pretrain_ddp.py
    #      then builds a fresh NoamLR from CLI args and starts from step 0.
    # Done unconditionally (including --dry-run) so the dry-run faithfully
    # exercises ckpt I/O — catches corrupt ckpts / insufficient disk before
    # the days-long real run.
    if not from_scratch:
        if resume:
            link = _symlink_ckpt_into_save_dir(ckpt, out_dir / "ckpt")
        else:
            link = _materialize_ckpt_for_fresh_schedule(ckpt, out_dir / "ckpt")

    # 7. Build the run.json manifest.
    commit, dirty = git_commit_with_env_override(REPO_ROOT)
    image_tag = os.environ.get("KERMT_IMAGE", "kermt:latest")
    image_digest = docker_image_digest(image_tag)
    cmd_replay_env: dict[str, str] = {}
    if gpus_str:
        cmd_replay_env["CUDA_VISIBLE_DEVICES"] = gpus_str
    cmd_replay_env["WORLD_SIZE"] = str(world_size)
    cmd_replay = format_cmd_replay(argv, env=cmd_replay_env)
    run_manifest = {
        "workflow": workflow,
        "mode": mode,  # pretrain_from_scratch | continue_pretrain_fresh_schedule | continue_pretrain_resume
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "container": {"image_tag": image_tag, "image_digest": image_digest},
        "repo": {"commit": commit, "dirty": dirty},
        "inputs": {
            "ckpt": str(ckpt) if ckpt else None,
            "prepare_data_manifest": str(prep_manifest_path),
            "ckpt_validator_out": (
                str(Path(args.ckpt_validator_out).resolve()) if args.ckpt_validator_out else None
            ),
        },
        "model_type": model_type,
        "pretrain_mode": pretrain_mode,
        "world_size": world_size,
        "cuda_visible_devices": gpus_str,
        "args_applied": applied,
        "arch": arch,
        "vocab_check": vocab_check,  # None for from-scratch
        "resume_state": resume_state,  # None unless --resume; carries the restored scheduler_step / epoch / batch_idx / wandb_run_id from the ckpt
        "save_dir": str(out_dir / "ckpt"),
        "logs_dir": str(out_dir / "logs"),
        "tensorboard_dir": str(out_dir / "logs" / "tb"),
        "argv": argv,
        "cmd_replay": cmd_replay,
        "ok_to_replay": (not dirty) and (commit != "unknown"),
        "dry_run": bool(args.dry_run),
        "ckpt_symlink": str(link) if link else None,
        "from_scratch": from_scratch,
    }
    (out_dir / "run.json").write_text(json.dumps(run_manifest, indent=2))

    # 8. Execute (unless --dry-run).
    if args.dry_run:
        run_manifest["status"] = "dry_run"
        return run_manifest

    env = runner_environment(REPO_ROOT, wandb="wandb_project" in applied)
    env["WORLD_SIZE"] = str(world_size)
    if gpus_str:
        env["CUDA_VISIBLE_DEVICES"] = gpus_str

    log_file = out_dir / "logs" / "pretrain_ddp.log"
    with log_file.open("w") as logf:
        proc = subprocess.run(argv, env=env, stdout=logf, stderr=subprocess.STDOUT)
    run_manifest["exit_code"] = proc.returncode
    run_manifest["status"] = "ok" if proc.returncode == 0 else "failed"
    (out_dir / "run.json").write_text(json.dumps(run_manifest, indent=2))
    return run_manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Workstation pretrain runner (continue-pretrain by default, "
                    "or pretrain-from-scratch with --from-scratch).")
    p.add_argument("--ckpt", default=None,
                   help="Path to the input pretrain checkpoint. Required for continue-pretrain; "
                        "omit when --from-scratch is set.")
    p.add_argument("--from-scratch", action="store_true",
                   help="Pretrain a fresh model on the corpus (no input ckpt; arch from "
                        "defaults_pretrain.json; vocab built by prepare_data). Requires "
                        "--pretrain-target-mode.")
    p.add_argument("--resume", action="store_true",
                   help="Resume an interrupted pretrain run (crashed / Ctrl-C / OOM). "
                        "Restores everything from the ckpt: model weights, optimizer "
                        "state, scheduler_step, epoch, batch_idx, wandb_run_id. Schedule "
                        "shape (epochs / warmup_epochs / init/max/final_lr) is inherited "
                        "from the ckpt's saved_args; CLI overrides of schedule flags are "
                        "REJECTED in this mode. Without --resume (default), continue-pretrain "
                        "loads only model weights + optimizer momentum from the ckpt and "
                        "starts a fresh schedule from CLI/defaults_pretrain.json — use that "
                        "default mode when continue-pretraining on a new corpus / new "
                        "objective / extended training (the common case).")
    p.add_argument("--pretrain-target-mode", choices=["vocab", "cmim", "hybrid"], default=None,
                   help="(--from-scratch only) which pretrain objective to use for the fresh "
                        "model: vocab (grover_base-style), cmim, or hybrid (vocab + contrast). "
                        "No default — must be set explicitly so the user makes an informed "
                        "choice about the head config.")
    p.add_argument("--prepare-manifest", required=True,
                   help="Path to a prepare_data.json (must be mode=pretrain)")
    p.add_argument("--out", required=True, help="Output run directory")
    p.add_argument("--ckpt-validator-out", default=None,
                   help="Optional cached check_checkpoint.py JSON; computed if absent")
    p.add_argument("--gpus", default=None,
                   help="Comma-separated GPU ids (e.g. '0,1'). Default: all visible")
    p.add_argument("--dry-run", action="store_true",
                   help="Write run.json and print the command without executing")
    # Training overrides — all default to None so we can distinguish user-given vs default-config.
    for f, t in [("epochs", int), ("batch-size", int), ("init-lr", float), ("max-lr", float),
                 ("final-lr", float), ("warmup-epochs", float), ("weight-decay", float),
                 ("dropout", float), ("save-interval", int), ("seed", int),
                 ("vocab-loss-weight", float), ("latent-dim", int),
                 ("contrastive-temperature", float)]:
        p.add_argument(f"--{f}", type=t, default=None)
    # Optional W&B logging (pass-through to pretrain_ddp.py; off unless project is set).
    p.add_argument("--wandb-project", type=str, default=None,
                   help="W&B project name. When set, pretrain_ddp.py logs train/val losses.")
    p.add_argument("--wandb-run-name", type=str, default=None,
                   help="Optional W&B run name (only used when --wandb-project is set).")
    args = p.parse_args(argv)

    try:
        manifest = run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(json.dumps({"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}, indent=2),
              file=sys.stdout)
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
