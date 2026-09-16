#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a KERMT checkpoint for a given agent workflow.

Mode-dispatched. Emits a single JSON object to stdout that the calling skill
parses to decide whether to proceed (`ok: true`) or surface a structured error
back to the user (`ok: false` with `errors[]`). The JSON shape is stable
across modes; only the contract for what counts as `ok` differs per mode.

Modes
-----
continue_pretrain   Continuing pretraining from an existing pretrain ckpt.
                    Requires encoder + at least one pretrain head
                    (vocab_head for grover_base / cmim, or contrast_head for
                    cmim / hybrid). Rejects encoder-only or finetuned ckpts.

upgrade_to_hybrid   Adding a cMIM decoder onto a grover_base ckpt to convert
                    it to a hybrid pretrain. Requires encoder; rejects ckpts
                    that already carry a contrast_head or task_ffn (would be
                    workflow 4 instead).

finetune_init       Starting a finetune from a pretrained ckpt. Requires
                    encoder. Pretrain heads (vocab / contrast) are tolerated
                    but unused. Already-finetuned ckpts (task FFN heads
                    present) are REJECTED — finetune-on-finetune via the
                    agent skill isn't supported because saved-task
                    identity can't be machine-verified against the new
                    training data.

inference           Running predictions with a previously-finetuned ckpt.
                    Requires encoder + task_ffn. Reports task_output_dims
                    so the runner can compare against the user's task spec.

embed               Extracting embeddings. Requires encoder only. Anything
                    additional in the ckpt is ignored.

Output (stdout)
---------------
{
  "ok": true | false,
  "model_type": "grover_base" | "cmim" | "hybrid" | "finetuned" | "unknown",
  "has_encoder": bool,
  "has_vocab_head": bool,
  "has_contrast_head": bool,
  "has_task_ffn": bool,
  "task_output_dims": [int, ...],   // empty unless has_task_ffn
  "arch": {                          // ckpt-derived; runner uses these, ignores defaults_*.json arch
    "hidden_size": int | null,
    "depth": int | null,
    "num_attn_head": int | null,
    "latent_dim": int | null,
    "activation": str | null,
    "backbone": str | null,
    "embedding_output_type": str | null,
    "self_attention": bool | null
  },
  "saved_args": { ... } | null,      // raw args dict if present, else null
  "errors": [str, ...],              // mode-contract violations / load failures
  "warnings": [str, ...]             // non-fatal observations (e.g. arch fallback)
}

Exit code: 0 on `ok: true`, 1 on `ok: false`. Loader exceptions are caught and
surfaced into `errors[]` with `ok: false` (still exit 1), never raised.

CLI
---
    check_checkpoint.py --mode <mode> --ckpt <path>
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_checkpoint  # noqa: E402


# ---------------------------------------------------------------------------
# State-dict key prefix conventions (kermt/model/models.py).
# ---------------------------------------------------------------------------

# Encoder weights appear under one of these prefixes depending on the ckpt's
# era and task class:
#   - `grover.*`           : legacy grover_base ckpts (predate the cMIM rename)
#   - `kermt.*`            : current grover_base / hybrid / finetune ckpts
#   - `latent_dist.kermt.*`: cmim ckpts (encoder lives only inside latent_dist)
ENCODER_PREFIXES = ("kermt.", "grover.", "latent_dist.kermt.")
VOCAB_HEAD_PREFIX = "vocab_module."
CONTRAST_DECODER_PREFIX = "decoder."  # SMILES transformer decoder, cmim/hybrid only
LATENT_DIST_PREFIX = "latent_dist."    # cmim/hybrid; encoder may share via latent_dist.kermt.*
TASK_FFN_PREFIXES = (
    "mol_atom_from_atom_ffn.",
    "mol_atom_from_bond_ffn.",
)
TASK_FFN_TASK_SPECIFIC_PREFIXES = (
    "mol_atom_from_atom_ffn_task_specific.",
    "mol_atom_from_bond_ffn_task_specific.",
)


ARCH_KEYS = (
    "hidden_size",
    "depth",
    "num_attn_head",
    "latent_dim",
    "activation",
    "backbone",
    "embedding_output_type",
    "self_attention",
)


def _strip_ddp_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip `module.` prefix from every key if the dict is DDP-wrapped."""
    if state_dict and all(k.startswith("module.") for k in state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _classify_model(state_dict: dict[str, Any]) -> dict[str, Any]:
    keys = list(state_dict.keys())
    has_encoder = any(k.startswith(ENCODER_PREFIXES) for k in keys)
    has_vocab_head = any(k.startswith(VOCAB_HEAD_PREFIX) for k in keys)
    has_contrast_head = any(k.startswith(CONTRAST_DECODER_PREFIX) for k in keys)
    has_task_ffn = any(k.startswith(TASK_FFN_PREFIXES) for k in keys)

    if has_encoder and has_task_ffn:
        model_type = "finetuned"
    elif has_encoder and has_contrast_head and has_vocab_head:
        model_type = "hybrid"
    elif has_encoder and has_contrast_head and not has_vocab_head:
        model_type = "cmim"
    elif has_encoder and not has_contrast_head:
        # Includes:
        #  - modern repo-trained Grover base (kermt.* + vocab_module.*)
        #  - legacy original-Grover base (grover.encoders.* with no heads saved)
        #  - any encoder-stripped ckpt extracted from a larger model
        # The `has_vocab_head` flag discriminates the sub-cases for skills that
        # need it. The continue_pretrain mode contract relies on this — a
        # grover_base with vocab heads can continue, an encoder-only one cannot.
        model_type = "grover_base"
    else:
        model_type = "unknown"

    return {
        "model_type": model_type,
        "has_encoder": has_encoder,
        "has_vocab_head": has_vocab_head,
        "has_contrast_head": has_contrast_head,
        "has_task_ffn": has_task_ffn,
    }


def _vocab_sizes(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract vocab head sizes from state-dict weight shapes.

    The pretrain heads have the following layout per kermt/model/models.py:
      - Atom vocab predictors:  vocab_module.av_task_atom.*  + vocab_module.av_task_bond.*
        (two readout streams sharing the same vocab_size). Output dim of each
        final-Linear is the atom vocab size.
      - Bond vocab predictors:  vocab_module.bv_task_atom.*  + vocab_module.bv_task_bond.*
        Output dim is the bond vocab size.
      - SMILES vocab decoder:   decoder.output_projection.weight  (cmim / hybrid only).
        Output dim is the smiles vocab size.

    Returns {atom: int|None, bond: int|None, smiles: int|None}. Each is None
    when the corresponding head isn't present in the ckpt (e.g. legacy
    encoder-only grover_base has none; cmim has smiles but not atom/bond).
    """
    sizes: dict[str, Any] = {"atom": None, "bond": None, "smiles": None}

    def _head_out_dim(prefix: str) -> int | None:
        # Pick the highest-numbered 2-D Linear weight under `prefix.*` — that's
        # the final output layer.
        candidates = [
            k for k in state_dict
            if k.startswith(prefix) and k.endswith(".weight")
            and hasattr(state_dict[k], "ndim") and state_dict[k].ndim == 2
        ]
        if not candidates:
            return None
        def _layer_index(k: str) -> int:
            # ".weight" -> ".<idx>.weight"; pick the rightmost numeric component.
            parts = k.split(".")
            for tok in reversed(parts[:-1]):
                if tok.isdigit():
                    return int(tok)
            return -1
        final = max(candidates, key=_layer_index)
        return int(state_dict[final].shape[0])

    sizes["atom"] = _head_out_dim("vocab_module.av_task_atom.")
    sizes["bond"] = _head_out_dim("vocab_module.bv_task_atom.")
    sizes["smiles"] = _head_out_dim("decoder.output_projection.")
    # If the decoder's output_projection isn't a Linear (e.g. some saves wrap
    # it differently), fall back to a search over decoder.* heads.
    if sizes["smiles"] is None:
        sizes["smiles"] = _head_out_dim("decoder.token_embedding.")
    return sizes


def _task_output_dims(state_dict: dict[str, Any]) -> list[int]:
    """Return one entry per (logical task × readout) head's final-Linear out-dim.

    Two layouts:
      - **MTL** (`mol_atom_from_atom_ffn_task_specific.<i>.*`): one entry per
        task-specific head's final-Linear out-dim. Typically `[1, 1, ..., 1]`
        for regression with N tasks across 2 readouts.
      - **Non-MTL** (`mol_atom_from_atom_ffn.*` only): one entry per shared FFN's
        final-Linear out-dim. Typically `[num_tasks, num_tasks]` (one per readout).

    When both layouts coexist in the same ckpt (MTL configuration: shared FFN
    feeds task-specific heads), only the task-specific dims are reported — the
    shared FFN there is an intermediate layer, not the model output.
    """
    has_task_specific = any(k.startswith(TASK_FFN_TASK_SPECIFIC_PREFIXES) for k in state_dict)

    heads: dict[str, list[str]] = {}
    for k in state_dict:
        if k.startswith(TASK_FFN_TASK_SPECIFIC_PREFIXES):
            parts = k.split(".")
            root = ".".join(parts[:2])  # e.g. "mol_atom_from_atom_ffn_task_specific.0"
            heads.setdefault(root, []).append(k)
        elif k.startswith(TASK_FFN_PREFIXES) and not k.startswith(TASK_FFN_TASK_SPECIFIC_PREFIXES):
            if has_task_specific:
                continue  # shared FFN is intermediate when task-specific heads exist
            root = k.split(".")[0]  # e.g. "mol_atom_from_atom_ffn"
            heads.setdefault(root, []).append(k)

    dims: list[int] = []
    for root in sorted(heads):
        weight_keys = sorted(
            (k for k in heads[root] if k.endswith(".weight")
             and hasattr(state_dict[k], "ndim") and state_dict[k].ndim == 2),
            key=lambda k: int(k.split(".")[-2]) if k.split(".")[-2].isdigit() else -1,
        )
        if weight_keys:
            dims.append(int(state_dict[weight_keys[-1]].shape[0]))
    return dims


def _arch_from_args(args_obj: Any) -> dict[str, Any]:
    """Pull arch params from the saved args Namespace / dict, leaving missing keys as None."""
    arch: dict[str, Any] = {k: None for k in ARCH_KEYS}
    if args_obj is None:
        return arch
    # args_obj is typically argparse.Namespace; tolerate dict form too.
    args_dict = vars(args_obj) if isinstance(args_obj, Namespace) else dict(args_obj) if isinstance(args_obj, dict) else {}
    for k in ARCH_KEYS:
        if k in args_dict:
            arch[k] = args_dict[k]
    return arch


def _arch_from_shapes(state_dict: dict[str, Any], arch: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Fill in still-missing arch params by introspecting state-dict tensor shapes.

    Only fills entries that are currently None — does not override anything pulled
    from saved_args. Returns the updated arch + a list of warnings for any key that
    could not be inferred.
    """
    warnings: list[str] = []

    if arch["hidden_size"] is None:
        # First 2-D linear weight under any encoder prefix.
        candidates = [
            k for k in state_dict
            if k.startswith(ENCODER_PREFIXES)
            and k.endswith(".weight")
            and hasattr(state_dict[k], "ndim")
            and state_dict[k].ndim == 2
        ]
        if candidates:
            arch["hidden_size"] = int(state_dict[candidates[0]].shape[0])
        else:
            warnings.append("hidden_size could not be inferred from state_dict shapes")

    if arch["latent_dim"] is None:
        # Look for a Linear inside latent_dist that's not the shared encoder.
        candidates = [
            k for k in state_dict
            if k.startswith(LATENT_DIST_PREFIX)
            and not k.startswith("latent_dist.kermt.")
            and k.endswith(".weight")
            and state_dict[k].ndim == 2
        ]
        if candidates:
            arch["latent_dim"] = int(state_dict[candidates[0]].shape[0])
        # Encoder-only models have no latent distribution; latent_dim remains None.

    # depth, num_attn_head, activation, backbone, embedding_output_type, self_attention
    # are not robustly inferable from shapes alone; report a warning for each that's
    # still None so the caller can prompt the user or refuse to proceed.
    for k in ("depth", "num_attn_head", "activation", "backbone", "embedding_output_type", "self_attention"):
        if arch[k] is None:
            warnings.append(f"{k} not present in saved_args and cannot be inferred from state_dict shapes")

    return arch, warnings


def _apply_mode_contract(mode: str, classification: dict[str, Any]) -> list[str]:
    """Return a list of error messages if `classification` violates the mode contract."""
    errors: list[str] = []
    mt = classification["model_type"]
    has_enc = classification["has_encoder"]
    has_vocab = classification["has_vocab_head"]
    has_contrast = classification["has_contrast_head"]
    has_ffn = classification["has_task_ffn"]

    if not has_enc:
        errors.append("checkpoint has no encoder weights — cannot use it for any KERMT workflow")
        return errors

    if mode == "continue_pretrain":
        if not (has_vocab or has_contrast):
            errors.append(
                f"continue_pretrain requires the ckpt to still carry pretrain heads (vocab "
                f"and/or contrast), but this ckpt has neither (model_type='{mt}', "
                f"has_vocab_head=False, has_contrast_head=False). Either provide a ckpt with "
                f"its pretrain heads attached, or convert this encoder-only ckpt to a hybrid "
                f"via mode 'upgrade_to_hybrid'."
            )
        if has_ffn:
            errors.append(
                "continue_pretrain expects a pretrain ckpt; this ckpt has task FFN heads "
                "(it has been finetuned). Use a pretrain checkpoint — finetune+continue is "
                "not a supported workflow."
            )
    elif mode == "upgrade_to_hybrid":
        if has_contrast:
            errors.append(
                f"upgrade_to_hybrid converts grover_base -> hybrid by adding a cMIM decoder. "
                f"This ckpt already has a contrast head (classified as '{mt}'). "
                f"To continue pretraining it, use mode 'continue_pretrain'."
            )
        if has_ffn:
            errors.append("upgrade_to_hybrid does not support finetuned checkpoints.")
    elif mode == "finetune_init":
        # Requires an encoder. Pretrain heads (vocab / contrast) are unused
        # at finetune time but harmless. Task FFN heads (i.e. an already-
        # finetuned ckpt) are NOT accepted — finetune-on-finetune isn't
        # supported by the kermt-finetune skill because the saved-task
        # identity can't be machine-verified against the new training data
        # (dimension match doesn't prove target identity, dataset identity,
        # or absence of train/test contamination).
        if has_ffn:
            errors.append(
                f"finetune_init requires a pretrain ckpt (grover_base / cmim / hybrid); "
                f"this ckpt is classified as '{mt}' with task FFN heads attached. "
                f"To resume a finetune on the SAME dataset, call "
                f"`python main.py finetune --checkpoint_path <ckpt> ...` directly — the "
                f"kermt-finetune skill doesn't support resume."
            )
    elif mode == "inference":
        if not has_ffn:
            errors.append(
                "inference requires a finetuned ckpt with task FFN heads. "
                f"This ckpt is classified as '{mt}' with no task heads. "
                "Run finetune (mode 'finetune_init') first."
            )
    elif mode == "embed":
        # Encoder is sufficient.
        pass
    else:
        errors.append(f"unknown mode '{mode}'")

    return errors


def validate(mode: str, ckpt_path: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "model_type": "unknown",
        "has_encoder": False,
        "has_vocab_head": False,
        "has_contrast_head": False,
        "has_task_ffn": False,
        "task_output_dims": [],
        "vocab_sizes": {"atom": None, "bond": None, "smiles": None},
        "arch": {k: None for k in ARCH_KEYS},
        "saved_args": None,
        "errors": [],
        "warnings": [],
    }

    # 1. Load the checkpoint.
    try:
        ckpt = load_checkpoint(ckpt_path)
    except FileNotFoundError:
        result["errors"].append(f"checkpoint not found: {ckpt_path}")
        return result
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"failed to load checkpoint {ckpt_path}: {type(exc).__name__}: {exc}")
        return result

    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        result["errors"].append(
            "checkpoint is not in the expected save_model_for_restart format "
            "(expected a dict with a 'state_dict' key)."
        )
        return result

    state_dict = _strip_ddp_prefix(ckpt["state_dict"])
    args_obj = ckpt.get("args")

    # 2. Classify and check mode contract.
    classification = _classify_model(state_dict)
    result.update(classification)

    contract_errors = _apply_mode_contract(mode, classification)
    result["errors"].extend(contract_errors)

    # 3. Task output dims (for inference / informational).
    if classification["has_task_ffn"]:
        result["task_output_dims"] = _task_output_dims(state_dict)

    # 3b. Vocab head sizes (for continue-pretrain vocab-size verification).
    result["vocab_sizes"] = _vocab_sizes(state_dict)

    # 4. Arch derivation: args first, shape introspection for what's still missing.
    arch = _arch_from_args(args_obj)
    arch, shape_warnings = _arch_from_shapes(state_dict, arch)
    result["arch"] = arch
    result["warnings"].extend(shape_warnings)

    # 5. Saved args as serializable dict (best-effort).
    if args_obj is not None:
        try:
            result["saved_args"] = vars(args_obj) if isinstance(args_obj, Namespace) else dict(args_obj)
            # Drop non-JSON-serializable values; agent skill only needs human-readable scalars.
            result["saved_args"] = {
                k: v for k, v in result["saved_args"].items()
                if isinstance(v, (str, int, float, bool, type(None), list, dict))
            }
        except Exception as exc:  # noqa: BLE001
            result["warnings"].append(f"could not serialize saved_args: {type(exc).__name__}: {exc}")

    result["ok"] = not result["errors"]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a KERMT checkpoint for a given workflow.")
    parser.add_argument("--mode", required=True,
                        choices=["continue_pretrain", "upgrade_to_hybrid", "finetune_init", "inference", "embed"])
    parser.add_argument("--ckpt", required=True, help="Path to the .pt checkpoint")
    args = parser.parse_args(argv)

    try:
        result = validate(args.mode, args.ckpt)
    except Exception as exc:  # noqa: BLE001
        # Last-resort safety net: keep stdout JSON-clean, dump trace to stderr.
        print(traceback.format_exc(), file=sys.stderr)
        print(json.dumps({
            "ok": False,
            "model_type": "unknown",
            "errors": [f"unhandled exception in validator: {type(exc).__name__}: {exc}"],
            "warnings": [],
            "arch": {k: None for k in ARCH_KEYS},
            "has_encoder": False,
            "has_vocab_head": False,
            "has_contrast_head": False,
            "has_task_ffn": False,
            "task_output_dims": [],
            "vocab_sizes": {"atom": None, "bond": None, "smiles": None},
            "saved_args": None,
        }, indent=2))
        return 1

    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
