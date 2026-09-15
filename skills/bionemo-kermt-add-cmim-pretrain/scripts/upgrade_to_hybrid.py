#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a grover_base ckpt into a hybrid ckpt by adding a
randomly-initialized cMIM decoder + latent_dist on top of the loaded encoder.

After upgrade, the saved checkpoint classifies as `model_type: hybrid` via
`check_checkpoint.py --mode continue_pretrain`, and the pretrain runner
(`run_pretrain_local.py`) drops it directly into the
`--pretrain_mode hybrid` dispatch path with no special-case logic.

What's preserved from the input ckpt
------------------------------------
- Encoder weights (kermt.encoders.* state-dict subset). For legacy
  grover_base ckpts where the encoder lives under `grover.encoders.*`,
  the prefix is renormalized to `kermt.encoders.*` to match the
  KermtHybridTask layout.
- The saved `args` Namespace — augmented with hybrid-specific decoder fields
  if absent (defaults from `config/defaults_pretrain.json`).

What's fresh-initialized
------------------------
- The cMIM decoder (`decoder.*` — SMILESTransformerDecoder) — Xavier-init.
- The latent distribution (`latent_dist.*` minus `latent_dist.kermt.*` —
  the encoder lives inside latent_dist by reference, so it shares the
  same weights as the top-level encoder).
- The vocab heads (`vocab_module.*`) — always fresh, sized to the
  vocab built from the user's pretrain corpus. The previous heads (if the
  input ckpt was a modern grover_base with vocab heads) are discarded —
  rebuilding them is acceptable because continue-pretrain will retrain
  them anyway, and matching the new vocab dimensions is more important
  than warm-starting from the old vocab.

CLI
---
    upgrade_to_hybrid.py
        --ckpt <input.pt>             # grover_base ckpt (encoder-only or
                                      # encoder + vocab heads; legacy or modern)
        --prepare-manifest <path>     # prepare_data.json from a prior
                                      # `prepare_data.py --mode pretrain` run.
                                      # The smiles_vocab + atom_vocab +
                                      # bond_vocab from this manifest size
                                      # the new heads.
        --out <upgraded.pt>           # destination for the upgraded ckpt
        [--latent-dim N]              # override defaults_pretrain.json
        [--seed 0]                    # for reproducible random init

Exit code 0 + a one-line JSON summary on success. Exit code 1 + JSON-shaped
errors on failure (input rejection, vocab missing, shape mismatch, etc.).
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

# sys.path tweak so `_utils` imports cleanly whether launched via `kermt_run`
# or as a bare `python scripts/upgrade_to_hybrid.py …`.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import count_vocab_entries, load_checkpoint, load_json, run_checkpoint_validator  # noqa: E402


SKILL_ROOT = Path(__file__).resolve().parent.parent
DEFAULTS_PATH = SKILL_ROOT / "config" / "defaults_pretrain.json"
CHECK_CHECKPOINT_PATH = SKILL_ROOT / "scripts" / "check_checkpoint.py"

# Fields KermtHybridTask + its sub-components read from args. Anything not
# present on the input ckpt's args gets filled from defaults_pretrain.json
# (decoder fields from add_cmim_decoder group; encoder fields from training
# group; the rest from hardcoded encoder-arch defaults below).
HYBRID_REQUIRED_DECODER_ARGS = (
    "decoder_num_layers", "decoder_num_attention_heads", "decoder_ffn_hidden_size",
    "decoder_dropout", "decoder_max_seq_len", "decoder_positional_encoding",
    "decoder_gate_self_attn", "decoder_gate_cross_attn",
)

# Fields KERMTEmbedding (and its sub-modules) read at __init__ time. Legacy
# grover_base ckpts may be missing several of these — pre-cMIM args weren't a
# strict superset of what current KERMTEmbedding wants.
ENCODER_REQUIRED_ARGS_WITH_DEFAULTS = {
    "dropout": 0.1,                      # not in legacy grover_base.pt
    "bond_drop_rate": 0.0,
    "self_attention": False,
    "attn_hidden": 4,
    "attn_out": 8,
    "use_cuikmolmaker_featurization": False,
    "input_layer": "fc",
    "dense": False,
    "bias": False,
    "undirected": False,
    "dist_coff": 0.1,
    "num_mt_block": 1,
    "embedding_output_type": "both",
}


# JSON loading + vocab counting delegated to _utils. Aliases kept for the
# existing internal callsites.
_load_json = load_json
_count_vocab = count_vocab_entries


def _run_validator(ckpt: Path) -> dict[str, Any]:
    """Invoke check_checkpoint.py --mode upgrade_to_hybrid and return the JSON.
    Thin alias around _utils.run_checkpoint_validator for the specific mode."""
    return run_checkpoint_validator(ckpt, mode="upgrade_to_hybrid", script_path=CHECK_CHECKPOINT_PATH)


def _augment_args(input_args: Namespace, decoder_defaults: dict[str, Any], latent_dim: int) -> Namespace:
    """Make sure the args Namespace has every field KermtHybridTask reads.
    Existing input args take precedence (the ckpt was trained with them, and
    encoder arch needs them); only missing hybrid-specific fields are filled
    from defaults_pretrain.json's add_cmim_decoder group."""
    out = Namespace(**vars(input_args)) if isinstance(input_args, Namespace) else Namespace(**input_args)
    # Encoder-side fields KERMTEmbedding reads at __init__. Legacy grover_base
    # ckpts may be missing several (e.g. `dropout`). Fill from hardcoded
    # defaults — these are safe at upgrade time because the loaded weights
    # determine the actual encoder behavior; missing dropout etc. only affects
    # rebuild/forward semantics, which the continue-pretrain runner will
    # override anyway via its own training defaults.
    for field, default_value in ENCODER_REQUIRED_ARGS_WITH_DEFAULTS.items():
        if not hasattr(out, field):
            setattr(out, field, default_value)
    # Hybrid-specific decoder fields (only fill in if missing)
    for field in HYBRID_REQUIRED_DECODER_ARGS:
        if not hasattr(out, field):
            setattr(out, field, decoder_defaults[field])
    # latent_dim and contrastive_temperature also fill in from defaults if absent
    if not hasattr(out, "latent_dim"):
        out.latent_dim = latent_dim
    if not hasattr(out, "contrastive_temperature"):
        out.contrastive_temperature = decoder_defaults.get("contrastive_temperature", 0.1)
    # pretrain_mode and use_cmim flags — set explicitly to hybrid semantics
    out.pretrain_mode = "hybrid"
    if hasattr(out, "use_cmim"):
        out.use_cmim = True
    # vocab_loss_weight default from training/loss defaults (1.0 — set inside the runner;
    # here we just need it on the Namespace so save_checkpoint records it)
    if not hasattr(out, "vocab_loss_weight"):
        out.vocab_loss_weight = 1.0
    return out


def _renormalize_encoder_keys(state_dict: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Extract the encoder subset of the input state_dict, renormalized so the
    keys match the KermtHybridTask layout (`kermt.encoders.*`).

    Three input shapes are handled:
      - `grover.encoders.*` (legacy original-GROVER ckpts) → rename to `kermt.encoders.*`
      - `kermt.encoders.*`  (modern repo-trained grover_base / hybrid) → keep as-is
      - `latent_dist.kermt.encoders.*` (cmim ckpts) → reject — those should
        go through `continue_pretrain`, not `upgrade_to_hybrid`.

    Vocab heads (`vocab_module.*`) and any other non-encoder keys are
    DROPPED here — the upgraded hybrid task gets fresh vocab heads sized to
    the user's prepare-data vocab.

    Returns (renormalized_state_dict, notes). `notes` is a list of strings
    describing key handling for the manifest.
    """
    out: dict[str, Any] = {}
    notes: list[str] = []
    legacy_count = modern_count = vocab_dropped = other_dropped = 0

    for k, v in state_dict.items():
        if k.startswith("latent_dist.kermt."):
            raise ValueError(
                "input ckpt has `latent_dist.kermt.*` keys — it appears to be a "
                "cmim or hybrid ckpt, not a grover_base. Use kermt-continue-pretrain "
                "(no upgrade needed) or kermt-pretrain-scratch instead."
            )
        if k.startswith("grover.encoders."):
            new_k = "kermt.encoders." + k[len("grover.encoders."):]
            out[new_k] = v
            legacy_count += 1
        elif k.startswith("kermt.encoders."):
            out[k] = v
            modern_count += 1
        elif k.startswith("vocab_module."):
            vocab_dropped += 1  # fresh heads on the new hybrid
        else:
            other_dropped += 1

    if legacy_count and modern_count:
        notes.append(f"unexpected mix of legacy + modern encoder prefixes: "
                     f"{legacy_count} grover.encoders.* + {modern_count} kermt.encoders.*")
    elif legacy_count:
        notes.append(f"renamed {legacy_count} legacy `grover.encoders.*` keys to `kermt.encoders.*`")
    elif modern_count:
        notes.append(f"kept {modern_count} modern `kermt.encoders.*` keys as-is")
    else:
        raise ValueError("no encoder state-dict keys found in input ckpt "
                         "(expected `grover.encoders.*` or `kermt.encoders.*`)")
    if vocab_dropped:
        notes.append(f"dropped {vocab_dropped} `vocab_module.*` keys "
                     f"(new vocab heads sized to user's prepare-data vocab)")
    if other_dropped:
        notes.append(f"dropped {other_dropped} other keys (not encoder, not vocab)")
    return out, notes


def upgrade(args: argparse.Namespace) -> dict[str, Any]:
    """Main upgrade flow. Returns a dict summary suitable for printing as JSON."""
    summary: dict[str, Any] = {
        "ok": False,
        "input_ckpt": str(Path(args.ckpt).resolve()),
        "output_ckpt": str(Path(args.out).resolve()),
        "vocab_sizes_used": {"atom": None, "bond": None, "smiles": None},
        "notes": [],
        "errors": [],
        "warnings": [],
    }

    # 1. Validate input via check_checkpoint --mode upgrade_to_hybrid.
    validator = _run_validator(Path(args.ckpt))
    if not validator.get("ok"):
        summary["errors"].extend(validator.get("errors", []) or ["check_checkpoint rejected the ckpt"])
        return summary
    summary["input_model_type"] = validator.get("model_type")

    # 2. Read prepare manifest for vocab paths + sizes.
    prep_path = Path(args.prepare_manifest)
    prep = _load_json(prep_path, name="prepare_data.json")
    if prep.get("mode") != "pretrain":
        raise ValueError(f"prepare manifest mode='{prep.get('mode')}', expected 'pretrain'")
    if not prep.get("ok"):
        raise ValueError(f"prepare manifest reports ok=False: {prep.get('errors')}")
    outs = prep.get("outputs", {})
    for required in ("atom_vocab", "bond_vocab", "smiles_vocab"):
        if required not in outs:
            raise ValueError(
                f"prepare manifest missing {required}. kermt-add-cmim-pretrain needs the "
                "smiles vocab to size the new decoder; pass a corpus through "
                "`prepare_data.py --mode pretrain` (without --skip-vocab / --skip-features) first."
            )
    atom_size = _count_vocab(Path(outs["atom_vocab"]))
    bond_size = _count_vocab(Path(outs["bond_vocab"]))
    smiles_size = _count_vocab(Path(outs["smiles_vocab"]))
    summary["vocab_sizes_used"] = {"atom": atom_size, "bond": bond_size, "smiles": smiles_size}

    # 3. Load defaults + input ckpt.
    defaults = _load_json(DEFAULTS_PATH, name="defaults_pretrain.json")
    decoder_defaults = defaults.get("add_cmim_decoder") or {}
    latent_dim = args.latent_dim if args.latent_dim is not None else decoder_defaults.get("latent_dim", 800)

    input_ckpt = load_checkpoint(args.ckpt)
    input_args = input_ckpt.get("args")
    if input_args is None:
        raise ValueError(
            "input ckpt has no `args` Namespace — cannot reconstruct the encoder architecture. "
            "Original GROVER ckpts typically saved args; if this one didn't, the upgrade has no "
            "way to recover hidden_size / depth / num_attn_head / etc."
        )
    input_sd = input_ckpt["state_dict"]

    # 4. Strip DDP `module.` prefix if present.
    if input_sd and all(k.startswith("module.") for k in input_sd):
        input_sd = {k[len("module."):]: v for k, v in input_sd.items()}
        summary["notes"].append("stripped DDP `module.` prefix from input state_dict")

    # 5. Filter input state_dict down to just the encoder keys. Vocab heads
    #    are intentionally dropped here — the new hybrid task (built in step 7)
    #    provides freshly-initialized vocab heads sized to the user's vocab,
    #    which is safer than carrying over heads sized to the input ckpt's
    #    (possibly different) vocab.
    encoder_state, rename_notes = _renormalize_encoder_keys(input_sd)
    summary["notes"].extend(rename_notes)

    # 6. Augment args with hybrid-specific decoder fields.
    new_args = _augment_args(input_args, decoder_defaults, latent_dim)
    # Ensure cuda flag is False for the construction step (we don't move to GPU here).
    new_args.cuda = False

    # 7. Build the fresh KermtHybridTask.
    torch.manual_seed(args.seed)
    from kermt.model.models import KermtHybridTask, KERMTEmbedding  # type: ignore
    encoder = KERMTEmbedding(new_args)
    hybrid_task = KermtHybridTask(
        new_args, kermt=encoder,
        latent_dim=latent_dim,
        contrastive_temperature=new_args.contrastive_temperature,
        smiles_vocab_size=smiles_size,
        atom_vocab_size=atom_size,
        bond_vocab_size=bond_size,
    )

    # 8. Load encoder weights into the new task. strict=False because decoder /
    # latent_dist / vocab_module weren't in the input — they keep their fresh init.
    load_result = hybrid_task.load_state_dict(encoder_state, strict=False)
    missing_keys = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)
    # The encoder is shared between hybrid_task.kermt and hybrid_task.latent_dist.kermt;
    # any `kermt.encoders.*` key that wasn't loaded into the latent_dist's encoder copy
    # is benign because they're the same module by reference. Same for missing-keys that
    # belong to decoder / latent_dist (non-kermt parts) / vocab_module — those are
    # supposed to be fresh.
    summary["encoder_load"] = {
        "missing_keys_total": len(missing_keys),
        "unexpected_keys_total": len(unexpected_keys),
        "encoder_missing": [k for k in missing_keys if "encoders." in k][:5],
        "non_encoder_missing_categories": _categorize_missing(missing_keys),
        "unexpected_sample": unexpected_keys[:5],
    }
    if unexpected_keys:
        summary["warnings"].append(
            f"{len(unexpected_keys)} unexpected key(s) in encoder load — "
            "likely arch drift between legacy GROVER and modern KERMTEmbedding."
        )

    # Surface unknown backbones (anything beyond the gtrans/dualtrans pair
    # that pretrain_ddp.py's argparse + the model code both accept). The
    # legacy `dualtrans` name is the same architecture as `gtrans` and is
    # explicitly handled in kermt/model/models.py:200.
    if getattr(new_args, "backbone", None) not in (None, "gtrans", "dualtrans"):
        summary["warnings"].append(
            f"upgraded ckpt has backbone='{new_args.backbone}', which is neither "
            "'gtrans' nor 'dualtrans'. pretrain_ddp.py's --backbone argparse will reject "
            "it. Either re-pretrain a gtrans grover_base from scratch via "
            "kermt-pretrain-scratch, or extend the parsing.py:379 choices."
        )

    # 9. Build a fresh Adam optimizer over the new model (matches save_checkpoint format).
    init_lr = getattr(new_args, "init_lr", 1e-5)
    weight_decay = getattr(new_args, "weight_decay", 1e-7)
    optimizer = torch.optim.Adam(hybrid_task.parameters(), lr=init_lr, weight_decay=weight_decay)

    # 10. Save in the save_checkpoint format that task/kermttrainer.py:save_checkpoint produces.
    state = {
        "args": new_args,
        "state_dict": hybrid_task.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler_step": 0,
        "batch_idx": 0,
        "epoch": 0,
        "data_scaler": None,
        "features_scaler": None,
        "wandb_run_id": None,
    }
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_path)

    summary["ok"] = True
    summary["upgraded_state_dict_keys"] = len(state["state_dict"])
    return summary


def _categorize_missing(missing_keys: list[str]) -> dict[str, int]:
    """Bucket missing keys by their state-dict prefix for the manifest."""
    cats: dict[str, int] = {}
    for k in missing_keys:
        first = k.split(".")[0]
        cats[first] = cats.get(first, 0) + 1
    return cats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Upgrade a grover_base ckpt to a hybrid (cMIM + vocab) ckpt.")
    p.add_argument("--ckpt", required=True, help="Path to the input grover_base checkpoint")
    p.add_argument("--prepare-manifest", required=True,
                   help="Path to a prepare_data.json (mode=pretrain) — its vocab sizes "
                        "size the new decoder + fresh vocab heads.")
    p.add_argument("--out", required=True, help="Destination for the upgraded hybrid ckpt")
    p.add_argument("--latent-dim", type=int, default=None,
                   help="Override defaults_pretrain.json's add_cmim_decoder.latent_dim (default 800)")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for the fresh decoder / latent / vocab head weights")
    args = p.parse_args(argv)

    try:
        summary = upgrade(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(json.dumps({"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}, indent=2))
        return 1
    except Exception as exc:  # noqa: BLE001
        print(traceback.format_exc(), file=sys.stderr)
        print(json.dumps({"ok": False, "errors": [f"unhandled: {type(exc).__name__}: {exc}"]}, indent=2))
        return 1

    print(json.dumps(summary, indent=2))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
