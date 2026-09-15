#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workstation embedding-extraction runner — wraps task/extract_embeddings.py
and emits a reproducible `run.json` manifest alongside the outputs.

Accepts any encoder-bearing ckpt (grover_base / cmim / hybrid / finetuned).
Validates via check_checkpoint.py --mode embed (encoder-only sufficient).
Reads the prepare_data manifest (mode=embed: clean CSV only — featurization
happens on the fly inside extract_embeddings.py).

Output layout:
    <out>/out/atom_from_atom.npy
    <out>/out/bond_from_atom.npy
    <out>/out/atom_from_bond.npy
    <out>/out/bond_from_bond.npy
    <out>/out/canonical_smiles.npy
    <out>/out/validity.npy

Blocking-by-default. Embedding extraction is minutes-scale.

CLI
---
    run_extract_embeddings.py
        --ckpt <encoder-bearing.pt>   # required
        --prepare-manifest <path>     # prepare_data.json (mode=embed)
        --out <run-dir>               # output dir
        [--ckpt-validator-out <path>]
        [--gpus 0]                    # single GPU id (default 0)
        [--batch-size N]              # override defaults_embed.runtime.batch_size
        [--dry-run]
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

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import (  # noqa: E402
    resolve_kermt_repo, assert_prepare_manifest_basics, docker_image_digest, format_cmd_replay,
    git_commit_with_env_override, load_json, merge_default_into_applied,
    resolve_single_gpu, run_checkpoint_validator, runner_environment,
)


REPO_ROOT = resolve_kermt_repo()
SKILL_ROOT = Path(__file__).resolve().parent.parent
DEFAULTS_PATH = SKILL_ROOT / "config" / "defaults_embed.json"
CHECK_CHECKPOINT_PATH = SKILL_ROOT / "scripts" / "check_checkpoint.py"
EXTRACT_EMBEDDINGS_PATH = REPO_ROOT / "task" / "extract_embeddings.py"


def _verify_prepare_manifest(manifest: dict[str, Any]) -> None:
    assert_prepare_manifest_basics(manifest, "embed")
    out = manifest.get("outputs", {})
    if "clean_csv" not in out:
        raise ValueError(
            "prepare_data manifest is missing required output 'clean_csv'. "
            "Was prepare_data run successfully?"
        )


def _apply_defaults(args: argparse.Namespace, defaults: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Merge defaults_embed.json with CLI overrides."""
    applied: dict[str, dict[str, Any]] = {}
    runtime = defaults.get("runtime", {})

    merge_default_into_applied(applied, args, "batch_size", runtime)

    return applied


def _build_argv(
    *, gpu: int, ckpt: Path, manifest: dict[str, Any], out_dir: Path,
    applied: dict[str, dict[str, Any]],
) -> list[str]:
    """Constructs the task/extract_embeddings.py argv. Note: extract_embeddings
    uses its own CLI (--checkpoint, --input_file, --output_path) — NOT main.py.
    --format defaults to npy inside extract_embeddings.py, so we don't pass it."""
    outputs_dir = out_dir / "out"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    argv: list[str] = [sys.executable, "-u", str(EXTRACT_EMBEDDINGS_PATH)]
    argv += ["--checkpoint", str(ckpt)]
    argv += ["--input_file", manifest["outputs"]["clean_csv"]]
    argv += ["--output_path", str(outputs_dir)]
    argv += ["--device", "cuda"]
    # clean_smiles.py preserves the input CSV's column layout, so the cleaned
    # CSV has SMILES at whichever column the input had. Forward the
    # auto-detected smiles_column from prepare_data.json so extract_embeddings
    # doesn't fall back to its default of column 0 (which would index a
    # non-SMILES column for inputs like openadmet/all.csv where SMILES is at
    # column 1). Default to 0 if the manifest is from an older prepare_data
    # version that didn't record the field.
    argv += ["--smiles_column", str(manifest.get("smiles_column", 0))]
    if "batch_size" in applied:
        argv += ["--batch_size", str(applied["batch_size"]["value"])]

    return argv


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "out").mkdir(parents=True, exist_ok=True)

    defaults = load_json(DEFAULTS_PATH, name="defaults_embed.json")
    prep_manifest_path = Path(args.prepare_manifest).resolve()
    manifest = load_json(prep_manifest_path, name="prepare_data.json")
    _verify_prepare_manifest(manifest)

    ckpt = Path(args.ckpt).resolve()
    if args.ckpt_validator_out:
        validator_out = load_json(Path(args.ckpt_validator_out), name="ckpt validator output")
    else:
        validator_out = run_checkpoint_validator(ckpt, mode="embed", script_path=CHECK_CHECKPOINT_PATH)
    if not validator_out.get("ok"):
        raise ValueError(
            f"check_checkpoint.py rejected the input ckpt: {validator_out.get('errors')}"
        )

    model_type = validator_out.get("model_type")
    arch = validator_out.get("arch", {})

    gpu = resolve_single_gpu(args.gpus, workflow="embed")
    applied = _apply_defaults(args, defaults)

    argv = _build_argv(gpu=gpu, ckpt=ckpt, manifest=manifest, out_dir=out_dir, applied=applied)

    commit, dirty = git_commit_with_env_override(REPO_ROOT)
    image_tag = os.environ.get("KERMT_IMAGE", "kermt:latest")
    image_digest = docker_image_digest(image_tag)
    cmd_replay = format_cmd_replay(argv, env={"CUDA_VISIBLE_DEVICES": gpu})
    run_manifest: dict[str, Any] = {
        "workflow": "embed",
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "container": {"image_tag": image_tag, "image_digest": image_digest},
        "repo": {"commit": commit, "dirty": dirty},
        "inputs": {
            "ckpt": str(ckpt),
            "prepare_data_manifest": str(prep_manifest_path),
            "ckpt_validator_out": (
                str(Path(args.ckpt_validator_out).resolve()) if args.ckpt_validator_out else None
            ),
        },
        "model_type": model_type,
        "gpu": gpu,
        "args_applied": applied,
        "arch": arch,
        "output_dir": str(out_dir / "out"),
        "logs_dir": str(out_dir / "logs"),
        "argv": argv,
        "cmd_replay": cmd_replay,
        "ok_to_replay": (not dirty) and (commit != "unknown"),
        "dry_run": bool(args.dry_run),
    }
    (out_dir / "run.json").write_text(json.dumps(run_manifest, indent=2))

    if args.dry_run:
        run_manifest["status"] = "dry_run"
        return run_manifest

    env = runner_environment(REPO_ROOT)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_file = out_dir / "logs" / "embed.log"
    with log_file.open("w") as logf:
        proc = subprocess.run(argv, env=env, stdout=logf, stderr=subprocess.STDOUT)
    run_manifest["exit_code"] = proc.returncode
    run_manifest["status"] = "ok" if proc.returncode == 0 else "failed"
    (out_dir / "run.json").write_text(json.dumps(run_manifest, indent=2))
    return run_manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Workstation embedding-extraction runner — wraps task/extract_embeddings.py."
    )
    p.add_argument("--ckpt", required=True, help="Path to encoder-bearing ckpt (any model_type).")
    p.add_argument("--prepare-manifest", required=True,
                   help="Path to a prepare_data.json produced with --mode embed.")
    p.add_argument("--out", required=True, help="Output run directory.")
    p.add_argument("--ckpt-validator-out", default=None,
                   help="Optional cached check_checkpoint.py JSON.")
    p.add_argument("--gpus", default=None, help="Single GPU id (default 0). Multi-GPU rejected.")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--batch-size", type=int, default=None)

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
