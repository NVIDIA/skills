#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workstation inference runner — wraps `main.py predict` (which calls
task/predict.py::make_predictions) and emits a reproducible `run.json` manifest.

Inputs:
  - finetuned ckpt (must have task FFN heads; validator refuses pretrain ckpts)
  - prepare_data manifest (mode=inference)
  - output dir

`main.py predict` requires a checkpoint *directory* (--checkpoint_dir walks it
to find every .pt) rather than a single --checkpoint_path. The runner sidesteps
that by symlinking the user's ckpt as `<out>/ckpt_link/model.pt` and passing
`--checkpoint_dir <out>/ckpt_link` — `task/predict.py` then loads just the one
ckpt. No modification to the existing parsing.py needed.

Blocking-by-default. Inference is minutes-scale; no detach machinery here.

CLI
---
    run_inference.py
        --ckpt <finetuned.pt>           # required
        --prepare-manifest <path>       # prepare_data.json (mode=inference)
        --out <run-dir>                 # output dir
        [--ckpt-validator-out <path>]   # cached check_checkpoint.py JSON
        [--gpus 0]                      # single GPU id (default 0)
        [--batch-size N]                # override defaults_inference.runtime.batch_size
        [--seed N]
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
DEFAULTS_PATH = SKILL_ROOT / "config" / "defaults_inference.json"
CHECK_CHECKPOINT_PATH = SKILL_ROOT / "scripts" / "check_checkpoint.py"
MAIN_PY_PATH = REPO_ROOT / "main.py"

RUNTIME_FLAGS = ("batch_size", "seed")


def _verify_prepare_manifest(manifest: dict[str, Any]) -> None:
    assert_prepare_manifest_basics(manifest, "inference")
    out = manifest.get("outputs", {})
    if "clean_csv" not in out:
        raise ValueError(
            "prepare_data manifest is missing required output 'clean_csv'. "
            "Was prepare_data run successfully?"
        )


def _apply_defaults(args: argparse.Namespace, defaults: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Merges defaults_inference.json with CLI overrides into args_applied."""
    applied: dict[str, dict[str, Any]] = {}
    runtime = defaults.get("runtime", {})

    for f in RUNTIME_FLAGS:
        merge_default_into_applied(applied, args, f, runtime)

    return applied


def _link_ckpt_into_dir(user_ckpt: Path, link_dir: Path) -> Path:
    """Symlink the user ckpt into a fresh subdir so we can pass --checkpoint_dir
    to main.py predict (it doesn't expose --checkpoint_path).
    The dir is exclusive to this run (under out_dir), so no other .pt sneaks in."""
    link_dir.mkdir(parents=True, exist_ok=True)
    # Clean prior contents (e.g. from a previous run reusing the same out dir).
    for prior in link_dir.iterdir():
        if prior.is_symlink() or prior.is_file():
            prior.unlink()
    link = link_dir / "model.pt"
    link.symlink_to(user_ckpt.resolve())
    return link


def _build_argv(
    *, gpu: int, out_dir: Path, manifest: dict[str, Any], ckpt_dir: Path,
    output_csv: Path, applied: dict[str, dict[str, Any]],
) -> list[str]:
    """Constructs the `main.py predict` argv."""
    outputs = manifest["outputs"]
    argv: list[str] = [sys.executable, "-u", str(MAIN_PY_PATH), "predict"]

    argv += ["--data_path", outputs["clean_csv"]]
    argv += ["--output_path", str(output_csv)]
    argv += ["--checkpoint_dir", str(ckpt_dir)]
    if "clean_npz" in outputs:
        argv += ["--features_path", outputs["clean_npz"]]
    argv += ["--gpu", str(gpu)]

    if "batch_size" in applied:
        argv += ["--batch_size", str(applied["batch_size"]["value"])]
    if "seed" in applied:
        argv += ["--seed", str(applied["seed"]["value"])]

    return argv


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "out").mkdir(parents=True, exist_ok=True)
    ckpt_link_dir = out_dir / "ckpt_link"

    # 1. Load defaults + prepare manifest.
    defaults = load_json(DEFAULTS_PATH, name="defaults_inference.json")
    prep_manifest_path = Path(args.prepare_manifest).resolve()
    manifest = load_json(prep_manifest_path, name="prepare_data.json")
    _verify_prepare_manifest(manifest)

    # 2. Validate ckpt — must be finetuned.
    ckpt = Path(args.ckpt).resolve()
    if args.ckpt_validator_out:
        validator_out = load_json(Path(args.ckpt_validator_out), name="ckpt validator output")
    else:
        validator_out = run_checkpoint_validator(ckpt, mode="inference", script_path=CHECK_CHECKPOINT_PATH)
    if not validator_out.get("ok"):
        raise ValueError(
            f"check_checkpoint.py rejected the input ckpt: {validator_out.get('errors')}. "
            "Inference requires a finetuned ckpt with task FFN heads; for a pretrain ckpt "
            "use kermt-finetune first."
        )

    model_type = validator_out.get("model_type")
    arch = validator_out.get("arch", {})
    task_output_dims = validator_out.get("task_output_dims", [])

    # 3. GPU + defaults.
    gpu = resolve_single_gpu(args.gpus, workflow="inference")
    applied = _apply_defaults(args, defaults)

    # 4. Symlink the ckpt + build argv.
    output_csv = out_dir / "out" / "predictions.csv"
    if not args.dry_run:
        _link_ckpt_into_dir(ckpt, ckpt_link_dir)
    else:
        # In dry-run we still record where the link would go for replayability.
        ckpt_link_dir.mkdir(parents=True, exist_ok=True)
    argv = _build_argv(
        gpu=gpu, out_dir=out_dir, manifest=manifest, ckpt_dir=ckpt_link_dir,
        output_csv=output_csv, applied=applied,
    )

    # 5. Build the run.json manifest.
    commit, dirty = git_commit_with_env_override(REPO_ROOT)
    image_tag = os.environ.get("KERMT_IMAGE", "kermt:latest")
    image_digest = docker_image_digest(image_tag)
    cmd_replay = format_cmd_replay(argv, env={"CUDA_VISIBLE_DEVICES": gpu})
    run_manifest: dict[str, Any] = {
        "workflow": "inference",
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
        "task_output_dims": task_output_dims,
        "output_csv": str(output_csv),
        "logs_dir": str(out_dir / "logs"),
        "ckpt_link_dir": str(ckpt_link_dir),
        "argv": argv,
        "cmd_replay": cmd_replay,
        "ok_to_replay": (not dirty) and (commit != "unknown"),
        "dry_run": bool(args.dry_run),
    }
    (out_dir / "run.json").write_text(json.dumps(run_manifest, indent=2))

    if args.dry_run:
        run_manifest["status"] = "dry_run"
        return run_manifest

    # 6. Execute.
    env = runner_environment(REPO_ROOT)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # main.py enables strict deterministic algorithms via
    # `torch.use_deterministic_algorithms(True)` (kermt/main.py:23); CuBLAS
    # then requires this env var to be set. Default to ':4096:8' (slightly
    # more memory than ':16:8' but allows larger matmuls).
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    log_file = out_dir / "logs" / "inference.log"
    with log_file.open("w") as logf:
        proc = subprocess.run(argv, env=env, stdout=logf, stderr=subprocess.STDOUT)
    run_manifest["exit_code"] = proc.returncode
    run_manifest["status"] = "ok" if proc.returncode == 0 else "failed"
    (out_dir / "run.json").write_text(json.dumps(run_manifest, indent=2))
    return run_manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Workstation inference runner — wraps main.py predict via subprocess."
    )
    p.add_argument("--ckpt", required=True, help="Path to a finetuned ckpt (with task FFN heads).")
    p.add_argument("--prepare-manifest", required=True,
                   help="Path to a prepare_data.json produced with --mode inference.")
    p.add_argument("--out", required=True, help="Output run directory.")
    p.add_argument("--ckpt-validator-out", default=None,
                   help="Optional cached check_checkpoint.py JSON.")
    p.add_argument("--gpus", default=None, help="Single GPU id (default 0). Multi-GPU rejected.")
    p.add_argument("--dry-run", action="store_true",
                   help="Write run.json + print the command without executing.")

    # Runtime overrides (defaults None so source attribution stays accurate).
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
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
