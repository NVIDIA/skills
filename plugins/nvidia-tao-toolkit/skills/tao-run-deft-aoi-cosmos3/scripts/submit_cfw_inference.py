#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Compose or execute a Docker submit for native Framework inference."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import pwd
import shlex
import subprocess
import sys
from typing import Any

from submit_cfw_train import default_bank, resolve_image


def _mount(host: pathlib.Path, container: str, *, readonly: bool) -> str:
    return f"type=bind,src={host},dst={container}" + (",readonly" if readonly else "")


def _under(path: pathlib.Path, root: pathlib.Path, label: str) -> pathlib.Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be under --workspace: {resolved}") from exc
    return resolved


def _is_dcp(path: pathlib.Path) -> bool:
    return (path / ".metadata").is_file() or (path / "model/.metadata").is_file()


def build_docker_argv(
    *,
    immutable_image: str,
    job_id: str,
    workspace: pathlib.Path,
    results_dir: pathlib.Path,
    model_path: pathlib.Path,
    media: pathlib.Path,
    prompt: str,
    media_type: str,
    max_new_tokens: int,
    framework_config: pathlib.Path | None,
    action_model_dir: pathlib.Path | None,
    base_model_path: pathlib.Path | None,
    gpus: str,
    offline: bool,
) -> list[str]:
    username = pwd.getpwuid(os.getuid()).pw_name
    command = [
        "docker", "run", "-d", "--name", job_id,
        "--label", f"tao-job={job_id}", "--gpus", gpus,
        "--shm-size=8g", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", f"USER={username}", "-e", f"LOGNAME={username}", "-e", "HOME=/tmp",
        "-e", f"TAO_JOB_ID={job_id}", "-e", f"TAO_RESULTS_ROOT={results_dir}",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-v", "/etc/passwd:/etc/passwd:ro", "-v", "/etc/group:/etc/group:ro",
        "--mount", _mount(workspace, str(workspace), readonly=True),
        "--mount", _mount(results_dir, str(results_dir), readonly=False),
        "-w", str(results_dir / "cwd"),
    ]
    if action_model_dir is not None and action_model_dir.parent != results_dir:
        command.extend([
            "--mount",
            _mount(action_model_dir.parent, str(action_model_dir.parent), readonly=False),
        ])
    if offline:
        command.extend(["-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1"])
    command.extend([
        "--entrypoint", "/workspace/.venv/bin/cosmos-framework-inference",
        immutable_image,
        "--model_path", str(model_path),
        "--torch_dtype", "bfloat16",
        "--device_map", "auto",
        "--num_gpus", "1",
        "--type", media_type,
        "--media", str(media),
        "--prompt", prompt,
        "--max_new_tokens", str(max_new_tokens),
        "--results_dir", str(results_dir),
        "--enable_lora", "false",
    ])
    if framework_config is not None:
        command.extend(["--config_file", str(framework_config)])
    if action_model_dir is not None:
        command.extend(["--export_dir", str(action_model_dir)])
    if base_model_path is not None:
        command.extend(["--vit_checkpoint_path", str(base_model_path)])
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-bank", type=pathlib.Path, default=default_bank())
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--workspace", required=True, type=pathlib.Path)
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--model-path", required=True, type=pathlib.Path)
    parser.add_argument("--media", required=True, type=pathlib.Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--type", choices=("image", "video"), default="image")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--framework-config",
        type=pathlib.Path,
        help=(
            "Framework Train's saved Hydra config.yaml beside the DCP; "
            "never the input SFT TOML."
        ),
    )
    parser.add_argument("--action-model-dir", type=pathlib.Path)
    parser.add_argument("--base-model-path", type=pathlib.Path)
    parser.add_argument("--gpus", default="all")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        workspace = args.workspace.expanduser().resolve(strict=True)
        results_dir = _under(args.results_dir, workspace, "--results-dir")
        model_path = _under(args.model_path, workspace, "--model-path")
        media = _under(args.media, workspace, "--media")
        if not model_path.exists() or not media.exists():
            raise ValueError("--model-path and --media must exist")
        framework_config = (
            _under(args.framework_config, workspace, "--framework-config")
            if args.framework_config else None
        )
        action_model_dir = (
            _under(args.action_model_dir, workspace, "--action-model-dir")
            if args.action_model_dir else None
        )
        base_model_path = (
            _under(args.base_model_path, workspace, "--base-model-path")
            if args.base_model_path else None
        )
        if (framework_config is None) != (action_model_dir is None):
            raise ValueError(
                "DCP inference requires --framework-config pointing to Train's saved "
                "Hydra config.yaml and --action-model-dir"
            )
        if framework_config is not None:
            if base_model_path is None:
                raise ValueError("DCP inference requires --base-model-path")
            if not framework_config.is_file() or not base_model_path.is_dir():
                raise ValueError(
                    "DCP inference requires an existing config and prepared Qwen3-VL PTM"
                )
            if not _is_dcp(model_path):
                raise ValueError("--model-path must contain Framework DCP metadata")
        if args.max_new_tokens <= 0:
            raise ValueError("--max-new-tokens must be positive")
        tagged, immutable = resolve_image(args.skill_bank.expanduser().resolve(strict=True))
        command = build_docker_argv(
            immutable_image=immutable,
            job_id=args.job_id,
            workspace=workspace,
            results_dir=results_dir,
            model_path=model_path,
            media=media,
            prompt=args.prompt,
            media_type=args.type,
            max_new_tokens=args.max_new_tokens,
            framework_config=framework_config,
            action_model_dir=action_model_dir,
            base_model_path=base_model_path,
            gpus=args.gpus,
            offline=args.offline,
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "backend": "cosmos-framework",
            "entrypoint": "cosmos-framework-inference",
            "resolved_image": tagged,
            "image_digest": immutable,
            "job_id": args.job_id,
            "results_dir": str(results_dir),
            "command_argv": command,
            "command": shlex.join(command),
            "executed": False,
        }
        if args.execute:
            results_dir.mkdir(parents=True, exist_ok=True)
            (results_dir / "cwd").mkdir(exist_ok=True)
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            payload["executed"] = True
            payload["backend_ref"] = completed.stdout.strip()
            (results_dir / "inference_submission.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        if args.output:
            args.output.expanduser().resolve().write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"submit_cfw_inference: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
