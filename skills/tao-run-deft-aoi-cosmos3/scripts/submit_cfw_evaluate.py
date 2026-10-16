#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Compose or execute a Docker submit for native Framework evaluation."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 in supported TAO host environments
    import tomli as tomllib

from submit_cfw_train import default_bank, resolve_image


CONTAINER_CONFIG = "/tao/config/evaluate.toml"
LEGACY_WORKSPACE = "/tao-workspace"


def _username() -> str:
    return os.environ.get("USER") or os.environ.get("LOGNAME") or "tao"


def _rerender_error(reason: str) -> ValueError:
    return ValueError(
        "stale or foreign Framework evaluate spec: "
        f"{reason}; re-render with render_cfw_evaluate.py for this workspace and "
        "never reuse a packaged spec"
    )


def _is_legacy_workspace_path(value: Any) -> bool:
    return isinstance(value, str) and (
        value == LEGACY_WORKSPACE or value.startswith(f"{LEGACY_WORKSPACE}/")
    )


def _require_renderer_contract(config: dict[str, Any]) -> None:
    tables: dict[str, dict[str, Any]] = {}
    for name in ("dataset", "model", "evaluation", "vision", "generation"):
        value = config.get(name)
        if not isinstance(value, dict):
            raise _rerender_error(f"missing or invalid [{name}] table")
        tables[name] = value
    dataset = tables["dataset"]
    model = tables["model"]
    evaluation = tables["evaluation"]
    vision = tables["vision"]
    generation = tables["generation"]
    if "tokenizer_model_name" in model or "mode" in generation:
        raise _rerender_error("legacy tokenizer/generation keys are present")
    path_values = (
        ("results_dir", config.get("results_dir")),
        ("dataset.annotation_path", dataset.get("annotation_path")),
        ("dataset.media_dir", dataset.get("media_dir")),
        ("model.model_name", model.get("model_name")),
        ("model.config_file", model.get("config_file")),
        ("model.export_dir", model.get("export_dir")),
        ("model.vit_checkpoint_path", model.get("vit_checkpoint_path")),
    )
    required_paths = {
        "results_dir",
        "dataset.annotation_path",
        "dataset.media_dir",
        "model.model_name",
    }
    path_mismatches: list[str] = []
    for name, value in path_values:
        if name in required_paths and (
            not isinstance(value, str) or not pathlib.Path(value).is_absolute()
        ):
            path_mismatches.append(name)
        if value and _is_legacy_workspace_path(value):
            raise _rerender_error(f"{name} uses legacy {LEGACY_WORKSPACE} paths")
    expected = (
        ("num_gpus", config.get("num_gpus"), 1),
        (
            "dataset.system_prompt",
            dataset.get("system_prompt"),
            "Compare the AOI image with its golden reference. "
            "Respond with exactly OK or NG.",
        ),
        ("model.dtype", model.get("dtype"), "bfloat16"),
        ("model.max_length", model.get("max_length"), 40960),
        ("model.tp_size", model.get("tp_size"), 1),
        ("model.attn_implementation", model.get("attn_implementation"), "sdpa"),
        ("evaluation.answer_type", evaluation.get("answer_type"), "freeform"),
        ("evaluation.num_processes", evaluation.get("num_processes"), 1),
        ("evaluation.batch_size", evaluation.get("batch_size"), 1),
        ("vision.num_frames", vision.get("num_frames"), 1),
        (
            "vision.video_decoder",
            vision.get("video_decoder"),
            "torchcodec-cuda-on-demand",
        ),
        (
            "vision.dataloader_num_workers",
            vision.get("dataloader_num_workers"),
            1,
        ),
        (
            "vision.dataloader_persistent_workers",
            vision.get("dataloader_persistent_workers"),
            True,
        ),
        ("generation.max_tokens", generation.get("max_tokens"), 4),
    )
    mismatches = path_mismatches + [
        name for name, actual, wanted in expected if actual != wanted
    ]
    if mismatches:
        raise _rerender_error(
            "missing or invalid renderer-owned keys: " + ", ".join(mismatches)
        )


def _mount(host: pathlib.Path, container: str, *, readonly: bool) -> str:
    suffix = ",readonly" if readonly else ""
    return f"type=bind,src={host},dst={container}{suffix}"


def _under(path: pathlib.Path, root: pathlib.Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be under --workspace: {path}") from exc


def _is_dcp(path: pathlib.Path) -> bool:
    return (path / ".metadata").is_file() or (path / "model/.metadata").is_file()


def validate_config(
    config: dict[str, Any], workspace: pathlib.Path
) -> tuple[pathlib.Path, pathlib.Path | None]:
    _require_renderer_contract(config)
    model = config.get("model", {})
    dataset = config.get("dataset", {})
    vision = config.get("vision", {})
    if model.get("enable_lora") is not False:
        raise ValueError("Framework evaluation requires model.enable_lora=false")
    if vision.get("video_decoder") != "torchcodec-cuda-on-demand":
        raise ValueError(
            "Framework evaluation must use the rendered decoder profile "
            "vision.video_decoder=torchcodec-cuda-on-demand"
        )
    for table, key in (
        (dataset, "annotation_path"),
        (dataset, "media_dir"),
        (model, "model_name"),
        (config, "results_dir"),
    ):
        value = table.get(key)
        if not isinstance(value, str) or not pathlib.Path(value).is_absolute():
            raise ValueError(f"{key} must be an absolute compute-frame path")
        _under(pathlib.Path(value).resolve(), workspace, key)
    model_path = pathlib.Path(model["model_name"]).resolve()
    if not model_path.exists():
        raise ValueError(f"model.model_name does not exist: {model_path}")
    config_file = model.get("config_file")
    export_dir = model.get("export_dir")
    writable_export_parent = None
    if config_file:
        vit_checkpoint = model.get("vit_checkpoint_path")
        if (
            not pathlib.Path(config_file).is_file()
            or not export_dir
            or not isinstance(vit_checkpoint, str)
            or not pathlib.Path(vit_checkpoint).is_dir()
        ):
            raise ValueError(
                "DCP evaluation requires model.config_file pointing to Train's saved "
                "Hydra config.yaml, export_dir, and a local HF vit_checkpoint_path"
            )
        if not _is_dcp(model_path):
            raise ValueError("model.model_name must contain Framework DCP metadata")
        _under(pathlib.Path(config_file).resolve(), workspace, "model.config_file")
        _under(pathlib.Path(export_dir).resolve(), workspace, "model.export_dir")
        _under(pathlib.Path(vit_checkpoint).resolve(), workspace, "model.vit_checkpoint_path")
        writable_export_parent = pathlib.Path(export_dir).resolve().parent
    return pathlib.Path(config["results_dir"]).resolve(), writable_export_parent


def build_docker_argv(
    *,
    immutable_image: str,
    job_id: str,
    config_host: pathlib.Path,
    workspace: pathlib.Path,
    results_dir: pathlib.Path,
    writable_export_parent: pathlib.Path | None,
    gpus: str,
    offline: bool,
) -> list[str]:
    username = _username()
    command = [
        "docker", "run", "-d", "--name", job_id,
        "--label", f"tao-job={job_id}", "--gpus", gpus,
        "--shm-size=8g", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", f"USER={username}", "-e", f"LOGNAME={username}", "-e", "HOME=/tmp",
        "-e", f"TAO_JOB_ID={job_id}", "-e", f"TAO_RESULTS_ROOT={results_dir}",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-v", "/etc/passwd:/etc/passwd:ro", "-v", "/etc/group:/etc/group:ro",
        "--mount", _mount(config_host, CONTAINER_CONFIG, readonly=True),
        "--mount", _mount(workspace, str(workspace), readonly=True),
        "--mount", _mount(results_dir, str(results_dir), readonly=False),
        "-w", str(results_dir / "cwd"),
    ]
    if writable_export_parent is not None and writable_export_parent != results_dir:
        command.extend([
            "--mount",
            _mount(writable_export_parent, str(writable_export_parent), readonly=False),
        ])
    if offline:
        command.extend(["-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1"])
    command.extend([
        "--entrypoint", "/workspace/.venv/bin/cosmos-framework-evaluate",
        immutable_image, "--config", CONTAINER_CONFIG,
    ])
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-bank", type=pathlib.Path, default=default_bank())
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--workspace", required=True, type=pathlib.Path)
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--gpus", default="all")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config_host = args.config.expanduser().resolve(strict=True)
        workspace = args.workspace.expanduser().resolve(strict=True)
        results_dir = args.results_dir.expanduser().resolve()
        with config_host.open("rb") as stream:
            config = tomllib.load(stream)
        configured_results, writable_export_parent = validate_config(config, workspace)
        if configured_results != results_dir:
            raise ValueError("--results-dir must exactly match the TOML results_dir")
        tagged, immutable = resolve_image(args.skill_bank.expanduser().resolve(strict=True))
        command = build_docker_argv(
            immutable_image=immutable,
            job_id=args.job_id,
            config_host=config_host,
            workspace=workspace,
            results_dir=results_dir,
            writable_export_parent=writable_export_parent,
            gpus=args.gpus,
            offline=args.offline,
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "backend": "cosmos-framework",
            "entrypoint": "cosmos-framework-evaluate",
            "resolved_image": tagged,
            "image_digest": immutable,
            "job_id": args.job_id,
            "config": str(config_host),
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
            (results_dir / "evaluate_submission.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        if args.output:
            args.output.expanduser().resolve().write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError, tomllib.TOMLDecodeError) as exc:
        print(f"submit_cfw_evaluate: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
