#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Render and validate a Cosmos Framework CR3 SFT profile."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import tempfile
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 in supported TAO host environments
    import tomli as tomllib


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROFILES = {
    "smoke": SKILL_ROOT / "references/cosmos_framework_sft_smoke.toml",
    "full": SKILL_ROOT / "references/cosmos_framework_sft_full.toml",
}
MARKERS = {
    "__TAO_CR3_MODEL__": "model_path",
    "__TAO_CR3_TRAIN_ANNOTATION__": "annotation_path",
    "__TAO_CR3_MEDIA_ROOT__": "media_root",
    "__TAO_CR3_RUN_NAME__": "run_name",
    "__TAO_CR3_RESUME__": "resume_checkpoint",
}
LORA_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
LEGACY_WORKSPACE = "/tao-workspace"


def _rerender_error(reason: str) -> ValueError:
    return ValueError(
        "stale or foreign Framework train spec: "
        f"{reason}; re-render with render_cfw_sft.py for this workspace and "
        "never reuse a packaged spec"
    )


def _is_legacy_workspace_path(value: Any) -> bool:
    return isinstance(value, str) and (
        value == LEGACY_WORKSPACE or value.startswith(f"{LEGACY_WORKSPACE}/")
    )


def _require_renderer_contract(config: dict[str, Any]) -> None:
    tables: dict[str, dict[str, Any]] = {}
    for name in ("model", "custom", "checkpoint"):
        value = config.get(name)
        if not isinstance(value, dict):
            raise _rerender_error(f"missing or invalid [{name}] table")
        tables[name] = value
    model = tables["model"]
    backbone = model.get("backbone", {})
    if not isinstance(backbone, dict):
        raise _rerender_error("missing or invalid [model.backbone] table")
    custom = tables["custom"]
    checkpoint = tables["checkpoint"]
    if "tokenizer_model_name" in model:
        raise _rerender_error("legacy model.tokenizer_model_name is present")
    expected = (
        ("model.attn_implementation", model.get("attn_implementation"), "cosmos"),
        ("custom.annotation_mode", custom.get("annotation_mode"), "bare_okng"),
        ("custom.images_per_record", custom.get("images_per_record"), 2),
    )
    mismatches = [name for name, actual, wanted in expected if actual != wanted]
    path_values = (
        ("model.backbone.model_name", backbone.get("model_name")),
        ("model.backbone.safetensors_path", backbone.get("safetensors_path")),
        ("custom.annotation_path", custom.get("annotation_path")),
        ("custom.media_root", custom.get("media_root")),
    )
    for name, value in path_values:
        if not isinstance(value, str) or not pathlib.Path(value).is_absolute():
            mismatches.append(name)
        elif _is_legacy_workspace_path(value):
            raise _rerender_error(f"{name} uses legacy {LEGACY_WORKSPACE} paths")
    if backbone.get("model_name") != backbone.get("safetensors_path"):
        mismatches.append("model.backbone.safetensors_path")
    load_path = checkpoint.get("load_path")
    if _is_legacy_workspace_path(load_path):
        raise _rerender_error(
            f"checkpoint.load_path uses legacy {LEGACY_WORKSPACE} paths"
        )
    if mismatches:
        raise _rerender_error(
            "missing or invalid renderer-owned keys: " + ", ".join(sorted(set(mismatches)))
        )


def validate_config(config: dict[str, Any], *, profile: str | None = None) -> None:
    _require_renderer_contract(config)
    job = config.get("job", {})
    model = config.get("model", {})
    trainer = config.get("trainer", {})
    checkpoint = config.get("checkpoint", {})
    optimizer = config.get("optimizer", {})
    custom = config.get("custom", {})
    if job.get("task") != "vlm" or job.get("experiment") != "tao_cr3_aoi":
        raise ValueError("Framework CR3 config must select job.task=vlm and experiment=tao_cr3_aoi")
    if model.get("precision") != "bfloat16":
        raise ValueError("Framework CR3 config must use BF16")
    if not model.get("lora_enabled"):
        raise ValueError("Framework CR3 config must enable native VLM LoRA")
    if model.get("lora_target_modules") != LORA_TARGETS:
        raise ValueError("Framework CR3 config must use the reviewed language-side LoRA targets")
    if model.get("parallelism", {}).get("data_parallel_replicate_degree") != 1:
        raise ValueError("Framework CR3 config must remain single-node")
    if trainer.get("distributed_parallelism") != "fsdp":
        raise ValueError("Framework CR3 config must use Framework FSDP2")
    if optimizer.get("keys_to_select") != ["lora_"]:
        raise ValueError("Framework CR3 optimizer must select LoRA parameters only")
    if checkpoint.get("save_iter", 0) < 1 or checkpoint.get("dcp_async_mode_enabled") is not False:
        raise ValueError("Framework CR3 config must use a positive synchronous DCP save cadence")
    if checkpoint.get("keys_to_skip_loading"):
        raise ValueError("Framework CR3 resume must restore native LoRA keys")
    load_path = checkpoint.get("load_path")
    if load_path != "???" and (
        not isinstance(load_path, str) or not pathlib.Path(load_path).is_absolute()
    ):
        raise ValueError("Framework CR3 checkpoint.load_path must be ??? or an absolute DCP path")
    if not isinstance(custom.get("seed"), int):
        raise ValueError("Framework CR3 config must record an integer seed")
    if custom.get("fsdp_master_dtype") != "bfloat16":
        raise ValueError("Framework CR3 config must request a BF16 FSDP master dtype")
    if custom.get("model_max_length") != 40960:
        raise ValueError("Framework CR3 config must keep model_max_length=40960")
    if profile == "smoke" and not 1 <= int(trainer.get("max_iter", 0)) <= 50:
        raise ValueError("smoke profile must run between 1 and 50 optimizer steps")


def load_toml(path: pathlib.Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def render_profile(
    profile: str,
    *,
    model_path: str,
    annotation_path: str,
    media_root: str,
    run_name: str,
    resume_checkpoint: str = "???",
    num_epochs: int = 10,
    train_records: int | None = None,
) -> str:
    template = PROFILES[profile].read_text(encoding="utf-8")
    values = locals()
    for marker, name in MARKERS.items():
        template = template.replace(json.dumps(marker), json.dumps(values[name]))
    unresolved = [marker for marker in MARKERS if marker in template]
    if unresolved:
        raise ValueError(f"unresolved Framework TOML markers: {unresolved}")
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if train_records is not None:
        if train_records <= 0:
            raise ValueError("train_records must be positive")
        steps = 5 if profile == "smoke" else train_records * num_epochs
        save_iter = steps if profile == "smoke" else train_records
        replacements = {
            r"(?m)^cycle_lengths = \[[0-9]+\]$": f"cycle_lengths = [{steps}]",
            r"(?m)^max_iter = [0-9]+$": f"max_iter = {steps}",
            r"(?m)^save_iter = [0-9]+$": f"save_iter = {save_iter}",
            r"(?m)^images_per_record = 2$": (
                "images_per_record = 2\n"
                f"num_epochs = {num_epochs}\ntrain_records = {train_records}"
            ),
        }
        for pattern, replacement in replacements.items():
            template, count = re.subn(pattern, replacement, template)
            if count != 1:
                raise ValueError(f"Framework profile budget field did not match exactly once: {pattern}")
    parsed = tomllib.loads(template)
    validate_config(parsed, profile=profile)
    return template


def atomic_write(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument(
        "--model-path",
        required=True,
        help="Compute-frame prepared Qwen3-VL safetensors PTM path",
    )
    parser.add_argument("--annotation-path", required=True, help="Compute-frame Train JSON-array path")
    parser.add_argument("--media-root", required=True, help="Compute-frame base for relative image paths")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument(
        "--annotation-host",
        type=pathlib.Path,
        help="Host-visible JSON path used to derive the exact epoch step budget; defaults to --annotation-path.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        help="Previous iteration Framework DCP compute-frame path; omit for iter1.",
    )
    parser.add_argument("--output", required=True, type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        annotation_host = (
            args.annotation_host.expanduser().resolve(strict=True)
            if args.annotation_host
            else pathlib.Path(args.annotation_path).expanduser().resolve(strict=True)
        )
        records = json.loads(annotation_host.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not records:
            raise ValueError("training annotation must be a non-empty JSON array")
        rendered = render_profile(
            args.profile,
            model_path=args.model_path,
            annotation_path=args.annotation_path,
            media_root=args.media_root,
            run_name=args.run_name,
            resume_checkpoint=args.resume_checkpoint or "???",
            num_epochs=args.num_epochs,
            train_records=len(records),
        )
        atomic_write(args.output.expanduser().resolve(), rendered)
    except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        print(f"render_cfw_sft: {exc}", file=os.sys.stderr)
        return 2
    print(args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
