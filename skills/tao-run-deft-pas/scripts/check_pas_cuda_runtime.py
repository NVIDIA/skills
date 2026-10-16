#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the CUDA framework and TAO entrypoints inside one PAS runtime.

GPU enumeration alone does not establish that the PyTorch/CUDA build in a TAO
image can initialize against the host driver.  This probe deliberately creates
and synchronizes one CUDA tensor on every requested visible device.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
from typing import Any, Callable


def probe_runtime(
    *,
    minimum_gpus: int,
    required_clis: list[str],
    torch_module: Any,
    which: Callable[[str], str | None] = shutil.which,
) -> dict[str, Any]:
    """Return runtime facts after real CUDA initialization and CLI checks."""
    if minimum_gpus < 1:
        raise ValueError("minimum_gpus must be at least 1")
    missing = [name for name in required_clis if not which(name)]
    if missing:
        raise RuntimeError("missing TAO CLI entrypoints: " + ", ".join(missing))
    if not torch_module.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    visible = int(torch_module.cuda.device_count())
    if visible < minimum_gpus:
        raise RuntimeError(
            f"visible CUDA devices {visible} are fewer than required {minimum_gpus}"
        )

    devices = []
    for index in range(minimum_gpus):
        try:
            properties = torch_module.cuda.get_device_properties(index)
            allocation = torch_module.empty(1, device=f"cuda:{index}")
            allocation.add_(1)
            torch_module.cuda.synchronize(index)
        except Exception as exc:  # driver/runtime errors vary across torch builds
            raise RuntimeError(
                f"CUDA framework initialization failed on visible device {index}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        devices.append(
            {
                "index": index,
                "name": str(properties.name),
                "compute_capability": f"{properties.major}.{properties.minor}",
                "memory_gb": round(int(properties.total_memory) / (1024**3), 2),
            }
        )

    return {
        "status": "PASS",
        "torch_version": str(torch_module.__version__),
        "cuda_build": str(torch_module.version.cuda),
        "visible_gpus": visible,
        "tested_gpus": devices,
        "required_clis": required_clis,
    }


def probe_clip_lora_contract(config_type: Any, lora_module: Any) -> dict[str, Any]:
    """Verify the CLIP schema plus adapter/checkpoint implementation contract."""
    peft = dataclasses.asdict(config_type().peft)
    required_targets = ["q_proj", "k_proj", "v_proj", "out_proj"]
    if peft.get("method") != "lora" or not isinstance(peft.get("enabled"), bool):
        raise RuntimeError("CLIP PEFT schema lacks the LoRA enable/method contract")
    for tower in ("vision", "text"):
        block = peft.get(tower, {})
        required_fields = (
            "mode",
            "target_modules",
            "num_last_blocks",
            "rank",
            "alpha",
            "dropout",
        )
        if not isinstance(block, dict):
            raise RuntimeError(
                f"CLIP PEFT {tower} contract is incomplete: tower is not an object"
            )
        missing = [key for key in required_fields if key not in block]
        wrong_targets = block.get("target_modules") != required_targets
        if missing or wrong_targets:
            details = []
            if missing:
                details.append("missing fields: " + ", ".join(missing))
            if wrong_targets:
                details.append(
                    "target_modules must be " + ", ".join(required_targets)
                )
            raise RuntimeError(
                f"CLIP PEFT {tower} contract is incomplete: " + "; ".join(details)
            )
    symbols = (
        "LoRALinear",
        "inject_lora",
        "merge_lora",
        "_register_lora_checkpoint_compatibility",
    )
    missing_symbols = [
        name
        for name in symbols
        if name != "LoRALinear" and not callable(getattr(lora_module, name, None))
    ]
    if not isinstance(getattr(lora_module, "LoRALinear", None), type):
        missing_symbols.append("LoRALinear")
    if missing_symbols:
        raise RuntimeError(
            "CLIP LoRA runtime lacks: "
            + ", ".join(sorted(set(missing_symbols)))
        )
    return {
        "schema": peft,
        "runtime_symbols": list(symbols),
        "checkpoint_behavior": "register-and-merge",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-gpus", type=int, required=True)
    parser.add_argument("--require-cli", action="append", default=[])
    parser.add_argument("--require-clip-lora", action="store_true")
    parser.add_argument("--image-ref")
    parser.add_argument("--image-digest")
    parser.add_argument("--output", type=pathlib.Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        import torch

        result = probe_runtime(
            minimum_gpus=args.min_gpus,
            required_clis=args.require_cli,
            torch_module=torch,
        )
        if args.output is not None:
            if not args.image_ref or not args.image_digest:
                raise ValueError(
                    "--output requires --image-ref and --image-digest"
                )
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_digest):
                raise ValueError(
                    "--image-digest must be an immutable sha256:<64 lowercase hex> digest"
                )
            result.update(
                {
                    "schema_version": "1",
                    "image_ref": args.image_ref,
                    "image_digest": args.image_digest,
                    "finetuning_methods": ["sft"],
                }
            )
        if args.require_clip_lora:
            if args.output is None:
                raise ValueError("--require-clip-lora requires --output")
            from nvidia_tao_pytorch.config.clip.default_config import CLIPExperimentConfig
            from nvidia_tao_pytorch.multimodal.clip.model import lora

            result["clip_lora"] = probe_clip_lora_contract(CLIPExperimentConfig, lora)
            result["finetuning_methods"].append("lora")
    except Exception as exc:
        print(
            f"PAS_CUDA_PROBE=FAIL reason={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=args.output.name + ".",
            suffix=".tmp",
            dir=str(args.output.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            pathlib.Path(temporary).replace(args.output)
        except Exception:
            pathlib.Path(temporary).unlink(missing_ok=True)
            raise
    print("PAS_CUDA_PROBE=PASS " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
