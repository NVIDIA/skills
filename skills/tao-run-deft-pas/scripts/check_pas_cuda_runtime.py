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
import json
import shutil
import sys
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-gpus", type=int, required=True)
    parser.add_argument("--require-cli", action="append", default=[])
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
    except Exception as exc:
        print(
            f"PAS_CUDA_PROBE=FAIL reason={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    print("PAS_CUDA_PROBE=PASS " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
