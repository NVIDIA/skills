# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""SafeTensors -> ONNX export for object-detection models, via torch.onnx.export.

Replaces the former `optimum-cli export onnx` path. optimum pinned transformers below
4.54.0, which made it impossible to reach the transformers releases that fix the RCE
advisories (5.3.0 / 5.5.0); optimum 2.1.0 also dropped the `onnx` subcommand entirely.

Scope is object detection, matching the skill. Every supported DETR-family detector
(DETR, Conditional DETR, Deformable DETR, RT-DETR, YOLOS) exposes the same contract:
one `pixel_values` input and `logits` + `pred_boxes` outputs. That uniformity is what
makes a single generic exporter sufficient here.

Writes `<output_dir>/model.onnx` — the same filename optimum produced, so downstream
steps are unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForObjectDetection

DEFAULT_SIZE = 640
DETECTION_SUFFIXES = ("ForObjectDetection", "ForConditionalDetection", "ForZeroShotObjectDetection")


def resolve_input_size(model_dir: str, override: int | None) -> tuple[int, int]:
    """Pick the export resolution: explicit override, else preprocessor_config.json, else 640."""
    if override:
        return override, override

    cfg_path = Path(model_dir) / "preprocessor_config.json"
    if cfg_path.is_file():
        try:
            size = json.loads(cfg_path.read_text()).get("size") or {}
        except (json.JSONDecodeError, OSError):
            size = {}
        if isinstance(size, dict):
            if "height" in size and "width" in size:
                return int(size["height"]), int(size["width"])
            # DETR-style resize: a single shortest/longest edge. Export square at the
            # shortest edge — nvinfer feeds fixed-size letterboxed frames anyway.
            edge = size.get("shortest_edge") or size.get("longest_edge")
            if edge:
                return int(edge), int(edge)
        elif isinstance(size, int):
            return size, size

    return DEFAULT_SIZE, DEFAULT_SIZE


def assert_detection_architecture(model_id: str, revision: str) -> None:
    """Fail fast on non-detection models, matching the skill's stated scope."""
    try:
        config = AutoConfig.from_pretrained(model_id, revision=revision)
    except Exception as exc:  # noqa: BLE001 - surface the loader's own message
        sys.exit(f"ERROR: could not read model config for {model_id!r}: {exc}")

    arch_list = getattr(config, "architectures", None) or []
    if arch_list and not any(a.endswith(DETECTION_SUFFIXES) for a in arch_list):
        sys.exit(
            f"ERROR: deepstream-import-vision-model supports object detection models only. "
            f"Detected architecture: {arch_list[0]}. Classification, segmentation, and other "
            f"vision tasks are not supported."
        )


class DetectionWrapper(torch.nn.Module):
    """Reduce the HF output object to the two tensors DeepStream's parser consumes."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        out = self.model(pixel_values=pixel_values)
        return out.logits, out.pred_boxes


def consolidate_external_data(onnx_path: Path) -> None:
    """Fold any sidecar `.onnx.data` back into the model file.

    torch.onnx.export splits tensors out for large models; trtexec expects one file.
    """
    sidecar = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if not sidecar.exists():
        return
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save(model, str(onnx_path))
    sidecar.unlink()
    print(f"[export] consolidated external data ({sidecar.name}) into {onnx_path.name}")


def verify(onnx_path: Path) -> bool:
    """Assert the DeepStream input/output contract; return whether batch stayed dynamic.

    Neither backend reliably honours a dynamic batch dimension: the dynamo exporter can
    specialize it where the model's own code captures `shape[0]`, so it is checked rather
    than assumed.
    """
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    inputs = {i.name: i for i in model.graph.input}
    outputs = [o.name for o in model.graph.output]
    print(f"[export] inputs={list(inputs)} outputs={outputs}")

    if "pixel_values" not in inputs:
        sys.exit(f"ERROR: exported graph has no 'pixel_values' input (got {list(inputs)})")
    for required in ("logits", "pred_boxes"):
        if required not in outputs:
            sys.exit(f"ERROR: exported graph is missing the '{required}' output (got {outputs})")

    dims = inputs["pixel_values"].type.tensor_type.shape.dim
    shape = [d.dim_param or d.dim_value for d in dims]
    print(f"[export] pixel_values shape={shape}")
    return bool(dims[0].dim_param)


def export_dynamo(wrapper, h, w, onnx_path, opset, static_batch, max_batch) -> None:
    """torch.export-based exporter. Handles models the TorchScript tracer cannot."""
    # Trace with batch=2 so torch.export cannot specialize the dimension to the constant 1.
    dummy = torch.randn(1 if static_batch else 2, 3, h, w)
    shapes = None if static_batch else {
        "pixel_values": {0: torch.export.Dim("batch", min=1, max=max_batch)},
    }
    torch.onnx.export(
        wrapper, (dummy,), str(onnx_path), dynamic_shapes=shapes, dynamo=True,
        input_names=["pixel_values"], output_names=["logits", "pred_boxes"],
        opset_version=opset,
    )


def export_torchscript(wrapper, h, w, onnx_path, opset, static_batch, max_batch) -> None:
    """Legacy TorchScript exporter.

    Still the more reliable path for a dynamic batch dimension on DETR-family detectors —
    RT-DETR specializes batch under dynamo but stays dynamic here. It does fail on
    text-conditioned models (Grounding DINO and friends), which is what dynamo is for.
    """
    axes = None if static_batch else {
        "pixel_values": {0: "batch"}, "logits": {0: "batch"}, "pred_boxes": {0: "batch"},
    }
    torch.onnx.export(
        wrapper, torch.randn(1, 3, h, w), str(onnx_path), dynamic_axes=axes,
        do_constant_folding=True, dynamo=False,
        input_names=["pixel_values"], output_names=["logits", "pred_boxes"],
        opset_version=opset,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a HuggingFace detection model to ONNX.")
    ap.add_argument("--model", required=True, help="HF model id or local model directory")
    ap.add_argument("--output-dir", required=True, help="directory to write model.onnx into")
    # 18, not 17: the dynamo exporter implements >=18 and auto-upgrades anyway, and its
    # downgrade path fails outright on Resize ("No Adapter To Version 17 for Resize").
    # Opset 18 is compatible with TRT 10.16 — see references/engine-build.md.
    ap.add_argument("--opset", type=int, default=18, help="ONNX opset (default 18)")
    ap.add_argument("--image-size", type=int, default=None,
                    help="square export size; default reads preprocessor_config.json, else 640")
    ap.add_argument("--max-batch", type=int, default=16,
                    help="upper bound for the dynamic batch dimension (default 16)")
    ap.add_argument("--static-batch", action="store_true",
                    help="export a fixed batch-1 graph instead of a dynamic batch dimension")
    ap.add_argument("--revision", default="main",
                    help="Hub revision (branch, tag, or commit SHA). Pin a SHA for reproducible builds.")
    ap.add_argument("--legacy-torchscript", action="store_true",
                    help="force the TorchScript exporter instead of trying dynamo first")
    args = ap.parse_args()

    assert_detection_architecture(args.model, args.revision)

    h, w = resolve_input_size(args.model, args.image_size)
    print(f"[export] model={args.model}  input=1x3x{h}x{w}  opset={args.opset}")

    model = AutoModelForObjectDetection.from_pretrained(args.model, revision=args.revision)
    # .eval() on the wrapper too — it is a fresh nn.Module and defaults to training mode,
    # which the exporter warns about and which changes dropout/batchnorm behaviour.
    wrapper = DetectionWrapper(model).eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

    # Neither backend handles every architecture. Try dynamo first (it copes with models the
    # tracer chokes on), then fall back to TorchScript when dynamo specializes the batch
    # dimension — which is what RT-DETR, the default model, does.
    strategies = [("legacy-torchscript", export_torchscript)] if args.legacy_torchscript \
        else [("dynamo", export_dynamo), ("legacy-torchscript", export_torchscript)]

    for index, (name, export_fn) in enumerate(strategies):
        last = index == len(strategies) - 1
        print(f"[export] backend={name}")
        try:
            export_fn(wrapper, h, w, onnx_path, args.opset, args.static_batch, args.max_batch)
        except Exception as exc:  # noqa: BLE001 - report and try the next backend
            if last:
                sys.exit(f"ERROR: {name} export failed: {exc}")
            print(f"[export] {name} failed ({type(exc).__name__}); trying the next backend")
            continue

        consolidate_external_data(onnx_path)
        if args.static_batch or verify(onnx_path):
            break
        if last:
            sys.exit(
                "ERROR: every backend baked in a static batch dimension. Re-run with "
                "--static-batch and build a fixed-batch engine for this model."
            )
        print(f"[export] {name} produced a static batch dimension; trying the next backend")

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"[export] wrote {onnx_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
