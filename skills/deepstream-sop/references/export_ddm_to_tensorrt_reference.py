#!/usr/bin/env python3
######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
######################################################################################################

"""
Export DDM-Net model to TensorRT engine.

Pipeline: PyTorch Lightning checkpoint (.ckpt) → ONNX → TensorRT engine

This script is tailored for the nvds_action_detector project:
  - Loads .ckpt checkpoints (PyTorch Lightning format)
  - Finds DDM-Net source via DDM_BASE_PATH env var (default: ./3rdparty/DDM)
  - Exports a fixed-shape engine. The batch dimension is fixed at build time
    (configurable via --batch-size, default 8) because dynamic batch is not
    supported reliably for this model — pairwise L2 distance ops inside DDM
    don't propagate dynamic shapes cleanly. Any other build-time batch is
    fine; the runtime backend (triton_model_repo/ddm/1/model.py) reads the
    batch dim back from the engine and chunks+pads inputs to match it.
  - Input shape  (B, T, 3, H, W) → output (B, 2) logits, where
    B = --batch-size and T = 2 * --frames-per-side + 1.

Recommended entry point: ``scripts/tensorrt/build_engine.sh`` (host-side helper
that launches an ephemeral container from the SOP image and invokes this
script with the right mounts). Invoke this Python script directly only when
already inside a TRT-capable container.

Usage (direct, in-container):
    # One-shot (ONNX + TRT, FP16 — the verified default)
    python export_ddm_to_tensorrt.py \\
        --checkpoint /path/to/checkpoint.ckpt \\
        --resolution 384 \\
        --output-dir ./trt_output \\
        --precision fp16 \\
        --verify

    # Build BF16 instead (better accuracy than FP16 if needed)
    python export_ddm_to_tensorrt.py \\
        --checkpoint /path/to/checkpoint.ckpt \\
        --resolution 384 \\
        --output-dir ./trt_output \\
        --precision bf16

    # Export ONNX only (skip the TRT build)
    python export_ddm_to_tensorrt.py \\
        --checkpoint /path/to/checkpoint.ckpt \\
        --resolution 384 \\
        --output-dir ./trt_output \\
        --skip-trt
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDMWrapper
# ---------------------------------------------------------------------------


class DDMWrapper(torch.nn.Module):
    """
    Wraps DDM model to return only the final layer's logits.

    Original model returns (results, rgbs, ddms) where each is a list of 6
    tensors. This wrapper returns only results[-1] with shape (B, num_classes).
    Softmax is NOT applied here — it is applied after TRT inference so we keep
    logits and avoid precision issues in the export.
    """

    def __init__(self, ddm_model: torch.nn.Module):
        super().__init__()
        self.model = ddm_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DDM returns (results, rgbs, ddms) lists (one tensor per stage); export only the
        # final-stage logits (B, num_classes). Softmax is applied AFTER TRT inference, not here.
        results, _rgbs, _ddms = self.model(x)
        return results[-1]


# ---------------------------------------------------------------------------
# Model loading (mirrors ddm_net.py in triton_model_repo)
# ---------------------------------------------------------------------------


def load_ddm_model(
    checkpoint_path: str,
    ddm_base_path: str,
    frames_per_side: int = 5,
    resolution: int = 224,
) -> torch.nn.Module:
    """
    Load DDM-Net model from a PyTorch Lightning .ckpt checkpoint.

    Args:
        checkpoint_path: Path to .ckpt checkpoint.
        ddm_base_path:   Root of the DDM checkout (contains DDM-Net/ subdirectory).
        frames_per_side: Temporal context window half-size (default 5 → 11 frames).
        resolution:      Input spatial resolution passed to resnetGEBD (224 or 384).

    Returns:
        DDM model in eval mode on CPU.
    """
    modeling_path = os.path.abspath(os.path.join(ddm_base_path, "DDM-Net"))
    _LOGGER.info(f"DDM-Net source: {modeling_path}")

    if not os.path.isdir(modeling_path):
        raise FileNotFoundError(
            f"DDM-Net source directory not found: {modeling_path}\n"
            f"Set --ddm-base-path or the DDM_BASE_PATH environment variable."
        )

    try:
        sys.path.insert(0, modeling_path)
        from modeling.resnetGEBD import resnetGEBD
    finally:
        sys.path.pop(0)

    _LOGGER.info(f"Creating resnetGEBD(backbone=resnet50, frames_per_side={frames_per_side}, resolution={resolution})")
    model = resnetGEBD(
        backbone="resnet50",
        pretrained=False,
        num_classes=2,
        frames_per_side=frames_per_side,
    )

    _LOGGER.info(f"Loading checkpoint: {checkpoint_path}")
    # PyTorch 2.6+ torch.load defaults to weights_only=True; allowlist the ckpt's argparse.Namespace
    with torch.serialization.safe_globals([argparse.Namespace]):
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_key = key[len("model."):]
            elif key.startswith("module."):
                new_key = key[len("module."):]
            else:
                new_key = key
            new_state_dict[new_key] = value
    else:
        _LOGGER.info("No 'state_dict' key found — treating checkpoint as raw state dict.")
        new_state_dict = ckpt

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        _LOGGER.warning("Missing keys: %s", missing)
    if unexpected:
        _LOGGER.warning("Unexpected keys: %s", unexpected)

    model.eval()
    _LOGGER.info("DDM-Net model loaded successfully.")
    return model


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def export_to_onnx(
    model: torch.nn.Module,
    onnx_path: str,
    batch_size: int,
    num_frames: int,
    resolution: int,
) -> None:
    """
    Export wrapped DDM model to ONNX with fixed (static) shapes.

    Args:
        model:       DDMWrapper in eval mode (on CPU).
        onnx_path:   Output .onnx file path.
        batch_size:  Fixed batch size (default 8 = SEQUENCE_BATCH).
        num_frames:  Fixed number of frames per window (default 11).
        resolution:  Spatial resolution (H = W).
    """
    dummy_input = torch.randn(batch_size, num_frames, 3, resolution, resolution)

    _LOGGER.info(f"ONNX export — input shape: {list(dummy_input.shape)}")
    _LOGGER.info(f"ONNX path: {onnx_path}")

    # Static batch size export — dynamic batch causes TRT build failures
    # for this model due to pairwise L2 distance ops inside DDM.
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["video_frames"],  # (B, T, C, H, W)
            output_names=["logits"],       # (B, 2)
            verbose=False,
        )

    _LOGGER.info("ONNX export done.")

    # Validate
    try:
        import onnx

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        inp = onnx_model.graph.input[0]
        batch_dim = inp.type.tensor_type.shape.dim[0]
        if batch_dim.HasField("dim_param"):
            _LOGGER.info(f"ONNX batch dim: DYNAMIC ({batch_dim.dim_param})")
        elif batch_dim.HasField("dim_value"):
            _LOGGER.info(f"ONNX batch dim: STATIC ({batch_dim.dim_value})")
        _LOGGER.info("ONNX model validation passed.")
    except ImportError:
        _LOGGER.warning("onnx package not available — skipping ONNX validation.")
    except Exception as exc:
        _LOGGER.warning(f"ONNX validation warning: {exc}")


# ---------------------------------------------------------------------------
# TensorRT engine build
# ---------------------------------------------------------------------------


def build_trt_engine(
    onnx_path: str,
    engine_path: str,
    batch_size: int,
    precision: str = "fp16",
    workspace_gb: int = 4,
    verbose: bool = False,
) -> bool:
    """
    Build a TensorRT engine from an ONNX model with static shapes.

    Because dynamic batch causes failures for this model, no optimization
    profile is set — TRT reads the fixed shapes directly from ONNX.

    Args:
        onnx_path:    Path to .onnx file.
        engine_path:  Output .engine file path.
        batch_size:   Fixed batch size the engine should be built for. Used
                      only to construct an optimization profile in the
                      defensive fallback branch (dynamic-batch ONNX input).
        precision:    "fp32", "fp16", or "bf16". Default "fp16" (verified config).
        workspace_gb: GPU memory TRT may use for optimization (GB).
        verbose:      Enable verbose TRT logging.

    Returns:
        True on success, False on failure.
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError("TensorRT is not installed. Run inside the appropriate Docker image.")

    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"precision must be one of fp32/fp16/bf16, got {precision!r}")

    trt_log_level = trt.Logger.VERBOSE if verbose else trt.Logger.INFO
    TRT_LOGGER = trt.Logger(trt_log_level)

    _LOGGER.info("=" * 60)
    _LOGGER.info("Building TensorRT engine")
    _LOGGER.info(f"  ONNX:      {onnx_path}")
    _LOGGER.info(f"  Engine:    {engine_path}")
    _LOGGER.info(f"  Precision: {precision.upper()}")
    _LOGGER.info(f"  Workspace: {workspace_gb} GB")
    _LOGGER.info("=" * 60)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    _LOGGER.info("Parsing ONNX model...")
    # Pass the path so the parser can resolve sidecar weight files (.onnx.data
    # generated by torch's external-data ONNX format for large models).
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read(), path=onnx_path):
            _LOGGER.error("ONNX parse failed:")
            for i in range(parser.num_errors):
                _LOGGER.error(f"  {parser.get_error(i)}")
            return False

    _LOGGER.info(f"ONNX parsed: {network.num_inputs} input(s), {network.num_outputs} output(s)")
    for i in range(network.num_inputs):
        t = network.get_input(i)
        _LOGGER.info(f"  Input  {i}: name={t.name!r} shape={list(t.shape)} dtype={t.dtype}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        _LOGGER.info(f"  Output {i}: name={t.name!r} shape={list(t.shape)} dtype={t.dtype}")

    config = builder.create_builder_config()
    # Workspace = scratch GPU memory the builder may use while timing/selecting kernels (DDM_TRT_BUILD_WORKSPACE_GB).
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            _LOGGER.info("FP16 precision enabled.")
        else:
            _LOGGER.warning("FP16 requested but not supported on this GPU — falling back to FP32.")
    elif precision == "bf16":
        # platform_has_fast_bf16 isn't exposed on every TRT 10.x release; gate
        # defensively via getattr so the check is a no-op on older bindings.
        has_bf16 = getattr(builder, "platform_has_fast_bf16", True)
        bf16_flag = getattr(trt.BuilderFlag, "BF16", None)
        if has_bf16 and bf16_flag is not None:
            config.set_flag(bf16_flag)
            _LOGGER.info("BF16 precision enabled.")
        else:
            _LOGGER.warning("BF16 requested but not available on this TRT/GPU — falling back to FP32.")

    # No optimization profile needed for static shapes.
    # If the ONNX has a dynamic batch dimension (-1), we must set a profile.
    input_shape = list(network.get_input(0).shape)
    if input_shape[0] == -1:
        _LOGGER.info(
            f"Dynamic batch dim detected in ONNX — adding optimization profile "
            f"with min=1, opt={batch_size}, max={batch_size}."
        )
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        rest = tuple(input_shape[1:])
        profile.set_shape(input_name, (1,) + rest, (batch_size,) + rest, (batch_size,) + rest)
        config.add_optimization_profile(profile)

    _LOGGER.info("Building engine — this can take 5–15 minutes, please wait...")
    try:
        # Build + serialize to a portable byte blob (written to engine_path below;
        # model.py reloads it via deserialize_cuda_engine).
        serialized = builder.build_serialized_network(network, config)
    except Exception as exc:
        _LOGGER.error(f"Engine build raised exception: {exc}", exc_info=True)
        return False

    if serialized is None:
        _LOGGER.error("Engine build returned None — check TRT logs above for errors.")
        return False

    with open(engine_path, "wb") as f:
        f.write(serialized)

    size_mb = Path(engine_path).stat().st_size / 1024 / 1024
    _LOGGER.info("=" * 60)
    _LOGGER.info(f"Engine saved: {engine_path} ({size_mb:.1f} MB)")
    _LOGGER.info("=" * 60)
    return True


# ---------------------------------------------------------------------------
# Engine verification
# ---------------------------------------------------------------------------


def verify_engine(
    engine_path: str,
    batch_size: int,
    num_frames: int,
    resolution: int,
) -> bool:
    """
    Load the TRT engine and run a forward pass with random input.

    Prints the output shape and a few values so you can confirm the engine
    is healthy before deploying it in Triton.

    Args:
        engine_path: Path to .engine file.
        batch_size:  Must match the engine's fixed batch size.
        num_frames:  Must match the engine's fixed frame count.
        resolution:  Must match the engine's spatial resolution.

    Returns:
        True if verification passed.
    """
    try:
        import tensorrt as trt
    except ImportError:
        _LOGGER.warning("tensorrt not available — skipping engine verification.")
        return True

    if not torch.cuda.is_available():
        _LOGGER.warning("CUDA not available — skipping engine verification.")
        return True

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    _LOGGER.info(f"Verifying engine: {engine_path}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        _LOGGER.error("Failed to deserialize engine.")
        return False

    context = engine.create_execution_context()
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    # Torch-owned input / output buffers on GPU. We hand TRT raw data_ptr()s.
    input_tensor = torch.rand(batch_size, num_frames, 3, resolution, resolution,
                              dtype=torch.float32, device=device).contiguous()
    output_shape = tuple(context.get_tensor_shape(output_name))
    output_tensor = torch.empty(output_shape, dtype=torch.float32, device=device)

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        context.set_tensor_address(input_name, input_tensor.data_ptr())
        context.set_tensor_address(output_name, output_tensor.data_ptr())
        ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            _LOGGER.error("TensorRT execute_async_v3 returned False")
            return False
    stream.synchronize()

    # Apply softmax to get boundary probability (mirrors the Triton model)
    probs = torch.softmax(output_tensor, dim=1)
    boundary_scores = probs[:, 1].cpu().numpy()

    _LOGGER.info(f"Verification passed!")
    _LOGGER.info(f"  Input:            {tuple(input_tensor.shape)}  (batch, frames, C, H, W)")
    _LOGGER.info(f"  Output (logits):  {tuple(output_tensor.shape)}")
    _LOGGER.info(f"  Boundary scores:  {boundary_scores.round(4)}")

    del context, engine, runtime
    return True


# ---------------------------------------------------------------------------
# DDM source discovery
# ---------------------------------------------------------------------------


def find_ddm_base_path() -> str:
    """
    Locate the DDM checkout (containing DDM-Net/).

    Resolution order:
      1. $DDM_BASE_PATH if set — operator override.
      2. /opt/nvidia/nvds_sop/3rdparty/DDM — SOP image install (Docker.build:95).
      3. ./3rdparty/DDM relative to cwd — repo-root dev workflow.

    If none of the candidates have a DDM-Net/ subdir, returns the first
    candidate so --help still renders cleanly; load_ddm_model() then raises a
    clear FileNotFoundError when it actually tries to import.
    """
    explicit = os.getenv("DDM_BASE_PATH")
    if explicit:
        return explicit

    candidates = [
        "/opt/nvidia/nvds_sop/3rdparty/DDM",
        os.path.join(os.getcwd(), "3rdparty/DDM"),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "DDM-Net")):
            return c
    return candidates[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    default_ddm_base = find_ddm_base_path()

    parser = argparse.ArgumentParser(
        description="Export DDM-Net to TensorRT (batch_size=8, static shapes)"
    )

    # Model / checkpoint
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to DDM-Net PyTorch Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        choices=[224, 384, 512],
        help="Input spatial resolution H=W",
    )
    parser.add_argument(
        "--ddm-base-path",
        default=default_ddm_base,
        help=f"Root of DDM checkout containing DDM-Net/ subdir "
             f"(env: DDM_BASE_PATH, default: {default_ddm_base})",
    )
    parser.add_argument(
        "--frames-per-side",
        type=int,
        default=5,
        help="Frames on EACH side of the center frame. "
             "Total temporal window T = 2 * <value> + 1. "
             "(default: 5 → T = 11)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Fixed batch size baked into the engine (default: 8). "
             "Dynamic batch is not supported for this model; the runtime "
             "backend chunks+pads to whatever the engine was built with.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="./trt_output",
        help="Directory where .onnx and .engine files are written (default: ./trt_output). "
             "Ignored if --engine-path is set.",
    )
    parser.add_argument(
        "--engine-path",
        default=None,
        help="Full path to write the .engine file. If set, takes precedence "
             "over --output-dir + auto-derived filename. The .onnx file is "
             "written next to it with the same stem.",
    )

    # TRT options
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp16",
        help="Engine precision (default: fp16, the verified setting). "
             "Pick bf16 if FP16 numerical drift hurts accuracy.",
    )
    parser.add_argument(
        "--workspace-size",
        type=int,
        default=16,
        help="TensorRT workspace size in GB (default: 16)",
    )
    parser.add_argument("--trt-verbose", action="store_true", help="Enable verbose TensorRT logging")

    # Pipeline control
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX export — use existing .onnx file")
    parser.add_argument("--skip-trt", action="store_true", help="Only export ONNX, skip TRT build")
    parser.add_argument("--verify", action="store_true", help="Verify engine with a random forward pass after build")

    args = parser.parse_args()

    num_frames = 2 * args.frames_per_side + 1

    if args.engine_path:
        engine_path = Path(args.engine_path)
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        # ONNX shares the engine's directory and stem (just a different suffix).
        onnx_path = engine_path.with_suffix(".onnx")
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"ddm_res{args.resolution}_frs{args.frames_per_side}_b{args.batch_size}"
        onnx_path = output_dir / f"{stem}.onnx"
        engine_path = output_dir / f"{stem}_{args.precision}.engine"

    _LOGGER.info("=" * 60)
    _LOGGER.info("DDM-Net TensorRT Export")
    _LOGGER.info(f"  Checkpoint:    {args.checkpoint}")
    _LOGGER.info(f"  DDM base:      {args.ddm_base_path}")
    _LOGGER.info(f"  Resolution:    {args.resolution}")
    _LOGGER.info(f"  Frames/side:   {args.frames_per_side}  (total: {num_frames})")
    _LOGGER.info(f"  Batch size:    {args.batch_size}  (FIXED — no dynamic batch)")
    _LOGGER.info(f"  Precision:     {args.precision.upper()}")
    _LOGGER.info(f"  ONNX path:     {onnx_path}")
    _LOGGER.info(f"  Engine path:   {engine_path}")
    _LOGGER.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: ONNX export
    # ------------------------------------------------------------------
    if not args.skip_onnx:
        _LOGGER.info("[Step 1/2] Exporting PyTorch model → ONNX")
        model = load_ddm_model(
            checkpoint_path=args.checkpoint,
            ddm_base_path=args.ddm_base_path,
            frames_per_side=args.frames_per_side,
            resolution=args.resolution,
        )
        wrapped = DDMWrapper(model)
        wrapped.eval()
        with torch.no_grad():
            export_to_onnx(wrapped, str(onnx_path), args.batch_size, num_frames, args.resolution)
        del model, wrapped  # free memory before TRT build
    else:
        if not onnx_path.exists():
            _LOGGER.error(f"--skip-onnx set but ONNX file not found: {onnx_path}")
            sys.exit(1)
        _LOGGER.info(f"[Step 1/2] Skipped — using existing ONNX: {onnx_path}")

    # ------------------------------------------------------------------
    # Step 2: TensorRT engine build
    # ------------------------------------------------------------------
    if not args.skip_trt:
        _LOGGER.info("[Step 2/2] Building TensorRT engine from ONNX")
        success = build_trt_engine(
            onnx_path=str(onnx_path),
            engine_path=str(engine_path),
            batch_size=args.batch_size,
            precision=args.precision,
            workspace_gb=args.workspace_size,
            verbose=args.trt_verbose,
        )
        if not success:
            _LOGGER.error("TensorRT engine build failed.")
            sys.exit(1)

        if args.verify:
            _LOGGER.info("[Verify] Running forward pass on built engine")
            verify_engine(str(engine_path), args.batch_size, num_frames, args.resolution)
    else:
        _LOGGER.info("[Step 2/2] Skipped — only ONNX file was produced.")

    _LOGGER.info("=" * 60)
    _LOGGER.info("Done!")
    if onnx_path.exists():
        _LOGGER.info(f"  ONNX:   {onnx_path}  ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    if engine_path.exists():
        _LOGGER.info(f"  Engine: {engine_path}  ({engine_path.stat().st_size / 1e6:.1f} MB)")
    _LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
