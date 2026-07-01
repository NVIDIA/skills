# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import os
import sys
import threading
from typing import Optional

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.append("../../")
sys.path.append(os.getcwd())

import json
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack, to_dlpack

DDM_MODEL_PATH = os.getenv("DDM_MODEL_PATH", "/models/gbed_models/ddm/checkpoint.pth.tar")
FRAMES_PER_SIDE = int(os.getenv("FRAMES_PER_SIDE", "5"))
SEQUENCE_BATCH = int(os.getenv("SEQUENCE_BATCH", "8"))
DS_ACTION_IN_RESOLUTION = int(os.getenv("DS_ACTION_IN_RESOLUTION", "224"))

DDM_TRT_OPTIMIZATION = os.getenv("DDM_TRT_OPTIMIZATION", "false").lower() == "true"
DDM_TRT_ENGINE_OUTPUT_PATH = os.getenv("DDM_TRT_ENGINE_OUTPUT_PATH", "/tmp/trt_opt/ddm.engine")
DDM_TRT_PRECISION = os.getenv("DDM_TRT_PRECISION", "fp32")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
DDM_TRT_EXPORT_SCRIPT = os.path.join(_REPO_ROOT, "scripts/tensorrt/export_ddm_to_tensorrt.py")
DDM_TRT_BUILD_WORKSPACE_GB = os.getenv("DDM_TRT_BUILD_WORKSPACE_GB", "4")

# One DDM score uses this many frames around the center frame.
SINGLE_BATCH_SEQUENCE_FRAME_NUMBER = 2 * FRAMES_PER_SIDE + 1

# A long DeepStream request carries SEQUENCE_BATCH center frames plus context.
LONG_WINDOW_SIZE = 2 * FRAMES_PER_SIDE + SEQUENCE_BATCH

try:
    import nvds_action_detector.ds_logger as ds_logger

    logger = ds_logger.get_logger(__name__)
except ImportError:
    logger.debug("######## loading ds_logger failed, Using standard Python logger")
    logger = logging.getLogger(__name__)


class DDMTensorRTEngine:
    """
    TensorRT engine wrapper using PyTorch CUDA tensors throughout.

    Input/output stay on GPU as torch tensors. TRT receives raw data_ptr()
    pointers directly into PyTorch-owned GPU memory. Torch's CUDA module
    owns the device primary context; TRT picks it up implicitly.

    Execution context, CUDA stream, and output buffer are owned per thread
    so concurrent DeepStream pipelines do not share mutable TRT state
    (TRT IExecutionContext is not thread-safe).
    """

    def __init__(self, engine_path: str, device_id: int = 0):
        import tensorrt as trt

        self._device_id = device_id
        self._device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id)

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        logger.info(f"Loading TRT engine: {engine_path}")
        # Reload: deserialize the serialized .engine blob into one ICudaEngine for this process.
        with open(engine_path, "rb") as f:
            self._runtime = trt.Runtime(TRT_LOGGER)
            self._engine = self._runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            raise RuntimeError(f"Failed to load TRT engine from {engine_path}")

        self._input_name = self._engine.get_tensor_name(0)
        self._output_name = self._engine.get_tensor_name(1)

        self._input_shape = tuple(self._engine.get_tensor_shape(self._input_name))
        self._output_shape = tuple(self._engine.get_tensor_shape(self._output_name))

        # Fixed batch comes from the engine itself, not assumed to match SEQUENCE_BATCH.
        self.trt_batch_size = self._input_shape[0]
        # Non-batch dims (frames, C, H, W) — used to decide if PyTorch fallback is needed.
        self.input_non_batch_shape = self._input_shape[1:]

        # TRT IExecutionContext is NOT thread-safe → keep one context/stream/output-buffer PER
        # DeepStream worker thread (created lazily in _get_thread_state). _thread_states tracks them
        # only so destroy() can free every per-thread context.
        self._thread_state = threading.local()
        self._thread_states = []
        self._thread_states_lock = threading.Lock()

        logger.info(f"TRT engine ready")
        logger.info(f"  Input:  {self._input_name!r} {self._input_shape}")
        logger.info(f"  Output: {self._output_name!r} {self._output_shape}")
        logger.info(f"  TRT batch size: {self.trt_batch_size}")

    def _get_thread_state(self):
        state = getattr(self._thread_state, "trt_state", None)
        if state is not None:
            return state

        # First inference on this thread: build its own execution context + CUDA stream + reusable
        # output buffer (per-thread requirement — see __init__).
        context = self._engine.create_execution_context()
        stream = torch.cuda.Stream(device=self._device_id)
        output_buf = torch.empty(self._output_shape, dtype=torch.float32, device=self._device)

        state = {
            "thread_id": threading.get_ident(),
            "context": context,
            "stream": stream,
            "output_buf": output_buf,
        }
        self._thread_state.trt_state = state
        with self._thread_states_lock:
            self._thread_states.append(state)
        logger.info(f"Created TRT thread-local state for thread {state['thread_id']}")
        return state

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run one TRT call on exactly trt_batch_size windows.

        Returns:
            logits: (trt_batch_size, 2) FP32 on self._device.
                    Valid until the next infer() call on this thread.
        """
        state = self._get_thread_state()
        trt_stream = state["stream"]
        # NVBug 6289256 — TRT/gst-CV CUDA ordering fix.
        # input_tensor is produced by torch ops (cat/contiguous in execute()) on the
        # caller's CUDA stream, so order the TRT stream after it (read-before-write hazard).
        # Crucially, execute_async_v3 can enqueue work on TRT *auxiliary* streams beyond
        # trt_stream — synchronizing only trt_stream leaves TRT GPU work in flight, and the
        # downstream DeepStream gst-CV NvBufSurfTransform/cudaMemcpy2DAsync then races
        # freed/in-use surface memory -> driver-layer SIGSEGV (surfaces on the cached-engine
        # path). A full-device sync drains ALL TRT work (incl. aux streams) before returning.
        producer = torch.cuda.current_stream(device=self._device_id)
        trt_stream.wait_stream(producer)
        with torch.cuda.stream(trt_stream):
            # Bind input/output by GPU device pointer (zero-copy; tensors stay on-device).
            state["context"].set_tensor_address(self._input_name, input_tensor.data_ptr())
            state["context"].set_tensor_address(self._output_name, state["output_buf"].data_ptr())
            # Enqueue the TRT run on THIS thread's stream.
            ok = state["context"].execute_async_v3(stream_handle=trt_stream.cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 failed")

        torch.cuda.synchronize(self._device_id)   # device-wide drain (TRT may use aux streams)
        return state["output_buf"]

    def infer_batched(self, windows: torch.Tensor) -> torch.Tensor:
        """
        Run TRT inference over an arbitrary number of windows, chunking into
        trt_batch_size calls and padding the last chunk if needed.

        Args:
            windows: (total_windows, SINGLE_WINDOW_SIZE, 3, H, W) contiguous
                     FP32 on self._device. total_windows may be any value >= 1.

        Returns:
            logits: (total_windows, 2) FP32 on self._device.
        """
        # Engine batch is STATIC (trt_batch_size = SEQUENCE_BATCH): every infer() must get exactly bs
        # windows. Split into bs-sized groups, zero-pad the final partial group up to bs, run each,
        # then trim the padded rows back off the concatenated logits.
        total = windows.shape[0]
        bs = self.trt_batch_size

        if total == bs:
            return self.infer(windows).clone()

        remainder = total % bs
        pad = (bs - remainder) % bs   # rows to append so the last group is full

        if pad:
            full_chunks = windows.split(bs, dim=0)[:-1]
            last = windows[total - remainder:]
            padding = torch.zeros(pad, *windows.shape[1:], dtype=windows.dtype, device=windows.device)
            last_padded = torch.cat([last, padding], dim=0)
            all_chunks = list(full_chunks) + [last_padded]
        else:
            all_chunks = windows.split(bs, dim=0)

        logits = torch.cat([self.infer(chunk).clone() for chunk in all_chunks], dim=0)
        return logits[:total]   # drop logits produced for the zero-padded windows

    def destroy(self):
        try:
            with self._thread_states_lock:
                thread_states = list(self._thread_states)
                self._thread_states.clear()
            for state in thread_states:
                state["stream"].synchronize()
                del state["context"]
                del state["stream"]
                del state["output_buf"]
            del self._engine
            del self._runtime
        except Exception as e:
            logger.warning(f"Error during TRT engine cleanup: {e}")


class TritonPythonModel:

    def initialize(self, args):
        """
        This function allows
        the model to initialize any state associated with this model.
        """
        start_time = time.time()
        self._model_config = json.loads(args["model_config"])
        logger.info(f"######## self._model_config: {self._model_config}")
        # params = self._model_config["parameters"]
        self._device_id = int(json.loads(args["model_instance_device_id"]))
        self._cuda_device = f"cuda:{self._device_id}"
        self._preprocess_pipeline = None
        # T.Compose(
        #     [
        #         T.Resize((224, 224), antialias=False),
        #         T.ToDtype(torch.float32, scale=True),
        #         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ]
        # )
        self._frames_per_side = FRAMES_PER_SIDE
        self._ddm_net = None
        self._trt_engine = None

        logger.info(
            f"######## DDM init config: optimization={DDM_TRT_OPTIMIZATION} "
            f"engine_path={DDM_TRT_ENGINE_OUTPUT_PATH!r} precision={DDM_TRT_PRECISION} "
            f"sequence_batch={SEQUENCE_BATCH} resolution={DS_ACTION_IN_RESOLUTION}"
        )

        if DDM_TRT_OPTIMIZATION:
            self._trt_engine = self._init_trt_engine()

        if self._trt_engine is None:
            # Either DDM_TRT_OPTIMIZATION=false, or TRT load+build both failed.
            # PyTorch becomes the chosen runtime for the rest of the process.
            # Fail fast with an actionable message if the checkpoint is missing,
            # instead of letting load_ddm_net_model raise an opaque load error
            # that surfaces only as a Triton model-load failure.
            if not os.path.isfile(DDM_MODEL_PATH):
                raise FileNotFoundError(
                    f"DDM checkpoint not found at {DDM_MODEL_PATH!r}. A fresh box has no "
                    "checkpoint until you fine-tune/download one. Obtain a DDM-Net "
                    "checkpoint (e.g. via the SOP Training Blueprint), place it under "
                    "MODEL_ROOT_DIR, and set DDM_MODEL_PATH in deploy/.env to point at it."
                )
            from ddm_net import load_ddm_net_model
            self._ddm_net = load_ddm_net_model(
                checkpoint=DDM_MODEL_PATH, frames_per_side=self._frames_per_side, gpu_id=self._device_id
            )

        logger.info(
            f"######## DDM model initialized successfully "
            f"(runtime={'TRT' if self._trt_engine is not None else 'PyTorch'}), "
            f"initialization time: {time.time() - start_time:.3f}s"
        )

    def _init_trt_engine(self) -> Optional[DDMTensorRTEngine]:
        """
        Step 1: load a cached engine from DDM_TRT_ENGINE_OUTPUT_PATH if it
        exists and is shape-compatible with the current env config. Compat
        means input non-batch shape (FRAMES_PER_SIDE, DS_ACTION_IN_RESOLUTION)
        AND batch size (SEQUENCE_BATCH). DDM_TRT_PRECISION is NOT checked —
        once an engine file exists, switching precision via env does not
        retrigger a build. To actually change precision, delete the cached
        engine file.
        Step 2: build one on the fly into the same path.
        Returns None on total failure so the caller can fall back to PyTorch.
        """
        expected_non_batch = (
            2 * FRAMES_PER_SIDE + 1,
            3,
            DS_ACTION_IN_RESOLUTION,
            DS_ACTION_IN_RESOLUTION,
        )
        expected_batch = SEQUENCE_BATCH

        if os.path.isfile(DDM_TRT_ENGINE_OUTPUT_PATH):
            engine = self._try_load_engine(
                DDM_TRT_ENGINE_OUTPUT_PATH, expected_non_batch, expected_batch
            )
            if engine is not None:
                return engine
            logger.info("######## Cached engine unusable — rebuilding on the fly")

        parent = os.path.dirname(os.path.abspath(DDM_TRT_ENGINE_OUTPUT_PATH))
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as exc:
            logger.warning(
                f"######## cannot create engine output dir {parent!r}: {exc} — "
                f"falling back to PyTorch"
            )
            return None
        if not os.access(parent, os.W_OK):
            logger.warning(
                f"######## engine output dir {parent!r} is not writable by this user — "
                f"falling back to PyTorch"
            )
            return None

        logger.info(
            f"######## Building DDM TRT engine on the fly at {DDM_TRT_ENGINE_OUTPUT_PATH!r} — "
            f"this can take 5-15 minutes, please wait..."
        )
        if not self._build_engine_subprocess(DDM_TRT_ENGINE_OUTPUT_PATH):
            logger.warning("######## TRT engine build failed — falling back to PyTorch")
            return None

        return self._try_load_engine(
            DDM_TRT_ENGINE_OUTPUT_PATH, expected_non_batch, expected_batch
        )

    def _try_load_engine(
        self,
        engine_path: str,
        expected_non_batch: Tuple[int, ...],
        expected_batch: int,
    ) -> Optional[DDMTensorRTEngine]:
        """
        Load an engine and verify shape compatibility with current env config.
        Returns None (and destroys the partial engine) on any mismatch or error.
        """
        try:
            engine = DDMTensorRTEngine(engine_path, device_id=self._device_id)
        except Exception as exc:
            logger.warning(f"######## TRT engine load failed: {exc}")
            return None

        if (
            engine.input_non_batch_shape != expected_non_batch
            or engine.trt_batch_size != expected_batch
        ):
            logger.warning(
                f"######## Engine shape mismatch: have batch={engine.trt_batch_size} "
                f"non_batch={engine.input_non_batch_shape}, want batch={expected_batch} "
                f"non_batch={expected_non_batch}"
            )
            engine.destroy()
            return None

        return engine

    def _build_engine_subprocess(self, engine_path: str) -> bool:
        """
        Run export_ddm_to_tensorrt.py in a subprocess. Writes to a sibling
        ``.tmp`` path and atomically renames on success so a killed build
        never leaves a half-written engine. Memory used by the export
        (PyTorch DDM model + TRT builder workspace) is fully released when
        the subprocess exits.
        """
        import subprocess

        if not os.path.isfile(DDM_TRT_EXPORT_SCRIPT):
            logger.warning(
                f"######## TRT export script not found at {DDM_TRT_EXPORT_SCRIPT!r}"
            )
            return False
        if not os.path.isfile(DDM_MODEL_PATH):
            logger.warning(
                f"######## DDM checkpoint not found at {DDM_MODEL_PATH!r} — cannot build"
            )
            return False

        tmp_path = engine_path + ".tmp"
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        cmd = [
            sys.executable,
            DDM_TRT_EXPORT_SCRIPT,
            "--checkpoint", DDM_MODEL_PATH,
            "--resolution", str(DS_ACTION_IN_RESOLUTION),
            "--precision", DDM_TRT_PRECISION,
            "--batch-size", str(SEQUENCE_BATCH),
            "--frames-per-side", str(FRAMES_PER_SIDE),
            "--engine-path", tmp_path,
            "--workspace-size", DDM_TRT_BUILD_WORKSPACE_GB,
        ]
        logger.info(f"######## TRT build command: {' '.join(cmd)}")
        build_start = time.time()
        try:
            # Inherit stdout/stderr so the export's progress logs are visible.
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            logger.warning(f"######## TRT export subprocess failed (rc={exc.returncode})")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            return False

        if not os.path.isfile(tmp_path):
            logger.warning("######## TRT export claimed success but produced no engine file")
            return False

        try:
            os.replace(tmp_path, engine_path)
        except OSError as exc:
            logger.warning(f"######## could not rename {tmp_path!r} -> {engine_path!r}: {exc}")
            return False

        logger.info(
            f"######## TRT engine built in {time.time() - build_start:.1f}s -> {engine_path!r}"
        )
        return True

    def execute(self, requests):
        # logger = pb_utils.Logger
        torch.set_default_device(self._cuda_device)
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        logger.debug(f"######## requests numbers: {len(requests)}")
        if len(requests) > 1:
            logger.info(f"######## requests numbers: {len(requests)} more than 1")
        batched_inputs, batch_nums = [], []
        for request in requests:
            # input_0: [N, C, 2 * FRAMES_PER_SIDE + SEQUENCE_BATCH, H, W]
            input_0 = pb_utils.get_input_tensor_by_name(request, "input_0")
            tensors = from_dlpack(input_0.to_dlpack())
            logger.debug(f"######## original input_0.shape: {tensors.shape}, input_0.device: {tensors.device}")
            # Convert from [N, C, T, H, W] to [N, T, C, H, W] for DDM-Net.
            tensors = torch.permute(tensors, (0, 2, 1, 3, 4))
            if self._preprocess_pipeline is not None:
                with torch.no_grad():
                    preprocessed_tensors = self._preprocess_pipeline(tensors)
                    logger.debug(
                        f"######## preprocessed input_0.shape: {tensors.shape}, input_0.device: {tensors.device}"
                    )
            else:
                preprocessed_tensors = tensors
            # inputs.append(tensors)

            long_window_size = tensors.shape[1]
            assert (
                long_window_size > SINGLE_BATCH_SEQUENCE_FRAME_NUMBER
            ), f"Long window size{long_window_size} less than single batch sequence frame number{SINGLE_BATCH_SEQUENCE_FRAME_NUMBER}"
            seq_batch_size = long_window_size - SINGLE_BATCH_SEQUENCE_FRAME_NUMBER + 1
            for i in range(seq_batch_size):
                window = preprocessed_tensors[:, i : i + SINGLE_BATCH_SEQUENCE_FRAME_NUMBER]
                batched_inputs.append(window)
            batch_nums.append(seq_batch_size)

        batched_inputs = torch.cat(batched_inputs, dim=0)

        # Make tensor contiguous if needed (this is necessary after permute)
        if not batched_inputs.is_contiguous():
            batched_inputs = batched_inputs.contiguous()

        # Validate tensor properties
        if not batched_inputs.is_contiguous():
            raise RuntimeError(
                f"Tensor is not contiguous after operations. Shape: {batched_inputs.shape}, Device: {batched_inputs.device}"
            )

        logger.debug(
            f"######## batched_inputs shape: {batched_inputs.shape}, device: {batched_inputs.device}, contiguous: {batched_inputs.is_contiguous()}"
        )

        with torch.no_grad():
            if self._trt_engine is not None:
                batched_outputs = self._trt_engine.infer_batched(batched_inputs)
            else:
                batched_outputs, _, _ = self._ddm_net(batched_inputs)
            if isinstance(batched_outputs, (tuple, list)):
                batched_outputs = batched_outputs[-1]
            # window_scores shape: (seq_batch_size,), (8)
            window_scores = F.softmax(batched_outputs, dim=1)[:, 1]
            logger.debug(f"######## window_scores, shape: {window_scores.shape}, data: {window_scores}")
        # window_scores shape: [8] -> we need to keep it as 2D tensor [seq_batch_size, 1] for proper de-batching
        window_scores = window_scores.unsqueeze(1)  # Shape: [seq_batch_size2, 1]
        logger.debug(f"######## window_scores after unsqueeze, shape: {window_scores.shape}, data: {window_scores}")

        # Split into individual outputs for each request
        # len(outputs) = len(batch_nums)
        # outputs = [tensor.unsqueeze(0) for tensor in torch.unbind(window_scores, dim=0)]  # Each becomes [seq_batch_size, 1]
        outputs = [x for x in torch.split(window_scores, batch_nums)]
        logger.debug(f"######## ddm decoupled outputs size: {len(outputs)}")
        responses = []

        for i, request in enumerate(requests):
            output = outputs[i]
            seq_batch_size = batch_nums[i]
            output = output.reshape(-1, seq_batch_size)
            logger.debug(f"######## output_0: shape: {output.shape}, data: {output}")
            out_tensor = pb_utils.Tensor("output_0", output.cpu().numpy())
            # out_tensor = pb_utils.Tensor.from_dlpack("output_0", to_dlpack(output))
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)
        return responses

    def finalize(self):
        logger.info("Cleaning up ddm model...")
        if self._trt_engine is not None:
            self._trt_engine.destroy()
            self._trt_engine = None
