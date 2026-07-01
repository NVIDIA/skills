# § 5 — Triton GEBD Model Repository (default: DDM)

> **Generates**: `nvds_action_detector/triton_model_repo/ddm/` directory
> **Reference files to copy**: `triton_config_template.pbtxt`, `triton_model_reference.py`, `ddm_net_reference.py`
> **Critical Rules**: DDM_TEMPORAL_CONFIGURABLE, DDM_TRT_OPTIONAL_PATH, DDM_TRT_STREAM_ORDERING
> **Pipeline stage**: Stage 1 backend — **Generic Event Boundary Detection (GEBD)** model served via Triton Python backend. The default GEBD model shipped here is **DDM** (MCG-NJU/DDM).

---

## GEBD ↔ DDM relationship

Stage 1 expects any **GEBD-style** network — i.e. a model that ingests a sliding
window of frames and emits a per-frame **boundary score** (probability that the
current frame is a temporal event boundary). The pipeline downstream of Stage 1
only consumes that score sequence and does not depend on a specific architecture.

DDM is the **default** GEBD model packaged with this skill: a temporal sliding-
window model from [MCG-NJU/DDM](https://github.com/MCG-NJU/DDM). All references
to "DDM" in this skill describe the default; substitute another GEBD model by:

1. Replacing `triton_model_repo/ddm/1/model.py` and `ddm_net.py` with the new
   model's Triton Python backend wrapper. (DDM's own Python backend also ships an
   optional in-process **TensorRT** path — see "Optional TensorRT path
   (DDM_TRT_OPTIMIZATION)" below — which is distinct from swapping `config.pbtxt`
   `backend:` to a native `onnxruntime` / `tensorrt` backend.)
2. Keeping the input/output **tensor contract**: input window shape
   `[batch, T, C, H, W]` (T = sliding-window length) and output tensor shape
   `[batch, T]` of per-frame boundary scores in `[0, 1]`.
3. Updating `DDM_MODEL_PATH` to the new checkpoint and (if window length
   differs) the `SLIDING_WINDOWS_SIZE` / `FRAMES_PER_SIDE` constants used by
   `ds_3d_action_pipeline.py` and `SOPVideoProcessor`.
4. Leaving `nvdspreprocess` template (§ 4) and the
   `InferenceOutputTensorParser` (§ 3) unchanged — they're model-agnostic as
   long as the tensor contract above holds.

The remainder of this section documents the DDM-specific defaults; treat them
as the reference implementation of the GEBD slot.

---

## Directory Structure

```
nvds_action_detector/triton_model_repo/
└── ddm/
    ├── config_template.pbtxt   # Template (vars replaced at startup by § 4 rendering)
    ├── config.pbtxt            # Generated at runtime (DO NOT hand-edit)
    └── 1/
        ├── model.py            # Triton Python backend entry point
        └── ddm_net.py          # DDM network loader (from MCG-NJU/DDM + pytorch2 patch)
```

---

## config_template.pbtxt

**CRITICAL**: (1) Do NOT add `EXECUTION_ENV_PATH` — Triton Python backend fails at
startup if the path doesn't exist; use `FORCE_CPU_ONLY_INPUT_TENSORS: "no"` instead.
(2) Input `dims` second element must be `-1` (variable sequence length), not `18` —
`model.py` handles sliding-window batching internally.

```
name: "ddm"
backend: "python"
max_batch_size: 4

dynamic_batching {
    max_queue_delay_microseconds: 1000
}

input [{
  name: "input_0"
  data_type: TYPE_FP32
  # dims: [C=3, T=-1 (variable), H, W]
  # T is variable because model.py splits the 18-frame window into
  # overlapping 11-frame sub-windows internally
  dims: [3, -1, ${DS_ACTION_IN_RESOLUTION}, ${DS_ACTION_IN_RESOLUTION}]
}]
output [{
  name: "output_0"
  data_type: TYPE_FP32
  dims: [-1]    # One boundary score per sub-window
}]
instance_group [{ count: 1 kind: KIND_GPU gpus: 0 }]
parameters {
  key: "FORCE_CPU_ONLY_INPUT_TENSORS"
  value { string_value: "no" }
}
```

---

## model.py (Triton Python Backend)

**CRITICAL**: `model.py` must import from a **local** `ddm_net.py` in the same
directory (`from ddm_net import load_ddm_net_model`). Do NOT import `libs.core`
or `libs.modeling` — they live in `3rdparty/DDM/` and are not on Triton's Python
path. `ddm_net.py` handles path setup via `DDM_BASE_PATH` env var.

The execute method must handle **sliding-window batching**: the input tensor has
variable sequence length `[B, C, T, H, W]` where `T` can be 18 (=LONG_WINDOW_SIZE).
It must be permuted to `[B, T, C, H, W]`, split into overlapping windows of size
`SINGLE_BATCH_SEQUENCE_FRAME_NUMBER` (11), run through the DDM net, and the output
scores processed with `F.softmax(..., dim=1)[:, 1]`.

```python
import triton_python_backend_utils as pb_utils
import torch, torch.nn.functional as F, os, sys, json, time, logging
from torch.utils.dlpack import from_dlpack, to_dlpack

sys.path.append("../../")
sys.path.append(os.getcwd())

# --- DDM model constants ---
DDM_MODEL_PATH = os.getenv("DDM_MODEL_PATH", "/models/gbed_models/ddm/checkpoint.pth.tar")
FRAMES_PER_SIDE = int(os.getenv("FRAMES_PER_SIDE", "5"))
SEQUENCE_BATCH = int(os.getenv("SEQUENCE_BATCH", "8"))
# FRAMES_PER_SIDE is checkpoint-bound — lower it (e.g. 1, 2, 3) for a smaller-context DDM
# checkpoint. SEQUENCE_BATCH is runtime grouping/stride. Constraint: SEQUENCE_BATCH > 1, so
# LONG_WINDOW_SIZE > SINGLE_BATCH_SEQUENCE_FRAME_NUMBER (model.py asserts this). Comments show defaults.
SINGLE_BATCH_SEQUENCE_FRAME_NUMBER = 2 * FRAMES_PER_SIDE + 1  # 11 per sub-window (defaults 5,8)
LONG_WINDOW_SIZE = 2 * FRAMES_PER_SIDE + SEQUENCE_BATCH       # 18 total window (defaults 5,8)

class TritonPythonModel:
    def initialize(self, args):
        """Load DDM model on the assigned GPU device."""
        self._device_id = int(json.loads(args["model_instance_device_id"]))
        self._cuda_device = f"cuda:{self._device_id}"
        # Import from LOCAL ddm_net.py — NOT from libs.* (see § 5 notes)
        from ddm_net import load_ddm_net_model
        self._ddm_net = load_ddm_net_model(
            checkpoint=DDM_MODEL_PATH, frames_per_side=FRAMES_PER_SIDE, gpu_id=self._device_id
        )
        # Dummy tensor for batch-size-1 workaround (DDM requires batch >= 2)
        self.dummy_tensor = torch.randint(
            0, 256, (1, SINGLE_BATCH_SEQUENCE_FRAME_NUMBER, 3, 224, 224),
            dtype=torch.float32, device=self._cuda_device
        )

    def execute(self, requests):
        """Run DDM inference with sliding-window batching.

        Input: [B, C, T, H, W] tensor from nvdspreprocess (T=18)
        Processing:
          1. Permute to [B, T, C, H, W]
          2. Split into overlapping sub-windows of size 11 (stride=1)
             → 18 - 11 + 1 = 8 sub-windows per request
          3. Concatenate all sub-windows across requests into one batch
          4. Run DDM net → softmax → extract boundary probability (class 1)
          5. Return per-request score vectors
        """
        torch.set_default_device(self._cuda_device)
        batched_inputs, batch_nums = [], []
        for request in requests:
            input_0 = pb_utils.get_input_tensor_by_name(request, "input_0")
            tensors = from_dlpack(input_0.to_dlpack())
            tensors = torch.permute(tensors, (0, 2, 1, 3, 4))  # [B,C,T,H,W] → [B,T,C,H,W]
            seq_batch_size = tensors.shape[1] - SINGLE_BATCH_SEQUENCE_FRAME_NUMBER + 1
            for i in range(seq_batch_size):
                batched_inputs.append(tensors[:, i : i + SINGLE_BATCH_SEQUENCE_FRAME_NUMBER])
            batch_nums.append(seq_batch_size)

        # DDM requires batch_size >= 2; pad with dummy if only 1 sub-window
        if len(batched_inputs) == 1:
            batched_inputs.append(self.dummy_tensor)
            batch_nums.append(1)

        batched_inputs = torch.cat(batched_inputs, dim=0).contiguous()
        with torch.no_grad():
            batched_outputs, _, _ = self._ddm_net(batched_inputs)
            if isinstance(batched_outputs, (tuple, list)):
                batched_outputs = batched_outputs[-1]
            # Softmax → class 1 probability = boundary score
            window_scores = F.softmax(batched_outputs, dim=1)[:, 1].unsqueeze(1)

        outputs = list(torch.split(window_scores, batch_nums))
        responses = []
        for i, request in enumerate(requests):
            output = outputs[i].reshape(-1, batch_nums[i])
            out_tensor = pb_utils.Tensor("output_0", output.cpu().numpy())
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses

    def finalize(self):
        del self._ddm_net
```

> The inline snippet above is the **PyTorch-only** teaching version. The file actually copied
> from `references/triton_model_reference.py` additionally contains the optional TensorRT path
> documented next; the two runtimes are mutually exclusive (exactly one is loaded per process).

---

## Optional TensorRT path (`DDM_TRT_OPTIMIZATION`)

> **CRITICAL [DDM_TRT_OPTIONAL_PATH]**: TensorRT is OPT-IN (`DDM_TRT_OPTIMIZATION=true`); PyTorch is the
> default. Acquire the engine as load XOR build XOR fall-back — never load both runtimes. Build with a
> static batch = `SEQUENCE_BATCH`; keep a per-thread `IExecutionContext` (TRT contexts are not thread-safe).

Engine lifecycle at Triton `initialize()` (`_init_trt_engine`):

```
DDM_TRT_OPTIMIZATION == true ?
  │ no ───────────────────────────────────────────► load PyTorch DDM (self._ddm_net)   [default]
  │ yes
  ▼
cached .engine at DDM_TRT_ENGINE_OUTPUT_PATH with matching input-shape + batch(SEQUENCE_BATCH) ?
  │ yes ─► reload (deserialize) the serialized .engine ──────────────┐  (precision NOT re-checked)
  │ no                                                                │
  ▼                                                                   │
build on the fly  (subprocess: scripts/tensorrt/export_ddm_to_tensorrt.py):                       │
  ckpt ─► ONNX (opset 18, static batch=SEQUENCE_BATCH) ─► TRT build (workspace=DDM_TRT_BUILD_WORKSPACE_GB)
       ─► write .engine.tmp ─► atomic rename ─► deserialize ─────────┤
                                                                      ▼
                                              ICudaEngine ready  ──(build AND load failed?)──► PyTorch fallback
                                                                      ▼
                          per worker thread (lazy): IExecutionContext + CUDA stream + output buffer
```

Runtime inference path (DDM served in-process via Triton CAPI):

```
nvstreammux ─► nvdspreprocess (5D tensor) ─► nvinferserver (Triton CAPI) ─► model.py execute()
                                                                                │
                                              if self._trt_engine:  infer_batched()   else:  self._ddm_net()
                                                                                │  (TRT: split windows into
                                                                                │   SEQUENCE_BATCH groups,
                                                                                │   zero-pad last, run, trim pad)
                                                                                ▼
                              C++ custom postprocess (§ 5b) ─► InferenceOutputTensorParser ─► score_queue ─► Stage 2
```

The DDM Triton Python backend can run the network through **TensorRT** instead of PyTorch.
It is **off by default** — enable with `DDM_TRT_OPTIMIZATION=true` (§ 11). PyTorch and TRT
are **never loaded together**; exactly one runtime is chosen per process.

**Engine acquisition at Triton `initialize()`** (`_init_trt_engine()`):
1. **Cached engine** — if a file exists at `DDM_TRT_ENGINE_OUTPUT_PATH` and its embedded input
   shape `(2*FRAMES_PER_SIDE+1, 3, DS_ACTION_IN_RESOLUTION, DS_ACTION_IN_RESOLUTION)` **and**
   batch size (`SEQUENCE_BATCH`) match the current env, it is reused as-is.
   ⚠️ **Precision is NOT re-checked** — switching `DDM_TRT_PRECISION` (fp32/fp16/bf16) keeps
   using an existing engine; delete the engine file to force a rebuild.
2. **Build on the fly** — otherwise a subprocess runs `scripts/tensorrt/export_ddm_to_tensorrt.py`
   (PyTorch ckpt → ONNX opset 18 → TRT engine), writing a `.tmp` file and atomically renaming on
   success. First build takes ~5–15 min; builder workspace from `DDM_TRT_BUILD_WORKSPACE_GB`.
3. **Fallback** — if load and build both fail (missing checkpoint, unwritable path, build crash),
   `self._trt_engine` stays `None` and the backend loads the PyTorch model instead, so the
   service still starts.

**Per-thread TRT state** — a TRT `IExecutionContext` is **not thread-safe**, so `DDMTensorRTEngine`
keeps a `threading.local()` execution context + CUDA stream + output buffer per DeepStream worker
thread (created lazily on first inference per thread).

**[DDM_TRT_STREAM_ORDERING]** — Order the per-thread TRT stream against the caller's stream,
then drain the whole device after the run. In `infer()`, capture the producer stream with
`torch.cuda.current_stream(device=...)`; call `trt_stream.wait_stream(producer)` before binding
the input; run `execute_async_v3` inside `with torch.cuda.stream(trt_stream)`; then call
`torch.cuda.synchronize(self._device_id)`. Do NOT rely on `trt_stream.synchronize()` alone —
`execute_async_v3` can enqueue work on TRT *auxiliary* streams beyond `trt_stream`, so a
per-stream sync leaves TRT GPU work in flight and the downstream DeepStream gst-CV
`NvBufSurfTransform` / `cudaMemcpy2DAsync` then races freed/in-use surface memory → driver-layer
SIGSEGV (fix for NVBug 6289256; surfaces on the cached-engine reload path). A device-wide sync
drains all TRT work — including aux streams — before the pipeline reuses the surfaces.

**Fixed batch** — the engine is built with a **static batch = `SEQUENCE_BATCH`** (DDM's pairwise-
distance op doesn't propagate a dynamic batch cleanly). `infer_batched()` chunks the sliding
windows into `SEQUENCE_BATCH`-sized groups, zero-pads the last partial group, and trims the padding.

**`execute()` branch:** `if self._trt_engine is not None: infer_batched(...) else: self._ddm_net(...)`.

**Dependencies & shipping:** onnx/onnxscript/onnxruntime are installed in the image (§ 9); the
export script is shipped via `package_file_list.txt` (§ 14). The default cache path
`/tmp/trt_opt/ddm.engine` rebuilds on every container restart — point `DDM_TRT_ENGINE_OUTPUT_PATH`
under a mounted `MODEL_ROOT_DIR` path to persist the engine.

**Manual ahead-of-time export** (optional; skips the first-run build):
```bash
python scripts/tensorrt/export_ddm_to_tensorrt.py \
    --checkpoint "$DDM_MODEL_PATH" --resolution "$DS_ACTION_IN_RESOLUTION" \
    --precision fp16 --batch-size "$SEQUENCE_BATCH" --frames-per-side "$FRAMES_PER_SIDE" \
    --engine-path "$DDM_TRT_ENGINE_OUTPUT_PATH" --verify
```

Full implementation: `references/triton_model_reference.py` (`DDMTensorRTEngine` + `_init_trt_engine`)
and `references/export_ddm_to_tensorrt_reference.py`. Troubleshooting log markers are in § 12.

---

## ddm_net.py (DDM model loader)

**CRITICAL**: Must exist at `triton_model_repo/ddm/1/ddm_net.py` alongside `model.py`.
Wraps `modeling.resnetGEBD` from DDM clone and handles `sys.path` insertion for Triton.
Copy from `references/ddm_net_reference.py`.

```python
import os, sys, logging, argparse
import torch

# DDM_BASE_PATH points to the DDM git clone (3rdparty/DDM/)
# The PyTorch 2 compatibility patch must have been applied (ddm_pytorch2.patch)
DDM_BASE_PATH = os.getenv("DDM_BASE_PATH", f"{os.getcwd()}/3rdparty/DDM/")

def load_ddm_net_model(checkpoint: str, frames_per_side: int = 5, gpu_id: int = 0):
    """Load DDM resnetGEBD model from checkpoint.

    Temporarily inserts DDM-Net modeling path into sys.path to import resnetGEBD,
    then removes it to avoid polluting the module namespace.
    """
    modeling_path = os.path.join(DDM_BASE_PATH, "DDM-Net")
    try:
        sys.path.insert(0, os.path.abspath(modeling_path))
        from modeling.resnetGEBD import resnetGEBD
    finally:
        sys.path.pop(0)

    model = resnetGEBD(backbone="resnet50", pretrained=False,
                       num_classes=2, frames_per_side=frames_per_side)

    # Load checkpoint with safe_globals for argparse.Namespace (PyTorch 2 compat)
    with torch.serialization.safe_globals([argparse.Namespace]):
        ckpt = torch.load(checkpoint, map_location="cpu")

    # Strip model. or module. prefix from state dict keys
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = {}
        for key, value in ckpt["state_dict"].items():
            for prefix in ("model.", "module."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            state_dict[key] = value
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model = model.to(f"cuda:{gpu_id}").eval()
    return model
```
