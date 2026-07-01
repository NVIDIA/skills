# § 4 — Config File Templates

> **Generates**: `configs/nvds_preprocess_template.txt`, `configs/nvds_inference_template.txt`
> **Critical Rules**: DDM_TEMPORAL_CONFIGURABLE
> **Pipeline stage**: Stage 1 configuration — controls how nvdspreprocess builds the 5D tensor
> and how nvinferserver connects to Triton CAPI

---

## DDM temporal window (configurable)

> **CRITICAL [DDM_TEMPORAL_CONFIGURABLE]**: render `SLIDING_WINDOWS_SIZE = 2*FRAMES_PER_SIDE + SEQUENCE_BATCH`
> into the DeepStream preprocess `network-input-shape` and nvinferserver `inputs.dims`.
> Never hard-code 18. Keep the Triton model `config.pbtxt` sequence dimension as `-1` for
> generic purpose; `model.py` decouples each long window into sequence batches.
> `FRAMES_PER_SIDE` is checkpoint-bound (match the DDM ckpt); `SEQUENCE_BATCH` (> 1) is the
> runtime stride. Lower `FRAMES_PER_SIDE` for a smaller-context checkpoint.

```
one sliding window = SLIDING_WINDOWS_SIZE frames = 2*FRAMES_PER_SIDE + SEQUENCE_BATCH

  ┌── FRAMES_PER_SIDE ──┐┌──── SEQUENCE_BATCH ────┐┌── FRAMES_PER_SIDE ──┐
  │  left context       ││  center scored frames   ││  right context      │
  └─────────────────────┘└────────────────────────┘└─────────────────────┘
        pre-roll            stride = SEQUENCE_BATCH        post-roll
                            (one window emits SEQUENCE_BATCH boundary scores;
                             score i maps to temporal_pts[FRAMES_PER_SIDE + i] — § 3)

  defaults: FRAMES_PER_SIDE=5, SEQUENCE_BATCH=8 → SLIDING_WINDOWS_SIZE = 2*5+8 = 18
  example : FRAMES_PER_SIDE=1, SEQUENCE_BATCH=8 → SLIDING_WINDOWS_SIZE = 2*1+8 = 10

renders into DeepStream-facing configs (R = DS_ACTION_IN_RESOLUTION):
  preprocess  network-input-shape = 1;3;${SLIDING_WINDOWS_SIZE};R;R
  inference   dims                = [3, ${SLIDING_WINDOWS_SIZE}, R, R]
  preprocess  stride              = ${SEQUENCE_BATCH}

Triton model config stays generic:
  triton config.pbtxt dims        = [3, -1, R, R]
  model.py decouples each long window into SEQUENCE_BATCH-sized DDM calls
```

---

## nvds_preprocess_template.txt

Stored at `configs/nvds_preprocess_template.txt`. Variables replaced at startup with `string.Template`.

```ini
[property]
enable=1
target-unique-ids=1
process-on-frame=1

# 3D temporal tensor for DDM model:
#   batch=1, channels=3 (RGB), frames=SLIDING_WINDOWS_SIZE (sliding window), H=224, W=224
#   SLIDING_WINDOWS_SIZE = 2 * FRAMES_PER_SIDE + SEQUENCE_BATCH (defaults 2*5+8 = 18)
# FRAMES_PER_SIDE must match the DDM checkpoint's temporal context; SEQUENCE_BATCH is runtime grouping.
# This tensor is fed to the Triton DDM Python backend (§ 5)
network-input-shape= 1;3;${SLIDING_WINDOWS_SIZE};${DS_ACTION_IN_RESOLUTION};${DS_ACTION_IN_RESOLUTION}

network-color-format=0     # 0=RGB
network-input-order=2      # 2=CUSTOM (5D tensor for temporal model)
tensor-data-type=0         # 0=FP32
tensor-name=input_0
processing-width=${DS_ACTION_IN_RESOLUTION}
processing-height=${DS_ACTION_IN_RESOLUTION}

scaling-pool-memory-type=2    # 2=NVBUF_MEM_CUDA_DEVICE (GPU memory)
scaling-pool-compute-hw=1     # 1=GPU compute for scaling
scaling-filter=${DS_ACTION_IN_RESIZE_METHOD_ENUM}
# 0=nearest 1=bilinear 2=cubic 3=super 4=lanczos

tensor-buf-pool-size=8
# Custom sequence preprocessing library (part of DeepStream SDK)
custom-lib-path=/opt/nvidia/deepstream/deepstream/lib/libnvds_custom_sequence_preprocess.so
custom-tensor-preparation-function=CustomSequenceTensorPreparation

[user-configs]
# ImageNet normalization: scale factors and mean offsets for RGB channels
channel-scale-factors=0.017125;0.0175070;0.017429
channel-mean-offsets=123.675;116.280;103.530
# Advance by SEQUENCE_BATCH because one long window emits that many scores.
stride=${SEQUENCE_BATCH}
subsample=0

[group-0]
src-ids=-1
process-on-roi=0
```

---

## nvds_inference_template.txt

Format: protobuf text (pbtxt). The `model_repo {}` block enables **Triton CAPI** mode
(in-process inference via C API — no separate Triton server needed).

```
infer_config {
  unique_id: 1
  gpu_ids: [0]
  max_batch_size: 1
  backend {
    triton {
      model_name: "ddm"          # Must match directory name in triton_model_repo/ (§ 5)
      version: -1                # Use latest version
      model_repo {
        root: "../nvds_action_detector/triton_model_repo"
        strict_model_config: true
      }
    }
    inputs [{
      name: "input_0"
      # dims: [C=3, T=SLIDING_WINDOWS_SIZE, H, W] — the 5D tensor from nvdspreprocess
      dims: [ 3, ${SLIDING_WINDOWS_SIZE}, ${DS_ACTION_IN_RESOLUTION}, ${DS_ACTION_IN_RESOLUTION} ]
    }]
    disable_warmup: true         # We do our own warmup via create_dummy_pipeline() (§ 3)
    output_mem_type: MEMORY_TYPE_DEFAULT
  }
  input_tensor_from_meta { is_first_dim_batch: true }
  postprocess { other {} }
  extra {
    output_buffer_pool_size: 8
    # Links to the C++ custom postprocess plugin (§ 5b)
    custom_process_funcion: "CreateInferServerCustomProcess"
  }
  custom_lib {
    # Compiled during Docker build: cd custom_postprocess/ && make
    path: "../nvds_action_detector/custom_postprocess/libnvds_custom_postprocess_tensor.so"
  }
}
output_control { output_tensor_meta: true }
```

---

## Config Template Rendering (at startup)

```python
import string

# Template rendering happens when ds_3d_action_pipeline.py is imported.
# Replaces ${VAR} placeholders with runtime values, writes rendered files
# that are referenced by pipeline element config-file properties.

TEMPLATES = [
    ("configs/nvds_preprocess_template.txt", "configs/nvds_preprocess_rendered.txt"),
    ("configs/nvds_inference_template.txt",  "configs/nvds_inference_rendered.txt"),
    ("nvds_action_detector/triton_model_repo/ddm/config_template.pbtxt",
     "nvds_action_detector/triton_model_repo/ddm/config.pbtxt"),
]

# DDM temporal window — FRAMES_PER_SIDE is checkpoint-bound, SEQUENCE_BATCH is runtime grouping/stride.
# SLIDING_WINDOWS_SIZE = 2 * FRAMES_PER_SIDE + SEQUENCE_BATCH feeds both the preprocess tensor shape
# and the nvinferserver input dims so they stay consistent when frames_per_side is lowered.
# Triton model config uses sequence dim -1; this rendering still provides DS_ACTION_IN_RESOLUTION.
for template_path, output_path in TEMPLATES:
    with open(template_path) as f:
        tmpl = string.Template(f.read())
    with open(output_path, "w") as f:
        f.write(tmpl.safe_substitute(
            DS_ACTION_IN_RESOLUTION=DS_ACTION_IN_RESOLUTION,
            DS_ACTION_IN_RESIZE_METHOD_ENUM=DS_ACTION_IN_RESIZE_METHOD_MAP[DS_ACTION_IN_RESIZE_METHOD],
            FRAMES_PER_SIDE=FRAMES_PER_SIDE,
            SEQUENCE_BATCH=SEQUENCE_BATCH,
            SLIDING_WINDOWS_SIZE=SLIDING_WINDOWS_SIZE,
        ))
```
