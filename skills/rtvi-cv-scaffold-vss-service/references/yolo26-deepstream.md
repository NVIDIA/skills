# YOLO26 in a DeepStream Pipeline

How the scaffolded service wires YOLO26 into a DeepStream `nvinfer`
primary GIE, and what the customer is responsible for providing.

## What the scaffolder generates

The scaffolder writes a **template** `pgie-yolo26-config.txt` with the
DeepStream `nvinfer` keys pre-set:

```ini
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-color-format=0
onnx-file=/opt/models/yolo26.onnx
model-engine-file=/opt/models/yolo26.onnx_b1_gpu0_fp16.engine
labelfile-path=/opt/configs/labels.txt
batch-size=1
network-mode=2                # 0=FP32 1=INT8 2=FP16
num-detected-classes=<NUM_CLASSES>
interval=0
gie-unique-id=1
process-mode=1                # 1=primary
network-type=0                # 0=detector
cluster-mode=2                # 2=NMS
maintain-aspect-ratio=1
parse-bbox-func-name=NvDsInferParseYolo26
custom-lib-path=/opt/parser/libnvdsinfer_custom_yolo26.so

[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold=0.25
topk=300
```

The scaffolder leaves three things as placeholders the customer must
provide:

1. **`onnx-file`** — the YOLO26 export. Path inside the container is
   `/opt/models/yolo26.onnx`; bind-mount the host path the customer
   chooses.
2. **`labelfile-path`** — one class name per line, in the order the model
   was trained. `num-detected-classes` must equal the line count.
3. **`custom-lib-path` + `parse-bbox-func-name`** — a custom output
   parser for YOLO26's output tensor layout. The scaffolder writes
   `pipeline/custom-parser/README.md` with the expected ABI:

   ```c
   extern "C" bool NvDsInferParseYolo26(
       std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
       NvDsInferNetworkInfo const &networkInfo,
       NvDsInferParseDetectionParams const &detectionParams,
       std::vector<NvDsInferParseObjectInfo> &objectList);
   ```

   YOLO26's anchor-free head layout differs from YOLOv11 — the customer
   compiles the parser from their training repo (or from the YOLO26
   reference parser if NVIDIA ships one) and bind-mounts the resulting
   `.so` at `/opt/parser/libnvdsinfer_custom_yolo26.so`.

## Tracker

The pipeline uses the NvDCF tracker out of the box. The scaffolder
writes `tracker-nvdcf.yml` referencing the DeepStream-shipped low-level
config:

```ini
[tracker]
ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
ll-config-file=/opt/configs/tracker-nvdcf.yml
tracker-width=960
tracker-height=544
```

NvDCF is the recommended tracker for VSS — it produces stable
`Object.id` across frames, which Behavior Analytics depends on.

## TensorRT engine cache

First container start runs the ONNX → TRT engine build, which on
RTX-class GPUs takes 2–10 minutes for a YOLO26-sized model. The scaffolded
Compose file mounts a named volume at `/opt/models` and overlays the
customer's read-only ONNX file at `/opt/models/yolo26.onnx`. DeepStream 9.1
serializes a newly built engine beside the ONNX as
`/opt/models/yolo26.onnx_b1_gpu0_fp16.engine`, even when a different missing
`model-engine-file` path is configured. Pointing `model-engine-file` at that
actual sibling path lets subsequent starts load the engine from the persistent
model volume. Change the filename when batch size, GPU ID, or precision
changes. Customers running on Jetson should rebuild the engine when the device
changes.

## What the scaffolder does *not* handle

- Producing the YOLO26 ONNX. The customer trains/exports their model
  outside this skill. If reference weights or export guidance are needed,
  see [Ultralytics/YOLO26](https://huggingface.co/Ultralytics/YOLO26).
- Compiling the custom parser. The skill provides the ABI signature and
  config wiring; the customer compiles their parser against the
  DeepStream version they target.
- Class-aware post-processing. If the customer needs class-conditional
  NMS thresholds, they edit the `[class-attrs-<id>]` blocks in
  `pgie-yolo26-config.txt`.

## References

- Ultralytics YOLO26 — `https://docs.ultralytics.com/models/yolo26`
- Hugging Face Ultralytics/YOLO26 — `https://huggingface.co/Ultralytics/YOLO26`
- DeepStream `Gst-nvinfer` — `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html`
- DeepStream custom parser interface — `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html#custom-bounding-box-parsing`
- NvDCF tracker config reference — `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html`
