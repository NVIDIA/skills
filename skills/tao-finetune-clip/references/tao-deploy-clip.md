# CLIP Deploy

CLIP deploy covers the TAO Deploy actions for an exported multimodal embedding model. Use the `clip` model skill for training, checkpoint evaluation, quantization, distillation, pruning, export, or non-TensorRT inference where those actions exist. Use this deploy workflow after export when the input artifact is an ONNX model and the desired output is a TensorRT engine or TensorRT-backed predictions.

Supported actions: `gen_trt_engine`, `evaluate`, `inference`.

## Quick Start

### Generate TensorRT Engine

```bash
docker run --gpus all --rm --shm-size=16g \
  -v /path/to/specs:/specs \
  -v /path/to/export:/models \
  -v /path/to/results:/results \
  nvcr.io/nvidia/tao/tao-toolkit:6.26.3-deploy \
  clip gen_trt_engine -e /specs/clip_deploy_gen_trt_engine.yaml
```

### Evaluate TensorRT Engine

```bash
docker run --gpus all --rm --shm-size=16g \
  -v /path/to/specs:/specs \
  -v /path/to/eval:/data \
  -v /path/to/results:/results \
  nvcr.io/nvidia/tao/tao-toolkit:6.26.3-deploy \
  clip evaluate -e /specs/clip_deploy_evaluate.yaml
```

### TensorRT Inference

```bash
docker run --gpus all --rm --shm-size=16g \
  -v /path/to/specs:/specs \
  -v /path/to/inference:/data \
  -v /path/to/results:/results \
  nvcr.io/nvidia/tao/tao-toolkit:6.26.3-deploy \
  clip inference -e /specs/clip_deploy_inference.yaml
```

Deploy action metadata is in `tao-deploy-clip.skill_info.yaml`. Deploy spec templates live in this references folder:

- `spec_template_deploy.yaml`

## Deploy Workflow

1. Train and export with the `clip` skill.
2. Keep the exported ONNX artifact and any sidecar files together in the mounted model directory.
3. Build the TensorRT engine with this workflow. For `export.encoder_type: separate`, run `gen_trt_engine` once for the `_vision.onnx` file and once for the matching `_text.onnx` file when text inference or retrieval evaluation is required.
4. Run TensorRT `evaluate` or `inference` from the engine artifact produced by `gen_trt_engine`.

Direct TAO Launcher spelling is `tao deploy clip gen_trt_engine`, `tao deploy clip evaluate`, `tao deploy clip inference`.

## Required Inputs

| Action | Required artifact or data | Spec key |
|---|---|---|
| `gen_trt_engine` | Exported ONNX model or ONNX bundle | `gen_trt_engine.onnx_file` |
| `gen_trt_engine` | Output engine path or engine directory | `gen_trt_engine.trt_engine` |
| `evaluate` | TensorRT engine path or directory | `evaluate.trt_engine` |
| `evaluate` | Validation image folder | `dataset.val.datasets[0].image_dir` |
| `evaluate` | Validation caption folder | `dataset.val.datasets[0].caption_dir` |
| `inference` | TensorRT engine path or directory | `inference.trt_engine` |
| `inference` | Image datasets or text file | `inference.datasets / inference.text_file` |

For direct Docker runs, mount input folders at the same paths used in the spec. For chained jobs, map exported ONNX artifacts into `gen_trt_engine.onnx_file` and map the engine artifact into `evaluate.trt_engine` or `inference.trt_engine` where those actions are available.

TensorRT `evaluate` requires both image and text embeddings. Provide a combined engine, a directory containing paired separate engines, or an engine path ending in `_vision.engine` with a matching `_text.engine` beside it. A single `_vision.engine` is valid for image-only inference, but not for retrieval evaluation or text inference.

## Spec Overrides

Carry structural model and dataset settings forward from the train/export spec. The deploy defaults are templates, not a substitute for the model-specific values used to produce the ONNX file.

Recommended starting overrides:

```python
{
    'gen_trt_engine.tensorrt.data_type': 'fp16',
    'gen_trt_engine.tensorrt.max_batch_size': 16,
    'evaluate.batch_size': 16,
    'inference.batch_size': 16,
}
```

Model-specific notes:

- Keep CLIP sidecar artifacts next to the engine path because evaluate and inference load model configuration from the engine location.
- For image-only inference, populate `inference.datasets` and leave `inference.text_file` unset.
- For text inference or retrieval evaluation with separate export, build and keep paired `_vision.engine` and `_text.engine` files in the same directory.

## Job Chain Mapping

| Action | Spec field | Parent or output |
|---|---|---|
| `gen_trt_engine` | `gen_trt_engine.onnx_file` | export job ONNX |
| `gen_trt_engine` | `gen_trt_engine.trt_engine` | new engine output path |
| `evaluate` | `evaluate.trt_engine` | engine job output |
| `inference` | `inference.trt_engine` | engine job output |

## Outputs

| Action | Output |
|---|---|
| `gen_trt_engine` | TensorRT engine file or engine directory at `gen_trt_engine.trt_engine` |
| `evaluate` | Retrieval metrics under `results_dir` |
| `inference` | Image and/or text embeddings under `results_dir` |

## Known Pitfalls

**Engine profile mismatch:** Runtime batch size for evaluate or inference must fit within the TensorRT min/opt/max profile used during `gen_trt_engine`.

**Text engine not loaded:** A retrieval evaluation or text inference run was pointed at a vision-only engine. Build the matching `_text.engine` from the exported text ONNX and keep it beside the `_vision.engine`, or point `evaluate.trt_engine` / `inference.trt_engine` at a directory containing the pair.

**Template class or shape mismatch:** Copy class count, input resolution, backbone, and post-processing settings from train/export before running TAO Deploy.

**INT8 calibration missing:** INT8 builds need an extracted calibration image directory, a writable cache path, and enough images for `cal_batch_size * cal_batches`.

**Mounted paths do not exist:** TAO Deploy checks local paths inside the container. Make sure every path in the spec has a matching Docker mount or job artifact mapping.
