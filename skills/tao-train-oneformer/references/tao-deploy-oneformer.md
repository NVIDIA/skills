# OneFormer Deploy

OneFormer deploy covers the TAO Deploy actions for an exported universal segmentation model. Use the `oneformer` model skill for training, checkpoint evaluation, quantization, distillation, pruning, export, or non-TensorRT inference where those actions exist. Use this deploy workflow after export when the input artifact is an ONNX model and the desired output is a TensorRT engine or TensorRT-backed predictions.

Supported actions: `gen_trt_engine`, `evaluate`, `inference`.

## Quick Start

### Generate TensorRT Engine

```bash
docker run --gpus all --rm --shm-size=16g \
  -v /path/to/specs:/specs \
  -v /path/to/export:/models \
  -v /path/to/results:/results \
  nvcr.io/nvidia/tao/tao-toolkit:6.26.3-deploy \
  oneformer gen_trt_engine -e /specs/oneformer_deploy_gen_trt_engine.yaml
```

### Evaluate TensorRT Engine

```bash
docker run --gpus all --rm --shm-size=16g \
  -v /path/to/specs:/specs \
  -v /path/to/eval:/data \
  -v /path/to/results:/results \
  nvcr.io/nvidia/tao/tao-toolkit:6.26.3-deploy \
  oneformer evaluate -e /specs/oneformer_deploy_evaluate.yaml
```

### TensorRT Inference

```bash
docker run --gpus all --rm --shm-size=16g \
  -v /path/to/specs:/specs \
  -v /path/to/inference:/data \
  -v /path/to/results:/results \
  nvcr.io/nvidia/tao/tao-toolkit:6.26.3-deploy \
  oneformer inference -e /specs/oneformer_deploy_inference.yaml
```

Deploy action metadata is in `tao-deploy-oneformer.skill_info.yaml`. Deploy spec templates live in this references folder:

- `spec_template_deploy_gen_trt_engine.yaml`
- `spec_template_deploy_evaluate.yaml`
- `spec_template_deploy_inference.yaml`

## Deploy Workflow

1. Train and export with the `oneformer` skill.
2. Keep the exported ONNX artifact and any sidecar files together in the mounted model directory.
3. Build the TensorRT engine with this workflow.
4. Run TensorRT `evaluate` or `inference` from the engine artifact produced by `gen_trt_engine`.

Direct TAO Launcher spelling is `tao deploy oneformer gen_trt_engine`, `tao deploy oneformer evaluate`, `tao deploy oneformer inference`.

## Required Inputs

| Action | Required artifact or data | Spec key |
|---|---|---|
| `gen_trt_engine` | Exported ONNX model | `gen_trt_engine.onnx_file` |
| `gen_trt_engine` | Output engine path | `gen_trt_engine.trt_engine` |
| `evaluate` | TensorRT engine | `evaluate.trt_engine` |
| `evaluate` | Label map | `dataset.label_map` |
| `evaluate` | Validation annotations | `dataset.val.annotations` |
| `evaluate` | Validation images | `dataset.val.images` |
| `evaluate` | Validation panoptic masks | `dataset.val.panoptic` |
| `evaluate` | Test image directory used by the shared deploy template | `dataset.test.images` |
| `inference` | TensorRT engine | `inference.trt_engine` |
| `inference` | Test image directory | `dataset.test.images` |
| `inference` | Label map | `dataset.label_map` |

For direct Docker runs, mount input folders at the same paths used in the spec. Image and panoptic tarballs must be extracted before deploy evaluate or inference, and the spec must point to the actual inner image/panoptic folder. For chained jobs, map exported ONNX artifacts into `gen_trt_engine.onnx_file` and map the engine artifact into `evaluate.trt_engine` or `inference.trt_engine` where those actions are available.

## Spec Overrides

Carry structural model and dataset settings forward from the train/export spec. The deploy defaults are templates, not a substitute for the model-specific values used to produce the ONNX file.

Recommended starting overrides:

```python
{
    'model.sem_seg_head.num_classes': 133,
    'dataset.contiguous_id': True,
    'gen_trt_engine.tensorrt.data_type': 'fp16',
    'dataset.val.batch_size': 1,
    'dataset.test.batch_size': 1,
}
```

Model-specific notes:

- Carry `model.sem_seg_head.num_classes` and `dataset.contiguous_id` from train/export. The packaged COCO panoptic path uses `dataset.contiguous_id: True` with `model.sem_seg_head.num_classes: 133`; use a smaller value only if the train/export dataset was reduced to that exact class set.
- Evaluate and inference share the deploy infer template but use different top-level engine fields.
- Set `gen_trt_engine.trt_engine` explicitly to a non-existing file path in
  the mounted results tree. Do not pre-create it as a declared file output.

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
| `gen_trt_engine` | TensorRT engine at `gen_trt_engine.trt_engine` |
| `evaluate` | Universal segmentation metrics under `results_dir` |
| `inference` | Segmentation predictions and visualizations under `results_dir` |

## Known Pitfalls

**Engine profile mismatch:** Runtime batch size for evaluate or inference must fit within the TensorRT min/opt/max profile used during `gen_trt_engine`.

**Template class or shape mismatch:** Copy class count, input resolution, backbone, and post-processing settings from train/export before running TAO Deploy.

**INT8 calibration missing:** INT8 builds need an extracted calibration image directory, a writable cache path, and enough images for `cal_batch_size * cal_batches`.

**Mounted paths do not exist:** TAO Deploy checks local paths inside the container. Make sure every path in the spec has a matching Docker mount or job artifact mapping.

**Older deploy image OneFormer engine generation failure:** Some 7.0.0 RC
deploy images parse the exported OneFormer ONNX with two inputs, `images` and
`task_tokens`, then assume every input is a 4D image tensor. This causes
`gen_trt_engine` to fail with `IndexError: Out of bounds` while reading the 2D
`task_tokens` input. Use a deploy image that creates profiles for both
`images` and `task_tokens`, then mark deploy validation per image by requiring
a produced engine before running downstream TensorRT actions.
