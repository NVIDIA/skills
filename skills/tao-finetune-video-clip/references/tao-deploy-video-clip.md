# Video-CLIP Deploy

Video-CLIP deploy converts the combined image-and-text ONNX export to TensorRT, evaluates multi-relevance text-to-video retrieval, and extracts text and/or gallery-video embeddings. Use the PyTorch `video_clip` actions for training, checkpoint evaluation, checkpoint inference, and export. Use this TAO Deploy workflow only after export.

Supported actions: `gen_trt_engine`, `evaluate`, `inference`.

## Quick Start

```bash
VIDEO_CLIP_DEPLOY_IMAGE=nvcr.io/nvidia/tao/tao-toolkit:7.2.0-deploy  # versions-key: images.tao_toolkit.deploy
```

Copy the matching template for each action:

- `spec_template_gen_trt_engine.yaml`
- `spec_template_deploy_evaluate.yaml`
- `spec_template_deploy_inference.yaml`

Mount the ONNX, engine, sidecar, data, and results paths exactly as written in the selected spec. The in-container commands are:

```bash
video_clip gen_trt_engine -e /specs/gen_trt_engine.yaml
video_clip evaluate -e /specs/evaluate.yaml
video_clip inference -e /specs/inference.yaml
```

Direct TAO Launcher spelling is `tao deploy video_clip <action> -e /path/to/spec.yaml`.

## Workflow

1. Export with PyTorch `video_clip export` and `export.encoder_type: combined`.
2. Keep the ONNX file with its exported `*_config.yaml` and `*_tokenizer/` sidecars.
3. Run `gen_trt_engine`. It copies the sidecars from the ONNX directory to the engine directory when those directories differ.
4. Run TensorRT `evaluate` with explicit-relevance queries and the matching vadr1_chunks metadata, or run TensorRT `inference` for text and/or gallery-video embeddings.

## Required Inputs

| Action | Required artifact or data | Spec key |
|---|---|---|
| `gen_trt_engine` | Combined image-and-text ONNX | `gen_trt_engine.onnx_file` |
| `gen_trt_engine` | Output engine path | `gen_trt_engine.trt_engine` |
| `evaluate` | TensorRT engine | `evaluate.trt_engine` |
| `evaluate` | JSON containing `gallery` and `queries` with `relevant_clip_ids` | `dataset.val.gt_queries` |
| `evaluate` | vadr1_chunks metadata resolving gallery chunk IDs | `dataset.val.metadata` |
| `inference` | TensorRT engine | `inference.trt_engine` |
| `inference` | Prompt file and/or gallery query JSON plus metadata | `inference.text_file` and/or `dataset.val.{gt_queries,metadata}` |

Use `dataset.val.video_root` for relative video paths. Use `dataset.val.path_prefix_mapping` when metadata contains absolute paths that must be remapped inside the container.

## Outputs

| Action | Output |
|---|---|
| `gen_trt_engine` | TensorRT engine at `gen_trt_engine.trt_engine`, plus copied export sidecars |
| `evaluate` | Retrieval metrics in `results.json` under `results_dir` |
| `inference` | `text_embeddings.h5` and/or `video_embeddings.h5` under `results_dir` |

## TensorRT Settings

The starter engine template uses FP32 as the conservative default. The deploy implementation also supports FP16 and preserves sensitive vision normalization reductions in FP32 for weakly typed builds. Runtime batches must fit the min/opt/max batch profile created by `gen_trt_engine`.

## Job Chain Mapping

| Action | Spec field | Parent or output |
|---|---|---|
| `gen_trt_engine` | `gen_trt_engine.onnx_file` | PyTorch export ONNX |
| `gen_trt_engine` | `gen_trt_engine.trt_engine` | New engine output path |
| `evaluate` | `evaluate.trt_engine` | Engine job output |
| `inference` | `inference.trt_engine` | Engine job output |

## Known Pitfalls

- **Sidecars missing:** `evaluate` and text inference cannot load preprocessing/tokenizer state without the matching export sidecars. Keep them beside the ONNX before engine generation and beside the engine afterward.
- **Wrong export type:** TAO Deploy expects the combined Video-CLIP ONNX with both image and text inputs.
- **Nothing to infer:** Set `inference.text_file` and/or both `dataset.val.gt_queries` and `dataset.val.metadata`.
- **Gallery mismatch:** Every gallery `chunk_id` in `gt_queries` must resolve through the supplied vadr1_chunks metadata.
- **Engine profile mismatch:** Evaluation and inference batch sizes must fit the engine optimization profile.
