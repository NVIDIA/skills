---
name: tao-finetune-video-clip
description: >-
  InternVideo2-CLIP L14 (TAO video_clip) for video-text retrieval, zero-shot classification,
  embedding extraction, LoRA fine-tuning, ONNX export, and TensorRT deployment. Use when
  the user asks to "fine-tune IV2CLIP", "run video_clip train/evaluate/inference/export",
  "build a Video-CLIP TensorRT engine", "InternVideo2-CLIP on KPI chunks", or "TAO
  video_clip on vadr1_chunks JSON".
license: Apache-2.0
compatibility: >-
  Requires docker + nvidia-container-toolkit and the pinned TAO video_clip PyTorch and
  Deploy containers (see references/skill_info.yaml and
  references/tao-deploy-video-clip.skill_info.yaml), or a local tao-pytorch checkout +
  tao-cli venv for PyTorch virtualenv runs. MobileCLIP + InternVideo2 weights must be on
  disk for offline eval (HF LFS may be blocked in CI). Metadata JSON uses vadr1_chunks
  with absolute video_path entries.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- video
- vision-language
- vlm
- multimodal
- retrieval
- embedding
- internvideo2
- iv2clip
- fine-tuning
- deployment
---

# InternVideo2-CLIP (TAO video_clip)

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

TAO task **`video_clip`** wraps OpenGVLab **InternVideo2-CLIP L14**. The PyTorch image provides `train`, `evaluate`, `inference`, `export`, and `default_specs`. TAO Deploy provides `gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference`.

Container images and per-action commands are in `references/skill_info.yaml` and `references/tao-deploy-video-clip.skill_info.yaml`. Starting specs are in `references/spec_template_*.yaml`.

> **Release note:** The pinned PyTorch image is the TAO 7.2 release-candidate build validated for Video-CLIP. It includes PyAV 17.1.0 as the primary decoder and ONNXScript 0.7.1 for export, with decord absent. The TAO Deploy image is pinned independently because `gen_trt_engine` and TensorRT-backed actions do not run in the PyTorch image.
>
> **Known-broken images:** interim builds cut before tao-pytorch commit `0cc31de4` ship a `video_clip` package with no `model.backbones` submodule, so `train`/`evaluate`/`inference` die at import while `video_clip --help` still exits 0. Images without PyAV also fail at data loading. Run both import checks in the preflight below before pulling data or launching a run.

## Train Action Policy

AutoML is not packaged for this model skill. Always use direct `video_clip` actions even when a higher-level request mentions AutoML. Non-train actions stay in this skill.

## Quick Start (local Docker)

Use the pinned TAO container declared in `references/skill_info.yaml`. Pull with `NGC_KEY` when the image is not cached locally.

```bash
VIDEO_CLIP_IMAGE_DEFAULT="nvcr.io/nvidia/tao/tao-toolkit:7.2.0-pyt"  # versions-key: images.tao_toolkit.pyt
VIDEO_CLIP_IMAGE="${VIDEO_CLIP_IMAGE:-$VIDEO_CLIP_IMAGE_DEFAULT}"
docker pull "$VIDEO_CLIP_IMAGE"
```

Expected workspace layout (host paths bind-mounted into the container):

```text
workspace/
├── data/
│   ├── train.json              # vadr1_chunks metadata; video_path = /data/videos/<name>.mp4
│   ├── val.json
│   ├── prompts.txt             # one text prompt per line (inference)
│   └── videos/                 # mp4 clips referenced by train.json / val.json
├── model/
│   ├── mobileclip_blt.pt       # MobileCLIP weights for model.text_encoder
│   └── hf/                     # offline InternVideo2_distillation_models snapshot (HF_HUB_OFFLINE=1)
├── specs/
│   ├── train.yaml
│   ├── evaluate.yaml
│   ├── inference.yaml
│   └── export.yaml
├── deploy_specs/
│   ├── gen_trt_engine.yaml
│   ├── evaluate.yaml
│   └── inference.yaml
└── results/
```

Docker options for all actions (skill-eval CI uses the same `$WORKSPACE_DIR` bind-mount pattern):

```bash
VIDEO_CLIP_IMAGE_DEFAULT="nvcr.io/nvidia/tao/tao-toolkit:7.2.0-pyt"  # versions-key: images.tao_toolkit.pyt
VIDEO_CLIP_IMAGE="${VIDEO_CLIP_IMAGE:-$VIDEO_CLIP_IMAGE_DEFAULT}"
RUN_ROOT="${RUN_ROOT:-$PWD}"
DOCKER_COMMON=(
  --rm --gpus all --shm-size=8g --network=host
  --shm-size=64g
  --ulimit memlock=-1
  --ulimit stack=67108864
  -e WANDB_DISABLED=true
  -e WANDB_MODE=disabled
  -e HF_HUB_OFFLINE=1
  -e HUGGINGFACE_HUB_CACHE=/model/hf
  -e TRANSFORMERS_OFFLINE=1
  -e TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  -v "$RUN_ROOT/data:/data:ro"
  -v "$RUN_ROOT/model:/model:ro"
  -v "$RUN_ROOT/specs:/specs:ro"
  -v "$RUN_ROOT/results:/results"
)
```

Preflight (host):

```bash
[ -f "$RUN_ROOT/model/mobileclip_blt.pt" ] || echo "MISSING: MobileCLIP weights"
[ -f "$RUN_ROOT/data/train.json" ] || echo "MISSING: train metadata"
[ -d "$RUN_ROOT/model/hf" ] || echo "MISSING: offline HF snapshot under model/hf"
docker run --rm "$VIDEO_CLIP_IMAGE" video_clip --help >/dev/null || echo "MISSING: video_clip in container"
docker run --rm "$VIDEO_CLIP_IMAGE" \
  python -c "import nvidia_tao_pytorch.multimodal.video_clip.model.adapters.internvideo2clip" \
  >/dev/null 2>&1 || echo "BROKEN IMAGE: video_clip package is incomplete (missing model.backbones) — stop, see Release note"
docker run --rm "$VIDEO_CLIP_IMAGE" python -c "import av; print(av.__version__)" \
  >/dev/null 2>&1 || echo "BROKEN IMAGE: PyAV is missing — use the pinned FC image; do not install decord"
nvidia-smi >/dev/null 2>&1 || echo "note: no GPU visible"
```

Train:

```bash
docker run "${DOCKER_COMMON[@]}" "$VIDEO_CLIP_IMAGE" \
  video_clip train -e /specs/train.yaml results_dir=/results
```

Evaluate:

```bash
docker run "${DOCKER_COMMON[@]}" "$VIDEO_CLIP_IMAGE" \
  video_clip evaluate -e /specs/evaluate.yaml results_dir=/results
```

Inference:

```bash
docker run "${DOCKER_COMMON[@]}" "$VIDEO_CLIP_IMAGE" \
  video_clip inference -e /specs/inference.yaml results_dir=/results
```

Export:

```bash
docker run "${DOCKER_COMMON[@]}" "$VIDEO_CLIP_IMAGE" \
  video_clip export -e /specs/export.yaml results_dir=/results
```

## TensorRT Deploy

Use the independently pinned TAO Deploy image after PyTorch export. Read `references/tao-deploy-video-clip.md` before running the deploy actions; its templates cover the engine build, retrieval evaluation, and embedding inference contracts.

```bash
VIDEO_CLIP_DEPLOY_IMAGE_DEFAULT="nvcr.io/nvidia/tao/tao-toolkit:7.2.0-deploy"  # versions-key: images.tao_toolkit.deploy
VIDEO_CLIP_DEPLOY_IMAGE="${VIDEO_CLIP_DEPLOY_IMAGE:-$VIDEO_CLIP_DEPLOY_IMAGE_DEFAULT}"

# Verify the independently pinned image before staging artifacts or using a GPU.
docker run --rm "$VIDEO_CLIP_DEPLOY_IMAGE" video_clip gen_trt_engine --help >/dev/null || \
  { echo "BROKEN IMAGE: Video-CLIP deploy entrypoint is unavailable" >&2; exit 1; }

docker run --gpus all --rm --shm-size=16g \
  -v "$RUN_ROOT/deploy_specs:/specs:ro" \
  -v "$RUN_ROOT/results/export:/models:ro" \
  -v "$RUN_ROOT/data:/data:ro" \
  -v "$RUN_ROOT/results/deploy:/results" \
  "$VIDEO_CLIP_DEPLOY_IMAGE" \
  video_clip gen_trt_engine -e /specs/gen_trt_engine.yaml
```

Keep the exported ONNX file, its matching `*_config.yaml`, and its matching `*_tokenizer/` directory together. `gen_trt_engine` copies the sidecars beside the engine so TensorRT `evaluate` and `inference` can reconstruct preprocessing and tokenization.

## Quick Start (virtualenv — local dev hosts)

On hosts with a `tao-pytorch` checkout and `tao-cli` venv (for example rtdetr-pytorch), run through **`tao-run-on-virtualenv`** instead of Docker:

```bash
export VENV="${VENV:?set to your tao-cli virtualenv (must contain bin/video_clip)}"
export TAO_PYTORCH_ROOT="${TAO_PYTORCH_ROOT:?set to your tao-pytorch checkout with multimodal/video_clip}"
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$TAO_PYTORCH_ROOT:$TAO_PYTORCH_ROOT/tao-core:$PYTHONPATH"
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export WANDB_DISABLED=true
export WANDB_MODE=disabled
```

Copy a template spec from `references/spec_template_*.yaml`, fill checkpoint paths and metadata, then run:

```bash
video_clip train -e /path/to/train.yaml
video_clip evaluate -e /path/to/evaluate.yaml
video_clip inference -e /path/to/inference.yaml
video_clip export -e /path/to/export.yaml
```

## Credentials

- **NGC_KEY** — pull the pinned TAO container from `nvcr.io` when it is not cached locally.
- **HF_TOKEN** (online Hugging Face download only): Hugging Face read token used
  when `model.vision_encoder` and `model.clip_head` are `null` and the resolver
  downloads the InternVideo2 snapshot named by `model.internvideo2clip_hf_id`.
  For offline CI/eval, stage the complete S3/local snapshot and point both fields
  at the local files; no Hugging Face token is then required.

Treat tokens as secrets. Export them into the environment or pass them through an
`--env-file` of bare `KEY=value` lines, rather than inlining values into generated
spec YAML, command lines, or anything written under `results_dir`.

## Data format (vadr1_chunks)

Metadata is a **top-level JSON list** of video records. Each record has `video_path`, `split`, and nested `chunks[]` with caption fields (`queries`, `action_queries`, `anomaly_queries`, `dense_caption`, `scene_caption`).

Point `dataset.*.video_text.metadata` at the user's JSON files (for example `/data/train.json` and `/data/val.json` in the templates). Each `video_path` must be an **absolute** path resolvable inside the runtime — for Docker, remap host clips to container paths like `/data/videos/<name>.mp4` and set `data_root: null` unless using `path_prefix_mapping`.

## Smoke overrides

For a short functional check (for example 2 epochs, 1 GPU, small batch):

```yaml
train.num_epochs: 2
train.num_gpus: 1
train.gpu_ids: [0]
train.optim.warmup_steps: 10
dataset.train.batch_size: 2
dataset.val.batch_size: 2
dataset.train.num_workers: 4
dataset.val.num_workers: 4
dataset.train.video_text.caption_fields: [queries, action_queries, anomaly_queries]
dataset.train.video_text.caption_mode: first
dataset.metrics.mode: classification
```

Use `dataset.metrics.mode: retrieval` only when `dataset.val.video_text.relevance_file` is provided.

## Inference

- `inference.mode: embeddings` writes `video_embeddings.h5` and `text_embeddings.h5` under `results_dir`.
- Provide `inference.query.text_file` (one prompt per line) and/or `inference.query.input_texts`.
- Gallery videos come from `dataset.inference.video_text.metadata`.

## Export

`export.encoder_type: combined` produces the image-and-text ONNX consumed by the Video-CLIP deploy workflow. Keep `export.batch_size: -1` for symbolic/dynamic batch dimensions; a positive value produces a fixed-batch ONNX. Export also writes matching `*_config.yaml` and `*_tokenizer/` sidecars; preserve all three artifacts. Default opset is **23** on the vendor branch. Export requires a trained `.pth` at `export.checkpoint`.

## LoRA

For vision-LoRA runs, start from `tao-pytorch` `experiment_spec_lora.yaml` or add a top-level `peft:` block (see shipped spec comments). Merge LoRA before export when checkpoints contain `lora_*` keys.

## Common pitfalls

- **`PATH` must prefer `$VENV/bin`** on virtualenv hosts so child processes resolve the venv Python.
- **`evaluate` uses `dataset.val`**, not a separate test split — Lightning stage `"test"` still loads val metadata.
- **Eval precision**: match `train.precision` (typically `bf16`) or flash-attn paths may fail under fp32 eval.
- **Empty `action_queries` on normal chunks** become literal `"Normal"` positives during training; exclude `Normal`/`Abnormal` in `dataset.metrics.exclude_categories` for classification eval.
- **No hard-negative / explicit-neg training** on the vendor branch unless the spec and branch explicitly enable it.
- **`video_clip --help` is not a health check.** It exits 0 on an image whose `video_clip` package is missing `model.backbones`; only the import smoke check in the preflight catches it.
- **PyAV is the primary Video-CLIP decoder in TAO 7.2.** The loader can fall back to the image's FFmpeg CLI and OpenCV support. A missing `av` import is an image defect: use the pinned FC image. Do not add or force-install decord, because it is not part of the supported TAO 7.2 decode contract.
- **TensorRT actions use TAO Deploy.** `gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference` must use the independently pinned deploy image and deploy templates, not the PyTorch image/specs.
- **Deploy sidecars are required.** TensorRT evaluation and text inference need the exported `*_config.yaml` and `*_tokenizer/` beside the engine. Keep them with the ONNX input so engine generation can copy them automatically.
- **PyTorch ≥ 2.6 defaults to `torch.load(weights_only=True)`** and rejects the TAO checkpoint’s numpy dtype objects with `_pickle.UnpicklingError`. `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is set in `DOCKER_COMMON` above; keep it for `evaluate`, `inference`, and `export`.
- **`model/hf/` is a snapshot, not an HF hub cache.** Offline packs stage InternVideo2 weights at repo-relative paths (`stage1/L14/L14_dist_1B_stage2/pytorch_model.bin`, `clip/L14/pytorch_model.bin`), while `HUGGINGFACE_HUB_CACHE` expects a `models--<org>--<repo>/snapshots/<sha>/` tree. Leaving `model.vision_encoder` / `model.clip_head` at `null` sends asset resolution to `hf_hub_download` and fails under `HF_HUB_OFFLINE=1` — point both at the files directly.
- **CI / skill-eval**: stage weights + remapped JSON under `$WORKSPACE_DIR` from S3; do not rely on HuggingFace LFS downloads at eval time.

## References

- Shipped defaults: `tao-pytorch` `nvidia_tao_pytorch/multimodal/video_clip/experiment_specs/` (when developing from source)
- TAO Deploy workflow: [`references/tao-deploy-video-clip.md`](references/tao-deploy-video-clip.md)
