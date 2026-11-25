---
name: tao-finetune-cosmos-embed
description: >-
  Cosmos-Embed1 video-text embedding for text-to-video retrieval, video-to-video search, semantic deduplication, and
  LoRA fine-tuning. Use when the user asks to "fine-tune Cosmos-Embed1", "LoRA fine-tune Cosmos-Embed1 on
  PhysicalAI-Event-Videos", "train Cosmos-Embed1 on vad_r1_chunks manifests", "score the text-to-video retrieval
  benchmark during training", "run cosmos-embed inference", "export Cosmos-Embed1 to ONNX", "embed videos", or
  "search videos with text".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit, the published Cosmos-Embed TAO container (pinned in this skill), and a HuggingFace token when downloading pretrained `nvidia/Cosmos-Embed1-*` weights.
metadata:
  author: NVIDIA Corporation
  version: "0.2.0"
allowed-tools: Read Bash
tags:
- video
- vision-language
- vlm
- multimodal
- retrieval
- embedding
- cosmos
- fine-tuning
- lora
---

# Cosmos-Embed

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

Cosmos-Embed1 is a joint video-text embedder for text-to-video retrieval, video-to-video search, zero-shot/kNN classification, and semantic deduplication. The packaged CLI is `cosmos-embed1` and supports `train`, `evaluate`, `inference`, and `export`.

Container image and per-action commands are in `references/skill_info.yaml`. Compact starting specs are in `references/spec_template_*.yaml`; the LoRA fine-tuning recipe on chunk manifests is in `references/lora-finetune-physicalai.md` with `references/spec_template_train_lora.yaml`.

> **Release note:** the pinned `7.3.0-cosmos-embed` image is the first that trains ViT LoRA adapters, accepts `caption_mode`, `label_mode`, and `train.grad_accum_iter`, scores the in-training `retrieval_benchmark`, and validates large ONNX exports by file path. On `7.1.0-cosmos-embed` the ViT adapters silently stay at zero and the vision ONNX export exits 1 after writing valid files.

## Train Action Policy

AutoML is not packaged for this model skill (no Cosmos-Embed schemas under `schemas/`). Use the direct `train`, `evaluate`, `inference`, and `export` actions even when a higher-level request includes `automl_policy: on`; the per-run override does not change model metadata, and non-train and deploy flows stay in this skill.

## Quick Start

Use the published container pinned below (also declared in `references/skill_info.yaml`); do not build from the Cosmos-Embed1 source tree for normal skill use.

```bash
COSMOS_EMBED_IMAGE_DEFAULT=nvcr.io/nvidia/tao/tao-toolkit:7.3.0-cosmos-embed  # versions-key: images.tao_toolkit.cosmos_embed
COSMOS_EMBED_IMAGE="${COSMOS_EMBED_IMAGE:-$COSMOS_EMBED_IMAGE_DEFAULT}"
docker pull "$COSMOS_EMBED_IMAGE"
```

Expected local workspace layout:

```text
workspace/
├── data/
│   ├── msrvtt_test_1k.json
│   └── video/
│       ├── video7020.mp4
│       └── ...
├── model/
│   └── Cosmos-Embed1-224p/        # optional if using HF repo id
├── specs/
│   ├── train.yaml                 # or train_lora.yaml from references/
│   ├── evaluate.yaml
│   ├── inference.yaml
│   ├── export_onnx.yaml
│   └── export_hf.yaml
└── results/
```

Use these Docker options for all actions unless the local Docker/platform skill gives a stricter environment-specific command:

```bash
set -a; source /path/to/.env; set +a   # omit if already exported
COSMOS_EMBED_IMAGE_DEFAULT=nvcr.io/nvidia/tao/tao-toolkit:7.3.0-cosmos-embed  # versions-key: images.tao_toolkit.cosmos_embed
COSMOS_EMBED_IMAGE="${COSMOS_EMBED_IMAGE:-$COSMOS_EMBED_IMAGE_DEFAULT}"
RUN_ROOT="${RUN_ROOT:-$PWD}"
COSMOS_EMBED_PRETRAINED_CACHE="${COSMOS_EMBED_PRETRAINED_CACHE:-$RUN_ROOT/pretrained_checkpoints}"
mkdir -p "$COSMOS_EMBED_PRETRAINED_CACHE"
DOCKER_COMMON=(
  --rm --gpus all --shm-size=8g --network=host
  --shm-size=64g
  --ulimit memlock=-1
  --ulimit stack=67108864
  -w /results
  -e HF_TOKEN
  -e WANDB_DISABLED=true
  -e WANDB_MODE=disabled
  -e HUGGINGFACE_HUB_CACHE=/hf_cache
  -v "$RUN_ROOT/data:/data:ro"
  -v "$RUN_ROOT/model:/model"
  -v "$RUN_ROOT/specs:/specs:ro"
  -v "$RUN_ROOT/results:/results"
  -v "$RUN_ROOT/hf_cache:/hf_cache"
  --mount "type=bind,source=$COSMOS_EMBED_PRETRAINED_CACHE,target=/usr/local/lib/python3.12/dist-packages/pretrained_checkpoints"
)
```

The image downloads the BERT Q-Former component under `/usr/local/lib/python3.12/dist-packages/pretrained_checkpoints` regardless of the Hugging Face cache variables, so the bind mount above must exist and be writable by the container UID chosen by the platform skill. Verify the cache location before reusing this path with another image version.

Keep `-w /results`: with a cold Triton cache, a working directory containing `specs/` makes GCC read `./specs` as a compiler spec file (`cannot read spec file './specs': Is a directory`).

Run `cosmos-embed1` directly; do not `pip install` anything into the container first. The pinned image ships a matched `wandb`/`onnx`/`protobuf` set, and its `PIP_CONSTRAINT` resolves `pip install "protobuf<7"` to `protobuf==4.24.4`, which breaks ONNX export. Keep `WANDB_DISABLED=true` for smoke or offline runs. The Q-Former path may download the public `google-bert/bert-base-uncased` even with the model checkpoint disabled, so provide `HF_TOKEN` through the environment or an `--env-file`, or mount a persistent HuggingFace cache. Never write the token into specs, logs, or reports.

Train:

```bash
docker run "${DOCKER_COMMON[@]}" "$COSMOS_EMBED_IMAGE" \
  cosmos-embed1 train -e /specs/train.yaml results_dir=/results
```

Evaluate:

```bash
docker run "${DOCKER_COMMON[@]}" "$COSMOS_EMBED_IMAGE" \
  cosmos-embed1 evaluate -e /specs/evaluate.yaml results_dir=/results
```

Inference:

```bash
docker run "${DOCKER_COMMON[@]}" "$COSMOS_EMBED_IMAGE" \
  cosmos-embed1 inference -e /specs/inference.yaml \
  'inference.query.input_texts=["a man is singing on stage"]' \
  inference.k=5 \
  results_dir=/results
```

Export ONNX (one graph per tower, dynamic batch axis; run once with `export.mode=video` and once with `export.mode=text`):

```bash
docker run "${DOCKER_COMMON[@]}" "$COSMOS_EMBED_IMAGE" \
  cosmos-embed1 export -e /specs/export_onnx.yaml \
  export.checkpoint=/results/train/checkpoints/iter_000000001.pt \
  export.mode=video export.batch_size=-1 export.on_cpu=true \
  export.onnx_file=/results/export/cosmos_embed1_vision.onnx \
  results_dir=/results
```

Export HuggingFace format:

```bash
docker run "${DOCKER_COMMON[@]}" "$COSMOS_EMBED_IMAGE" \
  cosmos-embed1 export -e /specs/export_hf.yaml \
  export.checkpoint=/results/train/checkpoints/iter_000000001.pt \
  export.hf_output_dir=/results/export_hf/cosmos_embed1_hf \
  results_dir=/results
```

For a LoRA checkpoint, every downstream spec (`evaluate`, `inference`, `export`) must carry the training spec's `model.lora` block and list exactly the modules that were trained in `target_modules`; export merges the adapters into the weights. See `references/lora-finetune-physicalai.md`.

## Smoke Overrides

For a quick functional check of any action, keep the same specs and shrink iterations, batch sizes, and workers with the command-line overrides listed in `references/smoke-overrides.md` (including the scheduler and no-checkpoint caveats).

## Data Format

The MSR-VTT path expects a local video glob and a JSON metadata file:

```yaml
dataset:
  train_dataset:
    dataset_type: msrvtt
    mp4_urls: /data/video/*.mp4
    metadata: /data/msrvtt_test_1k.json
```

List-format metadata rows must include at least `video` and `caption`:

```json
{"video_id": "video7020", "video": "video7020.mp4", "caption": "a woman creating a fondant baby and flower"}
```

The dataset loader derives the video id from the local `.mp4` filename and filters to videos present in the metadata. If a run finds zero videos, check that `mp4_urls` points to a container-local glob and that metadata `video` names match the filenames.

Chunk manifests (`dataset_type: vad_r1_chunks`, the PhysicalAI-Event-Videos `labels.json` layout) hold one record per labeled chunk of a parent video, with caption fields that may be lists of strings. `caption_field` names the fields that feed the caption pool, `caption_mode` turns the pool into samples, `label_mode` decides what counts as a negative, and `path_prefix_mapping` rewrites the manifest's video prefix to the mounted folder. `dataset_type: retrieval_benchmark` (validation and evaluation only) takes a benchmark release root holding `gt/`, `additional_gt/`, `clips/`. Details and staging layouts: `references/lora-finetune-physicalai.md`, `references/data-staging.md`.

## Model Weights

- Local HF directory: mount it under `/model` and set `model.pretrained_model_path=/model/Cosmos-Embed1-224p`.
- HuggingFace repo: set `model.pretrained_model_path=nvidia/Cosmos-Embed1-224p` (or `-336p`, `-448p`) and pass `HF_TOKEN` if access is gated.
- Fine-tuned checkpoint: set downstream actions to the resolver-selected `/results/train/checkpoints/iter_#########.pt` file.

Training writes full checkpoints under `results/train/checkpoints/iter_#########.pt`, updates `results/train/checkpoints/latest_checkpoint.txt`, and creates a `cosmos_embed1_model_latest.pth` symlink. For `evaluate.checkpoint`, `inference.checkpoint`, `export.checkpoint`, and `train.resume_training_checkpoint_path`, resolve and pass the exact `iter_#########.pt` file for the intended iteration. The action spec templates intentionally leave these checkpoint fields null so the model-skill runner or the user must provide the resolver-selected checkpoint. Use the latest symlink only when the user explicitly asks for latest. When the `retrieval_benchmark` callback is enabled, `results/train/selection.json` names the best-scoring checkpoint; prefer it over the latest one.

For single-GPU resume/retrain from a consolidated checkpoint, set `model.fsdp_shard_size: 1`. The container default is 8, which sends resumed training through an FSDP apply path that Cosmos-Embed1 does not implement for this model class. Do not resume a LoRA run through `train.resume_training_checkpoint_path`; export the selected checkpoint to HF format, point `model.pretrained_model_path` at it, and start a fresh LoRA stage with its own schedule.

Variants:

| Variant | Resolution | Frames | Embedding dim |
|---|---:|---:|---:|
| `Cosmos-Embed1-224p` | 224 x 224 | 8 | 256 |
| `Cosmos-Embed1-336p` | 336 x 336 | 8 | 768 |
| `Cosmos-Embed1-448p` | 448 x 448 | 8 | 768 |

Keep `model.network.embed_dim`, `model.input_hw`, and `model.network.spatial_resolution` aligned with the selected variant.

## Important Parameters

| Parameter | Notes |
|---|---|
| `train.num_gpus` | `1` for single GPU, `>1` auto-launches `torchrun`, `-1` auto-detects visible GPUs. |
| `train.max_iter` | Main training length in optimizer updates. Use `1` only for smoke testing. |
| `train.grad_accum_iter` | Micro-steps per optimizer update. `max_iter`, `validation_iter`, and `checkpoint_iter` count updates, not micro-steps. |
| `train.validation_iter` / `checkpoint_iter` | Keep equal (about one epoch) so every validated iteration has a checkpoint on disk. |
| `train.optim.optim` | `fused_adamw` is faster when available; `adamw` is safer for smoke and portability. |
| `train.optim.lr` | `1e-5` for full fine-tuning; `1e-4` for LoRA adapters (`1e-5` under-fits them). |
| `train.callbacks.retrieval_benchmark` | `{query_domain, selection_metric}`. Scores text-to-video retrieval every `validation_iter`, writes `train/retrieval_benchmark.jsonl` and `train/selection.json`. Requires `dataset.val_dataset.dataset_type: retrieval_benchmark`; raises `TypeError` otherwise. |
| `dataset.*.caption_field` | String or list of field names (`vad_r1_chunks`); list-valued field values contribute every string. |
| `dataset.*.caption_mode` | `all` (default; one sample per chunk-caption pair), `first`, or `random`. |
| `dataset.*.label_mode` | `category` (default; anomaly type) or `caption` (identical caption strings share a label). Use `caption` with free-form captions. |
| `model.lora.enabled` | Enables LoRA. Set `model.network.visual_encoder.transformer_engine=false` when LoRA is on. |
| `model.lora.lora_rank` / `lora_alpha` / `lora_dropout` | Start with `8` / `16` / `0.1`; sweep rank `4`, `8`, `16` and dropout `0.0`, `0.05`, `0.1` for small datasets. |
| `model.lora.bias` | Bias policy: `none`, `all`, or `lora_only`. Keep `none` unless intentionally training biases. |
| `model.lora.use_rslora` / `use_dora` | Optional LoRA variants. Enable one at a time and record the setting with the checkpoint. |
| `model.lora.target_modules` | Module names to adapt. The recipe in `spec_template_train_lora.yaml` adapts attention of the last 20 ViT blocks plus Q-Former and projection heads. Record the list with the checkpoint; downstream specs must repeat it exactly. |
| `model.lora.modules_to_save` | Optional modules to keep fully trainable alongside LoRA. Leave empty unless preserving a task-specific head. |
| `model.network.visual_encoder.checkpoint_activations` | Activation checkpointing for the ViT; enable when 448p training runs out of memory. |
| `evaluate.load_dataset_pkl` / `save_dataset_pkl` | Cache evaluation embeddings. |
| `inference.load_dataset_pkl` / `save_dataset_pkl` | Cache the search database for repeated retrieval. |
| `export.mode` | `video`, `text`, `combined`, or `huggingface`. Use `video` + `text` for TensorRT deployment. |
| `export.batch_size` | `-1` exports a dynamic batch axis; a positive value fixes it. |
| `export.on_cpu` | Recommended for export to avoid device mismatch issues. |

## Outputs

```text
results/
├── train/
│   ├── checkpoints/iter_#########.pt
│   ├── cosmos_embed1_model_latest.pth
│   ├── retrieval_benchmark.jsonl      # with the retrieval_benchmark callback
│   ├── selection.json                 # best checkpoint by the callback's metric
│   └── experiment.yaml
├── evaluate/
│   ├── metrics.json
│   └── experiment.yaml
├── inference/
│   ├── results.json
│   └── experiment.yaml
├── export/
│   ├── cosmos_embed1_vision.onnx      # + cosmos_embed1_vision.onnx_data (weights above 2 GB)
│   ├── cosmos_embed1_text.onnx
│   └── export_config.yaml
└── export_hf/
    └── cosmos_embed1_hf/
```

The per-tower ONNX input/output contract and TensorRT notes are in `references/lora-finetune-physicalai.md`.

## Known Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `MSRVTTDataset: 0 videos found` | `mp4_urls` is not a local glob or metadata filenames do not match videos. | Mount data into the container and set `mp4_urls=/data/video/*.mp4`. |
| Chunk manifest loads 0 videos or skips most entries | Manifest paths (`videos/<file>.mp4`) were not rewritten to the mounted folder. | Set `dataset.*.path_prefix_mapping` from the manifest prefix to the container path. |
| HF download/auth failure | Missing or invalid `HF_TOKEN`, or model agreement not accepted. | Accept the model terms and pass `-e HF_TOKEN`. |
| ONNX export fails with `cannot import name 'runtime_version' from 'google.protobuf'` | protobuf was downgraded in the container; the image's `PIP_CONSTRAINT` resolves `pip install "protobuf<7"` to 4.24.4, below what its `onnx` needs. | Rerun in a fresh container from the pinned image with no `pip install` step. |
| `cannot import name 'Imports' from 'wandb.proto.wandb_telemetry_pb2'` | A 7.0.x Cosmos-Embed image (`wandb==0.21.0` with `protobuf==7.x`). | Use the pinned image; do not pin protobuf at runtime. |
| Resume fails with `Model does not implement 'apply_fsdp'` | Single-GPU resume loaded a consolidated checkpoint while `model.fsdp_shard_size` stayed at the default 8. | Set `model.fsdp_shard_size=1` for local single-GPU resume/retrain. |
| LoRA injection failure | Transformer Engine visual encoder is enabled. | Set `model.network.visual_encoder.transformer_engine=false`. |
| Every `visual_encoder.*.lora_B` tensor is zero after training | Image older than the pinned one (reentrant activation checkpointing with a frozen ViT input; `attn.qkv` bypassed its LoRA wrapper). | Use the pinned image and re-run the `lora_B` check in `references/lora-finetune-physicalai.md`. |
| Evaluate/export of a LoRA checkpoint reports missing or unexpected adapter keys | The downstream spec's `model.lora.target_modules` differs from the trained list (or is empty). | Copy the training spec's `model.lora` block verbatim. |
| Vision ONNX export exits 1 with `Failed to serialize proto` after writing files | Older image checked the above-2 GB graph in memory. | Use the pinned image; on older images the files are valid, confirm with `onnx.checker.check_model(path)`. |
| `RetrievalBenchmarkCallback needs dataset.val_dataset.dataset_type=retrieval_benchmark` | The callback is enabled while `val_dataset` is `mock` or MSR-VTT. | Enable the callback and the benchmark `val_dataset` together, or comment both out. |
| `EADDRINUSE` on port 29500 | Two `cosmos-embed1` runs share a host and the default distributed port. | Give each run its own `MASTER_PORT` / `--master_port`. |
| `torchrun --standalone` hangs at rendezvous | The pod hostname does not resolve for worker ranks. | Launch with `python -m torch.distributed.run --master_addr 127.0.0.1 ... -m cosmos_embed1.scripts.train --config-path /specs --config-name <spec>`. |
| ONNX/HF export complains about missing components | Export checkpoint is partial or adapter-only. | Use a full checkpoint or configure pretrained visual/text sources before export. |
| CUDA OOM | Batch/resolution too high for the GPU. | Enable `visual_encoder.checkpoint_activations`, reduce batch size, use 224p, enable LoRA, or use more GPUs. |
