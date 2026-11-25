# LoRA fine-tuning Cosmos-Embed1 on chunk manifests

End-to-end recipe for `references/spec_template_train_lora.yaml`: a 448p LoRA fine-tune on the public [PhysicalAI-Event-Videos](https://huggingface.co/datasets/nvidia/PhysicalAI-Event-Videos) labels, then checkpoint selection, export, evaluation, and inference with the same spec. All commands run inside the pinned container with the `DOCKER_COMMON` options from `SKILL.md`; paths below are container paths.

## 1. Stage the data

1. Download the dataset and unzip `2026/train/videos.zip` next to `labels.json`, so the host has `data/PhysicalAI-Event-Videos/2026/train/{labels.json,videos/}`.
2. `labels.json` references clips as `videos/<file>.mp4`. The loader rewrites paths only through `dataset.train_dataset.path_prefix_mapping`, so map that prefix to the container path of the unzipped folder (the template already does).
3. Parent videos that are not in the package are skipped with a warning; the training log prints videos kept and dropped, chunks, samples, and unique captions. Check those counts before a long run.

## 2. How captions become training samples

`dataset_type: vad_r1_chunks` reads one record per labeled chunk. Three knobs control the text side:

| Knob | Values | Effect |
|---|---|---|
| `caption_field` | string or list of field names | Every named field joins the chunk's caption pool; a field whose value is a list of strings contributes every string. |
| `caption_mode` | `all` (default), `first`, `random` | `all` makes one sample per (chunk, caption) pair; `first` one sample per chunk with the first caption; `random` one sample per chunk with a caption drawn per access. |
| `label_mode` | `category` (default), `caption` | Label used by the label-aware contrastive loss. `category` labels by anomaly type, so every chunk of the same category is a non-negative. With free-form captions use `caption`, so only identical caption strings share a label. |

The template uses `caption_field: [caption_queries, anomaly_queries, action_queries]`, `caption_mode: all`, `label_mode: caption`. Shipped specs with a scalar `caption_field` behave as before (`all` equals `first` for single-caption chunks).

## 3. Train

```bash
docker run "${DOCKER_COMMON[@]}" "$COSMOS_EMBED_IMAGE" \
  cosmos-embed1 train -e /specs/train_lora.yaml results_dir=/results
```

- `train.grad_accum_iter` multiplies the per-GPU batch; `max_iter`, `validation_iter`, and `checkpoint_iter` count optimizer updates, not micro-steps.
- Keep `validation_iter == checkpoint_iter` so every validated iteration has a checkpoint on disk.
- `model.network.visual_encoder.transformer_engine` must be `false` for LoRA. If 448p runs out of memory, set `visual_encoder.checkpoint_activations: true`; it costs little wall-clock in this configuration.
- Multi-GPU runs launched outside the `cosmos-embed1` CLI (for example inside a pod whose hostname does not resolve) should use `python -m torch.distributed.run --nnodes=1 --nproc_per_node=<n> --master_addr 127.0.0.1 --master_port <port> -m cosmos_embed1.scripts.train --config-path /specs --config-name train_lora`; the Hydra entry point does not accept `-e`. Two concurrent runs on one host need distinct `--master_port` values.

### Confirm the adapters trained

Every ViT adapter listed in `target_modules` must have a non-zero `lora_B`. Run inside the container on a saved checkpoint:

```bash
python - <<'EOF'
import torch
ckpt = torch.load("/results/train/checkpoints/iter_000000640.pt", map_location="cpu", weights_only=False)
sd = ckpt.get("model", ckpt)
zero = [k for k, v in sd.items() if "lora_B" in k and hasattr(v, "abs") and float(v.float().abs().sum()) == 0]
print(len(zero), "zero lora_B tensors")
print("\n".join(zero[:20]))
EOF
```

Expected: zero tensors only under `itm_proj` when `train.use_text_matching_loss` is `false`. Any `visual_encoder.*` entry in that list means the ViT adapters received no gradient; use the pinned image.

## 4. Select the checkpoint

With `val_dataset.dataset_type: retrieval_benchmark` and `train.callbacks.retrieval_benchmark` enabled (both together; the callback raises `TypeError` for any other validation dataset), the trainer scores text-to-video retrieval every `validation_iter`:

- one log line per validation: `RETRIEVAL_BENCHMARK iter=... mAP=... hit@1=... hit@5=...`
- `<results_dir>/train/retrieval_benchmark.jsonl`: one record per validation
- `<results_dir>/train/selection.json`: the best checkpoint so far by `selection_metric` on `query_domain` (iteration 0 is recorded but never selected)

```bash
cat /results/train/selection.json
tail /results/train/retrieval_benchmark.jsonl
```

Without a benchmark root, keep `val_dataset: mock`, leave the callback commented out, and evaluate the saved `iter_*.pt` files with `cosmos-embed1 evaluate` on your own held-out set. Pick the earliest checkpoint at the metric peak rather than the last one.

## 5. Export, evaluate, infer with the same spec

Downstream actions must see the same `model.lora` block as training. `target_modules` has to list exactly the modules that were trained; a shorter or missing list makes PEFT inject the default adapter set and the checkpoint no longer matches.

```bash
SPEC=/specs/train_lora.yaml
CKPT=/results/train/checkpoints/iter_000000640.pt
# HF format: adapters merged into the weights
cosmos-embed1 export -e $SPEC export.mode=huggingface export.checkpoint=$CKPT export.hf_output_dir=/results/export_hf
# ONNX, one graph per tower, dynamic batch axis, adapters merged
cosmos-embed1 export -e $SPEC export.mode=video export.checkpoint=$CKPT export.batch_size=-1 export.on_cpu=true export.onnx_file=/results/export/vision.onnx
cosmos-embed1 export -e $SPEC export.mode=text  export.checkpoint=$CKPT export.batch_size=-1 export.on_cpu=true export.onnx_file=/results/export/text.onnx
# Evaluate on a benchmark release root (if available)
cosmos-embed1 evaluate -e $SPEC evaluate.checkpoint=$CKPT dataset.test_dataset.dataset_type=retrieval_benchmark \
    dataset.test_dataset.metadata=/data/retrieval_benchmark dataset.test_dataset.resolution=[448,448]
# Text->video retrieval over the training manifest
cosmos-embed1 inference -e $SPEC inference.checkpoint=$CKPT dataset.inference_dataset.dataset_type=vad_r1_chunks \
    dataset.inference_dataset.metadata=/data/PhysicalAI-Event-Videos/2026/train/labels.json \
    "dataset.inference_dataset.path_prefix_mapping={videos/: /data/PhysicalAI-Event-Videos/2026/train/videos/}" \
    dataset.inference_dataset.resolution=[512,512] "inference.query.input_texts=[a forklift colliding with a rack]"
```

The merged HF export runs without adapters: `inference.checkpoint=/results/export_hf model.lora.enabled=false`. Rankings from the `.pt` and from the merged export agree up to bf16 rounding.

ONNX contract (per tower, opset 17): `videos float32 [batch, 3, frames, H, W]` in `[0, 1]` (mean/std applied inside the graph) to `video_embedding [batch, embed_dim]`; `input_ids` and `attention_mask` `int64 [batch, 128]` to `text_embedding [batch, embed_dim]`. The vision graph exceeds 2 GB, so its weights land in a sidecar `.onnx_data` file next to the `.onnx`; keep both together. TensorRT engines built with plain `--fp16` from these graphs match the PyTorch embeddings closely (the towers use LayerNorm and softmax attention); validate parity on your own queries before deploying.

## 6. Continue training from a LoRA checkpoint

`train.resume_training_checkpoint_path` is for full-model checkpoints. To extend a LoRA run, export the selected checkpoint to HF format, set `model.pretrained_model_path` to that export directory, and start a fresh LoRA stage with its own warmup and cosine schedule. The second stage trains new adapters on top of the merged weights.

## LoRA and AutoML notes

Treat `model.lora.lora_rank`, `lora_alpha`, `lora_dropout`, and `train.optim.lr` as the first candidates for manual tuning or AutoML-style search before unfreezing larger model blocks. `target_modules` sets where adapters live; the template's list (attention of the last 20 ViT blocks plus Q-Former and projection heads) is a tested starting point. Record the exact list with every checkpoint, because every downstream spec must repeat it. `modules_to_save` keeps modules fully trainable alongside LoRA; leave it empty unless a task-specific head must be preserved.
