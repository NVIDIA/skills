---
name: tao-finetune-anomalygennext
description: >-
  Validate a user AnomalyGenNext dataset and recipe, freeze its validation set,
  and prepare a canonical texture fine-tuning recipe. Use when AnomalyGenNext
  task weights do not yet exist. Do not use for ordinary defect generation.
license: Apache-2.0
compatibility: Requires the AnomalyGenNext 1.1 container, Pillow, and its model checkpoints.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags: [tao, data, anomalygen-next, fine-tuning, synthetic-data]
---

# Fine-tune AnomalyGenNext

This leaf prepares a user-owned dataset and optional recipe for task-specific
AnomalyGenNext LoRA training. The pinned image is declared in
`references/skill_info.yaml`; do not replace it with the older 1.0 release.
Read `references/input-contract.md` when validating a dataset or adapting a
recipe, and `references/container-runtime.md` before submission.

## Inputs

```text
DATASET/
  defect_spec.jsonl
  TEXTURE/
    clean_image/*
    anomaly_image/DEFECT/*
    mask/DEFECT/*
VALIDATION/testcase.jsonl
```

Also supply a DCP-format Cosmos3-Nano base checkpoint directory containing
`checkpoint.json`, `model/.metadata`, and at least one `model/*.distcp` shard,
`Wan2.2_VAE.pth`, and
a complete checkpoint root containing the required Qwen tokenizer model assets
under `hf/` plus `facebook/dinov2-large/` for validation. The `hf/` directory
name is part of the upstream image's fixed checkpoint layout. The validation
JSONL must contain `image_filename`, `mask_filename`, and `anomaly_type`; each trained
`TEXTURE+DEFECT` needs at least three rows. A separately stored defect spec is
accepted with `--defect-spec`.

If the official Cosmos3-Nano checkpoint is not already staged in DCP format,
run the converter shipped in the pinned AnomalyGenNext image and write its
output to a mounted, persistent directory:

```bash
python -m cosmos_framework.scripts.convert_model_to_dcp \
  -o /models/Cosmos3-Nano-dcp \
  --checkpoint-path Cosmos3-Nano
```

The registered `Cosmos3-Nano` name resolves the official checkpoint. To
convert an already-staged Hugging Face checkpoint instead, replace it with
that directory's absolute container path. Use the converter output as
`--base-checkpoint`; preparation verifies `checkpoint.json`,
`model/.metadata`, and the generated `model/*.distcp` shards before GPU work.

Dataset and validation images and masks must use `.jpg`, `.jpeg`, or `.png`,
matching the extensions supported by the AnomalyGenNext 1.1 runtime loader.
Every mask pixel must be binary `0` or `255`; palette or grayscale masks with
intermediate values are rejected before training. Each mask must contain both
values, so empty/all-zero and full/all-255 masks are also rejected.

An optional user `recipe.yaml` is a template. Custom training settings remain,
while dataset, checkpoint, validation, type order, and iteration-zero validation
are replaced by validated identities. Without a template, the packaged recipe
uses the established 5000-step defaults.

## Prepare

After the common launch review, run the `prepare_recipe` action:

```bash
scripts/prepare_finetune_recipe.py \
  --dataset-root /data/my_dataset \
  --validation-testcase /data/validation/testcase.jsonl \
  --base-checkpoint /models/Cosmos3-Nano \
  --vae-path /models/Wan2.2_VAE.pth \
  --checkpoint-root /models/anomalygen-checkpoints \
  --dataset-name my_dataset \
  --recipe-template /data/recipe.yaml \
  --output /results/canonical_recipe.yaml
```

The action freezes absolute validation paths, validates the DCP base-checkpoint
shape, rejects image extensions the runtime cannot decode, validates
anomaly/mask pairing and binary masks, checks type agreement with
`defect_spec.jsonl`, refuses output reuse, and emits a recipe plus metadata.
`validation_iter` must be a multiple of `save_iter`; `max_iter` must reach a
post-baseline validation.

## Train and accept

Invoke `tao-launch-workflow`, review the platform, image, mounts, GPU shape,
runtime, and exact recipe, then submit the `train` action. Bind the entire
checkpoint root read-only at the fixed container path declared in
`skill_info.yaml`; the upstream trainer resolves the Qwen tokenizer model
assets at `checkpoints/hf` and DINOv2 at
`checkpoints/facebook/dinov2-large` beneath its image repository. Mounting only
the DINOv2 directory or setting `HF_HOME` does not satisfy that contract.

```bash
scripts/finetune_anomalygennext.sh \
  --recipe /results/canonical_recipe.yaml \
  --results-dir /new/training_results \
  --checkpoint-root /workspace/paidf-anomalygen/checkpoints \
  --num-gpus 1
```

The wrapper uses the upstream trainer already inside the pinned container. It
refuses existing publication outputs and, unless `--resume` is explicit, an
existing runtime directory. Accept success only when `training_handoff.json`
is `COMPLETE`: iteration zero and a later scored checkpoint must exist,
`best_checkpoint.txt` must select the maximum `Average.nn_score`, and the score
must improve strictly. The handoff hashes the copied adapter and recipe.

Read `references/nn-score-contract.md` before changing metric or acceptance
settings. A 1000-step run is the smallest quality smoke; shorter runs prove
wiring only.
