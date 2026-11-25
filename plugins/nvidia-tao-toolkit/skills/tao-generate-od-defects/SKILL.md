---
name: tao-generate-od-defects
description: >-
  Run AnomalyGenNext 1.1 inference from a native testcase or a prepared
  generation plan, then publish validated fine-grained and binary COCO labels.
  Use for synthetic object-detection defect generation, not model training.
license: Apache-2.0
compatibility: Requires the AnomalyGenNext 1.1 container, CUDA GPUs, base and task checkpoints.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags: [tao, data, anomalygen-next, object-detection, synthetic-data, coco]
---

# Generate AnomalyGenNext OD Defects

This leaf uses the upstream generator and pseudo-labeler in the pinned public
container. It does not prepare detector gaps, run AMP, train AnomalyGenNext, or
append outputs to a detector training set.
Read `references/execution-contract.md` when adapting input or completion
behavior, and `references/container-runtime.md` before submission.

## Inputs

Pass either:

- `--inputs-dir` pointing to the completed output of
  `tao-prepare-anomalygennext-inputs`; or
- `--input-data-path`, `--checkpoint`, and matching `--recipe` for a native
  AnomalyGenNext testcase.

Also provide the Cosmos3-Nano base checkpoint, the complete AnomalyGenNext
checkpoint root, and one or more GPUs. Every testcase image and aligned mask
must exist, every anomaly type must occur in the recipe, and the requested row
count must match the testcase. When a prepared integrity manifest exists,
every recorded hash is verified before generation.

## Run

Invoke `tao-launch-workflow`, review the exact image, mounts, GPU shape, runtime,
and output directory, then submit the `generate` action:

```bash
scripts/generate_od_defects.py \
  --inputs-dir /results/prepared_inputs \
  --base-checkpoint /models/Cosmos3-Nano \
  --checkpoint-root /workspace/paidf-anomalygen/checkpoints \
  --output-dir /temporary/generation \
  --published-root /persistent/generation \
  --no-guardrail \
  --num-gpus 1
```

When execution uses temporary storage, `--published-root` records the
persistent locations that will contain the saved results.

Guardrails are enabled by default and preserve the native invocation without a
guardrail flag. The wrapper appends `--no-guardrail` only when explicitly
disabled.
`--no-guardrail` disables text screening, image content-safety screening, and
face blurring together. The checkpoint root is mounted over the image's
canonical `/workspace/paidf-anomalygen/checkpoints` tree and validated before
GPU work. The Nano tokenizer, Edge processor, and DINOv2 assets are always
required; guardrail-enabled runs additionally require the Qwen3Guard and Cosmos
Guardrail repositories. Generation is forced offline after validation. The
selected platform must honor the declared `checkpoint_root.container_path`;
the generation leaf rejects a different mount point. The base-checkpoint
argument remains separate: it is the parent containing
`checkpoint.json` and the `model/` checkpoint directory.

## Completion

For each dataset, `generated + guardrail_blocked` must equal requested rows.
With `--no-guardrail`, `generated` must equal requested rows and
`guardrail_blocked` remains zero. The validation summary records the selected
guardrail mode.
The pseudo-label count must match generated images, each image needs an
annotation, categories must be declared anomaly types, and every COCO bbox must
be positive and inside its image. The action emits:

```text
DATASET/raw/
DATASET/pseudo_labels/coco_annotations.json
pseudo_labels/coco_annotations.json
pseudo_labels/coco_annotations_od_defect.json
validation_summary.json
status.json
```

The native COCO preserves `TEXTURE+DEFECT` categories. The binary file maps all
annotations to `defect`. Only a calling application may admit these outputs to
training; this leaf always reports `training_pool_mutated=false`.
