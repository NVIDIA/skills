---
name: codonfm-finetune
description: Fine-tune public CodonFM Encodon checkpoints on labeled coding-sequence or coding-variant data using LoRA, head-only, or full fine-tuning. Use when a user explicitly asks to fine-tune CodonFM or Encodon for regression or classification. Support generic public-v1 Encodon workflows only; reject Decodon, MissenseDataset, missense_synom_agg, and generation workflows.
metadata:
  author: "NVIDIA BioNeMo <bionemofeedback@nvidia.com>"
---

# Fine-tune public Encodon

Use `--pretrained_ckpt_path` for public v1. Do not substitute
`--checkpoint_path`: the public runner does not forward that argument to the
fine-tuning task.

## Instructions

Resolve the target label, dataset, checkpoint, and output directory from the
request and available files. Reuse existing data and weights. For training,
check the project's ML dependencies and a compatible NVIDIA GPU before launch.
If a required resource is unavailable, complete the available data preparation
and return the command with the missing prerequisite clearly identified.
When training is requested and the prerequisites are met, execute it and check
the resulting checkpoints and metrics. A request for preparation ends with the
validated inputs and command.

Use the user's labeled dataset when provided. For a demonstration of sequence
regression without a dataset, use the public human RiboNN translation-efficiency
data below and state that choice. This is not a substitute for a user's intended
assay or for labeled coding variants. If variant labels are missing, return the
required schema and a command template promptly; do not search for labels or
invent measured effects.

Default demonstration checkpoint: `nvidia/NV-CodonFM-Encodon-80M-v1`, revision
`399ca9fe17b57941a7bebc6788033919b417413c`, file
`NV-CodonFM-Encodon-80M-v1.safetensors` with sibling `config.json`.
The [public weights](https://huggingface.co/nvidia/NV-CodonFM-Encodon-80M-v1)
are about 307 MB. Download them when needed for the requested work; input
preparation can record an intended checkpoint path. These are the original Encodon weights; the `-TE-`
checkpoints use the separate TransformerEngine implementation.

For an unsupported Decodon or missense-aggregation request, inspect the public
parser/model configuration, explain the missing feature, and finish. Do not
implement the missing model or search private repositories.

## Examples

Prepare a small public-data example with the standard-library helper
[prepare_ribonn.py](scripts/prepare_ribonn.py), running from the repository root.
Set `CODONFM_DATA_PATH` to the CSV you want to create:

```bash
python skills/codonfm-finetune/scripts/prepare_ribonn.py \
    --output "$CODONFM_DATA_PATH"
```

With an existing raw file, add `--input "$RIBONN_DATA_PATH"`. The default reads
at most eight accepted rows per split; `--max-rows-per-split 0` processes the full
input. Remote streaming has a time budget and no automatic retries; use a local
file if it fails. The helper follows the CDS slicing in the
[RiboNN notebook](../../notebooks/4-EnCodon-Downstream-Task-riboNN.ipynb):

- Read the upstream `.csv` with a **tab** delimiter.
- Set `ref_seq = tx_sequence[utr5_size:utr5_size + cds_size]`, `id = transcript_id`,
  and `value = mean_te` unchanged. Do not take another logarithm.
- Preserve source fold groups: 0–7 become `train`, 8 becomes `val`, 9 becomes
  `test`. This is a demonstration holdout, not the notebook's cross-validation.
- Exclude invalid/non-finite rows and CDSs exceeding 2046 codons instead of
  silently truncating labeled examples. Record counts and source in the adjacent
  `.metadata.json`. A small subset does not establish predictive performance.

The pinned dataset URL is in the helper; its source is
[CenikLab/TE_classic_ML](https://github.com/CenikLab/TE_classic_ML/tree/main/data).
The notebook extracts frozen Encodon embeddings and trains a random-forest
regressor with fold-based cross-validation. This skill reuses its data source,
CDS extraction, and target for a separate fine-tuning example; it does not
reproduce the notebook's training procedure or results.

## Supported strategies

- `lora`: adapter fine-tuning; default choice for smaller datasets.
- `head_only_random`: freeze the backbone and train a new head.
- `head_only_pretrained`: train an existing compatible pretrained head.
- `full`: update the complete model.

Accept only `encodon_80m`, `encodon_600m`, or `encodon_1b`.

## Sequence-level regression or classification

Require `id`, `ref_seq`, `value`, and `split` columns. Extra columns are allowed.
Map the user's columns to this loader schema; the RiboNN helper is only for
RiboNN source data. Training needs `train` rows and, when validation is enabled,
`val` rows. A `test` split is needed only for later evaluation. Labels in unused
splits need not be populated. Regression targets must be finite numbers;
classification targets must be integer class indices from zero through
`num_classes - 1`. Use a downstream head for scalar targets.

Check sequence preparation with the user’s assay in mind. The loader converts
uppercase RNA `U` to `T`, and the tokenizer uppercases bases. Ambiguous bases and
incomplete codons can produce unknown tokens; overlength sequences are truncated.
Review these cases rather than silently dropping user records. Choose batches
and a training budget appropriate to the dataset; small training sets may be
resampled by the loader.

Set `CODONFM_CHECKPOINT_PATH` to the checkpoint file and `CODONFM_RUN_DIR` to
your chosen output directory. This example runs ten steps to check the workflow;
choose the training budget for the actual dataset and task:

```bash
python -m src.runner finetune \
    --exp_name property_finetune \
    --model_name encodon_80m \
    --pretrained_ckpt_path "$CODONFM_CHECKPOINT_PATH" \
    --data_path "$CODONFM_DATA_PATH" \
    --process_item codon_sequence \
    --dataset_name CodonBertDataset \
    --finetune_strategy lora \
    --lora_alpha 32 \
    --lora_r 16 \
    --lora_dropout 0.1 \
    --loss_type regression \
    --use_downstream_head \
    --lr 2e-5 \
    --max_steps 10 \
    --warmup_iterations 1 \
    --check_val_every_n_epoch 1 \
    --train_batch_size 4 \
    --val_batch_size 4 \
    --num_workers 0 \
    --num_nodes 1 \
    --num_gpus 1 \
    --out_dir "$CODONFM_RUN_DIR" \
    --checkpoints_dir "$CODONFM_RUN_DIR/checkpoints"
```

For classification, replace `--loss_type regression` with
`--loss_type classification` and pass the correct `--num_classes`.

## Generic coding-variant classification

Use `MutationDataset` only for an ordinary labeled variant head, not the newer
synonymous-codon aggregation loss. Require `id`, the reference-sequence column
(`ref_seq` by default), `ref_codon`, `alt_codon`, `codon_position`, and the chosen
label column. Select existing sequence/label columns with `--ref_seq_col` and
`--label_col`; these overrides apply to `MutationDataset` only. Starting from
the sequence-level command, change/add:

```text
--process_item mutation_pred_mlm
--dataset_name MutationDataset
--label_col label
--loss_type classification
--num_classes 2
--use_downstream_head
--extract-seq
--mask_mutation
--train_val_test_ratio 0.8 0.1 0.1
```

Always keep `--mask_mutation` for masked-codon variant inputs.
Use `--extract-seq` to construct the context around a variant in a full CDS;
already prepared contexts can omit it. Choose split ratios for the dataset;
a held-out test split is optional for training. Public v1 reuses existing
`train_idx.npy`, `val_idx.npy`, and `test_idx.npy` files without checking that
they belong to the current CSV. Verify their provenance before reusing them.

## Execute and outputs

Check prepared data directly against the selected loader's schema above using
ordinary CSV inspection. Verify required columns, finite labels in the splits
used for training, class indices when applicable, sequence preparation, and
variant reference positions. The RiboNN helper checks its output during
preparation. These checks do not require installing CodonFM's ML dependencies.
For preparation requests, report what was checked and provide the training
command. For execution requests, run it once data, weights, and compute are ready.

The existing [runner](../../src/runner.py) has an optional `--dryrun` flag that
builds runtime configuration and skips execution. It requires the ML dependencies
and does not read the dataset or load weights. It is not a data-validation step
or a prerequisite for preparing inputs and commands.

Set validation frequency for the planned training length: for a small example,
`--check_val_every_n_epoch 1` or a smaller `--val_check_interval` avoids public
v1's default interval of 1,000 batches exceeding an epoch.

- Checkpoints are written under the explicitly supplied `--checkpoints_dir`,
  including `last.ckpt` and configured best checkpoints.
- CSV metrics are written below `--out_dir/<exp_name>/version_*` unless W&B is
  enabled.
- W&B requires `--enable_wandb`, `--project_name`, and `--entity` together.
- Fine-tuning does not produce prediction arrays; run an evaluation task
  separately against the resulting checkpoint.

## Boundaries

- Do not use `MissenseDataset`, `missense_seq`, `missense_inference`,
  `missense_synom_agg`, or any `--missense_*` flag. They are absent publicly.
- Do not use Decodon model names, CLM preprocessing, organism tokens, or
  generation datasets.
- Require an explicit learning rate. Public v1 passes `lr=None` otherwise.
- Treat scientific and clinical validity as a separate validation problem;
  successful training does not certify the resulting model.
