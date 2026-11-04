---
name: codonfm-embed
description: Validate coding-sequence CSVs, extract public CodonFM/Encodon embeddings, and choose checkpoints for downstream property modeling.
license: Apache-2.0
metadata:
  author: "NVIDIA BioNeMo <bionemofeedback@nvidia.com>"
  tags: [biology, codonfm, embeddings]
---

# Extract public Encodon embeddings

## Purpose

Extract one frozen CLS vector per coding sequence with public Encodon v1.
Support input validation, command preparation, extraction, and checkpoint
selection for translation efficiency, expression, or mRNA stability modeling.
Extraction does not automatically train a downstream regressor.

## Prerequisites

- Validation needs Python 3 standard library only; no GPU, weights, or API key.
- Execution needs the public CodonFM checkout, its `requirements.txt` environment,
  a compatible NVIDIA GPU, and local checkpoint weights. A metadata JSON is not
  a checkpoint. A `.safetensors` file needs its sibling `config.json`; `.ckpt`
  checkpoints are also supported by the public loader.
- Run `python -m src.runner` from the CodonFM repository root. In an isolated
  workspace, use supplied source artifacts; source paths below are relative to
  that checkout or source archive, not this skill directory.

## Inputs

Input source precedence: explicit user prompt arguments, then supplied
files/checkpoint metadata, then inspected public runner defaults. Resolve
conflicting model names and checkpoint metadata before execution. Supplied 80M
metadata is useful for preparing an 80M command; it does not restrict an
open-ended recommendation to that size.

Required for validation: a CSV. Required for extraction: the CSV, checkpoint,
matching model name, and output directory. Optional: context length and batch
size overrides. Checkpoint-selection questions can be answered without a CSV.

| Input | Requirement or default |
| --- | --- |
| Sequence CSV | Columns `id`, `ref_seq`, `value`, `split`; extra columns allowed |
| `id` | Nonblank, unique IDs for unambiguous output association |
| `ref_seq` | Coding sequence, uppercase DNA `A/C/G/T`, length divisible by three; public dataset converts uppercase `U` to `T` |
| `value` | Numeric label; use `0.0` for new extraction-only data, preserve supplied labels |
| `split` | Only exact `test` values enter extraction; blank/other values are excluded |
| Checkpoint and model | Match weights/config to `encodon_80m`, `encodon_600m`, or `encodon_1b` |
| Context length | Public runner default `2048` tokens, including CLS and SEP |
| Output directory | A fresh run directory with an empty predictions directory |

## Instructions

1. **Choose the requested workflow.** For a checkpoint/performance question,
   read [checkpoint selection](references/checkpoint-selection.md) and answer
   from public benchmark evidence. For the strongest published downstream
   results, prefer the public **1B random-mask checkpoint** when resources allow;
   80M is a demonstration or resource-constrained choice. A small labeled set
   alone does not establish that 80M frozen features are better. Do not download
   weights or inspect the entire source tree just to make a recommendation.
2. **Inspect supplied source only where needed.** Confirm runner/config,
   `src/data/codon_bert_dataset.py`, `src/data/preprocess/codon_sequence.py`,
   `src/inference/encodon.py`, or `src/utils/pred_writer.py` for the relevant
   behavior. Read ZIP members with `zipfile.ZipFile.namelist()` and `.read()`;
   source inspection does not need extraction. If a checkout is needed, use a
   new directory from `tempfile.mkdtemp()` or `mktemp -d`, without deleting or
   overwriting an existing directory. For Decodon support questions, inspect
   runner/config and model/inference modules, cite the inspected files, explain
   the missing public implementation, and finish there.
3. **Validate the CSV before running extraction.** Run the bundled checker
   below with the intended context length. Report per-row verdicts using CSV
   row numbers as well as IDs, since IDs can repeat. Separate excluded rows,
   invalid inputs, duplicate-ID warnings, and truncation. Propose fixes without
   silently rewriting supplied data. The checker is a preflight, not model
   execution or proof of biological CDS validity.
4. **Deliver the requested preparation or execution.** For preparation, return
   a complete command with resolved paths (or clearly identified prerequisites),
   the test-row count, validation findings, and the output contract below.
   Include all task/dataset/process flags in the final answer, even if already
   shown in a tool call. For extraction, reuse/download the chosen checkpoint
   when needed, execute once resources are ready, and verify the saved arrays.
   If resources are missing, finish preparation and state what is missing.

## Available Scripts

| Script | Purpose | Arguments |
| --- | --- | --- |
| [validate_inputs.py](scripts/validate_inputs.py) | Read-only CSV validation and per-row verdicts | Required CSV path; optional `--context-length` (default `2048`) |

Run the preflight with Python; `CODONFM_SKILL_DIR` is the directory containing this file:

```bash
python "$CODONFM_SKILL_DIR/scripts/validate_inputs.py" "$CODONFM_DATA_PATH" \
    --context-length 2048
```

The checker prints JSON. Exit `0` means no findings, `1` means row findings to
review (including exclusions/warnings), and `2` means a file/schema error.
Neither warnings nor exclusions imply that the public runner will crash.

## Output Format

The checker emits JSON with `total_rows`, `test_rows`, `excluded_rows`,
`context_length`, `codon_limit`, `warnings`, and `rows`. Each row records its
one-based data-row number (excluding the header), ID, split, verdict, issues,
sequence/value validity, and retained/lost codons. A file/schema error emits
`error` and `csv` instead. These are preflight findings, not generated embeddings.

## Examples

Set `CODONFM_DATA_PATH` to the CSV, `CODONFM_CHECKPOINT_PATH` to the weights,
`CODONFM_MODEL_NAME` to the matching architecture, and `CODONFM_RUN_DIR` to a
fresh output directory. Substitute actual paths in a prepared command:

```bash
python -m src.runner eval \
    --task_type embedding_prediction \
    --process_item codon_sequence \
    --dataset_name CodonBertDataset \
    --exp_name embed_extract \
    --model_name "$CODONFM_MODEL_NAME" \
    --checkpoint_path "$CODONFM_CHECKPOINT_PATH" \
    --data_path "$CODONFM_DATA_PATH" \
    --context_length 2048 \
    --num_nodes 1 \
    --num_gpus 1 \
    --num_workers 0 \
    --val_batch_size 2 \
    --out_dir "$CODONFM_RUN_DIR" \
    --predictions_output_dir "$CODONFM_RUN_DIR/predictions"
```

For a low-cost demonstration, `encodon_80m` matches
`nvidia/NV-CodonFM-Encodon-80M-v1`, revision
`399ca9fe17b57941a7bebc6788033919b417413c`, file
`NV-CodonFM-Encodon-80M-v1.safetensors` and sibling `config.json`.

## Outputs

- Under `--predictions_output_dir`, `embeddings_merged.npy` contains frozen
  final-layer CLS vectors, shape `(processed_rows, hidden_size)`.
- `ids_merged.npy` is index-aligned: embedding row `i` belongs to ID row `i`.
  Use these IDs to join to the CSV; do not assume every CSV row was retained.
  Duplicate IDs make that join ambiguous even when extraction succeeds.
- For the one-GPU example, verify both arrays have the expected test-row count,
  embeddings are finite, and width matches checkpoint config (`1024` for 80M,
  `2048` for 600M/1B). Do not fabricate arrays for a preparation-only request.

The public checkout's downstream-model references are:

- `notebooks/4-EnCodon-Downstream-Task-riboNN.ipynb`
- `notebooks/5-EnCodon-Downstream-Task-mRFP-expression.ipynb`
- `notebooks/6-EnCodon-Downstream-Task-mRNA-stability.ipynb`

## Limitations

- Public v1 has no Decodon model/inference implementation, Decodon notebooks,
  `notebooks/te_predictor.py`, or `notebooks/mfe_predictor.py`.
- At context length `2048`, retain the first `2046` codons; any remaining
  3-prime sequence is lost. Increasing the flag does not validate a longer
  context. Disclose deliberate cropping or a separate chunking/aggregation
  strategy; neither is equivalent to embedding the complete sequence once.
- `--dryrun` builds runtime configuration, may create directories, and needs
  ML dependencies; it reads neither the CSV nor the weights and is not input
  validation.
- Do not claim a benchmark-trained regressor generalizes to a new organism,
  cell type, or assay without new labeled validation data.
- Do not invoke this skill for a generic expression-prediction request that
  does not mention CodonFM or Encodon.

## Troubleshooting

| Symptom | Cause and action |
| --- | --- |
| Missing `split` column | Eval requests the test split despite the dataset docstring calling this column optional; add an explicit split column to a corrected copy |
| Fewer output rows | Blank/non-`test` split values are silently filtered; set intended extraction rows to exact `test` in a corrected copy |
| Repeated output IDs | Duplicate input IDs are not rejected; assign unique IDs while preserving a mapping to the original rows |
| Oversized sequence | Preprocessing truncates at `context_length - 2` codons; report retained/lost lengths and agree on a sequence-handling strategy |
| Missing weights or dependencies | Complete validation/command preparation; metadata and `--dryrun` do not substitute for weights |
| Merge failure on a repeated run | The writer scans `.npy` files; use a fresh predictions directory to avoid stale shards or merged arrays |
