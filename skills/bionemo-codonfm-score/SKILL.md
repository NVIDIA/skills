---
name: codonfm-score
description: Validate, prepare, or run public CodonFM Encodon masked-codon variant scoring and review compatibility of its scoring workflows. Use only when the user explicitly requests CodonFM or Encodon, or that context is already established in the conversation. Do not select this skill for a generic variant-scoring request without that context; ask for the variant and intended analysis first.
metadata:
  author: "NVIDIA BioNeMo <bionemofeedback@nvidia.com>"
---

# Score variants with public Encodon

Run general masked-codon `mutation_prediction` only. This produces a research
signal, not a clinical diagnosis or an expression-direction prediction.

## Instructions

First confirm CodonFM or Encodon context in the user's request or established
conversation. If that context is missing, ask for any missing variant details
and the intended analysis before choosing a model or inspecting model-specific
files. The presence of this skill or source files alone does not establish
the user's intent.

For source reviews and command preparation, inspect the supplied source and
metadata without installing the ML runtime. Use an available Python 3
interpreter with standard-library `zipfile`, `json`, and `csv`; do not assume
the `python` alias or `unzip` exists. Read archive members directly with
`ZipFile.namelist()` and `ZipFile.read()` where possible. If extraction is
needed, use a fresh directory from `tempfile.mkdtemp()` or `mktemp -d` and
preserve existing checkouts and scratch directories. Check whether `rg` is
available; use `grep` or Python if it is absent. Read the source sections
needed for the requested command or compatibility question.

Check whether the request is executable in public v1 before installing or
downloading anything. For synonymous-codon aggregation or Decodon, inspect the
[parser](../../src/runner.py) and [model configuration](../../src/config.py),
explain the missing feature, and finish. Do not implement the missing workflow,
search private code, or keep retrying unsupported commands.

Resolve the variant CSV, checkpoint, and output directory from the request and
available files. Validate inputs before inference. Execution requires the
project's ML dependencies and a compatible NVIDIA GPU. If a required resource
is unavailable, return the validated inputs where possible and a command with
the missing prerequisite identified. When scoring is requested and resources
are ready, execute and verify the score arrays. A request for preparation ends
with the inputs and command. If variants are missing, report the required
schema; do not invent variants or silently switch to a public dataset.

Default to the public 80M checkpoint for demonstrations:
`nvidia/NV-CodonFM-Encodon-80M-v1`, revision
`399ca9fe17b57941a7bebc6788033919b417413c`, file
`NV-CodonFM-Encodon-80M-v1.safetensors` and sibling `config.json`.
Reuse an existing checkpoint or download it when needed for the requested work.
Preserve an explicitly requested model size.

## Preflight

1. Confirm `src/runner.py`, `src/data/mutation_dataset.py`, and
   `src/inference/encodon.py` exist.
2. Accept only `encodon_80m`, `encodon_600m`, or `encodon_1b` as
   `--model_name`. The public parser lists larger names, but its model
   configuration does not implement them.
3. For model execution, require a `.ckpt` file, or a `.safetensors` file with
   sibling `config.json`. Input preparation can use a planned path.
4. Validate the CSV headers before starting a GPU job.

## Inputs

Require these CSV columns:

- `id`: unique row identifier.
- `ref_seq`: reference coding sequence, not genomic DNA with introns, UTR-only
  sequence, or protein sequence.
- `ref_codon` and `alt_codon`: three-nucleotide codons.
- `codon_position`: zero-based codon position relative to the CDS.

With `--extract-seq`, `MutationDataset` extracts an appropriate sequence window
from `ref_seq`; it does not derive or require `alt_seq`.

Before running, normalize sequences and codons to uppercase DNA (`A/C/G/T`),
require CDS lengths divisible by three, and check every row satisfies:

```text
0 <= codon_position < len(ref_seq) / 3
ref_seq[3 * codon_position : 3 * codon_position + 3] == ref_codon
```

The public extractor asserts the second condition and otherwise stops the job.

## Examples

Set `CODONFM_DATA_PATH` to the variant CSV, `CODONFM_CHECKPOINT_PATH` to the
checkpoint, and `CODONFM_RUN_DIR` to your chosen output directory:

Use the interpreter from the configured ML environment for inference. The
example uses `python`; substitute that environment's interpreter path if the
alias is unavailable.

```bash
python -m src.runner eval \
    --exp_name variant_scoring \
    --model_name encodon_80m \
    --checkpoint_path "$CODONFM_CHECKPOINT_PATH" \
    --data_path "$CODONFM_DATA_PATH" \
    --process_item mutation_pred_mlm \
    --dataset_name MutationDataset \
    --task_type mutation_prediction \
    --extract-seq \
    --mask_mutation \
    --num_nodes 1 \
    --num_gpus 1 \
    --num_workers 0 \
    --val_batch_size 2 \
    --out_dir "$CODONFM_RUN_DIR" \
    --predictions_output_dir "$CODONFM_RUN_DIR/predictions"
```

Do not remove `--mask_mutation`: without it, the reference codon remains
visible at the scored position and invalidates masked-codon LLR scoring.
For preparation requests, inspect the CSV directly against the input schema
and reference-position checks above, then report the rows checked and provide
the scoring command. Extra columns are allowed; use `--ref_seq_col` if the
reference sequence has a different column name. These checks do not require
the ML runtime. The command above performs inference when resources are ready.

The existing `--dryrun` optionally builds runtime configuration and skips
execution. It requires the ML dependencies, can create the prediction directory,
and does not read the CSV or load weights. Do not use it as evidence that inputs,
checkpoint compatibility, or prediction quality have been validated.

## Outputs

`--predictions_output_dir` receives:

- `ref_likelihoods_merged.npy`
- `alt_likelihoods_merged.npy`
- `likelihood_ratios_merged.npy`
- `ids_merged.npy`

Load the arrays with NumPy and align scores by `ids_merged.npy`. The reported
LLR is `log p(ref_codon) - log p(alt_codon)`; a larger positive value means the
alternate codon is less probable in context. It does not say whether
expression goes up or down.

## Reporting

Keep the final answer concise and self-contained, with the requested command
or compatibility conclusion near the start. For command preparation, include
each row's validation result, the complete command, all four output filenames,
and the LLR definition and sign interpretation. Cite the inspected source
locations for the command, outputs, and scoring semantics. State whether
inference ran; report numerical scores only when execution produced them.

## Boundaries

- General `mutation_prediction` handles both synonymous and missense changes.
- Do not use `missense_prediction`, `missense_inference`, `MissenseDataset`,
  `mutation_pred_clm`, `--organism_token`, or `--causal`; those are newer
  unavailable public-release features.
- If a user asks specifically for synonymous-codon-aggregated missense
  scoring, explain that public v1 only provides the general ref/alt LLR. Do not
  silently substitute the two methods.
