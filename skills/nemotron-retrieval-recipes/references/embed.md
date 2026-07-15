# Embedding Recipe Reference

Load this reference for `nemotron embed ...` work or for questions about first-stage retrieval, bi-encoder training, low Recall@k, missing relevant documents, embedding NIMs, or re-indexing after model changes.

## Contents

- Grounding Paths
- When To Use Embed
- Commands
- Data And Credential Safety
- Stage Map
- Stage Contracts
- Model Profiles
- Profile Selection Invariants
- Operating Patterns
- NIM Smoke Test
- Tests And Checks

## Grounding Paths

- Recipe README: `src/nemotron/recipes/embed/README.md`
- CLI group: `src/nemotron/cli/commands/embed/_typer_group.py`
- Pipeline command: `src/nemotron/cli/commands/embed/run.py`
- Stage configs: `src/nemotron/recipes/embed/stage*/config/{default,llama}.yaml`
- Main outputs: `output/embed/`

## When To Use Embed

Use embedding fine-tuning when relevant documents are not retrieved into the candidate set, Recall@k is low, domain terms are poorly matched, or the user needs a better first-stage retrieval model. Embedding changes usually require re-embedding and re-indexing the deployment corpus.

## Commands

Use `uv run` when `nemotron` is not already available.

```bash
uv run nemotron embed info
uv run nemotron embed --help
uv run nemotron embed run -c default -d --from prep --to eval
```

For raw domain documents, preview only data generation and prep before any training plan:

```bash
uv run nemotron embed run -c default -d --from sdg --to prep
```

If training/eval pairs already exist, skip SDG and preview prep through eval:

```bash
uv run nemotron embed run -c default -d --from prep --to eval
```

Stage commands:

```bash
uv run nemotron embed sdg -c default corpus_dir=/path/to/docs
uv run nemotron embed prep -c default
uv run nemotron embed finetune -c default
uv run nemotron embed eval -c default
uv run nemotron embed deploy -c default

# The Llama profile adds the export stage.
uv run nemotron embed export -c llama
uv run nemotron embed deploy -c llama
```

Remote execution uses root `env.toml` profiles:

```bash
uv run nemotron embed finetune -c default --run my-cluster
uv run nemotron embed finetune -c default --batch my-cluster
```

## Data And Credential Safety

Stage 0 SDG can transmit the user's text corpus or fetched HF corpus content to NVIDIA-hosted API endpoints for synthetic data generation. Before running SDG on proprietary, confidential, regulated, or customer data, confirm the user's data-governance policy permits that transfer; otherwise use an approved private or air-gapped path.

Protect `NVIDIA_API_KEY` and `NGC_API_KEY` as secrets. The default
Stage 0 profile uses Data Designer's built-in endpoint unless
`NVIDIA_API_BASE_URL` is set; the `llama` profile uses the API Catalog
defaults. Keep credentials in environment variables, local
`.env` files excluded from version control, or an approved secrets manager;
never hardcode them in commands, scripts, configs, or committed logs. Rotate
any key that may have been exposed.

## Stage Map

| Stage | Command | Input | Output | Notes |
| --- | --- | --- | --- | --- |
| 0 SDG | `embed sdg` | Text corpus or HF URI | Profile-specific Stage 0 directory | Requires the profile's NVIDIA inference credential. |
| 1 prep | `embed prep` | Stage 0 output or existing QA data | Profile-specific Stage 1 directory | Converts data, mines hard negatives, and creates BEIR eval data. |
| 2 finetune | `embed finetune` | `train_mined.automodel_unrolled.json` | Profile-specific checkpoints | AutoModel contrastive training. |
| 3 eval | `embed eval` | BEIR eval data and checkpoint | Profile-specific `eval_results.json` | Compare base, fine-tuned, and optionally NIM retrieval metrics. |
| 4 export | `embed export` | Fine-tuned checkpoint | Skipped by default; ONNX/TensorRT for `llama` | Only the Llama profile needs this stage. |
| 5 deploy | `embed deploy` | PyTorch checkpoint or Llama export | NIM or vLLM on `host_port` | Requires Docker. Default local-artifact deployment does not forward `NGC_API_KEY`; `NIM_CUSTOM_MODEL` or model-download paths may require it. |

The pipeline order is `sdg`, `prep`, `finetune`, `eval`, `export`, `deploy`; `embed run` defaults to `--to eval`.


## Stage Contracts

| Stage | Required Inputs | Default Creates | `llama` Profile Difference | Cheapest Check |
| --- | --- | --- | --- | --- |
| 0 SDG | Text corpus, inference credential | `output/embed/nemotron-3-1b/stage0_sdg` | `output/embed/stage0_sdg`; Llama profile generation models | `uv run nemotron embed sdg -c default -d` |
| 1 prep | Stage 0 output | `output/embed/nemotron-3-1b/stage1_data_prep` | Flat artifact root and Llama hard-negative miner | `uv run nemotron embed prep -c default -d` |
| 2 finetune | Unrolled Automodel JSON | `output/embed/nemotron-3-1b/stage2_finetune/checkpoints` | Llama model, Llama optimizer master-weight setting | `uv run nemotron embed finetune -c default -d` |
| 3 eval | Fixed BEIR split and checkpoint | `output/embed/nemotron-3-1b/stage3_eval/eval_results.json` | Llama model/API identity and Llama artifact root | `uv run nemotron embed eval -c default -d` |
| 4 export | Fine-tuned checkpoint | Explicit no-op | Exports ONNX/TensorRT and must use `-c llama` | `uv run nemotron embed export -c llama -d` |
| 5 deploy | PyTorch checkpoint or exported Llama artifact, Docker | Default can mount safetensors through NIM `NIM_ENGINE_MODEL_PATH` or vLLM | Mounts ONNX/TensorRT through `NIM_CUSTOM_MODEL` | `uv run nemotron embed deploy -c default -d backend=vllm` |

## Model Profiles

Default (`-c default`):

- Nemotron 3 Embed is the default model family.
- Uses `nvidia/Nemotron-3-Embed-1B-BF16` from Hugging Face.
- Stage 0 uses Data Designer's built-in endpoint or the generic
  `NVIDIA_API_BASE_URL` override and
  `nvidia/nemotron-3-ultra-550b-a55b` for generation and judging.
- Model-dependent outputs are isolated below `output/embed/nemotron-3-1b/`.
- Every stage derives its paths from `artifact_root`; a pipeline-wide override
  relocates the complete artifact chain without changing model identity.
- Stage 4 has `enabled=false`; it returns a skipped result without loading or
  converting the checkpoint.
- Stage 5 can use a compatible NIM image configured by
  `NEMOTRON3_EMBED_NIM_IMAGE` with `backend=nim`, mounting the Stage 2
  consolidated checkpoint read-only at `/model` with `NIM_ENGINE_MODEL_PATH=/model`.
  Or use the checked-in `nvcr.io/nvidia/vllm:26.06-py3` image with
  `backend=vllm`; it mounts the same checkpoint at `/model` and detects its
  embedding configuration automatically. Neither local-artifact path forwards
  `NGC_API_KEY`.
- Deploy preflight requires hidden size 2,048, 16 layers, 24 attention heads,
  8 key/value heads, intermediate size 6,144, and vocabulary size 131,072.
- NIM uses a 512-token runtime limit and the default `padded-naive-fp16`
  pipeline; override `NEMOTRON3_EMBED_NIM_PIPELINE_ID` only when the target
  hardware supports another pipeline. vLLM derives sequence length, pooling,
  activation, and prompts from the checkpoint.
- The evaluator retries null/non-finite endpoint responses up to 32 times per
  affected input. Preserve retry warnings as serving-reliability evidence.
- Stage 2 pins Transformers 5.12.1. Stages 1 and 3 retain the supported 5.1-5.5
  range for their checkpoint paths.

Llama profile (`-c llama`):

- Uses `nvidia/llama-nemotron-embed-1b-v2` for mining, training, and checkpoint
  evaluation.
- Retains the former `output/embed/stage*` locations.
- Stage 4 remains enabled and uses the Llama bidirectional adapter to export
  ONNX and optionally TensorRT.
- Stage 5 uses `nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:1.10.1`, mounts
  the export at `/opt/nim/custom_model`, sets `NIM_CUSTOM_MODEL`, and
  retains NGC credential forwarding.

## Profile Selection Invariants

- Run `uv run nemotron embed info` when model identity is unclear, then carry
  the selected `-c` value and `artifact_root` through every stage.
- Both profiles use approved NVIDIA checkpoint code with
  `trust_remote_code=true`. Preserve each stage's checked-in Transformers
  constraint instead of imposing one range across the full pipeline.
- Do not copy performance overrides between profiles without a dry-run. The
  default Nemotron 3 eval batch is `4`; the Llama profile uses `128`.
- Preserve each profile's optimizer and checkpoint defaults. Nemotron 3 uses
  no FlashAdamW master weights and fixed 1000-step checkpoint/validation
  intervals; Llama uses 32-bit master weights and auto-scaled 100-step
  intervals.

```bash
uv run nemotron embed run -c default -d --from sdg --to eval
uv run nemotron embed deploy -c default -d backend=vllm
uv run nemotron embed deploy -c default -d

uv run nemotron embed run -c llama -d --from sdg --to export
uv run nemotron embed deploy -c llama -d
```

## Operating Patterns

- Skip SDG when the user already has generated QA pairs or wants NVIDIA's pre-generated dataset; start Stage 1 with `sdg_input_path`.
- For production-like chunks, align `sentences_per_chunk`, `passage_max_length`, and eval `max_length` with expected retrieval chunks.
- If increasing sequence length, reduce batch sizes before attempting to recover from OOM.
- Mine at least as many hard negatives as Stage 2 will consume: `hard_negatives_to_mine >= train_n_passages - 1`.
- Preserve the selected profile's `stage1_data_prep/eval_beir/` across comparisons so metrics are not shifted by new splits.
- Use `val_ratio=0` only for small datasets where preserving test size matters; use a validation split for larger datasets.
- Inspect existing `output/embed/` artifacts before rerunning a stage. Ask before deleting checkpoints, cached embeddings, or generated data.
- For deploy handoff, include the exact deploy command, `detach=true` when background service ownership is expected, container name, host port, smoke test, and container stop or replacement steps.

## Serving Smoke Tests

```bash
# Default profile
curl -X POST http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": ["hello"], "model": "nvidia/nemotron-3-embed-1b", "input_type": "query"}'

# With -c llama, use model nvidia/llama-3.2-nv-embedqa-1b-v2 instead.

# Default profile with backend=vllm. The checkpoint applies the prompt for
# the input type, so do not add `query: ` or `passage: ` yourself.
curl -X POST http://localhost:8000/v2/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["hello"], "model": "nvidia/nemotron-3-embed-1b", "input_type": "query"}'
```

For behavioral comparison, evaluate the local checkpoint and selected serving
backend in the same run and write to a separate directory:

```bash
uv run nemotron embed eval -c default eval_nim=true eval_base=false \
  eval_finetuned=true \
  output_dir=./output/embed/nemotron-3-1b/stage3_eval_nim_comparison

# For backend=vllm, switch the evaluation adapter to /v2/embed. It sends
# `input_type` and does not add query/passage prefixes itself.
uv run nemotron embed eval -c default eval_nim=true embedding_api_backend=vllm \
  eval_base=false eval_finetuned=true \
  output_dir=./output/embed/nemotron-3-1b/stage3_eval_vllm_comparison
```

This aggregate metric comparison is not proof of artifact identity. Deployment
mount/fingerprint checks establish which local checkpoint was selected. Set
`fail_on_nim_metric_drift=true` only when the configured tolerances should
gate the run.

## Tests And Checks

```bash
uv run nemotron embed --help
uv run nemotron embed finetune -c default -d
uv run pytest tests/recipes/embed tests/nemo_runspec/test_execution_uv_spec.py -q
```
