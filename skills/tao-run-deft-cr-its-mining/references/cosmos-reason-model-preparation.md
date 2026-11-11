# Cosmos Reason Baseline Model Preparation

Use this reference only for the `prepare_cosmos_reason_model` stage, after workflow validation and initialization and before baseline evaluation.

## Contract

`cosmos_reason.baseline_model_path` is the source-model lineage and must point to a local checkpoint inside the DEFT workspace. The preparation stage inspects its `config.json`:

- `qwen3_vl`: validate and reuse the source directory.
- `cosmos3_omni`: convert it to a complete Qwen3-VL safetensors checkpoint.
- any other `model_type`: stop with an unsupported-format error.

The prepared checkpoint, not the raw Omni source, is used for baseline evaluation, iteration 1 training, and every non-continual iteration. A continual iteration after iteration 1 still uses the previous iteration checkpoint.

## Inputs Owned By Other Skills

Resolve `DEFT_COSMOS_REASON_IMAGE` from `images.tao_toolkit.deft_cosmos_reason` in the installed skill bank's `versions.yaml`.

For an Omni source, first read and follow the current `tao-finetune-cosmos-reason` model-preparation contract. The pinned `DEFT_COSMOS_REASON_IMAGE` owns the Cosmos-RL preparation wrapper and its isolated Framework converter runtime. Inspect that exact local image and record its immutable ID or repo digest as `DEFT_COSMOS_REASON_IMAGE_DIGEST`. Do not select a second image, invent a digest, or invoke the Framework converter directly.

The workflow uses `Qwen/Qwen3-VL-8B-Instruct` as the Nano architecture donor. `prepare_cosmos_reason_model.py` resolves the requested donor revision to an immutable Hugging Face commit and records it. It passes `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` by environment only when already set; never print or persist a token. Edge and Super require a separately validated architecture mapping and are not eligible for the Nano default.

## Run

For an already-Qwen3-VL baseline:

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/prepare_cosmos_reason_model.py" \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml" \
  --run-dir "$RUN_DIR" \
  --runtime-image "$DEFT_COSMOS_REASON_IMAGE" \
  --runtime-image-digest "$DEFT_COSMOS_REASON_IMAGE_DIGEST"
```

The same command handles a Cosmos3 Nano Omni baseline:

```bash
"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/prepare_cosmos_reason_model.py" \
  --workspace "$WORKSPACE" \
  --workflow-yaml "$WORKSPACE/specs/workflow.yaml" \
  --run-dir "$RUN_DIR" \
  --runtime-image "$DEFT_COSMOS_REASON_IMAGE" \
  --runtime-image-digest "$DEFT_COSMOS_REASON_IMAGE_DIGEST"
```

The helper invokes the installed model skill's `scripts/prepare_cosmos3_vlm_checkpoint.py` with `--backend cosmos-rl`, the source and architecture inputs, the output/cache paths, and the exact runtime image and digest. That shared helper owns the packaged conversion entrypoint and its isolated converter runtime. Override `--model-preparation-script` only if the plugin runtime resolves that shared helper at a different absolute path.

Converted outputs are cached at `<workspace>/model/prepared/<source-fingerprint16>/<conversion-fingerprint16>/prepared`. The source fingerprint covers checkpoint metadata and safetensors contents; the conversion fingerprint covers the donor commit, Cosmos-RL backend, and runtime-image digest. Reuse requires a complete checkpoint and matching provenance. A stale or partial cache is replaced only at that exact deterministic target.

## Completion And Logging

The stage validates `config.json`, tokenizer files, processor configuration, and either a complete single-file safetensors checkpoint or every shard referenced by its index. It then constructs the model from config inside `DEFT_COSMOS_REASON_IMAGE` without loading the full weights. Completion requires `$RUN_DIR/baseline/model_preparation.json` with `status: ready` and `runtime_validation.status: passed`.

Log the stage with `iter-label=baseline`, `stage=prepare_cosmos_reason_model`, and the manifest as its artifact. On failure, log the error and stop before creating the baseline evaluation TOML.
