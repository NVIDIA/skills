# Cosmos Reason 3 Framework action contract

## Prepare the Cosmos Reason 3 checkpoint

The model this loop fine-tunes is the published **Cosmos Reason 3** reasoner
(`nvidia/Cosmos3-Nano` by default). That published directory ships in Cosmos3's
own native Omni format — recognizable by `model_type="cosmos3_omni"` and
`Cosmos3ForConditionalGeneration` — while this application's Framework
Train/Evaluate actions consume a Qwen3-VL VLM safetensors PTM.

So a one-time conversion turns the selected Cosmos Reason 3 reasoner into a
Qwen3-VL PTM. The reasoner stays the model's identity and lineage throughout;
the Qwen3-VL PTM is only the on-disk format Framework consumes.

After the user approves the launch review, prepare the selected checkpoint with
the helper owned by `tao-finetune-cosmos-reason`:

```bash
# If credentials are not already exported, source only the user-approved env
# file selected during Pre-Flight before running this block.

# --base-model-path-or-uri may be a URI or a LOCAL DIRECTORY. This workflow
# downloads the source first (~33 GB excluding demo assets; the repo is
# ungated) so the approved conversion has a stable, inspectable local input:
# Repeat --exclude per pattern. A second bare pattern is parsed as a positional
# FILENAMES argument, and the whole download fails with "File not found in
# repository" pointing at .../resolve/main/images/%2A — which reads like a
# broken repo rather than a CLI syntax error.
hf download nvidia/Cosmos3-Nano --local-dir "$COSMOS3_SOURCE_DIR" \
  --exclude 'assets/*' --exclude 'images/*'

PREPARED_MODEL_PARENT=$(dirname "$PREPARED_MODEL_HOST_PATH")
mkdir -p "$PREPARED_MODEL_PARENT"
probe="$PREPARED_MODEL_PARENT/.tao-write-probe.$$"
(umask 077 && : >"$probe" && rm -f "$probe") || {
  echo "FATAL: $PREPARED_MODEL_PARENT is not writable by uid $(id -u)" >&2
  exit 2
}

# Pre-Flight resolves this packaged Nano mapping to an immutable Hub revision.
COSMOS3_VLM_ARCHITECTURE_MODEL="${COSMOS3_VLM_ARCHITECTURE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
: "${COSMOS3_VLM_ARCHITECTURE_REVISION:?set the immutable revision resolved by the model planner}"
COSMOS3_CONVERSION_CACHE_DIR="${COSMOS3_CONVERSION_CACHE_DIR:-$PREPARED_MODEL_PARENT/cache}"
TAO_CFW_IMAGE_DIGEST=$(docker image inspect --format '{{.Id}}' "$TAO_CFW_IMAGE")
: "${TAO_CFW_IMAGE_DIGEST:?could not resolve the selected runtime image ID}"

"$PYTHON" \
  "$TAO_SKILL_BANK_PATH/skills/models/tao-finetune-cosmos-reason/scripts/prepare_cosmos3_vlm_checkpoint.py" \
  --base-model-path-or-uri "$COSMOS3_SOURCE_DIR" \
  --vlm-architecture-model-path-or-uri "$COSMOS3_VLM_ARCHITECTURE_MODEL" \
  --vlm-architecture-model-revision "$COSMOS3_VLM_ARCHITECTURE_REVISION" \
  --output-path "$PREPARED_MODEL_HOST_PATH" \
  --cache-dir "$COSMOS3_CONVERSION_CACHE_DIR" \
  --backend cosmos-framework \
  --runtime-image "$TAO_CFW_IMAGE" \
  --runtime-image-digest "$TAO_CFW_IMAGE_DIGEST"
```

The helper's `--runtime-image` and `--runtime-image-digest` must identify the
model skill's resolved Framework image. The same immutable image runs checkpoint
preparation, Train, Evaluate, and Inference.

Confirm `$COSMOS3_SOURCE_DIR` exists before launching; that check costs nothing
and saves several minutes plus a large cache write.

The model helper owns its internal Docker command and maps the invoking
UID:GID before writing the prepared checkpoint. Treat any root-owned output as
a helper regression and hard-stop; never launch a root repair container. A
valid existing checkpoint is reported as `status=reused_verified` and reused.

The helper may pull the selected Framework image, download the architecture
checkpoint, and write a large checkpoint, so never run it before approval. It
forwards whichever of `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` is set in the
session, by name and without printing values; the second is the legacy alias
for the same HuggingFace access token, so either one is enough.

Do not convert again when the helper reports `status=reused_verified`; it has
already verified a complete `qwen3_vl` safetensors directory. Mount or stage
that directory for the selected platform, then use its compute-frame path for:

- baseline `model.model_name`;
- Train `model.backbone.model_name` and `model.backbone.safetensors_path`;
- iteration Evaluate/Inference `model.vit_checkpoint_path` when loading DCP.

Keep the canonical Cosmos Reason 3 ID (`nvidia/Cosmos3-Nano` by default) as the
source-model lineage; it names the model, not the prepared PTM path. The
command above is the Nano default. For Edge or Super, supply the selected
source checkpoint plus a model-skill-approved, variant-matched
`--vlm-architecture-model-path-or-uri` and immutable
`--vlm-architecture-model-revision`; if that mapping and the selected runtime
have not been validated, hard-stop instead of applying Nano's Qwen3-VL 8B
default.

## Framework image

Resolve the Framework action image through the model skill and inspect its
repository digest. The same immutable Framework image runs checkpoint
preparation, Train, Evaluate, and Inference.

## Train

Render `references/cosmos_framework_sft_full.toml` with
`"$PYTHON" scripts/render_cfw_sft.py`. The validation profile is single-GPU BF16 VLM
LoRA, one ordered `[AOI, golden_reference]` pair per record, 10 epochs, and
language-side projection targets. The mounted `cfw_cr3_aoi_adapter.py` decodes
both paths and inserts both images before the inspection prompt.
Keep Train `model.attn_implementation="cosmos"` because `sdpa` collapses
Framework LoRA SFT to the majority label on this image. Evaluate remains
`sdpa` because its HF loader rejects `cosmos`.
The native command is:

```bash
cosmos-framework-train --sft-toml=/tao/config/train.toml
```

Iteration 1 leaves `checkpoint.load_path="???"`. Later iterations set it to
the preceding native DCP directory and retain
`checkpoint.keys_to_skip_loading=[]`. Submit through
`"$PYTHON" scripts/submit_cfw_train.py`; record the input SFT TOML, Train's saved
Hydra `config.yaml`, and selected DCP path.

## Evaluate

Render an exact nested TOML with `"$PYTHON" scripts/render_cfw_evaluate.py`. The native
command is:

```bash
cosmos-framework-evaluate --config /tao/config/evaluate.toml
```

The rendered Evaluate profile is single-GPU BF16, batch size 1, one frame per
image, `torchcodec-cuda-on-demand` decoding, temperature 0, and at most four
output tokens.
Proxy and Benchmark annotations remain two-image ShareGPT arrays; the native
Framework evaluator receives the ordered pair for each prompt.
Use the prepared PTM directly for baseline. For an iteration DCP:

- `model.model_name`: DCP directory;
- `model.config_file`: Train's saved Hydra `config.yaml` beside that DCP, never
  the input SFT TOML;
- `model.export_dir`: action-local writable model directory;
- `model.vit_checkpoint_path`: prepared Qwen3-VL PTM directory;
- `model.enable_lora=false`.

The public action handles the DCP locally when it starts. Mount the action
model parent read-write and the rest of the workspace read-only. Submit with
`"$PYTHON" scripts/submit_cfw_evaluate.py`.

## Inference

The native action remains Framework-owned. For standalone one-image use its
public inference command:

```bash
cosmos-framework-inference \
  --model_path MODEL --torch_dtype bfloat16 --device_map auto --num_gpus 1 \
  --type image --media IMAGE --prompt 'Return exactly OK or NG.' \
  --max_new_tokens 4 --results_dir RESULTS --enable_lora false
```

For a DCP add `--config_file`, `--export_dir`, and
`--vit_checkpoint_path` with the same meanings as Evaluate. Use
`"$PYTHON" scripts/submit_cfw_inference.py`.

PCB pair inference uses a one-record two-image annotation through the native
Framework Evaluate action so the AOI and golden reference reach the processor
together. Do not discard the golden image or invoke a different backend.

## Container invariants

Use the job-record id as the Docker name and label. Run detached, poll Docker
through `status`, read Docker output through `logs`, and stop through `cancel`.
Before every action, prove the recorded results directory is writable and
compose the application-owned identity arguments:

```bash
mkdir -p "$RESULTS_DIR"
probe="$RESULTS_DIR/.tao-write-probe.$$"
(umask 077 && : >"$probe" && rm -f "$probe") || {
  echo "FATAL: $RESULTS_DIR is not writable by uid $(id -u)" >&2
  exit 2
}

CR3_IDENTITY_ARGS=(
  --user "$(id -u):$(id -g)"
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" -e HOME=/tmp
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro
)
```

Insert `"${CR3_IDENTITY_ARGS[@]}"` into every Docker Train, Proxy evaluate,
Benchmark evaluate, and Inference launch, including resumed actions. Use a
writable stage working directory, keep user data off `/workspace`, mount the
workspace read-only, and expose only recorded results and action-model
directories read-write.
