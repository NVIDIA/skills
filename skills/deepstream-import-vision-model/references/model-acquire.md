
# NV Model Acquire — Steps 1-3

Acquire an ONNX model from Hugging Face, creating the mandatory model folder structure.

## Intake — choose the model

Before Step 1, present two explicit choices:

1. **Default model (recommended):** `PekingU/rtdetr_r50vd` from Hugging Face.
2. **Custom object-detection model:** collect a Hugging Face ID/URL or a versioned NVIDIA NGC
   catalog URL.

Do not replace this with only an open-ended source prompt. If Default is selected, set
`INPUT="PekingU/rtdetr_r50vd"`. If Custom is selected, require the source before continuing.
Dry runs show the choices but perform no browsing, downloads, Docker launches, or file writes.

## MANDATORY: Model Folder Structure

Create this layout at the start of Step 2 (once `$MODEL_NAME` is set by Step 1):
```
models/{model_name}/
  model/       config/       parser/       scripts/
  benchmarks/engines/
  reports/charts/      samples/
```
```bash
mkdir -p models/$MODEL_NAME/{model,parser,config,scripts,benchmarks/engines,reports/charts,samples}
```
Temporary staging dirs (`hf_model/`, `ngc_download/`, `build/`) are created inline where needed and cleaned up afterward — they are NOT part of this structure.

## Step 1: Parse the Model Source URL

Accept a model URL or ID in one of these formats and extract the required fields:

```bash
[ -z "$ARGUMENTS" ] && { echo "ERROR: No model URL or ID provided. Usage: /deepstream-import-vision-model <url>"; exit 1; }
INPUT="${ARGUMENTS}"

if echo "$INPUT" | grep -q "catalog.ngc.nvidia.com"; then
  # NGC catalog URL
  # e.g. https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/trafficcamnet_transformer_lite/files?version=deployable_resnet50_v2.0
  MODEL_SOURCE="ngc"
  NGC_ORG=$(echo "$INPUT"    | sed 's|.*/orgs/\([^/]*\)/.*|\1|')
  NGC_TEAM=$(echo "$INPUT"   | sed 's|.*/teams/\([^/]*\)/.*|\1|')
  MODEL_NAME=$(echo "$INPUT" | sed 's|.*/models/\([^/]*\)/.*|\1|')
  NGC_VERSION=$(echo "$INPUT" | sed 's|.*version=\([^&]*\).*|\1|')
  echo "Source: NGC  Org: $NGC_ORG  Team: $NGC_TEAM  Model: $MODEL_NAME  Version: $NGC_VERSION"
else
  # HuggingFace full URL or short ID (e.g. https://huggingface.co/onnx-community/yolov8n or onnx-community/yolov8n)
  MODEL_SOURCE="hf"
  SLUG=$(echo "$INPUT" | sed 's|https://huggingface.co/||' | sed 's|/resolve/.*||' | sed 's|/$||')
  HF_ORG=$(echo "$SLUG"    | cut -d/ -f1)
  MODEL_NAME=$(echo "$SLUG" | cut -d/ -f2)
  echo "Source: HF  Org: $HF_ORG  Model: $MODEL_NAME"
fi
```

- `MODEL_SOURCE` (`hf` or `ngc`) drives category selection in Step 2
- `MODEL_NAME` is used as the folder name throughout (`models/{MODEL_NAME}/`)
- Proceed to Step 2 with these variables set

## Step 2: Detect Model Source and Format

First, create the model directory structure (required for all sources), then route by source:
```bash
# Create permanent model directory structure (all sources — HF and NGC)
mkdir -p models/$MODEL_NAME/{model,parser,config,scripts,benchmarks/engines,reports/charts,samples}

# Route based on MODEL_SOURCE set in Step 1
if [ "$MODEL_SOURCE" = "ngc" ]; then
  echo "NGC model detected — skipping HF repo browse, proceeding to Step 2d"
  # Skip to Step 2d directly — do not run any HF curl commands below
fi
# The following HF browse, config download, and labels extraction only runs for MODEL_SOURCE=hf
```

- Browse the HF repository and classify available model files using the vetted helper script
  (validates inputs, uses HTTPS+TLSv1.2 only, honors `$HF_TOKEN`):
  ```bash
  FILES="$(bash .claude/skills/deepstream-import-vision-model/scripts/model/hf-list-files.sh "$HF_ORG" "$MODEL_NAME")"
  ONNX_FILES=$(echo "$FILES" | grep -E '\.onnx$' || true)
  ST_FILES=$(echo "$FILES" | grep -E '\.(safetensors|bin)$' || true)
  echo "ONNX files:      ${ONNX_FILES:-none}"
  echo "SafeTensors/bin: ${ST_FILES:-none}"
  echo "All files:       $FILES"

  # If ONNX list is empty in root, also check /onnx subdirectory
  if [ -z "$ONNX_FILES" ]; then
      ONNX_SUB="$(bash .claude/skills/deepstream-import-vision-model/scripts/model/hf-list-files.sh "$HF_ORG" "$MODEL_NAME" onnx | grep -E '\.onnx$' || true)"
      echo "ONNX in /onnx subdir: ${ONNX_SUB:-none}"
  fi
  ```
- Classify the repo into one of these categories:

  **Category A: ONNX files available** -> proceed to Step 2a (select ONNX variant)
  **Category B: SafeTensors/PyTorch only (no ONNX)** -> proceed to Step 2b (export to ONNX)
  **Category C: No usable model files** -> inform user, suggest alternative repos
  **Category D: NGC model (not on HuggingFace)** -> proceed to Step 2d (NGC download)

- Download `config.json` — required for architecture detection and label extraction.
  Uses the vetted helper script (validated inputs, HTTPS+TLS, honors `$HF_TOKEN`):
  ```bash
  # HF: download from API via vetted helper. NGC: extracted from archive in Step 2d.
  if [ "$MODEL_SOURCE" = "hf" ]; then
    bash .claude/skills/deepstream-import-vision-model/scripts/model/hf-download-config.sh \
        "$HF_ORG" "$MODEL_NAME" "models/$MODEL_NAME/config/config.json"
  else
    echo "NGC model — config.json will be extracted from the downloaded archive in Step 2d"
  fi
  # Note: models/$MODEL_NAME/config/ already exists from the MANDATORY mkdir at the top of Step 2
  ```
- Inspect `config.json` to identify:
  - Model type (e.g., `grounding-dino`, `detr`, `yolos`, `resnet`, `swin`)
  - Architecture class (e.g., `GroundingDinoForObjectDetection`)
  - Number of inputs (single input vs multi-modal)

- **Reject non-detection architectures (fail fast)**: Check the `architectures` field in `config.json` before continuing. If the architecture class ends in a non-detection suffix such as `ForImageClassification`, `ForSemanticSegmentation`, `ForInstanceSegmentation`, `ForPanopticSegmentation`, `ForDepthEstimation`, `ForMaskedLM`, `ForTokenClassification`, or `ForCausalLM`, **abort the pipeline with a clear error and exit non-zero**: `"deepstream-import-vision-model currently supports object detection models only. Detected architecture: {arch_class}. Classification, segmentation, and other vision tasks are not yet supported."` Do not prompt the user. Detection architectures end in `ForObjectDetection` (or, for some DETR-family variants, `ForConditionalDetection` / `ForZeroShotObjectDetection`).

- **Validate the architecture and extract `labels.txt`** — one shared helper does both, so the
  HF and NGC routes cannot drift apart. Run it as soon as `config.json` is in place (for HF that
  is now; for NGC it runs at the end of Step 2d):
  ```bash
  build/.venv_optimum/bin/python \
    .claude/skills/deepstream-import-vision-model/scripts/model/config-to-labels.py \
    --config models/$MODEL_NAME/config/config.json \
    --labels models/$MODEL_NAME/config/labels.txt
  ```
  It exits non-zero on a non-detection architecture (the fail-fast gate above) and on a missing
  label map. **Treat either as fatal** — do not prompt the user, and never fall back to hardcoded
  COCO, ImageNet, or any other default list.

### Step 2a: Select ONNX Variant (Category A)
- Identify available quantization variants (fp32, fp16, int8, int4, quantized, etc.)
- **Default preference: fp16**. Apply this logic:
  1. If fp16 variant exists -> **select it silently**, log: `"Selected: fp16 (default). All available: [list]"`
  2. If fp16 does NOT exist -> **auto-select deterministically** in this priority order: fp32 > int8 > int4 > quantized > first ONNX alphabetically. Log: `"Selected: {variant} (fp16 unavailable). All available: [list]"`. Do not prompt the user.
  3. If only one ONNX file exists -> log it and proceed without asking
- **Construct the resolved download URL** for the selected variant from the tree listing:
  ```bash
  # The tree API returns entries with a "path" field (relative to repo root)
  # Construct the download URL as:
  PATH_FROM_TREE="<path field from tree listing, e.g. onnx/model_fp16.onnx>"
  ONNX_URL="https://huggingface.co/$HF_ORG/$MODEL_NAME/resolve/main/$PATH_FROM_TREE"
  # Example: path="onnx/model_fp16.onnx" -> URL ends in /resolve/main/onnx/model_fp16.onnx
  # Store this URL for use in Step 3
  ```
- After URL construction, proceed to **Step 3** (download ONNX)

### Step 2b: Export SafeTensors to ONNX (Category B)

When the repo only has `.safetensors` (or `.bin`) files and no ONNX export, convert to ONNX using an **isolated virtual environment** to avoid polluting the host system.

#### 2b-i: Virtual Environment (already built by `setup.sh`, in-container)
- The shared venv at **`build/.venv_optimum`** is created **inside the DeepStream container** by the
  skill's `setup.sh` (one-time bootstrap) — do **not** create it on the host. Every export command runs
  in-container against it; use `PY=build/.venv_optimum/bin/python` and `build/.venv_optimum/bin/optimum-cli`.
- It's a single shared venv across all models (`torch`/`transformers`/`onnxruntime` are heavy
  and identical model-to-model). If it's missing, run the bootstrap (from SKILL.md Pre-flight):
  ```bash
  # in-container (via docker run … --entrypoint bash … setup.sh); NOT on the host
  .claude/skills/deepstream-import-vision-model/setup.sh    # builds build/.venv_optimum + wkhtmltopdf
  ```
- For a new model that needs **extra packages** (e.g. `timm` for DETR-family backbones or `onnxsim`), `pip install` them **into the existing shared venv** rather than creating a new one:
  ```bash
  source build/.venv_optimum/bin/activate
  pip install timm   # or: pip install onnxsim
  ```
- The venv lives under `build/.venv_optimum` at the repo root, keeping `models/` clean and excluded from git via the root `.gitignore`
- All subsequent Python/pip commands in Step 2b must run inside this venv
- Legacy per-model venvs at `build/.venv_$MODEL_NAME` from older runs are still cleaned up by `.claude/skills/deepstream-import-vision-model/scripts/model/cleanup.sh "$MODEL_NAME"` for backward compatibility

#### 2b-ii: Download Required Files
- Download from the HF repo into `models/$MODEL_NAME/hf_model/` using `-P` to avoid changing the working directory:
  ```bash
  mkdir -p models/$MODEL_NAME/hf_model
  HF_BASE="https://huggingface.co/$HF_ORG/$MODEL_NAME/resolve/main"
  # Download model files
  wget -P models/$MODEL_NAME/hf_model "$HF_BASE/model.safetensors"
  wget -P models/$MODEL_NAME/hf_model "$HF_BASE/config.json"
  wget -P models/$MODEL_NAME/hf_model "$HF_BASE/preprocessor_config.json"
  # For text+vision models, also download tokenizer files (failures are non-fatal):
  wget -P models/$MODEL_NAME/hf_model "$HF_BASE/tokenizer.json"         || true
  wget -P models/$MODEL_NAME/hf_model "$HF_BASE/tokenizer_config.json"  || true
  wget -P models/$MODEL_NAME/hf_model "$HF_BASE/vocab.txt"              || true
  wget -P models/$MODEL_NAME/hf_model "$HF_BASE/special_tokens_map.json" || true
  ```
- For sharded models (multiple `.safetensors` files), also download `model.safetensors.index.json` and all shards

#### 2b-iii: SafeTensors -> ONNX Export -- Max 3 Retries

> **optimum was removed from this skill.** It pinned `transformers` below 4.54.0, which blocked the
> releases fixing its RCE advisories (GHSA-29pf-2h5f-8g72, fixed in 5.3.0; GHSA-fgcw-684q-jj6r, fixed
> in 5.5.0), and optimum 2.1.0 dropped the `onnx` subcommand entirely. Export now goes through
> `torch.onnx.export` directly.

- Run the export wrapper (it resolves the shared venv and calls `safetensors_to_onnx.py`):
  ```bash
  bash .claude/skills/deepstream-import-vision-model/scripts/model/safetensors-to-onnx.sh \
    models/$MODEL_NAME/hf_model \
    models/$MODEL_NAME/onnx_export/
  ```
- It writes `models/$MODEL_NAME/onnx_export/model.onnx` and:
  - reads the export resolution from `preprocessor_config.json` (falling back to 640x640);
  - rejects non-detection architectures before doing any work;
  - **tries the dynamo backend, then falls back to TorchScript** if dynamo specialized the batch
    dimension — it prints `[export] backend=...` so you can see which one produced the graph;
  - consolidates any sidecar `model.onnx.data` back into the `.onnx`;
  - **verifies** the graph has the `pixel_values` input plus `logits` / `pred_boxes` outputs, and
    that the batch dimension really is dynamic. It exits non-zero rather than emitting a
    static-batch graph, because DeepStream needs `batch_size == num_streams` up to `MAX_BS`.
- Expected output for the default model — note the automatic fallback:
  ```
  [export] backend=dynamo
  [export] pixel_values shape=[2, 3, 640, 640]
  [export] dynamo produced a static batch dimension; trying the next backend
  [export] backend=legacy-torchscript
  [export] pixel_values shape=['batch', 3, 640, 640]
  ```
- Useful flags (passed straight through): `--opset N` · `--image-size N` · `--max-batch N` ·
  `--static-batch` · `--legacy-torchscript`
- If export succeeds, copy the ONNX file to the `model/` subdirectory:
  ```bash
  cp models/$MODEL_NAME/onnx_export/model.onnx models/$MODEL_NAME/model/$MODEL_NAME.onnx
  ```
- **Retry policy**: the two-backend fallback is automatic, so a failure here means both backends
  failed. Retry up to **3 times total** with adjustments between attempts:
  - **Retry 1**: Set `--image-size` explicitly if the error mentions a shape or resize mismatch
  - **Retry 2**: Try a different `--opset` (18 is the default; 17 and below fail on `Resize`)
  - **Retry 3**: Try `--static-batch` if both backends baked in the batch dimension. The engine
    must then be built at a fixed batch size — see `references/engine-build.md`.
  - After 3 failed attempts, fall back to **Step 2b-iv** (hand-written export for an unusual architecture)

#### 2b-iv: Fallback -- Hand-written torch.onnx.export (unusual architectures) -- Max 3 Retries
- If the wrapper fails after 3 retries because the model does not follow the standard
  `logits` / `pred_boxes` detection contract, write the export inline and adjust the wrapper outputs:
  ```bash
  build/.venv_optimum/bin/python -c "
  import torch
  from transformers import AutoModelForObjectDetection

  model = AutoModelForObjectDetection.from_pretrained('models/$MODEL_NAME/hf_model').eval()

  class Wrap(torch.nn.Module):
      def __init__(s, m): super().__init__(); s.m = m
      def forward(s, x):
          o = s.m(pixel_values=x)
          return o.logits, o.pred_boxes   # <- adjust for the architecture

  # Dummy input matching preprocessor_config.json dimensions
  dummy = torch.randn(1, 3, 800, 800)

  # TorchScript backend — the more reliable one for a dynamic batch dimension on
  # DETR-family detectors. Swap to the dynamo block below if this backend fails.
  torch.onnx.export(Wrap(model), dummy, 'models/$MODEL_NAME/model/$MODEL_NAME.onnx',
    opset_version=18, do_constant_folding=True, dynamo=False,
    input_names=['pixel_values'],
    output_names=['logits', 'pred_boxes'],
    dynamic_axes={'pixel_values': {0: 'batch'},
                  'logits': {0: 'batch'},
                  'pred_boxes': {0: 'batch'}})

  # dynamo backend (use for text-conditioned models the tracer cannot handle):
  #   batch = torch.export.Dim('batch', min=1, max=16)
  #   torch.onnx.export(Wrap(model), (torch.randn(2, 3, 800, 800),), '<out>.onnx',
  #     opset_version=18, dynamo=True,
  #     input_names=['pixel_values'], output_names=['logits', 'pred_boxes'],
  #     dynamic_shapes={'pixel_values': {0: batch}})
  "
  ```
- Adjust the wrapper's returned tensors, input/output names, and shapes for the architecture
- Always confirm the batch dimension survived:
  ```bash
  build/.venv_optimum/bin/python -c "
  import onnx; d = onnx.load('models/$MODEL_NAME/model/$MODEL_NAME.onnx').graph.input[0].type.tensor_type.shape.dim
  print([x.dim_param or x.dim_value for x in d])"   # expect ['batch', 3, H, W]
  ```
- **Retry policy**: If manual export fails, retry up to **3 times total** with adjustments:
  - **Retry 1**: Try a different `AutoModel` class, or return different tensors from the wrapper
  - **Retry 2**: Switch backend (TorchScript <-> dynamo), remembering `dynamic_axes` vs `dynamic_shapes`
  - **Retry 3**: Drop the dynamic batch entirely and build a fixed-batch engine
  - After 3 failed attempts, **stop and generate a failure report**

> **Gotchas for recent PyTorch/transformers** (verified on torch 2.13.0 + transformers 5.14.1
> inside `nvcr.io/nvidia/deepstream:9.1-triton-multiarch`):
> - **Neither backend works for every architecture — try `dynamo=True` first, fall back to `dynamo=False`.** `safetensors_to_onnx.py` does this automatically and reports which backend won.
> - **`dynamo=True` can silently produce a STATIC batch dimension.** The model's own code may specialize `shape[0]`; `torch.export` then reports *"you marked batch as dynamic but your code specialized it to a constant"*. **RT-DETR (`PekingU/rtdetr_r50vd`, the default model) does exactly this** — dynamo yields `[2, 3, 640, 640]`, the TorchScript path yields `['batch', 3, 640, 640]`. Always verify with `onnx.load()`; the exporter asserts it.
> - **`dynamo=False` does NOT crash on transformers 5.5+ for vision detectors.** An earlier revision of this document claimed it did, via `create_bidirectional_mask`. That applies to **text-conditioned** models (Grounding DINO and friends, which run a BERT text encoder) — not to pure-vision detectors. RT-DETR and YOLOS both export cleanly through TorchScript on transformers 5.14.1.
> - Under `dynamo=True` the parameter is `dynamic_shapes` (with `torch.export.Dim`), **not** `dynamic_axes`; under `dynamo=False` it is `dynamic_axes`. Passing the wrong one is silently ineffective.
> - **Use opset 18, not 17.** The dynamo exporter implements >= 18 and auto-upgrades, then its downgrade pass fails outright with `No Adapter To Version 17 for Resize`. Opset 18 is fine on TRT 10.16.
> - **Call `.eval()` on the wrapper module too**, not just the loaded model — a fresh `nn.Module` defaults to training mode, which changes dropout/batchnorm behaviour during export.
> - **External data files**: `torch.onnx.export` may produce `model.onnx.data` alongside the `.onnx`. Consolidate before TRT conversion: `m = onnx.load(path, load_external_data=True); onnx.save(m, consolidated_path)`. The exporter does this automatically.
> - No `ForeignNode` failure was observed for RT-DETR on TRT 10.16 with either backend — see `references/engine-build.md` for the historical issue.

#### 2b-v: Handle Multi-Modal Models (e.g., Grounding DINO)
- Models that take **both image AND text** inputs need special handling for DeepStream (nvinfer only supports image input)
- Strategy: **freeze the text prompt** into the ONNX graph as a constant
  1. Run the model once with a fixed text prompt (e.g., "person . car . truck .")
  2. Export ONNX with the text embeddings baked in as constants
  3. The resulting ONNX model only needs `pixel_values` as input
- If freezing is not possible, check `onnx-community/` for pre-converted single-input versions
- **Inform the user** about the frozen text prompt and its implications (fixed detection classes)

#### 2b-vi: onnxsim — Run After Export When Needed

If the model has dynamic shape paths that cause TRT `ForeignNode` fusion issues, simplify the ONNX graph with `onnxsim` **before** engine building:

```bash
source build/.venv_optimum/bin/activate
pip install onnxsim
python3 -m onnxsim \
  models/$MODEL_NAME/model/$MODEL_NAME.onnx \
  models/$MODEL_NAME/model/${MODEL_NAME}_sim.onnx
# Use the _sim.onnx for engine building if the original triggers ForeignNode errors
```

Only run `onnxsim` if TRT build fails with `ForeignNode` warnings — it is not needed for most models.

#### 2b-vii: Validate ONNX Output
- After export, validate the ONNX file:
  ```bash
  source build/.venv_optimum/bin/activate
  python3 -c "
  import onnx
  m = onnx.load('models/$MODEL_NAME/model/$MODEL_NAME.onnx')
  onnx.checker.check_model(m)
  print('Inputs:')
  for i in m.graph.input:
    dims = [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]
    print(f'  {i.name}: {dims}')
  print('Outputs:')
  for o in m.graph.output:
    dims = [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]
    print(f'  {o.name}: {dims}')
  print('ONNX validation passed!')
  "
  ```
- Verify:
  - Single image input (no text/mask inputs -- remove if needed)
  - Output shapes match expected detection format
  - Dynamic batch dimension is present

#### 2b-viii: Cleanup
- Deactivate the venv after export is complete:
  ```bash
  deactivate
  ```
- **Keep `build/.venv_optimum` across runs** — it is shared by every SafeTensors → ONNX export and rebuilding it for each model costs minutes and GBs. `cleanup.sh` intentionally does not remove it.
- `cleanup.sh` removes per-model artifacts (`models/$MODEL_NAME/hf_model`, `models/$MODEL_NAME/onnx_export`, and any legacy `build/.venv_$MODEL_NAME` left over from older runs):
  ```bash
  # Validated script; will refuse unsafe paths. Shared .venv_optimum is preserved.
  bash .claude/skills/deepstream-import-vision-model/scripts/model/cleanup.sh "$MODEL_NAME"
  # Preview without removing:
  # bash .claude/skills/deepstream-import-vision-model/scripts/model/cleanup.sh "$MODEL_NAME" --dry-run
  ```
- The ONNX file is now at `models/$MODEL_NAME/model/$MODEL_NAME.onnx` -- proceed to engine building

### Step 2d: NGC Model Download (Category D)

When the model comes from NVIDIA NGC (not HuggingFace), download using the `ngc` CLI if available, or fall back to `wget` for direct file download:

```bash
# Vetted helper: prefers ngc CLI if installed, else falls back to authenticated
# HTTPS+TLS via curl against the public NGC catalog API. All inputs validated
# against ^[A-Za-z0-9._-]+$. See .claude/skills/deepstream-import-vision-model/scripts/model/ngc-download.sh for details.
bash .claude/skills/deepstream-import-vision-model/scripts/model/ngc-download.sh \
    "$NGC_ORG" "$NGC_TEAM" "$MODEL_NAME" "$NGC_VERSION" \
    "models/$MODEL_NAME/ngc_download"

# Inspect downloaded files
echo "Downloaded files:"
ls -lhR models/$MODEL_NAME/ngc_download/
```

- Identify the ONNX file(s) in the downloaded archive (often inside a subdirectory named after the model version)
- If the download contains a `.etlt` or `.engine` file only (TAO encrypted format), check if a plain ONNX is also provided; if not, use the TAO-provided engine directly and skip Step 4 (engine build)
- Copy the ONNX to the model directory:
  ```bash
  NGC_ONNX=$(find models/$MODEL_NAME/ngc_download -name "*.onnx" | head -1)
  cp "$NGC_ONNX" models/$MODEL_NAME/model/$MODEL_NAME.onnx
  echo "ONNX: $NGC_ONNX -> models/$MODEL_NAME/model/$MODEL_NAME.onnx"
  ```
- Extract `config.json` from the archive and build `labels.txt` (same logic as HF path):
  ```bash
  NGC_CONFIG=$(find models/$MODEL_NAME/ngc_download -name "config.json" | head -1)
  if [ -z "$NGC_CONFIG" ]; then
    echo "ERROR: config.json not found in NGC archive — cannot create labels.txt"
    echo "Cannot proceed without a label map — aborting. Provide an NGC archive that contains config.json."
    exit 1
  else
    cp "$NGC_CONFIG" models/$MODEL_NAME/config/config.json
    echo "config.json extracted from: $NGC_CONFIG"

    # Same helper the HF route uses: gates the architecture, then writes labels.txt.
    # The NGC route reaches config.json later than the HF route, so without this a
    # classification model would build an engine and a parser before anything noticed.
    build/.venv_optimum/bin/python \
      .claude/skills/deepstream-import-vision-model/scripts/model/config-to-labels.py \
      --config models/$MODEL_NAME/config/config.json \
      --labels models/$MODEL_NAME/config/labels.txt || exit 1
  fi
  ```

## Step 3: Download the ONNX Model

The model directory structure was already created in the MANDATORY block at the top. Do NOT run `mkdir -p` again here — just download the file:

```bash
wget -O "models/$MODEL_NAME/model/$MODEL_NAME.onnx" "${ONNX_URL}"
```

Where `$ONNX_URL` is the resolved URL constructed at the end of Step 2a (Category A) or derived from the NGC download path (Category D). Categories B and D write the ONNX directly to `models/$MODEL_NAME/model/$MODEL_NAME.onnx` during export/copy — Step 3 only applies to Category A.
- Also download any external data files if the ONNX model references them (files with `.onnx_data` extension or similar)
- Verify the download completed successfully and report file size

## Timing

Record wall-clock time at the start and end of this skill:
```bash
STEP_START=$(date +%s.%N)
# ... all steps ...
STEP_END=$(date +%s.%N)
STEP_DURATION=$(python3 -c "print(round($STEP_END - $STEP_START, 2))")   # bc is not in the container; python3 always is
```

## Output Summary

When complete, print:
```
=== HF Model Acquire Complete ===  [Steps 1-3: ${STEP_DURATION}s]
Model:  $MODEL_NAME
ONNX:   models/$MODEL_NAME/model/$MODEL_NAME.onnx ({size} MB)
Input:  {input_name} {input_shape}
Output: {output_names} {output_shapes}
Labels: {num_classes} classes -> models/$MODEL_NAME/config/labels.txt
Ready for: Steps 4-5 — read references/engine-build.md models/$MODEL_NAME/model/$MODEL_NAME.onnx
```
(`{size}`, `{input_name}`, `{input_shape}`, `{output_names}`, `{output_shapes}`, `{num_classes}` are filled from the ONNX inspection output — all other fields use bash variables.)
