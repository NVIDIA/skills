# Cosmos3 Train and Bare OK/NG Evaluate

## Prepare the Cosmos Reason 3 checkpoint

The model this loop fine-tunes is the published **Cosmos Reason 3** reasoner
(`nvidia/Cosmos3-Nano` by default). Cosmos-RL cannot load that published
directory: it ships in Cosmos3's own native Omni format — recognizable by
`model_type="cosmos3_omni"` and `Cosmos3ForConditionalGeneration` — while the
Cosmos-RL AOI train/evaluate jobs load a Qwen3-VL VLM safetensors PTM.

So a one-time conversion turns the selected Cosmos Reason 3 reasoner into a
Qwen3-VL PTM. The reasoner stays the model's identity and lineage throughout;
the Qwen3-VL PTM is only the on-disk format Cosmos-RL can consume.

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
COSMOS_RL_IMAGE_DIGEST=$(docker image inspect --format '{{.Id}}' "$COSMOS_RL_IMAGE")
: "${COSMOS_RL_IMAGE_DIGEST:?could not resolve the selected runtime image ID}"

"$PYTHON" \
  "$TAO_SKILL_BANK_PATH/skills/models/tao-finetune-cosmos-reason/scripts/prepare_cosmos3_vlm_checkpoint.py" \
  --base-model-path-or-uri "$COSMOS3_SOURCE_DIR" \
  --vlm-architecture-model-path-or-uri "$COSMOS3_VLM_ARCHITECTURE_MODEL" \
  --vlm-architecture-model-revision "$COSMOS3_VLM_ARCHITECTURE_REVISION" \
  --output-path "$PREPARED_MODEL_HOST_PATH" \
  --cache-dir "$COSMOS3_CONVERSION_CACHE_DIR" \
  --runtime-image "$COSMOS_RL_IMAGE" \
  --runtime-image-digest "$COSMOS_RL_IMAGE_DIGEST"
```

Confirm `$COSMOS3_SOURCE_DIR` exists before launching; that check costs nothing
and saves several minutes plus a large cache write.

The model helper owns its internal Docker command and maps the invoking
UID:GID before writing the prepared checkpoint. Treat any root-owned output as
a helper regression and hard-stop; never launch a root repair container. A
valid existing checkpoint is reported as `status=reused_verified` and reused.

The helper may pull the selected backend image, download the architecture
checkpoint, and write a large checkpoint, so never run it before approval. It
forwards whichever of `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` is set in the
session, by name and without printing values; the second is the legacy alias
for the same HuggingFace access token, so either one is enough.

Do not convert again when the helper reports `status=reused_verified`; it has
already verified a complete `qwen3_vl` safetensors directory. Mount or stage
that directory for the selected platform, then use its compute-frame path for:

- baseline and iteration `model.model_name`;
- Train `policy.model_name_or_path`;
- LoRA evaluate `model.base_model_path`.

Keep the canonical Cosmos Reason 3 ID (`nvidia/Cosmos3-Nano` by default) as the
source-model lineage; it names the model, not the Cosmos-RL PTM path. The
command above is the Nano default. For Edge or Super, supply the selected
source checkpoint plus a model-skill-approved, variant-matched
`--vlm-architecture-model-path-or-uri` and immutable
`--vlm-architecture-model-revision`; if that mapping and the selected runtime
have not been validated, hard-stop instead of applying Nano's Qwen3-VL 8B
default.

## Run containers as the invoking user

Every container that writes into the bound results directory — Train, both
evaluate stages, AnomalyGen — must drop to the host user. Containers default to
root, so their outputs land root-owned inside a directory the operator owns,
and the run tree then cannot be deleted, re-rendered into, or cleaned up
without `sudo`. On Docker:

```bash
CR3_IDENTITY_ARGS=(
  --user "$(id -u):$(id -g)"
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" -e HOME=/tmp
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro
)
```

Insert `"${CR3_IDENTITY_ARGS[@]}"` into every Docker Train, Proxy evaluate,
and Benchmark evaluate launch, including resumed actions, and use the
stage-local writable working directory:

```bash
-w "$RESULTS_DIR/<label>/<stage>/cwd"
```

All four parts are one unit; dropping any of them fails.

`HOME=/tmp` matters: a mapped uid has no home inside the image, and libraries
that write caches to `$HOME` fail or scatter files otherwise.

`-w` matters for the same reason. The image's `WORKDIR` is `/workspace`, owned
by root, and TAO's status decorator falls back to creating `./results` in the
current directory when `TAO_API_JOB_ID` is unset. As root that silently
succeeded; as a mapped uid the first job dies with
`PermissionError: [Errno 13] Permission denied: './results'`. Point `-w` at a
writable directory you create inside the stage's bound results tree.

That leaves a `cwd/results/status.json` stub per stage. It is inert, but treat
`cwd/` as run scaffolding rather than an artifact — never record it in state.
The cleaner alternative, read from the container's `decorators.py`, is to set
`TAO_API_JOB_ID` together with `TAO_API_RESULTS_DIR` pointing into the bound
results tree, which takes the branch that skips `./results` entirely; that path
has not been run end-to-end here, so verify it before relying on it. Other platforms
express the same intent differently — a Kubernetes `securityContext`
`runAsUser`/`runAsGroup`, or SLURM's already-unprivileged execution — so apply
the selected platform's equivalent rather than copying these flags blindly.

## Train producer

Read the current `tao-finetune-cosmos-reason` model skill and
`references/skill_info.yaml`. The action contract is:

- image: resolve the `cosmos-rl` backend image from the Cosmos model skill's
  `references/skill_info.yaml` with
  `"$PYTHON" "$TAO_SKILL_BANK_PATH/scripts/resolve_tao_image.py"`;
- command: `cosmos-rl --config {config_path}
  /opt/cosmos_rl/tao_sft_example.py`;
- mode/format: `config` / TOML;
- output: `train.output_dir`.

Use `nvidia/Cosmos3-Nano` by default. Accept `nvidia/Cosmos3-Edge` or
`nvidia/Cosmos3-Super` only when the user explicitly selects that variant.
Keep the canonical selected ID as lineage across baseline evaluation, Train,
and LoRA evaluation, and use its matching prepared PTM for each job. Treat
checkpoint loading/conversion and compute shape as variant-specific preflight
evidence. Give hardware recommendations consistent with the prompt and
selected variant; report insufficient resources instead of silently
substituting a different variant.

Invoke the model train action with workflow argument `automl_policy: off`, then
dispatch its bundle through the selected platform's four verbs. Build a nested
spec from the model skill's current packaged Train template and apply only the
AOI values below. Do not copy literal dotted override names into TOML.

Required AOI values:

- `custom.train_dataset.annotation_path`: the workspace template initially
  points to `mining_pool.json`; after the baseline gate is shown unmet and
  zero-shot Proxy, RCA, and Mining selection have run, replace it in the staged
  Train spec with `iterN/assemble/train_iter_N.json`;
- `custom.train_dataset.media_path` and `media_root`: compute-frame media root;
- `custom.system_prompt`: exact OK/NG instruction;
- `train.output_dir`: bound stage results directory. A checkpoint outside the
  iteration result tree is a hard stop, so a carried-over template value such
  as `results/train_lora` fails at commit rather than at launch — rewrite it
  when staging, do not inherit it;
- `validation.enable=false` and no `custom.val_dataset`; Proxy is evaluated by
  a separate job and must not become Train validation data;
- `train.train_policy.type="sft"`;
- `policy.model_name_or_path`: prepared Qwen3-VL PTM compute-frame path;
- `policy.model_max_length >= 40960`;
- `policy.lora`: recorded rank/alpha with the recorded
  `target_modules` — by default the language-side projections
  `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
  "down_proj"]`, not `"all-linear"`;
- parallelism values derived from selected platform shape;
- LoRA unless the user explicitly approved full fine-tuning.

Four of these are per-run values that no template can carry, and each is
hard-stop grade if left at its template value. Rewrite them together when
staging the Train spec, then re-read the file before launching:
`custom.train_dataset.annotation_path`, `policy.model_name_or_path`,
`train.output_dir`, and `train.epoch`.

`references/example_lora_config.toml` shows the produced default shape at
iteration N, and `references/example_sft_config.toml` shows the
explicitly-approved full-parameter variant. Read them to review the serialized
result, not as spec sources: build every staged spec from the model skill's
current template so template changes are not silently overridden.

Resolve the concrete exported safetensors checkpoint under the stage output.
Do not record a broken `best` symlink or a launcher staging directory.

The model skill warns that one-epoch runs can leave only a broken `best`
symlink after checkpoint cleanup. That warning is about configurations relying
on `best`. With `train.ckpt.save_freq_in_epoch=1` and
`export_safetensors=true` — this workflow's shape — `epoch=1` was observed to
write a concrete `<output>/<timestamp>/safetensors/epoch_1/` export and create
no `best` symlink at all, so it is safe here. Commit that concrete directory.

## Proxy and Benchmark evaluate producers

Read the current model skill's `evaluate` action contract. Use it for both
evaluation stages:

- command: `cosmos-rl-evaluate --config {config_path}`;
- mode/format: `config` / TOML;
- inputs: `dataset.annotation_path`, `dataset.media_dir`, and
  `model.model_name`;
- output: `results_dir`.

Start from the model skill's current packaged Evaluate template and materialize
it once for Proxy and once for Benchmark. Apply these AOI overrides to both:

- `task.type=""`;
- `dataset.system_prompt`: return exactly `OK` or `NG`;
- `evaluation.answer_type="freeform"` and
  `evaluation.soft_accuracy.enabled=false`;
- `generation.max_tokens=4` and `generation.temperature=0`;
- `metrics.names=[]`;
- save individual results; disable evaluator confusion-matrix and metric
  summary output because `analyze_gaps.py` owns the discrete OK/NG metrics.

Both stages must use the same checkpoint and generation settings; only the
annotation, bound output directory, and save-folder label differ. For a LoRA
export, set `model.enable_lora=true` and keep `model.base_model_path` aligned
with Train's prepared Qwen3-VL PTM.

### Classify and, when needed, lift the evaluation image cap

Some `cosmos-rl` images build vLLM with
`limit_mm_per_prompt={"video": 1, "image": 1}`. Every AOI record carries two
images, so those images fail evaluation until the cap is raised. Other images,
including framework-evaluator builds, contain neither that literal nor a vLLM
engine construction and need no patch. Image tags are not a stable way to tell
the two apart.

Run this once per run, before the first evaluate job:

```bash
"$PYTHON" "$SKILL_ROOT/scripts/patch_eval_image_cap.py" \
  --image "$COSMOS_RL_IMAGE" \
  --output-dir "$RESULTS_DIR/patches/cosmos_rl_eval" \
  --summary "$RESULTS_DIR/patches/cosmos_rl_eval/summary.json"
```

It reads `base.py` out of the selected image and reports one source-driven
classification, independent of whether the tag is semver, a custom build name,
or a future tag:

- `patch_required`: the recognized cap is below two; the script rewrites only
  that literal and prints `MOUNT_ARG=<host>:<container>:ro`. Add it as a
  read-only mount to every `cosmos-rl-evaluate` job.
- `already_sufficient`: the recognized cap is at least two; no file or mount.
- `cap_absent`: the source contains neither the cap nor vLLM engine evidence;
  no file or mount.

If the source still references `limit_mm_per_prompt` or vLLM without the
recognized literal, the script fails with `classification=unknown`; verify the
new evaluator shape instead of guessing. Nothing is vendored into the skill,
and Train jobs never need this mount.

The evaluator writes `results.json` with per-sample `video_id`, `response`,
`question`, and `gt`. Run `analyze_gaps.py` afterward:

- `--evaluation-role proxy` writes per-sample RCCA artifacts and never gates;
- `--evaluation-role benchmark` writes aggregate metrics and the stop-gate
  `metric_result.json`.

The analysis layer normalizes the last standalone `OK`/`NG` token. Source
training labels remain exact.

## Output commits

Baseline begins with zero-shot frozen Benchmark against the base model and
submits no Train job; zero-shot Proxy follows only when that gate is unmet.
After Proxy RCA and Mining selection create `train_iter_N.json`, submit Train,
Benchmark evaluate, and — only when the loop continues — Proxy evaluate, as
separate platform jobs with separate job-records. Commit Train with
`--best-ckpt` and `--training-spec`; commit each evaluation with its
`results.json`. The `benchmark_metrics` commit records the KPI; a stopping
iteration completes there, and a continuing one completes at `proxy_rcca`.
