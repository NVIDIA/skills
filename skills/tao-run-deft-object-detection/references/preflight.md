# Pre-Flight Checks and Summary

Resolve everything you can before asking the user. Parameter precedence is strict: **explicit user value → workspace spec → documented default.** Never replace an explicit value with a recommendation. Record the winning value and its source in the Summary.

## Checks, in order

1. **Workspace and run directory.** Resolve to an absolute path (`WORKSPACE=$(realpath -m <workspace>)`); never hand a quoted `~/...` path to Python. Derive `RESULTS_DIR=${WORKSPACE}/results/run_$(date +%Y%m%d_%H%M%S)`. Do **not** create it before the user gate. If resuming, set `RESULTS_DIR` to the existing run directory (detect via `results/run_*/deft_state.json`).

2. **Host Python.** Probe with the bundled launcher:

   ```bash
   <skill_root>/scripts/deft_python.sh -c "import pandas,numpy,matplotlib,pyarrow,PIL,yaml"
   ```

   **All six.** `deft_python.sh` selects an interpreter only when *every* one of
   `pandas numpy matplotlib pyarrow PIL yaml` imports; a host missing just `matplotlib`
   makes it exit 2 with no interpreter at all, and every bundled script becomes unrunnable.
   Probing a shorter list will pass here and then fail at the first real call. If the probe
   fails, the host is not provisioned. Installing packages is out of scope for this skill —
   `deft_python.sh` never installs. Report what is missing and stop.

   `deft_python.sh` finds `$WORKSPACE/.venv` only when `WORKSPACE` (or `WORKSPACE_DIR`) is
   exported in the calling shell — otherwise pass the interpreter as `DEFT_PYTHON`.

   `deft_python.sh` auto-selects `$WORKSPACE/.venv/bin/python` once it exists. Pre-Flight is incomplete until the probe exits zero.

3. **Credentials.** Check presence only; never print a value.

   ```bash
   for var in NGC_KEY HF_TOKEN; do
     [ -n "${!var:-}" ] && printf '%s SET\n' "$var" || printf '%s UNSET\n' "$var"
   done
   ```

   | Variable | Required for |
   |---|---|
   | `NGC_KEY` | **Conditionally** — only when an `nvcr.io` image has to be pulled, or the Grounding DINO checkpoint downloaded. Both images already local and an `ngc` CLI holding its own credentials is a complete run with this unset. See check 5. |
   | `HF_TOKEN` | **Conditionally** — only when the encoder resolves to a HuggingFace id rather than a local snapshot directory. See check 9. |

   Defer both verdicts. Check 5 sets `WILL_PULL_AFTER_APPROVAL`, and only then does an unset `NGC_KEY` stop the run; check 9 resolves the encoder, and a local snapshot needs no HuggingFace access at all. Stopping a run whose inputs are all present is the failure to avoid here.

   `fetch_gdino_checkpoint.py` shells out to the `ngc` CLI, which is not always on `PATH` — check for it and add its directory before deciding a checkpoint cannot be fetched.

4. **Resolve and export the version-managed container image env vars.** The rest of this skill — the Pre-Flight Summary's `docker image inspect` line, every stage launch, and every `references/*.md` — reads two env vars. Resolve both from the installed skill bank's `versions.yaml`; never copy a tag into this document or carry one over from an earlier run.

   ```bash
   TAO_PYT_IMAGE=$(
     <skill_root>/scripts/deft_python.sh \
       <skill_bank>/scripts/resolve_versions_key.py images.tao_toolkit.pyt
   )
   TAO_DS_IMAGE=$(
     <skill_root>/scripts/deft_python.sh \
       <skill_bank>/scripts/resolve_versions_key.py images.tao_toolkit.data_services
   )
   : "${TAO_PYT_IMAGE:?versions key images.tao_toolkit.pyt did not resolve}"
   : "${TAO_DS_IMAGE:?versions key images.tao_toolkit.data_services did not resolve}"
   export TAO_PYT_IMAGE TAO_DS_IMAGE
   ```

   `resolve_versions_key.py` reads the `versions.yaml` of the bank it is part of, so this works from a git clone and from a plugin install alike. Set `TAO_SKILL_BANK_PATH` only to point at a *different* bank than the one the script lives in. The `:?` guards matter: an unset image variable makes `docker image inspect` fail in a way that reads as a missing image rather than a missing resolution.

   | Env var | versions-key | Used by |
   |---|---|---|
   | `TAO_PYT_IMAGE` | `images.tao_toolkit.pyt` | `train`, `inference` |
   | `TAO_DS_IMAGE` | `images.tao_toolkit.data_services` | `gap_analysis`, `embed`, `mine`, `kpi_analyze` |

   `versions.yaml` is the single place either URI is written, and step 4 resolves both from
   it, so a bump lands in one file and no document can drift from it.

5. **Image presence.** `docker image inspect "$TAO_PYT_IMAGE" "$TAO_DS_IMAGE"`. Record anything missing as `WILL_PULL_AFTER_APPROVAL`; do not pull before the gate.

6. **Zero-shot checkpoint — pull it from NGC unless the user supplied one.** The baseline
   scores this checkpoint without training and every iteration fine-tunes from it.

   **The user's own path always wins.** When they did not give one, fetch the published
   checkpoint rather than asking:

   Pre-Flight resolves the path only. `--plan` reports whether the checkpoint is
   already present or would be downloaded, and writes nothing — the download is a
   side effect and belongs after the gate, with the container pulls:

   ```bash
   # --plan reports what it would fetch; it prints a sentence, not a path.
   <skill_root>/scripts/deft_python.sh \
     <skill_root>/scripts/fetch_gdino_checkpoint.py --plan \
     --dest "$WORKSPACE/checkpoints/gdino"

   # After approval, the same command without --plan downloads and prints the path.
   ZERO_SHOT_CHECKPOINT=$(<skill_root>/scripts/deft_python.sh \
     <skill_root>/scripts/fetch_gdino_checkpoint.py \
     --dest "$WORKSPACE/checkpoints/gdino")
   ```

   Capture the variable from the real run, not from `--plan`: under `--plan` the script
   prints `WILL DOWNLOAD after approval: ... -> <dir>`, so capturing it puts an English
   sentence in a variable every later stage uses as a path.

   That resolves `nvidia/tao/grounding_dino:grounding_dino_swin_tiny_commercial_trainable_v1.1`
   (1.93 GB) and prints the checkpoint path on stdout. It is idempotent — an existing
   download is reused, so a resumed run re-costs nothing. Use a **`trainable`** release; the
   sibling `deployable` one is for TensorRT export and cannot be fine-tuned.

   Requires the `ngc` CLI and a configured account (check 3). On an air-gapped host, or any
   time the download cannot run, the user supplies `--zero-shot-checkpoint` directly — the
   script says so rather than failing obscurely.

   Record in the Summary which source won (`user` or `NGC <version>`), and hard-stop if the
   resolved path does not exist.

   Then confirm the spec matches the checkpoint's architecture. `model.backbone`,
   `num_queries`, `enc_layers`, `dec_layers`, `num_feature_levels`, and especially
   `class_embed_bias` must all agree, or the run dies at load time — after the container has
   started, with an error that reads like a bad checkpoint rather than a bad spec.

   `class_embed_bias` is the one that actually bites: it defaults to `False`, is absent from
   the shipped `infer.yaml` template, and a checkpoint trained with it `True` fails with
   *only* `class_embed.*.bias` keys reported unexpected. Read the values out of the
   checkpoint and set them explicitly — see `references/grounding-dino.md`.

   The checkpoint is coupled to the container, not just the spec. Record which TAO PyTorch
   image the checkpoint was trained with in the Summary; the pinned image is not
   automatically the right one.

7. **Train-spec template.** Must exist and parse as YAML, and `dataset.train_data_sources` must be a **list** (Grounding DINO ODVG shape). A mapping there means the spec is COCO-shaped and this workflow cannot append to it.

   **Seed training data is optional.** Unlike the AOI loop — where ChangeNet must learn the task from a mandatory seed set — Grounding DINO is zero-shot capable and can start cold. Inspect the list and branch:

   - **Non-empty** → validate every entry's `image_dir`, `json_file`, and `label_map` resolve on disk. Report the source count and total ODVG record count in the Summary. Iteration 1 appends to what is already there.
   - **Empty or absent** → note in the Summary that iteration 1 trains on mined data alone, and that the first iteration's dataset will be small. Not an error.

   Either way the source pool (check 8) stays mandatory — without it there is nothing to mine and the loop cannot add data at all.

8. **Source pool.** Two artifacts, both required:
   - `source_pool_embeddings` parquet — must be non-empty and carry `filepath` and `embedding`.
   - `source_pool_annotations` — an ODVG tree containing `*.jsonl` records keyed by `file_name`, and ideally a `*labelmap.json`. Staging synthesizes a labelmap from observed categories when none is found, but an explicit one is preferred.

   Hard-stop if either is missing or the parquet has zero rows. **The loop consumes a
   prepared pool; it does not build one.** Preparing the pool is its own run, completed
   before the loop launches — see `references/prep-source-pool.md`. `init_deft_state.py`
   refuses to write state without these paths, because the alternative is discovering the
   corpus is absent at `mine`, six stages and a training run later.

   Pass `--pool-report` with the prep run's `pool_report.json`. It records which classes the

   A pool prepared elsewhere may not carry `pool_report.json`. Do not re-label to get
   one — generate it from the pool's COCO with `validate_pool_coco.py --record`
   (`references/prep-source-pool.md`).
   pool actually holds annotations for, and init cross-checks that against the target classes.
   A pool prepared for a different class set does not make mining fail — it makes mining
   return neighbours of something else, and the affected class simply never improves.

9. **Resolve the encoder — local snapshot first, never an implicit online default.**

   The encoder that embeds each iteration's weak images must be the *same* one that produced the source-pool parquet. A mismatch is silent: mining succeeds and returns confidently wrong neighbours. Record the resolved values as `config.embedding_model` / `config.embedding_model_path` and reuse them verbatim on every iteration.

   Resolution order — keep an already-set `SIGLIP_MODEL_PATH` only when it is a directory containing `config.json`; otherwise search the known cache roots:

   ```bash
   resolved_siglip=""
   if [ -n "${SIGLIP_MODEL_PATH:-}" ] && [ -f "$SIGLIP_MODEL_PATH/config.json" ]; then
     resolved_siglip="$SIGLIP_MODEL_PATH"
   else
     for cache_root in \
       "${HF_HOME:+$HF_HOME/hub}" \
       "${HUGGINGFACE_HUB_CACHE:-}" \
       "$HOME/.cache/huggingface/hub" \
       "$WORKSPACE/source_pool/hf_cache/hub" \
       "$WORKSPACE/source_pool/hf_cache"; do
       [ -n "$cache_root" ] || continue
       for snapshot in "$cache_root/models--google--siglip-base-patch16-224/snapshots/"*; do
         if [ -f "$snapshot/config.json" ]; then
           resolved_siglip="$snapshot"
           break 2
         fi
       done
     done
   fi
   [ -n "$resolved_siglip" ] && export SIGLIP_MODEL_PATH="$(realpath "$resolved_siglip")"
   # init takes it as --embedding-model-path; bind it here or the resolution is lost.
   export EMBEDDING_MODEL_PATH="${SIGLIP_MODEL_PATH:-}"
   ```

   Rules:

   - Select a snapshot **only** from `models--google--siglip-base-patch16-224/snapshots/*` and verify its `config.json`. Never substitute a DINO, C-RADIO, or other model that happens to be cached — it will embed successfully and produce garbage matches.
   - `HF_HOME` may point outside the workspace, so do not limit the search to the workspace tree.
   - Fall back to the bare HuggingFace id `google/siglip-base-patch16-224` **only after** outbound HuggingFace access has been verified, and only then require `HF_TOKEN`. Never let the embed stage pick an online default implicitly.
   - If no local snapshot resolves and outbound access cannot be verified, hard-stop. Report the cache roots searched.

   Record the resolved snapshot path (or the verified HF id) in the Summary. If the user cannot say which encoder produced the source pool, surface that as an explicit risk row rather than assuming SigLIP.

10. **KPI inputs.** Image directory, ground-truth KITTI label directory, and class-mapping YAML must all exist. `image_dir` must not end in `/` — `kpi_analyze` derives its `Sequence Name` from the second-to-last path component.

11. **Class thresholds and mining config.** Per-class AP50 thresholds and the mining `multiplier` both have reference defaults — do not interrogate the user for them. Omitting `--ap50-thresholds-json` gates each target class at the reference ITS value (`car 0.99`, `bicycle 0.7`, `person 0.7`) and any other target class at `0.7`; `--multiplier` defaults to `3`. Surface the defaulted values in the Pre-Flight Summary so the user can override them, and treat a class gated by assumption as worth flagging: too loose a gate marks no image weak and the iteration mines nothing. If rare classes are configured, also require `source_detection_file` and `target_detection_file` as **COCO JSONs** — `class_stratified` mining needs them and TAO DS will not infer the format.

12. **GPU count.**

    ```bash
    if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --list-gpus | wc -l
    else docker run --rm --gpus all "$TAO_PYT_IMAGE" python3 -c 'import torch; print(torch.cuda.device_count())'
    fi
    ```

13. **Spec sanity.** `train.checkpoint_interval` must be `<= train.num_epochs`. `prepare_spec_for_train.py` lowers it automatically when an explicit epoch override would violate this, but flag the adjustment in the Summary so it is not a surprise.

**`max_iterations` defaults to `1`.** Confirm it with the user when they have not said how many iterations they want, but do not block on it: an unattended run takes the default.

## Defaults

- `train.num_epochs` — from the train-spec template
- `train.optim.lr` — from the train-spec template
- `multiplier` — `3`
- `max_iterations` — `1`
- `allocation_policy` — `class_stratified` when rare classes are given, else `global`
- `rare_class_list` — every target class holding a below-mean share of the pool's
  annotations. It is a property of the pool, so on a run that still has to prep it is
  left unset at init and the prep commit derives it from `pool_report.json`. Supply
  `--rare-class-list` only to override that.
- `distance_metric` — `euclidean`
- `candidate_expansion_factor` — `5`
- `embedding_model` — `SigLIP`; `embedding_model_path` is **resolved**, not defaulted (check 9)
- `iou_threshold` — `0.5`
- `kpi.conf_threshold` — `0.0` (frozen as `config.kpi_conf_threshold`; every phase is scored at it)
- workspace root — user prompt, else `~/workspace`

## Container identity

Every launch in this skill passes `--user "$(id -u):$(id -g)" $DOCKER_IDENTITY`. A uid that has no
`/etc/passwd` entry inside the image — which is every uid but 1000 on the TAO images —
then has no name, and `getpass.getuser()` falls back to `pwd.getpwuid()` and raises.
torch calls it while setting up the inductor cache directory, so the container dies at
import with `KeyError: 'getpwuid(): uid not found: <uid>'` before the requested action
starts. `$HOME` is unset for the same reason and defaults to `/`, which is not writable,
so matplotlib and cupy fail on their cache directories.

Export the identity beside the mounts and pass it to every launch:

```bash
DEFT_HOME="$WORKSPACE/.deft_home"
mkdir -p "$DEFT_HOME"
DOCKER_IDENTITY="-e USER=deft -e HOME=$DEFT_HOME"
export DOCKER_IDENTITY
```

Both are required, and neither is sufficient alone. `USER` satisfies
`getpass.getuser()`, which reads it before consulting `pwd`; any non-empty value works,
and a literal keeps container logs identical whoever runs the workflow. `HOME` defaults
to `/`, which is not writable, so without it matplotlib and cupy fail on their cache
directories even once the uid has a name.

`HOME` points into the workspace rather than at `/tmp` so the caches are owned by the
caller and removed with the workspace.

This is invisible on a uid-1000 workstation and fires on every shared or cloud host, so a
run that works locally proves nothing about it. Dropping `--user` is not the alternative:
see `references/pipeline-and-state.md`.

## Container mounts

Export them once, right after init, and use the variable in every launch:

```bash
EXTRA_MOUNTS=$(<skill_root>/scripts/deft_python.sh -c 'import json,sys
m=json.load(open(sys.argv[1]))["config"].get("extra_container_mounts") or []
print(" ".join(f"-v {p}:{p}" for p in m))' "${RESULTS_DIR}/deft_state.json") || exit 1
export EXTRA_MOUNTS
echo "EXTRA_MOUNTS=$EXTRA_MOUNTS"
```

Echo it and read the result. A command substitution that fails leaves `EXTRA_MOUNTS`
empty and every later launch still runs, mounting nothing extra — the same silent
failure this block exists to prevent.

`init_deft_state.py` derives this list from the run's own inputs — the checkpoints, the
KPI images and labels, the class mapping, the encoder repo, the pool, and both COCO
detection files. A path that none of those name is invisible to it: pass
`--extra-mount PATH` (repeatable) at init and it joins the list. Files are mounted by
their parent directory.

Every `docker run` in this skill is written as
`-v "$WORKSPACE:$WORKSPACE" $EXTRA_MOUNTS -w "$WORKSPACE"`. With inputs inside the
workspace it expands to nothing and the command is unchanged; with KPI data or
checkpoints outside it, it is the difference between a working stage and
`No .txt label files found`.

A HuggingFace snapshot directory is a tree of symlinks into a sibling `blobs/`, so the
mount must be the **repo root** (`.../models--org--name/`), not the snapshot. Mounting the
snapshot alone makes the loader report a missing model file for a model that is present.
`init_deft_state.py` handles this when it derives the list.

`init_deft_state.py` records `config.extra_container_mounts`: the directories that
inputs live in outside `$WORKSPACE`. Containers see only `"$WORKSPACE:$WORKSPACE"`,
so **every** `docker run` in this skill must add a `-v "$m:$m"` for each entry, in
addition to the workspace mount. KPI images, ground truth and checkpoints commonly
sit outside the workspace, and a container cannot read what is not mounted.

## Pre-Flight Summary

Print this and **STOP — wait for explicit approval.** This is the only user gate.

```
## DEFT OD Loop — Pre-Flight Summary

### Run config
| Field                  | Value                                          | Source            |
| ---------------------- | ---------------------------------------------- | ----------------- |
| Model                  | Grounding DINO (ODVG)                          | workflow          |
| Max iterations         | N                                              | user              |
| Stop condition         | max_iterations reached, or zero weak images.
                          mAP is reported, not gated — no target.         | workflow          |
| Epochs / LR            | N / X                                          | user/spec         |
| Encoder                | <model> @ <resolved snapshot or verified HF id>| resolved          |
| Allocation policy      | class_stratified / global                      | user/default      |
| Rare classes           | <list or none>                                 | user              |
| Mining multiplier      | N (budget = iter1 weak count x N)              | user/default      |
| AP50 thresholds        | {"car": 0.99, ...}                             | user/default      |
| GPUs                   | N                                              | detected          |
| Resuming               | yes — iter N complete / no                     | disk              |

### Inputs
| Field                     | Value                                        |
| ------------------------- | -------------------------------------------- |
| Zero-shot checkpoint      | <path>                                       |
| Train spec template       | <path> (N base source(s); 0 = mined-only)    |
| Source pool embeddings    | <path> (N rows, encoder: <model>)            |
| Source pool annotations   | <path> (N jsonl, labelmap: found/synthesized)|
| KPI images                | <path>                                       |
| KPI ground truth          | <path> (N label files)                       |
| Class mapping             | <path>                                       |

### Docker images
| Env var         | Image              | Status     |
| --------------- | ------------------ | ---------- |
| `TAO_PYT_IMAGE` | `<$TAO_PYT_IMAGE>` | OK/MISSING |
| `TAO_DS_IMAGE`  | `<$TAO_DS_IMAGE>`  | OK/MISSING |

### Per-iteration stages
gap_analysis -> embed -> mine -> stage -> train -> inference -> kpi_analyze
(baseline runs inference -> kpi_analyze only; no training)
```

Remind the user to enable auto-mode (shift+tab) before approving — the post-gate loop is continuously side-effecting.

## Run variables

Every stage reference launches containers with shell variables. They are set once here
and used unchanged for the rest of the run; nothing later re-derives them. Export them
before the first stage:

```bash
# Resolved at the gate above
export WORKSPACE RESULTS_DIR MAX_ITERATIONS NUM_GPUS NUM_EPOCHS LEARNING_RATE
export MULTIPLIER ALLOCATION_POLICY AP50_THRESHOLDS_JSON
export ZERO_SHOT_CHECKPOINT TRAIN_SPEC_TEMPLATE
export SOURCE_POOL_EMBEDDINGS SOURCE_POOL_ANNOTATIONS
export EMBEDDING_MODEL EMBEDDING_MODEL_PATH
export KPI_IMAGES_DIR GROUND_TRUTH_LABELS_DIR CLASS_MAPPING
export TAO_PYT_IMAGE TAO_DS_IMAGE EXTRA_MOUNTS DOCKER_IDENTITY

# Prep runs only
export POOL_IMAGES CODETR_CHECKPOINT CODETR_CLASSMAP CLASSES_YAML

# Read by the init block below; unset ones simply drop out of it
export TARGET_CLASSES KPI_CONF_THRESHOLD
export SOURCE_DETECTION_FILE TARGET_DETECTION_FILE   # class_stratified only
export POOL_REPORT RARE_CLASS_LIST                   # an already-prepared pool only
export EXTRA_MOUNTS_IN                               # space-separated extra mounts, if any
```

Two more are set per iteration rather than once, and every stage reference uses them:

```bash
N=<current iteration number>       # 1, 2, ... — the loop counter; paths read iter${N}
PHASE=<baseline|iter${N}>          # which phase's output tree this stage writes to
```

`N` is the iteration being run. `PHASE` is `baseline` for the zero-shot pass and
`iter${N}` inside the loop; the two differ only because the baseline is not an
iteration. Re-export both at the top of each iteration.

An unset variable is not an error in a `docker run`: it expands to empty, so
`-e ""` reaches TAO as a blank spec path and the failure reads as a spec problem rather
than a missing export. Set them all, and prefer `set -u` in the shell driving the run so
a typo fails at the point of use.

Each stage's own spec path is bound in that stage's reference, next to the path it
documents — `$CODETR_SPEC`, `$INFER_SPEC`, `$OD_GAP_SPEC`, `$EMBED_SPEC`, `$MINE_SPEC`
and `$KPI_SPEC`.

## Immediately After Approval

Perform the planned pulls and directory creation, then initialize state once:
**`--pool-report` is the exception: omit it on a prep run.** Unlike the pool artifacts
and `--source-detection-file`, a `--pool-report` path that does not exist yet is a hard
error, not a warning — validate_pool_coco.py writes it during prep, so pass the flag
only when the pool already exists. Omitting it is what downgrades the requirement.

**Order:** `init_deft_state.py` runs first and `prep` is the run's first committed
stage. On a pool that still needs preparing, the pool artifacts and
`--source-detection-file` do not exist yet — pass them anyway, as the paths prep
*will* write, together with prep's inputs (`--pool-images`, `--codetr-checkpoint`,
`--codetr-classmap`). Init reports them as warnings rather than errors, because prep
produces them. Supplying neither the artifacts nor prep's inputs is still an error:
nothing would create them.


Three groups of flags apply only to some runs. Build them first so the command below
stays one shape:

```bash
# class_stratified only. Both are mandatory under that policy — init refuses without
# them, so a run that omits them here fails at `mine`, several stages later.
STRATIFIED_FLAGS=""
if [ "$ALLOCATION_POLICY" = class_stratified ]; then
  STRATIFIED_FLAGS="--source-detection-file $SOURCE_DETECTION_FILE"
  STRATIFIED_FLAGS="$STRATIFIED_FLAGS --target-detection-file $TARGET_DETECTION_FILE"
fi

# Prep runs only: the pool does not exist yet and these are what builds it.
PREP_FLAGS=""
if [ -n "${POOL_IMAGES:-}" ]; then
  PREP_FLAGS="--pool-images $POOL_IMAGES --codetr-checkpoint $CODETR_CHECKPOINT"
  PREP_FLAGS="$PREP_FLAGS --codetr-classmap $CODETR_CLASSMAP"
fi

# A pool that already exists must declare what it holds. Under class_stratified both
# of these are then mandatory: prep is what would otherwise produce the report and
# derive the rare classes from it, and a prep that is not going to run cannot.
EXISTING_POOL_FLAGS=""
if [ -z "${POOL_IMAGES:-}" ] && [ "$ALLOCATION_POLICY" = class_stratified ]; then
  EXISTING_POOL_FLAGS="--pool-report $POOL_REPORT --rare-class-list $RARE_CLASS_LIST"
fi

# Any path the run's own inputs do not name. Repeatable; omit when there are none.
EXTRA_MOUNT_FLAGS=""
for path in ${EXTRA_MOUNTS_IN:-}; do
  EXTRA_MOUNT_FLAGS="$EXTRA_MOUNT_FLAGS --extra-mount $path"
done
```

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/init_deft_state.py \
  --workspace "$WORKSPACE" \
  --results-dir "$RESULTS_DIR" \
  --max-iterations "$MAX_ITERATIONS" \
  --num-gpus "$NUM_GPUS" \
  --num-epochs "$NUM_EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --zero-shot-checkpoint "$ZERO_SHOT_CHECKPOINT" \
  --train-spec-template "$TRAIN_SPEC_TEMPLATE" \
  --source-pool-embeddings "$SOURCE_POOL_EMBEDDINGS" \
  --source-pool-annotations "$SOURCE_POOL_ANNOTATIONS" \
  --embedding-model "$EMBEDDING_MODEL" \
  --embedding-model-path "$EMBEDDING_MODEL_PATH" \
  --kpi-images-dir "$KPI_IMAGES_DIR" \
  --ground-truth-labels-dir "$GROUND_TRUTH_LABELS_DIR" \
  --class-mapping "$CLASS_MAPPING" \
  --ap50-thresholds-json "$AP50_THRESHOLDS_JSON" \
  --multiplier "$MULTIPLIER" \
  --allocation-policy "$ALLOCATION_POLICY" \
  --target-classes "$TARGET_CLASSES" \
  --kpi-conf-threshold "$KPI_CONF_THRESHOLD" \
  $STRATIFIED_FLAGS $PREP_FLAGS $EXISTING_POOL_FLAGS $EXTRA_MOUNT_FLAGS

<skill_root>/scripts/deft_python.sh <skill_root>/scripts/audit_deft_run.py \
  --results-dir "$RESULTS_DIR"
```

Then copy the train-spec template to `${RESULTS_DIR}/train_grounding_dino.yaml` and begin the baseline.
