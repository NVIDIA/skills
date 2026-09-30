# Pre-flight and Initialization

Use this reference for a new run or to resolve an existing run. Discovery is
read-only. Installation, pulls, probes that start containers, extraction, and
run creation occur only after approval.

## Contents

- [Initial intake](#initial-intake)
- [Read-only discovery](#read-only-discovery)
- [Defaults](#defaults)
- [Approval summary](#approval-summary)
- [Approved initialization](#approved-initialization)
- [Resume](#resume)

## Initial intake

PAS supports Docker (a local daemon or an approved remote `DOCKER_HOST`),
SLURM, Kubernetes, Brev, and virtualenv. If the user did not select one,
include that single platform choice with the other missing intake; never
default to Docker. When Docker is selected, resolve whether its compute frame
is local or remote before approval and persist that choice; never infer local
filesystem semantics merely because the platform name is `docker`. Before full
preflight, do only bounded, lightweight discovery that can reduce user
questions:

1. Resolve an explicitly named workspace or the conventional `~/workspace`
   candidate without creating it.
2. Resolve the archive root using this exact precedence and scope:

   - If the user supplied both archive file paths, require the exact names
     `images_raw.tar` and `meta.tar.gz`, a shared non-symlink parent, and two
     non-symlink regular files. Validate only that explicit pair's parent
     location during intake. If it is invalid, report why and ask for a
     corrected pair; do not search for alternatives.
   - If the user supplied an archive directory, treat it as the sole approved
     search root. Inspect that root and directories at candidate depth `1` or
     `2` beneath it. Do not search outside it.
   - Otherwise use, in order, `~/pas` at candidate depth `0..2`, the resolved
     workspace root at depth `0` only, then `<workspace>/pas`,
     `<workspace>/input`, and `<workspace>/inputs` at depth `0..2`. Normalize
     absolute paths and deduplicate overlapping search-root/depth pairs and
     discovered candidate directories.
   - A candidate is a non-symlink directory whose direct children include both
     `images_raw.tar` and `meta.tar.gz` as non-symlink regular files. Use a
     non-symlink-following, depth-limited directory enumeration inside each
     search root, then check the two exact child names; do not run a filename
     search over `$HOME`. Missing roots are reported as checked/missing.
     Permission errors and unsafe symlinks are reported and never cause the
     search to widen.
   - Do not add the current repository, tutorial/notebook checkouts, or
     `<workspace>/data` as implicit search roots. If the user explicitly names
     one of those locations, its bounded subtree is user-authorized and may be
     checked.
   - If no candidate exists in that bounded set, ask for the archive root. Do
     not broaden the search automatically.

   Candidate depth counts directories relative to the search root: the root is
   depth `0`, its direct directory children are depth `1`, and their directory
   children are depth `2`. Do not follow a symlink supplied as a search root or
   encountered below one. This mirrors AOI's useful discovery pattern: recurse
   only below a semantically selected subtree, with an explicit PAS depth cap.

   The archive root is an input location. It is distinct from `DATASET_ROOT`,
   which is the approved extraction/rebuild destination under workspace data.
   The run consumes one archive pair from one root; multiple roots are
   alternatives, not multiple required datasets.

   Classify a candidate from direct-child metadata only:

   - `archive-only`: the required pair is present and extracted-dataset markers
     such as `images_raw/`, `images/`, `captions/`, `rebuild.py`, or
     `train_pairs.json` are absent. Auto-select it only when it is the sole
     candidate.
   - `archive-with-extracted-data`: the pair is present alongside one or more
     extracted-dataset markers. Describe it as a mixed/development location
     and require an explicit user choice unless the user supplied it.

   Record file sizes, but do not open, hash, compare, or validate either large
   archive yet.
3. Enumerate only `<workspace>/results/run_*/deft_state.json`. For each small
   state file, read only enough JSON to identify `workflow`, `schema_version`,
   `results_dir`, `current_iteration`, and `max_iterations`. Present it as a
   resume option only when `workflow` is exactly `tao-run-deft-pas`. Ignore and
   briefly label malformed, unidentified, AOI, or other-workflow states; do not
   offer them as PAS runs. Do not run the full audit yet.
4. Report discovery provenance before asking for missing intake. Use absolute
   paths and this compact shape, omitting sections that do not apply:

   ```text
   PAS DEFT intake (read-only)
     workspace: <absolute path> (source=<user | conventional ~/workspace>)
     archive lookup:
       search root: <absolute path> (reason=<user-supplied | conventional ~/pas | workspace root | workspace child>; candidate_depth=<0 | 0..2>; status=<searched | missing | inaccessible | unsafe-symlink>)
       broad home/repository search: not performed
       candidate: <absolute archive root> (found_under=<search root>; depth=<0..2>; type=<archive-only | archive-with-extracted-data>)
         images: <absolute path>/images_raw.tar (<size>)
         metadata: <absolute path>/meta.tar.gz (<size>)
       selection: <selected sole archive-only root | user choice required | none found>
     resume lookup:
       checked: <workspace>/results/run_*/deft_state.json
       PAS candidates: <none | concise list with iteration/budget>
       ignored: <none | concise path and non-PAS/unreadable reason>
   ```

   If more than one valid archive root remains, say explicitly that they are
   alternatives and ask the user to select one. Never label a list of directory
   names as "dataset archives"; name the archive root and the two files under
   it. Paths found this way come from the current filesystem and must be labeled
   `source=discovery` with the search-root reason, scope, and candidate depth,
   not implied to come from session memory or packaged defaults.
   Never resolve ambiguity by search order, directory name, modification time,
   or archive size.

If `max_iterations` or a time budget is absent, stop here and ask one
consolidated question for that value plus any ambiguous archive/run selection.
State the defaults below and explain that full read-only discovery and the one
approval summary come next. Do not ask whether the user wants a KPI target,
authenticated Hugging Face access, or other overrides; their documented
defaults apply unless the prompt already provides an override.

## Read-only discovery

Run this section only after required intake is resolved.

1. Resolve absolute paths for `WORKSPACE`, the two archives, and the intended
   dataset root. PAS workflow code and templates are bundled under
   `SKILL_ROOT`; never ask the user for an implementation checkout.
   `RESULTS_DIR` must be a child of `WORKSPACE`; `DATASET_ROOT` must be nested
   below a workspace data directory (for example
   `$WORKSPACE/data/pas_v31_tao_ft`), not directly below `WORKSPACE`. Neither
   may contain the other, and approved paths may not traverse symlinks.
2. Resolve either one identity-filtered PAS run directory from initial intake,
   or a new path such as `<workspace>/results/run_<UTC timestamp>`. Never select
   among multiple PAS runs by guessing; summarize the candidates or ask once.
   Never send a state from another workflow to the PAS audit as a resume
   candidate.
3. Verify `images_raw.tar` and `meta.tar.gz` are regular, non-empty, readable
   archives. Check them to end without printing their member lists:

   ```bash
   set -e
   tar -tf "$IMAGES_ARCHIVE" >/dev/null
   tar -tzf "$METADATA_ARCHIVE" >/dev/null
   ```

   Record the presence of adjacent `SHA256SUMS`. Its absence is a warning, not
   a blocker; `rebuild.py` verification remains mandatory.
4. Read the selected platform skill and run its read-only access preflight.
   For Docker, SLURM, Kubernetes, and Brev, also run the shared checker with
   the exact platform, pinned image, and approved GPU requirements. Do not pass
   local archive paths to a remote platform check; validate their staged
   compute paths after the approved staging step. Do not use
   `--skip-platform-access` for launch readiness. The shared checker's Docker
   image-specific CUDA framework probes are side-effecting, so defer those
   invocations until approval.
   Virtualenv uses its platform skill's venv/import/CLI checks instead of the
   container-oriented shared checker. Platform-native checks are:

   ```bash
   PAS_PYT_IMAGE=nvcr.io/nvidia/tao/tao-toolkit:7.2.0-pyt  # versions-key: images.tao_toolkit.deft_pas_pyt
   PAS_DS_IMAGE=nvcr.io/nvidia/tao/tao-toolkit:7.2.0-data-services  # versions-key: images.tao_toolkit.deft_pas_data_services
   TARGET_GPU_ARGS=()
   if [[ "$PLATFORM" == docker ]]; then
     IFS=, read -r -a GPU_ID_LIST <<< "$GPU_IDS"
     for GPU_ID in "${GPU_ID_LIST[@]}"; do
       TARGET_GPU_ARGS+=(--target-gpu-index "$GPU_ID")
     done
   fi
   "${TAO_SKILL_BANK_PATH:?}/scripts/check_tao_launch_preflight.py" \
     --skill-bank "$TAO_SKILL_BANK_PATH" --platform "$PLATFORM" \
     --container-image "$PAS_PYT_IMAGE" \
     --gpu-min-count "$NUM_GPUS" "${TARGET_GPU_ARGS[@]}"
   ```

   Docker inspects the selected local or `DOCKER_HOST` daemon, images, GPU, and
   paths from that daemon's compute frame; SLURM checks SSH, scheduler, Lustre,
   Pyxis/Enroot, and scheduler-declared allocation shape; Kubernetes checks
   context, namespace/PVC, and allocatable GPUs; Brev checks its CLI/API and
   target-instance reachability. Remote Docker/GPU compatibility is not
   inferred from the launcher's inventory: the approved compute-frame probe in
   step 2 verifies it. Virtualenv checks the venv interpreter, CUDA, and that
   `clip`, `embedding`, and `tmm` resolve under `<venv>/bin`. Missing
   system/native prerequisites are blockers, not reasons to choose another
   platform. Image acquisition and CUDA jobs remain planned actions until
   approval.
5. Check only whether credentials required by the selected platform and model
   exist in the current process environment. Never open or source a credential
   file, and never print, grep, copy, inspect, or echo a credential value. This
   can include `NGC_KEY`, `HF_TOKEN`,
   `BREV_API_TOKEN`, SLURM connection variables, Kubernetes context variables,
   or tier-C storage variables. `NGC_KEY` is required only if the selected
   platform must acquire an image. The default SigLIP2 model is public, so
   `HF_TOKEN` is optional unless authenticated access is required.
   Do not inspect credential-file metadata when neither variable is required.
   If the user explicitly asks for a permissions check, `stat` only that named
   file, warn about group/other readability, and still require credentials to
   be exported in the launching environment.
6. Resolve the approved GPU shape in the selected compute frame. For Docker,
   the shared preflight intersects physical `nvidia-smi` inventory with
   `CUDA_VISIBLE_DEVICES`, compares the process-visible count with `num_gpus`,
   and verifies every requested host index is visible. Its summary must show
   `physical`, `visible`, and `requested`; a request larger than the visible set
   is blocking. For SLURM, Kubernetes, and Brev, use the platform-native
   allocation or target inventory instead of the launcher's local inventory.
   Keep using the selected compute frame's GPU telemetry for utilization and
   memory. SigLIP2-so400m training commonly needs
   roughly 30–45 GB free per selected GPU at the bundled batch size. Treat this
   as a planning estimate, not a capability guarantee. Surface occupied GPUs;
   do not silently reshape the selected host `gpu_ids`. These are launcher
   device selectors. The immutable preparation step separately derives TAO's
   in-container ordinals as `0..num_gpus-1`, because Docker renumbers every
   exposed allocation into a dense container-local CUDA namespace.
7. Resolve all run values and their sources. Validate the metric contract
   vocabulary against `references/metric-contract.md`. Do not create a config
   to discover defaults; read the bundled templates.
8. For resume, use the bundled PAS runtime with the prior workspace venv. The
   audit verifies the bundled runtime hash recorded when the run was created:

   ```bash
   "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
     "$SKILL_ROOT/scripts/audit_deft_run.py" --results-dir "$RESULTS_DIR"
   ```

   Report the audit status and next action. Do not change immutable values in
   an existing run. If the dependency environment or matching skill version is
   unavailable, report the resume audit as blocked during discovery. Restore
   the same skill version/runtime rather than asking for a separate source
   checkout, and do not launch GPU work before the audit.

Ask one consolidated question for unresolved required inputs. `max_iterations`
is required unless a time budget was supplied. Estimate conservatively:
baseline setup and pool embedding can take tens of minutes, and one 8-GPU
iteration is commonly about 30 minutes at the bundled 10k/1-epoch settings;
hardware, pool size, and accumulated data can change this substantially.

## Defaults

| Field | Default |
|---|---|
| metric | `Rank-1`, query type `medium`, operator `>=`, no target |
| training epochs | `1` per iteration |
| GPU shape | `num_gpus=1`, `gpu_ids=0` |
| mining | budget `10000`, top-N `25`, cosine distance |
| gap generation | `256` queries per slice; query types `easy,medium` |
| evaluation split | `val` (`val_pairs.json`) |
| optimizer | vision LR `1e-7`, text LR `1e-7` |
| batch sizes | train `32`, validation `64`, evaluation `32` |
| image/text embedding adapter | `SigLIP` in both specs, with the shared public `google/siglip2-so400m-patch16-256` checkpoint |
| history selection | enabled, replay fraction `0.20` |
| continual behavior | dataset `true`, model `false` |
| visualization | contact sheets `true`, embedding plot `true` |
| Hugging Face token forwarding | disabled; enable only when the approved model/environment requires it |
| PyTorch image | `nvcr.io/nvidia/tao/tao-toolkit:7.2.0-pyt` | <!-- versions-key: images.tao_toolkit.deft_pas_pyt -->
| data-services image | `nvcr.io/nvidia/tao/tao-toolkit:7.2.0-data-services` | <!-- versions-key: images.tao_toolkit.deft_pas_data_services -->
| monitoring | attached, poll every `5 minutes` |

An ungated run evaluates every allowed iteration and completes with
`max_iterations`. Never invent a KPI target.

## Approval summary

Show this one compact summary and stop for explicit approval. It is the sole
approval boundary for the planned run; do not split platform, image, setup, or
launch confirmation into separate prompts.

```text
PAS DEFT pre-flight

Workflow
  skill: tao-run-deft-pas 0.4.0 (source=SKILL.md frontmatter)
  platform: <docker | slurm | kubernetes | brev | virtualenv> (source=<user | resume state>)
  Docker endpoint: <local daemon | remote DOCKER_HOST | n/a> (source=<user | environment | resume state>)
  workspace: <absolute path> (source=<user | default>)
  monitoring: attached=true; interval=5 minutes (source=default)

Run
  results: <absolute path> (new | resume; source=<user | discovery | default>)
  stop: metric=<value> (source=<user | template | default>);
        query=<value> (source=<user | template | default>);
        operator=<value> (source=<user | template | default>);
        target=<value | no target> (source=<user | default>);
        max_iterations=<N> (source=<user | derived from approved time budget>)
  train: epochs=<N> (source=<user | template | default>);
         num_gpus=<N> (source=<user | default>);
         gpu_ids=<list> (source=<user | default>)
  mining: budget=<N> (source=<user | template | default>);
          topn=<N> (source=<user | template | default>);
          metric=<name> (source=<user | template | default>);
          history=<bool> (source=<user | template | default>);
          replay=<f> (source=<user | template | default>)
  gap: queries_per_slice=<N> (source=<user | template | default>);
       query_types=<list> (source=<user | template | default>)
  data/model: eval_split=<val | test> (source=<user | template | default>);
              vision_lr=<f>; text_lr=<f> (source=<user | template | default>);
              train_batch=<N>; val_batch=<N>; eval_batch=<N>
              (source=<user | template | default>);
              image_text_embed_adapter=SigLIP
              (source=fixed by TAO 7.2 shared image/text support)
  continual: dataset=<bool> (source=<user | template | default>);
             model=<bool> (source=<user | template | default>)
  visualization: sheets=<bool> (source=<user | template | default>);
                 embeddings=<bool> (source=<user | template | default>)

Inputs
  archive root: <absolute path> (type=<archive-only | archive-with-extracted-data>;
                source=<user | discovery: checked-root reason>)
  images archive: <absolute path and size> (source=<user | discovery: checked-root reason>)
  metadata archive: <absolute path and size> (source=<user | discovery: checked-root reason>)
  SHA256SUMS: <absolute path | absent> (source=<user | discovery>)
  dataset root: <absolute intended path> (source=<user | default>)
  PAS runtime: bundled with skill (source=fixed by workflow);
               integrity=<verified | mismatch>
  credentials: <selected platform variables with set/not-needed/missing only>;
               NGC_KEY=<set | not needed | missing>;
               HF_TOKEN=<set | optional/unset | missing>
  token forwarding: requires_hf_token=<bool> (source=<user | default>)
  PyTorch image: nvcr.io/nvidia/tao/tao-toolkit:7.2.0-pyt  # versions-key: images.tao_toolkit.deft_pas_pyt
                 (source=versions.yaml; status=<available | acquire after approval>)
  data-services image: nvcr.io/nvidia/tao/tao-toolkit:7.2.0-data-services  # versions-key: images.tao_toolkit.deft_pas_data_services
                       (source=versions.yaml; status=<available | acquire after approval>)
  virtualenv profiles: pyt=<absolute path | n/a>; ds=<absolute path | n/a>;
                       ABI/packages/entrypoints/imports/pip/CUDA=<pass | fail | n/a>
  control environment: <absolute path>; distinct from execution profiles=<true | false>
  storage/staging: tier=<A | B | C>; compute targets=<resolved platform paths>
  GPUs: <selected host IDs or platform allocation> (source=<user | default>);
        physical=<count or n/a>; visible=<CUDA-visible count or n/a>;
        requested=<num_gpus>; memory=<free/total or platform inventory>

Planned writes/actions
  <platform image/venv preparation>; image-specific CUDA framework/CLI probes;
  <runtime venv install if needed>;
  config/state creation; archive extraction/rebuild; staged action submits;
  baseline and at most N iterations

Estimate: <baseline + bounded-loop estimate and assumptions>
```

Label every configurable parameter source, including defaulted parameters; do
not limit source labels to overrides. Use `user`, `template`, `default`,
`discovery: <checked-root reason>`, `fixed by workflow`, `versions.yaml`, or
`derived from approved time budget` as applicable. If the user changes a row
after approval, show only the changed rows and wait for approval again.

## Approved initialization

For a new run, perform the following in order.

1. Follow the selected platform skill's approved image/runtime acquisition.
   Docker and Brev acquire the pinned images through Docker; SLURM converts and
   caches both images as SQSH before allocating GPUs; Kubernetes makes both
   images pullable by the namespace. Virtualenv uses two immutable execution
   profiles: `pyt` for `clip` train/evaluate and `ds` for `embedding`/`tmm`.
   They may resolve to the same directory only when that one environment
   independently satisfies both contracts. The checked-in
   `virtualenv-runtime-manifest.json` binds CPython 3.12, Linux x86_64/glibc,
   exact TAO/PyTorch distributions, console-script ownership, imports, and
   CUDA 13.0. Inspect acquisition readiness without mutation:

   Docker:

   ```bash
   (
     if [ -z "${NGC_KEY:-}" ]; then
       echo "NGC_KEY is not set. Export it in the shell that launches the agent." >&2
       exit 2
     fi
     printf '%s' "$NGC_KEY" | docker login nvcr.io \
       --username '$oauthtoken' --password-stdin >/dev/null
   )
   docker pull nvcr.io/nvidia/tao/tao-toolkit:7.2.0-pyt  # versions-key: images.tao_toolkit.deft_pas_pyt
   docker pull nvcr.io/nvidia/tao/tao-toolkit:7.2.0-data-services  # versions-key: images.tao_toolkit.deft_pas_data_services
   ```

   Virtualenv:

   ```bash
   python3 "$SKILL_ROOT/scripts/manage_pas_virtualenv.py" plan \
     --profile pyt --virtualenv "$PAS_PYT_VIRTUALENV"
   python3 "$SKILL_ROOT/scripts/manage_pas_virtualenv.py" plan \
     --profile ds --virtualenv "$PAS_DS_VIRTUALENV"
   ```

   A supplied prebuilt profile is verified directly against the manifest's
   exact ABI, resolved distribution/version set, console-script metadata,
   imports, dependency consistency, CUDA build, and real GPU behavior. That
   read-only verification does not depend on an installation lock and never
   mutates the supplied environment. Creating a new profile is different: it
   is allowed only from the same complete, reviewed, hash-locked transitive
   requirements file. The packaged manifest currently marks lock generation
   as required, so `install` fails before creating anything and reports the
   exact missing approved resolver step, while `verify` remains available for
   a supplied compatible profile. Do not install unpinned packages, infer
   hashes, use a source checkout as runtime, or weaken the acquisition
   boundary. Never expose credentials or silently substitute a platform.

2. Run an image-specific CUDA framework smoke using the exact approved resource
   shape. GPU enumeration or `nvidia-smi` inside a container is not sufficient:
   it can succeed when the image's PyTorch/CUDA build cannot initialize against
   the host driver. Bind-mount and execute
   `scripts/check_pas_cuda_runtime.py` inside every pinned runtime that the
   approved run will use. Require `clip` for the PyTorch image and require both
   `embedding` and `tmm` for the data-services image:

   ```bash
   python3 /probe/check_pas_cuda_runtime.py \
     --min-gpus "$NUM_GPUS" --require-cli clip
   python3 /probe/check_pas_cuda_runtime.py \
     --min-gpus "$NUM_GPUS" --require-cli embedding --require-cli tmm
   ```

   The selected platform consumer owns the surrounding Docker, `srun`, pod, or
   Brev command and must allocate the approved GPUs to the probe exactly as it
   will to the action. A bounded verification that uses only one image may
   probe only that image; a normal DEFT loop must probe both. For virtualenv,
   perform the complete profile verification, including the same real probe:

   ```bash
   python3 "$SKILL_ROOT/scripts/manage_pas_virtualenv.py" verify \
     --profile pyt --virtualenv "$PAS_PYT_VIRTUALENV" --min-gpus "$NUM_GPUS"
   python3 "$SKILL_ROOT/scripts/manage_pas_virtualenv.py" verify \
     --profile ds --virtualenv "$PAS_DS_VIRTUALENV" --min-gpus "$NUM_GPUS"
   ```

   The verifier rejects a fake executable unless pinned distribution metadata
   owns the exact console script, and checks the Python/platform ABI, package
   versions, action imports, `pip check`, PyTorch CUDA build, and then tensor
   allocation. The probe must allocate
   and synchronize a CUDA tensor on every requested visible device. Any
   framework initialization failure, missing TAO CLI entrypoint, insufficient
   visible GPU count, or unsupported architecture is a hard stop before state,
   data staging, or action submission.
   For Docker, `GPU_IDS` is the exact approved host `gpu_ids` list (for example
   `0` or `0,2`). The Docker selector remains `device=0,2`, while the TAO spec
   uses container-local `gpu_ids: [0, 1]` for that two-device allocation.
3. Reuse a complete **control** workspace venv if the bundled-runtime import
   passes. This is separate from the two TAO execution profiles above and may
   use CPU PyTorch because it runs host-side analysis, not TAO actions. Otherwise
   create it and install only the bundled runtime's third-party dependencies:

   ```bash
   python3 -m venv "$WORKSPACE/.venv"
   "$WORKSPACE/.venv/bin/pip" install \
     pandas numpy matplotlib pyarrow pillow pyyaml scikit-learn
   "$WORKSPACE/.venv/bin/pip" install torch \
     --index-url https://download.pytorch.org/whl/cpu
   "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
     -c 'import pas_deft; print("bundled PAS runtime: OK")'
   ```

   Never install into the system interpreter. If package installation was not
   in the approved actions, obtain approval first.
   The control interpreter used by `run_deft_action.py` must also import
   `jsonschema`; install that small helper in an approved non-system
   environment if no existing Python provides it.
4. Materialize an immutable run config and `approval.json` from the bundled
   templates. Pass every approved override explicitly; this command writes
   only `${RESULTS_DIR}/config/` and refuses an initialized run:

   ```bash
   PREP_OPTIONAL_ARGS=()
   PLATFORM_ARGS=(--platform "$PLATFORM")
   if [ "$PLATFORM" = docker ] && [ "${DOCKER_REMOTE:-false}" = true ]; then
     PLATFORM_ARGS+=(--docker-remote)
   fi
   if [ "$PLATFORM" = virtualenv ]; then
     PLATFORM_ARGS+=(
       --pyt-virtualenv "$PAS_PYT_VIRTUALENV"
       --ds-virtualenv "$PAS_DS_VIRTUALENV"
     )
   fi
   if [ -n "${CHECKSUMS_FILE:-}" ]; then
     PREP_OPTIONAL_ARGS+=(--checksums-file "$CHECKSUMS_FILE")
   fi
   if [ -n "${METRIC_TARGET:-}" ]; then
     PREP_OPTIONAL_ARGS+=(--metric-target "$METRIC_TARGET")
   fi
   if [ "${REQUIRES_HF_TOKEN:-false}" = true ]; then
     PREP_OPTIONAL_ARGS+=(--requires-hf-token)
   fi

   "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
     "$SKILL_ROOT/scripts/prepare_deft_config.py" \
       --workspace "$WORKSPACE" --results-dir "$RESULTS_DIR" \
       --dataset-root "$DATASET_ROOT" \
       --images-archive "$IMAGES_ARCHIVE" \
       --metadata-archive "$METADATA_ARCHIVE" \
       "${PLATFORM_ARGS[@]}" \
       "${PREP_OPTIONAL_ARGS[@]}" \
       --max-iterations "$MAX_ITERATIONS" \
       --training-epochs "$TRAINING_EPOCHS" \
       --num-gpus "$NUM_GPUS" --gpu-ids "$GPU_IDS" \
       --mining-topn "$MINING_TOPN" --knn-metric "$KNN_METRIC" \
       --target-query-count "$TARGET_QUERY_COUNT" \
       --queries-per-slice "$QUERIES_PER_SLICE" \
       --gap-query-types "$GAP_QUERY_TYPES" \
       --eval-split "$EVAL_SPLIT" \
       --vision-lr "$VISION_LR" --text-lr "$TEXT_LR" \
       --train-batch-size "$TRAIN_BATCH_SIZE" \
       --val-batch-size "$VAL_BATCH_SIZE" \
       --eval-batch-size "$EVAL_BATCH_SIZE" \
       --text-embed-model "$TEXT_EMBED_MODEL" \
       --history-aware "$HISTORY_AWARE" --replay-fraction "$REPLAY_FRACTION" \
       --continual-dataset "$CONTINUAL_DATASET" \
       --continual-model "$CONTINUAL_MODEL" \
       --visualize "$VISUALIZE" \
       --visualize-embeddings "$VISUALIZE_EMBEDDINGS" \
       --metric-name "$METRIC_NAME" --metric-query-type "$QUERY_TYPE" \
       --metric-op "$METRIC_OP"
   ```

5. Initialize state once. Omit checksum, target, and token flags when they are
   not approved:

   ```bash
   INIT_OPTIONAL_ARGS=()
   PLATFORM_ARGS=(--platform "$PLATFORM")
   if [ "$PLATFORM" = docker ] && [ "${DOCKER_REMOTE:-false}" = true ]; then
     PLATFORM_ARGS+=(--docker-remote)
   fi
   if [ "$PLATFORM" = virtualenv ]; then
     PLATFORM_ARGS+=(
       --pyt-virtualenv "$PAS_PYT_VIRTUALENV"
       --ds-virtualenv "$PAS_DS_VIRTUALENV"
     )
   fi
   if [ -n "${CHECKSUMS_FILE:-}" ]; then
     INIT_OPTIONAL_ARGS+=(--checksums-file "$CHECKSUMS_FILE")
   fi
   if [ -n "${METRIC_TARGET:-}" ]; then
     INIT_OPTIONAL_ARGS+=(--metric-target "$METRIC_TARGET")
   fi
   if [ "${REQUIRES_HF_TOKEN:-false}" = true ]; then
     INIT_OPTIONAL_ARGS+=(--requires-hf-token)
   fi

   PAS_PYT_IMAGE=nvcr.io/nvidia/tao/tao-toolkit:7.2.0-pyt  # versions-key: images.tao_toolkit.deft_pas_pyt
   PAS_DS_IMAGE=nvcr.io/nvidia/tao/tao-toolkit:7.2.0-data-services  # versions-key: images.tao_toolkit.deft_pas_data_services

   "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
     "$SKILL_ROOT/scripts/init_deft_state.py" \
       --results-dir "$RESULTS_DIR" --workspace "$WORKSPACE" \
       --dataset-root "$DATASET_ROOT" \
       --images-archive "$IMAGES_ARCHIVE" \
       --metadata-archive "$METADATA_ARCHIVE" \
       "${INIT_OPTIONAL_ARGS[@]}" \
       --max-iterations "$MAX_ITERATIONS" \
       --metric-name "$METRIC_NAME" --metric-query-type "$QUERY_TYPE" \
       --metric-op "$METRIC_OP" \
       "${PLATFORM_ARGS[@]}" \
       --pyt-image "$PAS_PYT_IMAGE" \
       --ds-image "$PAS_DS_IMAGE" \
       --deft-config "$RESULTS_DIR/config/deft_config.yaml" \
       --tao-spec "$RESULTS_DIR/config/tao_spec.yaml"
   ```

6. Run the audit. Its only legal first action for a new run is
   `baseline/dataset_setup`. If initialization or audit fails, do not
   reinitialize the same directory; diagnose the invalid input and start a new
   run directory.

## Resume

For an existing run, do not repeat image/config/state setup merely because a
new shell lacks variables. `deft_python.sh` takes the workspace explicitly and
the action producer reads platform, workspace, images, mounts, and config paths
from state. Run the audit, read its one stage reference, and follow `next_action`.
If the user's requested values differ from immutable state, explain the
difference and start a separately approved run rather than mutating history.
