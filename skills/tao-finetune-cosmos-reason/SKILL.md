---
name: tao-finetune-cosmos-reason
description: >-
  Shared Cosmos3 frontend that explicitly routes Cosmos Framework and
  Cosmos-RL, validates runtime model/video-dataset/SLURM inputs, consumes an
  SQSH or packaged backend image, optionally plans explicit clean
  source builds, prepares checkpoints, validates the first update in-process,
  and returns token-weighted losses and task-aware accuracy.
license: Apache-2.0
compatibility: Requires Python 3.11+ with PyYAML and a supported container execution platform; SLURM runs additionally require SSH, sbatch/srun, Pyxis/Enroot, and shared storage.
metadata:
  author: NVIDIA Corporation
  version: "0.3.6"
allowed-tools: Read Bash
tags:
- model
- cosmos
- multimodal
- training
---

# Cosmos3 TAO training

Keep one shared model-facing frontend. Backend image fields and contract paths
live under `backend_contracts` in `references/skill_info.yaml`; image literals
are stamped from `versions.yaml`, while referenced backend YAMLs define native
runtime schemas. Never translate between them.

## Mandatory runtime intake

Before planning training, collect all of the following. Do not infer a path
from history, another user, a prior job, an image, or a developer checkout.

- `base_model_path_or_uri`. For a Hugging Face model ID or URL, accept an
  optional friendly `base_model_revision` such as a branch or tag. If omitted,
  resolve `main`; do not ask the user for a commit SHA. Resolve the selected
  ref read-only through the Hub API to its immutable commit and seal the model
  ID, requested ref, and resolved SHA in the plan. A complete local snapshot
  needs no revision and is sealed by its file fingerprints.
- For Cosmos3-Nano, an explicit input-checkpoint `model_type`: `qwen3_vl`
  or `cosmos3_omni`. If the user did not supply it, ask once before planning;
  never infer the choice from `config.json`, a model ID, a path name, or a
  previous run. Explain the two choices in plain language: `qwen3_vl` uses a
  compatible Hugging Face checkpoint directly, while `cosmos3_omni` requires
  an immutable conversion to exact Qwen3-VL safetensors before training.
  Record the answer as `base_model_format`. Cosmos3-Edge is inferred as
  `cosmos3_edge` from the resolved model ID and does not present this Nano-only
  choice.
- Do not expose Omni preparation implementation fields during normal intake.
  For Nano, use the packaged `Qwen/Qwen3-VL-8B-Instruct` architecture mapping,
  resolve both Hub models to immutable commits automatically, and run the
  TAO-owned `cosmos_rl.model_preparation.vlm_safetensors` entrypoint with the
  already selected backend image/SQSH. Both backend images must package that
  entrypoint and its pinned native Framework conversion runtime. An explicitly
  supplied `prepared_checkpoint_path` or donor is an advanced override: validate
  it, but never present a route A/B choice or ask for one by default.
- Accept `hf_model://nvidia/Cosmos3-Nano` directly. If a gated/private model
  cannot be resolved, ask the user only to set `HF_TOKEN` in the session
  environment; never ask them to discover a SHA or provide the token value in
  chat.
- explicit video sampling mode: either uniform `nframes` or `fps`. FPS mode
  may also set `min_frames` and `max_frames`; both modes may set clip-time,
  resize, and pixel-budget fields supported by the selected backend.
- training/validation annotation paths and media roots for conversation-style
  or task-aware video supervision, plus optional task selection.
- explicit `backend` for a comparison; `cosmos-framework` or `cosmos-rl`.
- `training_mode`; `dense` or `peft`. PEFT also requires rank, alpha, dropout,
  target modules, bias, RS-LoRA, modules-to-save, and adapter precision.
- user-owned `results_dir`, `checkpoint_dir`, `cache_dir`, and, for SLURM,
  `sqsh_cache_dir`, `ssh_key_path`, mounts, and scheduler settings.
- Runtime order: compute-readable `sqsh_path`, explicit image, then the selected
  backend image in `references/skill_info.yaml`. On SLURM reuse or convert it
  once under `sqsh_cache_dir`. Never
  compare an SQSH filename with an image tag or request source provenance/SHA.
- Repository paths, commits/trees, branch, base image, build context, and
  timestamp are advanced inputs required only for explicit `source-build`.
  See `references/cosmos-backend-operations.md`; never infer a build from
  runtime selection.

The planner preserves each original path and reports an accessible `realpath`.
Missing required paths fail. A missing supplied SQSH fails; an omitted SQSH
selects the packaged image. No historical fallback path or image is allowed.

### Nano checkpoint model-type choice

Treat the selected `base_model_format` as a user decision and verify that it
matches the supplied local checkpoint's `config.json.model_type`. A mismatch
fails; it is not permission to relabel or rewrite the source checkpoint.

For `qwen3_vl`, fingerprint and use a complete compatible Hugging Face Nano
checkpoint directly. For `cosmos3_omni`, tell the user in the launch review
that Cosmos-RL will prepare its compatible checkpoint automatically, name the
planned output path, and preserve the source checkpoint unchanged. Do not ask
the user for an architecture donor, preparation image, or preparation SQSH.
The backend-owned preparation step emits a verified `qwen3_vl` checkpoint
under the selected platform's user-owned `checkpoint_dir`:

- Docker: the local Docker host's `checkpoint_dir`.
- SLURM: the compute-node-verified shared `checkpoint_dir`, covered by an
  explicit container mount. Run conversion through the SLURM/Pyxis contract;
  do not write the converted checkpoint to controller-local storage.

Use the packaged Nano architecture mapping and selected backend runtime unless
an advanced override was explicitly supplied. Fingerprint the source,
architecture mapping, converted config/tokenizer/processor/index/shards, and
conversion provenance. Before conversion, bind the
sealed plan's backend-native training fields and `VLM_SAFETENSORS_PATH` to the
planned converted checkpoint inside the selected container. Validate that
exact output before training without mutating the sealed plan. The original
Omni/Hugging Face path remains provenance only and must never remain as the
runtime model path. Reuse is allowed only when the exact target has complete
matching conversion provenance and passes the same validation.

### Public Cosmos3-Edge checkpoint contract

Accept the public model at its resolved immutable revision or a local
snapshot; never request a second checkpoint.
Apply the model-aware runtime defaults from `references/skill_info.yaml` and
`references/cosmos-framework-backend.yaml`, preserving separate model and
processor-profile fingerprints and each default/override origin.

## Backend selection

Run `scripts/cosmos_workflow.py resolve` first.

| Request | Automatic selection |
|---|---|
| Cosmos3-Nano plain train | Cosmos-RL (compatibility default) |
| Cosmos3-Nano AutoML/HPO | Cosmos-RL |
| Nano Framework-DCP export | Cosmos Framework |
| Nano evaluate/inference/microservice with no explicit backend | Cosmos-RL |
| Nano quantize | Cosmos-RL |
| Cosmos3-Edge train/export/evaluate/inference/microservice | Cosmos Framework |

An explicit supported backend wins, so users can select Cosmos Framework for
Nano training without changing model ownership. Comparative runs reject
`auto`, so both sides of an experiment are deliberately forced.
Framework-trained checkpoints use the native exact-key exporter, then the
repository-backed TAO evaluation adapter. That does not make Framework a
Cosmos-RL version.

## Evaluation intake and inheritance

Run `scripts/evaluation_workflow.py` for every evaluate action and follow
`references/cosmos-reason-evaluate.md` completely. A parent training plan owns
all inheritable model, dataset, prompt, preprocessing, precision, checkpoint,
and scoring fields; never ask the user to repeat them. Ask once only for the
helper's `required_user_inputs`, execute its `automated_actions`, and launch
only a checksum-valid `ready=true` plan. Cosmos-RL policy checkpoints require
the emitted `cosmos_rl_checkpoint_pre_action`; Framework DCP inputs require
their emitted export pre-action. The ready plan includes a validated
`spec_bundle.execution`; pass it unchanged to the selected platform. Cosmos
owns runtime attestation and evaluator configuration; the platform owns launch.

## Framework checkpoint pre-action

Before Framework evaluate, inference, or microservice actions, run
`scripts/framework_checkpoint_action.py plan` and its emitted `prepare` and
`verify` steps. Follow `references/cosmos-backend-operations.md`; never ask the
user to export DCP manually. On SLURM stage only the helper dependency set
declared by `workflow_contract.action_helper_dependencies`; the platform
verifies the closed bundle. Use only the verified
`action_model_path`, reuse only matching complete exports, and record the
pre-action, manifest, fingerprints, and independent child result.

## Required gates

Execute these stages in order and persist their outputs.

1. Resolve model/backend/action and load the selected backend contract.
2. Check credentials by presence only. Never read or persist credential
   values. Require a token only for the operation that needs it.
3. Validate tools, storage, paths, and runtime selection in the mandatory
   intake order; existing-SQSH and packaged-image modes skip source gates.
4. Only for explicit `source-build`, validate/build clean sources and verify
   `/opt/tao/image-provenance.json`. Never mount host source into training.
5. Enforce the explicit Nano checkpoint model-type choice. If it is
   `cosmos3_omni`, show the conversion and platform-owned output path in the
   launch review, then prepare the model through the shared TAO integration
   entrypoint packaged in the selected clean backend image after approval.
   Resolve URI/model-ID refs to immutable
   Hub commits automatically. Validate exact tensor/config keys and fingerprint model,
   tokenizer, processor, weight index, every shard, and provenance. Assign the
   verified converted path—not the original source path—to training.
6. Validate inputs, counts, duplicates, overlap, tasks, and identities. A
   no-byte-hashing request selects both fast fingerprint flags; metadata stays
   hashed while weight/media payloads use path+size.
   Verify the resolved inputs again from an allocated compute node.
   When SLURM storage is not mounted on the launch host, let
   `cosmos_workflow.py` stream its checked-in `cosmos_common.py` inspector to a
   login host over SSH. It runs from stdin, preserves remote `realpath` values,
   and creates no remote script or source overlay. Do not require local Lustre,
   `sbatch`, or `srun` on an SSH-based launch host. Run this expensive input
   inspection exactly once with the `plan` verb and pass a local
   `--plan-artifact <path>` so the resolved request and inspection results are
   sealed for the remaining launch verbs.
7. Resolve the video runtime from the structural dataset contract and enforce
   every profile gate in `references/cosmos-reproducibility-gates.md`.
   Cosmos-RL `auto` selects source-baked `pynv-device-rgbp` for
   `video_conversation` and `system-pyav` for `task_aware_video_reasoning`;
   Framework selects `torchcodec-cuda-on-demand`. All defaults use bounded
   rank-local, on-demand memory with no disk prewarm. Repeated-media validation
   may use the backend-native grouped sharder and validation-only caches only
   while preserving records, explicit batch, weighting, prompts, and
   preprocessing. Throughput settings that relax batch-1 parity require user
   authorization.
8. Generate backend-native TOML, environment, topology, preflight commands,
   parity data, runtime profile, and metadata. Full specs contain no sample
   limit. Reuse one sealed `--plan-artifact` across read-only `preflight`,
   post-review `materialize`, and `render-slurm`; do not repeat original inputs.
   Materialize atomically in the verified compute frame, derive container paths
   only from explicit mounts, and never copy source patches to the cluster.
   Diagnostic subsets are explicit opt-ins, never launch prerequisites.
9. On SLURM reuse the supplied/derived SQSH or convert the selected image once
   before GPU submit. SQSH SHA and source provenance are not runtime gates.
   When Omni preparation is required, inspect the SQSH filesystem and reject it
   unless it contains the shared TAO launcher, the native Framework converter,
   and `/opt/tao/framework-converter-runtime.json`. The runtime artifact must
   report `validation_mode=imported_converter_module`, proving the isolated
   converter's transitive dependency graph imported during the image build;
   file presence alone is not sufficient.
   Verify Pyxis/Enroot, mounts, Python/packages, decoder, GPU, CUDA, NCCL, and
   storage in the training allocation.
10. Launch the requested training job directly; do not submit a separate smoke
    job unless the user explicitly requests one. For Cosmos-RL VLM training,
    require packaged source that emits a padding-aware `attention_mask` and
    runs the visual-gradient contract after the first backward pass and before
    the first optimizer update. Persist total/trainable/frozen counts and
    gradient norms for the vision encoder, visual projector, language model,
    and language head. Fail immediately when a trainable visual component has
    no gradient, a non-finite norm, or a zero norm. Report an explicitly frozen
    visual component as not applicable rather than as a failure.
11. Materialize the full spec once and verify its SHA256 in the compute frame
    before rendering the job from the same plan artifact. Monitor scheduler and
    structured TAO state to a terminal result, and preserve the child exit code
    independently of scheduler state. Require child exit zero, structured
    `SUCCESS`, finite global train/validation loss, checkpoint completion, and
    a final evaluator metric before reporting completion.
12. Resolve evaluation with `scripts/evaluation_workflow.py`. Inherit exact
    fine-tuning artifacts, collect only its remaining user inputs, run its
    backend-owned automated checkpoint pre-actions, and require `ready=true`.
    On Cosmos-RL, verify the selected HF export with
    `cosmos_rl_checkpoint_action.py`, rerun resolution with its manifest, and
    submit the emitted spec-bundle through the chosen platform; never select
    the native policy directory, copy the lifecycle into an application-owned
    launcher, or improvise result/status variables.
    Evaluate the selected checkpoint with identical prompt, preprocessing,
    generation, normalization, and task scoring. Extract final metrics with
    `scripts/extract_cosmos_metrics.py`.

## Dataset contracts

Resolve datasets by structure, not by project, benchmark, directory, or file
name. The supported families are:

- `video_conversation`: a JSON array with media and at least two ShareGPT,
  LLaVA, or OpenAI-style conversation turns;
- `task_aware_video_reasoning`: one or more item-envelope or array annotation
  files with media, task identity, and conversation/response targets.

Default `dataset_family` to `auto`, inspect every annotation, and require train
and validation to resolve to the same family. Capture record count, unique
media count, media reuse, extensions, byte-size distribution, task/metric
metadata, and any declared width, height, FPS, and duration. Select processor,
cache and resource profiles from those characteristics and the
model tier. Never branch on a customer dataset name.

Tasks declaring accuracy participate in deterministic accuracy; common binary
and multiple-choice task types are recognized. Generative tasks report their
declared metrics and are excluded from aggregate accuracy with a reason.
Aggregate accuracy is example-weighted over records with an accuracy definition.

## Dense and PEFT contracts

Dense SFT has no LoRA block and reports trainable, frozen, and total counts.
PEFT preserves rank, alpha, dropout, targets, bias, RS-LoRA, saved modules,
precision, and trainable count across backends; mismatched semantics block a pair.

For fair comparisons, force the same logical model, train/validation records,
media, prompt, frames, sequence length, precision, seed, epochs, effective
global batch, optimizer, learning rate, schedule, warmup, weight decay,
clipping, loss masking, validation/checkpoint cadence, evaluated checkpoint,
and generation/normalization settings. Classify differences as equivalent
syntax, unavoidable implementation difference, or invalid mismatch; an invalid
mismatch blocks the full pair.

## Metrics and completion

The required primary metrics are:

- complete-run globally reduced token-weighted training loss, with numerator
  and valid-label denominator;
- final-validation globally reduced token-weighted loss, with numerator and
  valid-label denominator;
- the repository evaluator's final validation metric and any supporting values
  it emits. Do not add a second post-evaluation gate.

Do not average console lines or rank means. A step loss is not the average
training loss. A validation heartbeat is not final validation loss. A
generative exact match is not accuracy unless the task defines it.

The native Framework callback and native Cosmos-RL logger own early failure,
checkpoint, progress, metric, and terminal events. Do not stage a status bridge
or patch status at container startup. `COMPLETED` from SLURM is failure when the
child exit code is nonzero or the terminal TAO state is not successful.

## SLURM invariants

Generated jobs use Bash, SQSH via Pyxis, no requeue by default, one launcher
task per node, runtime-supplied logs, and the training child exit code. A
single-node exclusive allocation preserves the requested CPU count in
the `SBATCH` contract but passes every CPU actually granted by SLURM to the
training step and records requested, allocated, and step counts. Before the
training child, the same allocation runs the planner-owned packaged-runtime
gate as the training job's first Pyxis step; a failed gate writes the independent
child exit artifact and blocks training. Generated jobs set
`SLURM_EXPORT_ENV=ALL`, and the platform consumer still submits with
`sbatch --export=ALL`. Every Pyxis step also receives the planner-owned
runtime variable names through `--container-env`, so a baked image `ENV` value
cannot override the selected decoder, frame-transfer, cache, or worker profile.
Framework topology is
shard=`gpus_per_node`, replica=`nodes`.
Cosmos-RL uses one controller on node zero and its policy-worker topology.
Asynchronous distributed checkpointing is rejected for multi-node runs.

Every job metadata record must validate against
`schemas/cosmos-job-metadata.schema.json` and contain paths, fingerprints,
runtime identity, config, resources, states, outputs, and timing without
credentials. Source provenance exists only for `source-build`; SQSH SHA is
optional and never a runtime gate.

## Source-affecting recovery

If a run exposes a code or image defect, stop the affected path, change the
owning repository, add a test, commit it, rebuild both image and SQSH from a
clean checkout, then restart every affected training job from its clean sealed
plan. Never edit a running container, patch an existing image, reuse an old
SQSH after a source change, or rely on a temporary launch script as the
implementation.

Use `references/cosmos-reproducibility-gates.md` as the source-owner/test map.

For infrastructure retries, the launch skill classifies the failure and opens
the new `--retry-of` record; SLURM supplies validated node inventory and
exclusions. Run `cosmos_workflow.py retry-plan` with the new record's
`<action-root>/config/train.toml`; it rebases all writable paths and reseals the
Cosmos request. Render that plan; never patch SBATCH.
