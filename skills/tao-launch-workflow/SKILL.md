---
name: tao-launch-workflow
description: >-
  Shared launch intake for any TAO workflow or action. Use when the user wants
  to run TAO AutoML, train, evaluate, infer, export, generate TensorRT engines,
  or launch DEFT/workflow jobs on an execution platform.
license: Apache-2.0
compatibility: Requires the packaged TAO skill bank helper scripts.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- tao
- workflow
- launch
---

# TAO Workflow Launch Intake

Use this skill before launching any TAO workflow or model action.

## Quick Start

Run the platform helper, ask for platform and monitoring preferences, then run
the selected platform detail helper before asking for credentials.

## Non-Negotiable Launch Gate

This gate is model-agnostic. Apply it to every TAO model, data action, and
application workflow before launching side-effecting work.

Do **not** create runner scripts, launch scripts, compatibility shims,
workspace folders, state files, logs, or dependency-install side effects until
the launch preflight passes.

Preflight passes only after all of these are true:

1. The execution platform is selected from the packaged platform helper.
2. Platform credentials and required credential groups are satisfied.
3. Model-specific credentials are satisfied.
4. The default container image is resolved from packaged model/action metadata,
   shown to the user, and either confirmed or replaced by an explicit
   `image=<override>`.
5. The platform access check succeeds from the launch host.
6. Dataset inputs are mapped to concrete spec keys and verified from the
   selected platform's point of view.
7. Required compute shape fields from the model/workflow skill are known.
8. Required local tools for the selected data/platform path are present, or the
   user approved installing the smallest missing dependency and preflight was
   rerun.
9. A launch review with image, platform, datasets, compute shape, expected
   runtime, and any generated/default configuration changes has been shown and
   confirmed by the user. For AutoML, the launch review must explicitly state
   recommendation count/budget, max concurrency, algorithm, metric, direction,
   and searched parameters/ranges even when defaults are used.

If any item is missing, ask for the missing input and stop before generating
artifacts. This applies to AutoML, normal train/eval/infer/export/TRT, and
DEFT/application workflows.

When preflight work clears a blocker, keep track of the original user request.
After the fix, rerun the relevant preflight and continue toward that request;
do not stop at "blocker fixed" unless the user explicitly asked only for the
repair.

## Initial Questions

After the user confirms what they want to do, ask for the execution platform
using the packaged helper. Do not scan platform docs, skill folders, or config
folders to build the choices.

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_platforms.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} --format text
```

Then ask:

- Which supported platform should run this workflow?
- Should I monitor the run in this chat? Monitoring means I keep polling the
  backend/job logs after launch and report progress until the job finishes,
  fails, or you ask me to stop, even if the job stays queued for hours or days.
  If disabled, I launch the job, give you the job id/log path, and stop
  polling. Default: monitor in chat.
- How often should I post status? Default: every 5 minutes. Use 1-2 minutes for
  smoke tests, 5 minutes for normal training, or 10-15 minutes for long runs.

Use `long_running_enabled=true` and `status_interval_minutes=5` when the user
accepts the defaults.

When monitoring is enabled, do not send a final summary just because several
polls have elapsed or the job is still `PENDING`. Keep the turn attached and
emit status every `status_interval_minutes` until a terminal state or explicit
user stop/detach request. If the runtime environment cannot keep the chat turn
open, say that clearly and leave a durable watcher/log path; do not imply that
chat updates will continue after the turn ends.

Final-answer rule: a `final` response ends chat-side monitoring. While
`long_running_enabled=true` and any launched job is non-terminal, status
messages must be sent as in-progress updates and the agent must continue
polling. Only send a final response when the workflow reaches terminal state,
the user explicitly asks to detach/stop monitoring, or the runtime genuinely
cannot keep the turn open; in that last case, say it is a runtime limitation
and provide the exact durable status command/log path.

## Missing-Input Prompt Shape

When asking for launch inputs, include concrete examples and both dataset input
modes. Do not ask only for "dataset root".

Use this structure and adapt spec keys to the selected model/action:

```text
I need these launch inputs before I can create specs or runner files:

1. Execution platform: brev, slurm, local-docker, or kubernetes.

2. Dataset inputs. You can provide either mode:
   A) Root mode: give train/eval roots and I map required files automatically.
      Example Cosmos-RL:
      train_root=/lustre/fsw/.../cosmos/train
      -> custom.train_dataset.annotation_path=train_root/annotations.json
      -> custom.train_dataset.media_path=train_root
   B) Direct spec mode: give the exact config/spec parameters yourself.
      Example:
      custom.train_dataset.annotation_path=/lustre/fsw/.../train_annotations.json
      custom.train_dataset.media_path=/lustre/fsw/.../videos_train.tar.gz
      custom.val_dataset.annotation_path=/lustre/fsw/.../eval_annotations.json
      custom.val_dataset.media_path=/lustre/fsw/.../eval_videos/

   Platform examples:
   - SLURM/Lustre: /lustre/fsw/.../data/train or lustre:///lustre/fsw/.../data/train
   - Brev/Kubernetes: s3://bucket/path/train and s3://bucket/path/eval
   - local-docker: /data/tao/<model>/train or file:///data/tao/<model>/eval

3. Container image. I will resolve the default from packaged model metadata and
   show it before launch, for example:
   default image for <model>/<action>: <resolved container image>
   Use this image, or provide image=<override> to pin a different TAO build.

4. Compute shape required by the model, for example GPUs/nodes.

5. Required credentials from platform/model docs, for example HF_TOKEN for
   gated Hugging Face models.

6. Monitoring preference. By default I monitor in this chat and post progress
   every 5 minutes; choose 1-2 minutes for smoke tests or 10-15 minutes for
   long training.
```

## Container Image Confirmation

Before creating specs, runner scripts, workspaces, logs, state files, or
submitting a job, resolve the image for the selected model/action:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/resolve_tao_image.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} \
  --model <network> --action <action> --format text
```

If the helper is unavailable, read `skills/models/<network>/config.json` through
`SkillBank().get_model_config(network_arch)`. Resolve image fields in this
order:

1. `actions.<action>.container_image`
2. `actions.<action>.image`
3. top-level `container_image`
4. top-level `image`

Show the exact image and ask:

```text
Container image for <network>/<action>:
default=<resolved image>

Use this image, or provide image=<override>?
```

If the user accepts, pass the resolved image as the job `image`. If the user
overrides, require a non-empty image reference and pass that value instead.
Do not silently launch on the default image. This confirmation applies to
training, AutoML recommendations, evaluation, inference, export, TensorRT
engine generation, and application workflows that submit TAO containers.

## Credential Filtering

After the user chooses a platform, get the credential list for only that
platform:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_platforms.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} \
  --platform <platform> --format text
```

Ask only for credentials returned by that command, plus model-specific
credentials from the selected model skill. Do not ask for Brev credentials on
SLURM, Kubernetes, or local Docker. Do not ask for SLURM credentials on Brev,
Kubernetes, or local Docker. Ask S3 credentials only when the selected
platform and the dataset/result URIs require `s3://` access.
Credentials may already be present in the process environment or in a
user-approved secret env file such as `~/.tao/secrets.env` or
`~/.config/tao/.env`; source such files only when needed and never print,
grep, cat, paste, or log their contents. Verify only variable presence.

For initial launch intake, ask for required credentials and required credential
groups only. Treat the helper's optional credentials/settings section as
reference material; do not request those values unless their `only_when`
condition applies, the selected workflow cannot proceed without them, or the
user asks to customize that setting.

When the helper output includes a "Required credential groups" section, satisfy
one credential from each group before proceeding. Explain each requested value
using the helper's description and "How to get it" text.

For SLURM, user-facing prompts should ask for `SSH_KEY_PATH` first. Mention
`SSH_AUTH_SOCK` only if the user says they already use an SSH agent.

## Dependency Remediation

If a required CLI/library is missing, say exactly what is missing and why it is
needed, then ask before installing. Examples:

- S3 dataset or results path -> require an S3-capable client such as `aws`.
- SDK-backed platform launch -> require the platform-specific
  `nvidia-tao-sdk[...]` extra.
- Local Docker SDK path -> require the Docker Python client and the configured
  Docker network.

After user approval and installation, rerun the same preflight. Do not create
runner files or launch jobs between the failed check and the rerun.

## Dataset Intake

Accept dataset inputs in either mode:

- **Dataset root mode:** the user gives train/eval/calibration roots, and the
  model skill maps required files by convention. Example for Cosmos-RL train:
  `custom.train_dataset.annotation_path=<root>/annotations.json` and
  `custom.train_dataset.media_path=<root>`.
- **Direct spec mode:** the user gives exact spec-key paths when annotations,
  media archives, videos, or image folders live in different places. Preserve
  those keys directly, for example
  `custom.train_dataset.annotation_path=/lustre/.../train_annotations.json`
  and `custom.train_dataset.media_path=/lustre/.../videos.tar.gz`.

Ask for dataset examples that match the selected platform:

- SLURM: shared cluster paths such as
  `/lustre/fsw/portfolios/<team>/<your-dir>/data/<model>/train` (where
  `<your-dir>` is your per-user directory on the cluster), or direct
  spec paths under `/lustre/...`.
- Brev, Kubernetes: usually `s3://bucket/path/train` and
  `s3://bucket/path/eval` unless the platform profile mounts shared storage.
- Local Docker: local paths visible to the Docker host, such as
  `/data/tao/<model>/train`, or direct spec paths visible inside the planned
  container mount.
- Remote Docker: absolute paths visible on the remote Docker host named by
  `DOCKER_HOST`, not paths on the local agent machine.

Do not assume "dataset root" is the only acceptable input. When direct spec
paths are supplied, validate the exact spec paths rather than appending default
filenames.

## Platform Preflight

Run the selected platform's preflight checks before any launch artifact is
created.

Prefer the packaged preflight helper when the needed inputs are available:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/check_tao_launch_preflight.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} \
  --platform <platform> \
  --container-image <selected-image> \
  --path train_annotation=<path> \
  --path train_media=<path>
```

Pass exact direct spec paths when the user supplied them. For root-mode inputs,
expand model-required files first, then pass those concrete annotation/media
paths to the helper.

If the helper reports a missing client tool such as `aws` for `s3://` path
verification, install the smallest needed package after user approval, then
rerun the same command with `--install-missing-tools` and do not proceed until
the rerun verifies the paths.

When the selected model skill warns that large S3 media should be staged, copy
or extract the data once to platform-visible storage before creating launch
artifacts, then validate those staged paths with the same preflight helper.
Record the source URI and staged path in the run workspace so AutoML summaries
can distinguish data staging time from training/evaluation time.

For `local-docker` and `remote-docker`, always pass the selected image with
`--container-image` after resolving `container_image` from
`skill_info.yaml`/`versions.yaml`. The helper verifies Docker reachability,
NVIDIA Container Toolkit registration, GPU memory, selected-image architecture
compatibility when known, and a GPU-visible smoke container before launch. For
`remote-docker`, pass `--docker-host` or export `DOCKER_HOST`; the helper queries
GPUs and validates bind-mounted paths through the remote daemon instead of using
local host state. If the selected or smoke image is not present on the target
Docker host, ask before pulling it or rerun with `--pull-smoke-image` after
approval.

When a model skill lists annotation-level required fields, pass them with
`--json-required-field <path-label>=<field>[,<field>...]` so schema/data
content issues fail during preflight rather than inside the first training
container. Do not add required annotation fields from old failure history; only
enforce fields documented as required by the current model skill.
For local JSON/JSONL annotation paths, the helper prints `records=<N>`; use the
train annotation count as `automl_settings["train_sample_count"]` for
sample-count-sensitive AutoML runs before recommendations are generated.
If the model skill documents a run-local patch strategy for a missing required
field, create the patched copy in the current run workspace, update the spec
paths to that copy, and rerun the content check before launch. Do not ask the
user to mutate source datasets unless the model skill says patching is
impossible.

Do not use `--skip-platform-access` for a real launch. That flag is only for
dry environment checks or for cases where the user has already provided explicit
manual proof of platform and storage access. If the helper cannot verify remote
API, CLI, cluster, or object-store access, treat preflight as failed and do not
generate launch artifacts.

For SLURM:

1. Require `SLURM_USER`, `SLURM_HOSTNAME`, a partition intent, and one of
   `SSH_KEY_PATH` or `SSH_AUTH_SOCK`. If the user says to use the cluster
   default partition, pass an empty partition/omit the partition directive; do
   not substitute a site-specific value such as `batch`.
   Use the selected platform helper's `Resource defaults` for runtime values.
   For the packaged SLURM defaults, generate launchers with
   `SLURM_TIME_HOURS=4` and `SLURM_TIMEOUT_HOURS=3.8`; never invent a
   12-hour default for the 4-hour partition list.
   Launching the orchestrator with `nohup` or in the background is allowed for
   durability, but it does not satisfy chat monitoring by itself. After launch,
   keep a foreground chat-side polling loop attached until terminal state or
   explicit detach.
2. Split comma-separated `SLURM_HOSTNAME`, resolve hosts where possible, and
   require passwordless `ssh -o BatchMode=yes` to at least one host.
3. If SSH fails, do not offer several equivalent choices. Ask for
   `SSH_KEY_PATH=/path/to/private_key` and show the passwordless setup steps:
   create a key if needed with
   `ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519`; install it with
   `ssh-copy-id -i ~/.ssh/id_ed25519.pub <SLURM_USER>@<login-host>`; trust the
   host with `ssh-keyscan -H <login-host> >> ~/.ssh/known_hosts`; set
   `chmod 600 ~/.ssh/id_ed25519`; verify with
   `ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 <SLURM_USER>@<login-host> 'hostname'`;
   then rerun with `SSH_KEY_PATH=~/.ssh/id_ed25519`.
4. After SSH passes, validate dataset annotation/media paths on the remote login
   host with `test -e` or an equivalent read-only command.
5. Only then create runner scripts, specs, workspaces, or submit jobs.
6. For multi-GPU Slurm jobs, rely on the SDK Slurm backend to request
   `--gpus-per-node=<N>`. Do not generate manual `--gpus=<N>` sbatch snippets;
   that can spread GPUs across nodes and leave allocated GPUs idle.
7. For full-matrix or multi-node launches, submit one smoke job first. Launch
   the full matrix only after the smoke reaches training, emits the requested
   metric/status record, and shows expected GPU utilization.

For AutoML status, prefer structured controller/brain state and job metadata
(`active_jobs.json`, `.automl/controller/*.json`, result JSON, and
`results_dir/train/status.json`) before scanning raw logs. Parse logs only as a
fallback or when the user specifically asks for log-level investigation.

For local Docker, validate Docker/GPU access and local dataset paths before
writing launch artifacts. For Brev and Kubernetes, validate API or
cluster access plus object-storage credentials and `aws s3 ls` readability for
`s3://` inputs before writing launch artifacts. For mounted shared-storage or
PVC paths on those remote platforms, require manual proof that the path is
mounted into the job environment; the helper fails closed rather than accepting
unverified remote mount paths.

## Runtime And Configuration Review

Before any side-effecting launch, show a concise review:

- selected platform and exact container image
- GPU ids/count and nodes, including any GPUs avoided because they are already
  occupied
- dataset roots or direct spec paths, with sample counts when available
- important model/workflow overrides that differ from template defaults
- estimated runtime and the assumptions behind it
- monitoring interval and whether chat-side monitoring will stay attached

For AutoML, also show the algorithm, metric/direction, recommendation budget,
search parameters, ranges, and generated/default recommendation details as
described in `skills/applications/tao-run-automl/SKILL.md`. Ask for confirmation after
this review. If the user supplied a time limit, flag any plan that exceeds it
and offer concrete reductions before launch.
