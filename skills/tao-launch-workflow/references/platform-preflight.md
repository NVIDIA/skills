# Platform Preflight — per-platform detail

The `tao-launch-workflow` skill points here for the full per-platform preflight steps.

Run the selected platform's preflight checks before any launch artifact is
created.

Prefer the packaged preflight helper when the needed inputs are available:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skill-bank}/scripts/check_tao_launch_preflight.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skill-bank} \
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
6. For multi-GPU Slurm jobs, request `--gpus-per-node=<N>` (the
   `tao-run-on-slurm` platform skill emits this in the sbatch launcher). Do not
   generate manual `--gpus=<N>` sbatch snippets; that can spread GPUs across
   nodes and leave allocated GPUs idle.
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
