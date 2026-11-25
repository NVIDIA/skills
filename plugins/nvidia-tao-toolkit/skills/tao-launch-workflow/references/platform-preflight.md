# Platform Preflight — per-platform detail

The `tao-launch-workflow` skill points here for the full per-platform preflight steps.

First verify the bank path and required shared helpers. Reuse the plugin's
`TAO_SKILL_BANK_PATH` when set. The bank is the `nvidia-tao-toolkit` plugin
folder, which holds `skills/`, `scripts/`, `catalog/` and `versions.yaml`. When
`TAO_SKILL_BANK_PATH` is not already set (no session hook exported it, as in
Codex), set it before the check to the plugin folder that contains this skill,
the folder two levels above this skill's folder, when that folder passes the
check. The first lines of the block do that once `<this-skill-folder>` is
replaced with the folder that holds this skill's `SKILL.md`. A checkout of the
repository is a bank only when it contains `plugins/nvidia-tao-toolkit`, and
the bank is then that folder, never the root of the checkout. When it has
none, install the plugin instead, as described in `tao-setup`. Individual
skill installs do not include shared helper scripts. Run:

```bash
# TAO_SKILL_BANK_PATH not set? Set it to the plugin folder that contains this skill: two levels above this skill's folder.
skill_dir="<this-skill-folder>"   # the folder that holds this skill's SKILL.md
if [ -z "${TAO_SKILL_BANK_PATH:-}" ] && [ -f "$skill_dir/../../catalog/skill-layers.json" ]; then
  TAO_SKILL_BANK_PATH="$(cd -P "$skill_dir/../.." && pwd -P)"
fi
export TAO_SKILL_BANK_PATH="${TAO_SKILL_BANK_PATH:-$HOME/tao-skill-bank/plugins/nvidia-tao-toolkit}"
for item in versions.yaml scripts/resolve_tao_model.py scripts/resolve_tao_image.py scripts/check_tao_launch_preflight.py scripts/redact_secrets.py scripts/tao_job_record.py scripts/skill_layers.py catalog/skill-layers.json; do
  [ -e "$TAO_SKILL_BANK_PATH/$item" ] || { echo "Incomplete TAO skill bank: missing $item under $TAO_SKILL_BANK_PATH. Set TAO_SKILL_BANK_PATH to the nvidia-tao-toolkit plugin folder that contains this skill (two levels above this skill's folder); a repository checkout without plugins/nvidia-tao-toolkit is not a bank." >&2; exit 1; }
done
# The shared helpers import PyYAML; stop here when python3 cannot import it.
python3 -c 'import yaml' 2>/dev/null || { echo "MISSING: PyYAML (python3 cannot import yaml; the shared helper scripts need it). Install it with: python3 -m pip install pyyaml" >&2; exit 1; }
```

Run the selected platform's preflight checks before any launch artifact is
created.

Use the bank path verified by `tao-launch-workflow` before calling the packaged
preflight helper. Stop if the checkout check failed:

```bash
"$TAO_SKILL_BANK_PATH/scripts/check_tao_launch_preflight.py" \
  --skill-bank "$TAO_SKILL_BANK_PATH" \
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
   12-hour default. A longer wall time is the user's choice and must fit
   the limit of their partition.
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
