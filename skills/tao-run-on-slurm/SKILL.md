---
name: tao-run-on-slurm
description: Remote SLURM GPU cluster execution over SSH with sbatch/srun, Pyxis/Enroot containers, and Lustre-backed
  results. Use when running TAO training/eval/inference jobs on an on-prem or DGX SLURM cluster. Trigger phrases include
  "run on SLURM", "submit sbatch", "DGX SLURM cluster", "Pyxis/Enroot container", "Lustre dataset".
license: Apache-2.0
compatibility: Requires SSH access to a SLURM login node (passwordless via key auth) and SLURM_USER + SLURM_HOSTNAME env vars.
  The TAO SDK with the slurm extra (pip install 'nvidia-tao-sdk[slurm]') is needed only if you want Job handles, S3 I/O wrapping,
  or run-folder durability via ActionWorkflow.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- platform
- slurm
---

# SLURM

Remote GPU compute platform for clusters managed by SLURM. Jobs are submitted
from the TAO service or SDK host to a login node over SSH, staged on a shared
filesystem, submitted with `sbatch`, and executed with `srun` container support.

## When to use

Use SLURM when the user has access to a managed GPU cluster, shared Lustre
storage, and scheduler-owned GPU allocation. Do not use SLURM for local files
that exist only on the agent machine; data and outputs must be reachable from
the cluster.

## Preflight + SSH

Confirm `SLURM_USER` and `SLURM_HOSTNAME` are exported and passwordless SSH to a
login host works (`ssh -o BatchMode=yes`). Optionally install the TAO SDK
wrapper for Job handles + S3 wrapping (`nvidia-tao-sdk[slurm]`, on public PyPI).
For private `nvcr.io` images, install `~/.config/enroot/.credentials` on the
cluster once per (cluster, user): Pyxis/Enroot does not read `NGC_KEY` from the
job env, and without persistent credentials, auth-gated pulls fail with "Could
not process JSON input" at job startup. Install it via the `printf | ssh`
heredoc so the `NGC_KEY` value never lands in shell history, intermediate files,
or chat output; never `cat`/`echo` the value.

If a preflight check fails, the agent prompts the user to authorize the
install/fix via Bash. Pip-installable Python requirements are the exception:
install them automatically, then rerun preflight.

See `references/slurm-ssh-credentials.md` for the full preflight script, the
enroot-credentials heredoc, prerequisite key setup (keypair, `ssh-copy-id`,
`known_hosts`, container key mounts, 2FA handling), and the SSH failure
remediation prompt.

## Storage

Use shared-filesystem URIs, not local or `file://` paths; `tao-core` rejects
local/file paths for remote backends.

- `lustre:///absolute/path` for user-provided datasets on Lustre.
- `slurm://` paths may appear in microservices metadata and are converted to
  Lustre paths before the container starts.

Accept either dataset roots (model skills map them to required files) or direct
spec-key paths. After SSH succeeds and before generating scripts, `test -e` each
required dataset path from the login host; if it fails, stop and ask for
corrected paths or staged data rather than producing scripts that fail in the
first training job. See `references/slurm-ssh-credentials.md` for root vs.
direct-spec modes, backend details, and the results-dir default.

## Container execution

`tao-core` runs TAO containers through Pyxis/Enroot:

1. Stage compact JSON files for specs, environment, and cloud metadata under
   `<job_dir>/specs`, `<job_dir>/env`, and `<job_dir>/meta`.
2. Optionally convert the Docker image to a cached SQSH image with
   `srun -n1 -p <conversion_partition> enroot import`.
3. Write an sbatch script under `<job_dir>/sbatch/job_<job_id>.sbatch`.
4. Submit `sbatch --export=ALL <script>`.
5. Run the container with `srun --container-image=<image> --container-mounts=/lustre`.

Accepted image formats: `/path/to/image.sqsh`, `registry#image:tag`,
`docker://registry#image:tag`, and ordinary `registry/image:tag` (converted to
Pyxis form when needed). SQSH conversion is cached by image name; for `:latest`
images the cached SQSH is reused unless `force_reconvert_latest` is enabled.

## Monitoring and cancellation

- Scheduler status comes from the stored SLURM job id via `squeue`/`sacct`;
  TAO terminal status comes from `status.json` in the shared results folder.
- While chat monitoring is enabled, keep polling at the requested interval for
  any non-terminal job (`PENDING`, `RUNNING`, or otherwise). Do not stop after a
  fixed elapsed time such as 30 minutes; long queue waits are normal on shared
  GPU partitions.
- Do not send a final response for a non-terminal SLURM job when chat
  monitoring is enabled. A final response is a detach action; use it only if the
  user asked to detach/stop or the job reached terminal state.
- Logs are read over SSH from
  `<job_dir>/slurm-logs/<slurm_job_name>-<slurm_job_id>/main.out` and `.err`.
- Cancel by looking up `backend_details.slurm_metadata.slurm_job_id` and running
  `scancel <slurm_job_id>` over SSH. Treat missing or already terminated jobs as
  successful cancellation.

Status mapping:

- `PENDING` -> `Pending`
- `RUNNING` or `COMPLETING` -> `Running`
- `COMPLETED` -> check `status.json`
- `FAILED`, `BOOT_FAIL`, `DEADLINE`, `OUT_OF_MEMORY`, `NODE_FAIL` -> retry if
  logs match retriable infrastructure patterns, otherwise `Error`
- `CANCELLED`, `PREEMPTED`, `REVOKED` -> `Canceled`
- `TIMEOUT` -> `Error`
- `SUSPENDED`, `STOPPED` -> `Paused`

## Required inputs

Ask for these in the SLURM intake; see `references/slurm-ssh-credentials.md`
for the full credential list, microservices schema keys, and defaults.

- **SLURM_USER** (required): SSH username for the login node.
- **SLURM_HOSTNAME** (required): Comma-separated login hostnames for failover.
- **SLURM_PARTITION** (required): Partition list for GPU submission. Packaged
  default `polar,polar3,polar4,grizzly`, treated as 4-hour queues.
- **SSH_KEY_PATH** (preferred, expected before launch): private key for
  non-interactive public-key auth. Ask for this first in remediation; prefer it
  over the `SSH_AUTH_SOCK` agent-socket fallback.
- **SLURM_BASE_RESULTS_DIR** (optional): base shared-filesystem path; default
  `/lustre/fsw/portfolios/edgeai/users/<your-dir>` (your per-user Lustre dir).
- **SLURM_ACCOUNT** (usually required by site policy): account for `#SBATCH --account`.

Do not ask for `SLURM_ACCOUNT` or `SLURM_BASE_RESULTS_DIR` in the initial
intake unless the user says their site requires an account, wants a custom
results root, or the workflow cannot proceed without overriding defaults.

## Resource defaults

Defaults from `tao-core`:

- `num_nodes`: 1
- `num_gpus`: 4
- `max_num_gpus_per_node`: 8
- `cpus_per_task`: 16
- `time_hours`: 4
- `timeout_hours`: 3.8
- `max_time_hours`: 4
- `container_mounts`: `/lustre`
- `use_requeue`: true
- `use_sqsh`: true

When generating launchers or wrapper scripts for SLURM, set the wall-time
defaults explicitly from the packaged platform resource defaults:

```bash
export SLURM_TIME_HOURS="${SLURM_TIME_HOURS:-4}"
export SLURM_TIMEOUT_HOURS="${SLURM_TIMEOUT_HOURS:-3.8}"
```

Do not default to 12 hours on SLURM. If the user supplies a longer
`SLURM_TIME_HOURS`, verify that the selected partition supports it before
submitting. For the packaged default partition list
`polar,polar3,polar4,grizzly`, reject requests above 4 hours and ask for a
different partition only if the user actually wants a longer wall time.

When `num_gpus` is greater than or equal to `max_num_gpus_per_node`, the
handler treats the request as exclusive per node and computes additional nodes
from total GPU count when necessary.

## Multi-node, SDK, and retries

For multi-node jobs (`num_nodes > 1`), the SDK builds the sbatch directives and
exports the PyTorch-distributed rendezvous env vars automatically: `WORLD_SIZE`,
`NUM_GPU_PER_NODE`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` (29500).
TAO entrypoints read `WORLD_SIZE` + `NUM_GPU_PER_NODE` and build torchrun
internally. Cosmos-RL has special multi-node role handling for controller,
policy, and rollout workers.

**Use Lustre, not S3, for SLURM job inputs.** The GPU allocation starts the
moment the job is dispatched, so a long `s3://` download at the top of the
script burns the allocation, can get the job killed for GPU-idle, and is billed
either way. Stage training data on the shared filesystem first and reference it
as `lustre:///...`. S3/HF/NGC pre-fetch is fine for small auxiliary inputs
(checkpoints, configs), not training datasets. K8s/Brev do not share this
scheduler-idle constraint.

Auto-retry of infrastructure failures (`NODE_FAIL`, `BOOT_FAIL`, NCCL transport
timeouts, CUDA driver init failures, GPU/IB link-down, OOM-killer node reaping,
Xid errors) is automatic in the SDK, with a stable user-facing `Job.id` across
retries. Plain training failures surface immediately so a broken spec does not
consume the retry budget. `#SBATCH --requeue` is enabled by default via
`SLURM_USE_REQUEUE=true`.

See `references/slurm-container-execution.md` for the full multi-node
env-var/sbatch directive detail and table, cluster requirements, the optional
TAO SDK path (`SlurmSDK`, `build_entrypoint`, `ActionWorkflow`) with code, the
Lustre-not-S3 rule in full, and the failure-mode checklist;
`references/slurm-execution-sdk.md` covers the `MAX_JOB_RETRIES` retry budget.
When the SDK is in scope, read `tao-skill-bank:tao-run-platform` for the
`SlurmSDK` kwarg reference.

## References

- `references/slurm-ssh-credentials.md` — preflight script, SSH/key setup,
  enroot credentials, full credential list, backend details, storage rules,
  SSH remediation prompt.
- `references/slurm-container-execution.md` — container execution steps,
  monitoring, status mapping, cancellation, multi-node detail, SDK use,
  Lustre-not-S3, auto-retry, failure modes.
- `references/slurm-preflight-storage.md` — extended preflight/storage notes.
- `references/slurm-execution-sdk.md` — extended execution/SDK notes.
- `references/detailed-guide.md` — navigation map for the split references.
