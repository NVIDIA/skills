---
name: tao-run-on-slurm
description: Remote SLURM GPU cluster execution over SSH with sbatch/srun, Pyxis/Enroot containers, and Lustre-backed
  results. Use when running TAO training/eval/inference jobs on an on-prem or DGX SLURM cluster. Trigger phrases include
  "run on SLURM", "submit sbatch", "DGX SLURM cluster", "Pyxis/Enroot container", "Lustre dataset".
license: Apache-2.0
compatibility: Requires SSH access to a SLURM login node (passwordless via key auth) and SLURM_USER + SLURM_HOSTNAME env vars.
  No nvidia-tao-sdk install is required; jobs are driven directly over ssh + sbatch/squeue/sacct/scancel.
metadata:
  author: NVIDIA Corporation
  version: "0.1.1"
allowed-tools: Read Bash
tags:
- platform
- slurm
---

# SLURM

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

Remote GPU compute platform for clusters managed by SLURM. Jobs are submitted
from the launch host to a login node over SSH, staged on a shared
filesystem, submitted with `sbatch`, and executed with `srun` container support.

## When to use

Use SLURM when the user has access to a managed GPU cluster, shared Lustre
storage, and scheduler-owned GPU allocation. Do not use SLURM for local files
that exist only on the agent machine; data and outputs must be reachable from
the cluster.

## Preflight + SSH

Confirm `SLURM_USER` and `SLURM_HOSTNAME` are exported and passwordless SSH to a
login host works (`ssh -o BatchMode=yes`).
The launch host needs `ssh`, not local `sbatch`, `srun`, Enroot, or a Lustre
mount. Preflight those scheduler, Pyxis, Enroot, and shared-storage dependencies
on the selected remote login/compute frame. Model-specific inspectors may be
streamed from the installed skill over SSH stdin; do not stage an ad-hoc source
patch or treat the launch host as the SLURM frame.
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

## Execution — the four verbs

`tao-run-on-slurm` is a platform **consumer**: it runs a spec-bundle over
`ssh + sbatch/squeue/sacct/scancel`, mutating only the job-record. Storage is
**tier A** (Lustre) — the dataset is staged to a shared path *before* submit and
read through Pyxis; never fetch S3 inside the allocation (the scheduler-idle
timeout kills GPU-idle jobs and bills the wasted time). `$BANK` =
`${TAO_SKILL_BANK_PATH}`; `$LOGIN` = a resolved `SLURM_HOSTNAME`.

### submit

1. **Reuse what's already staged — never redo (tier A):**
   - *Image:* `@@IMAGE@@` is a Lustre `.sqsh` — **reuse an existing one if present**
     (`ssh $LOGIN ls <sqsh>`); only if missing, convert once with `enroot import`
     (cached by name — see `references/slurm-container-execution.md`).
   - *Dataset:* **confirm it is already on Lustre** (`ssh $LOGIN test -e …`) and
     reference those paths; `tao-data-io` stages *only* a small auxiliary input
     that is not there yet — never re-stage existing data, and never the training
     set inside the allocation.
   Then author the spec at `<job_dir>/specs/spec.yaml` on Lustre with those paths.
2. **Credentials → sidecar (never inline):** if the run needs session creds
   (e.g. `HF_TOKEN`), write them to a mode-600 sidecar on Lustre and let the
   template shred it on exit; NGC image pulls use the one-time
   `~/.config/enroot/.credentials` (see `references/slurm-ssh-credentials.md`),
   not the job env:
   ```bash
   set -a; source /path/to/.env; set +a   # omit if already exported
   printf 'export HF_TOKEN=%s\n' "$HF_TOKEN" | ssh $LOGIN "umask 077; cat > <job_dir>/job_$JOB_ID.env"
   ```
3. **Open the record — mints the id, binds `results_dir` on Lustre, before launch:**
   ```bash
   JOB_ID=$("$BANK/scripts/tao_job_record.py" open --platform slurm --image "$IMAGE" \
     --network-arch "$ARCH" --action "$ACTION" --storage-tier A --results-root "$SLURM_BASE_RESULTS_DIR")
   ```
4. **Consume the optional model lifecycle.** If the validated spec-bundle has
   `execution`, preserve its order and semantics while mapping distributed
   intent to native SLURM/Pyxis. Stage only its checksum-closed
   `supporting_files`. The full generic lifecycle and staging contract is in
   `references/slurm-container-execution.md`.
5. **Render** `templates/slurm/singlenode.sbatch.tmpl` — substitute every
   `@@<NAME>@@` (`JOB_NAME=$JOB_ID`, `NUM_GPUS`, `CPUS_PER_TASK`, `TIME`, `LOG_DIR`,
   `IMAGE`, `CONTAINER_MOUNTS=<RUNTIME_SUPPLIED_MOUNTS>`, `COMMAND=<bundle command reading the shared-storage
   spec>`, `SBATCH_EXTRA=` account/partition lines, `ENV_FILE=` the sidecar path or
   empty, `EXTRA_ENV=` any cluster NCCL knobs) → `<job_dir>/sbatch/job_$JOB_ID.sbatch`.
   **Lint + syntax-check before submit:** `redact_secrets.py lint <sbatch>` must
   pass and `bash -n <sbatch>` must succeed.
6. **Submit + record RUNNING:**
   ```bash
   SLURM_ID=$(ssh $LOGIN "sbatch --parsable <job_dir>/sbatch/job_$JOB_ID.sbatch")
   "$BANK/scripts/tao_job_record.py" mark "$JOB_ID" --state RUNNING --backend-ref "$SLURM_ID"
   ```

A submit that skipped the gate or the open has no id — so it cannot launch.

### status

```bash
# sacct ANNOTATES states ("CANCELLED by 12345") and truncates them to the
# default column width, so a cancelled job reads back as "CANCELLED+" and
# matches nothing in the table below — reporting UNKNOWN instead of CANCELED.
# Widen the column, take the first word, drop the truncation marker.
st=$(ssh $LOGIN "sacct -j $SLURM_ID -X -n -o State%30" | awk '{print $1}' | tr -d '+')
# (use squeue while the job is still PENDING; sacct lags briefly after submit)
```

| SLURM state | vocab |
|---|---|
| `PENDING` | `PENDING` |
| `RUNNING` / `COMPLETING` | `RUNNING` |
| `COMPLETED` | `COMPLETE` (confirm `status.json` in `results_dir`) |
| `FAILED` / `TIMEOUT` / `OUT_OF_MEMORY` | `ERROR` (infra-vs-program classify → retry, M6) |
| `NODE_FAIL` / `BOOT_FAIL` | `ERROR`, `err_class=ERR_INFRA` (`--requeue` re-queues these) |
| `CANCELLED` / `PREEMPTED` / `REVOKED` | `CANCELED` |
| (not found) | `UNKNOWN` |

Native sub-state rides in the transition `message`. Poll at the chosen interval;
long queue waits are normal — do not stop on elapsed time.

### logs

```bash
ssh $LOGIN "tail -n ${N:-200} <log_dir>/$JOB_ID-$SLURM_ID/main.out"   # SLURM auto-creates the %x-%j subdir
```

### cancel

```bash
ssh $LOGIN "scancel $SLURM_ID"
"$BANK/scripts/tao_job_record.py" mark "$JOB_ID" --state CANCELED --source agent
```

Treat an already-terminated SLURM job as a successful cancel.

### Multi-node (nodes > 1)

Same four verbs, with three additions at submit:

1. **Render `templates/slurm/multinode.sbatch.tmpl`** instead of the single-node
   one — it's a strict superset (adds `--nodes` / `--wait-all-nodes` + the
   rendezvous block). `WORLD_SIZE` is the **node count** (TAO's misnomer); never
   change it to a global-rank count.
2. **NCCL probe first** — before the real job, run a cheap 2-node all-reduce
   (`scripts/nccl_allreduce_probe.py` under the container's torchrun) with a
   ~120s timeout. Before invoking torchrun, preserve the TAO rendezvous values
   as `TAO_NODE_COUNT=$WORLD_SIZE`,
   `TAO_GPUS_PER_NODE=$NUM_GPU_PER_NODE`, and
   `TAO_NODE_RANK=$SLURM_PROCID`; torchrun overwrites its standard
   `WORLD_SIZE` with the global process count. `NCCL_PROBE_OK` → proceed.
   **Timed out** (the collective hung)
   → set the cluster's NCCL knob in `EXTRA_ENV` and re-probe — on CS-OCI-ORD that
   is `export NCCL_P2P_DISABLE=1` (the intra-node P2P hang), often with
   `NCCL_SOCKET_IFNAME=eth0` / `NCCL_IB_DISABLE=1`. **Cache the working env per
   cluster** so later jobs skip the probe. Gate on **`gpus_per_node > 1` too** —
   the P2P hang triggers on a single node with 2+ GPUs.
3. Tier-A Lustre, sidecar creds, record, and lint are unchanged.

### Cosmos backend guardrails

Read [`references/cosmos-slurm-guardrails.md`](references/cosmos-slurm-guardrails.md)
before rendering a Cosmos command. It defines image staging, planner
materialization, Framework and Cosmos-RL launch contracts, worker/runtime
requirements, and exit/status handling.

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
2. Convert the Docker image to a cached SQSH image **before** the GPU job, with
   `srun -n1 -p <conversion_partition> enroot import`. This is a one-time cost
   per image, not an optional optimization — see *Acquire the image off the GPU
   allocation* below.
3. Write an sbatch script under `<job_dir>/sbatch/job_<job_id>.sbatch`.
4. Submit `sbatch --export=ALL <script>`.
5. Run the container with `srun --container-image=<image> --container-mounts=<RUNTIME_SUPPLIED_MOUNTS>`.

Accepted image formats: `/path/to/image.sqsh`, `registry#image:tag`,
`docker://registry#image:tag`, and ordinary `registry/image:tag` (converted to
Pyxis form when needed). SQSH conversion is cached by image name; for `:latest`
images the cached SQSH is reused unless `force_reconvert_latest` is enabled.

### Acquire the image off the GPU allocation

**The GPU is yours from the moment the allocation starts, not from when compute
begins.** Anything the job does before training — pulling a registry image,
converting it, fetching a dataset — runs on GPUs that are idle, billed, and
visible to the cluster's GPU-idle reaper. A first-time TAO pull plus enroot
conversion is minutes of that, which is long enough to be killed and long enough
to be expensive.

So the image must already be a local `.sqsh` when the GPU job starts. Passing a
`docker://` or `registry#image:tag` URI straight to `srun --container-image=`
makes Pyxis pull *and* convert inside the allocation — the exact trap. Convert
once on a **CPU partition**, then point every later job at the resulting file:

```bash
# One-time per image, on CPU — costs no GPU time.
ssh $LOGIN "test -e <sqsh>" || \
  ssh $LOGIN "srun --chdir=/tmp -n1 -c4 --mem=7200M \
    -p <cpu_partition> -t <minutes> \
    bash -c 'set -Eeuo pipefail
      export TMPDIR=/tmp
      export ENROOT_TEMP_PATH=/tmp/enroot-tao-\${SLURM_JOB_ID}
      export SLURM_ENROOT_TEMP_PATH=\${ENROOT_TEMP_PATH}
      mkdir -p \"\${ENROOT_TEMP_PATH}\"
      cd /tmp
      enroot import -o <sqsh> docker://<registry>#<image>:<tag>'"

# Every GPU job then references the file, never the registry.
srun --container-image=<sqsh> ...
```

The same rule governs data: stage it to Lustre before submit (tier A) rather
than fetching inside the allocation.

CS-OCI-ORD conversion uses `cpu_long`, 4 CPUs, 7200M memory, no exclusive node,
node-local Enroot temp paths, and at least 120 minutes. The execution reference
records the evidence and `QOSGrpMemLimit` recovery contract.

Partial conversions are self-detecting: the SQSH is validated by `hsqs` magic,
so a truncated file is rejected rather than silently used. Conversion runs once
and is then cached by image name.

**A failed conversion must not fall back to the registry image.** The tempting
recovery — pass `docker://…` to `srun` and let Pyxis handle it — puts the pull
back inside the GPU allocation, which is the cost the conversion existed to
avoid, and it does so precisely when something is already wrong. Treat a failed
or truncated conversion as fatal: fix it on the CPU partition and resubmit.

Diagnostic: if a job is unexpectedly slow to produce output, check what
`--container-image=` actually received. A registry URI there — rather than a
`.sqsh` path — means the pull happened on the GPUs.

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
- `SUSPENDED`, `STOPPED` -> `Running` (still scheduler-owned and may resume;
  the native sub-state rides in the transition message — same convention as
  docker `paused`)

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
  a shared-storage root supplied and verified at runtime.
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
- `container_mounts`: explicit source-to-target mounts supplied at runtime
- `use_requeue`: true
- `use_sqsh`: true

Launchers must use the packaged 4-hour wall and 3.8-hour child-timeout
defaults, never 12 hours. If the user supplies a longer
`SLURM_TIME_HOURS`, verify that the selected partition supports it before
submitting. For the packaged default partition list
`polar,polar3,polar4,grizzly`, reject requests above 4 hours and ask for a
different partition only if the user actually wants a longer wall time.

At or above `max_num_gpus_per_node`, allocate exclusive nodes and derive their
count from total GPUs.

## Multi-node and retries

For multi-node jobs (`num_nodes > 1`), the rendered
`templates/slurm/multinode.sbatch.tmpl` sets the sbatch directives and exports
the PyTorch-distributed rendezvous env vars: `WORLD_SIZE`, `NUM_GPU_PER_NODE`,
`NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` (29500). TAO entrypoints read
`WORLD_SIZE` + `NUM_GPU_PER_NODE` and build torchrun internally. Cosmos-RL has
special multi-node role handling for controller, policy, and rollout workers.
See the `### Multi-node (nodes > 1)` submit subsection above for the NCCL-probe
gate and per-cluster env caching.

**Use Lustre, not S3, for SLURM job inputs.** The GPU allocation starts the
moment the job is dispatched, so a long `s3://` download at the top of the
script burns the allocation, can get the job killed for GPU-idle, and is billed
either way. Stage training data on the shared filesystem first and reference it
as `lustre:///...`. S3/HF/NGC pre-fetch is fine for small auxiliary inputs
(checkpoints, configs), not training datasets. K8s/Brev do not share this
scheduler-idle constraint.

On an infrastructure failure (`NODE_FAIL`, `BOOT_FAIL`, NCCL transport timeouts,
CUDA driver init failures, GPU/IB link-down, OOM-killer node reaping, Xid
errors), classify infra-vs-program from the logs and create a new retry record
with `--retry-of` before re-submitting the staged workload (M6). Plain training
failures surface immediately so a broken spec does not consume the retry
budget. `#SBATCH --requeue` is enabled by default via
`SLURM_USE_REQUEUE=true`, so SLURM itself re-queues the job on `NODE_FAIL` or
pre-emption before any agent-level resubmit; workload contracts such as Cosmos
may require `--no-requeue`.

Treat an empty `sbatch --parsable` response or SSH disconnect as ambiguous:
reconcile by exact job name, never submit blindly, and validate inherited node
exclusions. The referenced execution guide defines the full decision table.
See `references/slurm-container-execution.md` for the full multi-node
env-var/sbatch directive detail and table, cluster requirements, the
Lustre-not-S3 rule in full, and the failure-mode checklist.

## References

- `references/slurm-ssh-credentials.md` — preflight script, SSH/key setup,
  enroot credentials, full credential list, backend details, storage rules,
  SSH remediation prompt.
- `references/slurm-container-execution.md` — container execution steps,
  monitoring, status mapping, cancellation, multi-node detail,
  Lustre-not-S3, retries, failure modes.
- `references/slurm-preflight-storage.md` — extended preflight/storage notes.
- `references/cosmos-slurm-guardrails.md` — Cosmos Framework and Cosmos-RL
  launch and status guardrails.
- `references/detailed-guide.md` — navigation map for the split references.
