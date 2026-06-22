# SLURM Execution Monitoring And SDK

Container execution, resource mapping, monitoring, cancellation, multi-node behavior, SDK usage, retries, and failure modes.

Load this file only when the compact `SKILL.md` points here for the current task. If this reference conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the compact/current source wins.

## Contents

- Resource Mapping
- Monitoring
- Cancellation
- Multi-node training (distributed)
- What the SDK generates
- Cluster requirements for multi-node
- Reference reading
- Optional: via the TAO SDK
- Auto-retry for infrastructure failures
- Failure Modes

## Container Execution

`tao-core` uses the SLURM handler to run TAO containers through Pyxis/Enroot:

1. Stage compact JSON files for specs, environment, and cloud metadata under
   `<job_dir>/specs`, `<job_dir>/env`, and `<job_dir>/meta`.
2. Optionally convert the Docker image to a cached SQSH image with
   `srun -n1 -p <conversion_partition> enroot import`.
3. Write an sbatch script under `<job_dir>/sbatch/job_<job_id>.sbatch`.
4. Submit `sbatch --export=ALL <script>`.
5. Run the container with `srun --container-image=<image> --container-mounts=/lustre`.

Image formats accepted by the handler:

- `/path/to/image.sqsh`
- `registry#image:tag`
- `docker://registry#image:tag`
- ordinary `registry/image:tag`, which is converted to Pyxis form when needed

SQSH conversion is cached by image name. For `:latest` images, cached SQSH is
used unless `force_reconvert_latest` is enabled.

## Resource Mapping

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

For multi-node jobs, the sbatch script exports `WORLD_SIZE`, `MASTER_ADDR`,
`MASTER_PORT`, `NODE_RANK`, and `NUM_GPU_PER_NODE`. Cosmos-RL has special
multi-node role handling for controller, policy, and rollout workers.

## Monitoring

- Scheduler status comes from the stored SLURM job id via `squeue` or `sacct`.
- TAO terminal status comes from `status.json` in the shared results folder.
- If the user enabled chat monitoring, continue polling at the requested
  interval while the job is `PENDING`, `RUNNING`, or otherwise non-terminal.
  Do not stop after a fixed elapsed time such as 30 minutes; long queue waits
  are normal on shared GPU partitions.
- Do not send a final response for a non-terminal SLURM job when chat
  monitoring is enabled. A final response is a detach action; use it only if
  the user asked to detach/stop or the job reached terminal state.
- Logs are read over SSH from:

```text
<job_dir>/slurm-logs/<slurm_job_name>-<slurm_job_id>/main.out
<job_dir>/slurm-logs/<slurm_job_name>-<slurm_job_id>/main.err
```

Status mapping:

- `PENDING` -> `Pending`
- `RUNNING` or `COMPLETING` -> `Running`
- `COMPLETED` -> check `status.json`
- `FAILED`, `BOOT_FAIL`, `DEADLINE`, `OUT_OF_MEMORY`, `NODE_FAIL` -> retry if
  logs match retriable infrastructure patterns, otherwise `Error`
- `CANCELLED`, `PREEMPTED`, `REVOKED` -> `Canceled`
- `TIMEOUT` -> `Error`
- `SUSPENDED`, `STOPPED` -> `Paused`

## Cancellation

Cancel by looking up `backend_details.slurm_metadata.slurm_job_id` and running
`scancel <slurm_job_id>` over SSH. Treat missing or already terminated SLURM
jobs as successful cancellation.

## Multi-node training (distributed)

SLURM is the platform of choice for large multi-node runs — pass `num_nodes > 1` and the SDK handles the sbatch directives + PyTorch-distributed env vars automatically.

```python
job = sdk.create_job(
    image='nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt',
    command='torchrun --nnodes=$WORLD_SIZE --nproc-per-node=$NUM_GPU_PER_NODE '
            '--node-rank=$NODE_RANK --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT '
            'train.py',
    gpu_count=8,           # GPUs per node
    num_nodes=4,           # 4 × 8 = 32 GPUs total
    inputs={'/data/train.json': 'lustre:///lustre/.../coco/train.json'},
    outputs=['/results/'],
)
```

### What the SDK generates

The handler builds an `sbatch` script with:

```
#SBATCH --nodes=N                    # node count
#SBATCH --ntasks-per-node=1          # one container per node (Pyxis spawns the GPU procs inside)
#SBATCH --ntasks=N                   # total tasks across the job
#SBATCH --gres=gpu:G                 # G GPUs per node
#SBATCH --wait-all-nodes=1           # don't start until all N nodes are allocated
```

Then exports the rendezvous env vars before `srun --container-image=...` launches the container on each node. These match the TAO PyTorch container contract (`nvidia_tao_pytorch/core/entrypoint.py`):

| Env var | Value | Read by |
|---|---|---|
| `WORLD_SIZE` | `N` (= node count, TAO's misnamed convention) | TAO container entrypoint |
| `NUM_GPU_PER_NODE` | `G` | TAO container entrypoint |
| `NODE_RANK` | `$SLURM_NODEID` | TAO container entrypoint, torchrun |
| `MASTER_ADDR` | first hostname from `scontrol show hostname $SLURM_JOB_NODELIST` | TAO container entrypoint, torchrun |
| `MASTER_PORT` | `29500` | TAO container entrypoint, torchrun |

```bash
export WORLD_SIZE=N
export NUM_GPU_PER_NODE=G
export MASTER_PORT=29500
NODELIST=$(scontrol show hostname $SLURM_JOB_NODELIST)
export MASTER_ADDR=$(echo $NODELIST | cut -d' ' -f1)   # first node = rank-0 / master
export NODE_RANK=$SLURM_NODEID                          # SLURM provides this per-node
```

`SLURM_JOB_NODELIST` and `SLURM_NODEID` come from SLURM itself — no manual registration step.

For TAO entrypoints (`dino train -e spec.yaml`, etc.) the container's entrypoint reads `WORLD_SIZE` + `NUM_GPU_PER_NODE` and constructs the torchrun command internally. For raw `torchrun` commands, use the standard PyTorch flags pointing at these env vars.

### Cluster requirements for multi-node

- **Pyxis + Enroot** must be installed on the cluster for `srun --container-image` to work. (Standard on DGX SuperPOD; check with your cluster admin elsewhere.)
- **InfiniBand / NVLink** is recommended for performance — set `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME` via `env_vars` if the defaults don't pick the right interface.
- **Shared filesystem** (Lustre) for staging the entrypoint script, env files, and results. Set `SLURM_BASE_RESULTS_DIR`.

### Reference reading

- SLURM multi-node + sbatch: <https://slurm.schedmd.com/sbatch.html>
- Pyxis (NVIDIA's SLURM container plugin): <https://github.com/NVIDIA/pyxis>
- Enroot (NVIDIA's container runtime for SLURM/Pyxis): <https://github.com/NVIDIA/enroot>
- PyTorch distributed (env-var rendezvous): <https://pytorch.org/docs/stable/elastic/run.html>
- NCCL networking tuning (NCCL_SOCKET_IFNAME, NCCL_IB_HCA): <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html>

## Optional: via the TAO SDK

The SDK install is covered in [Preflight](#preflight) — `pip install
'nvidia-tao-sdk[slurm]'`. Use it when you want Job handles, the
sbatch/`squeue`/`sacct` plumbing handled for you, run-folder durability via
`ActionWorkflow`, **or convenient cloud-storage I/O** (the SDK's
`build_entrypoint` inlines `script_runner` and dispatches `s3://`,
`hf_model://`, and `ngc://` URIs to the right downloader; without the SDK you
either pre-stage the data on Lustre or call `fsspec` / `huggingface-cli`
yourself).

When the SDK is in scope, read `tao-skill-bank:tao-run-platform` for the `SlurmSDK`
kwarg reference (`num_nodes`, `partition`, `account`), `build_entrypoint`,
and `ActionWorkflow`.

> **Use Lustre, not S3, for SLURM job inputs.** SLURM's scheduler enforces a
> GPU-idle timeout: the GPU allocation starts the moment your job is
> dispatched, and a long `s3://` download at the top of the script will burn
> minutes (or tens of minutes for large datasets) before training begins. The
> scheduler can kill the job for being GPU-idle, and the cluster bills you for
> the wasted allocation either way. Stage data onto the cluster's shared
> filesystem first and reference it as `lustre:///...` (or a plain absolute
> path the compute nodes can read). S3 / HF / NGC pre-fetch is fine for *small*
> auxiliary inputs (model checkpoints, configs); avoid it for training
> datasets. K8s/Brev don't have this constraint because they don't
> share SLURM's scheduler-idle policy.

```python
from tao_sdk.platforms.slurm import SlurmSDK
from tao_sdk.script_runner import build_entrypoint

ep = build_entrypoint(
    command='dino train -e {config_path}',
    specs=specs,                                           # config-mode (spec rewriting)
    job_id='dino-train-1',
)

sdk = SlurmSDK()  # reads SLURM_USER, SLURM_HOSTNAME, SLURM_BASE_RESULTS_DIR from env
job = sdk.create_job(
    image='nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt',
    command=ep['command'],
    gpu_count=8,
    num_nodes=2,                                           # multi-node supported
    partition='batch',                                     # optional override
    account='myproject',                                   # optional override
)

status = sdk.get_job_status(job.id)
logs = sdk.get_job_logs(job.id, tail=200)
```

The SDK takes care of staging the entrypoint script to Lustre, generating the
`sbatch` script with Pyxis `srun --container-image`, and parsing
`squeue`/`sacct` for status. Without the SDK, drive `sbatch` and `srun`
yourself.

### Auto-retry for infrastructure failures

Auto-retry is **fully automatic** — submit once, the SDK handles the rest. A
background `JobMonitor` thread (started in `SlurmSDK.__init__`) polls
`squeue`/`sacct` every `poll_interval` seconds (default 30s). When it sees an
*infrastructure-looking* failure it re-`sbatch`'s the already-staged remote
script and keeps watching, up to `MAX_JOB_RETRIES = 10` retries. The
user-facing `Job.id` is stable across retries; only the underlying SLURM job
id rotates. There is no `Job.retry()` / `Job.wait()` API to call — polling
and resubmission both happen in the background.

A failure is classified as retriable when:

- SLURM reports `NODE_FAIL` or `BOOT_FAIL`, **or**
- The job's logs match one of the retriable patterns (NCCL transport timeouts,
  CUDA driver init failures, GPU/IB link-down, OOM-killer reaping the node, et
  cetera — see `RETRIABLE_ERROR_PATTERNS` in the handler).

Plain training failures (`FAILED` with no matching pattern) are surfaced
immediately — no retry — so a broken spec doesn't silently consume 10 GPU
allocations.

State is persisted to `tao_session_state.db`, so if the user's process exits
between submit and completion, a later `SlurmSDK(state_file=...)` rehydrates
the job and resumes monitoring (and retrying) from where the previous process
left off.

In addition, `#SBATCH --requeue` is set by default (controlled by the
`SLURM_USE_REQUEUE` env var, defaults to `true`), so SLURM itself will
re-queue the job on `NODE_FAIL` or pre-emption *before* the handler-level
retry loop ever sees it. Set `SLURM_USE_REQUEUE=false` to opt out.

## Failure Modes

**SSH auth failure**: The passwordless-login setup in [Prerequisites](#prerequisites)
is incomplete. Check `SLURM_USER`, `SLURM_HOSTNAME`, `SSH_KEY_PATH`, key
permissions (`chmod 600`), `known_hosts` entries for every login host, and
whether the key is mounted into the service container. Re-run the
`ssh -o BatchMode=yes ...` verification step from the Prerequisites section to
confirm the fix before resubmitting.

**Local dataset path rejected**: Convert the data path to `lustre:///...` or
copy the dataset onto the cluster's shared filesystem.

**SQSH conversion timeout**: Increase `sqsh_conversion_timeout_minutes`, use a
smaller image, or pre-stage the SQSH image in the cache directory.

**Pyxis or Enroot unavailable**: The generated sbatch script depends on
`srun --container-image`. Ask the cluster admin to enable Pyxis/Enroot or use a
different platform.

**Bad node or transient GPU failure**: The handler retries infrastructure-like
failures such as CUDA driver errors, missing GPUs, NCCL/RDMA failures, Xid
errors, and node failures up to the configured retry limit.
