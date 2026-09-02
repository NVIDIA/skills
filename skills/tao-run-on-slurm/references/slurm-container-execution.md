# SLURM Container Execution, Monitoring, Multi-node, SDK, And Failures

Container execution steps, monitoring, status mapping, cancellation, multi-node env-var/sbatch detail, the TAO SDK path, the Lustre-not-S3 rule, auto-retry, and failure modes. If this reference conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the compact/current source wins.

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

## SQSH Conversion And Caching

Pyxis pulls Docker images through enroot on the compute node. Without a
pre-converted SQSH, every job pays the pull/extract cost inside its GPU
allocation — on clusters with GPU-idle reapers that can kill the job outright.
The SDK therefore converts once on a CPU partition and caches the result.
Keep SQSH conversion enabled; do not disable it to route around conversion
failures — fix the conversion instead.

**Environment knobs** (read at `SlurmSDK` construction):

| Env var | Default | Notes |
|---|---|---|
| `SLURM_USE_SQSH` | `true` | Leave enabled; direct pulls burn GPU allocation time. |
| `SLURM_SQSH_CACHE_DIR` | base results dir | Where `.sqsh` files land. |
| `SLURM_CONVERSION_PARTITION` | `cpu` | Must have wall-time ≥ conversion time. Large TAO images (9+ layers) take >30 min — pick a long-limit CPU partition (e.g. `cpu_long`) when the default partition caps at ~30 min. |
| `SLURM_CONVERSION_TIMEOUT_MINUTES` | `30` | Raise to 120 for large images. |
| `SLURM_CONVERSION_MEMORY_GB` | `32` | Extraction memory. |
| `SLURM_ENROOT_TEMP_PATH` | `<cache_dir>/.enroot-tmp` | Layer-extraction workspace. **Must be on a filesystem with xattr support** — some shared filesystems (certain Lustre configs) reject enroot's whiteout conversion with "Operation not permitted". Node-local `/tmp` is usually safe; beware shared-path symlinks that resolve to a restricted filesystem, which make an override a silent no-op. |
| `SLURM_FORCE_RECONVERT_LATEST` | `false` | Re-convert `:latest` tags. |

**Cache semantics:** conversion runs once per image name, then every job — and
every future session — reuses the cached file. The SDK verifies the SquashFS
magic bytes (`hsqs`) before reuse, so partial files from killed conversions
are rejected and re-converted automatically; no manual cleanup needed.
Concurrent sessions dedupe via a deterministic job name (`tao-sqsh-<hash>`) —
a second session waits on the in-flight conversion instead of double-writing.

**Monitoring a conversion:**

```bash
squeue -u $USER -n tao-sqsh-<hash>            # scheduler state
sattach <slurm_job_id>.0                       # live stdout (Ctrl+C detaches, job unaffected)
ls -lh <cache_dir>/<image>.sqsh*               # .partial appears in the final squashfs-write phase
```

The phase sequence in the live log is: authenticate → fetch manifest →
download layers → extract layers → convert whiteouts → create squashfs.
Time-limit kills usually land in extract; xattr failures land in convert
whiteouts.

**Pre-staging manually** (avoids the first job paying conversion latency):

```bash
sbatch -p <long_cpu_partition> -A <account> -t 2:00:00 --mem=64G \
  --wrap="enroot import -o <cache_dir>/<name>.sqsh docker://<registry>#<image>:<tag>"
```

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
    image='nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt',  # versions-key: images.tao_toolkit.pyt
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
    image='nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt',  # versions-key: images.tao_toolkit.pyt
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

Auto-retry is automatic in the SDK. `SlurmSDK` starts a monitor that polls
`squeue`/`sacct`, keeps the user-facing `Job.id` stable, and resubmits the
staged script for infrastructure-looking failures such as `NODE_FAIL`,
`BOOT_FAIL`, NCCL transport timeouts, CUDA driver init failures, GPU/IB
link-down, OOM-killer node reaping, Xid errors, and similar retriable patterns.

Plain training failures surface immediately so a broken spec does not consume
the retry budget. State persists in `tao_session_state.db`, and
`#SBATCH --requeue` is enabled by default via `SLURM_USE_REQUEUE=true`.

## Failure Modes

**SSH auth failure**: Check `SLURM_USER`, `SLURM_HOSTNAME`, `SSH_KEY_PATH`, key
permissions, `known_hosts`, and key mounts. Re-run the
`ssh -o BatchMode=yes ...` verification before resubmitting.

**Local dataset path rejected**: Convert it to `lustre:///...` or copy it onto shared storage.

**SQSH conversion timeout / whiteout failure**: Two failure modes — a
conversion-partition wall-time limit shorter than the conversion time
(`CANCELLED ... DUE TO TIME LIMIT` during layer extraction), or a shared
filesystem without xattr support rejecting whiteout conversion
(`enroot-aufs2ovlfs: ... Operation not permitted`). Both are fixed with the
environment knobs in [SQSH Conversion And Caching](#sqsh-conversion-and-caching):
`SLURM_CONVERSION_PARTITION`/`SLURM_CONVERSION_TIMEOUT_MINUTES` for the former,
`SLURM_ENROOT_TEMP_PATH` (point it at node-local scratch) for the latter.
Partial SQSH files are auto-rejected via magic-byte validation; no manual
cleanup needed.

**Pyxis or Enroot unavailable**: The generated sbatch script depends on
`srun --container-image`. Ask the cluster admin to enable Pyxis/Enroot or use a
different platform.

**Bad node or transient GPU failure**: The handler retries infrastructure-like
failures such as CUDA driver errors, missing GPUs, NCCL/RDMA failures, Xid
errors, and node failures up to the configured retry limit.
