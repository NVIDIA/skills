# SLURM Container Execution, Monitoring, Multi-node, And Failures

Container execution steps, monitoring, status mapping, cancellation, multi-node env-var/sbatch detail, the Lustre-not-S3 rule, retries, and failure modes. If this reference conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the compact/current source wins.

## Container Execution

`tao-core` uses the SLURM handler to run TAO containers through Pyxis/Enroot:

1. Stage compact JSON files for specs, environment, and cloud metadata under
   `<job_dir>/specs`, `<job_dir>/env`, and `<job_dir>/meta`.
2. Optionally convert the Docker image to a cached SQSH image with
   `srun -n1 -p <conversion_partition> enroot import`.
3. Write an sbatch script under `<job_dir>/sbatch/job_<job_id>.sbatch`.
4. Submit `sbatch --export=ALL <script>`.
5. Run the container with `srun --container-image=<image> --container-mounts=<RUNTIME_SUPPLIED_MOUNTS>`.

Image formats accepted by the handler:

- `/path/to/image.sqsh`
- `registry#image:tag`
- `docker://registry#image:tag`
- ordinary `registry/image:tag`, which is converted to Pyxis form when needed

SQSH conversion is cached by image name. For `:latest` images, cached SQSH is
used unless `force_reconvert_latest` is enabled.

### Model-owned action lifecycle

When the validated spec-bundle declares `execution`, keep its model-owned order
exactly: `pre_commands` -> primary `command` -> successful-only
`post_commands`. Bind only `{config_path}`, `{job_id}`, and `{results_dir}` in
commands, environment, and spec strings after the job record is opened. The
record-owned results directory is authoritative. Serialize the bound config
once and persist the producer-bundle and runtime-config SHA256 values. Export
declared non-secret `environment` plus platform runtime variables. For
`distributed.launcher=torchrun`, use one Pyxis
task per node and map `processes_per_node` to `--nproc-per-node`; SLURM owns
rendezvous and task/rank syntax. Run leader-scoped post-commands only on task
zero and preserve every failure in `completion.child_exit_code_path`.

For `supporting_files`, resolve sources relative to the producing skill root,
reject absolute paths and `..`, and stage only the closed set below
`<job_dir>/inputs/action-helpers/`. Verify every declared SHA256 and persist a
manifest of source, destination, SHA256, and size. Reuse only a byte-identical
manifest; never overwrite a different bundle. These are orchestration inputs,
not permission to shadow packages in the image or add a startup source patch.

### CS-OCI-ORD SQSH conversion profile

Use partition `cpu_long`, not `cpu`; the latter's roughly 30-minute wall can
kill TAO conversion and leave a truncated file. Request 4 CPUs, 7200M memory,
no exclusive node, and a timeout of at least 120 minutes. This profile is
preserved from successful conversion job `32370651` (21m28s elapsed).

Set `TMPDIR=/tmp` and both `ENROOT_TEMP_PATH` and
`SLURM_ENROOT_TEMP_PATH` to job-unique
`/tmp/enroot-tao-${SLURM_JOB_ID}`. Direct Enroot and Pyxis may read different
variables; node-local unique paths avoid cleanup races and unsupported shared
overlay whiteouts.

Do not inherit a training job's CPU count: omitted memory is charged per CPU on
this cluster, so oversized requests can remain pending with `QOSGrpMemLimit`.
Inspect `ReqCPUS` and `ReqMem` with `scontrol`/`sacct`, cancel the pending job,
and retry with a new job record and the explicit profile above. Do not wait for
the QOS condition to clear or move conversion into a GPU allocation.

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
- `SUSPENDED`, `STOPPED` -> `Running` (still scheduler-owned and may resume;
  the native sub-state rides in the transition message — same convention as
  docker `paused`)

## Cancellation

Cancel by looking up `backend_details.slurm_metadata.slurm_job_id` and running
`scancel <slurm_job_id>` over SSH. Treat missing or already terminated SLURM
jobs as successful cancellation.

## Multi-node training (distributed)

SLURM is the platform of choice for large multi-node runs — set `num_nodes > 1`
and render `templates/slurm/multinode.sbatch.tmpl`, which is a strict superset of
the single-node template: it adds the sbatch directives and PyTorch-distributed
rendezvous env vars below. For example, `NUM_GPUS=8` (GPUs per node) with
`WORLD_SIZE=4` node count gives 4 × 8 = 32 GPUs total; the training command is a
`torchrun` reading the exported env vars, e.g.:

```bash
torchrun --nnodes=$WORLD_SIZE --nproc-per-node=$NUM_GPU_PER_NODE \
  --node-rank=$NODE_RANK --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT \
  train.py
```

(TAO entrypoints such as `dino train -e spec.yaml` build the torchrun invocation
internally from `WORLD_SIZE` + `NUM_GPU_PER_NODE`.)

### What the rendered template generates

The rendered multi-node `sbatch` script has:

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

## Lustre, not S3, for job inputs

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

Stage the entrypoint/spec files to Lustre, render the `sbatch` script with Pyxis
`srun --container-image`, submit with `sbatch --parsable`, and parse
`squeue`/`sacct` for status — driving `sbatch`/`srun` directly over SSH.

### Retry for infrastructure failures

On an infrastructure-looking failure — `NODE_FAIL`, `BOOT_FAIL`, NCCL transport
timeouts, CUDA driver init failures, GPU/IB link-down, OOM-killer node reaping,
Xid errors, and similar retriable patterns — classify infra-vs-program from the
job logs (M6). A real resubmit gets a new job record with `--retry-of`; do not
reuse the previous record or its results directory. A scheduler-level requeue,
when the workload allows it, remains the same backend job and record.

If `sbatch --parsable` returns no id or the SSH connection is lost, do not
assume submission failed. Reconcile `squeue` and `sacct` by the exact unique job
name on every configured login host. Adopt exactly one match, stop and clean up
multiple matches, and resubmit only when no match appears after a bounded
accounting-propagation window. Validate every carried node exclusion against
`scontrol show nodes`; nonexistent exclusions can make the retry itself fail.

Plain training failures (`FAILED` with no matching pattern) surface immediately
so a broken spec does not consume the retry budget. `#SBATCH --requeue` is
enabled by default via `SLURM_USE_REQUEUE=true`, so SLURM itself re-queues the
job on `NODE_FAIL` or pre-emption before any agent-level resubmit; set
`SLURM_USE_REQUEUE=false` to opt out. A workload-specific contract can require
it off; the Cosmos training planner emits `#SBATCH --no-requeue`.

The launch skill owns classification and opens the new `--retry-of` record.
This platform snapshots eligible-node inventory, validates evidence-backed
exclusions, and owns `#SBATCH --exclude`. The producer may restore its sealed
semantic request but must rebase every writable path under the new record.
Never use a model-specific retry launcher or patch a rendered SBATCH file.

## Failure Modes

**SSH auth failure**: Check `SLURM_USER`, `SLURM_HOSTNAME`, `SSH_KEY_PATH`, key
permissions, `known_hosts`, and key mounts. Re-run the
`ssh -o BatchMode=yes ...` verification before resubmitting.

**Local dataset path rejected**: Convert it to `lustre:///...` or copy it onto shared storage.

**SQSH conversion timeout**: Increase `sqsh_conversion_timeout_minutes`, use a smaller image, or pre-stage the SQSH image.

**Pyxis or Enroot unavailable**: The generated sbatch script depends on
`srun --container-image`. Ask the cluster admin to enable Pyxis/Enroot or use a
different platform.

**Bad node or transient GPU failure**: The handler retries infrastructure-like
failures such as CUDA driver errors, missing GPUs, NCCL/RDMA failures, Xid
errors, and node failures up to the configured retry limit.
