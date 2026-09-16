---
name: tao-run-on-local-docker
description: Local or remote Docker execution for TAO SDK job containers using a Docker daemon with NVIDIA GPU runtime. Use
  when running TAO jobs on the current machine, a directly attached Docker host, or a remote GPU box exposed through
  DOCKER_HOST. Trigger phrases include "run locally", "local Docker", "remote Docker", "use my GPU", "run on my
  machine", "host Docker daemon".
license: Apache-2.0
compatibility: Requires NVIDIA driver branch 580, CUDA Toolkit 13.0, Docker, and NVIDIA Container Toolkit 1.19.0. The TAO SDK with the docker extra (pip install 'nvidia-tao-sdk[docker]') is needed only if you want Job handles, S3 I/O wrapping, or run-folder durability via ActionWorkflow.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- platform
- local
- docker
---

# Local Docker

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

Single-node execution platform that runs TAO jobs as named Docker containers on
a Docker daemon. The daemon can be local to the agent host or remote through
`DOCKER_HOST=ssh://user@host` / a Docker context. It is useful for development,
debugging, small runs, and workflows where a local coding agent submits jobs to
a remote GPU box.

Use local Docker when the data is local to the Docker host or accessible through
mounted volumes/cloud credentials. Do not use it for remote cluster scheduling,
multi-node training, or jobs that need SLURM queueing.

Use remote Docker when the agent is running on a workstation or laptop but the
Docker daemon and GPUs are on another single GPU server. In remote Docker mode,
all local filesystem paths in specs are interpreted on the remote Docker host,
not on the agent machine.

## Preflight

The workflow must verify the host GPU runtime before starting Docker jobs. If
the check fails, prompt the user to approve the install, run the printed install
command, and rerun the preflight.

```bash
# Host GPU runtime: NVIDIA driver 580, CUDA 13.0, NVIDIA Container Toolkit 1.19.0.
SB="${TAO_SKILL_BANK_PATH:-${TAO_SKILL_BANK_ROOT:-$PWD}}"
SETUP_SCRIPT="${SB}/skills/platform/tao-setup-nvidia-gpu-host/scripts/setup-nvidia-gpu-host.sh"

bash "$SETUP_SCRIPT" --backend docker --check-only || {
  echo "MISSING: TAO GPU host runtime is not ready."
  echo "After user approval, run:"
  echo "  bash \"$SETUP_SCRIPT\" --backend docker --install --yes"
  exit 1
}

# Mode 1 — direct docker (no Python). All you need is docker + the GPU runtime.
docker info >/dev/null 2>&1 || { echo "MISSING: docker daemon not reachable. Start Docker."; exit 1; }
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi >/dev/null 2>&1 || {
  echo "MISSING: NVIDIA Container Toolkit not installed/configured. See:"
  echo "  bash \"$SETUP_SCRIPT\" --backend docker --install --yes"
  exit 1
}

# Mode 2 — TAO SDK wrapper. Adds Job handles, S3 I/O wrapping, ActionWorkflow.
# Skip this block if Mode 1 is sufficient for the user's request.
# When Mode 2 is in scope, read `tao-skill-bank:tao-run-platform` for the DockerSDK
# kwarg contract, build_entrypoint, and monitoring patterns.
# nvidia-tao-sdk is on public PyPI; the pin below is stamped from the release manifest.
PIN="nvidia-tao-sdk[docker]==7.1.0rc42"  # versions-key: wheels.tao_sdk_docker
python -c "import tao_sdk" 2>/dev/null || python -m pip install "$PIN"
python -c "import docker" 2>/dev/null || python -m pip install "$PIN"
python -c "import tao_sdk, docker"

# DockerSDK attaches every job container to ${DOCKER_NETWORK:-tao_default}.
# Create the network if it is missing; the operation is local and idempotent.
DOCKER_NETWORK_NAME="${DOCKER_NETWORK:-tao_default}"
docker network inspect "$DOCKER_NETWORK_NAME" >/dev/null 2>&1 || \
  docker network create "$DOCKER_NETWORK_NAME" >/dev/null
```

If a check fails, the agent prompts the user to authorize the install/fix via Bash before proceeding. Pip-installable Python requirements and Docker network creation above are exceptions: install/create them automatically, then rerun preflight.

## Credentials

There are no platform credentials required beyond access to the Docker daemon.

Optional environment:

- **DOCKER_HOST**: Optional Docker daemon URL. If unset, the SDK uses the
  Docker Python client's normal environment/default socket resolution. Required
  for the `remote-docker` platform option.
- **DOCKER_NETWORK**: Docker network for job containers. Default is
  `tao_default`.
- **DOCKER_USERNAME**: Registry username. Default is `$oauthtoken` for NGC.
- **NGC_KEY**: Used when pulling private images from `nvcr.io`.
- **HOST_SSH_PATH**: Mounted into AutoML brain containers when they need SSH keys
  to monitor remote SLURM child jobs.
- **ACCESS_KEY**, **SECRET_KEY**, **S3_ENDPOINT_URL**, **S3_BUCKET_NAME**:
  Optional S3-compatible storage settings for jobs that still read/write cloud
  storage from a local container.

## Launch Preflight

Before generating scripts or starting containers:

1. Verify the Docker daemon is reachable, NVIDIA Container Toolkit is registered
   as a Docker runtime, GPUs and driver version are reported, and a smoke
   container can see GPUs before launch. For remote Docker, query GPUs through
   `docker run ... nvidia-smi` against the remote daemon; do not use local
   `nvidia-smi` from the agent machine.
2. Verify every local/file dataset annotation and media path exists on the
   Docker host.
3. Classify every bind mount as read-only or writable. Writable mounts must use
   the Docker host user's numeric UID:GID by default, and the container identity
   (`USER`/`LOGNAME`) plus HOME/framework-cache paths must be set and writable
   for that identity. Direct Docker must pass them explicitly (see "Non-root
   container identity"); the SDK prepares them under a writable `/results` bind
   (with an isolated `/tmp` fallback only for forced non-root jobs without one).
   For remote Docker, resolve the identity on the remote host rather than
   copying the agent laptop's numeric ids.
4. For `s3://` datasets/results, verify `ACCESS_KEY` and `SECRET_KEY` are set
   and the exact paths are readable with `aws s3 ls`. If `aws` is missing,
   report the missing dependency and ask before installing it; rerun preflight
   after installation.
5. Verify model-specific credentials such as `HF_TOKEN` before launch.
6. Check current GPU occupancy with `nvidia-smi` and avoid GPUs already used by
   other running jobs when the user requested that constraint. Show the selected
   GPU ids in the launch review.
7. For model/container combinations with known architecture limits, compare
   host GPU compute capability with the container stack before launch. If the
   selected image cannot JIT or run kernels for the host architecture, block
   early and ask for a compatible image or platform.

Use the packaged helper for these checks when possible:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/check_tao_launch_preflight.py \
  --platform local-docker \
  --container-image "<selected-image>" \
  --path train_annotation=/abs/path/to/annotations.json \
  --path train_media=/abs/path/to/media
```

For a remote Docker daemon, use the `remote-docker` platform and pass or export
`DOCKER_HOST`. The helper verifies remote GPU/runtime readiness and checks
remote-host dataset paths through read-only bind mounts:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/check_tao_launch_preflight.py \
  --platform remote-docker \
  --docker-host ssh://user@gpu-host \
  --container-image "<selected-image>" \
  --gpu-smoke-image ubuntu:22.04 \
  --path train_annotation=/remote/data/train/annotations.json \
  --path train_media=/remote/data/train
```

The `--path` values above must exist on the remote Docker host. Do not pass
paths that exist only on the local laptop or Codex host.

Resolve the UID:GID of the actual submitting user on the remote Docker host,
then pass that identity to the SDK explicitly. Do not reuse the client
laptop's UID:GID, and do not infer a container user from `stat` ownership of a
shared output directory: that directory may be `root:<shared-group>` or owned
by another group member.

```bash
REMOTE_RESULTS=/remote/results
# Use the same SSH account represented by DOCKER_HOST=ssh://user@gpu-host,
# or obtain these two values from the remote administrator.
REMOTE_UID="$(ssh user@gpu-host id -u)"
REMOTE_GID="$(ssh user@gpu-host id -g)"
case "$REMOTE_UID" in
  ''|*[!0-9]*|0)
    echo "A verified non-root remote submitting UID is required."
    exit 1
    ;;
esac
case "$REMOTE_GID" in
  ''|*[!0-9]*)
    echo "A verified numeric remote submitting GID is required."
    exit 1
    ;;
esac
TAO_DOCKER_CONTAINER_USER="$REMOTE_UID:$REMOTE_GID"
export TAO_DOCKER_CONTAINER_USER

# Prove that this exact identity can create and remove a child in the bind.
docker --host "$DOCKER_HOST" run --rm \
  --user "$TAO_DOCKER_CONTAINER_USER" \
  -v "$REMOTE_RESULTS:/ownership-probe" ubuntu:22.04 \
  sh -c 'p=/ownership-probe/.tao-write-delete-probe-$$; touch "$p" && rm "$p"' || {
  echo "Remote submitting identity cannot write/delete under $REMOTE_RESULTS."
  exit 1
}
```

### Non-root container identity

`--user <uid>:<gid>` is necessary but not sufficient. TAO images provision
non-root accounts only at UID 1000 (`ubuntu` and `taotoolkituser`, which
collide there), so every other numeric UID runs with no `/etc/passwd` entry.
That makes the failure invisible on a UID-1000 workstation and reproducible
everywhere else. `getpass.getuser()` reads `LOGNAME`/`USER`/`LNAME`/
`USERNAME` and only then falls back to `pwd.getpwuid()`, so with none of them
set the lookup raises before any TAO code runs:

```
File "/usr/lib/python3.12/getpass.py", line 169, in getuser
    return pwd.getpwuid(os.getuid())[0]
KeyError: 'getpwuid(): uid not found: 1002'
```

Torch reaches that call while initializing its inductor cache directory during
`import`, so the container exits 1 at startup. Docker also leaves `HOME=/` for
an unknown UID, which sends framework caches into image-owned paths.

Every direct-Docker launch that passes `--user` must therefore also pass the
identity and cache environment. These mirror what the SDK injects in
`docker_handler.py`; keep the two lists in sync when either changes.

```bash
HOST_UID="$(id -u)"; HOST_GID="$(id -g)"
TAO_HOME=/results/.tao-runtime/home        # must live on a writable mount
mkdir -p "$RESULTS_DIR/.tao-runtime/home"

docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$HOST_UID:$HOST_GID" \
  -e USER="$HOST_UID" -e LOGNAME="$HOST_UID" \
  -e HOME="$TAO_HOME" \
  -e XDG_CACHE_HOME="$TAO_HOME/.cache" \
  -e HF_HOME="$TAO_HOME/.cache/huggingface" \
  -e TORCH_HOME="$TAO_HOME/.cache/torch" \
  -e TRITON_CACHE_DIR="$TAO_HOME/.cache/triton" \
  -e TORCHINDUCTOR_CACHE_DIR="$TAO_HOME/.cache/torchinductor" \
  -e MPLCONFIGDIR="$TAO_HOME/.cache/matplotlib" \
  -v "$DATA_DIR:/data:ro" -v "$RESULTS_DIR:/results" -v "$SPECS_DIR:/specs:ro" \
  "$IMAGE" <action> train -e /specs/<spec>.yaml
```

Numeric `USER`/`LOGNAME` values are deliberate: they describe an identity that
genuinely has no passwd entry, and they are only consumed for cache-path
naming. Do not drop `--user` to work around a startup `getpwuid` failure — that
trades a startup error for root-owned outputs, which is the more expensive
failure to repair.

## Multi-GPU and multi-node

**Multi-node is not supported on local Docker.** One job runs on the local Docker daemon's host with no cross-host coordination.

Multi-GPU **on the local host** is supported via the NVIDIA Container Toolkit's `--gpus` flag (`--gpus all` or `--gpus '"device=0,1,2,3"'`). `DockerSDK.create_job(gpu_count=N)` plumbs through to `--gpus`. Single-host distributed init uses `localhost`; `torchrun --nproc-per-node=N` or PyTorch DDP work as usual.

## Backend Details

Use the SDK backend value `local-docker`. The local backend schema has no extra
backend details, so most routing is controlled by environment and job
parameters:

```json
{
  "backend_type": "local-docker",
  "num_gpu": 1
}
```

Following the Brev SDK design, platform/control-plane values stay in SDK
state and Docker labels. The SDK does not inject `BACKEND`, `HOST_PLATFORM`,
`MONGOSECRET`, `DOCKER_HOST`, or `DOCKER_NETWORK` into the training container.

## Container Execution

The TAO SDK local Docker handler starts containers through the Docker Python
client:

- Backend job name uses the `tao-job-<job_id>` form used by SDK handlers.
- Command is usually `["/bin/bash", "-c", "<job command>"]`.
- Containers run detached. The SDK keeps containers by default so status and
  logs remain inspectable, unless `DOCKER_AUTO_REMOVE=true`.
- With `run_as_user=None` (the default), the SDK maps a local job to the invoking
  UID:GID when it has an absolute writable `/results` bind, preserves local
  supplementary groups, and prepares HOME/framework caches under
  `/results/.tao-runtime/home`. `run_as_user=True` opts other local mount layouts
  into user mapping. If the SDK process itself is root, automatic mapping fails
  closed instead of mapping `0:0`; provide the verified submitting non-root
  UID:GID through `container_user`. `container_user` is also the explicit
  non-root Docker user override for remote hosts. `run_as_user=False` is the
  deliberate opt-out for an image proven to require root.
- `/dev/shm` is mounted as tmpfs.
- The configured Docker network is applied by the Docker daemon for the job
  container; it is not passed through as a process environment variable.
- Existing containers with the same job id are stopped and removed before a
  replacement starts.

For GPU access, the handler auto-detects the host type:

- Tegra or Jetson hosts use `runtime="nvidia"` plus
  `NVIDIA_VISIBLE_DEVICES` and `NVIDIA_DRIVER_CAPABILITIES=all`.
- Standard x86 hosts use Docker `device_requests` with GPU capabilities.

If `num_gpus` is `0`, no GPUs are assigned. If `num_gpus` is `-1`, all visible
GPUs are requested. Prefer explicit GPU counts for shared development machines.
When explicit device ids are available, prefer them over count-only selection
on shared machines so the launch does not steal GPUs occupied by other tasks.

## Storage

Local Docker accepts local and `file://` paths because the container runs on the
same Docker host. Make sure every path in the spec is either:

- mounted into the container by the handler or surrounding service,
- reachable from inside the container already, or
- a cloud URI with matching credentials.

For bind-mounted outputs, host-user ownership is a launch invariant, not a
permission-error workaround. Root containers commonly create checkpoint
subdirectories as `root:root` mode `0755`; the host user then cannot delete
files inside them even if the top-level output directory was pre-created.
Container auto-removal also leaves bind-mounted outputs untouched.

Only opt out of host-user mapping (`run_as_user=False`) when the selected image
demonstrably requires root. Record that exception in the launch review, isolate
its writable mounts, and normalize every output/cache mount back to the Docker
host UID:GID after all terminal exits and cancellations. For remote Docker,
pass the remote host's verified non-root identity through `container_user`;
never infer it from the client machine or output-directory owner. Do not begin
another experiment until ownership normalization
succeeds. If the agent lacks permission to perform or verify that repair, the
root-required image cannot be launched on local Docker.

AutoML's default checkpoint retention is stricter: its preflight rejects
`run_as_user=False`, named volumes, remote bind mounts, or an incompatible
explicit `container_user` before launching a trial, because the SDK cannot
guarantee host-side deletion. Use those routes for AutoML only when retention
is explicitly disabled and an external operator owns artifact cleanup.

For remote/shared filesystems, prefer the platform that owns that filesystem.
For example, use SLURM plus `lustre:///...` for Lustre paths on a cluster.

## Monitoring

- The SDK handler maps Docker container state directly: created -> Pending,
  running/restarting -> Running, paused -> Paused, exit code 0 -> Complete,
  nonzero exit -> Error.
- Logs come directly from the named container through the Docker Python client
  (`docker logs tao-job-<job_id>`).

If the container has exited, died, is being removed, or cannot be found, status
reconciliation treats the backend process as terminated.

## Cancellation

Cancellation stops the named container. GPU ownership is managed by Docker /
the NVIDIA runtime, not by TAO Core's local GPU manager.

## Optional: via the TAO SDK

If you want Job handles, S3 I/O wrapping via the SDK's `script_runner`, or
durability across sessions:

```python
import os

from tao_sdk.platforms.docker import DockerSDK

docker_host = os.environ.get('DOCKER_HOST', '')
is_remote = bool(docker_host) and not docker_host.startswith(('unix://', 'npipe://', '/'))
container_user = os.environ.get('TAO_DOCKER_CONTAINER_USER')
if is_remote and not container_user:
    raise RuntimeError('Set TAO_DOCKER_CONTAINER_USER to the remote output owner UID:GID')

sdk = DockerSDK()  # reads DOCKER_HOST, NGC_KEY, S3 creds from env
job = sdk.create_job(
    image='nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt',  # versions-key: images.tao_toolkit.pyt
    command='dino train -e /data/spec.yaml',
    gpu_count=1,
    mounts=[
        {'host_path': '/host/data', 'container_path': '/data', 'read_only': True},
        {'host_path': '/host/results', 'container_path': '/results'},
    ],
    container_user=container_user,
)

status = sdk.get_job_status(job.id)
logs = sdk.get_job_logs(job.id, tail=200)
```

This wraps the same `docker run` invocation under a `Job` handle. For S3 I/O,
call `build_entrypoint(...)` first and pass its command so `script_runner` can
perform the declared downloads/uploads. If you do not need job tracking or
that wrapper, use `docker run` directly — no SDK install required.

## Failure Modes

**Docker client not initialized**: Verify the Docker Python package is installed,
set `DOCKER_HOST` if you are not using the default local socket, and confirm the
process can talk to the daemon.

**GPU assignment failed**: Requested GPUs are unavailable, the NVIDIA Container
Toolkit is not configured, or the Docker daemon cannot create GPU device
requests. Use fewer GPUs, wait for another job to finish, or verify
`docker run --gpus ...` works on the host.

**Image pull auth failed**: Set a valid `NGC_KEY` for private `nvcr.io` images
or run `docker login nvcr.io -u '$oauthtoken'` on the Docker host.

**Container exited unexpectedly**: Check `docker logs tao-job-<job_id>`, the
configured `DOCKER_NETWORK`, and the command produced by the SDK action runner.

**`KeyError: getpwuid(): uid not found`**: The launch passed `--user` with a UID
that has no `/etc/passwd` entry in the image and did not pass `USER`/`LOGNAME`.
Add the identity and cache environment from "Non-root container identity";
do not fall back to running as root.

**Path missing inside container**: A local path on the host is not necessarily
mounted into the job container. Use a path convention supported by the action
runner or configure an explicit volume through the surrounding service.

**Root-owned bind-mounted results**: Stop launching new experiments, identify
every writable mount from `docker inspect`, and have the host administrator
repair existing ownership once. Future launches must use host UID:GID mapping
and writable HOME/cache redirects. `docker rm` and `DOCKER_AUTO_REMOVE` do not
repair or delete bind-mounted files.
