---
name: tao-run-on-docker
description: Docker conventions for running NVIDIA GPU container workloads — NGC authentication, --gpus flag, mount patterns,
  env-var passthrough, container inspection, data-root relocation for split-disk hosts, and common error modes. Use when
  another skill requires running an nvcr.io container or any docker run command on a GPU host. Trigger keywords — docker,
  docker run, nvcr.io, NGC, --gpus, nvidia-container-toolkit, container image, docker login, docker pull.
license: Apache-2.0
compatibility: Requires NVIDIA driver 580 or newer, CUDA Toolkit 13.0 or newer, Docker, and NVIDIA Container Toolkit 1.19.0 or newer, unless the selected model declares different minimums in runtime_requirements.gpu_host.
metadata:
  version: "0.1.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- platform
- docker
---

# Docker for NVIDIA GPU Workloads

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

This skill documents the generic Docker conventions that GPU container workloads rely on. Model and data skills specify **what** image and **what** command to run; this skill covers **how** to run docker in a way that satisfies GPU + NVIDIA container requirements.

Sources: official Docker CLI reference (<https://docs.docker.com/reference/cli/docker/>) and NVIDIA Container Toolkit docs.

## Prerequisites

1. **Host GPU runtime** — by default, NVIDIA driver `>=580`, CUDA Toolkit `>=13.0`, and NVIDIA Container Toolkit `>=1.19.0`. If the selected model's `references/skill_info.yaml` declares `runtime_requirements.gpu_host`, pass those values to `tao-setup-nvidia-gpu-host` instead. Model requirements override the defaults for that workflow.
2. **Docker** — `docker --version` must return ≥ 20.10. Install: <https://docs.docker.com/engine/install/>.
3. **NGC API key** for `nvcr.io/*` pulls. Get from <https://ngc.nvidia.com/>.

```bash
SB="${TAO_SKILL_BANK_PATH:-${TAO_SKILL_BANK_ROOT:-$PWD}}"
SETUP_SCRIPT="${SB}/skills/platform/tao-setup-nvidia-gpu-host/scripts/setup-nvidia-gpu-host.sh"

bash "$SETUP_SCRIPT" --backend docker --check-only || {
  echo "MISSING: TAO GPU host runtime is not ready."
  echo "After user approval, run (append --yes for non-interactive agent runs):"
  echo "  bash \"$SETUP_SCRIPT\" --backend docker --install"
  exit 1
}

docker --version
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
[ -n "$NGC_KEY" ] || echo "NGC_KEY unset — cannot pull nvcr.io images"
```

If the selected model declares `runtime_requirements.gpu_host`, append the
corresponding `--min-driver-version`, `--min-cuda-version`, and
`--min-container-toolkit-version` values to both the check and any approved
install command. Do not apply one model's override to unrelated workflows.

## NGC authentication

```bash
echo "$NGC_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

Persists in `~/.docker/config.json` across reboots. Re-run on `unauthorized` errors.

## `docker run` — canonical flags

```bash
HOST_RESULTS=/host/results
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
HOST_USER_NAME="$(id -un)"
[ "$HOST_UID" -ne 0 ] || { echo "Refusing writable Docker launch as UID 0" >&2; exit 1; }
HOST_IDENTITY_ARGS=(--user "$HOST_UID:$HOST_GID")
for group_id in $(id -G); do
  [ "$group_id" = "$HOST_GID" ] || HOST_IDENTITY_ARGS+=(--group-add "$group_id")
done
mkdir -p "$HOST_RESULTS/.tao-runtime/home/.cache"/{huggingface,torch,triton,torchinductor,matplotlib}

docker run \
  --gpus all \
  --rm \
  --shm-size=8g \
  "${HOST_IDENTITY_ARGS[@]}" \
  -v /host/data:/data \
  -v "$HOST_RESULTS:/results" \
  -e HOME=/results/.tao-runtime/home \
  -e USER="$HOST_USER_NAME" -e LOGNAME="$HOST_USER_NAME" \
  -e XDG_CACHE_HOME=/results/.tao-runtime/home/.cache \
  -e HF_HOME=/results/.tao-runtime/home/.cache/huggingface \
  -e TORCH_HOME=/results/.tao-runtime/home/.cache/torch \
  -e TRITON_CACHE_DIR=/results/.tao-runtime/home/.cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/results/.tao-runtime/home/.cache/torchinductor \
  -e MPLCONFIGDIR=/results/.tao-runtime/home/.cache/matplotlib \
  -e HF_TOKEN -e NGC_KEY \
  <image> \
  <command>
```

Notes:

- `--gpus '"device=0,1"'` — specific GPUs (double-quote-escaped). Without nvidia-container-toolkit: `could not select device driver "" with capabilities: [[gpu]]`.
- `--rm` — clean up the container at exit; omit when you want `docker logs` after exit.
- `--shm-size=8g` — torchrun + PyTorch DataLoaders exhaust the default 64 MB `/dev/shm` otherwise; size it for multi-GPU training and raise (e.g. `16g`) if you still hit `Bus error`.
- `--user "$(id -u):$(id -g)"` — required by default whenever a bind mount is writable. It prevents root-owned checkpoint trees that the submitting host user cannot clean up.
- Refuse UID `0` for the canonical writable-bind path. If the launcher itself is root, obtain the verified non-root submitting UID:GID explicitly; never infer it from the output-directory owner.
- `--group-add <gid>` — preserve supplementary host-group access to shared datasets and workspaces. The canonical array adds every host group except the primary GID.
- `HOME`, `USER`, `LOGNAME`, and cache redirects — keep frameworks from writing to image-owned locations such as `/root` after the user override. Prepare these directories on the writable mount before launch. `USER`/`LOGNAME` are load-bearing, not cosmetic: an arbitrary `--user` UID has no `/etc/passwd` entry in the image, and torch 2.x calls `getpass.getuser()` at import (`torch/_dynamo` → inductor cache-dir setup) — with neither env var set the container crashes with `KeyError: 'getpwuid(): uid not found: <uid>'` before any workload code runs. Any non-empty name satisfies it; the name does not need to exist in the image.
- `-v host:container` — bind mount; the command references container paths only.
- `-e VAR` — passthrough from parent shell (no value needed if already set). Use this form for secrets.

## Container name collision

`docker run --name X` fails if a container named `X` already exists. Defensive pattern before reusing a name:

```bash
docker stop my-worker 2>/dev/null; docker rm my-worker 2>/dev/null
docker run --name my-worker ...
```

## Detached + exec pattern

For multi-step workflows on the same container (download → run → post-process), avoid restart cost:

```bash
HOST_RESULTS=/host/results
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
[ "$HOST_UID" -ne 0 ] || { echo "Refusing writable Docker launch as UID 0" >&2; exit 1; }
HOST_IDENTITY_ARGS=(--user "$HOST_UID:$HOST_GID")
for group_id in $(id -G); do
  [ "$group_id" = "$HOST_GID" ] || HOST_IDENTITY_ARGS+=(--group-add "$group_id")
done
mkdir -p "$HOST_RESULTS/.tao-runtime/home/.cache"/{huggingface,torch,triton,torchinductor,matplotlib}

docker run -d --name <worker> \
  --gpus all --shm-size=8g \
  "${HOST_IDENTITY_ARGS[@]}" \
  -v <host-data>:/data \
  -v "$HOST_RESULTS:/results" \
  -e HOME=/results/.tao-runtime/home \
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -e XDG_CACHE_HOME=/results/.tao-runtime/home/.cache \
  -e HF_HOME=/results/.tao-runtime/home/.cache/huggingface \
  -e TORCH_HOME=/results/.tao-runtime/home/.cache/torch \
  -e TRITON_CACHE_DIR=/results/.tao-runtime/home/.cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/results/.tao-runtime/home/.cache/torchinductor \
  -e MPLCONFIGDIR=/results/.tao-runtime/home/.cache/matplotlib \
  --entrypoint sh \
  <image> -c "tail -f /dev/null"

docker exec <worker> <step_1>
docker exec <worker> <step_2>

docker stop <worker> && docker rm <worker>
```

## Pull-if-missing idiom

```bash
docker image inspect <image> >/dev/null 2>&1 || docker pull <image>
```

## Labels for discovery

Tag containers for filtered listing later:

```bash
docker run --label tao-toolkit ...
docker ps --filter 'label=tao-toolkit'
```

## Mount patterns

The container expects its data at conventional paths defined by the image (often `/data`, `/results`, `/workspace/checkpoints`). The host side is arbitrary. The command inside docker run references container paths only.

### Writable-mount ownership invariant

For every writable bind mount, run as the submitting host UID:GID by default.
Pre-creating the mount root is not sufficient when a root container can create
deeper `0755` directories: deletion is controlled by the parent-directory
permissions, so those subtrees still become inaccessible to the host user.
Container `--rm` and `docker rm` remove container state only; neither deletes or
repairs bind-mounted checkpoints.

An image may run as root only when its documentation or a preflight proves that
host-user execution is incompatible. Treat this as an explicit launch
exception. Isolate its writable outputs and, after every terminal exit or
cancellation, normalize ownership before another experiment starts. For an
image with `/bin/sh` and `chown`, the post-run repair is:

```bash
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
docker run --rm --user 0:0 --entrypoint /bin/sh \
  -v /host/results:/owned-output \
  <same-approved-image> \
  -c 'chown -R "$1:$2" /owned-output' sh "$HOST_UID" "$HOST_GID"
```

Apply the repair to every writable output/cache mount. If the agent cannot run
or verify the ownership normalization, it must not use the root-required
exception. Never substitute `chmod 777` as the normal fix.

## Env-var conventions

Common passthrough vars for TAO-style workloads (the calling skill declares which it needs):

- `NGC_KEY` — `nvcr.io` pulls; some runtimes also read at runtime
- `HF_TOKEN` — gated HuggingFace model downloads
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL` — S3 I/O inside the container
- `WANDB_API_KEY` — optional W&B logging

Use `-e VAR` (no `=value`) when the var is in the parent shell. Avoid placing secrets on the command line.

Alternative GPU selection: `-e NVIDIA_VISIBLE_DEVICES=0,1` (or `all`) and `-e NVIDIA_DRIVER_CAPABILITIES=all` instead of `--gpus`. The `--gpus` flag is preferred on standard x86 hosts; the env-var form is older and is what `runtime=nvidia` (Tegra/Jetson) requires.

## Container inspection

```bash
docker ps                                # running containers only
docker ps -a                             # all containers, including exited
docker ps --filter status=running --format '{{.Names}} {{.Image}}'
docker logs <name_or_id>                 # stdout/stderr
docker logs -f <name_or_id>              # follow (tail -f equivalent)
docker logs --tail 100 <name_or_id>      # last N lines
docker inspect <name_or_id>              # full config, mounts, env, network, state (JSON)
docker inspect --format '{{.State.Status}}' <name_or_id>
docker stats                             # live CPU/mem/network/block I/O
docker stats --no-stream                 # one snapshot, non-interactive
```

`docker inspect` is the canonical source of truth for a container's mounts, env, cmd, network, and exit code. Use it to debug why a container isn't behaving as expected.

## Image management

```bash
docker pull <image>
docker image ls
docker system df                # Docker-managed image/layer/volume usage
```

Pull once per host; `docker run` reuses cached image. NVIDIA images are typically 5-40GB.

## Split-disk data-root relocation

Some cloud GPU providers ship with a small root volume + larger ephemeral. Docker writes to `/var/lib/docker` on root by default — large images fill it. Check:

```bash
df -h /         # root volume size/free
lsblk           # all block devices and mount points
```

If `/` is smaller than your total image footprint and there's a larger disk mounted elsewhere, relocate **before pulling images**:

```bash
sudo systemctl stop docker
sudo mkdir -p <large_volume_path>/docker
sudo rsync -aP /var/lib/docker/ <large_volume_path>/docker/
sudo mv /var/lib/docker /var/lib/docker.old

sudo tee /etc/docker/daemon.json <<'EOF'
{ "data-root": "<large_volume_path>/docker" }
EOF

sudo systemctl start docker
docker info | grep 'Docker Root Dir'
sudo rm -rf /var/lib/docker.old
```

## Networks (multi-container patterns)

For microservice containers that talk to each other by name, create a docker network and attach containers:

```bash
docker network create tao-net
docker run --network tao-net --name api ...
docker run --network tao-net --name worker ...   # can resolve `api` by name
```

Most TAO training workloads don't need this — single container per job.

## Common error modes

**`could not select device driver "" with capabilities: [[gpu]]`** — NVIDIA Container Toolkit missing or Docker is not configured for the NVIDIA runtime. Run `tao-setup-nvidia-gpu-host` with `--backend docker --install` after user approval (append `--yes` for a non-interactive agent run), then restart Docker.

**`unauthorized: authentication required`** on `docker pull` — NGC key invalid/missing. Re-run `docker login nvcr.io`.

**`no space left on device`** — first identify which filesystem and storage
class is full; bind-mounted training outputs are not counted by `docker system
df` and are not fixed by pruning Docker images:

```bash
df -h / /var/lib/docker <results_root>
docker system df
docker inspect <tao-container> --format '{{json .Mounts}}'
du -xhd1 <results_root> 2>/dev/null | sort -h
find <results_root> -maxdepth 3 -printf '%u:%g %m %s %p\n' 2>/dev/null | head
```

For a bind mount, clean only confirmed terminal job directories using the SDK
retention path or a reviewed ownership repair; never assume `docker system
prune` touches them. For Docker's own root, relocate `data-root` as described
above. `docker system prune -a --volumes` is destructive and may remove unused
images and volumes belonging to other workflows, so run it only after explicit
user approval and a reviewed `docker system df` inventory.

**`Bus error` / `DataLoader worker exited unexpectedly`** — `/dev/shm` too small. Increase shared memory with `--shm-size` (e.g. `--shm-size=16g`).

**`permission denied` on bind-mounted paths** — container UID ≠ host UID, or `HOME`/a framework cache still points to an image-owned directory. Use the canonical host UID:GID mapping and writable HOME/cache redirects above. For a documented root-required image, complete the mandatory post-run ownership normalization before retrying.

**`KeyError: 'getpwuid(): uid not found: <uid>'` at import of torch/torchvision** — the container runs as a `--user` UID with no `/etc/passwd` entry and no `USER`/`LOGNAME` env var, so `getpass.getuser()` falls through to `pwd.getpwuid()` at import time. `-e HOME=...` alone does not fix it. Keep the UID:GID mapping and launch with the canonical identity env block (`-e USER=... -e LOGNAME=...` + writable `HOME` + cache redirects). Do not work around it by running as root; that recreates the root-owned-outputs hazard.

**`Error: No such container: <name>` after `docker run -d`** — container crashed on startup. `docker ps -a` shows exited; `docker logs <name>` for cause. Drop `--rm` while debugging.

## Scope boundary

This skill covers the *how* of running docker on a GPU host. Platform-specific layering (how to get onto the host, dispatch via a CLI wrapper) lives in:

- `tao-skill-bank:tao-run-on-brev` — running docker via `brev exec` on a Brev instance
- `tao-skill-bank:tao-run-platform` — optional Python layer wrapping docker invocations with Job handles, state persistence, and S3 I/O

Model and data skills specify **what** image and command; they defer to this skill for the **how**.
