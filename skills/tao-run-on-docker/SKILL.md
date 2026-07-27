---
name: tao-run-on-docker
description: Docker conventions for running NVIDIA GPU container workloads — NGC authentication, --gpus flag, mount patterns,
  env-var passthrough, container inspection, data-root relocation for split-disk hosts, and common error modes. Use when
  another skill requires running an nvcr.io container or any docker run command on a GPU host. Trigger keywords — docker,
  docker run, nvcr.io, NGC, --gpus, nvidia-container-toolkit, container image, docker login, docker pull.
license: Apache-2.0
compatibility: Requires NVIDIA driver branch 580, CUDA Toolkit 13.0, Docker, and NVIDIA Container Toolkit 1.19.0.
metadata:
  version: "0.1.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- platform
- docker
---

# Docker for NVIDIA GPU Workloads

This skill documents the generic Docker conventions that GPU container workloads rely on. Model and data skills specify **what** image and **what** command to run; this skill covers **how** to run docker in a way that satisfies GPU + NVIDIA container requirements.

Sources: official Docker CLI reference (<https://docs.docker.com/reference/cli/docker/>) and NVIDIA Container Toolkit docs.

## Prerequisites

1. **Host GPU runtime** — NVIDIA driver branch 580, CUDA Toolkit 13.0, and NVIDIA Container Toolkit 1.19.0. Check with the `tao-setup-nvidia-gpu-host` skill before any GPU workflow starts.
2. **Docker** — `docker --version` must return ≥ 20.10. Install: <https://docs.docker.com/engine/install/>.
3. **NGC API key** for `nvcr.io/*` pulls. Get from <https://ngc.nvidia.com/>.

```bash
TAO_SKILL_BANK_ROOT="${TAO_SKILL_BANK_ROOT:-$PWD}"
SETUP_SCRIPT="${TAO_SKILL_BANK_ROOT}/platform/tao-setup-nvidia-gpu-host/scripts/setup-nvidia-gpu-host.sh"

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

## NGC authentication

```bash
echo "$NGC_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

Persists in `~/.docker/config.json` across reboots. Re-run on `unauthorized` errors.

## `docker run` — canonical flags

```bash
docker run \
  --gpus all \                        # all GPUs (requires nvidia-container-toolkit)
  --rm \                              # delete container after exit (image is preserved)
  --shm-size=8g \                     # shared mem for torchrun / DataLoader
  -v /host/data:/data \               # bind-mount input
  -v /host/results:/results \         # bind-mount output
  -e HF_TOKEN -e NGC_KEY \            # env-var passthrough (values from parent shell)
  <image> \
  <command>
```

Notes:

- `--gpus '"device=0,1"'` — specific GPUs (double-quote-escaped). Without nvidia-container-toolkit: `could not select device driver "" with capabilities: [[gpu]]`.
- `--rm` — clean up the container at exit; omit when you want `docker logs` after exit.
- `--shm-size=8g` — torchrun + PyTorch DataLoaders exhaust the default 64 MB `/dev/shm` otherwise; size it for multi-GPU training and raise (e.g. `16g`) if you still hit `Bus error`.
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
docker run -d --name <worker> \
  --gpus all --shm-size=8g \
  -v <mounts...> -e <envs...> \
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
docker system df                # disk usage
docker system prune -a --volumes # reclaim space — destructive, removes unused images + volumes
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

**`no space left on device`** — root volume full. `docker system df` to inspect; relocate `data-root` (above) or `docker system prune -a --volumes`.

**`Bus error` / `DataLoader worker exited unexpectedly`** — `/dev/shm` too small. Increase shared memory with `--shm-size` (e.g. `--shm-size=16g`).

**`permission denied` on bind-mounted paths** — container UID ≠ host UID. Either `-u $(id -u):$(id -g)`, or pre-create host files owned by the host user, or `chmod 777` (dev only).

**`Error: No such container: <name>` after `docker run -d`** — container crashed on startup. `docker ps -a` shows exited; `docker logs <name>` for cause. Drop `--rm` while debugging.

## Scope boundary

This skill covers the *how* of running docker on a GPU host. Platform-specific layering (how to get onto the host, dispatch via a CLI wrapper) lives in:

- `tao-skill-bank:tao-run-on-brev` — running docker via `brev exec` on a Brev instance
- `tao-skill-bank:tao-run-platform` — optional Python layer wrapping docker invocations with Job handles, state persistence, and S3 I/O

Model and data skills specify **what** image and command; they defer to this skill for the **how**.

