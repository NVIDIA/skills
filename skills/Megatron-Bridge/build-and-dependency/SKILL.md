---
name: build-and-dependency
description: Dev environment setup for Megatron Bridge — container-based development, uv package management, lockfile regeneration, adding dependencies, Slurm container usage, and common build pitfalls.
when_to_use: Setting up a dev environment, adding or removing dependencies, regenerating uv.lock, running inside containers, Slurm container setup, 'uv sync fails', 'ModuleNotFoundError', 'lockfile conflict'.
---

# Build and Dependency

Two core principles: **build and develop inside containers**, and **always use uv**.

## Why Containers

Megatron Bridge depends on CUDA, NCCL, PyTorch with GPU support, Transformer
Engine, and optional components like TRT-LLM, vLLM, and DeepEP. Installing
these on a bare host is fragile and hard to reproduce. The project ships
production-quality Dockerfiles that pin every dependency.

**Use the container as your development environment.** This guarantees:
- Identical CUDA / NCCL / cuDNN versions across developers and CI.
- `uv.lock` resolves the same way locally and in CI (the lockfile is
  Linux-only; it cannot be regenerated on macOS).
- GPU-dependent operations work out of the box.

## Container Options

### Option 1: NeMo Framework Container (fastest)

Find available tags at https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags

```bash
skopeo list-tags docker://nvcr.io/nvidia/nemo \
  | python3 -c "import sys,json,re; tags=json.load(sys.stdin)['Tags']; [print(t) for t in sorted((t for t in tags if re.match(r'^\d{2}\.\d{2}', t)), reverse=True)]"
```

```bash
docker run --rm -it --gpus all --shm-size=24g \
  nvcr.io/nvidia/nemo:<tag> \
  bash
```

### Option 2: Build the Megatron Bridge Container

See @docker/README.md for build commands, build arguments, and the full NeMo-FW image stack.

### Running the Container

```bash
docker run --rm -it -w /opt/Megatron-Bridge \
  -v $(pwd):/opt/Megatron-Bridge \
  -v $HOME/.cache/uv:/root/.cache/uv \
  --gpus all \
  --shm-size=24g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  megatron-bridge:latest \
  bash
```

Mounting `$HOME/.cache/uv` avoids re-downloading wheels on every run.

## Containers on Slurm

On Slurm clusters with Enroot/Pyxis, pass containers directly to `srun`:

```bash
srun --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --no-container-mount-home \
  bash -c "cd /opt/Megatron-Bridge && uv run --no-sync python ..."
```

If you **bind-mount a custom source tree** into the container, only rank 0
should sync while others wait:

```bash
if [ "$SLURM_LOCALID" -eq 0 ]; then uv sync; else sleep 10; fi
```

Note: `--no-container-mount-home` is an `srun` flag, not an `#SBATCH` directive.
Set `UV_CACHE_DIR` to shared storage to avoid filling `/root/.cache/`.

## Always Use uv

**Never use `pip install`, `conda`, or bare `python`** — always go through `uv`.
All `uv` commands must be run inside a container. Never install or upgrade
dependencies outside the CI container.

### Essential Commands

| Task | Command |
|---|---|
| Install all deps from lockfile | `uv sync --locked` |
| Install with all extras and dev groups | `uv sync --locked --all-extras --all-groups` |
| Run a Python command | `uv run python script.py` |
| Run distributed training | `uv run python -m torch.distributed.run --nproc_per_node=N script.py` |
| Add a new dependency | `uv add <package>` |
| Add an optional dependency | `uv add --optional --extra <group> <package>` |
| Regenerate the lockfile | `uv lock` (Linux/container only) |
| Install pre-commit hooks | `uv run --group dev pre-commit install` |

### Adding Dependencies

Submit dependency changes as a **separate PR** before the feature PR:

```bash
# Optional dependency (preferred)
uv add --optional --extra <group> <package>

# Required dependency (needs strong justification — affects all downstream)
uv add <package>
```

Commit both modified files:

```bash
git add pyproject.toml uv.lock
git commit -s -m "[build] chore: add <package>"
```

### Regenerating uv.lock

The lockfile is Linux-only (resolves CUDA wheels). Run inside Docker:

```bash
docker run --gpus all --rm \
  -v $(pwd):/opt/Megatron-Bridge \
  megatron-bridge:latest \
  bash -c 'cd /opt/Megatron-Bridge && uv lock'
```

### Switching MCore Branches

```bash
# Switch to dev branch
./scripts/switch_mcore.sh dev
uv sync              # without --locked

# Switch back to main
./scripts/switch_mcore.sh main
uv sync --locked     # lockfile matches again
```

## Quick Start

```bash
# 1. Clone and init submodules
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge megatron-bridge
cd megatron-bridge
git submodule update --init 3rdparty/Megatron-LM

# 2. Build the container
docker build -f docker/Dockerfile.ci --target megatron_bridge -t megatron-bridge:latest .

# 3. Start a dev shell
docker run --rm -it -v $(pwd):/opt/Megatron-Bridge --gpus all --shm-size=24g megatron-bridge:latest bash

# 4. Install pre-commit hooks (inside container)
uv run --group dev pre-commit install

# 5. Sanity check
uv run python -m torch.distributed.run --nproc_per_node=1 \
  scripts/training/run_recipe.py \
  --recipe vanilla_gpt_pretrain_config \
  train.train_iters=5 train.global_batch_size=8 train.micro_batch_size=4 \
  scheduler.lr_warmup_iters=1 scheduler.lr_decay_iters=5 \
  logger.log_interval=1
```

## Common Pitfalls

| Problem | Cause | Fix |
|---|---|---|
| `uv sync --locked` fails on macOS | Lockfile resolves CUDA wheels that don't exist on macOS | Run inside Docker or on a Linux machine |
| `ModuleNotFoundError` after pip install | pip installed outside uv-managed venv | Use `uv add` + `uv sync`, never bare `pip install` |
| `uv sync --locked` fails after MCore branch switch | Lockfile generated against main MCore | Use `uv sync` (without `--locked`) on dev |
| `uv: command not found` inside container | Container doesn't have uv | Use the `megatron-bridge` image built from `Dockerfile.ci` |
| `No space left on device` during uv ops | Cache fills container's `/root/.cache/` | Set `UV_CACHE_DIR` to shared/persistent storage |
| Pre-commit fails with ruff errors | Code style violations | Run `uv run ruff check --fix . && uv run ruff format .` |
