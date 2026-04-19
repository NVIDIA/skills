---
name: developer-guide
description: Developer environment setup, CI/CD workflows, and CI failure debugging for Megatron Bridge. Covers container-based development, uv package management, pre-commit hooks, running tests, CI failure investigation, and common pitfalls. Use when onboarding, setting up a dev environment, troubleshooting build issues, investigating CI failures, or dealing with lockfile issues (corrupted, regenerating, or updating uv.lock).
---

# Developer Guide

This guide covers the recommended development workflow for Megatron Bridge.
Two core principles apply everywhere: **build and develop inside containers**,
and **always use uv** for package management.

---

## Why Containers

Megatron Bridge depends on CUDA, NCCL, PyTorch with GPU support,
Transformer Engine, and optional components like TRT-LLM, vLLM, and DeepEP.
Installing these on a bare host is fragile and hard to reproduce. The project
ships production-quality Dockerfiles that pin every dependency.

**Use the container as your development environment.** This guarantees:

- Identical CUDA / NCCL / cuDNN versions across all developers and CI.
- `uv.lock` resolves the same way locally and in CI (the lockfile is
  Linux-only; it cannot be regenerated on macOS).
- GPU-dependent operations (training, conversion, `uv lock`) work out of the
  box.

### Option 1: Use the NeMo Framework Container

The fastest way to get started is the pre-built
[NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags),
which ships with Megatron Bridge, Megatron-Core, and all GPU dependencies
pre-installed. No build step required:

```bash
docker run --rm -it --gpus all --shm-size=24g \
  nvcr.io/nvidia/nemo:latest \
  bash
```

### Option 2: Build the Megatron Bridge Container

If you need to test against your local source tree, build the image from the
repository root:

```bash
docker build \
  -f docker/Dockerfile.ci \
  --target megatron_bridge \
  -t megatron-bridge:latest \
  .
```

This builds the CI image with all dependencies installed via `uv sync --locked`.
See `docker/README.md` for the full NeMo Framework image stack
(fw-base -> megatron-bridge -> fw-final) and build argument reference.

Key build args:
- `BASE_IMAGE` — base PyTorch image (default: `nvcr.io/nvidia/pytorch:26.02-py3`)
- `MCORE_TRIGGERED_TESTING` — set to `true` when testing against a non-pinned MCore commit
- `UV_CACHE_PRUNE_ARGS` — optional args passed to `uv cache prune` during image build

### Running the Container

Interactive development shell:

```bash
docker run --rm -it -w /opt/Megatron-Bridge \
  -v $(pwd):/opt/Megatron-Bridge \
  --gpus all \
  --shm-size=24g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  megatron-bridge:latest \
  bash
```

### Containers on Slurm Clusters

On Slurm clusters with Enroot/Pyxis, containers are passed to `srun` directly:

```bash
srun --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --no-container-mount-home \
  bash -c "cd /opt/Megatron-Bridge && uv run --no-sync python ..."
```

If you use the built container (or the NeMo Framework container) as-is,
dependencies are already installed and no `uv sync` is needed. If you
**bind-mount a custom Megatron Bridge source tree** into the container
(e.g., for development), you need to `uv sync` so dependencies match
your local `pyproject.toml` and `uv.lock`. In that case, only rank 0
should sync while other ranks wait:

```bash
if [ "$SLURM_LOCALID" -eq 0 ]; then uv sync; else sleep 10; fi
```

Other key points:

- `--no-container-mount-home` is an **srun flag**, not an `#SBATCH` directive.
- Set `UV_CACHE_DIR` to shared storage to avoid filling the container's
  `/root/.cache/`.

---

## Always Use uv

Megatron Bridge uses [uv](https://docs.astral.sh/uv/) as its sole package
manager. The `uv.lock` file is checked into the repository for reproducible
builds. **Never use `pip install`, `conda`, or bare `python`** — always go
through `uv`.

**Never install or upgrade dependencies outside the CI container.** All `uv`
commands must be run inside a `megatron-bridge` container — either one you
built locally or a pre-built image.

### Why uv

- **Reproducibility**: `uv.lock` pins every transitive dependency, ensuring
  identical environments across developers, CI, and production containers.
- **Speed**: uv resolves and installs dependencies 10-100x faster than pip.
- **Single tool**: uv handles virtual environments, dependency resolution,
  locking, syncing, and running scripts — no need for separate tools.
- **CI integration**: `Dockerfile.ci` installs everything via
  `uv sync --locked`. If you use pip to install something locally, it will
  diverge from what CI tests against.
- **Cache-friendly**: Set `UV_CACHE_DIR` to a persistent host directory and
  mount it into the container to avoid re-downloading wheels on every
  `docker run`. This is especially useful when you mount a frequently
  changing workdir that triggers re-syncs:
  ```bash
  docker run --rm -it \
    -v $(pwd):/opt/Megatron-Bridge \
    -v $HOME/.cache/uv:/root/.cache/uv \
    --gpus all --shm-size=24g \
    megatron-bridge:latest bash
  ```

### Essential uv Commands

| Task | Command |
|---|---|
| Install all deps from lockfile | `uv sync --locked` |
| Install with all extras and dev groups | `uv sync --locked --all-extras --all-groups` |
| Run a Python command | `uv run python script.py` |
| Run training | `uv run python -m torch.distributed.run --nproc_per_node=N script.py` |
| Add a new dependency | `uv add <package>` |
| Add an optional dependency | `uv add --optional --extra <group> <package>` |
| Regenerate the lockfile | `uv lock` (must be done inside the container on Linux) |
| Run linting | `uv run ruff check --fix . && uv run ruff format .` |
| Install pre-commit hooks | `uv run --group dev pre-commit install` |

### uv run, Not bare python

Always launch scripts with `uv run`:

```bash
# Correct
uv run python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py ...

# Wrong — bypasses the uv-managed environment
python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py ...
torchrun --nproc_per_node=1 scripts/training/run_recipe.py ...
```

After running `uv sync` inside a container, you can also use bare `python`
since the virtual environment is already activated. But `uv run` is always the
safer default.

### Adding Dependencies

```bash
uv add some-package

# For an optional extra group (e.g., trtllm-specific deps)
uv add --optional --extra trtllm some-package
```

This updates `pyproject.toml` and `uv.lock`. Commit both files:

```bash
git add pyproject.toml uv.lock
git commit -s -m "build: add some-package dependency"
```

### Regenerating uv.lock

The lockfile is Linux-only (it resolves against CUDA wheels). **You cannot
regenerate it on macOS.** Run `uv lock` inside the Docker container or on a
Linux workstation:

```bash
docker run --gpus all --rm \
  -v $(pwd):/opt/Megatron-Bridge \
  megatron-bridge:latest \
  bash -c 'cd /opt/Megatron-Bridge && uv lock'
```

### uv sync After Switching MCore Branches

The lockfile is generated against the main MCore commit. When switching to the
dev branch:

```bash
./scripts/switch_mcore.sh dev
uv sync            # without --locked
```

When switching back to main:

```bash
./scripts/switch_mcore.sh main
uv sync --locked   # lockfile matches again
```

---

## Pre-commit Hooks

Install pre-commit hooks before your first commit:

```bash
uv run --group dev pre-commit install
```

The hooks run [ruff](https://docs.astral.sh/ruff/) for linting and formatting,
plus end-of-file and trailing-whitespace fixers. If hooks auto-fix files,
re-stage and re-run:

```bash
git add -u
pre-commit run
# If it auto-fixed files:
git add -u
pre-commit run
```

Repeat until all hooks pass.

Before committing, you can also run linting manually:

```bash
ruff check --fix <changed_files>
ruff format <changed_files>
pre-commit run --all-files
```

---

## Running Tests

Tests live under `tests/`:

| Path | Description |
|------|-------------|
| `tests/unit_tests/` | Fast, isolated unit tests grouped by domain (models, core, data, etc.) |
| `tests/functional_tests/` | Integration tests with models/datasets, tiered L0/L1/L2 |

**Pytest markers available:** `unit`, `integration`, `system`, `acceptance`, `docs`, `skipduringci`, `pleasefixme`

### Unit Tests

```bash
uv run pytest tests/unit_tests/ -x -v
```

Unit tests run without GPUs and do not depend on large artifacts. Or inside Docker:

```bash
docker run --rm --gpus all -v $(pwd):/workdir/ -w /workdir/ megatron-bridge \
  uv run pytest tests/unit_tests/
```

### Functional Tests

Functional tests require GPUs and are typically run inside the container:

```bash
uv run pytest tests/functional_tests/ -x -v
```

Longer functional tests use `L2_Launch_*.sh` launcher scripts in
`tests/functional_tests/`. Each launcher must be registered in
`.github/workflows/cicd-main.yml` under `matrix.include` to be picked up
by CI.

### Adding a Unit Test

1. Place it under `tests/unit_tests/<domain>/test_<name>.py`.
2. Use the appropriate pytest marker: `@pytest.mark.unit`.
3. Run locally: `uv run --no-sync --active pytest tests/unit_tests/<your_test>.py`

### Adding a Functional Test

1. Create a launch script under `tests/functional_tests/launch_scripts/active/`.
2. Follow the naming convention: `L0_Launch_<area>_<desc>.sh`, `L1_Launch_...`, or `L2_Launch_...`.
3. Tier guidance:
   - **L0** — smoke tests that run on every PR; must be fast and stable.
   - **L1** — broader coverage; runs nightly.
   - **L2** — heavy tests (large models, checkpoint conversion); runs on schedule or manual trigger.
4. Apply the `needs-more-tests` PR label to trigger L0 + L1 for a PR.

---

## Commit and PR Workflow

- **Never commit directly to `main`** — always create a feature branch.
- **Always sign commits**: `git commit -s -m "message"`.
- **PR title format**: `[{areas}] {type}: {description}`
  (e.g., `[model] feat: Add Qwen3 model bridge`).
- **Trigger CI**: Comment `/ok to test <commit-sha>` on the PR, or set up
  signed commits for automatic CI triggering.

See `CONTRIBUTING.md` for the full PR workflow, area/type labels, and DCO
requirements.

---

## CI Pipeline

The CI pipeline is defined in `.github/workflows/cicd-main.yml`. It is
triggered by schedule, pushes to `main`, `deploy-release/*`, and
`pull-request/<number>` branches, merge groups, and `workflow_dispatch`.

### Pipeline Structure

```text
pre-flight
  └── lint-check
        └── cicd-wait-in-queue          # requires maintainer approval for untrusted PRs
              └── cicd-container-build  # builds and caches the Docker image
                    ├── unit-tests-core
                    ├── unit-tests-diffusion
                    └── functional-tests (L0 always; L1 with needs-more-tests label; L2 on schedule)
```

- The CI branch `pull-request/<number>` is created automatically when a PR is opened against `main` or `deploy-release/*`.
- Concurrent runs for the same PR are cancelled automatically (concurrency group per PR number).
- Slack notifications are sent on completion for scheduled and nightly runs.

---

## CI Failure Investigation

For PR-scoped CI runs, branches follow the pattern `pull-request/<number>`.
This workflow can also be triggered by schedule, push to `main`/`deploy-release/*`, and `workflow_dispatch`.

### Locating the PR from a CI Branch

```bash
# Extract PR number from the CI branch name (e.g. pull-request/1234)
PR_NUMBER=$(git rev-parse --abbrev-ref HEAD | grep -oP '(?<=pull-request/)\d+')

# Or, given a branch name string directly:
PR_NUMBER=$(echo "pull-request/1234" | grep -oP '(?<=pull-request/)\d+')

# Fetch PR metadata
gh pr view "$PR_NUMBER" --repo NVIDIA-NeMo/Megatron-Bridge

# List files changed in the PR
gh pr diff "$PR_NUMBER" --repo NVIDIA-NeMo/Megatron-Bridge --name-only

# View PR checks / CI status
gh pr checks "$PR_NUMBER" --repo NVIDIA-NeMo/Megatron-Bridge
```

### Investigating a Failing CI Job

1. **Get the PR number** from the branch name (see above).
2. **Review the changeset** to understand what changed:
   ```bash
   gh pr diff "$PR_NUMBER" --repo NVIDIA-NeMo/Megatron-Bridge
   ```
3. **Identify the failing job** from `gh pr checks` output or from the GitHub Actions URL in the failure notification.
4. **Fetch job logs** for deeper inspection:
   ```bash
   # List runs for the PR's head SHA
   gh run list --repo NVIDIA-NeMo/Megatron-Bridge --branch "pull-request/$PR_NUMBER"

   # Download logs for a specific run to a local file
   gh run view <run_id> --repo NVIDIA-NeMo/Megatron-Bridge --log-failed > run.log
   ```
5. **Scan the log file in chunks.** Log files can exceed 10,000 lines — never load them whole into context. Read them in chunks of ~200 lines and stop as soon as the root cause is found:
   ```bash
   # Total line count
   wc -l run.log

   # Read chunk N (lines 1–200, 201–400, …)
   sed -n '1,200p' run.log
   sed -n '201,400p' run.log
   # … continue until the failure is located
   ```
   Scan from the end first if looking for the final error, then work backwards:
   ```bash
   # Last 200 lines
   tail -200 run.log
   ```
6. **Cross-reference the changeset** against the failing test or step to narrow down the root cause.

### Common Failure Patterns

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| Lint job fails | `ruff` or `pre-commit` violation | Run `ruff check --fix` + `ruff format` locally |
| Container build fails | Dependency conflict or stale `uv.lock` | Re-run `uv lock` inside Docker and commit updated lock |
| Unit tests fail | Code regression or missing import | Run failing test locally; check the PR diff for the relevant module |
| Functional test (L0) fails | Integration breakage | Check GPU runner logs; reproduce with the corresponding `L0_Launch_*.sh` script |
| `cicd-wait-in-queue` blocked | PR not yet approved for CI | A maintainer must comment `/ok to test <SHA>` or approve via the test queue |
| MCore submodule mismatch | Pinned commit out of sync | Update `3rdparty/Megatron-LM` submodule and re-lock |

---

## Common Pitfalls

| Problem | Cause | Fix |
|---|---|---|
| `uv sync --locked` fails on macOS | Lockfile resolves CUDA wheels that don't exist on macOS | Run inside Docker or on a Linux machine |
| `ModuleNotFoundError` after pip install | pip installed outside the uv-managed venv | Use `uv add` and `uv sync`, never bare `pip install` |
| `uv sync --locked` fails after MCore branch switch | Lockfile was generated against main MCore | Use `uv sync` (without `--locked`) on dev |
| Stale checkpoint auto-resume in Bridge | `nemo_experiments/` from a previous run exists | `rm -rf nemo_experiments` before starting fresh |
| Port collision on Slurm (EADDRINUSE) | `ntasks-per-node=8` with `torchrun --nproc_per_node=8` | Drop torchrun; use `ntasks-per-node=8` with `uv run python script.py` (srun-native) |
| `uv: command not found` inside container | Container doesn't have uv | Use the `megatron-bridge` image built from `Dockerfile.ci` |
| `No space left on device` during uv ops | Cache fills container's `/root/.cache/` | Set `UV_CACHE_DIR` to shared/persistent storage |
| Pre-commit fails with ruff errors | Code style violations | Run `uv run ruff check --fix . && uv run ruff format .` |

---

## Quick Start Checklist

1. Clone the repo and initialize submodules:
   ```bash
   git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge megatron-bridge
   cd megatron-bridge
   git submodule update --init 3rdparty/Megatron-LM
   ```

2. Build the container:
   ```bash
   docker build -f docker/Dockerfile.ci --target megatron_bridge -t megatron-bridge:latest .
   ```

3. Start a dev shell:
   ```bash
   docker run --rm -it -v $(pwd):/opt/Megatron-Bridge --gpus all --shm-size=24g megatron-bridge:latest bash
   ```

4. Install pre-commit hooks (inside container):
   ```bash
   uv run --group dev pre-commit install
   ```

5. Run a quick training sanity check:
   ```bash
   uv run python -m torch.distributed.run --nproc_per_node=1 \
     scripts/training/run_recipe.py \
     --recipe vanilla_gpt_pretrain_config \
     train.train_iters=5 train.global_batch_size=8 train.micro_batch_size=4 \
     scheduler.lr_warmup_iters=1 scheduler.lr_decay_iters=5 \
     logger.log_interval=1
   ```

6. Create a branch, make changes, and submit a PR:
   ```bash
   git switch -c your-feature-name
   # ... make changes ...
   git add -u && git commit -s -m "[area] type: description"
   git push origin your-feature-name
   ```
