---
name: build-and-dependency
description: Build and dependency management for NeMo-RL. Covers Docker image building and running, uv usage, venv setup, and adding dependencies.
when_to_use: Setting up a dev environment; building or running the Docker container; adding or removing a dependency; uv errors; 'how do I install', 'ModuleNotFoundError', 'build the image', 'run in Docker', 'uv sync fails'.
---

# Build and Dependency Guide

---

## Docker Images

Build the release image (includes all dependencies and pre-fetched venvs):

```bash
# Build from local source
docker buildx build -f docker/Dockerfile --tag nemo-rl:latest .

# Build from a specific git ref (no local clone needed)
docker buildx build -f docker/Dockerfile \
    --build-arg NRL_GIT_REF=main \
    --tag nemo-rl:latest \
    https://github.com/NVIDIA-NeMo/RL.git
```

Skip optional backends to reduce build time:

```bash
# Skip vLLM and SGLang
docker buildx build -f docker/Dockerfile \
    --build-arg SKIP_VLLM_BUILD=1 \
    --build-arg SKIP_SGLANG_BUILD=1 \
    --tag nemo-rl:latest .
```

See @docs/docker.md for full options.

---

## Always Use uv

**Never use `pip install` directly** — always go through `uv`.

```bash
# Run a script
uv run examples/run_grpo.py

# Run tests
uv run --group test bash tests/run_unit.sh

# Install all deps from lockfile
uv sync --locked
```

Exception: `Dockerfile.ngc_pytorch` is exempt from this rule.

---

## Adding Dependencies

```bash
# Add a runtime dependency
uv add <package>

# Add an optional dependency
uv add --optional --extra <group> <package>

# Regenerate the lockfile after changes
uv lock
```

Commit both `pyproject.toml` and `uv.lock` together:

```bash
git add pyproject.toml uv.lock
git commit -s -m "build: add <package> dependency"
```

---

## Common Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| `uv sync --locked` fails | Dependency conflict or stale lockfile | Re-run `uv lock` and commit updated lock |
| `ModuleNotFoundError` after pip install | pip installed outside uv-managed venv | Use `uv add` + `uv sync`, never bare `pip install` |
| Docker build fails at vLLM | vLLM build time overhead | Pass `--build-arg SKIP_VLLM_BUILD=1` |
