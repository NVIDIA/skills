---
name: kermt-setup
description: Bootstrap the KERMT agent environment — verify host docker + nvidia-container-toolkit, build the kermt:latest image from the repo's Dockerfile if it doesn't yet exist, and run a GPU smoke test inside the container. Every other kermt-* skill depends on this; invoke it first.
license: Apache-2.0
compatibility: Requires docker, nvidia-container-toolkit, and a CUDA-capable NVIDIA GPU. Designed for Claude Code, Codex, and Nemotron.
metadata:
  owner: evax@nvidia.com
  classification: atomic-skill
  risk_tier: skill
# This file is intentionally short (~110 lines, ~1200 tokens) — well within the
# 500-line / 5000-token budget for skill files. Longer reference material lives
# alongside /skill/scripts/kermt_container.sh.
---

# kermt-setup

Bootstrap the KERMT agent environment. Run this once on a fresh machine (or
after the Dockerfile or `environment.yml` changes) before invoking any other
`kermt-*` skill.

## Skill and runtime paths

Set `SKILL_DIR` to the absolute path of this installed skill directory. Export
`KERMT_REPO` as the absolute path to the KERMT checkout used for model
execution. The bundled container helper mounts that checkout at
`/workspace` and this skill at `/skill` (read-only). Commands inside
the container use `/skill/scripts/`.

## Hardware requirements

- **GPU**: at least one CUDA-capable NVIDIA GPU visible to the host. The image
  is based on `nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04`, so the host driver
  must support CUDA 12.6. Verify with host `nvidia-smi` before invoking.
- **Host docker**: docker engine + nvidia-container-toolkit. Without the
  toolkit, `docker run --gpus all` will fail at step 2 of the workflow below.
- **Disk**: ≈ 50 GB free for the built kermt image (`docker image inspect
  --format '{{.Size}}'` reports ≈ 44 GB; the `docker images` Size column
  can show ~100 GB because it counts shareable buildx attestation layers
  that are deduplicated across images). Plan for ~50 GB of unique on-disk
  storage; add a comfortable buffer if you're also keeping build cache.
- **Memory**: the build itself peaks at ~4 GB RAM during conda env solve.
- This skill does not run training/inference workloads itself; per-workflow
  hardware requirements (VRAM, GPU count) are declared in the respective
  `kermt-<workflow>` skills.

## When to invoke

- User explicitly asks (`/kermt-setup`, "set up kermt", "build the kermt image",
  etc.).
- Or another `kermt-*` skill detected that the image does not exist and routed
  here. (Most other skills call `kermt_ensure_image` themselves, so this is
  usually only needed for the first-time setup, debugging, or a forced rebuild.)

## Inputs

The skill takes no required arguments. Optional overrides (via env vars before
invoking, or by setting them in the user's shell):

- `KERMT_IMAGE` — image tag to build/verify (default: `kermt:latest`).
- `KERMT_REPO` — host path of the kermt repo checkout (default: auto-derived
  from the script's location).

If the user has not specified a repo path and the current working directory is
not inside a kermt repo clone, ask for the repo path before proceeding.

## Workflow

All work goes through the bundled `scripts/kermt_container.sh` on the host. The script's
subcommand dispatch can be invoked directly without sourcing — that is the
preferred form for skill use.

Let `HELPER="$SKILL_DIR/scripts/kermt_container.sh"`.

1. **Verify docker is installed and the daemon is reachable.**
   ```
   "$HELPER" check_docker
   ```
   Exit 0 → continue. Non-zero → surface the error to the user (typically
   "docker not on PATH" or "daemon not reachable"); do not attempt step 2.

2. **Verify GPU passthrough works.**
   ```
   "$HELPER" check_gpu
   ```
   This runs `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04
   nvidia-smi` and checks the exit status. Non-zero → tell the user to install
   `nvidia-container-toolkit` on the host and confirm a CUDA-capable NVIDIA GPU
   is visible to the host (`nvidia-smi` on the host should also work). Stop
   here; without GPU passthrough the kermt image will build but no workflow
   will run.

3. **Build or verify the kermt image.**
   ```
   "$HELPER" ensure_image
   ```
   If the image already exists, this returns immediately. Otherwise it builds
   from `$KERMT_REPO/Dockerfile`. **Warn the user before invoking** that the
   first build takes ~10–20 minutes on a typical workstation and streams build
   logs to the console. Do not run this in the background — the user wants to
   see progress and any build failures must surface immediately.

4. **GPU smoke test inside the container.** Quote the whole `python` command
   as a single string — the helper passes args through `bash -c "$*"`, so
   unquoted multi-word commands get re-parsed and any embedded quotes are
   collapsed.
   ```
   "$HELPER" run -- 'python -c "import torch; print(\"cuda_available:\", torch.cuda.is_available()); print(\"device_count:\", torch.cuda.device_count())"'
   ```
   Expected output: `cuda_available: True` and a positive `device_count`. If
   `cuda_available` is `False` despite step 2 passing, something is wrong with
   the container's CUDA wiring — report the full output to the user and stop;
   do not declare the environment ready.

5. **Summary to user.** Report:
   - Image tag and ID (`docker image inspect $KERMT_IMAGE --format '{{.Id}}'`).
   - Image size (`docker image inspect $KERMT_IMAGE --format '{{.Size}}'`).
   - GPU count detected inside the container.
   - "Ready" — the user can now invoke other `kermt-*` skills.

## Hard rules

- Do **not** pull or push docker images. The kermt image is built locally only.
- Do **not** auto-delete or prune older `kermt:*` tags without the user's
  explicit confirmation — the user may be running a finetune or pretrain in
  another container that depends on a specific tag.
- Do **not** modify the host's docker daemon configuration, daemon.json, or
  user-group membership.
- Do **not** modify the `Dockerfile` or `environment.yml` as part of this
  skill. If the build fails because of a Dockerfile issue, surface the error
  and stop; let the user decide whether to edit.
- Do **not** rebuild the image when it already exists (i.e. do not pass a
  `--no-cache` or `--pull` flag to ensure_image) unless the user explicitly
  asks for a forced rebuild.

## Forced rebuild

If the user explicitly asks to rebuild (e.g. after changing the Dockerfile or
`environment.yml`), the cleanest path is to remove the old image first, then
rerun `ensure_image`:

```
docker image rm $KERMT_IMAGE
"$HELPER" ensure_image
```

Confirm with the user before running `docker image rm`.
