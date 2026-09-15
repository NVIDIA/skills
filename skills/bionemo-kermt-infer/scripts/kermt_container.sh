#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# kermt_container.sh — bootstrap helper for the kermt agent skills.
#
# Two ways to use this file:
#
# 1. As a subcommand dispatcher (recommended for skills):
#       "$SKILL_DIR/scripts/kermt_container.sh" ensure_image
#       "$SKILL_DIR/scripts/kermt_container.sh" run --ckpt /host/ckpt.pt -- python -c 'import torch; print(torch.cuda.device_count())'
#       "$SKILL_DIR/scripts/kermt_container.sh" run_detached --name foo --run-dir runs/foo -- bash train.sh
#
# 2. Sourced into a shell or another script, then call the kermt_* functions
#    directly:
#       source "$SKILL_DIR/scripts/kermt_container.sh"
#       kermt_ensure_image
#       kermt_run --ckpt /host/ckpt.pt -- python ...
#
# Configuration (override via env vars before invocation):
#   KERMT_IMAGE   docker image tag (default: kermt:latest)
#   KERMT_REPO    host path to the kermt repo checkout (default: auto-derived
#                 from this script's location)
#   KERMT_GPUS    value passed to docker --gpus (default: all)
#
# Mount flags accepted by kermt_run / kermt_run_detached:
#   --data <path>       bind to /data    (read-only). If <path> is a file,
#                       its PARENT directory is mounted at /data so
#                       commands can use /data/<basename>; if <path> is a
#                       directory, it is mounted at /data directly.
#   --ckpt <path>       bind to /ckpt    (read-only; the path is mounted as-is)
#   --vocab-dir <dir>   bind to /vocab   (read-only)
#   --run-dir <dir>     bind to /runs    (read-write; created on host if missing)
#   --model-dir <dir>   bind to /model   (read-write; created on host if missing).
#                       Target for released-model downloads (fetch_released_model.py).
#
# Additional flags for kermt_run_detached:
#   --name <name>       docker container name (default: kermt-<UTC-timestamp>-<pid>)
#
# Everything after `--` is the command passed to the container. It runs inside
# the `kermt` conda environment (the image's default env).

set -o pipefail

: "${KERMT_IMAGE:=kermt:latest}"
: "${KERMT_GPUS:=all}"

# The skill may be installed outside the KERMT checkout. Mount its own helpers
# separately so container commands always execute the distributed skill copy.
_kermt_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_kermt_bundle_dir="$(cd "$_kermt_script_dir/.." && pwd)"
if [[ -z "${KERMT_REPO:-}" ]]; then
  for _kermt_start in "$_kermt_script_dir" "$PWD"; do
    _kermt_candidate="$_kermt_start"
    while [[ "$_kermt_candidate" != / ]]; do
      if [[ -f "$_kermt_candidate/main.py" && -d "$_kermt_candidate/kermt" ]]; then
        KERMT_REPO="$_kermt_candidate"
        break 2
      fi
      _kermt_candidate="$(dirname "$_kermt_candidate")"
    done
  done
  unset _kermt_start _kermt_candidate
fi
unset _kermt_script_dir

_kermt_require_repo() {
  if [[ -z "${KERMT_REPO:-}" || ! -f "$KERMT_REPO/main.py" || ! -d "$KERMT_REPO/kermt" ]]; then
    echo "[kermt] Set KERMT_REPO to the KERMT checkout containing main.py and kermt/." >&2
    return 1
  fi
  KERMT_REPO="$(cd "$KERMT_REPO" && pwd)" || return $?
  export KERMT_REPO
}

# -----------------------------------------------------------------------------
# Host environment checks
# -----------------------------------------------------------------------------

kermt_check_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "[kermt] error: docker not found on PATH. Install Docker first." >&2
    return 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "[kermt] error: docker daemon not reachable. Is the docker service running, and is your user in the 'docker' group?" >&2
    return 1
  fi
}

kermt_check_system() {
  # Probe host system and report GPU presence + VRAM + compute capability +
  # driver / CUDA version + disk space. Emits a single JSON document to
  # stdout that the calling skill consumes; exits 0 with `ok: false` and a
  # populated `gaps` array when anything is below the per-workflow minimum,
  # exits 1 only on unexpected internal errors. Uses host nvidia-smi + df +
  # host python3 (stdlib only).
  python3 - "$KERMT_REPO" "$KERMT_IMAGE" <<'PYEOF'
import json, os, shutil, subprocess, sys

repo, image = sys.argv[1], sys.argv[2]

result = {
    "ok": True,
    "gpus": [],
    "disk": {"path": repo, "free_gb": None, "min_gb": 20},
    "host": {"docker": None, "nvidia_smi": None, "container_toolkit": None},
    "image": {"tag": image, "present_locally": None},
    "gaps": [],
}

def _gap(msg):
    result["ok"] = False
    result["gaps"].append(msg)

# docker presence
try:
    r = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
    result["host"]["docker"] = "ok" if r.returncode == 0 else f"failed: {r.stderr.strip().splitlines()[-1] if r.stderr else 'unknown'}"
    if r.returncode != 0:
        _gap("docker daemon not reachable (is the service running, and is your user in the 'docker' group?)")
except FileNotFoundError:
    result["host"]["docker"] = "not found"
    _gap("docker not on PATH; install Docker first")
except Exception as e:
    result["host"]["docker"] = f"error: {e}"
    _gap(f"docker probe failed: {e}")

# nvidia-smi (host driver)
try:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version,uuid",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    if r.returncode == 0:
        result["host"]["nvidia_smi"] = "ok"
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                try:
                    vram_mb = int(parts[1])
                except ValueError:
                    vram_mb = None
                result["gpus"].append({
                    "name": parts[0],
                    "vram_mb": vram_mb,
                    "compute_cap": parts[2],
                    "driver": parts[3],
                    "uuid": parts[4],
                })
        if not result["gpus"]:
            _gap("nvidia-smi succeeded but reported no GPUs")
    else:
        result["host"]["nvidia_smi"] = "failed"
        _gap("nvidia-smi found but failed; is the NVIDIA driver loaded?")
except FileNotFoundError:
    result["host"]["nvidia_smi"] = "not found"
    _gap("nvidia-smi not on PATH; install the NVIDIA driver")
except Exception as e:
    result["host"]["nvidia_smi"] = f"error: {e}"
    _gap(f"nvidia-smi probe failed: {e}")

# disk free at the repo location
try:
    free_bytes = shutil.disk_usage(repo).free
    free_gb = free_bytes // (1024**3)
    result["disk"]["free_gb"] = free_gb
    if free_gb < result["disk"]["min_gb"]:
        _gap(f"disk free at {repo} is {free_gb} GB; need at least {result['disk']['min_gb']} GB for the kermt image")
except Exception as e:
    _gap(f"could not check disk space at {repo}: {e}")

# image presence (informational only)
try:
    r = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True, timeout=10)
    result["image"]["present_locally"] = (r.returncode == 0)
except Exception:
    result["image"]["present_locally"] = None

# nvidia-container-toolkit probe — only meaningful if both docker and a
# locally-present image are available. Pick kermt:$tag first; fall back to
# the small CUDA base image if that's the only one present; otherwise skip
# (avoid pulling anything).
def _probe_image():
    for img in (image, "nvidia/cuda:12.6.3-base-ubuntu22.04"):
        r = subprocess.run(["docker", "image", "inspect", img], capture_output=True)
        if r.returncode == 0:
            return img
    return None

probe_img = _probe_image()
if probe_img:
    try:
        r = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all", probe_img, "nvidia-smi"],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0:
            result["host"]["container_toolkit"] = f"ok (probed via {probe_img})"
        else:
            result["host"]["container_toolkit"] = f"failed (probed via {probe_img})"
            _gap("`docker run --gpus all` failed; install nvidia-container-toolkit and ensure the host driver supports it")
    except Exception as e:
        result["host"]["container_toolkit"] = f"error: {e}"
        _gap(f"nvidia-container-toolkit probe failed: {e}")
else:
    result["host"]["container_toolkit"] = "skipped (no probe image present locally; run ensure_image first)"

print(json.dumps(result, indent=2))
PYEOF
}

kermt_check_gpu() {
  # Probes whether `docker --gpus all` is wired up (nvidia-container-toolkit).
  # Image-selection priority (never pulls anything):
  #   1) $KERMT_IMAGE if it exists locally,
  #   2) else nvidia/cuda:12.6.3-base-ubuntu22.04 if it exists locally,
  #   3) else skip with a warning (return 0). The smoke test inside kermt_run
  #      will catch broken GPU passthrough later anyway.
  local probe_img=""
  if docker image inspect "$KERMT_IMAGE" >/dev/null 2>&1; then
    probe_img="$KERMT_IMAGE"
  elif docker image inspect nvidia/cuda:12.6.3-base-ubuntu22.04 >/dev/null 2>&1; then
    probe_img="nvidia/cuda:12.6.3-base-ubuntu22.04"
  else
    echo "[kermt] check_gpu: skipped — neither '$KERMT_IMAGE' nor 'nvidia/cuda:12.6.3-base-ubuntu22.04' is present locally. Run 'ensure_image' first, or this probe will be exercised by the in-container smoke test." >&2
    return 0
  fi
  if ! docker run --rm --gpus all "$probe_img" nvidia-smi >/dev/null 2>&1; then
    echo "[kermt] error: 'docker run --gpus all' failed (probe image: $probe_img). Install nvidia-container-toolkit and ensure the host has a CUDA-capable NVIDIA driver." >&2
    return 1
  fi
}

# -----------------------------------------------------------------------------
# Image build / verification
# -----------------------------------------------------------------------------

kermt_ensure_image() {
  _kermt_require_repo || return $?
  kermt_check_docker || return $?
  if docker image inspect "$KERMT_IMAGE" >/dev/null 2>&1; then
    local id
    id=$(docker image inspect "$KERMT_IMAGE" --format '{{.Id}}' 2>/dev/null | cut -c1-19)
    echo "[kermt] image '$KERMT_IMAGE' already present (${id:-unknown})"
    return 0
  fi
  echo "[kermt] image '$KERMT_IMAGE' not found; building from $KERMT_REPO/Dockerfile"
  echo "[kermt] first build typically takes 10-20 minutes on a typical workstation; subsequent runs reuse the cached image"
  docker build -t "$KERMT_IMAGE" -f "$KERMT_REPO/Dockerfile" "$KERMT_REPO"
}

# -----------------------------------------------------------------------------
# Mount-flag parser, internal
# -----------------------------------------------------------------------------
# Reads flags from the caller's positional args until it hits '--', appending
# `-v src:dst[:ro]` pairs into the caller-provided array name (passed as $1).
# Returns the number of caller-provided args consumed via _kermt_consumed.
# This is bash-specific (uses nameref via `declare -n`).

_kermt_parse_mounts() {
  local -n _out="$1"
  shift
  _kermt_consumed=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --)
        return 0
        ;;
      --data)
        [[ -e "$2" ]] || { echo "[kermt] --data path not found: $2" >&2; return 1; }
        # If the user passes a file, mount its parent directory at /data so
        # downstream commands can refer to /data/<basename>. Mounting a
        # single file at /data makes the path-as-directory pattern in the
        # skill examples (`--csv /data/<basename>`) fail with "not found".
        if [[ -d "$2" ]]; then
          _out+=("-v" "$(realpath "$2"):/data:ro")
        else
          _out+=("-v" "$(realpath "$(dirname "$2")"):/data:ro")
        fi
        shift 2; _kermt_consumed=$((_kermt_consumed + 2))
        ;;
      --ckpt)
        [[ -e "$2" ]] || { echo "[kermt] --ckpt path not found: $2" >&2; return 1; }
        _out+=("-v" "$(realpath "$2"):/ckpt:ro")
        shift 2; _kermt_consumed=$((_kermt_consumed + 2))
        ;;
      --vocab-dir)
        [[ -d "$2" ]] || { echo "[kermt] --vocab-dir not found or not a directory: $2" >&2; return 1; }
        _out+=("-v" "$(realpath "$2"):/vocab:ro")
        shift 2; _kermt_consumed=$((_kermt_consumed + 2))
        ;;
      --run-dir)
        mkdir -p "$2" || { echo "[kermt] failed to create --run-dir: $2" >&2; return 1; }
        _out+=("-v" "$(realpath "$2"):/runs")
        shift 2; _kermt_consumed=$((_kermt_consumed + 2))
        ;;
      --model-dir)
        mkdir -p "$2" || { echo "[kermt] failed to create --model-dir: $2" >&2; return 1; }
        _out+=("-v" "$(realpath "$2"):/model")
        shift 2; _kermt_consumed=$((_kermt_consumed + 2))
        ;;
      *)
        return 0
        ;;
    esac
  done
}

# -----------------------------------------------------------------------------
# Foreground / detached run
# -----------------------------------------------------------------------------

# Capture host-side git state for the repo and emit `-e KERMT_REPO_COMMIT=…
# -e KERMT_REPO_DIRTY=true|false` flags. Used by the run / run_detached
# wrappers so the runner's run.json manifest gets honest commit info even
# though `git -C /workspace` inside the container fails due to bind-mount
# ownership.
_kermt_git_env_flags() {
  local commit="unknown"
  local dirty="false"
  if command -v git >/dev/null 2>&1 && [[ -d "$KERMT_REPO/.git" ]]; then
    local c
    c=$(git -C "$KERMT_REPO" rev-parse HEAD 2>/dev/null) && commit="$c"
    # `--untracked-files=no` filters out user-private notes (e.g. a CLAUDE.md
    # or RELEASE_PLAN_v2.0.md at the repo root) that wouldn't affect
    # reproducibility — only modifications to tracked files do.
    if [[ -n "$(git -C "$KERMT_REPO" status --porcelain --untracked-files=no 2>/dev/null | head -n 1)" ]]; then
      dirty="true"
    fi
  fi
  printf '%s\n%s\n%s\n%s\n' "-e" "KERMT_REPO_COMMIT=$commit" "-e" "KERMT_REPO_DIRTY=$dirty"
}

# Forward HF_TOKEN into the container when it is set, so fetch_released_model.py
# can authenticate to Hugging Face. The current release is public (no token
# needed); this only guards against shared-IP rate limits or a future gated
# repo. Emits nothing when HF_TOKEN is unset.
_kermt_hf_env_flags() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf '%s\n%s\n' "-e" "HF_TOKEN=$HF_TOKEN"
  fi
}

kermt_run() {
  kermt_ensure_image || return $?
  local mount_args=()
  _kermt_parse_mounts mount_args "$@" || return $?
  shift "$_kermt_consumed"
  if [[ "${1:-}" != "--" ]]; then
    echo "[kermt] expected '--' separating mount flags from the command (got '${1:-}')" >&2
    return 1
  fi
  shift
  if [[ $# -eq 0 ]]; then
    echo "[kermt] no command supplied after '--'" >&2
    return 1
  fi
  local git_args=()
  while IFS= read -r line; do git_args+=("$line"); done < <(_kermt_git_env_flags)
  local hf_args=()
  while IFS= read -r line; do hf_args+=("$line"); done < <(_kermt_hf_env_flags)
  docker run --rm --gpus "$KERMT_GPUS" \
    --user "$(id -u):$(id -g)" \
    -v "$KERMT_REPO:/workspace" \
    -v "$_kermt_bundle_dir:/skill:ro" \
    "${mount_args[@]}" \
    -w /workspace \
    -e KERMT_REPO=/workspace \
    -e PYTHONPATH=/workspace \
    -e HOME=/tmp/kermt-home \
    "${git_args[@]}" \
    "${hf_args[@]}" \
    "$KERMT_IMAGE" \
    conda run -n kermt --no-capture-output bash -c "$*"
}

kermt_run_detached() {
  kermt_ensure_image || return $?
  local name=""
  local mount_args=()
  # Pull --name out first, then let the shared mount parser handle the rest.
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --name) name="$2"; shift 2 ;;
      --) break ;;
      --data|--ckpt|--vocab-dir|--run-dir|--model-dir) break ;;
      *) break ;;
    esac
  done
  _kermt_parse_mounts mount_args "$@" || return $?
  shift "$_kermt_consumed"
  if [[ "${1:-}" != "--" ]]; then
    echo "[kermt] expected '--' separating mount flags from the command (got '${1:-}')" >&2
    return 1
  fi
  shift
  if [[ $# -eq 0 ]]; then
    echo "[kermt] no command supplied after '--'" >&2
    return 1
  fi
  if [[ -z "$name" ]]; then
    name="kermt-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  fi
  local cid
  local git_args=()
  while IFS= read -r line; do git_args+=("$line"); done < <(_kermt_git_env_flags)
  local hf_args=()
  while IFS= read -r line; do hf_args+=("$line"); done < <(_kermt_hf_env_flags)
  cid=$(docker run -d --gpus "$KERMT_GPUS" \
    --user "$(id -u):$(id -g)" \
    --name "$name" \
    -v "$KERMT_REPO:/workspace" \
    -v "$_kermt_bundle_dir:/skill:ro" \
    "${mount_args[@]}" \
    -w /workspace \
    -e KERMT_REPO=/workspace \
    -e PYTHONPATH=/workspace \
    -e HOME=/tmp/kermt-home \
    "${git_args[@]}" \
    "${hf_args[@]}" \
    "$KERMT_IMAGE" \
    conda run -n kermt --no-capture-output bash -c "$*") || return $?
  echo "[kermt] container started: name=$name id=$cid"
  echo "[kermt] follow logs:    docker logs -f $name"
  echo "[kermt] wait for exit:  docker wait $name"
  echo "[kermt] stop:           docker stop $name"
  echo "$cid"
}

# -----------------------------------------------------------------------------
# Subcommand dispatch when invoked directly (not sourced)
# -----------------------------------------------------------------------------

if [[ "${BASH_SOURCE[0]:-$0}" == "${0}" ]]; then
  cmd="${1:-}"; shift || true
  case "$cmd" in
    check_docker)  kermt_check_docker "$@" ;;
    check_gpu)     kermt_check_gpu "$@" ;;
    check_system)  kermt_check_system "$@" ;;
    ensure_image)  kermt_ensure_image "$@" ;;
    run)           kermt_run "$@" ;;
    run_detached)  kermt_run_detached "$@" ;;
    ""|-h|--help)
      cat >&2 <<EOF
usage: $0 <subcommand> [args...]

Subcommands:
  check_docker        Verify docker is installed and the daemon is reachable.
  check_gpu           Verify 'docker --gpus all' works (nvidia-container-toolkit).
  check_system        Emit a JSON probe of host GPU + VRAM + compute_cap +
                      driver + disk space + container toolkit + image presence.
                      Exits 0 with ok=false + a 'gaps' list when anything's
                      below the per-workflow minimum.
  ensure_image        Build kermt:latest from \$KERMT_REPO/Dockerfile if missing.
  run [flags] -- ...  Run a command inside the container (foreground, --rm).
  run_detached [flags] -- ...
                      Run detached; prints container name + id + log hint.

Mount flags (for run / run_detached):
  --data <path>       bind to /data    (read-only)
  --ckpt <path>       bind to /ckpt    (read-only)
  --vocab-dir <dir>   bind to /vocab   (read-only)
  --run-dir <dir>     bind to /runs    (read-write; created on host if missing)
  --model-dir <dir>   bind to /model   (read-write; released-model download target)

Additional flags for run_detached:
  --name <name>       container name (default: kermt-<timestamp>-<pid>)

Environment overrides:
  KERMT_IMAGE         default kermt:latest
  KERMT_REPO          checkout path; otherwise discovered above the skill or working directory
  KERMT_GPUS          default all
EOF
      exit 1
      ;;
    *)
      echo "[kermt] unknown subcommand: $cmd" >&2
      echo "[kermt] run '$0 --help' for usage" >&2
      exit 1
      ;;
  esac
fi
