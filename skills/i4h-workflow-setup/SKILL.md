---
name: i4h-workflow-setup
version: "0.6.0"
description: Verify host requirements and run `workflows/agentic/setup.sh`. Use when asked to set up, install, or bootstrap the agentic workflow, or hits missing `.venv`, third-party checkout, or engine errors.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - agentic-workflow
    - setup
    - installation
---

# i4h Workflow — Setup

## Purpose

Verify host requirements and run the idempotent `workflows/agentic/setup.sh` to bootstrap the agentic workflow. Use when asked to set up, install, or bootstrap the workflow, or when hitting a missing `.venv`, third-party checkout, or engine error.

## Base Code

These steps drive the i4h-workflows base code (the `workflows/agentic/` tree). To reuse an existing checkout, set `I4H_WORKFLOWS` to its path (no clone happens). Otherwise this resolves the current repo, or clones to `~/i4h-workflows` — pick that default without prompting. Run every command below from the resolved root:

```bash
# Resolve the i4h-workflows base code (provides workflows/agentic/).
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/workflows/agentic" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/i4h-workflows}"
  [ -d "$ROOT/workflows/agentic" ] || git clone https://github.com/isaac-for-healthcare/i4h-workflows "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"; cd "$ROOT"
```

## Basics

- `workflows/agentic/setup.sh` is the idempotent setup entry point.
- Cosmos setup is separate; invoke only when the user asks for Cosmos or video transfer.

## Preflight

```bash
command -v uv
command -v git
nvidia-smi
df -h .
```

Required: Linux, `uv`, `git`, NVIDIA driver/GPU, disk space for third-party checkouts. Docker is optional unless using Cosmos or local VLM containers.

Run the steps below in order. Each step is a separate bash call; variables persist in the local agent's tmux session.

### Step 1 — resolve repo and run dir

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
RUN_DIR="${REPO_ROOT}/workflows/agentic/runs/setup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${REPO_ROOT}/workflows/agentic/runs/.latest"
```

### Step 2 — run setup

On a freshly cloned repo the harness may ask to approve `setup.sh` once. Set `I4H_WORKFLOWS` to a pre-existing trusted clone to skip the fresh-clone gate.

```bash
"${REPO_ROOT}/workflows/agentic/setup.sh" > "${RUN_DIR}/logs/setup.log" 2>&1
```

Watch progress with `tail -f "${RUN_DIR}/logs/setup.log"`. For component-specific retries (also idempotent):

```bash
"${REPO_ROOT}/workflows/agentic/third_party/setup.sh"
"${REPO_ROOT}/workflows/agentic/policy/gr00t_n16/setup.sh"
```

## Verify

```bash
"${REPO_ROOT}/workflows/agentic/policy/run.sh" --list-envs
"${REPO_ROOT}/workflows/agentic/arena/run.sh" --list-envs
"${REPO_ROOT}/workflows/agentic/policy/run.sh" --env scissor_pick_and_place --dry-run
"${REPO_ROOT}/workflows/agentic/arena/run.sh" --env scissor_pick_and_place --dry-run
```

## Prerequisites

- Linux host with `uv`, `git`, and an NVIDIA driver/GPU (check via the Preflight commands).
- Disk space for the third-party checkouts (`df -h .`).
- Docker only when using Cosmos or local VLM containers; otherwise optional.

## Limitations

- Linux only; requires an NVIDIA driver/GPU.
- Cosmos setup is separate — `setup.sh` does not run it; invoke Cosmos only when the user asks for it.
- Verify steps use `--list-envs` and `--dry-run`; they confirm envs register and build, not that a full rollout works.

## Troubleshooting

- **Error:** `uv` / `git` / `nvidia-smi` not found — Cause: missing host requirement. Fix: install it and re-run Preflight before `setup.sh`.
- **Error:** a single component fails partway through setup — Cause: a checkout or component venv did not finish. Fix: re-run that component's script (e.g. `third_party/setup.sh`, `arena/setup.sh`, `policy/setup.sh`); all are idempotent.
- **Error:** `--list-envs` or `--dry-run` fails after setup — Cause: setup incomplete or a component venv missing. Fix: inspect `${RUN_DIR}/logs/setup.log` (or `workflows/agentic/runs/.latest/logs/setup.log`) and re-run `setup.sh` or the failing component script.
- **Error:** `setup.sh` fails instantly redirecting to a missing log dir — Cause: Step 2 ran before Step 1, so `${RUN_DIR}/logs` does not exist. Fix: run Step 1 first, then Step 2.

## Final Response

Report setup status, failed component (if any), relevant log path, next recommended smoke test.
