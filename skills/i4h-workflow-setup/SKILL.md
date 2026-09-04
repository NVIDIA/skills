---
name: i4h-workflow-setup
description: Preflight and set up the root-level workflow runtime. Use for installation, missing component environments, or third-party failures; do not use for rollout validation.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  tags:
    - isaac-for-healthcare
    - i4h
    - setup
    - dependencies
---

# i4h Workflow Setup

## Purpose

Prepare the complete runtime, then prove dependency-light workflow discovery works.

## Instructions

1. Resolve the base checkout.
2. Run every host preflight check.
3. Run default full setup.
4. Verify discovery and `lint --all`.

## Resolve the checkout

```bash
export I4H_WORKFLOWS_REPO_URL="${I4H_WORKFLOWS_REPO_URL:-https://github.com/isaac-for-healthcare/i4h-workflows}"
I4H_REPO_DIR_NAME="${I4H_WORKFLOWS_REPO_URL%/}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME##*/}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME##*:}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME%.git}"
[ -n "$I4H_REPO_DIR_NAME" ] || { echo "Cannot derive a checkout name from I4H_WORKFLOWS_REPO_URL" >&2; exit 2; }
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/workflows/i4h_workflows" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/$I4H_REPO_DIR_NAME}"
  [ -d "$ROOT/workflows/i4h_workflows" ] || git clone "$I4H_WORKFLOWS_REPO_URL" "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"
cd "$ROOT"
```

Treat this resolver as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `workflows/i4h_workflows`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

## Preflight

```bash
uname -srm
command -v uv
command -v git
nvidia-smi
df -h .
./setup.sh --help
```

Require supported Linux on `x86_64` or `aarch64`, `uv`, Git, network access for missing packages/assets, sufficient disk and model-cache space, and an NVIDIA driver compatible with the repository's Isaac Sim version. Read the exact current requirements from `./README.md`; do not rely on remembered version numbers. Require Docker only when the requested local VLM or Cosmos service needs it.

## Sync

Run the default full setup:

```bash
./setup.sh
```

The explicit equivalent is:

```bash
./setup.sh all
```

Use `./setup.sh clean` only when the user explicitly requests a full rebuild or cleanup because it deletes generated caches, component environments, and third-party checkouts. Setup is idempotent; do not clean as a generic repair. The setup script restores pinned third-party sources and the upstream Isaac Sim skill catalog; do not copy that catalog into `skills/`.

## Verify

```bash
./run.sh list
./run.sh lint --all
test -f ./third_party/IsaacSim-045ca8b/skills/SKILLS.md
```

Use `run.sh show <workflow> --mode <mode>` when the user named a workflow. These checks prove discovery and contracts without importing Isaac Sim. Use `i4h-workflow-validate` for an actual rollout.

## Troubleshooting

On failure, report the first host, network, checkout, or uv-sync error and rerun full setup only after correcting that cause.

## Prerequisites

Require supported Linux, network access, `uv`, Git, disk space, and a compatible NVIDIA GPU/driver for simulator use.

## Limitations

Setup does not download every optional checkpoint or prove any rollout succeeds. `clean` is destructive to generated environments and third-party checkouts.

## Examples

- `Set up the i4h workflow on this machine and tell me if any host requirements are missing.` → preflight, run default full setup, then run discovery and `lint --all`.

## Completion gate

Report every host check, synced target, first failing component if any, discovery/lint results, and the smallest next smoke command. Do not claim simulator or policy success from setup alone.
