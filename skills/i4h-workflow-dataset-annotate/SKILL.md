---
name: i4h-workflow-dataset-annotate
version: "0.6.0"
description: Use a VLM to verify whether each episode satisfies the env's task description. Use when the user asks to annotate, label episodes, filter demos, or gate finetuning on a success classifier.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - dataset
    - annotation
    - vlm
---

# i4h Workflow — Annotate Dataset

## Purpose

Use a VLM to verify whether each episode satisfies the env's task description. Use when the user asks to annotate, label episodes, filter demos, or gate finetuning on a success classifier.

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

- Annotation is optional. Do not run it during validation unless the user requests labels.
- For natural-language prompts such as "Run Annotation on all recorded episodes", annotate all episodes in one selected HDF5 recording, not every historical HDF5 under `workflows/agentic/runs/`. If the user does not name an HDF5, choose the latest annotatable recording: first look inside `runs/.latest` when it contains an HDF5, otherwise pick the newest non-annotation `.hdf5` under `workflows/agentic/runs/`. Only batch across multiple HDF5 files when the user explicitly asks for all historical recordings, every HDF5 file, or a batch annotation run.
- **Env config (source of truth):** the annotator reads the success criterion (`policy.task_description`) from `workflows/agentic/config/environments/<env>.yaml`. Pass `--task-description` to override.
- Talks to an OpenAI-compatible endpoint via `--base-url` (default `http://localhost:8000/v1`) and `--model` (default `Qwen/Qwen3-VL-8B-Instruct`). Point both at a running vision-model server. Do not use text-only/code models such as `qwen3-coder-next`; offline annotation sends image inputs and requires a VLM.
- Keep every annotation artifact inside `workflows/agentic/runs/<run>/`. Do not create or access `/tmp/annotate_*` or other external temp directories.

## Start VLM

> **Skip this section if an OpenAI-compatible endpoint serving a vision model is already running** — just set `VLM_BASE_URL`/`VLM_MODEL` in Run to point at it. A local-agent server running `qwen3-coder-next` does not qualify because it is text-only. `annotator/vllm.sh` defaults to port `8000`, so starting it on top of an existing server collides; don't.

Run the steps below in order. Each step is a separate bash call; variables persist in the local agent's tmux session.

For readiness, use `workflows/agentic/annotator/vllm.sh ensure`. Do not replace it with raw `docker ps`, fixed sleeps, ad hoc model-listing HTTP probes, or separate manual wait steps; the helper owns the start-and-wait policy.

### Step 1 — start VLM (if needed)

Run this exact command. Do not add `sleep`, `status`, `curl`, or shell control operators around it:

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
"${REPO_ROOT}/workflows/agentic/annotator/vllm.sh" ensure
```

## Run (Offline HDF5)

Run the steps below in order. Each step is a separate bash call; variables persist in the local agent's tmux session.

### Step 1 — setup and resolve HDF5

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
ENV_ID=scissor_pick_and_place
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"

# LLM endpoint + model (OpenAI-compatible vLLM). Defaults match annotator/vllm.sh; override to use an
# external vision server — e.g. VLM_BASE_URL=http://localhost:8000/v1 VLM_MODEL=qwen3-vl-32b
VLM_BASE_URL="${VLM_BASE_URL:-http://localhost:8000/v1}"
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"

# Point HDF5_PATH at a real recording (absolute path). Recordings come from teleop, mimic, or
# validate (which writes data/verify.hdf5 under each runs/eval_* dir). If HDF5_PATH is not set,
# choose one recording: an HDF5 inside runs/.latest if present, otherwise the newest non-annotation
# HDF5 under runs/. "All recorded episodes" means all episodes inside this one HDF5.
HDF5_PATH="${HDF5_PATH:-}"
if [ ! -f "${HDF5_PATH}" ]; then
  LATEST_RUN="$(readlink -f "${RUNS_ROOT}/.latest" 2>/dev/null || true)"
  if [ -n "${LATEST_RUN}" ] && [ -d "${LATEST_RUN}" ]; then
    HDF5_PATH="$(
      find "${LATEST_RUN}" -name '*.hdf5' -type f -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | awk 'NR==1 { $1=""; sub(/^ /, ""); print; exit }'
    )"
  fi
fi
if [ ! -f "${HDF5_PATH}" ]; then
  HDF5_PATH="$(
    find "${RUNS_ROOT}" \( -path '*/annotate_*' -o -path '*/.latest' \) -prune -o \
      -name '*.hdf5' -type f -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr | awk 'NR==1 { $1=""; sub(/^ /, ""); print; exit }'
  )"
fi
if [ ! -f "${HDF5_PATH}" ]; then
  echo "annotate: set HDF5_PATH to an existing .hdf5 (got '${HDF5_PATH:-<unset>}'). Candidates:" >&2
  find "${RUNS_ROOT}" \( -path '*/annotate_*' -o -path '*/.latest' \) -prune -o \
    -name '*.hdf5' -type f -printf '%TY-%Tm-%Td %TH:%TM  %p\n' 2>/dev/null | sort -r | head
  exit 1
fi

RUN_DIR="${RUNS_ROOT}/annotate_${ENV_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/data" "${RUN_DIR}/logs" "${RUN_DIR}/tmp"
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/.latest"
```

### Step 2 — annotate offline

```bash
TMPDIR="${RUN_DIR}/tmp" "${REPO_ROOT}/workflows/agentic/annotator/run.sh" \
  --env "${ENV_ID}" \
  --base-url "${VLM_BASE_URL}" \
  --model "${VLM_MODEL}" \
  --output "${RUN_DIR}/annotations.jsonl" \
  offline \
  --hdf5-path "${HDF5_PATH}" \
  --filter "${RUN_DIR}/data/filtered.hdf5" \
  > "${RUN_DIR}/logs/annotator.log" 2>&1
```

### Step 3 — summarize annotations

```bash
SUCCESS_COUNT=$(grep -c '"success": true' "${RUN_DIR}/annotations.jsonl" 2>/dev/null || true)
FAILURE_COUNT=$(grep -c '"success": false' "${RUN_DIR}/annotations.jsonl" 2>/dev/null || true)
printf 'annotations: success=%s failure=%s\n' "${SUCCESS_COUNT}" "${FAILURE_COUNT}"
grep -E "Traceback|Error|FAILED" "${RUN_DIR}/logs/annotator.log" || true
```

### Step 4 — stop VLM (only if you started it in Start VLM)

Skip when using an external server (e.g. the local-agent one) — it would kill that server.

```bash
"${REPO_ROOT}/workflows/agentic/annotator/vllm.sh" stop
```

## Live Mode

Annotate the latest camera frames from a **running** policy/Arena session over Zenoh (cameras default to the env config). Use only when such a session is already up and the user asks for live judging.

Run the steps below in order. Each step is a separate bash call; variables persist in the local agent's tmux session.

### Step 1 — setup

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
ENV_ID=scissor_pick_and_place
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"
VLM_BASE_URL="${VLM_BASE_URL:-http://localhost:8000/v1}"
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
RUN_DIR="${RUNS_ROOT}/annotate_live_${ENV_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/tmp"
```

### Step 2 — annotate live

```bash
TMPDIR="${RUN_DIR}/tmp" "${REPO_ROOT}/workflows/agentic/annotator/run.sh" \
  --env "${ENV_ID}" \
  --base-url "${VLM_BASE_URL}" \
  --model "${VLM_MODEL}" \
  --output "${RUN_DIR}/live.jsonl" \
  live \
  --count 5 \
  --interval 2.0 \
  --timeout 30.0
```

- `--count 0` runs forever; `--interval` is seconds between snapshots; `--timeout` is how long to wait for first frames from every camera.
- `--min-success-frames N` (needs a finite `--count`) exits non-zero unless at least N sampled snapshots pass — use it as a gate.
- `--dump-frames-dir DIR` saves sampled frames; add `--dump-frames-only` to dump without calling the VLM.
- `--cameras a,b` overrides the env's Zenoh camera names.

## Verify

- `annotations.jsonl` exists.
- Filtered HDF5 exists when `--filter` was passed.
- Tally success/failure counts from the JSONL before reporting.

## Prerequisites

- Workflow set up via [[i4h-workflow-setup]] (the `.venv` must exist).
- An existing HDF5 recording to annotate (set `HDF5_PATH` to an absolute path; the Run block lists candidates if it's unset or wrong).
- A reachable OpenAI-compatible endpoint serving a vision model — either start the annotator's own (`annotator/vllm.sh start`) or point `VLM_BASE_URL`/`VLM_MODEL` at an existing vision server. The current `qwen3-coder-next` local-agent endpoint is not sufficient because it is text-only.
- Annotation is optional — only run it when the user requests labels.

## Limitations

- Annotation is optional and is not run during validation unless requested.
- Requires a reachable OpenAI-compatible vLLM server; defaults to `localhost:8000/v1`.
- Live mode applies only when a policy/Arena session is already running and the user requests live judging.
- The annotator reads task text from the env YAML; override per-run with `--task-description`.

## Troubleshooting

- **Error:** `.venv` not found / module import fails - Cause: workflow not set up. Fix: run [[i4h-workflow-setup]] first.
- **Error:** connection refused at `localhost:8000/v1` - Cause: no vLLM at `VLM_BASE_URL`. Fix: start one (`annotator/vllm.sh start`) or set `VLM_BASE_URL`/`VLM_MODEL` to a running server.
- **Error:** model not found / 404 from the endpoint - Cause: `VLM_MODEL` is not the id the server actually serves. Fix: set `VLM_MODEL` to the served name (e.g. `qwen3-vl-32b`; check `curl ${VLM_BASE_URL}/models`).
- **Error:** image input unsupported / bad request from a text model - Cause: the endpoint is serving a text-only/code model such as `qwen3-coder-next`. Fix: use a vision model endpoint such as Qwen3-VL for annotation.
- **Error:** input HDF5 not found - Cause: `HDF5_PATH` unset or not a real file. Fix: pick an absolute path from the candidates the Run block prints.
- **Error:** filtered HDF5 missing - Cause: `--filter` was not passed. Fix: add `--filter <path>` to write the filtered dataset.

## Final Response

Report env, input HDF5, annotations path, filtered HDF5 (if any), success/failure counts, VLM blockers.
