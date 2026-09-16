---
name: i4h-workflow-dataset-annotate
description: Grade or filter workflow HDF5 episodes with an OpenAI-compatible vision model. Use for visual success labels; do not use for replay, policy evaluation, or recordings without frames.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  tags:
    - isaac-for-healthcare
    - i4h
    - dataset
    - annotation
    - vlm
---

# Annotate Workflow Recordings

## Purpose

Grade sampled camera frames against a natural-language success criterion while keeping VLM labels separate from simulator success.

## Instructions

1. Run the checkout resolver and select one HDF5 and one criterion.
2. Test camera sampling and the vision endpoint.
3. Run grading and optional filtering on every selected episode.
4. Compare verdict and output counts.

## Resolve input and criterion

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
find runs -name '*.hdf5' -type f -printf '%T@ %p\n' | sort -nr | head
```

Treat the resolver above as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `workflows/i4h_workflows`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

Use the explicit/current-chain HDF5. “All recorded episodes” means every episode in that selected file, not every historical run. Inspect it and use the user's explicit success criterion when supplied; otherwise combine the source Scene manifest instruction with the workflow's visible terminal goal semantics. Phrase placement success as the object reaching and remaining at its target, not as the robot continuing to hold it.

## Resolve the endpoint

Use a caller-provided OpenAI-compatible vision endpoint/model first. Local Agent exposes that configuration as `I4H_AGENT_VL_BASE_URL`, `I4H_AGENT_VL_MODEL`, and either `I4H_AGENT_VL_API_KEY` or `I4H_AGENT_API_KEY`. Map those generic agent variables to the annotator without printing the credential:

```bash
VLM_ARGS=()
if [ -n "${I4H_AGENT_VL_BASE_URL:-}" ] && [ -n "${I4H_AGENT_VL_MODEL:-}" ]; then
  I4H_VLM_URL="${I4H_AGENT_VL_BASE_URL%/}"
  case "$I4H_VLM_URL" in */v1) ;; *) I4H_VLM_URL="$I4H_VLM_URL/v1" ;; esac
  export I4H_VLM_URL
  export OPENAI_API_KEY="${I4H_AGENT_VL_API_KEY:-${I4H_AGENT_API_KEY:-EMPTY}}"
  VLM_ARGS=(--model "$I4H_AGENT_VL_MODEL")
fi
```

If no caller-provided endpoint/model is available, start the repository's local service:

```bash
tools/annotator/scripts/vllm.sh ensure
```

Record whether this invocation started it. Do not hard-code a model name in the skill; use the CLI/service defaults unless the user supplies one.

## Dry-run sampling when needed

```bash
uv run --project tools/annotator i4h-annotator \
  --task "<success criterion>" \
  --dry-run \
  offline /absolute/path/to/recording.hdf5
```

Use this to verify cameras and sampled frames without transmitting images.

## Grade and filter

```bash
RUN_DIR="$(pwd)/runs/<workflow>/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
uv run --project tools/annotator i4h-annotator \
  --task "<success criterion>" \
  "${VLM_ARGS[@]}" \
  offline /absolute/path/to/recording.hdf5 \
  --write
```

Add global `--base-url`, `--model`, `--camera`, or `--frames` only when selected. Add offline `--node` only for a requested segment. Add `--filter "$RUN_DIR/filtered.hdf5"` only when filtering was requested; a summarize-only prompt must grade all episodes without requiring at least one success. Keep credentials in environment variables; never print them.

Stop the local VLM only if this invocation started it:

```bash
tools/annotator/scripts/vllm.sh stop
```

## Verify

Inspect the annotator summary. If filtering was requested, also inspect the filtered file:

```bash
uv run --project tools/dataset i4h-dataset inspect "$RUN_DIR/filtered.hdf5" --segments
```

Require a verdict for every selected episode and reconcile pass/fail counts plus filtered counts when applicable. Treat endpoint errors, absent cameras, partial writes, and unexplained zero-episode output as failure. An all-failure verdict set is a valid completed grading run for summarize-only prompts; it is not a valid filtered dataset.

## Troubleshooting

Check camera sampling before endpoint/authentication errors. Never accept partial writes or a filtered file with an unexplained zero count.

## Prerequisites

Require a readable workflow HDF5 with camera frames and, unless dry-running, a reachable OpenAI-compatible vision endpoint.

## Limitations

Visual grading cannot recover missing frames or prove simulator state that is not visible.

## Examples

- `Run annotation on all recorded episodes and summarize.` → select the current HDF5, grade every episode, verify the filtered file, and report pass/fail counts.

## Completion gate

Report source HDF5, selected criterion/camera/model/endpoint origin, graded pass/fail counts, filtered path/count when requested, dry-run result if used, and local-service cleanup.
