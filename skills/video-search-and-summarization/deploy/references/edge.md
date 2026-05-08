# Edge Deployment Reference (DGX Spark, AGX Thor, IGX Thor)

Base-profile deployment for edge platforms. **Shared-mode deployments MUST use
NVIDIA Nemotron Edge 4B as the LLM** — the Nano 9B NIM image has a broken
arm64 manifest (x86_64 binaries inside an arm64-tagged layer) and the Nano 9B
FP8 does not yet ship a DGX-Spark-optimized NIM profile.

Two supported paths on edge hardware:

1. **NVIDIA Nemotron Edge 4B** (FP8) — **required** for shared-mode on all edge
   platforms (Spark, AGX Thor, IGX Thor). Fits in ~25% of GPU memory, lets the
   VLM share the remaining budget. Uses a simplified planning prompt
   (`config_edge.yml`) that skips clarifying questions. **Requires `HF_TOKEN`**
   in the environment (weights pulled from Hugging Face).
2. **NVIDIA Nemotron Nano 9B v2 FP8** — fallback ONLY when a DGX-Spark-optimized
   FP8 NIM becomes available (track via the upstream blueprint's compose
   overrides). Do not use until that lands — the current `:1` tag will fail on
   arm64.

## When to pick which

| Situation | Model |
|---|---|
| DGX Spark shared mode | **Edge 4B (mandatory)** |
| IGX/AGX Thor shared mode | **Edge 4B (mandatory)** |
| DGX Spark remote-llm mode (LLM at launchpad endpoint) | Remote LLM — no local model needed |
| Ambiguous / multi-turn user queries on edge | Edge 4B (accept: no clarifying Q's) |
| Non-edge hardware (H100, L40S, RTX PRO) | Nano 9B v2 (standard NIM) |

## Prerequisites

- `NGC_CLI_API_KEY` (NIM containers)
- **`HF_TOKEN` — required** (Edge 4B weights pull from Hugging Face; shared mode on
  edge hardware is blocked without it)
- `NVIDIA_API_KEY` (agent-side)
- GPU freed: `docker ps` should show no running VSS or LLM containers before
  starting. Reboot the device if in doubt.

### HF_TOKEN verification

Before running the deploy, verify the token can reach the Edge 4B repo:

```bash
curl -sf -H "Authorization: Bearer $HF_TOKEN" \
    https://huggingface.co/api/models/nvidia/NVIDIA-Nemotron-Edge-4B-v2.1-EA-020126_FP8 \
    >/dev/null && echo "HF_TOKEN works" || echo "HF_TOKEN missing/invalid/no access"
```

If the model is gated, the token's owner must request access on the HF page.

## DGX Spark — Edge 4B + local Cosmos-Reason2-8B VLM

Start the LLM as a standalone vLLM container (port 30081):

```bash
export HF_TOKEN=$HF_TOKEN

docker run --gpus all -d --name nemotron-edge -p 30081:8000 \
    -e HF_TOKEN=$HF_TOKEN \
    nvcr.io/nvidia/vllm:26.02-py3 \
    python3 -m vllm.entrypoints.openai.api_server \
    --model nvidia/NVIDIA-Nemotron-Edge-4B-v2.1-EA-020126_FP8 \
    --trust-remote-code \
    --gpu-memory-utilization 0.25 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --port 8000
```

Key flags:
- `--gpu-memory-utilization 0.25` — leaves ~75% for the VLM NIM (which uses
  `NIM_KVCACHE_PERCENT=0.4` on Spark shared).
- `--tool-call-parser qwen3_coder` — Edge 4B is Qwen3-lineage; the parser
  must match the template.
- `--enable-auto-tool-choice` — agent workflow uses tool-calls.

Then deploy the agent workflow (LLM treated as "remote" since it's a
standalone vLLM, not a NIM):

```bash
export NVIDIA_API_KEY=$NVIDIA_API_KEY
export NGC_CLI_API_KEY=$NGC_CLI_API_KEY
export LLM_ENDPOINT_URL=http://localhost:30081
export VSS_AGENT_CONFIG_FILE=./deployments/developer-workflow/dev-profile-base/vss-agent/configs/config_edge.yml

deployments/dev-profile.sh up -p base \
    --use-remote-llm \
    --llm nvidia/NVIDIA-Nemotron-Edge-4B-v2.1-EA-020126_FP8 \
    --hardware-profile DGX-SPARK \
    --vlm-env-file deployments/nim/cosmos-reason2-8b/hw-DGX-SPARK-shared.env
```

The `--vlm-env-file` caps the VLM's KV cache at 40% so both models coexist.

## DGX Spark — Nano 9B v2 FP8 (both NIMs, no standalone vLLM)

```bash
# Make sure the Edge vLLM container is not running:
# docker stop nemotron-edge && docker rm nemotron-edge

deployments/dev-profile.sh up -p base \
    --hardware-profile DGX-SPARK \
    --llm nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8 \
    --vlm nvidia/cosmos-reason2-8b
```

Uses the default `config.yml` (full planning prompt with clarifying questions).

## AGX Thor / IGX Thor — Edge 4B + rtvi-vlm

On Thor, the VLM used by the blueprint is `rtvi-vlm` (not cosmos-reason2-8b),
and the LLM runs from a jetson-specific vLLM image:

```bash
export HF_TOKEN=$HF_TOKEN

docker run --gpus all -d --name nemotron-edge -p 30081:8000 \
    --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    -e HF_TOKEN=$HF_TOKEN \
    ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor \
    python3 -m vllm.entrypoints.openai.api_server \
    --model nvidia/NVIDIA-Nemotron-Edge-4B-v2.1-EA-020126_FP8 \
    --trust-remote-code \
    --gpu-memory-utilization 0.25 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --port 8000
```

Then:

```bash
export NVIDIA_API_KEY=$NVIDIA_API_KEY
export NGC_CLI_API_KEY=$NGC_CLI_API_KEY
export LLM_ENDPOINT_URL=http://localhost:30081
export VSS_AGENT_CONFIG_FILE=./deployments/developer-workflow/dev-profile-base/vss-agent/configs/config_edge.yml

# Uses the default 35% GPU budget for rtvi-vlm on Thor
deployments/dev-profile.sh up -p base \
    --use-remote-llm \
    --llm nvidia/NVIDIA-Nemotron-Edge-4B-v2.1-EA-020126_FP8 \
    --hardware-profile AGX-THOR
```

For **IGX Thor**: replace `AGX-THOR` with `IGX-THOR` in the `--hardware-profile` flag.

## AGX/IGX Thor — Nano 9B v2 FP8

```bash
# docker stop nemotron-edge && docker rm nemotron-edge
deployments/dev-profile.sh up -p base \
    --hardware-profile AGX-THOR \
    --llm nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8
```

## Caveats

- **Edge 4B skips clarifying questions.** `config_edge.yml` deliberately
  simplifies the planning prompt for smaller models. If the user asks
  ambiguously (e.g. "summarize the video" without specifying which), the
  agent won't ask back — it'll pick one or fail. Switch to Nano 9B v2 FP8
  if this matters for your use case.
- **Edge 4B is not a NIM.** It's a plain vLLM container — no
  `nvcr.io/nim/...` tag. `dev-profile.sh --use-remote-llm` points the
  agent at the local port 30081 as if it were a remote endpoint.
- **Tool-call parser.** Edge 4B requires `--tool-call-parser qwen3_coder`
  (Qwen3-lineage). Omitting it or using `llama3_json` breaks the agent's
  tool calls.
- **HF_TOKEN gate.** Edge 4B weights are pulled from Hugging Face at first
  run; a gated model, so your token needs access.
- **`config_edge.yml` may not be present** in older checkouts — verify
  `deployments/developer-workflow/dev-profile-base/vss-agent/configs/config_edge.yml`
  exists before running. If missing, pull the latest `feat/skills` or
  main branch.
- **The planning prompt in `config_edge.yml` must go BEYOND "don't ask
  clarifying questions".** Edge 4B's default behavior on terse planning
  prompts is to emit `[USER] <template>` — which `vss_agents/agents/top_agent.py`
  treats as direct-to-user clarification and short-circuits away from tool
  calls. The E2E video probe then returns planning output instead of actual
  agent responses. Your `config_edge.yml`'s `workflow.prompt` must include
  explicit tool-call rules and per-query plan shapes. Known-working shape:

  ```yaml
  workflow:
    prompt: |
      You are a routing agent for a video surveillance system.

      CRITICAL PLANNING RULES:
      - You MUST produce a numbered execution plan that calls tools.
      - NEVER output [USER] for video-related questions. ALWAYS call the
        appropriate tool.
      - For "What videos are available?" / "List videos":
        Plan must be "1. Call vst_video_list to retrieve the list of videos."
      - For "Generate a report for video X":
        Plan must include "1. Call report_agent with video_name=X"
      - For video content questions:
        Plan must include "1. Call video_understanding with the video name and question"

      ## Routing Rules:
        (copy the rest of config.yml's workflow.prompt verbatim)
  ```

  The key invariant: the Edge 4B model will not infer "call a tool" from
  a prose description of tools; it needs exact-phrase plan templates to
  match its pattern-completion behavior. This was surfaced during the
  Harbor eval run on SPARK (shared mode).

## Known ARM64 gotcha

`nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2:1` (the default `base` NIM
tag) ships a broken arm64 manifest — it declares arm64 but contains
x86_64 binaries. This is why the Edge 4B path is the recommended default
on Spark: it avoids the NIM entirely. If you must use a local NIM for the
LLM, pin to the Spark variant:

```
nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2-dgx-spark:1.0.0-variant
```

(currently not wired into the blueprint's `compose.yml` — follow-up to track).
