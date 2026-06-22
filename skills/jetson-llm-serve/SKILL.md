---
name: jetson-llm-serve
description: Stand up vLLM or SGLang serving on Jetson, using upstream vLLM on Thor and Orin JetPack 7.2+, and NVIDIA-AI-IOT vLLM on older Orin.
version: 0.0.1
license: "Apache-2.0"
metadata:
  author: "Jetson Team"
  tags: [jetson, llm, serving]
  languages: [markdown]
  data-classification: public
---

# Jetson LLM Serve

Encodes the [Jetson AI Lab GenAI tutorial](https://www.jetson-ai-lab.com/tutorials/genai-on-jetson-llms-vlms/): on Orin JetPack 7.2 / L4T r39+, use upstream vLLM 0.20+ (`vllm/vllm-openai:latest`); on older Orin, pick the NVIDIA-AI-IOT prebuilt vLLM container; on Thor, use upstream vLLM 0.20+ or validated native vLLM 0.20+, and use NVIDIA SGLang 26.01 (`nvcr.io/nvidia/sglang:26.01-py3`, SGLang 0.5.5.post2) when SGLang is requested. Set MAXN, make Hugging Face credentials/cache available, and launch an OpenAI-compatible server. Works for both LLMs and VLMs.

## Purpose

Provide a Jetson-appropriate serving recipe for an LLM or VLM using vLLM or SGLang, including runtime path, launch command, endpoint, and verification step.

## When to use

- "Run / serve / host this model on a Jetson."
- "Start a vLLM server I can hit from Open WebUI / my app."
- After `jetson-inference-mem-tune` produced launch flags and the user wants to actually start the server.

For recipe-only questions, answer from this document without starting
containers. Run live pre-flight checks only when the user asks you to check this
device or execute the deployment.

## Prerequisites

- Run on the Jetson host or a shell with Docker access to the Jetson GPU runtime.
- Know the target Jetson generation (`thor` or `orin`) and the model identifier or local checkpoint path.
- Use `HF_TOKEN` only when the model is gated/private; public models should omit the token environment variable.
- Use `jetson-inference-mem-tune` first when memory headroom or launch flags are uncertain.

## Instructions

For recipe questions, provide a complete launch recipe instead of trying to call
`jetson-llm-serve` as a tool. A complete answer includes:

- The Jetson-appropriate runtime path: upstream vLLM 0.20+ (`vllm/vllm-openai:latest`) or NVIDIA SGLang 26.01 (`nvcr.io/nvidia/sglang:26.01-py3`, SGLang 0.5.5.post2) on Thor, NVIDIA-AI-IOT vLLM container on older Orin, or upstream vLLM 0.20+ on Orin JetPack 7.2 / L4T r39+.
- The model checkpoint / Hugging Face repo the user named.
- A `docker run` + server command sketch with `--host 0.0.0.0 --port 8000`.
- The OpenAI-compatible endpoint: `http://<jetson-ip>:8000/v1`.
- A verification step such as `curl http://localhost:8000/v1/models`.

For VLM questions, explicitly say the VLM uses the same vLLM serving flow as an
LLM with a different vision-language checkpoint. Do not omit `vLLM` or the
Jetson container when answering VLM prompts.

## Step 1 — Pick the runtime path (per Jetson family)

Use upstream vLLM 0.20+ on Thor (`vllm/vllm-openai:latest`, or a validated native vLLM 0.20+ install). On Orin JetPack 7.2 / L4T r39+, use upstream vLLM 0.20+ (`vllm/vllm-openai:latest`). On older Orin releases, use the **NVIDIA-AI-IOT prebuilt vLLM image** ([packages](https://github.com/orgs/NVIDIA-AI-IOT/packages)) because it ships the correct CUDA / cuDNN / TensorRT stack for that JetPack. Use NVIDIA SGLang 26.01 (`nvcr.io/nvidia/sglang:26.01-py3`, SGLang 0.5.5.post2) on Thor when the user asks for SGLang, RAG, tool-use, or programmable serving; do not recommend native upstream SGLang on Orin unless a JetPack-matched release explicitly supports it.

| Jetson family               | Runtime path                                      |
|-----------------------------|---------------------------------------------------|
| Thor (T5000, T4000)         | upstream vLLM 0.20+ (`vllm/vllm-openai:latest`) or NVIDIA SGLang 26.01 (`nvcr.io/nvidia/sglang:26.01-py3`, SGLang 0.5.5.post2) |
| AGX Orin / Orin NX / Nano   | Orin JetPack 7.2 / L4T r39+: upstream vLLM 0.20+ (`vllm/vllm-openai:latest`); older Orin: `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` |

To detect the silicon era for image tags:

1. Source the detector so exports survive in your shell:
   ```bash
   . skills/jetson-diagnostic/scripts/detect_jetson.sh
   ```
2. Check `JETSON_GENERATION` (`thor` or `orin`) and choose the matching runtime path from the table above.
3. Use `JETSON_PRODUCT_LINE` for a finer bucket such as `thor-agx` or `orin-nano`; `JETSON_SKU` remains the legacy identifier.

Do not use `bash skills/jetson-diagnostic/scripts/detect_jetson.sh` when you need exported variables in the caller; running with `bash` uses a subshell.

## Step 2 — Set MAXN power mode

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
```

Skip this only if the user explicitly asks for a power-constrained run; otherwise benchmark and serving numbers will be inconsistent.

## Step 3 — Run the server

On Thor with vLLM, use upstream vLLM 0.20+ (`vllm/vllm-openai:latest`) or a validated native vLLM 0.20+ install:

```bash
docker run --rm -it --runtime nvidia --network host --ipc host --name vllm \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="$HF_TOKEN" \
  vllm/vllm-openai:latest \
  vllm serve <hf-repo-id> \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.75 \
    --tensor-parallel-size 1
```

On Orin JetPack 7.2 / L4T r39+, use upstream vLLM 0.20+ (`vllm/vllm-openai:latest`). On older Orin releases, use the NVIDIA-AI-IOT container:

```bash
docker run --rm -it --runtime nvidia --network host --name vllm \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="$HF_TOKEN" \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
  vllm serve <hf-repo-id> \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 1
```

`HF_TOKEN` is required only for gated/private Hugging Face models; omit the `-e HF_TOKEN="$HF_TOKEN"` line for public models that do not need Hub authentication. Passing `HF_TOKEN` as an environment variable can expose it through Docker inspect output, process metadata, or logs on shared systems. Prefer the narrowest-scoped token possible, rotate/revoke it after shared-container use, and use a mounted credential file or Docker secret when the deployment environment supports that pattern.

Wait for `Application startup complete.` Server is on `http://0.0.0.0:8000/v1`.

For SGLang on Thor, use NVIDIA SGLang 26.01 (`nvcr.io/nvidia/sglang:26.01-py3`), which packages SGLang 0.5.5.post2 and lists Jetson Thor support. Do not judge Thor SGLang support from older prerelease SGLang results:

```bash
docker run --rm -it --runtime nvidia --network host --ipc host --name sglang \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="$HF_TOKEN" \
  nvcr.io/nvidia/sglang:26.01-py3 \
  python3 -m sglang.launch_server \
    --model-path <hf-repo-id> \
    --host 0.0.0.0 \
    --port 8000 \
    --mem-fraction-static 0.60 \
    --max-running-requests 8
```

Use SGLang when the user needs RAG/tool-use workflows, structured generation, or
SGLang-specific scheduling. For plain high-throughput OpenAI-compatible serving,
prefer vLLM unless the user asks for SGLang.

### SKU-appropriate defaults

| Knob                       | Orin Nano / NX | AGX Orin / Thor |
|----------------------------|----------------|-----------------|
| `--max-model-len`          | `4096`         | `8192`          |
| `--gpu-memory-utilization` | `0.85`         | `0.85`          |
| `--tensor-parallel-size`   | `1`            | `1`             |

If the server OOMs at startup, lower `--gpu-memory-utilization` by 0.05 and re-launch (or run `jetson-inference-mem-tune` for a workload-aware recommendation).

## Quantization preferences (matters more than the runtime)

For vLLM and SGLang, choose checkpoint formats by Jetson family:

| Jetson family | First choice | Acceptable fallback |
|---------------|--------------|---------------------|
| Thor | **NVFP4** when the model/runtime supports it | W4A16 |
| Orin Nano / NX | **W4A16** | AWQ or GPTQ 4-bit |
| AGX Orin | **W4A16** | AWQ or GPTQ 4-bit |

For llama.cpp and `Ollama`, use GGUF model quantization names instead: recommend **INT4 / Q4_K_M GGUF** on both Orin and Thor, and choose a smaller INT4 GGUF model if memory is tight. Do not call GGUF Q4_K_M a W4A16/AWQ/GPTQ model. NVFP4 is Thor-preferred and Thor-tuned for runtimes that support it.

## VLM mode

VLMs use the same flow as LLMs: same container, same `vllm serve` invocation, different vision-language checkpoint. The container handles image preprocessing. For a VLM-specific browser UI, use the [`live-vlm-webui`](https://github.com/orgs/NVIDIA-AI-IOT/packages) container; for a generic chat UI for either, use Open WebUI pointed at `http://<jetson-ip>:8000/v1`.

## Do not fabricate device capacity

Do not invent RAM totals, free-memory values, model sizes, JetPack versions, or
SKU/variant names when giving a serving recipe. If capacity matters, either run
the live pre-flight checks (when execution is allowed) or hand off to
`jetson-inference-mem-tune` / `jetson-memory-audit`. If live data is not
available, say the value is unknown and provide conservative defaults instead
of quoting a made-up number.

## Pre-flight checklist (the agent should verify before running Step 3)

- [ ] On a Jetson (`/proc/device-tree/model` contains `NVIDIA Jetson`).
- [ ] `nvpmodel -q` reports a recognized max-performance mode: `MAXN` or `MAXN_*` such as `MAXN_SUPER`. Wattage-named modes should be reported as warnings unless the user explicitly confirms they are the intended benchmark mode for that device.
- [ ] On Thor, check whether MIG is enabled before launching (`nvidia-smi -L` and `nvidia-smi mig -lgi`). If MIG is enabled, warn that vLLM/SGLang may see only a MIG slice or no CUDA device.
- [ ] On Thor with MIG or display/camera contention, inspect GPU users with `sudo lsof /dev/nvidia*`. Display managers, `Xorg`/GNOME, or `nvargus-daemon` may hold GPU device files; do not stop services or change MIG mode unless the user explicitly approves.
- [ ] No container named `vllm` already running (`docker ps --format '{{.Names}}'`); otherwise `docker rm -f vllm` first.
- [ ] Docker exposes the NVIDIA runtime (`docker info | grep -i 'runtimes.*nvidia'`), or a GPU-enabled container can run `nvidia-smi`.
- [ ] `~/.cache/huggingface` exists; `HF_TOKEN` is set if the model is gated.

## Limitations

- This skill provides serving commands and pre-flight checks; it does not benchmark the deployed server.
- Container tags such as `latest` are mutable. For release or compliance deployments, pin a digest and record it with the deployment notes.
- vLLM and SGLang memory limits still depend on model architecture, quantization, context length, and concurrent request count. Use `jetson-inference-mem-tune` when a command OOMs or memory headroom matters.
- Thor vLLM requires upstream vLLM 0.20+ or newer. Older upstream vLLM images may not support Thor / SM 11.0 correctly.
- Thor SGLang should use NVIDIA SGLang 26.01 or newer release notes that explicitly list Jetson Thor support. NVIDIA SGLang 26.01 contains SGLang 0.5.5.post2.
- On Thor, MIG, desktop display, or camera services can hide the full GPU from containers. This skill should detect and warn only; disabling MIG or stopping services such as `gdm3` or `nvargus-daemon` requires explicit user approval.
- Host-native vLLM/SGLang on Thor should be used only when that install is already validated on the target JetPack.

## Hand off to

- `jetson-llm-benchmark` to actually measure the deployed server.
- `jetson-speculative-decoding` to add EAGLE-3 / draft-model speculation by appending `--speculative-config '{...}'` to the `vllm serve` command above.
- `jetson-inference-mem-tune` if the server OOMs or is memory-bound.

## Source

[Jetson AI Lab — Introduction to GenAI on Jetson: How to Run LLMs and VLMs](https://www.jetson-ai-lab.com/tutorials/genai-on-jetson-llms-vlms/) and [NVIDIA-AI-IOT GHCR packages](https://github.com/orgs/NVIDIA-AI-IOT/packages).
