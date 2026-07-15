---
name: jetson-inference-mem-tune
description: Pick the serving stack and per-runtime memory flags (vLLM, SGLang, llama.cpp, TensorRT Edge-LLM) for an LLM/VLM workload on any NVIDIA Jetson.
version: 0.0.1
license: "Apache-2.0"
metadata:
  author: "Jetson Team"
  tags: [jetson, inference, memory]
  languages: [python]
  data-classification: public
---

# Jetson Inference Memory Tuning

Recommends an inference runtime and the specific memory-related flags to pass to it, given the Jetson SKU/variant and the user's workload. Does not include quantization recipe selection — that lives in the model-benchmarking skill — but it does point at the precision floor each runtime can serve efficiently.

## Purpose

Turn a live `jetson-memory-audit` snapshot into runtime and launch-flag recommendations for LLM/VLM serving on Jetson. Use this when the user needs to fit a model, reduce OOM risk, or switch to a lower-memory serving stack.

## When to use

- "Which serving stack should I use on Orin Nano 8 GB to run a 7B model?"
- "vLLM is OOMing — what should `--gpu-memory-utilization` and `--max-model-len` be?"
- "Same model, less memory — can I switch from vLLM to llama.cpp?"
- After `jetson-memory-audit` shows a model server is the top NvMap / PSS consumer.

## Prerequisites

- Start with a current `jetson-memory-audit/scripts/audit.sh` JSON snapshot from the target Jetson.
- Know the intended workload: `llm-server`, `vlm-server`, `embedding`, or `rag`.
- If the user gives a desired free-memory target, pass it as `--target-mb`; otherwise let the script use SKU defaults.

## Available Scripts

| Script | Purpose | Arguments |
|--------|---------|-----------|
| `scripts/recommend.py` | Reads an audit JSON and emits runtime plus launch-flag recommendations. | `--audit PATH`, `--runtime`, `--workload`, `--target-mb`, `--human`. |

If your agent runtime supports `run_script`, invoke `run_script("scripts/recommend.py", ["--audit", "/tmp/audit.json", "--runtime", "auto", "--workload", "llm-server"])` and summarize the returned JSON. Otherwise run it with `python3` from the repository root.

## Instructions

1. Run `jetson-memory-audit/scripts/audit.sh` to capture the device baseline.
2. Run `scripts/recommend.py --audit /tmp/audit.json --runtime auto --workload llm-server --target-mb 6000` to get a JSON of runtime + flag recommendations.
3. The agent presents the suggested runtime and the exact CLI flags. The user (or an outer agent) launches / restarts the server with those flags.
4. Re-run the audit to verify.

## Expected workflow

Use `scripts/recommend.py` for the specific prompt and answer from the JSON it emits. If direct execution is blocked, run it as `python3 {baseDir}/scripts/recommend.py ...`.

- For vLLM OOM prompts, run with `--runtime vllm --workload llm-server` and include concrete `--gpu-memory-utilization=<0.x>` and `--max-model-len=<number>` values from `launch_flags`.
- For "lowest memory" or Orin Nano 8 GB prompts, run with `--runtime auto --workload llm-server`; prefer the runtime in the JSON and explicitly mention the GGUF / 4-bit tradeoff when it selects `llama-cpp`.
- For SGLang prompts, run with `--runtime sglang` and quote `--mem-fraction-static`, `--max-running-requests`, and any context/KV-cache note.
- For "switch from vLLM to llama.cpp" prompts, run with `--runtime llama-cpp` and quote `-ngl`, `-c`, and `--no-mmap`.

## Limitations

- Recommendations are only as fresh as the audit JSON. Re-run `jetson-memory-audit` after stopping services, changing power mode, or restarting model servers.
- The script estimates memory pressure from SKU defaults and audit totals; model-specific KV-cache, quantization, and tokenizer behavior can still require benchmarking.
- This skill emits flags only. It does not start, stop, or restart model servers.

## Error handling

- Exit `2`: the audit JSON could not be read, parsed, or did not contain valid numeric memory fields. Ask the user to rerun `jetson-memory-audit/scripts/audit.sh`.
- Exit `3`: unsupported runtime or workload request. Re-run with one of the `--runtime` and `--workload` values listed in `scripts/recommend.py --help`.
- Empty or missing `launch_flags`: do not invent fallback flags. Report the script failure and ask for a fresh audit or a supported runtime.

## Output contract for `recommend.py`

```json
{
  "sku": "orin-nx",
  "variant": "orin-nx-16gb",
  "mem_total_gb": 16,
  "runtime": "vllm",
  "rationale": "Highest throughput at this memory budget given continuous batching + paged attention.",
  "launch_flags": [
    "--gpu-memory-utilization=0.55",
    "--max-model-len=4096",
    "--max-num-seqs=8",
    "--enable-prefix-caching"
  ],
  "alternatives": [
    { "runtime": "llama-cpp", "rationale": "Lower memory floor with GGUF Q4_K_M.", "launch_flags": ["-ngl 28", "-c 4096", "--no-mmap"] }
  ],
  "notes": ["Lower --gpu-memory-utilization further if you also run a small VLM alongside."]
}
```

## Runtimes covered

| Runtime              | Best for                                                  | Key memory knobs                                                                           | Preferred install path |
|----------------------|-----------------------------------------------------------|--------------------------------------------------------------------------------------------|------------------------|
| **llama.cpp**        | Tightest budget; GGUF; Orin Nano-class                    | `-ngl`, `-c`, `--mlock`, `--no-mmap`                                                       | `ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-{orin,thor}` |
| **vLLM**             | High-throughput serving with continuous batching          | `--gpu-memory-utilization`, `--max-model-len`, `--max-num-seqs`, `--enable-prefix-caching` | Thor and Orin JetPack 7.2 / L4T r39+: upstream vLLM 0.20+ (`vllm/vllm-openai`) container or validated native vLLM 0.20+. Older Orin: NVIDIA-AI-IOT image |
| **SGLang**           | Programmable workflows (RAG, tool use, structured output) | `--mem-fraction-static`, `--mem-fraction-dynamic`, `--max-running-requests`                | Thor: NVIDIA SGLang 26.01 (`nvcr.io/nvidia/sglang:26.01-py3`, SGLang 0.5.5.post2). Orin: JetPack-matched environment |
| **TensorRT Edge-LLM**| NVIDIA-tuned production serving                           | Build profile per SKU; paged-KV; KV reuse                                                  | Vendor docs for the target JetPack |

> For Orin JetPack 7.2 / L4T r39+, upstream vLLM 0.20+ is supported. For older Orin releases, prefer NVIDIA-AI-IOT prebuilt vLLM images where available because they ship the matching CUDA/cuDNN/TensorRT stack for JetPack. For Thor, prefer upstream vLLM 0.20+ (`vllm/vllm-openai`) or a validated native vLLM 0.20+ install; for SGLang use NVIDIA SGLang 26.01 (`nvcr.io/nvidia/sglang:26.01-py3`, SGLang 0.5.5.post2) or newer NVIDIA SGLang release notes that explicitly list Jetson Thor support. Do not force an Orin-specific Jetson container path on Thor, and do not assume native upstream SGLang support on Orin.

## Quantization recommendations

Use runtime-specific quantization names. vLLM and SGLang usually consume Hugging Face checkpoints such as W4A16, AWQ, GPTQ, FP16, or NVFP4. llama.cpp and Ollama consume GGUF models, so recommend INT4/Q4_K_M-style GGUF instead.

| Runtime family | Jetson family | First choice | Fallback |
|----------------|---------------|--------------|----------|
| vLLM / SGLang | Thor | NVFP4 when the model/runtime supports it | W4A16 |
| vLLM / SGLang | Orin Nano / NX | W4A16 | AWQ or GPTQ 4-bit |
| vLLM / SGLang | AGX Orin | W4A16 | AWQ or GPTQ 4-bit |
| llama.cpp / Ollama | Orin and Thor | GGUF INT4 / Q4_K_M | Smaller INT4 GGUF model if memory is tight |

Do not describe GGUF Q4_K_M as W4A16/AWQ/GPTQ. Do not compare Thor NVFP4 results with Orin W4A16 results unless the output includes a `quant` field.

## Runtime command guidance

Use `recommend.py` as the source of truth for memory knobs, then place its `launch_flags` into the matching serving command. Keep the command guidance in this skill instead of separate small reference files so agents ingest one complete instruction set.

For vLLM on Orin with JetPack 7.2 / L4T r39+, use upstream vLLM 0.20+ (`vllm/vllm-openai:latest`). On older Orin releases, use the NVIDIA-AI-IOT image:

```bash
docker run --rm -it --runtime nvidia --network host --name vllm \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="$HF_TOKEN" \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
  vllm serve <hf-model-id-or-local-path> \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.60 \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --enable-prefix-caching
```

For vLLM on Thor, use upstream vLLM 0.20+ (`vllm/vllm-openai:latest`) unless host-native vLLM 0.20+ is already installed and validated:

```bash
docker run --rm -it --runtime nvidia --network host --ipc host --name vllm \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="$HF_TOKEN" \
  vllm/vllm-openai:latest \
  vllm serve <hf-model-id-or-local-path> \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.75 \
    --max-model-len 8192 \
    --max-num-seqs 32 \
    --enable-prefix-caching
```

Thor vLLM note: do not judge Thor support from pre-0.20 vLLM results; upstream vLLM support starts at vLLM 0.20+.

For SGLang on Thor, use NVIDIA SGLang 26.01 (`nvcr.io/nvidia/sglang:26.01-py3`). NVIDIA SGLang 26.01 contains SGLang `0.5.5.post2` and explicitly lists Jetson Thor support. Avoid judging Thor support from older prerelease SGLang results. Avoid recommending `gpt-oss` or FP8 paths on Thor unless newer NVIDIA SGLang release notes say those known issues are fixed.

```bash
docker run --rm -it --runtime nvidia --network host --ipc host --name sglang \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="$HF_TOKEN" \
  nvcr.io/nvidia/sglang:26.01-py3 \
  python3 -m sglang.launch_server \
    --model-path <hf-model-id-or-local-path> \
    --host 0.0.0.0 \
    --port 8000 \
    --mem-fraction-static 0.60 \
    --max-running-requests 8
```

For llama.cpp, use the NVIDIA-AI-IOT llama.cpp image when available, or the `llama-server` binary from a JetPack-matched build. Start with GGUF INT4 / Q4_K_M on both Orin and Thor; choose a smaller INT4 GGUF model if the audit shows tight memory.

```bash
docker run --rm -it --runtime nvidia --network host --name llama-cpp \
  -v "$PWD/models:/models:ro" \
  ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-<orin-or-thor> \
  llama-server \
    -m /models/<model>.gguf \
    --host 0.0.0.0 \
    --port 8000 \
    -ngl 28 \
    -c 4096 \
    --no-mmap \
    --flash-attn
```

## Procedure (the script encodes this)

1. Pick the lightest runtime that satisfies the user's required features (continuous batching? structured generation? CPU offload?).
2. Pick the lowest precision that meets the user's accuracy bar (model-benchmarking skill).
3. Sweep the runtime's memory knobs (start with `gpu-memory-utilization` for vLLM, `n-gpu-layers` and `ctx-size` for llama.cpp) to find the minimum footprint that sustains target throughput.
4. Re-measure with `jetson-memory-audit`.

## Safety

Read-only. The skill never starts, stops, or restarts a model server. It emits flags; the user (or an outer orchestration agent) is responsible for invoking the runtime.
