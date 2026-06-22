---
name: jetson-llm-benchmark
description: Benchmark Jetson LLM/VLM serving performance across vLLM, llama.cpp, and Ollama with structured JSON output.
version: 0.0.2
license: "Apache-2.0"
metadata:
  author: "Jetson Team"
  tags: [jetson, llm, benchmark]
  languages: [bash]
  data-classification: public
---

# Jetson LLM Benchmark

Reproducible Jetson benchmarks with **structured JSON output** so an agent can compare runs. Encodes the workflow from the [Jetson AI Lab GenAI Benchmarking tutorial](https://www.jetson-ai-lab.com/tutorials/genai-benchmarking/).

## Purpose

Measure deployed LLM latency and throughput on a Jetson target using the correct
runtime-specific benchmark wrapper. Use the JSON output to compare models,
runtime flags, power modes, and before/after tuning changes.

## Prerequisites

- Run on the Jetson device that hosts the model runtime.
- For vLLM, start the OpenAI-compatible vLLM server first and know the served
  model ID.
- For Ollama, ensure the Ollama daemon is reachable at `--endpoint` and the
  named model is already pulled.
- For llama.cpp/GGUF, provide a readable `.gguf` model path on the host.
- Put the device in the intended power mode before measuring. MAXN is preferred
  for comparable performance numbers.

## Available Scripts

| Script | Purpose | Arguments |
|--------|---------|-----------|
| `scripts/bench_vllm.sh` | Runs `vllm bench serve` against a running OpenAI-compatible vLLM server. | `--model`, `--endpoint`, `--concurrency`, `--input-len`, `--output-len`, `--num-prompts`, `--no-warmup`, `--container`, `--native`. |
| `scripts/bench_llama_cpp.sh` | Runs `llama-bench` for a local GGUF model through the Jetson-appropriate NVIDIA-AI-IOT llama.cpp container. | `--model`, `--n-prompt`, `--n-gen`, `--n-gpu-layers`, `--threads`, `--container`. |
| `scripts/bench_ollama.sh` | Benchmarks a local or containerized Ollama daemon through the `/api/generate` REST API. | `--model`, `--endpoint`, `--num-prompts`, `--input-len`, `--output-len`, `--no-warmup`. |

If your agent runtime supports `run_script`, invoke the selected wrapper directly with the user-provided model identifier or local model path, then summarize the returned JSON. Otherwise run the wrapper with `bash {baseDir}/scripts/<wrapper-name> ...`.

## Instructions

Always use the matching wrapper script for the runtime — do **not** call the underlying `vllm bench serve`, `llama-bench`, or `curl` against `/api/generate` by hand:

- vLLM → `scripts/bench_vllm.sh` (required for the vLLM path)
- llama.cpp / GGUF → `scripts/bench_llama_cpp.sh` (required for the GGUF path)
- Ollama → `scripts/bench_ollama.sh` (required for the Ollama path)

These wrappers handle warmup, the NVIDIA-AI-IOT container selection, and JSON emission. Calling the underlying tool directly will not satisfy the output contract below.

For "how do I benchmark/measure" questions, first run the matching wrapper with
`--help` to verify the exact options, then answer with the wrapper command. Do
not run a full benchmark unless the user asks you to execute it or the required
server/model path is already confirmed.

## Expected Workflow

Pick exactly one wrapper based on the runtime the user named, and invoke that
wrapper with `--help` before composing the answer. Do not merely mention the
script name. If the runtime does not execute scripts relative to the skill
directory, use `{baseDir}/scripts/<wrapper-name>`.

- Existing vLLM OpenAI-compatible server at `localhost:8000`:
  `{baseDir}/scripts/bench_vllm.sh --help`, then show a command using
  `--concurrency 1,8` and the served model ID.
- llama.cpp / GGUF / `llama-server`: `{baseDir}/scripts/bench_llama_cpp.sh
  --help`, then show a command for the GGUF model path and report that
  prompt/generation speed maps to TTFT, ITL/TPOT, and throughput.
- Ollama: `{baseDir}/scripts/bench_ollama.sh --help`, then show a command with
  `--model <ollama-tag>`. Do not use vLLM or llama.cpp wrappers for Ollama.

## When to use

- "Benchmark / measure / compare X on this Jetson."
- After `jetson-llm-serve` to actually quantify the deployment.
- Before/after applying flags from `jetson-inference-mem-tune` to confirm the change helped.

## Three paths — pick by runtime

### A. vLLM (preferred for parity with how things are served)

Server must already be running (use `jetson-llm-serve`). Run **`bench_vllm.sh`**:

```bash
scripts/bench_vllm.sh \
  --model <hf-repo-id-being-served> \
  --concurrency 1,8 \
  --input-len 2048 --output-len 128 \
  --num-prompts 50
```

Uses the Jetson-appropriate benchmark client path: upstream vLLM 0.20+ container
`vllm/vllm-openai:latest` on Thor and Orin JetPack 7.2 / L4T r39+,
or the NVIDIA-AI-IOT vLLM benchmark container
`ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` on older Orin. Pass
`--native` only when host-native vLLM is already installed and validated. It
runs against `http://localhost:8000/v1`. **Always do a warmup pass first** (~10
prompts, discarded) before the measured run — Jetson has cold caches and JIT'd
kernels.

### B. Ollama (for models served by a running Ollama daemon)

No benchmark container needed. Uses Ollama's `/api/generate` REST API directly —
timing data (TTFT, ITL, throughput) comes from the response JSON, so no
`--verbose` parsing is required.

**Prerequisite:** the Ollama daemon must be reachable at `--endpoint` (default
`http://localhost:11434`). This works whether Ollama is installed natively or
running in a container that exposes that port. If the daemon is not running,
the script will tell you whether Ollama is installed but stopped (`ollama serve`
to fix) or not installed at all (install instructions printed). Run
**`bench_ollama.sh`** (do not roll your own `curl` against `/api/generate`):

```bash
scripts/bench_ollama.sh \
  --model <ollama-model-name> \
  --num-prompts 20 \
  --input-len 512 --output-len 128
```

Runs sequential single-stream requests (concurrency=1). Ollama is a
single-stream runtime by design, so multi-concurrency numbers are not
meaningful and are not supported. Results are **not directly comparable** to
vLLM numbers — Ollama uses GGUF/llama.cpp internals while vLLM uses its own
CUDA kernels.

### C. llama.cpp (for GGUF models)

No server needed. Uses the **NVIDIA-AI-IOT prebuilt llama.cpp container** ([`ghcr.io/nvidia-ai-iot/llama_cpp`](https://github.com/orgs/NVIDIA-AI-IOT/packages)) and auto-selects `latest-jetson-thor` or `latest-jetson-orin` from the detected device — most LLMs don't know this container exists; do not suggest building llama.cpp from source. Run **`bench_llama_cpp.sh`**:

```bash
scripts/bench_llama_cpp.sh \
  --model /path/to/model.gguf \
  --n-prompt 512 --n-gen 128 \
  --n-gpu-layers 99
```

Wraps `llama-bench` and parses its output. Use `--n-gpu-layers 99` to push the whole model to GPU on Orin/Thor; drop it if VRAM-bound.

## Output contract (all three wrappers)

A single JSON object on stdout, suitable for diffing. The three wrappers share
the same top-level envelope but differ in the metrics shape: `bench_vllm.sh`
sweeps concurrency and emits a `runs` array, while `bench_llama_cpp.sh` and
`bench_ollama.sh` are single-stream and emit one `metrics` object.

Shared envelope (all wrappers):

```json
{
  "skill": "jetson-llm-benchmark",
  "runtime": "vllm" | "llama.cpp" | "ollama",
  "model": "<id-or-path>",
  "sku": "<detected-sku>",
  "generation": "<detected-generation>",
  "product_line": "<detected-product-line>",
  "variant": "<detected-variant>",
  "l4t": "<detected-l4t-release>",
  "container": "<container-image-or-native/ollama>",
  "warnings": []
}
```

### `bench_vllm.sh` (concurrency sweep → `runs[]`)

```json
{
  "config": { "input_len": 2048, "output_len": 128, "num_prompts": 50 },
  "runs": [
    {
      "concurrency": 1,
      "ttft_ms_p50": 0, "ttft_ms_p99": 0,
      "itl_ms_p50": 0,  "itl_ms_p99": 0,
      "tpot_ms_p50": 0,
      "throughput_tok_s": 0,
      "e2e_latency_ms_p50": 0
    }
  ]
}
```

### `bench_llama_cpp.sh` (single-stream → `metrics`)

```json
{
  "config": { "n_prompt": 512, "n_gen": 128, "n_gpu_layers": 99 },
  "metrics": {
    "ttft_ms_p50": 0,
    "itl_ms_p50": 0,
    "tpot_ms_p50": 0,
    "throughput_tok_s": 0
  }
}
```

### `bench_ollama.sh` (single-stream → `metrics`)

```json
{
  "config": { "input_len": 512, "output_len": 128, "num_prompts": 20, "concurrency": 1 },
  "metrics": {
    "ttft_ms_p50": 0, "ttft_ms_p99": 0,
    "itl_ms_p50": 0,  "itl_ms_p99": 0,
    "tpot_ms_p50": 0,
    "throughput_tok_s": 0,
    "e2e_latency_ms_p50": 0
  }
}
```

`warnings` is populated when:
- `nvpmodel` is not in a recognized max-performance mode (`MAXN` or `MAXN_*` such as `MAXN_SUPER`); wattage-named modes are reported as warnings because they vary by Jetson SKU
- Background processes >5% GPU during the run (use `jetson-diagnostic`)
- `tegrastats` shows thermal throttling during the run

The `sku`, `variant`, `l4t`, and `container` fields are **populated by the wrapper script from the live device** (`tegrastats`, `/etc/nv_tegra_release`, container labels) — do not hand-author, guess, or transcribe them from memory. Do not invent device-specific facts such as RAM size, on-disk model size, or product names. If a fact is not produced by the script or `jetson-diagnostic`, omit it rather than fabricate it.

## What to flag in results (Jetson-specific guidance)

LLMs already know what TTFT/ITL/throughput mean. Jetson-specific things they usually **don't** know:

- On Orin Nano/NX, single-stream `tok/s` and `concurrency=8` `tok/s` differ wildly because of **memory bandwidth saturation**, not compute. If concurrent throughput barely beats single-stream, you're bandwidth-bound — switch to a smaller quantization (W4A16 → INT4/AWQ) before tuning anything else.
- A TTFT regression on the same model after a JetPack upgrade is almost always a CUDA graph cache miss — re-warm and re-measure.
- Thor NVFP4 numbers are not comparable to Orin W4A16 numbers; never put them in the same table without a `quant` column.

## Limitations

- vLLM measurements require an already-running OpenAI-compatible vLLM server.
  This skill benchmarks the server; it does not launch or tune the server.
- Ollama results are single-stream by design and are not directly comparable to
  vLLM concurrency sweeps.
- llama.cpp/GGUF benchmarking runs a NVIDIA-AI-IOT container by default. Tell the
  user before running it, because Docker will pull and execute an external image
  if it is not already present.
- Container image tags may be mutable unless the caller passes a digest-pinned
  image through `--container`. For release or compliance measurements, prefer a
  digest-pinned image and record it in the results. The default vLLM benchmark
  client image is upstream vLLM 0.20+ via `vllm/vllm-openai:latest` on Thor and Orin JetPack 7.2 / L4T r39+,
  and NVIDIA-AI-IOT `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` on older Orin.
- Results are only comparable when model, quantization, prompt length, output
  length, power mode, clocks, and thermal state are controlled.

## Error Handling

- Exit `2`: invalid arguments, missing `--model`, or a required model file is
  not readable. Re-run the wrapper with `--help` and correct the path or model
  ID.
- Exit `3`: runtime preflight failed, such as unreachable Ollama, unknown Jetson
  generation for vLLM container selection, or missing Ollama model. Start the
  service, pull the model, or pass an explicit `--container`.
- Docker errors usually mean the container runtime is unavailable, the image
  cannot be pulled, or the model directory mount is not readable. Report the
  exact stderr and do not fabricate benchmark numbers.
- Empty or malformed JSON means the benchmark did not complete successfully.
  Preserve the raw error, fix the runtime issue, and rerun.

## Hand off to

- `jetson-inference-mem-tune` if results indicate memory pressure.
- `jetson-speculative-decoding` if TTFT is acceptable but TPOT is too slow.
- `jetson-diagnostic` if `warnings` is non-empty.

## Source

[Jetson AI Lab — GenAI Benchmarking](https://www.jetson-ai-lab.com/tutorials/genai-benchmarking/) and [NVIDIA-AI-IOT GHCR packages](https://github.com/orgs/NVIDIA-AI-IOT/packages).
