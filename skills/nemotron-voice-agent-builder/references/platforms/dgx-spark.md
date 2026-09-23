# DGX Spark

Read when `preflight.md` classifies the target as `dgx_spark`. DGX Spark runs the vLLM plus
NeMo-Speech.cpp stack in `platforms/single-gpu.md`, which owns the pre-proposal checklist,
models, flags, setup, budget, and verification. This file covers only what is specific to
GB10.

DGX Spark does **not** use the workstation NIM path. Do not carry a NIM image, a NIM
profile, `NIM_TAGS_SELECTOR`, or a speech function id onto this platform.

## Identify the Platform

Either signal confirms GB10:

- the GPU name reported by `nvidia-smi` contains `NVIDIA GB10`
- `/sys/class/dmi/id/product_name` contains `DGX_Spark`, matched with the underscore
  exactly as the file reports it

`preflight.md` §2 owns how this ranks against the generic aarch64 rule.

## Prerequisites

- DGX Spark running its current supported software release, including the CUDA and
  container-runtime components
- Docker Engine and Docker Compose
- `HF_TOKEN` with access to the locked Nemotron repository, its draft repository, and the
  speech model repositories
- The Hugging Face CLI for the one-time speech model download
- Disk capacity for the LLM weights, the draft model, the speech GGUFs, and the compiled
  kernel cache

Resolve the supported release and component versions from the current platform
documentation rather than from this file.

## What Differs From Jetson Thor

| Aspect | DGX Spark |
| --- | --- |
| Speculative decoding | serve the published DGX Spark draft model alongside the NVFP4 target |
| Pipeline coverage | cascaded and Omni, including an Omni deployment with concurrent subagent pipelines |

Everything else in `platforms/single-gpu.md` applies unchanged, including the NVFP4
quantization, the speech container, the memory method, and the start order.

The draft model is specific to this platform. Do not reuse the workstation Blackwell draft
model here, and do not carry the DGX Spark draft onto Jetson Thor.

DGX Spark is aarch64, so confirm an `arm64` image exists for every container the project
generates, including any TURN service it owns (`networking/remote-webrtc.md` §Coturn).

## Anti-Patterns

- Assuming a NIM supports GB10 because another model does.
- Reusing a workstation image, profile, or draft model without confirming it on the card.
- Leaving the draft model out of the memory budget.
- Generating an `amd64`-only image for an aarch64 host.
