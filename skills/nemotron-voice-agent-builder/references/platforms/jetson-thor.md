# Jetson Thor

Read when `preflight.md` classifies the target as `jetson_thor`. Jetson Thor runs the vLLM
plus NeMo-Speech.cpp stack in `platforms/single-gpu.md`, which owns the pre-proposal
checklist, models, flags, setup, budget, and verification. This file covers only what is
specific to Thor.

Jetson Thor does **not** use the workstation NIM path. Do not carry a NIM image, a NIM
profile, `NIM_TAGS_SELECTOR`, or a speech function id onto this platform, and do not copy
an x86 image onto it.

Orin-class Jetsons are not supported. They are not a smaller Thor, because the models in
this skill do not fit them. Route Orin to cloud through `preflight.md` §2.

## Identify the Platform

Thor is a Tegra platform, so it has no discrete PCI GPU entry and no `/dev/nvidia0`. The
identity comes from the device tree:

- `/proc/device-tree/model` contains both `Jetson` and `Thor`
- `/sys/class/dmi/id/product_name` may carry the same pair when present

Requiring both words avoids matching an Orin device tree, so confirm it is Thor rather than
Orin before proposing anything local.

## Prerequisites

- Jetson Thor flashed with the JetPack release the platform documentation currently
  supports, including the CUDA, TensorRT, and container-runtime components
- Docker Engine and Docker Compose
- `HF_TOKEN` with access to the locked Nemotron repository and the speech model
  repositories
- The Hugging Face CLI for the one-time speech model download
- Disk capacity for the LLM weights, the speech GGUFs, and the compiled kernel cache

Resolve the supported JetPack release from the current platform documentation rather than
from this file, and use the component versions compatible with the release actually
installed.

## What Differs From DGX Spark

| Aspect | Jetson Thor |
| --- | --- |
| Speculative decoding | none. Serve the NVFP4 target model alone, with no draft model |
| Pipeline coverage | cascaded and single-agent Omni. An Omni deployment that runs concurrent subagent pipelines is not supported here |

Everything else in `platforms/single-gpu.md` applies unchanged, including the NVFP4
quantization, the speech container, the memory method, and the start order.

Do not add a draft model to Thor to match DGX Spark. When the use case needs the
concurrent-subagent Omni shape, propose it on NVIDIA cloud and say plainly that the local
platform does not support it.

Thor is aarch64, so confirm an `arm64` image exists for every container the project
generates. The speech container publishes both architectures from one tag. A TURN service
is the common gap, so check it before generating one
(`networking/remote-webrtc.md` §Coturn).

Thor is also the platform where `platforms/single-gpu.md` §Host memory at startup and
§First boot compiles kernels are most likely to appear. Read both before treating a slow
start as a failure.

## Anti-Patterns

- Following build.nvidia.com NIM launch instructions on Thor.
- Copying an x86 image onto Thor, or generating an `amd64`-only service.
- Treating an Orin device as a supported Thor.
- Adding a draft model, or reusing the DGX Spark draft model.
- Proposing the concurrent-subagent Omni shape as a local Thor deployment.
- Restarting vLLM during a first-boot kernel compile.
