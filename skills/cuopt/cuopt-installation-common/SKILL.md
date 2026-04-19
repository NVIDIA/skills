---
name: cuopt-installation-common
version: "26.06.00"
description: Install cuOpt — system and environment requirements only. Domain concepts; no install commands or interface guidance.
---

# cuOpt Installation (common)

Domain concepts for installing and running cuOpt. No install commands or interface details here.

## System requirements

- **GPU**: NVIDIA with Compute Capability ≥ 7.0 (Volta or newer). Examples: V100, A100, H100, RTX 20xx/30xx/40xx. Not supported: GTX 10xx (Pascal).
- **CUDA**: 12.x or 13.x. Package and runtime must match (e.g. cuopt built for CUDA 12 with a CUDA 12 driver).
- **Driver**: Compatible NVIDIA driver for the CUDA version in use.

## Required questions (environment)

Ask these if not already clear:

1. **Environment** — Local machine with GPU, cloud instance, Docker/Kubernetes, or no GPU (need remote/server)?
2. **CUDA version** — What is installed or planned? (e.g. `nvcc --version`, `nvidia-smi`.)
3. **Usage** — In-process (library/API) vs server (REST)? Which language or runtime (Python, C, server)?
4. **Package manager** — pip, conda, or Docker preferred?

## Notes

- Python API and C API are separate installables; having one does not provide the other.
- Server deployment typically uses Docker or a dedicated server package; client can be any language.
