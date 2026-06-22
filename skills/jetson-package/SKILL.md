---
name: jetson-package
description: Pick Jetson-compatible containers, vLLM runtime images, and Jetson AI Lab PyPI indexes; maps Orin SM 8.7 vs Thor SM 11.0 and JetPack-specific package choices.
version: 0.0.1
license: "Apache-2.0"
metadata:
  author: "Jetson Team"
  tags: [jetson, package, containers]
  languages: [bash]
  data-classification: public
---

# Jetson Package & Environment

Agents often suggest `docker pull` images or `pip install` wheels that claim **aarch64** support but were never built for Jetson’s GPU **streaming multiprocessor (SM)** targets. On Jetson, **default to NVIDIA-curated artifacts** unless the user explicitly opts out.

## Purpose

Choose Jetson-compatible containers and Python package indexes before installing GPU-native ML stacks. This skill prevents agents from recommending generic ARM wheels or stale container tags that do not include the right CUDA, JetPack, or SM target for the device.

## When to use

- "Which Docker image / container should I use on this Jetson?"
- "Where do I get PyTorch / vLLM / CUDA wheels for Jetson?"
- "`pip install` failed" or "wrong CUDA / SM" after installing a generic ARM wheel.
- Before `docker run` or `pip install` for ML stacks on Orin or Thor.
- User or agent looks for `l4t-cuda` containers on NGC — redirect to `nvcr.io/nvidia/cuda` (multi-arch).
- "Which PyTorch container should I use on Jetson?" — answer depends on Thor vs Orin and JetPack version.

## Canonical sources (use these first)

1. **Prebuilt containers (GHCR)** — [NVIDIA-AI-IOT packages](https://github.com/orgs/NVIDIA-AI-IOT/packages): `llama_cpp`, `ollama`, `live-vlm-webui`, older-Orin `vllm`, and related images built for Jetson JetPack stacks. Prefer these over random `arm64` images on Docker Hub. For vLLM, use upstream `vllm/vllm-openai` on Thor and Orin JetPack 7.2 / L4T r39+.
2. **NGC CUDA / PyTorch containers** — Tag selection depends on Jetson generation. Do not treat example PyTorch tag shapes as pinned recommendations; look up the current tag in the [NGC PyTorch catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) before giving a command.

   | Jetson | CUDA base | PyTorch |
   |--------|-----------|---------|
   | **Thor** | `nvcr.io/nvidia/cuda:<ver>-devel-ubuntu<ver>` (multi-arch, arm64 included) | `nvcr.io/nvidia/pytorch:<current-tag>-py3` (main multi-arch tag; verify current NGC tag) |
   | **Orin + r36 / JetPack 6** | same multi-arch CUDA base | `nvcr.io/nvidia/pytorch:<current-tag>-py3-igpu` — verify the current NGC tag and use the `-igpu` suffix for Orin iGPU (SM 8.7) when NGC publishes it |
   | **Orin + r39+ (future)** | same | likely main multi-arch tag once Orin becomes SBSA; verify when r39 ships |

**`l4t-cuda` is the legacy Orin-era CUDA container line.** If a user cannot find `l4t-cuda` on NGC, redirect them to the current multi-arch `nvcr.io/nvidia/cuda` image instead of third-party images.
3. **Python package indexes (devpi)** — [Jetson AI Lab PyPI](https://pypi.jetson-ai-lab.io/): browse the tree (for example `jp6/cu126`, `jp6/cu128`) and pick the index that matches your **JetPack / CUDA userland**. Prefer these over PyPI-only wheels for GPU-native stacks.

## GPU architecture reminder (why generic ARM fails)

| Jetson family | CUDA compute capability | Build target | Note |
|---------------|-------------------------|--------------|------|
| Orin (AGX / NX / Nano) | **8.7** | `sm_87` | Many desktop `aarch64` wheels omit Jetson Orin kernels. |
| Thor (T5000 / T4000) | **11.0** | `sm_110` | Requires CUDA / wheels / containers that include Blackwell Jetson support. |

A wheel or container may install on **ARM64 Linux** and still be **unusable or slow** if CUDA kernels were not compiled for your Jetson’s SM.

Use CUDA build target names when discussing wheel compatibility: `sm_87` for Jetson Orin and `sm_110` for Jetson Thor. Do not infer the generation from a prompt or a hostname — run `scripts/artifact_hints.sh` and use its detected `generation`, `variant`, `l4t`, and `cuda_sm_hint` fields before recommending wheels or container tags.

## GPU Python wheels on Jetson

Default PyPI wheels for GPU-native packages are usually not the right answer on Jetson, even when they claim `aarch64` support. For `onnxruntime-gpu`, PyTorch, vLLM, and similar packages, use the Jetson AI Lab package index as the canonical source and choose the subtree that matches the device's JetPack / CUDA userland.

For `onnxruntime-gpu`, lead with Jetson AI Lab rather than plain PyPI:

```bash
pip install --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/ onnxruntime-gpu
```

Adjust the `jp6/cu126` portion to match the detected JetPack / CUDA line. Do not present `pip install onnxruntime-gpu` from default PyPI as an equivalent Jetson GPU option.

## Do not fabricate device facts

Do not invent SKU names, RAM sizes, JetPack versions, CUDA versions, or GPU SM targets. Quote only what `scripts/artifact_hints.sh` or the user's supplied environment reports. If a field is unavailable, omit it or say it is unknown.

## Prerequisites

- Run package-detection scripts on a Jetson target, not on the host workstation.
- Network access is needed to inspect GHCR, NGC, or Jetson AI Lab package indexes.
- Source device facts from `scripts/artifact_hints.sh`, `jetson-diagnostic`, or user-provided environment output before recommending tags or wheels.

## Available Scripts

| Script | Purpose | Arguments |
|--------|---------|-----------|
| `scripts/artifact_hints.sh` | Emits detected Jetson SKU/generation, CUDA SM hint, canonical package URLs, and a preferred vLLM image hint. | `--human` for a readable summary; no argument for JSON. |

If your agent runtime supports `run_script`, use it to run `scripts/artifact_hints.sh` and read the JSON output. Otherwise run the script with `bash` from the repository root.

## Instructions

1. Run `scripts/artifact_hints.sh` (JSON on stdout). It sources `skills/jetson-diagnostic/scripts/detect_jetson.sh` and returns `sku`, `generation`, `product_line`, `variant`, `l4t`, a preferred **vLLM** image, `cuda_sm_hint`, and canonical URLs.
2. For **pip**, open the devpi root in a browser, pick the **jp6** subtree that matches your CUDA line, and set `--extra-index-url` / `PIP_EXTRA_INDEX_URL` — see `references/pypi-jetson-ai-lab.md`.
3. For **containers**, see `references/ghcr-images.md` and `jetson-llm-serve` for vLLM.

## Limitations

- This skill points to package catalogs and emits compatibility hints; it does not verify that a specific model checkpoint fits in memory.
- NGC and GHCR tags change. Treat placeholder tag shapes such as `<current-tag>-py3` as lookup instructions, not literal tags.
- If `generation` or `cuda_sm_hint` is unknown, do not guess a container tag.

## Hand off to

- `jetson-llm-serve` — run upstream/native vLLM 0.20+ on Thor and Orin JetPack 7.2 / L4T r39+, or `vllm:latest-jetson-orin` on older Orin.
- `jetson-llm-benchmark` — measure after the stack is installed.
- `jetson-diagnostic` — if installs succeed but runtime fails, snapshot first.

## Safety

Read-only: points to catalogs and emits hints; does not install or pull.

## Sources

[NVIDIA-AI-IOT GitHub Packages](https://github.com/orgs/NVIDIA-AI-IOT/packages), [pypi.jetson-ai-lab.io](https://pypi.jetson-ai-lab.io/).
