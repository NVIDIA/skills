<!-- SPDX-License-Identifier: Apache-2.0 AND CC-BY-4.0 -->
<!-- Copyright (c) 2026 NVIDIA Corporation. All rights reserved. -->

# NVIDIA Agent Skills

**Official, NVIDIA-verified skills for AI agents.**

[![NVIDIA](https://img.shields.io/badge/NVIDIA-Verified-76B900?style=flat&logo=nvidia&logoColor=white)](https://nvidia.com)
[![Agent Skills Spec](https://img.shields.io/badge/Agent%20Skills-Specification-blue)](https://agentskills.io)
[![License](https://img.shields.io/badge/License-Apache%202.0%20%2B%20CC--BY--4.0-green.svg)](LICENSE)

> 📖 **Docs:** [docs.nvidia.com/skills](https://docs.nvidia.com/skills) &nbsp;·&nbsp;
> 📺 **Livestream:** [From Vulnerable to Verified](https://www.youtube.com/watch?v=sVpKonYJ4D4&list=PL5B692fm6--vEL0FwctKghCpyEnBGAQJA&index=1) &nbsp;·&nbsp;
> 📝 **Blog:** [NVIDIA Verified Agent Skills: Capability Governance for AI Agents](https://developer.nvidia.com/blog/nvidia-verified-agent-skills-provide-capability-governance-for-ai-agents/)

---

Skills are portable instruction sets that teach AI agents how to use NVIDIA CUDA-X libraries, AI Blueprints, and platform tools correctly. This repository is a catalog: skills are maintained in their respective product repos and mirrored here daily via an automated sync pipeline. We are making NVIDIA skills available publicly and building this catalog in the open; see the [Roadmap](#roadmap) for what is planned next.

---

## Quickstart

Install NVIDIA skills with the default [`skills` CLI](https://github.com/vercel-labs/skills) flow:

```bash
npx skills add nvidia/skills
```

The CLI runs through `npx` and prompts you to choose a skill and install destination. You do not need to clone this repo or copy skill folders by hand.

The skill is available the next time your agent loads skills and encounters a relevant task. For example, ask your agent to "solve a linear programming problem with cuOpt" and the skill guides it through the cuOpt Python API.

### Install One Skill Without Prompts

Use this when you already know the skill name and want to skip prompts.

```bash
npx skills add nvidia/skills --skill cuopt-numerical-optimization-api-python --yes
```

Replace `cuopt-numerical-optimization-api-python` with any skill name from the [Skill Catalog](#skill-catalog).

### Install for a Specific Agent

Use `--agent` to target a specific AI coding agent. These are common client targets; for the full list of supported clients, see the [`skills` CLI Supported Agents table](https://github.com/vercel-labs/skills#supported-agents).

**Claude Code**

```bash
npx skills add nvidia/skills --skill cuopt-numerical-optimization-api-python --agent claude-code
```

**Codex**

```bash
npx skills add nvidia/skills --skill cuopt-numerical-optimization-api-python --agent codex
```

**Cursor**

```bash
npx skills add nvidia/skills --skill cuopt-numerical-optimization-api-python --agent cursor
```

**Kiro**

```bash
npx skills add nvidia/skills --skill cuopt-numerical-optimization-api-python --agent kiro-cli
```

Use `--agent` more than once to install the same skill into multiple agents.

```bash
npx skills add nvidia/skills \
  --skill cuopt-numerical-optimization-api-python \
  --agent claude-code \
  --agent codex \
  --agent cursor \
  --agent kiro-cli
```

### Browse the Catalog

Use this when you want to see available NVIDIA skills before installing anything.

```bash
npx skills add nvidia/skills --list
```

For non-interactive installs, global installs, agent-specific installs, updates, removals, and fallback manual copying, see [Advanced installation](docs/advanced-install.mdx).

---

## Skill Catalog

<!-- skills-table-start -->
| Product | Description | Skills | Source | Version |
|---------|-------------|:------:|--------|---------|
| **AIQ** | NVIDIA AI-Q Blueprint - deploy local AI-Q services and run shallow or deep research workflows as agent skills. | 2 | [Source](https://github.com/NVIDIA-AI-Blueprints/aiq/tree/develop/skills/aiq-research) | [`9f573a2`](https://github.com/NVIDIA-AI-Blueprints/aiq/commit/9f573a258f6423c2a663e6acbb9e0d86b072409f) · 2026-05-29 |
| **CUDA-Q** | CUDA Quantum — onboarding guide for installation, test programs, GPU simulation, QPU hardware, and quantum applications. | 1 | [Source](https://github.com/NVIDIA/cuda-quantum/tree/main/skills/cudaq-guide) | [`0b223e1`](https://github.com/NVIDIA/cuda-quantum/commit/0b223e125795384d76f5d727ac3b26576495e1b5) · 2026-06-01 |
| **cuDF** | Official NVIDIA-authored guidance for NVIDIA cuDF GPU DataFrames, pandas acceleration, dask-cuDF, ETL, joins, groupby, CSV/Parquet I/O, nullable semantics, and multi-GPU DataFrame workloads. | 1 | [Source](https://github.com/rapidsai/cudf/tree/main/skills/accelerated-computing-cudf) | [`09cf237`](https://github.com/rapidsai/cudf/commit/09cf237aa414ad98c7794710a156b3610db09c4c) · 2026-06-01 |
| **cuOpt** | GPU-accelerated optimization — vehicle routing, linear programming, quadratic programming, installation, server deployment, and developer tools. | 12 | [Source](https://github.com/NVIDIA/cuopt/tree/main/skills/cuopt-developer) | [`ffbe51a`](https://github.com/NVIDIA/cuopt/commit/ffbe51a3921fa36362ca0ebec9eece597a87b9d9) · 2026-05-30 |
| **cuPyNumeric** | NumPy and SciPy on multi-node multi-GPU systems — skills to help with installing cuPyNumeric, migrating existing NumPy code, and doing parallel I/O | 4 | [Source](https://github.com/nv-legate/cupynumeric/tree/main/skills/cupynumeric-hdf5) | [`e95f556`](https://github.com/nv-legate/cupynumeric/commit/e95f55666e4b3fecc7ded25ef3eb30aa7e9f5a6f) · 2026-05-30 |
| **DALI** | GPU-accelerated data loading and processing with NVIDIA DALI. | 1 | [Source](https://github.com/NVIDIA/DALI/tree/main/skills/dali-dynamic-mode) | [`7a61864`](https://github.com/NVIDIA/DALI/commit/7a618647943cc563c17134385463293c8212eb26) · 2026-06-01 |
| **DeepStream** | Agentic skills for guided DeepStream development. | 2 | [Source](https://github.com/NVIDIA-AI-IOT/DeepStream_Coding_Agent/tree/main/skills/deepstream-dev) | [`3daf16a`](https://github.com/NVIDIA-AI-IOT/DeepStream_Coding_Agent/commit/3daf16ae5168dd685608bb929fb4f5513f308045) · 2026-05-28 |
| **Dynamo** | NVIDIA Dynamo deployment bring-up on Kubernetes — pick and deploy recipes, start router modes, validate disagg NIXL/UCX/NCCL interconnect, and triage day-2 failures. | 4 | [Source](https://github.com/ai-dynamo/dynamo/tree/main/skills/dynamo-interconnect-check) | [`554d676`](https://github.com/ai-dynamo/dynamo/commit/554d67613c39922fb100b8107128d53f6bb8ddd9) · 2026-06-01 |
| **Earth2Studio** | Open-source deep-learning framework for exploring, building and deploying AI weather/climate workflows. | 4 | [Source](https://github.com/NVIDIA/earth2studio/tree/main/skills/earth2studio-data-fetch) | [`6b6c666`](https://github.com/NVIDIA/earth2studio/commit/6b6c666405e1d453d34dce32ea33d7eed3734a99) · 2026-05-29 |
| **Holoscan SDK** | Install and set up the Holoscan SDK on any platform (container, Debian, Python, Conda, or source). | 6 | [Source](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/skills/holoscan-install-debian) | [`e3a702d`](https://github.com/nvidia-holoscan/holoscan-sdk/commit/e3a702d29ce0f51fe53108cc3f010f96b6f7384e) · 2026-05-31 |
| **Medical AI Skills** | Agent-ready medical AI skills built on MONAI for DICOM handling, NVIDIA-hosted medical imaging model workflows, segmentation, synthesis, and evidence-oriented evaluation. | 12 | [Source](https://github.com/NVIDIA-Medtech/medical-AI-skills/tree/dev/skills/dicom-metadata-extract) | [`6d88654`](https://github.com/NVIDIA-Medtech/medical-AI-skills/commit/6d886544853ad6a92e26a600be56d3e74d4d144b) · 2026-05-31 |
| **Megatron-Core** | Large-scale distributed training — model parallelism, pipeline parallelism, and mixed precision. | 5 | [Source](https://github.com/NVIDIA/Megatron-LM/tree/main/skills/mcore-create-issue) | [`24cc2a8`](https://github.com/NVIDIA/Megatron-LM/commit/24cc2a85667825aae1413c4021a7075df8475e42) · 2026-06-01 |
| **NeMo AutoModel** | NeMo AutoModel - PyTorch-native distributed training for LLMs/VLMs with Hugging Face support, recipes, launchers, and validation workflows. | 4 | [Source](https://github.com/NVIDIA-NeMo/Automodel/tree/main/skills/nemo-automodel-distributed-training) | [`ec5de8b`](https://github.com/NVIDIA-NeMo/Automodel/commit/ec5de8bf538fff2cdfd12e1e955b997ff530a610) · 2026-06-01 |
| **NeMo MBridge** | NeMo MBridge - PyTorch-native bridge between Hugging Face and Megatron-Core for checkpoint conversion, training recipes, and NVIDIA GPU performance workflows. | 20 | [Source](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/skills/nemo-mbridge-mlm-bridge-training) | [`504d9b0`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/504d9b0ef64b31da2f03f20a04ec5c6a95e0d03a) · 2026-06-01 |
| **NeMo Platform** | NeMo Platform brings NVIDIA NeMo libraries together under one CLI, Python SDK, and web UI | 2 | [Source](https://github.com/NVIDIA-NeMo/nemo-platform/tree/main/skills/nemo-evaluator-plugin) | [`fc9b0c4`](https://github.com/NVIDIA-NeMo/nemo-platform/commit/fc9b0c44f9a1bf77f4be4fc652b5b4614e20b786) · 2026-06-01 |
| **NeMo Retriever** | NeMo Retriever - deploy NeMo Retriever Library locally, extract information from corpus of data, and answer questions against the corpus. | 1 | [Source](https://github.com/NVIDIA/NeMo-Retriever/tree/main/skills/nemo-retriever) | [`1e594ac`](https://github.com/NVIDIA/NeMo-Retriever/commit/1e594ac5dbccec1d741dceec3dfa1e2cf18819bd) · 2026-05-31 |
| **NeMo-RL** | RLHF training on Ray — GRPO, DPO, and SFT for LLMs and VLMs with FSDP2 and Megatron-Core. | 5 | [Source](https://github.com/NVIDIA-NeMo/RL/tree/main/skills) | [`48b2cd2`](https://github.com/NVIDIA-NeMo/RL/commit/48b2cd2ac79c3cef4dbf9b4458a04d38eebc1d16) · 2026-05-30 |
| **NemoClaw** | Secure agent sandboxing — run OpenClaw inside NVIDIA OpenShell with managed inference, policy management, remote deployment, sandbox monitoring. | 10 | [Source](https://github.com/NVIDIA/NemoClaw/tree/main/skills/nemoclaw-user-agent-skills) | [`df7d054`](https://github.com/NVIDIA/NemoClaw/commit/df7d054be5c00e90a286e7beaa4e87ddcf992304) · 2026-05-31 |
| **Nemotron** | Author end-to-end model development, customization, evaluation, and deployment pipelines using the NVIDIA AI stack. | 2 | [Source](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/skills/nemotron-customize) | [`306b2f1`](https://github.com/NVIDIA-NeMo/Nemotron/commit/306b2f1217e000b5972155c1f2b1ba6660c994bd) · 2026-05-29 |
| **Nemotron Speech** | Deploy and operate NVIDIA Nemotron Speech (Riva) NIMs — ASR, TTS, and NMT, cloud-hosted via build.nvidia.com or self-hosted on your own GPU. | 1 | [Source](https://github.com/nvidia-riva/Nemotron-speech-skills/tree/main/skills/nemotron-speech) | [`006d14c`](https://github.com/nvidia-riva/Nemotron-speech-skills/commit/006d14c944b48c6776269a0a5863a38c9ac5de6b) · 2026-05-29 |
| **NVIDIA Digital Health Examples** | Agent skills for the clinical ASR evaluation flywheel — term curation, synthetic clinical-speech benchmark generation, KER (Keyword Error Rate) scoring, and fine-tune guidance. | 4 | [Source](https://github.com/NVIDIA/digital-health-examples/tree/main/skills/digital-health-clinical-asr-setup) | [`404c5ca`](https://github.com/NVIDIA/digital-health-examples/commit/404c5ca905d73d2df46fcfe2490aace21659ecff) · 2026-05-28 |
| **Physical AI** | Generate labeled synthetic training data for physical-AI inspection and perception models. | 7 | [Source](https://github.com/NVIDIA/physical-ai-data-factory/tree/main/skills/physical-ai-defect-image-generation) | [`f858927`](https://github.com/NVIDIA/physical-ai-data-factory/commit/f858927dd4feaa2f060b7e027e1d29eaab5391fd) · 2026-05-31 |
| **PhysicsNeMo** | NVIDIA PhysicsNeMo - Open-source deep-learning framework for building, training, and fine-tuning deep learning models using state-of-the-art Physics-ML methods. | 1 | [Source](https://github.com/NVIDIA/physicsnemo/tree/main/skills/physicsnemo-discover) | [`8e46db6`](https://github.com/NVIDIA/physicsnemo/commit/8e46db617f552e47bfac3f8e84e5665dd9a0780e) · 2026-05-29 |
| **RAG Blueprint** | RAG pipeline — deploy, configure, troubleshoot, and manage retrieval augmented generation with Docker Compose or Helm. | 3 | [Source](https://github.com/NVIDIA-AI-Blueprints/rag/tree/develop/skills/rag-blueprint) | [`bed5165`](https://github.com/NVIDIA-AI-Blueprints/rag/commit/bed5165f4a9d7ba11b46a765fc36bd46e911babd) · 2026-05-30 |
| **Skill Card Generator** | Reads an agent skill's source files and produces a skill card plus a review table. Use when a skill directory exists and a governance card needs to be generated or updated. | 1 | [Source](https://github.com/NVIDIA/Trustworthy-AI/tree/main/skills/skill-card-generator) | [`09b0d6b`](https://github.com/NVIDIA/Trustworthy-AI/commit/09b0d6b03ee587cc8672ad3133a4eb84aeadbd2a) · 2026-05-28 |
| **TileGym** | Tile-based GPU programming — adding new kernels, cross-framework conversion, and performance optimization. | 1 | [Source](https://github.com/NVIDIA/TileGym/tree/main/skills/tilegym-adding-cutile-kernel) | [`2bf003b`](https://github.com/NVIDIA/TileGym/commit/2bf003b049c04f45b222423d328cb51ba5cf2670) · 2026-05-29 |
| **Video Search and Summarization** | VSS Blueprint — deploy profiles, search and summarize video, generate analysis reports, manage alerts and incidents, query VIOS sensors, and use the RTVI VLM microservice. | 15 | [Source](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/tree/develop/skills) | [`42fee9f`](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/commit/42fee9fb3d6fd44809bbdef5cd75ebb4309e956d) · 2026-06-01 |
<!-- skills-table-end -->

---

## Getting Help & Contributing

For skill-related issues, feature requests, new skill ideas, discussions, and contributions — use the source repo for the relevant product:

<!-- help-table-start -->
| Product | Issues | Discussions | Contributing | Security |
|---------|--------|-------------|--------------|----------|
| **AIQ** | [Issues](https://github.com/NVIDIA-AI-Blueprints/aiq/issues) | [Discussions](https://github.com/NVIDIA-AI-Blueprints/aiq/discussions) | [Contributing](https://github.com/NVIDIA-AI-Blueprints/aiq/blob/develop/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-Blueprints/aiq/blob/develop/SECURITY.md) |
| **CUDA-Q** | [Issues](https://github.com/NVIDIA/cuda-quantum/issues) | [Discussions](https://github.com/NVIDIA/cuda-quantum/discussions) | [Contributing](https://github.com/NVIDIA/cuda-quantum/blob/main/Contributing.md) | [Security](https://github.com/NVIDIA/cuda-quantum/blob/main/SECURITY.md) |
| **cuDF** | [Issues](https://github.com/rapidsai/cudf/issues) | [Discussions](https://github.com/rapidsai/cudf/discussions) | [Contributing](https://github.com/rapidsai/cudf/blob/main/CONTRIBUTING.md) | [Security](https://github.com/rapidsai/cudf/blob/main/SECURITY.md) |
| **cuOpt** | [Issues](https://github.com/NVIDIA/cuopt/issues) | [Discussions](https://github.com/NVIDIA/cuopt/discussions) | [Contributing](https://github.com/NVIDIA/cuopt/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/cuopt/blob/main/SECURITY.md) |
| **cuPyNumeric** | [Issues](https://github.com/nv-legate/cupynumeric/issues) | [Discussions](https://github.com/nv-legate/cupynumeric/discussions) | [Contributing](https://github.com/nv-legate/cupynumeric/blob/main/CONTRIBUTING.md) | [Security](https://github.com/nv-legate/cupynumeric/blob/main/SECURITY.md) |
| **DALI** | [Issues](https://github.com/NVIDIA/DALI/issues) | [Discussions](https://github.com/NVIDIA/DALI/discussions) | [Contributing](https://github.com/NVIDIA/DALI/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/DALI/blob/main/SECURITY.md) |
| **DeepStream** | [Issues](https://github.com/NVIDIA-AI-IOT/DeepStream_Coding_Agent/issues) | [Discussions](https://github.com/NVIDIA-AI-IOT/DeepStream_Coding_Agent/discussions) | [Contributing](https://github.com/NVIDIA-AI-IOT/DeepStream_Coding_Agent/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-IOT/DeepStream_Coding_Agent/blob/main/SECURITY.md) |
| **Dynamo** | [Issues](https://github.com/ai-dynamo/dynamo/issues) | [Discussions](https://github.com/ai-dynamo/dynamo/discussions) | [Contributing](https://github.com/ai-dynamo/dynamo/blob/main/CONTRIBUTING.md) | [Security](https://github.com/ai-dynamo/dynamo/blob/main/SECURITY.md) |
| **Earth2Studio** | [Issues](https://github.com/NVIDIA/earth2studio/issues) | [Discussions](https://github.com/NVIDIA/earth2studio/discussions) | [Contributing](https://github.com/NVIDIA/earth2studio/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/earth2studio/blob/main/SECURITY.md) |
| **Holoscan SDK** | [Issues](https://github.com/nvidia-holoscan/holoscan-sdk/issues) | [Discussions](https://github.com/nvidia-holoscan/holoscan-sdk/discussions) | [Contributing](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/CONTRIBUTING.md) | [Security](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/SECURITY.md) |
| **Medical AI Skills** | [Issues](https://github.com/NVIDIA-Medtech/medical-AI-skills/issues) | [Discussions](https://github.com/NVIDIA-Medtech/medical-AI-skills/discussions) | [Contributing](https://github.com/NVIDIA-Medtech/medical-AI-skills/blob/dev/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-Medtech/medical-AI-skills/blob/dev/SECURITY.md) |
| **Megatron-Core** | [Issues](https://github.com/NVIDIA/Megatron-LM/issues) | [Discussions](https://github.com/NVIDIA/Megatron-LM/discussions) | [Contributing](https://github.com/NVIDIA/Megatron-LM/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/Megatron-LM/blob/main/SECURITY.md) |
| **NeMo AutoModel** | [Issues](https://github.com/NVIDIA-NeMo/Automodel/issues) | [Discussions](https://github.com/NVIDIA-NeMo/Automodel/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/Automodel/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/Automodel/blob/main/SECURITY.md) |
| **NeMo MBridge** | [Issues](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues) | [Discussions](https://github.com/NVIDIA-NeMo/Megatron-Bridge/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/SECURITY.md) |
| **NeMo Platform** | [Issues](https://github.com/NVIDIA-NeMo/nemo-platform/issues) | [Discussions](https://github.com/NVIDIA-NeMo/nemo-platform/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/nemo-platform/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/nemo-platform/blob/main/SECURITY.md) |
| **NeMo Retriever** | [Issues](https://github.com/NVIDIA/NeMo-Retriever/issues) | [Discussions](https://github.com/NVIDIA/NeMo-Retriever/discussions) | [Contributing](https://github.com/NVIDIA/NeMo-Retriever/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/NeMo-Retriever/blob/main/SECURITY.md) |
| **NeMo-RL** | [Issues](https://github.com/NVIDIA-NeMo/RL/issues) | [Discussions](https://github.com/NVIDIA-NeMo/RL/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/RL/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/RL/blob/main/SECURITY.md) |
| **NemoClaw** | [Issues](https://github.com/NVIDIA/NemoClaw/issues) | [Discussions](https://github.com/NVIDIA/NemoClaw/discussions) | [Contributing](https://github.com/NVIDIA/NemoClaw/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/NemoClaw/blob/main/SECURITY.md) |
| **Nemotron** | [Issues](https://github.com/NVIDIA-NeMo/Nemotron/issues) | [Discussions](https://github.com/NVIDIA-NeMo/Nemotron/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/SECURITY.md) |
| **Nemotron Speech** | [Issues](https://github.com/nvidia-riva/Nemotron-speech-skills/issues) | [Discussions](https://github.com/nvidia-riva/Nemotron-speech-skills/discussions) | [Contributing](https://github.com/nvidia-riva/Nemotron-speech-skills/blob/main/CONTRIBUTING.md) | [Security](https://github.com/nvidia-riva/Nemotron-speech-skills/blob/main/SECURITY.md) |
| **NVIDIA Digital Health Examples** | [Issues](https://github.com/NVIDIA/digital-health-examples/issues) | [Discussions](https://github.com/NVIDIA/digital-health-examples/discussions) | [Contributing](https://github.com/NVIDIA/digital-health-examples/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/digital-health-examples/blob/main/SECURITY.md) |
| **Physical AI** | [Issues](https://github.com/NVIDIA/physical-ai-data-factory/issues) | [Discussions](https://github.com/NVIDIA/physical-ai-data-factory/discussions) | [Contributing](https://github.com/NVIDIA/physical-ai-data-factory/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/physical-ai-data-factory/blob/main/SECURITY.md) |
| **PhysicsNeMo** | [Issues](https://github.com/NVIDIA/physicsnemo/issues) | [Discussions](https://github.com/NVIDIA/physicsnemo/discussions) | [Contributing](https://github.com/NVIDIA/physicsnemo/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/physicsnemo/blob/main/SECURITY.md) |
| **RAG Blueprint** | [Issues](https://github.com/NVIDIA-AI-Blueprints/rag/issues) | [Discussions](https://github.com/NVIDIA-AI-Blueprints/rag/discussions) | [Contributing](https://github.com/NVIDIA-AI-Blueprints/rag/blob/develop/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-Blueprints/rag/blob/develop/SECURITY.md) |
| **Skill Card Generator** | [Issues](https://github.com/NVIDIA/Trustworthy-AI/issues) | [Discussions](https://github.com/NVIDIA/Trustworthy-AI/discussions) | [Contributing](https://github.com/NVIDIA/Trustworthy-AI/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/Trustworthy-AI/blob/main/SECURITY.md) |
| **TileGym** | [Issues](https://github.com/NVIDIA/TileGym/issues) | [Discussions](https://github.com/NVIDIA/TileGym/discussions) | [Contributing](https://github.com/NVIDIA/TileGym/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/TileGym/blob/main/SECURITY.md) |
| **Video Search and Summarization** | [Issues](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/issues) | [Discussions](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/discussions) | [Contributing](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/blob/develop/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/blob/develop/SECURITY.md) |
<!-- help-table-end -->

For issues with **this catalog repo itself** (README, structure, listing a new product): [open an issue here](../../issues).

---

## Verifying Skills

Every published skill ships with a detached OMS signature (`skill.oms.sig`). The sync pipeline drops any skill missing the required artifacts before publishing, so every skill in the catalog carries:

- `SKILL.md` — the skill instructions consumed by the agent
- `skill-card.md` — skill identity and governance card
- `skill.oms.sig` — detached OMS signature (verifiable against `nv-agent-root-cert.pem`)
- A Tier-3 evaluation dataset — accepted at `evals/evals.json`, `evals/*.json`, `eval/*.json`, or `benchmark/evals.json`
- `BENCHMARK.md` — generated benchmark report capturing verifiable uplift data

Verify a skill against the NVIDIA trust anchor [`nv-agent-root-cert.pem`](nv-agent-root-cert.pem):

```bash
pip install model-signing
model_signing verify certificate SKILL_DIR \
  --signature SKILL_DIR/skill.oms.sig \
  --certificate_chain nv-agent-root-cert.pem \
  --ignore_unsigned_files
```

A successful verification confirms that the skill contents have not been modified since signing by NVIDIA.

See [Verify Signed Agent Skills](docs/signing-agent-skills.mdx) for signature layout, the trust pipeline, and policy options.

---

## Roadmap

- ✅ Public skills catalog with NVIDIA-verified skills across multiple products
- ✅ Automated sync pipeline with skills mirrored from product repos daily
- ✅ Security scanning for all published skills covering instruction safety and supply-chain integrity
- ✅ Skills signing so every published skill carries a verifiable NVIDIA signature
- ✅ Skills universal evaluation criteria and task-specific criteria
- ✅ Skill Card with machine-readable metadata for identity, provenance, quality, and behavioral boundaries
- 🔲 Compliance gates before external publication
- 🔲 Syndication to external marketplaces and MCP hubs

---

## Repository Structure

```
NVIDIA/skills/
├── skills/                      # 110 verified skills across 24 products,
│   │                              synced from upstream product repos
│   ├── README.md                 # Browser-facing install guidance
│   ├── <product-prefix>-*/       # Flat layout — one dir per skill, product-prefixed
│   │                               # e.g. aiq-*, cuopt-*, cupynumeric-*, dali-*,
│   │                               # deepstream-*, digital-health-*, dynamo-*,
│   │                               # earth2studio-*, mcore-*, nemo-automodel-*,
│   │                               # nemo-data-designer-plugin, nemo-evaluator-plugin,
│   │                               # nemo-mbridge-* (20 skills), nemo-retriever,
│   │                               # nemoclaw-user-* (10 skills), nemotron-*,
│   │                               # physicsnemo-*, rag-*, skill-card-generator,
│   │                               # tilegym-*, accelerated-computing-cudf, cudaq-guide
│   ├── omniverse-*/              # Physical AI — manually staged (see manual-components.yml)
│   ├── physical-ai-*/            # Physical AI — manually staged
│   ├── NeMo-RL/                  # Legacy nested layout (5 skills under one dir)
│   └── video-search-and-summarization/  # Legacy nested layout (15 skills under one dir)
├── components.d/                # Product registry — one file per component, teams onboard here
│   ├── README.md                 # Schema and onboarding instructions
│   └── <product>.yml             # one file per registered product (24 today)
├── plugins/                     # Packaged plugin distributions
│   └── nvidia-skills/            # Curated NVIDIA skills bundle (Claude Code, Codex)
├── plugins.d/                   # Plugin build registry — config for `build-plugins.py`
│   ├── README.md
│   ├── _defaults.yml
│   └── nvidia-skills.yml
├── .claude-plugin/              # Claude Code marketplace metadata
│   └── marketplace.json
├── .agents/plugins/             # Agent marketplace metadata (other clients)
│   └── marketplace.json
├── docs/                        # Long-form documentation (published via Fern)
│   ├── README.md                 # How to build the docs locally
│   ├── index.mdx
│   ├── advanced-install.mdx
│   ├── agent-skill-trust-pipeline.mdx
│   ├── release-checklist.mdx
│   ├── scanning-agent-skills.mdx
│   ├── signing-agent-skills.mdx
│   └── skill-cards.mdx
├── fern/                        # Fern docs site configuration
├── .github/
│   ├── workflows/                # Sync pipeline, plugin validation, DCO check, author verify
│   └── scripts/                  # regenerate-readme.sh, build-plugins.py,
│                                 # manual-components.yml (temp Physical AI catalog
│                                 # exception, removed after Computex 2026),
│                                 # marketplace/metadata.json (skill metadata sidecar)
├── nv-agent-root-cert.pem       # Trust anchor for OMS signature verification
├── skills.sh.json               # Skills.sh marketplace grouping config
├── CHANGELOG.md
├── CONTRIBUTING.md              # Contribution guidelines
├── SECURITY.md                  # Security reporting policy
├── CODE_OF_CONDUCT.md           # Community code of conduct
└── LICENSE                      # Apache 2.0 / CC BY 4.0
```

Skills are maintained in their respective product repos (see the **Source** column in the [Skill Catalog](#skill-catalog)) and synced to this repo daily. Products only appear under `skills/` after the sync pipeline confirms each skill carries:

- `skill.oms.sig` — detached OMS-format signature (verifiable against `nv-agent-root-cert.pem`)
- `skill-card.md` — skill identity and governance card
- A Tier-3 evaluation dataset — accepted at `evals/evals.json`, `evals/*.json`, `eval/*.json`, or `benchmark/evals.json`

When evaluation runs produce a `BENCHMARK.md`, it ships alongside the skill so consumers can see verifiable benchmark uplift data.

---

## Standards & Compatibility

This repository adheres to the [Agent Skills specification](https://agentskills.io/specification):

- Skills are portable directories with a `SKILL.md` file at their root.
- Metadata uses YAML frontmatter with required `name` and `description` fields.
- Skills follow a progressive disclosure model — lightweight metadata loads at startup, full instructions load on activation.
- Validate your skill using the [`skills-ref`](https://github.com/agentskills/agentskills/tree/main/skills-ref) reference library.

---

## License

This project is dual-licensed under the [Apache License 2.0](LICENSE) and [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE).
