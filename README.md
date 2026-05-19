<!-- SPDX-License-Identifier: Apache-2.0 AND CC-BY-4.0 -->
<!-- Copyright (c) 2026 NVIDIA Corporation. All rights reserved. -->

# NVIDIA Agent Skills

**Official, NVIDIA-verified skills for AI agents.**

[![NVIDIA](https://img.shields.io/badge/NVIDIA-Verified-76B900?style=flat&logo=nvidia&logoColor=white)](https://nvidia.com)
[![Agent Skills Spec](https://img.shields.io/badge/Agent%20Skills-Specification-blue)](https://agentskills.io)
[![License](https://img.shields.io/badge/License-Apache%202.0%20%2B%20CC--BY--4.0-green.svg)](LICENSE)

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

For non-interactive installs, global installs, agent-specific installs, updates, removals, and fallback manual copying, see [Advanced installation](docs/advanced-install.md).

---

## Skill Catalog

<!-- skills-table-start -->
| Product | Description | Skills | Catalog | Source | Version |
|---------|-------------|:------:|---------|--------|---------|
| **CUDA-Q** | CUDA Quantum — onboarding guide for installation, test programs, GPU simulation, QPU hardware, and quantum applications. | 1 | [`skills/CUDA-Q/`](skills/CUDA-Q) | [Source](https://github.com/NVIDIA/cuda-quantum/tree/main/.claude/skills) | [`30bbe2c`](https://github.com/NVIDIA/cuda-quantum/commit/30bbe2c371eb6ddd9c57157fadafae898fa25455) · 2026-05-19 |
| **cuOpt** | GPU-accelerated optimization — vehicle routing, linear programming, quadratic programming, installation, server deployment, and developer tools. | 12 | [`skills/cuopt/`](skills/cuopt) | [Source](https://github.com/NVIDIA/cuopt/tree/main/skills) | [`33727c1`](https://github.com/NVIDIA/cuopt/commit/33727c15b4bee9d6a244755c6bbc91afcb74c366) · 2026-05-19 |
| **DALI** | GPU-accelerated data loading and processing with NVIDIA DALI. | 1 | [`skills/DALI/`](skills/DALI) | [Source](https://github.com/NVIDIA/DALI/tree/main/.agents/skills) | [`579d6a3`](https://github.com/NVIDIA/DALI/commit/579d6a3ec660dc0746dd1e59500fd724f0ab4b6a) · 2026-05-19 |
| **DeepStream** | Agentic skills for guided DeepStream development. | 2 | [`skills/deepstream/`](skills/deepstream) | [Source](https://github.com/NVIDIA-AI-IOT/DeepStream_Coding_Agent/tree/main/skills) | [`703b4d6`](https://github.com/NVIDIA-AI-IOT/DeepStream_Coding_Agent/commit/703b4d6b43c0ac02b37704c8929a81b5d6bd69f7) · 2026-05-14 |
| **Megatron-Bridge** | Bridge between NeMo and Megatron — data processing, model conversion, and training utilities. | 29 | [`skills/Megatron-Bridge/`](skills/Megatron-Bridge) | [Source](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/skills) | [`f8c6113`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/f8c61132b7211cd93003347631e7986abfb13a62) · 2026-05-19 |
| **Megatron-Core** | Large-scale distributed training — model parallelism, pipeline parallelism, and mixed precision. | 12 | [`skills/Megatron-Core/`](skills/Megatron-Core) | [Source](https://github.com/NVIDIA/Megatron-LM/tree/main/skills) | [`14aaa7e`](https://github.com/NVIDIA/Megatron-LM/commit/14aaa7e0efbe27c9e9862d2a2f7f8d5fb342f3a8) · 2026-05-19 |
| **Model-Optimizer** | Model optimization — quantization, sparsity, and distillation for efficient inference. | 8 | [`skills/Model-Optimizer/`](skills/Model-Optimizer) | [Source](https://github.com/NVIDIA/Model-Optimizer/tree/main/.claude/skills) | [`a5bc6f8`](https://github.com/NVIDIA/Model-Optimizer/commit/a5bc6f812340d4267027f5fe22346b3fe779272d) · 2026-05-20 |
| **NeMo Evaluator** | LLM evaluation — launch evaluations, access MLflow results, NeMo Evaluator Launcher assistant, and bring-your-own benchmarks. | 4 | [`skills/NeMo-Evaluator-Launcher/`](skills/NeMo-Evaluator-Launcher) | [Source](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/.claude/skills) | [`6b20f86`](https://github.com/NVIDIA-NeMo/Evaluator/commit/6b20f86ef4220bb985d22b1878fd1705b2085001) · 2026-05-19 |
| **NeMo Gym** | RL training environments — add benchmarks, resources servers, agent wiring, and reward profiling. | 5 | [`skills/NeMo-Gym/`](skills/NeMo-Gym) | [Source](https://github.com/NVIDIA-NeMo/Gym/tree/main/.claude/skills) | [`a645786`](https://github.com/NVIDIA-NeMo/Gym/commit/a645786cf02b15d07e400a3f856bdf001e5bc6db) · 2026-05-19 |
| **NemoClaw** | Secure agent sandboxing — run OpenClaw inside NVIDIA OpenShell with managed inference, policy management, remote deployment, sandbox monitoring, and contributor/maintainer workflows. | 23 | [`skills/NemoClaw/`](skills/NemoClaw) | [Source](https://github.com/NVIDIA/NemoClaw/tree/main/.agents/skills) | [`9e00c58`](https://github.com/NVIDIA/NemoClaw/commit/9e00c58755864de901f5ebf5dd4a53a42adb9120) · 2026-05-19 |
| **Nemotron Voice Agent** | Real-time conversational AI — deploy speech-to-speech voice agents on Workstation, Jetson Thor, or Cloud NIMs. | 1 | [`skills/nemotron-voice-agent/`](skills/nemotron-voice-agent) | [Source](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/tree/main/.agents/skills) | [`a87826c`](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/commit/a87826cc1ab03103cea7f1f24dc94f456500e5c2) · 2026-04-22 |
| **RAG Blueprint** | RAG pipeline — deploy, configure, troubleshoot, and manage retrieval augmented generation with Docker Compose or Helm. | 1 | [`skills/rag/`](skills/rag) | [Source](https://github.com/NVIDIA-AI-Blueprints/rag/tree/main/skill-source/.agents/skills) | [`405e905`](https://github.com/NVIDIA-AI-Blueprints/rag/commit/405e90533400da5ad48e80db717e79e32a5dff23) · 2026-05-19 |
| **TensorRT-LLM** | LLM inference optimization — model onboarding, performance analysis and optimization, kernel writing, CI diagnostics, code contribution, and codebase exploration. | 23 | [`skills/TensorRT-LLM/`](skills/TensorRT-LLM) | [Source](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills) | [`b892451`](https://github.com/NVIDIA/TensorRT-LLM/commit/b892451afc7949400335d2ee4cb3047eabcdf51c) · 2026-05-19 |
| **TileGym** | Tile-based GPU programming — adding new kernels, cross-framework conversion, and performance optimization. | 7 | [`skills/TileGym/`](skills/TileGym) | [Source](https://github.com/NVIDIA/TileGym/tree/main/.agents/skills) | [`ae09bb6`](https://github.com/NVIDIA/TileGym/commit/ae09bb6e3e6abd1000c799795f1bcff79763f502) · 2026-05-19 |
| **Video Search and Summarization** | VSS Blueprint — deploy profiles, search and summarize video, generate analysis reports, manage alerts and incidents, query VIOS sensors, and use the RTVI VLM microservice. | 10 | [`skills/video-search-and-summarization/`](skills/video-search-and-summarization) | [Source](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/tree/main/skills) | [`02f0f9f`](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/commit/02f0f9f9cc49546635beed13ac1f44978042af8f) · 2026-05-18 |
<!-- skills-table-end -->

---

## Getting Help & Contributing

For skill-related issues, feature requests, new skill ideas, discussions, and contributions — use the source repo for the relevant product:

<!-- help-table-start -->
| Product | Issues | Discussions | Contributing | Security |
|---------|--------|-------------|--------------|----------|
| **CUDA-Q** | [Issues](https://github.com/NVIDIA/cuda-quantum/issues) | [Discussions](https://github.com/NVIDIA/cuda-quantum/discussions) | [Contributing](https://github.com/NVIDIA/cuda-quantum/blob/main/Contributing.md) | [Security](https://github.com/NVIDIA/cuda-quantum/blob/main/SECURITY.md) |
| **cuOpt** | [Issues](https://github.com/NVIDIA/cuopt/issues) | [Discussions](https://github.com/NVIDIA/cuopt/discussions) | [Contributing](https://github.com/NVIDIA/cuopt/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/cuopt/blob/main/SECURITY.md) |
| **DALI** | [Issues](https://github.com/NVIDIA/DALI/issues) | [Discussions](https://github.com/NVIDIA/DALI/discussions) | [Contributing](https://github.com/NVIDIA/DALI/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/DALI/blob/main/SECURITY.md) |
| **DeepStream** | [Issues](https://github.com/NVIDIA-AI-IOT/DeepStream_Coding_Agent/issues) | [Discussions](https://github.com/NVIDIA-AI-IOT/DeepStream_Coding_Agent/discussions) | [Contributing](https://github.com/NVIDIA-AI-IOT/DeepStream_Coding_Agent/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-IOT/DeepStream_Coding_Agent/blob/main/SECURITY.md) |
| **Megatron-Bridge** | [Issues](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues) | [Discussions](https://github.com/NVIDIA-NeMo/Megatron-Bridge/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/SECURITY.md) |
| **Megatron-Core** | [Issues](https://github.com/NVIDIA/Megatron-LM/issues) | [Discussions](https://github.com/NVIDIA/Megatron-LM/discussions) | [Contributing](https://github.com/NVIDIA/Megatron-LM/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/Megatron-LM/blob/main/SECURITY.md) |
| **Model-Optimizer** | [Issues](https://github.com/NVIDIA/Model-Optimizer/issues) | [Discussions](https://github.com/NVIDIA/Model-Optimizer/discussions) | [Contributing](https://github.com/NVIDIA/Model-Optimizer/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/Model-Optimizer/blob/main/SECURITY.md) |
| **NeMo Evaluator** | [Issues](https://github.com/NVIDIA-NeMo/Evaluator/issues) | [Discussions](https://github.com/NVIDIA-NeMo/Evaluator/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/SECURITY.md) |
| **NeMo Gym** | [Issues](https://github.com/NVIDIA-NeMo/Gym/issues) | [Discussions](https://github.com/NVIDIA-NeMo/Gym/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/Gym/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/Gym/blob/main/SECURITY.md) |
| **NemoClaw** | [Issues](https://github.com/NVIDIA/NemoClaw/issues) | [Discussions](https://github.com/NVIDIA/NemoClaw/discussions) | [Contributing](https://github.com/NVIDIA/NemoClaw/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/NemoClaw/blob/main/SECURITY.md) |
| **Nemotron Voice Agent** | [Issues](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/issues) | [Discussions](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/discussions) | [Contributing](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/blob/main/SECURITY.md) |
| **RAG Blueprint** | [Issues](https://github.com/NVIDIA-AI-Blueprints/rag/issues) | [Discussions](https://github.com/NVIDIA-AI-Blueprints/rag/discussions) | [Contributing](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/SECURITY.md) |
| **TensorRT-LLM** | [Issues](https://github.com/NVIDIA/TensorRT-LLM/issues) | [Discussions](https://github.com/NVIDIA/TensorRT-LLM/discussions) | [Contributing](https://github.com/NVIDIA/TensorRT-LLM/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/TensorRT-LLM/blob/main/SECURITY.md) |
| **TileGym** | [Issues](https://github.com/NVIDIA/TileGym/issues) | [Discussions](https://github.com/NVIDIA/TileGym/discussions) | [Contributing](https://github.com/NVIDIA/TileGym/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/TileGym/blob/main/SECURITY.md) |
| **Video Search and Summarization** | [Issues](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/issues) | [Discussions](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/discussions) | [Contributing](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/blob/main/SECURITY.md) |
<!-- help-table-end -->

For issues with **this catalog repo itself** (README, structure, listing a new product): [open an issue here](../../issues).

---

## Roadmap

- ✅ Public skills catalog with NVIDIA-verified skills across multiple products
- ✅ Automated sync pipeline with skills mirrored from product repos daily
- ✅ Security scanning for all published skills covering instruction safety and supply-chain integrity
- 🔲 Skills signing so every published skill carries a verifiable NVIDIA signature
- 🔲 Skills universal evaluation criteria and task-specific criteria
- 🔲 Skill Card with machine-readable metadata for identity, provenance, quality, and behavioral boundaries
- 🔲 Compliance gates before external publication
- 🔲 Syndication to external marketplaces and MCP hubs

---

## Repository Structure

```
NVIDIA/skills/
├── skills/                  # All skills, mirrored daily from product repos
│   ├── README.md             # Install guidance for people browsing this folder directly
│   ├── CUDA-Q/               # CUDA-Q skills
│   ├── cuopt/                # cuOpt skills
│   ├── Megatron-Bridge/      # Megatron-Bridge skills
│   ├── Megatron-Core/        # Megatron-Core skills
│   ├── Model-Optimizer/      # Model-Optimizer skills
│   ├── NeMo-Evaluator/       # NeMo Evaluator skills
│   ├── NeMo-Evaluator-Launcher/
│   ├── NeMo-Gym/             # NeMo Gym skills 
│   ├── NemoClaw/             # NemoClaw skills 
│   ├── nemotron-voice-agent/ # Nemotron Voice Agent skills
│   ├── TensorRT-LLM/         # TensorRT-LLM skills 
│   └── ...
├── components.d/            # Product registry — one file per component, teams onboard here
│   ├── cuda-q.yml
│   ├── cuopt.yml
│   ├── megatron-bridge.yml
│   ├── ...
│   ├── tensorrt-llm.yml
│   └── README.md             # Schema and onboarding instructions
├── docs/
│   └── advanced-install.md   # Advanced skills CLI usage
├── .github/workflows/       # Automated sync pipeline
├── CONTRIBUTING.md          # Contribution guidelines
├── SECURITY.md              # Security reporting policy
├── CODE_OF_CONDUCT.md       # Community code of conduct
└── LICENSE                  # Apache 2.0
```

Skills are maintained in their respective product repos (see the **Source** column in the [Skill Catalog](#skill-catalog)) and automatically synced to this repo daily.

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
