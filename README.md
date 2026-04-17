<!-- SPDX-License-Identifier: Apache-2.0 AND CC-BY-4.0 -->
<!-- Copyright (c) 2026 NVIDIA Corporation. All rights reserved. -->

# NVIDIA Agent Skills

**Official, NVIDIA-verified skills for AI agents.**

[![NVIDIA](https://img.shields.io/badge/NVIDIA-Verified-76B900?style=flat&logo=nvidia&logoColor=white)](https://nvidia.com)
[![Agent Skills Spec](https://img.shields.io/badge/Agent%20Skills-Specification-blue)](https://agentskills.io)
[![License](https://img.shields.io/badge/License-Apache%202.0%20%2B%20CC--BY--4.0-green.svg)](LICENSE)

---

Skills are portable instruction sets that teach AI agents how to use NVIDIA CUDA-X libraries, AI Blueprints, and platform tools correctly. From solving vehicle routing problems with cuOpt, to deploying RAG pipelines, to onboarding models into TensorRT-LLM, every skill listed here is **published and verified by NVIDIA**.

This repository is a **catalog** — skills are maintained in their respective product repos and mirrored here daily via an automated sync pipeline. It follows the open [Agent Skills specification](https://agentskills.io/specification), making skills compatible with any AI agent or framework that supports the standard.

We are building this in the open. The catalog, sync pipeline, and roadmap are public so the community can see what we ship, what we plan to ship, and how we secure skills for use. As we add signing, scanning, and evaluation, each improvement lands here incrementally.

---

## Quickstart

Browse the [Available Skills](#available-skills) table below, then install directly from this repo — every skill is mirrored here under `skills/`:

```bash
# Example: install the cuOpt LP/MILP skill
git clone --depth 1 --filter=blob:none --sparse https://github.com/NVIDIA/skills.git
cd skills && git sparse-checkout set skills/cuopt/cuopt-lp-milp-api-python
cp -r skills/cuopt/cuopt-lp-milp-api-python ~/.claude/skills/
```

That's it — the skill activates automatically the next time your agent encounters a relevant task. For example, ask your agent to "solve a linear programming problem with cuOpt" and the skill guides it through the cuOpt Python API.

### Install by Agent

| Agent / Framework | Installation |
|-------------------|-------------|
| Claude Code | Copy the skill directory into `~/.claude/skills/` |
| Codex | Copy the skill directory into your project's `.codex/skills/` folder |
| Cursor | Copy the skill directory into your project's `.cursor/skills/` folder |
| Other agents | Copy the skill directory into your agent's skills folder |

---

## Available Skills

| Product | Description | Skills | Catalog | Source |
|---------|-------------|:------:|---------|--------|
| **CUDA-Q** | CUDA Quantum — onboarding guide for installation, test programs, GPU simulation, QPU hardware, and quantum applications. | 1 | [`skills/CUDA-Q/`](skills/CUDA-Q) | [Source](https://github.com/NVIDIA/cuda-quantum/tree/main/.claude/skills) |
| **cuOpt** | GPU-accelerated optimization — vehicle routing, linear programming, quadratic programming, installation, server deployment, and developer tools. | 19 | [`skills/cuopt/`](skills/cuopt) | [Source](https://github.com/NVIDIA/cuopt/tree/main/skills) |
| **TensorRT-LLM** | LLM inference optimization — model onboarding, performance analysis and optimization, kernel writing, CI diagnostics, code contribution, and codebase exploration. | 20 | [`skills/TensorRT-LLM/`](skills/TensorRT-LLM) | [Source](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills) |
| **Model-Optimizer** | Model optimization — quantization, sparsity, and distillation for efficient inference. | 5 | [`skills/Model-Optimizer/`](skills/Model-Optimizer) | [Source](https://github.com/NVIDIA/Model-Optimizer/tree/main/.claude/skills) |
| **Megatron-Core** | Large-scale distributed training — model parallelism, pipeline parallelism, and mixed precision. | 3 | [`skills/Megatron-Core/`](skills/Megatron-Core) | [Source](https://github.com/NVIDIA/Megatron-LM/tree/main/.claude/skills) |
| **Megatron-Bridge** | Bridge between NeMo and Megatron — data processing, model conversion, and training utilities. | 9 | [`skills/Megatron-Bridge/`](skills/Megatron-Bridge) | [Source](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/skills) |
| **Nemotron Voice Agent** | Real-time conversational AI — deploy speech-to-speech voice agents on Workstation, Jetson Thor, or Cloud NIMs. | 1 | [`skills/nemotron-voice-agent/`](skills/nemotron-voice-agent) | [Source](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/tree/main/.agents/skills) |
| **NeMo Gym** | RL training environments — add benchmarks, resources servers, agent wiring, and reward profiling. | 1 | [`skills/NeMo-Gym/`](skills/NeMo-Gym) | [Source](https://github.com/NVIDIA-NeMo/Gym/tree/main/.claude/skills) |
| **NeMo Evaluator** | LLM evaluation — launch evaluations, access MLflow results, NeMo Evaluator Launcher assistant, and bring-your-own benchmarks. | 4 | [`skills/NeMo-Evaluator/`](skills/NeMo-Evaluator) | [Source](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/.claude/skills) |
| **RAG Blueprint** | RAG pipeline — deploy, configure, troubleshoot, and manage retrieval augmented generation with Docker Compose or Helm. | 1 | [`skills/rag/`](skills/rag) | [Source](https://github.com/NVIDIA-AI-Blueprints/rag/tree/main/skill-source/.agents/skills) |

---

## Getting Help & Contributing

For skill-related issues, feature requests, new skill ideas, discussions, and contributions — use the source repo for the relevant product:

| Product | Issues | Discussions | Contributing | Security |
|---------|--------|-------------|--------------|----------|
| **CUDA-Q** | [Issues](https://github.com/NVIDIA/cuda-quantum/issues) | [Discussions](https://github.com/NVIDIA/cuda-quantum/discussions) | [Contributing](https://github.com/NVIDIA/cuda-quantum/blob/main/Contributing.md) | [Security](https://github.com/NVIDIA/cuda-quantum/blob/main/SECURITY.md) |
| **cuOpt** | [Issues](https://github.com/NVIDIA/cuopt/issues) | [Discussions](https://github.com/NVIDIA/cuopt/discussions) | [Contributing](https://github.com/NVIDIA/cuopt/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/cuopt/blob/main/SECURITY.md) |
| **TensorRT-LLM** | [Issues](https://github.com/NVIDIA/TensorRT-LLM/issues) | [Discussions](https://github.com/NVIDIA/TensorRT-LLM/discussions) | [Contributing](https://github.com/NVIDIA/TensorRT-LLM/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/TensorRT-LLM/blob/main/SECURITY.md) |
| **Model-Optimizer** | [Issues](https://github.com/NVIDIA/Model-Optimizer/issues) | — | [Contributing](https://github.com/NVIDIA/Model-Optimizer/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/Model-Optimizer/blob/main/SECURITY.md) |
| **Megatron-Core** | [Issues](https://github.com/NVIDIA/Megatron-LM/issues) | [Discussions](https://github.com/NVIDIA/Megatron-LM/discussions) | [Contributing](https://github.com/NVIDIA/Megatron-LM/blob/main/CONTRIBUTING.md) | — |
| **Megatron-Bridge** | [Issues](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues) | [Discussions](https://github.com/NVIDIA-NeMo/Megatron-Bridge/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/CONTRIBUTING.md) | — |
| **Nemotron Voice Agent** | [Issues](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/issues) | [Discussions](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/discussions) | [Contributing](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/blob/main/SECURITY.md) |
| **NeMo Gym** | [Issues](https://github.com/NVIDIA-NeMo/Gym/issues) | [Discussions](https://github.com/NVIDIA-NeMo/Gym/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/Gym/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/Gym/blob/main/SECURITY.md) |
| **NeMo Evaluator** | [Issues](https://github.com/NVIDIA-NeMo/Evaluator/issues) | [Discussions](https://github.com/NVIDIA-NeMo/Evaluator/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/SECURITY.md) |
| **RAG Blueprint** | [Issues](https://github.com/NVIDIA-AI-Blueprints/rag/issues) | [Discussions](https://github.com/NVIDIA-AI-Blueprints/rag/discussions) | [Contributing](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/SECURITY.md) |

For issues with **this catalog repo itself** (README, structure, listing a new product): [open an issue here](../../issues).

---

## Roadmap

- ✅ Public skills catalog with NVIDIA-verified skills across multiple products
- ✅ Automated sync pipeline with skills mirrored from product repos daily
- ✅ Security scanning for all published skills covering instruction safety and supply-chain integrity
- 🔲 Skills signing so every published skill carries a verifiable NVIDIA signature
- 🔲 Skills universal evaluation criteria and task-specific criteria
- 🔲 Skills Card with machine-readable metadata for identity, provenance, quality, and behavioral boundaries
- 🔲 Compliance gates before external publication
- 🔲 Syndication to external marketplaces and MCP hubs

---

## Repository Structure

```
NVIDIA/skills/
├── skills/                  # All skills, mirrored daily from product repos
│   ├── cuopt/               # cuOpt skills (19)
│   ├── TensorRT-LLM/        # TensorRT-LLM skills (20)
│   ├── Megatron-Bridge/      # Megatron-Bridge skills (9)
│   ├── Model-Optimizer/      # Model-Optimizer skills (5)
│   ├── NeMo-Evaluator/       # NeMo Evaluator skills
│   ├── NeMo-Evaluator-Launcher/
│   ├── NeMo-Gym/             # NeMo Gym skills
│   ├── CUDA-Q/               # CUDA-Q skills
│   ├── Megatron-Core/        # Megatron-Core skills
│   ├── nemotron-voice-agent/ # Nemotron Voice Agent skills
│   └── ...
├── components.yml           # Product registry — teams onboard here
├── .github/workflows/       # Automated sync pipeline
├── CONTRIBUTING.md          # Contribution guidelines
├── SECURITY.md              # Security reporting policy
├── CODE_OF_CONDUCT.md       # Community code of conduct
└── LICENSE                  # Apache 2.0
```

Skills are maintained in their respective product repos (see the **Source** column in [Available Skills](#available-skills)) and automatically synced to this repo daily.

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

