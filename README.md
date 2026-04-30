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

<!-- skills-table-start -->
| Product | Description | Skills | Catalog | Source | Version |
|---------|-------------|:------:|---------|--------|---------|
| **CUDA-Q** | CUDA Quantum — onboarding guide for installation, test programs, GPU simulation, QPU hardware, and quantum applications. | 1 | [`skills/CUDA-Q/`](skills/CUDA-Q) | [Source](https://github.com/NVIDIA/cuda-quantum/tree/main/.claude/skills) | [`33abdcc`](https://github.com/NVIDIA/cuda-quantum/commit/33abdccb94d1fdfb7655083c0a55b9b6e3ffd260) · 2026-04-29 |
| **cuOpt** | GPU-accelerated optimization — vehicle routing, linear programming, quadratic programming, installation, server deployment, and developer tools. | 19 | [`skills/cuopt/`](skills/cuopt) | [Source](https://github.com/NVIDIA/cuopt/tree/main/skills) | [`b2159a5`](https://github.com/NVIDIA/cuopt/commit/b2159a52f43969623c568da13afc9cb21219413f) · 2026-04-29 |
| **Megatron-Bridge** | Bridge between NeMo and Megatron — data processing, model conversion, and training utilities. | 25 | [`skills/Megatron-Bridge/`](skills/Megatron-Bridge) | [Source](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/skills) | [`d1120d7`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/d1120d7c5cb0b2e4f4d611731bcb2f1d318f7f8c) · 2026-04-29 |
| **Megatron-Core** | Large-scale distributed training — model parallelism, pipeline parallelism, and mixed precision. | 7 | [`skills/Megatron-Core/`](skills/Megatron-Core) | [Source](https://github.com/NVIDIA/Megatron-LM/tree/main/skills) | [`c5201a0`](https://github.com/NVIDIA/Megatron-LM/commit/c5201a026ac27e42d0633ff36d2f1c56e1585009) · 2026-04-29 |
| **Model-Optimizer** | Model optimization — quantization, sparsity, and distillation for efficient inference. | 8 | [`skills/Model-Optimizer/`](skills/Model-Optimizer) | [Source](https://github.com/NVIDIA/Model-Optimizer/tree/main/.claude/skills) | [`9bb917d`](https://github.com/NVIDIA/Model-Optimizer/commit/9bb917d57c65e52bbd5a2f66660b3ab253a52a61) · 2026-04-29 |
| **NeMo Evaluator** | LLM evaluation — launch evaluations, access MLflow results, NeMo Evaluator Launcher assistant, and bring-your-own benchmarks. | 4 | [`skills/NeMo-Evaluator-Launcher/`](skills/NeMo-Evaluator-Launcher) | [Source](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/.claude/skills) | [`df2de44`](https://github.com/NVIDIA-NeMo/Evaluator/commit/df2de4489729aa42377c298df4c182bc445f8b60) · 2026-04-29 |
| **NeMo Gym** | RL training environments — add benchmarks, resources servers, agent wiring, and reward profiling. | 2 | [`skills/NeMo-Gym/`](skills/NeMo-Gym) | [Source](https://github.com/NVIDIA-NeMo/Gym/tree/main/.claude/skills) | [`fa95df1`](https://github.com/NVIDIA-NeMo/Gym/commit/fa95df1f1ab6392915ab6213cbf431a8d0b98bf1) · 2026-04-29 |
| **NemoClaw** | Secure agent sandboxing — run OpenClaw inside NVIDIA OpenShell with managed inference, policy management, remote deployment, sandbox monitoring, and contributor/maintainer workflows. | 21 | [`skills/NemoClaw/`](skills/NemoClaw) | [Source](https://github.com/NVIDIA/NemoClaw/tree/main/.agents/skills) | [`a794971`](https://github.com/NVIDIA/NemoClaw/commit/a794971c429e3651a50ef3c6c332d6a32d08ad5a) · 2026-04-29 |
| **Nemotron Voice Agent** | Real-time conversational AI — deploy speech-to-speech voice agents on Workstation, Jetson Thor, or Cloud NIMs. | 1 | [`skills/nemotron-voice-agent/`](skills/nemotron-voice-agent) | [Source](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/tree/main/.agents/skills) | [`a87826c`](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/commit/a87826cc1ab03103cea7f1f24dc94f456500e5c2) · 2026-04-22 |
| **RAG Blueprint** | RAG pipeline — deploy, configure, troubleshoot, and manage retrieval augmented generation with Docker Compose or Helm. | 1 | [`skills/rag/`](skills/rag) | [Source](https://github.com/NVIDIA-AI-Blueprints/rag/tree/main/skill-source/.agents/skills) | [`1c03212`](https://github.com/NVIDIA-AI-Blueprints/rag/commit/1c0321235e344c68e8b63133d0fdf8edb5592bfc) · 2026-04-18 |
| **TensorRT-LLM** | LLM inference optimization — model onboarding, performance analysis and optimization, kernel writing, CI diagnostics, code contribution, and codebase exploration. | 24 | [`skills/TensorRT-LLM/`](skills/TensorRT-LLM) | [Source](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills) | [`3b7af1c`](https://github.com/NVIDIA/TensorRT-LLM/commit/3b7af1c21f1071d57ada08641f0f76c4827ae961) · 2026-04-29 |
<!-- skills-table-end -->

---

## Getting Help & Contributing

For skill-related issues, feature requests, new skill ideas, discussions, and contributions — use the source repo for the relevant product:

<!-- help-table-start -->
| Product | Issues | Discussions | Contributing | Security |
|---------|--------|-------------|--------------|----------|
| **CUDA-Q** | [Issues](https://github.com/NVIDIA/cuda-quantum/issues) | [Discussions](https://github.com/NVIDIA/cuda-quantum/discussions) | [Contributing](https://github.com/NVIDIA/cuda-quantum/blob/main/Contributing.md) | [Security](https://github.com/NVIDIA/cuda-quantum/blob/main/SECURITY.md) |
| **cuOpt** | [Issues](https://github.com/NVIDIA/cuopt/issues) | [Discussions](https://github.com/NVIDIA/cuopt/discussions) | [Contributing](https://github.com/NVIDIA/cuopt/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/cuopt/blob/main/SECURITY.md) |
| **Megatron-Bridge** | [Issues](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues) | [Discussions](https://github.com/NVIDIA-NeMo/Megatron-Bridge/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/SECURITY.md) |
| **Megatron-Core** | [Issues](https://github.com/NVIDIA/Megatron-LM/issues) | [Discussions](https://github.com/NVIDIA/Megatron-LM/discussions) | [Contributing](https://github.com/NVIDIA/Megatron-LM/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/Megatron-LM/blob/main/SECURITY.md) |
| **Model-Optimizer** | [Issues](https://github.com/NVIDIA/Model-Optimizer/issues) | [Discussions](https://github.com/NVIDIA/Model-Optimizer/discussions) | [Contributing](https://github.com/NVIDIA/Model-Optimizer/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/Model-Optimizer/blob/main/SECURITY.md) |
| **NeMo Evaluator** | [Issues](https://github.com/NVIDIA-NeMo/Evaluator/issues) | [Discussions](https://github.com/NVIDIA-NeMo/Evaluator/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/SECURITY.md) |
| **NeMo Gym** | [Issues](https://github.com/NVIDIA-NeMo/Gym/issues) | [Discussions](https://github.com/NVIDIA-NeMo/Gym/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/Gym/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/Gym/blob/main/SECURITY.md) |
| **NemoClaw** | [Issues](https://github.com/NVIDIA/NemoClaw/issues) | [Discussions](https://github.com/NVIDIA/NemoClaw/discussions) | [Contributing](https://github.com/NVIDIA/NemoClaw/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/NemoClaw/blob/main/SECURITY.md) |
| **Nemotron Voice Agent** | [Issues](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/issues) | [Discussions](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/discussions) | [Contributing](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/blob/main/SECURITY.md) |
| **RAG Blueprint** | [Issues](https://github.com/NVIDIA-AI-Blueprints/rag/issues) | [Discussions](https://github.com/NVIDIA-AI-Blueprints/rag/discussions) | [Contributing](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/SECURITY.md) |
| **TensorRT-LLM** | [Issues](https://github.com/NVIDIA/TensorRT-LLM/issues) | [Discussions](https://github.com/NVIDIA/TensorRT-LLM/discussions) | [Contributing](https://github.com/NVIDIA/TensorRT-LLM/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/TensorRT-LLM/blob/main/SECURITY.md) |
<!-- help-table-end -->

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
│   ├── CUDA-Q/               # CUDA-Q skills (1)
│   ├── cuopt/                # cuOpt skills (19)
│   ├── Megatron-Bridge/      # Megatron-Bridge skills (9)
│   ├── Megatron-Core/        # Megatron-Core skills (6)
│   ├── Model-Optimizer/      # Model-Optimizer skills (5)
│   ├── NeMo-Evaluator/       # NeMo Evaluator skills
│   ├── NeMo-Evaluator-Launcher/
│   ├── NeMo-Gym/             # NeMo Gym skills (1)
│   ├── NemoClaw/             # NemoClaw skills (21)
│   ├── nemotron-voice-agent/ # Nemotron Voice Agent skills (1)
│   ├── TensorRT-LLM/         # TensorRT-LLM skills (20)
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

