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
| Product | Description | Skills | Catalog | Source | Version |
|---------|-------------|:------:|---------|--------|---------|
| **AIQ** | NVIDIA AI-Q Blueprint - deploy local AI-Q services and run shallow or deep research workflows as agent skills. | 2 | [`skills/aiq-research/`](skills/aiq-research) | [Source](https://github.com/NVIDIA-AI-Blueprints/aiq/tree/develop/skills/aiq-research) | [`66b009e`](https://github.com/NVIDIA-AI-Blueprints/aiq/commit/66b009e3b28219eb2a09910c347d84c0675c2028) · 2026-05-27 |
| **cuOpt** | GPU-accelerated optimization — vehicle routing, linear programming, quadratic programming, installation, server deployment, and developer tools. | 6 | [`skills/cuopt/`](skills/cuopt) | [Source](https://github.com/NVIDIA/cuopt/tree/main/skills) | [`d415bb6`](https://github.com/NVIDIA/cuopt/commit/d415bb6447a86cd268e824ef64ac05dcbc140ca0) · 2026-05-27 |
<!-- skills-table-end -->

---

## Getting Help & Contributing

For skill-related issues, feature requests, new skill ideas, discussions, and contributions — use the source repo for the relevant product:

<!-- help-table-start -->
| Product | Issues | Discussions | Contributing | Security |
|---------|--------|-------------|--------------|----------|
| **AIQ** | [Issues](https://github.com/NVIDIA-AI-Blueprints/aiq/issues) | [Discussions](https://github.com/NVIDIA-AI-Blueprints/aiq/discussions) | [Contributing](https://github.com/NVIDIA-AI-Blueprints/aiq/blob/develop/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-Blueprints/aiq/blob/develop/SECURITY.md) |
| **cuOpt** | [Issues](https://github.com/NVIDIA/cuopt/issues) | [Discussions](https://github.com/NVIDIA/cuopt/discussions) | [Contributing](https://github.com/NVIDIA/cuopt/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/cuopt/blob/main/SECURITY.md) |
<!-- help-table-end -->

For issues with **this catalog repo itself** (README, structure, listing a new product): [open an issue here](../../issues).

---

## Verifying Skills

Every published skill ships with a detached OMS signature (`skill.oms.sig`). The sync pipeline drops any skill missing `skill.oms.sig`, `skill-card.md`, or `evals.json` before publishing, so anything in the catalog has all three artifacts.

Verify a skill against the NVIDIA trust anchor [`nv-agent-root-cert.pem`](nv-agent-root-cert.pem):

```bash
pip install model-signing
model_signing verify certificate SKILL_DIR \
  --signature SKILL_DIR/skill.oms.sig \
  --certificate_chain nv-agent-root-cert.pem \
  --ignore_unsigned_files
```

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
├── docs/                    # Long-form documentation (published via Fern)
│   ├── README.md             # How to build the docs locally
│   ├── index.mdx
│   ├── advanced-install.mdx  # Advanced skills CLI usage
│   ├── agent-skill-trust-pipeline.mdx
│   ├── release-checklist.mdx
│   ├── scanning-agent-skills.mdx
│   ├── signing-agent-skills.mdx
│   └── skill-cards.mdx
├── fern/                    # Fern docs site configuration
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
