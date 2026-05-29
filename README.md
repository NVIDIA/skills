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
| **AIQ** | NVIDIA AI-Q Blueprint - deploy local AI-Q services and run shallow or deep research workflows as agent skills. | 2 | [`skills/aiq-research/`](skills/aiq-research) | [Source](https://github.com/NVIDIA-AI-Blueprints/aiq/tree/develop/skills/aiq-research) | [`1323f65`](https://github.com/NVIDIA-AI-Blueprints/aiq/commit/1323f65804a9a0b14b341677bb25d431456a2a8b) · 2026-05-28 |
| **cuOpt** | GPU-accelerated optimization — vehicle routing, linear programming, quadratic programming, installation, server deployment, and developer tools. | 12 | [`skills/cuopt-developer/`](skills/cuopt-developer) | [Source](https://github.com/NVIDIA/cuopt/tree/main/skills/cuopt-developer) | [`ea7acf0`](https://github.com/NVIDIA/cuopt/commit/ea7acf0577575f0911812446cc7d63e99e5d6016) · 2026-05-29 |
| **cuPyNumeric** | NumPy and SciPy on multi-node multi-GPU systems — install and verify cuPyNumeric for Python. | 1 | [`skills/cupynumeric-install/`](skills/cupynumeric-install) | [Source](https://github.com/nv-legate/cupynumeric/tree/main/skills/cupynumeric-install) | [`7774592`](https://github.com/nv-legate/cupynumeric/commit/77745926fe8f73c852e640517a526f4d2bad6d9d) · 2026-05-28 |
| **DALI** | GPU-accelerated data loading and processing with NVIDIA DALI. | 1 | [`skills/dali-dynamic-mode/`](skills/dali-dynamic-mode) | [Source](https://github.com/NVIDIA/DALI/tree/main/skills/dali-dynamic-mode) | [`b1a2cd9`](https://github.com/NVIDIA/DALI/commit/b1a2cd9fa66da8f4d07a2e71c12f5f8e8a0d0872) · 2026-05-29 |
| **DeepStream** | Agentic skills for guided DeepStream development. | 2 | [`skills/deepstream-dev/`](skills/deepstream-dev) | [Source](https://github.com/NVIDIA-AI-IOT/DeepStream_Coding_Agent/tree/main/skills/deepstream-dev) | [`3daf16a`](https://github.com/NVIDIA-AI-IOT/DeepStream_Coding_Agent/commit/3daf16ae5168dd685608bb929fb4f5513f308045) · 2026-05-28 |
| **Dynamo** | NVIDIA Dynamo deployment bring-up on Kubernetes — pick and deploy recipes, start router modes, validate disagg NIXL/UCX/NCCL interconnect, and triage day-2 failures. | 4 | [`skills/dynamo-interconnect-check/`](skills/dynamo-interconnect-check) | [Source](https://github.com/ai-dynamo/dynamo/tree/main/skills/dynamo-interconnect-check) | [`5b4bc1d`](https://github.com/ai-dynamo/dynamo/commit/5b4bc1dd70965017a737c71b19db5a0aeaa88727) · 2026-05-29 |
| **NeMo AutoModel** | NeMo AutoModel - PyTorch-native distributed training for LLMs/VLMs with Hugging Face support, recipes, launchers, and validation workflows. | 4 | [`skills/nemo-automodel-distributed-training/`](skills/nemo-automodel-distributed-training) | [Source](https://github.com/NVIDIA-NeMo/Automodel/tree/main/skills/nemo-automodel-distributed-training) | [`38e301c`](https://github.com/NVIDIA-NeMo/Automodel/commit/38e301cc3c14c9799078bcf36a5b9dce7048a758) · 2026-05-29 |
| **NeMo MBridge** | NeMo MBridge - PyTorch-native bridge between Hugging Face and Megatron-Core for checkpoint conversion, training recipes, and NVIDIA GPU performance workflows. | 20 | [`skills/nemo-mbridge-mlm-bridge-training/`](skills/nemo-mbridge-mlm-bridge-training) | [Source](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/skills/nemo-mbridge-mlm-bridge-training) | [`3746fe2`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/3746fe2a43c3c5e0484d40fb9c2461508207e045) · 2026-05-29 |
| **NemoClaw** | Secure agent sandboxing — run OpenClaw inside NVIDIA OpenShell with managed inference, policy management, remote deployment, sandbox monitoring. | 10 | [`skills/nemoclaw-user-agent-skills/`](skills/nemoclaw-user-agent-skills) | [Source](https://github.com/NVIDIA/NemoClaw/tree/main/skills/nemoclaw-user-agent-skills) | [`6fb96bd`](https://github.com/NVIDIA/NemoClaw/commit/6fb96bd1bb9a5b7b6a9fd688a2addcba1a8fec13) · 2026-05-29 |
| **Nemotron** | Author end-to-end model development, customization, evaluation, and deployment pipelines using the NVIDIA AI stack. | 1 | [`skills/nemotron-customize/`](skills/nemotron-customize) | [Source](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/skills/nemotron-customize) | [`2b7a80b`](https://github.com/NVIDIA-NeMo/Nemotron/commit/2b7a80ba2ee45ffb2bd1c10496c0355d6a5a4bbb) · 2026-05-29 |
| **Physical AI Skill Hub** | Agent skills for Omniverse and Physical AI workflows. | 5 | [`skills/omniverse-cad-to-simready/`](skills/omniverse-cad-to-simready) | [Source](https://github.com/NVIDIA-dev/physical-ai-skill-hub-dev/tree/main/skills/omniverse-cad-to-simready) | — |
| **RAG Blueprint** | RAG pipeline — deploy, configure, troubleshoot, and manage retrieval augmented generation with Docker Compose or Helm. | 3 | [`skills/rag-blueprint/`](skills/rag-blueprint) | [Source](https://github.com/NVIDIA-AI-Blueprints/rag/tree/develop/skills/rag-blueprint) | [`315bd3e`](https://github.com/NVIDIA-AI-Blueprints/rag/commit/315bd3e80b4d797e19c2c1b698918efe5c86fe7f) · 2026-05-29 |
| **Skill Card Generator** | Reads an agent skill's source files and produces a skill card plus a review table. Use when a skill directory exists and a governance card needs to be generated or updated. | 1 | [`skills/skill-card-generator/`](skills/skill-card-generator) | [Source](https://github.com/NVIDIA/Trustworthy-AI/tree/main/skills/skill-card-generator) | [`09b0d6b`](https://github.com/NVIDIA/Trustworthy-AI/commit/09b0d6b03ee587cc8672ad3133a4eb84aeadbd2a) · 2026-05-28 |
| **TileGym** | Tile-based GPU programming — adding new kernels, cross-framework conversion, and performance optimization. | 1 | [`skills/tilegym-adding-cutile-kernel/`](skills/tilegym-adding-cutile-kernel) | [Source](https://github.com/NVIDIA/TileGym/tree/main/skills/tilegym-adding-cutile-kernel) | [`2bf003b`](https://github.com/NVIDIA/TileGym/commit/2bf003b049c04f45b222423d328cb51ba5cf2670) · 2026-05-29 |
<!-- skills-table-end -->

---

## Getting Help & Contributing

For skill-related issues, feature requests, new skill ideas, discussions, and contributions — use the source repo for the relevant product:

<!-- help-table-start -->
| Product | Issues | Discussions | Contributing | Security |
|---------|--------|-------------|--------------|----------|
| **AIQ** | [Issues](https://github.com/NVIDIA-AI-Blueprints/aiq/issues) | [Discussions](https://github.com/NVIDIA-AI-Blueprints/aiq/discussions) | [Contributing](https://github.com/NVIDIA-AI-Blueprints/aiq/blob/develop/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-Blueprints/aiq/blob/develop/SECURITY.md) |
| **cuOpt** | [Issues](https://github.com/NVIDIA/cuopt/issues) | [Discussions](https://github.com/NVIDIA/cuopt/discussions) | [Contributing](https://github.com/NVIDIA/cuopt/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/cuopt/blob/main/SECURITY.md) |
| **cuPyNumeric** | [Issues](https://github.com/nv-legate/cupynumeric/issues) | [Discussions](https://github.com/nv-legate/cupynumeric/discussions) | [Contributing](https://github.com/nv-legate/cupynumeric/blob/main/CONTRIBUTING.md) | [Security](https://github.com/nv-legate/cupynumeric/blob/main/SECURITY.md) |
| **DALI** | [Issues](https://github.com/NVIDIA/DALI/issues) | [Discussions](https://github.com/NVIDIA/DALI/discussions) | [Contributing](https://github.com/NVIDIA/DALI/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/DALI/blob/main/SECURITY.md) |
| **DeepStream** | [Issues](https://github.com/NVIDIA-AI-IOT/DeepStream_Coding_Agent/issues) | [Discussions](https://github.com/NVIDIA-AI-IOT/DeepStream_Coding_Agent/discussions) | [Contributing](https://github.com/NVIDIA-AI-IOT/DeepStream_Coding_Agent/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-IOT/DeepStream_Coding_Agent/blob/main/SECURITY.md) |
| **Dynamo** | [Issues](https://github.com/ai-dynamo/dynamo/issues) | [Discussions](https://github.com/ai-dynamo/dynamo/discussions) | [Contributing](https://github.com/ai-dynamo/dynamo/blob/main/CONTRIBUTING.md) | [Security](https://github.com/ai-dynamo/dynamo/blob/main/SECURITY.md) |
| **NeMo AutoModel** | [Issues](https://github.com/NVIDIA-NeMo/Automodel/issues) | [Discussions](https://github.com/NVIDIA-NeMo/Automodel/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/Automodel/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/Automodel/blob/main/SECURITY.md) |
| **NeMo MBridge** | [Issues](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues) | [Discussions](https://github.com/NVIDIA-NeMo/Megatron-Bridge/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/SECURITY.md) |
| **NemoClaw** | [Issues](https://github.com/NVIDIA/NemoClaw/issues) | [Discussions](https://github.com/NVIDIA/NemoClaw/discussions) | [Contributing](https://github.com/NVIDIA/NemoClaw/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/NemoClaw/blob/main/SECURITY.md) |
| **Nemotron** | [Issues](https://github.com/NVIDIA-NeMo/Nemotron/issues) | [Discussions](https://github.com/NVIDIA-NeMo/Nemotron/discussions) | [Contributing](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/SECURITY.md) |
| **Physical AI Skill Hub** | [Issues](https://github.com/NVIDIA-dev/physical-ai-skill-hub-dev/issues) | [Discussions](https://github.com/NVIDIA-dev/physical-ai-skill-hub-dev/discussions) | [Contributing](https://github.com/NVIDIA-dev/physical-ai-skill-hub-dev/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-dev/physical-ai-skill-hub-dev/blob/main/SECURITY.md) |
| **RAG Blueprint** | [Issues](https://github.com/NVIDIA-AI-Blueprints/rag/issues) | [Discussions](https://github.com/NVIDIA-AI-Blueprints/rag/discussions) | [Contributing](https://github.com/NVIDIA-AI-Blueprints/rag/blob/develop/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA-AI-Blueprints/rag/blob/develop/SECURITY.md) |
| **Skill Card Generator** | [Issues](https://github.com/NVIDIA/Trustworthy-AI/issues) | [Discussions](https://github.com/NVIDIA/Trustworthy-AI/discussions) | [Contributing](https://github.com/NVIDIA/Trustworthy-AI/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/Trustworthy-AI/blob/main/SECURITY.md) |
| **TileGym** | [Issues](https://github.com/NVIDIA/TileGym/issues) | [Discussions](https://github.com/NVIDIA/TileGym/discussions) | [Contributing](https://github.com/NVIDIA/TileGym/blob/main/CONTRIBUTING.md) | [Security](https://github.com/NVIDIA/TileGym/blob/main/SECURITY.md) |
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
├── skills/                      # Verified skills, mirrored from product repos
│   ├── README.md                 # Install guidance for people browsing this folder
│   ├── aiq-deploy/               # AIQ — deploy AI-Q services
│   ├── aiq-research/             # AIQ — research workflows
│   ├── cuopt-developer/          # cuOpt — internals & contribution
│   ├── cuopt-install/            # cuOpt — installation
│   ├── cuopt-routing-api-python/ # cuOpt — VRP/TSP routing (Python)
│   ├── dali-dynamic-mode/        # DALI — dynamic-mode pipeline authoring
│   ├── deepstream-dev/           # DeepStream — guided development
│   ├── dynamo-recipe-runner/     # Dynamo — Kubernetes recipe deploys
│   ├── nemo-automodel-*/         # NeMo AutoModel — distributed training (4 skills)
│   ├── nemoclaw-user-*/          # NemoClaw — secure agent sandboxing (10 skills)
│   ├── nemotron-customize/       # Nemotron — model customization
│   ├── rag-blueprint/            # RAG Blueprint — RAG pipeline
│   ├── skill-card-generator/     # Trustworthy-AI — skill card generation
│   └── tilegym-adding-cutile-kernel/  # TileGym — cuTile kernel authoring
│                                 # Each skill lives flat at top level (no product parent
│                                 # dir), so `npx skills add --skill <name>` resolves cleanly
├── components.d/                # Product registry — one file per component, teams onboard here
│   ├── README.md                 # Schema and onboarding instructions
│   ├── aiq.yml
│   ├── cuopt.yml
│   └── …                         # one file per registered product
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
│   ├── workflows/                # Sync pipeline, plugin validation, DCO check
│   └── scripts/                  # regenerate-readme.sh, build-plugins.py
├── nv-agent-root-cert.pem       # Trust anchor for OMS signature verification
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
