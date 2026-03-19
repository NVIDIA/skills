# NVIDIA Agent Skills

**Official, NVIDIA-verified skills for AI coding agents.**

[![NVIDIA](https://img.shields.io/badge/NVIDIA-Verified-76B900?style=flat&logo=nvidia&logoColor=white)](https://nvidia.com)
[![Agent Skills Spec](https://img.shields.io/badge/Agent%20Skills-Specification-blue)](https://agentskills.io)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

Skills are portable instruction sets that teach AI agents how to perform specialized tasks — from solving vehicle routing problems with GPU-accelerated cuOpt, to onboarding HuggingFace models into TensorRT-LLM AutoDeploy, to deploying real-time voice agents on Jetson and cloud NIMs. Every skill in this repository is **published and verified by NVIDIA**.

This repository follows the open [Agent Skills specification](https://agentskills.io/specification), making skills compatible with any AI agent or framework that supports the standard.

---

## Getting Started

### 1. Browse the skills

Explore the [`skills/`](skills/) directory to find NVIDIA-verified skills for your use case.

### 2. Install a skill

Each skill is a self-contained directory that can be added to any compatible agent. How you install depends on your agent or framework:

| Agent / Framework | Installation |
|-------------------|-------------|
| Claude Code | `/plugin install <skill-name>@nvidia-agent-skills` |
| Codex | Copy the skill directory into your project's `.codex/skills/` folder |
| Cursor | Copy the skill directory into your project's `.cursor/skills/` folder |
| Other agents | Copy the skill directory into your agent's skills folder |
| Manual | Clone this repo and point your agent to the skill path |

### 3. Use a skill

Once installed, skills activate automatically when relevant — or invoke them using your agent's command interface. Refer to your agent's documentation for specifics.

---

## Available Skills

| Product | Description | Skills | Source |
|---------|-------------|:------:|--------|
| **cuOpt** | GPU-accelerated optimization — vehicle routing, linear programming, quadratic programming, installation, server deployment, and developer tools. | 19 | [NVIDIA/cuopt](https://github.com/NVIDIA/cuopt) |
| **TensorRT-LLM** | LLM inference optimization — model onboarding to AutoDeploy, CI pipeline failure analysis, and test failure diagnostics. | 3 | [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) |
| **Nemotron Voice Agent** | Real-time conversational AI — deploy speech-to-speech voice agents on Workstation, Jetson Thor, or Cloud NIMs. | 1 | [NVIDIA-AI-Blueprints/nemotron-voice-agent](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent) |
| **NeMo Gym** | RL training environments — add benchmarks, resources servers, agent wiring, and reward profiling. | 1 | [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym) |

Browse all skills in the [`skills/`](skills/) directory.

---

## Community

We take a **GitHub-first approach** to community engagement. GitHub is the primary place to ask questions, report issues, request features, and collaborate with NVIDIA and fellow developers.

### How to Communicate with Us

| Channel | Purpose | Link |
|---------|---------|------|
| **GitHub Issues (this repo)** | New skill requests and issues with this repository | [Open an Issue](../../issues) |
| **GitHub Issues (source repo)** | Bugs or issues with an existing skill — file in the repo where the skill lives (see [Available Skills](#available-skills) for links) | See source repo |
| **GitHub Discussions** | General Q&A, ideas, show & tell, and community conversation | [Join Discussions](../../discussions) |
| **Pull Requests** | Contribute improvements, fixes, or documentation | [Contributing Guide](CONTRIBUTING.md) |

### Why GitHub-First?

- **Transparent** — every conversation is public and searchable, so answers help everyone.
- **Traceable** — issues and PRs link directly to the code changes they produce.
- **Inclusive** — anyone with a GitHub account can participate, no special access required.
- **Persistent** — discussions and decisions live alongside the code, not buried in chat history.

> **Note:** For NVIDIA-internal inquiries or partnership-related topics, reach out via your existing NVIDIA contacts. For security vulnerabilities, please follow our [Security Policy](SECURITY.md) and report privately — do not open a public issue.

---

## Contributing

We welcome contributions from the community! Whether it's fixing a typo, improving documentation, or proposing a new skill — we'd love to hear from you.

### Submitting a Skill for NVIDIA Verification

Community-authored skills can be submitted for NVIDIA review and verification:

1. **Build your skill** following the [Agent Skills specification](https://agentskills.io/specification).
2. **Test locally** with your preferred agent to ensure it works as expected.
3. **Open a Pull Request** against this repository with your skill in the `community/` directory.
4. **NVIDIA reviews** for quality, security, and compatibility.
5. **If approved**, the skill is merged and earns the NVIDIA-verified badge.

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

### What Makes a Good Skill?

- Solves a clear, specific problem
- Follows the [Agent Skills specification](https://agentskills.io/specification)
- Includes a well-written `SKILL.md` with accurate name and description
- Works reliably across agents and environments
- Does not require proprietary dependencies without clear documentation

---

## Repository Structure

```
nvidia-agent-skills/
├── skills/              # NVIDIA-verified skills
│   ├── skill-name/
│   │   ├── SKILL.md     # Skill definition (required)
│   │   ├── scripts/     # Executable scripts (optional)
│   │   ├── references/  # Reference documents (optional)
│   │   └── assets/      # Static assets (optional)
│   └── ...
├── community/           # Community-submitted skills under review
├── spec/                # Local copy of the Agent Skills spec
├── CONTRIBUTING.md      # Contribution guidelines
├── SECURITY.md          # Security reporting policy
├── CODE_OF_CONDUCT.md   # Community code of conduct
└── LICENSE              # Apache 2.0
```

---

## Standards & Compatibility

This repository adheres to the [Agent Skills specification](https://agentskills.io/specification):

- Skills are portable directories with a `SKILL.md` file at their root.
- Metadata uses YAML frontmatter with required `name` and `description` fields.
- Skills follow a progressive disclosure model — lightweight metadata loads at startup, full instructions load on activation.
- Validate your skill using the [`skills-ref`](https://github.com/agentskills/agentskills/tree/main/skills-ref) reference library.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE) unless otherwise noted in individual skill directories.

---

<p align="center">
  <strong>Built by NVIDIA. Powered by the community.</strong><br>
  <a href="../../issues">Report a Bug</a> · <a href="../../discussions">Start a Discussion</a> · <a href="CONTRIBUTING.md">Contribute</a>
</p>
