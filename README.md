# NVIDIA Agent Skills

**Official, NVIDIA-verified skills for AI coding agents.**

[![NVIDIA](https://img.shields.io/badge/NVIDIA-Verified-76B900?style=flat&logo=nvidia&logoColor=white)](https://nvidia.com)
[![Agent Skills Spec](https://img.shields.io/badge/Agent%20Skills-Specification-blue)](https://agentskills.io)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

Skills are portable instruction sets that teach AI agents how to perform specialized tasks — from solving vehicle routing problems with GPU-accelerated cuOpt, to onboarding HuggingFace models into TensorRT-LLM AutoDeploy, to deploying real-time voice agents on Jetson and cloud NIMs. Every skill in this repository is **published and verified by NVIDIA**.

This repository follows the open [Agent Skills specification](https://agentskills.io/specification), making skills compatible with any AI agent or framework that supports the standard.

---

## Quickstart

Clone the repo and copy a skill into your agent's skills directory:

```bash
git clone https://github.com/nvidia/agent-skills.git
cp -r agent-skills/skills/cuopt-lp-milp-api-python ~/.claude/skills/
```

That's it — the skill activates automatically the next time your agent encounters a relevant task. For example, ask your agent to "solve a linear programming problem with cuOpt" and the skill guides it through the cuOpt Python API.

### Install by Agent

| Agent / Framework | Installation |
|-------------------|-------------|
| Claude Code | `/plugin install <skill-name>@nvidia-agent-skills` |
| Codex | Copy the skill directory into your project's `.codex/skills/` folder |
| Cursor | Copy the skill directory into your project's `.cursor/skills/` folder |
| Other agents | Copy the skill directory into your agent's skills folder |
| Manual | Clone this repo and point your agent to the skill path |

Browse all skills in the [`skills/`](skills/) directory.

---

## Available Skills

| Product | Description | Skills | Source |
|---------|-------------|:------:|--------|
| **cuOpt** | GPU-accelerated optimization — vehicle routing, linear programming, quadratic programming, installation, server deployment, and developer tools. | 19 | [NVIDIA/cuopt](https://github.com/NVIDIA/cuopt) |
| **TensorRT-LLM** | LLM inference optimization — model onboarding to AutoDeploy, CI pipeline failure analysis, and test failure diagnostics. | 3 | [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) |
| **Nemotron Voice Agent** | Real-time conversational AI — deploy speech-to-speech voice agents on Workstation, Jetson Thor, or Cloud NIMs. | 1 | [NVIDIA-AI-Blueprints/nemotron-voice-agent](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent) |
| **NeMo Gym** | RL training environments — add benchmarks, resources servers, agent wiring, and reward profiling. | 1 | [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym) |

---

## Getting Help

- **Questions and discussion:** [GitHub Discussions](../../discussions)
- **New skill requests:** [Open an Issue](../../issues) in this repo
- **Bugs with an existing skill:** File an issue in the source repo where the skill lives (see [Available Skills](#available-skills) for links)
- **Security vulnerabilities:** See [SECURITY.md](SECURITY.md) — do not use GitHub Issues

---

## Contributing

We welcome contributions — new skills, improvements, or documentation fixes. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting pull requests.

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
