# Contributing to NVIDIA Agent Skills

We welcome contributions: new skills, improvements to existing ones, or documentation fixes. Please read this guide before submitting pull requests.

All participants are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Local Setup

```bash
git clone https://github.com/nvidia/agent-skills.git
cd agent-skills
```

To test a skill, copy it into your agent's skills directory and try it on a real task:

```bash
# Example: test the cuOpt LP/MILP skill with Claude Code
cp -r skills/cuopt-lp-milp-api-python ~/.claude/skills/
```

To validate a skill's `SKILL.md` structure, use the [`skills-ref`](https://github.com/agentskills/agentskills/tree/main/skills-ref) reference library.

---

## How to Contribute

### Where to File Issues

- **New skill requests** — [open an issue in this repo](../../issues). Describe the use case, the product or workflow it would cover, and any relevant context.
- **Bugs or issues with an existing skill** — file the issue in the **source repo** where the skill originates (e.g., [NVIDIA/cuopt](https://github.com/NVIDIA/cuopt) for cuOpt skills, [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) for TensorRT-LLM skills). The maintainers of each product are best positioned to triage and fix skill-specific issues.
- **Issues with this repository itself** (README, structure, contribution process) — [open an issue here](../../issues).

### Finding Something to Work On

Look for issues labeled [`good first issue`](../../labels/good%20first%20issue) — these are scoped, well-defined tasks ideal for new contributors. If you're unsure where to start, ask in [GitHub Discussions](../../discussions).

### Improving Existing Skills

1. Fork the repository.
2. Create a feature branch: `git checkout -b fix/skill-name-description`
3. Make your changes and test locally with a compatible agent.
4. Open a Pull Request with a clear description of the change.

### Submitting a New Skill for NVIDIA Verification

We accept community-authored skills into the `community/` directory. Once reviewed and approved by NVIDIA, verified skills are promoted to `skills/`.

#### Requirements

- Follow the [Agent Skills specification](https://agentskills.io/specification).
- Include a `SKILL.md` with valid YAML frontmatter (`name` and `description` are required).
- Test your skill locally with your preferred agent before submitting.
- Do not include secrets, credentials, or proprietary dependencies without clear documentation.
- Keep skills focused — one skill should solve one clear problem.

#### Submission Process

1. Fork this repository.
2. Add your skill to `community/<your-skill-name>/`.
3. Ensure your skill directory follows this structure:
   ```
   community/your-skill-name/
   ├── SKILL.md          # Required: skill definition
   ├── scripts/          # Optional: executable scripts
   ├── references/       # Optional: reference documents
   └── assets/           # Optional: static assets
   ```
4. Open a Pull Request with:
   - A description of what the skill does
   - Example usage
   - Any dependencies or requirements
5. NVIDIA will review for quality, security, and compatibility.
6. If approved, the skill is merged and may be promoted to `skills/` with the NVIDIA-verified badge.

---

## Pull Request Checklist

Before submitting a PR, make sure:

- [ ] All commits are signed off (`git commit -s`) — see [Signing Your Work](#signing-your-work)
- [ ] `SKILL.md` has valid YAML frontmatter with `name` and `description`
- [ ] Skill tested locally with at least one agent
- [ ] No secrets, credentials, or API keys included
- [ ] PR description includes what the skill does and an example of how to use it

---

## Review Process

NVIDIA maintainers review all pull requests. Here's what to expect:

- **Acknowledgement** within a few business days
- **Review feedback** on quality, security, and compatibility with the Agent Skills spec
- If changes are requested, update your branch and the reviewer will re-review
- Once approved, an NVIDIA maintainer will merge the PR

If your PR hasn't received a response after a week, feel free to leave a comment to bump it.

---

## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```

---

## Code of Conduct

All participants are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md). Be respectful, constructive, and inclusive.

---

## Questions?

Open a [GitHub Discussion](../../discussions) — we're happy to help.
