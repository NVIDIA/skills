# NVFLARE Agent Skills

This directory contains NVFLARE-owned agent skills for supported coding agents.

Each skill lives in its own directory with a `SKILL.md` file, supporting
references, and its co-located evaluation suite:

```text
skills/
  nvflare-your-skill/
    SKILL.md
    references/
    evals/
      evals.json
      files/
```

Evaluation suites are grading-oracle data, not runtime guidance. They are
co-located with the owning skill to follow the Agent Skills Specification. They
are not referenced as agent instructions, so their presence does not change a
skill's triggering or workflow behavior.

`nvflare-shared/` is an internal, non-triggered skill: it holds references and
templates shared by the other skills and is installed alongside them, but it is
not user-selectable. It still follows the skill structure (a valid `SKILL.md`
with `status: internal`, plus `references/` and `assets/`).

`SKILL.md` frontmatter uses the portable fields from the
[agentskills.io spec](https://agentskills.io/specification) plus the NVIDIA
catalog extensions documented below. NVFLARE's required fields (`author`,
`version`, `min-flare-version`, `blast-radius`, and public-skill `category`) are
nested under the `metadata:` map:

```yaml
---
name: nvflare-your-skill
description: Short trigger-oriented description.
metadata:
  version: "0.1.0"
  author: "NVIDIA FLARE Team <federatedlearning@nvidia.com>"
  min-flare-version: "2.9.0"
  blast-radius: read_only
  category: Orientation
---
```

The skill name above is illustrative; actual skill directories use their
published skill names. Other than the NVIDIA catalog fields accepted by the
validator, do not place NVFLARE custom fields at the top level; nest them under
`metadata:`.

Every skill must include a team `author` identity under `metadata:` in the
`Name <group@nvidia.com>` form. Public skills must also include `category` as
product-facing runtime metadata. Draft, internal, and private skills
(`metadata.status`) may omit `category` while they are not publishable.

Bundled skills declare `version` as a string under `metadata`, following the
Agent Skills Specification. A root-level `version` is ignored by conforming
consumers and rejected by the NVFLARE validator.

`blast-radius` must be one of:

- `read_only`
- `edits_files`
- `runs_simulator`
- `submits_poc`
- `submits_production`

## Installing the skills

NVFLARE skills are installed with the standard [`skills`](https://agentskills.io)
tool via `npx skills add`. Both `claude-code` and `codex` are supported agent
targets. Install the whole set together so cross-skill references resolve:
`nvflare-shared/` is loaded by the other skills through relative references, so
installing skills individually can leave those references dangling.

From a local checkout (pre-publish):

```bash
npx skills add ./skills --skill '*' -a claude-code -a codex -y
```

From the published repository (no manual `git clone` needed — `npx skills add`
fetches the repo itself; append `#<branch>` to install from a specific branch):

```bash
npx skills add NVIDIA/<skills-repo> --skill '*' -a claude-code -a codex -y
```

Installation is git-based and does not depend on `pip install nvflare`; the
skills are not shipped inside the Python wheel. Pass every agent you use with
repeated `-a` flags. Omitting an agent skips installation for that agent; there
is no NVFLARE-specific installer command. The standard installer copies the
complete skill directory, including each co-located `evals/` directory. Those
files are evaluation metadata, not runtime guidance: `SKILL.md` remains the
instruction entry point, and repository tooling treats `evals/` separately
from the guidance it validates.
