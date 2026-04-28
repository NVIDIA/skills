<!-- SPDX-License-Identifier: Apache-2.0 AND CC-BY-4.0 -->
<!-- Copyright (c) 2026 NVIDIA Corporation. All rights reserved. -->

# Contributing

This repository is a catalog of NVIDIA-verified agent skills. Skills are maintained by each product team in their own repos.

To contribute to a skill or propose a new one, use the contributing guidelines in the relevant source repo. See the [Available Skills](README.md#available-skills) and [Getting Help & Contributing](README.md#getting-help--contributing) sections in the README for links.

For changes to the catalog itself (fixing links, adding a new product listing), open a [pull request](../../pulls) or [issue](../../issues).

## Recommended Skill Directory Path

When publishing skills in your product repo, use one of these canonical paths so agents can discover them consistently:

- **`.agents/skills/`** — recommended default; agent-agnostic (matches the [agentskills.io](https://agentskills.io/specification) spec; recognized by Claude Code, Cursor, Codex, Windsurf, and other compatible agents).
- **`skills/`** at repo root — acceptable for OSS product repos where skills are a first-class product artifact.

Avoid agent-specific paths (`.claude/skills/`, `.codex/skills/`, `.cursor/skills/`) for new entries — they create duplication. Existing products on those paths can keep them; `components.yml` handles per-repo paths via the `skills[].path` field.

## IP Review and License (External Skills)

For skills published to `github.com/nvidia/skills`, NVIDIA contributors confirm three things per onboarding PR:

1. The skills have been cleared for open source release per NVIDIA's internal IP review process (six-question check).
2. The skill is licensed under Apache 2.0, CC-BY 4.0, or dual-license (Apache 2.0 + CC-BY 4.0).
3. No new license or new third-party component has been introduced beyond what the source repo already carries.

NVIDIA contributors: see the internal onboarding guide for the IP review process details and license selection. The [pull request template](.github/PULL_REQUEST_TEMPLATE.md) prompts for these affirmations on every onboarding PR.

## Signing Your Work

All pull requests require a DCO sign-off on every commit. This certifies that the contribution is your original work or you have rights to submit it under the same license.

```bash
git commit -s -m "Fix broken link"
```

This appends `Signed-off-by: Your Name <your@email.com>` to the commit. Unsigned commits will not be accepted.

If you forgot to sign off (existing commits without the trailer), retroactively sign all commits in your branch with:

```bash
git rebase --signoff origin/main && git push --force-with-lease
```

See the full [Developer Certificate of Origin](https://developercertificate.org/) for details.
