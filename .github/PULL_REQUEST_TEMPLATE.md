<!-- SPDX-License-Identifier: Apache-2.0 AND CC-BY-4.0 -->
<!-- Copyright (c) 2026 NVIDIA Corporation. All rights reserved. -->

## Onboarding type

- [ ] New product onboarding (`components.yml` entry)
- [ ] Other (catalog change, README fix, infrastructure, etc.)

## For new product onboarding — author affirmations

By submitting this PR, I confirm on behalf of my team:

- [ ] **Skills cleared for open source release** per NVIDIA's internal IP review process (six-question check, all answers affirmative)
- [ ] **License selected:** Apache 2.0 / CC-BY 4.0 / Dual (Apache 2.0 + CC-BY 4.0). Specify: _____
- [ ] **No new license or new third-party component** introduced beyond what the source repo already carries
- [ ] **Source repo is public and under an NVIDIA-owned GitHub org**
- [ ] `.agents/skills/` or `skills/` path used for new entries (or existing path retained for legacy entries per `components.yml`)

> NVIDIA contributors: see the internal onboarding guide for the IP review process details and license selection.

## Reviewer checklist (OSS Skills PIC)

- [ ] Author confirmations above are checked
- [ ] `components.yml` entry valid (required fields, unique `catalog_dir`, path exists in source repo)
- [ ] `SKILL.md` frontmatter spec-compliant (at least one sampled)
- [ ] No new license or third-party dependency requiring OSRB filing

## All PRs

- [ ] All commits signed off with DCO (`git commit -s`).
      If you forgot, run `git rebase --signoff origin/main && git push --force-with-lease` to retroactively sign all commits in your branch.

## Other context (for non-onboarding PRs)

<!-- Brief description of the change -->
