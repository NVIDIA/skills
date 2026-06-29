<!-- SPDX-License-Identifier: Apache-2.0 AND CC-BY-4.0 -->
<!-- Copyright (c) 2026 NVIDIA Corporation. All rights reserved. -->

# Contributing

This repository is a catalog of NVIDIA-verified agent skills. Skills are maintained by each product team in their own repos.

To contribute to a skill or propose a new one, use the contributing guidelines in the relevant source repo. See the [Available Skills](README.md#available-skills) and [Getting Help & Contributing](README.md#getting-help--contributing) sections in the README for links.

For changes to the catalog itself (fixing links, adding a new product listing), open a [pull request](../../pulls). For catalog-level bug reports, feature proposals, or documentation problems, file an [issue](../../issues/new/choose) using one of the catalog issue templates (Bug Report, Feature Request, Documentation Request or Correction). Questions and general discussion belong in [Discussions](../../discussions); security vulnerabilities follow the disclosure process in [SECURITY.md](SECURITY.md).

## Recommended Skill Directory Path

When publishing skills in a product repo, keep the source of truth in `skills/` at the repo root. This is the recommended default for product-owned or OSS skills, where skills are a first-class artifact.

Use `.agents/skills/` for installed skills that agents discover at runtime, or for repos that are intentionally structured as agent-readable skills packs.

Avoid agent-specific paths in the repo (`.claude/skills/`, `.codex/skills/`, `.cursor/skills/`) for new entries — they create duplication. Existing products on those paths can keep them; `components.d/<slug>.yml` handles per-repo paths via the `skills[].path` field.

At install time, your tooling or packaging can copy or symlink from `skills/` into the appropriate agent discovery locations (for example `.agents/skills/`, `.claude/skills/`, `.codex/skills/`) as required by each tool.

## IP Review and License (External Skills)

For skills published to `github.com/nvidia/skills`, NVIDIA contributors confirm three things per onboarding PR:

1. The skills have been cleared for open source release per NVIDIA's internal IP review process (six-question check).
2. The skill is licensed under Apache 2.0, CC-BY 4.0, or dual-license (Apache 2.0 + CC-BY 4.0).
3. No new license or new third-party component has been introduced beyond what the source repo already carries.

NVIDIA contributors: see the internal onboarding guide for the IP review process details and license selection. The [pull request template](.github/PULL_REQUEST_TEMPLATE.md) prompts for these affirmations on every onboarding PR.

## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

  ```bash
  git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* If you forgot to sign off on earlier commits, retroactively sign all commits in your branch with:

  ```bash
  git rebase --signoff origin/main && git push --force-with-lease
  ```

* Full text of the DCO (https://developercertificate.org/):

  ```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.


  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
  ```

