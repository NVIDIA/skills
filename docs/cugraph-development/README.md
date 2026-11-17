<!-- SPDX-License-Identifier: Apache-2.0 AND CC-BY-4.0 -->
<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# cuGraph skill development and upstream handoff

This draft stages the implementation at `skills/accelerated-computing-cugraph/` for review against NVIDIA/skills. It is not a completed catalog onboarding: the owning product team must select its public source repository, complete evaluation and signing, and register the source before this skill can be published through catalog sync.

## Repository requirements reviewed on 2026-11-17

NVIDIA/skills is a **catalog**, with skill content synchronized from product-owned source repositories. See root CONTRIBUTING.md, components.d/README.md, .github/PULL_REQUEST_TEMPLATE.md, and .github/scripts/verify_source_onboarding.py.

The implementation uses the required flat directory layout, name/description frontmatter, permitted licenses, a skill card, and an evaluation dataset. Its measured benchmark and signature are deliberately pending. Do not copy an old signature or mark a benchmark PASS without measurements.

Before upstream onboarding:

1. The owning team selects a **public NVIDIA-owned GitHub source repository** and clears the derived content and roadmap statements for open-source release. The internal reference and a personal fork are not eligible catalog sources.
2. Move this skill into that source repository's root `skills/` directory. Keep attribution and licenses. Confirm dependencies and licensing with the owning team; this draft does not attest approvals on their behalf.
3. Run CPU tests and Linux GPU validation: single-GPU semantics, 2-GPU direct and Dask fixtures, 2-node/4-GPU fixtures, empty partitions, and coordinated failure handling. Add a nonuniform weighted graph with dangling/isolated vertices and compare to CPU/SG oracles. Record the exact package lock, GPU/network hardware, commands, logs and tolerances.
4. Run the agent evaluations in `evals/evals.json` through the source team's current NVSkills pipeline. Update the skill card and measured BENCHMARK.md. Obtain a real `skill.oms.sig` covering the final files. The catalog gate requires a measured PASS verdict, eval data, card, and valid signature/content hashes.
5. Register `components.d/cugraph.yml` with the actual public source repository/ref/path and unique `catalog_dir: accelerated-computing-cugraph`. Ensure the component resolves to a supported `product.primary` value through an exact name match or explicit `product_primary`; propose a schema addition if needed. The catalog generates its README, marketplace metadata, versions, and plugin copies. Do not hand-edit generated copies or invent a source location now.
6. Sign off commits with DCO. For fork-based source PRs, follow CONTRIBUTING.md's current NVSkills workflow, service-account access, and maintainer `/nvskills-ci` trigger requirements. A catalog-only registration PR has different signing setup requirements.

This review draft is intentionally not added to components.d or catalog-exceptions.yml. An unregistered directory would be pruned by the catalog sync workflow, so it must not be merged as-is. Do not bypass that requirement with an unapproved catalog exception. Product source placement, registration, signing, benchmark PASS, and public-release clearance remain required before publication.

## Local validation

From the repository root:

```bash
python3 -m unittest discover -s docs/cugraph-development/tests -v
python3 -m compileall -q skills/accelerated-computing-cugraph/scripts
python3 skills/accelerated-computing-cugraph/scripts/pagerank_mg.py --help
python3 skills/accelerated-computing-cugraph/scripts/pagerank_dask.py --help
python3 skills/skill-card-generator/scripts/validate_submission.py \
  skills/accelerated-computing-cugraph/skill-card.md
```

The skill-creator `quick_validate.py` can also check frontmatter when available. The script checks are intentionally GPU-free; they do not validate CUDA, NCCL, cuGraph results, or distributed runtime behavior. Follow both distributed references for hardware checks.

## Design decisions

- Explicit cuGraph/pylibcugraph acceleration, no nx-cuGraph dispatch path.
- Kumo Engine recommended for GNNs; cuGraph-PyG scoped to distributed training and existing-stack maintenance.
- Dask shown as transitional, with a direct communicator-backed alternative.
- 26.10 source-verified APIs, including the optional-edge-ID MGGraph expensive-check workaround.
- Fresh examples and evaluations; no inherited benchmark success or signature.
