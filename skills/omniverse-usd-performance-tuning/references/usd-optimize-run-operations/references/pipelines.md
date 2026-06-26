<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Pipeline Recipes - Upstream Handoff

This local reference preserves the digitaltwin workflow milestone. Scene
Optimizer mechanics for this step are owned by upstream `usd-optimize`.

- Public repository: [https://github.com/NVIDIA-Omniverse/usd-optimize/](https://github.com/NVIDIA-Omniverse/usd-optimize/)
- Package path: `.agents/operations/PIPELINES.md`
- Upstream web URL: [https://github.com/NVIDIA-Omniverse/usd-optimize/blob/main/.agents/operations/PIPELINES.md](https://github.com/NVIDIA-Omniverse/usd-optimize/blob/main/.agents/operations/PIPELINES.md)

Resolve the upstream guide without cloning the source repo:

1. `$USD_OPTIMIZE_ROOT/.agents/operations/PIPELINES.md`
2. `$USD_OPTIMIZE_ROOT/.agents/operations/PIPELINES.md`

If no package root is available, download and extract the prebuilt Usd Optimize release package (current asset name + download: `references/upstreams/usd-optimize.md`) (direct
archive URLs are in `references/upstreams/usd-optimize.md`), or use the package
path/URL supplied by the user. If the user supplies an extracted
package root directly, resolve this same package path under that root. If
GitHub raw fetch is available, the web URL above is acceptable for docs-only
reads. Do not clone the source repo just to read upstream SO guidance.

## Local Responsibilities

- Keep workflow phase order, prototype-first ordering, and broad optimization milestone ordering in `workflow.md`.
- Use `config-from-evidence.md` for local evidence-to-request routing and `operation-safety.md` for approvals.
- Use `batch-mode.md` for digitaltwin's scheduler-backed multi-target policy: adaptive concurrency, dependency-aware target groups, status artifacts, and resume prompts.

Named pipeline parameters and per-operation defaults belong upstream. If a
digitaltwin workflow needs to cite a chain, cite the upstream path and record
only the local evidence, target set, approval state, and report fields here.

## Local Routing Keys

The local workflow may route evidence to these operation keys before handing
mechanics to upstream: `computeExtents`, `decimateMeshes`,
`deduplicateGeometry`, `fitPrimitives`, `generateNormals`, `merge`,
`meshCleanup`, `optimizeMaterials`, `optimizeTimeSamples`, `pruneLeaves`,
`pythonScript`, `removeSmallGeometry`, and `removeUnusedUVs`. `merge`
(Merge Static Meshes) is the intent-gated within-prototype prim-count
consolidation step — see `workflow.md` Phase 4 and
`usd-structure-assessment/references/apply-restructure/references/hierarchy-dedupe-rewrite-tool-spec.md`
§9 for its op-chain and eligibility guard.
