<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# usd-optimize-run-validators - Local Validation Policy and Upstream Handoff

This local reference preserves the digitaltwin workflow milestone. Scene
Optimizer mechanics for this step are owned by upstream `usd-optimize` and the
prebuilt Usd Optimize package.

## When to Use

Use when the digitaltwin workflow reaches the `usd-optimize-run-validators` milestone or
when a user directly asks to run Usd Optimize validators on a USD asset.

## Instructions

1. If this is the entry reference, run the local runtime gate and consume
   `<output_path>/setup-preflight.json` before validation.
2. Apply `usd-validation-runner/README.md` selected-scope policy, deferred-validator policy, and explicit
   approval for expensive checks.
3. Apply `runtime-artifact-token-budget.md`; never read full validator CSVs or
   full `run.log` into context.
4. Resolve the upstream validator runner from an extracted package root before
   using web docs. Do not clone the source repo just to read Usd Optimize validator
   guidance.
5. Preserve logical milestone name `usd-optimize-run-validators` and pass artifacts to
   `usd-optimize-interpret-validators`.

## Output Format

Return a concise status or report that names the input asset, selected runtime,
artifacts written, blockers, and the next interpretation step.

## Upstream Source

- Public repository: [https://github.com/NVIDIA-Omniverse/usd-optimize/](https://github.com/NVIDIA-Omniverse/usd-optimize/)
- Package path: `.agents/skills/run-validators/SKILL.md`
- Upstream web URL: [https://github.com/NVIDIA-Omniverse/usd-optimize/blob/main/.agents/skills/run-validators/SKILL.md](https://github.com/NVIDIA-Omniverse/usd-optimize/blob/main/.agents/skills/run-validators/SKILL.md)

Resolve the upstream guide without cloning the source repo:

1. `$USD_OPTIMIZE_ROOT/.agents/skills/run-validators/SKILL.md`
2. `$USD_OPTIMIZE_ROOT/.agents/skills/run-validators/SKILL.md`

If no package root is available, download and extract the prebuilt Usd Optimize release package (current asset name + download: `references/upstreams/usd-optimize.md`), or use
the package archive path, direct archive URL, or extracted package root supplied
by the user. Current public direct archive URLs are listed in
`references/upstreams/usd-optimize.md`. If the user supplies an extracted
package root directly, resolve this same package path under that root. If
GitHub raw fetch is available, the web URL above is acceptable for docs-only
reads. Do not clone the source repo just to read upstream SO guidance.

## Local Responsibilities

- Runtime context gate and `setup-preflight.json` consumption.
- `operationsAvailable` and runtime-family awareness from setup.
- Validation scoping, selected validators, masked-stage spot-check policy, and
  expensive-check approval gates.
- Runtime artifact token budget for CSV/log/summary handling.
- Digitaltwin milestone routing into `usd-optimize-interpret-validators`.
