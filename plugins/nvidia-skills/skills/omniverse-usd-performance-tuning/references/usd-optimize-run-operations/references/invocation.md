<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Invocation Reference

How to execute Usd Optimize operations once the runtime is selected and the
operation plan is approved. Read `<output_path>/setup-preflight.json` to
determine which runtime and API surface to use.

This is the local source of truth for Usd Optimize operation invocation.
Other workflow docs should link here instead of repeating Python API snippets.

**Standalone is the sole runtime that executes operations.** Kit is not an
operation-execution runtime; it is retained only as an opt-in render-profiling
adjunct (Kit→omniperf, see `workflow.md` Phase 1a/6a) and never runs Scene
Optimizer operations. Do not bootstrap `KitApp` to execute ops.

## Standalone Runtime

When `setup-preflight.json` indicates standalone, invocation mechanics are
owned by the Usd Optimize package itself. Resolve the upstream guide:

1. `$USD_OPTIMIZE_ROOT/.agents/operations/INVOCATION.md`
2. `$USD_OPTIMIZE_ROOT/.agents/operations/INVOCATION.md`

If no package root is available, download and extract the published
the prebuilt Usd Optimize release package (asset name + download in
`references/upstreams/usd-optimize.md`), or use the package path/URL supplied
by the user.

**Local responsibilities still apply:**

- Cross-check every operation key against `operationsAvailable` in
  `setup-preflight.json` before execution. If missing, report
  `blocked_missing_usd_optimize_operation`.
- Apply destructive-operation approval gates via `operation-safety.md`.
- Write optimized stages and runtime artifacts under the local output
  workspace chosen by setup.

## Verified Python API Shapes

Verified against the `usd_optimize_...@1.0.4` GitHub release asset
(2026-06-11; current asset resolution: `references/upstreams/usd-optimize.md`).
Import from `usd_optimize.core`; the `omni.scene.optimizer.core` module path
and the `SceneOptimizerCore` class name survive upstream only as deprecated
aliases and must not appear in new configs.

Direct API with per-operation results (1.0.4-verified shape):

```python
from usd_optimize.core import ExecutionContext, UsdOptimizeCore
from pxr import Usd

stage = Usd.Stage.Open(input_path)
context = ExecutionContext()
context.set_stage(stage)
results = UsdOptimizeCore.getInstance().executeConfig(context, [
    {"operation": "meshCleanup", "mergeVertices": True},
])
for success, error, output in results:
    if not success:
        raise RuntimeError(error)
stage.Export(output_path)
```

## Invalid Call Shape

Do not pass a plain `pxr.Usd.Stage` directly as the second argument to
`SceneOptimizerCore.executeOperation` or `executeConfig`. The binding expects an
`ExecutionContext`; the stage must be attached with `context.set_stage(stage)`.
The bad shape below reproduces the failure seen in Horde testing:

```python
SceneOptimizerCore.getInstance().executeOperation("printStats", stage, {})
# AttributeError: 'Stage' object has no attribute '_impl'
```

If `_impl` appears in an operation log, stop the operation pass, mark the
attempt as an invalid SO invocation, and rerun through the supported shapes
above. Do not export or report a successful optimized stage from that failed
pass.

## Save Policy

- Export optimized output to a NEW `.usdc` path under `<output_path>/`.
  Never overwrite the source stage.
- Use `stage.Export(path)` for clean output. Use `Sdf.Layer.Export()` only
  for individual layer cleanup (Phase 4.5).
- Use in-place `Save()` only for newly created layers or explicitly
  user-approved source edits.
- Do not flatten unless the user asks for a flattened deliverable.

## Per-Operation Parameters

Per-operation parameter tables, defaults, and implementation caveats are owned
by upstream `usd-optimize`. The same package paths listed in the standalone
section above contain the full operation reference. If GitHub raw fetch is
available, the web URL below is acceptable for docs-only reads:

- [https://github.com/NVIDIA-Omniverse/usd-optimize/blob/main/.agents/operations/INVOCATION.md](https://github.com/NVIDIA-Omniverse/usd-optimize/blob/main/.agents/operations/INVOCATION.md)

Do not clone the source repo just to read docs.
