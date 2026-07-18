<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Scheduler-Backed Batch Mode

This is Phase 4b of the canonical workflow. The deterministic mechanics are
scheduler-backed (`usd-optimize-run-operations/scripts/run_batch.py`); the agent still decides
concurrency, tags, and op chains. Optional helper wrappers accept one asset
path; the agent writes a batch plan and the scheduler runs the targets.

Do not serialize independent optimization targets by default. Run them in
adaptive batches sized by target weight and available system resources, then
adjust concurrency after each completed batch.

## Runner (resource-aware scheduler)

The deterministic mechanics of this pattern are owned by
`usd-optimize-run-operations/scripts/run_batch.py`. The split is: the **agent decides** (stays
prose) `max_workers` from the resource budget, the `archetype` tags, which op
chains, and the intent-gated opt-ins; the **runner owns** (deterministic tool)
spawning standalone single-asset workers, dependency ordering (prototype-first),
per-target/per-op timeouts, the GPU-cliff guard, status emission, and `--resume`.

- **The plan and status artifacts are both contracts.** The runner reads the
  batch plan (schema: `usd-optimize-run-operations/scripts/batch-plan.schema.json`) and writes
  `status.json` (schema: `usd-optimize-run-operations/scripts/status.schema.json`); CLI bars and any
  future dashboard are *views* over the status. The plan makes the descent →
  apply-restructure manifest → scheduler handoff a real contract — **one target =
  one own-layer file = one job** — and the manifest `phase4_targets[]` frontier
  metadata (`role`/`target_class`, `level`, `archetype`) flows into per-target
  jobs. Invoke with `--plan <batch-plan.json>` `--max-workers N` and optionally
  `--preflight <setup-preflight.json>` (CUDA signal) or
  `--cuda available|unavailable`.
- **`state: done` is NOT coverage `disposition: optimized`.** Worker completion
  proves the op ran, not that it changed anything. The runner records per-target
  before/after deltas (from each worker's `summary_path`) and derives
  `disposition` (`optimized` only on a real delta; `no_op` when unchanged;
  `unknown` when no summary). The Phase-4e `target_coverage.entries[]` /
  `coverage_ledger` are built from `disposition`, never from bare `done`.
- **Per-target/per-rule timeout:** every worker runs under
  `subprocess.run(timeout=...)`; a hung worker is killed (`state: timeout`, a
  distinct outcome from a worker that ran to a non-zero exit `state: failed`)
  without stalling the rest of the batch. Each target also records
  `duration_seconds` and `exit_code`. Scoped validator probes invoked through
  `usd_validation_executor.py` carry the same timeout contract.
- **GPU-cliff guard:** `gpu_bound` targets are skipped
  (`state: skipped_gpu_unavailable`) on a CPU-only/WSL host. CUDA is read from
  the setup-preflight `runtime_context`, **not** from SO's own `hasNvidiaGpu()`,
  and the signal fails closed (absent ⇒ treated as CPU-only) so a GPU op never
  silently enters a long CPU fallback.
- **`--resume` replaces the remainder script:** resuming off `status.json`
  re-runs only targets whose state is not terminal (`done` /
  `skipped_gpu_unavailable`).
- Workers are standalone single-asset processes (optimization/validation never
  uses Kit; standalone is the sole optimization runtime). The opt-in Kit→omniperf profiling path is capped to 1-2
  runs and sits outside this worker fan-out.

## Edit-target invariant (why per-target parallelizes)

Each target is **opened as its own root layer** so Usd Optimize's edit target
*is* that file's bytes. Never run SO on the composed assembly to optimize a
referenced library — the edits land as overrides on the assembly layer while the
library keeps its heavy geometry (override bloat, not reduction). One target =
one own-layer file = one job; this is precisely *why* per-target work
parallelizes. De-class abstract `class` prototype namespaces (`Class → Def`)
before the chain and restore after, and require every library file to resolve
standalone. See `apply-restructure/references/restructure-mode.md` § Edit-Target
Invariant.

## Targets

Targets come from:

- `apply-restructure` mode=`restructure`: prototype USDs, shared layers, and
  newly loadable sub-assets recorded in
  `<output_dir>/apply-restructure-manifest.json` `phase4_targets[]`, plus any
  `target_class: "assembly_root"` entry for mesh data retained in the assembly.
  Do not filter the manifest to prototype files only.
- Composed stages with no restructure: referenced sub-assets from
  `usd-structure-assessment` Phase 1.2 `assets.manifest`.
- Monolithic-as-is: the original stage (`N=1`).

## Adaptive Concurrency

Use target count only after estimating target weight. A fixed target-count cap
is too conservative for small mechanical parts and too aggressive for large
floor-scale facility sections.

Before the first batch, build a lightweight batch manifest:

- Independent target list, grouped by dependency class.
- Per-target weight signals: file size, mesh count, vertex/face count,
  material/texture count, prototype/instance count, and expected op-chain cost.
- Resource budget: CPU cores, available RAM, available VRAM when Kit/rendering
  is involved, free disk, and expected log/artifact volume.
- Initial concurrency choice and reason.

Initial concurrency guidance:

| Target class | Starting point |
|---|---|
| Monolithic target | `1` |
| Heavy facility/floor-scale target, multi-GB target, or high mesh/texture count | `1`, then increase only after a healthy pilot |
| Medium sub-assets | `2-4` depending on memory and disk headroom |
| Small mechanical parts or small fixture libraries | Start above `5` when resources allow; use CPU, memory, disk, and log headroom rather than the old fixed cap |
| Unknown weight | Start conservatively at `2`, or `1` if opening one target already consumes significant memory |

After each batch, inspect duration, failed targets, peak RAM/VRAM if available,
disk growth, log size, and output count. Increase concurrency when the pilot is
healthy and targets are small. Decrease concurrency or switch to serial when a
batch hits memory pressure, GPU pressure, disk/log pressure, runtime crashes,
or long-tail target variance.

If the remaining work is likely to exceed the user's time/resource budget, pause
and ask whether to continue, resume later from `status.json`, or stop. Do not
pause solely because target count exceeds five; pause because the observed budget
or risk says continuing automatically is unsafe.

## Prototype-First Ordering

When targets include prototypes and non-prototype assets, run prototypes first,
wait for completion, then run non-prototype assets. Parallelize within each
dependency group according to the adaptive concurrency policy. Prototype changes
propagate to instances, so running instance-site work first wastes time. Treat
an `assembly_root` target with retained meshes as a non-prototype mesh target:
run the evidence-selected per-target mesh op chain on it before final
assembled-root cleanup.

## Output Naming

Hash the absolute input path in every per-target output, summary, and log
filename. Basename-only naming is unsafe because many industrial scenes contain
repeated names such as `Body.usd` or `Default_V5.usd`.

Recommended pattern:

```text
<stem>.<sha1-absolute-path-prefix-12>.optimized.usdc
<stem>.<sha1-absolute-path-prefix-12>.summary.json
<stem>.<sha1-absolute-path-prefix-12>.log
```

After every batch, verify that the number of produced optimized files matches
the number of targets in that batch. If not, report a collision or failed write
instead of declaring success.

## Resume Prompt

When the adaptive budget says the remaining work should not continue
automatically, show:

- Already optimized targets (terminal in `status.json`).
- Deferred targets (still `queued`).
- Observed runtime/resource pressure from completed batches.
- The `status.json` path to resume from.
- Options: resume the remaining targets now (`run_batch.py --resume`), stop here,
  or explicitly optimize all remaining targets anyway.

Default behavior is to stop until the user chooses; the resource budget is the
guardrail. `--resume` re-runs only the non-terminal targets, so there is no
improvised remainder script to generate.

## Failure Handling

Aggregate per-target summaries into one batch summary. Surface failed targets
with log and summary paths. Do not auto-retry failed targets.

The final batch manifest should record every batch's target list, concurrency,
duration, output paths, summary/log paths, failures, resource observations, and
the reason for any concurrency adjustment.
