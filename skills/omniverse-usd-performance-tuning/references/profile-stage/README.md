# Profile Stage

## When to Use

Use when profiling a USD stage before/after optimization; do not use to interpret regressions alone.

## Instructions

See `references/_shared/standard-instructions.md`.

## Pre-flight Checklist

Before running profile measurements, re-read and confirm:

- [ ] `references/runtime-artifact-token-budget.md` — keep raw profile output
  on disk, read bounded summaries only.
- [ ] Output workspace policy from `references/output-workspace.md`.
- [ ] Profiling mode (quick vs full) matches what was used for baseline —
  never compare across modes.
- [ ] For full mode: multi-sample warm protocol (discard first, average rest).
## Output Format

See `references/_shared/standard-output-format.md`.

Use this reference to capture measurable performance data. Run it **before**
optimization to establish a baseline, and **after** to verify improvement.

## Purpose

Capture repeatable quick or full performance metrics for a USD stage so
optimization decisions and before/after comparisons are evidence-based.

## Runtime artifact token budget

Follow
`skills/omniverse-usd-performance-tuning/references/runtime-artifact-token-budget.md`
for Kit logs, Tracy captures, and CSV exports. Do not load full `.tracy` files,
Tracy CSVs, or Kit logs into context. Extract compact metrics and keep the raw
captures on disk.

## Prerequisites

- A readable USD stage path.
- `pxr` Python API for quick mode.
- Kit, Isaac Sim, or compatible runtime plus Tracy support for full mode.
- Same profiling mode and environment for any baseline/after comparison.

## Examples

- "Profile this USD stage in quick mode before optimization."
- "Capture a full Kit runtime profile after mesh cleanup."

## Quick Mode (USD-level, always available)

Requires only the `pxr` Python API. No Kit, no GPU needed. Measures:

- **Stage open time** (cold + warm) — composition cost.
- **Prim traversal time** — scene graph complexity.
- **Attribute resolution time** — value resolution across composition arcs.
- **Transform computation time** — XformCache world transforms.
- **Material binding resolution time** — ComputeBoundMaterial cost.

### Usage

```python
from pxr import Usd, UsdGeom, UsdShade, UsdUtils
from statistics import median
import gc
import time

stage_path = "/path/to/stage.usd"

def open_once_ms(path):
    t0 = time.perf_counter()
    stage = Usd.Stage.Open(path)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    del stage
    gc.collect()
    return elapsed_ms

# Stage-open timing. Prefer running this script in a fresh process for each
# baseline/after capture. Treat cold_open_ms as the first measured open in this
# capture process, not a guaranteed OS-cold read.
cold_open_ms = open_once_ms(stage_path)
_warmup_open_ms = open_once_ms(stage_path)
warm_open_samples_ms = [open_once_ms(stage_path) for _ in range(5)]
warm_open_ms = median(warm_open_samples_ms)
warm_open_spread_pct = (
    (max(warm_open_samples_ms) - min(warm_open_samples_ms)) / warm_open_ms * 100
    if warm_open_ms
    else 0.0
)

stage = Usd.Stage.Open(stage_path)

# Traversal. `Usd.Stage.Traverse()` walks the pseudo-root's namespace, which
# contains the authored scene graph and nothing else. Prototype prims
# (/__Prototype_*) are not in that namespace — reach them through
# `stage.GetPrototypes()`, and reach the geometry they stand in for through
# `Usd.TraverseInstanceProxies()`.
#
# Keep the timed loop free of per-prim Python work. Anything you put inside it
# is charged to the metric and scales with prim count, so it costs the larger
# capture more than the smaller one and invents an improvement.
all_prims = list(stage.Traverse())

t0 = time.perf_counter()
for _ in range(10):
    list(stage.Traverse())
traverse_ms = (time.perf_counter() - t0) * 1000 / 10

# Instance-proxy traversal (only meaningful when instance_count > 0).
all_prims_with_proxies = list(stage.Traverse(Usd.TraverseInstanceProxies()))

# Attribute resolution
t0 = time.perf_counter()
for prim in all_prims:
    for attr in prim.GetAttributes():
        attr.Get()
resolve_ms = (time.perf_counter() - t0) * 1000

# Transform computation
xf_cache = UsdGeom.XformCache()
xformable = [p for p in all_prims if p.IsA(UsdGeom.Xformable)]
t0 = time.perf_counter()
for prim in xformable:
    xf_cache.GetLocalToWorldTransform(prim)
xform_ms = (time.perf_counter() - t0) * 1000

# Stage stats
stats = UsdUtils.ComputeUsdStageStats(stage)
```

### Quick mode output

Real capture from an instanced 23.7k-prim stage (USD 0.26.5):

```json
{
  "mode": "quick",
  "stage_path": "/path/to/stage.usdc",
  "cold_open_ms": 1013.2,
  "warm_open_ms": 854.1,
  "warm_open_samples_ms": [853.7, 854.6, 845.9, 854.1, 865.2],
  "warm_open_sample_count": 5,
  "warm_open_spread_pct": 2.3,
  "open_timing_context": "fresh_process",
  "traverse_ms": 42.52,
  "attribute_resolution_ms": 867.0,
  "transform_ms": 106.6,
  "prim_count": 23704,
  "prim_count_with_instance_proxies": 385703,
  "layer_count": 1,
  "instance_count": 104023,
  "prototype_count": 5225,
  "total_attributes": 149949
}
```

`traverse_ms` is the mean of ten `list(stage.Traverse())` passes and is the one
traversal metric. Use it for before/after comparison.

There is no separate "authored" traversal metric because there is nothing to
separate. `stage.Traverse()` returns authored prims only; it never yields a
`/__Prototype_*` path, whatever `prototype_count` says.

`prim_count` is how many prims `stage.Traverse()` yields — the authored scene
graph. `prim_count_with_instance_proxies` counts instance proxies as well and
is the rendered-geometry footprint (what Hydra walks); it is much larger once
`instance_count > 0`, as in the capture above. `prototype_count`, from
`UsdUtils.ComputeUsdStageStats`, covers the prototype side. Report all three so
the optimization-report can attribute a regression to the right axis.

### Retired metrics

An earlier version of this recipe reported `traverse_ms` from a loop that
filtered `/__Prototype_*` paths out of `stage.Traverse()`, plus an unfiltered
`traverse_full_ms`, plus a `prim_count_authored` alongside `prim_count`. The
filter was dead code: it never matched a prim, so `prim_count_authored` always
equalled `prim_count` and the only difference between the two timings was the
per-prim `str()` and `startswith()` the filter ran inside the timed loop. That
cost grows with prim count, so it inflated a large baseline far more than a
small optimized stage and reported improvement that was the filter getting
cheaper.

`traverse_full_ms` and `prim_count_authored` are gone. Profiles captured with
the old recipe cannot be compared against profiles captured with this one —
recapture the baseline. If you are re-reading an old profile, its
`traverse_full_ms` is the honest number and its `traverse_ms` is not.

### Stage-open Timing Protocol

Use this protocol for `cold_open_ms` and `warm_open_ms`; do not treat a single
post-optimization warm open as a verdict.

- Prefer a fresh process for each baseline and after capture. If the capture
  must run inside the same long-running process that performed optimization,
  set `open_timing_context` to `same_process_warm` and lower confidence.
- For each stage path, record one first-open timing, run one unreported warmup
  open, then measure at least five warm opens. Set `warm_open_ms` to the
  median and include `warm_open_samples_ms`, `warm_open_sample_count`, and
  `warm_open_spread_pct` when possible.
- If the optimized file was just written, run the same warmup/sample protocol
  before comparing it to the baseline. Do not compare a first after-write open
  to a warmed baseline.
- If warm samples are noisy (for example, max-min exceeds 15% of median) or the
  before/after delta is within the measured spread, the warm-load evidence is
  inconclusive: in `compare-profiles` classify that row's verdict as `neutral`
  (the verdict enum has no `inconclusive` value) and record the inconclusive
  timing context in the notes, rather than reporting a regression.

## Full Mode (Kit runtime, requires Isaac Sim + GPU)

Captures actual rendering performance via Tracy. Measures everything in
quick mode plus:

- **FPS** (steady-state frame rate).
- **Frame time** (mean, p50, p95, min, max).
- **Hydra sync time** — USD → Hydra scene population.
- **RTX render time** — GPU rendering passes.
- **Shader compilation time** — first-run shader cache cost.
- **Stage load event timing** — from Kit's internal instrumentation.

### Prerequisites

- Isaac Sim or Kit SDK with RTX renderer.
- Kit `omni.kit.profiler.tracy` profiler extension (Tracy is a Kit profiler, not a Usd Optimize component).
- GPU with display (headless with virtual display works).

### Usage

Launch Isaac Sim with Tracy profiler:

```python
from isaacsim import SimulationApp
app = SimulationApp({
    'headless': True,
    'extra_args': [
        '--/app/profilerBackend=tracy',
        '--/app/profileFromStart=true',
        '--/profiler/enabled=true',
        '--/profiler/gpu=true',
        '--/profiler/gpu/tracyInject/enabled=true',
        '--/app/profilerMask=1',
        '--enable', 'omni.kit.profiler.tracy',
    ]
})
```

Capture the trace with the Tracy `capture` binary bundled in
`omni.kit.profiler.tracy` extension. Export with `csvexport`.

Treat Tracy CSV exports as large artifacts: run an analyzer that emits compact
startup/runtime summaries, or read only bounded heads/tails and targeted zone
matches. Never paste the full CSV into the report.

For detailed capture procedure and analysis, refer to the external
profiling skills at `NVIDIA/omniperf/.agents/skills/profiling/SKILL.md`
and `NVIDIA/omniperf/.agents/skills/nsys-analyze/SKILL.md`.

### Full mode output

```json
{
  "mode": "full",
  "stage_path": "/path/to/stage.usd",
  "quick_metrics": { "...same as quick mode..." },
  "kit_metrics": {
    "fps_mean": 43.2,
    "frame_time_mean_ms": 23.1,
    "frame_time_p95_ms": 25.8,
    "hydra_sync_ms": 4.4,
    "rtx_render_ms": 3.1,
    "stage_load_ms": 580,
    "shader_compile_ms": 8200,
    "tracy_zone_count": 101707,
    "trace_file": "/path/to/trace.tracy"
  }
}
```

## Full mode: startup vs runtime separation

When capturing Tracy data, separate the zone report into two sections:

- **Startup zones** — count=1 or proportional to extension/device count.
  Report total startup time.
- **Runtime zones** — count matches frame count. Report per-frame averages.

Classification: if zone count is within ±10% of the rendered frame count,
it is a runtime zone. Otherwise it is startup.

Output should include:

```json
{
  "startup_zones": [
    {"name": "compileShaderGroupForDevice", "total_ms": 6998, "count": 178}
  ],
  "runtime_zones": [
    {"name": "App Update", "mean_ms": 15.7, "count": 139},
    {"name": "hydraRenderViews", "mean_ms": 9.6, "count": 104}
  ],
  "startup_total_ms": 25646,
  "runtime_mean_frame_ms": 15.7
}
```

This separation enables `compare-profiles` to correctly classify tradeoffs
(startup cost increase + runtime improvement = net positive, not a regression).

## When to use which mode

- **Quick mode** for structural optimization (instancing, layer packaging,
  reference remapping). Measures composition cost which is what these changes affect.
- **Full mode** for geometry optimization (mesh cleanup, decimation, material
  consolidation). Measures rendering cost which is what these changes affect.
- **Always run the same mode before and after** for a valid comparison.

## What quick mode can and cannot prove (standalone-path caveat)

Quick mode is the only available mode when the Phase 0 runtime is
standalone Usd Optimize (no Kit). The agent must be explicit in the
final `optimization-report` about which claims quick-mode metrics support
and which they do not.

**Quick mode CAN prove:**

- Stage open time (cold + warm) — composition + I/O cost.
- Prim / layer / instance / prototype counts — structural complexity.
- Attribute resolution + transform compute — composition-arc evaluation cost.
- Aggregate disk-size deltas on prototype / sub-asset files (compared
  separately, not part of quick mode itself).

**Quick mode CANNOT prove:**

- Steady-state FPS or frame time (no renderer).
- VRAM footprint (no GPU allocator).
- Hydra sync / RTX render / shader compile costs.
- Real draw-call count under the renderer (SO analysis-mode
  `rtxMeshCount` reports a count, but the renderer's actual draw-call
  count depends on Hydra batching, instance promotion, and material
  switch grouping that only the runtime sees).

When this reference ran in quick mode only, **the report's `verdict` should
explicitly note that render-time claims (FPS, frame time, VRAM, draw-call
count) are unmeasured**. Improvements predicted by `rtxMeshCount` or
prototype sharing are plausible but not verified. See
`skills/omniverse-usd-performance-tuning/references/optimization-report/references/optimization-report-template.md` §"Structural-only path (SO
unavailable)" and §"Quick-mode-only caveat" for the report wording.

## Rules

- Use the Stage-open Timing Protocol above for `cold_open_ms` and
  `warm_open_ms`.
- Do not compare quick mode baseline to full mode post-optimization (or vice versa).
- Store profile results as JSON for the compare-profiles skill.
## Limitations

- Quick mode measures USD-level structure and composition, not rendered FPS.
- Full mode requires a compatible Kit runtime, GPU/display setup, and Tracy capture tooling.
- A single profile cannot determine improvement; compare matching baseline and after results.

## Troubleshooting

- If `pxr` imports fail, run `setup-usd-performance-tuning` to resolve the
  standalone USD Python runtime; it owns the probe and writes
  `setup-preflight.json`. Nothing is being chosen here — standalone is the
  runtime for all optimization and validation work, and the Kit adjunct is only
  relevant to full mode below.
- If full mode cannot load Tracy, verify `omni.kit.profiler.tracy` is enabled in the selected Kit runtime.
- If warm-open samples vary widely, rerun the protocol in fresh processes; if
  variance persists, treat warm-load evidence as inconclusive (verdict row
  `neutral` + inconclusive context in notes).
