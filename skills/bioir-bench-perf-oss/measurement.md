---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
{}
---

# Measurement protocol

The shared helpers ship in [`harness/`](harness/). Copy them into
`$WORKDIR/bench/` and import them from both harnesses; do not invent a
different timer, scorer, or aggregate on one side of the comparison.

- [`timing.py`](harness/timing.py) — `time_model_forward`, `residues_all_chains`
- [`gpu_telemetry.py`](harness/gpu_telemetry.py) — `GpuSampler`, file-level
  aggregate
- [`scoring.py`](harness/scoring.py) — `score_lddt`, `score_dockq`,
  `dockq_band`
- [`check_results.py`](harness/check_results.py) — the result-JSON contract
- [`summarize_speedup.py`](harness/summarize_speedup.py) — `speedup.json`
- [`plot_latency.py`](harness/plot_latency.py) — both Phase 6 charts

`tests/skills/` runs them on CPU in CI. The model-specific staging and runner
scripts stay per bench, because each depends on its pinned OSS tree.

## Dataset and MSAs

The only dataset is the one `rebuild_dataset.py` builds
([samples.md](samples.md#build)). Local root: `benchmarks/dataset/`. See
[samples.md](samples.md) for fetch, specs, and the catalog.

- Convert each in-scope spec item to an `InputRequest` (BioIR on the
  processor path) and to the OSS script input (OSS side and feature
  dumps). Resolve paths against `$DATASET_ROOT`. Do not call
  `load_requests` on `spec_*.json`.
- Attach every unpaired and paired A3M the spec lists (and Boltz CSV
  when present). A protein forward without those files is invalid even
  if it is faster.
- Attach every template the spec lists, on **both** sides or neither
  ([samples.md](samples.md#templates-are-in)). Do not use
  `examples/data/samples/` or FASTA-only inputs.
- Progress bars, logging, and feature assertions stay **outside** the
  timing window — in the sample loop, never between the CUDA syncs
  ([progress reporting](workflow.md#progress-reporting--every-harness-shows-a-bar)).
- Record resolved MSA paths on every result row
  (`unpaired_msas`, `paired_msas`) so a dropped alignment is visible,
  plus `templates` and `template_status` so a dropped template is
  visible too.

## Timing window

Match `FoldingEngine.execute` when `profile_inference=True`
(`bionemo_ir/pipeline/engine.py`):

1. Build or load the feature batch (untimed).
1. Move the batch to the GPU (untimed).
1. `torch.cuda.synchronize()`.
1. `t0 = time.perf_counter()`.
1. `output = model(device_batch, **runtime_args)` under
   `torch.inference_mode()`.
1. `torch.cuda.synchronize()`.
1. `elapsed_s = time.perf_counter() - t0`.

Outside the window: parse, tokenize, featurize, H2D, postprocess, write,
`stage_timing_s`, `time_taken`, and processor wall time. Record those as
diagnostics only.

`time_model_forward` in [`harness/timing.py`](harness/timing.py) is this window,
and also returns peak allocated and reserved GiB.

**Processor path.** Do not re-wrap `model.forward`. Set
`engine_kwargs={"profile_inference": True}` and read
`row["model_inference_time"]`. That field is already this window.
Still call `torch.cuda.reset_peak_memory_stats()` immediately before
`processor([row])` and read peak memory after it returns — the engine
does not record GPU bytes.

**OSS feature path** (a folding model with Pipeline = No; no current key). There
is no engine. Call `time_model_forward` on the BioIR module after
adapting the dumped OSS batch ([no-pipeline.md](no-pipeline.md)).

### Allocator flags — leave PyTorch on its default

Never set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, on
either side, in any scenario. Expandable segments trade throughput
for a lower reserved ceiling: the allocator grows segments by
remapping virtual address ranges, which adds driver work on the
hot path and can slow `forward` measurably. A latency benchmark
has to exercise the allocator the way a normal deployment does.

Run both columns on the allocator default and record no
`PYTORCH_CUDA_ALLOC_CONF` in `bench_config.json`. If a sample only
fits with a non-default allocator, that is an out-of-memory result
to report ([Per-sample GPU fields](#per-sample-gpu-fields)), not a
flag to switch on — and never a flag on one side only.

Memory-reduction work is a different task: `scan-mem-opt-patterns`
sets this flag deliberately, to separate fragmentation from live
bytes. Keep it out of timed runs.

## BioIR pretrained config

The default constructor is the optimized path. Every folding
class does:

```python
self.config = config or self.get_pretrained_config(self.model_name)
```

So `ModelCls(model_name=..., config=None)` (omit `config=`) is
`get_pretrained_config`. Processor path: omit `engine_kwargs["config"]`
so `FoldingEngine` constructs the same way. OSS feature path: omit
`config=` on the module constructor.

Do **not**:

- Build a `BaseConfig` by hand
- Copy OSS dtypes or attention backends onto BioIR
- Pass a config that changes fields beyond the documented model-profile
  overrides or an eager arm's `disable_cuda_graphs()`

For a profile's input-parity, recycle-lock, or approximate-option override,
derive the official pretrained config, change only the documented fields,
pass it through `engine_kwargs["config"]` (or module `config=`), and record
the delta in `bench_config.json`. Approximate options also require the paired
exact arm and fidelity report in [SKILL.md](SKILL.md#approximate-options).

That default config carries the CUDA-graph policies ([below](#cuda-graphs)).
An eager arm's `disable_cuda_graphs()` is the one permitted graph change.

## CUDA graphs

The default BioIR model captures every graph region up to its token limit:
1024, or 2048 for a diffusion region on a device with at least 64 GiB. It
needs no `optimize()` call. `docs/ref/api.md` (CUDA Graphs) lists each model's
regions and policy.

- The BioIR column uses the default model and locks
  `graph_activation="default"`.
- An eager arm derives the pretrained config, calls
  `config.disable_cuda_graphs()`, and passes the config as
  `engine_kwargs["config"]` (processor path) or `config=` (OSS feature
  path). It locks `graph_activation="disable_cuda_graphs()"`.
- Keep the model's policies. Do not replace one through
  `accelerated_configs`, call `with_graph_cache`, or raise a region's token
  limit for coverage. Inputs above the limit run eagerly; record them per
  row.
- Keep graph caches between samples. Do not reset trackers or reload
  weights to clean up; a new shape captures during its own warmup forward.

The audit reads the live model. On the processor path, get it before the
first request:

```python
model = processor.get_stage_udf("FoldingEngineStage").folding.engine.model
```

### Audit CUDA-graph routing, not cache emptiness

A cached graph does not show that the current call replayed: earlier
shapes stay resident while an out-of-range input runs eagerly. Snapshot
each region's `execution_counts` before and after every forward:

```python
from collections import Counter

from bionemo_ir._torch.graph_optimization import GraphRegion

GRAPH_COUNTERS = ("warmup", "capture", "replay", "eviction", "fallback", "out_of_range")


def graph_counts(model) -> dict[str, Counter]:
    """Every graph region's execution counters, keyed by module path."""
    return {
        path: Counter(region.tracker.execution_counts if region.tracker is not None else {})
        for path, region in model.named_modules()
        if isinstance(region, GraphRegion)
    }


def graph_audit(before: dict[str, Counter], after: dict[str, Counter]) -> dict[str, dict]:
    """Per-region counter deltas of one forward and its execution path."""
    audit = {}
    for path, counts in after.items():
        delta = {name: counts[name] - before[path][name] for name in GRAPH_COUNTERS}
        if delta["out_of_range"]:
            execution_path = "eager_out_of_range"
        elif delta["fallback"]:
            execution_path = "eager_capture_fallback"
        elif delta["replay"]:
            execution_path = "cuda_graph"
        else:
            execution_path = "eager"
        audit[path] = {**delta, "execution_path": execution_path}
    return audit
```

Store the warmup and measured audits on every row. `eager` means the
region did not run or bypassed its cache: it is not enabled, or it is the
confidence Pairformer of a single-sample request. A region that captured
during the warmup forward must replay on the measured forward with zero
`warmup` and `capture`; a capture there is a recapture, so record it.

## Compare BioIR graph policies

A comparison of BioIR graph policies or revisions is BioIR against BioIR,
not an OSS speedup. Keep the dataset, features, locked knobs, and GPU fixed,
and run each arm in its own process:

- `eager` — `config.disable_cuda_graphs()` before construction;
- a role subset — set the other regions' `graph_optimization_config` to
  `None` before construction;
- `default` — the default model.

For two revisions, give each process an explicit source root and record its
commit plus source, native-library, checkpoint, and kernel-pack hashes.
Record each sample's cold first forward separately from its warm repeat.
Then run one forward per request over a mixed-size stream with caches
retained, so the result includes cache reuse, eviction, recapture, and
eager inputs above the token limit. Compare raw output tensors as well as
written-CIF quality.

## OSS `torch.compile`

Two OSS scenarios. **Eager is required.** Compile is a second column
only when a probe shows it works.

**Compile the target submodules once. Run that compiled model on
every in-scope sample.** Do not call `torch.compile` per sample.
Do not compile the whole `nn.Module`.

```text
eager OSS model  ──►  full manifest, one sample at a time  ──►  oss_eager.json

fresh OSS model
    │
    ├─ compile_oss_hot_modules(...)   ONCE (targets only)
    ├─ synthetic direct-module probe  (fast rejection filter)
    ├─ real two-sample integration probe
    └─ SAME compiled model ──► every in-scope sample ──► oss_compile.json
```

1. Run the full in-scope manifest **eager** (one sample per
   forward) → `oss_eager.json`. Keep this model instance eager.
1. Construct a **fresh** OSS model. Call
   `compile_oss_hot_modules` **once** on it.
1. Feed deterministic synthetic tensors **directly to each compiled
   child** at two dynamic sizes. This is the fast compiler probe below;
   it avoids repeating full featurization, rollout, and writing while
   walking the retry ladder.
1. After the synthetic probe passes, run one real integration probe on
   the smallest and same-bin second sample (warmup 1 + measure 1). See
   pass rules below.
1. If the probe passes, **reuse that same compiled model** and
   loop every in-scope sample (still one batch at a time) →
   `oss_compile.json`. Do not compile again. Do not build a new
   compiled model per row. **Count Dynamo compiles on every
   warmup and every measure** ([below](#track-recompiles-on-warmup)).
1. If the probe fails, **do not drop compile yet.** Walk the
   [retry ladder](#compile-retry-ladder) on a **fresh** model
   each try. Mark the real semantic dynamic dimensions explicitly and
   classify every other direct input dimension as static. Only after
   those attempts fail, skip the compile column. Eager still counts.
   Record every attempt in `implementation-notes.md`.

### Fast synthetic compile probe

Use synthetic data to validate `torch.compile` targets before spending
minutes on an end-to-end sample. The probe calls the selected compiled
modules directly; it does **not** call the top-level model, data
pipeline, diffusion rollout, writer, or scorers.

Each model profile provides a
`make_synthetic_compile_inputs(role, shape, module)` adapter. It must:

- derive fixed channel widths, dtypes, device, and required flags from
  the actual module/config—never hard-code guessed widths
- return the target's real positional/keyword input structure,
  including a shape-consistent tensor dictionary when the child takes
  one
- invoke the child under its production precision context, including an
  outer autocast or a local `autocast(enabled=False)` boundary; record
  that context with the fixture
- vary only semantic dynamic dimensions (token, atom, and MSA depth);
  keep batch, channels, heads, coordinates, diffusion-sample count,
  and config booleans fixed
- use a fixed seed and synthetic finite values; no benchmark sample
  content or model-quality claim comes from this probe
- provide two different sizes, preferably in the same token bin

Compile each target once on a fresh model. Call it in A, A, B, B order
while counting Dynamo frames around every call. Initial A and first B
may compile under automatic `dynamic=None` adaptation. Second A and
second B must compile zero new frames. Recursively assert every tensor
output is finite.

Before installing the compiled wrapper, run the same first synthetic
input through the eager child and retain its outputs. Record max
absolute error and relative L2 versus that eager baseline as
**diagnostics only**. `torch.compile` rewrites the graph and can
drift intermediate tensors while still producing features that are
good for downstream folding. That drift is **not** a probe failure.

Fail the synthetic probe only when:

- the compiled call raises or OOMs
- any tensor contains NaN or Inf
- an immediate same-shape repeat recaptures

Do not reject a target because relative L2 or max abs versus eager
exceeds a tight tolerance. Downstream fitness is judged later by
lDDT and DockQ on real samples, not by child-tensor closeness to
eager. Dynamo's eager backend remains an optional diagnostic when
you want to isolate an Inductor lowering from the selected module.

Run targets separately first (Pairformer, then DiffusionModule), then
the combined target set only when each passes. A failing target goes
straight through the retry ladder using this cheap fixture. After the
largest target set passes synthetically, build a fresh model and run
the real two-sample integration probe once to verify module discovery,
shared aliases, the production call path, and output writing.

Synthetic timings are **never benchmark results** and never appear in
latency tables or charts. If a faithful direct-input fixture cannot be
built for a target, record that and use the real probe; do not feed a
different signature merely to make compile pass.

**The real integration probe uses two samples**, not one. Smallest
in-scope first, then a different-shape item (prefer the same
`token_bin` when one exists). **Probe passes** only when all of these
hold:

- `torch.compile` on the hot submodules does not raise
- both samples complete warmup 1 + measure 1 without exception
  or OOM
- sample 1 warmup **may** compile (`warmup_compile_delta > 0`)
- sample 1 measure does **not** compile
- sample 2 warmup may compile while `dynamic=None` automatically
  specializes or widens for the changed shape
- sample 2 measure does **not** compile
- outputs are finite (no NaN / Inf in coordinates)

Child-tensor drift versus eager is recorded, not gated. Judge compile
quality on the real samples with lDDT and DockQ.

Do not treat "compile is slower than eager" as a probe failure —
still publish both columns. Record every automatic warmup
specialization. A measured-forward compile event is a failure because
the discarded warmup did not establish a stable graph for that row.

Wrap **submodules**, never the top-level `nn.Module`. Do not
compile in-place on the eager instance and lose the eager path.
Before replacement, find every parent `_modules` slot that references
the selected child by object identity and point all aliases at the same
compiled wrapper. A shallow attribute can share a module with the
actual inference driver; replacing only the shallow slot creates a
dead compile column that still runs eager. Record alias paths and
assert every selected wrapper observed inputs.

| Family                      | Compile these                      |
| --------------------------- | ---------------------------------- |
| OF2-style (OpenFold2 / AF2) | Evoformer                          |
| AF3-style (Boltz, OF3)      | Pairformer **and** DiffusionModule |

Pinned Protenix-v2 is an exception: skip its compile column under the
current model profile because compilation is too slow and fails
downstream structural-quality fitness despite finite, stable measured
forwards.

```python
import torch

class DefaultCompiledModule(torch.nn.Module):
    """Use torch.compile's automatic dynamic=None behavior unchanged."""

    def __init__(self, module, role):
        super().__init__()
        self.role = role
        self.compiled = torch.compile(module)

    def forward(self, *args, **kwargs):
        return self.compiled(*args, **kwargs)


def compile_oss_hot_modules(model, family: str) -> list[str]:
    """Compile one shallowest module for every required semantic role."""
    aliases = (
        {"evoformer": {"evoformer", "evoformerstack"}}
        if family == "of2"
        else {
            "pairformer": {
                "pairformer",
                "pairformerstack",
                "pairformermodule",
            },
            "diffusion": {"diffusion", "diffusionmodule"},
        }
    )
    named = list(model.named_modules())
    wrapped = []
    for role, accepted_tails in aliases.items():
        candidates = [
            (name, child)
            for name, child in named
            if name
            and name.rsplit(".", 1)[-1].lower().replace("_", "")
            in accepted_tails
        ]
        if not candidates:
            raise RuntimeError(f"no OSS module found for compile role {role!r}")
        name, child = min(
            candidates,
            key=lambda item: (item[0].count("."), len(item[0])),
        )
        compiled = DefaultCompiledModule(child, role)
        alias_slots = []
        for parent_name, parent in named:
            for attr, referenced in parent._modules.items():
                if referenced is child:
                    alias_slots.append((parent, attr))
        if not alias_slots:
            raise RuntimeError(f"no parent slot found for {name!r}")
        for parent, attr in alias_slots:
            setattr(parent, attr, compiled)
        wrapped.append(name)
    return wrapped
```

Do not pass `dynamic=True` or `dynamic=False`; omission is intentional,
not shorthand. With `dynamic=None`, Dynamo initially assumes static
shapes and automatically recompiles toward dynamic shapes when later
tensor sizes change. Do not override
`automatic_dynamic_shapes` / `assume_static_by_default`, and do not
call `mark_dynamic`, `maybe_mark_dynamic`, or `mark_static`. Record
`compile_dynamic: null`, the observed Dynamo defaults, target paths,
and alias paths in the result JSON.

### Record the recompile limits, and pin them

`dynamic=None` means Dynamo recompiles as shapes change, and a spec that
walks 29 to 1734 residues does exactly that. Three settings govern what
happens when it recompiles too often, and none of them used to reach the
results:

| setting                       | what it does                                   |
| ----------------------------- | ---------------------------------------------- |
| `recompile_limit`             | per code object; past it that frame goes eager |
| `accumulated_recompile_limit` | the same, across all frames                    |
| `fail_on_recompile_limit_hit` | raise instead of falling back                  |

"Goes eager" means for the **rest of the process**, not just that call —
so a column that keeps the name `torch.compile` can finish as eager plus
guard overhead, and nothing raises or logs at default verbosity.

Pin `recompile_limit` to **128** (`OSS_RECOMPILE_LIMIT=128`), turn the
failure on (`OSS_FAIL_ON_RECOMPILE_LIMIT=1`), and record all three
**effective** values — read back off `torch._dynamo.config`, not echoed
from the environment — beside `compile_dynamic: null`.

`compile_dynamic` stays `null`. Pinning a limit is not a shape policy: it
bounds how many specializations may happen before the run has to admit it
stopped compiling.

**This is provenance, not a fix.** On the spec the limit is nowhere near
binding: 17 recompile events over 6 frames, at most 3 for any one frame,
against the pinned image's 512 — and forcing 128 moved no H200 sample by
more than 1%. Pin it so a future run cannot drift into a fallback
unnoticed, and record it so a reader never has to wonder. Do not reach for
it as an explanation.

**Never lower the limit to a remembered default.** An earlier attempt
installed 8 "because that is torch's default"; the stock
`nvcr.io/nvidia/pytorch` image ships **512**, so it silently cut the limit
by 64x on every run — manufacturing the very fallback it was meant to
detect. Read the value, then set it; never assume it.

**Never guess the attribute name.** torch renamed these in 2.7 and kept
the old spellings as deprecated aliases, and assigning a name that does
not exist on `torch._dynamo.config` **raises** rather than being ignored.
Probe for `recompile_limit` then `cache_size_limit`, and treat "this build
has neither" as a fact to record, not an error to swallow.

### A compile column can fail to reproduce

The real hazard is upstream of any limit: the same compiled module,
measured twice on equivalent nodes, can differ by 2x while every other
column holds.

Boltz-2 on H100 published **2.92x** against `torch.compile` beside
**1.74x** on H200 — an asymmetry between two sm90 parts that a reader was
right to call an error. The largest sample took **186.64 s** in the run
the page was drawn from and **88.52 s** in each of two later runs on an
equivalent node: same driver, ECC clean, BioIR within 0.7% and OSS eager
within 0.3%. Only the compiled kernel moved. Two runs agreeing against one
made the original the outlier, and the square was redrawn at 1.78x — after
which H100 and H200 agree, which is what the two parts should do.

The same signature appeared across the fleet, on one build of the spec
run everywhere — `oss_compile ÷ oss_eager` by residue count:

```text
                29    330    635    959   1142   1339   1734
l40s          0.72   0.78   0.73   0.57   0.53   0.61   0.76
h200          0.73   0.75   0.68   0.64   0.64   0.76   0.68
l40           0.58   0.91   0.84   0.98   1.15   1.54   2.05
h100          0.74   0.88   1.14   1.34   1.42   1.43   1.40
```

Five of fourteen SKUs crossed 1.0 — compile **slower than the eager column
it is meant to beat** — and were published that way. They are not slower
silicon: on L40 the compiled pass ran at a *higher* SM clock and *lower*
mean power than eager while taking twice as long, `measure_compile_delta`
was 0 on every sample, and `peak_allocated_gib` matched eager to the
megabyte. The GPU was waiting, not throttling, and no compilation happened
inside the timed region.

So: **a single measurement of a compile column is not evidence.** Before
publishing one that disagrees with its architectural twin, or that reads
above 1.0 against eager, re-run it on a second node. Cheap, and it is the
only thing that separates a real result from one bad compile.

The synthetic probe calls every selected child as A, A, B, B. Initial
A and first B may compile. Second A and second B must not. This is the
fastest way to distinguish expected automatic adaptation from an
unstable graph before running featurization or diffusion rollout.

### Track recompiles on warmup

A realistic default `torch.compile` column may specialize several
times as shapes and control-flow branches appear. The first warmup is
the initial compile. Later warmups may trigger automatic static-to-
dynamic widening or branch specialization, even inside a previously
seen coarse `token_bin`. Record those events exactly.

The measured forward immediately following each warmup must not
compile. This keeps compilation cost out of the latency window while
proving that the graph selected for that sample is reusable. If a
measured forward compiles, mark the row `recapture` and do not include
it in the compile aggregate.

Count Dynamo frame compiles around **both** the warmup and the
measured forward. Copy this helper into `$WORKDIR/bench/`:

```python
def dynamo_compile_count() -> int:
    """Successful Dynamo frame compiles in this process."""
    from torch._dynamo.utils import counters
    frames = counters["frames"]
    if "ok" in frames:
        return int(frames["ok"])
    return int(sum(int(v) for v in frames.values()))


def run_oss_timed_forward(model, batch, runtime_args):
    """Warmup 1 + measure 1, with compile deltas on each."""
    n0 = dynamo_compile_count()
    time_model_forward(model, batch, runtime_args)  # warmup; discard time
    warmup_delta = dynamo_compile_count() - n0

    n1 = dynamo_compile_count()
    output, elapsed_s, alloc, reserved = time_model_forward(
        model, batch, runtime_args
    )
    measure_delta = dynamo_compile_count() - n1
    return {
        "output": output,
        "elapsed_s": elapsed_s,
        "peak_alloc_gb": alloc,
        "peak_reserved_gb": reserved,
        "warmup_compile_delta": warmup_delta,
        "measure_compile_delta": measure_delta,
        "warmup_recompile": warmup_delta > 0,
        "measure_recompile": measure_delta > 0,
    }
```

Use `run_oss_timed_forward` for the compile probe and every
compile-column sample. Eager BioIR / OSS rows do not need the
deltas (`null`).

### How to read the deltas

- First compiled sample, warmup: `warmup_compile_delta >= 1` is
  the initial compile (Pairformer and DiffusionModule may each
  increment). Expected. `status` stays `"ok"` if measure does
  not compile.
- First compiled sample, measure: `measure_compile_delta == 0`.
- Later samples: warmup compiles are allowed while the default
  `dynamic=None` policy adapts to sizes or control-flow branches.
  They may occur inside an already-seen coarse `token_bin`; record
  them rather than relabeling them as failures.
- `measure_compile_delta > 0` on any sample: recapture leaked
  into the headline. `status="recapture"`. Invalid measured time.
- Any warmup-only specialization stays `status="ok"` because warmup
  is untimed and the immediate measured repeat proves reuse. Still
  record exact deltas and sample ids.

Optional debug log (compile process only):

```bash
export TORCH_LOGS="recompiles"
```

Save stderr under `$WORKDIR/debug/oss_compile_recompiles.log`.
Parse it; do not guess.

**After the compile sweep**, set file-level
`compile_stats`:

- `first_compile_sample` — id whose warmup did the first compile
- `warmup_recompile_ids` — ids with `warmup_compile_delta > 0`
  (includes the first sample)
- `measure_recompile_ids` — ids with `measure_compile_delta > 0`
- `same_bin_recompile_ids` — later rows that compiled after an earlier
  row had already established that `token_bin`; diagnostic only
- `n_later_warmup_recompiles` — count after the first sample
- `measurement_stable` — `true` when every measured-forward delta is
  zero and no sample failed
- `automatic_shape_adaptation` — `true` when any later warmup
  compiled; include all ids and deltas

Keep warmup-adapted rows in the compile aggregate. Skip only
`recapture` / `oom` / failed rows. If every sample warmup compiles,
publish that fact prominently: the reported latency is per-shape
steady state and excludes substantial adaptive compile cost.

### Compile retry ladder

A first-try recapture, graph break, or OOM is **not** enough to
skip the compile column. Retry on a **fresh** OSS model. Keep
`torch.compile(module)` with `dynamic` omitted on every attempt. Never
"fix" a failure with `dynamic=True`, `dynamic=False`, manual shape
marks, a wholly static graph, or `torch.compile(model)`.

Walk these in order. Stop at the first attempt that passes the
probe. Record each try (what changed, exception / recapture
text) in `implementation-notes.md`. Use the synthetic direct-module
fixture for every ladder rung; do not repeat full featurization and
rollout just to reject another compiler setting. Run the real
two-sample integration probe only after a synthetic attempt passes.

1. **Baseline.** Target submodules with `torch.compile(module)`.
   Verify `compile_dynamic` is `null` in metadata and do not mutate
   global Dynamo shape settings.
1. **Verify automatic adaptation.** Run synthetic A, A, B, B inputs.
   Initial A and first B may compile; immediate repeats must not.
   Assert finite outputs. Record eager deltas as diagnostics; do not
   fail the rung on graph-induced drift.
1. **Fence Python-only setup helpers.** If the traceback enters a
   chunk tuner, shape cache, or other control-plane helper rather than
   tensor math, wrap only that helper with `torch.compiler.disable`
   and retry on a fresh model. It runs eagerly outside the graph and
   must return the same config value. Record the helper. Never apply
   the fence to Pairformer, DiffusionModule, Evoformer, their tensor
   blocks, or use `suppress_errors` to hide a backend failure.
1. **Find the real children.** If `compile_oss_hot_modules`
   wrapped nothing (or the wrong module), dump `named_modules()`
   tails and compile the actual Pairformer / DiffusionModule /
   Evoformer objects with the same default policy.
1. **Fewer targets.** Compile one target at a time. AF3: Pairformer
   alone, then DiffusionModule alone,
   then both if each passed. OF2: Evoformer only (already one
   target). Publish the **largest** set that probes clean.
   Record `oss_compile.targets` as that set.
1. **OOM.** Retry with the smaller set from (5). Do not switch
   to static to save memory.

**Still forbidden** (not retries):

- `dynamic=True`, `dynamic=False`, manual shape marks, or
  `fullgraph=True` to force a trace
- `torch.compile` on the whole model
- Per-sample `torch.compile`
- Labeling `aot_eager` or an eager fallback as `oss_compile`

If the ladder is exhausted, set
`oss_compile.enabled=false` with `probe_reason` listing every
attempt. Eager still stands. Do not invent a compile column.

A compile on a **measured** forward means the preceding warmup did not
stabilize that row. Record `status="recapture"` and exclude it.
Warmup-only automatic adaptations—including same-bin adaptations—stay
`"ok"` and remain in the aggregate. Do not call `torch.compile` again
mid-sweep.

`torch.compile(...)` itself belongs only in compile-scenario setup.
Dynamo may compile internally on later warmups as automatic adaptation;
the harness must never call `torch.compile` inside the sample loop.

## Warmup and repeats

**Warmup 1, measure 1** for every family and both backends.

- Construct the model / `build_processor` once. Construction is untimed.
- Run one untimed forward (JIT, CUDA-graph capture, first
  `torch.compile`).
- Run one timed `model.forward()`. That value is the headline.
- Persist it as `forward_s: [<seconds>]`, and report **seconds**
  everywhere — table, charts, progress lines. Do not keep a
  millisecond copy of the same number; one unit end to end is what
  keeps a table and a chart from disagreeing.
- Do not median **repeats**. There is only one measured forward per
  sample. Phase 6 **does** report median and geometric mean of
  per-sample speedups across the set
  ([speedup aggregates](#speedup-aggregates)).

**OSS `torch.compile`:** later warmups may automatically specialize or
widen under `dynamic=None`, including within a coarse `token_bin`.
Track every delta
([Track recompiles on warmup](#track-recompiles-on-warmup)); every
measured-forward delta must be zero.

**BioIR CUDA graphs:** an accepted shape (`num_tokens <= 1024`) may
recapture on that sample's warmup (static CUDA graph). That is
expected; record it as a diagnostic if you see it. Larger shapes use
the eager fallback and must not be forced into capture.
This does not apply to the OSS compile column.

## GPU inventory

Run once per WORKDIR, before any timed call. Save the `nvidia-smi`
output to `$WORKDIR/ref_data/gpu_inventory.csv`. Power limit and
clock rates are required fields, not optional debug.

```bash
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used,memory.free,driver_version,uuid,power.limit,enforced.power.limit,power.default_limit,power.max_limit,clocks.max.sm,clocks.max.graphics,clocks.max.memory,clocks.applications.graphics,clocks.applications.memory,clocks.current.sm,clocks.current.graphics,clocks.current.memory,power.draw,pstate,clocks_event_reasons.active,clocks_event_reasons.gpu_idle,clocks_event_reasons.sw_power_cap,persistence_mode \
  --format=csv > "$WORKDIR/ref_data/gpu_inventory.csv"
```

Copy those numbers into `bench_config.json` as well:

- `gpu_name`, `sm`, `driver`
- `power_limit_w`, `enforced_power_limit_w`,
  `power_default_limit_w`, `power_max_limit_w`
- `clocks_max_mhz` — SM, graphics, memory
- `clocks_applications_mhz` — graphics / memory, or `null` when
  nvidia-smi returns `[N/A]` or a deprecation string
- `persistence_mode`

Idle `clocks.current.*` in this CSV are **not** the run clocks. On
an H100 they are often a few hundred MHz while `clocks.max.sm` is
near 2 GHz. Phase 6 quotes the inventory for the cap and the max
clocks, and quotes in-forward samples for what the GPU actually
ran at.

```python
import os

import torch

cap = torch.cuda.get_device_capability(0)
inventory = {
    "device_name": torch.cuda.get_device_name(0),
    "sm": f"{cap[0]}{cap[1]}",
    "device_count": torch.cuda.device_count(),
    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
```

v1 is one visible GPU. If `device_count > 1`, pin
`CUDA_VISIBLE_DEVICES` to a single index and re-snapshot. Replica
mode places one engine per GPU (`docs/ref/architecture.md`); a
multi-GPU forward is out of scope.

## Per-sample GPU fields

Record memory after each timed forward:

- `peak_alloc_gb` — `torch.cuda.max_memory_allocated() / 1024**3`
- `peak_reserved_gb` — `torch.cuda.max_memory_reserved() / 1024**3`
- `nvidia_smi_used_mib` — `memory.used` immediately after the sync
- `weights_alloc_gb` — peak after model load, before the first
  forward

`nvidia-smi` **utilization %** is noisy on a single forward. Do
not use it as the headline GPU number.

**Power (W) and clock rates (MHz) are headline GPU numbers.** A
single snapshot after `cuda.synchronize()` is usually idle again
(SM clock already dropped). Sample on a sidecar thread during the
measured forward, outside the timing window. Do not call
`nvidia-smi` on the thread that runs `timed_forward`.

[`harness/gpu_telemetry.py`](harness/gpu_telemetry.py) implements this.

Per timed sample, write `gpu_telemetry`:

- `n_polls` — sidecar samples spanning the measured forward
- `n_busy` — polls where `clocks_event_reasons.gpu_idle` is not
  `Active`
- `power_draw_w` — `{min, max, mean}` over **busy** polls
- `clocks_sm_mhz` — `{min, max, mean}` over busy polls (this is
  the compute clock)
- `clocks_graphics_mhz`, `clocks_memory_mhz` — same shape
- `throttle` — named event reasons that were `Active` on any busy
  poll, excluding `gpu_idle` (`sw_power_cap`, `hw_slowdown`,
  `sw_thermal_slowdown`, `applications_clocks_setting`, …)

If `n_busy == 0` (forward shorter than the poll interval), set
the clock / power objects to `null` and say so. Do not report the
idle inventory clock as the run clock.

File-level `gpu` on every result JSON repeats the inventory cap
and max clocks. File-level `gpu_telemetry` aggregates `ok` rows:
min of per-sample mins, max of per-sample maxes, mean of
per-sample means, plus the union of throttle reasons. Phase 6
prints both, and names the gap between `clocks.max.sm` and the
observed SM mean.

Wire it as `sampler.start(); try: elapsed = timed_forward(...) finally:
telemetry = sampler.stop()` and store `telemetry` as `gpu_telemetry`.

After an OOM, record `status="oom"` for that sample and every larger
sample on that backend. Do not skip to a still-larger shape.

## Result JSON

Write one file per scenario:

- `$WORKDIR/results/bioir_forward.json`
- `$WORKDIR/results/oss_eager.json` — always
- `$WORKDIR/results/oss_compile.json` — only if the compile probe
  passed and every published row has `measure_compile_delta == 0`

Set `"backend"` to `bioir_*`, `oss_eager`, or `oss_compile`.

**`residues` is the sum over all polymer chains**, not one chain
and not `max(chain lengths)`. Protein, RNA, and DNA only
(CCD / SMILES ligands are not residues). For each of those
polymers add `len(sequence) * n_copies`, where `n_copies` is
`len(chain_id)` if `chain_id` is a list, else `1`. Separate
polymer entries (A and B) both count. Spec `seq_len` is already
this sum — assert equality; do not replace it with chain A.

[`residues_all_chains`](harness/timing.py) computes it.

Write that integer on the manifest and on every result row.
Example: `8ic7-assembly1_A_B` is `867 + 867 = 1734`, not `867`.

```json
{
  "backend": "bioir_build_processor",
  "model_source": "boltz-2",
  "gpu": {
    "device_name": "...",
    "sm": "90",
    "driver": "...",
    "power_limit_w": 700.0,
    "enforced_power_limit_w": 700.0,
    "power_default_limit_w": 650.0,
    "power_max_limit_w": 700.0,
    "clocks_max_mhz": {"sm": 1980, "graphics": 1980, "memory": 1593},
    "clocks_applications_mhz": {"graphics": null, "memory": null},
    "persistence_mode": "Enabled"
  },
  "config": {"recycling_steps": 3, "num_sampling_steps": 200, "diffusion_samples": 5},
  "warmup": 1,
  "repeats": 1,
  "samples": {
    "5sbj-assembly1": {
      "residues": 30,
      "unpaired_msas": ["/abs/path/msa/5sbj-assembly1_0.a3m"],
      "paired_msas": [],
      "templates": [],
      "template_status": "none_declared",
      "n_templates_supplied": 0,
      "n_templates_attached": 0,
      "status": "ok",
      "forward_s": [2.09],
      "peak_alloc_gb": 12.4,
      "peak_reserved_gb": 14.1,
      "weights_alloc_gb": 4.2,
      "gpu_telemetry": {
        "n_polls": 48,
        "n_busy": 44,
        "power_draw_w": {"min": 380.1, "max": 612.4, "mean": 540.2},
        "clocks_sm_mhz": {"min": 1620, "max": 1830, "mean": 1784},
        "clocks_graphics_mhz": {"min": 1620, "max": 1830, "mean": 1784},
        "clocks_memory_mhz": {"min": 1593, "max": 1593, "mean": 1593},
        "throttle": []
      },
      "lddt": 0.81,
      "dockq": 0.62,
      "dockq_status": "ok",
      "dockq_scope": "all_interfaces",
      "dockq_mapping": "AB:AB",
      "dockq_interfaces": {"A-B": {"DockQ": 0.62, "iRMSD": 1.9,
                                   "LRMSD": 5.4, "fnat": 0.51,
                                   "clashes": 0}},
      "dockq_json": "/abs/path/debug/dockq_5sbj-assembly1.json",
      "pred_cif": "/abs/path/5sbj-assembly1.cif",
      "gt_path": "/abs/path/ground_truth/5sbj-assembly1.cif"
    }
  }
}
```

`forward_s` is seconds, matching `model_inference_time`, and
seconds is also the reporting unit — `forward_s[0]` is the headline
number, printed to three decimals. Do not convert to milliseconds
anywhere.

`residues` on every row must be `residues_all_chains` (sum of
all protein / RNA / DNA chains). Do not store a single-chain
length.

`n_templates_attached` must equal `min(n_templates_supplied, cap)`
and must match across the two result files for the same sample id.
A row templated on one side and bare — or filtered down — on the
other is a hard failure, not a latency win
([templates.md](templates.md#verification)).

Row `status` is `ok`, `oom`, `recapture`, or `failed` (a crash, with its
stderr in `implementation-notes.md`). Run
[`check_results.py`](harness/check_results.py) on every result file of the
comparison before Phase 6. It enforces this section per file, then checks that
the files cover the same samples and agree on residues, template counts, and
DockQ scope.

On `oss_compile.json` also require file-level `compile_stats` and
per-sample `warmup_compile_delta`, `measure_compile_delta`,
`warmup_recompile`, `measure_recompile`. Any sample may have
`warmup_recompile: true` and still `status: "ok"` if
`measure_compile_delta == 0`; this is automatic adaptation under the
default policy. Eager / BioIR files omit `compile_stats` or set the
deltas to `null`.

```json
"compile_stats": {
  "first_compile_sample": "5sbj-assembly1",
  "warmup_recompile_ids": ["5sbj-assembly1", "7qsj-assembly1"],
  "measure_recompile_ids": [],
  "n_later_warmup_recompiles": 1,
  "measurement_stable": true,
  "compile_dynamic": null
}
```

Every result JSON also carries file-level `gpu_telemetry` over
`ok` rows: min of per-sample mins, max of per-sample maxes, mean
of per-sample means for power draw and SM / graphics / memory
clocks, plus the union of throttle reasons. Phase 6 prints that
next to the inventory cap. Missing `gpu_telemetry` on an `ok` row
is a harness bug.

```json
"gpu_telemetry": {
  "n_ok": 15,
  "n_busy_total": 620,
  "power_draw_w": {"min": 390.2, "max": 612.4, "mean": 528.1},
  "clocks_sm_mhz": {"min": 1680, "max": 1830, "mean": 1766},
  "clocks_graphics_mhz": {"min": 1680, "max": 1830, "mean": 1766},
  "clocks_memory_mhz": {"min": 1593, "max": 1593, "mean": 1593},
  "throttle": []
}
```

## Quality: lDDT and DockQ

Two metrics, both required, both outside the timing window. They
answer different questions and one cannot stand in for the other:
lDDT scores the structure as a whole, DockQ scores the
**interfaces**. A complex can hold every chain's fold and still
dock them wrongly, which reads as a good lDDT next to a DockQ in
the "incorrect" band — so a run reporting lDDT alone cannot tell
whether the two stacks agree on the assembly.

Install both before Phase 4
([environment.md](environment.md#step-8--install-dockq-interface-scorer)).
[`harness/scoring.py`](harness/scoring.py) runs both and applies the rules
below; build any scorer-only DockQ input copy before calling it.

**Per sample, per side:**

- `lddt` — `ost compare-structures`, on every sample with a
  ground truth.
- `dockq` — `GlobalDockQ` from DockQ's `--json` output, on every
  supported protein-protein or protein-small-molecule complex with a
  ground truth. This is the mean over native interfaces; the
  per-interface numbers go in `dockq_interfaces`.
- `dockq_status` — `ok` | `single_chain` | `no_gt` |
  `no_native_interfaces` | `unsupported_chain_types` | `failed`.
- `dockq_scope` — `all_interfaces` or `protein_only`, meaning
  whether `--small_molecule` was passed. Must be the same value on
  both sides for a given sample.
- `dockq_mapping` — `best_mapping_str`, the model:native chain
  assignment DockQ chose. Record it; a nonsensical mapping explains
  a surprising score.
- `dockq_model_input`, `dockq_native_input`, and
  `dockq_normalization` — exact scorer inputs and any scorer-only
  missing-occupancy or manifest-declared ligand `HETATM`
  normalization. Never overwrite a prediction.

**Rules:**

1. `dockq: null` on a single-chain sample, with
   `dockq_status: "single_chain"`. **Never `0.0`** — zero is a real
   DockQ value meaning "interfaces are wrong", and a monomer has no
   interface to get wrong. DockQ exits 1 on an interface-free
   native; that message is the monomer answer, not an error.
1. DockQ 2.1.3 does not support RNA/DNA interfaces. When either
   polymer type is present, do not invoke it: use `dockq: null`,
   `dockq_status: "unsupported_chain_types"`, and
   `dockq_scope: "unsupported"` on both sides. `--small_molecule`
   extends DockQ to protein-ligand, not RNA-ligand.
1. Never invent either number. A non-zero exit that is *not* the
   no-interface case leaves `dockq: null`,
   `dockq_status: "failed"`, and stderr in `implementation-notes.md`.
1. Same flags on both sides. `dockq_args` is locked in
   `bench_config.json`, and a per-sample `--small_molecule`
   decision comes from the manifest, not from what happened to
   work.
1. A writer may emit a manifest-declared CCD ligand as `ATOM` when its
   residue name is also an amino acid. DockQ then treats that ligand as
   a protein chain and mapping fails. In a scorer-only copy, change
   only those declared ligand chains to `HETATM`; record the operation
   and apply the rule to both sides. Never infer ligand chains from
   residue names.
1. When DockQ identifies a limited model/native sequence mismatch,
   lock the smallest required `--allowed_mismatches` value globally
   for the comparison. Do not tune it independently per side.
1. Aggregate as a **mean over scored samples**, reported with the
   count, per side. Do not average across sides, and do not fill a
   `null` with the set mean.
1. Report DockQ against its own bands, not as a percentage:
   `< 0.23` incorrect, `0.23–0.49` acceptable, `0.49–0.80` medium,
   `>= 0.80` high.

**A DockQ gap between the two sides is a finding, not noise.** Both
sides fold the same inputs with the same config, so a systematic
interface-quality difference points at an input asymmetry — MSA
depth, pairing, or a dropped template ([msa.md](msa.md),
[templates.md](templates.md#verification)) — before it points at
the engine.

## Locked config

`$WORKDIR/ref_data/bench_config.json` is written in Phase 1 and is
read-only afterwards. Both harnesses load it. A harness that accepts a
flag that silently overrides a locked key is a bug.

Required keys:

- `model_source` — `FoldingSupportMatrix` key
- `feature_path` — `processor` (`build_processor`) or `oss` (OSS features)
- `oss_root`, `oss_url`, `oss_ref`, `oss_commit` — pin from
  [environment.md](environment.md#oss-checkout-pins)
  (`3rdparty/` gitlink, `v2.2.1`, or `v2.2.0`)
- `oss_entry`
- `bioir_python`, `oss_python`, `bioir_torch`, `oss_torch`
- `isolation` — `two_venv` or `shared`
- `checkpoint_id` / path / hash on each side
- `runtime_args` (BioIR names) and `oss_runtime_args` (OSS names)
- AF3-style families map both sets of runtime args to
  `num_sampling_steps=200` and `diffusion_samples=5`
- `boltz-1`, `boltz-2`, and `openfold3` use `recycling_steps=3`;
  `protenix-v2` maps BioIR `recycling_steps=5` to OSS `model.N_cycle=6`
- `precision` / autocast policy
- `seed`
- `warmup` — always `1`
- `repeats` — always `1`
- `oss_compile` — `{enabled: true|false, dynamic: null, targets: [...],
  probe_sample, probe_second_sample, probe_ok, probe_reason,
  retries: [...], measurement_stable: true|false,
  n_later_warmup_recompiles: int, warmup_recompile_ids: [...]}`
- `bioir_config` — `"get_pretrained_config"` (required)
- `graph_activation` — `"default"`, or `"disable_cuda_graphs()"` for eager.
  Record enabled roles, `graph_config="model_default"`, and
  `num_tokens_max=1024`; never
  serialize a model-level graph-config override
- `cuda_visible_devices`
- `gpu_name`, `sm`, `driver`
- `power_limit_w`, `enforced_power_limit_w`,
  `power_default_limit_w`, `power_max_limit_w`
- `clocks_max_mhz` — `{sm, graphics, memory}`
- `clocks_applications_mhz` — `{graphics, memory}` or `null`s
- `persistence_mode`
- `executor_backend` — always `null` (serial). See below.
- `container` — default `nvcr.io/nvidia/pytorch:26.09-py3`
- `bioir_install` — `editable` | `wheel` | `pip`
- `cutedsl_force_cubin` — always `true`
- `ost_cmd` — `$OST_ENV/bin/ost` (dedicated conda env)
- `ost_env` — default `$WORKDIR/envs/ost`
- `ost_version` — `ost --version` output, e.g. `2.10.0`
- `dockq_cmd` — `$DOCKQ_ENV/bin/DockQ` (dedicated venv)
- `dockq_env` — default `$WORKDIR/envs/dockq`
- `dockq_version` — from package metadata (no `--version` flag)
- `dockq_args` — locked flag set, identical on both sides
- `dockq_input_normalization` — scorer-only operations: add
  `_atom_site.occupancy=1.0` when absent and mark manifest-declared
  small-molecule chains `HETATM` when a writer mislabeled them `ATOM`
- `deepspeed_evo_attn` — `true` for OpenFold2 / OpenFold3 OSS
  (source-built `evoformer_attn` only); `false` otherwise
- `oss_kernels` — accelerated paths the OSS config enables, e.g.
  `{"cueq_triangle": true, "deepspeed_evo_attn": true,
  "flash_attn": false}`. cuEq must be `true` whenever the OSS tree
  integrates it
- `template_config` — per side, the ingestion form, `use_templates`,
  the cap (`n_templates` / `max_templates`), the structure directory,
  and every filter value (identity, coverage, length, date, resolved
  fraction). "Defaults" is not an answer
  ([templates.md](templates.md#6-neutralize-every-filter-and-equalize-the-cap))
- `oss_patches` — recorded input-side patches applied on top of the
  pin: `[{"file": "ref_data/patches/<name>.patch", "what": ...,
  "why": ..., "reverted": true}]`, or `[]`. Compute-path patches are
  forbidden
- `bioir_patches` — same shape, for a BioIR-side deviation
- `apt_packages` — distro packages OSS needed (or `[]`)
- `apt_install_script` — generated `ref_data/apt_install.sh`, or `null`
- `apt_installed_by` — `apt` | `sudo` | `human` | `none`
- `cuda12_remaps` — `{from, to}` list, or `[]`
- `dataset_manifest_sha256`, `dataset_build_sha256` — MANIFEST.json and
  BUILD.json digests; a built tree has no version string that identifies it
- `dataset_root` — `benchmarks/dataset`
- `dataset_spec` — `spec_full.json` or `spec_monomer.json`
- `dataset_manifest_sha256` — SHA-256 of `MANIFEST.json`

## Comparison table

| sample | residues | BioIR (s) | OSS eager (s) | OSS compile (s) | vs eager | vs compile | BioIR peak (GB) | OSS eager peak (GB) |
| ------ | -------- | --------- | ------------- | --------------- | -------- | ---------- | --------------- | ------------------- |

Latency cells are **seconds to three decimals**.

`vs eager = oss_eager_s / bioir_s`. `vs compile` only when that
column exists. Values `> 1` favor BioIR. If compile was skipped, write
`—` in the compile cells. If a side OOMs, write `OOM` and skip that
speedup cell.

Phase 6 then prints speedup aggregates — overall and by residue
range — and writes `$WORKDIR/results/speedup.json`
([speedup aggregates](#speedup-aggregates)).

Quality is a **second table**, not extra columns on the latency one
([quality rules](#quality-lddt-and-dockq)):

| sample | chains | BioIR lDDT | OSS lDDT | BioIR DockQ | OSS DockQ | interfaces |
| ------ | ------ | ---------- | -------- | ----------- | --------- | ---------- |

Write `n/a` in the DockQ cells of a single-chain sample (`chains`
is `1`), never `0.00`. Close with the per-side mean lDDT and mean
DockQ, each with the number of samples it covers, since the DockQ
mean covers only the multi-chain subset.

## Speedup aggregates

Phase 6 prints speedup as **geometric mean and median**, not a
single overall geomean. Ratio is `oss_s / bioir_s` on in-scope
rows where both sides are `status: "ok"`. Values `> 1` favor
BioIR. Compute vs OSS eager always, and vs OSS compile when that
JSON exists.

Residue bins use `residues` (sum of all polymer chains):

- **short** — `residues < 512`
- **medium** — `512 <= residues <= 1024`
- **long** — `residues > 1024`

Print every bin, including empties. An empty bin is `n=0` with
**no** geomean or median — do not invent a ratio. Name the sample
ids in every non-empty bin.

Per comparison (vs eager, vs compile) print:

- overall (`n`, geomean, median)
- short / medium / long (`n`, geomean, median, ids)

Write the same structure to `$WORKDIR/results/speedup.json` (or
`$WORKDIR/results/<role>/speedup.json` when roles are split).
Speedup numbers live in that JSON and in the Phase 6 printout,
not in the model profile notes.

Run [`harness/summarize_speedup.py`](harness/summarize_speedup.py) with
`$BIOIR_PYTHON` from `$WORKDIR` (or pass the role directory). Ratios are three
decimals.

Two-role benches (OpenFold2 monomer / multimer) run this once per
role directory. Do not merge the two roles into one geomean.

## Charts (latency vs residues)

Phase 6 **must** plot the headline times. A table alone is not
the report. Run [`harness/plot_latency.py`](harness/plot_latency.py) with
`$BIOIR_PYTHON` from `$WORKDIR`.

Required artifacts:

- `$WORKDIR/results/latency_vs_residues.png` — `forward_s` vs
  residue count. Series: BioIR, OSS eager, OSS compile (omit
  compile if that JSON is absent).
- `$WORKDIR/results/speedup_vs_residues.png` — `oss_s / bioir_s`
  vs residue count. Series: vs eager, and vs compile when present.
  Horizontal line at `1.0`.
- `$WORKDIR/results/speedup.json` — geomean and median, overall
  and in residue bins
  ([speedup aggregates](#speedup-aggregates)).

Rules:

- X is `residues` from the result row — **sum of all polymer
  chains**, same integer as the manifest (`residues_all_chains`).
  Sort points by X. Do not plot chain A only.
- Y is `forward_s` in **seconds** (or the speedup of two `ok`
  rows), matching the table's unit. Skip
  `oom` / `recapture` / missing keys. Do not
  interpolate or invent a point.
- Annotate `sample_id` on each point when there are ≤ 20
  in-scope samples.
- Title must name the model key and GPU from `bench_config.json`.
- After writing the PNGs, **read both image files in the final
  reply** so the human sees the charts, not only a path.

## Executor: serial, not Ray

the bench set is small (tens of spec items, one request at
a time). Use the serial processor:

```python
EngineProcessorConfig(..., executor_backend=None)
```

Ray (`executor_backend="ray"`) only pays off on a large dataset, where
CPU stages can overlap many GPU forwards. On this set the actor /
`ray.init` / `map_batches` overhead dominates and the per-row times
are not the model cost (`docs/ref/architecture.md`). Do not switch to
Ray for v1.

## Anti-patterns

- Building a BioIR `BaseConfig` by hand, or passing anything other
  than `get_pretrained_config(...)` as `engine_kwargs["config"]`.
- Timing `processor(rows)` wall clock and calling it `model.forward`.
- Timing OSS `predict.py` end-to-end and comparing it to
  `model_inference_time`.
- Reimplementing OSS featurize + load from library internals
  when the repo already has a `predict` / `infer` script. Wrap
  that script; only the `forward` clock is new.
- Leaving H2D or the postprocessor inside one side's window.
- Disabling OSS kernels "for fairness". The OSS column is the source
  model's recommended inference path.
- Falling back to eager / math SDP / no-cuEq after a crash and still
  labeling the row `oss_e2e`.
- Reporting the warmup forward (graph capture / `torch.compile`) as
  the measured sample.
- Skipping OSS eager because compile worked, or publishing compile
  after a failed probe.
- `torch.compile(model)` on the whole OSS module, or setting
  `dynamic=True` / `dynamic=False` on a target.
- Dropping the compile column after one recapture without walking
  the retry ladder.
- Calling `torch.compile` again for each sample instead of compiling
  the target submodules once and reusing that model.
- Hiding automatic warmup specializations. Track every
  `warmup_compile_delta` and state when the column represents
  per-shape steady state.
- Publishing OSS compile measured forwards that recapture after their
  discarded warmup.
- Using Ray (`executor_backend="ray"`) on this small dataset. Ray is
  for large inference jobs; v1 uses serial (`executor_backend=None`).
- Substituting `examples/data/samples/` or a demo FASTA for
  the bench set.
- Timing OSS `main` or a tag other than the pin (`3rdparty/`
  gitlink, Boltz `v2.2.1`, OpenFold `v2.2.0`).
- Installing PyPI DeepSpeed for OpenFold2 / OpenFold3, or
  `DS_BUILD_OPS=1` (build only `DS_BUILD_EVOFORMER_ATTN`). Do not
  point `CUTLASS_PATH` at `$REPO/3rdparty/cutlass` (CUTLASS 4).
- Skipping an OSS apt package after `apt-get` / `sudo` failed.
  Stop and give the human the command.
- Installing OSS `*-cu12` / `[cu12]` / `+cu12*` wheels on the
  CUDA 13 container. Remap to `cu13`.
- Plotting or storing `residues` as a single chain (or the max
  chain) instead of the sum over all protein / RNA / DNA chains.
- Sharing the GPU with another job.
- Running BioIR without `CUTEDSL_FORCE_CUBIN=1` (CuTeDSL JIT on the
  clock). In developer mode, skipping `git lfs pull` so packs are
  pointer files.
- Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on a
  timed run. It can cost throughput, so it never belongs in a
  latency column — least of all on one side only.
- Inventing or rounding away a number that is not in the JSON artifact.
- Ending Phase 6 with a table and no latency-vs-residues chart, or
  plotting points that are not in the result JSONs.
- Reporting GPU name without power limit, observed power draw, max
  SM clock, and observed SM clock. Quoting idle inventory clocks
  (often a few hundred MHz) as the run clock.
- Printing only an overall speedup geomean. Phase 6 also needs
  median, and geomean plus median in residue bins short `< 512`,
  medium `512–1024`, long `> 1024`, with `n=0` on empty bins.
