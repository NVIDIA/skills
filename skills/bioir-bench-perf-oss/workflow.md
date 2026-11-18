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

# Workflow — phase procedures

The full procedure behind each phase in [SKILL.md](SKILL.md). SKILL.md keeps
the mandatory gates and what each phase hands to the next; read the section
here when you start that phase. Deeper rules live in the references each
section links.

## Phase 0 — WORKDIR, environments, GPU, OSS root

Ask for `$WORKDIR` (default `workdir/bench_perf_<model>/` at the repo
root). Resolve `$OSS_ROOT` from the
[OSS checkout pins](environment.md#oss-checkout-pins) — do not
guess, and do not clone `main`.

- OpenFold3: follow its checkpoint, source, data, and patch profile in
  [models/of3.md](models/of3.md).
- Boltz-2: follow its source pin, template enablement, MSA, and kernel
  profile in [models/boltz2.md](models/boltz2.md).
- OpenFold2 / AlphaFold2: follow the protein-only AF2 model-1
  checkpoint, recycle, MSA/template, and Evoformer-compile profile in
  [models/of2.md](models/of2.md). WORKDIR default `/tmp/openfold2`.
- Protenix-v2: follow the Processor path source, checkpoint, CCD, recycle
  mapping, and input profile in [models/protenix.md](models/protenix.md).
  WORKDIR default `/tmp/protenix`.
- Boltz-1/2: [jwohlwend/boltz](https://github.com/jwohlwend/boltz) `v2.2.1`
- OpenFold2:
  [aqlaboratory/openfold `v2.2.0`](https://github.com/aqlaboratory/openfold/tree/v2.2.0)

```bash
WORKDIR=${WORKDIR:-workdir/bench_perf_<model>}
mkdir -p "$WORKDIR"/{bench,ref_data,oss_data,results,debug,venv_oss,envs,oss}
```

```text
$WORKDIR/
├── oss/                        # Boltz / OpenFold2 clones at the pinned tag
├── bench/                      # BioIR + OSS harnesses (written here)
├── venv_oss/                   # OSS interpreter when isolation is two_venv
├── envs/ost/                   # dedicated OpenStructure conda env
├── envs/dockq/                 # dedicated DockQ venv
├── miniforge3/                 # bootstrapper only; do not install ost in base
├── ref_data/
│   ├── sample_manifest.json    # canonical sample set + MSA paths
│   ├── bench_config.json       # locked runtime / GPU / checkpoint / pythons
│   ├── gpu_inventory.csv
│   ├── env_bioir.txt           # pip freeze
│   ├── env_oss.txt
│   └── env_conflicts.md
├── oss_data/                   # dataset re-shaped for the OSS loader only
│   ├── msa/                    # alignments regrouped / renamed as it wants
│   ├── templates/              # template alignments or structures likewise
│   ├── queries/                # generated per-sample input files
│   ├── cache/                  # anything the OSS preprocessor writes
│   └── index.json              # generated path -> dataset source, per chain
├── results/
│   ├── bioir_forward.json
│   ├── oss_eager.json              # always
│   ├── oss_compile.json            # only if compile probe passed
│   ├── latency_vs_residues.png     # Phase 6 chart (required)
│   ├── speedup_vs_residues.png     # Phase 6 chart (required)
│   └── speedup.json                # geomean + median, overall and bins
├── implementation-notes.md
└── NOTES.md
```

Harnesses live in `$WORKDIR`. Do not modify `bionemo_ir/` or `$OSS_ROOT`.

`oss_data/` holds the benchmark dataset re-shaped to whatever layout and
naming the OSS loader demands. What that means is per tree — one
directory per chain, a fixed set of recognized filename stems, a native
YAML / CSV / JSON query, a synthesized alignment, a preprocessor cache —
so treat the subdirectories above as a starting shape and keep the index
JSON, which is the part that makes the mapping auditable. Build it with a
script (`bench/stage_oss_*.py`) rather than by hand.

Three rules keep it honest:

- **Symlink, never copy or edit.** Entries point at
  `$DATASET_ROOT`. Same bytes on both sides, and nothing is written into
  the dataset. Point every OSS output directory (caches included) here,
  since some preprocessors create directories next to their inputs.
- **BioIR keeps reading the original dataset paths.** `oss_data/` exists
  because the OSS loader needs a shape, not because the inputs differ. If
  a reshape would change *content* (a parse-time row cap that truncates
  what BioIR reads in full, a template the OSS filters reject), that is a
  parity break: fix it or record it, do not paper over it.
- **Assert the mapping, don't assume it.** Loaders select inputs by
  filename stem and directory name, and the common failure is silent —
  files skipped, or chains collapsed onto one alignment. Verify on the
  featurized batch ([msa.md](msa.md), [templates.md](templates.md)).

### Environments (do this before GPU timing)

Follow [environment.md](environment.md) Steps 1–8 in order: container and
BioIR install with `CUTEDSL_FORCE_CUBIN=1`, probe BioIR, probe OSS at its pin,
diff and classify the freezes, two venvs or a shared env with BioIR as the
lock, CUDA 12 → 13 remaps, OSS apt packages, the DeepSpeed `evoformer_attn`
source build for OpenFold2 / OpenFold3, a dry import of both sides, then
OpenStructure and DockQ in their own environments. Do not skip to a harness
because "python is already there". Record every field in
[What to record](environment.md#what-to-record) in `bench_config.json`.

### GPU inventory

Save before any timed call. See
[measurement.md](measurement.md#gpu-inventory).

```bash
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used,driver_version,uuid,power.limit,enforced.power.limit,power.default_limit,power.max_limit,clocks.max.sm,clocks.max.graphics,clocks.max.memory,clocks.applications.graphics,clocks.applications.memory,clocks.current.sm,clocks.current.graphics,clocks.current.memory,power.draw,pstate,clocks_event_reasons.active,clocks_event_reasons.gpu_idle,clocks_event_reasons.sw_power_cap,persistence_mode \
  --format=csv
"$BIOIR_PYTHON" -c "import torch, bionemo_ir; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(), bionemo_ir.__file__)"
"$OSS_PYTHON" -c "import torch; print(torch.cuda.get_device_name(0), torch.__version__)"
```

Power limit and clock rates are part of the inventory, not optional.
Save the CSV as `$WORKDIR/ref_data/gpu_inventory.csv`. Idle SM clocks
in that CSV are nameplate context only — they are often a few hundred
MHz. Harnesses sample power and SM / graphics / memory clocks
**during** every measured forward
([measurement.md](measurement.md#gpu-inventory)).

Look up `model_source` in `docs/ref/support-matrix.md`. Set
`feature_path` in `bench_config.json`:

- Pipeline **Yes** → `processor` (folding factory)
- Pipeline **No** and folding → `oss`
  ([no-pipeline.md](no-pipeline.md)). Do not call `build_processor`.
- Pipeline **No** and not folding (`boltz-2-affinity`) → stop.
  Extend the skill first
  ([Extending to other model kinds](#extending-to-other-model-kinds)).

### Weights preflight — prove one load before any timed run

Resolve and load the checkpoint **once, offline, before Phase 4**.
Many of these checkpoints sit behind gated or authenticated repos, so
a harness that reaches the hub per sample can fail every sample in
turn and leave a full results file of zeros — after the whole
environment build. Never discover that from inside the bench loop.

First prove that the pin and the BioIR key still name the **same model
variant**. Read the OSS checkpoint registry / compatibility spec and
its current default checkpoint, then compare that with BioIR's
resolved checkpoint. A source release can retire the checkpoint BioIR
still ships or switch its default to a new architecture while keeping
the same package and CLI name. An explicit file path may bypass the
registry check, so validation success is not enough: the subsequent
strict state-dict load is the gate.

If the OSS pin rejects the BioIR checkpoint, or its current checkpoint
has a different architecture / training variant, stop. Do not average
per-block weights into a new shared parameter, add or delete keys, use
`strict=False`, or call two different checkpoints "the same model".
That is model onboarding, not a benchmark input patch. Either onboard
the new variant into BioIR or, with the user's explicit choice, change
the benchmark's declared OSS revision and model variant.

State-dict compatibility is necessary but not sufficient. Compare
non-weight semantics changed by the pin too: tensor orientation,
normalization placement, feature bookkeeping, inference presets, and
default gates. If BioIR combines an old checkpoint with behavior
ported from a newer incompatible release, it is a third, unvalidated
variant; changing only the OSS revision cannot make that comparison
apple-to-apple.

Stage weights with `scripts/fetch_weights.sh --model <family>` when
the hub cannot resolve them, or point BioIR at a checkpoint already
on disk (`BIOIR_CHECKPOINTS=<root>` scans `<root>/<model>/*.pt`; the
per-family variable named by the hub's default repo id takes a direct
file path — `bionemo_ir/hubs/local.py`). Repo checkouts often already
carry the file under `examples/<model>/checkpoints/`; look there
before downloading anything.

The gate: one construct-and-load in the bench interpreter, with the
network unavailable or the hub untouched, logging the resolved local
path, followed by the same strict load in the OSS interpreter. A 401 /
`GatedRepoError`, incompatible-version declaration, missing /
unexpected state-dict keys, or any per-sample hub request is a setup
failure — fix it before timing, and do not paper over it with a token
or a non-strict load in the harness. Record both checkpoint paths,
hashes, registry names, and declared compatibility ranges in
`bench_config.json`.

## Phase 1 — Manifest + locked config

### Sample manifest

Build or reuse the dataset
at `benchmarks/dataset/`
([samples.md](samples.md#build)), and check it with
`rebuild_dataset.py --verify` against `MANIFEST.json`. Never put a
token in the harness or notes.

Pick the spec ([samples.md](samples.md#which-spec)). Enumerate
**every** item in that spec (not only in-scope rows). Record:

- `sample_id` (spec `id`), `spec`, `chemical_class`, `seq_len`,
  `token_bin`
- `residues` — sum of all protein / RNA / DNA chains (see
  [measurement.md](measurement.md#result-json)); must equal
  spec `seq_len`
- per-polymer `polymer_type`, `chain_id`, sequence length
- resolved unpaired and paired A3M paths (against `$DATASET_ROOT`)
- `boltz_yaml` / `boltz_msa_csv` when present
- `msa_status`: `attached` | `none_declared` | `missing_file`
  (blocker)
- `gt_path` — `$DATASET_ROOT/<spec gt>`, or `null`
- `has_templates`, `templates[]` (resolved paths + `chain_id`),
  `template_status`
  ([samples.md](samples.md#templates-are-in))
- `in_scope` — see filters below

Filter to what the model can run
(`docs/ref/support-matrix.md`):

- AF2 / OF2 monomer — `spec_monomer.json` (protein monomers only)
- AF2 multimer — `protein-protein` items from `spec_full.json`
  whose polymers are all protein (no RNA / DNA / ligand). See
  [models/of2.md](models/of2.md#protein-only).
- Boltz-1/2, OpenFold3, `protenix-v2` — `spec_full.json`

Every filter is about model capability, never about templates. Attach
each item's listed A3Ms / Boltz CSVs and templates.

Write `$WORKDIR/ref_data/sample_manifest.json`. A later harness that
processes a different set is a hard failure.

Default is every **in-scope** spec item, templates included. The user
may narrow it further; do not silently drop RNA/DNA/ligand or
template-bearing samples on Boltz / OpenFold3.

### Locked inference config

**Start from the OSS inference script.** The OSS repo almost
always ships one (`predict.py`, `infer.py`, `run_predict.py`,
`main.py predict`, a CLI entry in `pyproject.toml`). Find it
first. Copy or import that path into `$WORKDIR/bench/` and
build the harness **on top of it** — model load, featurizer,
writer, default flags. Do not rewrite the OSS e2e stack from
the library internals. Record `oss_entry` (path + how you
invoke it) in `bench_config.json`.

Print the resolved OSS config from that script and the BioIR
`get_default_runtime_args(model_source)`. Lock the union in
`$WORKDIR/ref_data/bench_config.json`.

A preset name is not a locked config. Serialize its **expanded,
effective** memory, chunking, offload, kernel, and inference-head
values too. Presets can keep the same name while changing a chunk
ceiling or backend default between pins, which invalidates old timing
and memory rows even when every command-line argument is unchanged.

BioIR `runtime_args` names (`docs/ref/api.md`):

- Boltz-1/2, OpenFold3 — `recycling_steps`, `num_sampling_steps`,
  `diffusion_samples` (OpenFold3 maps those onto cycle / rollout)
- Protenix-v2 — `recycling_steps`, `num_sampling_steps`,
  `diffusion_samples`; this bench locks `recycling_steps=5`
- OpenFold2 / AlphaFold2 — `recycling_steps` only. This bench locks
  `recycling_steps=3` as **exactly three trunk iterations** on both
  sides, with early stopping off. See
  [models/of2.md](models/of2.md#recycle-lock--three-trunk-iterations-no-early-stop).
  Do not leave the multimer default of 20 iters / 0.5 CA-distance
  stop, and do not treat OF2 `recycling_steps` as the AF3-style
  `+ 1` cycle count.

**AF3-style folding config is fixed for this bench.** For Boltz-1,
Boltz-2, and OpenFold3, lock:

```json
{
  "recycling_steps": 3,
  "num_sampling_steps": 200,
  "diffusion_samples": 5
}
```

For Protenix-v2, lock:

```json
{
  "recycling_steps": 5,
  "num_sampling_steps": 200,
  "diffusion_samples": 5
}
```

Use these exact BioIR runtime-argument names. Map the OSS script's
equivalent names onto the same semantics; do not leave its defaults
at another recycle, diffusion-step, or sample count. OpenFold3's
internal `num_cycles` mapping is handled by its `forward`; pass
`recycling_steps=3` to BioIR. For Protenix-v2, pass
`recycling_steps=5` to BioIR and set OSS `model.N_cycle=6`; see
[models/protenix.md](models/protenix.md#runtime-lock-and-recycle-semantics).
Protenix-v2 `build_processor` rejects `return_full_data=False`. Its
registry default is the release's nine recycles, so the lock above is
required.

Also lock `dataset_root`, `dataset_spec`, `dataset_manifest_sha256` and
`dataset_build_sha256` (BUILD.json — a built tree is not identified by a version
string; see [samples.md](samples.md#provenance--record-the-build-not-a-tag)),
precision, seed, `CUDA_VISIBLE_DEVICES`, warmup/repeats, checkpoint ids,
`bioir_config="get_pretrained_config"`, `graph_activation="default"`
(or `"disable_cuda_graphs()"` for eager), enabled roles, `feature_path`
(`processor` | `oss`),
`ost_cmd`, `dockq_cmd`, and `dockq_args`. The processor path seeds via
`init_context.random_seed` on the **feature-generator** stage; the OSS
feature path seeds the OSS featurizer the way the OSS e2e script does.

OpenFold2 / AlphaFold2 use the same `recycling_steps=3` lock as a
fixed trunk-iteration count (not `+ 1`, not early stop). Details
and the two-field pretrained-config delta are in
[models/of2.md](models/of2.md#recycle-lock--three-trunk-iterations-no-early-stop).
For every family, seed and all supported inference knobs must match
between BioIR and OSS.

**BioIR model config.** Use the default constructor:
`ModelCls(model_name=..., config=None)`. The class fills `self.config` from
`get_pretrained_config(model_name)` — the default optimized stack (bf16 where
the class sets it, auto triangle / pairwise backends,
`docs/ref/support-matrix.md`). The processor path omits
`engine_kwargs["config"]` so `FoldingEngine` does the same. Do not construct a
`BaseConfig` by hand. Do not pass OSS dtypes or attention backends into BioIR.
Supported exceptions are documented model-profile overrides for input parity,
fixed recycle counts, or approximate options, and an eager arm's
`disable_cuda_graphs()`. Derive the official pretrained config, change only the
documented fields, and serialize the delta. For approximate options, include the
paired exact arm and fidelity report required by
[SKILL.md](SKILL.md#approximate-options). Boltz-2's custom-template case is
documented in [models/boltz2.md](models/boltz2.md).

**BioIR CUDA graphs.** Every graph region captures by default: lock
`graph_activation="default"` for the BioIR column, or
`"disable_cuda_graphs()"` for an eager arm. Follow
[measurement.md](measurement.md#cuda-graphs) for the eager config, caches,
and the audit; [`docs/ref/api.md`](../../docs/ref/api.md#cuda-graphs)
lists each model's regions and policy.

**OSS scenarios.** Always run **eager**. Protenix-v2 is an explicit
compile exception: its pinned combined child compile was prohibitively
slow and collapsed downstream lDDT/DockQ despite stable measured
forwards. Skip that compile column unless a user requests a reprobe
after relevant pins change; see
[models/protenix.md](models/protenix.md#oss-compile-status--skipped).

For other model profiles, compile the **target submodules once** on a
fresh OSS model (OF2: Evoformer; AF3-style: Pairformer +
DiffusionModule — never the whole module). Use
`torch.compile(module)` exactly: omit `dynamic`, leaving its documented
`None` default so Dynamo begins specialized and automatically widens
after shape changes. Do not set `dynamic=True` or `dynamic=False`, and
do not apply manual dynamic/static input marks.

Probe each child with synthetic direct inputs in the sequence
shape-A, shape-A, shape-B, shape-B
([measurement.md](measurement.md#oss-torchcompile)). The first call at
each new shape may compile as the default policy adapts; each immediate
repeat must not. Reproduce the child's production autocast-enabled or
autocast-disabled context in every direct call. Then run the smallest
real sample and a second
different-shape sample. If the probe raises, produces NaN/Inf, or
recompiles on an immediate repeat, walk the
[retry ladder](measurement.md#compile-retry-ladder) without changing
the default dynamic policy. Child-tensor drift versus eager is
expected under `torch.compile` and is not a probe failure; judge
downstream fitness with lDDT and DockQ. If a retry works, **reuse that same
compiled model** for every in-scope sample. Count compile events on
every warmup and measured forward. Later-sample warmup
specializations are reported, not hidden; measured-forward recaptures
fail the compile column. Never drop eager because compile worked and
never `torch.compile` again per sample.

## Phase 2 — Map the dataset into `oss_data`

The dataset is shaped for BioIR's schema. The OSS loader almost never
reads it as-is, and when it disagrees it tends to skip inputs rather
than raise. Do this mapping **before** writing the harnesses, in one
script, so the harness consumes an index instead of guessing layout.

Write `bench/stage_oss_data.py` (or one script per input kind). For
each in-scope sample and each chain it must:

1. **Resolve** the inputs the manifest already recorded — alignments,
   templates, ground truth — against `$DATASET_ROOT`.
1. **Re-shape** them into what the loader accepts, following
   [msa.md](msa.md#3-learn-how-the-loader-selects-and-keys-files) and
   [templates.md](templates.md): the directory that carries chain
   identity, the filename stems the parser recognizes, the native query
   file, the synthesized alignment.
1. **Symlink** rather than copy, so both sides read the same bytes and
   `$DATASET_ROOT` stays untouched. Generate a file only when the
   parser genuinely cannot read the shipped format, and record what it
   was generated from.
1. **Assert what would otherwise pass silently** — every declared input
   staged, one distinct key per chain, and file depth within any
   parse-time cap the loader applies before the other side's global cap
   ([msa.md](msa.md#failure-modes-seen-in-practice)).
1. **Write `oss_data/index.json`** keyed by sample and chain: source
   path, staged path, and the numbers you asserted on (rows, rows used,
   template ids, selected chain). This is what the report cites and
   what the next run reuses instead of re-deriving.

Point every OSS output directory (preprocessor caches, parsed-structure
dumps) into `oss_data/cache/`. Some preprocessors default to a temp
directory or to the input's parent, which either vanishes between runs
or writes into the dataset.

Rerun the script until it is idempotent and silent. Then check the
staged tree once by hand — one monomer and one multi-chain sample,
confirming distinct per-chain inputs — because a mapping bug here looks
exactly like a fast model.

## Phase 3 — Write the harnesses

Copy the shared helpers from [`harness/`](harness/) into `$WORKDIR/bench/`
([measurement.md](measurement.md)): the timer, GPU telemetry, both scorers, the
result checker, the speedup summary, and the charts. Both sides import the same
files. Then write the harnesses for `feature_path` from Phase 0: both
processor-path blocks, or the OSS feature-path block.

Both harnesses have to turn spec items into their side's input. That
mapping — field names, enable flags, files to synthesize, and the
assertions that prove an alignment or template actually attached — is
in [msa.md](msa.md) and [templates.md](templates.md). Do that work
here, not after the first suspiciously fast row.

### Processor path, BioIR side — `$WORKDIR/bench/run_bioir.py`

```python
from bionemo_ir.data.schemas import InputRequest, MSARecord, Polymer, Template
from bionemo_ir.pipeline.processor.engine_proc import (
    EngineProcessorConfig,
    build_processor,
)
from bionemo_ir.pipeline.stages.configs import (
    FeatureGeneratorStageConfig,
    WriterStageConfig,
)
from scoring import score_dockq, score_lddt  # harness/, copied into bench/

# Omit engine_kwargs["config"] unless the model profile documents one
# minimal input-parity feature flag or this is the eager arm
# (disable_cuda_graphs()). Then derive the official pretrained config and
# record only that delta.
engine_kwargs = {"profile_inference": True}

config = EngineProcessorConfig(
    model_source="<model_source>",
    executor_backend=None,  # serial: small dataset; Ray is for large jobs only
    runtime_args=locked["runtime_args"],
    engine_kwargs=engine_kwargs,
    feature_generator_stage=FeatureGeneratorStageConfig(
        init_context={"random_seed": locked["seed"]},
    ),
    writer_stage=WriterStageConfig(output_path=str(out_dir), format="cif"),
)
processor = build_processor(config)
# Live model for the CUDA-graph audit (measurement.md).
model = processor.get_stage_udf("FoldingEngineStage").folding.engine.model

# sample.request = spec_item_to_request(item, DATASET_ROOT)  # samples.md
for sample in in_scope_manifest:
    record = {"record": sample.request, "__record_id": sample.sample_id}
    torch.cuda.reset_peak_memory_stats()
    rows = list(processor([record]))
    row = rows[0]
    forward_s = float(row["model_inference_time"])
    # Scoring is AFTER the timed window (writer already ran).
    cif = parse_cif_path(row)
    sid = sample.sample_id
    scores = score_dockq(  # decides single_chain / no_gt / unsupported itself
        locked["dockq_cmd"], cif, sample.gt_path, debug_dir / f"dockq_{sid}.json",
        locked["dockq_args"], polymer_types=sample.polymer_types,
        n_chains=sample.n_chains,
    )
    if sample.gt_path:
        scores |= score_lddt(locked["ost_cmd"], cif, sample.gt_path,
                             debug_dir / f"ost_{sid}.json")
```

- **One `InputRequest` per `processor([record])` call.** Do not
  collect the manifest and call `processor(all_rows)`. That
  materializes every sample's features in host RAM and OOM-kills
  before the engine.
- Assert MSA and template files before the call, and verify on the
  featurized batch that both actually attached
  ([msa.md](msa.md#verification),
  [templates.md](templates.md#verification)).
- Read `row["model_inference_time"]` — that **is** `model.forward()`.
- Ignore `time_taken` and `stage_timing_s` for the headline.
- Reset peak memory before the call; record alloc / reserved after.
- Score the written CIF **after** the call, with
  `ost compare-structures` and — on multi-chain samples — `DockQ`.
  Missing GT → both `null` (still a valid latency row). A scorer
  that failed → record the stderr and leave that metric `null`; do
  not invent a score. A single-chain sample is
  `dockq: null, dockq_status: "single_chain"`, never `0.0`
  ([measurement.md](measurement.md#quality-lddt-and-dockq)).

### Processor path, OSS side — `$WORKDIR/bench/run_oss.py`

**Base this on the OSS inference script**, not a from-scratch
forward. Copy it into `$WORKDIR/debug/` (or import its
helpers) and wrap it. Do not edit `$OSS_ROOT`. Reuse its
input conversion, MSA load, featurizer, checkpoint load, and
CIF writer. You only add: one-sample loop, GPU-sync clock
around `model.forward()`, compile-once column, and the same two
scorers after write (`ost compare-structures`, plus `DockQ` on
supported protein interfaces with the locked `dockq_args`).

1. Convert **one** in-scope spec item at a time using **that
   script's** input format **including** every resolved A3M
   (and Boltz CSV). Do not build one OSS batch from the
   whole manifest. Do not invent a second parser. Do not
   read `examples/data/samples/`.
1. Run OSS featurization untimed (that one sample).
1. Move that batch to GPU untimed.
1. **Eager (required).** Time `model.forward()` (warmup 1, measure 1).
   Write the pred CIF, then `ost compare-structures` vs `gt_path`
   (outside the clock). Write `$WORKDIR/results/oss_eager.json`.
1. **Compile once.** Skip this and the compile steps below for Protenix-v2 (see
   the OSS scenarios compile exception) and record the compile column as
   intentionally skipped. On a **fresh** OSS model, wrap the target submodules
   with `compile_oss_hot_modules` **one time** (OF2: Evoformer; AF3-style:
   Pairformer + DiffusionModule). Each target uses `torch.compile(module)` with
   `dynamic` omitted (`None`), no manual shape marks, and unmodified global
   Dynamo shape settings. Do not `torch.compile(model)`.
1. **Fast synthetic compile probe.** Feed deterministic,
   shape-correct synthetic tensors directly to each compiled child at
   two sizes in A, A, B, B order. Initial A and first B may compile;
   immediate repeats must not. Test targets separately, then together.
   Walk compiler retries with this fixture—never rerun the full data
   pipeline merely to reject a compile setting. Synthetic timings are
   not benchmark results.
1. **Real integration probe.** After the synthetic probe passes,
   warmup 1 + measure 1 on the smallest in-scope sample, then on a
   second sample in the **same** `token_bin` when one exists (else the
   next bin). Count Dynamo compiles on both warmups. Probe rules are in
   [measurement.md](measurement.md#oss-torchcompile).
1. **If the probe fails, retry.** Walk the ladder in
   [measurement.md](measurement.md#compile-retry-ladder)
   (real child names, Python-only fences, fewer targets). Every try
   keeps the default `dynamic=None` policy on a fresh model. Do not
   give up after the first adaptation, recapture, or graph break.
1. **Compile column (only after a passing probe or retry).**
   Keep **that same compiled model**. Loop every in-scope
   sample (one batch at a time, warmup 1 / measure 1 per
   sample). Track `warmup_compile_delta` /
   `measure_compile_delta` on every row. Do not compile again.
   Automatic warmup specializations, including same-bin events, stay
   in the result and must be reported. Write
   `$WORKDIR/results/oss_compile.json` only when every published
   measured-forward delta is zero. If the **whole ladder** fails,
   skip this column — eager still stands.

Assert the OSS batch still contains MSA-derived features (non-empty
MSA depth / pair rows). An empty MSA tensor after "successful" load
is a failure.

If OSS must be monkey-patched to expose `forward`, copy the entry
script into `$WORKDIR/debug/`, patch the copy, revert nothing upstream.

### OSS feature path — OSS features, swap the module

Follow [no-pipeline.md](no-pipeline.md). Write three scripts:

1. `$WORKDIR/bench/dump_oss_features.py` (`oss_python`) — call
   the **OSS inference script's** featurizer only; save
   `$WORKDIR/ref_data/oss_features/<id>.pt`.
1. `$WORKDIR/bench/run_oss.py` (`oss_python`) — load dump; run OSS
   eager (one sample at a time); compile target submodules
   **once** on a fresh model; probe; if it works, run **that**
   compiled model on every in-scope dump.
1. `$WORKDIR/bench/run_bioir.py` (`bioir_python`) — load the **same**
   dump, `adapt_oss_batch_to_bioir`, construct BioIR `Protenix`
   (`include_load_weights=True`), then time `model.forward`.

Do not call `build_processor`. Do not re-featurize on the BioIR side.

### Progress reporting — every harness shows a bar

These runs take tens of minutes and a silent process is
indistinguishable from a hung one. Wrap the sample loop in `tqdm` in
**both** harnesses (and any dump / scoring loop), so the user can see
where a run is and whether it is still moving.

Count the bar in **samples**, so it reads `4/17` — the question
being asked is "which sample is it on", and a residue total answers
a question nobody has. Name the sample and its residue count in the
postfix: the manifest is sorted ascending, so seeing the current
size is what tells the reader whether the expensive tail is still
ahead. Treat the ETA as indicative only, since per-sample cost
grows steeply with size.

```python
from tqdm import tqdm

bar = tqdm(total=len(in_scope), unit="sample", dynamic_ncols=True,
           mininterval=5.0, desc="bioir")  # stderr; results go to stdout
for sample in in_scope:
    bar.set_postfix_str(f"{sample['sample_id']} ({sample['residues']} res)")
    ...  # warmup, then the one measured call
    bar.write(f"{sid}: {forward_s:.3f} s | peak {peak_gb} GB | "
              f"lddt {lddt} | dockq {dockq}")
    bar.update(1)
bar.close()
```

Rules:

- **Nothing inside the timing window.** The bar lives in the loop,
  never between the CUDA syncs; `set_postfix_str` before, `update`
  after.
- **`bar.write()`, not `print()`**, for per-sample lines, so results
  and bar do not interleave into garbage.
- **Keep it on when redirected.** A backgrounded run still needs
  progress in the log; `mininterval=5.0` keeps a redirected bar from
  filling the file with refreshes. Do not gate on `isatty()`.
- **Silence an inner framework bar.** A tree driving a Lightning
  `Trainer` prints its own per-batch bar, which fights the outer one
  at one sample per `predict` call; pass `enable_progress_bar=False`.
  It is display-only and changes no compute.
- **Degrade, never crash.** If `tqdm` is missing from an interpreter,
  fall back to plain prints rather than failing the bench.
- Flush per-sample lines (`flush=True`) so a killed run still shows
  what completed.

### Five-sample preparation smoke gate

After dataset mapping and both harnesses are complete, but before any
full sweep, run a **five-sample BioIR + OSS eager smoke benchmark**,
then an OSS `torch.compile` smoke on the same selection when possible
(never for Protenix-v2, which has no compile column).
This is mandatory: it catches an expensive tail failure while still
testing the real checkpoint, featurizers, writers, scorers, timing
window, and compiled dynamic-shape serving path.

Choose exactly five unique samples spanning the residue range:

1. Sort the in-scope manifest by total residues.
1. Start with ranks nearest 0%, 25%, 50%, 75%, and 100%.
1. Keep the smallest and largest samples fixed.
1. If templates exist but none of the three interior selections has
   templates, replace the nearest interior rank with the
   template-bearing sample closest to the median residue count.
1. Likewise ensure a declared-MSA sample and a multichain sample are
   represented when those classes exist, replacing interior ranks
   only.
1. Record the selected IDs, ranks, residue counts, and replacement
   reasons in `bench_config.json`.

Use the final locked configuration, one warmup and one measured
forward per side. Do not shorten diffusion, recycling, MSA, template,
or sampling settings for the smoke. Write separate artifacts:

```text
results/bioir_smoke.json
results/oss_eager_smoke.json
results/oss_compile_smoke.json
```

Build a fresh OSS model for the compile smoke, call
`compile_oss_hot_modules(...)` once with `torch.compile(module)` and
no `dynamic` argument or manual input marks, and first feed its
children synthetic direct inputs at two sizes. Repeat each size
immediately so the probe distinguishes automatic adaptation from
unstable recompilation. Use that fast fixture for the
[compile retry ladder](measurement.md#compile-retry-ladder).
After a synthetic attempt passes, run the two-real-sample integration
probe once, then the five smoke samples. Never compile per sample or
set `dynamic=True` or `dynamic=False`. If the ladder is
exhausted, omit `oss_compile_smoke.json` and report every attempted
target and failure; the eager gate can still pass.

For each selected sample, show the user:

- residues and chain count
- MSA and populated-template status
- BioIR, OSS eager, and valid OSS compile forward latency in seconds
- peak allocated GPU memory
- lDDT and DockQ when applicable
- explicit success or failure

Any missing BioIR/eager output, attachment mismatch, non-finite value,
scoring failure, or OOM fails the gate. Fix it and rerun all five; do
not quietly replace the failing sample. A compile-only failure follows
the retry ladder and is reported as an unavailable optional column,
not relabeled as eager.

After a successful smoke, print the comparison and **ask the user
whether to launch the full in-scope sweep**. Do not start the full run
until they answer. A declined full run leaves the smoke artifacts as
the handoff.

## Phase 4 — Run BioIR

The full run requires user approval after the five-sample smoke gate.

On the OSS feature path, dump features first:

```bash
cd "$WORKDIR" && "$OSS_PYTHON" bench/dump_oss_features.py
```

Then, either path:

```bash
export CUTEDSL_FORCE_CUBIN=1
cd "$WORKDIR" && "$BIOIR_PYTHON" bench/run_bioir.py
```

Warmup and repeats: [measurement.md](measurement.md#warmup-and-repeats).
Write `$WORKDIR/results/bioir_forward.json`. Preserve stdout/stderr
under `$WORKDIR/debug/`.

## Phase 5 — Run OSS e2e

The full run requires the same approval; do not infer it from a
successful BioIR smoke.

```bash
cd "$WORKDIR" && "$OSS_PYTHON" bench/run_oss.py
```

Same manifest, same locked knobs, same GPU. On the OSS feature path,
`run_oss.py` must load the dumps from Phase 4, not featurize a second time.

Always write `$WORKDIR/results/oss_eager.json`. Write
`oss_compile.json` only after a passing compile probe **or a
passing retry**. Exhaust the retry ladder before skipping
compile. A compile failure after that does **not** block the
bench — report eager and every attempt.

If an OSS **eager** kernel crashes, that backend is blocked. Do not
disable the kernel and still label the row `oss_e2e`.

## Phase 6 — Compare and report

First run `check_results.py` on every result file of the comparison and fix
the harness until it reports no problems. Then run `summarize_speedup.py` and
`plot_latency.py` from `$WORKDIR` ([measurement.md](measurement.md)).

Print (not only file):

1. **Setup** — model key, `feature_path`, OSS url / ref / commit
   (must match the pin), both checkpoint
   hashes, GPU name / SM / driver, **power limit (W)**, **max SM
   clock and observed SM clock (MHz)**, `bioir_python` / `oss_python`,
   both torch versions, `isolation`, `ost_cmd`, `dockq_cmd`
   and `dockq_args`, locked
   `runtime_args`, warmup=1 / measure=1, BioIR graph config
   (`graph_activation`, enabled roles, `model_default`, per-region
   token limits and execution paths), OSS compile probe pass/fail
   (two-sample warmup compile
   deltas, targets if pass), `compile_stats.measurement_stable`
1. **Dataset** — `dataset_root`, spec file, MANIFEST digest,
   BUILD.json digest, in-scope count, per-sample MSA
   attached / none and `template_status` (naming any synthesized
   alignment), exclusions with reason
1. **Forward latency** — table from
   [measurement.md](measurement.md#comparison-table)
1. **Speedup** — vs OSS eager, and vs OSS compile when that column
   exists (`oss_s / bioir_s`; `> 1` favors BioIR). Print **geometric
   mean and median** over all in-scope `ok` rows, then the same two
   statistics in residue bins: **short** `< 512`, **medium**
   `512–1024` inclusive, **long** `> 1024`. Name `n` and the sample
   ids in every bin. An empty bin is `n=0`, not an invented ratio.
   See [measurement.md](measurement.md#speedup-aggregates).
1. **lDDT** — OpenStructure scores for BioIR / OSS eager / OSS
   compile. `null` when GT is missing or `ost` failed (cite stderr).
   Never invent a number. On **Boltz-2**, if BioIR and OSS lDDT
   diverge on MSA-bearing proteins, check
   [the known OSS deletion bug](models/boltz2.md#pinned-v221-deletion-divergence)
   before blaming the engine.
1. **DockQ** — interface scores for every multi-chain sample, per
   side, with the chain mapping DockQ chose and the per-side mean
   over the samples it covers. `n/a` on single-chain rows, never
   `0.00`. Name the quality band (incorrect / acceptable / medium /
   high) rather than reading DockQ as a percentage, and treat a
   side-to-side gap as an input-parity question first
   ([measurement.md](measurement.md#quality-lddt-and-dockq)).
1. **GPU** — weights footprint, per-sample peak alloc / reserved,
   any OOM, **power limit / enforced / default (W)**, and
   **observed power draw (W)** plus **SM / graphics / memory
   clocks (MHz)** from in-forward samples (min / max / mean per
   scenario, busy samples only). Name max SM vs observed SM.
   Clocks and power are first-class report fields. An idle
   inventory snapshot is not the run clock.
1. **Commands** — exact invocations and result JSON paths
1. **Charts** — latency vs residue count (sum of all chains), then
   speedup vs residue count. Generate them from the result JSONs
   ([measurement.md](measurement.md#charts-latency-vs-residues)).
   **Show the PNGs in the final reply** (read the image files so
   they render for the human). A path-only mention is not enough.
1. **Open questions** — path to `implementation-notes.md`

Report latency in **seconds** (three decimals) everywhere — table,
charts, and per-sample lines. A Phase 6 that prints only overall
geomean, or GPU name without power and clocks, is incomplete.

## Extending to other model kinds

v1 is a **folding** bench: `InputRequest` → structure CIF →
`ost compare-structures` lDDT plus DockQ on the interfaces,
X-axis = residue count (sum of
all polymer chains). A new
kind of BioIR model needs its own dataset, metric, and size
axis. Reuse the machinery below; do not pretend CIF / lDDT /
DockQ / these folding specs apply.

### Keep as-is for any kind

- BioIR freeze first; isolate OSS; never retune BioIR packages
- Dedicated env per third-party scorer (same idea as
  `$WORKDIR/envs/ost` and `$WORKDIR/envs/dockq`)
- Serial, one sample per `forward` / `processor()` call
- GPU-synced host clock around `model.forward()` only
- Warmup 1, measure 1; no invented numbers
- OSS eager always; compile **target submodules once**, then
  run every sample on that compiled model
- `CUTEDSL_FORCE_CUBIN=1` on every BioIR process
- Allocator left on its default — no `expandable_segments`
- Charts at the end, shown to the human, from result JSONs

**Rewrite per kind** (lock these in `bench_config.json` before
Phase 4):

- **Task / output** — what `forward` returns (coords, affinity
  scalar, embedding, …)
- **Dataset** — not these folding specs unless they actually
  carry the right labels
- **Quality metric** — community tool for that task, not
  hand-rolled. Folding = OpenStructure lDDT + DockQ on
  multi-chain items. Affinity ≠ lDDT.
- **Size axis** — residue count is folding-specific (sum of all
  polymer chains). Pick the
  quantity that drives compute (tokens, atoms, pocket+ligand
  size, …) and plot latency against it
- **Feature loading** — factory (`build_processor`) or OSS features
  - BioIR module (`docs/ref/support-matrix.md` Pipeline column)
- **MSA / template rules** — only if that task uses them

**Known next key:** `boltz-2-affinity` (`Boltz2Affinity`,
Pipeline = No, `docs/ref/support-matrix.md`). Ligand *structure*
on Boltz-1/2 / OpenFold3 is already in v1. Affinity is a
different head: no CIF writer, no `ost`, no folding GT under
`ground_truth/`. When someone benches it, dump OSS affinity features
on the OSS feature path, time `model.forward()`, score with the
affinity metric the OSS paper uses, and chart vs a ligand /
complex size — do not reuse this folding manifest.

If the user names a non-folding key, **stop and extend this
skill** (or a sibling) before writing harnesses. Do not score
affinity with lDDT or fold-shaped `runtime_args`.
