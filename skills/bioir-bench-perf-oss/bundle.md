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

# Compact benches into a portable `bioir-perf` package

**Optional.** Skip this playbook unless the user asks to pack completed
folding benches so they can run on another GPU SKU, in a fresh
container, or from a host that is not the machine that developed the
harnesses.

The parent skill still owns protocol, pins, scoring, and timing.
This file only describes how to fold those locked pieces into a
SKU-portable tree. Do not invent a second protocol.

Current target: `bioir-perf/` (host `bench.sh`, in-container
`bench.py`, per-model dirs). The user may name another folder; keep
the same layout. Do not name the pack `release/`.

## When to run this

Run after at least one model has a working WORKDIR harness (Phases
0–6, or a passing five-sample smoke). Typical ask:

- "compact the benches so they run on other SKUs"
- "pack `boltz2` / `of3` / `protenix` / `of2` into `bioir-perf`"
- "host script pulls docker, downloads the dataset, mounts it, and
  runs all models"

Do **not** start this instead of Phase 0–6. A package with no locked
pin, no MSA/template mapping, or no scorer contract is not a bench.

## Copy is not a sweep — smoke is mandatory

Packing copies harness logic. It is not a full-dataset SKU sweep
and it must not relaunch the original `$WORKDIR` Phase 0–6
forwards.

The packed tree **does** require a GPU smoke of every model before
the pack is done. That smoke is the step that finds harness bugs
(missing `--no-deps` wheels, bare `model(batch)` OOM, un-prepared
Lightning datamodules, RDKit `libXrender`). Do not skip it, do not
treat it as a later SKU-only ask, and do not ship a tree that has
only been statically checked. See [Mandatory packed-tree
smoke](#mandatory-packed-tree-smoke).

If the user says skip re-run of **historical** `$WORKDIR` results,
strip leftover settings (for example
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) from **source
and metadata only**. Leave those historical JSON / markdown files
as-is unless they ask to rewrite them. That exemption does not
cancel the packed-tree smoke.

## What it produces

A host can:

1. Build the dataset if it is missing.
1. `docker pull` the default image
   (`nvcr.io/nvidia/pytorch:26.09-py3` unless overridden).
1. Bind-mount BioIR (source directory or `.whl`), this pack, the
   dataset, a scratch workdir, and a checkpoint cache.
1. Run in-container `bench.py`, which loops every requested model.

Inside the container, each model directory is enough to install,
stage, time BioIR, time OSS, and write the required **JSON and
Markdown** reports.

## Layout

```text
bioir-perf/
  bench.sh                 # host: dataset, docker pull, mount, bench.py
  container_entrypoint.sh  # host UID + passwordless sudo, then exec
  bench.py                 # container: scorers, weights, per-model loop
  install_scorers.sh       # isolated ost + DockQ (+ kalign if missing)
  README.md                # operator usage from this directory
  lib/                     # dataset, scoring, telemetry, processor path,
                           # report, patches.sh
  models/
    boltz2/
    of3/
      patches/             # two input-only template fixes
    of2/
      patches/             # lazy relax import
    protenix/
```

Every supported folding key in this skill gets its own directory:

- `install_deps.sh` — clone the pin, overlay venv, kernels, extra
  weights (AF2 JAX→PT, Protenix CCD)
- `run_bioir.py` — timed BioIR column (processor or OSS feature path)
- `run_oss.py` — timed OSS eager column; `--compile` only where the
  profile allows it
- `report.py` — required `speedup.json` +
  `comparison_report.md`; optional PNG charts

Add extra files only when the model needs them (`stage.py`, OSS feature path
`dump_features.py`, OF3 `runner.yaml`, `patches/`). Do not dump the
whole `$WORKDIR` tree into `bioir-perf/`.

### Patches must ship, and must be committed

A model that needs source changes carries them as git patches in
`models/<key>/patches/`, applied by its `install_deps.sh` right after
the pin is checked out. Copy them from the skill's canonical
[models/misc/](models/misc/); the pack copy and the skill copy must
stay byte-identical.

**Check `git status` shows them before you push the pack.** A
`.gitignore` copied from a model repo very likely carries a blanket
`*.patch` rule for throwaway diffs, which silently swallows these. Add
a negation next to that rule:

```gitignore
*.patch
!models/*/patches/*.patch
```

This is the failure it prevents: the patches change what the OSS
column reads or imports, so a clone without them does not fail — it
benchmarks a different upstream than the profile recorded and reports
the numbers as if nothing were missing.

Apply them through `apply_patches` in `lib/patches.sh`, never with a
bare `git apply ... || true`. That idiom hides both a missing file and
a patch that no longer fits the pin, which is the same silent-skip
outcome. The helper takes a worktree and a list of patch files, and
treats a missing or unfittable patch as fatal so `install_deps.sh`
stops under `set -euo pipefail` before the overlay venv is built:

```bash
# shellcheck source=../../lib/patches.sh
source "${SCRIPT_DIR}/../../lib/patches.sh"
apply_patches "${OSS_ROOT}" \
  "${SCRIPT_DIR}/patches/of3-p2-template-cache-dir.patch" \
  "${SCRIPT_DIR}/patches/of3-p2-template-gap-rows.patch"
```

It has to tolerate an already-patched tree, because `install_deps.sh`
reuses an existing clone and every re-run arrives at one. Detect that
with `git apply --reverse --check`, which is the only thing that
separates "already applied" from "no longer applies" — a plain
forward `git apply` fails identically in both cases, and that is what
pushed the original scripts to `|| true`.

## Host vs container

`bench.sh` runs on the **host**. It must not assume BioIR, torch, or
OSS venvs exist outside the image.

- Resolve BioIR from `--bioir` / `BIOIR_ROOT`: a source directory
  (`pip install -e`, no `--no-build-isolation` — see below) or a
  `.whl` (`pip install` the file). If unset, use the parent of the
  script directory when that parent has `pyproject.toml`. Do not
  hardcode a pack-folder name (`bioir-perf/`, `release/`) in usage
  or in the docker argv. Mount the pack independently of BioIR. Working
  directory is the pack; run `python ./bench.py`.
- Build the dataset into `--dataset-dir` when `spec_full.json` is
  missing: copy the skill's `dataset/*.json` beside it and run
  `rebuild_dataset.py` ([samples.md](samples.md#build)). Mount that
  directory read-only-capable into the container. Required when
  `BIOIR_ROOT` is a wheel (there is no source-tree `benchmarks/`
  default).
- Pull `--image`, pin one GPU (`--gpus device=$GPU_ID`), set
  `NVIDIA_VISIBLE_DEVICES=0` / `CUDA_VISIBLE_DEVICES=0` inside so
  the visible device is always `cuda:0`.
- Run as the **host UID/GID**. `bench.sh` starts the container as
  root only long enough for `container_entrypoint.sh` to create
  that user (if missing) and a passwordless sudoers drop-in
  (`NOPASSWD:ALL`), then drops privileges. Bind-mounted workdir
  files are host-owned; do not `sudo chown` after a run. Inside,
  `sudo -n` works with no password (apt, if needed). Do **not**
  pass docker `--user`: that skips sudoers setup and sudo prompts
  for a password the image user does not have.
- Set `HOME` to `$BENCH_WORKDIR/.container_home` and `USER` /
  `LOGNAME` to the host login. `bench.py` uses `pip install
  --user` when not root so BioIR lands in that HOME, not
  `/usr/local`.
- Mount the pack, BioIR (directory, or the wheel's parent),
  dataset, workdir, and checkpoints. Pass `BIOIR_ROOT`,
  `DATASET_ROOT`, `BENCH_WORKDIR`, `CHECKPOINTS_DIR`,
  `BIOIR_CACHE`, `CUTEDSL_FORCE_CUBIN=1`, `HOST_USER_UID`,
  `HOST_USER_GID`, `HOST_USER_NAME`. Forward `HF_TOKEN` when
  present (OpenFold3 is gated).
- **Mount on short fixed container paths, not the host path.**
  `/bioir`, `/bench` (the pack), `/dataset`, `/work`, `/ckpt`.
  Every in-container env var carries the container path, so
  `BIOIR_ROOT` for a wheel is `/bioir/<name>.whl`. The prefix has
  to be stable because a workdir is full of absolute paths that
  outlive the run: OSS venv shebangs and `pyvenv.cfg`,
  editable-install `.pth` files, and the absolute symlinks
  `fetch_weights.sh` writes into the checkpoint cache. Reusing a
  workdir with `--skip-deps` after the host path moved (another
  SKU, another mount point) only works when the container prefix
  did not move with it.
- **A tree nested inside one already mounted resolves through its
  parent** instead of getting a second mount: the pack inside the
  BioIR checkout is `/bioir/release`, the default dataset is
  `/bioir/benchmarks/dataset`, and a workdir-relative
  checkpoint cache is `/work/checkpoints`. Two container paths
  onto one host tree is a bug, not a convenience — an editable
  install and an OSS clone would each resolve a different prefix
  for the same files. Match on `"$host"/*`, so a sibling like
  `/repo-tools` never absorbs `/repo`.
- Default workdir is host-local scratch, not `/tmp/protenix` or
  `/tmp/openfold2`.

`bench.py` runs **in the container** (or on a machine that already
has the image's Python). It:

1. Installs BioIR if `import bionemo_ir` fails: editable
   `pip install -e` when `BIOIR_ROOT` is a directory, or
   `pip install` when it is a `.whl`. Adds `--user` when the process
   is not root (host-UID remap).

   **Not `--no-build-isolation`.** The bundle is portable by
   definition — it runs wherever the user put it, and cannot assume
   the interpreter already carries `cmake`, `nanobind` and
   `setuptools`. That flag skips installing precisely those, so the
   nanobind extension fails to configure and the user is told to go
   install nanobind by hand before the bundle will run. Let pip build
   in isolation and fetch them; `build-system.requires` names them
   and pins nanobind exactly.
1. Runs `install_scorers.sh` into `$BENCH_WORKDIR/envs`.
1. Fetches public checkpoints via `scripts/fetch_weights.sh`
   (`--source public`) when `BIOIR_ROOT` is a source tree. A wheel
   has no `scripts/`, so it must **not** simply give up and leave a
   cold cache — every `lib.checkpoints` accessor then dies on the
   first model. Fall back to `lib/weights.py`, which downloads
   through the registries the wheel does ship
   (`bionemo_ir.hubs.hf.HF_CHECKPOINTS` and
   `bionemo_ir.hubs.metadata.HF_MODEL_METADATA`), at the commits in
   `bionemo_ir.hubs.integrity.HF_REVISIONS`. Never copy HuggingFace
   URLs into the pack; the wheel already knows them. Protenix-v2 has
   no Hub entry; the user stages it.
   `hf_hub_download` writes a
   `models--org--repo/snapshots/...` tree, so link each asset to
   a flat fallback name `lib.checkpoints` probes (`boltz2_conf.ckpt`,
   `ccd.pkl`, `mols`, `of3-p2-155k.pt`, `protenix-v2.pt`).
   Otherwise an OSS venv, which has no `bionemo_ir`, cannot resolve
   what the BioIR column just downloaded. Run the fetch under
   `BIOIR_PYTHON`.
   A source-tree fetch has a different, explicit staging contract:
   raw public downloads live under
   `$MODEL_CACHE_DIR/public/<family>/`, checkpoints are symlinked to
   `${BIOIR_CHECKPOINTS:-$BIOIR_CACHE/checkpoints}/<model-key>/`,
   and metadata is symlinked to
   `${BIOIR_METADATA:-$BIOIR_CACHE/metadata}/<ENV_VAR>`.
   `lib.checkpoints` must probe those canonical checkpoint and
   metadata directories, while retaining the wheel's flat-name
   fallback. Do not probe only
   `$CHECKPOINTS_DIR/model_cache/<family>/`: public downloads add a
   `public/` component, and `MODEL_CACHE_DIR` may be overridden.
   The fetcher also writes `$MODEL_CACHE_DIR/weights.env`, but
   `bench.py` invokes it as a subprocess, so that file does not alter
   the parent Python environment. Treat it as a shell-caller
   convenience, not as the packed resolver contract.
   AlphaFold2 PyTorch files are **not** published, on that public
   table or on HuggingFace, so OF2 `install_deps.sh` downloads the
   GCS JAX tarball (about 5 GB) and converts it whenever
   `params_model_1.pt` and `params_model_1_multimer_v3.pt` are
   missing. The converter ships in the pack at
   `models/of2/jax_to_pt.py`; it must not be borrowed from a BioIR
   source tree, or a wheel run fails on a cold cache with `OF2
   conversion needs a BioIR source tree`. Every import in it comes
   from the OpenFold checkout `install_deps.sh` already cloned.
1. For each model: `install_deps.sh` → `run_bioir.py` → `run_oss.py`
   → `report.py` (JSON + Markdown).

`--compile` is an extra OSS column for boltz2 / of3 / of2.
**Protenix never compiles** (unstable and slow), even if the flag is
set.

## Environment parameterization

No packed script may hardcode `/tmp/protenix`, `/tmp/openfold2`, or a
developer checkout path. Drive everything from env:

- `BIOIR_ROOT` — BioIR source directory (dev) or a `bionemo-ir`
  wheel file
- `DATASET_ROOT` — the built dataset root
- `BENCH_WORKDIR` — shared scratch (venvs, features, results)
- `MODEL_WORKDIR` — per-model scratch (`$BENCH_WORKDIR/<model>`)
- `CHECKPOINTS_DIR` / `BIOIR_CACHE` — weight cache
- `BIOIR_PYTHON` / `OSS_PYTHON` — isolated interpreters
- `OST_CMD` / `DOCKQ_CMD` / `KALIGN` — scorer and template binaries
- `CUTEDSL_FORCE_CUBIN=1` — every BioIR process

Set `PYTHONPATH` to the package root (the directory that contains
`lib/` and `bench.py`) so `from lib...` works. Per-model helpers
(`common.py`, `stage.py`) stay on that model's directory via
`sys.path` insert.

## Shared `lib/`

Start `lib/` from the skill's [`harness/`](harness/) (timer, GPU telemetry,
both scorers, result checker, speedup summary, charts), then keep one copy of:

- spec → manifest / `InputRequest` / DockQ contract / smoke IDs
- OpenStructure lDDT + DockQ
- GPU inventory and in-forward NVIDIA-SMI sampling
- CUDA Event `model.forward()` timing
- `build_processor` serial runner
- CUDA-graph region audit
  ([measurement.md](measurement.md#audit-cuda-graph-routing-not-cache-emptiness))
- speedup geomean/median, Markdown table; optional PNG charts

Smoke IDs stay locked with [samples.md](samples.md) and the model
profiles. Do not pick a new five-sample set while packing.

- Boltz-2 / OpenFold3: `R1117`, `T1152`, `7uww-assembly1`, `T1125`,
  `8ic7-assembly1_A_B`
- Protenix: percentile + template-replacement rule from the five-sample
  gate (`R1117`, `R1136`, `T1118v1`, `8a8o-assembly1_A_B`,
  `8IC7_A_B`)
- OF2 monomer: `7pv5-assembly1`, `T1137s1`, `7uww-assembly1`,
  `8wnj-assembly1`, `8uxt-assembly1`
- OF2 multimer: `7wr3-assembly1_A_C`

## Per-model compact rules

Copy the **locked** pin, not `main`. Follow the matching profile:

- Boltz-2 — [models/boltz2.md](models/boltz2.md). Processor path. Tag
  `v2.2.1`. `recycling_steps=3`, `num_sampling_steps=200`,
  `diffusion_samples=5`. `trunk.use_templates_v2=True`. MSA cap
  8192. `use_kernels=True`.
- OpenFold3 — [models/of3.md](models/of3.md). Processor path. Tag `0.4.3`
  commit `0bb17be5199846e806b6347b6e17c6249c88ff1b`. Ship the two
  input-only template patches. Predict preset **without** `low_mem`.
  cuEq + DeepSpeed evoformer.
- Protenix-v2 — [models/protenix.md](models/protenix.md). Processor path. Pin
  `2475421477ab414b571149ad4a875c390ff8a35d`. BioIR
  `recycling_steps=5` → OSS `model.N_cycle=6`. Seed 101.
  `LAYERNORM_TYPE=torch`. Never `torch.compile`.
- OpenFold2 / AF2 model-1 — [models/of2.md](models/of2.md). Processor path.
  `aqlaboratory/openfold` `v2.2.0` commit
  `e938c184a291bf053af3b14c1e3e8bb29aee57e2`. Protein-only (monomer
  spec + protein-protein). `max_recycling_iters=2`,
  `recycle_early_stop_tolerance=-1`. BF16 ExtraMSA +
  Evoformer wrapper. NumPy `np.string_ = np.bytes_` alias.

Shared still applies: serial one-sample, GPU-sync only, warmup 1 /
measure 1, BioIR graph regions on by default, both scorers
isolated, CUDA 12 extras remapped to CUDA 13,
DeepSpeed evoformer from source with CUTLASS 3.6.0 (not BioIR
CUTLASS 4), never `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
See [environment.md](environment.md) and
[measurement.md](measurement.md).

## Required outputs: JSON and Markdown

The packed bench's required operator artifacts are **JSON and
Markdown**. PNG charts and canvases are optional extras. A missing
matplotlib must not fail `report.py`, and a smoke pass must not
require a PNG.

Per model, under `$BENCH_WORKDIR/<model>/results/`:

- Column JSON from the timed runners:
  `bioir_forward.json` or `bioir_smoke.json`,
  `oss_eager.json` or `oss_eager_smoke.json`, and
  `oss_compile.json` or `oss_compile_smoke.json` when that column
  ran (not Protenix).
- `speedup.json` — geomean and median, overall and residue bins,
  plus GPU inventory and in-forward power/clocks.
- `comparison_report.md` — GPU identity (name, SM, driver, CUDA,
  memory, UUID), power caps, max clocks vs observed SM, busy-poll
  power/clocks per column, quality, per-sample table.

OF2 writes the same JSON + Markdown pair under
`results/monomer/` and `results/multimer/`.

`bench.py` must fail the model if those JSON and Markdown files
are missing after `report.py`. The Markdown **GPU** section is
required (not only the device name): power limit / enforced /
default / max, max SM vs observed SM mean, and in-forward busy
min/mean/max power and clocks per column. Optional PNGs
(`latency_vs_residues.png`, `speedup_vs_residues.png`) may be
written when Agg matplotlib imports.

## How to compact

1. Confirm the user wants a portable package and the destination
   folder (default `bioir-perf/`).
1. List models. Default is every folding key this skill supports:
   `boltz2`, `of3`, `protenix`, `of2`. Drop a key only if the user
   names a subset.
1. Extract from each `$WORKDIR` **and** the model profile: pin,
   patches, `runtime_args`, stage mapping, compile policy, scorer
   contract. Prefer the profile when a WORKDIR script drifted.
1. Rewrite paths to the env vars above. Delete leftover allocator
   exports.
1. Copy each model's patches into `models/<key>/patches/` and confirm
   `git status` lists them. See [Patches must ship, and must be
   committed](#patches-must-ship-and-must-be-committed) — a stock
   `*.patch` ignore rule drops them and the bench then silently times
   unpatched upstream.
1. Keep NVIDIA SPDX Apache-2.0 headers on every source file.
1. Write operator `README.md` with commands that run **from this
   directory** (`./bench.sh --smoke --compile`), not
   `./bioir-perf/bench.sh`.
1. `chmod +x` the host and install scripts.
1. Encode the packed-tree notes below in `bench.py`,
   `install_deps.sh`, and the per-model runners. Do not leave them
   as operator folklore.
1. Static-check the tree (ruff, rumdl, shellcheck / shfmt,
   `python -m compileall`). Run `--help` on `bench.py` and each
   `run_*.py` / `report.py`.
1. From a cold temporary cache, create source-fetch-shaped checkpoint
   and metadata symlinks and call every `lib.checkpoints` accessor
   with all model-specific path variables unset. Boltz checkpoint,
   CCD, molecules, OpenFold3, and Protenix must all resolve before
   the packed-tree GPU smoke. This catches a fetch that succeeds but
   stages into directories the runners never probe.
1. **Mandatory:** run the packed-tree smoke on GPU for every model
   (BioIR, OSS eager, OSS `torch.compile`). Packing is not done
   until it passes. A full-dataset SKU sweep remains a separate
   ask.

## Packed-tree reproduction notes

These are not a second protocol. They are the checks the first
packed `bioir-perf` smoke needed so a later SKU can rerun without
rediscovering the same failures.

### Shared runtime

- Image: `nvcr.io/nvidia/pytorch:26.09-py3`. Run `bench.py` with
  that interpreter, or `./bench.sh` from the host.
- Every BioIR process: `CUTEDSL_FORCE_CUBIN=1`.
- Never set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  Unset it if the shell inherited it. `bench.py` must pop it from
  child env.
- `PYTHONPATH` must include the package root so `from lib...`
  works.
- **Root in the container may not be able to write the workdir.**
  A bind mount from an NFS export with `root_squash` maps root to
  nobody, so `mkdir` and `chown` return `EACCES` on a directory
  the host user owns and can write. Observed as
  `mkdir: cannot create directory
  '/bench/bench_work/.container_home/.local': Permission denied`
  — the host-side `mkdir` of `.container_home` had just
  succeeded. So create `HOME/.local/bin` **on the host** in
  `bench.sh`, where the owning user runs, and keep the container
  side best-effort (`|| true` plus a warning). Never let root-side
  workdir setup abort the run; the target user writes it fine
  after the drop.
- Scorers live under `$BENCH_WORKDIR/envs`. Set `OST_CMD`,
  `DOCKQ_CMD`, and `KALIGN` (kalign is used by OF2 and Protenix
  template staging).
- The NGC image is headless. RDKit Draw (Boltz, OF3, Protenix)
  wants `libXrender.so.1`. Prefer `sudo -n apt-get install
  libxrender1` when the host-UID remap granted passwordless
  sudo. When apt is still blocked, prepend the OpenStructure
  conda prefix the package already installs:
  `$BENCH_WORKDIR/envs/ost/lib`. That is the scorer env, not an
  unrelated conda tree. `bench.py` should do this prepend after
  `install_scorers.sh`.
- `--sample` **replaces** the results JSON. It does not merge.
  Use `--smoke` for the five-sample set. `--warmup 0` times the
  first forward (CUDA-graph capture, if any, is inside the
  headline). Default warmup is still 1.
- `lib.measure.measure_forward` always wraps the timed callable in
  `torch.inference_mode()`. OpenFold3's eval path is unsafe without
  it. Boltz / OF2 / Protenix keep their production autocast
  **inside** the callable.

### Processor path, BioIR side

Serial engine UDFs are lazy. Before inference, get the live model through
`processor.get_stage_udf("FoldingEngineStage").folding.engine.model` and
bind the graph audit to it. The processor
reports `model_inference_time` from inside the engine. Sample power and
clocks only while its model forward executes, as on the OSS side.

### Boltz-2

- Overlay install is `--no-deps`. Also install
  `chembl_structure_pipeline`, `dm-tree`, `einx`, `fairscale`,
  `gemmi` or `process_inputs` is unimportable.
- `process_inputs(data=...)` takes a **list of YAML paths**, not a
  directory.
- Official checkpoints pickle OmegaConf. Load with
  `weights_only=False` and `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.
- Identical protein sequences must share one MSA path (Boltz
  schema). Homodimer `8ic7-assembly1_A_B` is `id: [A, B]` with one
  `.a3m`.
- Rewrite MSA paths to `$DATASET_ROOT/casp15/msa/<basename>` when
  that file exists. Example-tree YAML often still points at stale
  `/.../T1152_*.csv`.
- OSS `parse_mmcif` cannot resolve ligand CCDs in some dataset
  templates (`T1152` / `11CI.cif` `A1DEZ`). Stage a **polymer-only**
  CIF copy under `$MODEL_WORKDIR/oss_data/templates/<id>/` and hash
  polymer atoms before and after so the copy does not change the
  polymer.
- Production precision is bf16 autocast around `predict_step`.

### OpenFold3

- Overlay `--no-deps` extras that 0.4.3 imports at predict time:
  `lmdb`, `pdbeccdutils`, `func_timeout`, `ijson`,
  `memory_profiler`, `kalign-python`, `boto3`, `awscrt`.
- Official Lightning `predict` calls `prepare_data()` then
  `setup("predict")` before `predict_dataloader()`. A harness that
  only calls `predict_dataloader()` hits `InferenceDataModule` with
  no `datasets_by_mode`.
- Timed OSS call must run under `torch.inference_mode()` (already
  in `measure_forward`). `.eval()` does not disable autograd.
  OpenFold3 sets
  `inplace_safe = not (self.training or torch.is_grad_enabled())`.
  A bare `model(batch)` keeps the graph, skips in-place eval
  kernels, and OOMs mid-size samples: `7uww-assembly1` (635
  residues) peaked at 77.7 GiB allocated on an 80 GB H100. With
  `inference_mode`, the same GPU holds the largest smoke id
  (`8ic7-assembly1_A_B`, 1734 residues) at about 30 GiB allocated /
  43 GiB reserved.
- Official inference is Lightning `precision: 32-true` (true
  fp32). Training YAMLs use `bf16-mixed`. Do **not** wrap OF3 OSS
  in `autocast(bf16)`. DeepSpeed evoformer may cast QKV to bf16
  inside the kernel and cast back; that is not a global bf16
  policy. BioIR OF3 trunks are bf16 by design — a known stack
  difference, not a harness bug.
- Predict preset only. Do not add `low_mem`. Lock `chunk_size:
  1024`, cuEq triangle kernels, DeepSpeed evoformer, Triton
  triangle off. CUDA 13: no OF3 `cu12` extra.
- `HF_TOKEN` is required for the gated P2 checkpoint.

### Protenix-v2

- Never `torch.compile`, even if `--compile` is set.
- Pin `2475421477ab414b571149ad4a875c390ff8a35d`. Prefer cloning
  `3rdparty/protenix` then checking out the pin.
- Import `update_inference_configs` from `runner.inference` on
  this pin. Later upstream moved it to `protenix.model.protenix`;
  that name is absent here and the OSS column will fail at import.
- **The OSS CCD cache build needs `pdbeccdutils` and runs in the
  overlay venv, not the BioIR interpreter.**
  `prepare_ccd.py` calls upstream
  `scripts/gen_ccd_cache.py`, which imports it. Install it
  `--no-deps` (as OF3 does): its floors `gemmi>=0.6.6` and
  `scipy>=1.14.1` would otherwise pull the overlay off gemmi 0.6.5
  and scipy 1.13.1 and change what the data pipeline parses. Those
  same floors are why it can never go in the BioIR interpreter,
  which pins both. The import chain also reaches `PIL`, `networkx`,
  and `scipy`, already present via rdkit, torch, and the overlay
  list.
- Run `prepare_ccd.py` under `$OSS_PYTHON` only. The BioIR column
  runs `build_processor`, reads CCD topology from Biotite's bundled
  CCD, and resolves conformers through `BIOIR_PROTENIX_CCD_MOLS_PATH`;
  point it at the same `common/components.cif.rdkit_mol.pkl` the OSS
  column uses.
- `install_deps.sh` curls `common/components.cif` (490 MB) and
  owns it. `prepare_ccd.py` only checks that it is there before
  scanning it — it must not try to symlink it into place, because
  source and target are the same path and a real file makes that
  an unconditional `FileExistsError`.
- The overlay list is **not** the full set `protenix.data` needs.
  A clean container from a wheel also needs `ml_collections`,
  `scikit-learn`, and `optree` in the OSS interpreter, while
  `fair-esm` is stubbed rather than installed. A dev box hides
  this because its base interpreter already has them. See
  [models/protenix.md](models/protenix.md).
- Never pass `return_full_data=False` to the BioIR column;
  `build_processor` rejects it for Protenix.
- `LAYERNORM_TYPE=torch`, seed 101. BioIR `recycling_steps=5`
  corresponds to OSS `model.N_cycle=6`.

### OpenFold2

- `$MODEL_WORKDIR/oss/openfold` must be the `aqlaboratory/openfold`
  clone at `e938c184a291bf053af3b14c1e3e8bb29aee57e2`. A nested
  `oss/oss` symlink breaks imports.
- Selective bf16: ExtraMSA + Evoformer `PrecisionWrapper` only
  (OpenFold `--precision=bf16`). Not full-model bf16.
- `np.string_ = np.bytes_`. No CUDA graphs. Convert AF2 JAX
  `params_model_1` and `params_model_1_multimer_v3` on first
  `install_deps.sh`.

## Mandatory packed-tree smoke

This is the critical verification of the generated code. Do it on
the GPU used to pack (or the first target SKU) **before** calling
the pack complete.

Run **all** folding keys: `boltz2`, `of3`, `protenix`, `of2`.

Required columns:

- BioIR
- OSS eager
- OSS `torch.compile` for boltz2 / of3 / of2

**Protenix never compiles** (unstable and slow), even if
`--compile` is set. A Protenix smoke is BioIR + OSS eager only.

Protocol: `--smoke --compile`, serial one-sample, GPU-sync only,
warmup 1 / measure 1 unless the user set `--warmup`.
`CUTEDSL_FORCE_CUBIN=1`. Do not set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

From the package directory (host):

```bash
./bench.sh --smoke --compile
```

Already inside the image, from the pack directory (`bench.py`
applies the same defaults). `BIOIR_ROOT` is a source directory or
a `.whl`:

```bash
export CUTEDSL_FORCE_CUBIN=1
# Dev: /path/to/bionemo-ir   Wheel: /path/to/bionemo_ir-*.whl
export BIOIR_ROOT=/path/to/bionemo-ir
export DATASET_ROOT=/path/to/dataset
export BENCH_WORKDIR=/tmp/bioir_bench
export CHECKPOINTS_DIR=/tmp/bioir_bench/checkpoints
export BIOIR_CACHE="$CHECKPOINTS_DIR"
unset PYTORCH_CUDA_ALLOC_CONF
python ./bench.py \
  --models boltz2,of3,protenix,of2 --smoke --compile
```

Do not run `python "$BIOIR_ROOT/.../bench.py"` — the pack is not
inside BioIR when `BIOIR_ROOT` is a wheel. Reuse an existing
overlay with `--skip-deps --skip-weights`. OpenFold3 still needs
`HF_TOKEN`. OF2 converts AF2 JAX params on first `install_deps.sh`
when `BIOIR_ROOT` is a source tree.

A pass writes, for every model, **JSON and Markdown**:

- `$BENCH_WORKDIR/<model>/results/bioir_smoke.json`
- `$BENCH_WORKDIR/<model>/results/oss_eager_smoke.json`
- `$BENCH_WORKDIR/<model>/results/oss_compile_smoke.json`
  (not Protenix)
- `$BENCH_WORKDIR/<model>/results/speedup.json`
- `$BENCH_WORKDIR/<model>/results/comparison_report.md`

OF2 writes those under `results/monomer/` and
`results/multimer/`. Confirm every expected JSON and Markdown file
exists and every sample `status` is `ok` (quality `dockq_status`
may still be `unsupported` / `single_chain`). PNG charts are
optional. If any column or required report file fails, fix the
packed harness, record the fix in the notes above, and re-run that
model. Do not drop the failing column.

`--sample` overwrites the results JSON; it does not merge. Use
`--smoke` for this gate.

## After that smoke, on another SKU

From the package directory:

```bash
./bench.sh --smoke --compile
./bench.sh --models boltz2,of3 --smoke --compile
./bench.sh --image nvcr.io/nvidia/pytorch:26.09-py3 --workdir /data/bioir_bench
```

Required results land under `$BENCH_WORKDIR/<model>/results/` as
JSON and Markdown (`speedup.json`, `comparison_report.md`, plus
the per-column JSON). Both files must include GPU inventory and
in-forward power/clocks. PNG charts are optional.
