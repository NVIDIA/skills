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

# Protenix-v2 benchmark profile

Use this profile with [`../SKILL.md`](../SKILL.md). Protenix-v2 is a
folding model with a native BioIR pipeline (`ProtenixFactory`,
Pipeline = Yes in `docs/ref/support-matrix.md`). Benchmark it on
**Processor path**, like OpenFold3 and Boltz-2: each side featurizes itself.

```text
dataset spec -> BioIR build_processor (native featurizer) -> BioIR forward
dataset spec -> pinned OSS runner/inference featurizer   -> OSS forward
```

The native pipeline passed Level 1 feature equivalence against the
upstream v2.0.0 featurizer
([`docs/advanced/validate-data-pipeline.md`](../../../docs/advanced/validate-data-pipeline.md)),
so Processor path needs no shared OSS feature dump.

## Source and checkpoint lock

- OSS source: repository gitlink `3rdparty/protenix`.
- Required pin: ByteDance Protenix tag `v2.0.0`, commit
  `2475421477ab414b571149ad4a875c390ff8a35d`.
- Model name on both sides: `protenix-v2`.
- Checkpoint: the v2 weights are proprietary and BioIR has no Hub fallback.
  Obtain `protenix-v2.pt` from the Protenix authors and stage it yourself.
- BioIR construction: serial
  `build_processor(EngineProcessorConfig(model_source="protenix-v2", ...))`.
  Start from the optimized pretrained config; pass an explicit config only for
  the
  [documented approximate option](#approximate-option-frame-aware-prediction-reuse-default-on)
  or an eager arm.
- Set `PROTENIX_V2_CKPT` to the staged checkpoint. Record the
  resolved path and SHA-256 and strict-load the same file in both
  interpreters before timing.
- BioIR reads CCD topology from Biotite's bundled CCD and never
  downloads the conformer pickle; set `BIOIR_PROTENIX_CCD_MOLS_PATH`.
  Pinned OSS reads `$OSS_ROOT/common/components.cif` and its
  `.rdkit_mol.pkl`. Point both sides at the same pickle and record both
  SHA-256 values; a different CCD or pickle changes ligand features.
  Biotite's CCD differs from the release `components.cif` on heavy-atom
  names of `NOT` and one leaving flag of `OYA`; avoid both in samples.
- Require the hashes listed in `docs/ref/model-weights.md`
  ("Protenix metadata"):
  `components.cif` `bb31ae5c…` and `components.cif.rdkit_mol.pkl`
  `d1cfb71f…`. A truncated pickle loads without error and silently
  changes the reference conformer of every component it lacks.
- Default scratch work directory for this profile: `/tmp/protenix`.

The checkpoint is wrapped as `{"model": state_dict}` and its keys may have
a `module.` prefix. Pinned OSS strips the prefix in
`InferenceRunner.load_checkpoint`; BioIR's hub loader does the equivalent
before its strict conversion. Never use `strict=False`.

## Runtime lock and recycle semantics

BioIR's public argument is a **recycle count**:

```python
num_cycles = recycling_steps + 1
```

Pinned OSS Protenix's `model.N_cycle` is the number of trunk loop
iterations. Therefore map:

```text
OSS model.N_cycle = BioIR recycling_steps + 1
```

The skill baseline (`recycling_steps=5`) consequently means OSS
`model.N_cycle=6`.

For this benchmark, lock:

```json
{
  "bioir": {
    "recycling_steps": 5,
    "num_sampling_steps": 200,
    "diffusion_samples": 5
  },
  "oss": {
    "model.N_cycle": 6,
    "sample_diffusion.N_step": 200,
    "sample_diffusion.N_sample": 5
  }
}
```

This is five recycles / six total trunk cycles on both sides. Setting OSS
`N_cycle=5` would run one fewer cycle and is not parity.
`ProtenixFactory.get_default_runtime_args()` follows the release
(`recycling_steps=9`), so pass all three values explicitly.

Also lock seed 101, one model seed, `use_msa=true`,
`use_rna_msa=false`, `msa_pair_as_unpair=true`, `use_template=true`,
one discarded warmup, and one measured forward. The release declares no
RNA A3Ms; do not enable RNA-MSA search or invent one. There is no Amber
relaxation stage in this path.

BioIR reads its data settings from `ProtenixConfig`
(`bionemo_ir/models/protenix/config.py`): `use_msa`, `use_rna_msa`,
`msa_pair_as_unpair`, `max_msa_rows`, `max_paired_per_species`,
`use_templates`, and `max_templates`. The pretrained defaults
(`use_rna_msa=False` included) match the upstream inference defaults.
Serialize the resolved values in `bench_config.json`; do not override
them.

Both sides resample MSA rows on every recycle. BioIR does it through
`msa_module_config.sample_rows=True`, seeded from the request seed. The
OSS release also applies Monte Carlo dropout to the recycled pair state
for a random 40% of requests; BioIR does not. That is a documented,
accepted accuracy difference, so a single seed can diverge per sample.
Compare lDDT and DockQ over seeds or over the sample set, not one seed
per sample.

Seed the OSS featurizer the way `runner/inference.py` does: call
`seed_everything(seed)` before iterating the dataloader. The release
draws each residue's random reference pose from the global RNG, so an
unseeded harness folds a different pose, and often a different
structure, in every process. BioIR seeds featurization and sampling from
the request seed and reproduces across processes.

A per-sample gap from one seed is not yet a finding. On CASP15 `T1152` without
templates at seed 101, OSS scored GlobalDockQ 0.44 and BioIR 0.23, yet over ten
seeds the top-ranked sample averaged 0.341 for OSS, 0.362 for OSS without MC
dropout, and 0.374 for BioIR; about a quarter of diffusion samples dock the
GT-like interface on either side. Before reporting a per-sample lDDT or DockQ
gap, rerun that sample with at least five seeds per side and compare every
diffusion sample, not only the top-ranked one.

## Feature parity

On 66 CASP15 targets with unpaired and paired A3Ms, CCD ligands, ions,
RNA, and DNA, but no templates, the native featurizer matched pinned
v2.0.0 tensor for tensor at the model boundary, with three expected
exceptions:

- `ref_pos`: each side draws its own random pose per residue. Compare
  intra-residue distances, not coordinates. Swapping `ref_pos` between
  the two feature dicts reproduces the other side's structures bit for
  bit, so the pose is the only input that changes the output.
- SMILES ligand conformers: the release embeds them without a seed.
- 25 OSS-only keys the BioIR model never reads. The model rebuilds
  `d_lm`, `v_lm`, and `pad_info` itself; the rest are raw or
  bookkeeping fields.

After a change to the native featurizer, repeat that comparison before
timing. Build the OSS dict with `get_inference_dataloader()` and
`update_input_feature_dict()`, and the BioIR dict by running the
processor's stages up to `FoldingEngineStage` and reading `__data__`.

## Approximate option: frame-aware prediction reuse (default on)

The option was evaluated with 100, 150, 200 and 400 steps on Protenix; the
200-step lock above is the one this profile uses, so the BioIR column runs with
`edm_sampling_config.frame_reuse_period = 4` (about 3-12 percent faster, less
on large inputs), and a second, exact BioIR column (`frame_reuse_period = 0`)
runs for the paired comparison. Follow
[Approximate options](../SKILL.md#approximate-options):
construct from `Protenix.get_pretrained_config("protenix-v2")`, set only that
field in each column, record both values in `bench_config.json`, and report
the paired lDDT/DockQ difference. The harness's `--exact` switch sets it to 0.
See `docs/ref/sampling.md` for what it does.

## OSS inference path and effective config

Base the harness on `3rdparty/protenix/runner/inference.py`:

- Build config from `configs_base`, `data_configs`, `inference_configs`,
  and `model_configs["protenix-v2"]`, then apply the locked command-line
  values.
- Use `get_inference_dataloader()` for featurization and `DataDumper` for
  CIF/confidence output.
- For each OSS forward, preserve the native call boundary from
  `InferenceRunner.predict`: move the already-featurized batch to CUDA
  before timing, enter the resolved BF16 autocast context, and call the
  top-level model. Do not time the dataloader, H2D, or `DataDumper`.
- Call `update_inference_configs(configs, N_token)` before each OSS
  sample so the pin's dynamic chunking policy is applied. Serialize the
  expanded thresholds and the resulting per-sample chunk values.
- Use the recommended OSS kernels and inference switches:
  `triangle_attention=cuequivariance`,
  `triangle_multiplicative=cuequivariance`, `dtype=bf16`,
  `enable_tf32=true`, `enable_efficient_fusion=true`, and
  `enable_diffusion_shared_vars_cache=true`.
- Installing cuEquivariance is not enough: the two triangle config
  fields above must resolve to `cuequivariance` in the locked config.
- Install the matching `cuequivariance-torch` frontend as well as the
  base and CUDA-operator wheels. On SM90, pinned Protenix reports that
  cuEquivariance triangle attention is unavailable and uses its native
  reference implementation; record the requested and effective kernels
  separately. Triangle multiplicative update still uses cuEquivariance.
- On a CUDA-13 container use the CUDA-13 cuEquivariance artifacts; do
  not install the upstream requirements file's `*-cu12` wheel.
- Keep `LAYERNORM_TYPE=fast_layernorm` only when its pinned extension
  builds and probes successfully in the isolated OSS interpreter.
  A deliberate `LAYERNORM_TYPE=torch` compatibility column must be
  named and approved as a fallback; do not silently relabel it as the
  recommended OSS path.

With torch `2.12.0a0+5aff3928d8.nv26.05`, the pinned fast LayerNorm
extension builds but fails the real forward with `Cannot access data
pointer of Tensor that doesn't have storage`. The compatible smoke/full
column is therefore named `oss_*_torch_layernorm`, uses
`LAYERNORM_TYPE=torch`, and preserves every other OSS setting. Keep the
failed native probe in the benchmark notes; do not publish it as a
latency row.

The upstream `requirements.txt` pins torch 2.7.1 and
`cuequivariance-ops-torch-cu12==0.8.0`. Keep OSS package changes out of
the BioIR interpreter. Record the actual CUDA-13 remap and all version
differences in `bench_config.json`.

## Convert `spec_full.json` to Protenix input

All 17 items in `$DATASET_ROOT/spec_full.json` are in scope. BioIR
converts each item to an `InputRequest`
([`../samples.md`](../samples.md#load-path-processor)) with every listed A3M
and template attached. For OSS, convert each item to the
AlphaFold-Server-like list accepted by
`runner/inference.py`:

- `protein` -> `proteinChain`
- `rna` -> `rnaSequence`
- `dna` -> `dnaSequence`
- `ccd_ligand` -> `ligand` with `CCD_<code>`
- preserve explicit chain ids as an `id` list and set `count` to its
  length
- attach each protein's resolved `unpairedMsaPath` and
  `pairedMsaPath`; attach an RNA `unpairedMsaPath` only when the spec
  declares one

The spec sometimes stores one chain id as a string and sometimes as a
one-element list. Normalize both to a list before computing `count`.
Write one generated query JSON per sample plus an aggregate index.
Symlink every A3M from the release into `oss_data/msa/`; never copy or
edit the release.

Pinned Protenix's inference MSA loader takes one unpaired and one paired
A3M path per protein entity. Assert:

- every path declared by the spec is present in the generated query;
- every staged symlink resolves to the recorded release path;
- `N_msa > 1` for declared-MSA samples unless the source A3M genuinely
  contains only the query row;
- `msa`, `has_deletion`, `deletion_value`, `profile`, and
  `deletion_mean` are present and finite;
- paired multichain inputs retain distinct per-chain entries.

## Caller-supplied templates

`T1152`, `T1118v1`, and `T1158v1` carry caller-supplied template
**structures**.

BioIR takes them directly: map each spec template to
`Template(path=..., format="cif", chain_id=...)` on `Polymer.templates`
in spec order. The pipeline accepts CIF and PDB, aligns each supplied
template without search-hit filters, and fills slots in the supplied
order up to `ProtenixConfig.max_templates`.

`chain_id` names a CIF chain by `label_asym_id` and a PDB chain by its
author id. Pinned OSS keys template chains by author id. When a CIF's
label and author ids differ, give BioIR the label id and keep the author
id in the OSS hit name. A mismatched id selects another chain without
notice, or drops the template with only a warning.

Pinned Protenix's JSON accepts a template-search alignment
(`templatesPath`) instead and then resolves an mmCIF by the hit name.
Stage this deterministic OSS adapter:

1. Symlink every supplied CIF into
   `oss_data/templates/mmcif/<lowercase-entry>.cif`.
1. Select the requested `chain_id`; when it is null, choose the
   best-aligning protein chain in that supplied CIF and record the
   identity/alignment used for the choice.
1. Generate one `hmmsearch.a3m`-compatible hit per supplied template.
   Its description must satisfy pinned
   `HmmsearchA3MParser` (`<entry>_<chain>/start-end`,
   `mol:protein`, and `length:<n>`), and the query entity receives that
   file as `templatesPath`.
1. Use the real Kalign binary when Protenix realigns query and template.
   Do not synthesize coordinates or a dummy template tensor.
1. Scope the input-side compatibility patch to this explicit caller
   list: bypass search-oriented release-date, duplicate, coverage, and
   keep/drop filters; set the processing cutoff far enough in the
   future to accept the supplied CIF. Do not alter template feature
   math, cap, or model compute.

The profile's cap is four on both sides, matching pinned inference and
the BioIR default `max_templates=4`. Verify attachment per protein chain
from `template_pseudo_beta_mask` on each side's featurized batch: count
rows with any populated token in that chain and require
`min(n_supplied_for_chain, 4)`, with the same entry and chain per slot.
A global nonzero mask is insufficient, especially for `T1152`, which
supplies the same structure separately to two protein entities.

Record every generated alignment, chosen chain, symlink target, and
filter override in `oss_data/index.json` and
`implementation-notes.md`.

## OSS data-pipeline dependencies

Importing `protenix.data` is what the OSS featurizer costs, and the
dependency set is larger than the overlay installs. On a dev box the
base interpreter happens to carry the rest, so this only surfaces in a
clean container running from a wheel, where `run_oss.py` dies on
`ModuleNotFoundError` before it reaches a single sample.

The OSS column imports `protenix.data.inference.infer_dataloader`,
`protenix.model.protenix`, `runner.inference`, and `runner.dumper`. The
BioIR column imports none of them.

Three are hard imports and must be installed:

- `ml_collections` — `protenix.config.config`, which `make_oss_config`
  needs.
- `scikit-learn` — `KDTree` in `protenix/data/core/featurizer.py`, real
  featurization work that cannot be stubbed.
- `optree` — module scope in `protenix/model/utils.py`.

Install them into the OSS interpreter. A `--user` install in the base
is invisible inside the overlay venv, so install into the venv itself.
Do not add them to the BioIR interpreter.

Three more appear in a static scan but need nothing:

- `lmdb` — `protenix/utils/file_io.py` imports it under
  `if TYPE_CHECKING`, plus one function-local use.
- `orjson` — function-local inside `try/except ImportError`, with a
  `json` fallback.
- `deepspeed` — function-local in `protenix/model/triangular/layers.py`,
  reached only by `DS4Sci_EvoformerAttention`, which this profile does
  not select.

### Skip fair-esm

Do not install `fair-esm`. Nothing in this profile computes ESM
embeddings:

- The `protenix-v2` block in `configs_model_type.py` declares no `esm`
  key, so it inherits `configs_base.py`'s
  `{"enable": False, "model_name": "esm2-3b", "embedding_dim": 2560}`.
- `infer_dataloader.py` reads `esm_info.get("enable", False)` and puts
  every `ESMFeaturizer` construction and call behind that flag.
- The only contact with the package is a module-scope
  `from esm import FastaBatchedDataset, pretrained` in
  `protenix/data/esm/compute_esm.py`, which `protenix/data/esm/__init__.py`
  imports, which `infer_dataloader.py` pulls in for `ESMFeaturizer`.

So the requirement is module surface, not functionality. `common.py`
registers a stub `esm` module at import time — before any entry point
adds the OSS tree to `sys.path` — and its symbols raise when used, so a
config that turns ESM on gets an explicit error rather than features
computed without embeddings.

**The stub must raise `AttributeError` for `_`-prefixed and dunder
names.** `inspect.getmodule` walks `sys.modules` and probes `__file__`
on every entry, so a stub that answers dunders breaks unrelated work:
observed as a `torch.library.custom_op` registration inside `import
torch` failing with `protenix used esm.__file__`. Only real ESM symbols
get a refusing placeholder.

Importing this chain the first time also JIT-builds Protenix's
`fast_layer_norm_cuda_v2` extension with `nvcc`, which took about two
and a half minutes. That pause is the extension build, not a hang.

## BioIR harness

Use the generic Processor path harness
([`../workflow.md`](../workflow.md#processor-path-bioir-side--workdirbenchrun_bioirpy))
with `model_source="protenix-v2"`:

- `executor_backend=None`, one `processor([record])` per sample;
- `runtime_args` = the three locked BioIR values above;
- `engine_kwargs={"profile_inference": True}`, with no
  `engine_kwargs["config"]`;
- `FeatureGeneratorStageConfig(init_context={"random_seed": 101})`;
- CIF writer under `$WORKDIR/results/`.

`build_processor` rejects `return_full_data=False` for Protenix: the
postprocessor needs per-atom confidence. It also rejects
`consume_input_features=True` when `keep_input_features` omits the
postprocessor's inputs; the registry defaults set both. The default
`compact_confidence=True` already sets `ProtenixConfig.compact_output`.

The default pretrained config captures graph regions automatically. Keep
the model policies and audit each forward
([CUDA graphs](../measurement.md#cuda-graphs)).

Expect one trunk capture per input shape of at most 1024 tokens,
replayed on every later recycle and request of that shape. The trunk
samples MSA rows at a fixed depth, `min(N_msa, 16384)` rows with the
unsampled ones masked, so the sampled count never changes the graph
key. Above 1024 tokens the trunk runs
eagerly and counts `out_of_range`. A trunk that captures on every
recycle is a build without fixed-depth sampling: it runs slower than
eager and segfaults in `cuGraphLaunch` after a few same-size requests
(six on `T1152`).

For large pair representations, lock and serialize the benchmark's
semantic-preserving chunk policy: pair transition, diffusion pair
transition, and pair-weighted averaging at 256; outer-product mean at
64; contact probability at 256; each with `min_size=1024`. Set it on
`CHUNK_REGISTRY` (`bionemo_ir/_torch/utils/auto_chunk.py`) in the BioIR
harness process before `build_processor`, and apply the same values for
smoke and full runs. These chunks avoid materializing the corresponding
full pair intermediates and do not change features, model weights, or
requested sampling work.

## OSS compile status — skipped

Do not run or publish a `torch.compile` column for pinned Protenix-v2.
OSS eager with `LAYERNORM_TYPE=torch` is the required reference.

The faithful combined `pairformer_stack` + `diffusion_module` probe
passed synthetic A, A, B, B stability and every five-sample measured
forward captured zero new frames. It was nevertheless unusable:

- probe plus smoke startup/compilation took about 81 minutes on H100;
- warmup-only shape adaptations continued through the 1,734-token row;
- mean lDDT collapsed from `0.7108` eager to `0.1308` compiled;
- mean DockQ over three applicable rows collapsed from `0.7709` to
  `0.2310`.

Those are downstream-fitness failures, not publishable latency results.
An isolated Pairformer retry was stopped because compilation remained
too slow. Do not continue the target-isolation or retry ladder unless
the user explicitly asks to reprobe after a Protenix, PyTorch, or
cuEquivariance pin changes.

Keep one transferable fixture lesson: a future direct probe must call
`pairformer_stack` under CUDA BF16 autocast and `diffusion_module` with
autocast disabled (`skip_amp.sample_diffusion=true`). Calling
Pairformer outside its production autocast context creates an
FP32-query/BF16-value SDPA mismatch before the compiler is tested.

## Output and scoring

- OSS: pinned `DataDumper` with `sorted_by_ranking_score=true` writes
  `<sample>_sample_0.cif` as the highest-ranked diffusion sample. Score
  that file.
- BioIR: the Protenix postprocessor writes the diffusion sample with the
  highest `ranking_score`. That CIF is the comparable file. It writes
  every declared CCD ligand, including residue-named codes such as
  `TYR` and `2MG`, as a HETATM non-polymer, so DockQ needs no scorer-only
  HETATM copy.

Score after timing. Run OpenStructure lDDT on every
sample with ground truth. Run DockQ on supported protein-protein and
protein-small-molecule interfaces; use `--small_molecule` for ligand
complexes and record `unsupported_chain_types` when DockQ cannot score
an RNA/DNA interface. Single-chain structures use
`dockq=null, dockq_status="single_chain"`.

Two DockQ outcomes get a status instead of a score:

- `Need at least two chains in the two inputs`: the native has fewer
  chains than the model, such as a one-chain ground truth for a homomer
  query. Record `no_native_interfaces` on both sides.
- Exit status 0 with no JSON written: DockQ could not map model chains
  to the native. Record `failed` with its stderr; never read a stale
  JSON from an earlier run.

## Five-sample smoke selection

For the 17-row release manifest, the percentile rule initially selects
indices 0, 4, 8, 12, and 16 after sorting by residue count. Replace the
middle interior row with the template-bearing row nearest the median.
The locked smoke set is:

- `R1117` — smallest; RNA-ligand; no declared MSA
- `R1136` — lower quartile; RNA-ligand
- `T1118v1` — template-bearing replacement nearest the median
- `8a8o-assembly1_A_B` — upper quartile; paired protein MSA
- `8ic7-assembly1_A_B` — largest

Store ranks, residue counts, and the replacement reason in
`bench_config.json`. Run the full locked 200-step / five-sample
diffusion settings in smoke; do not shorten them.
