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

# OpenFold2 / AlphaFold2 OSS benchmark profile

This is the model-specific companion to
[`bench-perf-oss`](../SKILL.md). Generic environment isolation,
manifest construction, timing, scoring, synthetic compile probing,
charts, and reporting stay in the parent skill. This profile locks
the OpenFold `v2.2.0` pin, the AlphaFold2 **model 1** checkpoints
(monomer and multimer), protein-only sample filters, recycle
semantics, MSA/template staging, and the Evoformer compile target.

Use this profile for BioIR keys `alphafold2_1` (monomer) and
`alphafold2_multimer_1` (protein–protein). Do not use OpenFold-trained
`openfold2_*` finetuning checkpoints for this bench. Nucleic acids and
ligands are out of scope: AlphaFold2 / OpenFold2 fold protein chains
only (`docs/ref/support-matrix.md`).

Default WORKDIR: `/tmp/openfold2`.

## Source pin

The OSS reference is:

- repository: `https://github.com/aqlaboratory/openfold.git`
- tag: `v2.2.0`
- commit: `e938c184a291bf053af3b14c1e3e8bb29aee57e2`
- checkout: `$WORKDIR/oss/openfold`
- official entry: `run_pretrained_openfold.py`

Clone the tag and verify it before installing:

```bash
git clone --branch v2.2.0 --depth 1 \
  https://github.com/aqlaboratory/openfold.git \
  "$WORKDIR/oss/openfold"
git -C "$WORKDIR/oss/openfold" describe --tags --exact-match
test "$(git -C "$WORKDIR/oss/openfold" rev-parse HEAD)" = \
  "e938c184a291bf053af3b14c1e3e8bb29aee57e2"
git -C "$WORKDIR/oss/openfold" apply \
  "$REPO/skills/bench-perf-oss/models/misc/of2-lazy-relax-import.patch"
```

Do not clone `main`. Consume the tree via `PYTHONPATH=$OSS_ROOT` — do
not `pip install` the package. OpenFold's `setup.py` compiles CUDA
extensions that are not the recommended inference kernel for this
bench.

## Checkpoints — AlphaFold2 model 1, JAX converted to PyTorch

This bench uses **DeepMind AlphaFold2 model 1**, not OpenFold-trained
weights.

The OpenFold repo publishes two public weight sources. Do not mix
them:

- **AlphaFold2 JAX params** (this bench) —
  `scripts/download_alphafold_params.sh` fetches
  `https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar`
  (GCS; CC BY 4.0). That tarball is the OF2-documented AF2 source.
- **OpenFold-trained PyTorch params** —
  `scripts/download_openfold_params.sh` copies
  `s3://openfold/openfold_params/` (`finetuning_*.pt`). Those are a
  different training run. Do not load them as `alphafold2_1` /
  `alphafold2_multimer_1`.

Convert JAX → PT with `jax_to_pt.py` in the OSS interpreter (it
imports `openfold.config`, `openfold.model.model.AlphaFold`, and
`openfold.utils.import_weights.import_jax_weights_`). **Do not rename
the `.npz` first** — the converter derives the JAX version from the
basename (`params_model_1.npz` → `model_1`;
`params_model_1_multimer_v3.npz` → `model_1_multimer_v3`).

**A packed tree carries its own copy of that converter**, at
`models/of2/jax_to_pt.py`, and `install_deps.sh` downloads the GCS
tarball and runs it whenever the two `.pt` files are missing. Do not
reach into a BioIR source tree for
`examples/folding/openfold2/jax_to_pt.py`: every import in it comes
from the pinned OpenFold checkout that `install_deps.sh` has already
cloned, nothing in it is BioIR-specific, and depending on the source
copy is what made a wheel-based run fail with `OF2 conversion needs a
BioIR source tree` on a cold `CHECKPOINTS_DIR`. Consequently OF2 needs
no `BIOIR_ROOT` at all.

```bash
# The 2022-12-06 tarball stores npz files at the archive root,
# not under params/. Extract without renaming.
tar -xf alphafold_params_2022-12-06.tar \
  params_model_1.npz params_model_1_multimer_v3.npz

PYTHONPATH="$WORKDIR/oss/openfold" "$OSS_PYTHON" \
  models/of2/jax_to_pt.py \
  --jax_path params_model_1.npz \
  --config_preset model_1 \
  --output_dir "$WORKDIR/checkpoints"

PYTHONPATH="$WORKDIR/oss/openfold" "$OSS_PYTHON" \
  models/of2/jax_to_pt.py \
  --jax_path params_model_1_multimer_v3.npz \
  --config_preset model_1_multimer_v3 \
  --output_dir "$WORKDIR/checkpoints"
```

Stage the converted files under the BioIR names and point both
interpreters at the same bytes:

- Monomer: BioIR key `alphafold2_1`, JAX stem `params_model_1`,
  OSS `--config_preset model_1`, env `ALPHAFOLD2_1_CKPT`
- Multimer: BioIR key `alphafold2_multimer_1`, JAX stem
  `params_model_1_multimer_v3`, OSS `--config_preset
  model_1_multimer_v3`, env `ALPHAFOLD2_MULTIMER_1_CKPT`

BioIR remaps the OpenFold state-dict through
`bionemo_ir/models/openfold2/convert.py`. OSS loads the same `.pt`
with `import_openfold_weights_` (`openfold_checkpoint_path`). One
offline strict load in each interpreter is the weights-preflight
gate. Never `strict=False`.

Record SHA-256 of both converted files in `bench_config.json`.

## Protein only

AlphaFold2 / OpenFold2 do not fold RNA, DNA, or ligands. Filter
before writing the manifest:

- **Monomer role** — every item in `spec_monomer.json` (all protein
  monomers).
- **Multimer role** — `spec_full.json` items whose
  `chemical_class` is `protein-protein` and whose polymers are all
  `polymer_type="protein"`. The release set is
  `7spq-assembly1_A_A-2`, `7wr3-assembly1_A_C`,
  `8a8o-assembly1_A_B`, `8ic7-assembly1_A_B`.
- Exclude `ligand`, `protein-ligand`, RNA, and DNA rows from
  `spec_full.json`. Those are in-scope for Boltz / OpenFold3, not
  for this profile.

Write two manifests (or one manifest with a `role` field). A harness
must assert the in-scope set for the role it is running. Do not mix
monomer weights with a multi-chain FASTA or multimer weights with a
single chain.

## Recycle lock — three trunk iterations, no early stop

OF2 does **not** use the AF3-style `recycling_steps + 1` mapping.
BioIR `OpenFold2.forward` takes `recycling_steps` as a **cap on the
feature recycle axis**:

```python
num_iters = feed_dict["aatype"].shape[-1]  # max_recycling_iters + 1
if recycling_steps is not None:
    num_iters = min(num_iters, recycling_steps)
```

OSS OpenFold uses the same axis with no runtime cap:

```python
num_iters = batch["aatype"].shape[-1]
```

Multimer defaults are not comparable: BioIR and OSS both set
`max_recycling_iters=20` and `recycle_early_stop_tolerance=0.5`, so
a run can exit after one cycle. For this bench both sides run a
**fixed** trunk count.

Lock:

- `recycling_steps = 3` — exactly three Evoformer + structure-module
  cycles on both sides
- `max_recycling_iters = 2` on both feature pipelines so the recycle
  axis length is `2 + 1 = 3` (do not generate four slices and drop
  one)
- `recycle_early_stop_tolerance = -1` on both models (negative
  disables the AF2Complex CA-distance stop)

BioIR: derive `get_pretrained_config(model_name)`, change only those
two fields, pass that config, and pass
`runtime_args={"recycling_steps": 3}`. Record the delta; this is the
documented exception to `config=None`.

OSS: after `model_config(preset, use_deepspeed_evoformer_attention=True)`,
set the same two fields (or write them in `--experiment_config_json`).
Do not leave the multimer preset at 20 iters / 0.5 tolerance.

Assert `output["num_recycles"] == 3` (BioIR) / three iteration
completions (OSS) on every measured forward. A row that stopped
early is invalid.

## Environment and kernels

Use an overlay venv (`python3.12 -m venv --system-site-packages
$WORKDIR/venv_oss`) so OSS reuses the container CUDA 13 torch.
Do not install OpenFold's `pytorch=2.5` / `pytorch-cuda=12.4` pins.
Do not `pip install flash-attn`. Do not install PyTorch Lightning
either — see [Do not install PyTorch Lightning — stub
it](#do-not-install-pytorch-lightning--stub-it).

Recommended OSS inference kernel is DeepSpeed Evoformer attention.
Build it from source into `$OSS_PYTHON` as in
[deepspeed-evoformer.md](../deepspeed-evoformer.md). Prefer the OF2
`environment.yml` pin `deepspeed==0.14.5`; if that pin cannot import
torch 2.12 / CUDA 13, record the fallback SHA and keep
`evoformer_attn` only. Enable it:

```python
config = model_config(
    preset,
    use_deepspeed_evoformer_attention=True,
)
```

Pass `--use_deepspeed_evoformer_attention` when wrapping
`run_pretrained_openfold.py`. Verify
`config.globals.use_deepspeed_evo_attention is True` and
`installed_ops["evoformer_attn"] == 1` before any forward.

OpenFold `v2.2.0` does not integrate cuEquivariance. Record
`cueq_triangle: false`. BioIR uses its default CuTeDSL / auto
triangle backends under `CUTEDSL_FORCE_CUBIN=1`.

kalign is required on the OSS template path. Prefer the distro
`kalign` package. If apt is blocked, bioconda `kalign2` is the
fallback. Record `kalign_binary_path`.

Do **not** install OpenMM or pdbfixer for this bench. Score
unrelaxed CIFs. See [Skip Amber relaxation](#skip-amber-relaxation).

`dm-tree==0.1.6` in OF2 `environment.yml` is an sdist that needs
bazel and fails on Python 3.12. Install a manylinux wheel
(`dm-tree>=0.1.8`).

### Do not install PyTorch Lightning — stub it

`openfold/utils/script_utils.py` imports
`convert_zero_checkpoint_to_fp32_state_dict` from
`pytorch_lightning.utilities.deepspeed` at module scope, so
`load_models_from_command_line` is unreachable without the name and
`run_oss.py` dies on `ModuleNotFoundError: pytorch_lightning`.

It is unreachable code for this bench. The only call sits inside
`if os.path.isdir(path)` — the sharded DeepSpeed-checkpoint branch —
while the pack passes the converted `params_model_1.pt` **file**, so
loading takes the plain `torch.load` path and
`import_openfold_weights_`. That single import is also Lightning's only
appearance in the 45 OpenFold modules this harness reaches.

So `models/of2/common.py` registers stub `pytorch_lightning`,
`pytorch_lightning.utilities`, and
`pytorch_lightning.utilities.deepspeed` modules at import time, with a
`convert_zero_checkpoint_to_fp32_state_dict` that raises. Installing
Lightning instead would pull its whole stack, and its own torch
expectations, into an overlay that reuses the container's CUDA 13
torch — for one symbol nothing calls. Give the two package-like stubs a
`__path__` so an unrelated submodule import still fails as a normal
`ModuleNotFoundError`.

Nothing else in that chain is missing. `flash_attn` and `deepspeed` are
both probed with `importlib.util.find_spec` in
`openfold/model/primitives.py` and imported only when present, so the
absent flash-attn is fine and `_flash_attn` raises a clear error if a
config ever selects it. The rest — `Bio`, `ml_collections`, `modelcif`,
`numpy`, `scipy`, `torch`, `tree` — is covered by the overlay list plus
the container base.

### NumPy 2 and native-multimer output compatibility

OpenFold `v2.2.0` references the NumPy 1.x `np.string_` alias in
`openfold/data/msa_pairing.py` while padding paired heteromer MSAs.
NumPy 2 removed that alias in favor of `np.bytes_`, so a heteromer
fails during featurization even when monomer and homomer probes pass.
Install this compatibility alias in the benchmark harness immediately
after importing NumPy and before heteromer featurization:

```python
if not hasattr(np, "string_"):
    setattr(np, "string_", np.bytes_)
```

This restores only the removed type name; it does not change MSA
content or model math. Do not downgrade the BioIR environment or
install DockQ's NumPy pin into the OSS environment to work around it.
Record the NumPy version and alias in `bench_config.json`, and include
heteromer `7wr3-assembly1_A_C` in the multimer preflight.

The same pin's `prep_output()` gap-removal loop predates native
`DataPipelineMultimer` features. Native multimer features already
reset `residue_index` per chain and carry chain identity in `asym_id`;
running the old loop on them creates negative ModelCIF residue
indices. For multimer output only, suppress that gap-removal loop
while leaving the processed batch's `residue_index` and `asym_id`
unchanged. This is untimed postprocessing. Do not add artificial
residue gaps to the model input.

`openfold.model.primitives` imports `attn_core_inplace_cuda` at
module load even when inference uses PyTorch softmax. Build it
in-place with `python setup.py build_ext --inplace` under
`PYTHONPATH=$OSS_ROOT` **before** `jax_to_pt.py`; do not
`pip install openfold`. The converter imports `AlphaFold` and will
fail with `ModuleNotFoundError: attn_core_inplace_cuda` otherwise.

### Precision — selective BF16

Use OpenFold's selective `--precision=bf16` inference policy:
convert only `ExtraMSAStack` and `EvoformerStack` to BF16, cast
their FP32 inputs to BF16, and cast their BF16 outputs back to
FP32. Keep the rest of the OSS model in FP32. This is separate
from `--use_deepspeed_evoformer_attention`, which casts only the
attention Q/K/V and biases.

OpenFold `v2.2.0` predates the inference `--precision` CLI flag.
Reproduce the later OpenFold implementation from
`openfold/utils/precision_utils.py` at
`be2ec1841f16c966c65ae0e7599ebbadc725757d` in the harness without
patching the pinned source:

```python
model.extra_msa_stack = PrecisionWrapper(model.extra_msa_stack, "bf16")
model.evoformer = PrecisionWrapper(model.evoformer, "bf16")
```

The wrapper recursively casts only floating-point tensors; integer
and boolean arguments are unchanged. Apply it after loading weights
and before `torch.compile`. Do not use whole-model autocast or cast
the structure/confidence modules. Record FP32 wrapper boundaries
and BF16 inner-module parameter dtypes in every result artifact.

### Skip Amber relaxation

`--skip_relaxation` skips Amber on the **run**, but it does not
skip the **import**. `openfold.utils.script_utils` imported
`openfold.np.relax.relax` at module load, and that module imports
`pdbfixer` and OpenMM. Wrapping `load_models_from_command_line`
then dies with `ModuleNotFoundError: pdbfixer` before any
forward.

Apply the recorded patch
[`models/misc/of2-lazy-relax-import.patch`](misc/of2-lazy-relax-import.patch)
after checkout so `relax` is imported only inside
`relax_protein()`:

```bash
git -C "$WORKDIR/oss/openfold" apply \
  "$REPO/skills/bench-perf-oss/models/misc/of2-lazy-relax-import.patch"
```

Do not install OpenMM/pdbfixer to "make the import work". Do not
stub a `.so`. The harness still passes `--skip_relaxation` and
writes unrelaxed ModelCIF. Revert the patch after the run if the
checkout must return to a clean pin; record the patch path in
`bench_config.json`.

## Processor path — BioIR

```python
config = OpenFold2.get_pretrained_config(model_name)  # alphafold2_1 or alphafold2_multimer_1
config.max_recycling_iters = 2
config.recycle_early_stop_tolerance = -1

EngineProcessorConfig(
    model_source=model_name,
    executor_backend=None,
    runtime_args={"recycling_steps": 3},
    engine_kwargs={"profile_inference": True, "config": config},
)
```

One `InputRequest` per `processor([record])`. Headline latency is
`row["model_inference_time"]`. Seed via
`feature_generator_stage.init_context.random_seed`. The OF2 `trunk`
region captures by default from recycle 1
([CUDA graphs](../measurement.md#cuda-graphs)).

## OSS entry

Wrap `run_pretrained_openfold.py`; do not reimplement featurize +
`forward` from internals. Reuse its FASTA parse, `DataPipeline` /
`DataPipelineMultimer`, `FeaturePipeline`, checkpoint load, and
CIF writer (`--cif_output --skip_relaxation`). Add only: one-sample
loop, GPU-sync clock around `model(...)`, compile-once on
`evoformer`, and the two scorers after write.

Lock:

- `--config_preset model_1` or `model_1_multimer_v3`
- `--openfold_checkpoint_path` pointing at the converted `.pt`
- `--use_precomputed_alignments $WORKDIR/oss_data/msa/<role>/`
- `--use_deepspeed_evoformer_attention`
- `--skip_relaxation --cif_output` (unrelaxed ModelCIF; see
  [Skip Amber relaxation](#skip-amber-relaxation))
- `--data_random_seed` identical to BioIR
- `--experiment_config_json` with `max_recycling_iters=2` and
  `recycle_early_stop_tolerance=-1`
- `--max_template_date 9999-12-31` and `--kalign_binary_path`
- CLI `main()` **rejects** `--openfold_checkpoint_path` in
  multimer mode. Call `load_models_from_command_line` directly so
  the converted `.pt` loads for both roles.

## Map the dataset into `oss_data`

BioIR keeps reading `$DATASET_ROOT`. OSS is alignment-directory
driven. Stage with `bench/stage_oss_data.py` and write
`oss_data/index.json`.

### MSAs

OSS `DataPipeline._parse_msa_data` loads every `*.a3m` and every
`*.sto` except `uniprot_hits` / `hmm_output`. Identity is the
**parent directory**, not the filename stem.

Monomer, per sample (FASTA tag is the spec `chain_id`; alignment
identity is that parent directory):

```text
oss_data/queries/<sample_id>.fasta
oss_data/msa/<sample_id>/<chain_id>/
└── uniref90_hits.a3m   # symlink of the spec unpaired A3M
```

Multimer, per FASTA description (use the spec `chain_id` as the
FASTA tag so it matches `alignment_dir/<desc>/`):

```text
oss_data/msa/<sample_id>/<chain_id>/
├── uniref90_hits.a3m   # unpaired A3M
└── uniprot_hits.sto    # paired A3M rewritten as Stockholm
```

Heteromers raise if `uniprot_hits.sto` is missing. Convert each
declared `paired_msas` A3M to Stockholm (query row first). Homomers
(`7spq-assembly1_A_A-2`, identical sequences) skip pairing in OSS;
still stage the unpaired A3M.

Dataset paired A3M headers are UniRef-style (`UniRef100_...`).
OSS pairing extracts species only from UniProt ids
`tr|ACC|ENTRY_SPECIES` (species 1–5 characters). Rewrite Stockholm
sequence names to `tr|R#####|PAIR_P####` with a **shared per-row
species** so OSS reconstructs the shipped row alignment. Unpaired
files stay as `uniref90_hits.a3m` (A3M is accepted). Convert
A3M→STO by dropping lowercase insertions.

Ground-truth CIFs come from spec field `gt`, not `{id}.cif`.
Interface ids such as `7spq-assembly1_A_A-2` point at
`ground_truth/7spq-assembly1.cif`.

Do not run JackHMMER / HHblits / MSA server. Assert parsed MSA
depth greater than one whenever the spec listed a file.

### Templates

OSS is **alignment-driven**. `--use_custom_template` is not usable
here: it requires chain A and identical sequence length. Synthesize
a query→template alignment instead.

Monomer: write `pdb70_hits.hhr` that `parse_hhr` accepts. Hit names
must match `^[a-zA-Z0-9]{4}_[a-zA-Z0-9.]+` because
`_get_pdb_id_and_chain` splits on that. Stage each CIF under
`oss_data/templates/mmcif/<pdb_id>.cif` using a four-character stem
(rename in the staging tree if the dataset stem is not four
characters; record the mapping).

Multimer: write `hmm_output.sto` in the same per-chain alignment
directory. Description lines must match the hmmsearch parser
(`pdb_chain/start-end ... protein length:N`).

Turn off search-oriented gates:

- `max_template_date=9999-12-31`
- empty `release_dates_path` / `obsolete_pdbs_path`
- `max_hits` equal to BioIR `max_templates` (4)

If `_assess_hhsearch_hit` still drops a caller-supplied template
(`DateError`, `DuplicateError`, `AlignRatioError`), apply a
recorded, reverted, input-side patch so every supplied template
passes the prefilter. Do not change model math.

Verify by counting populated template slots on the featurized batch
**before** `model.forward()`, same cap on both sides. BioIR builds
direct-CIF features in
`bionemo_ir/pipeline/models/openfold2/template_logic.py` (Kalign
realign + top-k). OSS must attach the same CIF/chain per slot.

Template-bearing monomer ids: `T1104`, `T1106s1`, `T1112`,
`T1137s1`, `T1114s3`. The four protein–protein multimers in this
release declare no templates.

## Compile — Evoformer only

On a fresh OSS model, install the selective BF16 wrappers first,
then compile the actual Evoformer inside its eager precision
boundary once:

```python
model.evoformer.model = torch.compile(model.evoformer.model)
```

`dynamic` omitted (`None`). No axis marks. No
`torch.compile(model)`.

Keep the OpenFold precision wrapper eager so it performs the
FP32→BF16 input and BF16→FP32 output casts. Confirm its inner
`model` slot is `torch._dynamo.eval_frame.OptimizedModule` with
BF16 parameters.

Do **not** add another `nn.Module` whose `forward` only calls
`torch.compile(child)` (the parent skill's
`DefaultCompiledModule`). That extra Python frame trips Dynamo on
this pin + torch 2.12: after `recompile_limit`, the graph rewrite
raises `SyntaxError: 'True' is an illegal expression for
augmented assignment`.

Probe with synthetic direct Evoformer inputs in A, A, B, B order
**only if that call stays finite**. OF2's Evoformer plus the
DeepSpeed kernel and chunk tuner can hit Dynamo's `recompile_limit`
and the same inplace graph `SyntaxError` on synthetic two-size
feeds even with a correctly compiled child. That is not a bench
failure: record `synthetic_ok: false` and use the first two real
samples as the integration probe. Keep that compiled model for
every in-scope sample of that role. Count Dynamo frames on warmup
and measure. Warmup adaptations are reported; measured-forward
recaptures fail the compile column.

Monomer and multimer are different `nn.Module` graphs. Compile each
role's model separately.

If a later multimer-shape warmup hangs in Inductor's process pool
(GPU idle, parent waiting on a futex, and every compile worker asleep),
terminate the whole process group and restart the **entire role** on a
fresh model with:

```bash
export TORCHINDUCTOR_COMPILE_THREADS=1
```

This serializes code generation during untimed warmup; it does not
change `torch.compile`'s shape policy, generated model graph, or the
measured-forward timing window. Record the retry and compile-thread
count in `bench_config.json`. Do not combine early rows from the hung
process with later rows from the retry. Warmup compile deltas remain
reported, and every measured-forward delta must still be zero.

## Two-role layout

```text
$WORKDIR/ref_data/
  bench_config.json          # both roles, both checkpoints
  sample_manifest_monomer.json
  sample_manifest_multimer.json
$WORKDIR/results/
  monomer/                   # bioir, oss_eager, oss_compile, speedup.json
  multimer/
```

Harnesses take `--role monomer|multimer`. Five-sample smoke uses
the monomer spec (fifteen protein monomers, residue span). After
that smoke passes, ask before the full monomer sweep **and** before
the four-sample multimer sweep. Do not silently skip either role.

## Quality

lDDT via `ost compare-structures`. DockQ on the four protein–protein
multimers (`--short --n_cpu 1 --allowed_mismatches 1`). Monomers are
`dockq_status: single_chain`. No `--small_molecule` — this profile
has no ligands.

## Phase 6

Follow the parent skill. Per role, print GPU power and SM clocks
from the inventory plus in-forward samples, and speedup vs eager /
compile as geomean **and** median overall and in residue bins
short `< 512`, medium `512–1024`, long `> 1024`. Write those
numbers only under `$WORKDIR/results/{role}/`. Do not copy run
figures into this profile.
