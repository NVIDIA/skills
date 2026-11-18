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

# OpenFold3 P2 benchmark profile

This is the model-specific companion to the generic benchmark workflow.
Use the generic procedures for [environment isolation](../environment.md),
[MSA mapping](../msa.md), [template mapping](../templates.md), and
[measurement](../measurement.md); use this file for OpenFold3's exact pin,
configuration, data contracts, patches, and validation values.

## Supported variant and source pin

The BioIR `openfold3` key ships `of3-p2-155k.pt`. The latest official
OpenFold3 release that supports this checkpoint is:

- tag: `0.4.3`
- commit: `0bb17be5199846e806b6347b6e17c6249c88ff1b`
- declared checkpoint range: `>=0.4,<0.4.4dev0`

Pin the submodule to that commit. Do not use v0.5 merely because it is a
newer package release: v0.5 defaults to the private OpenBind checkpoint
and defines a different model.

```bash
git submodule update --init --recursive 3rdparty/openfold-3
test -z "$(git -C 3rdparty/openfold-3 status --porcelain)"
git -C 3rdparty/openfold-3 switch --detach 0.4.3
test "$(git -C 3rdparty/openfold-3 rev-parse HEAD)" = \
  "0bb17be5199846e806b6347b6e17c6249c88ff1b"
```

If the submodule is dirty, stop before switching. Preserve that state in
a user-approved stash or separate worktree; never reset it. Record both
the parent gitlink and the selected checkpoint-compatible commit.

## P2 model contract

A valid P2 configuration has all of these properties:

- The token diffusion transformer has per-block
  `attention_pair_bias.layer_norm_z` parameters.
- Cross-attention atom transformers use one shared pair normalization.
- `model.version_tensor` is absent. The OSS loader's missing-version
  warning is expected; missing or unexpected model parameters are not.
- Ending-node triangle attention uses the **untransposed** pair bias in
  all four pair stacks:
  - template
  - MSA
  - trunk pairformer
  - confidence pairformer

BioIR's default `OpenFold3Config` must preserve this contract. Setting
only MSA to untransposed while leaving the other three stacks on v0.5
semantics creates a third, unvalidated model.

Run model construction and checkpoint loading in both interpreters
before preprocessing any sample. Both sides must resolve the same local
checkpoint SHA-256. The OSS load may warn only about
`model.version_tensor`; every weight key and shape must otherwise match.

## OSS runtime configuration

Use only the `predict` preset, then serialize its expanded values in the
benchmark lock. Do not add `low_mem` or the deprecated `pae_enabled`
preset.

The P2 inference lock is:

- 3 recycles (4 model cycles)
- 200 diffusion steps
- 5 full-rollout samples
- prediction chunk ceiling 1024
- cuEquivariance triangle kernels enabled
- DeepSpeed Evoformer attention enabled
- Triton triangle kernels disabled when cuEq + DeepSpeed are selected
- MSA server disabled

```yaml
model_update:
  presets:
    - predict
  custom:
    settings:
      memory:
        eval:
          chunk_size: 1024
          use_cueq_triangle_kernels: true
          use_deepspeed_evo_attention: true
          use_triton_triangle_kernels: false
```

cuEq and DeepSpeed compose in OpenFold3: cuEq handles supported triangle
shapes and falls back to DeepSpeed or PyTorch where appropriate. Build
the DeepSpeed op as described in
[deepspeed-evoformer.md](../deepspeed-evoformer.md).

On CUDA 13, do not install OpenFold3's cu12 extra. Install the package
without that extra, then install matching
`cuequivariance`, `cuequivariance-torch`, and
`cuequivariance-ops-torch-cu13` versions in the OSS environment.

Reference kernel validation on H100 (sm_90), torch
`2.12.0a0+5aff3928d8.nv26.05`, CUDA 13.2:

- DeepSpeed `0.19.6+da3ca68`
- CUTLASS `v3.6.0`
- three Evoformer translation units built in about six minutes with
  `MAX_JOBS=16`
- `installed_ops["evoformer_attn"] == 1`
- functional maximum error `7.07e-3`

## Approximate option: frame-aware prediction reuse (default on)

The 200-step lock above is a step count `frame_reuse_period` was evaluated with,
so the BioIR column runs with `edm_sampling_config.frame_reuse_period = 4`. The
option skips 28 of the 200 denoiser calls in the churn-free tail.
`docs/ref/sampling.md` reports the diffusion sampler 15 to 18 percent faster at
200 steps; the forward gain is smaller, because the trunk and the confidence
head do not change. A second, exact BioIR column (`frame_reuse_period = 0`) runs
for the paired comparison.

Follow [Approximate options](../SKILL.md#approximate-options). Take the
config from the processor config, change only that field, and pass it as
the engine config:

```python
model_config = processor_config.get_model_pretrained_config()
model_config.edm_sampling_config.frame_reuse_period = 4  # 0 for the exact column
processor_config.engine_kwargs["config"] = model_config
```

- Give both BioIR columns the same `random_seed`, set in the tokenizer and
  feature-generator `init_context`. The processor seeds the sampling
  rollout from it, so the exact and reuse rollouts share their random draws
  and the paired difference comes from the option, not from sampling noise.
- Record `frame_reuse_period` and the 200 steps in `bench_config.json`. The
  harness's `--exact` switch sets the field to 0.
- The rank-1 sample (the best mean pLDDT of the five) can change between
  the two columns. Score both on the same rule and report the paired lDDT
  and DockQ difference.

## Map the dataset into `oss_data`

The mapping script owns both MSA and template staging and writes one
auditable `oss_data/index.json`. BioIR continues to read the original
dataset paths; the OSS staging tree contains symlinks only.

### MSAs

The dataset's flat filenames are not directly usable by OpenFold3:

- The direct parser silently ignores stems absent from
  `MSASettings.max_seq_counts`, `aln_order`, or `paired_msa_order`.
- A direct A3M's representative identity is its parent directory stem.
  Reusing one flat parent causes distinct chains to collapse onto the
  first chain's MSA.

For each sample and polymer chain group, stage:

```text
oss_data/msa/<sample>/<chain-group>/
├── colabfold_main.a3m
└── colabfold_paired.a3m
```

Create `colabfold_paired.a3m` only when the dataset declares a paired
MSA. In each query chain:

- `main_msa_file_paths` points to the staged main A3M.
- `paired_msa_file_paths` points to the staged paired A3M when present.
- `use_msas`, `use_main_msas`, and `use_paired_msas` reflect the
  declared files exactly.
- `use_msa_server` is explicitly false at experiment level.

Lock these P2 limits:

- `max_rows = 16384`
- `max_rows_paired = 8191`
- `subsample_main = false`
- `paired_msa_order = ["colabfold_paired"]`

OpenFold3 0.4.3 retains legacy precomputed-paired bookkeeping. Paired
rows are cropped and used to deduplicate main rows, but the paired-depth
counter stays zero, so paired rows are not emitted and
`num_paired_seqs == 1`. BioIR's P2 featurizer intentionally mirrors
this. Do not port v0.5's `[query] + [paired] + [main]` behavior into a
P2 comparison.

For the release benchmark, the reference multichain check is:

- `T1152`: MSA shape `[1, 15872, 126, 32]`
- `T1152`: `num_paired_seqs == 1`

Also assert each staged source path, parsed non-query row count, final
depth, and per-chain identity. A successful run with a query-only
fallback is a failure whenever the dataset declared a deeper A3M.

### Templates

OpenFold3 0.4.3 already supports native CIF-direct input; synthesized
template alignments are unnecessary. For every protein chain with
templates:

- stage each CIF under
  `oss_data/templates/<sample>/<chain-group>/<original-name>.cif`
- preserve the original CIF basename because 0.4.3 uses its stem as the
  entry ID and resolves it against `structure_directory`
- set `template_cif_paths` to those exact staged files
- set `template_cif_chain_ids` from the dataset (`null` means
  auto-select)
- never also set `template_alignment_file_path`

Use this preprocessing lock:

```yaml
template_preprocessor_settings:
  mode: predict
  structure_file_format: cif
  fetch_missing_structures: false
  max_seq_id: null
  min_align: null
  min_len: null
  max_release_date: null
  min_release_date_diff: null
  cif_direct_min_score: 0.0
  max_templates: 4
```

Route `output_directory` to a versioned directory under
`oss_data/cache/`, set `use_templates=true` on a programmatic runner,
and keep the BioIR custom-template score floor at the same `0.0`.

Tag 0.4.3 needs two input-only fixes:

- [of3-p2-template-cache-dir.patch](misc/of3-p2-template-cache-dir.patch)
  passes the configured inference cache directory into template
  sampling.
- [of3-p2-template-gap-rows.patch](misc/of3-p2-template-gap-rows.patch)
  removes rows gapped on the template side before the keep/drop check.

Apply them only to a disposable 0.4.3 benchmark checkout:

```bash
PATCH_DIR="$REPO/skills/bench-perf-oss/models/misc"
git -C "$OSS_ROOT" apply --check \
  "$PATCH_DIR/of3-p2-template-cache-dir.patch" \
  "$PATCH_DIR/of3-p2-template-gap-rows.patch"
git -C "$OSS_ROOT" apply \
  "$PATCH_DIR/of3-p2-template-cache-dir.patch" \
  "$PATCH_DIR/of3-p2-template-gap-rows.patch"
```

These patches change input attachment only; they do not alter model
math. Record their digests and changed files in `bench_config.json`.

Verify values, not shapes. OpenFold3 always allocates four slots, so an
all-empty tensor has the same shape as an attached template tensor.
Count slots whose `template_backbone_frame_mask` contains any valid
token **before** calling `model.forward()`: the P2 forward path
mutates/reuses batch tensors, so inspecting the same mask afterwards
can report a false zero. Keep this pre-forward check outside the timing
window. Expected release-dataset checks are:

- `T1152`: `1/1` populated slots
- `T1118v1`: `1/1`
- `T1158v1`: `1/1`

The expected slot count is the maximum templates on any one chain,
capped at four—not the sum across chains.

Keep these regression cases in the mask-level check:

- A supplied template can align at roughly 4% identity over 13% query
  coverage; the default `0.1` direct-CIF score floor rejects it.
- Near-complete alignments covering 55/56 or 1325/1339 query residues
  must remain attached. Whole-chain equality and one-sided gap
  filtering incorrectly drop them.

## Harness preflight

Before the five-sample smoke benchmark, assert:

- imported `openfold3` resolves under the selected OSS checkout
- checkout HEAD is the exact P2 commit
- the only checkout modifications are the two recorded template patches
- checkpoint path and SHA-256 match BioIR
- all 17 generated queries pass OpenFold3 schema validation
- every declared MSA path is staged and non-empty
- all three template-bearing samples pass the populated-mask gate

The harness may add only per-sample orchestration, synchronized timing
around `model.forward()`, memory collection, scoring, and the optional
compile-once wrapper. It must reuse OpenFold3's query parser,
featurizer, checkpoint loader, sampling loop, and CIF writer.

For the smoke compile column, start from a fresh model and compile the
actual Pairformer and diffusion-module call paths once, using the
default automatic policy:

```python
model.sample_diffusion.diffusion_module = torch.compile(
    model.sample_diffusion.diffusion_module
)
model.pairformer_stack = torch.compile(model.pairformer_stack)
```

Do not pass `dynamic`, change global Dynamo shape settings, or mark
input axes. Test the roles separately first with synthetic direct
inputs in A, A, B, B order:

- Pairformer inputs are synthetic `s`, `z`, `single_mask`, and
  `pair_mask`; derive `c_s`, `c_z`, dtype, and device from the selected
  module/config.
- DiffusionModule inputs must satisfy its complete direct signature and
  batch dictionary. Derive token/atom axes and fixed channel widths
  from the module/config; keep the five-sample diffusion axis fixed.

The synthetic probe is a fast compiler diagnostic only. After the
largest target set passes it, probe the smallest and a same-bin real
sample before the five-sample sweep. Initial A and first B may compile
while `dynamic=None` adapts; immediate repeats must not. Do not mark
the raw inference batch or child inputs.

P2 stores the same diffusion child in both `model.diffusion_module`
and `model.sample_diffusion.diffusion_module`; inference calls the
second path. The direct assignment above therefore compiles the path
used by the sampler even though the shallow alias remains unchanged.
If a generic helper replaces by object identity, replacing both aliases
with the same compiled child is also valid; in either case assert that
the sampler's child received calls.

On P2 source `0.4.3` with torch 2.12, the exact default calls require
no chunk-tuner fence. The 17-sample sweep succeeded on every row. The
first sample captured 88 Dynamo frames; automatic later-sample warmup
specializations occurred at:

- `5sbj-assembly1`: +56 frames
- `T1152`: +3
- `R1136`: +7
- `8a8o-assembly1_A_B`: +3

Every measured forward captured zero new frames. This is stable
per-sample measurement but not a one-compile-serves-all graph: report
all five warmup ids and state that `dynamic=None` adapted four times
after the initial sample.

The combined default-compile column was `0.884x` OSS eager by
geometric mean (about 12% slower), while BioIR was `2.44x` faster than
that compile column. Mean lDDT was `0.624` and mean DockQ over nine
applicable samples was `0.560`, consistent with OSS eager at the
reported precision. Keep this slower compile column; speed is an
outcome, not a compile-validity gate. Child-tensor drift versus eager
is likewise diagnostic: `torch.compile` may rewrite the graph, and
downstream lDDT/DockQ is the fitness check.

## OpenBind boundary

`of3-ob-2025-06-30-174k` is not a drop-in replacement:

- it requires OpenFold3 v0.5 or newer within its declared range
- token diffusion uses one shared pair normalization
- the checkpoint includes `model.version_tensor`
- ending-node triangle attention uses the transposed pair bias
- precomputed paired rows are emitted and counted
- both 0.4.3 template patches are already fixed

Never bypass the registry or use `strict=False` to load P2 into v0.5.
That produces missing shared-normalization keys and unexpected
per-block keys, which is model onboarding—not benchmarking.
