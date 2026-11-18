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

# OSS feature path — no BioIR data pipeline

Use this when `docs/ref/support-matrix.md` says Pipeline = **No**
**and** the model is still a folding compute path. No current folding
key uses it: Boltz-1/2, OpenFold2 / AlphaFold2, OpenFold3, and
`protenix-v2` all have a factory and run on Processor path. `boltz-2-affinity`
is the same *shape* (module only) but a different task — do not run it
with this folding dump / CIF / lDDT path. See
[Extending to other model kinds](workflow.md#extending-to-other-model-kinds).

`build_processor` cannot run. Both timed forwards consume **the same
OSS-featurized batch**. The BioIR side is the `nn.Module` constructor
API (`docs/ref/api.md`), not the processor.

```text
the spec item (A3Ms / templates)
        │  MSAs + templates attached (same contract)
        ▼
OSS data pipeline  ──untimed──►  feature dict
        │
        ├─► OSS model.forward(batch)      timed   (oss_python)
        └─► adapt keys ──► BioIR model.forward    timed   (bioir_python)
```

One feature dict for both forwards gives equal templates:
whatever the OSS pipeline built, both sides see. Verify the
template features are non-empty once, on that shared batch.

Do not invent a BioIR tokenizer / feature factory for the bench.

## Construct the BioIR module

Use the default constructor (`docs/ref/api.md`). Check the class's
`include_load_weights` default and confirm the weights load from the
local checkpoint (`docs/ref/model-weights.md`).

```python

# Default constructor: config=None → get_pretrained_config.
# Do not pass a handmade BaseConfig.
model = ModelCls(model_name="<model_source>")
model = model.cuda().eval()
```

Its graph regions capture by default; keep their policies
([CUDA graphs](measurement.md#cuda-graphs)).

Pass the `runtime_args` its `forward` accepts (`docs/ref/api.md`), and
map each onto the OSS equivalent with the same semantics. Write the
mapping into a model profile under `models/`.

## Featurize once, time twice

Two venvs cannot import OSS and BioIR in one process. Dump features
from `oss_python`, load them in both harnesses.

```text
$WORKDIR/ref_data/oss_features/<sample_id>.pt
```

Each file is the GPU-ready host dict (CPU tensors) plus the resolved
MSA paths used to build it. Write a provenance line
(`source=oss`, OSS commit, command). A dump that came from BioIR is
invalid.

Reuse the OSS inference script's featurizer, as in the processor path.
Do not write a second feature builder from OSS internals.

```python
# oss_python — untimed, via the OSS infer/predict helpers
batch = oss_featurize(oss_input)          # must have seen the A3Ms
torch.save({"batch": cpu(batch), "msas": msa_paths}, path)

# oss_python — eager first (required)
batch = to_device(torch.load(path)["batch"])
time_model_forward(oss_model, batch, oss_runtime_args)

# oss_python — compile target submodules ONCE on a fresh model,
# then reuse oss_compiled for every in-scope dump. Do not compile
# per sample. Do not torch.compile(oss_compiled) as a whole.
compile_oss_hot_modules(oss_compiled, family="af3")
for sample_id, path in in_scope_dumps:
    batch = to_device(torch.load(path)["batch"])
    row = run_oss_timed_forward(oss_compiled, batch, oss_runtime_args)
    # first warmup (and first of a new token_bin) may compile;
    # not every sample.

# bioir_python — timed
from adapt_features import adapt_features  # $WORKDIR/bench/
raw = torch.load(path)["batch"]
batch = to_device(adapt_features(raw))
time_model_forward(bioir_model, batch, bioir_runtime_args)
```

Use the same `time_model_forward` helper as
[measurement.md](measurement.md). There is no
`profile_inference` field on this path.

## Feature adapter

OSS key names may not match the BioIR module's `forward`. Write
`$WORKDIR/bench/adapt_features.py` that maps OSS → BioIR keys.

- Record every rename / reshape / dtype cast in
  `implementation-notes.md`.
- Do not recompute MSA or structure features in the adapter. It is a
  rename/layout step only.
- If a required BioIR key is missing from the OSS dict, stop. Do not
  fill zeros to keep the bench green.
- After adapt, assert MSA-derived tensors are still non-empty.

## Inputs

Still the spec items with MSAs
([samples.md](samples.md)). Convert each in-scope item to the OSS
query / CIF / A3M (or Boltz YAML+CSV) layout the OSS pipeline
expects. Rewrite stale Boltz `msa:` absolutes to
`$DATASET_ROOT/casp15/msa/`. The MSA contract still applies — the
OSS featurizer must receive every listed A3M / CSV. Do not call
`load_requests` + `build_processor`. Do not use
`examples/data/samples/`.

## Isolation

`bioir_python` needs `bionemo_ir` and the dumped `.pt` files only.
It must not import the OSS package. `oss_python` featurizes and times
the OSS module; it must not import `bionemo_ir`.
