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

# DeepSpeed `evoformer_attn` — source build

Applies to **OpenFold2 / OpenFold3** OSS. Boltz and Protenix-v2 skip
this page unless their OSS script asks for the same kernel.

The published `deepspeed` wheel does **not** ship
`DS4Sci_EvoformerAttention`. It has to be compiled from source, and
the compile has two traps that both end in a green `pip` and no
kernel. Read the whole page before starting: the build is ~5–10
minutes and the failure modes are silent.

## Why it can be required, not optional

When the model profile enables this kernel for the measured inference
path, silently falling back measures a different, slower path. Treat
the fallback as a separate column requiring user approval. The model
profile also determines whether this kernel composes with another
backend; see [models/of3.md](models/of3.md#oss-runtime-configuration)
for OpenFold3.

BioIR does **not** use this kernel (CuTeDSL / auto triangle
backends). Never install DeepSpeed into `$BIOIR_PYTHON`.

## Prerequisites

- Toolchain on the machine: `g++`, `ninja`, `nvcc`. Check with
  `command -v`; missing ones are an
  [apt package](environment.md#oss-system-packages-apt)
  (`build-essential`, `ninja-build`). The NGC PyTorch image already
  has all three.
- CUTLASS **3.x headers** — clone `v3.6.0`. Do **not** point
  `CUTLASS_PATH` at the repo's `3rdparty/cutlass`: that tree is
  CUTLASS 4 for BioIR / CuTeDSL, and `evoformer_attn` wants ≥ 3.1
  from a 3.x checkout.
- DeepSpeed source — prefer the version OSS pins (OF2's
  `environment.yml` pins `deepspeed==0.14.5`), otherwise the tip.

```bash
git clone --branch v3.6.0 --depth 1 \
  https://github.com/NVIDIA/cutlass.git "$WORKDIR/oss/cutlass"
git clone --depth 1 https://github.com/deepspeedai/DeepSpeed.git \
  "$WORKDIR/oss/deepspeed"
```

## Build

Uninstall any PyPI `deepspeed` the OSS install pulled in first, then
build **non-editable** from the source directory:

```bash
"$OSS_PYTHON" -m pip uninstall -y deepspeed || true

export CUTLASS_PATH="$WORKDIR/oss/cutlass"
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=$("$OSS_PYTHON" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0))')  # bench GPU only
export MAX_JOBS=16
DS_BUILD_OPS=0 DS_BUILD_EVOFORMER_ATTN=1 \
  "$OSS_PYTHON" -m pip install "$WORKDIR/oss/deepspeed" \
  --no-build-isolation -c "$WORKDIR/ref_data/constraints_bioir.txt" -v \
  > "$WORKDIR/debug/ds_build.log" 2>&1
```

`DS_BUILD_OPS=0` turns every op off; `DS_BUILD_EVOFORMER_ATTN=1`
turns only `evoformer_attn` back on. `TORCH_CUDA_ARCH_LIST` pinned to
the bench GPU keeps the compile to one architecture. Drop `-c` if
this run has no constraints file.

Two mandatory pip flags, and the reason each one exists:

- **`-v`** — without it pip prints `Successfully installed` and
  nothing else, so a build that compiled zero kernels looks
  identical to one that worked. The op decision is only visible in
  the build output.
- **No `-e`** — see below.

### Never `pip install -e` this

`pip install -e` on DeepSpeed routes through `setup.py develop`,
which current setuptools has deprecated into a shim that **re-invokes
pip with build isolation**. The nested pip gets a fresh env with no
torch, so DeepSpeed's `setup.py` takes its `except ImportError`
branch, falls back to the **CPU accelerator** (whose op list has no
`evoformer_attn`), compiles nothing, and overwrites
`deepspeed/git_version_info_installed.py` with the CPU result —
discarding the outer invocation's correct decision. `pip` exits **0**.

The log of a poisoned editable build shows the outer run deciding
correctly and the nested pip then undoing it:

```text
Running setup.py develop for deepspeed
  Install Ops={... 'evoformer_attn': 1 ...}          <- outer, correct
  ext_modules=[Extension('deepspeed.ops.evoformer_attn_op')]
  running develop
    Obtaining file:///.../deepspeed                  <- nested pip
    Installing build dependencies: started           <- isolation, no torch
    Building editable for deepspeed (pyproject.toml)
```

and the artifact it leaves behind:

```python
# deepspeed/git_version_info_installed.py
installed_ops={'deepspeed_not_implemented': False, 'async_io': False, ...}
accelerator_name='cpu'
torch_info={'version': '0.0', 'cuda_version': '0.0', ...}
```

`accelerator_name='cpu'` and `torch_info['version'] == '0.0'` on a
CUDA box are the fingerprint. A non-editable `pip install <dir>`
runs `setup.py` once, in the outer environment, and compiles.

### Compile time — budget 5–10 minutes

Three CUTLASS translation units get built, and they are slow:

- `csrc/deepspeed4science/evoformer_attn/attention.o` — the forward
  kernel, ~15 MB of object code on its own
- `attention_back.o` — backward; compiled even though the bench only
  runs forward, and there is no flag to skip it
- `attention_cu.o`

Expect **~5–10 minutes wall clock** for a single arch with
`MAX_JOBS=16`, most of it in `attention.o`. Consequences:

- Do not kill the build at 60–90 seconds thinking it hung. Watch the
  `.o` files appear under
  `oss/deepspeed/build/temp.linux-*/csrc/deepspeed4science/evoformer_attn/`.
- A **re-run after a completed compile links in seconds** —
  setuptools reuses those objects. Fast is normal on a second
  attempt and does not mean the kernel was skipped; a fast *first*
  attempt does. For a true cold rebuild, move the build directory
  under `$WORKDIR/oss/deepspeed` into a fresh backup directory under
  `$WORKDIR/debug`, after checking the resolved source and destination.
  Keep the backup until the rebuilt kernel passes verification.
- Build in **Phase 0 / environment setup**, never lazily. Leaving the
  op to JIT puts this same compile on the first timed forward, or
  worse, inside the measured window.

## Verify before any forward

All three checks, in this order. The first one is the gate the skill
requires; the other two catch a stale or mis-linked `.so`.

```bash
"$OSS_PYTHON" - <<'PY'
from deepspeed.git_version_info import installed_ops, version, accelerator_name, torch_info
print("deepspeed", version, "| accelerator", accelerator_name)
print("torch_info", torch_info)
assert accelerator_name == "cuda", accelerator_name
assert torch_info["version"] != "0.0", torch_info
assert installed_ops.get("evoformer_attn"), installed_ops
print("evoformer_attn prebuilt ok")
PY

"$OSS_PYTHON" -c "
import os, deepspeed
so = os.path.join(os.path.dirname(deepspeed.__file__), 'ops')
print([f for f in os.listdir(so) if 'evoformer_attn_op' in f])
"
```

Then a real numeric check — the op can be present and still be the
wrong kernel for the arch:

```bash
"$OSS_PYTHON" - <<'PY'
import math, torch
from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention

torch.manual_seed(0)
B, N, L, H, D = 1, 4, 64, 4, 32
dt, dev = torch.bfloat16, "cuda"
Q, K, V = (torch.randn(B, N, L, H, D, dtype=dt, device=dev) for _ in range(3))
mask_bias = torch.zeros(B, N, 1, 1, L, dtype=dt, device=dev)
mask_bias[..., L // 2:] = -1e4
pair_bias = torch.randn(B, 1, H, L, L, dtype=dt, device=dev)

out = DS4Sci_EvoformerAttention(Q, K, V, [mask_bias, pair_bias])

s = torch.einsum("bnqhd,bnkhd->bnhqk", Q.float(), K.float()) / math.sqrt(D)
s = s + mask_bias.float() + pair_bias.float()
ref = torch.einsum("bnhqk,bnkhd->bnqhd", s.softmax(-1), V.float())

err = (out.float() - ref).abs().max().item()
print("max abs err vs fp32 reference:", err)
assert err < 2e-2, err
print("evoformer_attn functional ok")
PY
```

bf16 against an fp32 reference lands around `7e-3`. An error near
`1.0`, or a `RuntimeError` about an unsupported architecture, means
the kernel was built for the wrong `TORCH_CUDA_ARCH_LIST`.

Only after all three pass, enable the flag in the OSS inference
config (`use_deepspeed_evo_attention` /
`use_deepspeed_evoformer_attention`) — together with cuEq, per
[environment.md](environment.md#oss-kernel-flags--installing-is-not-enabling)
— and record `deepspeed_evo_attn: true` plus the full `oss_kernels`
set in `bench_config.json`.

## Recovery

- Interrupted install leaves `~eepspeed` stubs that make pip warn
  `Ignoring invalid distribution`. Inspect the affected interpreter's
  site-packages directory and move only the identified stubs into a fresh
  backup directory under `$WORKDIR/debug`. Do not use a wildcard cleanup or
  modify an interpreter outside the benchmark workspace.
- A poisoned editable attempt leaves a bad
  `deepspeed/git_version_info_installed.py` **in the source tree**,
  where an editable import will keep reading it. Move that exact file from
  the benchmark-owned OSS checkout into a fresh backup directory under
  `$WORKDIR/debug` before rebuilding.
- `is_compatible()` is a useful pre-flight and is independent of the
  build path:

```bash
"$OSS_PYTHON" -c "
from deepspeed.ops.op_builder.evoformer_attn import EvoformerAttnBuilder
print(EvoformerAttnBuilder().is_compatible(verbose=True))
"
```

  `False` here is a real prerequisite problem (missing `CUTLASS_PATH`,
  CUTLASS < 3.1, no `nvcc`). `True` plus an empty `installed_ops` is
  the editable-install trap, not a prerequisite problem.
