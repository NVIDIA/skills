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

# Environment setup — BioIR vs OSS

v1 runs two stacks. Check both, then choose one isolation mode.

**Priority: BioIR first, then OSS.** Finish the BioIR install and
freeze before touching OSS. Packages that BioIR already has
(torch, CUDA wheels, cuEq, numpy, …) stay at BioIR versions. Resolve
OSS against that freeze: reuse BioIR pins when they import, isolate
OSS when they cannot. Never change a BioIR package to satisfy an OSS
pin. Do not `pip install` OSS pins into the BioIR interpreter.

## CUDA 12 packages → CUDA 13

The default container (`nvcr.io/nvidia/pytorch:26.09-py3`) and BioIR
(`pyproject.toml`) are **CUDA 13** (`cuequivariance-ops-*-cu13`,
`nvidia-cutlass-dsl[cu13]`, `cuda-python>=13`). OSS READMEs and
extras often still pin CUDA 12.

**If a `cu13` build exists, using `cu12` is a hard failure.** CUDA 13
is the default on this image, so a CUDA 12 artifact is never the
right choice merely because an OSS README names it. For every
package, extra, wheel, or index URL that names CUDA 12, install the
CUDA 13 counterpart instead.

The container's CUDA is the target, not a literal `13` — read it
once and remap against that major so this rule survives the next
image bump:

```bash
CUDA_MAJOR=$("$BIOIR_PYTHON" -c "import torch;print(torch.version.cuda.split('.')[0])")
echo "target CUDA major: $CUDA_MAJOR"   # 13 on pytorch:26.09-py3
```

**Check availability before you install**, so "cu13 is missing" is a
finding and not an assumption. A CUDA-tagged dist publishes one
variant per CUDA major under its own name:

```bash
"$OSS_PYTHON" -m pip index versions "cuequivariance-ops-torch-cu${CUDA_MAJOR}"
```

If that resolves, the `cu13` wheel is mandatory. Only when the
`cu13` name genuinely does not exist on any configured index (and
BioIR has no equivalent already installed) do you stop and report —
see the Do-not list below.

Rewrite (keep the rest of the name and version specifier):

- `*-cu12` / `*_cu12` → `*-cu13` / `*_cu13`
  (`cuequivariance-ops-torch-cu12` → `cuequivariance-ops-torch-cu13`,
  `cuequivariance-ops-cu12` → `cuequivariance-ops-cu13`,
  `nvidia-cublas-cu12` → `nvidia-cublas-cu13`)
- extras `[cu12]` → `[cu13]` (`nvidia-cutlass-dsl[cu12]`)
- `cuda12` in the dist name → `cuda13`
- PyTorch extra index `https://download.pytorch.org/whl/cu12*` →
  `cu13*` **or** keep the container / BioIR torch (already CUDA 13)
- `torch==…+cu12*` / `+cu121` / `+cu124` / `+cu128` → do not
  install; reuse BioIR / NGC torch

Do **not**:

- Treat OSS `cu12` vs BioIR `cu13` as a hard conflict that forces a
  second venv. Remap first; only split if the **cu13** package still
  fights BioIR.
- Fall back to the `cu12` wheel if `cu13` is missing from the index.
  Stop and report. Reuse BioIR's already-installed `cu13` package
  when it imports. Installing `cu12` "just to get moving" silently
  puts a second CUDA runtime next to the container's, and any
  latency number measured afterwards is not trustworthy.
- Install a CUDA 12 build of a package BioIR **already provides** as
  `cu13`. Reuse the installed one; do not add a parallel copy.
- Let a transitive dependency drag a `cu12` wheel in unnoticed. Pin
  the shared stack with a constraints file, then re-run the gate
  below after every install.

After OSS install, `$OSS_PYTHON -m pip freeze` must not list
CUDA-tagged `*cu12*` / `*cuda12*` wheels. `torch.version.cuda` must
match the container major. Record remaps as `cuda12_remaps`
(`[{from, to}, …]`) in `bench_config.json` and
`implementation-notes.md`.

Run this gate **after each OSS install step**, not once at the end —
that is how you catch the transitive case while it is still one
package to undo:

```bash
if "$OSS_PYTHON" -m pip freeze | grep -iE '[-_]cu12|cuda12|\+cu12'; then
  echo "FAIL: CUDA 12 wheel still installed — remap to cu13"
  exit 1
fi
"$OSS_PYTHON" -c "
import torch
assert str(torch.version.cuda).startswith('13'), torch.version.cuda
print('PASS torch.version.cuda =', torch.version.cuda)
"
```

## Container and BioIR install

The prepared development container is built from
`nvcr.io/nvidia/pytorch:26.09-py3`; start it with `docker/dev.sh` as described
in `docs/dev.md`. Ask whether the user is in **developer mode** (editable
checkout) or **not** (released / wheel / pip).

**Always** export this before any BioIR import, probe, or timed run:

```bash
export CUTEDSL_FORCE_CUBIN=1
```

That forces the packaged CUBIN path and keeps CuTeDSL JIT off the
clock (`docs/ref/architecture.md`,
`bionemo_ir/dsl_kernels/cute_cache.py`). JIT compile time is
non-deterministic and makes `model.forward()` numbers unstable. A
harness that unsets this, or a process that does not inherit it, is
an invalid bench. Record it in `bench_config.json` as
`cutedsl_force_cubin: true`.

### Not developer mode — wheel or pip

Do **not** editable-install the checkout.

```bash
# Local wheel from dist/ (docs/dev.md)
"$BIOIR_PYTHON" -m pip install --no-deps dist/bionemo_ir-*.whl
# or a full resolve:
"$BIOIR_PYTHON" -m pip install dist/bionemo_ir-*.whl
# or the published package:
"$BIOIR_PYTHON" -m pip install bionemo-ir
```

CUBINs are already inside the wheel extension. Still set
`CUTEDSL_FORCE_CUBIN=1`.

### Developer mode — editable + git-lfs

The fused kernels live in LFS-tracked packs under
`cpp/kernels/cutedsl_*/cubins/packs/*.tar.xz`. Pointer files are not CUBINs.

```bash
git lfs install
git lfs pull --include='cpp/kernels/cutedsl_*/cubins/packs/*.tar.xz'
# or: git lfs pull
```

If a pack still starts with `version https://git-lfs.github.com/spec`, stop and
pull LFS.

On a configured host, create the locked project environment and use its
interpreter:

```bash
uv sync --locked
BIOIR_PYTHON="$PWD/.venv/bin/python"
```

Inside the prepared development container, the image already contains the
locked runtime, build, and development dependencies. Add only the editable
checkout to its writable environment:

```bash
uv pip install --no-deps -e .
BIOIR_PYTHON="$(command -v python)"
```

Do not use the container command in a raw NGC image or general interpreter;
`--no-deps` is correct only when the prepared image already supplies the
dependency closure. If `nvcc` is not on `PATH`, set the wheel tag explicitly:

```bash
# Configured host.
CUDA_TAG=cu132 uv sync --locked
# Prepared development container.
CUDA_TAG=cu132 uv pip install --no-deps -e .
```

Verify the extension and that FORCE_CUBIN is on:

```bash
CUTEDSL_FORCE_CUBIN=1 "$BIOIR_PYTHON" -c "
import os
assert os.environ.get('CUTEDSL_FORCE_CUBIN') == '1'
import bionemo_ir.libs._cutedsl_kernels as k
print('cutedsl_kernels ok', k)
"
```

## Interpreters

Record these in `$WORKDIR/ref_data/bench_config.json`:

- `bioir_python` — interpreter that imports `bionemo_ir`
- `oss_python` — interpreter that imports the OSS package

Phase 4 uses only `bioir_python`. Phase 5 uses only `oss_python`.
Mixing them is a failed run.

## Step 1 — Probe BioIR

BioIR needs Python 3.12+, CUDA, and the editable / wheel install
(`docs/dev.md`).

```bash
if [ -z "${BIOIR_PYTHON:-}" ]; then
  BIOIR_PYTHON=$(command -v python)
fi
"$BIOIR_PYTHON" - <<'PY'
import sys, torch, bionemo_ir
from bionemo_ir.pipeline.processor.engine_proc import build_processor
print("python", sys.executable, sys.version.split()[0])
print("bionemo_ir", bionemo_ir.__file__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability())
try:
    import bionemo_ir.libs._cutedsl_kernels as k
    print("cutedsl_kernels", "ok")
except Exception as e:
    print("cutedsl_kernels", type(e).__name__, e)
PY
```

If `import bionemo_ir` fails, install into this interpreter only —
editable in developer mode, wheel/pip otherwise (see above). Never
skip `CUTEDSL_FORCE_CUBIN=1`.

If the extension is missing, stop — do not continue on CPU.

Save `$WORKDIR/ref_data/env_bioir.txt`:

```bash
"$BIOIR_PYTHON" -m pip freeze > "$WORKDIR/ref_data/env_bioir.txt"
```

## OSS checkout pins

Do **not** ask for a random `$OSS_ROOT` and do not clone `main`.
Resolve the tree from this table, then probe it (Step 2).

| BioIR key                                      | OSS tree                                                                               | How to get it                                                                      |
| ---------------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `openfold3`                                    | [`3rdparty/openfold-3`](../../3rdparty/openfold-3)                                     | checkpoint-compatible commit from its registry; see [models/of3.md](models/of3.md) |
| `protenix-v2`                                  | [`3rdparty/protenix`](../../3rdparty/protenix)                                         | gitlink SHA from `git ls-tree HEAD 3rdparty/protenix`                              |
| `boltz-1`, `boltz-2`                           | [jwohlwend/boltz](https://github.com/jwohlwend/boltz) **`v2.2.1`**                     | clone that tag into `$WORKDIR/oss/boltz`                                           |
| `openfold2_*` (and AF2 e2e that uses OpenFold) | [aqlaboratory/openfold `v2.2.0`](https://github.com/aqlaboratory/openfold/tree/v2.2.0) | clone that tag into `$WORKDIR/oss/openfold`                                        |

For an in-repo submodule, use the gitlink unless its model profile
selects a different checkpoint-compatible commit. Never substitute a
newer `origin/main`. Record the parent gitlink, selected `oss_ref`, and
selected `oss_commit` (full SHA) in `bench_config.json`.

Do not clone upstream HEAD or `git pull` inside `3rdparty/`. Follow the
model profile before assigning `$OSS_ROOT`.

```bash
git submodule update --init --recursive 3rdparty/openfold-3 3rdparty/protenix
# Protenix-v2
OSS_ROOT="$REPO/3rdparty/protenix"
expected=$(git -C "$REPO" ls-tree HEAD 3rdparty/protenix | awk '{print $3}')
test "$(git -C "$OSS_ROOT" rev-parse HEAD)" = "$expected"
```

A selected-ref mismatch is a blocker. Fix a gitlink mismatch with
`git submodule update` on that path only. Never overwrite a dirty
submodule.

When the selected ref differs from the gitlink, or the submodule is
dirty, preserve the user's tree and make a clean detached worktree:

```bash
git -C "$SUBMODULE_ROOT" worktree add --detach \
  "$WORKDIR/oss/$MODEL_NAME" "$expected"
OSS_ROOT="$WORKDIR/oss/$MODEL_NAME"
test "$(git -C "$OSS_ROOT" rev-parse HEAD)" = "$expected"
test -z "$(git -C "$OSS_ROOT" status --porcelain)"
```

The first environment probe must record both the parent gitlink and
the checkout SHA. A parent rebase that moves the gitlink invalidates
an already-populated submodule and every prior timing artifact, even
when the old source still imports.

**Boltz-1/2 — tag `v2.2.1`.** Same repo for both keys.

```bash
git clone --branch v2.2.1 --depth 1 \
  https://github.com/jwohlwend/boltz.git "$WORKDIR/oss/boltz"
OSS_ROOT="$WORKDIR/oss/boltz"
test "$(git -C "$OSS_ROOT" describe --tags --exact-match)" = "v2.2.1"
```

**OpenFold2 — tag `v2.2.0`.** Use
[aqlaboratory/openfold](https://github.com/aqlaboratory/openfold/tree/v2.2.0),
not the `nz/OpenFold` weight mirror.

```bash
git clone --branch v2.2.0 --depth 1 \
  https://github.com/aqlaboratory/openfold.git "$WORKDIR/oss/openfold"
OSS_ROOT="$WORKDIR/oss/openfold"
test "$(git -C "$OSS_ROOT" describe --tags --exact-match)" = "v2.2.0"
```

If a clone already exists, check it out to the pin and verify;
do not reuse `main` or another tag. A wrong revision is a hard
failure — do not time it.

## Step 2 — Probe OSS

`$OSS_ROOT` is the pin above. Read its `pyproject.toml` / `requirements*.txt` /
`setup.cfg` and the e2e entry imports. List the packages actually
needed to construct the model and call `forward` (ignore training-only
extras).

**Classify by import graph, not by what a package is for.** Package
metadata says what a dependency is *nominally* used for; only the
imports say whether the run can start without it. A helper the bench
never exercises — a checkpoint downloader, a cloud-storage client, a
telemetry or experiment-tracking shim — is still **mandatory** if
some module on the entry-point chain imports it at module scope.
Reasoning from purpose ("we stage weights locally, so the download
SDK is optional") is how you end up discovering dependencies one
`ImportError` at a time.

So probe the **whole chain the run touches**, not just the top-level
package: entry point, runner, config, validator, model. Import them
in one pass and let it tell you what is missing.

```bash
OSS_PYTHON=${OSS_PYTHON:-$BIOIR_PYTHON}
"$OSS_PYTHON" - <<'PY'
import importlib, sys
print("python", sys.executable, sys.version.split()[0])
# Every module the e2e path imports, deepest entry points included —
# not just "<oss_package>".
for name in ("torch", "<oss_package>", "<oss_package>.<entry_point>",
             "<oss_package>.<runner>", "<oss_package>.<config>"):
    try:
        m = importlib.import_module(name)
        print(name, getattr(m, "__file__", "?"), getattr(m, "__version__", ""))
    except Exception as e:
        print(name, "MISSING", type(e).__name__, e)
PY
```

Save the output under `$WORKDIR/debug/`. A missing OSS package is not
a license to install it into `BIOIR_PYTHON`.

Probe the checkpoint registry in the same pass. Record the current
default checkpoint, every compatibility range, and the checkpoint
BioIR resolves for the model key. Package / CLI names are not model
identity: a release can keep `openfold3` while changing its default
from one trained variant to another and marking the old weights
legacy. An explicit path often bypasses registry validation, so run a
strict state-dict load in both interpreters before treating the
environments as ready. Incompatible keys or a declared incompatible
version is a model-variant blocker, not an environment problem to
solve with `strict=False`.

## Step 3 — Diff and classify conflicts

BioIR's freeze is the lock. Compare the OSS pin set **to that lock**,
not the other way around. Write `$WORKDIR/ref_data/env_conflicts.md`.

For each OSS pin:

1. **BioIR already has a suitable version** — keep the BioIR package.
   Do not reinstall or upgrade it for OSS.
1. **OSS needs something BioIR does not have** (soft extra) — install
   it only in the OSS interpreter, after BioIR still imports.
1. **OSS pin fights a BioIR package** (hard conflict) — leave BioIR
   unchanged and put OSS in a second venv (Step 4).

**Hard conflicts** — OSS wants a different major/minor or CUDA tag
than the BioIR lock **after** [CUDA 12 → 13 remaps](#cuda-12-packages--cuda-13).
These require a second environment:

- `torch`, `torchvision`, `torchaudio`
- `triton`, `nvidia-cudnn-*`, CUDA wheels
- `cuequivariance`, `cuequivariance-ops-torch-*`
- `deepspeed` built against another torch
- `numpy` major (1 vs 2) when either side imports compiled extensions

**`flash-attn` is optional.** Neither BioIR nor the OSS bench needs
it. If it conflicts, remove it only from an environment created for this
benchmark. For an existing or shared interpreter, explain the conflict and
obtain the user's approval before uninstalling:

```bash
"$BIOIR_PYTHON" -m pip uninstall -y flash-attn
"$OSS_PYTHON" -m pip uninstall -y flash-attn
```

Do not install it to "match" OSS, and do not split venvs because of
it.

**Soft conflicts** — extra packages OSS needs that BioIR does not
pin (`pytorch-lightning`, `gemmi`, `ml_collections`, …). Install
these only in the OSS environment. If the pin is CUDA 12, install
the CUDA 13 counterpart
([above](#cuda-12-packages--cuda-13)).

**Same version** — leave it. Do not reinstall.

If any hard conflict exists, go to
[Step 4 — two environments](#step-4--two-environments). If none exist,
[Step 5 — shared environment](#step-5--shared-environment) is allowed.

## OSS system packages (apt)

Some OSS deps are **distro packages**, not pip: shared libraries
(RDKit / OpenGL), compilers for the DeepSpeed evoformer kernel, and
anything the OSS README or Dockerfile lists as `apt install`. Install
them on the **machine / container**, not into a venv.

The agent often cannot install them (no root, no `sudo`, or a
sandbox policy that blocks `apt-get` / `sudo` outright). Assume you
will have to hand the command to the human, so **collect the whole
set in one pass and generate a ready-to-run script**. Do not
discover packages one `ImportError` at a time — each round trip
costs the human another turn.

### Step A — scan for every missing soname at once

Do not wait for the first `ImportError`. Walk the OSS
interpreter's `site-packages`, read each ELF's `DT_NEEDED`, and ask
the dynamic loader for every soname. `ldd` / `objdump` may also be
blocked by policy, so parse the ELF in Python. Save as
`$WORKDIR/bench/scan_missing_libs.py`:

```python
import ctypes, re, struct, sys
from pathlib import Path

# auditwheel renames wheel-private copies to libfoo-<8 hex>.so.X and
# resolves them through RPATH, not the system loader path.
AUDITWHEEL_RE = re.compile(r"-[0-9a-f]{8}\.so")


def dt_needed(path: Path) -> list[str]:
    """DT_NEEDED sonames of a 64-bit little-endian ELF."""
    data = path.read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2:
        return []
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shentsize = struct.unpack_from("<H", data, 0x3A)[0]
    e_shnum = struct.unpack_from("<H", data, 0x3C)[0]
    dynamic = None
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        if struct.unpack_from("<I", data, off + 0x04)[0] == 6:  # SHT_DYNAMIC
            dynamic = (
                struct.unpack_from("<Q", data, off + 0x18)[0],  # sh_offset
                struct.unpack_from("<Q", data, off + 0x20)[0],  # sh_size
                struct.unpack_from("<I", data, off + 0x28)[0],  # sh_link
            )
            break
    if dynamic is None:
        return []
    dyn_off, dyn_size, dyn_link = dynamic
    off = e_shoff + dyn_link * e_shentsize  # .dynstr
    str_off = struct.unpack_from("<Q", data, off + 0x18)[0]
    strings = data[str_off : str_off + struct.unpack_from("<Q", data, off + 0x20)[0]]
    needed = []
    for pos in range(dyn_off, dyn_off + dyn_size, 16):
        d_tag, d_val = struct.unpack_from("<qQ", data, pos)
        if d_tag == 0:  # DT_NULL
            break
        if d_tag == 1:  # DT_NEEDED
            needed.append(strings[d_val : strings.find(b"\x00", d_val)].decode())
    return needed


def loader_finds(soname: str) -> bool:
    try:
        ctypes.CDLL(soname)
        return True
    except OSError:
        return False


# Default to every directory on this interpreter's sys.path, so the
# scan covers the venv, the user site, and the container's
# dist-packages without hardcoding any of them.
roots = [Path(p) for p in (sys.argv[1:] or sys.path) if p and Path(p).is_dir()]

needed_by: dict[str, set[str]] = {}
bundled: set[str] = set()
for root in roots:
    for so in root.rglob("*.so*"):
        if not so.is_file():
            continue
        bundled.add(so.name)
        for name in dt_needed(so):
            needed_by.setdefault(name, set()).add(so.name)

for soname in sorted(needed_by):
    if loader_finds(soname) or AUDITWHEEL_RE.search(soname) or soname in bundled:
        continue
    print(soname, "needed by", ", ".join(sorted(needed_by[soname])[:3]))
```

Run it with the **OSS** interpreter so the scan sees exactly what
that stack will load. No arguments needed — `sys.path` already
covers the venv, the user site, and any container `dist-packages`:

```bash
"$OSS_PYTHON" bench/scan_missing_libs.py \
  | tee "$WORKDIR/debug/scan_missing_libs.log"
```

Filtering matters: a wheel that bundles its own Boost / cairo /
OpenBLAS reports dozens of unresolved sonames that are **not**
missing. Only unmangled sonames absent from every scanned tree are
real apt packages.

### Step B — map sonames to packages

Debian / Ubuntu names, whichever OSS tree is being benched:

- `libXrender.so.1` → `libxrender1`
- `libXext.so.6` → `libxext6`
- `libX11.so.6` → `libx11-6`
- `libSM.so.6` → `libsm6`, `libICE.so.6` → `libice6`
- `libGL.so.1` → `libgl1`, `libEGL.so.1` → `libegl1`
- `libglib-2.0.so.0` / `libgthread-2.0.so.0` → `libglib2.0-0`
- `libgomp.so.1` → `libgomp1`
- `libxcb.so.1` → `libxcb1`
- `libfontconfig.so.1` → `libfontconfig1`,
  `libfreetype.so.6` → `libfreetype6`

Also add, only when the tool is actually missing (`command -v`):

- DeepSpeed `evoformer_attn` compile (OpenFold2 / OpenFold3):
  `build-essential` (`g++`), `ninja-build` (`ninja`)
- unknown soname → find the owner with
  `apt-file search <soname>` on a machine that has it, or the
  Debian package search page. Never guess a package name into the
  generated script; list the soname as unresolved and say so.

Which OSS tree needs what is a property of its dependencies, not of
this skill — derive it from Step A rather than from this list. As a
rough expectation: trees whose data pipeline imports RDKit (Boltz-1/2,
OpenFold3, Protenix-v2) tend to pull the X11 / cairo set through
RDKit's bundled drawing library, while OpenFold2 usually needs only
the compiler toolchain for the DeepSpeed kernel. Confirm with the
scan; do not pre-install from this paragraph.

### Step C — generate the install script

Always write the script, even when you expect to run it yourself —
it is the artifact the human needs and the record of what the env
required.

```bash
# Substitute the real set from Steps A+B; these names are only an example.
pkgs=(libxrender1 libxext6 libx11-6)
cat > "$WORKDIR/ref_data/apt_install.sh" <<EOF
#!/usr/bin/env bash
# Distro packages required by the OSS stack for this bench.
# Generated by bench-perf-oss. Run on the host / in the container.
set -euo pipefail
sudo apt-get update
sudo apt-get install -y ${pkgs[*]}
EOF
chmod +x "$WORKDIR/ref_data/apt_install.sh"
cat "$WORKDIR/ref_data/apt_install.sh"
```

### Step D — confirm the privileged install scope

Show the generated script and explain which missing dependency requires
each package. Run it only when the user has authorized this system change
in the current session. Root access or passwordless elevation alone does
not establish authorization. Otherwise use the hand-off in Step E.

After authorization, choose the available rung. Run the update and install
as separate commands, and stop if the update fails. Do not prompt for a
password (`sudo -n` is non-interactive).

```bash
if [ "$(id -u)" -eq 0 ] && command -v apt-get >/dev/null; then
  apt-get update || exit 1
  apt-get install -y "${pkgs[@]}"
elif command -v sudo >/dev/null; then
  sudo -n apt-get update || exit 1
  sudo -n apt-get install -y "${pkgs[@]}"
fi
```

### Step E — if you cannot install, give the human the command

**This is a required step, not a fallback of last resort.** On most
benching machines the agent is not root, has no passwordless
`sudo`, or runs under a harness that refuses to execute `apt-get` /
`sudo` at all. When any of that is true, the agent's job is to
**produce the command and ask the human to run it** — not to work
around the missing library and not to stop silently.

1. **Stop the OSS track.** Do not skip the package, do not
   `pip install` a substitute, do not stub or symlink the `.so`, do
   not point `LD_LIBRARY_PATH` at an unrelated conda prefix, and do
   not start any timed run.
1. **Print the command inline in the reply.** The human may never
   open `$WORKDIR`, so a path alone is not a hand-off. Give one
   copy-pasteable block, and mention the generated script as a
   convenience second.
1. **Justify it in one or two sentences** — which import or compile
   failed, which soname it wanted, and which OSS dependency pulls
   it. A bare `sudo apt-get install …` with no reason is not
   actionable.
1. **Say what happens next**: they run it, reply, and you re-verify
   (Step F) and continue.

Reply template — fill the placeholders from Steps A and B, keep the
shape:

The OSS dependency needs the missing sonames, so the import or compile
fails with the recorded error. Explain why package installation is
unavailable, then give the user the commands with the actual package names:

```bash
sudo apt-get update
# Continue only if the update succeeded.
sudo apt-get install -y <pkgs>
```

Mention the script saved at `$WORKDIR/ref_data/apt_install.sh` and ask the
user to reply after running it so the scan can resume.

Keep it to the failing import, the sonames, the reason the agent
cannot install, and the command. Do not paste the whole traceback.

While waiting, keep making progress on work that does not need the
OSS import: the sample manifest, the BioIR half of the locked
config, and the BioIR probe are all independent of these packages.
Do not idle, and do not start Phase 4 / 5.

If the human comes back with a failure (wrong distro, package not
found, no `sudo` for them either), do not improvise a workaround.
Re-derive the package name for their distro (Step B) and hand over a
corrected command.

### Step F — verify after install

Re-run the scanner (expect no unresolved sonames) and the OSS
import probe. Continue only after both pass.

```bash
"$OSS_PYTHON" bench/scan_missing_libs.py ... | tee "$WORKDIR/debug/scan_missing_libs_after.log"
for p in "${pkgs[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || echo "MISSING $p"; done
```

If a package the human installed still does not resolve its
soname, say so explicitly and re-derive the name — do not proceed
on a partially satisfied set.

Do **not**:

- `apt-get upgrade` CUDA, NVIDIA driver, or libc stacks (breaks BioIR)
- `conda install` these into BioIR / OSS / Miniforge base (OpenStructure
  conda is Step 7 only)
- continue with a missing `.so` and call the env ready

Record the package list, the generated script path, and who
installed them (`apt_packages`, `apt_install_script`,
`apt_installed_by`: `apt` | `sudo` | `human` | `none`) in
`bench_config.json` and `implementation-notes.md`.

## Step 4 — Two environments

Default when anything in Step 3 is hard.

```bash
# Keep the current BioIR interpreter untouched.
python3.12 -m venv "$WORKDIR/venv_oss"
OSS_PYTHON="$WORKDIR/venv_oss/bin/python"
"$OSS_PYTHON" -m pip install -U pip
```

Install [apt packages](#oss-system-packages-apt) first if OSS needs
them. Then install OSS from `$OSS_ROOT` into that venv using the
project's own docs (conda env file, `pip install -e .`, pinned
requirements).
Prefer BioIR-compatible versions of shared stacks when OSS can run
on them (same `+cu` torch, same numpy major). Rewrite any CUDA 12
pin to CUDA 13
([above](#cuda-12-packages--cuda-13)). Only then add OSS-only
kernels the README names (cuEq, DeepSpeed evo). For OpenFold2 /
OpenFold3, source-build DeepSpeed with **only**
`DS_BUILD_EVOFORMER_ATTN` (see
[below](#openfold2--openfold3--deepspeed-evoformer-kernel)). Do **not** install
`flash-attn`. Do not go back and retune BioIR to match this venv.

Verify:

```bash
"$OSS_PYTHON" -c "import torch; from <oss_package> import <Model>; print(torch.__version__, 'ok')"
"$BIOIR_PYTHON" -c "import bionemo_ir, torch; print(torch.__version__, 'bioir ok')"
```

The two `torch.__version__` strings may differ. That is expected.
Record both.

Do **not**:

- clone OSS `main` or a tag other than the pin
- `git pull` inside `3rdparty/openfold-3` or `3rdparty/protenix`
- `pip install -e $OSS_ROOT` into `BIOIR_PYTHON` when torch pins differ
- upgrade / downgrade BioIR's torch to satisfy OSS
- share `site-packages` via `PYTHONPATH` across the two venvs
- install OSS `*-cu12` / `[cu12]` / `+cu12*` wheels (remap to
  CUDA 13 first)

## Step 5 — Shared environment

Allowed only when Step 3 found no hard conflict.

Prefer `PYTHONPATH=$OSS_ROOT/src:$PYTHONPATH` or
`pip install --no-deps -e $OSS_ROOT` so OSS reuses the BioIR lock
instead of resolving its own pin set. Install
[apt packages](#oss-system-packages-apt) if OSS needs them. Then
install each missing **soft** dependency one at a time (CUDA 12
pins → CUDA 13, [above](#cuda-12-packages--cuda-13)), re-running
the BioIR probe after each install. If a soft install changes a
BioIR package or breaks `import bionemo_ir`, revert that install
(restore BioIR) and fall back to Step 4. Do not "fix" BioIR to make
the shared env work.

```bash
"$BIOIR_PYTHON" -c "import bionemo_ir; print('bioir still ok')"
```

## OpenFold2 / OpenFold3 — DeepSpeed Evoformer kernel

For **OpenFold2 and OpenFold3** OSS, the recommended triangle
attention path is DeepSpeed `DS4Sci_EvoformerAttention`. A PyPI
wheel does **not** ship that op. OF2 / OF3 `pip` / conda deps still
pull that wheel — **replace it** with a from-source install into
`$OSS_PYTHON`, compiling **only** the Evoformer / triangle-attention
kernel.

Do **not** `DS_BUILD_OPS=1`. Do not build FusedAdam, transformer,
sparse attention, CUTLASS ops, or other DeepSpeed kernels. Do not
leave the kernel to JIT on the first timed forward.

Do **not** set `CUTLASS_PATH=$REPO/3rdparty/cutlass`. That tree is
CUTLASS 4 (BioIR / CuTeDSL). DeepSpeed `evoformer_attn` expects
CUTLASS **≥ 3.1** headers from a 3.x checkout. Clone `v3.6.0`.

```bash
# After OSS_PYTHON exists (Step 4 or 5). OF2 / OF3 only.
git clone --branch v3.6.0 --depth 1 \
  https://github.com/NVIDIA/cutlass.git "$WORKDIR/oss/cutlass"
export CUTLASS_PATH="$WORKDIR/oss/cutlass"

# Prefer the DeepSpeed version OSS pins (OF2 environment.yml is
# deepspeed==0.14.5). Otherwise clone latest:
git clone --depth 1 https://github.com/deepspeedai/DeepSpeed.git \
  "$WORKDIR/oss/deepspeed"

# Uninstall the PyPI wheel first if OSS already pulled it.
"$OSS_PYTHON" -m pip uninstall -y deepspeed || true

# Non-editable, -v, one arch. An editable pip install silently builds
# nothing — see deepspeed-evoformer.md.
export TORCH_CUDA_ARCH_LIST=$("$OSS_PYTHON" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0))') MAX_JOBS=16
DS_BUILD_OPS=0 DS_BUILD_EVOFORMER_ATTN=1 \
  "$OSS_PYTHON" -m pip install "$WORKDIR/oss/deepspeed" \
  --no-build-isolation -v > "$WORKDIR/debug/ds_build.log" 2>&1
```

`DS_BUILD_OPS=0` turns every op off; `DS_BUILD_EVOFORMER_ATTN=1`
turns only `evoformer_attn` back on. `TORCH_CUDA_ARCH_LIST` should
name the bench GPU (DeepSpeed prunes `< sm_70`).
If the compile fails to find CUDA libs, expose `LIBRARY_PATH` /
`LD_LIBRARY_PATH` from the conda prefix.
Missing `g++` / `ninja` is an
[apt package](#oss-system-packages-apt) (`build-essential`,
`ninja-build`) — try apt, then hand the command to the human.

**Budget 5–10 minutes** for the compile (three CUTLASS translation
units, most of it in `attention.o`) and do not kill it early. A
first attempt that finishes in seconds compiled nothing.

Verify **before** any forward:

```bash
"$OSS_PYTHON" - <<'PY'
from deepspeed.git_version_info import installed_ops, accelerator_name, torch_info
from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention
assert accelerator_name == "cuda", accelerator_name
assert torch_info["version"] != "0.0", torch_info
assert installed_ops.get("evoformer_attn"), installed_ops
print("evoformer_attn", installed_ops["evoformer_attn"], "ok")
PY
```

`accelerator_name` / `torch_info` are part of the gate, not extra
noise: a build whose `setup.py` could not import torch reports
`cpu` / `'0.0'` and drops the op while `pip` still exits 0.

Then enable it on the OSS inference script (`use_deepspeed_evo_attention`
/ `use_deepspeed_evoformer_attention`). Record `deepspeed_evo_attn: true`
in `bench_config.json`. A fallback to PyTorch attention is a different
column, and only with user approval.

BioIR does **not** use this kernel (CuTeDSL / auto triangle backends).
Do not install DeepSpeed into `$BIOIR_PYTHON` for this op.

Boltz and Protenix-v2 skip this step unless their OSS script
requires the same kernel.

Full build, trap, and verification detail:
[deepspeed-evoformer.md](deepspeed-evoformer.md).

## OSS kernel flags — installing is not enabling

Installing a kernel package is half the job. Most OSS trees gate each
accelerated path behind a **config flag that ships off**, so a run
left on defaults quietly measures the slow path and reports it as
OSS's number.

The rule: for every accelerated kernel the OSS tree integrates,
**enable it in the OSS inference config**, following what the OSS
docs recommend rather than what the config file happens to default
to. In particular, **if OSS integrates cuEquivariance, the inference
config always enables cuEq.** BioIR is measured with its own fast
kernels; OSS gets the same treatment.

Also: these paths usually **compose**. cuEq and the DeepSpeed
evoformer kernel are not an either/or choice — enable both unless
the OSS docs say they conflict.

The exact flags, combinations, and preset overrides belong in the
model profile. For OpenFold3, see [models/of3.md](models/of3.md).

### Verify the flag landed

Two checks. The kernel package must import in the OSS interpreter:

```bash
"$OSS_PYTHON" -c "import cuequivariance_torch as cet; print('cueq', cet.__version__)"
```

and the **resolved** config the model actually runs must show the
flag on — read it back after the runner merges YAML, presets, and
CLI, not from the YAML you wrote:

```python
mem = config.settings.memory.eval
assert mem.use_cueq_triangle_kernels, mem
assert mem.use_deepspeed_evo_attention, mem
```

A flag silently dropped by a preset or a schema rename is the common
failure, and it is invisible in the latency number.

### Do not tune OSS internals

Leave OSS-internal kernel thresholds alone unless the OSS docs tell
you to change them. If a kernel falls back on shorter samples, that
is OSS's own dispatch behavior and belongs in the report as an
observation, not as a knob to turn.

Record the enabled set as `oss_kernels` in `bench_config.json`, e.g.
`{"cueq_triangle": true, "deepspeed_evo_attn": true, "flash_attn":
false}`, and name any kernel the tree supports that you could **not**
enable, with the reason.

## Step 6 — Dry import both sides

Before Phase 1, both probes must print `ok`. Save the commands and
output under `$WORKDIR/debug/env_probe_bioir.log` and
`env_probe_oss.log`.

Also confirm they see the **same GPU**:

```bash
"$BIOIR_PYTHON" -c "import torch; print(torch.cuda.get_device_name(0))"
"$OSS_PYTHON"   -c "import torch; print(torch.cuda.get_device_name(0))"
```

A mismatch usually means different `CUDA_VISIBLE_DEVICES`. Fix it
before any timed run.

## Step 7 — Install OpenStructure (after BioIR and OSS)

Do this **after** BioIR and OSS interpreters are installed and
probed. OpenStructure ships a large conda-forge stack (Boost,
gemmi, its own Python). Put it in a **dedicated conda env** so
those packages never mix with BioIR or OSS torch / CUDA pins.

Do **not**:

- `pip install` OpenStructure into `$BIOIR_PYTHON` or `$OSS_PYTHON`
- `conda install openstructure` into Miniforge **base**
- reuse a repo-root `miniconda3` / `miniforge3` base `ost`

Score after the writer, never inside the `model.forward()` clock.
Use only `ost compare-structures` — do not write a hand-rolled
lDDT.

**Reuse a dedicated env if it already exists:**

```bash
OST_ENV=${OST_ENV:-$WORKDIR/envs/ost}
if [ -x "$OST_ENV/bin/ost" ]; then
    OST_CMD="$OST_ENV/bin/ost"
    echo "OpenStructure found:"
    "$OST_CMD" --version
else
    echo "OpenStructure NOT found — creating dedicated env"
fi
```

**`openstructure` lives on `bioconda`, not `conda-forge`.** A
conda-forge-only channel list fails with
`PackagesNotFoundError: openstructure`. List both, conda-forge first,
because bioconda resolves its dependencies against it. Add
`--override-channels` so the bootstrapper's own channel config
(often `defaults`) cannot leak in and change what resolves.

**Any existing conda / mamba / micromamba may act as the
bootstrapper** — it only has to solve and create the prefix env.
Download Miniforge when the machine has none. Reusing a solver
binary is fine; reusing an `ost` **inside** that base is not, even
when one is sitting there (a repo-root `miniconda3` often has one).
The scorer this skill records must be the prefix env's own binary.

**Create the env only if missing**; `openstructure` goes into
`$OST_ENV`, never into base.

```bash
OST_ENV=${OST_ENV:-$WORKDIR/envs/ost}

# Bootstrapper: first existing conda-like solver, else fetch Miniforge.
CONDA_BIN=$(command -v conda mamba micromamba 2>/dev/null | head -1)
if [ -z "$CONDA_BIN" ]; then
    REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo /nonexistent)
    for c in "$REPO_ROOT"/mini{forge3,conda3}/bin/conda \
             "$HOME"/mini{forge3,conda3}/bin/conda; do
        [ -x "$c" ] && CONDA_BIN="$c" && break
    done
fi
if [ -z "$CONDA_BIN" ]; then
    MINIFORGE_DIR=${MINIFORGE_DIR:-$WORKDIR/miniforge3}
    MINIFORGE_INSTALLER=$(mktemp "$WORKDIR/ref_data/miniforge.XXXXXX.sh")
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -O "$MINIFORGE_INSTALLER"
    bash "$MINIFORGE_INSTALLER" -b -p "$MINIFORGE_DIR"
    # Retain the installer in the benchmark workspace for provenance.
    CONDA_BIN="$MINIFORGE_DIR/bin/conda"
fi

# Prefix env — isolated from base and from BioIR / OSS.
"$CONDA_BIN" create -y -p "$OST_ENV" \
    --override-channels -c conda-forge -c bioconda openstructure
OST_CMD="$OST_ENV/bin/ost"
"$OST_CMD" --version
```

The solve pulls a large stack (Boost, its own Python) and takes
minutes. If `openstructure` resolves nowhere, say so and stop — do
not substitute a different scorer.

Call `ost` by absolute path. Do not `conda activate` this env in
the BioIR or OSS shells — activation would change `PATH` /
`PYTHONHOME` and can break those interpreters.

Call it as:

```bash
"$OST_CMD" compare-structures \
    -m "$pred_cif" -r "$gt_path" \
    -o "$WORKDIR/debug/ost_${sample_id}.json" \
    --lddt --fault-tolerant \
    --min-pep-length 4 --min-nuc-length 4
```

Read `lddt` from the JSON. On non-zero exit, keep `lddt` null and
record stderr — do not invent a score or fall back to a custom
metric.

Record `ost_cmd` in `bench_config.json` and
`implementation-notes.md`. A missing `ost` after this step is a
hard blocker — do not time without a scorer.

## Step 8 — Install DockQ (interface scorer)

lDDT scores a whole predicted structure; it does not tell you
whether the **interfaces** are right, and a complex can score well
globally while every chain-chain contact is wrong. DockQ is the
required second metric on supported protein interfaces, so install it
here, alongside `ost` and before any timed run
([measurement.md](measurement.md#quality-lddt-and-dockq)).

Unlike OpenStructure this is a plain pip package, so the temptation
is to drop it into an existing interpreter. Do not: DockQ pins
**`numpy < 2.0`** (plus biopython, networkx, parallelbar), and this
stack runs numpy 2.x. Installing it into `$BIOIR_PYTHON` silently
downgrades numpy underneath the frozen BioIR env, which is the one
thing this playbook never allows.

Do **not**:

- `pip install DockQ` into `$BIOIR_PYTHON` or `$OSS_PYTHON`
- create the venv with `--system-site-packages`, which lets the
  numpy downgrade reach back into the parent environment

Creating the venv **from** `$BIOIR_PYTHON` is fine — a venv gets its
own `site-packages`, so the interpreter is a known-good starting
point without sharing packages.

```bash
DOCKQ_ENV=${DOCKQ_ENV:-$WORKDIR/envs/dockq}
if [ ! -x "$DOCKQ_ENV/bin/DockQ" ]; then
    "$BIOIR_PYTHON" -m venv "$DOCKQ_ENV"
    "$DOCKQ_ENV/bin/pip" install -q --upgrade pip
    # BioIR/OSS setup may leave a session-wide PIP_CONSTRAINT that pins
    # NumPy 2.x. It must not govern this isolated NumPy<2 scorer env.
    env -u PIP_CONSTRAINT -u PYTHONPATH \
        "$DOCKQ_ENV/bin/pip" install DockQ
fi
DOCKQ_CMD="$DOCKQ_ENV/bin/DockQ"
env -u PYTHONPATH "$DOCKQ_CMD" --help \
    > "$WORKDIR/debug/dockq_probe.log" 2>&1 || {
    echo "DockQ install failed — see debug/dockq_probe.log"; exit 1; }
DOCKQ_VERSION=$(env -u PYTHONPATH "$DOCKQ_ENV/bin/python" -c \
    "import importlib.metadata as m; print(m.version('dockq'))")
echo "DockQ $DOCKQ_VERSION at $DOCKQ_CMD"
```

There is **no `--version` flag** — probe with `--help` and read the
version from package metadata. The distribution is `dockq`
lowercase while the console script is `DockQ`. Unset any benchmark
`PYTHONPATH` for install and invocation so an OSS source tree cannot
leak package metadata or imports into the scorer env.

The install compiles a Cython extension, so it needs a C compiler
(`build-essential` / `gcc`). A compiler failure here is an
[apt problem](#oss-system-packages-apt), not a reason to skip the
metric. Let pip use **build isolation** here (the default) — the
build needs `cython` and `numpy < 2.0`, and the
`--no-build-isolation` habit from the BioIR and DeepSpeed steps
would make it compile against the wrong numpy.

Call it by absolute path, per sample, after the writer:

```bash
env -u PYTHONPATH "$DOCKQ_CMD" "$pred_cif" "$gt_path" \
    --json "$WORKDIR/debug/dockq_${sample_id}.json" --short
```

Model comes **first**, native second — reversing them silently
scores the wrong direction. From the JSON, read `GlobalDockQ` (the
**mean** DockQ over native interfaces; `best_dockq` is the sum),
`best_result` (per-interface metrics, keyed by chain pair), and
`best_mapping_str` (the `model:native` assignment it chose).

DockQ 2.1.3 depends on mmCIF metadata that some valid OSS writers omit
or misclassify. Before invocation, check both inputs with a real CIF
parser and write a scorer-only copy under `$WORKDIR/debug/` when
needed:

- add `_atom_site.occupancy=1.0` when that optional column is absent
- set `_atom_site.group_PDB=HETATM` for chains the manifest explicitly
  declares as small molecules when a writer emitted their atoms as
  `ATOM` (a CCD ligand can itself have an amino-acid name such as TYR)

Do not infer ligand identity from residue names. Do not alter
coordinates, chain ids, or residue ids, and never edit the written
prediction in place. Apply the same manifest-driven rule to both
sides. Record the normalized input path and every operation in
`dockq_normalization`.

Five things about the invocation matter for a bench:

- **Let DockQ pick the chain mapping.** Without `--mapping` it
  searches for the model→native assignment that maximizes the
  score, which is what you want when the two stacks label chains
  differently. Record the mapping it reports. Pass `--mapping` only
  to pin a known-correct assignment on a large homomer for speed,
  and then pass the same one on both sides.
- **mmCIF chains come from `label_asym_id`**, not `auth_asym_id`,
  so a predicted CIF and a ground-truth CIF can disagree about
  chain names. That is the mapping search's job, not something to
  fix by rewriting the CIF.
- **Ligands need `--small_molecule`** and their own chain id, and
  only LRMSD is reported for them. Set the flag from the manifest,
  identically on both sides.
- **Lock sequence mismatch tolerance.** If DockQ reports a limited
  model/native sequence mismatch for a chain that should map, use the
  smallest `--allowed_mismatches` value that admits it and lock that
  value for every side. Do not increase it until an arbitrary mapping
  passes.
- **DockQ 2.1.3 is a protein docking scorer.** Its small-molecule path
  supports protein-ligand interfaces, but RNA/DNA complexes can fail
  inside protein sequence alignment. Do not reinterpret that as
  DockQ `0.0`: skip invocation when any RNA/DNA polymer is present and
  record `dockq=null`, `dockq_status="unsupported_chain_types"` on
  both sides.
- **`--capri_peptide` is not trustworthy** (upstream says so).
  Do not use it.
- **Exit 1 with "Could not find interfaces in the native" is the
  monomer answer, not a failure.** DockQ needs at least one native
  interface, so single-chain samples exit non-zero by design.
  Distinguish that message from a real error and record the sample
  as `single_chain`, never as `0.0`
  ([measurement.md](measurement.md#quality-lddt-and-dockq)).

Record `dockq_cmd`, `dockq_env`, `dockq_version`, and `dockq_args`
in `bench_config.json`. A missing `DockQ` after this step is a hard
blocker for any supported protein-protein or protein-small-molecule
complex.

## What to record

`$WORKDIR/ref_data/bench_config.json` must include:

- `bioir_python`, `oss_python`
- `bioir_torch`, `oss_torch` (full `torch.__version__`, including `+cu`)
- `isolation` — `two_venv` or `shared`
- `conflicts` — list of hard conflicts, or `[]`
- `container` — `nvcr.io/nvidia/pytorch:26.09-py3` (or override)
- `bioir_install` — `editable` | `wheel` | `pip`
- `cutedsl_force_cubin` — always `true`
- `ost_cmd` — `$OST_ENV/bin/ost` (dedicated env, Step 7)
- `ost_env` — conda prefix, default `$WORKDIR/envs/ost`
- `ost_version` — `ost --version` output, e.g. `2.10.0`
- `dockq_cmd` — `$DOCKQ_ENV/bin/DockQ` (dedicated venv, Step 8)
- `dockq_env` — venv path, default `$WORKDIR/envs/dockq`
- `dockq_version` — from package metadata (there is no `--version`)
- `dockq_args` — the exact flag set used on **both** sides, e.g.
  `["--short"]` or `["--short", "--small_molecule"]`
- `dockq_input_normalization` — scorer-only missing-occupancy rule
  described above
- `deepspeed_evo_attn` — `true` for OpenFold2 / OpenFold3 OSS
  (source-built `evoformer_attn` only); `false` otherwise
- `oss_kernels` — every accelerated path the OSS config enables,
  e.g. `{"cueq_triangle": true, "deepspeed_evo_attn": true,
  "flash_attn": false}`
- `apt_packages` — distro packages OSS needed (or `[]`)
- `apt_install_script` — `$WORKDIR/ref_data/apt_install.sh` (always
  generated when `apt_packages` is non-empty), or `null`
- `apt_installed_by` — `apt` | `sudo` | `human` | `none`
- `cuda12_remaps` — `{from, to}` list, or `[]`

`$WORKDIR/implementation-notes.md` gets one entry for every install
decision (why two venvs, why a package was `--no-deps`, why an OSS
pin was left unmatched so BioIR could stay put).
