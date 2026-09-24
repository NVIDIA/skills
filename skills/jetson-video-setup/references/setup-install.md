# Direct installation guide

Use this guide only when the user explicitly asks to install, set up, repair,
or create a fresh environment. A readiness or report-only request is read-only.
Inspect the live candidate, show the exact command, and use the standard
package or Python tool directly.

## Product boundaries

| Product | Acquisition | Verification |
|---|---|---|
| Native Video Codec SDK | `nvidia-video-codec-sdk` from the configured public Jetson APT release | Package-owned `AppEncCuda` followed by package-owned `AppDec` |
| PyNvVideoCodec | Public `PyNvVideoCodec` wheel and dependencies in one isolated venv | Wheel-owned `basic/encode.py` followed by `advanced/decode_perf.py` or `advanced/decode.py` |

The products are independent. Do not install the native SDK for a Python-only
request or create a Python environment for a native-only request. CUDA, driver,
compiler, SDK, and Python package versions are separate facts.

## APT safety

Before any APT change:

1. Inspect `/etc/nv_tegra_release` and the configured source files.
2. Run `apt-cache policy` for every proposed package and record the exact
   candidate and origin.
3. For NVIDIA SDK or CUDA packages, require the configured,
   signature-authenticated public Jetson source at
   `https://repo.download.nvidia.com/jetson/common` or
   `https://repo.download.nvidia.com/jetson/som`, with the `rNN.N/main` suite
   for the installed release. Never add or modify a source or signing key.
4. Pin each explicitly requested package to its reviewed candidate and run the
   exact `apt-get -s install --no-remove` command. Review every dependency and
   origin in the simulated transaction, but do not copy the transitive closure
   into the apply command. Stop on a removal, downgrade, unauthenticated
   package, changed candidate, or unrelated package set.
5. Apply the same top-level package specs using noninteractive `sudo -n`.
   Recheck their candidates immediately before applying, and do not retype or
   otherwise change a successfully simulated package spec. If passwordless
   authorization is unavailable, stop and report it; never switch to an
   interactive password prompt, `su`, or another escalation method.
6. Run `/usr/bin/dpkg --audit` after the mutation. A nonzero exit or any output
   blocks further setup until the package state is repaired by the operator.

Refresh package metadata once with `sudo -n apt-get update` only when the
configured source is correct and local metadata has no usable candidate or the
user explicitly requested a refresh. Reinspect candidates after the refresh;
do not apply a command based on the old result.

## Install the native Video Codec SDK

Inspect these packages and select only missing prerequisites:

- `nvidia-video-codec-sdk`
- `cmake`, `g++`, and either `ninja-build` or an available CMake generator
- `pkg-config`
- `libavcodec-dev`, `libavformat-dev`, `libavutil-dev`, and
  `libswresample-dev` for AppDec

Use the APT safety sequence above. A typical reviewed pair is shown below;
replace each `VERSION` with the exact live candidate and do not copy the
placeholder literally:

```bash
sudo -n env DEBIAN_FRONTEND=noninteractive \
  apt-get -s install -y --no-remove -- PACKAGE=VERSION
sudo -n env DEBIAN_FRONTEND=noninteractive \
  apt-get install -y --no-remove -- PACKAGE=VERSION
/usr/bin/dpkg --audit
```

Install one complete reviewed package set rather than repeatedly reacting to
compiler errors. For an explicitly requested reinstall, use `--reinstall` with
the same current candidate; do not uninstall the working package or base
dependencies first.

After APT succeeds, repeat package ownership, build-tool, CUDA, and `pkg-config`
checks, then perform the official native proof in
[setup-workflow.md](setup-workflow.md). An installed package that cannot build
or run its samples is not ready.

## Reuse an existing PyNvVideoCodec environment

Use the interpreter precedence in `SKILL.md`: an explicit path, a path
established in the current conversation, then the applicable conventional
profile path. Inspect it with the commands in
[setup-workflow.md](setup-workflow.md). If the expected distribution, loaded
module, dependencies, and samples are present, make no package change and run
the official proof directly.

Never scan for another venv, infer one from shell activation, or reinstall a
valid environment. If the conventional path exists but is invalid, report its
specific defect, leave it untouched, and ask for a different new absolute path.

## Create a PyNvVideoCodec environment

Under the user home directory, use `.venvs/nvcodec` for a new default smoke
environment and `.venvs/nvcodec-full` for a new `full-samples` environment when
the user does not supply a path. An explicit alternative must be absolute and
in a durable user-owned location. The path and every parent component must be
non-symlinked; the final path must not exist. Do not delete or repair an
existing directory.

Complete this prerequisite phase before creating the final venv directory:

- Inspect `python3-venv`, `python3-dev`, `g++`, and `make`. Install every
  missing item through the APT safety sequence. `python3 -m venv --help` is not
  proof that `ensurepip` is usable; require `python3 -I -m ensurepip --version`
  after installation.
- Resolve an existing `nvcc` from `PATH`, `/usr/local/cuda/bin/nvcc`, or the
  release-specific `/usr/local/cuda-*/bin/nvcc` locations before treating it as
  missing. Derive `CUDA_ROOT` from the verified compiler and check
  `include/curand.h` under `CUDA_ROOT`.
- Install the release-matched minimal CUDA build or CURAND development package
  only when that capability is absent from every existing toolkit. On R39.2
  the packages are `cuda-minimal-build-13-2` and `libcurand-dev-13-2`; derive
  their exact candidates and HTTPS Jetson origins rather than copying a
  version.

Do not install the full CUDA toolkit merely as a workaround when a suitable
toolkit is already present. Scope CUDA include/library variables to the PyCUDA
build; never add a CUDA stub directory to the runtime loader path.

Only after those checks pass, confirm the chosen final path is still absent,
create it once, and inspect it with standard Python commands:

```bash
python3 -m venv "$VENV"
"$VENV/bin/python" -I -m pip --version
```

For the default smoke profile, review the resolver result and then install.
Scope the verified toolkit paths to these PyCUDA build commands:

```bash
env PATH="$CUDA_ROOT/bin:/usr/bin:$PATH" \
  CPATH="$CUDA_ROOT/include" LIBRARY_PATH="$CUDA_ROOT/lib64" \
  "$VENV/bin/python" -I -m pip install --dry-run \
  'PyNvVideoCodec==2.1.0' 'numpy>=1.24' 'pycuda==2026.1'
env PATH="$CUDA_ROOT/bin:/usr/bin:$PATH" \
  CPATH="$CUDA_ROOT/include" LIBRARY_PATH="$CUDA_ROOT/lib64" \
  "$VENV/bin/python" -I -m pip install \
  'PyNvVideoCodec==2.1.0' 'numpy>=1.24' 'pycuda==2026.1'
"$VENV/bin/python" -I -m pip check
```

If the installed pip does not support `--dry-run`, report that limitation and
show the exact install command and package sources before applying it. Do not
disable TLS, signature, hash, or certificate checks, and do not put credentials
in an index URL or command.

Only when the user requests samples that import Torch, create a separate
`full-samples` venv and add the release-compatible CUDA-enabled Torch package
from its official index. For the currently documented CUDA 13 profile this is:

```bash
"$VENV/bin/python" -I -m pip install --dry-run \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.9.1+cu130'
"$VENV/bin/python" -I -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.9.1+cu130'
```

Confirm the version against the release-matched official PyNvVideoCodec
documentation before installing if the local release differs. Do not add
Torch to the default smoke profile.

Finally repeat the exact-interpreter inspection and run the wheel-owned proof
in [setup-workflow.md](setup-workflow.md). Return the absolute interpreter path
to the user and downstream skills.

## Failure handling

- Stop at the first failed mutation or inconsistent package state.
- Do not retry an unchanged failing command more than once.
- Preserve an independently successful product when both were requested.
- Name the exact command, failure, selected candidate or interpreter, and one
  actionable remedy. Do not convert an installation failure into “codec
  unsupported”.

## Primary sources

- <https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/read-me/index.html>
- <https://docs.nvidia.com/video-technologies/pynvvideocodec/read-me/index.html>
