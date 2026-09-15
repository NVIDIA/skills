# Installation procedure

Commands run from the product checkout identified in SKILL.md. Consult that checkout's
`README.md` and `pyproject.toml` if its revision differs from this procedure.

## Preflight

```bash
ldd --version | head -1
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4.31
nvidia-smi
nvcc --version
uv --version
docker --version
git --version
command -v wget unzip
docker run --rm --gpus=all ubuntu:24.04 nvidia-smi
```

Require glibc >= 2.38 and GLIBCXX_3.4.31 before the expensive build; Ubuntu 24.04 meets this
floor. A successful Docker build does not prove the resulting library can load on the host.
Require driver >= 580 and the CUDA toolkit >= 12.8. The toolkit supplies headers and link
libraries for pycuda's source build; pip runtime wheels alone are insufficient.

If GPU access fails, check the
[NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Docker socket access and GPU access are separate checks. Host package installation, daemon
restarts, and Docker group changes require the user's normal administrative workflow; report
what is missing when those privileges are unavailable. Newly granted group membership requires
a new login/shell. Reuse tools and access already present.

## Pipeline and SAM3

Capture absolute paths so subsequent directory changes do not change the venv target:

```bash
PIPELINE_ROOT="$PWD"
PIPELINE_PARENT="$(dirname "$PIPELINE_ROOT")"
```

For a new environment:

```bash
uv venv --python 3.12
uv sync --extra foundationpose
```

For an existing environment, preserve out-of-band SAM3, TAO Deploy, and pycuda:

```bash
uv sync --inexact --extra foundationpose
```

Reuse an existing sibling SAM3 checkout; otherwise:

```bash
git clone https://github.com/facebookresearch/sam3 "$PIPELINE_PARENT/sam3"
```

Install the sibling, not the unrelated package named `sam3` on PyPI:

```bash
uv pip install --python "$PIPELINE_ROOT/.venv/bin/python" -e "$PIPELINE_PARENT/sam3"
"$PIPELINE_ROOT/.venv/bin/python" -c "import numpy, cv2, scipy; print(numpy.__version__, cv2.__version__, scipy.__version__)"
```

Expect numpy 1.26.x. If an out-of-band install changed it, restore the project's pins with
`uv sync --inexact --extra foundationpose` from the product root. Preserve the setuptools < 81
pin: SAM3 imports `pkg_resources`.

The checkpoint is gated at [facebook/sam3](https://huggingface.co/facebook/sam3).
Use existing access or a usable cached checkpoint. If access is missing, the user must request
the model grant and authenticate with `"$PIPELINE_ROOT/.venv/bin/hf" auth login` in their shell.
Do not print tokens. The `hf` executable arrives through SAM3's dependencies and may not be on
PATH before venv activation.

## FoundationPose build and weights

Reuse the sibling checkout if present; otherwise:

```bash
git clone https://github.com/nvidia-isaac/foundation-pose-inference-library.git \
  "$PIPELINE_PARENT/foundation-pose-inference-library"
```

Enter that checkout. Create `.env` from `.env.example` only if it does not exist; preserve any
existing configuration. On a fresh setup set `FP_UID` and `FP_GID` to the current user's IDs.
Keep `FP_WEIGHTS_DIR=./weights` unless using an intentional custom layout.

```bash
cd "$PIPELINE_PARENT/foundation-pose-inference-library"
mkdir -p data weights engine_cache
./run_dev.sh build
./run_dev.sh run --rm build
scripts/download_weights.sh
cd "$PIPELINE_ROOT"
uv sync --inexact --extra foundationpose
```

Create bind-mount directories before running Docker so the daemon does not create root-owned
ones. The downloader requires wget and unzip. The documented weights are public; on download
failure inspect HTTP/network errors and retry a transient failure once before reporting it.

The pipeline does not read the library's `.env`. It looks under `FOUNDATIONPOSE_ROOT` for
`build/libfoundation_pose_nvidia.so`, `weights/refiner_net.onnx`, and `weights/score_net.onnx`.
If these move, use consistent `--fp-library`, `--fp-refine-model-path`, and
`--fp-score-model-path` overrides. Changing `FP_WEIGHTS_DIR` alone does not update pipeline paths.

## TAO Deploy

From the product root:

```bash
uv pip install --python .venv/bin/python --no-deps nvidia-tao-deploy==7.1.0
uv pip install --python .venv/bin/python pycuda
```

`--no-deps` is required for this stack: TAO's dependency pins conflict with SAM3's numpy < 2
requirement. Do not add a `tao` extra to the project; uv resolves even unselected extras and this
conflict makes all syncs fail. The project's declared dependencies supply `omegaconf` and
`matplotlib`; if absent, sync with `--inexact --extra foundationpose` rather than applying another
ad-hoc install. Older TAO Deploy releases may lack the usable `cv/depth_net` configuration.

## Runtime libraries

Set the absolute `FOUNDATIONPOSE_ROOT` and `LD_LIBRARY_PATH` from SKILL.md before checking:

```bash
ldd "$FOUNDATIONPOSE_ROOT/build/libfoundation_pose_nvidia.so"
```

There should be no `not found` entries. Locate missing NVIDIA runtime libraries in this venv's
installed wheels. Non-NVIDIA dependencies may be system libraries; do not assume every missing
library comes from a Python package. Never mix another venv's CUDA/cuDNN libraries into the path.

`FOUNDATIONPOSE_ROOT` is needed at import time; `--foundationpose-root` alone does not replace it.
No `PYTHONPATH` modification is needed for the installed pipeline package.
