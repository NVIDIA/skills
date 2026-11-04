---
name: holoscan-install-wheel
description: "Install Holoscan SDK 5.x CUDA 13 Python wheels in an isolated environment, choosing the full stack or version-matched components."
license: Apache-2.0
metadata:
  author: "Holoscan Team <holoscan-team@nvidia.com>"
  version: "2.0.0"
  github-url: "https://github.com/nvidia-holoscan/holoscan-sdk"
  tags:
    - holoscan
    - installation
    - wheel
---

# Holoscan 5.x Python Wheel Installation

## Release and platform checks

This skill targets **Holoscan SDK 5.x**. Select the requested 5.x release before
choosing artifacts. Read its release notes and the matching
[SDK installation guide](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation).
Select the 5.x documentation version; an unversioned page may describe 4.x.
For an unreleased checkout, use its `docs/sdk_installation.mdx` and `DEVELOP.md`.
Early Access binary commands describe planned artifacts, so confirm publication
for the exact release before installing. If unavailable, explain that status
and offer `holoscan-install-source`; keep the requested 5.x version.

The 5.0 artifact set targets CUDA 13 and dGPU systems on x86_64 or arm64.
Native packages use Ubuntu 24.04; wheels use glibc 2.39 and regular CPython
3.12–3.14. Check the selected release for changes to that matrix. There is no
CUDA 12 variant or separate Tegra/iGPU artifact in this set. A driver reporting
CUDA 12 support needs a compatibility assessment or driver upgrade before a
CUDA 13 installation; the `nvidia-smi` CUDA field is a driver capability, not
the installed toolkit version. Never infer compatibility from that field alone.

## Choose a distribution

Wheels contain Python bindings, typing information, and native runtime libraries.
C++ headers, CMake files, examples, and sample data are separate. For native
C++ development use Debian packages or a source build.

| Goal | Distribution |
| --- | --- |
| Full stack | `holoscan` (selects `holoscan-cu13`) |
| Core only | `holoscan-core-cu13` |
| Optional modules | `holoscan-vision-cu13`, `holoscan-visualization-cu13`, `holoscan-inference-cu13` |

Inspect the host, `python3 --version`, and `getconf GNU_LIBC_VERSION`. Confirm
regular CPython 3.12–3.14, glibc 2.39 or newer, a compatible Linux architecture,
and the CUDA 13 driver/runtime prerequisites. An active Conda environment does
not imply a supported Conda package. Prefer a separate venv for this workflow.
If these prerequisites are unmet, report the failed check and stop before
installation. Compare the driver version against the selected release's
documented CUDA 13 driver requirement; a toolkit version alone is insufficient.

Confirm the exact published 5.x distribution version on PyPI or the selected
release's documented index; reserved package names are not installable SDKs.
If absent, offer source instead of resolving an unpinned package to 4.x.
Keep all components on the same SDK version.

## Install and verify

If an installation or verification command fails, stop and report the failing
command and output. Do not continue to the example or report success after a
failed prerequisite, package-index, dry-run, install, dependency, or import check.

Choose a fresh venv path, or reuse an environment the user selected after
checking its contents. Explain the chosen packages before installing:

```bash
python3 -m venv ~/holoscan/venv-5x
source ~/holoscan/venv-5x/bin/activate
# SDK_VERSION is the exact published Python distribution version for 5.x.
python -m pip index versions holoscan
python -m pip install --dry-run "holoscan==${SDK_VERSION}"
python -m pip install "holoscan==${SDK_VERSION}"
python -m pip check
python -m pip list --format=freeze
python -c 'import holoscan as hs; print(hs.__version__); print(hs.RealtimeClock); print(len(hs.registry.operators()))'
```

For Core-only use `holoscan-core-cu13` in the index, dry-run, and install
commands. Add optional component wheels at the exact matching SDK version.
Vision resolves CV-CUDA and NPP Python dependencies. The host supplies the
driver and CUDA 13 runtime; Inference expects TensorRT 10 in the application
environment. Verify optional providers separately when selected:
`holoscan_vision`, `holoscan_visualization`, and `holoscan_inference`.

For execution coverage, obtain `examples/first_pipeline/python/first_pipeline.py`
from the source ref corresponding to this release. Inspect it before execution
and run it in the venv; it is bounded and needs no RacerX data or display.
Prefer an existing matching checkout. Do not construct a Git tag from a Python
version string: Early Access packaging versions and source refs can differ.
The 4.x `hello_world.py`, `tensor_interop`, and `video_replayer` smoke sequence
does not describe the 5.x Python examples.

Report the Python, SDK, and component versions; import, dependency, and graph
checks; and any skipped module tests. An import check alone is not full runtime
qualification. Missing runtime libraries should be traced to the selected
component's prerequisites before changing unrelated system paths.
