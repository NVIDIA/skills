---
name: cudnn-install
description: Install, verify, or upgrade the cuDNN backend and Python or C++ frontend. Check existing PyTorch, JAX, and NVIDIA NGC installations, select package extras, and diagnose cuDNN library conflicts.
---

# Install cuDNN

## Select the environment

Ask the user before choosing an install path, unless they already gave the answer:

- Are you developing in **Python**, **C++**, or both?
- Do you want to keep the installed versions, or upgrade the **frontend**,
  **backend**, or both to the latest compatible stable release?

Inspect the OS, CPU architecture, GPU, driver, CUDA version, active Python
interpreter or C++ toolchain, and container. Ask only for details that remain
unknown. For Python, confirm the target virtual environment and whether the
user needs a wheel or a source/editable install. Check for PyTorch and JAX.

Use the current NVIDIA guides and their platform-specific child pages:

- [Backend installation](https://docs.nvidia.com/cudnn/installation/backend)
- [Frontend installation](https://docs.nvidia.com/cudnn/installation/frontend)
- [Supported platforms and versions](https://docs.nvidia.com/cudnn/latest/reference/support-matrix)

Append `.md` to a documentation page URL to read its Markdown. For a source
install, read the selected checkout's
[pyproject.toml](https://github.com/NVIDIA/cudnn-frontend/blob/develop/pyproject.toml),
[CMakeLists.txt](https://github.com/NVIDIA/cudnn-frontend/blob/develop/CMakeLists.txt),
and [cuDNN CMake discovery](https://github.com/NVIDIA/cudnn-frontend/blob/develop/cmake/cuDNN.cmake)
before forming commands. These paths belong to the cuDNN Frontend source repo,
not to this installed skill. Without a checkout, read them from the source
repo at the selected tag or commit; the links use `develop` for navigation.
Resolve a selected branch or tag once and use the same commit for all reads.
For a released wheel, inspect that release's package metadata instead; a wheel
install does not require cloning the source repo. Do not apply `develop`
dependencies to a release. If metadata is unavailable, state the gap rather
than guessing version bounds, extras, or build requirements.

## Find cuDNN before installing it

The backend provides native cuDNN libraries. The frontend Python distribution
is `nvidia-cudnn-frontend`, imported as `cudnn`. Installing the frontend does not
prove that a usable backend is present.

1. Inspect installed package names, versions, requirements, and file locations
   with the target interpreter's
   [package metadata API](https://docs.python.org/3/library/importlib.metadata.html)
   or `python -m pip`.
   Look for `nvidia-cudnn-cu12`, `nvidia-cudnn-cu13`, JIT variants, `torch`,
   `jax`, `jaxlib`, and CUDA-specific JAX plugins. Check Conda packages if used.
2. GPU builds of PyTorch and JAX require cuDNN and may already supply it through
   bundled libraries or package dependencies. CPU-only builds do not establish
   cuDNN availability. In separate processes, check installed frameworks:
   - PyTorch: check the framework, CUDA, device, and backend versions:

     ```python
     import torch

     print(torch.__version__, torch.version.cuda)
     print(torch.cuda.is_available(), torch.backends.cudnn.version())
     ```

   - JAX: package/plugin requirements and `jax.devices()`; confirm a CUDA device.
     A package listing or CPU device alone does not prove cuDNN can load.
   Capture import or loader errors; do not interpret them as an absent package.
3. NVIDIA NGC framework containers often include cuDNN. Check the image tag,
   release notes, `/etc/nvidia-container-release` if present, package records,
   and actual libraries. Do not reinstall solely because pip has no record.
4. For native installs, check package-manager records and installed files:
   cudnn_version.h, libcudnn shared libraries on Linux, or cudnn64_9.dll on
   Windows. Inspect `CUDNN_PATH`,
   `CUDAToolkit_ROOT`, `LD_LIBRARY_PATH`, and Windows `PATH` as applicable.
   C++ and source builds need headers and development libraries as well as
   runtime libraries. Framework runtime availability alone is insufficient.

Record the backend's owner, version, CUDA flavor, and library path. Reuse a
working compatible install. If multiple copies exist, identify which library
the process loads before changing search paths. On Linux, inspect
`/proc/self/maps` after loading cuDNN; on Windows, inspect loaded module paths.
Do not use the CUDA maximum shown by `nvidia-smi` as the installed toolkit version.

## Choose versions and install

Find the latest stable frontend and backend releases at execution time. Check
the driver, GPU, CUDA flavor, framework constraints, and required API support.
State installed and proposed versions. If the newest release conflicts with
the environment, explain the constraint and offer a compatible release or a
separate environment. Do not silently change PyTorch/JAX or remove their pins.

Keep installs inside the selected environment. Preserve lockfiles and container
image constraints. Preview dependency changes with the package manager's dry
run where available. Ask about unresolved version choices before making changes;
do not ask again when the user already selected the upgrade scope.

### Backend

For Python, use the matching CUDA package from the backend guide, for example:

```bash
python -m pip install nvidia-cudnn-cu13
```

Use `nvidia-cudnn-cu12` for a CUDA 12 target. Install only one CUDA flavor in the
environment; these wheels share library paths. Use `--upgrade` only for the
selected upgrade, or `==<version>` for a compatible pin. Check framework pins
before replacing a framework-managed backend. Do not use `--no-deps` to hide
conflicts. Do not select a JIT backend unless the user needs it.

For C++, prefer the documented Linux development packages or Windows installer
for the detected platform. Follow repository setup and package names from the
current guide. A pip backend needs explicit include/library paths for native
use. On Windows, ensure the selected cuDNN `bin` directory is on `PATH`.
Use `CUDNN_PATH` and `CUDAToolkit_ROOT` for nondefault locations; verify their
layout against the selected source revision's CMake discovery file linked above.
Do not copy libraries into unrelated CUDA
or framework directories. For NGC, prefer a compatible image tag over replacing
its framework-managed stack; explain any requested in-container changes.

### Python frontend and dependencies

For a wheel, use the target interpreter:

```bash
python -m pip install nvidia-cudnn-frontend
```

Add `--upgrade` if the user selected the latest compatible frontend. For a
source install, run `python -m pip install -e .` from the selected checkout,
after checking its build requirements and CUDA/cuDNN development files.

Read these project metadata sections rather than installing every group:

| Section | Use |
| --- | --- |
| project.requires-python | Enforce the selected revision's Python minimum. |
| project.dependencies | Base runtime dependencies; pip installs them automatically. |
| project.optional-dependencies | Add only extras needed for the requested APIs. |
| dependency-groups | Repo development and framework tools; not wheel extras. |
| build-system.requires | Source build tools; normally supplied by build isolation. |

At the time this skill was added, the base dependencies are
`nvidia-cutlass-dsl[cu13]` and `apache-tvm-ffi`, with platform markers. Extras are
`cutedsl` (`cuda-python`), `comm` (`nvshmem4py-cu13`), `cutile` (`cuda-tile`), and
`triton` (`triton`, with platform/Python markers). The `cutedsl` extra is retained
for compatibility; CuTeDSL is already a base dependency. Preserve the current
version bounds and markers. Do not infer the backend CUDA flavor from the DSL
extra: check compiler/runtime compatibility separately for the requested kernels.

Select extras as, for example, `"nvidia-cudnn-frontend[comm]"` or `".[cutile]"`.
The `cutile` engines need a compatible `tileiras` compiler; this extra does not
install it. Check the requested engine's docs before adding it.

The repo's `torch` group contains `torch` and `torch-c-dlpack-ext`; `jax` contains
JAX with its declared minimum; `dev` contains test, notebook, and format tools.
From the repo, a pip version that supports dependency groups can use
`python -m pip install --group torch`, `--group jax`, or `--group dev` as needed.
These are not `[torch]`, `[jax]`, or `[dev]` extras. Select CUDA-enabled framework
wheels/plugins separately using their official install instructions. Check
constraints before a group install can replace an existing framework.

### C++ frontend

The C++ frontend is header-only. Use the selected release's `include/` directory
and `#include <cudnn_frontend.h>`, or integrate the `cudnn_frontend` CMake target.
Check the repo's compiler standard, CMake minimum, and CUDA/backend requirements.
Do not install Python extras for a C++-only workflow.

For a repo build with the selected backend and toolkit:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Add `-DCUDNN_PATH=<root>` and `-DCUDAToolkit_ROOT=<root>` when needed. Confirm
that CMake resolved the intended headers and libraries. If the user requested
an upgrade, update the frontend checkout/tag as well as the selected backend;
do not overwrite local changes.

## PyTorch compatibility check

`PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK=1` bypasses PyTorch's cuDNN version
compatibility check in versions that support it. It does not install cuDNN,
select a library, satisfy package pins, or make incompatible libraries work.
It can permit a process to continue with a backend that causes errors or
incorrect results. Leave it unset for normal installation and validation.

For an incompatibility error, compare PyTorch's compile/runtime cuDNN versions
and the actual loaded path. First restore its expected backend or remove a
conflicting library search path. If the user needs a controlled experiment,
explain the mismatch and limit the override to that test process. On Linux:

```bash
PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK=1 python <test-script.py>
```

Set it before PyTorch initializes cuDNN. Do not add it to shell startup files,
project defaults, or container images. Report use of the override and repeat
validation without it before declaring the install compatible.

## Verify and report

For Python, run `python -m pip check` and a fresh-process check:

```python
import cudnn

print(cudnn.__file__)
print(cudnn.__version__, cudnn.backend_version())
handle = cudnn.create_handle()
cudnn.destroy_handle(handle)
```

Check the import path against the intended checkout for editable installs.
Run a small GPU convolution through each installed framework the user uses;
for JAX, wait for completion with `block_until_ready()`. Check the loaded backend
path in these processes too. A successful import alone is not a GPU execution test.

For C++, compile and run an existing frontend GPU sample or the backend guide's
`mnistCUDNN` sample. Report compile-only checks separately from GPU execution.
Use the selected source revision's
[test guide](https://github.com/NVIDIA/cudnn-frontend/blob/develop/test/AGENTS.md)
for further repo tests; do not copy obsolete test paths from the web guide.

Report the selected environment, frontend/backend versions, library paths,
installed extras/groups, changes made, and checks passed or unavailable.
If no NVIDIA GPU is accessible, mark GPU execution unverified.
