---
name: holoscan-setup
description: "Assess a host and choose a Holoscan SDK 5.x installation method using release availability, CUDA 13 compatibility, and component needs."
license: Apache-2.0
metadata:
  author: "Holoscan Team <holoscan-team@nvidia.com>"
  version: "2.0.0"
  github-url: "https://github.com/nvidia-holoscan/holoscan-sdk"
  tags:
    - holoscan
    - installation
    - setup
---

# Holoscan 5.x Setup

## Establish the target

Determine the requested 5.x release or source branch and whether the user needs
C++, Python, specific modules, or SDK development. Read the release notes and
[SDK installation guide](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation)
with the matching 5.x version selected. An unversioned page may describe 4.x;
for a development checkout use its `docs/sdk_installation.mdx` and `DEVELOP.md`.
Treat Early Access binary methods as tentative until the exact artifacts are
confirmed published. Package-name registration alone is not availability.

## Inspect and assess

Use read-only checks on the intended Linux host:

```bash
uname -m
cat /etc/os-release
nvidia-smi
python3 --version
getconf GNU_LIBC_VERSION
docker --version
docker info
```

Report unavailable tools separately from incompatible hardware. Inspect an
existing venv or SDK checkout only at user-selected or relevant known paths.
Use `python -m pip list` in the intended environment and `dpkg-query -W 'holoscan*'`
on Debian systems to distinguish 4.x installations from usable 5.x ones.
A GPU passthrough check that runs/pulls an image is a separate execution step;
a Docker version check alone does not establish passthrough.

The 5.0 baseline is CUDA 13 with dGPU support, Ubuntu 24.04 native packages,
and x86_64/SBSA arm64. Wheels require glibc 2.39 and regular CPython 3.12–3.14.
The `nvidia-smi` CUDA value describes driver capability, not toolkit presence.
Check driver/runtime compatibility for the selected artifact; a CUDA 12-only
host cannot be routed to a nonexistent 5.x CUDA 12 package. ARM64 does not
establish Tegra/iGPU support. Confirm hardware-specific support before offering
installation on Jetson, IGX, DGX Spark, or Grace systems.

To check an already-pulled image, run the bundled helper from this skill's
installation directory with the **full confirmed 5.x tag**, for example:

```bash
bash scripts/check_ngc_image.sh v5.0.0-cuda13
```

It inspects the local image only, makes no registry request, and does not prove
publication, GPU passthrough, or runtime success. Report its result accordingly.

## Recommend and route

Present the relevant choices with separate publication and host-readiness status:

| Method | 5.x workflow |
| --- | --- |
| Development container | Published CUDA 13 multiarch image; full C++ and Python stack under `/usr` |
| Debian | Published Ubuntu 24.04 C++ `holoscan` metapackage or Core/optional modules |
| Python wheels | Published `holoscan` metapackage or CUDA 13 component wheels in a venv |
| Source | Explicit 5.x ref; build container and checkout-specific options |
| Conda | Unavailable for 5.0; check the selected release's documentation for later 5.x support |

Recommend source when the requested Early Access binaries are unavailable.
Recommend the development container for a ready-to-use environment when the
image and host support are confirmed; native C++ and Python needs can justify
Debian or wheels. If the user already selected a supported method, honor that
choice and proceed within their authorization. Otherwise ask one concise
method-selection question before installation.

Use the corresponding `holoscan-install-container`, `holoscan-install-debian`,
`holoscan-install-wheel`, or `holoscan-install-source` skill when available.
Pass the release/source ref, host architecture, driver/runtime status, selected
components, existing installation details, and verified artifact availability.
If a companion skill is unavailable, consult the release instructions directly.
For a Conda-specific request, `holoscan-install-conda` explains the limitation
and alternatives; installing the 4.x Conda package does not meet a 5.x request.

## Report the outcome

Report the selected SDK version/ref, method, component versions, and checks
actually executed. Separate availability checks, imports, graph execution, and
module/GPU qualification. Missing binaries should produce an actionable source
alternative; missing host support should produce the prerequisite to resolve.
