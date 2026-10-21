---
name: holoscan-install-debian
description: "Install Holoscan SDK 5.x native C++ Debian packages on Ubuntu 24.04, choosing the full SDK or matching modular packages."
license: Apache-2.0
metadata:
  author: "Holoscan Team <holoscan-team@nvidia.com>"
  version: "2.0.0"
  github-url: "https://github.com/nvidia-holoscan/holoscan-sdk"
  tags:
    - holoscan
    - installation
    - debian
---

# Holoscan 5.x Debian Installation

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

## Select packages and inspect availability

This path supplies **C++** libraries, headers, CMake configuration, and C++
examples under `/usr`. Python bindings use `holoscan-install-wheel` separately.
Support in the planned artifact set is Ubuntu 24.04 x86_64 and SBSA arm64;
SBSA does not imply Tegra/iGPU support.

| Goal | Package |
| --- | --- |
| Full SDK | `holoscan` |
| Core libraries only | `holoscan-core` |
| Optional modules | `holoscan-vision`, `holoscan-visualization`, `holoscan-inference` |

Inspect `/etc/os-release`, `uname -m`, `nvidia-smi`, and existing packages.
Use the release-specific network/local installer from
[Holoscan Downloads](https://developer.nvidia.com/holoscan-downloads) to configure
the CUDA APT repository when needed. Choose `ubuntu2404/x86_64` or
`ubuntu2404/sbsa` as appropriate; use the published release-specific installer
instructions and package names.

After repository setup is authorized, refresh metadata and inspect candidates:

```bash
sudo apt update
apt-cache policy holoscan holoscan-core holoscan-vision holoscan-visualization holoscan-inference
apt-cache madison holoscan holoscan-core
```

Choose the exact APT version corresponding to the requested 5.x release.
A candidate from 4.x is not an available 5.x package. Keep every optional
module on the same Core release. Preview dependency changes before installing:

```bash
# APT_VERSION is the exact available 5.x Debian version, including its revision.
apt-get --simulate install "holoscan=${APT_VERSION}"
```

For Core-only, substitute `holoscan-core`. Add selected optional module packages
with their matching versions. Review the simulated transaction for removals or
an existing 4.x installation before applying it.
If simulation fails or proposes changes outside the requested scope, stop and
report the failing command/output or conflicting changes. Resolve the conflict
or obtain authorization for the expanded scope before installing. An existing
4.x installation alone does not block an upgrade the user requested.

After the simulated transaction is acceptable, install the selected package set:

```bash
sudo apt install "holoscan=${APT_VERSION}"
```

## Verify C++ support

```bash
dpkg-query -W 'holoscan*'
dpkg -L holoscan-core-cuda-13
```

The unsuffixed component packages select CUDA 13 native packages, such as
`holoscan-core-cuda-13`; inspect their installed versions and file lists.
Use the package file list to locate the installed `hello_world/cpp/hello_world`
and `hello_core/cpp/hello_core` executables under the system library hierarchy.
Run them and expect `Hello Holoscan!` and `holoscan core sample complete`.
Use the installed `hello_world` source and its `CMakeLists.txt` for a small
consumer build in a user-writable build directory; verify `find_package(holoscan)`
and the `holoscan::core` target. Check that `cmake --version` reports 4.0 or
newer first; Ubuntu 24.04's default CMake 3.28 cannot configure the consumer.
Inspect package contents rather than exporting the old `/opt/nvidia/holoscan/lib`
path. Record missing example packages as a verification gap.

Core declares CUDA development dependencies; Vision supplies its CV-CUDA
runtime, and Inference recommends TensorRT 10 runtime libraries. Check the
application's selected modules and dependencies before claiming full support.
Report installed package versions, C++ checks, and any untested GPU/module paths.
If Python import is requested, explain the C++/Python package split and use the
wheel workflow with the same SDK release.
