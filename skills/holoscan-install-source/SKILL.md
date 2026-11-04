---
name: holoscan-install-source
description: "Build and verify Holoscan SDK 5.x from an explicit source ref with the repository run wrapper; use for Early Access or SDK development."
license: Apache-2.0
metadata:
  author: "Holoscan Team <holoscan-team@nvidia.com>"
  version: "2.0.0"
  github-url: "https://github.com/nvidia-holoscan/holoscan-sdk"
  tags:
    - holoscan
    - installation
    - source
---

# Build Holoscan 5.x from Source

## Choose the source and prerequisites

Use source for unpublished Early Access features, SDK changes, custom components,
or debug builds. This workflow uses the checkout's build container with Ubuntu
24.04 and CUDA 13, targeting x86_64 or arm64 dGPU. Check Linux, disk capacity,
GPU/driver, Git, Docker, buildx, and NVIDIA Container Toolkit availability.
Explain the container download and build cost before starting.

Select the explicit **5.x source ref** from the release notes or the user's
requested development branch. Inspect an existing checkout before reusing it;
keep its branch and local changes intact. Resolve the requested ref to its full
commit, using the intended remote's ref for a remote branch and fetching that
ref if necessary. From the existing repository root, compare its HEAD with the
resolved ref before building:

```bash
git status --short
git rev-parse HEAD
git rev-parse --verify "${SDK_REF}^{commit}"
```

`SDK_REF` must identify the selected tag, commit, or remote-qualified branch.
If the ref cannot be resolved or the commits differ, stop and report the
mismatch. Preserve the existing branch and working tree; offer a separate
checkout at the requested ref. Report any local changes even when HEAD matches,
and preserve them when the user requested building their modified checkout.

For a new public checkout:

```bash
# SDK_REF is a confirmed published 5.x tag or development branch.
git ls-remote --heads --tags https://github.com/nvidia-holoscan/holoscan-sdk.git
mkdir -p ~/holoscan
git clone --branch "$SDK_REF" https://github.com/nvidia-holoscan/holoscan-sdk.git ~/holoscan/holoscan-sdk-5x
cd ~/holoscan/holoscan-sdk-5x
cat VERSION
./run --help
./run build --help
```

Verify `VERSION` is 5.x. Read this checkout's `AGENTS.md`, `DEVELOP.md`, and
`docs/sdk_installation.mdx`; these govern an unreleased branch. In an internal
checkout the exported source root is the directory containing `VERSION` and
`run`; run the commands from that directory. Never assume the repository's
default branch or a 4.x example is the requested 5.x source.

## Build

```bash
./run check_system_deps
./run build --dryrun
./run build
```

The wrapper builds, configures, compiles, and installs. It enables Python,
examples, operator testing, and optional modules by default. Useful choices:

| Option | Meaning |
| --- | --- |
| `--type debug` / `--type rel-debug` | Debug / release with debug information |
| `--arch arm64` | ARM64 target; consult the checkout's cross-compilation prerequisites |
| `--buildpath build-custom` | Explicit build directory |
| `--installprefix install-custom` | Explicit install prefix |
| `--build-python false` | C++-only build |
| `--config "-DHOLOSCAN_BUILD_VISION=OFF"` | Disable an optional component when supported by the checkout |

Use the same target selection for build, directory queries, and launch. The
standard trees are `build-<arch>` and `install-<arch>`; use `./run get_build_dir`
and `./run get_install_dir` rather than the 4.x `build-cu<N>-<arch>` naming.
Custom paths must be passed explicitly. Start with `--reconfigure` for stale
configuration; inspect what a cleanup command removes before using it.

## Verify

For the default build directory:

```bash
./run test --name 'EXAMPLE_CPP_HELLO_WORLD_TEST|EXAMPLE_CPP_HELLO_CORE_TEST|EXAMPLE_PYTHON_FIRST_PIPELINE_TEST'
```

Confirm the selected tests were found and executed, rather than treating zero
tests as success. A C++-only build omits the Python case. For a custom directory:

```bash
./run launch build-custom --run-cmd "ctest -N"
./run launch build-custom --run-cmd "ctest -R 'EXAMPLE_CPP_HELLO_WORLD_TEST|EXAMPLE_CPP_HELLO_CORE_TEST|EXAMPLE_PYTHON_FIRST_PIPELINE_TEST' --output-on-failure"
```

These cover greeting, bounded C++ Core execution, and the 5.x Python pipeline.
Run additional module/GPU tests appropriate to the selected components; state
what remains untested. Cross-compilation does not qualify execution on the
target architecture.

Use the absolute install prefix as `CMAKE_PREFIX_PATH` for consumer projects
using `find_package(holoscan)` and `holoscan::core`. Report the source commit,
build options, build/install directories, and test results. For GPU passthrough
or buildx failures, report the failing command and diagnostic output, then
resolve the prerequisite identified by the checkout's checks. If the cause
remains unknown, stop before building and request the missing host information
instead of retrying indefinitely or changing unrelated system settings.
