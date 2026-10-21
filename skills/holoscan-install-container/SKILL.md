---
name: holoscan-install-container
description: "Install and verify the Holoscan SDK 5.x CUDA 13 development container after checking release availability and host compatibility."
license: Apache-2.0
metadata:
  author: "Holoscan Team <holoscan-team@nvidia.com>"
  version: "2.0.0"
  github-url: "https://github.com/nvidia-holoscan/holoscan-sdk"
  tags:
    - holoscan
    - installation
    - container
---

# Holoscan 5.x Development Container

## 1. Check the release and platform

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

## 2. Select and pull the image

Inspect Linux architecture, GPU model, driver, Docker, and NVIDIA Container
Toolkit. The planned image is one multi-architecture amd64/arm64 dGPU image:
`nvcr.io/nvidia/clara-holoscan/holoscan:v<SDK_VERSION>-cuda13`.
Check the [NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/containers/holoscan)
and release notes for the exact published tag and supported host. There are no
`cuda12-dgpu` or `cuda12-igpu` tags for this 5.x workflow. GPU passthrough tests
may pull an image; explain the download before running one if needed.

Use the exact published tag within the installation scope the user authorized.
When the method and release are already selected, proceed with that choice.
Ask only when a missing release or host fact prevents safe selection.
For a plan-only request, return the commands without pulling or running images:

```bash
# SDK_VERSION is the exact published 5.x tag version from the release notes.
IMAGE="nvcr.io/nvidia/clara-holoscan/holoscan:v${SDK_VERSION}-cuda13"
docker pull "$IMAGE"
docker image inspect --format 'ID={{.Id}} RepoDigests={{json .RepoDigests}}' "$IMAGE"
docker run --rm --gpus all "$IMAGE" nvidia-smi
```

Record the image ID and repository digests from inspection for the final report.
If no repository digest is available, report the image ID and that limitation.

The development image contains the installed C++ and Python SDK under `/usr`,
all three optional modules, examples, and RacerX data. It is distinct from the
source-build container managed by `./run`.

## 3. Verify the installed SDK

Use a bounded Core example and Python API discovery first. Locate the examples
instead of assuming the 4.x `/opt/nvidia/holoscan` layout or a fixed multiarch
library directory:

```bash
docker run --rm --gpus all "$IMAGE" bash -eu -c '
  example=$(find /usr/lib -type f -path "*/holoscan/examples/hello_world/cpp/hello_world" -print -quit)
  test -n "$example"
  "$example"
  python3 -c "import holoscan as hs; print(hs.__version__); print(hs.RealtimeClock); print(len(hs.registry.operators()))"
'
```

Expect `Hello Holoscan!`, the requested 5.x version, and successful API access.
Run the installed `first_pipeline/python/first_pipeline.py` example when present
for graph execution coverage. Check the release's installed example tree and
README before adding module/GPU tests; a Core import alone does not qualify
Vision, Visualization, Inference, graphics, or camera support.

## 4. Report results and reuse the image

```bash
docker run --rm -it --gpus all "$IMAGE" bash
```

Add mounts for the user's project or devices as the application needs them.
For GUI workloads follow the selected release's display/graphics instructions;
keep initial verification independent of display and datasets.

- GPU passthrough failure: check the host driver and Container Toolkit setup.
- Missing tag: report publication status and offer source; keep the 5.x target.
- Missing examples or Python components: inspect the pulled image tag/digest;
  verify it is the development image for the selected release.

Report the image tag/digest, architecture, SDK version, tests run, and skipped
hardware or module checks.
