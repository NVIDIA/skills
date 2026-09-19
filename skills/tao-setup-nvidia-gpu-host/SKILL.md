---
name: tao-setup-nvidia-gpu-host
description: >-
  Host setup for TAO GPU backends. Checks and, after user approval, installs
  minimum-compatible NVIDIA driver, CUDA Toolkit, and NVIDIA Container Toolkit
  versions for Docker/local-Docker and Kubernetes GPU worker hosts. TAO-wide
  defaults can be overridden by the selected model's runtime profile. The
  `--check-only` path works on any Linux distribution; `--install` automates
  debian-family (Ubuntu/Debian/Pop!_OS/Mint/Zorin/Raspbian), rhel-family
  (Fedora/RHEL/Rocky/AlmaLinux), and suse-family (openSUSE/SLES) hosts, and
  prints actionable manual-install steps for everything else. Use when the user
  asks to "set up an NVIDIA GPU host", "check TAO Docker GPU runtime", or
  prepare a Kubernetes GPU worker for TAO.
license: Apache-2.0
compatibility: Runs `--check-only` on any Linux distribution. `--install` automates Ubuntu 22.04/24.04 + Debian 12 (apt), Fedora + RHEL/Rocky/AlmaLinux 9/10 (dnf), and openSUSE Leap / SLES 15 (zypper). Requires sudo/root, internet access to NVIDIA package repositories (and download.docker.com on rhel-family), and an x86_64 or aarch64 (sbsa) host. Other distributions (Arch, Alpine, Gentoo, NixOS, …) get a clear error that names the version targets and the NVIDIA install-guide URL.
metadata:
  author: NVIDIA Corporation
  version: "0.1.2"
allowed-tools: Read Bash
tags:
- setup
- nvidia
- cuda
- docker
- kubernetes
---

# NVIDIA GPU Host Setup

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

Use this setup skill before TAO workflows run on the `docker`, `local-docker`,
or `kubernetes` backend. The TAO-wide default minimums are:

- NVIDIA driver `>=580` (open kernel module preferred)
- CUDA Toolkit `>=13.0`
- NVIDIA Container Toolkit `>=1.19.0`
- Docker engine — only installed for `docker` / `local-docker` backends and
  only when Docker is missing. The package picked depends on the distro
  family (`docker.io` on Debian-family by default, `moby-engine` /
  `docker-ce` from `download.docker.com` on RHEL-family, `docker` on
  SUSE-family). Pass `--skip-docker-install` to opt out.

The check is safe and read-only by default — it works on any Linux
distribution because it only probes `nvidia-smi`, the CUDA toolkit path,
the installed container-toolkit package version (via `dpkg`/`rpm`/the
`nvidia-ctk` binary version), and the Docker daemon's NVIDIA runtime.

Installation must be explicitly authorized by the user and rerun with
`--install`. The install path is automated for these distro families:

| Family | Tested distros | Manager | Notes |
|---|---|---|---|
| debian | Ubuntu 22.04 / 24.04, Debian 12 (and derivatives Pop!_OS, Mint, Zorin, Raspbian, KDE Neon, etc. via `UBUNTU_CODENAME` / `VERSION_CODENAME`) | `apt-get` | Adds NVIDIA `cuda-keyring` + Container Toolkit `.list`. Docker via `docker.io` (override `$DOCKER_PACKAGE_DEBIAN`). |
| rhel | Fedora 39+, RHEL / Rocky / AlmaLinux 9 and 10 | `dnf` (or `yum`) | Adds NVIDIA `cuda-<distro>.repo` + Container Toolkit `.repo`. Docker via Fedora `moby-engine` when available, otherwise `docker-ce` from `download.docker.com`. |
| suse | openSUSE Leap 15, SLES 15 | `zypper` | Adds the same NVIDIA `.repo` files. Docker via the distribution `docker` package. |
| other (Arch, Alpine, Gentoo, NixOS, FreeBSD, …) | n/a | n/a | `--install` exits with a clear error listing the version targets and the NVIDIA install-guide URLs. Install manually, then rerun `--check-only`. |

## Quick Start

From the skill bank root:

```bash
# Check the local Docker backend host.
bash skills/platform/tao-setup-nvidia-gpu-host/scripts/setup-nvidia-gpu-host.sh --backend docker --check-only

# Install or repair after user approval.
bash skills/platform/tao-setup-nvidia-gpu-host/scripts/setup-nvidia-gpu-host.sh --backend docker --install

# Check a Kubernetes GPU worker host.
bash skills/platform/tao-setup-nvidia-gpu-host/scripts/setup-nvidia-gpu-host.sh --backend kubernetes --check-only
```

> ⚠️ **Note — running non-interactively (agent/skill runs):** a skill run has no terminal, so the
> installer's `Continue? [y/N]` prompt cannot be answered. After running `--check-only` to preview and
> getting the user's approval, append the assume-yes flag (`--yes`) to the `--install` command so it
> proceeds without a prompt — this auto-confirms installation of system packages (NVIDIA driver, CUDA
> Toolkit, NVIDIA Container Toolkit, and Docker for Docker backends) and modifies the host, so only do
> this on a host you control. A person running `--install` directly at a terminal gets the prompt instead.

## Workflow Contract

Docker and Kubernetes workflows must run the check before submitting GPU work:

```bash
SB="${TAO_SKILL_BANK_PATH:-${TAO_SKILL_BANK_ROOT:-$PWD}}"
SETUP_SCRIPT="${SB}/skills/platform/tao-setup-nvidia-gpu-host/scripts/setup-nvidia-gpu-host.sh"

bash "$SETUP_SCRIPT" --backend docker --check-only || {
  echo "MISSING: TAO GPU host runtime is not ready."
  echo "After user approval, run (append --yes for non-interactive agent runs):"
  echo "  bash \"$SETUP_SCRIPT\" --backend docker --install"
  exit 1
}
```

Never install silently. If the check fails, explain what is missing, ask the
user to authorize the fix, then run the install command and rerun the check.

## Model Runtime Override Contract

Platform defaults apply when a model has no override. A model that needs a
different validated host stack declares it in
`references/skill_info.yaml`:

```yaml
runtime_requirements:
  gpu_host:
    min_driver_version: '<version>'
    min_cuda_version: '<version>'
    min_container_toolkit_version: '<version>'
```

Read those values before the final platform preflight and pass them to the
matching `--min-*-version` flags. Model minimums take precedence for that
workflow only; do not rewrite the platform defaults or requirements for other
models. Version checks use numeric lower bounds, so later compatible releases
pass. Always retain the selected-image GPU smoke test because a version bound
cannot prove support for a particular GPU architecture.

## What The Installer Does

The installer dispatches on the detected distribution family. On every
supported family it adds NVIDIA's CUDA and Container Toolkit repositories
(if missing), installs packages that satisfy the active minimums, optionally
installs Docker, wires the NVIDIA Docker runtime, and adds the invoking user to
the `docker` group.

Common steps (all families):

1. Adds NVIDIA's CUDA repository if missing (apt `cuda-keyring` deb,
   `cuda-<distro>.repo` for dnf/zypper).
2. Adds NVIDIA's Container Toolkit repository if missing (`.list` for apt,
   `.repo` for dnf/zypper).
3. Installs the matching kernel header / devel package for the running
   kernel.
4. Installs the current open-driver and Container Toolkit packages from the
   configured repositories plus the CUDA Toolkit package selected by
   `--min-cuda-version`, then verifies all three against the active minimums.
5. For Docker backends and when Docker is missing, installs Docker
   (override / opt-out flags below), enables/starts the daemon, then runs
   `nvidia-ctk runtime configure --runtime=docker` and restarts Docker
   when `systemctl` is available.
6. Adds the invoking user (`$SUDO_USER` if available, else `$USER`) to the
   `docker` group so subsequent shells can run `docker` without `sudo` —
   opt out with `--skip-docker-group`. **The new group membership does not
   take effect in the current shell**: log out and back in, or run
   `newgrp docker` in each new shell.
7. Attempts `modprobe nvidia` so verification can pass before reboot.

Family-specific package selections:

| Step | debian-family | rhel-family | suse-family |
|---|---|---|---|
| Kernel headers | `linux-headers-$(uname -r)` | `kernel-devel-$(uname -r)`, `kernel-headers-$(uname -r)` | `kernel-default-devel` |
| Driver | current `nvidia-open` (override: `$NVIDIA_DRIVER_PACKAGE_DEBIAN`) | current `nvidia-driver-cuda`, `kmod-nvidia-open-dkms` (override: `$NVIDIA_DRIVER_PACKAGE_RHEL`, `$NVIDIA_DRIVER_KMOD_RHEL`) | current `nvidia-open-driver-G06-signed-kmp-default` (override: `$NVIDIA_DRIVER_PACKAGE_SUSE`) |
| CUDA toolkit | package derived from the active minimum, such as `cuda-toolkit-13-0` | same | same |
| Container Toolkit | current `nvidia-container-toolkit` + base/tools/libs, then minimum-version validation | same | same |
| Docker | `docker.io` (override: `$DOCKER_PACKAGE_DEBIAN`) | `moby-engine`+`moby-cli` on Fedora when available, else `docker-ce docker-ce-cli containerd.io` from `download.docker.com` | `docker` |

## Verification

After installation, verify:

```bash
nvidia-smi
nvcc --version
docker info --format '{{json .Runtimes}}' | grep nvidia
sudo docker run --rm --runtime=nvidia --gpus all "$TAO_IMAGE" nvidia-smi -L
```

The detected driver, CUDA Toolkit, and Container Toolkit versions must meet the
active TAO-wide or model-specific minimums. Then run the selected image's GPU
smoke test; version comparison alone is not sufficient compatibility proof.

For a Cosmos backend, extend that smoke with the backend contract's Python and
entrypoint checks. Cosmos Framework must execute
`/workspace/.venv/bin/python` as a non-root UID, import
`cosmos_framework.callbacks.tao_status`, find native torchrun, and verify the
A100 PatchEmbed compatibility marker when the host reports compute capability
8.0. Cosmos-RL must resolve its requested action executable, import
system PyAV for video workflows, resolve the restricted FFmpeg `h264_cuvid`
decoder, load `libnvcuvid.so.1`, verify the backward-safe linear Qwen3-VL
PatchEmbed marker, and verify its checkpoint loader accepts the prepared
`qwen3_vl` directory. A container that only passes `nvidia-smi` is not ready
for Cosmos training.

## Kubernetes Notes

For self-managed Kubernetes clusters, run the host installer on every GPU
worker node or bake the same package set into the node image before installing
the NVIDIA GPU Operator or device plugin.

The workflow check also warns if `kubectl` is available but the cluster reports
no `nvidia.com/gpu` allocatable capacity. In that case, install/configure the
NVIDIA GPU Operator after the worker host runtime is ready:

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
helm install --wait gpu-operator -n gpu-operator --create-namespace nvidia/gpu-operator
```

Managed Kubernetes providers may own driver installation through node images or
GPU Operator policy. Do not overwrite a provider-managed GPU node without user
approval and a rollback plan.

## Failure Modes

**Unsupported distribution family**: `--install` automates debian-, rhel-,
and suse-family hosts. On Arch, Alpine, Gentoo, NixOS, FreeBSD, or anything
without `/etc/os-release` (e.g. macOS), the script exits with a clear error
that lists the four version targets and the upstream NVIDIA install-guide
URLs:

- `https://docs.nvidia.com/cuda/cuda-installation-guide-linux/`
- `https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html`
- `https://docs.docker.com/engine/install/`

Install those four pieces using your distribution's package manager and
rerun the script with `--check-only` to verify. The check is universally
portable — it only queries the binaries / package databases — so once the
runtime is in place the workflow contract is satisfied regardless of the
underlying distro.

**Unsupported Ubuntu/Debian derivative**: When `ID` is e.g. `pop`, `mint`,
`zorin`, `raspbian`, or another debian-family derivative, the script maps
the host onto the upstream Ubuntu/Debian CUDA repo via `UBUNTU_CODENAME` /
`VERSION_CODENAME` (`focal`/`jammy`/`noble` → Ubuntu 20.04/22.04/24.04;
`bullseye`/`bookworm`/`trixie` → Debian 11/12/12). If the host's codename
doesn't match a known upstream release, `--install` exits with the same
manual-install guidance described above.

**Docker not installed**: `--check-only` reports `MISSING: Docker is not
installed` and prints the exact rerun command appropriate to the detected
distro family. The default `--install` path installs Docker (`docker.io` /
`moby-engine` / `docker-ce` / `docker` depending on family), enables/starts
the daemon, configures the NVIDIA runtime, and adds the invoking user to
the `docker` group. If you prefer to manage Docker yourself, install it
before rerunning the script or pass `--skip-docker-install`.

**Docker installed but `docker run` still needs sudo**: The script adds the
invoking user to the `docker` group, but Linux only refreshes group
membership on a new login session. Log out and back in, or run
`newgrp docker` in each new shell, until the new membership is active.

**Docker runtime still missing**: Restart Docker, then rerun
`nvidia-ctk runtime configure --runtime=docker`.

**Detected version is below the active minimum**: Rerun the same command with
`--install` after approval, preserving any model-specific `--min-*-version`
flags. Package-name environment overrides select distribution-specific driver
packages but do not weaken the minimum-version checks.

**Driver installed but `nvidia-smi` fails**: Load the module with
`sudo modprobe nvidia` or reboot. Secure Boot may require MOK enrollment on
systems where it is enabled.

**Kubernetes still has no GPU capacity**: Confirm the driver works on each GPU
node with `nvidia-smi`, then check the GPU Operator/device plugin pods and node
labels.
