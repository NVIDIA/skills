<!-- SPDX-License-Identifier: Apache-2.0 -->

# Installation choices and documentation

## Installation choices

[NVAT downloads](https://developer.nvidia.com/nvat-downloads) is the authority
for available OS/architecture/package combinations. The source guides describe
Ubuntu 22.04 and 24.04 on x86-64 and aarch64. Check the selected release's
support matrix before treating another combination as supported.

| Need | Component |
| --- | --- |
| Command-line tool | `nvattest`, with its `libnvat` dependency |
| C/C++ application development | `libnvat` and `libnvat-dev` headers |
| Rust application development | Safe `nv-attestation-sdk` wrapper, C library, and the binding build requirements in that release |

For a network package installation, configure the official repository before
installing NVAT, or reuse a matching repository that is already configured.
The `sudo` commands below require administrator privileges to install public
repository-signing keys, refresh APT metadata, and install NVAT system packages.
Run them on the intended target with authorized administrator access.
For example, the official Ubuntu 24.04 x86-64 network instructions are:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install nvattest
```

Select the detected OS, architecture, and installer type on
[NVAT downloads](https://developer.nvidia.com/nvat-downloads) and verify its
current instructions before running commands. The example above is specific
to Ubuntu 24.04 x86-64. The download selector names the server ARM64 target
`arm64-sbsa`; do not construct repository URLs by substituting `aarch64`.
Local repository installers and tarball installations have different steps;
use the selected artifact's instructions. The CLI package does not provide
development headers or establish CoRIM capability by itself.

For source builds, select a public release tag or commit and follow that
revision's CLI or C development guide. Record the revision. The CLI normally
builds the C library with it; `USE_SYSTEM_NVAT` changes this behavior and
requires a compatible installed library. Check that revision's CMake options
to determine whether an installed `find_package(nvat)` integration exists.
Prevent a source-built CLI from accidentally loading a runtime library from
another release.

For CC evidence collection, the documented setup requires a supported GPU
attached to an Ubuntu 22.04/24.04 CVM. NVML and container device exposure are
separate prerequisites. Non-CC Corelib collection has different hardware,
privilege, and release requirements; read its documentation before selecting it.

For Rust, inspect the selected revision's `Cargo.toml`, build script, and
examples. The local guide describes Rust 1.80+, Clang/libclang for bindgen, and
the native C library. Confirm registry availability for the chosen version;
do not assume a crates.io release. A Cargo Git dependency selects the package
by name and a tag or revision; a local path must point to the crate directory.
Avoid copying unverified dependency keys from a prose example.

## Check GPU and NVML access

Run these checks where evidence will be collected, including inside a target
container or CVM. Confirm that each utility is available before using it:

```sh
nvidia-smi -L
python3 -c 'import ctypes; ctypes.CDLL("libnvidia-ml.so.1"); print("NVML library loads")'
```

The first command checks whether the utility can enumerate GPUs. The second
checks whether the process can load the NVML library; it does not establish
that evidence collection is permitted. If Python is unavailable, inspect the
library with the platform's loader tools. A missing `nvidia-smi` executable
alone does not prove that no GPU is present.

For a container, compare device visibility inside it with the host and inspect
the configured GPU device/library exposure. For a CVM, confirm GPU assignment
and the supported CC configuration. Use the relevant collection command as the
functional check once those prerequisites are satisfied. Diagnose missing
access or libraries before proposing driver, firmware, or GPU-mode changes.

## Select the Blackwell workflow

For Blackwell, FSP responders currently use CoRIM, while GSP responders use
TCG RIM. This may change depending on the deployment model; confirm support
for the installed SDK release. Use known deployment configuration or validated
certificate information to establish the responder. Ask for it when unknown;
do not choose a path solely from the `BLACKWELL` architecture value.

| Responder | Current appraisal path | Collection format |
| --- | --- | --- |
| Blackwell GSP | Legacy TCG RIM, using `attest` | Legacy evidence from `collect-evidence` |
| Blackwell FSP | CoRIM, using `attest-v2` | CoRIM-compatible CMW from a supported collection path |

## Legacy evidence collection

For "How do I collect evidence for Hopper?", start with the documented legacy
NVML collection path. Confirm `nvattest version`, `nvattest collect-evidence
--help`, and GPU/NVML access in the supported CC environment. The command is
`collect-evidence`; there is no `collect` alias in the current CLI.

```bash
nvattest --format json collect-evidence \
  --device gpu \
  --gpu-evidence-source nvml \
  > hopper.collection.json 2> hopper.collection.log
```

Check the exit status immediately. When a relying party supplies a challenge,
add `--nonce` with that challenge in hex and retain it for subsequent
verification. Do not substitute Corelib for Hopper: the current CLI's Corelib
architecture selector supports Blackwell, not Hopper.

Legacy JSON output wraps an `evidences` array with `result_code` and
`result_message`. GPU entries contain fields such as `arch`, `nonce`,
`evidence`, and `certificate`; Hopper entries identify `HOPPER`. These are
serialized legacy GPU evidence, not CMW or an attestation verdict.

For saved-file legacy appraisal, the GPU evidence reader expects the evidence
array. After successful collection, extract `.evidences` from the CLI wrapper
and use the matching `attest --gpu-evidence-source file --gpu-evidence-file`
workflow with the original nonce. Check the release's documented schema and
command help before reuse.

| Requested input/output | Command family |
| --- | --- |
| Hopper legacy/TCG GPU evidence | `collect-evidence` produces the legacy JSON wrapper |
| Blackwell GSP legacy/TCG GPU evidence | `collect-evidence` produces the legacy JSON wrapper |
| Legacy/TCG evidence appraisal | `attest`, using the documented GPU evidence input |
| CMW for a supported CoRIM workflow | `collect-evidence-v2` produces CMW |
| CoRIM appraisal producing EAR | `attest-v2` consumes CMW |

Both collections can be JSON while having different schemas. Renaming a file
or extracting its `evidences` array does not convert legacy evidence to CMW.
If the user specifically requests CoRIM on Hopper, check the actual release,
evidence profile, and conversion support. The GPU generation alone does not
establish whether it is supported.

## Documentation map

Base: <https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/>

| Question | Official page |
| --- | --- |
| Overview and supported workflows | [Overview](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/overview.html) |
| CLI installation and prerequisites | [CLI introduction](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-cli/introduction.html) |
| CLI commands and flags | [Command reference](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-cli/command-reference.html) |
| Hopper/legacy GPU evidence collection | [collect-evidence](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-cli/command-reference.html#collect-evidence) |
| Legacy/TCG RIM appraisal | [attest](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-cli/command-reference.html#attest) |
| Building the CLI | [CLI development](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-cli/development.html) |
| C installation and API entry points | [C introduction](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-c/introduction.html) |
| C configuration | [C configuration](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-c/configuration.html) |
| C usage and resource management | [C user guide](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-c/user_guide.html) |
| CoRIM concepts and C workflow | [CoRIM verifier](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-c/corim_verifier.html) |
| CoRIM trust stores and HTTP configuration | [CoRIM configuration](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-c/corim_configuration.html) |
| EAR fields and result semantics | [EAR format](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-c/ear_format.html) |
| Rust setup | [Rust introduction](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-rust/introduction.html) |
| Rust API usage | [Rust user guide](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-rust/user_guide.html) |
| Rust CoRIM APIs | [Rust CoRIM verifier](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-rust/corim_verifier.html) |
| Public versions | [Releases](https://github.com/NVIDIA/attestation-sdk/releases) |

If a hosted page is unavailable, the public repository contains Markdown under
`docs/nv-attestation-sdk-cpp/`, C declarations under
`nv-attestation-sdk-cpp/include/nvat.h.in`, CLI definitions under
`nv-attestation-cli/src/`, and Rust bindings under `nv-attestation-sdk-rust/`.
Use [the public source](https://github.com/NVIDIA/attestation-sdk) at the matching
revision. An internal checkout can help investigate a future feature, but does
not establish that users can install it publicly.
