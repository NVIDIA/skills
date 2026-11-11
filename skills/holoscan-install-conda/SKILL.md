---
name: holoscan-install-conda
description: "Handle Holoscan SDK 5.x Conda installation requests: explain 5.0 package unavailability, check later releases, and select a supported installation method."
license: Apache-2.0
metadata:
  author: "Holoscan Team <holoscan-team@nvidia.com>"
  version: "2.0.0"
  github-url: "https://github.com/nvidia-holoscan/holoscan-sdk"
  tags:
    - holoscan
    - installation
    - conda
---

# Holoscan 5.0 and Conda

This skill checks Conda availability for the requested Holoscan SDK 5.x release.
SDK 5.0 has no Conda package. For later releases, it follows the release's Conda
instructions when support is documented and matching packages are published.
Otherwise, it offers container, Debian, wheel, or source alternatives while
preserving the requested SDK version and C++ or Python goal.

Conda-forge Holoscan 4.x packages are a different release line and cannot satisfy
a 5.x installation request.

1. Establish the requested SDK release. For 5.0, explain that Conda installation
   is unavailable; do not install Miniforge, create an environment, or run a
   Holoscan Conda solve. For a later release, follow its documented Conda method
   only if that release explicitly supports it and the matching packages are
   published. If availability is unknown, report the uncertainty and offer the
   verified alternatives below rather than inventing an installation command.
2. When Conda is unavailable or unconfirmed, preserve the requested SDK version
   and the user's C++ or Python goal. Offer the matching skill when available,
   or use the release's installation guide directly:

   | Goal | Method |
   | --- | --- |
   | Ready-to-use C++ and Python environment | `holoscan-install-container` |
   | Native C++ headers and libraries | `holoscan-install-debian` |
   | Python environment | `holoscan-install-wheel` |
   | Early Access binaries unavailable, or SDK development | `holoscan-install-source` |

3. Confirm which alternative the user wants if they have not already chosen
   one. If the host or goal is unclear, use `holoscan-setup` to assess it.
4. If the user explicitly needs 4.x, consult that release's Conda instructions
   and report the version change before proceeding. Keep an existing Conda
   installation intact; its presence does not establish availability for the
   requested release.

Use the selected release notes and
[installation guide](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation)
to establish current support. Do not treat a 4.x search result as 5.x support.
