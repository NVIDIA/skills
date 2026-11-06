<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Pinocchio 4.0 runtime for the release-4.6 controller

The release-4.6 `ros-jazzy-isaac-ros-deploy-ros2-control` controller library
requires `libpinocchio_parsers.so.4.0.0`, while the Jazzy APT repository can
install Pinocchio 4.1 transitively through the G1 image packages. Install the
compatible C++ runtime alongside APT's version; do not rename or symlink 4.1
libraries to 4.0 filenames.

After installing the ROS dependencies, run the following inside the activated
Ubuntu 24.04 amd64 container, before starting the robot. The package hashes were
verified against the signed official ROS Jazzy snapshot dated 2026-06-18. The
matching Coal and EigenPy packages are required by this runtime. Stop on any
download, checksum, or extraction failure.

```bash
bash <<'BASH'
set -euo pipefail
test "$(dpkg --print-architecture)" = amd64
PINOCCHIO_STAGE=$(mktemp -d)
trap 'rm -rf "$PINOCCHIO_STAGE"' EXIT
cd "$PINOCCHIO_STAGE"
while read -r package version checksum; do
    archive="${package}_${version}_amd64.deb"
    curl --fail --location --show-error --output "$archive" \
        "http://snapshots.ros.org/jazzy/2026-06-18/ubuntu/pool/main/r/$package/$archive"
    printf '%s  %s\n' "$checksum" "$archive" > SHA256SUMS
    sha256sum --check SHA256SUMS
done <<'PACKAGES'
ros-jazzy-pinocchio 4.0.0-2noble.20260604.104112 e0aefff66a7a1ddc1eadcce74cef411912e780ca65075f05f3b20cfd4895bffe
ros-jazzy-coal 3.0.3-2noble.20260512.195409 228d16c27a2ab4f366002d29dd81736ba94bdc58a355f336d08a151b3a60a614
ros-jazzy-eigenpy 3.12.0-1noble.20260225.051537 2fe6cd5403adf5d7006cc1c78796281a6efc86092db9a90e89de37c83e0acf2c
PACKAGES
sudo mkdir -p /opt/pinocchio-4.0.0
for archive in ./*.deb; do
    sudo dpkg-deb --extract "$archive" /opt/pinocchio-4.0.0
done
BASH
```

In the robot launch shell, after sourcing ROS and the workspace, select these
libraries and verify the controller loads in a fresh Python process:

```bash
export PINOCCHIO_RUNTIME_PREFIX=/opt/pinocchio-4.0.0/opt/ros/jazzy
export LD_LIBRARY_PATH="$PINOCCHIO_RUNTIME_PREFIX/lib/x86_64-linux-gnu:$PINOCCHIO_RUNTIME_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 - <<'PY'
import ctypes
import os
ctypes.CDLL(
    '/opt/ros/jazzy/lib/libisaac_ros_deploy_ros2_control_controllers.so',
    mode=os.RTLD_NOW,
)
print('Pinocchio 4.0 controller load passed')
PY
```

Require a successful load before launching the robot. Keep this environment in
the robot launch shell, not Isaac Sim or global shell configuration. Extraction
does not change APT's installed packages or add a snapshot APT source. Reuse the
runtime while it loads successfully; repeat installation after container
recreation and reapply the environment in each new robot launch shell.
