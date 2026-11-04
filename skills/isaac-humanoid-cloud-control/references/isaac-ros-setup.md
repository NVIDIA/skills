# Isaac ROS CLI installation and Docker setup

Use this recovery path when `isaac-ros` is missing, its environment has not been
initialized, or the G1 Docker image layers are missing. A request to set up or
run this simulated demo includes installing its missing CLI prerequisite and
initializing a previously unconfigured Docker environment. Explain the changes
and proceed within that authorization; do not ask again when installation was
already approved. Honor inspection-only or no-host-changes requests. This
reference does not authorize changing an existing venv/baremetal environment,
replacing unrelated configuration, or stopping shared containers.

The commands below target the workflow's Ubuntu 24.04 x86_64 host and Isaac ROS
release-4.6. Inspect `/etc/os-release` and `uname -m` first. On a different
platform, report the mismatch rather than applying this repository stanza.
Run host setup on the host, not inside an existing development container.

## Install the missing CLI

1. Check `command -v isaac-ros`. If present, keep it and proceed to mode/config
   validation; do not reinstall or upgrade a working CLI as routine recovery.
   If the Debian package is installed but its executable is absent from PATH,
   inspect `dpkg -L isaac-ros-cli` and correct the shell environment first.
2. Inspect existing NVIDIA Isaac ROS APT sources and their keyring. Reuse a
   matching release-4.6 source. Do not overwrite or mix another configured
   release; report that conflict before changing an existing installation.
3. For a fresh host without that source, configure the public release-4.6
   repository. Run each command separately and check its result. Reuse existing
   valid keyring/source files rather than overwriting them with these commands:

   ```bash
   sudo apt-get update
   sudo apt-get install -y curl gnupg software-properties-common
   sudo add-apt-repository -y universe
   APT_SETUP_DIR=$(python3 "$SKILL_DIR/scripts/prepare_cli_repository.py")
   ```

   Stop if staging fails. The helper downloads only the fixed official HTTPS
   endpoint, rejects redirects, verifies the pinned SHA-256 of the public key
   bundle, and dearmors it without root in a newly created mode-0700 directory.
   Key rotation requires review and an updated pin; never bypass a mismatch.
   Inspect `repository.list` in the printed directory: it must contain only the
   release-4.6 noble source with its dedicated `signed-by` keyring. These are
   public package-verification keys, not user credentials. After confirming
   that the destination files do not already exist, install each staged file:

   ```bash
   sudo install -o root -g root -m 0644 "$APT_SETUP_DIR/repository.gpg" /usr/share/keyrings/nvidia-isaac-ros.gpg
   sudo install -o root -g root -m 0644 "$APT_SETUP_DIR/repository.list" /etc/apt/sources.list.d/nvidia-isaac-ros.list
   sudo apt-get update
   ```

Remove only the private staging directory after installation. Keep an existing
matching configuration; never overwrite a conflicting key or source.

4. Install and verify the CLI:

   ```bash
   apt-cache policy isaac-ros-cli
   sudo apt-get install -y isaac-ros-cli
   command -v isaac-ros
   isaac-ros --help
   ```

   Require an install candidate from the intended release-4.6 source before
   installing. Missing sudo access, network failures, an unavailable package,
   or dependency conflicts are actionable failures: report the failing command
   and its output. Do not claim success or continue on a failed installation.

## Initialize Docker mode

1. Inspect Docker Engine, Compose, NVIDIA Container Toolkit, and current user
   access. If prerequisites are missing, complete the official Docker Engine
   and NVIDIA Container Toolkit installation/configuration linked below as part
   of the authorized bootstrap. Inspect running containers before any daemon
   restart; do not interrupt unrelated workloads without authorization.
   If adding the operator to the `docker` group is necessary, refresh group
   membership in a new shell before retrying. Do not run activation as root to
   work around a user-permission failure.
2. Verify Docker and GPU access:

   ```bash
   docker info
   docker compose version
   docker run --rm --gpus all ubuntu:24.04 nvidia-smi
   ```

3. Inspect `isaac-ros status --output json` when supported. For older CLI
   versions, inspect `/etc/isaac-ros-cli/environment.conf`. If already configured
   for Docker, retain that configuration. If uninitialized, run:

   ```bash
   sudo isaac-ros init docker
   ```

   Verify Docker mode after initialization. A command failure is not proof of
   an uninitialized environment: inspect the error/configuration first. If the
   existing mode is venv or baremetal, report it and obtain explicit direction
   before changing modes; never activate either for this simulated workflow.

## Configure the G1 image and workspace

Reuse an explicitly selected or discovered `ISAAC_ROS_WS`. On a fresh host with
no workspace, use the conventional default:

```bash
export ISAAC_ROS_WS="$HOME/workspaces/isaac_ros-dev"
mkdir -p "$ISAAC_ROS_WS/src" "$ISAAC_ROS_WS/.isaac-ros-cli"
```

Preserve the export in subsequent host shells. Inspect the CLI's default,
system, user, and workspace configuration layers. Merge the missing image keys
into `$ISAAC_ROS_WS/.isaac-ros-cli/config.yaml` while preserving all unrelated
settings and existing image keys:

```yaml
docker:
  image:
    additional_image_keys:
      - realsense
      - gr00t_workflow
```

This is the content for a new override file, not a command to replace an
existing file. Lists append across CLI configuration layers: do not add a key
again if an earlier layer already supplies it. Do not copy the entire default
configuration or modify CLI-installed files. Retain these image layers for the
G1 package dependencies while keeping camera localization disabled in the
simulation launch.

Read `isaac-ros-activate` at `skills.isaac-ros-activate.skill_md` in the
resolved dependency manifest; no separately installed skill is required. Run
its mode gate through the bundled adapter:

```bash
python3 "$SKILL_DIR/references/isaac-ros-activate/scripts/run.py" --expected docker
```

Then follow that runtime skill for the long-lived host PTY. For
new/changed layers, build and activate with:

```bash
isaac-ros activate --build-local
```

Keep this shell for later ROS commands. For an already-built matching image,
use `isaac-ros activate`. Do not silently reuse a running container built without
the requested layers; arrange an authorized recreation if required. Verify
`mode=docker` and `activation=active` using `isaac-ros status --output json`, or
use the owning activation skill's environment-file fallback on older versions.

Continue with the G1 binary package and source-only Deploy extension setup in
the release runbook. Then rerun the host `scripts/preflight.py` with the resolved
paths and map. Initial missing-prerequisite findings are recoverable; start the
demo only after the repaired host passes. Keep genuine package/CUDA conflicts
visible rather than patching the CLI or replacing image keys to bypass them.

## Sources

Verified against the public release-4.6 documentation on 2026-10-01:

- [Getting Started: repository, CLI, and Docker initialization](https://nvidia-isaac-ros.github.io/v/release-4.6/getting_started/index.html)
- [Configuration layers and Docker image builds](https://nvidia-isaac-ros.github.io/v/release-4.6/concepts/dev_env/index.html)
- [G1 cloud-control Docker image keys](https://nvidia-isaac-ros.github.io/v/release-4.6/repositories_and_packages/isaac_ros_physical_ai/isaac_ros_unitree_g1_cloud_control_bringup/index.html)
- [Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [NVIDIA Container Toolkit installation and Docker configuration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
