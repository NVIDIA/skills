# Prerequisites and container setup

Use a Linux host with an NVIDIA GPU supported by Isaac Sim 6.0, its working driver, Docker,
NVIDIA Container Toolkit, and enough free disk for the simulator, ROS image and
artifacts. Check `nvidia-smi` and Docker GPU access. Stop only processes owned by
this invocation; avoid driver changes or restarting Docker with unrelated workloads.

## Checkpoint export environment

Reuse the training skill's recorded manipulation repository and its
`isaaclab_training/.venv`; no Isaac Lab source checkout is required. Pass that
package directory as `--training-root` and its plan as `--training-plan` so the
same selected uv and cache settings are used. The plan must match `--training-root`;
without a plan, select uv with `--uv /absolute/path/to/uv`. An explicit fresh-install
request still takes priority. If missing, use the companion training skill's setup.
Keep that skill installed alongside deployment: `training_support.py` imports
its canonical environment builder and installation/dependency probes.

As inspected on 2026-10-31, `release-r1` locks LEAPP to **0.6.1**
from NVIDIA's Python index (`https://pypi.nvidia.com/`), not a cloned codebase.
Keep the branch default; no checkout commit pin is required. Inspect the selected
project and lockfile before setup, and record HEAD. The export gate and dependency
audit support LEAPP 0.6.1; stop for compatibility review if the branch changes it.
Training does not require LEAPP; add the optional export dependencies only when
needed. From the package root, use the training skill's `training_cmd` helper and
saved consent:

```bash
training_cmd uv sync --locked --inexact --group sim --group leapp --no-install-package onnxruntime --dry-run
# Review: retain existing locked packages and GPU ONNX; add the missing export dependencies.
training_cmd uv sync --locked --inexact --group sim --group leapp --no-install-package onnxruntime
```

Never sync an environment used by an active trainer. Record hashes of
`pyproject.toml` and `uv.lock` before/after; do not modify either, installed metadata,
or training source. Review the dry run against the selected lockfile rather than
assuming a fixed list of newly installed packages. `--inexact` preserves video
and other training supplements.

LEAPP declares the CPU package name `onnxruntime`, but NVIDIA explicitly supports
[the GPU distribution](https://nvidia-isaac.github.io/leapp/leapp_runtime.html).
Keep only `onnxruntime-gpu==1.26.0`, which also satisfies the simulator's dependency.
The shared dependency audit records this specific provider substitution only
when LEAPP matches its 0.6.1 lock, the GPU version is 1.26.0, and LEAPP's requirement
accepts that version. This is a documented distribution-name exception: plain
`uv pip check` still reports the absent CPU distribution. Do not call the raw
metadata conflict-free, install both providers, or waive other missing packages.
A new release/provider combination requires review and numerical export validation.
Metadata acceptance alone is insufficient; fresh numerical export validation is required.

Preflight verifies the editable training package, locked Isaac Lab wheel and
simulator, all installed dependencies, LEAPP import, and ONNX providers. Every
child—including preparation, export, validation and Docker—uses the training
skill's sanitized environment. Export uses the absolute `.venv/bin/python`;
the simulator adds only its own ROS bridge path plus explicit plan overrides.
Run a fresh export with numerical validation before using the bundle in ROS.
Training smoke evidence does not substitute for export validation.
See [driver options](#driver-options) for export-only execution.

The exporter resolves registered task configuration, not saved YAML directly.
`prepare.py` validates the supported saved interface, supplies explicit inference
settings through native `agent.actor.*` and `agent.critic.*` Hydra overrides, and
records the originals. Fail unsupported architectures or interfaces rather than
guessing. The `.pt` must be a checkpoint
the user trusts, because upstream PyTorch checkpoint loading can deserialize code.

The export wrapper supplies temporary reset-index and IK-helper compatibility
without editing upstream files; see [Troubleshooting](troubleshooting.md). Require
both LEAPP numerical validation and bundle interface/hash checks.

## Isaac Sim container

Pull the official simulator image before invoking the driver:

```bash
docker pull nvcr.io/nvidia/isaac-sim:6.0.0
```

Use `--sim-image` to select a compatible prepared image or an immutable registry
digest. The driver records image inspection metadata and launches the resolved
local image ID, then checks `/isaac-sim/VERSION` for the required 6.0.0 build.
The image provides its Python runtime, renderer, physics engine and ROS Jazzy
bridge. The separate ROS container provides Lyrical; both use Fast DDS UDPv4 and
the selected ROS domain on host networking.

The default NVIDIA image needs ffmpeg for recording. Pass `--install-sim-runtime`
to install Ubuntu's ffmpeg package in the newly created simulator container.
For a prepared image already containing `/usr/bin/ffmpeg` with libx264 support,
omit that flag.
Preflight records the ffmpeg version and encodes one black frame with libx264;
recording and video verification also run inside the simulator container.
No host video encoder is needed.

After checking the image build and executables, the driver runs the image's
`/isaac-sim/isaac-sim.compatibility_check.sh --/app/quitAfter=10 --no-window`.
Require the explicit `System checking result: PASSED` marker. The checker may
remain open after printing its result, so the driver stops and restarts that same
owned container before export. No second simulator image is downloaded for this
check. Image setup and checks are part of this skill's driver.

The simulator container runs as root with the image's explicit root allowance so
it can read private session files. It receives a read-only skill mount and only
this invocation's directory as a writable host mount; the directory is mounted at
its original absolute path so saved manifest paths remain valid. User-home,
training-installation and Docker-socket mounts are not needed. Caches and user
configuration stay inside the newly created container.

The training/export environment remains separate and uses its locked Isaac Sim
6.1 Python dependencies. Selecting the Docker deployment runtime does not change
the checkpoint's training version or export contract.

## Isaac ROS container

The driver launches the ROS container directly with Docker. With the host GPU
container prerequisites above satisfied, pull the default NVIDIA NGC AMD64 image
validated for this workflow:

```bash
docker pull nvcr.io/nvidia/isaac/ros:isaac_ros_1742a70509cba44cb1943ab200bbece1-amd64
```

Use `--runtime-image` for a compatible prepared image or immutable registry
digest. Pull that image or finish preparing it before invoking the driver;
preflight requires it to exist locally before spending time on export.

The driver creates a dedicated named container with GPU access, host networking,
read-only skill scripts and a writable invocation directory. With
`--install-runtime`, it installs these packages inside the new container:

```text
ros-lyrical-isaac-ros-deploy-reference-applications
ros-lyrical-rmw-fastrtps-cpp
cuda-cupti-13-2
```

Dependent packages come from APT; no source build is needed. Omit
`--install-runtime` when the selected image already contains these packages.

## Driver options

Use `--sudo-docker` only if Docker needs existing passwordless sudo access.
Installation flags and image selection are described in the container sections
above; environment selection is covered under checkpoint export.

- `--export-only` validates and exports without creating either deployment
  container; omit both install flags. Export still launches the training
  environment's simulator to trace the policy.
- `--preflight-only` creates the simulator container and runs its encoder and
  compatibility checks without exporting or running an insertion trial.

Cleanup stops both owned containers on completion, failure or interruption,
retaining them for inspection and restoring session ownership.

### Trial and retry options

`--attempts 3` permits at most three completed trials with the same configuration
and stops after the first success. Retain and report every measured failure.
Repeats may help diagnose variability among valid synchronized trials; this
stop-on-success mode does not establish a policy success rate. For repeatability
checks after a success, use separate invocations and report all outcomes.

A synchronization timeout, mismatched command or invalid timing evidence is an
infrastructure failure, not a completed policy trial. Diagnose and correct the
cause before starting a new session; retries must not bypass synchronization.
Controller or policy tuning is a separate task.
