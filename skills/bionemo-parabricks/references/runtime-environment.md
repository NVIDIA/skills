# Parabricks runtime environment

Use this reference when assessing NVIDIA Parabricks readiness, installation gaps,
GPU memory, Docker/container access, storage, or runtime recommendations on a
local or target host.

## First Step

Confirm whether the current machine is the Parabricks execution host. If the
user will run on a remote Linux server, cloud instance, scheduler node, or
container platform, ask for facts from that target environment instead of
assuming local results apply.

When the current machine is the execution host, prefer the bundled diagnostic
script and use its report as the basis for recommendations. Run these examples
from the skill directory containing `SKILL.md`; the helper is at
`scripts/check_parabricks_runtime.sh` relative to that directory:

```bash
bash scripts/check_parabricks_runtime.sh
```

The helper requires Bash and GNU coreutils `timeout` (or `gtimeout`). If neither
timeout utility is available, it exits with status 2 before running probes.
`--timeout SECONDS` must be positive; probes that ignore termination are killed
after a one-second grace period. The legacy `check_parabricks_runtime.py` entry
point forwards to this Bash helper with the same arguments and exit status.

When input, output, or temporary directories are known, include them in the
storage check:

```bash
bash scripts/check_parabricks_runtime.sh \
  --path <input-dir> \
  --path <output-dir> \
  --path <tmp-dir>
```

Wait for the helper's report before choosing additional commands. It already
collects OS, CPU/RAM, GPU/driver, Docker version/daemon access, and any requested
storage facts. Use the discovery commands below only if the helper cannot run
or a specific fact remains unresolved. If workload paths are unknown, report
that limitation; a filesystem check does not establish where inputs or scratch
space will reside.

If Docker is missing or daemon access fails, report the observed error and stop
Docker diagnostics. Readiness does not require inspecting, connecting to,
mounting, or changing permissions on the host Docker socket. Do not dump
environment variables, including NVIDIA/CUDA-prefixed variables, or read
credential/configuration files to discover authentication or host access.
Use the helper's GPU and runtime results and request administrator-supplied
diagnostics when the target environment is inaccessible.

The default check does not launch containers. `--run-container-check` enables
the CUDA probe; `--run-parabricks-check` enables the Parabricks probe and requires
`--parabricks-version` or a full `--parabricks-image` reference. Supplying an
image selection alone does not launch anything.
Only enable probes when the user has authorized the
network/image-access side effects. These checks may pull images or require NGC
authentication. Select a CUDA probe image compatible with the target release
and driver using `--cuda-test-image`; `--parabricks-version` selects only the
Parabricks image and does not change the CUDA probe image:

```bash
bash scripts/check_parabricks_runtime.sh \
  --run-container-check \
  --cuda-test-image nvidia/cuda@sha256:<verified-compatible-digest> \
  --run-parabricks-check \
  --parabricks-image nvcr.io/nvidia/clara/clara-parabricks@sha256:<verified-digest>
```

For machine-readable output, use:

```bash
bash scripts/check_parabricks_runtime.sh --format json
```

Do not install, upgrade, or modify packages. If prerequisites are missing,
provide commands for the user or system administrator to run, and make clear
that the commands must be reviewed for their OS, package manager, security
policy, and Parabricks version.

## Discovery Commands

Use the diagnostic script first when possible. If the script cannot run, if the
target host is remote, or if extra detail is needed, use commands that fit the
target operating system and available tools.

For operating system and kernel:

```bash
uname -a
cat /etc/os-release
```

For NVIDIA GPU and driver details:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv
nvidia-smi -L
```

Read the driver-supported CUDA version from the plain `nvidia-smi` banner;
`cuda_version` is not a supported `--query-gpu` field. This is the driver's
CUDA capability, not evidence of an installed host CUDA toolkit. The helper's
JSON `cuda_version` field has this meaning and is `null` when unavailable.

For CPU and memory on Linux:

```bash
lscpu
cat /proc/meminfo | grep MemTotal
cat /proc/cpuinfo | grep processor | wc -l
```

For storage:

```bash
df -h
df -h <input-dir> <output-dir> <tmp-dir>
```

For Docker, use the CLI checks below. Run `docker info` only if `docker --version`
succeeds; a missing CLI or failed daemon-access check is a readiness blocker.

```bash
docker --version
docker info
```

When Docker is available and daemon access succeeds, use the opt-in helper
probes above. Select an image appropriate to the target release and driver.
Record its trusted registry digest for reproducibility; `--cuda-test-image`
and `--parabricks-image` require an explicit release tag or a full SHA-256 digest.
The helper rejects untagged images and `latest`; tags can still be moved, so use
a verified digest when immutability matters. The script does not log
in to a registry, install a runtime, or inspect credentials.

## Requirements For The Selected Release

Start with the exact image tag and its archived manual. The helper reports host
facts and general resource warnings; it does not certify compatibility with
every Parabricks release. Its 100 GB free-space warning is a heuristic, not a
documented WGS storage requirement. Size scratch/output space for the dataset
and planned concurrency.

For **4.2 (`4.2-1`)**, the
[archived requirements](https://archive.docs.nvidia.com/clara/parabricks/4.2.0/GettingStarted.html)
specify driver **525.60.13 or newer**, Docker **20.10 or newer**, and supported
NVIDIA GPUs with at least 16 GB each. `fq2bam` needs 24 GB by default, or 16 GB
with `--low-memory`. For eight GPUs, the documented host baseline is 392 GB
RAM and 48 CPU threads. The manual's container probe uses CUDA 12.0.0; do not
test a 4.2 host against the helper's default CUDA 12.9.1 image.

For authorized container probes on a Parabricks 4.2 host, explicitly select the
CUDA image from that manual:

```bash
bash scripts/check_parabricks_runtime.sh \
  --run-container-check \
  --cuda-test-image nvidia/cuda:12.0.0-base-ubuntu20.04 \
  --run-parabricks-check \
  --parabricks-version 4.2-1
```

If a CUDA probe fails, inspect the reported image and error before treating
`runtime-incomplete` as evidence of a Container Toolkit problem. An incompatible
probe image can fail on a host that supports the selected Parabricks release;
rerun with a compatible image before recommending runtime changes.

For any selected release, check its manual for:

- A Linux operating system that supports the NVIDIA Container Toolkit.
- Docker version 20.10 or higher.
- An NVIDIA driver compatible with that Parabricks container's CUDA version.
- NVIDIA GPU support and per-device memory for the selected tool and options.
- CPU RAM and CPU thread recommendations for multi-GPU systems.
- No unsupported GPU mode for the target Parabricks version. Verify whether
  vGPU or MIG limitations apply in the current docs before making a strong
  claim.

## Interpret Results

Use the diagnostic script output to report:

- OS/distribution and whether it looks like a supported Linux runtime.
- GPU count, model names, compute capability if available, memory per GPU, and
  whether GPUs are visible through `nvidia-smi`.
- Driver version and driver-supported CUDA version, distinguished from host
  toolkit installation and the CUDA runtime bundled inside the selected image.
- Whether Docker is installed and new enough.
- Whether Docker can access GPUs through NVIDIA Container Toolkit.
- CPU thread count and system RAM.
- Free space on input, output, and temporary filesystems when paths are known.
- Whether the selected Parabricks container can be pulled and can run `pbrun`.
- Any missing information that prevents a recommendation.

## Missing Prerequisites

When something is missing, do not run install commands. Provide the next action
for the user.

Examples:

- Missing NVIDIA driver: direct the user to install a CUDA-compatible NVIDIA
  data center or supported GPU driver for their OS, then rerun `nvidia-smi`.
- Missing Docker: provide the official Docker Engine installation page for the
  user's distro and a user-run command only after confirming the OS/package
  manager.
- Missing NVIDIA Container Toolkit: provide the official NVIDIA Container
  Toolkit installation page and tell the user to configure Docker GPU runtime.
- Docker cannot access GPUs: suggest verifying driver health, container toolkit
  installation, Docker daemon configuration, and user permissions.
- Missing Parabricks image access: ask whether the user is authenticated to
  NVIDIA NGC and whether network/proxy policy permits pulling from `nvcr.io`.

When offering commands, label them as "user-run commands" and keep them
separate from diagnostic commands the agent can safely execute.

## Recommendation Style

Use qualitative recommendations:

- `ready`: required runtime components appear available.
- `hardware-constrained`: GPU memory, CPU RAM, CPU threads, or storage may
  limit the requested workflow.
- `runtime-incomplete`: required software such as Docker, NVIDIA Container
  Toolkit, driver support, or image access is missing or unverified.
- `I/O-constrained`: storage layout or free space is likely to dominate runtime
  or failure risk.
- `not supported`: the target OS/GPU/runtime mode appears unsupported for the
  selected Parabricks version.
- `not enough information`: key facts are absent.

Prefer the script's `Assessment`, `Recommendations`, and `Open questions`
sections as the starting point. Add workflow-specific guidance only after
checking the requested Parabricks tool, data size, paths, and selected version.

Do not provide exact runtime predictions unless the user supplies comparable
benchmark data including dataset size, read length, coverage, GPU model/count,
storage type, Parabricks version, and command options.

## fq2bam-Specific Guidance

For `fq2bam`, pay special attention to:

- GPU memory per device.
- Number of GPUs requested.
- System RAM and CPU threads.
- Docker GPU runtime availability.
- FASTQ size and expected output size.
- Reference and known-sites file location.
- Temporary directory location and free space.
- Whether `--low-memory` should be considered for constrained GPU memory when
  supported by the selected Parabricks version.

If the environment looks constrained or incomplete, recommend a safer first run
or prerequisite remediation before aggressive performance tuning.

## Output Template

Structure the response as:

```text
Environment summary:
<OS/GPU/driver/Docker/container toolkit/CPU/RAM/storage facts>

Assessment:
<ready | hardware-constrained | runtime-incomplete | I/O-constrained | not supported | not enough information>

Recommendations:
<specific runtime, command, or setup changes>

User-run install/setup commands:
<only include when OS/package manager is known; otherwise link to official docs>

Open questions:
<only facts still needed for a better recommendation>
```

## Key References

- Release notes and archived manuals:
  <https://docs.nvidia.com/clara/parabricks/about-parabricks/release-notes>
- Parabricks 4.2 requirements:
  <https://archive.docs.nvidia.com/clara/parabricks/4.2.0/GettingStarted.html>
- Current installation requirements (use only for that release):
  <https://docs.nvidia.com/clara/parabricks/get-started/installation-requirements>
- NVIDIA Container Toolkit installation:
  <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>
- Docker Engine installation:
  <https://docs.docker.com/engine/install/>
- NGC CLI documentation:
  <https://docs.ngc.nvidia.com/cli/>
