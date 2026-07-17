---
license: Apache-2.0
name: doca-gpunetio
description: >
  Use this skill when the user is doing hands-on DOCA GPUNetIO
  programming — wiring a CUDA kernel on an NVIDIA GPU to a doca-eth
  queue via doca_gpu_eth_rxq / doca_gpu_eth_txq, standing up the
  per-CUDA-device doca_gpu context, designing the persistent CUDA
  kernel that drains the GPU-visible queue, running the dual
  capability check (DOCA cap-query plus cudaGetDeviceProperties),
  registering cudaMalloc pools via doca_buf_arr_create_*, or
  debugging DOCA_ERROR_* returns from the GPUNetIO API. Trigger
  even when the user does not explicitly mention "DOCA GPUNetIO"
  or "persistent kernel" — typical implicit phrasings include
  "CUDA kernel reading packets directly from the NIC",
  "GPU-initiated networking on BlueField", "DOCA_ERROR_DRIVER on
  doca_gpu_create", "nvidia_peermem not loaded",
  "kernel-per-packet is too slow", or "which GPU supports GPU-side
  packet I/O". Refuse and route elsewhere for general CUDA
  programming, DOCA Ethernet queue bring-up, DOCA DPA, or
  DOCA install — those belong to other skills.
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK at /opt/mellanox/doca on Linux (Ubuntu 22.04/24.04 or
  RHEL/SLES) with a BlueField DPU or ConnectX NIC. Reads the local install
  via `pkg-config doca-gpunetio`. Requires an NVIDIA GPU with CUDA toolkit
  (matched to DOCA per the DOCA Compatibility Policy) and the
  nvidia_peermem kernel module loaded for GPUDirect RDMA; some samples
  need an InfiniBand-capable RNIC.

---

# DOCA GPUNetIO

**Where to start:** This skill assumes DOCA is already installed,
the CUDA toolkit is installed and matched to the DOCA install, and
the user is doing **hands-on GPUNetIO work** — i.e. wiring a DOCA
network queue into a CUDA kernel on an NVIDIA GPU. Open
[`TASKS.md`](TASKS.md) if the user wants to *do* something
(configure / build / modify / run / test / debug); open
[`CAPABILITIES.md`](CAPABILITIES.md) when the question is *what
can GPUNetIO express* on this version + this GPU. If the user has
not installed DOCA yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first; if the user has
not set up the underlying Ethernet RX/TX queues yet, that is a
DOCA Ethernet question (no library skill ships for it yet in this
bundle — route via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
to the public DOCA Ethernet guide).

## Example questions this skill answers well

The CLASSES of GPUNetIO questions this skill is built to answer,
each with one worked example. The agent should treat the *class*
as the load-bearing piece — the worked example is a single
instance.

- **"How do I get a CUDA kernel to receive packets directly from
  the NIC?"** — worked example: *"persistent kernel on one GPU
  reads packets from a `doca_gpu_eth_rxq` built on top of a
  representor `doca_eth_rxq` and counts them per-flow"*. Answered
  by the persistent-kernel pattern in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the GPU-side bring-up workflow in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"Can I run GPUNetIO on this GPU?"** — worked example: *"my
  host has one Ampere card and one Turing card; which one
  supports GPU-initiated networking?"*. Answered by the dual
  capability-discovery rule (DOCA cap-query AND
  `cudaGetDeviceProperties` against the CUDA device ordinal) in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the device-enumeration step in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"Why does my GPUNetIO setup fail with
  `DOCA_ERROR_NOT_SUPPORTED` even though doca-eth came up
  fine?"** — worked example: *"`nvidia_peermem` is not loaded so
  GPUDirect RDMA is unavailable"*. Answered by the env preconditions
  in [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the env checklist in
  [`TASKS.md ## configure`](TASKS.md#configure) step 1.
- **"How do I move data between CUDA-allocated buffers and a DOCA
  queue?"** — worked example: *"use `cudaMalloc` for the receive
  buffer pool and register it with DOCA via `doca_buf_arr_create_*`
  before starting the context"*. Answered by the CUDA-allocator
  + DOCA-registration overlay in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the buffer-prep step in
  [`TASKS.md ## configure`](TASKS.md#configure) step 4.
- **"Is the GPUNetIO API I'm reading about on my installed DOCA +
  CUDA combination?"** — worked example: *"is the persistent-kernel
  helper available with the CUDA toolkit version I have?"*.
  Answered by the version-compatibility overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)
  which cross-links the canonical detection chain in
  [`doca-version`](../../doca-version/SKILL.md) and adds the
  GPUNetIO-specific *DOCA must match CUDA* overlay.
- **"What does this `DOCA_ERROR_*` from a GPUNetIO call mean and
  which layer caused it?"** — worked example: *"`DOCA_ERROR_DRIVER`
  on `doca_gpu_*_create` — is it DOCA, CUDA, or the underlying
  doca-eth queue?"*. Answered by the GPUNetIO overlay on the
  cross-library taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug) that escalates to
  [`doca-debug`](../../doca-debug/SKILL.md).

## Audience

This skill serves **external developers building applications
that consume the DOCA GPUNetIO library** — i.e., users whose code
calls `doca_gpu_*` from host C/C++ to stand up the per-GPU
context and the GPU-visible queue handles, and whose CUDA kernel
(`.cu` translation unit) uses those handles from device code to
submit / receive packets. The canonical target shape is the GPU
Packet Processing reference application: a CUDA persistent
kernel on an NVIDIA GPU that polls a GPU-visible RX queue and
processes packets in-place on the GPU. It is *not* for NVIDIA
developers contributing to DOCA GPUNetIO itself.

**Language scope.** DOCA GPUNetIO ships as a C / CUDA library
with `pkg-config` module name `doca-gpunetio`. The host-side API
is C; the device-side API is CUDA C++ used inside a `.cu`
kernel. The shipped samples and the GPU Packet Processing
reference application are written in C + CUDA C++ (NVIDIA's
choice). Other-language consumers are limited in practice — the
device-side API has no FFI escape hatch because the kernel must
be a CUDA translation unit — but a Rust / Go / Python host-side
wrapper that drives the host-side `doca_gpu_*` setup and
launches a CUDA kernel built separately is still useful, and the
skill keeps the lifecycle, capability-discovery, env-precondition,
and error-taxonomy guidance language-neutral.

## When to load this skill

Load this skill when the user is doing hands-on DOCA GPUNetIO
work, in any host language plus CUDA. Concretely:

- Initializing a `doca_gpu` against a specific CUDA device
  ordinal on a host with one or more NVIDIA GPUs.
- Creating a GPU-visible queue handle (`doca_gpu_eth_rxq`,
  `doca_gpu_eth_txq`) on top of an existing `doca_eth_rxq` /
  `doca_eth_txq` from DOCA Ethernet, and passing the handle into
  a CUDA kernel for device-side use.
- Writing or modifying the persistent CUDA kernel that drains
  the GPU-visible RX queue in a long-running loop (the canonical
  GPU Packet Processing shape).
- Allocating GPU buffers via `cudaMalloc` and registering them
  with DOCA via the `doca_buf_arr_create_*` family before
  `doca_ctx_start()`.
- Checking which GPUNetIO features are supported on the active
  `doca_devinfo` (DOCA cap-query family) AND on the candidate
  CUDA device (`cudaGetDeviceProperties` and CUDA-driver-version
  checks).
- Debugging a `DOCA_ERROR_*` returned from a GPUNetIO call — in
  particular disambiguating *DOCA capability missing* from *CUDA
  device too old* from *`nvidia_peermem` not loaded* from *CUDA
  driver + DOCA version skew*.
- Designing host-side bindings for non-C languages that drive a
  CUDA kernel they built separately — the env-precondition and
  capability-discovery rules in this skill still apply.

Do **not** load this skill for general DOCA orientation, install
of DOCA or the CUDA toolkit, the underlying DOCA Ethernet queue
setup, or non-GPUNetIO library questions. For those, route
through [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
to the matching upstream guide.

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive
GPUNetIO-specific material lives in two companion files:

- `CAPABILITIES.md` — what GPUNetIO can express on this version
  + this GPU: the `doca_gpu` per-device context, the GPU-visible
  RX / TX queue handles layered on doca-eth, the persistent
  CUDA-kernel pattern as the default usage shape, the
  capability-query surface (the doca-eth
  `doca_eth_rxq_cap_is_type_supported` / `doca_eth_rxq_cap_get_*`
  family in `doca_eth_rxq.h`, plus the matching
  `doca_eth_txq_cap_*` family, on the DOCA side, plus
  `cudaGetDeviceProperties` on the CUDA side), the GPUNetIO error taxonomy mapped onto the cross-library
  `DOCA_ERROR_*` set, the observability surface (CUDA-side
  counters + DOCA-side per-task completion), and the safety
  policy that gates env preconditions (CUDA + DOCA version
  match, `nvidia_peermem`, CUDA buffer registration).
- `TASKS.md` — step-by-step workflows for the six in-scope
  GPUNetIO verbs: `configure`, `build`, `modify`, `run`, `test`,
  `debug`. Plus a `## rollback` overlay (GPUNetIO-specific
  five-step teardown that signals the persistent kernel to
  drain, unregisters GPU buffers in reverse-register order, and
  leaves the parent doca-eth queue intact) and the 5-phase
  universal debug-loop instantiation appended to `## debug`.
  Plus a `Deferred task verbs` block that points out-of-scope
  questions at the right next skill.

The skill assumes a host where DOCA is already installed at the
standard location, an NVIDIA GPU is physically present, the CUDA
toolkit is installed and its version is matched to the DOCA
install per the DOCA Compatibility Policy, and the underlying
DOCA Ethernet RX/TX queues are *already* configured (this skill
sits on top of doca-eth, not below it). It does not cover
installing DOCA or the CUDA toolkit — that path goes through
[`doca-setup`](../../doca-setup/SKILL.md).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA GPUNetIO application source code or CUDA
  kernel source, in any language.** The verified GPUNetIO
  source is the shipped C + CUDA samples at
  `/opt/mellanox/doca/samples/doca_gpunetio/` and the GPU Packet
  Processing reference application. The agent's job is to route
  the user to those files and prescribe a minimum-diff
  modification on them via the universal modify-a-sample
  workflow in
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md),
  layered with the GPUNetIO-specific overrides in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **Standalone build manifests** (`meson.build`,
  `CMakeLists.txt`, …) parked inside the skill. The agent
  constructs the build manifest *in the user's project
  directory* against the user's installed DOCA + CUDA toolkit,
  where `pkg-config --modversion doca-gpunetio` and `nvcc
  --version` are the two sources of truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree, even
  one labeled "reference", is misleading: users will read it as
  buildable.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is
   in scope.
2. **For the GPUNetIO capability matrix, the `doca_gpu` per-device
   context, the persistent-kernel pattern, the dual capability
   query, the env-precondition policy, the error taxonomy, the
   observability surface, and the safety policy, see
   [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify, run,
   test, debug — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other,
[`doca-version`](../../doca-version/SKILL.md) for the canonical
DOCA version-handling rules (with the GPUNetIO overlay that DOCA
must match CUDA), and
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public DOCA
GPUNetIO guide, the DOCA Compatibility Policy, the CUDA toolkit
docs, or the on-disk install layout" rather than
"GPUNetIO-specific guidance".

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  the routing table for every public DOCA documentation source
  and the on-disk layout of an installed DOCA package. The
  GPUNetIO public guide is at
  <https://docs.nvidia.com/doca/sdk/DOCA-GPUNetIO/index.html>;
  the GPU Packet Processing reference application is reachable
  from there. The CUDA toolkit and DOCA Compatibility Policy
  links live in the same routing table.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, CUDA toolkit install / verification, and
  the *I have no install yet* path with the public NGC DOCA
  container. This skill assumes its preconditions are satisfied
  AND that CUDA is installed at a version that matches DOCA.
- [`doca-version`](../../doca-version/SKILL.md) — canonical DOCA
  version-handling rules. This skill's `## Version
  compatibility` cross-links the four-way match rule and adds
  the GPUNetIO-specific *DOCA-and-CUDA must match* overlay per
  the DOCA Compatibility Policy.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md) —
  the bundle's structured-tools precedence rule (detect / prefer
  / fall back / report). The Command appendix in
  [TASKS.md](TASKS.md) honors this contract.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns shared by every library:
  the canonical `pkg-config` + meson build pattern, the
  universal modify-a-shipped-sample first-app workflow, the
  universal lifecycle, the cross-library `DOCA_ERROR_*`
  taxonomy, and the program-side debug order. This skill
  layers GPUNetIO specifics on top.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). GPUNetIO-specific debug (CUDA + DOCA
  version skew, `nvidia_peermem` missing, persistent-kernel
  silent hangs, CUDA-allocator + DOCA-registration mismatches)
  overlays on top of that ladder.

DOCA Ethernet is GPUNetIO's mandatory companion library: GPU-visible
RX / TX queue handles are layered on top of `doca_eth_rxq` /
`doca_eth_txq` from DOCA Ethernet. No `libs/doca-eth/` skill
ships in this bundle yet; for the underlying Ethernet queue
setup, route via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
to the public *DOCA Ethernet* guide and to the shipped
`/opt/mellanox/doca/samples/doca_eth/` samples.

