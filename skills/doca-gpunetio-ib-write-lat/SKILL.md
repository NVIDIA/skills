---
license: Apache-2.0
name: doca-gpunetio-ib-write-lat
description: >
  Use this skill when the user is measuring GPU-kernel-initiated RDMA
  WRITE latency through doca-gpunetio — building and running the
  `gpunetio_ib_write_lat` client + server pair under
  `doca/tools/gpunetio_ib_write_lat/`, checking GPU-NIC pairing,
  reading the half-iter / full-iter / CUDA-side usec columns,
  characterizing median / p99 / jitter for a real-time control loop,
  picking GPUNetIO vs GPI vs CPU-initiated `perftest`, or weighing the
  latency-vs-batching trade-off. Trigger even without 'GPUNetIO' or
  'ib_write_lat': 'GPU kernel RDMA latency benchmark', 'how fast can a
  CUDA kernel post a WRITE', 'p99 RDMA latency on H100 + ConnectX',
  'kernel-launched WR tail latency', or 'compare GPU-init vs CPU-init
  perftest'. Route elsewhere for bandwidth runs
  (doca-gpunetio-ib-write-bw), the GPI surface (doca-gpi), library
  debugging (doca-gpunetio), or DOCA install.
metadata:
  kind: tool
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux
  (Ubuntu 22.04/24.04 or RHEL/SLES) with an InfiniBand-capable
  ConnectX or BlueField RNIC. Requires NVIDIA GPU with CUDA
  Toolkit and `nvidia_peermem` loaded; client and server hosts
  each need a GPU-NIC pair on a common PCIe / NVLink fabric.
  Reads pkg-config doca-gpunetio / doca-rdma / doca-common and
  inspects /opt/mellanox/doca/tools/gpunetio_ib_write_lat.
---

# DOCA GPUNetIO ib_write_lat

**Where to start:** This is a tool skill for the GPUNetIO-
flavored `ib_write_lat` benchmark shipped under
`doca/tools/gpunetio_ib_write_lat/` (a client + server pair,
built from source against the installed DOCA via `meson`).
It measures the latency of an RDMA WRITE work request when
the WR is posted **from a CUDA kernel through the
doca-gpunetio device-side surface**, in a ping-pong cadence.
Open [`TASKS.md`](TASKS.md) and start at
[`## configure`](TASKS.md#configure) for the GPU-NIC pairing
precondition and the build pattern; jump to
[`## run`](TASKS.md#run) for the single-iteration smoke
flow. Open [`CAPABILITIES.md`](CAPABILITIES.md) when the
question is *what this tool actually measures*, *how it
differs from the GPI sister tool on the same physical
operation*, or *how to interpret the half-iter / full-iter
/ CUDA-side usec output and the median / p99 / jitter
characterization*. If DOCA is not installed yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first; if the
user is still deciding between GPUNetIO and GPI as a
programming surface, the picture in
[`../../libs/doca-gpunetio/CAPABILITIES.md#capabilities-and-modes`](../../libs/doca-gpunetio/CAPABILITIES.md#capabilities-and-modes)
and
[`../../libs/doca-gpi/CAPABILITIES.md#capabilities-and-modes`](../../libs/doca-gpi/CAPABILITIES.md#capabilities-and-modes)
is the first stop.

## Example questions this skill answers well

The CLASSES of `doca-gpunetio-ib-write-lat` questions this
skill is built to answer, each with one worked example. The
class is the load-bearing piece; the worked example is one
instance.

- **"What GPU-init RDMA-WRITE latency / jitter can the
  GPUNetIO path deliver for a real-time / control-loop
  workload?"** — worked example: *"measure per-iteration
  WRITE latency between two hosts with an H100 +
  ConnectX-7 on each side, target the median and the p99
  separately"*. Answered by the GPU-NIC pairing
  precondition in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the bring-up flow in
  [`TASKS.md ## configure`](TASKS.md#configure) +
  [`TASKS.md ## run`](TASKS.md#run).
- **"This is the GPUNetIO tool — how does the latency
  number differ from the GPI programming surface?"** —
  worked example: *"the team is using GPI; should I expect
  GPUNetIO to beat / tie / lose vs GPI?"*. Answered by the
  *"same physical operation, different runtime framework"*
  rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the cross-link to the GPI library skill
  [`../../libs/doca-gpi/CAPABILITIES.md`](../../libs/doca-gpi/CAPABILITIES.md)
  (note: `doca/tools/` ships no GPI `ib_write_lat`
  benchmark binary — GPI is a programming surface, not a
  shipped benchmark tool).
- **"Median vs p99 vs jitter — which one is the actual
  answer for a real-time control loop?"** — worked
  example: *"my control loop has a deadline; the median
  is well under the budget but p99 spikes; do I quote
  the median or the p99?"*. Answered by the
  median-vs-p99-vs-jitter rule in
  [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability)
  + the eval-loop overlay in
  [`TASKS.md ## test`](TASKS.md#test).
- **"What is the latency-vs-batching trade-off specific
  to GPU-init RDMA?"** — worked example: *"my CUDA kernel
  could batch multiple WRs to amortize the GPU-side
  overhead; what does that buy me on latency vs what does
  it cost me?"*. Answered by the
  latency-vs-batching trade-off in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
- **"What version of DOCA + CUDA Toolkit do I need for
  this binary to build and run?"** — worked example: *"my
  install has DOCA at one semver and CUDA at another;
  will the ToT-shipped `gpunetio_ib_write_lat` even
  link?"*. Answered by the version overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).
- **"How do I read the half-iter / full-iter / CUDA-side
  usec columns?"** — worked example: *"the binary printed
  half-iter, full-iter, and a CUDA-side number — what is
  the right column to quote for one-way latency vs
  round-trip vs cross-check?"*. Answered by the column-
  semantics rule in
  [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability).

## Audience

This skill serves **external developers and performance
engineers who need a reproducible measurement of the
latency of an RDMA WRITE WR when the WR is posted from a
CUDA kernel through doca-gpunetio**, on the user's actual
install and GPU-NIC pair. Concretely:

- A developer designing a GPU-resident real-time control
  loop and deciding whether the GPUNetIO path's tail
  latency fits the deadline.
- A platform operator validating a tuning change (NUMA
  pinning, GPU PCIe placement, IB device choice, GID
  index, NIC firmware burn) by re-running this benchmark
  against the new state.
- An SRE / performance engineer producing a *"this is the
  GPUNetIO-driven WRITE latency on this GPU-NIC pair
  today, with median + p99 + jitter"* artifact downstream
  consumers can cite.
- An AI agent answering *"is the doca-gpunetio latency
  budget acceptable for this real-time workload class"*
  honestly — with measured numbers, the build +
  invocation that produced them, and the GPU + NIC +
  DOCA version that scopes them — rather than guessing.

It is **not** for users debugging the `doca-gpunetio`
library itself (route to
[`../../libs/doca-gpunetio/SKILL.md`](../../libs/doca-gpunetio/SKILL.md)),
and **not** a substitute for the `perftest` upstream
`ib_write_lat` (which measures CPU-initiated WRITE
latency).

## Language scope

The `doca-gpunetio-ib-write-lat` tool is shipped as **C
plus CUDA `.cu` translation units** under
`doca/tools/gpunetio_ib_write_lat/`, split into a
`client/` subtree, a `server/` subtree, and a `common/`
subtree shared between them (per the verified file layout:
`client/{main.c,perftest.{c,h},meson.build}`,
`server/{main.c,perftest.{c,h},meson.build}`,
`common/{common.c,common.h,kernel.cu}`). The host-side
build is `meson` against the installed DOCA `pkg-config`
modules (`doca-gpunetio`, `doca-rdma`, `doca-common`,
plus the CUDA Toolkit dependency); the device-side build
is `nvcc` against the DOCA GPU NetIO device-side header
set. There is no Python / Rust / Go binding — the tool is
a pair of CLI binaries.

## When to load this skill

Load this skill when the user is — or the agent needs to
— build and run the `gpunetio_ib_write_lat` client +
server on real hosts with DOCA installed plus a CUDA
Toolkit matched to the DOCA install, and a GPU + IB device
pair on each host's PCIe topology. Concretely:

- Measuring kernel-initiated RDMA WRITE latency between
  two hosts (or a host and a BlueField DPU) with the
  GPUNetIO surface.
- Characterizing tail latency (p99 / p99.9) and jitter
  for a real-time / control-loop workload class.
- Deciding whether the GPUNetIO path is the right runtime
  surface for a class of workload vs the GPI programming
  surface (the [`doca-gpi`](../../libs/doca-gpi/SKILL.md)
  library — `doca/tools/` ships no GPI benchmark binary)
  or the classic CPU-initiated `perftest` path.
- Capturing a documented baseline (build + invocation +
  DOCA version + GPU + NIC + as-deployed environment +
  numbers) for later regression hunts.
- Diagnosing a build / link / run failure that surfaces
  the GPUNetIO + RDMA bring-up sequence under this tool's
  shipped scaffolding.

Do **not** load this skill for general DOCA orientation,
library API work, or installation. For those, use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
[`../../libs/doca-gpunetio/SKILL.md`](../../libs/doca-gpunetio/SKILL.md),
or [`doca-setup`](../../doca-setup/SKILL.md). Do not load
it for *application-level* real-time deadline analysis —
this benchmark measures the WR latency through GPUNetIO,
not the user's full pipeline.

## What this skill provides

This is a **thin loader**. Substantive material lives in
two companion files:

- `CAPABILITIES.md` — what the tool measures (the
  ping-pong WRITE latency primitive driven by both sides'
  CUDA kernels through doca-gpunetio), the
  runtime-surface selection rule (GPUNetIO vs GPI vs
  CPU-initiated), the GPU-NIC pairing precondition, the
  latency-vs-batching trade-off intrinsic to GPU-init
  RDMA, the median / p99 / jitter reporting taxonomy,
  the version overlay (DOCA `.pc` PLUS CUDA Toolkit),
  the layered error taxonomy, the observability surface
  (stdout report including the timeout knob the
  `gpunetio_rdma_write_lat_*` kernel functions surface
  per the verified `common.h`), and the safety overlay.
- `TASKS.md` — step-by-step workflows for the in-scope
  task verbs: `install`, `configure`, `build`, `modify`,
  `run` (smoke-before-bulk; single-iteration verification;
  reading the report columns), `test` (the eval loop —
  median / p99 / jitter / steady-state), `debug` (walk
  the error taxonomy layer by layer), `use` (how a
  latency result feeds a real-time class-of-workload
  decision), plus a `Deferred task verbs` block.

The skill assumes a host where DOCA is already installed,
a CUDA Toolkit matched to the install is present, and the
operator has whatever privileges the public install
profile expects for binding a `doca_dev`, a `doca_gpu`,
and an OOB TCP socket.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or
scripts bundle. It deliberately does not contain — and
pull requests should not add:

- **Specific flag strings or expected latency numbers**
  beyond what the tool's shipped `--help` and `main.c`
  ARGP registration establish. The flag surface is small
  (device name, GPU PCIe address, GID index, server IP
  on the client side); the agent re-reads the binary's
  `--help` on the installed version.
- **Pre-written DOCA GPUNetIO or CUDA kernel source
  code** that would compete with the shipped tool tree.
  The shipped `client/`, `server/`, and `common/`
  subtrees are the verified worked example.
- **Wrappers, parsers, or scripts** in any language that
  consume the tool's stdout. The output format is small
  and documented in
  [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability).
- **A `samples/`, `bindings/`, or `reference/` subtree.**
  This is a thin loader for a shipped tool tree.

## Loading order

1. Read this `SKILL.md` first to confirm the user's
   question is in scope (the user actually wants to
   measure kernel-initiated WRITE latency through
   GPUNetIO, not the GPI variant, not the CPU-initiated
   variant, and not a library API question).
2. **For what the tool measures, the surface-selection
   rule against the GPI sister tool and the CPU-initiated
   `perftest`, the latency-vs-batching trade-off, the
   median / p99 / jitter reporting taxonomy, the version
   overlay, the error taxonomy, the observability
   surface, and the safety overlay, see
   [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — `install`,
   `configure`, `build`, `modify`, `run`, `test`,
   `debug`, `use` — see [TASKS.md](TASKS.md).**

## Related skills

- [`../../libs/doca-gpunetio/SKILL.md`](../../libs/doca-gpunetio/SKILL.md) —
  the library this tool wraps. The per-GPU `doca_gpu`
  context, the GPU-visible RDMA handles, the CUDA-side
  persistent-kernel pattern, the dual capability-
  discovery rule (DOCA cap-query AND
  `cudaGetDeviceProperties`), and the env preconditions
  (`nvidia_peermem` loaded, CUDA buffers registered
  with DOCA) live there.
- [`../../libs/doca-rdma/SKILL.md`](../../libs/doca-rdma/SKILL.md) —
  the underlying RDMA library. The RDMA queue this tool
  binds is created and connected via `doca-rdma`; the
  queue lifecycle, the transport type (RC vs UC vs UD),
  the permission matrix, and the connection method are
  owned there.
- [`../../libs/doca-verbs/SKILL.md`](../../libs/doca-verbs/SKILL.md) —
  the raw-verbs escape hatch beneath `doca-rdma` /
  `doca-gpunetio`. This tool stays on the higher-level
  surfaces.
- [`../doca-gpunetio-ib-write-bw/SKILL.md`](../doca-gpunetio-ib-write-bw/SKILL.md) —
  bandwidth analog of this tool on the same runtime
  framework. Same physical operation; different metric
  class (latency vs BW). The two together carry the full
  GPUNetIO-side latency / throughput picture.
- [`doca-gpi`](../../libs/doca-gpi/SKILL.md) — the GPI
  programming surface (CUDA-kernel-initiated RDMA). The
  alternative runtime framework for the same physical
  operation; `doca/tools/` ships no GPI `ib_write_lat`
  benchmark binary, so the GPI comparison is against the
  library surface, not a sibling tool. The selection rule
  in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  is the decision aid; the agent's job is to teach when
  to pick which.
- [`doca-version`](../../doca-version/SKILL.md) — the
  canonical version-detection chain, four-way match
  rule. The `## Version compatibility` section here is a
  thin overlay.
- [`doca-setup`](../../doca-setup/SKILL.md) — env
  preparation, install verification, GPU + CUDA Toolkit
  pairing, `nvidia_peermem` load, hugepages, NUMA, and
  the NGC DOCA container path.
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  routing to the public DOCA documentation set and the
  CUDA Toolkit pointer.
- [`doca-debug`](../../doca-debug/SKILL.md) — the
  cross-cutting debug ladder.
- [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) —
  the bundle-wide hardware-safety meta-policy.
