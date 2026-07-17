---
license: Apache-2.0
name: doca-gpunetio-ib-write-bw
description: >
  Use this skill when the user is building, running, or interpreting
  the doca/tools/gpunetio_ib_write_bw client+server benchmark — a CUDA
  kernel on the server posts RDMA WRITE work requests through the
  doca-gpunetio device-side surface to measure sustained GPU-driven
  WRITE bandwidth on a GPU+IB-device pair. Trigger even when the user
  does not explicitly mention "doca-gpunetio-ib-write-bw" or
  "GPUNetIO" — typical implicit phrasings include "measure WRITE BW
  when the GPU posts the WRs", "BW swings between runs on the same
  flags", "is the NIC saturated or am I CPU-bound on the CUDA
  kernel", "meson compile fails for the GPUNetIO bw tool",
  "nvidia_peermem isn't picking up my GPU buffer", or "GPU-initiated
  WRITE throughput vs CPU-initiated perftest". Refuse and route
  elsewhere for general doca-gpunetio library work, DOCA install, the
  GPU-initiated WRITE latency analog, the CPU-initiated upstream
  perftest, or application-level end-to-end throughput — those belong
  to other skills.
metadata:
  kind: tool
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) with a BlueField DPU or ConnectX NIC. Reads
  `pkg-config doca-gpunetio doca-rdma doca-common` and inspects
  /opt/mellanox/doca/tools/gpunetio_ib_write_bw/. Requires an NVIDIA
  GPU with CUDA toolkit + nvcc installed, the `nvidia_peermem` kernel
  module loaded, and an InfiniBand-capable RNIC paired with the GPU on
  a common PCIe/NVLink fabric.
---

# DOCA GPUNetIO ib_write_bw

**Where to start:** This is a tool skill for the GPUNetIO-
flavored `ib_write_bw` benchmark shipped under
`doca/tools/gpunetio_ib_write_bw/` (a client + server pair,
built from source against the installed DOCA via `meson`).
It measures sustained RDMA WRITE bandwidth when the WRs are
posted **from a CUDA kernel through the doca-gpunetio
device-side surface**, with the GPU on the data path. Open
[`TASKS.md`](TASKS.md) and start at
[`## configure`](TASKS.md#configure) for the GPU-NIC
pairing precondition and the build pattern; jump to
[`## run`](TASKS.md#run) for the smoke-before-bulk flow.
Open [`CAPABILITIES.md`](CAPABILITIES.md) when the question
is *what this tool actually measures*, *how the result
decomposes (GPU occupancy vs NIC issue rate vs link
saturation)*, or *how the result reads against the GPI
sister tool and the upstream CPU-initiated `perftest`
`ib_write_bw`*. If DOCA is not installed yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first; if the
user is still deciding between the GPI and GPUNetIO
programming surfaces, the picture in
[`../../libs/doca-gpunetio/CAPABILITIES.md#capabilities-and-modes`](../../libs/doca-gpunetio/CAPABILITIES.md#capabilities-and-modes)
and
[`../../libs/doca-gpi/CAPABILITIES.md#capabilities-and-modes`](../../libs/doca-gpi/CAPABILITIES.md#capabilities-and-modes)
is the first stop.

## Example questions this skill answers well

The CLASSES of `doca-gpunetio-ib-write-bw` questions this
skill is built to answer, each with one worked example. The
class is the load-bearing piece; the worked example is one
instance.

- **"What sustained RDMA-WRITE bandwidth can the GPUNetIO
  path deliver on this GPU-NIC pair?"** — worked example:
  *"measure sustained WRITE BW between two hosts with an
  H100 + ConnectX-7 on each side"*. Answered by the
  GPU-NIC pairing precondition in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the bring-up flow in
  [`TASKS.md ## configure`](TASKS.md#configure) +
  [`TASKS.md ## run`](TASKS.md#run). The same shape
  answers *"measure GPUNetIO-driven WRITE BW between a
  host GPU and a BlueField DPU"*.
- **"Where is the bottleneck — GPU compute occupancy, NIC
  issue rate, or link saturation?"** — worked example:
  *"I see 120 Gbit/s on a 200 Gbit/s link; is the NIC
  saturated, am I CPU-bound on the client, or is the CUDA
  kernel not driving enough WRs in flight?"*. Answered by
  the throughput-decomposition rules in
  [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability)
  + the eval-loop overlay in
  [`TASKS.md ## test`](TASKS.md#test).
- **"How does the result differ from the classic CPU-
  initiated `perftest` `ib_write_bw`?"** — worked example:
  *"my team has a CPU-initiated WRITE BW number on this
  same NIC; should I expect the GPUNetIO number to match
  or be different?"*. Answered by the *"GPU-initiated
  path adds (or removes) overhead vs the CPU-initiated
  path"* rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
- **"Is the doca-gpunetio path the right surface for my
  sustained-throughput workload class?"** — worked example:
  *"my application streams sensor data from GPU memory at
  line rate to a remote consumer"*. Answered by the
  *"when GPUNetIO is the right surface vs GPI vs CPU-
  initiated"* rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the use-side decision in [`TASKS.md ## use`](TASKS.md#use).
- **"My BW number swings between runs. What do I check
  before quoting it?"** — worked example: *"three runs at
  the same flags gave 145, 187, and 160 Gbit/s; is the
  benchmark noisy or is my platform inconsistent?"*.
  Answered by the measurement-soundness rules in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  layer 5 + the steady-state guidance in
  [`TASKS.md ## test`](TASKS.md#test).
- **"What version of DOCA + CUDA Toolkit do I need for this
  binary to build and run?"** — worked example: *"my
  install has DOCA at one semver and CUDA at another; will
  the ToT-shipped `gpunetio_ib_write_bw` even link?"*.
  Answered by the version overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)
  which cross-links the canonical detection chain in
  [`doca-version`](../../doca-version/SKILL.md).

## Audience

This skill serves **external developers and performance
engineers who need a reproducible measurement of sustained
RDMA WRITE bandwidth when the WRs are posted from a CUDA
kernel through doca-gpunetio**, on the user's actual install
and GPU-NIC pair. Concretely:

- A developer comparing the GPUNetIO path against the GPI
  path or the host-initiated `perftest`-style path before
  committing an application design to one of them.
- A platform operator validating a tuning change (NUMA
  pinning, GPU PCIe placement, IB device choice, GID
  index, NIC firmware burn) by re-running this benchmark
  against the new state.
- An SRE / performance engineer producing a *"this is the
  GPUNetIO-driven WRITE BW on this GPU-NIC pair today"*
  artifact downstream consumers can cite.
- An AI agent answering *"is the doca-gpunetio path a win
  for my sustained-throughput workload class"* honestly —
  with a measured number, the build + invocation that
  produced it, and the GPU + NIC + DOCA version that
  scopes it — rather than guessing from datasheet
  headlines.

It is **not** for users debugging the `doca-gpunetio`
library itself (route to
[`../../libs/doca-gpunetio/SKILL.md`](../../libs/doca-gpunetio/SKILL.md)),
and **not** a substitute for the `perftest` upstream
`ib_write_bw` (which measures CPU-initiated WRITE BW).

## Language scope

The `doca-gpunetio-ib-write-bw` tool is shipped as **C plus
a CUDA `.cu` translation unit** under
`doca/tools/gpunetio_ib_write_bw/`, split into a `client/`
subtree and a `server/` subtree. The verified surface (per
`client/{main.c,common.h,common.c,kernel.cu,perftest.c}` and
`server/{main.c,common.h,common.c,perftest.c}`): host-side
build via `meson` against the installed DOCA `pkg-config`
modules (`doca-gpunetio`, `doca-rdma`, `doca-common`); the
device-side build via `nvcc` against the DOCA GPU NetIO
device-side header set; the OOB descriptor exchange via a
TCP socket between client and server. There is no Python /
Rust / Go binding — the tool is a pair of CLI binaries.
The skill's job is to keep the operator-side workflow
language-neutral; the device-side CUDA surface is not
wrappable in another language.

## When to load this skill

Load this skill when the user is — or the agent needs to —
build and run the `gpunetio_ib_write_bw` client + server on
real hosts with DOCA installed plus a CUDA Toolkit matched
to the DOCA install, and a GPU + IB device pair on the
host's PCIe topology. Concretely:

- Measuring sustained kernel-initiated RDMA WRITE
  bandwidth between two hosts (or a host and a BlueField
  DPU) with the GPUNetIO surface.
- Deciding whether the GPUNetIO path is the right runtime
  surface for a class of workload vs the GPI programming
  surface (the [`doca-gpi`](../../libs/doca-gpi/SKILL.md)
  library — `doca/tools/` ships no GPI benchmark binary) or
  the classic CPU-initiated `perftest` path.
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
it for *application-level* end-to-end throughput either —
this benchmark measures the WR-submission path through
GPUNetIO, not the user's full pipeline.

## What this skill provides

This is a **thin loader**. Substantive material lives in
two companion files:

- `CAPABILITIES.md` — what the tool measures (the
  sustained-WRITE-BW primitive driven by a server-side
  CUDA kernel through doca-gpunetio), the
  runtime-surface selection rule (GPUNetIO vs GPI vs
  CPU-initiated), the GPU-NIC pairing precondition, the
  throughput-decomposition guide (GPU compute occupancy
  vs NIC issue rate vs link saturation), the version
  overlay (DOCA `.pc` PLUS CUDA Toolkit), the layered
  error taxonomy (config-syntax / build-time / GPU-NIC-
  pairing / GPUNetIO-lifecycle / RDMA-connection /
  measurement-soundness / version / cross-cutting), the
  observability surface (stdout report, DOCA log levels,
  OOB-socket exchange), and the safety overlay (the
  *"GPU-side handle is a credential"* rule from
  doca-gpunetio; the cross-cutting hardware-safety
  meta-policy).
- `TASKS.md` — step-by-step workflows for the in-scope
  task verbs: `install` (preconditions — DOCA install,
  CUDA Toolkit, GPU + NIC pair, OOB connectivity),
  `configure` (build-tree under
  `doca/tools/gpunetio_ib_write_bw/` and the `meson`
  build wrapping the shipped DOCA), `build` (the
  `meson setup` + `meson compile` pattern from the
  public DOCA build documentation), `modify` (do not
  patch the shipped tool source; modify the invocation
  and the surrounding environment instead), `run` (smoke-
  before-bulk; client + server bring-up order; reading
  the per-iteration report), `test` (the eval loop —
  steady-state, NUMA placement, NIC saturation cross-
  check), `debug` (walk the error taxonomy layer by
  layer), `use` (how a BW result feeds a class-of-
  workload decision), plus a `Deferred task verbs`
  block routing out-of-scope questions.

The skill assumes a host where DOCA is already installed,
a CUDA Toolkit matched to the install is present, and the
operator has whatever privileges the public install profile
expects for binding a `doca_dev`, a `doca_gpu`, and an OOB
TCP socket.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or scripts
bundle. To keep the boundary clean, it deliberately does
not contain — and pull requests should not add:

- **Specific flag strings or expected throughput numbers**
  beyond what the tool's shipped `--help` and `main.c` ARGP
  registration establish. The flag surface is small
  (device name, GPU PCIe address, GID index, server IP on
  the client side); the agent re-reads the binary's
  `--help` on the installed version before quoting flag
  strings. Throughput numbers are device-, firmware-,
  version-, and topology-specific.
- **Pre-written DOCA GPUNetIO or CUDA kernel source code**
  that would compete with the shipped tool tree. The
  shipped `client/{main.c,kernel.cu,perftest.c,common.{c,h}}`
  and `server/{main.c,perftest.c,common.{c,h}}` files are
  the verified worked example; the agent's job is to
  route the user there and prescribe minimum-diff
  modification per the universal modify-a-sample workflow
  in
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
- **Wrappers, parsers, or scripts** in any language that
  consume the tool's stdout. The output format is small
  and documented in
  [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability);
  if the user wants to script against it, the right
  answer is *"read the live source, write the parser
  against your installed binary"*.
- **A `samples/`, `bindings/`, or `reference/` subtree.**
  This is a thin loader for a shipped tool tree;
  substantive material lives in the source tree and in
  the GPUNetIO library docs.

## Loading order

1. Read this `SKILL.md` first to confirm the user's
   question is in scope (the user actually wants to
   measure sustained kernel-initiated WRITE BW through
   GPUNetIO, not learn GPUNetIO as a library or do a
   CPU-initiated measurement).
2. **For what the tool measures, the surface-selection
   rule against the GPI sister tool and the CPU-initiated
   `perftest`, the throughput-decomposition guide, the
   version overlay, the error taxonomy, the observability
   surface, and the safety overlay, see
   [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — `install`, `configure`,
   `build`, `modify`, `run`, `test`, `debug`, `use` — see
   [TASKS.md](TASKS.md).**

## Related skills

- [`../../libs/doca-gpunetio/SKILL.md`](../../libs/doca-gpunetio/SKILL.md) —
  the library this tool wraps. The per-GPU `doca_gpu`
  context, the GPU-visible `doca_gpu_eth_*` and RDMA-side
  handles, the CUDA-side persistent-kernel pattern, the
  dual capability-discovery rule (DOCA cap-query AND
  `cudaGetDeviceProperties`), and the env preconditions
  (`nvidia_peermem` loaded, CUDA buffers registered with
  DOCA) live there.
- [`../../libs/doca-rdma/SKILL.md`](../../libs/doca-rdma/SKILL.md) —
  the underlying RDMA library. The RDMA queue this tool
  binds is created and connected via `doca-rdma`; the
  queue lifecycle, transport type (RC vs UC vs UD),
  permission matrix, and connection method are owned
  there.
- [`../../libs/doca-verbs/SKILL.md`](../../libs/doca-verbs/SKILL.md) —
  the raw-verbs escape hatch beneath `doca-rdma` /
  `doca-gpunetio`. This tool stays on the higher-level
  surfaces; `doca-verbs` is the right place only if the
  user needs a specific WR flag / QP attribute the
  GPUNetIO + RDMA surfaces do not expose.
- [`../doca-gpunetio-ib-write-lat/SKILL.md`](../doca-gpunetio-ib-write-lat/SKILL.md) —
  the latency analog of this tool. Same physical
  operation; same runtime framework; different metric
  class (BW vs latency). The two together carry the
  full GPUNetIO-side throughput / latency picture.
- [`doca-gpi`](../../libs/doca-gpi/SKILL.md) — the GPI
  programming surface (CUDA-kernel-initiated RDMA), the
  alternative runtime framework for the same physical
  operation. `doca/tools/` ships no GPI `ib_write_lat` /
  `ib_write_bw` benchmark binary, so the GPI comparison is
  against the library surface, not a sibling tool. The
  selection rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  is the decision aid.
- [`doca-version`](../../doca-version/SKILL.md) — the
  canonical version-detection chain, four-way match rule,
  NGC container semantics, and headers-win-over-docs
  rule. The `## Version compatibility` section in this
  skill is a thin overlay; the body lives there.
- [`doca-setup`](../../doca-setup/SKILL.md) — env
  preparation, install verification, GPU + CUDA Toolkit
  pairing, `nvidia_peermem` load, hugepages, NUMA, and
  the *I have no install yet* path with the public NGC
  DOCA container.
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  routing to the public DOCA documentation set (DOCA GPU
  NetIO, DOCA RDMA pages on `docs.nvidia.com`) and the
  `docs.nvidia.com/cuda/` pointer for the CUDA Toolkit.
- [`doca-debug`](../../doca-debug/SKILL.md) — the
  cross-cutting debug ladder. The tool surfaces its own
  error taxonomy; when the cause is below DOCA, the
  taxonomy hands off here.
- [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) —
  the bundle-wide hardware-safety meta-policy. The
  `## Safety policy` overlay cross-links it.
