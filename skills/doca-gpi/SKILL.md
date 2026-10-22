---
license: Apache-2.0
name: doca-gpi
description: >
  Use this skill for DOCA GPI work — configuring a GPU Packet
  Initiator so device code issues RDMA operations straight out of
  GPU-resident buffers via doca_gpi objects: domain creation and
  channel limits, attribute objects, the doca_gpu_gpi_channel*
  handoff consumed inside device code, and DOCA_ERROR_* triage on
  doca_gpi_* entry points such as doca_gpi_gpu_channel_get. Also
  load it for implied asks like "how many channels fit in a GPI
  domain" or "device code issuing one-sided RDMA". Refuse and route
  to the proper owner for Send/Receive semantics (doca-gpunetio),
  host-managed QP lifecycle (doca-rdma), Arm-side origination
  (doca-rdmi), or generic CUDA tutelage.
metadata:
  author: NVIDIA <doca@nvidia.com>
  kind: library
  doca_version: "3.5"
  tags:
    - doca
    - gpi
    - gpu
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) with a BlueField DPU or ConnectX NIC
  attached, plus an NVIDIA GPU with CUDA Toolkit installed
  (GPUDirect-style PCIe path between GPU and NIC). Reads the local
  install via `pkg-config doca-gpi` (co-requires doca-gpunetio,
  doca-dpa, doca-verbs) and inspects
  /opt/mellanox/doca/{lib,include,samples,applications}. The validation
  steps require a live DOCA install.
---

# DOCA GPI

## Examples

- **"CUDA kernel needs to post RDMA from GPU memory."** — this is
  in scope. Use GPI domain / channel, not the GPUNetIO send/recv
  surface.
- **"How do I hand a GPU handle to my CUDA kernel?"** — use
  `doca_gpu_gpi_channel*`. Do not route this to DPA-side RDMI.

## Instructions

Follow Loading order below. Open the tasks companion for configure /
modify / debug. Open the capabilities companion for GPI vs GPUNetIO
vs RDMI. Do not open every companion file.

**Where to start:** This skill assumes DOCA is already installed
and the user is doing **hands-on GPI work** on a host that has both
a BlueField / ConnectX device and an NVIDIA GPU reachable over
PCIe. Open the tasks companion if the user wants to *do*
something (install / configure / build / modify / run / test /
debug / use); open the capabilities companion when the
question is *what can GPI express on this version* — the domain +
channel object model, the GPU-side handle handoff, the relationship
to doca-gpunetio and doca-verbs, the attribute objects, and the
safety overlay. If the user has not installed DOCA yet, route to
`doca-setup/SKILL.md` first.

## Example questions this skill answers well

The CLASSES of GPI questions this skill is built to answer, each
with one worked example. The agent should treat the *class* as the
load-bearing piece — the worked example is a single instance.

- **"Should I use `doca-gpi` or `doca-gpunetio` for this case?"** —
  worked example: *"my CUDA kernel needs to post RDMA writes
  directly to a remote DPU's memory — do I want the higher-level
  Send/Receive surface or the lower-level channel/queue surface?"*.
  Answered by the *channel-level vs Send/Receive-level* selection
  rule in
  ## Capabilities and modes
  surface-selection table.
- **"How do I bring up a GPI channel and connect it to a remote
  peer?"** — worked example: *"create the GPI, set domain + channel
  attribute sizing, create the channel, exchange endpoint
  connection info with the remote, connect the endpoint"*. Answered
  by the channel-object lifecycle in
  ## Capabilities and modes
  + the configure walk in
  ## configure.
- **"What is the GPU-side handle and how do I hand it to my CUDA
  kernel?"** — worked example: *"`doca_gpi_gpu_channel_get`
  returns a `doca_gpu_gpi_channel*` — how do I get that into my
  CUDA kernel's argument list?"*. Answered by the GPU-handoff
  pattern in
  ## Capabilities and modes
  + the run-side wiring in ## run,
  cross-linked into
  `doca-gpunetio/SKILL.md` for the CUDA-side
  programming surface itself.
- **"What does my CUDA + GPU + DOCA version stack need to look
  like?"** — worked example: *"I have BlueField-3 + A100; which
  CUDA Toolkit and which DOCA version do I need?"*. Answered by
  the version-overlay in
  ## Version compatibility
  + the install-checks in ## install.
- **"How do I size the channels and work queues I want?"** —
  worked example: *"I want 64 channels in a domain, each with a
  1024-entry send queue; which setters express that?"*. Answered
  by the attribute-object sizing rule
  (`doca_gpi_domain_attr_set_num_channels`,
  `doca_gpi_channel_attr_set_sq_wqe_num`) in
  ## Capabilities and modes
  + the sizing step in
  ## configure.
- **"What does this `DOCA_ERROR_*` from a `doca_gpi_*` call
  mean?"** — worked example: *"`DOCA_ERROR_*` from
  `doca_gpi_gpu_channel_get`"*. Answered by the GPI overlay
  on the cross-library taxonomy in
  ## Error taxonomy
  + the layered ladder in ## debug
  that escalates to `doca-debug/SKILL.md`.

## Audience

This skill serves **external developers building GPU-resident DOCA
applications that need to drive RDMA queues directly from CUDA
kernels** — i.e., users whose accelerator-side code wants to post
RDMA work from GPU memory without round-tripping through the host
CPU. The canonical caller is a CUDA kernel that runs on an NVIDIA
GPU on the same host as a BlueField / ConnectX device, has
GPUDirect-style access to the DPU's RDMA queues through the
DOCA GPU-NetIO stack, and uses the GPI channel + queue handle to
drive RDMA initiation. This skill is *not* for NVIDIA developers
contributing to DOCA GPI itself, and it is not the right surface
for the higher-level Send/Receive Ethernet-shaped GPU NetIO API —
that belongs to `doca-gpunetio/SKILL.md`.

## Language scope

DOCA GPI ships as a C library with the `pkg-config` module name
`doca-gpi`. The library's **host-side** surface (`doca_gpi_*`)
is C; the **GPU-side** surface — the `doca_gpu_gpi_channel*`
handle and the device-side calls a CUDA kernel uses against that
handle — is compiled with `nvcc` against the DOCA GPU NetIO
device-side header set documented in
`doca-gpunetio/SKILL.md`. Other-language
consumers (Rust, Go, Python, …) consume the host-side `*.so`
through FFI; the skill's contribution in that case is to keep the
channel / queue lifecycle, the GPU-handle handoff, the version
discipline, and the safety overlay language-neutral, and to route
the agent to the public C ABI as the authoritative surface that
any wrapper will eventually call. The GPU-side surface is *not*
wrappable in another language — it is compiled and linked into
the CUDA binary itself.

## When to load this skill

Load this skill when the user is doing **hands-on DOCA GPI work**
on a host with both a BlueField / ConnectX device and an NVIDIA
GPU. Concretely:

- Deciding between `doca-gpi` (the lower-level channel/queue
  surface) and `doca-gpunetio` (the higher-level Send/Receive
  surface) for a new GPU-initiated RDMA workload.
- Creating a `doca_gpi` on a `doca_dev`, configuring it via the
  `doca_gpi_set_*` family (domain count, GID index, port) and
  sizing domains and channels through the
  `doca_gpi_domain_attr_*` / `doca_gpi_channel_attr_*` setters
  before `doca_gpi_start()`.
- Creating a channel with `doca_gpi_channel_create` and
  retrieving its GPU-side handle with `doca_gpi_gpu_channel_get`,
  then handing the GPU-side handle to a CUDA kernel.
- Exchanging endpoint connection info with a remote peer using
  `doca_gpi_channel_ep_conn_info_create` /
  `doca_gpi_channel_ep_connect` to establish the GPU-driven
  channel end-to-end.
- Attaching memory regions to a GPI domain with
  `doca_gpi_domain_attach_local_mmap` /
  `doca_gpi_domain_attach_remote_mmap` (each backed by a
  `doca_mmap` the application created).
- Debugging a `DOCA_ERROR_*` returned by a `doca_gpi_*` call
  and deciding whether the cause is a lifecycle ordering bug, a
  GPU datapath mis-assignment, a CUDA-version mismatch, or a
  layer below DOCA.

Do **not** load this skill for general DOCA orientation, install
of DOCA itself, host-CPU-initiated RDMA, or the higher-level GPU
NetIO Send/Receive Ethernet-shaped API. For those, use
`doca-public-knowledge-map/SKILL.md`,
`doca-setup/SKILL.md`,
`doca-rdma/SKILL.md`, and
`doca-gpunetio/SKILL.md` respectively.
When one question spans the GPI channel lifecycle and CUDA-side
GPU NetIO behavior, load both skills: this skill owns GPI object
creation, connection, and teardown, while `doca-gpunetio` owns
kernel launch and device-side execution. If that boundary remains
ambiguous after reading both scopes, stop and ask which side is
failing instead of choosing one implicitly.

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive GPI-specific
material lives in two companion files:

- the capabilities companion — what GPI can express on this version: the
  `doca_gpi` / `doca_gpi_domain` / `doca_gpi_channel` object
  model, the GPU-side handle handoff, the relationship to
  `doca-gpunetio/SKILL.md` (which owns the
  GPU-side `doca_gpu_gpi_channel*` device surface) and to
  `doca-verbs/SKILL.md` and
  `doca-dpa/SKILL.md` (the transport and DPA layers
  GPI builds on), the domain and channel attribute objects, the
  maturity statement (every `doca_gpi_*` symbol is
  `DOCA_EXPERIMENTAL`), the GPI overlay on the cross-library
  `DOCA_ERROR_*` taxonomy, the observability surface (CUDA-side
  channel polling, mmap-attach exchange), and the safety policy
  that gates GPU-side RDMA initiation.
- the tasks companion — step-by-step workflows for the eight in-scope
  verbs: `install`, `configure`, `build`, `modify`, `run`,
  `test`, `debug`, `use`. Plus a `Deferred task verbs` block that
  points out-of-scope questions at the right next skill.

The skill assumes a host where DOCA is already installed at the
standard location, a CUDA Toolkit compatible with the installed
DOCA is present, and the user has the privileges their public
install profile expects. It does not cover installing DOCA — that
path goes through `doca-setup/SKILL.md`.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA GPI application source code, in any
  language.** The agent's job is to route the user to verified
  reference code (the shipped DOCA GPU-NetIO samples on the
  installed package set are the canonical worked examples for the
  GPU-side handoff) and to prescribe a minimum-diff modification
  via the universal modify-a-sample workflow in
  `doca-programming-guide/SKILL.md`.
  Because every GPI symbol is tagged `DOCA_EXPERIMENTAL`
  in the public header, the skill refuses to author GPI
  source from documentation prose.
- **Standalone build manifests** (`meson.build`, `CMakeLists.txt`,
  `Cargo.toml`, …) parked inside the skill. The agent constructs
  the build manifest *in the user's project directory* against
  the user's installed DOCA, where `pkg-config --modversion
  doca-gpi` is the source of truth.
- **CUDA kernel templates.** The CUDA-side surface is owned by
  `doca-gpunetio/SKILL.md`; GPI's GPU-side
  handle is *consumed by* the CUDA programming model documented
  there. This skill names the GPI-specific handoff (the
  `doca_gpu_gpi_channel*` type, the channel-connect call) but
  does not author CUDA kernels.
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree, even
  one labeled "reference", is misleading: users will read it as
  buildable.

## Loading order

1. Read this skill first to confirm the user's question is
   in scope.
2. **For the GPI object model, the GPU-side handoff pattern, the
   domain and channel attribute objects, the version compatibility
   rule, the error taxonomy, observability, and safety policy, see
   the capabilities companion.**
3. **For step-by-step workflows — install, configure, build,
   modify, run, test, debug, use — see the tasks companion.**

Both companion files cross-link to each other and to
`doca-public-knowledge-map/SKILL.md`
whenever the right answer is "look it up in the public docs or
the installed package layout" rather than "GPI-specific
guidance".

## Related skills

- `doca-gpunetio/SKILL.md` — the GPU NetIO
  library that exposes Send/Receive-shaped Ethernet I/O for CUDA
  kernels and **owns the GPU-side device surface GPI hands off
  to**: the `doca_gpu_gpi_channel*` type and its device-side
  `.cuh` API live in doca-gpunetio, and `doca_gpi.h` includes
  `doca_gpunetio.h`. Both can coexist in the same application.
  The selection table in
  ## Capabilities and modes
  is the load-bearing decision aid.
- `doca-verbs/SKILL.md` and
  `doca-dpa/SKILL.md` — the transport and DPA
  layers GPI builds on. `dependencies/meson.build` lists
  `doca-dpa`, `doca-gpunetio`, and `doca-verbs` (plus the
  `libmlx5` / `libibverbs` externals); `doca_gpi_get_dpa` returns
  the GPI-owned `doca_dpa*` for tuning DPA attributes. The RDMA
  transport type, GID / port selection, and permission semantics
  live at the verbs layer; GPI consumes it rather than binding a
  `doca-rdma` queue.
- `doca-rdmi/SKILL.md` — the sister DPA-side
  initiator surface. Both GPI and RDMI exist for "drive RDMA
  initiation from an accelerator without the host CPU on the
  data path"; GPI is the GPU case, RDMI is the DPA case.
- `doca-public-knowledge-map/SKILL.md` — the
  routing table for every public DOCA documentation source and
  the on-disk layout of an installed DOCA package.
- `doca-setup/SKILL.md` — env preparation,
  install verification, and the *I have no install yet* path
  with the public NGC DOCA container.
- `doca-programming-guide/SKILL.md` —
  general DOCA programming patterns shared by every library: the
  canonical `pkg-config` + meson build pattern, the universal
  modify-a-shipped-sample first-app workflow, the universal
  Core-context lifecycle, the cross-library `DOCA_ERROR_*`
  taxonomy. This skill layers GPI specifics on top.
- `doca-debug/SKILL.md` — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). GPI-specific debug overlays on top of it.
- `doca-hardware-safety/SKILL.md` —
  the bundle-wide hardware-safety meta-policy. The `## Safety
  policy` overlay in the capabilities companion cross-links it.
- `doca-version/SKILL.md` — the version
  detection / four-way match rule every per-artifact `##
  Version compatibility` anchor builds on. This skill quotes
  the GPI-specific overlay only (DOCA-side `.pc` PLUS the CUDA
  Toolkit axis).
- `doca-structured-tools-contract/SKILL.md` —
  the JSON-schema contracts for the agent-preferred structured
  helpers; the `## Command appendix` in the tasks companion defers to
  them before falling back to the manual chain.

