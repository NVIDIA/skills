# DOCA GPUNetIO ib_write_bw — Capabilities

**Where to start:** This file is loaded by [`SKILL.md`](SKILL.md).
It documents *what `gpunetio_ib_write_bw` actually measures*,
*how it differs from the GPI sister tool and the CPU-initiated
upstream `perftest` `ib_write_bw`*, *which DOCA / CUDA versions
it needs*, *the layered error and observability surfaces*, and
*the safety overlay*. The pattern overview names the recurring
`gpunetio_ib_write_bw`-class questions; pick the pattern first,
then drill into the H2 that owns the substance. For the *how*
of executing each pattern, jump to [TASKS.md](TASKS.md).

## Pattern overview

Every `gpunetio_ib_write_bw`-class question this skill teaches
resolves into one of six patterns. The patterns are CLASSES —
they apply to every GPU-NIC pair the benchmark can target.

| `gpunetio_ib_write_bw` pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Pick the runtime surface | Decide *before* building whether the workload's WR-init path should be GPUNetIO (this tool), GPI (the [`doca-gpi`](../../libs/doca-gpi/SKILL.md) library programming surface — `doca/tools/` ships no GPI benchmark binary), or classic CPU-initiated `perftest` `ib_write_bw`. The three measure the same physical operation but answer different runtime questions. | [`## Capabilities and modes`](#capabilities-and-modes) surface-selection table |
| 2. Confirm the GPU-NIC pairing | The GPUNetIO path requires the GPU and the IB device to be reachable through the same PCIe / NVLink fabric for the WR-submission path to be efficient. A wrong pairing produces a number, but not the property the operator was asking about. | [`## Capabilities and modes`](#capabilities-and-modes) GPU-NIC pairing rule + [TASKS.md ## configure](TASKS.md#configure) |
| 3. Build against the install | The tool ships under `doca/tools/gpunetio_ib_write_bw/` as a client/ + server/ pair with a `meson.build` that wraps `doca-gpunetio`, `doca-rdma`, `doca-common` and the CUDA Toolkit. Mismatched DOCA + CUDA surfaces at build time. | [`## Version compatibility`](#version-compatibility) + [TASKS.md ## build](TASKS.md#build) |
| 4. Decompose the throughput | A reported BW is meaningful only when the operator can name which constraint binds it: GPU compute occupancy (the CUDA kernel can't issue WRs fast enough), NIC issue rate (the device hits its WR-submission ceiling), or link saturation (the physical IB link is full). Quoting a number without naming the binding constraint is the canonical apples-to-oranges failure. | [`## Capabilities and modes`](#capabilities-and-modes) throughput-decomposition table + [`## Observability`](#observability) |
| 5. Diagnose a tool failure | Walk the layered error taxonomy in [`## Error taxonomy`](#error-taxonomy) — config-syntax / build-time / GPU-NIC-pairing / GPUNetIO-lifecycle / RDMA-connection / measurement-soundness / version / cross-cutting. | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |
| 6. Interpret the reported throughput | The tool prints sustained throughput at the chosen message size; the operator must know whether to quote that as packet-rate, bit-rate, or work-request-rate, and must capture the full tuple alongside. | [`## Observability`](#observability) + [TASKS.md ## test](TASKS.md#test) |

Two cross-cutting rules that apply to *every* pattern above:

- **The result is GPU-NIC-pair-specific, version-specific, and
  topology-specific.** A number captured on one host is not
  transferable to another without re-capturing the (GPU model,
  NIC model, PCIe topology, DOCA version, CUDA Toolkit
  version, firmware level, IB transport, RDMA queue size)
  tuple. Quoting without the tuple is the cross-platform
  regression-hunt failure mode.
- **GPUNetIO is the higher-level GPU-init networking surface.**
  Most GPU-side networking applications belong on GPUNetIO,
  not on the lower-level GPI surface. The
  surface-selection table below is the decision aid; this
  tool only makes sense when the application has *already
  committed* to GPUNetIO.

## Capabilities and modes

`gpunetio_ib_write_bw` is shipped as a **client + server pair
of source trees** under `doca/tools/gpunetio_ib_write_bw/`
(`client/` and `server/` each carry their own `main.c`,
`common.c/h`, `perftest.c`, and the client carries a
`kernel.cu`). The pair is built from source against the
installed DOCA via `meson`.

### What the benchmark actually measures

The client brings up a doca-gpu context, a doca-rdma queue
backed by a GPU-resident buffer registered with DOCA, and a
GPU-visible RDMA handle through the doca-gpunetio device-
side surface (only the client registers the `--gpu` PCIe
param). The CUDA kernel on the client (`client/kernel.cu`,
`rdma_write_bw`) posts a stream of RDMA WRITE work requests
against the server's remote buffer (`server_remote_buf_arr`)
at the configured message size for the configured iteration
count. The server side hosts the remote buffer and a
doca-rdma queue but does not post WRs of its own — its
`main.c` registers only `--device` / `--gid-index` and no
GPU. The OOB TCP socket exchanges connection details
between the two halves before the benchmark begins.

The measured quantity is **sustained throughput of GPU-driven
RDMA WRITE work** between the client's GPU-resident local
memory and the server's remote memory.

### Surface selection: GPUNetIO vs GPI vs CPU-initiated

The same RDMA WRITE operation can be measured from three
different runtime surfaces. The agent must surface this choice
to the operator before quoting any number:

| Surface | Tool | When this surface is the right answer |
| --- | --- | --- |
| GPUNetIO (this skill) | `doca-gpunetio-ib-write-bw` | The CUDA kernel drives RDMA WRITE through the higher-level `doca-gpunetio` framework (the Send / Receive-style and direct-RDMA paths exposed by [`../../libs/doca-gpunetio/CAPABILITIES.md`](../../libs/doca-gpunetio/CAPABILITIES.md)). Right when the application sits on doca-gpunetio and wants the BW it will see in practice. |
| GPI | [`doca-gpi`](../../libs/doca-gpi/SKILL.md) (library programming surface; `doca/tools/` ships no GPI `ib_write_bw` or `ib_write_lat` benchmark binary, so the GPI vs GPUNetIO BW comparison is against the library surface, not a sibling tool). | Right when the user has *already committed* to GPI as their programming surface. |
| CPU-initiated `perftest` | Upstream `perftest` `ib_write_bw` (out of scope here; not in `doca/tools/`) | Right when the comparison the operator needs is *"how much overhead does the GPU-initiated path add (or remove) versus the classic CPU-initiated path?"*. |

**Decision rule for the agent.** Surface the choice; ask which
programming surface the application will run on; do not
silently default to GPUNetIO just because the user mentioned
a GPU.

### The GPU-NIC pairing precondition

The GPUNetIO WR-submission path is efficient only when the
GPU and the IB device are reachable through a common PCIe
complex or an NVLink fabric the platform exposes for that
purpose. A misplaced pair (GPU on one NUMA node, NIC on the
other, no NVLink bridge) still completes the benchmark but
the reported BW reflects PCIe-crossover overhead, not the
property the operator was trying to measure. The pre-flight
check is the GPU-NIC pairing verification in
[`TASKS.md ## configure`](TASKS.md#configure) and the
[`doca-gpunetio`](../../libs/doca-gpunetio/SKILL.md) library;
the rule is independent of the WR-init runtime surface. The
agent quotes the pairing
alongside any reported number.

In addition, GPUNetIO carries the *"`nvidia_peermem` loaded
for GPUDirect RDMA"* env precondition documented in
[`../../libs/doca-gpunetio/CAPABILITIES.md#safety-policy`](../../libs/doca-gpunetio/CAPABILITIES.md#safety-policy);
without it, the GPU-resident buffers cannot be registered
with DOCA and the benchmark cannot bring up the WR path.

### Throughput decomposition

A reported sustained BW is only meaningful with the binding
constraint named. The class shapes the agent must keep
distinct:

| Binding constraint | Symptom on the BW plot | What to check next |
| --- | --- | --- |
| Link saturation | Reported BW sits near the physical IB link's documented capacity for the chosen transport; further tuning (message size, queue depth, kernel-side concurrency) does not move it. | This is the physical ceiling; quote it as the answer. |
| NIC issue rate | Reported BW is well below link capacity but the message-size sweep flattens (the BW does not grow as message size grows); raising the queue depth or the in-flight WR count does not help. | Confirm the NIC's documented WR-submission rate for this transport; route the platform question through [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md). |
| GPU compute occupancy | Reported BW grows with CUDA kernel block / thread count up to a point, then plateaus well below link capacity; `nvidia-smi dmon` shows the SM under-utilized. | Re-walk the CUDA kernel's persistent-kernel pattern per [`../../libs/doca-gpunetio/CAPABILITIES.md#capabilities-and-modes`](../../libs/doca-gpunetio/CAPABILITIES.md#capabilities-and-modes); the kernel may not be issuing WRs at the rate the device can accept. |
| PCIe crossover (wrong pairing) | Reported BW is well below all three of the above and does not respond to message-size / queue-depth changes the way the platform documents. | Walk the GPU-NIC pairing precondition above; the answer is platform-side, not benchmark-side. |

This decomposition is the load-bearing piece of *"interpret
the throughput"*. A number with no decomposition is the
canonical apples-to-oranges quote.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the
headers-win-over-docs rule, see
[`doca-version`](../../doca-version/SKILL.md). The body lives
there; this skill does not duplicate it.

**The `doca-gpunetio-ib-write-bw`-specific overlay** is:

- **The tool builds against `doca-gpunetio`** per the shipped
  `meson.build` (per the verified source, the `client/` and
  `server/` subtrees each carry their own `meson.build` and
  the top-level `meson.build` wires them together). The
  version of the installed `doca-gpunetio`
  (`pkg-config --modversion doca-gpunetio`) is the
  authoritative pin for this tool's API surface. The
  four-way match in
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)
  must hold across `doca-common`, `doca-rdma`, and
  `doca-gpunetio` on the install before the build is
  attempted.
- **The CUDA Toolkit is a second axis.** The shipped
  `client/kernel.cu` is compiled with `nvcc`; the toolkit
  version must be the one paired with the installed DOCA
  per the DOCA release notes (looked up via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).
- **The CUDA driver is a third axis the agent surfaces but
  does not own.** GPUDirect-style buffer registration
  requires both the CUDA driver and the `nvidia_peermem`
  kernel module loaded per
  [`../../libs/doca-gpunetio/CAPABILITIES.md#safety-policy`](../../libs/doca-gpunetio/CAPABILITIES.md#safety-policy).
  An older CUDA driver than the toolkit was built against
  surfaces as a runtime `DOCA_ERROR_DRIVER` route per
  [`## Error taxonomy`](#error-taxonomy) layer 8.
- **The remote peer must run a binary built against a
  compatible DOCA + CUDA pair.** Mixing a client built on
  one DOCA version with a server built on a different one is
  unsupported; the OOB-socket descriptor exchange uses the
  ABI the binary was built with.
- **No version literal from memory.** The agent quotes the
  version observed from the user's host
  (`pkg-config --modversion doca-gpunetio`,
  `doca_caps --version`) and from the DOCA release notes
  published on `docs.nvidia.com/doca/`.

## Error taxonomy

The error surface for `gpunetio_ib_write_bw` is broader than a
pure library call because the tool builds from source against
the installed DOCA, talks to a remote peer over a TCP socket,
and drives a CUDA kernel against an RDMA queue. The error
layers the agent should distinguish, in escalating order:

1. **Config-syntax.** Invocation does not parse: missing
   `-d <ibdev>`, missing `--gpu <bdf>`, an `-c` server-IP
   that is not a valid IPv4, a non-numeric `--gid-index`.
   The tool's `main.c` carries the ARGP schema; re-read the
   binary's `--help` on the installed build.
2. **Build-time.** `meson setup` or `meson compile` failed
   under `doca/tools/gpunetio_ib_write_bw/`. Common causes:
   `doca-gpunetio.pc` not found, `nvcc` not on `PATH`,
   GCC / GLIBC mismatch. Re-route through
   [`doca-setup`](../../doca-setup/SKILL.md) and the
   GPUNetIO library skill's
   [`../../libs/doca-gpunetio/TASKS.md`](../../libs/doca-gpunetio/TASKS.md)
   verification before re-running the build.
3. **GPU-NIC pairing.** Binary built; the runtime cannot
   bind the GPU to the IB device because the platform does
   not expose a path between them. Re-walk the pairing
   precondition in
   [`## Capabilities and modes`](#capabilities-and-modes)
   before re-running.
4. **GPUNetIO-lifecycle.** The doca-gpu context or the
   GPU-visible RDMA handle fails to come up. Most common
   sub-causes: `nvidia_peermem` not loaded, CUDA buffers
   not registered with DOCA before `doca_ctx_start()`, the
   per-CUDA-device `doca_gpu` not initialized on the right
   CUDA device. Route to
   [`../../libs/doca-gpunetio/CAPABILITIES.md#error-taxonomy`](../../libs/doca-gpunetio/CAPABILITIES.md#error-taxonomy).
5. **RDMA-connection.** The OOB socket comes up but the
   queue connection fails: GID index mismatch, RDMA
   permissions do not include WRITE, the remote
   memory-mmap export was rejected. Route to
   [`../../libs/doca-rdma/CAPABILITIES.md`](../../libs/doca-rdma/CAPABILITIES.md).
6. **Measurement-soundness.** The benchmark completes and
   prints throughput, but the number is unsound. Common
   sub-layers: (a) the GPU-NIC pairing is wrong, (b) the
   system was not at idle (background traffic on the IB
   link, concurrent CUDA workloads), (c) the warm-up was
   too short, (d) the user quoted the number without
   naming the binding constraint per
   [`## Capabilities and modes`](#capabilities-and-modes)
   throughput-decomposition. Re-walk
   [`## Observability`](#observability) before quoting.
7. **Version.** Cross-cutting partial-install / mixed-
   version per
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
   Symptoms: client and server were built against different
   DOCA versions, the CUDA Toolkit version disagrees with
   the runtime CUDA driver, `pkg-config --modversion
   doca-gpunetio` disagrees with `doca_caps --version`.
8. **Cross-cutting.** Cause is below DOCA — mlx5 driver,
   CUDA driver, `nvidia_peermem` not loaded, firmware,
   NUMA, hugepages, kernel boot parameters (IOMMU mode for
   GPUDirect-style memory mapping). Route to
   [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
   and
   [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug).

For the cross-library `DOCA_ERROR_*` taxonomy itself, see
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).

## Observability

The tool's observability surface is the printed throughput
report on stdout (per the shipped `perftest.c` source — the
agent re-reads the column layout from the installed binary),
plus the DOCA log surface and the OOB-socket exchange.
Specifically:

- **Stdout result.** Per the shipped `perftest.c` the
  benchmark prints sustained throughput at the configured
  message size + iteration count. The exact column header
  is install-specific; re-read against the user's binary.
  The agent quotes the BW alongside the message size, the
  iteration count, and the binding constraint per
  [`## Capabilities and modes`](#capabilities-and-modes)
  throughput-decomposition.
- **DOCA log levels.** `DOCA_LOG_LEVEL` and
  `--sdk-log-level` apply per the cross-cutting rule in
  [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
  The tool's `main.c` sets the SDK logger to `WARNING` by
  default; raise it when hunting bring-up issues.
- **OOB-socket exchange.** The pre-handshake hangs surface
  on the TCP socket, not the RDMA path; checking the socket
  is up and the firewall is not blocking is the first
  observable signal that bring-up reached the exchange
  step.
- **Pre-run echo.** The tool logs the device name, GPU
  PCIe address, GID index, and client/server choice at
  startup via `DOCA_LOG_INFO`. A captured log self-documents
  the invocation the throughput belongs to; preserve it
  when capturing a baseline.
- **`nvidia-smi dmon` cross-check.** When the binding
  constraint hypothesis is *"GPU compute occupancy is the
  limit"*, `nvidia-smi dmon` against the chosen GPU is the
  signal that confirms or refutes the SM utilization
  reading. It is not a DOCA observability surface, but it
  is the right cross-check.
- **`ibdev2netdev` + `ibstat` for the NIC side.** When the
  binding constraint is *"NIC issue rate"* or *"link
  saturation"*, the device's own counters are the cross-
  check. Routed via
  [`doca-setup CAPABILITIES.md ## Observability`](../../doca-setup/CAPABILITIES.md#observability).

For env-side observability (PCIe scans, link introspection,
`mlxconfig`) see
[`doca-setup CAPABILITIES.md ## Observability`](../../doca-setup/CAPABILITIES.md#observability).
For program-side observability (DOCA log levels) see
[`doca-programming-guide CAPABILITIES.md ## Observability`](../../doca-programming-guide/CAPABILITIES.md#observability).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

`gpunetio_ib_write_bw` is a measurement tool. It allocates
GPU-resident memory, registers it with DOCA via the
GPUDirect-style binding path, exchanges descriptors with a
remote peer over a TCP socket, and drives sustained RDMA
WRITE traffic from a CUDA kernel. The artifact-specific
safety overlay:

- **GPU-side handles and memory-mmap exports are
  credentials.** Treat the doca-gpunetio device-side handle
  and the doca-rdma mmap-export descriptor the same way the
  GPUNetIO and RDMA libraries name: valid only for the
  lifetime of the started context, transported over a
  trusted OOB segment, and never re-used across runs. The
  shipped tool uses cleartext TCP for the OOB exchange; run
  this benchmark only on a trusted segment, never on a
  shared production fabric.
- **Smoke-before-bulk; never trust the first sweep.** A
  swept run on the wrong GPU-NIC pair, wrong GID index,
  wrong RDMA permissions, or wrong message-size range burns
  the operator's time on unusable data. The agent's rule is
  the [`TASKS.md ## run`](TASKS.md#run) smoke step before
  any sweep.
- **Quote the (DOCA version + CUDA Toolkit + GPU + NIC +
  topology + firmware + binding constraint) tuple.** A
  throughput number quoted without the tuple — and without
  naming whether GPU occupancy, NIC issue rate, or link
  saturation was the binding constraint — is unreplicable
  and unfalsifiable.
- **Do not invent flags.** The flag surface is small (per
  the shipped `main.c`: `-c`, `-d`, `--gpu`,
  `--gid-index`); the installed binary's `--help` is the
  authoritative source. Prose-derived flags are the most
  common hallucination failure for this skill.
- **Sustained traffic on a shared fabric is a deployment
  concern.** This benchmark drives line-rate WRITE traffic;
  on a shared IB fabric it will affect other tenants.
  Schedule the run inside an explicit maintenance window
  per [`doca-hardware-safety CAPABILITIES.md ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy).
- **Hardware-safety meta-policy applies to host-side
  changes.** Any host-side change the benchmark surfaces a
  need for (kernel command-line `iommu=` change for
  GPUDirect, hugepage reservation change, BlueField BFB
  reflash, NIC firmware burn) is a hardware-touching change
  that runs through the cross-cutting meta-policy in
  [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md),
  not through this tool's invocation.

## Public-source pointer

The canonical public sources for this tool's surface are the
**DOCA GPU NetIO** programming guide page and the **DOCA
RDMA** page on `docs.nvidia.com/doca/sdk/`, plus the shipped
source tree at `doca/tools/gpunetio_ib_write_bw/` on the
user's install (or in the public DOCA SDK download).
Routing to those lives in
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
Do not invent flags, GPUNetIO symbols, RDMA queue
attributes, or expected throughput literals beyond what
those sources document.
