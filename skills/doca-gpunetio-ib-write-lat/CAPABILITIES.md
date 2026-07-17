# DOCA GPUNetIO ib_write_lat — Capabilities

**Where to start:** This file is loaded by [`SKILL.md`](SKILL.md).
It documents *what `gpunetio_ib_write_lat` actually measures*,
*how it differs from the GPI sister tool and the CPU-initiated
upstream `perftest` `ib_write_lat`*, *which DOCA / CUDA
versions it needs*, *the latency-vs-batching trade-off
specific to GPU-init RDMA*, *the layered error and
observability surfaces*, and *the safety overlay*. The pattern
overview names the recurring `gpunetio_ib_write_lat`-class
questions; pick the pattern first, then drill into the H2
that owns the substance. For the *how*, jump to
[TASKS.md](TASKS.md).

## Pattern overview

Every `gpunetio_ib_write_lat`-class question this skill
teaches resolves into one of six patterns.

| `gpunetio_ib_write_lat` pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Pick the runtime surface | Decide *before* building whether the workload's WR-init path should be GPUNetIO (this tool), GPI (the [`doca-gpi`](../../libs/doca-gpi/SKILL.md) library programming surface — `doca/tools/` ships no GPI `ib_write_lat` benchmark binary), or classic CPU-initiated `perftest` `ib_write_lat`. The three measure the same physical operation but answer different runtime questions. | [`## Capabilities and modes`](#capabilities-and-modes) surface-selection table |
| 2. Confirm the GPU-NIC pairing | The GPUNetIO path requires the GPU and the IB device to be reachable through the same PCIe / NVLink fabric. A wrong pairing produces a number, but not the property the operator was asking about. | [`## Capabilities and modes`](#capabilities-and-modes) GPU-NIC pairing rule + [TASKS.md ## configure](TASKS.md#configure) |
| 3. Build against the install | The tool ships under `doca/tools/gpunetio_ib_write_lat/` as client/ + server/ + common/ subtrees with `meson.build` files that wrap `doca-gpunetio`, `doca-rdma`, `doca-common` and the CUDA Toolkit. | [`## Version compatibility`](#version-compatibility) + [TASKS.md ## build](TASKS.md#build) |
| 4. Characterize the latency distribution | A reported latency is meaningful only when the operator names which statistic it is — median, p99, p99.9, or jitter. The right statistic depends on the workload class (control loop, request-response, batch). | [`## Capabilities and modes`](#capabilities-and-modes) median-vs-p99-vs-jitter rule + [`## Observability`](#observability) |
| 5. Diagnose a tool failure | Walk the layered error taxonomy in [`## Error taxonomy`](#error-taxonomy). | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |
| 6. Trade latency vs batching | A CUDA kernel can issue multiple WRs per launch to amortize GPU-side overhead; that lowers the per-WR latency floor but increases batch tail latency. The trade-off is intrinsic to GPU-init RDMA, not a measurement artifact. | [`## Capabilities and modes`](#capabilities-and-modes) latency-vs-batching trade-off + [TASKS.md ## test](TASKS.md#test) |

Two cross-cutting rules that apply to *every* pattern above:

- **The result is GPU-NIC-pair-specific, version-specific, and
  topology-specific.** A latency number captured on one host
  is not transferable to another without re-capturing the
  (GPU model, NIC model, PCIe topology, DOCA version, CUDA
  Toolkit version, firmware level, IB transport, RDMA queue
  size, GID index) tuple.
- **A single number is never the answer for a real-time
  workload.** *"The latency is X usec"* is the cross-tool
  failure mode; *"median X usec, p99 Y usec, p99.9 Z usec,
  jitter J usec, all over N iterations on the given tuple"*
  is the defensible quote.

## Capabilities and modes

`gpunetio_ib_write_lat` is shipped as a **client + server +
common** triple of source trees under
`doca/tools/gpunetio_ib_write_lat/` (the verified layout
is `client/`, `server/`, and `common/`). The pair is built
from source against the installed DOCA via `meson`.

### What the benchmark actually measures

Both the client side and the server side launch CUDA kernels
that drive RDMA WRITEs through the doca-gpunetio device-side
surface in a ping-pong cadence: the client posts a WRITE,
the server's kernel polls for completion and posts the
reciprocal WRITE, the client's kernel polls for completion;
that round trip is one iteration. The OOB TCP socket
exchanges connection details before the kernels start.

Per the verified `common/common.h`, the kernel-side
functions
(`gpunetio_rdma_write_lat_server`,
`gpunetio_rdma_write_lat_client`) accept a configurable
iteration count, message size, and a per-iteration timeout
in nanoseconds. The tool reports its result format per
`common/common.h` (`REPORT_FMT_LAT`, `RESULT_FMT_LAT`):
`#bytes`, `#iterations`, `t_half_iter[usec]`,
`t_full_iter[usec]`, `t_cuda[usec]`.

### Surface selection: GPUNetIO vs GPI vs CPU-initiated

The same RDMA WRITE operation can be measured from three
runtime surfaces. The agent must surface this choice:

| Surface | Tool | When this surface is the right answer |
| --- | --- | --- |
| GPUNetIO (this skill) | `doca-gpunetio-ib-write-lat` | The CUDA kernel drives RDMA WRITE through the higher-level `doca-gpunetio` framework. Right when the application sits on doca-gpunetio and wants the latency it will see in practice. |
| GPI | [`doca-gpi`](../../libs/doca-gpi/SKILL.md) (library programming surface; no shipped GPI `ib_write_lat` benchmark binary) | The CUDA kernel drives the RDMA queue **directly** through the `doca-gpi` channel + queue handle. Right when the application is committed to GPI as its programming surface. |
| CPU-initiated `perftest` | Upstream `perftest` `ib_write_lat` (out of scope here) | Right when the comparison the operator needs is *"how much overhead does the GPU-initiated path add (or remove) versus the classic CPU-initiated path?"*. |

**Decision rule for the agent.** Surface the three options
to the user and ask which programming surface their
application will actually run on. *"Pick GPUNetIO because
the user said GPU"* and *"pick GPI because the user said
low latency"* are both bait — the decision is about which
programming surface the application will use, not about
the metric.

### The GPU-NIC pairing precondition

Same as the GPUNetIO BW sister tool — see
[`../doca-gpunetio-ib-write-bw/CAPABILITIES.md#capabilities-and-modes`](../doca-gpunetio-ib-write-bw/CAPABILITIES.md#capabilities-and-modes)
for the verification steps; the precondition is identical
(GPU + NIC reachable through a common PCIe complex or an
NVLink path the platform documents). Plus the
`nvidia_peermem` env precondition from
[`../../libs/doca-gpunetio/CAPABILITIES.md#safety-policy`](../../libs/doca-gpunetio/CAPABILITIES.md#safety-policy).

A wrong pairing manifests in this tool as additional
latency on the WR-submission path *and* as a divergence
between the host-side and CUDA-side reported usec — the
PCIe-crossover overhead shows up on the host-side number
without affecting the CUDA-side number in the same way.
The agent surfaces this signal honestly.

### Median vs p99 vs jitter — picking the right statistic

A latency benchmark reports a distribution; the user has
to pick which statistic answers their question:

| Statistic | Reads as | When this is the right answer |
| --- | --- | --- |
| Median (or mean) | Typical-case latency for a steady workload | The right number for request-response patterns where occasional tail events are acceptable. |
| p99 / p99.9 | Tail latency the user will see on the 1% / 0.1% slowest iterations | The right number for real-time control loops with a deadline; quoting the median for a deadline-bound workload is the cross-statistic failure mode. |
| Jitter (min / max spread, or stdev) | Run-to-run variability | The right number for workloads that need predictability rather than a fast median (e.g. a phase-locked control loop where consistency matters more than speed). |

The tool's reported columns (`t_half_iter`, `t_full_iter`,
`t_cuda`) carry per-iteration timing — the agent's job is
to teach the user that quoting any single number is
incomplete for a real-time workload class.

### Latency-vs-batching trade-off

A CUDA kernel can issue multiple WRs per launch to amortize
GPU-side overhead (the kernel launch latency, the CUDA
runtime cost). Batching trades:

- **Lower per-WR amortized latency** — the GPU-side
  overhead is paid once per launch, not once per WR;
  larger batches push the per-WR floor down.
- **Higher per-batch tail latency** — the batch's first
  WR pays the per-batch wait; the per-batch p99 stretches
  linearly with batch size.

This trade-off is intrinsic to GPU-init RDMA: the per-WR
latency this benchmark reports already reflects the chosen
batch size (the tool's shipped configuration is small per
`common.h`'s `NUM_ITER` / `POST_LIST` / `NUM_MSG_SIZE`
constants, but those are configurable). A real-time
control loop typically wants the smallest batch (one WR
per launch); a sustained-throughput workload wants the
batch size that minimizes amortized cost — those are two
different runs of the same tool.

## Version compatibility

For the canonical DOCA version-detection chain, the
four-way match rule, NGC container semantics, and the
headers-win-over-docs rule, see
[`doca-version`](../../doca-version/SKILL.md).

**The `doca-gpunetio-ib-write-lat`-specific overlay** is:

- **The tool builds against `doca-gpunetio`** per the
  verified per-subtree `meson.build`. The version of the
  installed `doca-gpunetio` (`pkg-config --modversion
  doca-gpunetio`) is the authoritative pin. The four-way
  match in
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)
  must hold across `doca-common`, `doca-rdma`, and
  `doca-gpunetio` on the install.
- **The CUDA Toolkit is a second axis.** The shipped
  `common/kernel.cu` is compiled with `nvcc`; the
  toolkit version must be paired with the installed DOCA
  per the DOCA release notes (looked up via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).
- **The CUDA driver is a third axis.** GPUDirect-style
  buffer registration requires both the CUDA driver and
  the `nvidia_peermem` kernel module loaded per
  [`../../libs/doca-gpunetio/CAPABILITIES.md#safety-policy`](../../libs/doca-gpunetio/CAPABILITIES.md#safety-policy).
- **Both halves of the benchmark must run against
  compatible DOCA + CUDA pairs.** Mixing a client built
  on one DOCA version with a server built on a different
  one is unsupported; the OOB-socket descriptor exchange
  uses the ABI the binary was built with.
- **No version literal from memory.** The agent quotes
  versions observed from the user's hosts
  (`pkg-config --modversion doca-gpunetio`,
  `doca_caps --version`) and from the DOCA release notes
  on `docs.nvidia.com/doca/`.

## Error taxonomy

The error surface for `gpunetio_ib_write_lat` is broader
than a pure library call because the tool builds from
source, talks to a remote peer over a TCP socket, and
drives CUDA kernels on both sides. The layers, in
escalating order:

1. **Config-syntax.** Invocation does not parse: missing
   `-d <ibdev>`, missing `--gpu <bdf>`, an `-c` server-IP
   that is not a valid IPv4, a non-numeric `--gid-index`.
   Re-read `--help` on the installed binary.
2. **Build-time.** `meson setup` or `meson compile`
   failed under
   `doca/tools/gpunetio_ib_write_lat/`. Common causes:
   `doca-gpunetio.pc` not found, `nvcc` not on `PATH`,
   GCC / GLIBC mismatch. Re-route through
   [`doca-setup`](../../doca-setup/SKILL.md) and
   [`../../libs/doca-gpunetio/TASKS.md`](../../libs/doca-gpunetio/TASKS.md).
3. **GPU-NIC pairing.** Binary built; the runtime cannot
   bind the GPU to the IB device. Re-walk the pairing
   precondition.
4. **GPUNetIO-lifecycle.** The doca-gpu context or the
   GPU-visible RDMA handle fails to come up. Most
   common: `nvidia_peermem` not loaded; CUDA buffer
   registration with DOCA happened *after*
   `doca_ctx_start()` instead of before; the per-CUDA-
   device `doca_gpu` not initialized on the right CUDA
   device. Route to
   [`../../libs/doca-gpunetio/CAPABILITIES.md#error-taxonomy`](../../libs/doca-gpunetio/CAPABILITIES.md#error-taxonomy).
5. **RDMA-connection.** The OOB socket comes up but the
   queue connection fails: GID index mismatch, RDMA
   permissions do not include WRITE, the remote mmap
   export was rejected. Route to
   [`../../libs/doca-rdma/CAPABILITIES.md`](../../libs/doca-rdma/CAPABILITIES.md).
6. **Measurement-soundness.** The benchmark completes and
   prints numbers, but the numbers are unsound. Common
   sub-layers: (a) the GPU-NIC pairing is wrong (PCIe
   crossover inflates the host-side number), (b) the
   system was not at idle, (c) the warm-up was too short,
   (d) the user quoted the median when the workload
   class wanted p99, (e) the user quoted a single number
   without naming median / p99 / jitter, (f) the
   per-iteration timeout the kernel-side function takes
   was set too short and iterations were dropped. Re-walk
   [`## Observability`](#observability).
7. **Version.** Cross-cutting partial-install /
   mixed-version per
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
8. **Cross-cutting.** Cause is below DOCA — mlx5 driver,
   CUDA driver, `nvidia_peermem`, firmware, NUMA,
   hugepages, kernel boot parameters (IOMMU mode).
   Route to
   [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
   and
   [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug).

For the cross-library `DOCA_ERROR_*` taxonomy, see
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).

## Observability

The tool's observability surface is the printed report on
stdout, plus the DOCA log surface and the OOB-socket
exchange. Specifically:

- **Stdout result line.** Per the verified
  `common/common.h` (`REPORT_FMT_LAT`, `RESULT_FMT_LAT`)
  the benchmark prints `#bytes`, `#iterations`,
  `t_half_iter[usec]`, `t_full_iter[usec]`, and
  `t_cuda[usec]` per (message size × iteration count)
  combination. The exact format is install-specific; the
  agent re-reads it against the user's binary.
  - `t_half_iter` = one-way (one-direction) WRITE latency.
  - `t_full_iter` = round-trip latency for a
    request-response pattern.
  - `t_cuda` = CUDA-side measured time; cross-check
    against the host-side number. A large divergence is
    itself a diagnostic signal, not a number to quote.
- **Distribution, not just summary.** A latency
  benchmark's value is in the distribution. The agent's
  rule: capture all iterations the run reported (the
  tool's stdout per iteration) and compute or report
  median, p99, p99.9, and jitter — not a single number
  paraphrased from the summary line.
- **DOCA log levels.** `DOCA_LOG_LEVEL` and
  `--sdk-log-level` apply per the cross-cutting rule in
  [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
  The tool's `main.c` sets the SDK logger to `WARNING`
  by default; raise it when hunting bring-up issues.
- **OOB-socket exchange.** Pre-handshake hangs surface
  on the TCP socket. The verified `common.h` declares
  `MAX_IP_LEN`-bounded IP strings for the connection
  helpers; checking the socket is up and the firewall is
  not blocking is the first observable signal.
- **Per-iteration timeout.** The kernel-side functions
  per `common.h` accept a `timeout_ns` argument; iterations
  that exceed it are not measured. If the operator sees
  fewer reported iterations than configured, the timeout
  is the suspect — not "the network is slow".
- **`nvidia-smi dmon` cross-check.** When the
  measurement looks unsound and the binding hypothesis is
  *"something is eating SM time mid-iteration"*,
  `nvidia-smi dmon` against the GPU is the cross-check.

For env-side observability (PCIe scans, link
introspection, `mlxconfig`) see
[`doca-setup CAPABILITIES.md ## Observability`](../../doca-setup/CAPABILITIES.md#observability).
For program-side observability see
[`doca-programming-guide CAPABILITIES.md ## Observability`](../../doca-programming-guide/CAPABILITIES.md#observability).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

`gpunetio_ib_write_lat` is a measurement tool. It
allocates GPU-resident memory, registers it with DOCA via
the GPUDirect-style binding path, exchanges descriptors
with a remote peer over a TCP socket, and drives RDMA
WRITE WRs from CUDA kernels on both sides. The artifact-
specific overlay:

- **GPU-side handles and mmap exports are credentials.**
  Same rule as the GPUNetIO library and the BW sister
  tool: handles valid only for the lifetime of the
  started context, transported over a trusted OOB
  segment, never re-used across runs.
- **Smoke-before-bulk; never trust the first sweep.** A
  swept run on the wrong GPU-NIC pair burns the
  operator's time on unusable distributions.
- **Quote the (DOCA version + CUDA Toolkit + GPU + NIC +
  topology + firmware + median + p99 + jitter) tuple.** A
  latency number quoted without the tuple and without the
  distribution is unreplicable and unfalsifiable.
- **Do not invent flags.** The flag surface is small (per
  the shipped `main.c`: `-c`, `-d`, `--gpu`,
  `--gid-index`); the installed binary's `--help` is the
  authoritative source.
- **Real-time / control-loop workloads need application-
  level testing.** A passing benchmark on this tool is the
  *floor* of what GPUNetIO will deliver on the WR-init
  path; the application's full pipeline can add latency
  (compute on the GPU between WRs, host-side handoffs,
  scheduling jitter) that this benchmark does not see.
  The agent surfaces this constraint to the user when
  the result is being used to decide a real-time
  deadline.
- **Hardware-safety meta-policy applies to host-side
  changes.** Any host-side change the benchmark
  surfaces — kernel command-line `iommu=` change for
  GPUDirect, hugepage reservation, BlueField BFB
  reflash, NIC firmware burn — runs through the
  cross-cutting meta-policy in
  [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md).

## Public-source pointer

The canonical public sources are the **DOCA GPU NetIO**
programming guide page and the **DOCA RDMA** page on
`docs.nvidia.com/doca/sdk/`, plus the shipped source tree
at `doca/tools/gpunetio_ib_write_lat/` on the user's
install (or in the public DOCA SDK download). Routing
lives in
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
Do not invent flags, GPUNetIO symbols, RDMA queue
attributes, or expected latency literals beyond what
those sources document.
