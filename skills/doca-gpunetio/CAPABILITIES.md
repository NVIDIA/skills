# DOCA GPUNetIO capabilities, version overlay, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your question
(per-GPU context / GPU-visible queue handles / persistent-kernel
pattern / dual capability discovery / env preconditions / errors)
and read that section end-to-end. The tables in each section are
the load-bearing content; the prose around them is interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each pattern
(the verbs `configure / build / modify / run / test / debug`),
jump to [TASKS.md](TASKS.md). For the canonical DOCA
version-handling rules that this skill layers a GPUNetIO overlay
on top of, see [`doca-version`](../../doca-version/SKILL.md).

## Pattern overview

Every GPUNetIO question this skill teaches resolves into one of
SIX patterns. The patterns are CLASSES — they apply across every
GPUNetIO release and every host + GPU + NIC combination.

| Pattern | When it applies (class shape) | Where the substance lives |
| --- | --- | --- |
| 1. Set up the underlying Ethernet queues first | doca-eth `doca_eth_rxq` / `doca_eth_txq` must be configured before any GPUNetIO call; GPUNetIO does not own the queue, it exposes it to the GPU | [`## Capabilities and modes`](#capabilities-and-modes) layering rule + [TASKS.md ## configure](TASKS.md#configure) step 2 |
| 2. Create the per-GPU `doca_gpu` context | One context per GPU (created from the GPU's PCIe bus-id string); a multi-GPU host needs one `doca_gpu` per device | [`## Capabilities and modes`](#capabilities-and-modes) per-device-context rule + [TASKS.md ## configure](TASKS.md#configure) step 3 |
| 3. Build the GPU-visible queue handles on top of doca-eth | `doca_gpu_eth_rxq` / `doca_gpu_eth_txq` are passed into a CUDA kernel for device-side packet I/O | [`## Capabilities and modes`](#capabilities-and-modes) handle table + [TASKS.md ## configure](TASKS.md#configure) step 5 |
| 4. Use the persistent-kernel pattern, not kernel-per-packet | One long-running CUDA kernel polls the GPU-visible RX queue, processes in place, optionally pushes results out via the GPU-visible TX queue | [`## Capabilities and modes`](#capabilities-and-modes) persistent-kernel section + [TASKS.md ## modify](TASKS.md#modify) |
| 5. Honor env preconditions: CUDA toolkit matched to DOCA, `nvidia_peermem` loaded, CUDA buffers registered | Mismatched CUDA + DOCA combos fail at link or runtime in confusing ways; missing `nvidia_peermem` disables GPUDirect RDMA | [`## Safety policy`](#safety-policy) env-precondition matrix + [TASKS.md ## configure](TASKS.md#configure) step 1 |
| 6. Diagnose a GPUNetIO error | Map `DOCA_ERROR_NOT_SUPPORTED` / `_DRIVER` / `_AGAIN` / `_INVALID_VALUE` / `_BAD_STATE` to a root cause without leaving the GPUNetIO layer prematurely | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **GPUNetIO sits on top of doca-eth, not below it.** The agent
  recommending that the user *"create a GPU queue first and the
  Ethernet queue follows from it"* has the layering backwards
  for every version of GPUNetIO. Configure doca-eth first, then
  layer GPUNetIO handles on top.
- **Capability is a TWO-axis question.** *"Is this feature
  supported on this host"* requires BOTH a DOCA cap-query against
  the active `doca_devinfo` AND a CUDA device-capability check
  (`cudaGetDeviceProperties`). Either axis missing the support
  fails the feature. An agent that quotes only one axis will
  miss half the *"silently fails on this GPU"* cases.

## Capabilities and modes

The two orthogonal selection axes for any GPUNetIO design are
*which CUDA device* (`doca_gpu` per device ordinal) and *which
doca-eth queue is being exposed to the GPU* (RX, TX, or both).
Choose both before writing any CUDA kernel code, then drill into
the relevant capability-query.

**The per-GPU `doca_gpu` context — one per GPU (created from the GPU's PCIe bus-id string).**

| Object | Lifetime | What it owns | Key calls |
| --- | --- | --- | --- |
| `doca_gpu` | One per GPU; created for a specific GPU on the host | The DOCA-side bookkeeping for that GPU, the registration of CUDA buffers into DOCA, the GPU-visible queue handles built on top of doca-eth queues for that GPU | `doca_gpu_create(const char *gpu_bus_id, ...)` — takes the GPU's PCIe bus-id string (e.g. from `cudaDeviceGetPCIBusId`), not an int ordinal; plus the destroy / re-create flow on rebind |

A multi-GPU host needs one `doca_gpu` per device the user wants
to drive — there is no *"global GPU context"*. The agent must
ask which CUDA device the user intends to drive before
recommending any `doca_gpu_*` call, and must surface
`cudaGetDeviceProperties(devOrdinal)` as the canonical way to
confirm the device is GPUNetIO-capable.

**The GPU-visible queue handles — layered on doca-eth.**

| Handle | Layered on | Direction | Where it lives at runtime |
| --- | --- | --- | --- |
| `doca_gpu_eth_rxq` | `doca_eth_rxq` (from doca-eth) | RX (receive) | The handle is passed into a CUDA kernel; device-side code reads packets directly from the queue from inside the kernel |
| `doca_gpu_eth_txq` | `doca_eth_txq` (from doca-eth) | TX (send) | The handle is passed into a CUDA kernel; device-side code submits packets directly to the queue from inside the kernel |

The GPU-visible handles do not own the underlying queue — they
expose it to the GPU. Tearing down the underlying doca-eth queue
without first tearing down the GPU-visible handle is a
use-after-free on the GPU side; the agent must surface this
ordering explicitly when discussing lifecycle.

**Persistent-kernel pattern — the canonical first-app shape.**
The default GPUNetIO usage shape is *one long-running CUDA
kernel that polls the GPU-visible RX queue in a loop, processes
each packet in place on the GPU, and optionally pushes results
out via the GPU-visible TX queue*. This is the pattern the GPU
Packet Processing reference application uses; the agent should
treat it as the default and treat *"launch a kernel per packet"*
as an anti-pattern that defeats the entire reason to use
GPUNetIO (the kernel-launch overhead per packet swamps the
network savings).

| Anti-pattern | Why it's wrong | Right shape |
| --- | --- | --- |
| Launch a CUDA kernel for every received packet | Kernel-launch overhead dominates; defeats GPU-initiated networking entirely | One long-running persistent kernel that drains the GPU-visible RX queue in a loop |
| Drive the queue from host code with `cudaMemcpy` between every batch | Round-trips through the host CPU; that is what GPUNetIO exists to avoid | Process packets in-place on the GPU in the persistent kernel |
| One persistent kernel across multiple GPUs | A CUDA kernel is bound to one CUDA device; multi-GPU means multiple persistent kernels, one per `doca_gpu` | One persistent kernel per `doca_gpu` per direction, sized to the GPU it targets |

**Dual-axis capability discovery — the only rule.** Before
sizing any queue, assuming a feature is available, or
recommending the persistent-kernel pattern works on the user's
device, run BOTH a DOCA cap-query AND a CUDA device-capability
check. Either axis missing the support fails the feature.

| Axis | What to call | Why the agent must ask |
| --- | --- | --- |
| DOCA side | `doca_eth_rxq_cap_is_type_supported(devinfo, ...)` against the active `doca_devinfo` (the doca-eth cap function in `doca_eth_rxq.h`, plus the `doca_eth_rxq_cap_get_*` sizing family and the matching `doca_eth_txq_cap_is_type_supported` / `doca_eth_txq_cap_get_*` in `doca_eth_txq.h`) — GPUNetIO itself exposes no `cap_is_supported` symbol; the queue capability is a doca-eth query | DOCA-side compatibility of the underlying RX/TX queue type with GPU exposure is device-conditional; do not assume the feature is on every NIC + DOCA combo |
| CUDA side | `cudaGetDeviceProperties(props, devOrdinal)` against the candidate CUDA device | GPU-initiated networking depends on per-device CUDA capabilities; older non-Ampere GPUs may not support the GPU-side primitives even if DOCA is happy |

**Configuration shape.** *Mandatory* preconditions before any
`doca_gpu_*` call: the underlying `doca_eth_rxq` / `doca_eth_txq`
must be configured and the doca-eth context must be at the right
lifecycle stage for the GPUNetIO layer to attach; the CUDA
toolkit must be installed and `cudaSetDevice(devOrdinal)` must
succeed against the target GPU; the CUDA buffers that will hold
the receive payload pool must be allocated via `cudaMalloc` (or
equivalent CUDA allocator) and registered with DOCA via the
`doca_buf_arr_create_*` family. *Optional* configurations (queue
sizing, persistent-kernel block / thread counts on the CUDA
side) are program-side tunables that ride on top of the same
cap-query rule.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match rule, NGC container semantics, and the headers-win-over-docs rule, see [`doca-version`](../../doca-version/SKILL.md). The body lives there; this skill does not duplicate it.

**The GPUNetIO-specific overlay** is:

- **DOCA must match CUDA per the DOCA Compatibility Policy.** The public DOCA Compatibility Policy at <https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html> defines which CUDA toolkit version a given DOCA release supports. Mismatched CUDA + DOCA combos fail at link time (missing CUDA-side symbols) or at runtime (`DOCA_ERROR_DRIVER` from `doca_gpu_create`) in confusing ways. The agent must surface BOTH `pkg-config --modversion doca-gpunetio` AND `nvcc --version` (and `cat /usr/local/cuda/version.txt` when present) and compare them against the DOCA Compatibility Policy before any GPUNetIO bring-up.
- **The DOCA cap-query AND `cudaGetDeviceProperties` are both runtime authorities.** Per the cross-cutting cap-query rule in [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability), the DOCA-side cap query is the runtime authority for *"is this GPUNetIO feature on this DOCA install"*. The CUDA-side device-properties query is the runtime authority for *"is this GPU capable of GPU-initiated networking at all"*. Either being false fails the feature; the agent must report both.
- **`doca-gpunetio.pc`, `doca-common.pc`, and the matching CUDA toolkit must all line up at the four-way-match check** (per [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)). A common partial-install pattern is that DOCA was upgraded but the CUDA toolkit on the host was not, or vice versa; the agent must surface that as a four-way-match failure and route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) before any GPUNetIO-layer diagnosis.

## Error taxonomy

GPUNetIO-specific overlays on the cross-library `DOCA_ERROR_*`
taxonomy. The cross-library taxonomy itself lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
the rows below are the *GPUNetIO surface* meaning that the agent
must disambiguate before falling back to the cross-library
response.

| Error | GPUNetIO context where it shows up | GPUNetIO-specific cause |
| --- | --- | --- |
| `DOCA_ERROR_NOT_SUPPORTED` | `doca_gpu_create`; `doca_gpu_eth_rxq_*` / `_txq_*` creation; the cap-query family | The GPU does not support GPUNetIO (e.g. older non-Ampere GPU per `cudaGetDeviceProperties`), or `nvidia_peermem` is not loaded so GPUDirect RDMA is unavailable, or the underlying doca-eth queue type is not exposable to the GPU on this NIC + DOCA combo. Run BOTH `doca_eth_rxq_cap_is_type_supported(devinfo, ...)` (the doca-eth query in `doca_eth_rxq.h`) AND `cudaGetDeviceProperties(devOrdinal)`; surface which axis is false. |
| `DOCA_ERROR_INVALID_VALUE` | `doca_gpu_create` with a GPU PCIe bus-id string; buffer registration via `doca_buf_arr_create_*` | The PCIe bus-id string does not match a GPU present on the host (mismatched between `nvidia-smi -L` / `cudaDeviceGetPCIBusId` and the string passed to DOCA), or the CUDA-allocated buffer has alignment / size that does not match the doca-eth queue expectations. Re-resolve the bus-id and the queue's required alignment; do not paper over with a retry. |
| `DOCA_ERROR_BAD_STATE` | Any GPUNetIO call against a doca-eth context that is at the wrong lifecycle stage; teardown ordering between the GPU-visible handle and the underlying doca-eth queue | Lifecycle violation. The most common case is calling `doca_gpu_eth_rxq_*` against a doca-eth context that has not been started, or destroying the underlying `doca_eth_rxq` before destroying the GPU-visible handle. Walk the layering rule in [`## Capabilities and modes`](#capabilities-and-modes). |
| `DOCA_ERROR_AGAIN` | TX submit from inside the CUDA kernel (device-side); RX drain from the persistent kernel | The GPU-visible queue is full from the GPU side. This is *not* a hardware error; the persistent kernel must drain completions before re-submitting (the device-side equivalent of the host-side *"would-block, retry after progress"* pattern). |
| `DOCA_ERROR_DRIVER` | `doca_gpu_create`; any GPUNetIO call when CUDA + DOCA versions are skewed | The CUDA driver layer reported failure to DOCA. Most common cause is a CUDA + DOCA version mismatch per the DOCA Compatibility Policy. Route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) layer 5 (driver) AND to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) for the version-skew side. |

The agent's rule: **never recommend a retry loop on
`DOCA_ERROR_*` without first identifying which of the rows above
is the cause**. `_AGAIN` is the only one that wants a device-side
drain-then-retry; the others want investigation (env / version /
lifecycle), not retry.

## Observability

GPUNetIO observability surface is **two-sided**: there is a
DOCA-side observability surface (per-task completions, the DOCA
logger, capability-query snapshots) AND a CUDA-side observability
surface (CUDA device counters, the persistent-kernel's own
in-kernel logging, CUDA error returns from launch / sync). The
agent must reach for both, not just one — a hung persistent
kernel that produces no DOCA log line is almost always visible
on the CUDA side via `cuda-gdb` or kernel-internal counters.

Three primary signals the agent should reach for:

1. **DOCA-side per-task completions.** Every queue submit
   produces a completion (success or failure) that the persistent
   kernel can read from the device side. Absence of a completion
   for submitted packets is *always* a queue-not-drained bug;
   the persistent kernel needs an explicit drain step in its
   loop.
2. **Capability snapshot at configure time.** The output of
   `doca_eth_rxq_cap_is_type_supported(devinfo, ...)` (the
   doca-eth query in `doca_eth_rxq.h`) AND
   `cudaGetDeviceProperties(devOrdinal)` together is the baseline
   of *"what the library + the GPU said was possible"* before
   any kernel was launched. Save both; if a runtime call later
   returns `DOCA_ERROR_NOT_SUPPORTED` the diff against this
   baseline is the bug.
3. **CUDA-side health.** `nvidia-smi -q -d UTILIZATION` while
   the persistent kernel runs shows whether the GPU is actually
   busy with work; a persistent kernel that reports 0% util is
   stuck on a queue drain. `cuda-gdb` attached to the host
   process can step into the device-side kernel when DOCA-side
   completions stop arriving without an explicit error.

For cross-cutting observability primitives (`--sdk-log-level`,
the `DOCA_LOG_LEVEL` env var, the trace build flavor) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
For the install-tree observability (logger names, package
layout, sample tree) defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

GPUNetIO's safety surface is **env-precondition-driven**. Unlike
RDMA (permission-driven) or Comch (representor-visibility-driven),
the single most common GPUNetIO first-app failure is that the
host's CUDA + DOCA combination is not what the user thinks it is
— and the agent's job is to verify that before any
`doca_gpu_*` call, not after the first `DOCA_ERROR_DRIVER`.

The **env-precondition matrix** the agent must walk for any new
GPUNetIO setup:

| Precondition | What must be true | How the agent verifies | Where to fix |
| --- | --- | --- | --- |
| CUDA toolkit installed and matched to DOCA | The CUDA toolkit on the host is installed at a version listed as compatible with the installed DOCA release per the DOCA Compatibility Policy | `nvcc --version`; `cat /usr/local/cuda/version.txt`; cross-check against `pkg-config --modversion doca-gpunetio` and the [DOCA Compatibility Policy](https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html) | [`doca-setup`](../../doca-setup/SKILL.md) for the install-side; route to [`doca-version`](../../doca-version/SKILL.md) for the four-way-match check |
| `nvidia_peermem` loaded for GPUDirect RDMA | The `nvidia_peermem` kernel module is loaded so GPUDirect RDMA between the NIC and the GPU works; without it GPU-initiated networking falls back or fails entirely | `lsmod \| grep nvidia_peermem`; missing → `sudo modprobe nvidia_peermem`; persistent via system config | [`doca-setup`](../../doca-setup/SKILL.md) for the env-side; do not modify the program |
| CUDA device enumerable | `cudaGetDeviceCount` returns at least 1 and `cudaGetDeviceProperties(devOrdinal)` succeeds for the device the user intends to drive | `nvidia-smi -L`; programmatic via the CUDA runtime; `cudaSetDevice(devOrdinal)` must succeed | [`doca-setup`](../../doca-setup/SKILL.md) for the env-side; verify the NVIDIA driver is loaded |
| doca-eth side already up | The underlying `doca_eth_rxq` / `doca_eth_txq` exists and the doca-eth context is at the right lifecycle stage to expose the queue to the GPU | The doca-eth bring-up workflow (no library skill yet ships in this bundle; route via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) to the public DOCA Ethernet guide) is the upstream verb | Configure doca-eth first; do NOT skip ahead to `doca_gpu_*` calls |
| CUDA buffers registered with DOCA | The GPU-side payload pool allocated via `cudaMalloc` is registered with DOCA via the `doca_buf_arr_create_*` family before `doca_ctx_start()` | Walk the registration sequence in [TASKS.md ## configure](TASKS.md#configure) step 4; a missing registration surfaces as `DOCA_ERROR_BAD_STATE` from the first device-side submit | Program-layer fix — the registration call goes in the host-side bring-up code, before the persistent kernel is launched |

**Do not partial-install one side.** A CUDA-only install
without DOCA, or a DOCA install with no CUDA toolkit, fails
GPUNetIO in non-obvious ways: pkg-config may succeed for DOCA
while the link line fails on CUDA symbols, or DOCA may succeed
at create-time while the first runtime call returns
`DOCA_ERROR_DRIVER` from CUDA below. The fix is to align both
sides per the DOCA Compatibility Policy, not to silence the
error.

**Lifecycle ordering is GPU-aware.** The GPU-visible handle must
be destroyed BEFORE the underlying doca-eth queue is destroyed;
the `doca_gpu` context must be destroyed BEFORE the CUDA context
is unbound from the process. Out-of-order teardown surfaces as
`DOCA_ERROR_BAD_STATE` on subsequent calls but also leaves the
GPU in an undefined state for the next process; the agent must
surface this ordering explicitly.

## Deferred topic boundaries

This skill scopes itself to the DOCA GPUNetIO library. Adjacent
topics the agent will get asked but should route elsewhere:

- **CUDA programming in general** (CUDA kernel design, CUDA
  memory model, stream / event semantics, multi-stream
  scheduling) — outside this skill. Route to the upstream CUDA
  toolkit documentation; this skill assumes the user already
  understands CUDA C++ and is asking *how to wire a DOCA queue
  into a CUDA kernel*.
- **DOCA Ethernet queue setup** (`doca_eth_rxq` / `doca_eth_txq`
  bring-up, RSS, representor selection on the NIC side) —
  GPUNetIO depends on it but does not redefine it. Route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  to the public DOCA Ethernet guide; no library skill yet ships
  for it in this bundle.
- **DOCA DPA** (DPA-side networking primitives, DPA kernel
  programming) — the *sibling* DOCA library for offloading
  device-side compute onto a different non-CPU target. The
  agent should NAME DPA when asked about *"DOCA libraries that
  integrate with non-CPU compute"* (see the worked example in
  [TASKS.md ## modify](TASKS.md#modify)) but should not invent
  DPA API surfaces here. Route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  to the public DOCA DPA guide.
- **DOCA Core context and progress engine internals** — owned
  by
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
  This skill *uses* the Core lifecycle; it does not redefine
  it.
- **Cross-cutting `DOCA_ERROR_*` taxonomy** — owned by
  [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
  This skill adds the GPUNetIO overlay, not the taxonomy itself.
- **Cross-cutting debug ladder** (install / version / build /
  link / runtime / program / driver) — owned by
  [`doca-debug ## debug`](../../doca-debug/TASKS.md#debug). This
  skill's `## debug` redirects there for layer 1-4; layers 5-7
  carry the GPUNetIO-specific overlay (including the
  CUDA-driver-layer route).
