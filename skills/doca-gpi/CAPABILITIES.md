# DOCA GPI capabilities, version compatibility, errors, observability, safety

**Where to start:** The pattern overview below names the recurring
GPI-class patterns. Pick the pattern first, then drill into the H2
that owns the substance. For the *how* of executing each pattern,
jump to [TASKS.md](TASKS.md).

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For step-by-step workflows that *use* these
capabilities (install, configure, build, modify, run, test, debug,
use) see [TASKS.md](TASKS.md). For where the underlying public
documentation and installed package paths live, defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) — do
not duplicate URLs or install paths in this file.

## Pattern overview

Every GPI-class question this skill teaches resolves into one of
FIVE patterns. The patterns are CLASSES — they apply across every
GPU-initiated RDMA use case, not just the worked example shown.

| GPI pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Pick `doca-gpi` vs `doca-gpunetio` | Decide *before* writing any code whether the CUDA kernel drives an RDMA channel directly (channel level — GPI) or sends / receives at the Ethernet-shaped Send/Receive layer (GPU NetIO) | [`## Capabilities and modes`](#capabilities-and-modes) surface-selection table |
| 2. Stand up the GPI instance | Lifecycle: `doca_gpi_create` on a `doca_dev` → configure via `doca_gpi_set_*` → `doca_gpi_start` → create domain(s) and channel(s) from their attribute objects → retrieve GPU handles → use → `doca_gpi_stop` → `doca_gpi_destroy` | [TASKS.md ## configure](TASKS.md#configure) |
| 3. Connect a channel endpoint to a remote peer | The channel owns the *GPU-side wiring*; an endpoint is connected by exchanging connection info out of band via `doca_gpi_channel_ep_conn_info_create` + `doca_gpi_channel_ep_connect`, and memory is shared via `doca_gpi_domain_attach_local_mmap` / `doca_gpi_domain_attach_remote_mmap` | [`## Capabilities and modes`](#capabilities-and-modes) endpoint-connection bullet + [TASKS.md ## configure](TASKS.md#configure) |
| 4. Hand off to the CUDA kernel | `doca_gpi_gpu_channel_get` returns a `doca_gpu_gpi_channel*` the CUDA kernel uses directly; the host releases ownership and the device-side path is then GPU-driven | [`## Capabilities and modes`](#capabilities-and-modes) GPU-handoff bullet + [TASKS.md ## run](TASKS.md#run) |
| 5. Interpret a `DOCA_ERROR_*` from a GPI call | Map the error to a layer (configuration / lifecycle / GPU datapath / CUDA-version / verbs below / driver) and route | [`## Error taxonomy`](#error-taxonomy) GPI overlay + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **Sizing is set on attribute objects, not on the `doca_gpi`
  handle, and GPI exposes no runtime capability query.** Channel
  count and endpoint / bind counts are set on a
  `doca_gpi_domain_attr` (`doca_gpi_domain_attr_set_num_channels`,
  `_set_num_ep`, `_set_num_binds`, `_set_bind_size`); per-channel
  work-queue sizing is set on a `doca_gpi_channel_attr`
  (`doca_gpi_channel_attr_set_sq_wqe_num`, `_set_srq_wqe_num`,
  `_set_gpu_wqe_num`). `doca_gpi.h` ships **no** `doca_gpi_cap_*`
  devinfo query, so the agent must not invent a runtime maximum;
  an out-of-range value surfaces as a `DOCA_ERROR_*` from the
  create / start call, and supported ranges come from the device
  and release notes, not from agent memory.
- **All `doca_gpi_set_*` attributes must be set before
  `doca_gpi_start()`.** The header states `doca_gpi_start` *"Must
  be called after setting all the GPI attributes"*, and
  `doca_gpi_get_dpa` *"can be called only if gpi not started"*.
  The agent walks the user through configuration (and, where the
  application drives the GPU datapath, the
  [`doca-gpunetio`](../doca-gpunetio/SKILL.md) setup) *before*
  `doca_gpi_start()`, not after.

## Capabilities and modes

DOCA GPI is the **GPU-Packet-Initiator** surface: a CUDA kernel
running on an NVIDIA GPU drives RDMA work directly against the
BlueField / ConnectX device's queues, with no host CPU on the
data path. The host-side library configures a `doca_gpi`
instance, one or more domains, and a set of channels; the
GPU-side handle then lets the CUDA kernel issue work.

### `doca-gpi` vs `doca-gpunetio`

Two GPU-side surfaces ship in DOCA. They are NOT interchangeable;
pick one before writing any code:

| Library | Surface shape | When to pick |
| --- | --- | --- |
| [`doca-gpunetio`](../doca-gpunetio/SKILL.md) | Higher-level Send/Receive-shaped Ethernet I/O for CUDA kernels; richer per-packet API; the canonical surface for GPU-side packet processing | The CUDA kernel sends or receives Ethernet packets through a queue and the application is happy with the Send/Receive abstraction |
| `doca-gpi` (this skill) | Lower-level channel surface; the CUDA kernel drives RDMA work *directly* via the GPU-side channel handle; channel endpoints are connected to remote peers via `doca_gpi_channel_ep_connect` | The CUDA kernel needs to initiate RDMA operations and the application wants direct control over the per-channel work submission rather than the Send/Receive abstraction |

**Decision rule for the agent.** If the user's intent is *"my
CUDA kernel needs to drive an RDMA queue directly from GPU
memory, without the Send/Receive abstraction in between"*, GPI
is the right surface. If the user's intent is *"my CUDA kernel
sends or receives Ethernet packets"*, GPU NetIO is the right
surface. Both can coexist in one application; the agent does not
force one when the other fits better.

### The GPI instance

`doca_gpi` is the top-level GPI object. Every symbol below is
`DOCA_EXPERIMENTAL`. Verified surface (`doca_gpi.h`):

| Lifecycle phase | Calls | Note |
| --- | --- | --- |
| Create | `doca_gpi_create(dev, &gpi)` | Created on a `doca_dev`; the GPI binds to that device for its lifetime |
| Configure | `doca_gpi_set_num_domains(gpi, N)`, `doca_gpi_set_enable_err_monitor(gpi, on)`, `doca_gpi_set_gid_index(gpi, gid)`, `doca_gpi_set_port_num(gpi, port)` | Must all run *before* `doca_gpi_start`; the GID index and port select the IB / RoCE path the channels will use |
| DPA access | `doca_gpi_get_dpa(gpi, &dpa)` | Returns the DPA object GPI initialized internally (for setting DPA attributes / affinity); the header notes it "can be called only if gpi not started" |
| Start | `doca_gpi_start(gpi)` | The header states it "must be called after setting all the GPI attributes" |
| Stop | `doca_gpi_stop(gpi)` | Stops the started GPI instance |
| Destroy | `doca_gpi_destroy(gpi)` | Releases all resources associated with the GPI object created by `doca_gpi_create` |

Channels and memory binds are owned by **domains**, created from
a `doca_gpi_domain_attr`:

| Call | Purpose | Note |
| --- | --- | --- |
| `doca_gpi_domain_attr_create(&attr)` / `doca_gpi_domain_attr_destroy(attr)` | Allocate / free the domain attribute object | Sizing is set on this object before `doca_gpi_domain_create` |
| `doca_gpi_domain_attr_set_num_ep`, `_set_num_binds`, `_set_bind_size`, `_set_enable_ro`, `_set_num_channels` | Size endpoints, memory binds, bind size, relaxed ordering, and channel count for the domain | `doca_gpi.h` exposes no `doca_gpi_cap_*` query; an out-of-range value surfaces as a `DOCA_ERROR_*` at create time |
| `doca_gpi_domain_create(gpi, attr, &domain)` / `doca_gpi_domain_destroy(domain)` | Create / destroy a domain on the GPI instance | |
| `doca_gpi_domain_attach_local_mmap(domain, mmap, &bind_id)` | Attach a local `doca_mmap` to the domain, returning a `doca_gpi_mmap_handle_t` | The application is responsible for creating the `doca_mmap` for the memory area |
| `doca_gpi_domain_attach_remote_mmap(domain, ep, mmap, bind_id)` | Attach a remote peer's `doca_mmap` against an endpoint and bind id | The application creates the remote `doca_mmap` and exchanges it with the peer out of band |
| `doca_gpi_domain_detach_mmap(domain, bind_id)` | Detach a previously attached mmap | |

### The channel and endpoint surface

A GPI domain exposes channels sized via
`doca_gpi_domain_attr_set_num_channels`; each channel is created
from a `doca_gpi_channel_attr` on a GPU device. Every symbol
below is `DOCA_EXPERIMENTAL`. Verified channel surface:

| Call | Purpose | Note |
| --- | --- | --- |
| `doca_gpi_channel_attr_create(&attr)` / `doca_gpi_channel_attr_destroy(attr)` | Allocate / free the channel attribute object | |
| `doca_gpi_channel_attr_set_sq_wqe_num`, `_set_srq_wqe_num`, `_set_gpu_wqe_num`, `_set_num_counters`, `_set_num_signals`, `_set_low_latency`, `_set_pending_credits_writeback_shift`, `_set_srq_repost_mask` | Size the send / shared-receive / GPU work-queue depth, counters, signals, and behavior flags | All `uint16_t` WQE counts; set on the attr before `doca_gpi_channel_create` |
| `doca_gpi_channel_create(gpu_dev, domain, attr, &channel)` | Create the channel on a `doca_gpu` device and the supplied domain | Takes the GPU device as its first argument |
| `doca_gpi_channel_destroy(channel)` | Release a channel created with `doca_gpi_channel_create` | |
| `doca_gpi_gpu_channel_get(channel, &gpu_channel)` | Retrieve the GPU-side handle the CUDA kernel uses; type is `struct doca_gpu_gpi_channel*` | This is the only host→device handoff; the type is defined by [`doca-gpunetio`](../doca-gpunetio/SKILL.md) |
| `doca_gpi_channel_ep_conn_info_create(channel, ep_idx, &conn_info_size, &conn_info)` | Build the connection-info blob for an endpoint, to exchange out of band | Pair with `doca_gpi_channel_ep_conn_info_destroy(conn_info)` once the peer's blob has been consumed |
| `doca_gpi_channel_ep_connect(channel, ep_idx, conn_info, conn_info_size)` | Connect the endpoint using the peer's connection-info blob | After this call the GPU side can issue work on the endpoint |
| `doca_gpi_channel_get_error_state(channel)` | Check the channel error state | Returns `DOCA_SUCCESS` if no error, `DOCA_ERROR_DRIVER` on error |

### The CUDA-side surface

The GPU-side handle returned by
`doca_gpi_gpu_channel_get` — type
`struct doca_gpu_gpi_channel*` — is consumed by a CUDA kernel
compiled with `nvcc` against the DOCA GPU NetIO device-side
header set. The CUDA-side programming model (kernel launch, GPU
memory mapping, the device-side API surface that *uses* the
GPI channel) is owned by
[`doca-gpunetio`](../doca-gpunetio/SKILL.md) — the
`doca_gpu_gpi_channel` type and its device-side `.cuh` API are
defined there; this skill does not duplicate it. Two rules that
DO belong here because they are GPI-specific:

- The GPU handle returned by
  `doca_gpi_gpu_channel_get` is **the only legal bridge**
  between host-side configuration and CUDA-side execution. The
  agent does not invent a different handoff (e.g. casting a host
  pointer to a CUDA-managed pointer) — the header names this
  entry point.
- The GPU handle is valid only once the channel has been created
  on a configured, started GPI instance. The agent does not
  invent an ordering the header does not state; if
  `doca_gpi_gpu_channel_get` returns a `DOCA_ERROR_*`, walk the
  lifecycle in [`## Capabilities and modes`](#capabilities-and-modes)
  and confirm the channel was created before the handle was
  requested.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the headers-win-over-docs
rule, see [`doca-version`](../../doca-version/SKILL.md). The body
lives there; this skill does not duplicate it.

**The GPI-specific overlay** is:

- **The entire GPI surface is `DOCA_EXPERIMENTAL`.** Every
  `doca_gpi_*` function in `doca_gpi.h` carries the
  `DOCA_EXPERIMENTAL` annotation — there is no `DOCA_STABLE`
  subset. The agent must surface this whenever the user wants to
  "ship to production": a future DOCA release may rename or
  change the shape of any GPI call, so a version pin and a
  re-test gate are mandatory, and no part of the API should be
  treated as a frozen, long-term-stable contract.
- **CUDA Toolkit is a second compatibility axis.** GPI's GPU-
  side handle is consumed by `nvcc`-compiled code, against the
  DOCA GPU-NetIO device-side header set. The DOCA-side `.pc`
  version (`pkg-config --modversion doca-gpi`) is one axis; the
  installed CUDA Toolkit version is a second axis. The
  authoritative DOCA ↔ CUDA pairing for a given DOCA release
  lives in the release notes; route through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  for the release-notes URL pattern rather than quoting a CUDA
  version pin from agent memory.
- **`doca-gpi.pc` plus `doca-gpunetio.pc` plus `doca-dpa.pc`
  plus `doca-verbs.pc` must all match `doca_caps --version`** at
  the four-way-match check (per
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)).
  `dependencies/meson.build` lists `doca-dpa`, `doca-gpunetio`,
  and `doca-verbs` (plus the `libmlx5` / `libibverbs` externals)
  as GPI's DOCA dependencies; a partial install where one of
  these `.pc` files reports a different version is the most
  common partial-install pattern for GPI users.
- **The closest public docs surface for the GPU-side handoff is
  the DOCA GPU NetIO programming guide.** Until a dedicated
  *DOCA GPI* page is published, the agent uses the sister DOCA
  GPU NetIO guide at
  [docs.nvidia.com/doca/sdk/doca-gpunetio/index.html](https://docs.nvidia.com/doca/sdk/doca-gpunetio/index.html)
  for the underlying GPU-NetIO concepts and explicitly frames
  GPI as the lower-level channel/queue surface rather than a
  re-export of GPU NetIO. The
  [DOCA SDK index](https://docs.nvidia.com/doca/sdk/) is the
  authoritative starting point for whether a *GPI*-specific
  page now exists; route through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  for the up-to-date URL pattern rather than quoting a URL
  literal from agent memory.

## Error taxonomy

The cross-library `DOCA_ERROR_*` taxonomy (what each family
means and which debug layer it routes to) lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
The GPI-specific overlay names the families the agent will see
most often from `doca_gpi_*` calls and what they specifically
indicate:

| Family | Most common GPI cause | First action |
| --- | --- | --- |
| `DOCA_ERROR_BAD_STATE` | A `doca_gpi_set_*` attribute call ran *after* `doca_gpi_start()`, or `doca_gpi_get_dpa` was called on an already-started instance (the header notes it "can be called only if gpi not started") | Walk the lifecycle in [`## Capabilities and modes`](#capabilities-and-modes); confirm every `doca_gpi_set_*` call and `doca_gpi_get_dpa` landed before `doca_gpi_start()` |
| `DOCA_ERROR_INVALID_VALUE` | An attribute setter (`doca_gpi_domain_attr_set_*`, `doca_gpi_channel_attr_set_*`) was given an unsupported value, or a pointer argument is NULL | GPI exposes no `doca_gpi_cap_*` query, so re-derive the supported range from the device and release notes rather than agent memory; do not invent a runtime maximum |
| `DOCA_ERROR_NO_MEMORY` | `doca_gpi_create` failed to allocate internal state | Inspect the system's available memory; this is rarely an application bug, usually a host-side resource issue |
| `DOCA_ERROR_INITIALIZATION` | `doca_gpi_create` failed to initialize internal state | Same as above — inspect host resources |
| `DOCA_ERROR_IN_USE` | `doca_gpi_destroy` ran while domains or channels are still alive | Destroy every channel (`doca_gpi_channel_destroy`) and domain (`doca_gpi_domain_destroy`), and detach mmaps (`doca_gpi_domain_detach_mmap`), before destroying the GPI instance |
| `DOCA_ERROR_NOT_SUPPORTED` | The installed DOCA version does not export the requested `doca_gpi_*` symbol, or the device does not support the GPU datapath this code requires | Confirm the symbol exists in the installed headers per [`## Version compatibility`](#version-compatibility); confirm GPU-datapath support on the device via `doca_caps` ([`doca-caps`](../../tools/doca-caps/SKILL.md)) |
| `DOCA_ERROR_DRIVER` | The layer below DOCA (mlx5 driver, firmware, the verbs / RDMA stack GPI depends on) reported a failure | Stop. This is not a GPI-spec problem. Capture `dmesg | tail` and `mlxconfig -d <pcie> q`; route to [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug) layer 7 |

Quote `doca_error_get_descr()` verbatim — do not paraphrase. The
cross-cutting debug ladder
([`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug))
is the canonical layered diagnosis path that the agent escalates
to once the GPI-specific cause has been narrowed.

## Observability

GPI's observability surface is **split between host and CUDA
kernel**, and the agent must keep both visible when walking a
problem.

1. **Host-side: error monitoring and channel error state.** The
   host side enables the GPI error path with
   `doca_gpi_set_enable_err_monitor` at configure time and reads
   per-channel health with `doca_gpi_channel_get_error_state`,
   which returns `DOCA_SUCCESS` if no error and `DOCA_ERROR_DRIVER`
   on a driver-level failure. The host side does NOT observe
   per-work-request completions — those happen on the GPU side.
   It only sees lifecycle transitions, channel error state, and
   any host-initiated call that fails.
2. **GPU-side: the CUDA kernel reads the channel directly.**
   The CUDA kernel that holds the `doca_gpu_gpi_channel*`
   handle reads completions, posts work, and observes per-channel
   state through the device-side API surface owned by
   [`doca-gpunetio`](../doca-gpunetio/SKILL.md). The host side
   is blind to those completions; the only signal the host gets
   that work happened is application-level (counters the CUDA
   kernel updates in shared memory, host-side timing, observable
   network traffic on the wire).
3. **Attribute snapshot at configure time.** The channel and
   domain attribute values the application set on its
   `doca_gpi_domain_attr` / `doca_gpi_channel_attr` objects are
   the record of *what was requested*. Save them as the baseline;
   GPI exposes no `doca_gpi_cap_*` query, so if
   `doca_gpi_domain_create` / `doca_gpi_channel_create` returns a
   `DOCA_ERROR_*`, the diff against this snapshot — not a
   fabricated runtime maximum — is where the bug lives.
4. **Out-of-band exchanges.** Two artifacts cross the wire out of
   band and are observable when *"the peer says it can't reach my
   memory"*: the remote `doca_mmap` the peer attaches via
   `doca_gpi_domain_attach_remote_mmap`, and the endpoint
   connection-info blob produced by
   `doca_gpi_channel_ep_conn_info_create` and consumed by
   `doca_gpi_channel_ep_connect`. Diff what the local side built
   against what the peer received over its out-of-band channel.

For cross-cutting observability primitives (`--sdk-log-level`,
the `doca-<lib>-trace` build flavor, the `DOCA_LOG_LEVEL` env
var) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
For the install-tree observability (logger names, package
layout) defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

GPI's safety surface is **GPU-side initiator**: a CUDA kernel
that has been handed a `doca_gpu_gpi_channel*` can issue RDMA
work directly against a remote peer's memory with no host CPU on
the data path. A wrong configuration or a stale handle silently
issues remote operations the user did not intend. Three policies
follow from that:

1. **The GPU handle is a credential, not a pointer.** Treat
   `doca_gpu_gpi_channel*` like any other capability handle: it
   is valid only for the lifetime of the channel and GPI instance
   that produced it. Reusing a handle across a
   `doca_gpi_stop` / `doca_gpi_start` restart, sharing a handle
   between unrelated CUDA kernels, or persisting it across GPU
   driver reloads is undefined behavior. The agent enforces the
   *"one handle, one channel, one consuming kernel"* discipline.
2. **Do not invent a sizing limit.** GPI exposes no
   `doca_gpi_cap_*` devinfo query, so there is no runtime maximum
   to quote. Sizing is set on `doca_gpi_domain_attr` /
   `doca_gpi_channel_attr` objects, and an out-of-range value
   surfaces as a `DOCA_ERROR_*` from `doca_gpi_domain_create` /
   `doca_gpi_channel_create` — an error the application may have
   logged-and-ignored. The agent refuses to recommend a number it
   cannot source from the device or the release notes, and never
   fabricates a cap.
3. **Out-of-band exchange artifacts are wire-format secrets.**
   The remote `doca_mmap` attached via
   `doca_gpi_domain_attach_remote_mmap` and the endpoint
   connection-info blob from `doca_gpi_channel_ep_conn_info_create`
   let a remote peer address into bound GPU memory and connect to
   the channel. Treat their out-of-band transport like any other
   RDMA descriptor: over a secure channel, only to authenticated
   peers, with the *"production environments need a secure
   channel"* discipline the public DOCA RDMA guide names (see
   [`doca-rdma CAPABILITIES.md ## Safety policy`](../doca-rdma/CAPABILITIES.md#safety-policy)).
   The GPI overlay does not relax that rule.

For changes that touch hardware state below the GPI library
itself — `mlxconfig`-class writes, firmware burns, BlueField BFB
reflash, host kernel boot parameters (IOMMU mode is particularly
load-bearing for GPUDirect-style memory mapping) — the
cross-cutting meta-policy in
[`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md)
applies without modification. GPI does not redefine those rules;
the agent walks the hardware-safety ladder first whenever the
symptom involves device state, then the GPI overlay above for
the API-surface specifics.

## Deferred topic boundaries

This skill scopes itself to the DOCA GPI library. Adjacent
topics the agent will get asked but should route elsewhere:

- **The CUDA programming model** (kernel launch, stream
  ordering, GPU memory allocation) — outside this skill. The
  upstream CUDA documentation is the right answer; this skill
  assumes the user already builds and launches CUDA kernels.
- **The DOCA GPU NetIO Send/Receive surface** — owned by
  [`doca-gpunetio`](../doca-gpunetio/SKILL.md). GPI is the
  lower-level channel/queue surface; GPU NetIO is the higher-
  level Send/Receive surface. The selection table at the top
  of [`## Capabilities and modes`](#capabilities-and-modes)
  routes the agent there.
- **The doca-rdma queue lifecycle and permission matrix** —
  owned by [`doca-rdma`](../doca-rdma/SKILL.md). GPI does not
  create or bind a `doca-rdma` queue — its transport layer is
  `doca-verbs` ([`doca-verbs`](../doca-verbs/SKILL.md)). The
  host-CPU-initiated RDMA queue object, its transport type, and
  its permission matrix are not GPI's concern.
- **DPA-resident accelerator initiation** — owned by
  [`doca-rdmi`](../doca-rdmi/SKILL.md). GPI is the GPU case;
  RDMI is the DPA case. The CLASS of "accelerator-initiated
  one-sided RDMA" applies to both.
- **Cross-library `DOCA_ERROR_*` taxonomy** — owned by
  [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
  This skill adds the GPI overlay, not the taxonomy itself.
- **Cross-library capability-snapshot tooling** — owned by
  [`doca-caps`](../../tools/doca-caps/SKILL.md). This skill
  references the tool; it does not redefine its invocation
  patterns.
