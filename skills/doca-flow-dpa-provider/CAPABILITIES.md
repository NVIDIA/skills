# DOCA Flow DPA Provider capabilities, version overlay, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your
question (the three-program model / the per-port
`doca_flow_dpa_ctx` / the three queue types / the
pipe-export and external-resource-export handshake / the
DPA-side device API surface / the export lifecycle that
gates everything) and read that section end-to-end. The
tables in each section are the load-bearing content; the
prose around them is interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each
pattern (the verbs `install / configure / build / modify /
run / test / debug / use`), jump to [TASKS.md](TASKS.md).
For the canonical DOCA version-handling rules that this
skill layers a provider overlay on top of, see
[`doca-version`](../../doca-version/SKILL.md). For the
host-side Flow surface this library exports, see
[`doca-flow`](../doca-flow/SKILL.md). For the host-side DPA
lifecycle and the two-side-program rule this library
inherits, see [`doca-dpa`](../doca-dpa/SKILL.md).

## Pattern overview

Every DPA Flow Provider question this skill teaches resolves
into one of SIX patterns. The patterns are CLASSES — they
apply across every DOCA release and every host + BlueField +
DPACC combination.

| Pattern | When it applies (class shape) | Where the substance lives |
| --- | --- | --- |
| 1. Pick the right surface — do you actually need this library at all? | The library is a NICHE bridge for the case where a `doca-flow` pipe must be reachable from a DPA kernel; pure host-side Flow does not need it, pure generic DPA does not need it, and most first-time askers actually want one of those two | [`## Capabilities and modes`](#capabilities-and-modes) three-program rule + [TASKS.md ## configure](TASKS.md#configure) step 1 |
| 2. Walk the three-program model before drafting any code | A deployment using this library has THREE coupled translation units (host-side `doca-flow` programming, host-side `doca-dpa` / DPACC kernel launch, DPA-side kernel that consumes the exported device address); skipping any of the three is the most common design error | [`## Capabilities and modes`](#capabilities-and-modes) three-program rule + [TASKS.md ## configure](TASKS.md#configure) step 2 |
| 3. Allocate the per-port DPA Flow context and its queues | One `doca_flow_dpa_ctx` per BlueField port the user wants DPA-reachable Flow on; queues are allocated per port via `doca_flow_dpa_queues_create` with the three queue types (general, resources-write, resources-read) | [`## Capabilities and modes`](#capabilities-and-modes) per-port-context + queue-types tables + [TASKS.md ## configure](TASKS.md#configure) step 4 |
| 4. Run the export handshake in the documented order | `doca_flow_dpa_pipe_export_prepare` → add entries → `doca_flow_dpa_pipe_export` → `doca_flow_dpa_pipe_get_device_addr`; `_export_prepare` MUST happen before any entry is added to the pipe, otherwise those entries do not get exported | [`## Capabilities and modes`](#capabilities-and-modes) pipe-export-handshake table + [`## Safety policy`](#safety-policy) export-lifecycle rule + [TASKS.md ## configure](TASKS.md#configure) step 5 |
| 5. Hand the device address to a DPA kernel and consume it through the DPA-side API | The DPA kernel receives the `doca_flow_dpa_addr` as a launch argument (or in DPA-side state); it then calls `doca_flow_pipe_hash_*`, `doca_flow_external_resource_*`, and `doca_flow_queue_poll_completion` on that handle | [`## Capabilities and modes`](#capabilities-and-modes) DPA-side API table + [TASKS.md ## run](TASKS.md#run) + [TASKS.md ## use](TASKS.md#use) |
| 6. Diagnose a provider error | Map `DOCA_ERROR_INVALID_VALUE` / `_NOT_SUPPORTED` / `_NO_MEMORY` / `_BAD_STATE` / `_AGAIN` / `_IO_FAILED` / `_DRIVER` to a root cause without leaving the provider layer prematurely | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **The three-program model is non-negotiable.** A
  Flow-DPA-Provider deployment is THREE coupled translation
  units: a host-side `doca-flow` program (builds the pipe
  spec, opens the port), a host-side `doca-dpa` / DPACC-
  driven program (creates the per-DPA context, loads the DPA
  kernel image), and a DPA-side kernel (consumes the exported
  pipe device address). An agent that treats it as one
  program — for example by proposing that the host code "just
  call into the DPA function directly" — has the model wrong
  for every version of the provider. The host *exports*; the
  DPA kernel *executes*; the host-side `doca-flow` programs
  the pipe; the three sides are coupled by the device-address
  handoff and by the queue allocation that frames every
  DPA-side request.
- **The export lifecycle is order-sensitive.**
  `doca_flow_dpa_pipe_export_prepare` MUST be called BEFORE
  any entry is added to the pipe; otherwise the documented
  contract is that those entries will not be exported and the
  DPA kernel cannot see them. Skipping this rule is the
  single most common provider first-app bug. See the
  documented ordering on `doca_flow_dpa_pipe_export_prepare`
  in the installed header
  `doca_flow_dpa_provider.h` ("This operation must be done
  before any entry is added to the pipe, otherwise such
  entries will not be exported") and the export-lifecycle
  rule in [`## Safety policy`](#safety-policy).

## Capabilities and modes

The two orthogonal selection axes for any provider design are
*which BlueField port carrying the Flow pipe* (`doca_flow_dpa_ctx`
per `flexio_process` that maps to a DPA-capable BlueField AND
a `doca_flow_port` that maps to the same device) and *which
Flow pipe or external resource is being exported*. Choose
both before writing any host-side export code, then drill
into the relevant capability surface.

**Three-program model — host Flow + host DPA + DPA kernel.**

| Side | What runs there | Toolchain | What this skill covers |
| --- | --- | --- | --- |
| Host (Flow) | C / C++ (or any language that can FFI a C library) using `doca-flow` to bring up the port and construct the pipe spec | Host system compiler + `pkg-config doca-flow` | This skill *consumes* the result; the build-up of the pipe is owned by [`doca-flow`](../doca-flow/SKILL.md). The agent surfaces that the pipe is the same pipe in both libraries — the provider does NOT create its own pipe object |
| Host (DPA / DPACC) | C / C++ using `doca-flow-dpa-provider` (this skill) + the host-side DPA execution surface (`doca-dpa` and the underlying FlexIO process) to allocate queues, run the export handshake, and hand the device address to the DPA kernel | Host system compiler + `pkg-config doca-flow-dpa-provider` + the DPACC compiler for the DPA-side translation unit | All of `## Capabilities and modes` / `## Error taxonomy` / `## Observability` / `## Safety policy` below |
| DPA side | The kernel function body that runs on the DPA processor; consumes the `doca_flow_dpa_addr` device pointer and calls the DPA-side device API (`doca_flow_pipe_hash_*`, `doca_flow_external_resource_*`, `doca_flow_queue_poll_completion`) | DPACC (DPA-side compiler) plus the device-side library that is linked when the kernel includes `doca_flow_dpa_provider_dev.h` | This skill names the DPA-side surface and routes via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) to the public DOCA DPA / DPACC guides for the kernel-writing mechanics; it does not redefine those surfaces |

The agent's rule: when the user asks *"how do I write the DPA
kernel itself"*, that is the DPA-side question — route to
the public DPA / DPACC guide via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
When the user asks *"how do I export my host-side Flow pipe
so a DPA kernel can read its counters and modify its
entries"*, that is this skill's scope.

**Per-port `doca_flow_dpa_ctx` context — one per BlueField
port the host wants DPA-reachable Flow on.**

| Object | Lifetime | What it owns | Key calls |
| --- | --- | --- | --- |
| `doca_flow_dpa_ctx` | Per BlueField port + per FlexIO process the host is driving; created against a `flexio_process` that maps to a DPA-capable BlueField, paired with a `doca_flow_port` that was brought up on the same device | The DOCA-side bookkeeping for that port's DPA-reachable Flow surface, the registration of the allocated queues, the registration of the exported pipes (`doca_flow_dpa_pipe`) and exported external resources (`doca_flow_dpa_external_resource`) | `doca_flow_dpa_init(process, &ctx)` / `doca_flow_dpa_destroy(ctx)` (per the installed `doca_flow_dpa_provider.h`) |

A multi-port host that wants DPA-reachable Flow on more than
one port needs one `doca_flow_dpa_ctx` per port — there is
no global "DPA Flow context". The agent must ask which port
(which `doca_flow_port`, against which `flexio_process`)
before recommending any `doca_flow_dpa_*` call.

**Queue types — the three roles a DPA-side request can
take.**

| Enum value | Purpose | Why the agent must surface it |
| --- | --- | --- |
| `DOCA_FLOW_DPA_QUEUE_TYPE_GENERAL` | General (entries) queue used for hash-pipe entry manipulation requests issued from the DPA | Carries `doca_flow_pipe_hash_*` requests on the DPA side; if the user wants entry-level control, this queue type must be in the queue-config array |
| `DOCA_FLOW_DPA_QUEUE_TYPE_RESOURCES_WRITE` | Resource-update (write) queue used for in-DPA writes to external resources | Carries `doca_flow_external_resource_memory_update` and the index-selector modify requests; required for any DPA-side mutation of memory or index-selector resources |
| `DOCA_FLOW_DPA_QUEUE_TYPE_RESOURCES_READ` | Resource-query (read) queue used for in-DPA reads of external resources | Carries `doca_flow_external_resource_memory_read*`; values are populated only AFTER the matching completion is polled on this queue type. The agent must surface that the read API is two-phase (issue, then poll for the populated value) |

The queue-config is an *array* passed to
`doca_flow_dpa_queues_create`; each entry names the type,
the queue depth (which must be a power of 2), and the number
of queues of that type. The agent's anti-pattern alert:
allocating only `GENERAL` queues and then expecting in-DPA
memory reads to work is a guaranteed
`DOCA_ERROR_NOT_SUPPORTED` later.

**Pipe-export handshake — the load-bearing ordering this
skill exists to teach.**

| Step | Call | Why it has to happen at this position |
| --- | --- | --- |
| 1. Build the pipe on the host | `doca_flow_pipe_create` (owned by [`doca-flow`](../doca-flow/SKILL.md)) — the pipe object that will be exported | The provider does NOT define its own pipe; it bridges an existing `doca-flow` pipe. If the pipe spec is wrong, fix that in `doca-flow` BEFORE coming here |
| 2. Mark the pipe as a candidate for export | `doca_flow_dpa_pipe_export_prepare(ctx, pipe, nr_entries, &dpa_pipe)` — must be done BEFORE any entry is added to the pipe (per the installed header), otherwise those entries will not be exported | The provider must register the pipe's planned entry budget BEFORE the host adds entries; doing it after the entries are added silently produces an exported pipe with no entries the DPA kernel can see |
| 3. Add entries on the host | `doca_flow_pipe_add_entry` (owned by `doca-flow`), called AFTER step 2 | The host still owns the entry-add surface; the provider just made the pipe DPA-aware in step 2 |
| 4. Commit the export | `doca_flow_dpa_pipe_export(ctx, dpa_pipe)` — actually copies the pipe's state into DPA address space | Until this call, the device address is not usable from the DPA kernel even though `dpa_pipe` exists on the host side. Re-exporting an already-exported pipe returns `DOCA_ERROR_BAD_STATE` per the installed header |
| 5. Get the device address | `doca_flow_dpa_pipe_get_device_addr(ctx, dpa_pipe, &dev_addr)` — populates the `doca_flow_dpa_addr` opaque 64-bit device-side pointer | The device address is what the DPA kernel actually consumes; the agent surfaces that the kernel does NOT take a host-side pipe pointer, it takes this opaque DPA address. The host hands the address to the kernel through whatever DPA launch-argument mechanism the host-side DPA library provides |

The matching destroy call is
`doca_flow_dpa_pipe_destroy`; lifecycle ordering on teardown
is the reverse of the configure order and is documented in
[`## Safety policy`](#safety-policy).

**External-resource export handshake — same shape, different
surface.**

| Step | Call | Why it has to happen at this position |
| --- | --- | --- |
| 1. Build the external resource on the host | The matching `doca-flow` external-resource API (an index-selector resource or a memory resource) — owned by [`doca-flow`](../doca-flow/SKILL.md) | Same separation of concerns as the pipe — the provider does not define its own external resource |
| 2. Export to the DPA | `doca_flow_dpa_external_resource_export(ctx, resource_type, resource_ctx, &dpa_resource)` | Creates the DPA-side handle. Returns `DOCA_ERROR_BAD_STATE` if the queues required for the resource type were not yet created (see queue-config above) |
| 3. Get the device address | `doca_flow_dpa_external_resource_get_device_addr(ctx, dpa_resource, &dev_addr)` | The DPA-side device pointer the kernel consumes. The DPA kernel calls `doca_flow_external_resource_*` against this address |

The matching destroy call is
`doca_flow_dpa_external_resource_destroy`.

**DPA-side consumption — what the kernel calls against the
exported handles.**

| Surface (DPA-side) | What it does | Which queue type carries it |
| --- | --- | --- |
| `doca_flow_pipe_hash_enable_index` | Re-enable a hash-pipe entry that was previously disabled | `GENERAL` |
| `doca_flow_pipe_hash_disable_index` | Disable a hash-pipe entry | `GENERAL` |
| `doca_flow_pipe_hash_replace_index` | Replace a path-selector value within an existing hash-pipe entry | `GENERAL` |
| `doca_flow_external_resource_index_selector_modify` | Modify a single index in an index-selector external resource (enable / skip-once / skip-twice / disable) | `RESOURCES_WRITE` |
| `doca_flow_external_resource_index_selector_modify_range` | Modify a range of indices in an index-selector external resource — issues one low-level request per 32-index chunk per the installed header | `RESOURCES_WRITE` |
| `doca_flow_external_resource_memory_update` | Update a memory-resource value at an offset | `RESOURCES_WRITE` |
| `doca_flow_external_resource_memory_read` | Read a memory-resource value at an offset; the value is populated only AFTER polling the `RESOURCES_READ` completion queue | `RESOURCES_READ` |
| `doca_flow_external_resource_memory_read_range` | Read a range of memory-resource values; one low-level request per 8-index chunk per the installed header | `RESOURCES_READ` |
| `doca_flow_queue_poll_completion` | Poll a queue's completion ring; populates a `doca_flow_dpa_completion_stats` with `num_completed` and `num_failed` | All three queue types (parameter `queue_type`) |
| `doca_flow_queue_flush` | Flush all pending requests in a queue | All three queue types |

Each DPA-side request takes a `flush` boolean — when true,
the call flushes pending requests after posting. The agent
must surface that a request that fails to drain (no
`poll_completion` call on the matching queue type) eventually
returns `DOCA_ERROR_AGAIN` from the DPA-side post call. This
is the same pattern as the cross-library *"would-block,
retry after progress"* contract.

**Capability discovery.** Provider capability is conditional
on the underlying DOCA Flow capability set AND the
underlying DOCA DPA capability set. The agent must surface
that there is no separate `doca_flow_dpa_provider_cap_*`
family — capability discovery for this skill is the *join*
of
[`doca-flow CAPABILITIES.md ## Capabilities and modes`](../doca-flow/CAPABILITIES.md#capabilities-and-modes)
("is the pipe shape supported on this device") and
[`doca-dpa CAPABILITIES.md ## Capabilities and modes`](../doca-dpa/CAPABILITIES.md#capabilities-and-modes)
("does this BlueField expose a DPA processor"). A pipe shape
that fails to construct in `doca-flow` cannot be exported by
this skill; a BlueField that fails the DPA cap query cannot
host a `doca_flow_dpa_ctx`.

**Configuration shape.** *Mandatory* preconditions before any
`doca_flow_dpa_pipe_export_prepare` call: the host-side
`doca-flow` port and the host-side DPA execution surface
must both be brought up on the same `doca_dev`; a
`flexio_process` against the same device must exist (this is
the handle this library takes — `struct flexio_process *` is
the first argument of `doca_flow_dpa_init`); the
`doca_flow_dpa_ctx` must be created via
`doca_flow_dpa_init`; the queue-config array passed to
`doca_flow_dpa_queues_create` must include the queue types
the DPA kernel will actually use (allocating only `GENERAL`
when the kernel reads memory resources is the common
failure); the pipe spec must be constructed in `doca-flow`
but no entries added yet. *Optional* configurations (number
of queues per type, queue depth) are program-side tunables
that ride on top of the queue-type rule.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the
headers-win-over-docs rule, see
[`doca-version`](../../doca-version/SKILL.md). The body lives
there; this skill does not duplicate it.

**The provider-specific overlay** is:

- **DOCA Flow, DOCA DPA, and DPACC must agree per the DOCA
  Compatibility Policy.** This library bridges the two
  host-side surfaces and its DPA-side component is built by
  DPACC. Mismatched versions across the
  `doca-flow.pc` / `doca-dpa.pc` / `doca-flow-dpa-provider.pc`
  / installed `dpacc` set fail at link time (missing symbols
  on either side) or at runtime (`DOCA_ERROR_DRIVER` from a
  `doca_flow_dpa_*` call) in ways that look like hardware
  bugs but are version-skew bugs. The agent must surface ALL
  FOUR of `pkg-config --modversion doca-flow-dpa-provider`,
  `pkg-config --modversion doca-flow`, `pkg-config
  --modversion doca-dpa`, and the installed `dpacc` version,
  cross-check them against the DOCA Compatibility Policy at
  <https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html>,
  and route any disagreement to
  [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
  before any provider-layer diagnosis.
- **Use the current `doca_flow_dpa_queues_create` API.** The
  installed v3.5 public header (`doca_flow_dpa_provider.h`)
  only forward-declares `struct doca_flow_dpa_pipe_queues`;
  it does not expose `doca_flow_dpa_pipe_queues_create` /
  `_pipe_queues_destroy` and carries no `DOCA_DEPRECATED`
  marker for them in this release. New consumer code authored
  against this skill should always use the current
  `doca_flow_dpa_queues_create` API; the agent surfaces the
  older pipe-queues surface only when the user explicitly asks
  about an older installed sample that still references it.
- **The cross-cutting cap-query rule still applies.** Per
  [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability),
  the `doca_flow_*_cap_*` queries in `doca-flow` and the
  `doca_dpa_cap_*` queries in `doca-dpa` are the runtime
  authority for *"is this combined surface supported on
  this hardware + this DOCA install"*; the provider sits on
  top of both and does not add a third capability axis.
- **Public-doc routing.** Search
  <https://docs.nvidia.com/doca/sdk/> for the *DOCA Flow DPA
  Provider* page and cross-reference both
  <https://docs.nvidia.com/doca/sdk/doca-flow/index.html> and
  <https://docs.nvidia.com/doca/sdk/doca-dpa/index.html>
  since the library sits between the two. Route to
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  for the canonical URL list rather than quoting a specific
  URL from memory.

## Error taxonomy

Provider-specific overlays on the cross-library
`DOCA_ERROR_*` taxonomy. The cross-library taxonomy itself
lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
the rows below are the *provider surface* meaning the agent
must disambiguate before falling back to the cross-library
response. Every row reflects an error code documented in the
installed headers `doca_flow_dpa_provider.h` and
`doca_flow_dpa_provider_dev.h`.

| Error | Provider context where it shows up | Provider-specific cause |
| --- | --- | --- |
| `DOCA_ERROR_INVALID_VALUE` | `doca_flow_dpa_init`; `doca_flow_dpa_queues_create`; both `_pipe_export_prepare` and `_pipe_export`; both `_pipe_get_device_addr` and `_external_resource_*` variants | A NULL handle, a queue-config violation (e.g. queue_size not a power of 2), or a device-address-out parameter that is missing. The fix is at the call site; do not retry the same arguments |
| `DOCA_ERROR_NOT_SUPPORTED` | `doca_flow_dpa_init` (device-side uses features the BlueField does not support); `_queues_create` (failed to match requested queue config to device capabilities); `_pipe_export_prepare` / `_pipe_export` (the pipe shape does not support an export — for example, a non-hash pipe being asked to expose hash-entry control to the DPA); `_external_resource_*` (the external resource type is not exportable) | The pipe spec OR the queue config OR the BlueField generation is asking for something the underlying capability set does not advertise. Re-run capability discovery in `doca-flow` (for the pipe shape) and `doca-dpa` (for the BlueField generation); do NOT retry the same export against the same device |
| `DOCA_ERROR_NO_MEMORY` | `doca_flow_dpa_init`; `_queues_create`; `_pipe_export_prepare`; `_external_resource_export` | Allocation failure for the provider's internal bookkeeping (DPA context, queue arena, export context). Inspect host memory state; do not silently retry |
| `DOCA_ERROR_BAD_STATE` | `_queues_create` (queues were already created for this port); `_pipe_export_prepare` (queues were not yet created); `_pipe_export` (preparation step failed or the pipe was already exported); `_external_resource_export` (queues not yet created or wrong queue-type set for this resource type) | Lifecycle violation. Walk the export lifecycle in [`## Safety policy`](#safety-policy); the most common shape is calling `_pipe_export_prepare` before `_queues_create`, or calling `_pipe_export` twice on the same pipe |
| `DOCA_ERROR_DRIVER` | `doca_flow_dpa_init` (lower-level DPA-related layer failure); `_queues_create`; `_pipe_export` (error in DPA copy operations); on the DPA side, `doca_flow_queue_poll_completion` and `doca_flow_queue_flush` use `_DRIVER` for lower-layer failures | The DPA driver layer (or the joint DOCA + DPACC version skew) reported failure. Most common cause is a DOCA-Flow + DOCA-DPA + DPACC version mismatch per the DOCA Compatibility Policy; route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) layer 5 (driver) AND to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) for the version-skew side |
| `DOCA_ERROR_AGAIN` *(DPA-side)* | Every DPA-side post call (`doca_flow_pipe_hash_*`, `doca_flow_external_resource_*`) returns this when the matching work queue is full and the request could not be submitted | The DPA-side kernel is producing requests faster than the host (or the kernel itself) is polling the matching completion queue. Drain via `doca_flow_queue_poll_completion(queue_type)` on the right queue type, then retry. Same as the cross-library *"would-block, retry after progress"* pattern; this is NOT a queue-resize event |
| `DOCA_ERROR_IO_FAILED` *(DPA-side)* | DPA-side `doca_flow_pipe_hash_*` and `doca_flow_external_resource_*` post paths | The submission itself was rejected by the lower DPA layer. This is rarer than `_AGAIN` and typically points at queue state corruption (e.g. queue destroyed while in-flight requests existed) rather than backpressure |
| `DOCA_ERROR_NOT_SUPPORTED` *(DPA-side)* | `doca_flow_queue_poll_completion`, `doca_flow_queue_flush` (given queue type not supported); `doca_flow_pipe_hash_disable_index` (disable isn't supported for the given hash pipe); `doca_flow_external_resource_memory_read*` (read not supported in this configuration) | The DPA-side call asks for an operation the underlying pipe / resource / queue type does not expose. Re-read the pipe and resource construction in `doca-flow`; the fix is on the host side (build a pipe / resource that supports the operation), not on the DPA side |

The agent's rule: **never recommend a retry loop on a host-
side `DOCA_ERROR_*` without first identifying which of the
rows above is the cause**. Only the DPA-side `_AGAIN` row
wants a drain-then-retry pattern; the host-side rows want
investigation (lifecycle / capability / queue-config /
version), not retry.

## Observability

Provider observability surface is **two-sided**: there is a
host-side observability surface (the host-side DOCA logger
plus per-call return codes from the provider's `doca_flow_dpa_*`
family, plus the underlying Flow observability and DPA
observability surfaces) AND a DPA-side observability surface
(the `doca_flow_dpa_completion_stats` populated by
`doca_flow_queue_poll_completion`, plus the cross-library DPA
developer tools the agent inherits from
[`doca-dpa CAPABILITIES.md ## Observability`](../doca-dpa/CAPABILITIES.md#observability)).
The agent must reach for both, not just one — a DPA-side
request that is silently dropped because the matching
completion queue is full will not show up on the host's
return code, but it WILL show up in the DPA-side
`doca_flow_dpa_completion_stats` once polled.

Four primary signals the agent should reach for:

1. **Per-call return codes on the host side.** Every
   `doca_flow_dpa_*` host-side call returns a `doca_error_t`;
   the documented set per call is in
   [`## Error taxonomy`](#error-taxonomy). The agent must
   quote `doca_error_get_descr()` verbatim when reporting
   the error to the user, never paraphrase it.
2. **DPA-side completion statistics.** The
   `doca_flow_dpa_completion_stats` struct populated by
   `doca_flow_queue_poll_completion` carries
   `num_completed` (some of which may have failed) and
   `num_failed`. A monotonically growing `num_failed` is
   *the* canonical signal that the DPA-side request stream
   has a bug; the agent must surface that
   `num_failed > 0` is not the same as the post call
   returning a non-success code (the post call may have
   returned success and the request later failed at
   execution time).
3. **The underlying Flow observability surface.** Per-pipe
   and per-entry counters in `doca-flow` are still the
   ground truth for whether traffic is matching the
   exported pipe. The provider does not bypass them; reach
   for them per
   [`doca-flow CAPABILITIES.md ## Observability`](../doca-flow/CAPABILITIES.md#observability).
4. **The underlying DPA observability surface.** The DPA
   developer tools named in
   [`doca-dpa CAPABILITIES.md ## Observability`](../doca-dpa/CAPABILITIES.md#observability)
   (the DPA debugger, the DPA process-state inspector, the
   DPA statistics tool — all routed via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md))
   are the right surface when the DPA-side kernel is
   running but not making forward progress on the exported
   handles. Stuck-DPA cases are inherited from `doca-dpa`,
   not redefined here.

For cross-cutting observability primitives
(`--sdk-log-level`, the `DOCA_LOG_LEVEL` env var, the trace
build flavor) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
For the install-tree observability (logger names, package
layout, sample tree) defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

The provider's safety surface is **export-lifecycle-driven
AND three-program-driven**. Misusing the export handshake
does not just produce a runtime error — it can leave the
host-side Flow pipe and the DPA-side view of it
inconsistent, which surfaces later as traffic going to the
wrong place. The three most common provider first-app
failures are (1) calling `doca_flow_dpa_pipe_export_prepare`
AFTER adding entries to the pipe (those entries are not
exported, the DPA kernel sees an empty pipe and the host
blames the data plane); (2) allocating only `GENERAL` queues
and then expecting the DPA-side memory-read API to work; and
(3) tearing down the `doca_flow_dpa_ctx` while the DPA-side
kernel still holds the device address. The agent's job is to
enforce these orderings BEFORE the first export call, not
after the first `DOCA_ERROR_BAD_STATE`.

The **export lifecycle** the agent must walk for any new
provider setup:

1. **`doca_flow_dpa_init(process, &ctx)` against the matching
   FlexIO process.** The `flexio_process` must map to the
   same BlueField device as the `doca_flow_port` that the
   pipe will live on; mismatching devices silently exports
   a pipe to a DPA on a different device than the one
   carrying the traffic.
2. **`doca_flow_dpa_queues_create(ctx, port, queue_cfgs,
   num_queue_cfgs)` BEFORE any export call.** Include
   exactly the queue types the DPA kernel will use. Per the
   header, calling it twice on the same port returns
   `DOCA_ERROR_BAD_STATE`; queue depth must be a power of 2.
3. **`doca_flow_dpa_pipe_export_prepare(ctx, pipe,
   nr_entries, &dpa_pipe)` BEFORE any entry is added to
   `pipe`.** This is the load-bearing rule documented on
   the call in the installed header: *"This operation must
   be done before any entry is added to the pipe, otherwise
   such entries will not be exported."* Skipping it does
   NOT produce a host-side error — it silently produces an
   exported pipe whose entries the DPA cannot see.
4. **Add entries to the pipe via `doca-flow` after step 3,
   not before.** Re-validate the spec before each add per
   [`doca-flow CAPABILITIES.md ## Safety policy`](../doca-flow/CAPABILITIES.md#safety-policy).
5. **`doca_flow_dpa_pipe_export(ctx, dpa_pipe)`.** Calling
   it twice on the same `dpa_pipe` returns
   `DOCA_ERROR_BAD_STATE` per the installed header.
6. **`doca_flow_dpa_pipe_get_device_addr(ctx, dpa_pipe,
   &dev_addr)`.** The 64-bit `doca_flow_dpa_addr` is the
   handle the DPA kernel actually consumes.
7. **Teardown in reverse.** Destroy the exported pipe
   before destroying the `doca_flow_dpa_ctx`; destroy the
   `doca_flow_dpa_ctx` before destroying the underlying
   `doca_flow_port` and the `flexio_process`. Out-of-order
   teardown can leave the DPA processor referencing a stale
   device address — the agent must surface this even when
   the host-side teardown sequence appears to succeed.

**Do not partial-rebuild one side.** Inherited from
[`doca-dpa CAPABILITIES.md ## Safety policy`](../doca-dpa/CAPABILITIES.md#safety-policy):
a host-side rebuild against a new DOCA install with the
DPA-side image still built by an old `dpacc`, or the inverse,
fails the export handshake in non-obvious ways (the host
call returns success at link time, then `DOCA_ERROR_DRIVER`
at runtime). Rebuild BOTH sides against the matched DOCA +
DPACC versions per the DOCA Compatibility Policy.

**Validate the host-side pipe before exporting it.**
Inherited from
[`doca-flow CAPABILITIES.md ## Safety policy`](../doca-flow/CAPABILITIES.md#safety-policy):
the host-side `doca-flow` pipe specification MUST be
validated (via the Flow pipe-validate API or the dry-run
sample) BEFORE step 3 of the export lifecycle. A pipe that
fails Flow validation will fail export, and the export-side
error message is less specific than the validate-side error
message.

**The DPA-side kernel is the load-bearing
mutation surface.** Because the DPA kernel can disable hash
entries, replace path-selector values, and write to memory
resources inline with the datapath, a buggy DPA-side
kernel can take traffic offline as surely as a buggy
host-side Flow spec can. The agent's discipline: stage on a
non-production representor first, run a single hash-replace
or memory-update against a known-safe entry, poll the
completion queue to confirm `num_failed == 0`, then widen.
The replica-first rule from
[`doca-hardware-safety CAPABILITIES.md ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy)
applies.
