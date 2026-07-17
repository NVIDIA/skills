# DOCA RDMA Initiator capabilities, version compatibility, errors, observability, safety

**Where to start:** The pattern overview below names the recurring
RDMI-class patterns. Pick the pattern first, then drill into the H2
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

Every RDMI-class question this skill teaches resolves into one of
SIX patterns. The patterns are CLASSES — they apply across every
accelerator-initiated one-sided RDMA use case, not just the worked
example shown.

| RDMI pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Pick `doca-rdmi` vs `doca-rdma` | Decide *before* writing any code whether the initiator is accelerator-side (DPA / GPU) and one-sided, or host-CPU and possibly two-sided | [`## Capabilities and modes`](#capabilities-and-modes) surface-selection table |
| 2. Stand up the connection / poster | DOCA Core lifecycle on each object: create → configure (attach CQ or DPA completion; set verbs PD) → start → use → stop → destroy | [TASKS.md ## configure](TASKS.md#configure) |
| 3. Hand off to the DPA datapath | The DPA-side handle returned by `doca_rdmi_connection_get_dpa_handle` / `doca_rdmi_poster_get_dpa_handle` is the bridge between host-side configuration and DPA-side execution | [`## Capabilities and modes`](#capabilities-and-modes) DPA-handoff bullet + [TASKS.md ## run](TASKS.md#run) |
| 4. Drive completions correctly | Either a `doca_dpa_completion` on the DPA datapath (kernel-polled) or a `doca_verbs_cq` on the host side; never both implicitly | [`## Observability`](#observability) + [TASKS.md ## run](TASKS.md#run) |
| 5. Confirm the symbol is on this install and stable enough to ship | Every RDMI symbol is currently tagged `DOCA_EXPERIMENTAL` in the version map; quoting a symbol without confirming the install ships it (and surfacing the experimental flag) is the most common hallucination failure mode | [`## Version compatibility`](#version-compatibility) + [TASKS.md ## configure](TASKS.md#configure) |
| 6. Interpret a `DOCA_ERROR_*` from an RDMI call | Map the error to a layer (lifecycle / configuration / capability / DPA datapath / verbs below / driver) and route | [`## Error taxonomy`](#error-taxonomy) RDMI overlay + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **Treat every RDMI symbol as EXPERIMENTAL until confirmed
  otherwise.** The library's public version map exposes
  `doca_rdmi_*` symbols under the `EXPERIMENTAL` tag. Quoting a
  symbol or a behavior without surfacing the experimental
  qualifier — and without checking that the symbol is present in
  the installed headers — is the failure mode this rule prevents.
  See [`## Version compatibility`](#version-compatibility).
- **The DPA datapath has its own lifecycle.** RDMI hands off to
  DPA via a handle; the agent must walk the user through the
  `doca_ctx_set_datapath_on_dpa()` call before
  `doca_ctx_start()`, and through the DPA-side completion attach
  *before* the start, not after. Out-of-order returns
  `DOCA_ERROR_BAD_STATE`, not a clear symptom.

## Capabilities and modes

DOCA RDMI is the **initiator-side RDMA surface** for DOCA
applications whose data-movement code runs on an accelerator
datapath rather than on the host CPU. The host-side library
configures two objects — a *connection* and a *poster* — and hands
DPA-side handles to the accelerator kernel. The accelerator kernel
then drives RDMA work directly.

### `doca-rdmi` vs `doca-rdma` vs `doca-verbs`

Three RDMA-shaped libraries ship in DOCA. They are NOT
interchangeable; pick one before writing any code:

| Library | When to pick | Initiator | Direction |
| --- | --- | --- | --- |
| [`doca-rdma`](../doca-rdma/SKILL.md) | General-purpose RDMA from the host CPU or the BlueField CPU; supports the full eleven-task set (send / receive / read / write / atomic / sync-event variants); both two-sided and one-sided | Host or BlueField CPU | Two-sided or one-sided |
| `doca-rdmi` (this skill) | One-sided RDMA initiated from an accelerator datapath (DPA today; GPU via `doca-gpi`); focused connection + poster object pair; the accelerator kernel drives the queues directly | DPA kernel (this skill) or GPU kernel ([`doca-gpi`](../doca-gpi/SKILL.md)) | Initiator-side only |
| `doca-verbs` | The low-level verbs surface; RDMI builds on a `doca_verbs_context` for the underlying QP / CQ / PD primitives; rarely the right surface for application code — prefer the higher-level library above | Host or BlueField CPU | Either |

**Decision rule for the agent.** If the user's intent is *"my
accelerator kernel posts RDMA writes / reads / sends against a
remote responder, and I do not want the host CPU on the data
path"*, RDMI is the right surface. Anything else — two-sided
flows, host-CPU-initiated traffic, the rich completion / mmap
permission matrix — routes to
[`doca-rdma`](../doca-rdma/SKILL.md). When the user is unsure,
walk the table above before recommending any `doca_rdmi_*` call.

### The connection object

`doca_rdmi_connection` is the connection-state object. It is
created on a `doca_verbs_context` and a `doca_dev_rep`
representation, converted to a generalized DOCA Core context via
`doca_rdmi_connection_as_ctx()`, configured with at least one of
the attach calls below, and then started. Public surface verified
against the shipped headers (`doca_rdmi_connection.h`):

| Lifecycle phase | Calls | Note |
| --- | --- | --- |
| Create | `doca_rdmi_connection_create(verbs_ctx, rep, &conn)`; optionally `doca_rdmi_connection_set_pd(conn, pd)` | The verbs PD must be the same PD used elsewhere in the application's verbs surface |
| Configure | `doca_rdmi_connection_init_receive(conn, buf, size)` to hand the library the application's receive buffer; `doca_rdmi_connection_dpa_completion_attach(conn, dpa_comp)` OR `doca_rdmi_connection_cq_attach(conn, cq)` to bind a completion source; `doca_rdmi_connection_set_input(conn, op_type, in_buf, size)` for PRM-passthrough inputs (FW-control) | The attach call must run BEFORE `doca_ctx_start()`; the API doc strings explicitly require this |
| DPA-side handoff | `doca_rdmi_connection_get_dpa_handle(conn, &dpa_handle)` returns a `doca_dpa_dev_rdmi_connection_t` the DPA kernel uses via `doca_rdmi_dev_connection.h` | Only valid after `doca_ctx_start()` and on a context configured with `doca_ctx_set_datapath_on_dpa()` |
| Run | The DPA kernel drives receives; the host side calls `doca_rdmi_connection_recv_ack(conn, num_acked)` to release the corresponding RQ slots | The ack count must equal the number of receive work requests the DPA actually consumed |
| QP-state outputs | `doca_rdmi_connection_set_qp_create_out`, `_set_qp_modify_out`, `_set_qp_query_out` carry PRM-side QP-state outputs back into the verbs command pipeline | These calls have `@note This API will likely change once we have On-Behalf API support` per the shipped headers — surface the instability to the user |
| Destroy | `doca_rdmi_connection_destroy(conn)` after `doca_ctx_stop()` | The same lifecycle rule as every other DOCA Core context |

### The poster object

`doca_rdmi_poster` is the poster-side object that issues RDMA work
requests. Verified surface (`doca_rdmi_poster.h`):

| Lifecycle phase | Calls | Note |
| --- | --- | --- |
| Create | `doca_rdmi_poster_create(verbs_ctx, &poster)`; optionally `doca_rdmi_poster_set_pd(poster, pd)` | Same verbs context the connection uses |
| Configure | `doca_rdmi_poster_dpa_completion_attach(poster, dpa_comp)` OR `doca_rdmi_poster_cq_attach(poster, cq)`; `doca_rdmi_poster_set_input(poster, op_type, in_buf, size)` for PRM-passthrough inputs | Same before-start rule as the connection |
| DPA-side handoff | `doca_rdmi_poster_get_dpa_handle(poster, &dpa_handle)` returns a `doca_rdmi_dev_poster_t` the DPA kernel uses via `doca_rdmi_dev_poster.h` | Same datapath-on-DPA rule |
| Post (host-side helper) | `doca_rdmi_poster_post(poster, cqe_buf, num_cqe, mkey)` is the host-side helper that pushes CQEs for the poster's completion side | The bulk-post path is the DPA kernel itself; this host-side helper exists for the CQE-feeding case |
| Destroy | `doca_rdmi_poster_destroy(poster)` after `doca_ctx_stop()` | Standard |

### The DPA-side surface

The DPA-side headers — `doca_rdmi_dev_connection.h`,
`doca_rdmi_dev_poster.h`, `doca_rdmi_dev_cqe.h` — are the
device-side counterpart consumed by the DPA kernel. They are
compiled with the DOCA DPA toolchain owned by
[`doca-dpa`](../doca-dpa/SKILL.md); this skill names the handoff
(the handle type, the *valid only after start* rule) but does not
duplicate the DPA programming model. Two rules that DO belong
here because they are RDMI-specific:

- The DPA handle returned by `doca_rdmi_connection_get_dpa_handle`
  / `doca_rdmi_poster_get_dpa_handle` is **the only legal bridge**
  between host-side configuration and DPA-side execution. The
  agent does not invent a different handoff (e.g. casting a host
  pointer to a DPA address) — the doc string explicitly names
  this entry point.
- The DPA-side completion-attach call
  (`doca_rdmi_connection_dpa_completion_attach`,
  `doca_rdmi_poster_dpa_completion_attach`) must run **before**
  `doca_ctx_start()`. The shipped header note is explicit:
  *"This function must be called before DOCA RDMI Connection
  context is started"* — and *"This API is relevant only for
  contexts that are set on DPA datapath, using
  `doca_ctx_set_datapath_on_dpa()` before calling
  `doca_ctx_start()`."*

### Configuration shape

*Mandatory* before `doca_ctx_start()` on either object: a verbs
context (via `doca_verbs_*` setup), a verbs PD (if the application
shares a PD), and **exactly one** completion source — either a
`doca_dpa_completion` attached via
`doca_rdmi_*_dpa_completion_attach` for the DPA datapath, or a
`doca_verbs_cq` attached via `doca_rdmi_*_cq_attach` for the
host-side completion path. Attaching both is not documented in the
public headers as a supported configuration; the agent should not
recommend it.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the headers-win-over-docs
rule, see [`doca-version`](../../doca-version/SKILL.md). The body
lives there; this skill does not duplicate it.

**The RDMI-specific overlay** is:

- **Every public `doca_rdmi_*` symbol is currently tagged
  `EXPERIMENTAL`.** The library's public version map (the
  authoritative source for what the shipped `.so` actually
  exports) places `doca_rdmi_connection_*` and
  `doca_rdmi_poster_*` under the `EXPERIMENTAL` symbol set. The
  agent must surface this whenever it recommends an RDMI symbol:
  the API can change shape between DOCA releases, and the
  shipped header doc strings explicitly say so for the QP-state
  outputs ("This API will likely change once we have On-Behalf
  API support"). Build the user's expectations around that.
- **The runtime authority is the installed headers, not the
  public docs.** Per the cross-cutting rule in
  [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability),
  the agent confirms a symbol exists by checking the installed
  header (look up the install-path pattern in
  [`doca-public-knowledge-map ## Layout of an installed DOCA package`](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package))
  before recommending it. A *"function not found"* link error on
  a `doca_rdmi_*` symbol is most commonly a wrong-version
  mismatch — confirm the version per
  [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure)
  before any RDMI-layer diagnosis.
- **`doca-rdmi.pc` plus `doca-common.pc` plus `doca-verbs.pc`
  plus `doca-dpa.pc` must all match `doca_caps --version`** at
  the four-way-match check (per
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)).
  RDMI sits on top of the verbs surface and (for the DPA
  datapath) the DPA surface — a partial install where one of
  these `.pc` files reports a different version is the most
  common partial-install pattern for RDMI users. Route to
  [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
  layer 2 before any RDMI-layer diagnosis.
- **The closest public docs surface is the DOCA RDMA programming
  guide.** Until a dedicated *DOCA RDMA Initiator* page is
  published, the agent uses the sister DOCA RDMA guide at
  [docs.nvidia.com/doca/sdk/doca-rdma/index.html](https://docs.nvidia.com/doca/sdk/doca-rdma/index.html)
  for the underlying RDMA concepts and explicitly frames RDMI as
  a sister surface, not as a re-export of `doca-rdma`. The
  [DOCA SDK index](https://docs.nvidia.com/doca/sdk/) is the
  authoritative starting point for whether an *RDMI*-specific
  page now exists; route through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  for the up-to-date URL pattern rather than quoting a URL
  literal from agent memory.

## Error taxonomy

The cross-library `DOCA_ERROR_*` taxonomy (what each family means
and which debug layer it routes to) lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
The RDMI-specific overlay names the families the agent will see
most often from `doca_rdmi_*` calls and what they specifically
indicate:

| Family | Most common RDMI cause | First action |
| --- | --- | --- |
| `DOCA_ERROR_BAD_STATE` | A configuration call (`_init_receive`, `_dpa_completion_attach`, `_cq_attach`, `_set_input`) ran *after* `doca_ctx_start()`; or a DPA-side call ran on a context that was not set on the DPA datapath via `doca_ctx_set_datapath_on_dpa()` | Walk the lifecycle in [`## Capabilities and modes`](#capabilities-and-modes); confirm every attach/configure call landed before `doca_ctx_start()` and that the datapath was set before start |
| `DOCA_ERROR_INVALID_VALUE` | An object handle is NULL, a buffer size is zero, or a verbs context / PD argument is mismatched between connection and poster | Re-walk the create/configure step in [`## Capabilities and modes`](#capabilities-and-modes); the verbs PD must be consistent across the connection, poster, and any sibling verbs objects in the same application |
| `DOCA_ERROR_NOT_SUPPORTED` | The installed DOCA version does not export the requested `doca_rdmi_*` symbol, or the device does not support the DPA datapath this code requires | Confirm the symbol exists in the installed headers per [`## Version compatibility`](#version-compatibility); confirm DPA support on the device via `doca_caps` ([`doca-caps`](../../tools/doca-caps/SKILL.md)) |
| `DOCA_ERROR_NO_MEMORY` | The library could not allocate internal bookkeeping (queues, descriptors) — typically a budget exceeded on the verbs surface below RDMI | Inspect verbs-side resource limits; route to the verbs-layer / driver-layer ladder in [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug) |
| `DOCA_ERROR_DRIVER` | The layer below DOCA (mlx5 driver, firmware, the verbs QP transitions) reported a failure | Stop. This is not an RDMI-spec problem. Capture `dmesg | tail` and `mlxconfig -d <pcie> q`; route to [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug) layer 7 |

Quote `doca_error_get_descr()` verbatim — do not paraphrase. The
cross-cutting debug ladder
([`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug))
is the canonical layered diagnosis path that the agent escalates
to once the RDMI-specific cause has been narrowed.

## Observability

RDMI's observability surface is **split between host and
accelerator**, and the agent must keep both visible when walking a
problem.

1. **Host-side: the DOCA Core progress engine (PE) and the verbs
   CQ.** The host side of RDMI uses the universal Core-context
   PE for lifecycle events on the connection / poster objects.
   When a `doca_verbs_cq` is attached
   (`doca_rdmi_connection_cq_attach`,
   `doca_rdmi_poster_cq_attach`), CQEs land on that CQ and the
   host application reads them through the verbs CQ surface — not
   through a DOCA-side completion API. The agent should NOT
   describe a host-side DOCA completion event that is
   simultaneously consumed by the DPA when the attach was a CQ
   attach.
2. **DPA-side: `doca_dpa_completion`.** When a
   `doca_dpa_completion` is attached
   (`doca_rdmi_connection_dpa_completion_attach`,
   `doca_rdmi_poster_dpa_completion_attach`), the DPA kernel
   polls the completion through the
   [`doca-dpa`](../doca-dpa/SKILL.md) device-side surface. The
   host side is blind to those completions; the only host-side
   signal that work was consumed is the matching ack
   (`doca_rdmi_connection_recv_ack`) and the eventual stop/
   destroy lifecycle on the context.
3. **Receive bookkeeping.** The application's receive buffer
   passed to `doca_rdmi_connection_init_receive` is the
   library's RQ window. `doca_rdmi_connection_get_app_rq_info`
   surfaces the descriptor (mkey, offset, size) the DPA kernel
   needs to reference into that buffer. If the DPA consumed *N*
   work requests and the host never calls
   `doca_rdmi_connection_recv_ack(conn, N)`, the RQ slot budget
   exhausts and new receives back up — the symptom is *"the DPA
   posted, the work request never arrived"*. The fix is on the
   host side, not the DPA side.

For cross-cutting observability primitives (`--sdk-log-level`,
the `doca-<lib>-trace` build flavor, the `DOCA_LOG_LEVEL` env
var) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
For the install-tree observability (logger names, package
layout) defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

RDMI's safety surface is **initiator-side**: an accelerator kernel
that has been handed a DPA handle can post sends, writes, and reads
against a remote responder's memory with no host CPU on the path.
A wrong configuration or a stale handle silently issues remote
memory operations that the user did not intend. Three policies
follow from that:

1. **EXPERIMENTAL means "validate the shape on the installed
   version, every release."** Because every public `doca_rdmi_*`
   symbol is tagged `EXPERIMENTAL`, any RDMI-using component must
   re-confirm symbol presence and lifecycle order against the
   installed headers between DOCA upgrades. The agent must not
   recommend an RDMI deployment that hard-codes a version
   assumption; the per-release re-verification step in
   [`TASKS.md ## test`](TASKS.md#test) is the load-bearing gate.
2. **The DPA handle is a credential, not a pointer.** Treat
   `doca_dpa_dev_rdmi_connection_t` /
   `doca_rdmi_dev_poster_t` like any other capability handle:
   it is valid only for the lifetime of the context that
   produced it, and it grants the DPA kernel the right to issue
   remote operations through that connection. Reusing a handle
   across context restarts, sharing a handle between unrelated
   kernels, or persisting it across BlueField cold reboots is
   undefined behavior — and a security regression on the
   responder side. The agent enforces the *"one handle, one
   context, one consuming kernel"* discipline.
3. **Validate receives before posting.** Before the first work
   request posts from the DPA kernel, the host must have called
   `doca_rdmi_connection_init_receive` with a buffer large
   enough for the application's intended RQ window AND the
   host-side ack loop (`doca_rdmi_connection_recv_ack`) must be
   ready to release consumed slots. Posting against an un-init
   receive buffer or against an exhausted RQ window is the
   canonical *"the DPA posted, nothing happened, no error
   surfaced"* failure mode.

For changes that touch hardware state below the RDMI library
itself — `mlxconfig`-class writes, firmware burns, BlueField BFB
reflash, host kernel boot parameters that affect the DPA — the
cross-cutting meta-policy in
[`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md)
applies without modification. RDMI does not redefine those rules;
the agent walks the hardware-safety ladder first whenever the
symptom involves device state, then the RDMI overlay above for
the API-surface specifics.

## Deferred topic boundaries

This skill scopes itself to the DOCA RDMA Initiator library.
Adjacent topics the agent will get asked but should route
elsewhere:

- **General RDMA networking concepts** (queue pairs, memory
  regions, RoCE vs InfiniBand) — outside this skill. The
  upstream RDMA / IB documentation is the right answer; this
  skill assumes the user already understands the underlying
  abstractions.
- **Two-sided RDMA, host-CPU-initiated RDMA, the full eleven-task
  matrix** — owned by [`doca-rdma`](../doca-rdma/SKILL.md). The
  selection table at the top of [`##
  Capabilities and modes`](#capabilities-and-modes) routes the
  agent there.
- **The DPA programming model** (DPA toolchain, kernel build,
  device-side memory model) — owned by
  [`doca-dpa`](../doca-dpa/SKILL.md). RDMI returns DPA handles
  and names the device-side header set, but does not author DPA
  kernels.
- **GPU-side packet / RDMA initiation** — owned by
  [`doca-gpi`](../doca-gpi/SKILL.md) and
  [`doca-gpunetio`](../doca-gpunetio/SKILL.md). RDMI is the
  DPA-side counterpart; GPI/GPUNetIO is the GPU-side
  counterpart.
- **Cross-library `DOCA_ERROR_*` taxonomy** — owned by
  [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
  This skill adds the RDMI overlay, not the taxonomy itself.
- **Cross-library capability-snapshot tooling** — owned by
  [`doca-caps`](../../tools/doca-caps/SKILL.md). This skill
  references the tool; it does not redefine its invocation
  patterns.
