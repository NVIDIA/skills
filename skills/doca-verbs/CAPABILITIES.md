# DOCA Verbs capabilities, version compatibility, errors, observability, safety

**Where to start:** The pattern overview below names the recurring
raw-verbs CLASS patterns. Pick the pattern first, then drill into
the H2 that owns the substance. Every section in this file rests on
ONE invariant: **`doca-verbs` is a targeted escape hatch, not a
default.** If the user's case fits the matching higher-level DOCA
library ([`doca-rdma`](../doca-rdma/SKILL.md) for RDMA work,
[`doca-eth`](../doca-eth/SKILL.md) for Ethernet queues,
[`doca-rmax`](../doca-rmax/SKILL.md) for timing-precise media),
that is the answer; the rest of this file applies only after the
drop-down decision in [SKILL.md](SKILL.md) has been made.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For step-by-step workflows that *use* these
capabilities (configure, build, modify, run, test, debug) see
[TASKS.md](TASKS.md). For where the underlying public
documentation and installed package paths live, defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
do not duplicate URLs or install paths in this file.

## Pattern overview

Every raw-verbs question this skill teaches resolves into one of
SIX patterns. The patterns are CLASSES — they apply across every
raw-verbs use case, not just the worked example shown.

| Raw-verbs pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Decide drop-down vs stay-up | The agent first re-checks whether the matching higher-level DOCA library ([`doca-rdma`](../doca-rdma/SKILL.md) / [`doca-eth`](../doca-eth/SKILL.md) / [`doca-rmax`](../doca-rmax/SKILL.md)) covers the case; only if it explicitly does not does the conversation continue here | [`## Capabilities and modes`](#capabilities-and-modes) higher-level-vs-doca-verbs table + [TASKS.md ## configure](TASKS.md#configure) step 1 |
| 2. Place the surface against libibverbs | The agent teaches that `doca-verbs` is *not* a synonym for libibverbs even though the object names rhyme; mixing handles across the boundary is unsupported | [`## Safety policy`](#safety-policy) no-mixing rule + [TASKS.md ## modify](TASKS.md#modify) porting block |
| 3. Bring the verbs context up inside DOCA Core | The agent walks `doca_dev` + `doca_pe` + `doca_verbs_context_create` + per-object PD / QP / CQ / SRQ / AH creation in DOCA Core terms (not in libibverbs poll-CQ terms) | [`## Capabilities and modes`](#capabilities-and-modes) object-model section + [TASKS.md ## configure](TASKS.md#configure) |
| 4. Capability-query the specific verb / opcode / QP feature | The user wanted raw verbs because some specific thing was missing upstairs — the agent must confirm that *specific* thing is supported here on the device, via `doca_verbs_query_device` + the matching `doca_verbs_device_attr_get_*` against the active `doca_devinfo` | [`## Capabilities and modes`](#capabilities-and-modes) cap-query rule + [TASKS.md ## configure](TASKS.md#configure) |
| 5. Observe what the HW actually did | Three valid surfaces: the DOCA progress engine (preferred default, integrates with the rest of DOCA Core), manual completion-queue handling via `doca_verbs_bridge_poll_cq`, or completion-channel event delivery via `doca_verbs_comp_channel_create` + `doca_verbs_get_cq_event` — the agent picks one explicitly | [`## Observability`](#observability) + [TASKS.md ## debug](TASKS.md#debug) |
| 6. Interpret a `DOCA_ERROR_*` from a raw-verbs call | Map the error to a layer (lifecycle / cap / permission / completion-status); the IO_FAILED case has its own overlay because the answer lives on the CQE, not on the submit return | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **Smoke-before-scale.** Always start with one QP + one WR + one
  completion before adding any second QP, second connection, or
  second WR opcode. Raw verbs amplifies the cost of a hidden
  configuration bug; a single-shot smoke isolates the cause
  cleanly. The full eval-loop overlay is in
  [TASKS.md ## test](TASKS.md#test).
- **Discover the version-installed surface, do not assume.** Every
  pattern above gates on `pkg-config --modversion doca-verbs` and
  on the `doca_verbs_query_device` + `doca_verbs_device_attr_get_*`
  capability queries against the active `doca_devinfo`. Quoting a
  verb / opcode / QP option / SRQ feature without checking is the
  most common hallucination failure mode for raw verbs.

## Capabilities and modes

DOCA Verbs is a **DOCA Core Context.** Every verbs context follows
the universal `cfg-create → cfg-set-* → init → start → use → stop →
destroy` lifecycle (see
[`doca-common CAPABILITIES.md ## ctx`](../doca-common/CAPABILITIES.md#ctx)).
On top of that lifecycle, the verbs surface layers an ibverbs-like
object model accessed through `doca_verbs_context_create` and the
per-object create / destroy families.

**Higher-level-library vs `doca-verbs` — the load-bearing selection
table.** This is the table the agent walks BEFORE any code-level
discussion.

| Axis | Higher-level DOCA library | doca-verbs (this skill) |
| --- | --- | --- |
| Default for | The vast majority of work in scope: [`doca-rdma`](../doca-rdma/SKILL.md) covers Send / Receive / Read / Write / Write-Imm / Atomic / Sync-Event RDMA tasks; [`doca-eth`](../doca-eth/SKILL.md) covers Ethernet TX / RX queue patterns; [`doca-rmax`](../doca-rmax/SKILL.md) covers timing-precise media / data-over-IP streams | The narrow case where the higher-level library's task abstractions do not expose the specific verb / opcode / WR flag / QP attribute / SRQ option / AH attribute / CC-group attachment the user needs |
| Surface shape | Task-level (`doca_<library>_task_*`) — submit a task, get a completion event | Raw verbs primitives (`doca_verbs_pd` / `_qp` / `_cq` / `_srq` / `_ah_attr` / `_comp_channel` / `_eth_sq` / `_eth_rq` / `_cc_group`) inside DOCA Core — closer to what libibverbs exposes, but inside the DOCA Core lifecycle |
| Completion handling | DOCA progress engine (`doca_pe_progress`), event per task | DOCA progress engine OR manual completion-queue handling via `doca_verbs_bridge_poll_cq` OR completion-channel event delivery via `doca_verbs_comp_channel_create` + `doca_verbs_get_cq_event`, picked explicitly by the user |
| Capability discovery | `doca_<library>_cap_*` (per-task supported, per-transport supported, per-property max) against `doca_devinfo` | `doca_verbs_query_device` returns a `doca_verbs_device_attr` handle; the `doca_verbs_device_attr_get_*` family (max QP, max QP WR, max SGE, max CQ / CQE, max MR / PD / AH / SRQ / SRQ-WR / SRQ-SGE, per-QP-type supported, max-rd-atomic, ECE / CC-group / CQE-inline / ordering-semantic / DPA-external-datapath / GPU-external-datapath / plane-index / send-DBR-mode / TS-source-type supported, etc.) is the per-attribute query family |
| Right answer for *"I want raw QP control"* | Often: the higher-level task abstractions cover the case once the user names the actual semantic | Sometimes: when the user has confirmed the higher level genuinely does not surface the option |
| Right answer for *"I want a higher-level task abstraction"* | Yes — this is the home | No — climb back up |

**Verbs object model inside DOCA Core.** The verbs primitives have
the same *conceptual* role as in libibverbs, but they are created
and torn down through the DOCA Core lifecycle, not through
libibverbs `ibv_*` calls. The per-object surface as exposed under
the experimental ABI tier on this release:

| Object | What it is | Lifecycle (the agent must read the exact symbol shape from `doca_verbs*.h` on the user's install) |
| --- | --- | --- |
| `doca_verbs_context` | The per-library context — the DOCA Core context that owns the verbs surface. Wraps a `doca_dev` and surfaces to the PE | `doca_verbs_context_create(dev, &ctx)` → use → `doca_verbs_context_destroy(ctx)`. Bridge import / export via `doca_verbs_context_export_handle` / `_import_from_handle` for cross-process / cross-runtime cases |
| `doca_verbs_pd` (Protection Domain) | The protection domain QPs, MRs, and AHs live within | `doca_verbs_pd_create(ctx, &pd)` → use → `doca_verbs_pd_destroy(pd)`. Conversion to `doca_dev` via `doca_verbs_pd_as_doca_dev`; export / import via `doca_verbs_pd_export_handle` / `_import_from_handle` |
| `doca_verbs_qp` (Queue Pair) | The QP. Created from a `doca_verbs_qp_init_attr` configured via the `doca_verbs_qp_init_attr_set_*` setters; transitioned via `doca_verbs_qp_modify` using a `doca_verbs_qp_attr` configured via the `doca_verbs_qp_attr_set_*` setters; queried via `doca_verbs_qp_query` / `_query_ece`; ECE attributes set via `doca_verbs_qp_set_ece` | `doca_verbs_qp_init_attr_create` → `_set_*` → `doca_verbs_qp_create(ctx, init_attr, &qp)` → `doca_verbs_qp_modify(qp, qp_attr)` to drive the QP state machine → use → `doca_verbs_qp_destroy(qp)` |
| `doca_verbs_cq` (Completion Queue) | The CQ that holds completion entries for the QPs feeding into it | `doca_verbs_cq_attr_create` → `_set_*` (size, overrun, collapsed, entry size, completion channel) → `doca_verbs_cq_create(ctx, cq_attr, &cq)` → use → `doca_verbs_cq_destroy(cq)`. Direct WQ / DBR / CQN accessors via `doca_verbs_cq_get_wq` / `_get_dbr_addr` / `_get_cqn` |
| `doca_verbs_srq` (Shared Receive Queue) | The shared receive queue that multiple QPs can pull receive WRs from | `doca_verbs_srq_init_attr_create` → `_set_*` → `doca_verbs_srq_create(ctx, init_attr, &srq)` → use (`doca_verbs_bridge_post_srq_recv`) → `doca_verbs_srq_destroy(srq)`. Query via `doca_verbs_srq_query`; SRQ number via `doca_verbs_srq_get_srqn` |
| `doca_verbs_ah_attr` (Address-Handle attributes) | The Address-Handle attribute block for routing send WRs over IB or RoCE (address type is one of `DOCA_VERBS_ADDR_TYPE_IPv4` / `_IPv6` / `_IB_GRH` / `_IB_NO_GRH`) | `doca_verbs_ah_attr_create` → `_set_*` (DGID / DLID / SL / SGID-index / static-rate / hop-limit / traffic-class / UDP-source-port / address-type) → attached to a `doca_verbs_qp_attr` via `doca_verbs_qp_attr_set_ah_attr` → `doca_verbs_ah_attr_destroy` |
| `doca_verbs_comp_channel` (Completion Channel) | An event-delivery channel that fires when a CQ has new completions. Alternative to PE-driven completions and to manual CQ poll | `doca_verbs_comp_channel_create(ctx, &chan)` → `doca_verbs_comp_channel_get_handle` → poll the OS handle → `doca_verbs_get_cq_event` to drain → `doca_verbs_comp_channel_destroy(chan)`. CQ ACK via `doca_verbs_ack_cq_events`; arming via `doca_verbs_req_notify_cq` |
| `doca_verbs_cc_group` (Congestion-Control Group) | A congestion-control group that QPs can be attached to | `doca_verbs_cc_group_attr_create` → `_set_hint` → `doca_verbs_cc_group_create(ctx, group_attr, &cc_group)` → attach to a QP via `doca_verbs_qp_attr_set_cc_group` → modify / query / destroy. The matching cap-query is `doca_verbs_query_cc_group_caps` |
| `doca_verbs_eth_sq` (Ethernet Send Queue) | An Ethernet-side SQ (the verbs-tier counterpart to the [`doca-eth`](../doca-eth/SKILL.md) TX queue) — only when the user has confirmed `doca-eth` does not expose the option they need (e.g., explicit TS-source-type, plane-index, multi-pkt-send-WQE) | `doca_verbs_eth_sq_init_attr_create` → `_set_*` (PD / CQ / WR-num / max-SGEs / max-inline-data / queue-id / user-index / external-datapath / DPA / TS-source-type / plane-index / flush-in-error / allow-multi-pkt-send-WQE / sig-all) → `doca_verbs_eth_sq_create(ctx, init_attr, &sq)` → use → `doca_verbs_eth_sq_destroy(sq)`. Query via `doca_verbs_eth_sq_query`; recover from error via `doca_verbs_eth_sq_recover_from_error` |
| `doca_verbs_eth_rq` (Ethernet Receive Queue) | An Ethernet-side RQ (the verbs-tier counterpart to the [`doca-eth`](../doca-eth/SKILL.md) RX queue) — same drop-down logic as `doca_verbs_eth_sq` | `doca_verbs_eth_rq_init_attr_create` → `_set_*` (PD / CQ / WR-num / max-SGEs / queue-id / user-index / external-datapath / DPA / TS-source-type / SRQ / flush-in-error) → `doca_verbs_eth_rq_create(ctx, init_attr, &rq)` → use → `doca_verbs_eth_rq_destroy(rq)` |

The exact symbol names are install-bound; the agent must read them
from the verbs headers shipped on the user's install rather than
quote them from memory. The right symbol-lookup procedure is in
[TASKS.md ## configure](TASKS.md#configure) step 2.

**Capability discovery — the only rule.** Before assuming any verb,
opcode, WR flag, or QP feature is available, run
`doca_verbs_query_device(devinfo, &dev_attr)` and inspect the
matching `doca_verbs_device_attr_get_*` for what the user wants:

- General sizing: `_get_max_qp`, `_get_max_qp_wr`, `_get_max_sge`,
  `_get_max_cq`, `_get_max_cqe`, `_get_max_mr`, `_get_max_pd`,
  `_get_max_ah`, `_get_max_srq`, `_get_max_srq_wr`,
  `_get_max_srq_sge`, `_get_max_pkeys`, `_get_max_qp_rd_atom`,
  `_get_max_qp_init_rd_atom`.
- Per-feature flags: `_get_atomic_cap`,
  `_get_is_qp_type_supported`,
  `_get_is_ece_supported`,
  `_get_is_dpa_external_datapath_supported`,
  `_get_is_gpu_external_datapath_supported`,
  `_get_is_cc_group_supported`,
  `_get_is_cqe_inline_supported`,
  `_get_is_ordering_semantic_supported`,
  `_get_is_send_dbr_mode_supported`,
  `_get_is_allow_multi_pkt_send_wqe_supported`,
  `_get_is_plane_index_supported`,
  `_get_is_eth_sq_ts_source_type_supported`,
  `_get_is_eth_rq_srq_supported`,
  `_get_is_eth_rq_ts_source_type_supported`.
- Per-CC-group: `_get_max_cc_group`,
  `_get_max_cc_group_hint_max_size`; plus the per-vendor
  `doca_verbs_query_cc_group_caps` + the
  `doca_verbs_cc_group_caps_get_*` family for the actual hint
  schema.
- Per-Ethernet-queue: `_get_max_eth_sq_wr_num`,
  `_get_max_eth_rq_wr_num`,
  `_get_ts_free_running_clock_frequency`.

Per the cross-cutting rule in
[`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability),
the cap-query is the runtime authority — the public docs are the
*promise*, the cap-query is the *reality*. **Free the attribute
handle when done** with `doca_verbs_device_attr_free` (failing to
free is a slow leak that the agent must surface in the cleanup
walk). The agent must not quote raw verbs feature support without
naming the cap query that established it.

**Configuration shape.** *Mandatory* before
`doca_ctx_start()` on a verbs context: a `doca_verbs_context`
created against a `doca_dev`; at least one `doca_verbs_pd`
attached; for QP-class work, at least one `doca_verbs_qp` created
with its matching `doca_verbs_qp_init_attr`, plus the send and
receive `doca_verbs_cq`s and the per-QP `doca_verbs_qp_attr`
transitions via `doca_verbs_qp_modify` to drive the QP state
machine (RESET → INIT → RTR → RTS for RC; analogous transitions
for UC); MRs covering any memory the QP will read or
write — all configured through the `doca_verbs_*` setters
appropriate for each object. *Optional but commonly needed*:
explicit QP-type selection (RC / UC — the only two QP types this
release defines: `DOCA_VERBS_QP_TYPE_RC` / `_UC`), queue depths,
per-WR flag selection, ECE attribute tuning, CC-group attachment,
SRQ attachment, completion-channel attachment to a CQ. Query the
active value of any setter with the matching
`doca_verbs_*_attr_get_*` call.

**Climb back up to the higher-level library.** Raw verbs is a
*targeted* surface, not a long-term home. Once the specific need
that drove the drop-down is covered (the WR flag is set; the
custom QP attribute is honored; the legacy libibverbs port is
integrated), the agent should *explicitly* ask whether the user
can move the rest of the application back up to
[`doca-rdma`](../doca-rdma/SKILL.md) /
[`doca-eth`](../doca-eth/SKILL.md) /
[`doca-rmax`](../doca-rmax/SKILL.md). The maintenance cost of raw
verbs (capability-query density, manual lifecycle management,
no-mixing rule with libibverbs) is real; staying down at the verbs
level for work the higher-level library already covers is not free.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match
rule, NGC container semantics, and the headers-win-over-docs rule,
see [`doca-version`](../../doca-version/SKILL.md). The body lives
there; this skill does not duplicate it.

**The raw-verbs-specific overlay** is:

- **`doca-verbs.pc` joins the four-way match.** On any host where
  both libraries are installed, the agent must verify
  `pkg-config --modversion doca-verbs`,
  `pkg-config --modversion doca-rdma` (when the user is using
  doca-verbs to extend an RDMA-class workload),
  `pkg-config --modversion doca-eth` (for Ethernet-class
  workloads), and `pkg-config --modversion doca-common` all match
  `doca_caps --version`. A common partial-install pattern is that
  the user upgraded `doca-verbs` independently of the rest of
  DOCA; the cap-query at runtime can then return values the
  higher-level library does not honor. Route to
  [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
  layer 2 before any verbs-layer diagnosis.
- **The DOCA Verbs surface ships under the EXPERIMENTAL ABI tier
  on this release.** Per the source-tree split observable in the
  library's `version.map` on disk, the entire `doca_verbs_*`
  symbol family is exposed under the experimental section. The
  agent should surface this: experimental-tier symbols may
  rename, reshape, or graduate between releases. Quote the symbol
  surface from the installed headers, not from the public docs;
  pin against the release the user is actually building on.
- **Use the cap-query at runtime, not at configure time alone.**
  Per the cross-cutting rule in
  [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability),
  the cap-query is the runtime authority for *"is this verb /
  opcode / WR flag / QP feature supported on this device + this
  DOCA version"*. For raw verbs the surface is wide enough that
  reading a feature off a doc page and skipping the cap-query is
  almost guaranteed to produce a runtime surprise.
- **Headers win over docs.** When the user reports *"the doc says
  this verb / opcode / flag is supported but the symbol isn't in
  my headers"*, the headers on the user's install
  ($(pkg-config --variable=includedir doca-common)) are the
  authoritative truth for what the *built* library exposes. The
  agent must not assert a symbol exists without confirming it
  there — per the headers-win-over-docs rule in
  [`doca-version`](../../doca-version/SKILL.md).

## Error taxonomy

The cross-library `DOCA_ERROR_*` taxonomy (what each family means
and which debug layer it routes to) lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
The raw-verbs-specific overlay names the families the agent will
see most often from `doca_verbs_*` calls and what they specifically
indicate:

| Family | Most common verbs cause | First action |
| --- | --- | --- |
| `DOCA_ERROR_BAD_STATE` | Lifecycle violation in verbs terms — e.g., `doca_verbs_qp_create` called before `doca_verbs_pd_create`; WR submit (`doca_verbs_bridge_post_send` / `_post_recv` / `_post_srq_recv`) called before the QP transition the verbs surface requires; MR destroyed before context destroy; `doca_verbs_qp_modify` called with an out-of-order state transition | Walk the lifecycle in [`## Capabilities and modes`](#capabilities-and-modes) object-model section; confirm each step's preconditions BEFORE retrying. The QP state machine is exposed directly via `doca_verbs_qp_modify` + the `doca_verbs_qp_attr_set_current_state` / `_set_next_state` pair; misordered transitions return `BAD_STATE` and the right response is to query the current state via `doca_verbs_qp_query` and re-walk |
| `DOCA_ERROR_NOT_SUPPORTED` | The verb / opcode / WR flag / QP attribute / SRQ option / AH attribute / CC-group hint / Ethernet-queue option the user requested is not on this device + firmware + DOCA version | Run `doca_verbs_query_device` against the active `doca_devinfo`; inspect the matching `doca_verbs_device_attr_get_*`; if the feature flag is false (or the max is zero), that is the answer — climb back up to the matching higher-level library if it covers a viable alternative |
| `DOCA_ERROR_INVALID_VALUE` | Bad WR flags / opcode / address-handle parameter / WR field that the runtime rejects at submit time; `doca_verbs_qp_modify` with mutually-exclusive attributes set; `doca_verbs_ah_attr_set_*` with an SGID index out of range; `doca_verbs_cq_attr_set_cq_size` with a size beyond the device max | Re-check the user's WR / attribute construction against the headers; do not assume the libibverbs field layout transfers — this is one of the highest-frequency raw-verbs program bugs. For sizing rejections, re-run the matching `doca_verbs_device_attr_get_*` cap-query |
| `DOCA_ERROR_NOT_PERMITTED` | Missing privileges to open the device, register the MR, or transition the QP — usually a host-side env issue (RDMA stack module loads, user group, ulimits) | Route to [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug) — the layer below the verbs API is the suspect |
| `DOCA_ERROR_IO_FAILED` | Work-request completed with error status. **The submit return is not the answer; the completion-queue entry is.** The agent MUST direct the user to inspect the CQE's error field for the specific cause | Drain the CQ (via `doca_pe_progress` if using the DOCA progress engine, or via `doca_verbs_bridge_poll_cq` if using the manual path, or via `doca_verbs_get_cq_event` if using completion-channel delivery); read the CQE error field verbatim; then map THAT to the next action |

The agent's rule: **never recommend a retry loop on `DOCA_ERROR_*`
from a verbs call without first identifying which of the rows above
is the cause**. Raw verbs amplifies the cost of "retry until it
works" — the retry can mask a configuration bug that gets worse at
scale.

Quote `doca_error_get_descr()` verbatim — do not paraphrase. The
cross-cutting debug ladder
([`doca-debug ## debug`](../../doca-debug/TASKS.md#debug)) is the
canonical layered diagnosis path that the agent escalates to once
the raw-verbs-specific cause has been narrowed.

## Observability

Raw verbs has **three valid completion-handling surfaces**, and the
agent's job is to make the choice explicit rather than let it
drift:

1. **DOCA progress engine (`doca_pe_progress`).** The same engine
   the higher-level DOCA libraries and every other DOCA Core
   context uses. Completions surface as events on the PE; the rest
   of the user's DOCA application keeps its single-PE loop. This
   is the recommended default for raw verbs unless the user has
   named a specific reason it does not fit. Per
   [`doca-common CAPABILITIES.md ## progress engine`](../doca-common/CAPABILITIES.md#progress-engine).
2. **Manual completion-queue handling.** The raw-verbs surface
   exposes manual poll via `doca_verbs_bridge_poll_cq`; the user
   can poll a CQ themselves in whatever loop they want. This is
   the right answer ONLY when the user's reason for dropping to
   raw verbs was *"I need custom CQ handling"* — otherwise the PE
   path is simpler and integrates with the rest of the user's
   DOCA application.
3. **Completion-channel event delivery.** `doca_verbs_comp_channel_create`
   yields a channel whose handle the user `poll` / `epoll`s; when
   the handle fires, the user drains with `doca_verbs_get_cq_event`
   and ACKs with `doca_verbs_ack_cq_events`. Re-arms with
   `doca_verbs_req_notify_cq`. This is the right answer when the
   user needs an event-driven loop that wakes on completion
   without busy-polling a CQ or driving the PE.

The agent must not silently mix any two of the three on the same
CQ in the same program. *"Some completions go through the PE,
others I poll manually, others come on the comp channel"* is
unsupported and a source of dropped completions.

Five primary signals the agent should reach for:

1. **Per-WR completion entries.** Whether surfaced through the PE,
   polled manually, or delivered on a comp channel, the CQE
   carries the work-request status. `DOCA_ERROR_IO_FAILED` is the
   indicator that *the submit succeeded but the completion reports
   an error* — the answer then lives in the CQE error field, not
   in the submit return.
2. **QP state transitions.** Raw verbs exposes the QP state
   machine directly via `doca_verbs_qp_modify` + the
   `_set_current_state` / `_set_next_state` pair; misordered
   transitions return `DOCA_ERROR_BAD_STATE`. The current state is
   queryable via `doca_verbs_qp_query`. A debugging session
   without confirmation of the QP's current state is blind to
   half the lifecycle.
3. **Capability snapshot at configure time.** The output of
   `doca_verbs_query_device` + the iterated
   `doca_verbs_device_attr_get_*` is a snapshot of *what the
   library said was possible* before any WR was submitted. Save
   it as the baseline; if a WR later returns
   `DOCA_ERROR_NOT_SUPPORTED` the diff against this snapshot is
   the bug. **Free the attribute handle** with
   `doca_verbs_device_attr_free` after snapshotting.
4. **Async events.** `doca_verbs_get_async_event_handle` +
   `doca_verbs_get_async_event` + `doca_verbs_ack_async_event`
   surface device-level async events (QP errors, port-state
   changes, fatal errors). Wire the handle into the user's poll
   loop the same way as the comp-channel handle when those events
   are operationally relevant.
5. **Commands in flight (RESTRICTED / trap-tier — NOT public).**
   `doca_verbs_command_get_state`, `_get_op_type`,
   `_get_user_data` give visibility into command-in-flight
   state, but they are **not part of the public verbs surface**:
   they are declared only in the restricted trap header
   `include/restricted/multi_path/doca_verbs_trap.h`, not in
   `include/public/doca_verbs.h`. Do not present them as
   generally available observability for public verbs consumers;
   they require the restricted multi-path/trap tier.

For cross-cutting observability primitives (`--sdk-log-level`, the
`doca-<lib>-trace` build flavor, the `DOCA_LOG_LEVEL` env var) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability)
and the foundation log surface in
[`doca-common CAPABILITIES.md ## log`](../doca-common/CAPABILITIES.md#log).
For the install-tree observability (logger names, package layout)
defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

Raw verbs' safety surface centers on **one rule that has no
equivalent at the higher-level DOCA-library tier**: **do not mix
libibverbs handles with `doca-verbs` handles on the same hardware
resources.** The two libraries look similar by intent — both expose
QP / CQ / PD / MR / SRQ / AH — but they are *different libraries*
operating against the same kernel uverbs interface from different
sides:

- `libibverbs` (`/usr/include/infiniband/verbs.h`) is the kernel
  uverbs userspace interface. It is NOT integrated with the DOCA
  Core lifecycle, the DOCA progress engine, the DOCA error model,
  or the DOCA device model (`doca_dev` / `doca_devinfo`).
- `doca-verbs` is the same conceptual surface BUT lives inside
  the DOCA Core model. The verbs objects it returns are managed
  through `doca_verbs_context_create` and the per-object
  `doca_verbs_*_create` / `_destroy` families; completions surface
  through `doca_pe_progress` (or via the explicit manual / comp-
  channel paths the skill documents); the errors are
  `DOCA_ERROR_*`; the device handle it consumes is `doca_dev`,
  not `ibv_context`.

Mixing the two — e.g., calling `ibv_modify_qp()` on a QP created
through `doca-verbs`, or passing a `doca_dev`-derived MR into a
libibverbs `ibv_post_send` — is unsupported. The visible symptom
can be silent (the call appears to succeed) and the program then
fails far from the line that mixed the boundary. The agent's first
response to *"can I keep my existing ibv_* code next to new
doca_verbs_* code on the same QP / CQ / PD / MR?"* must be **no**.
The right pattern is in [TASKS.md ## modify](TASKS.md#modify)
porting block: port the libibverbs code over to `doca-verbs`
handles, then run purely through one library at a time.

Additional rules:

- **Pick exactly one completion-handling path per CQ.** PE-driven
  vs manual poll vs comp-channel event delivery: pick one
  explicitly per CQ. Mixed paths drop completions silently. The
  rule lives in [`## Observability`](#observability).
- **Free the device-attribute handle.** `doca_verbs_query_device`
  returns a handle that must be freed via
  `doca_verbs_device_attr_free`. Failing to free is a slow leak
  that surfaces as memory growth across reconfiguration cycles,
  not as an immediate error.
- **Permissions and mmap-export rules inherit from `doca-common`.**
  The PD / MR / device permission envelope is the same as the
  underlying [`doca-common`](../doca-common/SKILL.md) surface; the
  agent should not re-invent them here — defer to
  [`doca-common CAPABILITIES.md ## buf`](../doca-common/CAPABILITIES.md#buf)
  and the `doca_mmap_*` permission family for the matrix.
- **Cross-link to `doca-hardware-safety`.** Any verbs-layer change
  that involves a `mlxconfig`-class firmware-stored configuration
  change (e.g., enabling/disabling a specific QP feature at the
  device level) follows the meta-policy in
  [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) —
  pre-flight inventory, OOB access, maintenance window, cold
  power cycle, replica-first, rollback documented.

## Deferred topic boundaries

This skill scopes itself to the raw-verbs surface inside DOCA
Core. Adjacent topics the agent will get asked but should route
elsewhere:

- **General DOCA RDMA work (Send / Receive / Read / Write / Atomic
  task patterns).** Owned by
  [`doca-rdma`](../doca-rdma/SKILL.md). This skill exists *only*
  for cases the higher-level library does not cover.
- **General DOCA Ethernet queue work.** Owned by
  [`doca-eth`](../doca-eth/SKILL.md). This skill's
  `doca_verbs_eth_sq_*` / `doca_verbs_eth_rq_*` surface exists
  *only* for the cases doca-eth does not expose.
- **Timing-precise media / data-over-IP streaming.** Owned by
  [`doca-rmax`](../doca-rmax/SKILL.md). Most Rivermax cases stay
  there.
- **General libibverbs programming** (raw ibverbs lifecycle, queue
  pair theory, memory-region semantics outside DOCA Core) —
  outside this skill. Route to the upstream RDMA / IB
  documentation; this skill assumes the user already understands
  the abstractions and is asking *how to express them inside the
  DOCA Core model*.
- **DOCA Core context and progress engine internals** — owned by
  [`doca-common`](../doca-common/SKILL.md). This skill *uses* the
  Core context lifecycle and the PE drive loop; it does not
  redefine them.
- **Cross-library `DOCA_ERROR_*` taxonomy** — owned by
  [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
  This skill adds the raw-verbs overlay, not the taxonomy itself.
- **Cross-cutting debug ladder** (install / version / build / link
  / runtime / program / driver) — owned by
  [`doca-debug ## debug`](../../doca-debug/TASKS.md#debug). This
  skill's `## debug` redirects there for layers 1-4 and layer 7;
  layers 5-6 carry the raw-verbs-specific overlay.
- **Cross-library `doca_caps` invocation patterns** — owned by
  [`doca-caps`](../../tools/doca-caps/SKILL.md). This skill
  references the *raw-verbs capability query family*
  (`doca_verbs_query_device` + `doca_verbs_device_attr_get_*`),
  which is per-library; the *cross-library capability snapshot
  tool* is a separate surface routed there.
