## use

Goal: walk the **verbs object-use sequence** — the exact sequence
of `doca_verbs_*` calls that follows a healthy
[`## configure`](../TASKS.md#configure) on the live raw-verbs context.

This verb is the *use-the-verbs-context* counterpart to
[`## configure`](../TASKS.md#configure): it names the sequence so the agent
can quote it the same way every time, whether the user is on the
RDMA-side, the Ethernet-queue side, or porting from libibverbs.

The use sequence, in order, on top of the doca-common foundation
walked in [`doca-common TASKS.md ## use`](../../doca-common/TASKS.md#use):

1. **`doca_verbs_context_create(dev, &ctx)`** — wrap the
   `doca_dev` opened in [`doca-common TASKS.md ## use`](../../doca-common/TASKS.md#use)
   step 4 in a verbs context. The `ctx` handle is the root of
   the per-library object set.
2. **`doca_verbs_pd_create(ctx, &pd)`** — create the protection
   domain. Every QP / MR / AH attached to this verbs context
   lives inside this PD.
3. **(Optional) Create the completion channel.** If the user
   picked the comp-channel completion path (per
   [`CAPABILITIES.md ## Observability`](../CAPABILITIES.md#observability)),
   call `doca_verbs_comp_channel_create(ctx, &chan)`. The channel
   must exist before the CQ that will use it.
4. **Create the CQ(s).** Build a `doca_verbs_cq_attr` via the
   `doca_verbs_cq_attr_create` + `_set_*` family (size, overrun,
   collapsed, entry size). For the comp-channel path, attach the
   already-created channel with
   `doca_verbs_cq_attr_set_comp_channel` BEFORE
   `doca_verbs_cq_create(ctx, cq_attr,
   &cq)`. One CQ per send / receive direction is the common case;
   shared send-and-receive CQs are valid when the user has named
   that as the shape.
5. **(Optional) Create the SRQ(s).** Build a
   `doca_verbs_srq_init_attr` via the `doca_verbs_srq_init_attr_create`
   + `_set_*` family, then `doca_verbs_srq_create(ctx, init_attr,
   &srq)`. Use when multiple QPs share a receive queue.
6. **(Optional) Create the CC group.** If the user is attaching
   a congestion-control group, build a
   `doca_verbs_cc_group_attr` via `doca_verbs_cc_group_attr_create`
   + `_set_hint`, then `doca_verbs_cc_group_create(ctx, group_attr,
   &cc_group)`. Attach to the QP via
   `doca_verbs_qp_attr_set_cc_group` during the QP modify.
7. **Build the QP.** `doca_verbs_qp_init_attr_create(&init_attr)`
   → call the `doca_verbs_qp_init_attr_set_*` setters to wire
   `set_pd(pd)`, `set_send_cq(send_cq)`, `set_receive_cq(receive_cq)`,
   `set_sq_wr(...)`, `set_rq_wr(...)`, `set_send_max_sges(...)`,
   `set_receive_max_sges(...)`, `set_max_inline_data(...)`,
   `set_qp_type(...)`, plus any feature-specific setter the
   cap-query allowed (`set_send_dbr_mode`, `set_ordering_semantic`,
   `set_cqe_inline`, `set_dpa`, …) → `doca_verbs_qp_create(ctx,
   init_attr, &qp)`. Destroy the `init_attr` after with
   `doca_verbs_qp_init_attr_destroy` (it holds only the init-time
   spec; the QP object owns its own state).
8. **Drive the QP state machine.** Build a `doca_verbs_qp_attr`
   via `doca_verbs_qp_attr_create` + `_set_current_state` /
   `_set_next_state` plus the per-state required setters (path
   MTU / PKey index / port num / SQ-PSN / RQ-PSN / dest QP num /
   AH attributes / atomic mode / retry / RNR / timeout for an RC
   QP; the subset varies by QP type). Apply each transition via
   `doca_verbs_qp_modify(qp, qp_attr)` until the QP is in the
   in-use state (RTS for RC). Destroy the `qp_attr` after with
   `doca_verbs_qp_attr_destroy`.
9. **Connect the PE.** `doca_pe_connect_ctx(pe, ctx_as_doca_ctx)`
   so completions on this context surface through
   `doca_pe_progress`. This step must happen BEFORE
   `doca_ctx_start`; the verbs context's `doca_ctx *` is
   accessible through the standard DOCA Core bridge accessors.
10. **`doca_ctx_start(ctx_as_doca_ctx)`** — transition the verbs
    context to RUNNING. From this point WR submission is allowed.
11. **Post WRs.** The submit family is the `doca_verbs_bridge_*`
    surface: `doca_verbs_bridge_post_send(qp, wr, ...)` for send
    WRs, `doca_verbs_bridge_post_recv(qp, wr, ...)` for receive
    WRs, `doca_verbs_bridge_post_srq_recv(srq, wr, ...)` for SRQ
    receives. The submit return is `DOCA_SUCCESS` /
    `DOCA_ERROR_*`; on success the WR is in flight and completion
    is reported on the chosen completion path.
12. **Drain completions.** PE path: `doca_pe_progress(pe)` in the
    main loop. Manual path: `doca_verbs_bridge_poll_cq(cq, ...)`
    in the main loop. Comp-channel path: `doca_verbs_req_notify_cq`
    to arm → `epoll`/`poll` on the
    `doca_verbs_comp_channel_get_handle` fd → on event,
    `doca_verbs_get_cq_event(chan, ...)` to drain →
    `doca_verbs_ack_cq_events(cq, n)` to ACK → re-arm with
    `doca_verbs_req_notify_cq`.
13. **(Optional) Wire async events.** `doca_verbs_get_async_event_handle(ctx)`
    returns an OS handle; `epoll` it and call
    `doca_verbs_get_async_event` / `doca_verbs_ack_async_event` on
    fire to surface device-level events (QP fatal, port state
    change) that do not appear on the data-plane CQ.
14. **Shutdown.** Drain inflight WRs (loop the chosen completion
    path until quiet); transition the QP through the destroy-
    appropriate state (typically ERR via
    `doca_verbs_qp_modify(qp, qp_attr_to_err)`); `doca_ctx_stop`;
    destroy in reverse order — QP → SRQ → CQ → CC group → comp
    channel → PD → `doca_verbs_context_destroy(ctx)`. Per
    [`CAPABILITIES.md ## Safety policy`](../CAPABILITIES.md#safety-policy),
    destroying a parent (PD, context) before its children (QP,
    CQ) is `DOCA_ERROR_BAD_STATE`.

The agent's rule: **the order of the selected objects is the
contract.** Optional steps 3, 5, and 6 impose ordering only when
their objects are used: a completion channel precedes its CQ, and
an SRQ or CC group precedes the QP configuration that references
it. Violating a selected object's API preconditions can return
`DOCA_ERROR_BAD_STATE`; misordering step 14 destroy-targets can do
the same. The cost of getting the order right once is much lower
than the cost of debugging a misordered destroy in production.

