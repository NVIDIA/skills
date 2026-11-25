# Raw ibverbs initialization limits

Use for constrained configuration repair, MTU-preserving prelaunch checks, or raw TX QP startup failures. Reuse supplied runtime, topology and diagnostics. The behavior below is verified in public source revision `2c315f1`; confirm the matching checkout's functions in `src/engines/ibverbs/daqiri_ibverbs_engine.cpp`. Diagnostic builds can shift line numbers. Hardware-free semantic validation does not allocate a QP or exercise netdev changes.

## Preserve the selected MTU

Inspect `ensure_port_mtus`. Initialization sizes the possible frame from configured queue capacity, even when the benchmark sends smaller packets:

- For each queue, sum the referenced MRs' configured `buf_size` values. Use neither allocation stride nor payload size alone.
- Take the largest queue sum separately for RX and TX. Add `flow_max_decap_wire_overhead` for RX and `flow_max_encap_wire_overhead` for TX, then take the larger direction.
- In this revision, a required frame at most 1514 bytes skips adjustment. Otherwise the requested netdev MTU is `required_frame - 14`; the engine raises a smaller current MTU and never lowers it. Shutdown does not automatically restore it.

Project this for every selected interface in each role, including queues sharing that interface. Compare against the captured current MTU. A small payload, successful validation, or eventual restoration does not prove preservation at initialization.

Where capacity changes are authorized, right-size only the necessary MR `buf_size` values. Retain capacity for each actual packet segment, header and applicable transform geometry; check other consumers of a shared MR. Preserve unrelated pool counts, queues, roles and application fields. Leave already-safe capacities unchanged. If packet requirements cannot fit the preserved MTU within the permitted edits, explain that concrete conflict. Do not change networking to evade it.

## Diagnose TX QP allocation

Locate the failing stage. Successful MR allocation/registration followed by `ibv_create_qp (RAW_PACKET)` failure is a QP startup failure; it does not establish GPU-memory exhaustion, packet loss or a driver defect.

Trace `setup_tx_queue` into `create_tx_raw_qp`:

- Cyclic TX initially takes packet slots from the smallest referenced MR `num_bufs`. The device WR check caps slots at `max_qp_wr / 2`, then requests twice the effective slots as send WRs, reserving SEND/WAIT capacity. A smaller batch does not by itself reduce these slot/WR requests. Shared TX derives credits separately; follow that branch when selected.
- The provider also sizes SQ descriptors for requested SGEs and inline capacity. Inspect `cpu_inline_enabled`, `MLX5_CPU_INLINE_MAX_FRAME` and `gather_wqe_ds`: one-region direct HOST_PINNED/HUGE queues reserve the small-frame inline ceiling; other paths reserve header/gather capacity. Actual small packets do not necessarily shrink that initialization request.

An advertised `max_qp_wr` is insufficient to prove the combined request fits. Compare supplied provider-probe rows for the same provider, QP type, inline and SGE geometry. Successful creation at one inline size does not establish support at another. Use successful matching geometry to choose a justified lower TX capacity, deriving its slot/WR request rather than importing a buffer-count limit from another platform. Preserve RX resources, polling mode, packet shape and unrelated fields. Check shared-MR consumers before changing a TX pool.

Render fresh outputs with the existing generator, inspect the semantic diff and run the matching native validator. Report those results separately from native initialization and delivery. When authorized, a bounded same-runtime run can check effective SQ geometry, startup and counts; a read-only task stops at the validated proposal and its evidence limits.
