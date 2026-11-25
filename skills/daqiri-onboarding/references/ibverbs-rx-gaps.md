# Raw ibverbs / DevX RX Gaps

Read this when worker RX remains below TX after a controlled drain, a partial tail persists, or clean interface counters seem to contradict missing data. The observations below were verified on IGX indirect raw/striding-RQ and hardware-reorder tests. Establish the selected engine, queue mode, memory kind and matching source before applying them elsewhere.

## Separate boundaries before assigning a cause

Compare accepted application TX, engine TX posted/completed, NIC ingress, RX-worker data/placements, published bursts/windows and application consumption. Keep snapshots and units separate. Application TX-thread exit is not final hardware TX completion; retain final TX-completion and NIC-ingress evidence when validating the RX drain. An application-only remainder and a TX-to-worker deficit can have different causes.

- For indirect raw RX, `timeout_us: 0` can retain a final incomplete burst until worker shutdown. The application may already have stopped by the time that burst is flushed. A longer application drain alone does not force full-batch-only publication. A positive RX queue timeout released that remainder in the verified profile. Consult `docs/api-reference/configuration.md` and `rx_poll_queue` / `rx_flush_burst` in `src/engines/ibverbs/daqiri_ibverbs_engine.cpp`; do not apply this timeout to direct mode, where it is forbidden.
- A TX-to-worker deficit remains unexplained by that application publication tail. Possible boundaries include steering, receive-credit starvation, CQ progress and unprocessed work. Do not assert a specific cause from the gap alone.
- Zero selected `ethtool` drop/out-of-buffer counters do not establish zero loss on DAQIRI's custom DevX RQ. Verified tests exposed nonzero private RQ out-of-buffer deltas while interface counters and CQ errors stayed zero. If the build lacks this telemetry, state that missing coverage; do not invent a native flag. A matching instrumented diagnostic build may be needed when deeper debugging is authorized.

## Counter lifetime matters

For private queue counters, capture a baseline after this RQ is configured and an end reading in the same owned lifetime. Reused counter IDs can carry previous counts. Do not treat a raw absolute reading as a per-run delta or rely on unverified clear/reset semantics. Failed or unsupported private-counter queries are unavailable evidence, never zero drops. Validate layout, units, wrap/reset and ownership before reconciling a deficit. Even exact event-count reconciliation is not sequence or payload verification.

## Test a targeted mitigation

For a first smoke result, choose a bounded rate and verify delivery and drop evidence. Preserve the user's requested rate when measuring performance. Co-size packet capacity, metadata capacity and batching against actual occupancy and consumer lifetimes; raising one pool can expose exhaustion in another. Verify actual engine/application/helper affinities. A tested deeper profile is evidence for that workload, not a universal losslessness guarantee.

Inspect `TokenBucketPacer::wait_for_bytes` in matching `examples/raw_bench_common.cpp` when nominal pacing still produces drops. A cumulative byte/time schedule can catch up after GPU warmup or idle and emit a short burst. A no-catch-up pacing experiment passed an idle regression, but its native offered-load/rate characterization was unrun; do not treat that experiment as a validated production fix.

For hardware reorder, `missing_packets=0` on published batches and zero timed-out/dropped batches do not cover input lost before placement or entirely absent windows. Require accepted-input versus worker/placement/application accounting and applicable private-RQ deltas. Verify partial/residual window semantics independently; do not transfer an indirect-burst timeout explanation to every reorder gap.
