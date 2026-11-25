# Acquisition Failure Diagnosis

Start with the exact error, emitting engine, relevant config and saved output. Distinguish an observed symptom from a root cause. For API semantics use `docs/api-reference/cpp.md` (RX freeing and error codes), `docs/concepts.md` (Zero-Copy Ownership) and the selected engine's source when needed.

For raw ibverbs `ibv_create_qp (RAW_PACKET)` startup failures, read [initialization limits](ibverbs-init-limits.md). A valid YAML can still exceed provider/SQ allocation limits; distinguish this stage from packet delivery or buffer recycling.

| Observation | What to establish next |
|---|---|
| `NO_FREE_BURST_BUFFERS` | Burst metadata pool occupancy / high-water mark, `rx_meta_buffers`, and metadata returns. It is not the packet-buffer pool. |
| `NO_FREE_PACKET_BUFFERS` | Affected MR/segment's available buffers, effective `num_bufs`, and packet/segment releases. Increasing metadata capacity cannot repair a packet-buffer leak. |
| App calls only `free_rx_burst` | That releases metadata. Packet/segment buffers need their appropriate frees; `free_all_packets_and_burst_rx` is the ordinary combined RX release. Check the actual mode before changing code. |
| Buffers used asynchronously | Keep ownership until the GPU/consumer is finished. Releasing earlier can recycle live data; holding a bounded pipeline is legitimate. Distinguish live bounded retention from monotonically growing unreleased ownership. |
| App receives zero packets | RX-only configs can legitimately wait for external traffic. Check flow matching/queue targets, intended sender and path before declaring a build or hardware failure. |
| Monitor exits zero with an error or no samples | Delivery and measurement have different prerequisites. Check dependencies inside the execution context (`mlnx_perf` needs `ethtool`), and mark throughput unmeasured. |

“Retains several bursts” alone does not prove exhaustion. The effective pool can differ from YAML defaults or startup requests. Compare lifetime, acquisition/release counts and high-water occupancy against the effective per-pool capacity; include buffers posted to hardware and held internally. Do not infer a particular pool or cause from a generic “no-free-buffer” line.

Before recommending larger pools, identify which pool is exhausted and audit ownership on normal, error, filtered and shutdown paths. Prior successful packets do not rule out a partial leak. If evidence is insufficient, propose one discriminating check rather than asserting a fix.

For packet gaps, counters, timing, memory or path claims, use [evidence](evidence.md). Do not start traffic or change host settings for a saved-output discussion.
