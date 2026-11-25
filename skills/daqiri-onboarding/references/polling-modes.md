# Polling Mode Selection

Read when choosing polling or investigating latency. Use supplied engine,
scheduling, ownership and reorder requirements before discovering more facts.

Raw ibverbs direct polling removes the worker/handoff path when one owning
application thread can service each queue continuously. Blocking or descheduling
that thread stops progress and recycling. Indirect workers can tolerate more
application jitter; choose from the workload rather than promise a latency gain.

A raw direct queue must omit `cpu_core` and `batch_size`; direct RX must also
omit `timeout_us`. Zero values still count as present. Preserve unrelated
routing, allocation and the selected memory path: Spark/GB10 UMA uses
`host_pinned`; IGX dGPU acquisition uses `device` packet/payload buffers.

Direct RX returns currently ready packets, up to 256; an empty poll returns
`Status::NOT_READY`. Keep polling and return RX packets/bursts through the normal
free APIs. Raw direct TX accepts exactly one packet per `BurstParams`, with one
acquired-but-unsubmitted packet per queue. Raw `send_tx_burst` consumes the burst
on `SUCCESS` and `NO_SPACE_AVAILABLE`; the latter already freed its reservation
and metadata. Other failures retain caller ownership. Submission does not prove
completion or delivery; consult the API for drain/completion handling. RDMA also supports direct queues but has a distinct
completion/buffer-ownership contract; do not apply raw TX rules to RoCE.

A direct RX queue cannot be a reorder target. If ordering is required, retain
a supported indirect RX/reorder path and address consumer scheduling and output
lifetime. Changing a mode can require dependent configuration and application
changes even when requested as one scalar edit. Native-validate the output.

For authorized measurements, use the existing `daqiri_bench_raw_latency` and
matching template/harness. Match packet size, memory, load and CPU conditions;
report p50/p99, CPU, throughput and delivery evidence. State the timing boundary.
Hardware-split intervals need the documented clock alignment; the RX hardware
timestamp is a loopback-ingress proxy, not a measured TX egress timestamp.

Sources: `docs/concepts.md` Queue Polling Modes;
`docs/api-reference/configuration.md` RX/TX Queues;
`docs/api-reference/cpp.md` direct APIs;
`docs/benchmarks/raw_benchmarking.md` Direct-polling latency sweep.
