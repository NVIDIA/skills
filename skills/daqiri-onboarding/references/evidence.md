# Evidence and Delivery Claims

For saved evidence, interpret that run. Do not search the machine for validators, inspect live hardware, or attempt fresh validation unless the user requests it. A saved validation result whose config identity is unknown remains a stated limit; revalidating a copy would not establish what happened historically. Read the matching benchmark guide only for unresolved semantics.

Use supplied structured count/counter observations directly when they answer the question. Read CSV/JSON evidence in its own schema and keep host, port, interval and counter levels separate. Do not run a text-log helper on these files.

For `TX/RX complete` benchmark logs or `INTEGRITY_RESULT` finite-probe logs that need reconciliation, run once on the explicit log or single run directory:

```bash
python3 <skill-directory>/scripts/summarize_raw_run.py <evidence-directory-or-bench.log>
```

Native RX summaries may count reordered aggregates. For scoped ordinary raw
summaries, supply the matching configuration with `--config` to establish
source-packet units. Unscoped or reordered summaries retain unknown units and
deficits. Finite `INTEGRITY_RESULT` probes report source packets.

The helper detects the log format and can select `bench.log` or a single `tx.log`/`rx.log` pair directly inside a run directory. It does not combine nested runs. It reconciles app summaries, keeps engine snapshots separate, preserves before/end/delta values for selected NIC counters, flags monitor failures and optionally summarizes `config.used.yaml`. Explicit `--before`, `--after`, `--monitor` and `--config` flags select auxiliary files; nested numeric JSON counter snapshots retain their scopes. Read only the additional sections needed to resolve a remaining question.

Exit 0 means recognized and parsed, not benchmark success; exit 2 means partial, ambiguous or invalid input; exit 3 means unsupported format or no unambiguous log selection. For exit 3, read the named structured evidence directly or select the known run/log once; do not retry guessed paths or feed every file through the helper. Missing optional scopes are explicit, never zero-filled. Config parsing is not native validation. Reported finite-probe fields require their probe source/capture scope for interpretation; they do not certify another run. Counter deltas do not prove provenance or payload identity. A positive before/end value with delta zero means no new events, not an absolute zero counter. Monitor rows still require qualified stable intervals before a throughput claim.

Interpret the boundaries in order:

1. `TX complete` counts reported transmissions; `RX complete` counts packets consumed by the benchmark application. Compare their totals without adding per-queue and `queues=all` rows twice.
2. `ibverbs RX ... packets=` printed during teardown and the later `RX worker stopped ... data=` are different snapshots. Keep their times/scopes separate. `cqe_seen` includes completion events; use `data` for data-packet accounting, excluding fillers and errors. A worker reaching TX's count does not close an application deficit.
3. If app RX is below TX, report the exact deficit. A shutdown tail, queue backlog, a drop, or another ownership problem requires additional evidence to distinguish. Zero selected drop counters only means no increments in those counters. Avoid absolute claims such as “nothing was dropped,” “all packets delivered,” or “not lost” while the deficit is unresolved.
4. Even equal app counts do not establish payload identity or absence of compensating loss/duplicates. Sequence/payload checks and a defined drain interval are needed for stronger delivery claims.

Positive `app_ring_full_drops` or `handoff_full_drops` are reported software discards, even with zero NIC drop counters. Report this direct evidence before considering a shutdown-tail explanation for any residual gap. For hardware reorder, distinguish packets, placements, completed/timed-out/dropped batches and application output units; do not infer their equivalence from similar labels.

For a coherent same-scope capture, compare the software-discard count with the application deficit and allow conditional subtraction as event accounting. This does not identify unique missing sequence IDs, locate active-run versus shutdown timing, or prove that the remainder is an observed tail. A bounded drain cannot recover discarded events, so do not wait indefinitely for application RX to equal worker RX when discards were reported. Use sequence/drop correlation for identity and a defined drain boundary for accounting.

| Path / claim | Supporting evidence and limits |
|---|---|
| Cabled raw | Compare sender TX PHY with receiver RX PHY for the same interval and test flow; establish topology. RX confirms TX; do not add both rates. |
| Hardware loopback | `vport_loopback_*` is the relevant path. PHY may include background packets. It does not prove a physical-cable path. |
| TCP/UDP | A connected socket or active physical link does not prove cable traversal. Inspect route / namespace and peer delivery. For UDP also inspect kernel errors and app count gaps; for TCP inspect retransmission/backpressure evidence. |
| RoCE | Require nonzero completions and examine RDMA-CM, CQ and retry errors. Completions can represent messages spanning multiple wire packets. |
| Throughput | For raw, use documented stable one-second `mlnx_perf` samples from at least ten seconds; omit startup/shutdown and count aliases once. Missing dependencies or no valid samples invalidate the measurement even if the monitor exits zero. Do not divide app counts by process wall time. |
| GPU path | Inspect actual memory kind and platform, not an MR's name. GB10 `host_pinned` is GPU-accessible shared system memory; it does not establish discrete-GPU device-memory registration or GPU computation. |

For a customer update with a hard 250-word limit, draft about 180 words covering outcome, decisive evidence/remaining gap and next check. Count the actual final draft using a local word counter, then emit that checked draft; do not add an unchecked inventory afterward. Omit unrelated startup/untested-feature lists. A useful smoke test can pass while losslessness and performance remain unverified.
