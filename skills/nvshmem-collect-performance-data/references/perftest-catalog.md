# NVSHMEM Perftest Catalog

Use this catalog to explain the standard benchmark families and help the user select a meaningful test. Assume these normal perftests are available; verify only a selected executable when preparing its launch. Treat the selected NVSHMEM version's `.args` file and executable `--help` as authoritative for exact options.

## Selection Menu

For an unspecified request, offer these three choices and wait for an explicit selection:

1. **Default device sanity suite (recommended):** device put/get bandwidth and latency with two PEs.
2. **Custom device suite:** scalar operations, atomics, completion/signaling, device collectives, tile collectives, or TMA.
3. **Host or feature-specific suite:** host-issued/on-stream operations, CUDA graph, mmap/EGM, or allocation timing.

Do not recite every executable or option unless the user asks for the full catalog. Summarize the families first, then explain only the variants relevant to the chosen family.

## Default Device Sanity Suite

| Executable below `device/pt-to-pt` | Measures | Required scale |
| --- | --- | --- |
| `shmem_put_bw` | Device-initiated bulk put bandwidth | Two PEs |
| `shmem_put_latency` | Device-initiated bulk put completion latency | Two PEs |
| `shmem_get_bw` | Device-initiated bulk get bandwidth | Two PEs |
| `shmem_get_latency` | Device-initiated bulk get latency | Two PEs |

Use this suite for a general performance sanity package. For `shmem_put_bw`, use the first nonempty `.args` line without `--bidir` and record its full scope, CTA-count, and threads-per-CTA configuration. Keep same-node and two-node results separate.

## Device Point-to-Point Families

### Bulk RMA

- `shmem_put_bw`
- `shmem_put_latency`
- `shmem_get_bw`
- `shmem_get_latency`

Choose these for ordinary contiguous put/get curves. Relevant choices include message-size range, thread-group scope, CTA/thread count, and bidirectional put bandwidth. Do not add bidirectional mode unless requested.

### Scalar Put, Get, and Store

- `shmem_p_bw`
- `shmem_p_latency`
- `shmem_p_ping_pong_latency`
- `shmem_g_bw`
- `shmem_g_latency`
- `shmem_st_bw`

Choose `p` and `g` tests for scalar NVSHMEM operations rather than bulk RMA. Use ping-pong when round-trip synchronization behavior matters. Treat `shmem_st_bw` as a specialized device-store path; keep it distinct from bulk put bandwidth.

### Atomics

- `shmem_atomic_bw`
- `shmem_atomic_latency`
- `shmem_atomic_ping_pong_latency`
- `shmem_put_atomic_ping_pong_latency`

Ask which atomic operation matters when the request does not specify one. Standard choices can include add, bitwise operations, compare-swap, increment, set, swap, and fetch variants, subject to the selected executable's help and datatype support. Do not combine results from different atomic operations.

### Completion and Signaling

- `shmem_put_ping_pong_latency`
- `shmem_put_signal_ping_pong_latency`
- `shmem_signal_ping_pong_latency`
- `shmem_flush_bench`

Choose these to isolate round-trip completion, put-with-signal, signal-only, or flush behavior. Explain that ping-pong latency is not interchangeable with one-way put latency.

### TMA

- `shmem_put_tma_bw`

Choose this only for an explicit TMA question. Confirm that the selected NVSHMEM build, GPU architecture, CUDA toolchain, and benchmark options support the path. Keep its result separate from ordinary `shmem_put_bw`.

## Device Collectives

All executables are below `device/coll`:

- `barrier_latency`
- `sync_latency`
- `bcast_latency`
- `fcollect_latency`
- `alltoall_latency`
- `reduction_latency`
- `reducescatter_latency`

Require the user to select nodes, total PEs, PEs per node, and GPU placement. For reductions, also resolve datatype and operation when relevant. Never infer a production scale from the point-to-point default of two PEs.

## Device Tile Collectives

All executables are below `device/tile`:

- `tile_allgather_latency`
- `tile_allreduce_latency`

Use these for explicit tile-API performance questions. Resolve PE layout plus the tile-specific configuration accepted by the installed executable. Do not present them as general collective replacements.

## Host Point-to-Point Tests

All executables are below `host/pt-to-pt`:

- `bw`
- `latency`
- `stream_latency`

Use `bw` and `latency` for host-issued read/write operation measurements. Use `stream_latency` for stream-ordered operation latency. Resolve direction and issue mode from the request or the selected standard `.args` line. Do not substitute these for the four default device tests.

## Host On-Stream Collectives

All executables are below `host/coll`:

- `barrier_all_on_stream`
- `barrier_on_stream`
- `sync_all_on_stream`
- `sync_on_stream`
- `alltoall_on_stream`
- `broadcast_on_stream`
- `fcollect_on_stream`
- `reduction_on_stream`
- `reducescatter_on_stream`

Use these when the application launches collectives from the host onto a CUDA stream. Require an explicit node/PE layout. Offer CUDA graph mode only when the selected `.args` file or `--help` documents it and the user wants graph-captured behavior.

## Allocation Timing

- `host/init/malloc`

Use this for symmetric allocation timing or scaling. Require total PEs, PEs per node, allocation sizes, and placement. Do not mix allocation timing into a communication bandwidth or latency summary.

## Special Modes and Preset Suites

Explain a special mode only after its related benchmark family is selected.

| Mode | Use when | Constraint |
| --- | --- | --- |
| Bidirectional | Both PEs should issue bandwidth traffic | Keep separate from normal one-way results |
| CUDA graph | Measuring graph-captured on-stream operations | Require documented support in the selected test |
| mmap | Measuring user-buffer or mapped-memory behavior | Keep buffer type and handle mode in the report |
| EGM | Measuring supported EGM-backed mapped memory | Require an explicit EGM request and supported platform |
| Datatype | Comparing typed scalar, atomic, or collective operations | Do not aggregate unlike datatypes |
| Thread-group scope | Comparing thread, warp, or block issue scope | Record scope, CTAs, and threads per CTA |
| Atomic operation | Comparing a specific atomic primitive | Record the exact operation and datatype |

Standard preset lists can include:

- `perftest-ib.list`
- `perftest-p2p-nvlink.list`
- `perftest-p2p-pcie.list`
- `perftest-p2p-cudagraph.list`
- `perftest-mmap-sanity.list`
- `perftest-mmap-full.list`

Use a preset only when the user selects it or its stated fabric/feature matches the request. Preserve the list path and do not describe a preset as the four-test default sanity suite.

## Output Forms

Prefer machine-readable output when the selected NVSHMEM version supports it. Current statistical rows have this shape:

```text
&&&& PERF_STATS <job>___<subjob>___size__<bytes>___<metric> <sign><unit> mean=<value> stddev=<value-or-NA> min=<value> max=<value> repetitions=<count>
```

Older machine-readable rows have this shape:

```text
&&&& PERF <job>___<subjob>___size__<bytes>___<metric> <value> <sign><unit>
```

Parse the encoded job, subjob or scope, byte size, metric, unit, and values. Require consistent identity and units across a curve. Do not treat the plus/minus sign before the unit as an uncertainty value.

When neither machine-readable form appears, parse the human table by its headers rather than fixed column positions. Typical tables begin with a `# <job>` title and columns such as `size(B)`, `scope`, a metric with units, and—when repetitions are enabled—`stddev`, `min`, `max`, and `repetitions`. Preserve `NA` standard deviation for a single repetition. Reject a table when the selected metric or its units cannot be identified unambiguously.

## Quick Routing Guide

| User goal | Recommend |
| --- | --- |
| General NVSHMEM performance sanity | Default four-test device suite |
| Large-message RMA throughput | Device put/get bandwidth |
| Small-message RMA cost | Device put/get latency |
| Request/response behavior | Relevant ping-pong latency test |
| Atomic throughput or cost | Atomic bandwidth/latency with an explicit operation |
| Collective scaling | Matching device or host on-stream collective with explicit scale |
| CUDA stream or graph behavior | Host on-stream test, optionally with CUDA graph |
| TMA path validation | `shmem_put_tma_bw` with capability checks |
| Symmetric allocation overhead | `host/init/malloc` |
| NVLink versus PCIe path | Matching preset or selected device tests with explicit GPU pair |
| Inter-node fabric path | Selected tests with one or more explicit PEs per node and transport evidence |
