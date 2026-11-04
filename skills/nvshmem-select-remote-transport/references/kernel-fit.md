# Kernel and API Fit Analysis

Use this reference after system eligibility is known. Kernel fit can rank only candidates that
passed hard plugin, fabric, provider, prerequisite, and any applicable release-specific gates.

## Contents

- [Analysis boundary](#analysis-boundary)
- [Evidence to extract](#evidence-to-extract)
- [Proxy versus GDAKI](#proxy-versus-gdaki)
- [Scale and queue-pair pressure](#scale-and-queue-pair-pressure)
- [Decision patterns](#decision-patterns)
- [Common analysis errors](#common-analysis-errors)
- [Communication profile template](#communication-profile-template)

## Analysis Boundary

- Analyze exactly the user-selected kernel, its reachable device call graph, and directly corresponding host/on-stream calls.
- Classify only operations that can target a remote PE. Local peer loads/stores and NVLink P2P paths do not select a remote plugin.
- Do not edit code, expand to adjacent kernels, or infer application-wide behavior from one helper without call-site evidence.
- If source is unavailable, accept a user-supplied profile using the template below. Mark unverified fields explicitly.

## Evidence to Extract

### API Surface

Record each of:

- host API;
- `_on_stream` API;
- device scalar/thread API;
- warp/block/team collective API;
- custom-QP API;
- waits, barriers, fence, quiet, or flush.

IBGDA accelerates eligible device-side network operations; it does not replace the separately
selected host remote transport. GPUNetIO can provide a CPU path and optional GPU path in one
plugin. For host/on-stream-heavy applications, evaluate the CPU path directly.

### Operation Family

Record the exact type and width for:

- put/get and NBI variants;
- scalar `p`/`g`;
- put-with-signal and signal-only operations;
- standard and extended atomics;
- synchronization and completion;
- collectives and their actual backend.

Operation coverage is a hard gate, not a performance preference.

### Message Shape

Record typical, minimum, and maximum physical bytes per NVSHMEM call. Distinguish:

- logical elements from physical messages;
- application packing/aggregation from one call per element;
- one issuing lane from every thread issuing independently;
- blocking versus NBI only when it changes the transport work/completion pattern.

Do not label a kernel fine-grained merely because it processes small elements. A kernel that
packs thousands of elements into one 16 KiB put is a moderate-message workload, not millions of
8-byte network operations.

### Concurrency and Fan-Out

Record:

- active CTAs, warps, and issuing lanes per CTA;
- maximum simultaneous issuers after warp coalescing or leader-only control flow;
- remote peers per iteration;
- whether peers differ across issuers;
- whether operations serialize on a CTA barrier, global quiet, or dependency.

Estimate physical message rate only when loop counts, aggregation, and synchronization make it
possible. Otherwise use qualitative `low`, `moderate`, or `high` with supporting evidence.

### Completion and Ordering

Frequent `quiet`, fence, or target-dependent completion can dominate transport choice and increase
QP synchronization costs. Record:

- completion after every operation versus amortized completion;
- get/fetch dependencies that force immediate completion;
- source-buffer reuse requirements;
- cross-CTA or grid-wide synchronization.

### Host and CPU Conditions

Record whether:

- a CPU proxy core is available and correctly pinned;
- the kernel is persistent/long-running;
- host/on-stream calls dominate total remote bytes or operations;
- CPU oversubscription, container limits, or multiple PEs share proxy resources.

CPU constraints can strengthen a GDAKI inference, but they do not make an ineligible GDAKI path
available.

## Proxy Versus GDAKI

### Evidence Favoring a Proxy Path

- Host or on-stream APIs dominate remote communication.
- A small number of GPU lanes issue operations.
- Application data is already packed into larger transfers.
- An isolated operation's latency matters more than aggregate message rate.
- Frequent dependencies prevent parallel network submission.
- GDAKI setup, plugin, operation, or connection-scale requirements are unresolved.

IBRC, IBDevX, UCX, libfabric, and the GPUNetIO CPU path are proxy/CPU-submission candidates. Fabric
and provider gates choose among them before kernel fit.

### Evidence Favoring GDAKI

- Many warps/CTAs issue independent remote operations concurrently.
- Physical messages are small enough that a single CPU proxy is likely to serialize submission.
- The kernel is persistent or communication is integrated into GPU work.
- Peer fan-out allows multiple QPs/NICs to be used concurrently.
- Completion is sufficiently amortized to preserve parallelism.

GDAKI fit is an inference, not proof of speedup. Public guidance also notes that an individual GPU
thread creates a WQE more slowly than a CPU thread, so low-concurrency latency can favor a proxy.

### Mixed Workloads

For mixed host and device traffic, recommend a configuration rather than a single label:

- IBGDA plus an independently selected host remote transport; or
- GPUNetIO with its CPU path and optional GDAKI path.

Compare operation coverage and connection scale for both halves. Do not recommend IBGDA alone.

## Scale and Queue-Pair Pressure

### RC

RC QPs can offer low latency but consume resources per connected peer. Estimate the connection
shape from total PEs, local PEs, QPs per peer, selected NICs, and whether every PE communicates
with every other PE.

Do not invent a universal PE threshold. Mark RC scale conditional when resource limits are not
known.

### DCI/DCT

IBGDA can use DCI/DCT to reduce connection growth. This can favor IBGDA over an RC-only GDAKI path
at large scale, at the cost of connection switching and possible latency. Verify exact release
defaults and required QP types.

### QP Mapping and Contention

CTA, warp, SM, or round-robin QP mapping can materially affect high-message-rate workloads, but it
is tuning after transport selection. Record QP pressure as evidence; route exact QP settings to a
separate tuning task and keep them out of the minimal selection block.

## Decision Patterns

| Observed workload | Expected architectural fit after hard gates |
| --- | --- |
| EFA or Slingshot/CXI deployment | Native libfabric provider; kernel fit tunes expectations but does not override provider gate |
| 1-2 lanes issuing packed 64 KiB puts | Proxy-friendly; compatibility-first native provider |
| 128 CTAs issuing independent 8-64 B puts to many peers | GDAKI-favorable if physical messages are truly independent |
| 128 CTAs but application aggregates into 16 KiB buffers | Moderate-message workload; do not assume GDAKI advantage from logical count |
| Persistent device kernel with little CPU availability | GDAKI-favorable if eligible and operation coverage is complete |
| Host/on-stream RMA dominates | CPU/proxy transport; evaluate provider and host coverage |
| Large all-to-all PE scale with many RCs per peer | Consider IBGDA DCI/DCT; verify resource limits |
| NVLink-only peer-reachable job | `none` may be eligible; verify every relevant PE pair |
| Collectives only | Resolve actual collective backend before ranking point-to-point plugins |

## Common Analysis Errors

- Counting CUDA threads rather than threads that actually call NVSHMEM.
- Counting logical elements rather than physical NVSHMEM operations.
- Ignoring packing, warp coalescing, leader-only issuance, or batching.
- Assuming `_block` means every thread submits a network operation on a proxy transport.
- Treating NBI as automatically parallel when each operation is followed by immediate quiet.
- Treating single-node execution as NVLink-only without peer-access evidence.
- Assuming a transport selected for device calls also covers host/on-stream calls.
- Ranking a transport whose plugin or exact operation coverage is unknown.
- Claiming GPUNetIO is faster because a design goal says equivalent or better.
- Claiming IBDevX is faster because it uses DevX.

## Communication Profile Template

Use this when source inspection is unavailable:

```text
Application/kernel:
NVSHMEM version and prefix:
Total PEs / local PEs:
Fabric/provider:
API surface: host | on-stream | device | mixed
Operations and datatypes:
Typical/min/max bytes per NVSHMEM call:
Issuing CTAs, warps, and lanes:
Remote peers per issuer:
Application packing/aggregation:
Completion/ordering frequency:
Host versus device operation share:
Persistent kernel: yes | no | unknown
Required features: atomics | signals | VMM | registered buffers | custom QPs | other
Optimization goal: default | latency | throughput | compatibility
Unknowns:
```
