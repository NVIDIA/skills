# DOCA GPUNetIO workflows

**Where to start:** The verbs run `configure → build → modify →
run → test → debug`. Skip ahead only when the user is already
past a verb. The `## test` verb is an iterative loop (single-packet
smoke → persistent-kernel burst loop → multi-GPU scale-out → loop
back if a precondition changes), not a one-shot pass — see the
eval-loop overlay in `## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the GPUNetIO capability surface, the
`doca_gpu` per-device context, the GPU-visible queue handles, the
persistent-kernel pattern, the dual capability-query rule, the
env-precondition matrix, the error taxonomy, the observability
surface, and the safety policy, see
[CAPABILITIES.md](CAPABILITIES.md). For the cross-library DOCA
patterns layered under everything below (the universal lifecycle,
the cross-library `DOCA_ERROR_*` taxonomy, the modify-a-shipped-
sample workflow), see
[`doca-programming-guide`](../../doca-programming-guide/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through
the steps in order, verifying preconditions before recommending
the next call.

## configure

Goal: stand up a `doca_gpu` context on a target CUDA device,
layer GPU-visible queue handles on top of an already-configured
doca-eth queue pair, and confirm both the DOCA side and the CUDA
side are in a state where a persistent kernel can be launched.

Steps the agent should walk the user through:

1. **Confirm the env preconditions.** Per the env-precondition
   matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy):
   `nvcc --version` returns a CUDA toolkit version listed as
   compatible with the installed DOCA release by the DOCA
   Compatibility Policy; `lsmod | grep nvidia_peermem` shows the
   module loaded (or `sudo modprobe nvidia_peermem` to load it);
   `nvidia-smi -L` enumerates the candidate GPU; the underlying
   doca-eth queues exist (route via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
   to the public DOCA Ethernet guide for the upstream verb). If
   ANY of these fails, this is an env / version problem to fix
   via [`doca-setup TASKS.md ## configure`](../../doca-setup/TASKS.md#configure)
   + [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure),
   NOT a code change in the GPUNetIO program.
2. **Bring up doca-eth FIRST.** GPUNetIO sits on top of doca-eth:
   the user must have a configured `doca_eth_rxq` (and
   `doca_eth_txq` if TX is in scope) on the target representor /
   PCIe device, with the doca-eth context at the right lifecycle
   stage for the GPUNetIO layer to attach. Per the layering rule
   in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   the agent must NOT skip ahead to `doca_gpu_*` calls if doca-eth
   is not already up — that ordering is fixed for every version
   of GPUNetIO.
3. **Run dual capability discovery against the target device.**
   Per the dual-axis rule in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes):
   on the DOCA side, run the doca-eth query
   `doca_eth_rxq_cap_is_type_supported(devinfo, ...)` from
   `doca_eth_rxq.h` (and the matching
   `doca_eth_txq_cap_is_type_supported` / `doca_eth_*_cap_get_*`
   sizing family) against the active `doca_devinfo` for the NIC —
   GPUNetIO exposes no `cap_is_supported` symbol of its own; on the CUDA
   side, run `cudaGetDeviceProperties(devOrdinal)` against the
   candidate CUDA device. Quote BOTH results back to the user.
   If either says *not supported*, that axis is the answer — do
   not proceed.
4. **Allocate the GPU payload pool and register it with DOCA.**
   The receive payload pool lives in GPU memory: allocate via
   `cudaMalloc` (or the CUDA allocator the user's app already
   uses) on the target device, then register the buffer with
   DOCA via the `doca_buf_arr_create_*` family BEFORE
   `doca_ctx_start()`. Per the safety policy in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   a missing or out-of-order registration surfaces as
   `DOCA_ERROR_BAD_STATE` from the first device-side submit, not
   at create time.
5. **Create the `doca_gpu` context and the GPU-visible queue
   handles.** Create one `doca_gpu` for the target GPU —
   `doca_gpu_create(const char *gpu_bus_id, ...)` takes the GPU's
   PCIe bus-id STRING (e.g. from `nvidia-smi -L` / `cudaDeviceGetPCIBusId`),
   not an integer device ordinal; then create the GPU-visible RX handle
   (`doca_gpu_eth_rxq`) on top of the existing `doca_eth_rxq`
   (and, if TX is in scope, the matching `doca_gpu_eth_txq`).
   Start the doca-eth context if it has not been started already.
   Per the lifecycle ordering in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   record the order so teardown happens in reverse: GPU-visible
   handle → `doca_gpu` → CUDA context → doca-eth queue.
6. **Sanity check before launching any CUDA kernel.** Confirm
   with the user: which CUDA device ordinal the persistent kernel
   will run on; which doca-eth queue the GPU-visible handle is
   layered on; whether RX-only, TX-only, or both; how the
   persistent kernel will know when to exit (a host-side flag
   read from device memory is the canonical signal). If any of
   those are unclear, stop and ask — do not invent.

For the canonical DOCA universal lifecycle that underlies steps
2-5, see
[`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure).
This skill adds the GPUNetIO overlay; do not re-explain the
lifecycle here.

## build

Goal: compile a GPUNetIO-using consumer (host-side C/C++ + at
least one CUDA `.cu` translation unit) against the user's
installed DOCA + CUDA toolkit, with `pkg-config` + `nvcc` as the
joint sources of truth for include + link flags.

The build pattern for any DOCA C/C++ consumer is fully documented
in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
GPUNetIO adds a CUDA layer on top: the device-side code lives in
`.cu` files compiled by `nvcc`, and the link line pulls in both
the DOCA libraries and the CUDA runtime. This skill carries only
the GPUNetIO-specific overlay:

| Slot | Value | Why it matters |
| --- | --- | --- |
| `pkg-config` module name | `doca-gpunetio` | The library's `.pc` file installed by the DOCA host packages |
| Include flags | `pkg-config --cflags doca-gpunetio` + `nvcc --include-path` for CUDA | Resolves DOCA headers under whichever include directory `pkg-config --cflags` reports on this install (do not hardcode the include path — it can move across DOCA install profiles) and the CUDA runtime headers from the CUDA toolkit install |
| Link flags | `pkg-config --libs doca-gpunetio` plus the CUDA runtime link line (`-lcudart`) | Pulls in whatever `pkg-config --libs` resolves on this install (do not predict the `-l<name>` form by hand — `.so` basenames use underscores, `.pc` names use hyphens, and `pkg-config` is the only correct translator) and the CUDA runtime |
| Companion DOCA libs | `doca-eth` (mandatory; GPUNetIO is layered on top); `doca-argp` for arg parsing in samples | Adding `doca-eth` to the build is mandatory because every shipped GPUNetIO sample includes the underlying Ethernet queue setup |
| CUDA compile rule | `.cu` translation units compiled by `nvcc`; host-side C/C++ compiled by the system compiler; linked together | A common partial-build failure is compiling everything with the system C compiler — the `.cu` file silently degrades to host-only code and the device-side primitives never execute |
| Minimum DOCA version | Query with `pkg-config --modversion doca-gpunetio`; never hardcode | Cross-version build/runtime mixing breaks per [CAPABILITIES.md ## Version compatibility](CAPABILITIES.md#version-compatibility) |
| Minimum CUDA version | Query with `nvcc --version`; cross-check against the DOCA Compatibility Policy linked from [CAPABILITIES.md ## Version compatibility](CAPABILITIES.md#version-compatibility) | Mismatched CUDA + DOCA combos fail at link time or runtime with `DOCA_ERROR_DRIVER` |

For non-C host-side consumers (Rust, Go, Python) that drive
GPUNetIO setup and launch a CUDA kernel built separately, the
host-side link line and version rules above still apply; the CUDA
kernel itself is a separate compilation unit and is out of scope
for this skill, but the `nvcc` version check is the load-bearing
input the wrapper still needs.

## modify

Goal: take the closest-fitting shipped DOCA GPUNetIO sample as
the verified starting point and apply a **minimum diff** to make
it match the user's intent, without rewriting from scratch.

The universal modify-a-shipped-sample workflow is in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify);
this skill provides the GPUNetIO-specific slot fill.

| Slot | What the agent asks the user | GPUNetIO-specific consideration |
| --- | --- | --- |
| 1. Starting sample | Which sample under `/opt/mellanox/doca/samples/doca_gpunetio/`? Or the GPU Packet Processing reference application? | Match the user's intent (RX-only / TX-only / both; single-flow / multi-flow) to a sample whose code shape already matches. Do NOT bridge across the persistent-kernel boundary — a sample that uses a persistent kernel is the right base for any persistent-kernel app |
| 2. Persistent kernel body | Which packet-processing logic goes inside the persistent kernel? | The persistent-kernel body is the in-place edit point. The agent's anti-pattern alert: do NOT propose launching one CUDA kernel per packet — that defeats GPUNetIO entirely per [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) |
| 3. CUDA device ordinal | Which GPU is the persistent kernel pinned to? Single-GPU or multi-GPU? | Per the per-GPU rule in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes), multi-GPU means one `doca_gpu` per device AND one persistent kernel per device — that is a re-architecture, not a tweak |
| 4. RX-only vs TX-only vs both | Which directions does the persistent kernel need? | Each direction needs its own GPU-visible queue handle and its own underlying doca-eth queue; adding TX after RX is up means re-running cap-query for the TX side |
| 5. Buffer pool sizing | How big is the GPU payload pool that backs the RX queue? | Per the env-precondition matrix in [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy), the pool is allocated via `cudaMalloc` and registered with DOCA via `doca_buf_arr_create_*` BEFORE `doca_ctx_start()`; resizing means re-running the registration sequence |
| 6. Termination signal | How does the persistent kernel know when to exit? | A host-side flag read from device memory is the canonical signal; without it the kernel runs forever and `doca_ctx_stop` blocks. The agent must surface this on every modify pass |

The agent emits an *intent description + the six filled slots*;
the *actual* unified diff against the sample source is produced
the way every other library skill in this bundle handles modify
— the agent walks the user through the diff line-by-line against
the sample source they read on disk, and has the user paste back
the result for validation. The agent's anti-pattern alert: a
*"clean rewrite"* from scratch is almost always slower to first
green than a minimum-diff modify on a shipped GPUNetIO sample,
and removes the user's ability to bisect against a known-good
baseline.

## run

Goal: actually launch the persistent kernel against the
configured GPU-visible queue and confirm packets flow on the
wire.

Steps the agent should walk the user through:

1. **Confirm the peer is generating traffic.** GPUNetIO needs
   packets arriving at the doca-eth RX queue; running the
   persistent kernel against an empty queue produces a
   misleading 0% GPU util and looks like a hang. The agent must
   surface this on every first run.
2. **Launch the persistent kernel** with the GPU-visible RX (and
   optionally TX) handle passed in as a kernel argument. The
   kernel is launched once and runs for the duration of the
   data-movement session; do not re-launch per batch. The
   block / thread sizing depends on the user's GPU per
   `cudaGetDeviceProperties`.
3. **Drive the host-side `doca_pe_progress` loop** in parallel
   with the persistent kernel — the DOCA-side completions and
   the connection / lifecycle events still flow through the host
   PE even when the data plane is GPU-driven. A common
   silent-failure mode is forgetting the host-side PE drive and
   wondering why state transitions are not reported.
4. **Capture the structured log on first failure.** Set
   `DOCA_LOG_LEVEL=trace` for the first run (see
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability));
   pair it with `nvidia-smi -q -d UTILIZATION` while the kernel
   runs to confirm the GPU is actually busy. A persistent kernel
   that reports 0% util is stuck on a queue drain.

For the runtime version + `LD_LIBRARY_PATH` cross-checks that
underlie *"the program built but does nothing"*, see
[`doca-version TASKS.md ## run`](../../doca-version/TASKS.md#run).

## test

Goal: prove the configured GPUNetIO setup actually moves packets
from the wire into the persistent kernel on the user's hardware,
and that the env-precondition + capability set was sized right.

This is **a loop, not a one-shot pass.** Each iteration narrows
either the env-precondition set, the capability set, the
persistent-kernel design, or the buffer sizing. The loop
terminates when either (a) the persistent kernel processes
packets at the user's intended rate with the expected accuracy,
or (b) the agent has narrowed the failure cause to a layer
outside GPUNetIO itself (CUDA driver / doca-eth queue / NIC
firmware / network) and escalated to the matching skill.

Iteration shape:

1. **Single-packet smoke from the GPU side.** Send ONE packet
   from a peer; confirm the persistent kernel observes one RX
   completion. If yes, advance. If no — and `nvidia-smi`
   reports the kernel is running — the GPU-visible RX handle is
   not actually layered on the doca-eth queue that the peer's
   traffic is hitting; re-run [`## configure`](#configure)
   step 5.
2. **Persistent-kernel burst loop.** Send 1000 packets in a
   burst; confirm the persistent kernel processes all of them
   without dropping. Catches buffer-pool-sizing bugs:
   `DOCA_ERROR_AGAIN` from the device-side submit means the
   queue is full and the persistent kernel needs to drain
   faster, OR the buffer pool is undersized — per [`##
   modify`](#modify) slot 5.
3. **TX smoke (if TX is in scope).** Submit one TX packet from
   the persistent kernel via the GPU-visible TX handle; confirm
   it arrives at the peer. The fast-path TX submit from device
   code is independent of the RX path and can fail differently
   (different cap-query axis).
4. **Multi-GPU scale-out (if used).** Re-run steps 1-3 on each
   `doca_gpu` independently — different GPUs in the same host
   can fail the dual capability check differently (e.g. an
   Ampere card supports GPUNetIO while a Turing card in the same
   chassis does not). Per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   the per-GPU rule, each card is its own configuration.
5. **Negative test.** Intentionally point the persistent kernel
   at a CUDA device that should NOT support GPUNetIO (e.g. a
   pre-Ampere card) and confirm the dual capability discovery
   from [`## configure`](#configure) step 3 reports the
   `DOCA_ERROR_NOT_SUPPORTED` cleanly — validates the agent's
   capability-discovery itself is correct.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| `DOCA_ERROR_NOT_SUPPORTED` on a GPU we expected to work | DOCA cap-query succeeded but the device fails | The CUDA-side axis was missed; re-run `cudaGetDeviceProperties` against the device ordinal and check `nvidia_peermem` is loaded |
| `DOCA_ERROR_DRIVER` on `doca_gpu_create` | DOCA + CUDA versions are skewed | Re-run the version chain per [`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test); cross-check against the DOCA Compatibility Policy |
| Single-packet smoke passed; burst loop hangs | The persistent kernel is not draining the queue fast enough OR the buffer pool is too small | Raise the buffer pool size in [`## modify`](#modify) slot 5; or restructure the persistent kernel to drain in larger batches |
| Persistent kernel runs but reports 0% GPU util | The kernel is stuck on a queue drain or the doca-eth queue is not receiving | Confirm the peer is generating traffic per [`## run`](#run) step 1; then re-run the dual capability discovery |
| `doca_ctx_stop` blocks on teardown | The persistent kernel is still running because the termination signal was never set | Walk the host-side flag write that the kernel polls per [`## modify`](#modify) slot 6 |

Loop termination: stop iterating once two consecutive iterations
of the same kind don't change anything — that means the cause is
below GPUNetIO (CUDA driver, NIC firmware, network). Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
with the captured layer-1-through-5 evidence including BOTH the
DOCA log and the `nvidia-smi` output.

## debug

Goal: when a DOCA GPUNetIO call (or the persistent kernel)
returns a `DOCA_ERROR_*` or does not make forward progress,
narrow the cause to a specific layer and act on it.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver — *before* recommending
GPUNetIO-specific fixes. This skill's overlay names the
GPUNetIO-specific manifestation at layers 5 (runtime), 6
(program), and 7 (driver):

**Layer 5 (runtime) — GPUNetIO overlay.**

- The persistent kernel never observing a completion is
  *almost always* one of three things: the underlying doca-eth
  queue is not actually receiving traffic; the GPU-visible
  handle is layered on a different queue than the one the peer
  is hitting; or the host-side PE is not being progressed.
  Confirm the env-side preconditions per [`## configure`](#configure)
  step 1 before any code change.
- `DOCA_ERROR_AGAIN` from device-side submit is *always* a
  queue-full / drain-rate problem. Do not recommend a retry
  loop in the kernel; recommend a faster drain step in the
  persistent kernel body.
- A *"persistent kernel hung; cannot stop the program"*
  pattern is almost always a missing host-side termination flag
  per [`## modify`](#modify) slot 6.

**Layer 6 (program) — GPUNetIO overlay.**

- Lifecycle ordering: the GPU-visible handle must be destroyed
  BEFORE the underlying doca-eth queue is destroyed; the
  `doca_gpu` context must be destroyed BEFORE the CUDA context
  is unbound from the process. Out-of-order returns
  `DOCA_ERROR_BAD_STATE` per
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
- Buffer registration: a CUDA-allocated buffer that was not
  registered with DOCA via `doca_buf_arr_create_*` before
  `doca_ctx_start()` returns `DOCA_ERROR_BAD_STATE` from the
  first device-side submit, not at create time. Walk the
  registration sequence in [`## configure`](#configure) step 4.
- Kernel-per-packet anti-pattern: if the user reports
  *"throughput is much lower than expected"* and the code
  launches a kernel per packet, that is the bug. Refactor to
  the persistent-kernel pattern per
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).

**Layer 7 (driver) — GPUNetIO overlay.**

- `DOCA_ERROR_DRIVER` from `doca_gpu_*` is most often a CUDA
  driver layer reporting failure to DOCA. Capture
  `nvidia-smi -q` and `dmesg | tail`; cross-check the CUDA
  driver version against the DOCA Compatibility Policy.
- `nvidia_peermem` not loaded silently degrades GPUDirect RDMA
  and surfaces as `DOCA_ERROR_NOT_SUPPORTED` at create time.
  Route to
  [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug)
  layer 5 (driver) for the kernel-module fix.

Once the layer is identified, route to the matching debug verb
on the matching skill: install / build / link / driver to
[`doca-setup ## debug`](../../doca-setup/TASKS.md#debug); version
to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug);
cross-cutting runtime to
[`doca-debug ## debug`](../../doca-debug/TASKS.md#debug);
program-layer Core-context patterns to
[`doca-programming-guide TASKS.md ## debug`](../../doca-programming-guide/TASKS.md#debug).

**5-phase universal debug-loop instantiation (GPUNetIO).** Layer
identification above is phase 1 of the
[universal debug-loop contract](../../doca-debug/CAPABILITIES.md#universal-debug-loop-contract).
The agent MUST walk the remaining four phases on every GPUNetIO
debug answer before declaring done:

1. **Layer identification** — above (RX / lifecycle / driver).
2. **Triple capture (READ-ONLY).** Capture (a) `doca_eth_rxq`
   parent state via `doca_caps --list-devs` and
   `ethtool -S <netdev>` (rx queue counters), (b) DOCA log lines
   at `DOCA_LOG_LEVEL=DEBUG` for the offending submit / drain,
   (c) GPU-side state via `nvidia-smi -q` + `cuda-memcheck`
   (or `compute-sanitizer`) on the persistent kernel. Capture
   ALL THREE before mutating; the triple is the rollback target.
3. **Single-variable mutation SMALLER than the original
   change.** Examples: drop the kernel's per-iteration drain
   width by half (not refactor to per-packet); register one
   extra buffer (not the full pool); switch one queue to the
   `doca-eth` smoke sample (not the production topology).
   Larger mutations void the experiment.
4. **Re-capture and compare.** Re-run the triple from phase 2;
   diff against the baseline. The compare diff IS the evidence,
   not the agent's prose interpretation.
5. **Exit with named green signal OR escalate.** Green = the
   specific counter / log line / GPU state the bug was
   negating. Examples: `ethtool -S` `rx_packets` increments AND
   GPU buffer holds the bytes; no `compute-sanitizer` errors;
   stop-flag termination clean. If two consecutive iterations
   don't change anything, the cause is below GPUNetIO — escalate
   to [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
   with the captured triple.

## rollback

GPUNetIO contexts are stateful (persistent kernel + GPU buffer
registration + doca-eth queue) and the agent's failure mode is
to leave half-registered GPU buffers behind after a botched
configure → start sequence. The
[universal verification contract](../../doca-setup/CAPABILITIES.md#universal-verification-contract)
step 1 (preconditions) requires *"the rollback path is
documented"* on every change-recommending answer; this is the
GPUNetIO instantiation.

**Snapshot before mutate.** Before any change-recommending
GPUNetIO answer, capture the GPU-side allocation map (`nvidia-smi
-q -d MEMORY` + `cudaGetDeviceProperties`), the doca-eth queue
attachment map (`doca_caps --list-devs` + the eth RXQ identity),
and the persistent-kernel stop-flag location. The triple IS the
rollback target.

1. **Signal the persistent kernel to drain and stop FIRST.**
   Flip the host-side termination flag from
   [`## modify`](#modify) slot 6 *before* any context destroy.
   `cudaDeviceSynchronize()` until the kernel returns; if it
   does not return within the expected drain window, the kernel
   is hung — that is the
   [deploy-loop bridge](../../doca-setup/CAPABILITIES.md#deploy-loop-bridge--step-5-not-green-is-the-debug-loop-trigger)
   trigger, not a rollback trigger; fire the debug-loop on the
   hung-kernel symptom before continuing the rollback.
2. **Unregister GPU buffers in reverse-register order.**
   `doca_buf_arr_destroy` on every array created with
   `doca_buf_arr_create_*`. Buffers MUST be unregistered before
   `cudaFree`; the reverse order is non-negotiable per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
3. **Stop and destroy the GPUNetIO context.** `doca_ctx_stop`
   on the `doca_gpu` context, then `doca_gpu_destroy`. The
   parent doca-eth RXQ remains valid and must not be torn down
   by this step.
4. **`cudaFree` the underlying GPU allocations.** Only after the
   DOCA-side unregister landed clean. Skipping this step leaks
   GPU memory at process exit and surfaces as
   `CUDA_ERROR_LAUNCH_FAILED` on the next program run with the
   same NUMA-pinned GPU.
5. **Re-verify the doca-eth parent is intact.** Re-run the
   eth RXQ smoke from [`doca-eth TASKS.md ## test`](../doca-eth/TASKS.md#test)
   to confirm the parent queue still receives packets at the
   pre-GPUNetIO rate. If not, the GPUNetIO rollback corrupted
   the parent — that is a bug to surface, not a retry trigger.
6. **Document the rollback verb in the verification contract
   preconditions block.** The step 1 line for a GPUNetIO add
   reads: *"the rollback path is the five-step reversal in
   [`## rollback`](#rollback); the agent has captured the GPU
   allocation map and persistent-kernel stop-flag location."*
   Without that line, the contract is incomplete and the agent
   is NOT eligible to declare done.

The rollback is bounded — on the second non-green re-verify at
step 5, the agent MUST surface the unresolved residual gap
instead of recommending another GPUNetIO retry.

## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as follows
so the agent does not invent guidance:

- **install.** Installing DOCA, installing the CUDA toolkit,
  choosing matched versions, post-install verification,
  `pkg-config` wiring, loading `nvidia_peermem` — defer to
  [`doca-setup`](../../doca-setup/SKILL.md) and to the
  install-tree layout in
  [`doca-public-knowledge-map ## Layout of an installed DOCA package`](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
  This skill assumes DOCA + CUDA are already installed and
  matched.
- **deploy.** Deploying GPUNetIO-using applications at scale
  (multi-GPU multi-NIC clusters, Kubernetes operator workflows
  for GPU + DPU workloads, multi-tenant GPU sharing for
  GPU-initiated networking) — out of scope for Phase 1 and
  reserved for a future platform skill. For single-host
  first-run testing, the right verb in this skill is `## run`;
  do not invent a "deploy" workflow.
- **CUDA kernel design and tuning.** Block / thread sizing,
  warp-level optimization, shared memory layout inside the
  persistent kernel — out of scope. Route to the upstream CUDA
  toolkit documentation via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md);
  this skill prescribes that the user *uses* the persistent-kernel
  pattern, not how to tune it.
- **DOCA Ethernet queue setup.** Bringing up the underlying
  `doca_eth_rxq` / `doca_eth_txq`, RSS configuration,
  representor selection — GPUNetIO depends on it but does not
  redefine it. Route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  to the public DOCA Ethernet guide; no library skill yet ships
  for it in this bundle.

## Command appendix

Every command below is **cross-cutting on DOCA GPUNetIO** — it
answers a recurring class of question that comes up in the verbs
above. The agent should treat the *class* as load-bearing; the
worked example is a single instance. Run-as user is the
unprivileged user unless noted. Rows that need elevated privileges call that out explicitly.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST (`doca-env --json`
   for version + devices + libraries + drivers + hugepages in one
   shot; `doca-capability-snapshot` for per-device capability flags;
   `version-matrix.json` for *"available since"* lookups).
2. If the probe succeeds, the structured tool's output is the
   authoritative answer and the agent SHOULD NOT also run the
   manual command in the row below. Report *"using structured
   `<tool>`"*.
3. If the probe fails, fall back to the manual command in the
   row. Report *"falling back to manual chain"*.
4. The schemas the structured tools emit are defined in
   [`doca-structured-tools-contract ## Schemas`](../../doca-structured-tools-contract/SKILL.md#schemas);
   the version-handling semantics (four-way match, NGC,
   headers-win) are owned by
   [`doca-version`](../../doca-version/SKILL.md).

| Command (worked example) | Owning step | Class of question it answers | What healthy output looks like |
| --- | --- | --- | --- |
| `pkg-config --modversion doca-gpunetio` | `## configure` step 1; `## build` minimum-version slot | What is the build-time DOCA GPUNetIO version? | A semver string matching `doca_caps --version`. Disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2) |
| `pkg-config --cflags --libs doca-gpunetio` | `## build` | What include + link flags does the linker need? | Trust whatever `pkg-config --cflags --libs` produces on this install. Do not hardcode either the `-I` include path or the `-l<name>` flag form — both can drift between DOCA install profiles and DOCA majors; the on-disk `.so` basenames use underscores on every release where we have ground truth, while the `.pc` package names use hyphens, and `pkg-config` is the only thing that resolves both correctly. Hand-crafted `-l` lines silently break when DOCA upgrades. |
| `nvcc --version` | `## configure` step 1; `## build` minimum-CUDA slot | What is the installed CUDA toolkit version? | A release string the agent compares against the DOCA Compatibility Policy linked from [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility) |
| `nvidia-smi -L` | `## configure` step 1; `## test` step 4 | Which CUDA devices are enumerable on this host? | One row per GPU with device ordinal, name, and UUID. Empty = NVIDIA driver not loaded; route to [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug) |
| `lsmod \| grep nvidia_peermem` | `## configure` step 1; `## debug` layer 7 | Is the `nvidia_peermem` module loaded for GPUDirect RDMA? | One line showing the module loaded. Empty = `sudo modprobe nvidia_peermem` (env-side fix, not a code change) |
| `ls /opt/mellanox/doca/samples/doca_gpunetio/` | `## modify` slot 1 | Which GPUNetIO samples ship in this install, and which is the closest starting point? | A list of sample directories named after the RX / TX / persistent-kernel pattern they demonstrate |
| `nvidia-smi -q -d UTILIZATION` | `## run` step 4; `## debug` layer 5 | Is the persistent kernel actually busy on the GPU? | Non-zero GPU util while the kernel runs. 0% = stuck on a queue drain or the doca-eth queue is not receiving |
| `dmesg \| tail -n 40` (sudo) | `## debug` layer 7 | What did the kernel / driver log around the last GPUNetIO call? | Empty or recent benign messages. Repeated mlx5 / nvidia errors → driver-layer bug; route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) |
| `DOCA_LOG_LEVEL=trace ./<binary>` | `## run` step 4 | What did the structured DOCA logger emit for the first failing call? | A trace-level line on every lifecycle transition. Silence after the persistent kernel launches = host-side PE not progressed |
| `cat /usr/local/cuda/version.txt` | `## configure` step 1; `## debug` layer 2 | What does the CUDA install tree itself claim its version is? | A version string matching `nvcc --version`. Disagreement = partial CUDA install; route to [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug) |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the GPUNetIO-specific rows on top.
