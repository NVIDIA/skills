# DOCA UROM workflows

**Where to start:** The verbs run `configure → build → modify →
run → test → debug`. Skip ahead only when the user is already
past a verb. The `## test` verb is an iterative loop
(single-pair smoke → multi-op smoke → small collective → full
HPC pattern → loop back if a precondition changes), not a
one-shot pass — see the eval-loop overlay in `## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the UROM capability surface, the
paired-contract model, the Service + Worker context model, the
enqueue-side operation surface, the capability-query rule, the
env-precondition matrix, the error taxonomy, the observability
surface, and the safety policy, see
[CAPABILITIES.md](CAPABILITIES.md). For the cross-library DOCA
patterns layered under everything below (the universal Core
lifecycle, the cross-library `DOCA_ERROR_*` taxonomy, the
modify-a-shipped-sample workflow), see
[`doca-programming-guide`](../../doca-programming-guide/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through
the steps in order, verifying preconditions before recommending
the next call.

## configure

Goal: stand up a UROM Service context (and the Worker context(s)
attached to it) on the host against a
BlueField that is running the DOCA UROM Service, confirm the
underlying RDMA substrate is healthy, and put the host + DPU
pair into a state where the first remote memory operation can
be enqueued and completed.

Steps the agent should walk the user through:

1. **Confirm the DPU-side UROM Service is deployed and running
   on the target BlueField — BEFORE any host-side code change.**
   Per the env-precondition matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   this is the load-bearing precondition the agent MUST surface
   FIRST: the host library cannot offload to a service that is
   not there. Walk the user through the DPU-side service health
   check per the public *DOCA UROM Service* guide via
   [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).
   If the service is not running, this is a service-side env
   problem to fix BEFORE writing any host-side `doca_urom_*`
   code — NOT a host-side library bug. A baseline agent that
   jumps to `doca_urom_*` calls without confirming the
   DPU-side service is up has the model wrong for every
   version of UROM.
2. **Confirm the host-side DOCA install and that the BlueField
   is enumerable as a `doca_dev`.** Use the procedure in
   [`doca-setup CAPABILITIES.md ## Version compatibility`](../../doca-setup/CAPABILITIES.md#version-compatibility).
   Quote the version observed (`pkg-config --modversion
   doca-urom`, then `doca_caps --version`); do not assume
   "latest". Cross-check against the host library + DPU service
   version-pairing rule in
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)
   per the [DOCA Compatibility Policy](https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html).
   Disagreement = a paired-version gap to fix BEFORE proceeding,
   not a code workaround opportunity.
3. **Confirm the underlying RDMA transport substrate is
   healthy.** UROM rides on top of `doca-rdma` (or the
   equivalent RDMA / RoCE / IB transport the BlueField is
   configured for). Walk the user through the RDMA substrate
   bring-up per
   [`doca-rdma TASKS.md ## configure`](../doca-rdma/TASKS.md#configure)
   — at minimum, confirm the BlueField port carrying the
   inter-node traffic is up and the underlying fabric routes
   between the nodes the user intends to communicate between.
   A UROM enqueue over a broken RDMA fabric surfaces as
   `DOCA_ERROR_IO_FAILED` and the user will incorrectly blame
   UROM.
4. **Create and start the Service, then run plugin discovery.**
   Create the Service with `doca_urom_service_create`, bind it to
   the target BlueField's `doca_dev` via
   `doca_urom_service_set_dev`, and start it with
   `doca_ctx_start(doca_urom_service_as_ctx(service))`. Then call
   `doca_urom_service_get_plugins_list(service, &plugins, &count)`
   — it requires a STARTED Service (`DOCA_ERROR_BAD_STATE`
   otherwise) — and record which plugins the DPU side supports
   (these map to operation families: puts, gets, atomic variants,
   active messages, collective primitives). UROM has NO
   `doca_urom_cap_*` devinfo capability family; the plugins list
   IS the capability surface. The matrix to compare against lives
   in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
   Quote the discovered plugins back to the user; do not assume
   from prior installs or from agent memory.
5. **Decide the operation kind to enqueue.** Per the
   operation-shape table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   pick by direction and semantics: one-sided data movement
   (put / get) vs atomic vs active message vs collective
   primitive. The pick decides the next-step code shape; do
   not pick *for* the user when their intent is ambiguous —
   ask which HPC / UCX / MPI pattern they are offloading.
6. **Create and start the Worker context(s) attached to the
   Service.** Create each Worker with `doca_urom_worker_create`,
   attach it to the Service created in step 4 via
   `doca_urom_worker_set_service`, select the discovered
   plugin(s) via `doca_urom_worker_set_plugins`, optionally set
   `doca_urom_worker_set_id` /
   `doca_urom_worker_set_max_inflight_tasks`, and start it with
   `doca_ctx_start(doca_urom_worker_as_ctx(worker))`. A host
   driving UROM toward more than one BlueField needs one Service
   (with its Workers) per target BlueField, not a *"global"* one,
   per the per-BlueField rule in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
   Both the Service and the Worker are standard DOCA Core context
   creates — the universal lifecycle from
   [`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure)
   applies.
7. **Register the host-side memory descriptors the operations
   will touch.** Local buffers must be registered through the
   host-side UROM API (and exported to the peer via the
   matching export step when one-sided remote access is
   involved, mirroring the same shape
   [`doca-rdma CAPABILITIES.md ## Safety policy`](../doca-rdma/CAPABILITIES.md#safety-policy)
   documents for the underlying RDMA substrate). The exact
   symbol names for memory registration / export are
   install-bound and must be read from the headers under
   $(pkg-config --variable=includedir doca-common) and the
   shipped samples under
   `/opt/mellanox/doca/samples/doca_urom/`.
8. **Sanity check before the first enqueue.** Confirm with
   the user: which BlueField (which `doca_dev`) the host is
   offloading to; which DPU-side UROM Service version that
   BlueField is running; which operation family the first
   enqueue will exercise; which local buffer + remote handle
   the operation will touch; how the host will observe the
   completion. If any of those are unclear, stop and ask — do
   not invent.

If any step fails with a `DOCA_ERROR_*`, route through the
error taxonomy in
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
before retrying. The first hypothesis on `DOCA_ERROR_NOT_PERMITTED`
is always *DPU-side UROM Service not deployed and running*,
not a host-OS permission problem.

## build

Goal: produce a host-side binary that links DOCA UROM against
the user's installed DOCA, using the canonical cross-library
build pattern.

The build pattern for any DOCA C / C++ consumer is **identical**
across libraries — `pkg-config` for include + link flags, meson
or CMake as the build system — and is fully documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
This skill carries only the UROM-specific overlay:

| Slot | Value for UROM | Why it matters |
| --- | --- | --- |
| `pkg-config` module name | `doca-urom` | The library's `.pc` file installed by the DOCA host packages |
| Required runtime libs | `libdoca-common`, `libdoca-urom`, plus the underlying RDMA-substrate libraries referenced transitively by `pkg-config --libs doca-urom` | UROM rides on top of the same RDMA substrate `doca-rdma` documents; the transitive `*.so` dependencies surface through `pkg-config` and must NOT be hand-edited away in the user's link line |
| Header check | `doca_urom.h` (or the matching public header set) resolvable under $(pkg-config --variable=includedir doca-common) | If `pkg-config --cflags doca-urom` resolves but the include is missing, the install is partial; route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2 |
| Companion DOCA libs | `doca-argp` for argument parsing (if the consumer uses the standard DOCA arg style); `doca-rdma` only when the consumer also directly drives the RDMA substrate (UROM does NOT require the consumer to link `doca-rdma` itself — the substrate is reached transitively) | Adding unnecessary companion libs bloats the link line and obscures real partial-install issues |
| Minimum DOCA version | Query with `pkg-config --modversion doca-urom`; never hardcode in build files | Cross-version build/runtime mixing breaks per [CAPABILITIES.md ## Version compatibility](CAPABILITIES.md#version-compatibility); the *DPU-side service version must agree* overlay is the load-bearing UROM-specific axis on top of the cross-library four-way match |
| Sample tree | `/opt/mellanox/doca/samples/doca_urom/` | The shipped C samples are the verified UROM source code on the user's install; `meson.build` files inside each sample subdirectory show the exact build wiring `pkg-config doca-urom` produces |

For non-C consumers (Rust, Go, Python), the link surface is the
same `*.so` files; the FFI wrapper layer is the language-specific
binding and is out of scope for this skill — but the slots above
are still the load-bearing inputs the wrapper needs. UCX-based
HPC stacks (OpenMPI, MPICH, custom UCX consumers) that wire
UROM in as a transport go through the same `pkg-config doca-urom`
anchor at build time.

## modify

Goal: take a shipped DOCA UROM sample as the verified starting
point and apply a minimum-diff modification to express the
user's intent.

The universal modify-a-shipped-sample workflow lives in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify).
Use it as-is. The UROM-specific overlay is the *modify-from-sample
schema fill* — the seven slots the agent must elicit from the
user before recommending any code-level edit:

| Slot | What the agent asks the user | UROM-specific consideration |
| --- | --- | --- |
| 1. Starting sample | Which sample under `/opt/mellanox/doca/samples/doca_urom/`? | Pick the closest in *operation family* (put / get vs atomic vs active message vs collective) to the user's intent. Do NOT bridge across operation families — a smaller diff is always safer than a re-architecture; if the user's intent crosses a family, that is a different starting sample |
| 2. Target BlueField + UROM Service version | Which BlueField is the host offloading to, and what version is the DPU-side UROM Service running? | Per the paired-contract version rule in [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility), confirm the host-side `pkg-config --modversion doca-urom` matches what the DPU-side service supports. If the modify changes the operation family, re-confirm support on BOTH sides |
| 3. Operation family changes | Which operation families is the user adding or removing relative to the sample? | Each added family needs its plugin confirmed present via `doca_urom_service_get_plugins_list` (and selected on the Worker via `doca_urom_worker_set_plugins`) before it can be assumed to work on this device + this DOCA + this UROM Service version; one plugin check per family added |
| 4. Memory descriptor changes | Which local buffers / remote handles change, and what is the payload size profile? | Per the descriptor validation in [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy), oversized payloads or mismatched remote handles return `DOCA_ERROR_INVALID_VALUE` at enqueue. Check the comm-channel limit set by `doca_urom_service_set_max_comm_msg_size` for the payload limit |
| 5. HPC stack integration | Is this UROM consumer standalone, or wired underneath OpenMPI / MPICH / custom UCX? | Stack integration is a separate concern — this skill teaches how to drive `doca-urom` directly; stack-side integration belongs in the upstream MPI / UCX documentation. The agent must NOT invent stack-side glue code |
| 6. Single-pair smoke retained | Does the modify keep a small smoke test (one put + one get round-trip between two nodes) before the user's real pattern? | A modify that goes straight from sample-to-full-collective skips the cheapest place to catch service-side / version / substrate gaps. Per the safety-policy smoke rule in [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy), the smoke stays in the diff |
| 7. Re-validate against capabilities | Re-run `doca_urom_service_get_plugins_list` from `## configure` step 4 against the modified configuration | Per the cross-cutting rule in [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability), the plugins-list query is the runtime authority |

The agent emits an *intent description + the seven filled
slots*; the *actual* unified diff against the sample source is
produced the way every other library skill in this bundle
handles modify — the agent walks the user through the diff
line-by-line against the sample source they read on disk, and
has the user paste back the result for validation.

The agent's anti-pattern alert: a *"clean rewrite"* from
scratch is almost always slower to first green than a
minimum-diff modify on a shipped UROM sample, and removes the
user's ability to bisect against a known-good baseline.

## run

Goal: actually execute the built binary against the user's
installed DOCA on the host, with the DPU-side UROM Service
already running on the target BlueField and at least one peer
node available to exchange remote memory operations with.

Steps the agent should walk the user through:

1. **Confirm the DPU-side UROM Service is still healthy.** A
   service that was running at configure time may have been
   stopped, restarted, or upgraded since; re-check it per the
   public *DOCA UROM Service* guide via
   [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services)
   before assuming the host can offload to it. A host-side
   `doca_urom_service_create` succeeding does NOT prove the
   service is still healthy.
2. **Confirm the peer node is reachable on the RDMA fabric.**
   UROM, like RDMA, needs at least one peer; running the binary
   on one side alone with no peer produces a misleading hang or
   `DOCA_ERROR_IO_FAILED`. Both sides must be on networks that
   route the underlying RDMA transport (IB / RoCE) to each
   other; route to
   [`doca-rdma TASKS.md ## run`](../doca-rdma/TASKS.md#run)
   for the substrate-side peer check.
3. **Capture the structured log.** Set `DOCA_LOG_LEVEL=trace`
   for the first run (see
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability)).
   This is the cheapest way to make the lifecycle, enqueue,
   and completion transitions visible on first failure.
4. **Drive the host-side `doca_pe_progress` loop in parallel
   with enqueues.** Completion events for enqueued operations
   flow through the DOCA progress engine; a host that
   enqueues and then blocks without progressing the PE will
   see no completions and conclude *"the offload is broken"*
   incorrectly. This is the cross-library *"PE not progressed"*
   failure mode applied to UROM.
5. **Observe at least one completion before scaling.**
   Confirm the first enqueue's completion event fires on the
   host-side PE; only then proceed to bulk enqueue or
   collective patterns. The infrastructure-side surfaces
   (DPU service observability + RDMA substrate counters per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability))
   are the diagnostics for *enqueue succeeded but completion
   never fires*.

For the runtime version + `LD_LIBRARY_PATH` cross-checks that
underlie *"the program built but does nothing"*, see
[`doca-version TASKS.md ## run`](../../doca-version/TASKS.md#run).

## test

Goal: prove the configured UROM deployment can actually move
data correctly between the host + BlueField pair and at least
one peer node, with the DPU offloading the operations as
intended, and that the capability and version axes were sized
right.

This is **a loop, not a one-shot pass.** Each iteration narrows
either the env-precondition set, the capability set, the
operation family, the memory descriptor shape, or the HPC
pattern complexity. The loop terminates when either (a) the
user's intended HPC / UCX / MPI pattern flows end-to-end with
the expected completions on the host PE and the expected effect
on the peer-side memory, or (b) the agent has narrowed the
failure cause to a layer outside UROM itself (DPU-side service,
RDMA substrate, version skew, MPI / UCX stack design) and
escalated to the matching skill / guide.

Iteration shape:

1. **Single-pair smoke (one put + one get round-trip).**
   Between TWO nodes, enqueue one put with a small payload
   into the peer's exported memory; verify the host-side
   completion event fires AND the peer-side memory contains
   the expected bytes. Then enqueue one get to read the same
   region back; verify both completions. This is the cheapest
   place to identify *service-side*, *version-axis*,
   *RDMA-substrate*, or *memory-descriptor* gaps before any
   HPC stack effort is wasted. Per the safety-policy smoke
   rule in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   DO NOT scale a broken smoke into a full collective pattern.
2. **Multi-operation smoke.** Loop 100 puts (or 100 atomics,
   depending on the operation family the user is targeting)
   with `doca_pe_progress()` between submits; confirm every
   completion callback fires. Catches missing-progress bugs
   and undersized enqueue queues (`DOCA_ERROR_AGAIN`).
3. **Small collective smoke (if collectives are in scope).**
   Run a small all-to-all or all-reduce across a small number
   of nodes (2-4); verify the collective completes and the
   per-node memory has the expected pattern. Catches
   service-side collective-variant gaps that the single-pair
   smoke cannot exercise.
4. **Capability negative test.** Intentionally try to enqueue
   an operation family the agent expects to be *not supported*
   on this device + DOCA + UROM Service combo (per the
   triple-axis cap-query result from
   [`## configure`](#configure) step 4) and confirm the
   reported `DOCA_ERROR_NOT_SUPPORTED` matches the discovery.
   Validates the agent's capability-discovery itself is
   correct.
5. **Full HPC pattern.** Once 1-4 are green, scale to the
   user's actual intended HPC pattern (full collective across
   the cluster, dense MPI window). At this point any failure
   is most often a stack-side design issue (MPI algorithm,
   UCX wiring) rather than a UROM bug, and the agent should
   route accordingly.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| `DOCA_ERROR_NOT_PERMITTED` on the first enqueue, `doca_dev` access otherwise fine | Host-side `doca_urom_service_create` + Worker start succeeded; first enqueue rejects | DPU-side UROM Service is most likely not deployed / not running / at a non-pairing version. Re-check the service per the public service guide BEFORE diagnosing as a host-OS permission problem |
| `DOCA_ERROR_NOT_SUPPORTED` on an operation the agent expected to work | `doca_urom_service_get_plugins_list` listed the plugin; runtime rejected anyway | The plugins list answered for the *library / discovery* axis; the DPU-side *service* axis (a different UROM Service version, or the Worker not having selected the plugin via `doca_urom_worker_set_plugins`) is the real gate. Re-narrow to the host-library + DPU-service version pair per the DOCA Compatibility Policy |
| `DOCA_ERROR_IO_FAILED` on enqueue / completion | UROM API surface error that points downward | The underlying RDMA substrate has failed; route to [`doca-rdma TASKS.md ## debug`](../doca-rdma/TASKS.md#debug) layer 5-7. UROM did not cause it; UROM exposed it |
| Enqueue returns `DOCA_SUCCESS`; completion never fires | PE not progressed, OR the DPU-side service queue is stuck, OR the algorithm body on the service side is hung | Confirm `doca_pe_progress()` is being driven in the host loop; consult the service-side observability per [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability) |
| Bulk enqueue starts returning `DOCA_ERROR_AGAIN` after N successful submits | First N submissions succeed, then `AGAIN` | The Worker's in-flight task queue is sized below the user's intended in-flight depth. Raise it via `doca_urom_worker_set_max_inflight_tasks` at configure time OR drain completions between bursts |
| Single-pair smoke passes; collective smoke fails | The operation family the collective uses (most often a specific atomic or active-message variant) is supported in single-pair shape but not in the collective primitive the user picked, OR the UROM Service version does not have that collective | Re-run the cap query for the specific collective variant; the answer may be that the user needs to upgrade the UROM Service or pick a different collective algorithm at the MPI / UCX layer |

Loop termination: stop iterating once two consecutive
iterations of the same kind don't change anything — that means
the cause is below UROM (DPU-side service, RDMA substrate,
firmware, fabric, MPI / UCX stack). Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
with the captured layer-1-through-5 evidence, the public *DOCA
UROM Service* guide cross-link for service-side investigation,
and the RDMA-substrate `dmesg` / `ibv_devinfo` snapshot.

## debug

Goal: when a `doca_urom_*` call returns a `DOCA_ERROR_*` (or
the program doesn't make forward progress), narrow the cause to
a specific layer and act on it.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver — *before* recommending
UROM-specific fixes. This skill's overlay names the
UROM-specific manifestation at layers 5 (runtime) and 6
(program):

**Layer 5 (runtime) — UROM overlay.**

- The single most common runtime failure is *DPU-side UROM
  Service not deployed and running* on the BlueField the host
  is enqueueing to. The host-side library cannot offload to a
  service that is not there; symptoms surface as
  `DOCA_ERROR_NOT_PERMITTED` at the first enqueue even though
  `doca_urom_service_create` looked fine. The agent's FIRST diagnostic
  move on this error is to walk the user to the public *DOCA
  UROM Service* guide via
  [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services)
  to confirm the service is up — NOT to dig into host-OS
  permissions or `doca_dev` access.
- `DOCA_ERROR_IO_FAILED` from UROM enqueue / completion is
  almost always an underlying RDMA substrate failure (link
  down, RoCE / IB config skew between host and BlueField or
  between BlueFields). Route to
  [`doca-rdma TASKS.md ## debug`](../doca-rdma/TASKS.md#debug)
  layer 5-7 for substrate-layer diagnosis; do NOT mask
  substrate failures with retry loops in the UROM consumer.
- *No completion event after a `DOCA_SUCCESS` enqueue* is
  almost always either a missing `doca_pe_progress()` call in
  the host loop OR a DPU-side service that has stopped
  processing enqueues. Confirm the PE is being driven before
  routing to service-side investigation.
- Confirm both sides agreed on transport type. A mixed-transport
  pair (IB on one side, RoCE on the other) surfaces as
  `DOCA_ERROR_IO_FAILED` from UROM with the real cause at the
  RDMA substrate layer; the fix is at substrate configure
  time, not in the UROM call.

**Layer 6 (program) — UROM overlay.**

- Memory-descriptor matrix: the most common UROM program-layer
  bug is a mismatched remote handle, an unregistered local
  buffer, or an oversized payload surfaced as
  `DOCA_ERROR_INVALID_VALUE`. Walk the user's local
  registration + peer export + payload-size choices against
  the discovered plugin for the operation family.
- Lifecycle order: configure → start → enqueue → progress /
  observe → stop → destroy, and Workers must be destroyed
  before the Service they attach to. Out-of-order returns
  `DOCA_ERROR_BAD_STATE` (or `DOCA_ERROR_IN_USE` when Workers
  are still attached); the most common case is enqueueing
  before the context has finished starting, or destroying the
  Service / Worker while operations are still in flight (the
  in-flight ones may leak DPU-side resources the next
  `doca_urom_service_create` has to recover from).
- Paired-version mismatch: a host-side `pkg-config --modversion
  doca-urom` upgraded without the matching DPU-side UROM
  Service upgrade (or vice versa) returns
  `DOCA_ERROR_NOT_SUPPORTED` for an operation family that
  DOES exist on one side but not the other. Cross-check both
  versions against the DOCA Compatibility Policy; route the
  fix to whichever side is out of date — host install via
  [`doca-setup`](../../doca-setup/SKILL.md), DPU service via
  the public *DOCA UROM Service* guide.

Once the layer is identified, route to the matching debug verb
on the matching skill: install / build / link / driver to
[`doca-setup ## debug`](../../doca-setup/TASKS.md#debug);
version to
[`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug);
cross-cutting runtime to
[`doca-debug ## debug`](../../doca-debug/TASKS.md#debug);
program-layer Core-context patterns to
[`doca-programming-guide TASKS.md ## debug`](../../doca-programming-guide/TASKS.md#debug);
RDMA substrate failures to
[`doca-rdma TASKS.md ## debug`](../doca-rdma/TASKS.md#debug);
DPU-side service problems via the public *DOCA UROM Service*
guide.

## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as follows
so the agent does not invent guidance:

- **install.** Installing DOCA on the host, choosing packages,
  post-install verification, `pkg-config` wiring — defer to
  [`doca-setup`](../../doca-setup/SKILL.md) and to the
  install-tree layout in
  [`doca-public-knowledge-map ## Layout of an installed DOCA package`](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
  This skill assumes DOCA is already installed on the host.
- **deploy.** Deploying the DOCA UROM Service container on the
  BlueField, multi-tenant UROM Service operation, scaling the
  service to multiple BlueFields, Kubernetes operator
  workflows for UROM-using HPC clusters — out of scope for
  this skill. The DPU-side service is a SEPARATE artifact;
  route to the public *DOCA UROM Service* guide via
  [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).
- **MPI / UCX stack integration and collective algorithm
  design.** Designing a collective algorithm, wiring UROM in
  as a UCX transport, tuning MPI internals — out of scope.
  Route to the upstream MPI / UCX documentation; this skill
  prescribes how the underlying remote memory operations
  flow through `doca-urom`, not how an MPI / UCX stack
  selects which operations to issue.
- **Substrate RDMA bring-up.** Bringing the RDMA / RoCE / IB
  transport between host and BlueField (or between
  BlueFields) up — owned by
  [`doca-rdma`](../doca-rdma/SKILL.md). This skill *uses*
  that substrate; it does not stand it up.
- **kernel-level driver install / firmware burn.** Installing
  the `mlx5_core` driver, burning new ConnectX firmware, or
  modifying `mlxconfig` parameters that need a reset — out of
  scope for this skill. Route to
  [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)
  layer 5 (driver), then to the upstream MLNX OFED / firmware
  documentation reachable through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
- **rollback.** Coordinated rollback of UROM-using HPC
  applications across multiple nodes — out of scope for Phase
  1 and reserved for a future platform skill. For a single
  in-session UROM configuration rollback, the right verb in
  this skill is destroying the context (`doca_ctx_stop` →
  `doca_ctx_destroy`) and re-running `## configure` with the
  corrected parameters; do not invent a "rollback" workflow.

## Command appendix

Every command below is **cross-cutting on DOCA UROM** — it
answers a recurring class of question that comes up in the
verbs above. The agent should treat the *class* as
load-bearing; the worked example is a single instance. Run-as
user is the unprivileged user unless noted. Rows that need elevated privileges call that out explicitly.

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
| `pkg-config --modversion doca-urom` | `## configure` step 2; `## build` minimum-version slot | What is the build-time DOCA UROM host-side library version? | A semver string matching `doca_caps --version`. Disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2). Surface this version ALONGSIDE the DPU-side UROM Service version per the paired-contract rule |
| `pkg-config --cflags --libs doca-urom` | `## build` | What include + link flags does the host-side linker need? | Trust whatever `pkg-config --cflags --libs` produces on this install. Do not hardcode either the `-I` include path or the `-l<name>` flag form — both can drift between DOCA install profiles and DOCA majors; the on-disk `.so` basenames use underscores on every release where we have ground truth, while the `.pc` package names use hyphens, and `pkg-config` is the only thing that resolves both correctly. Hand-crafted `-l` lines silently break when DOCA upgrades. |
| `doca_caps --list-devs` | `## configure` step 2; `## configure` step 4 | Which DOCA devices does the host see, and which map to BlueFields that could be running the UROM Service? | One entry per `doca_dev` with the BlueField identity and per-library capability flags. No entry for the intended BlueField = `doca_dev` enumeration is failing; route to [`doca-setup`](../../doca-setup/SKILL.md) BEFORE any UROM-layer diagnosis |
| `doca_caps --version` | `## configure` step 2; `## test` step 4 | What is the *runtime* DOCA version on this host? | A semver string matching `pkg-config --modversion doca-urom`. Disagreement = partial install per [`doca-version`](../../doca-version/SKILL.md) |
| `ls /opt/mellanox/doca/samples/doca_urom/` | `## modify` slot 1 | Which UROM samples ship in this install, and which is the closest starting point? | A list of sample subdirectories named after the operation family / HPC pattern they demonstrate; the per-sample `meson.build` shows the canonical build wiring |
| `cat /opt/mellanox/doca/applications/VERSION` | `## configure` step 2; `## debug` layer 1 | What does the install tree itself claim its version is? | A semver string matching the other two version sources |
| (route via [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services)) DPU-side UROM Service health / version check | `## configure` step 1; `## run` step 1; `## debug` layer 5 | Is the DOCA UROM Service deployed and running on the BlueField the host is offloading to, and at what version? | The public *DOCA UROM Service* guide documents the exact health-check shape; the agent's job is to NAME the existence of the service-side check and route the user there, NOT to redefine it. This is the load-bearing precondition the agent MUST verify on the FIRST `DOCA_ERROR_NOT_PERMITTED` from `doca_urom_*` |
| `dmesg \| tail -n 40` (sudo) | `## debug` layer 7 | What did the kernel / driver log around the last UROM call? | Empty or recent benign messages. Repeated mlx5 / IB errors → RDMA substrate bug; route to [`doca-rdma TASKS.md ## debug`](../doca-rdma/TASKS.md#debug) and to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) |
| `ibv_devinfo` (sudo) | `## configure` step 3; `## debug` layer 7 | What does the underlying `libibverbs` see for the BlueField port carrying the UROM offload traffic? | One device row with `state: PORT_ACTIVE` and a sane MTU. A down port = UROM cannot offload; fix at the substrate layer per [`doca-rdma`](../doca-rdma/SKILL.md) BEFORE any UROM-layer diagnosis |
| `DOCA_LOG_LEVEL=trace ./<binary>` | `## run` step 3 | What did the structured DOCA logger emit for the first failing host-side UROM call? | A trace-level line on every host-side lifecycle transition and every enqueue. Silence after `doca_ctx_start()` = either host PE not progressed OR DPU-side service queue stuck — reach for the service-side observability next |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the UROM-specific rows on top.
