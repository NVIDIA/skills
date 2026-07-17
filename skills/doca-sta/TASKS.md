# DOCA STA workflows

**Where to start:** The verbs run `configure → build → modify →
run → test → debug`. Skip ahead only when the user is already
past a verb. The `## test` verb is an iterative loop (cap-query
re-check → substrate / steering precondition re-check →
single-IO admin-then-read smoke → multi-queue smoke → loop back
if a precondition or sizing changed), not a one-shot pass — see
the eval-loop overlay in `## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the STA capability surface, the
target object model (subsystem / namespaces / backend NVMe-PCI
disks), the NVMe queue-pair shape, the RDMA-only transport, the
capability-query rules,
the error taxonomy, the observability surface, and the safety
policy, see [CAPABILITIES.md](CAPABILITIES.md). For the
cross-library DOCA patterns layered under everything below (the
universal Core lifecycle, the cross-library `DOCA_ERROR_*`
taxonomy, the modify-a-shipped-sample workflow), see
[`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
For the RDMA substrate that NVMe-over-RDMA transport lands on,
see [`doca-rdma`](../doca-rdma/SKILL.md). For the steering
side that decides which NVMe-oF packets reach STA-managed
queues, see [`doca-flow`](../doca-flow/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through
the steps in order, verifying preconditions before recommending
the next call.

## configure

Goal: stand up a `doca_sta` Core context on a BlueField, confirm
the device supports STA, define the target object model
(subsystems / namespaces / backend NVMe-PCI disks), size the NVMe
queue pair against the reported `doca_sta_get_max_*` bounds, and
confirm both the
substrate-library and the steering preconditions are met before
the NVMe-oF Connect handshake is attempted.

Steps the agent should walk the user through:

1. **Confirm the env preconditions and substrate library FIRST.**
   Per the precondition matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   walk three checks BEFORE any `doca_sta_*` call: (a) DOCA is
   installed and consistent — `pkg-config --modversion doca-sta`
   resolves and matches `doca_caps --version` per the four-way
   match rule owned by
   [`doca-version`](../../doca-version/SKILL.md); (b) for the
   RDMA transport (STA's only transport),
   `pkg-config --modversion doca-rdma`
   resolves AND `doca_rdma_cap_*` on the chosen device reports a
   non-empty surface per
   [`doca-rdma CAPABILITIES.md ## Capabilities and modes`](../doca-rdma/CAPABILITIES.md#capabilities-and-modes);
   (c) the user has a plan for how NVMe-oF traffic will reach
   the STA-managed queue — either a DOCA Flow rule programmed
   per [`doca-flow TASKS.md ## configure`](../doca-flow/TASKS.md#configure),
   or the env-side equivalent on the user's setup. If any of
   the three is missing, route to the owning skill (the env
   side to [`doca-setup TASKS.md ## configure`](../../doca-setup/TASKS.md#configure))
   first and do NOT propose a `doca_sta_*` workaround.
2. **Confirm the device supports STA.** Per the cap-check rule in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   call `doca_sta_cap_is_supported` against the active
   `doca_devinfo` and quote the result. This is the single device
   gate — there is no transport choice to make, because STA's
   only transport is NVMe-over-RDMA. If the device is not
   supported, that is the answer — this device or DOCA version
   cannot accelerate an STA target.
3. **Define the target object model.** Per the target-object
   table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   lay out the NVMe-oF target the BlueField will present:
   `doca_sta_subsystem_create()` (one NQN per subsystem),
   `doca_sta_subsystem_add_ns()` for each namespace (with a block
   size validated by `doca_sta_is_logical_block_size_supported()`),
   and `doca_sta_be_create()` for each backend controller (a local
   NVMe-PCI disk) that stores the namespace data. The remote
   initiator that connects in is out of scope.
4. **Size the NVMe queue pair against the reported maxima.** Per
   the queue-pair sizing table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   ASK the user for the intended per-connection I/O queue count
   and queue depth, then gate each input on the matching
   `doca_sta_get_max_*` query (`doca_sta_get_max_qps`,
   `doca_sta_get_max_io_queue_size`, `doca_sta_get_max_io_size`).
   Surface the
   queried ceilings before the user commits — oversizing fails
   at `doca_ctx_start()` with `DOCA_ERROR_NOT_SUPPORTED` or
   `DOCA_ERROR_INVALID_VALUE`, and undersizing leaves throughput
   on the floor. Each NVMe-oF connection always carries exactly
   one admin queue plus the configured number of I/O queues —
   that ratio is not negotiable.
5. **Create and configure the `doca_sta` context.** This is a
   standard DOCA Core context create — the universal lifecycle
   from
   [`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure)
   applies. Mandatory before `doca_ctx_start()`: the underlying
   `doca_dev` opened against a device that passes
   `doca_sta_cap_is_supported`; the devices added with
   `doca_sta_add_dev()`; the IO-context count set with
   `doca_sta_set_max_sta_io()` at or below the library maximum;
   the subsystems, namespaces, and backends from step 3 defined
   within the `doca_sta_get_max_*` bounds. Register the per-queue
   and
   per-IO completion callbacks BEFORE start — callbacks
   registered after `doca_ctx_start()` are not observed on the
   first lifecycle transitions and the agent must surface this.
6. **Sanity check before any Connect handshake.** Confirm with
   the user: that the remote initiator / host NVMe stack is a
   separate peer out of scope for doca-sta (doca-sta accelerates
   the target data path only, per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes));
   which `doca_sta_subsystem` (NQN) the initiators will connect
   to; how the user will read per-queue
   state transitions and per-IO completions on the DOCA Core
   progress engine per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability).
   If any of those are unclear, stop and ask.

If any step fails with a `DOCA_ERROR_*`, route through the STA
error taxonomy in
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
before retrying.

## build

Goal: produce a binary that links DOCA STA against the user's
installed DOCA, using the canonical cross-library build pattern,
with the substrate library (`doca-rdma`) linked alongside — STA's
transport is RDMA, so `doca-rdma` is always required.

The build pattern for any DOCA C/C++ consumer is fully
documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
This skill carries only the STA-specific overlay:

| Slot | Value for STA | Why it matters |
| --- | --- | --- |
| `pkg-config` module name | `doca-sta` for the STA surface itself; AND `doca-rdma` for the RDMA substrate (always required — STA's transport is RDMA) | Linking only `doca-sta` without `doca-rdma` produces `undefined reference` errors for the substrate symbols — surface the substrate-link requirement up front per [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy) |
| Include flags | `pkg-config --cflags doca-sta doca-rdma` | Resolves DOCA headers under whichever include directory `pkg-config --cflags` reports on this install (do not hardcode the include path — it can move across DOCA install profiles) for the STA surface and the RDMA substrate surface in one pass |
| Link flags | `pkg-config --libs doca-sta doca-rdma` | Pulls in whatever `pkg-config --libs` resolves on this install (do not predict the `-l<name>` form by hand — `.so` basenames use underscores, `.pc` names use hyphens, and `pkg-config` is the only correct translator), plus whatever the `doca-rdma` `.pc` adds (resolved by `pkg-config`, do not hand-craft); ordering is handled by `pkg-config`, do not hand-craft the `-l` list |
| Version anchors | `pkg-config --modversion doca-sta` MUST agree with `doca_caps --version`; `pkg-config --modversion doca-rdma` MUST also agree with the same line | Per the substrate-library version-match rule in [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility), a STA install that compiles against one DOCA RDMA major and runs against another is a partial-install hazard; route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2 |
| Companion DOCA libs | `doca-argp` for argument parsing (if the consumer uses the standard DOCA arg style); the cross-library `doca-common` is pulled in transitively | Adding unnecessary companion libs bloats the link line and obscures real partial-install issues |
| Initiator/host NVMe stack glue | NOT shipped by this skill | Per [`SKILL.md`](SKILL.md), the initiator / host NVMe stacks (SPDK `bdev_nvme`, kernel `nvme` host) are upstream projects out of scope; doca-sta is target-side acceleration, not an initiator transport provider |

For non-C consumers (Rust, Go, Python), the wrapper consumes
`libdoca_sta.so` (and `libdoca_rdma.so` for the RDMA substrate)
through FFI; the build-time version visibility goes through the
language's own FFI generator (e.g. `bindgen` against the
doca-sta headers, which assume the RDMA substrate headers). The
substrate rule, lifecycle
rule, and capability-discovery rules still apply per
[`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility),
and the wrapper must surface the version-pair to the user.

## modify

Goal: because DOCA STA ships **no public samples** (there is no
`/opt/mellanox/doca/samples/doca_sta/` directory and STA is absent
from the libraries / extension_libraries sample profiles), the
first-app path is to build a minimal STA target directly against
the public headers — not to modify a shipped sample. The agent
assembles the smallest correct program from the `doca_sta_*` API
and the public DOCA STA guide.

The universal modify-a-shipped-sample workflow in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify)
does NOT apply here (there is no sample to start from); instead
the agent walks these slots to scope the from-scratch minimal
target:

| Slot | What the agent asks the user | STA-specific consideration |
| --- | --- | --- |
| 1. Public-header surface | Which `doca_sta_*` symbols does this install expose? | There is no shipped sample; read the public headers at `$(pkg-config --variable=includedir doca-common) doca_sta*.h` (`doca_sta.h`, `doca_sta_subsystem.h`, `doca_sta_be.h`, `doca_sta_io_qp.h`, …) and the public DOCA STA guide as the authoritative surface. Do NOT fabricate a sample path. |
| 2. Target object model | How many subsystems / namespaces / backends does the target need? | Define each `doca_sta_subsystem` (NQN), its namespaces (`doca_sta_subsystem_add_ns()` with a `doca_sta_is_logical_block_size_supported()`-valid block size), and the `doca_sta_be` backend disks within the `doca_sta_get_max_subsys` / `doca_sta_get_max_ns_per_subs` / `doca_sta_get_max_be` bounds. |
| 3. Device support | Is STA supported on the chosen device? | Gate on `doca_sta_cap_is_supported` against the active `doca_devinfo` BEFORE writing any setup code. STA's transport is RDMA-only — there is no transport to choose. |
| 4. Queue-pair sizing | What I/O queue count and depth does the target need? | Gate each input on the matching `doca_sta_get_max_*` query (`doca_sta_get_max_qps`, `doca_sta_get_max_io_queue_size`, `doca_sta_get_max_io_size`); the reported max is the ceiling. |
| 5. Build manifest | A fresh `meson.build` (or equivalent) in the user's project | Wire `pkg-config doca-sta doca-rdma` (the RDMA substrate is always required); do NOT hand-roll a Makefile that drops the version-check and substrate-link rails. |
| 6. Initiator boundary | Is the user expecting doca-sta to act as an NVMe-oF initiator / host? | No — doca-sta is target-side acceleration only. The remote initiator (SPDK `bdev_nvme`, kernel `nvme` host) is out of scope; route the user to the upstream project. |

The agent emits an *intent description + the six filled slots*,
then helps the user assemble the minimal target program directly
against the public headers, validating each `doca_sta_*` call
against the header it came from. Because there is no shipped
sample to bisect against, the agent must lean harder on the
single-IO smoke in [`## test`](#test) as the known-good baseline.

## run

Goal: actually execute the built binary against the user's
installed DOCA on a BlueField, with a remote NVMe-oF initiator
reachable on the fabric to connect into the accelerated target
and drive the NVMe protocol semantics.

Steps the agent should walk the user through:

1. **Confirm the remote initiator can reach the target.**
   doca-sta accelerates the target data path; without a remote
   initiator (the NVMe-oF host) connecting in, no Connect
   handshake arrives and the target idles. For NVMe-over-RDMA —
   STA's only transport — both ends must route IB / RoCE to each
   other per [`doca-rdma TASKS.md ## run`](../doca-rdma/TASKS.md#run).
   This is a fabric / env precondition, NOT a code problem.
2. **Confirm the steering side is in place.** Per the
   precondition matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   NVMe-oF traffic only reaches the STA-managed queue when a
   DOCA Flow rule (or the env-side equivalent) steers it
   there. The most common symptom of a missing Flow rule is
   that the Connect handshake never completes and the program
   blocks indefinitely — do NOT recommend retrying the
   `doca_ctx_start()` until the Flow rule is confirmed per
   [`doca-flow TASKS.md ## run`](../doca-flow/TASKS.md#run).
3. **Start the target before the initiator connects.** Bring
   the STA target up (subsystems / namespaces / backends defined,
   `doca_ctx_start()` returned success) before the remote
   initiator attempts to connect. An initiator that connects
   before the target is listening produces a symptom identical to
   a steering bug and wastes bisection time.
4. **Have the remote initiator connect.** Per
   [`SKILL.md`](SKILL.md), doca-sta accelerates the target data
   path; the NVMe-oF Connect handshake, the Discovery
   exchange, and the admin / I/O command stream are driven by the
   remote initiator's NVMe host stack (SPDK `bdev_nvme`, kernel
   `nvme` host, …), which is out of scope for this skill. Confirm
   the initiator is pointed at the target's NQN.
5. **Capture the structured log on the first run.** Set
   `DOCA_LOG_LEVEL=trace` for the first run (see
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability)).
   This is the cheapest way to make the per-queue lifecycle
   transitions, the Connect handshake outcome, and the
   per-IO completion events visible on first failure — and
   to confirm the DOCA Core progress engine is being
   progressed per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability).

For the runtime version + `LD_LIBRARY_PATH` cross-checks that
underlie *"the program built but does nothing"*, see
[`doca-version TASKS.md ## run`](../../doca-version/TASKS.md#run).

## test

Goal: prove the configured STA context can actually establish
an NVMe-oF connection and move one admin command and one I/O
end-to-end on the user's hardware, before the user opens the
target to production traffic from remote initiators.

This is **a loop, not a one-shot pass.** Each iteration
narrows either the capability set, the substrate / steering
preconditions, the queue-pair sizing, or the target object
model. The loop
terminates when either (a) a single admin command (e.g.
Identify Controller) on the admin queue plus a single Read or
Write I/O on one I/O queue both complete successfully, or
(b) the agent has narrowed the failure cause to a layer
outside STA itself (substrate library, steering, fabric, remote
initiator, driver / firmware) and escalated to the matching
skill.

Iteration shape:

1. **Capability re-check.** Re-run `doca_sta_cap_is_supported`
   against the active `doca_devinfo`, and re-run the matching
   `doca_sta_get_max_*` query for each sizing value the user
   requested. If the device is unsupported (or a getter returns
   a smaller bound than the user's request) that is the answer —
   the user's device or DOCA version does not support the
   request. Per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   do NOT escalate further until the cap-query baseline is
   re-established.
2. **Substrate and steering re-check.** Walk the precondition
   matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
   once more: substrate library (the RDMA substrate),
   device access (group / sudo), DOCA Flow rule in place for
   the NVMe-oF 5-tuple. The vast majority of *"the initiator's
   Connect handshake never completes"* failures are here, not in
   the STA code.
3. **Single-IO smoke — admin command FIRST, then one I/O.**
   Per the smoke-before-scale-up rule in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   drive ONE NVMe admin command (typically Identify Controller)
   on the admin queue and confirm the completion event arrives
   on the DOCA Core PE with success; THEN drive ONE NVMe Read
   or Write on a single I/O queue and confirm its completion
   arrives. Failure on admin narrows to handshake / fabric
   / steering; failure on I/O after admin succeeded narrows to
   I/O-queue sizing / mmap / SPDK or kernel-nvme glue. Both
   together give a much cleaner bisection than starting at
   production scale.
4. **Multi-queue smoke.** Once the single-IO smoke is green,
   add a second I/O queue on the same connection, repeat one
   Read or Write on each, and confirm both completions
   arrive. Catches per-queue bugs that a single-queue smoke
   cannot (queue-count cap miscount, per-queue progress not
   wired, queue-pair state machine confused about which
   queue is which).
5. **Negative test — capability mismatch.** Intentionally
   request an over-cap queue depth or queue count, or a
   logical block size, that the `doca_sta_get_max_*` /
   `doca_sta_is_logical_block_size_supported` checks in step 1
   reject, and confirm the
   reported `DOCA_ERROR_NOT_SUPPORTED` (or
   `DOCA_ERROR_INVALID_VALUE` for an over-cap value) matches
   the queried answer. Validates the agent's capability
   discovery is itself correct.
6. **Sustained-run loop (optional, only after the smoke is
   green).** Drive a small steady-state workload through the
   established connection for minutes — not seconds — and
   confirm: no spurious `DOCA_ERROR_IO_FAILED` (transport is
   stable); no `DOCA_ERROR_AGAIN` storm (in-flight budget is
   appropriately sized for the steady rate); per-queue
   completions continue to arrive. Catches sizing-envelope
   bugs that a short smoke cannot.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| Connect handshake never completes | `doca_ctx_start()` returns success but the per-queue state never transitions past CREATED, OR the remote initiator reports the NVMe-oF Connect timed out | Re-walk the substrate + steering precondition matrix in [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy) BEFORE re-checking STA code — this is almost never a STA bug |
| `DOCA_ERROR_NOT_SUPPORTED` at start | The chosen device does not pass `doca_sta_cap_is_supported`, or a sizing value exceeds the reported `doca_sta_get_max_*` bound | Re-run `doca_sta_cap_is_supported` and the matching `doca_sta_get_max_*` query against the active `doca_devinfo`; pick a supported device OR lower the sizing value below the reported maximum |
| Single I/O fails with `DOCA_ERROR_IO_FAILED` after admin command succeeded | The admin queue worked but the I/O queue path is broken — transport-layer error, peer-side controller reset, or the substrate (e.g. RDMA queue-pair) hit a fault | Capture `dmesg | tail` per [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug) and the matching substrate-skill error taxonomy in [`doca-rdma CAPABILITIES.md ## Error taxonomy`](../doca-rdma/CAPABILITIES.md#error-taxonomy); do NOT retry blindly |
| `DOCA_ERROR_AGAIN` on submit during the sustained-run loop | The per-queue in-flight budget is full | This is the cross-library *"would-block, retry after progress"* pattern — drain completions via `doca_pe_progress()` before re-submitting, or raise the queue depth / in-flight budget within the device cap per [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) |
| Same code passes on host A, fails on host B | Different DOCA version, different substrate version, or different device cap surface | Re-run the version chain per [`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test) four-way match on host B; re-run the STA cap queries on host B against the active `doca_devinfo`; re-walk the substrate-library version-match rule in [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility) |

Loop termination: stop iterating once two consecutive
iterations of the same kind do not change the picture — that
means the cause is below STA (substrate, steering, fabric,
NVMe stack on top, driver / firmware). Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
with the captured STA-layer trace, the cap-query baseline,
and the substrate-side trace as evidence.

## debug

Goal: when a `doca_sta_*` call (or the per-queue / per-IO
event stream on the progress engine) returns a `DOCA_ERROR_*`
or does not behave as expected, narrow the cause to a single
layer before recommending any code change.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver — *before* recommending STA-specific
fixes. This skill's overlay names the STA-specific
manifestation at layers 5 (runtime), 6 (program), and 7
(driver / substrate):

**Layer 5 (runtime) — STA overlay.**

- `DOCA_ERROR_BAD_STATE` on the first STA call after start is
  *almost always* a lifecycle violation on the per-connection
  queue-pair state machine: an I/O was submitted before the
  queue-pair transitioned to CONNECTED, or the context was
  reconfigured after start. Walk the universal lifecycle in
  [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes)
  AND the per-queue state-transition note in
  [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability)
  before recommending any code change.
- `DOCA_ERROR_AGAIN` on I/O submit is the in-flight budget
  full. This is *not* a hardware error — drive
  `doca_pe_progress()` to drain completions, or raise the
  per-queue in-flight budget within the device cap per
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
  Do not recommend a retry loop without the progress call.
- *"The Connect handshake never completes"* is *rarely* an
  STA bug. Per the safety policy in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
  walk the precondition matrix (substrate library present,
  device access, steering rule in place) BEFORE any code
  change. The most common cause is a missing or wrong DOCA
  Flow rule for the NVMe-oF 5-tuple, owned by
  [`doca-flow TASKS.md ## debug`](../doca-flow/TASKS.md#debug).

**Layer 6 (program) — STA overlay.**

- Lifecycle order: configure → start → per-queue CONNECTED
  callback → use → stop → destroy. Out-of-order returns
  `DOCA_ERROR_BAD_STATE` per
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
  The most common case is submitting an I/O before the
  queue-pair reports CONNECTED — surface the transition-event
  hookup before any other diagnosis.
- Cap-query miss: `DOCA_ERROR_NOT_SUPPORTED` or
  `DOCA_ERROR_INVALID_VALUE` at configure / start is almost
  always a value (queue depth, queue count, logical block size)
  that the `doca_sta_get_max_*` /
  `doca_sta_is_logical_block_size_supported` checks would have
  rejected, or a device that does not pass
  `doca_sta_cap_is_supported`. Re-read the value, lower it, and
  re-run configure per
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
- Initiator-boundary confusion: the agent must NOT propose an
  initiator-side fix inside this skill. Per
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
  the remote initiator / host NVMe stack (SPDK `bdev_nvme`,
  kernel `nvme` host) drives the initiator side — if the failure
  is at that layer, route the user to the upstream project's own
  debug guide and do not invent a `doca_sta_*` call to substitute.

**Layer 7 (driver / substrate) — STA overlay.**

- `DOCA_ERROR_IO_FAILED` on a per-IO completion is a
  transport-layer error: link drop, RDMA peer disconnect,
  firmware fault, or peer-side controller reset.
  Capture `dmesg | tail` and `mlxconfig -d <pcie> q` per
  [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug)
  layer 5 (driver); for the NVMe-over-RDMA path overlay the
  substrate-side error taxonomy in
  [`doca-rdma CAPABILITIES.md ## Error taxonomy`](../doca-rdma/CAPABILITIES.md#error-taxonomy).
  Do NOT retry blindly in the STA code.
- `DOCA_ERROR_DRIVER` from any STA call is the layer below
  DOCA reporting failure. Capture
  `pkg-config --modversion doca-sta`,
  `pkg-config --modversion doca-rdma` (if applicable), and
  `doca_caps --version`; cross-check the version triple per
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility);
  route to
  [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug)
  layer 5 (driver). A version-skew between STA and the
  substrate is the canonical partial-install hazard here.
- `DOCA_ERROR_NOT_PERMITTED` on the `doca_dev` open or the
  `doca_sta` context create is access-side, not code-side.
  Confirm sudo or the appropriate group membership per
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy);
  the fix lives in
  [`doca-setup TASKS.md ## configure`](../../doca-setup/TASKS.md#configure),
  not in a `doca_sta_*` call.

Once the layer is identified, route to the matching debug
verb on the matching skill: install / build / link / driver
to [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug);
version to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug);
cross-cutting runtime to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug);
program-layer Core-context patterns to
[`doca-programming-guide TASKS.md ## debug`](../../doca-programming-guide/TASKS.md#debug);
substrate (RDMA) to
[`doca-rdma TASKS.md ## debug`](../doca-rdma/TASKS.md#debug);
steering to
[`doca-flow TASKS.md ## debug`](../doca-flow/TASKS.md#debug).

## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as follows
so the agent does not invent guidance:

- **install.** Installing DOCA, choosing packages, post-install
  verification, `pkg-config` wiring — defer to
  [`doca-setup`](../../doca-setup/SKILL.md) and to the install-tree
  layout in
  [`doca-public-knowledge-map ## Layout of an installed DOCA package`](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
  This skill assumes DOCA is already installed.
- **deploy.** Deploying NVMe-oF-using applications at scale
  across many hosts / DPUs, multi-tenant subsystem fan-out,
  Kubernetes operator workflows for NVMe-oF workloads — out
  of scope for Phase 1 and reserved for a future platform
  skill. For single-host first-run testing, the right verb in
  this skill is [`## run`](#run); do not invent a "deploy"
  workflow.
- **Initiator / host NVMe stack work.** The remote initiator's
  NVMe host stack — SPDK `bdev_nvme`, the kernel `nvme` host, or
  any other NVMe-oF initiator — is a separate peer with its own
  integration patterns. Per [`SKILL.md`](SKILL.md), doca-sta is
  target-side acceleration, not an initiator transport provider,
  and this skill does not ship initiator glue.
  Route the user to the upstream project's own integration
  documentation reachable via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
- **Initiator-side NVMe host semantics.** Host-side controller
  configuration, host block-layer semantics, and host multipath
  policy are owned by the remote initiator's NVMe stack, not by
  doca-sta. (Target-side namespaces and backends *are* doca-sta's
  job — see [`## configure`](#configure) step 3.) Surface the
  boundary per
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  and route initiator-side questions to the upstream project.
- **firmware burn / BFB re-image.** STA depends on the
  underlying ConnectX firmware and BlueField BFB; if the
  debug ladder lands on a driver-layer issue, the fix is via
  `mlxconfig` / `mlxfwreset` / re-imaging the BFB, all of
  which belong to the env-side skill rather than this one.
  Route to
  [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug)
  layer 5.

## Command appendix

Every command below is **cross-cutting on DOCA STA** — it
answers a recurring class of question that comes up in the
verbs above. The agent should treat the *class* as
load-bearing; the worked example is a single instance.
Run-as user is the unprivileged user unless noted. Rows that
need elevated privileges call that out explicitly.

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
| `pkg-config --modversion doca-sta` | [`## configure`](#configure) step 1; [`## build`](#build) version-anchor slot | What is the build-time DOCA STA version? | A semver string matching `doca_caps --version`. Disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2) |
| `pkg-config --modversion doca-rdma` | [`## configure`](#configure) step 1 (NVMe-over-RDMA path); [`## build`](#build) version-anchor slot | What is the build-time DOCA RDMA substrate version, and does it agree with `doca-sta`? | A semver string matching `pkg-config --modversion doca-sta` and `doca_caps --version`. Disagreement = substrate-vs-STA partial-install hazard per [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility) |
| `pkg-config --cflags --libs doca-sta doca-rdma` | [`## build`](#build) | What include + link flags does the linker need for the STA surface plus the RDMA substrate? | Trust whatever `pkg-config --cflags --libs` produces on this install. Do not hardcode either the `-I` include path or the `-l<name>` flag form — both can drift between DOCA install profiles and DOCA majors; the on-disk `.so` basenames use underscores on every release where we have ground truth, while the `.pc` package names use hyphens, and `pkg-config` is the only thing that resolves both correctly. Hand-crafted `-l` lines silently break when DOCA upgrades. |
| `ls /opt/mellanox/doca/samples/doca_sta/ 2>/dev/null` | [`## modify`](#modify) slot 1 | Does this install ship STA samples? | Expect **no such directory** — DOCA STA ships no public samples and is absent from the sample profiles. Build from the public headers (next row) instead; do NOT fabricate a sample path |
| `doca_caps --list-devs` | [`## configure`](#configure) step 2 | Which devices on this host can be used as a `doca_dev` for STA, and what do they advertise? | One row per visible device with PCIe address and capability flags; cross-check against `doca_sta_cap_is_supported` for the per-device STA support |
| `ls "$(pkg-config --variable=includedir doca-common)/doca_sta*.h"` | [`## configure`](#configure) step 2; [`## modify`](#modify) slot 1 | Which `doca_sta_*` cap-check, sizing, and setter symbols does this install actually expose? | One or more header files; grep inside for the `doca_sta_cap_is_supported`, `doca_sta_get_max_*`, and `doca_sta_set_*` declarations rather than quoting symbols from memory per [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) |
| `DOCA_LOG_LEVEL=trace ./<binary>` | [`## run`](#run) step 5 | What did the structured DOCA logger emit for the first failing STA call? | Trace-level lines on every STA-layer lifecycle transition, the Connect handshake outcome, every per-IO completion. Silence after `doca_ctx_start()` on the `doca_sta` = either PE not progressed OR the steering rule is missing — reach for the substrate / steering trace next |
| `dmesg \| tail -n 40` (sudo) | [`## debug`](#debug) layer 7 | What did the kernel / driver log around the last STA / substrate call? | Empty or recent benign messages. Repeated mlx5 / IB errors → driver-layer bug; route to [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug). Repeated NVMe-oF reset / disconnect lines → peer-side fault, NOT a doca-sta bug |
| `ibv_devinfo` (sudo, NVMe-over-RDMA path only) | [`## configure`](#configure) step 1; [`## debug`](#debug) layer 7 | What does the underlying `libibverbs` see for this device on the RDMA substrate side? | One device row with `state: PORT_ACTIVE` and a sane MTU; absence indicates the RDMA substrate is not actually up regardless of what doca-sta reports |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the STA-specific rows on top. The substrate-
library commands (RDMA-side cap queries, `ibv_devinfo`,
RDMA-side trace) live in
[`doca-rdma TASKS.md ## Command appendix`](../doca-rdma/TASKS.md#command-appendix)
and are referenced from there, not duplicated here.
