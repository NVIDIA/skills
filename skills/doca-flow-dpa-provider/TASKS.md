# DOCA Flow DPA Provider workflows

**Where to start:** The verbs run `install → configure →
build → modify → run → test → debug → use`. Skip ahead only
when the user is already past a verb. The `## test` verb is
an iterative loop (single-entry hash-control smoke →
single-memory-update smoke → memory-read with completion poll
→ widening loop → loop back if a precondition changes), not
a one-shot pass — see the eval-loop overlay in `## test`
below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the provider capability surface,
the three-program model, the per-port `doca_flow_dpa_ctx`,
the three queue types, the pipe-export and external-resource-
export handshakes, the DPA-side device API surface, the
error taxonomy, the observability surface, and the safety
policy, see [CAPABILITIES.md](CAPABILITIES.md). For the
host-side Flow surface this library exports, see
[`doca-flow`](../doca-flow/SKILL.md). For the host-side DPA
lifecycle, see [`doca-dpa`](../doca-dpa/SKILL.md). For the
cross-library DOCA patterns layered under everything below,
see
[`doca-programming-guide`](../../doca-programming-guide/SKILL.md).

Each verb below describes the **shape of the workflow**, not
a copy-paste recipe. The agent's job is to walk the user
through the steps in order, verifying preconditions before
recommending the next call.

## install

Goal: confirm the host has the `doca-flow-dpa-provider`
library installed alongside its prerequisites, AND that the
joint version set is consistent.

This library does not ship its own installer separate from
DOCA — it lands on disk when the user installs the matching
DOCA host packages. The install verb here is *verification*,
not orchestration. Defer the full install workflow (package
selection, NGC container fallback, post-install verification)
to [`doca-setup TASKS.md ## install`](../../doca-setup/TASKS.md#configure)
and the install-tree layout to
[`doca-public-knowledge-map ## Layout of an installed DOCA package`](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).

Verification gates the agent must walk for this skill:

1. **`pkg-config --modversion doca-flow-dpa-provider` returns
   a version.** If it fails with "Package
   doca-flow-dpa-provider was not found", the provider host
   package is not installed; route to `doca-setup` and stop
   until the package lands.
2. **The matching prerequisites are installed.**
   `pkg-config --modversion doca-flow` and `pkg-config
   --modversion doca-dpa` must BOTH return; the provider is
   a bridge and useless without either side. Missing either
   one routes back to the matching skill
   ([`doca-flow`](../doca-flow/SKILL.md) /
   [`doca-dpa`](../doca-dpa/SKILL.md)) for the install.
3. **The DPACC compiler is installed at a matching version.**
   `which dpacc && dpacc --version` returns a string the
   agent compares against the DOCA Compatibility Policy at
   <https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html>.
   Missing or mismatched `dpacc` is a hard blocker — the
   DPA-side translation unit cannot be built. Route to
   [`doca-setup`](../../doca-setup/SKILL.md).
4. **The four `.pc` versions agree.** The four numbers above
   (provider, flow, dpa, dpacc) must be consistent per the
   four-way match rule in
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility);
   a single mismatched one is a partial install and a
   future `DOCA_ERROR_DRIVER` waiting to fire.

If any gate fails, do NOT proceed to `## configure`; the
fixes live in `doca-setup` and `doca-version`, not in this
skill's program-side workflows.

## configure

Goal: stand up a `doca_flow_dpa_ctx` against a BlueField
port that already has a host-side `doca-flow` port and a
host-side `flexio_process` brought up, allocate the queue
set the DPA kernel will need, export the user's host-side
pipe to the DPA address space, and confirm everything is in
a state where the DPA kernel can consume the device address.

Steps the agent should walk the user through:

1. **Confirm the install gates from `## install` are
   green.** If the user has not run them, run them now;
   skipping the version check is the leading cause of a
   later `DOCA_ERROR_DRIVER` that looks like a hardware
   bug.
2. **Walk the three-program model BEFORE writing any
   provider code.** Per the three-program rule in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   the user is writing THREE translation units: a host-side
   `doca-flow` program (port + pipe), a host-side
   `doca-flow-dpa-provider` + `doca-dpa` program (queues +
   export + device-address handoff), and a DPA-side kernel
   (consumes the device address via the DPA-side device
   API). Surface this model EXPLICITLY before any
   `doca_flow_dpa_*` call is drafted. If the user is asking
   *"how do I write the DPA-side kernel itself"*, route via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
   to the public DOCA DPA / DPACC guides.
3. **Stand up the host-side `doca-flow` port and `flexio_process`
   FIRST.** Per
   [`doca-flow TASKS.md ## configure`](../doca-flow/TASKS.md#configure),
   bring up the port and confirm the device handle the
   `flexio_process` was created against matches the port's
   `doca_dev`. Mismatched devices is the silent-misroute
   failure mode the agent must surface.
4. **Initialize the provider context.** Call
   `doca_flow_dpa_init(process, &flow_dpa_ctx)` with the
   FlexIO process handle and capture the returned
   `doca_flow_dpa_ctx *`. The matching destroy call is
   `doca_flow_dpa_destroy(flow_dpa_ctx)` and must run
   AFTER every exported pipe / external resource has been
   destroyed.
5. **Allocate the queues the DPA kernel will use.** Call
   `doca_flow_dpa_queues_create(flow_dpa_ctx, port,
   queue_cfgs, num_queue_cfgs)` with a `queue_cfgs` array
   that includes the right queue types per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   queue-types table. The agent's anti-pattern alert:
   allocating only `DOCA_FLOW_DPA_QUEUE_TYPE_GENERAL` and
   then expecting the DPA kernel to call
   `doca_flow_external_resource_memory_read` is a
   guaranteed `DOCA_ERROR_NOT_SUPPORTED` later. Each
   queue's `queue_size` must be a power of 2.
6. **Run the pipe-export handshake in the documented
   order.** Per the export-lifecycle rule in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy):
   call `doca_flow_dpa_pipe_export_prepare(flow_dpa_ctx,
   pipe, nr_entries, &dpa_pipe)` BEFORE any entry is added
   to `pipe`; add entries via `doca-flow`; call
   `doca_flow_dpa_pipe_export(flow_dpa_ctx, dpa_pipe)` to
   commit the export; call
   `doca_flow_dpa_pipe_get_device_addr(flow_dpa_ctx,
   dpa_pipe, &dev_addr)` to populate the
   `doca_flow_dpa_addr` the DPA kernel will consume.
7. **(If applicable) Run the external-resource-export
   handshake.** Same shape as step 6 but using
   `doca_flow_dpa_external_resource_export(flow_dpa_ctx,
   external_resource_type, external_resource_ctx,
   &dpa_resource)` and
   `doca_flow_dpa_external_resource_get_device_addr(...,
   &dev_addr)`. Index-selector resources use the
   `RESOURCES_WRITE` queue type on the DPA side; memory
   resources use both `RESOURCES_WRITE` (for updates) and
   `RESOURCES_READ` (for the two-phase read-then-poll).
8. **Sanity check before launching the DPA kernel.**
   Confirm with the user: which BlueField port; which Flow
   pipe (and how many entries it will hold); which external
   resources are being exported; which queue types and
   depths were allocated; which DPA-side kernel function
   will consume the device address; how the kernel will
   terminate. If any of those are unclear, stop and ask —
   do not invent.

For the canonical DOCA universal lifecycle that underlies
the host-side Core contexts on either side, see
[`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure).

## build

Goal: compile a provider-using consumer (host-side C / C++ +
at least one DPA-side translation unit) against the user's
installed DOCA + DPACC compiler, with `pkg-config` + `dpacc`
as the joint sources of truth for include + link flags + DPA
binary embedding.

The build pattern for any DOCA C / C++ consumer is fully
documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
The provider stacks the DPACC layer documented in
[`doca-dpa TASKS.md ## build`](../doca-dpa/TASKS.md#build) on
top of the Flow build documented in
[`doca-flow TASKS.md ## build`](../doca-flow/TASKS.md#build).
This skill carries only the provider-specific overlay:

| Slot | Value | Why it matters |
| --- | --- | --- |
| `pkg-config` module name (host side) | `doca-flow-dpa-provider` | The library's `.pc` file installed by the DOCA host packages; gives the host-side include + link flags for the provider host API |
| Companion pkg-config modules | `doca-flow`, `doca-dpa` (and transitively `doca-common`) | The provider is a bridge — the host program links against all three; missing any one is a build error |
| DPA-side toolchain | `dpacc` (DPACC compiler), installed alongside DOCA | Compiles the DPA-side translation unit that includes `doca_flow_dpa_provider_dev.h`; the host system compiler is NOT a substitute |
| Host-side include flags | `pkg-config --cflags doca-flow-dpa-provider doca-flow doca-dpa` | Resolves the provider headers (`doca_flow_dpa_provider.h`) plus the Flow and DPA headers that the provider header includes (`doca_flow.h`, `doca_error.h`, `doca_compat.h`) |
| DPA-side include path | The DPACC-prescribed include path that resolves `doca_flow_dpa_provider_dev.h` plus the DPA-side base headers (`dpaintrin.h`, `doca_error.h`) — route via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) for the per-install layout | The DPA-side translation unit must define `DOCA_DPA_DEVICE` before including the DPA-side header, per the documented contract on `doca_flow_dpa_provider_dev.h` |
| Host-side link flags | `pkg-config --libs doca-flow-dpa-provider doca-flow doca-dpa` | Pulls in the provider, Flow, and DPA host-side libraries. No mention of the DPA-side device library on the host link line; that goes through `dpacc` |
| DPA-side link / embed step | The DPACC-prescribed embed step that bakes the DPA-side binary (which links the DPA-side device library that backs `doca_flow_dpa_provider_dev.h`) into the host executable as a DPA application image | Without this step the host has no DPA application image to load at runtime, and the device address handoff has nowhere to land |
| Minimum DOCA version (and DPACC version) | Query with `pkg-config --modversion doca-flow-dpa-provider` plus the matching `doca-flow`, `doca-dpa`, and `dpacc` versions; cross-check against the DOCA Compatibility Policy | The provider's working set is the joint cross-product; never hardcode |

For non-C host-side consumers (Rust, Go, Python) that drive
the provider's host-side setup and embed a DPA application
image built separately by `dpacc`, the host-side link line
and version rules above still apply; the DPA-side build is a
separate compilation unit and is out of scope for this
skill, but the `dpacc` version check is the load-bearing
input the wrapper still needs.

## modify

Goal: take the closest-fitting shipped DOCA Flow DPA Provider
sample as the verified starting point and apply a **minimum
diff** to make it match the user's intent, without rewriting
from scratch.

The universal modify-a-shipped-sample workflow is in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify);
this skill provides the provider-specific slot fill.

| Slot | What the agent asks the user | Provider-specific consideration |
| --- | --- | --- |
| 1. Starting sample | Which sample under the installed DOCA samples tree that exercises the provider (route to [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) for the per-install path; do not hardcode the directory)? | Pick a sample whose **shape** matches the user's intent: same Flow pipe type (hash vs basic), same external-resource set (none vs index-selector vs memory), same queue mix. Provider samples are three-side programs; the sample's DPA-side translation unit is the third half of the verified base |
| 2. Flow pipe spec | What pipe shape is being exported, and what is changing? | Per the validate-before-commit rule in [`doca-flow CAPABILITIES.md ## Safety policy`](../doca-flow/CAPABILITIES.md#safety-policy), the modified pipe spec must validate in `doca-flow` BEFORE the provider's export prepare step. A pipe shape that does not support export will fail `_export_prepare` with `DOCA_ERROR_NOT_SUPPORTED` |
| 3. Queue config | Which queue types and what depths does the modified DPA kernel need? | Per the queue-types table in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes), the queue-config array passed to `doca_flow_dpa_queues_create` must list the queue types the kernel will call against; a kernel that adds a memory read where the original sample only did entry control needs a new `RESOURCES_READ` entry in the array |
| 4. DPA-side kernel body | What does the DPA kernel actually do with the exported handle? Is it new behavior or a tweak on the sample's behavior? | The DPA-side translation unit is the in-place edit point. The agent's anti-pattern alert: do NOT propose moving the per-entry decision to the host side to "simplify" — that defeats the entire reason to use this library (host round-trip per entry is exactly what the export was meant to avoid) |
| 5. Device-address handoff | How does the host pass the `doca_flow_dpa_addr` (and any external-resource addresses) to the DPA kernel? | Same shape as any DPA launch-argument; per the two-side-program rule in [`doca-dpa CAPABILITIES.md ## Capabilities and modes`](../doca-dpa/CAPABILITIES.md#capabilities-and-modes), changing the launch-argument shape on either side requires re-aligning the other side |
| 6. Lifecycle ordering | Does the modify pass still preserve the export lifecycle (queues before export prepare, export prepare before entries, export before get-device-addr, kernel destroyed before context destroyed)? | Per [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy), this is the load-bearing invariant the modify cannot break. A refactor that moves the entry-add loop above the export-prepare call silently produces an exported pipe whose entries the DPA kernel cannot see |
| 7. Rebuild BOTH sides | After any modify, rebuild the DPA-side image via `dpacc` AND rebuild the host executable that embeds it | Inherited from [`doca-dpa CAPABILITIES.md ## Safety policy`](../doca-dpa/CAPABILITIES.md#safety-policy); partial rebuild is the canonical way to introduce `DOCA_ERROR_DRIVER` at runtime |

The agent emits an *intent description + the seven filled
slots*; the *actual* unified diff against the sample source
is produced the way every other library skill in this bundle
handles modify — the agent walks the user through the diff
line-by-line against the sample source they read on disk,
and has the user paste back the result for validation.

## run

Goal: actually launch the DPA kernel against the exported
pipe / resource and confirm the host-side and DPA-side
surfaces agree on what is happening.

Steps the agent should walk the user through:

1. **Confirm `## configure` is complete.** Specifically:
   queues created, pipe exported, device address captured
   in a host-side variable that the DPA launch path will
   pass to the kernel. If any step is missing, the kernel
   will start against a stale or NULL handle and produce
   misleading errors.
2. **Hand the device address to the kernel as a launch
   argument** (or in DPA-side state that the kernel reads).
   The launch itself is owned by the host-side DPA library
   per [`doca-dpa TASKS.md ## run`](../doca-dpa/TASKS.md#run);
   this skill's contract is only the device address.
3. **Drive the host-side `doca_pe_progress` loop** in
   parallel with any outstanding DPA work, exactly as
   `doca-dpa` documents. A host that submits the kernel
   and then blocks without progressing the PE will not see
   any completions — the provider does not change this
   pattern.
4. **Confirm the DPA-side kernel is calling
   `doca_flow_queue_poll_completion` on every queue type it
   posts to.** A kernel that posts hash-replace requests on
   the `GENERAL` queue and never polls that queue's
   completion ring will eventually see `DOCA_ERROR_AGAIN`
   on the next post call; the fix is on the DPA-side, not
   the host-side.
5. **Capture the structured log on first failure.** Set
   `DOCA_LOG_LEVEL=trace` for the first run (see
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability));
   if the host-side log is silent but the DPA-side kernel
   appears stuck, reach for the DPA-side developer tools
   inherited from
   [`doca-dpa CAPABILITIES.md ## Observability`](../doca-dpa/CAPABILITIES.md#observability).

For the runtime version + `LD_LIBRARY_PATH` cross-checks
that underlie *"the program built but does nothing"*, see
[`doca-version TASKS.md ## run`](../../doca-version/TASKS.md#run).

## test

Goal: prove the configured provider setup actually exports a
pipe (and any external resources) such that a DPA-side
kernel can read and mutate them through the documented
device API, and that the host-side Flow counters reflect the
DPA-side mutations the way the user intends.

This is **a loop, not a one-shot pass.** Each iteration
narrows either the export lifecycle, the queue-config, the
two-side-program signature for the device-address handoff,
the DPA-side polling discipline, or the Flow-side pipe spec.
The loop terminates when either (a) the host observes the
intended traffic effect AND the DPA-side
`doca_flow_dpa_completion_stats` shows `num_failed == 0`, or
(b) the agent has narrowed the failure cause to a layer
outside the provider itself (DPA-side kernel logic bug, Flow
pipe spec bug, BlueField generation gap) and escalated to
the matching skill.

Iteration shape:

1. **Single hash-control smoke.** From the DPA-side kernel,
   issue ONE `doca_flow_pipe_hash_disable_index` (or
   `_enable_index`) against a single known entry of the
   exported pipe; flush; poll the `GENERAL` completion
   queue; confirm `num_completed == 1`, `num_failed == 0`.
   If yes, advance. If no — and the host-side log shows no
   error — the export prepare-before-entries lifecycle was
   broken; route to [`## debug`](#debug) layer 5.
2. **Single memory-update smoke (if memory resources were
   exported).** Issue ONE `doca_flow_external_resource_memory_update`
   on a known offset; poll `RESOURCES_WRITE`; confirm
   `num_completed == 1`, `num_failed == 0`. Catches
   queue-config gaps where `RESOURCES_WRITE` was not
   allocated.
3. **Single memory-read smoke (if memory resources were
   exported AND reads are needed).** Issue ONE
   `doca_flow_external_resource_memory_read`; poll
   `RESOURCES_READ`; confirm the populated value matches
   the last update. Catches the two-phase contract
   (read-then-poll) that the installed header documents.
4. **Widening loop.** Once each kind of operation has
   smoked cleanly on a single entry / offset, replay the
   user's intended kernel against the same exported pipe;
   confirm the host-side Flow counters on the exported
   pipe move the way the user expects. A counter that
   does not move while `num_failed == 0` is a host-side
   Flow spec bug, not a provider bug.
5. **Lifecycle negative test.** Intentionally call
   `doca_flow_dpa_pipe_export_prepare` AFTER an entry has
   been added to the pipe (in a copy of the test program)
   and confirm the DPA-side kernel sees an empty exported
   pipe — validates that the agent's earlier
   lifecycle-ordering check is real, not just notional.
   Then restore the correct ordering.
6. **Queue-config negative test.** Intentionally call a
   DPA-side memory-read while the queue config did not
   include `RESOURCES_READ` and confirm the reported
   `DOCA_ERROR_NOT_SUPPORTED` matches the queue-types
   table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).

Loop termination: stop iterating once two consecutive
iterations of the same kind don't change anything — that
means the cause is below the provider (Flow pipe spec, DPA
kernel logic, BlueField firmware). Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
with both the host-side DOCA log and the DPA-side
completion-statistics output captured.

## debug

Goal: when a `doca_flow_dpa_*` call (host-side) or a
DPA-side device-API call against an exported handle returns
a `DOCA_ERROR_*` or does not make forward progress, narrow
the cause to a specific layer and act on it.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build →
link → runtime → program → driver — *before* recommending
provider-specific fixes. This skill's overlay names the
provider-specific manifestation at layers 5 (runtime), 6
(program), and 7 (driver):

**Layer 5 (runtime) — provider overlay.**

- A DPA-side request that returns `DOCA_ERROR_AGAIN` is
  *always* a queue-full problem on the matching queue type.
  The fix is a drain-then-retry pattern via
  `doca_flow_queue_poll_completion(queue_type)` on the
  right queue type, NOT a tight retry loop and NOT a queue
  resize on the first occurrence.
- A DPA-side memory-read whose value is "wrong" or stale
  most often means the kernel forgot to poll
  `RESOURCES_READ` before reading the populated `value`
  — the call is two-phase per the installed header. Walk
  the DPA-side polling loop with the user.
- A host-side counter on the exported pipe that fails to
  move while the DPA-side completion stats show
  `num_completed > 0`, `num_failed == 0` is the
  *pipe-export-prepare-after-entries* failure mode from
  the lifecycle rule in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  — the exported pipe has no entries because the entries
  were added before the prepare call. Re-walk
  `## configure` step 6.

**Layer 6 (program) — provider overlay.**

- Lifecycle ordering: queues must be created BEFORE the
  first `_pipe_export_prepare`; export prepare must
  precede entry-add; entry-add must precede export; export
  must precede `_pipe_get_device_addr`. Out-of-order
  returns `DOCA_ERROR_BAD_STATE` per
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
- Wrong queue type in the config: a kernel that calls
  `doca_flow_external_resource_memory_update` while the
  queue config did not include `RESOURCES_WRITE` fails
  with `DOCA_ERROR_NOT_SUPPORTED`. Update the queue config
  passed to `doca_flow_dpa_queues_create` in step 5 of
  `## configure`, then rebuild and re-test.
- Wrong handle to the kernel: a kernel that takes a
  `doca_flow_dpa_addr` that was minted against a different
  `doca_flow_dpa_ctx` (e.g. a different port) silently
  reads from the wrong pipe. The agent surfaces this when
  the user reports *"my kernel is reading values that
  don't match the pipe I think I attached to"*.
- Three-program signature mismatch: changing the DPA-side
  kernel's expected launch-argument shape without
  re-aligning the host-side handoff returns
  `DOCA_ERROR_INVALID_VALUE` from the host-side DPA
  launch call (inherited from
  [`doca-dpa CAPABILITIES.md ## Error taxonomy`](../doca-dpa/CAPABILITIES.md#error-taxonomy)).

**Layer 7 (driver) — provider overlay.**

- `DOCA_ERROR_DRIVER` from any host-side `doca_flow_dpa_*`
  call is most often a joint
  `doca-flow` + `doca-dpa` + `doca-flow-dpa-provider` +
  `dpacc` version skew. Capture all four versions via
  `pkg-config --modversion` and `dpacc --version`;
  cross-check against the DOCA Compatibility Policy at
  <https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html>.
- A BlueField in the wrong mode that does NOT expose the
  DPA surfaces as `DOCA_ERROR_NOT_SUPPORTED` at
  `doca_flow_dpa_init` (or at `doca_dpa` create — the
  inherited DPA failure mode). Route to
  [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug)
  layer 5 (driver) for the env-side fix.

Once the layer is identified, route to the matching debug
verb on the matching skill: install / build / link / driver
to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug);
version to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug);
cross-cutting runtime to
[`doca-debug ## debug`](../../doca-debug/TASKS.md#debug);
program-layer Core-context patterns to
[`doca-programming-guide TASKS.md ## debug`](../../doca-programming-guide/TASKS.md#debug);
Flow-side pipe-spec problems to
[`doca-flow TASKS.md ## debug`](../doca-flow/TASKS.md#debug);
DPA-side kernel hangs to
[`doca-dpa TASKS.md ## debug`](../doca-dpa/TASKS.md#debug).

## use

Goal: integrate the provider into a real application so that
DPA-side logic running on the BlueField participates in the
DOCA Flow datapath, with the host-side and DPA-side
lifecycles cleanly coupled.

The integration patterns the agent should walk the user
through:

- **Per-flow stateful work on the accelerator.** The
  canonical use case: a DOCA Flow pipe owns the
  match-and-action specification; per-entry counters live
  on the hardware; a DPA kernel reads the counters via
  `doca_flow_external_resource_memory_read*` or watches the
  hash-pipe directly, decides per-flow what to do next
  (rate-limit, disable, redirect), and applies the change
  via `doca_flow_pipe_hash_*` or
  `doca_flow_external_resource_*` — all without a host
  round-trip. The agent walks the user through the queue
  allocation, the two device addresses (one for the pipe,
  one for each external resource), and the DPA-side polling
  discipline that keeps the queues from filling up.
- **Stateful steering decisions driven by DPA-side
  observability.** When the steering decision depends on
  data only the DPA can see in time (because the host round-
  trip is too slow), exporting the pipe and modifying it
  inline is the right shape. The agent surfaces that this
  is a *niche* pattern; if the decision can be made by the
  host PE on a per-batch cadence, host-side `doca-flow`
  plus periodic re-programming is simpler and the user
  should not be here.
- **Bulk index-selector or memory updates from the DPA
  side.** The `_modify_range` and `_memory_read_range`
  variants exist precisely because per-element posts on
  large structures saturate the work queue. The agent
  surfaces the chunking contract documented on those calls
  (one low-level request per 32-index chunk for the
  index-selector range; one per 8-index chunk for the
  memory-range read) so the user sizes their completion
  polls correctly.

For each integration pattern, the agent must surface that
the validate-before-commit discipline from
[`doca-flow CAPABILITIES.md ## Safety policy`](../doca-flow/CAPABILITIES.md#safety-policy)
and the two-side-program rebuild discipline from
[`doca-dpa CAPABILITIES.md ## Safety policy`](../doca-dpa/CAPABILITIES.md#safety-policy)
BOTH apply, with the additional export-lifecycle constraint
documented in
[`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
on top.

## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as
follows so the agent does not invent guidance:

- **deploy.** Deploying provider-using applications at scale
  (multi-BlueField clusters, Kubernetes operator workflows
  for DPU workloads, multi-tenant DPA sharing) — out of
  scope for Phase 1 and reserved for a future platform
  skill. For single-host first-run testing, the right verb
  in this skill is `## run`; do not invent a "deploy"
  workflow.
- **rollback.** Coordinated rollback of an exported pipe
  across multiple BlueFields and host nodes — out of scope
  for Phase 1 and reserved for a future platform skill.
  For single-DPU spec rollback within a session, the
  right move is to destroy the exported `doca_flow_dpa_pipe`
  (and any exported external resources) and rebuild from
  `## configure` step 6; do not invent a "rollback"
  workflow.
- **DPA-side kernel programming and DPACC usage.** Writing
  the DPA-side function body, DPA-side memory layout,
  DPACC compile flags, DPA-side debugging from inside the
  kernel — out of scope. Route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  to the public *DOCA DPA*, *DOCA DPACC Compiler*, and
  *DOCA Flow* guides plus the *DPA Tools* umbrella.
- **Host-side Flow pipe spec design.** Owned by
  [`doca-flow`](../doca-flow/SKILL.md). This skill
  *exports* a pipe; it does not redefine how to write the
  pipe spec.
- **Default factory PCC algorithms, RDMA / RoCE traffic
  setup, and other adjacent datapath surfaces.** Route via
  the per-library skill (`doca-rdma`, `doca-pcc`,
  `doca-gpunetio`, …) — this skill is the Flow-side bridge
  to the DPA only.

## Command appendix

Every command below is **cross-cutting on DOCA Flow DPA
Provider** — it answers a recurring class of question that
comes up in the verbs above. The agent should treat the
*class* as load-bearing; the worked example is a single
instance.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST
   (`doca-env --json` for version + devices + libraries +
   drivers + hugepages in one shot; `doca-capability-snapshot`
   for per-device capability flags; `version-matrix.json`
   for *"available since"* lookups).
2. If the probe succeeds, the structured tool's output is
   the authoritative answer and the agent SHOULD NOT also
   run the manual command in the row below. Report *"using
   structured `<tool>`"*.
3. If the probe fails, fall back to the manual command in
   the row. Report *"falling back to manual chain"*.
4. The schemas the structured tools emit are defined in
   [`doca-structured-tools-contract ## Schemas`](../../doca-structured-tools-contract/SKILL.md#schemas);
   the version-handling semantics are owned by
   [`doca-version`](../../doca-version/SKILL.md).

| Command (worked example) | Owning step | Class of question it answers | What healthy output looks like |
| --- | --- | --- | --- |
| `pkg-config --modversion doca-flow-dpa-provider` | `## install` step 1; `## build` minimum-version slot | What is the build-time DOCA Flow DPA Provider host-side library version? | A semver string matching `doca_caps --version`. Disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2) |
| `pkg-config --cflags --libs doca-flow-dpa-provider doca-flow doca-dpa` | `## build` | What include + link flags does the host-side linker need for the joint provider + Flow + DPA build? | Includes resolve under whichever include directory `pkg-config --cflags` reports on this install (do not hardcode the path); libs include whatever `pkg-config --libs doca-flow-dpa-provider doca-flow doca-dpa` resolves on this install (do not predict the `-l<name>` form by hand). NO mention of DPA-side device libraries (those are linked into the DPA-side translation unit by `dpacc`, not into the host) |
| `which dpacc && dpacc --version` | `## install` step 3; `## build` minimum-DPACC slot | Is the DPACC compiler installed and at what version? | A version string the agent compares against the DOCA Compatibility Policy linked from [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility). Missing `dpacc` = the DPA-side translation unit cannot be built |
| `doca_caps --list-devs` | `## configure` step 3 | Which DOCA devices does the host see, and which expose a DPA processor (the hardware substrate the exported handle is consumed on)? | One entry per `doca_dev` with the BlueField identity and the per-library capability flags including the DPA support axis and the Flow support axis. No DPA-capable entry = the BlueField is not present, not in the right mode, or not on a generation that exposes the DPA |
| `DOCA_LOG_LEVEL=trace ./<binary>` | `## run` step 5 | What did the structured DOCA logger emit for the first failing host-side provider call? | A trace-level line on every host-side lifecycle transition (`doca_flow_dpa_init`, `_queues_create`, `_pipe_export_prepare`, `_pipe_export`, `_pipe_get_device_addr`). Silence after `_pipe_export` while the DPA-side kernel is launched = either host PE not progressed OR the DPA kernel is running but stuck — reach for the DPA-side tooling next |
| (route via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)) DPA-side developer tools — DPA debugger, DPA process-state inspector, DPA statistics tool | `## debug` layer 5; `## debug` layer 6 | What is the DPA processor itself doing right now, from the DPA side (for cases where the kernel is suspected to be stuck or to be polling the wrong queue type)? | The public *DPA Tools* umbrella documents the per-tool output; the agent's job is to NAME the existence of these tools and route the user there, not to redefine their surface |
| `ls` against the installed DOCA samples tree (route via [`doca-public-knowledge-map ## Layout of an installed DOCA package`](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package)) | `## modify` slot 1 | Which provider samples ship in this install, and which is the closest starting point? | A list of sample directories that each contain BOTH host-side and DPA-side source plus a `meson.build` that wires `dpacc` and `pkg-config doca-flow-dpa-provider doca-flow doca-dpa` together |

For commands shared across libraries (`pkg-config
--modversion`, `doca_caps`, `cat
/opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the provider-specific rows on top.
