# DOCA Verbs workflows

**Where to start:** The verbs run `configure → build → modify → run
→ test → debug`. EVERY workflow below begins by re-confirming the
drop-down decision in [SKILL.md](SKILL.md) — if the matching
higher-level DOCA library (`doca-rdma/SKILL.md` /
`doca-eth/SKILL.md` /
`doca-rmax/SKILL.md`) covers the user's case, the
right answer is to stop and climb back up, not to walk these verbs.
The `## test` verb is an iterative loop (smoke → narrow → loop back
if the WR-flag set, QP feature set, or completion path changed),
not a one-shot pass.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the higher-level-library-vs-doca-verbs
split, the libibverbs-vs-doca-verbs boundary, the verbs object
model, capability discovery, error taxonomy, observability, and
safety policy that these workflows assume, see
[CAPABILITIES.md](CAPABILITIES.md). For the foundation primitives
(`doca_dev`, `doca_pe`, `doca_ctx`, `doca_mmap`, `doca_buf`,
`doca_log`) every verbs context rests on, see
`doca-common/SKILL.md`. For the cross-library
DOCA patterns layered under everything below (the universal
lifecycle, the cross-library `DOCA_ERROR_*` taxonomy, the
modify-a-shipped-sample workflow), see
`doca-programming-guide/SKILL.md`.

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through the
steps in order, verifying preconditions before recommending the
next call.

## install

See [references/install.md](references/install.md).

## DOCA 3.5 field notes

Observed on DOCA 3.5 with BlueField-3 (developer review); check them
before debugging further:

- **Async events block.** `doca_verbs_get_async_event()` blocks until
  an event is pending; the header says it is a blocking function even
  though the guide calls async events non-blocking notifications.
  Poll the context's async-event handle (`doca_verbs_get_async_event_handle()`)
  with `poll()` / `epoll` first, then call `doca_verbs_get_async_event()`
  only when it is readable, and acknowledge every event with
  `doca_verbs_ack_async_event()`.
- **Multi-Rail on BlueField-3.** The 3.5 Multi-Rail app fails with
  default settings: `EWMA (measured-BW) LB mode is supported only on
  Spectrum-X devices`, and `doca_verbs_multi_rail_update_qp` returns
  `DOCA_ERROR_NOT_SUPPORTED`. Use `lb_mode: port_speed` (listed in the
  app's `--help`). The app has no guide yet; treat `--help` as the
  reference.
- **Multi-Rail build is skipped without json-c.** meson prints
  `Skipping compilation of DOCA Application - verbs_multi_rail_server_client
  - Missing json-c support`; install `libjson-c-dev` (not pulled in by
  `doca-all`) and rebuild.

## configure

Goal: bring up a `doca-verbs` context on a host or BlueField, with
a QP / CQ / PD / MR / SRQ / AH set the user has explicitly chosen,
*after* confirming the drop-down decision is correct.

Steps the agent should walk the user through:

1. **Re-confirm drop-down.** Ask: *"have you confirmed the matching
   higher-level DOCA library (`doca-rdma/SKILL.md`
   for RDMA work, `doca-eth/SKILL.md` for Ethernet
   queues, `doca-rmax/SKILL.md` for timing-precise
   media) does not expose the verb / opcode / WR flag / QP
   attribute / SRQ option / CC-group hint / Ethernet-queue option
   you need?"* If no — stop, route back. If yes — ask the user to
   name the specific thing the higher-level surface does not
   cover; that name is the load-bearing input for steps 4 and 5.
   This step is the cheapest place to catch the *"recommended raw
   verbs unnecessarily"* failure mode.
2. **Version and ABI evidence before feature advice.** Name and follow
   `doca-version TASKS.md ## configure` (`doca-version/TASKS.md#configure`).
   Request `pkg-config --modversion doca-verbs`,
   `pkg-config --modversion doca-common`, and `doca_caps --version`
   from the target install; compare them using that version-coherence
   procedure before promising a feature. Missing or mismatched evidence
   means stop and diagnose the install, not guess a version.
   Also state that `doca_verbs_*` is experimental on the documented
   release: signatures and availability can change between releases.
3. **Read the symbols from that same install.** Discover the headers
   with `pkg-config --variable=includedir doca-common`, and list
   `/opt/mellanox/doca/samples/` for the shipped examples. The installed
   headers and ABI declarations win over prose; never extrapolate an
   experimental symbol into another release. Version agreement is only
   a prerequisite: the device query in step 5 must still confirm the
   specific feature on the active `doca_devinfo`.
4. **Bring up the doca-common foundation FIRST.** Walk the
   universal foundation skeleton in
   `doca-common TASKS.md ## use` (`doca-common/TASKS.md#use`):
   `doca_devinfo_create_list` → pick devinfo → `doca_dev_open` →
   `doca_pe_create`. Without this foundation in place, the
   verbs-layer steps below have nothing to attach to.
5. **Capability-query the SPECIFIC verb / option the user named
   in step 1.** Call `doca_verbs_query_device(devinfo, &dev_attr)`
   and inspect the matching `doca_verbs_device_attr_get_*` for
   what the user wants — `_get_is_qp_type_supported` for the QP
   type; `_get_max_qp_wr` / `_get_max_sge` / `_get_max_inline_data`
   for QP sizing; `_get_is_ece_supported`,
   `_get_is_cc_group_supported`, `_get_is_cqe_inline_supported`,
   `_get_is_ordering_semantic_supported`,
   `_get_is_send_dbr_mode_supported`,
   `_get_is_dpa_external_datapath_supported`,
   `_get_is_gpu_external_datapath_supported` for the feature
   flags; the Ethernet-queue family
   (`_get_max_eth_sq_wr_num`, `_get_max_eth_rq_wr_num`,
   `_get_is_eth_sq_ts_source_type_supported`,
   `_get_is_eth_rq_ts_source_type_supported`,
   `_get_is_eth_rq_srq_supported`,
   `_get_is_allow_multi_pkt_send_wqe_supported`,
   `_get_is_plane_index_supported`) when the user is dropping to
   doca-verbs for an Ethernet-queue case. If the relevant flag is
   false (or the max is zero) — that is the answer. The user's
   device / firmware / DOCA version does not support what they
   wanted, and dropping to raw verbs cannot manufacture support
   that the cap-query denies. Climb back up to the matching
   higher-level library if it covers a viable alternative. **Free
   the handle** with `doca_verbs_device_attr_free` after
   snapshotting.
6. **Sketch the verbs object set BEFORE any code.** Which
   `doca_verbs_context`? Which `doca_verbs_pd`? Which
   `doca_verbs_qp`(s), with which feature set (transport type,
   QP type, ECE / CC-group / CQE-inline / ordering-semantic
   choices)? Which `doca_verbs_cq`(s), with which
   completion-handling path (DOCA progress engine vs manual
   `doca_verbs_bridge_poll_cq` vs comp-channel event delivery —
   pick one explicitly per
   [CAPABILITIES.md ## Observability](CAPABILITIES.md#observability))?
   Which `doca_verbs_srq`(s)? Which `doca_verbs_ah_attr` for IB
   or RoCE routing? Which MR(s) covering which memory? If any of
   those are unclear, stop and ask — do not invent.
7. **Configure the verbs context.** Walk the universal Core
   lifecycle (`cfg-create → cfg-set-* → init → start → use →
   stop → destroy`, per
   `doca-common CAPABILITIES.md ## ctx` (`doca-common/CAPABILITIES.md#ctx`));
   apply the verbs-specific setters in the order the headers
   require. The high-level shape is:
   `doca_verbs_context_create(dev, &ctx)` → `doca_verbs_pd_create(ctx, &pd)`
   → if selected, `doca_verbs_comp_channel_create(ctx, &chan)` and
   `doca_verbs_cq_attr_set_comp_channel` on the CQ attributes →
   CQ init-attr / create → QP init-attr / create → drive the QP
   state machine via `doca_verbs_qp_modify(qp, qp_attr)` through the
   transitions the QP type requires (RESET → INIT → RTR → RTS for RC)
   → `doca_pe_connect_ctx(pe, ctx_as_doca_ctx)` → `doca_ctx_start`.
   This is the order in [CAPABILITIES.md](CAPABILITIES.md) and
   [references/use.md](references/use.md); when the shipped sample
   differs on your release, follow the sample. Per the no-mixing rule in
   [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy),
   do not pull any `ibv_*` handle into this context.
8. **Sanity check before any WR submission.** Confirm with the
   user: which QP, which CQ it reports to, which MR(s) the WR
   will reference, which WR opcode + flags. If any of those are
   unclear, stop and ask — raw verbs amplifies the cost of
   guessed parameters.

If any step fails with a `DOCA_ERROR_*`, route through the error
taxonomy in
[CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy)
before retrying.

## build

Goal: produce a binary that links `doca-verbs` against the user's
installed DOCA, using the canonical cross-library build pattern.

The build pattern for any DOCA C/C++ consumer is **identical**
across libraries — `pkg-config` for include + link flags, meson
or CMake as the build system — and is fully documented in
`doca-programming-guide TASKS.md ## build` (`doca-programming-guide/TASKS.md#build`).
This skill carries only the raw-verbs-specific overlay:

| Slot | Value for doca-verbs | Why it matters |
| --- | --- | --- |
| `pkg-config` module name | `doca-verbs` (NOT `doca-rdma`; NOT `libibverbs`; NOT `doca-eth` even when the workload is Ethernet-side) | The library's `.pc` file installed by the DOCA host packages. A typo to `doca-rdma` silently links the wrong library; a fallback to `libibverbs` puts the user on the wrong side of the boundary in [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy) |
| Required runtime libs | `libdoca_common` plus the verbs runtime referenced by `pkg-config --libs doca-verbs` | The raw-verbs library depends on Core (`doca-common`); it does NOT auto-pull `doca-rdma` / `doca-eth` / `doca-rmax`, and adding any of them on the link line as a *"just in case"* hides the drop-down decision the agent and user already made |
| Header check | The verbs headers resolvable under $(pkg-config --variable=includedir doca-common) (`doca_verbs.h` and the adjacent `doca_verbs_*.h` family) | If `pkg-config --cflags doca-verbs` resolves but the include is missing, the install is partial — route to `doca-version TASKS.md ## debug` (`doca-version/TASKS.md#debug`) |
| Minimum required DOCA version | Query with `pkg-config --modversion doca-verbs`; never hardcode in build files | Cross-version build/runtime mixing breaks per [CAPABILITIES.md ## Version compatibility](CAPABILITIES.md#version-compatibility), and verbs-only upgrades are a documented partial-install pattern. The doca-verbs ABI is on the experimental tier and may shift between releases |
| Coexistence with the higher-level library on the same link line | Allowed only if BOTH libraries are independently used in the same binary; never *just in case* | Adding `doca-rdma` (or `doca-eth` / `doca-rmax`) to a binary that only uses `doca-verbs` is a code-smell that suggests the drop-down was not actually needed |

For non-C consumers (Rust, Go, Python), the link surface is the
same `*.so` files; the FFI wrapper layer is the language-specific
binding and is out of scope for this skill — but the five slots
above are still the load-bearing inputs the wrapper needs.

## modify

Goal: take a shipped DOCA Verbs sample as the verified starting
point and apply a minimum-diff modification to express the user's
intent — OR, for the libibverbs-porting case, walk the integration
path step by step (this is the load-bearing case for this skill in
particular).

The universal modify-a-shipped-sample workflow lives in
`doca-programming-guide TASKS.md ## modify` (`doca-programming-guide/TASKS.md#modify`).
Use it as-is for the *modify-a-sample* path. The verbs-specific
overlays are the *modify-from-sample schema fill* AND the
*libibverbs-porting block* below.

**Modify-from-sample schema fill — the five slots the agent must
elicit from the user before recommending any code-level edit:**

| Slot | What the agent asks the user | Verbs-specific consideration |
| --- | --- | --- |
| 1. Starting sample | Which sample under the installed DOCA samples tree? Run `ls /opt/mellanox/doca/samples/` and pick the one whose QP shape (reliable vs unreliable; connected vs unconnected; RDMA-side vs Ethernet-side) most closely matches the user's intent | Pick the closest in *QP shape* (RC / UC — the only QP types this release defines) and *completion-handling path* (DOCA PE vs manual CQ poll vs comp-channel event, per [CAPABILITIES.md ## Observability](CAPABILITIES.md#observability)). Do NOT bridge across both axes in a single modify pass — a smaller diff is always safer than a re-architecture |
| 2. Verbs / opcodes added or removed | Which WR opcodes? Which QP features (ECE, CC-group, CQE-inline, ordering-semantic, send-DBR-mode)? Which SRQ options? Which AH attributes? Which Ethernet-queue options (TS-source-type, plane-index, multi-pkt-send-WQE, flush-in-error)? | Each added opcode / WR flag / QP attribute / SRQ option / AH attribute / CC-group hint / Ethernet-queue option needs its own `doca_verbs_device_attr_get_*` query from `## configure` step 5 against the active `doca_devinfo` before any code-level change |
| 3. MR / PD changes | Which MR(s) / PD(s) change? | Refer to the no-mixing rule in [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy) — if the user is tempted to reuse an `ibv_pd` / `ibv_mr` from existing code, the porting block below is the right path |
| 4. Completion-handling change | Switch PE → manual CQ poll → comp-channel, or vice versa? | This is a re-architecture, not a tweak. If yes, recommend the user start from the sample that already uses the target path instead of patching one over |
| 5. QP type / queue sizing | Change QP type (RC / UC) or queue depths? | Same option set as `doca-rdma/SKILL.md` when relevant; re-run `## configure` step 5 — QP-type support is device-conditional even at the verbs level (`doca_verbs_device_attr_get_is_qp_type_supported`) |

**Libibverbs porting block — what the agent walks the user through
when they arrive with existing `ibv_*` code.** This is NOT a
textual replacement; it is an integration into the DOCA Core
model:

1. **Confirm the case really wants `doca-verbs`, not the matching
   higher-level DOCA library.** Most libibverbs ports land cleaner
   on the higher-level surface because the task abstractions cover
   the case once the user names what they actually do. Re-walk
   the drop-down decision in [SKILL.md](SKILL.md) before any
   porting work. A worked instance: a libibverbs codebase that
   sends `IBV_SEND_INLINE` from a custom QP attribute likely maps
   to `doca-rdma` with the matching `doca_rdma_task_send_*`
   abstraction and a per-task inline-data flag; only if that
   abstraction does NOT expose the inline path does the user need
   to drop to `doca-verbs` and set the verbs-side equivalent
   (`doca_verbs_qp_init_attr_set_max_inline_data` + a WR with the
   inline flag).
2. **Replace `ibv_context` ownership with `doca_dev` ownership.**
   The device handle the verbs context consumes is `doca_dev`,
   discovered through the standard DOCA Core path
   (`doca_dev_open` on a `doca_devinfo`), not through
   `ibv_open_device`. Mixing the two handles on the same hardware
   resource is the no-mixing rule in
   [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy).
3. **Build the verbs context.** Call
   `doca_verbs_context_create(dev, &ctx)`; create the PD with
   `doca_verbs_pd_create(ctx, &pd)`; create the CQ(s) via the
   `doca_verbs_cq_attr_*` + `doca_verbs_cq_create` family; create
   the QP via the `doca_verbs_qp_init_attr_*` +
   `doca_verbs_qp_create` family; drive the QP state machine to
   the in-use state with `doca_verbs_qp_modify` using a
   `doca_verbs_qp_attr` built via the
   `doca_verbs_qp_attr_*` setters; build the AH attributes via
   `doca_verbs_ah_attr_*` setters and attach to the QP via
   `doca_verbs_qp_attr_set_ah_attr` for RoCE or IB QPs.
4. **Integrate with the DOCA Core lifecycle.** Wrap the verbs
   object set in a `doca_ctx_*` lifecycle per
   `doca-common TASKS.md ## configure` (`doca-common/TASKS.md#configure`).
   Calls that lived in `ibv_*` setup move into the
   verbs-init-attr / qp-attr / ctx-start path; calls that lived
   in cleanup move into the destroy path (reverse order —
   destroy the QP before the CQ before the PD before the
   context).
5. **Drive completions through the DOCA progress engine.** Replace
   the libibverbs CQ-polling loop with `doca_pe_progress` against
   the verbs context, per
   [CAPABILITIES.md ## Observability](CAPABILITIES.md#observability).
   If the user has a documented reason to keep manual CQ polling,
   the verbs-side equivalent is `doca_verbs_bridge_poll_cq`; if
   the user has a documented reason to use event-driven delivery,
   the verbs-side equivalent is `doca_verbs_comp_channel_create`
   + `doca_verbs_get_cq_event` + `doca_verbs_ack_cq_events`. The
   agent must make the choice explicit, not let it drift.
6. **Map every `ibv_*` error path onto `DOCA_ERROR_*`.** Per the
   raw-verbs error overlay in
   [CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy),
   including the `IO_FAILED → inspect the CQE` rule.

The agent emits an *intent description + the five filled slots*
(modify-from-sample case) or *the six-step porting walk*
(libibverbs case); the *actual* unified diff against the sample
source is produced by the modify-from-sample renderer (deferred to
a future round). Until the renderer ships, the agent walks the
user through the diff line-by-line against the sample source they
read on disk, and has the user paste back the result for
validation.

## run

Goal: actually execute the built binary against the user's
installed DOCA on a host or BlueField, including a peer to connect
to (for QP-class workloads) or a traffic generator (for
Ethernet-queue-class workloads).

Steps the agent should walk the user through:

1. **Confirm the peer / traffic source is reachable.** For
   QP-class workloads, raw verbs needs a peer; the peer's RDMA
   stack and transport must match what this side asked for
   (IB / RoCE; QP type RC / UC and feature set per `## configure` step 5). A
   solo run produces a misleading hang. For Ethernet-queue-class
   workloads, the traffic generator (host-side `pktgen`, a remote
   sender, the DPU-side `doca-flow` rules feeding the queue) must
   be set up first.
2. **Run the listening side first.** Same shape as
   `doca-rdma TASKS.md ## run` (`doca-rdma/TASKS.md#run`) — the
   listening side must be in its accept-ready state before the
   connecting side starts. Raw verbs does not change that
   ordering.
3. **Capture the structured log.** Set `DOCA_LOG_LEVEL=trace` (or
   `--sdk-log-level DEBUG`) for the first run per the two-tier
   model in
   `doca-common CAPABILITIES.md ## log` (`doca-common/CAPABILITIES.md#log`).
   This is the cheapest way to make verbs-layer lifecycle
   transitions visible on first failure.
4. **Confirm completions are arriving on the chosen path.** A run
   that submits WRs but produces no CQEs is almost always one of:
   a missing `doca_pe_progress` call (DOCA PE path), a missing
   `doca_verbs_bridge_poll_cq` call (manual path), or a missing
   `epoll`/`doca_verbs_get_cq_event` chain (comp-channel path).
   Confirm the chosen path from
   [CAPABILITIES.md ## Observability](CAPABILITIES.md#observability)
   is actually being driven on both sides.

## test

Goal: prove the configured raw-verbs context can actually move data
correctly between the two sides on the user's hardware, and that
the specific verb / opcode / WR flag / QP attribute that justified
the drop-down actually fires the way the user expected.

This is **a loop, not a one-shot pass.** Each iteration narrows
either the cap-query, the QP / WR setup, the completion-handling
path, or the user's no-mixing-with-libibverbs hygiene. The loop
terminates when either (a) the user's intended raw-verbs pattern
flows end-to-end with the expected completions, or (b) the agent
has narrowed the failure cause to a layer outside `doca-verbs`
itself (driver / firmware / network / *the higher-level library
was the right answer all along*) and escalated to the matching
skill.

Iteration shape — the smoke-before-scale principle is
non-negotiable for raw verbs:

1. **One QP, one WR, one completion.** The cheapest possible
   smoke. Bring up exactly ONE `doca_verbs_qp`, post exactly ONE
   WR (matched on the peer where two-sided), drain exactly ONE
   completion through the chosen path
   ([CAPABILITIES.md ## Observability](CAPABILITIES.md#observability)).
   If this fails, do not scale up — narrow.
2. **Re-confirm the cap-query passed for THIS device.** Re-run
   `doca_verbs_query_device` + the matching
   `doca_verbs_device_attr_get_*` for the specific verb / opcode
   / WR flag / QP attribute against the active `doca_devinfo`. If
   false → that's the answer; the user's device or DOCA version
   does not support the verb. Update the user's intent, climb
   back to the matching higher-level library, or update the
   install. Free the handle with `doca_verbs_device_attr_free`.
3. **Verify completion-handling path is wired.** If the agent
   picked the DOCA PE path: confirm `doca_pe_progress` is in the
   user's main loop on both sides. If the agent picked the
   manual CQ poll: confirm the `doca_verbs_bridge_poll_cq` loop
   runs and reads CQEs. If the agent picked the comp-channel
   path: confirm `doca_verbs_req_notify_cq` is called to arm the
   channel, the OS handle from `doca_verbs_comp_channel_get_handle`
   is `epoll`'d, and `doca_verbs_get_cq_event` /
   `doca_verbs_ack_cq_events` complete the cycle. The most common
   failure mode here is *picked one path in `## configure` but
   the user wrote a different path in the main loop*.
4. **Inspect the CQE on `DOCA_ERROR_IO_FAILED`.** Per the
   raw-verbs error overlay in
   [CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy),
   IO_FAILED means *the submit succeeded but the completion
   reports an error*. The answer is in the CQE error field; the
   agent must direct the user there before recommending any
   code-level change.
5. **Confirm the no-mixing-with-libibverbs hygiene.** If the user
   was porting libibverbs code, walk the porting block from
   [`## modify`](#modify) and confirm no leftover `ibv_*` call
   touches an object owned by `doca-verbs`. Mixed handles are the
   highest-cost-to-find raw-verbs bug.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| `DOCA_ERROR_NOT_SUPPORTED` on the verb / attribute / option we expected to work | A `doca_verbs_device_attr_get_*` returned true at configure time, but the runtime rejects the WR or the modify call | Re-run the same `doca_verbs_query_device` + matching `doca_verbs_device_attr_get_*` against the active selected `doca_devinfo`; verify that the queried attribute, selected device, and requested runtime configuration still match. If support is absent, climb back up or update the environment |
| `DOCA_ERROR_IO_FAILED` on the WR | Submit returned `DOCA_SUCCESS`; completion arrives with error status | Stop reading the submit return; read the CQE error field. The cross-cutting taxonomy ladder in `doca-debug ## debug` (`doca-debug/TASKS.md#debug`) takes over from the CQE error |
| Submit succeeded but no completion at all | One of: the PE is not progressed, the `doca_verbs_bridge_poll_cq` loop is missing, the comp-channel is not armed (`doca_verbs_req_notify_cq` not called) or its handle is not `epoll`'d, or the peer disconnected silently | Map to the path picked in [CAPABILITIES.md ## Observability](CAPABILITIES.md#observability); only ONE of PE / manual / comp-channel should be active on this CQ |
| Intermittent `DOCA_ERROR_BAD_STATE` on `doca_verbs_qp_modify` / WR submit | QP state transitions misordered | Re-walk the verbs object lifecycle from [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes); raw verbs exposes the QP state machine directly via `doca_verbs_qp_modify` + `_set_current_state` / `_set_next_state`, and the agent must confirm each transition's precondition |
| Same code that worked yesterday now fails with `NOT_PERMITTED` | RDMA stack module loads / user group / ulimits regressed | Route to `doca-setup TASKS.md ## debug` (`doca-setup/TASKS.md#debug`); this is an env regression, not a verbs code change |
| The user's case turned out to be covered by the higher-level library after all | The drop-down was unnecessary; the smoke surfaced a higher-level alternative | This is a successful outcome of the loop. Climb back up to `doca-rdma/SKILL.md` / `doca-eth/SKILL.md` / `doca-rmax/SKILL.md` and retire the raw-verbs path |

Loop terms are exact: an iteration's **kind** is the `Iteration
trigger` row selected above; its **prescribed change** is that row's
`What changes next iteration` action; and its **evidence** is the
cap-query, QP state, submit result, and selected completion-path
capture taken before and after that change. The outcome is
**resolved** when the named smoke turns green, **shape-changed** when
the re-run selects a different trigger row or materially changes
that evidence, and **unchanged** when the prescribed change was
applied but the re-run selects the same trigger row with the same
outcome and materially unchanged evidence. The baseline occurrence
and that post-change re-run are the two consecutive iterations; on
`unchanged`, stop and escalate to
`doca-debug ## debug` (`doca-debug/TASKS.md#debug`) with the
captured layer-1-through-5 evidence.

## debug

Goal: when a `doca_verbs_*` call returns a `DOCA_ERROR_*` (or the
program doesn't make forward progress), narrow the cause to a
specific layer and act on it.

The cross-library debug ladder lives in
`doca-debug TASKS.md ## debug` (`doca-debug/TASKS.md#debug`).
Walk through it in order — install → version → build → link →
runtime → program → driver — *before* recommending
raw-verbs-specific fixes. This skill's overlay names the
raw-verbs-specific manifestation at layers 5 (runtime) and 6
(program):

**Layer 5 (runtime) — raw-verbs overlay.**

- Walk the QP state machine: was the QP transitioned through every
  state the verbs surface requires before the first WR was posted?
  Out-of-order transitions return `DOCA_ERROR_BAD_STATE`, not a
  self-describing symptom. Query the current state via
  `doca_verbs_qp_query` to confirm where the QP actually is.
- Confirm exactly ONE completion-handling path is active on the
  CQ — the DOCA progress engine OR `doca_verbs_bridge_poll_cq` OR
  the `doca_verbs_comp_channel` chain, never two. Mixed paths
  drop completions silently.
- On `DOCA_ERROR_IO_FAILED`, the submit return is not the answer.
  Direct the user to the CQE error field per
  [CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy).
- Wire `doca_verbs_get_async_event_handle` +
  `doca_verbs_get_async_event` + `doca_verbs_ack_async_event` into
  the run loop if the symptom looks like a device-level event
  (QP fatal error, port state change) — those will not surface on
  the data-plane CQ.

**Layer 6 (program) — raw-verbs overlay.**

- **No mixing with libibverbs.** This is the single
  highest-frequency raw-verbs program bug. Search the user's code
  for any `ibv_*` call that touches an object owned by
  `doca-verbs`; route to the porting block in
  [`## modify`](#modify) when found.
- Lifecycle order: configure → start → use → stop → destroy.
  Out-of-order returns `DOCA_ERROR_BAD_STATE`. The most common
  case in verbs context is destroying an MR or PD before the QP
  that referenced it, or destroying the `doca_verbs_context`
  before the per-object handles it owned.
- Cap-query mismatches: the program's selected device, queried
  attribute, or requested runtime configuration does not match
  the device-level capability snapshot. Re-run the query against
  the active `doca_devinfo` per
  [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes)
  cap-query rule.
- **Climb-back-up check.** Before exhausting a layer-6 debug
  session, ask: *does the higher-level
  `doca-rdma/SKILL.md` /
  `doca-eth/SKILL.md` /
  `doca-rmax/SKILL.md` cover this case? If yes,
  is the raw-verbs path still required?* Sometimes the cheapest
  fix at layer 6 is to drop the raw-verbs path entirely.
- Freed-handle hygiene: confirm `doca_verbs_device_attr_free` was
  called after every `doca_verbs_query_device`. A slow leak shows
  up as memory growth across reconfiguration cycles and is easy
  to miss in short smoke runs.

Once the layer is identified, route to the matching debug verb on
the matching skill: install / build / link / driver to
`doca-setup ## debug` (`doca-setup/TASKS.md#debug`);
cross-cutting runtime to
`doca-debug ## debug` (`doca-debug/TASKS.md#debug`);
program-layer Core-context patterns to
`doca-programming-guide TASKS.md ## debug` (`doca-programming-guide/TASKS.md#debug`);
higher-level RDMA-layer patterns the user might climb back to,
to `doca-rdma TASKS.md ## debug` (`doca-rdma/TASKS.md#debug`);
higher-level Ethernet-queue patterns to
`doca-eth TASKS.md ## debug` (`doca-eth/TASKS.md#debug`);
higher-level Rivermax patterns to
`doca-rmax TASKS.md ## debug` (`doca-rmax/TASKS.md#debug`).

## use

See [references/use.md](references/use.md).

## Deferred task verbs

The full deferred-verb routing lives in [references/deferred-verbs.md](references/deferred-verbs.md). Load that companion rather than inventing a verb this skill does not own.
- **`install` / env prep.** Defer to `doca-setup/SKILL.md`.
- **Higher-level RDMA / Ethernet / Rivermax.** After a working QP, hand off to `doca-rdma/SKILL.md`, `doca-eth/SKILL.md`, or `doca-rmax/SKILL.md`.

## Command appendix

See [references/command-appendix.md](references/command-appendix.md).
