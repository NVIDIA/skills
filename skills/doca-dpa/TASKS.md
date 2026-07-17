# DOCA DPA workflows

**Where to start:** The verbs run `configure → build → modify →
run → test → debug`. Skip ahead only when the user is already
past a verb. The `## test` verb is an iterative loop
(single-kernel-launch smoke → multi-launch streaming → multi-thread
scale-out → loop back if a precondition changes), not a one-shot
pass — see the eval-loop overlay in `## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the DPA capability surface, the
two-side-program model, the `doca_dpa` per-instance context, the
loaded DPA application image, the DPA execution context, the
host-initiated launch + completion model, the dual capability-query
rule, the env-precondition matrix, the error taxonomy, the
observability surface, and the safety policy, see
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

Goal: stand up a `doca_dpa` context against a BlueField with a
DPA processor, load the user's DPA application image (compiled
by `dpacc` from their DPA-side source) onto it, create at least
one DPA execution context (`doca_dpa_thread`), and confirm both
the host side and the DPA side are in a state where a host-side
kernel launch can succeed.

Steps the agent should walk the user through:

1. **Confirm the env preconditions.** Per the env-precondition
   matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy):
   `pkg-config --modversion doca-dpa` returns a version the
   installed DPACC compiler is listed as compatible with under
   the DOCA Compatibility Policy; the target BlueField is
   present and exposes its DPA processor to the host (the
   `doca_dev` enumeration succeeds AND a `doca_dpa_cap_*` query
   against its `doca_devinfo` returns supported for the
   baseline DPA surface); the user / process can open the
   target `doca_dev`. If ANY of these fails, this is an env /
   version / mode problem to fix via
   [`doca-setup TASKS.md ## configure`](../../doca-setup/TASKS.md#configure)
   + [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure),
   NOT a code change in the host-side DPA program.
2. **Walk the two-side-program model BEFORE writing any host
   code.** Per the two-side-program rule in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   the user is writing TWO translation units: a host-side
   program that calls `doca_dpa_*`, and a DPA-side program
   (the kernel function bodies) that `dpacc` compiles into a
   binary the host executable embeds as a `doca_dpa_app`. The
   agent must surface this model EXPLICITLY before any
   host-side launch call is drafted. If the user is asking
   *"how do I write the DPA-side kernel itself"*, that is the
   DPACC + DPA-side libraries surface — route via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
   to the public DOCA DPA / DPACC / DPA-Comms / DPA-Verbs
   guides; do not redefine those surfaces here.
3. **Run dual capability discovery against the target
   BlueField.** Per the dual-axis rule in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes):
   on the DOCA side, run the matching `doca_dpa_cap_*` against
   the active `doca_devinfo` for the BlueField the host is
   driving; on the install side, confirm `pkg-config
   --modversion doca-dpa` agrees with `doca_caps --version`
   and with the installed `dpacc` version per the DOCA
   Compatibility Policy. Quote BOTH results back to the user.
   If either says *not supported*, that axis is the answer —
   do not proceed.
4. **Create the `doca_dpa` Core context against the chosen
   `doca_dev`.** This is a standard DOCA Core context create —
   the universal lifecycle from
   [`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure)
   applies. Choose the `doca_dev` deliberately: a multi-BlueField
   host needs one `doca_dpa` per BlueField, not a "global" one,
   per the per-instance rule in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
5. **Load the DPA application image and create DPA execution
   contexts.** Load the `doca_dpa_app` produced by `dpacc` from
   the user's DPA-side source into the `doca_dpa`; then create
   one or more `doca_dpa_thread` execution contexts on top.
   Start the `doca_dpa` via `doca_ctx_start()`. Per the
   lifecycle ordering in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   record the order so teardown happens in reverse: stop /
   destroy `doca_dpa_thread` → release the loaded
   `doca_dpa_app` → destroy the `doca_dpa` → close the
   `doca_dev`.
6. **Attach a `doca_dpa_completion` before launching anything.**
   Per the launch + completion section in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   `doca_dpa_kernel_launch_update_*` is asynchronous from the
   host's point of view — without a completion attached, the
   host has no portable way to know the DPA kernel actually
   finished. Set this up at configure time, not at first-failure
   time.
7. **Sanity check before the first launch.** Confirm with the
   user: which BlueField (which `doca_dev`) the launches will
   target; which DPA application image (`doca_dpa_app`) is
   loaded and which kernel function entry points it exposes;
   which `doca_dpa_thread` each launch will run on; how the
   host launch arguments map to the DPA-side function
   signature; how the kernel will terminate (a DPA kernel that
   never exits will eventually pin the DPA processor and the
   host will see no completion). If any of those are unclear,
   stop and ask — do not invent.

For the canonical DOCA universal lifecycle that underlies
steps 4-5, see
[`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure).
This skill adds the DPA overlay; do not re-explain the
lifecycle here.

## build

Goal: compile a DPA-using consumer (host-side C / C++ + at
least one DPA-side translation unit) against the user's
installed DOCA + DPACC compiler, with `pkg-config` + `dpacc`
as the joint sources of truth for include + link flags + DPA
binary embedding.

The build pattern for any DOCA C / C++ consumer is fully
documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
DPA adds a DPACC layer on top: the DPA-side translation units
are compiled by `dpacc` into a binary embedded into the host
executable, and the link line on the host side pulls in the
DOCA DPA library. This skill carries only the DPA-specific
overlay:

| Slot | Value | Why it matters |
| --- | --- | --- |
| `pkg-config` module name | `doca-dpa` | The library's `.pc` file installed by the DOCA host packages; gives the host-side include + link flags for the DOCA DPA host API |
| DPA-side toolchain | `dpacc` (DPACC compiler), installed alongside DOCA | Compiles DPA-side translation units into the binary that the host executable embeds as the `doca_dpa_app`. The host system compiler is NOT a substitute |
| Include flags | `pkg-config --cflags doca-dpa` for the host side; the DPACC-prescribed include path for the DPA side | Resolves DOCA headers under whichever include directory `pkg-config --cflags` reports on this install (do not hardcode the include path — it can move across DOCA install profiles) for the host side; the DPA-side include path comes from the DPACC install layout via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) |
| Link flags | `pkg-config --libs doca-dpa` on the host side, plus the DPACC-prescribed embed step that bakes the DPA-side binary into the host executable | Pulls in whatever `pkg-config --libs` resolves on this install (do not predict the `-l<name>` form by hand — `.so` basenames use underscores, `.pc` names use hyphens, and `pkg-config` is the only correct translator) on the host side. The DPACC embed step is the load-bearing build action — without it the host has no DPA application image to load at runtime |
| Companion DOCA libs | `doca-argp` for arg parsing in samples; the shipped samples include both host-side and DPA-side translation units | The shipped samples are the verified two-side-program build template; do not invent a one-side-only manifest |
| DPA device-side archives NOT to add to the host link line | `libdoca_dpa_dev_comm.a`, `libdoca_dpa_dev_verbs.a` | These are DPA-SIDE archives shipped within `doca-dpa` (NOT separate pkg-config modules): their symbols are linked into the DPA image from inside the DPA kernel by `dpacc`, NOT from the host link line. Adding them to the host link line is a common first-build error |
| Minimum DOCA version | Query with `pkg-config --modversion doca-dpa`; never hardcode | Cross-version build/runtime mixing breaks per [CAPABILITIES.md ## Version compatibility](CAPABILITIES.md#version-compatibility) |
| Minimum DPACC version | Cross-check the installed `dpacc` version against the DOCA Compatibility Policy linked from [CAPABILITIES.md ## Version compatibility](CAPABILITIES.md#version-compatibility) | Mismatched DOCA + DPACC combos fail at link time or at launch time with `DOCA_ERROR_DRIVER` |

For non-C host-side consumers (Rust, Go, Python) that drive
host-side DPA setup and embed a DPA application image built
separately by `dpacc`, the host-side link line and version
rules above still apply; the DPA-side build is a separate
compilation unit and is out of scope for this skill, but the
`dpacc` version check is the load-bearing input the wrapper
still needs.

## modify

Goal: take the closest-fitting shipped DOCA DPA sample as the
verified starting point and apply a **minimum diff** to make
it match the user's intent, without rewriting from scratch.

The universal modify-a-shipped-sample workflow is in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify);
this skill provides the DPA-specific slot fill.

| Slot | What the agent asks the user | DPA-specific consideration |
| --- | --- | --- |
| 1. Starting sample | Which sample under `/opt/mellanox/doca/samples/doca_dpa/`? | Pick a sample whose **shape** matches the user's intent: same kernel-launch-style (single launch vs streaming), same use of `doca_dpa_completion`, same DPA device-side component set (vanilla, or comm via `libdoca_dpa_dev_comm.a`, or verbs via `libdoca_dpa_dev_verbs.a`). DPA samples are two-side programs; the sample's DPA-side translation unit is the second half of the verified base |
| 2. DPA-side kernel function body | What is the DPA kernel function actually computing? | The DPA-side translation unit is the in-place edit point for the kernel body. The agent's anti-pattern alert: do NOT propose moving the kernel logic to the host side to *"simplify"* — that defeats the entire reason to use DPA. Keep the kernel on the DPA |
| 3. Host launch-argument shape | What arguments does the kernel take, what shapes / sizes? | Per the two-side-program rule in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes), the host launch call and the DPA-side kernel signature MUST agree on count, size, and type. Any change to one side requires updating the other; track this as a single edit, not two |
| 4. Number of `doca_dpa_thread` execution contexts | One DPA thread or many? Persistent or per-launch? | Per the anti-pattern note in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes), fresh-thread-per-launch defeats the DPA. Persistent threads processing a stream of launches are the right shape; multi-thread is a re-architecture, not a tweak |
| 5. Completion topology | How does the host observe each launch completing? | The `doca_dpa_completion` attached at configure time is the host-side observability surface. Resizing or restructuring the completion queue means re-running step 6 of [`## configure`](#configure); a missing or undersized completion queue surfaces as `DOCA_ERROR_AGAIN` at launch submit time |
| 6. Termination signal for the DPA kernel | How does the kernel know when to exit? | A DPA kernel that runs forever pins the DPA processor and the host sees no completion. The agent must surface the kernel-exit condition on every modify pass — a host-side flag in DPA-visible memory, a per-launch one-shot, or whatever the sample's pattern is |
| 7. Rebuild BOTH sides | After any modify, rebuild the DPA-side image via `dpacc` AND rebuild the host executable that embeds it | Per the *do not partial-rebuild one side* rule in [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy), rebuilding only one side is the canonical way to introduce `DOCA_ERROR_DRIVER` at launch time |

The agent emits an *intent description + the seven filled
slots*; the *actual* unified diff against the sample source is
produced the way every other library skill in this bundle
handles modify — the agent walks the user through the diff
line-by-line against the sample source they read on disk, and
has the user paste back the result for validation. The agent's
anti-pattern alert: a *"clean rewrite"* from scratch is almost
always slower to first green than a minimum-diff modify on a
shipped DPA sample, and removes the user's ability to bisect
against a known-good baseline.

## run

Goal: actually launch a DPA kernel from the host against the
loaded DPA application image and confirm the launch completes.

Steps the agent should walk the user through:

1. **Confirm the loaded image exposes the kernel function the
   host will launch.** The host's view of *"which kernels
   exist"* is exactly the entry-point set in the loaded
   `doca_dpa_app`. If the user's host launch call names a
   function `dpacc` did not compile in, that is a build-side
   bug, not a launch-side bug; route back to [`## build`](#build).
2. **Submit the launch from the host** via the matching
   `doca_dpa_kernel_launch_update_*` call against the chosen
   `doca_dpa_thread`. Remember: the call is asynchronous; a
   successful return means the launch was submitted, not that
   the kernel finished. Per the launch + completion section
   in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   the host must drain the attached `doca_dpa_completion` to
   know the kernel actually finished.
3. **Drive the host-side `doca_pe_progress` loop** in parallel
   with any outstanding DPA launches. Completions arrive
   through the progress engine; a host that submits launches
   and then blocks without progressing the PE will see no
   completions and conclude *"the DPA is hung"* incorrectly.
   This is the cross-library *"PE not progressed"* failure
   mode applied to DPA.
4. **Capture the structured log on first failure.** Set
   `DOCA_LOG_LEVEL=trace` for the first run (see
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability));
   if the host-side log is silent but the launch never
   completes, the DPA kernel is running but stuck — that is the
   moment to reach for the DPA-side developer tools (the DPA
   debugger and the DPA process-state inspector named in
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability)
   and routed via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
   to the public *DPA Tools* umbrella).

For the runtime version + `LD_LIBRARY_PATH` cross-checks that
underlie *"the program built but does nothing"*, see
[`doca-version TASKS.md ## run`](../../doca-version/TASKS.md#run).

## test

Goal: prove the configured DPA setup actually launches a kernel
on the DPA processor and reports a completion to the host on
the user's hardware, and that the env-precondition +
capability set was sized right.

This is **a loop, not a one-shot pass.** Each iteration
narrows either the env-precondition set, the capability set,
the two-side-program signature, or the completion-queue
sizing. The loop terminates when either (a) the host launches
a kernel at the user's intended rate with the expected
completions, or (b) the agent has narrowed the failure cause
to a layer outside DPA itself (DPA-side kernel hang / DPACC
build / BlueField mode / DOCA install) and escalated to the
matching skill.

Iteration shape:

1. **Single-kernel-launch smoke.** Launch ONE invocation of a
   trivial DPA kernel (no-op, or a counter increment) and
   confirm the host observes one completion. If yes, advance.
   If no — and the host-side log shows no error — the DPA
   kernel is running but stuck or the completion queue is not
   wired up; route to [`## debug`](#debug) layer 5 and consult
   the DPA-side tooling. Per the safety-policy rule in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   DO NOT scale a broken smoke into a high-throughput design.
2. **Streaming-launch loop.** Submit 100 launches in a tight
   loop and confirm the host observes all 100 completions
   without losing any. Catches completion-queue-sizing bugs:
   `DOCA_ERROR_AGAIN` from the host-side submit means the
   completion queue is full and the host must drain faster,
   OR the queue is undersized — per [`## modify`](#modify) slot
   5.
3. **Multi-thread scale-out (if used).** Re-run steps 1-2 on
   each additional `doca_dpa_thread` — each thread is its own
   execution context and can fail differently (e.g. different
   completion-queue topology, different per-thread state). Per
   the persistent-thread rule in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   one thread per stream of launches is the right pattern.
4. **Two-side-program signature negative test.** Intentionally
   pass mismatched arguments from the host launch call versus
   the DPA-side kernel signature and confirm the host gets
   `DOCA_ERROR_INVALID_VALUE` cleanly — validates that the
   agent's earlier two-side-program signature check is real,
   not just notional. Then restore the matched signatures.
5. **Cap-query negative test.** Intentionally call a
   `doca_dpa_cap_*` axis the agent expects to be *not
   supported* on this BlueField + DOCA combo and confirm the
   reported `DOCA_ERROR_NOT_SUPPORTED` matches the dual
   capability discovery from [`## configure`](#configure) step
   3 — validates the agent's capability-discovery itself is
   correct.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| `DOCA_ERROR_NOT_SUPPORTED` on a BlueField the agent expected to work | DOCA cap-query failed against the active `doca_devinfo` | The BlueField generation / mode axis was missed; re-run the env-side BlueField mode check; if the BlueField truly does not expose the DPA, the answer is the hardware, not a code or DOCA upgrade |
| `DOCA_ERROR_DRIVER` on first kernel launch | DOCA + DPACC versions are skewed OR the DPA-side image was built against a different DOCA install than the host runtime | Re-run the version chain per [`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test); rebuild BOTH sides via `dpacc` + the host build against the matched versions; cross-check against the DOCA Compatibility Policy |
| Single-kernel-launch smoke passed; streaming loop hangs | The host is not draining the completion queue fast enough OR the completion queue is too small | Raise the completion queue size in [`## modify`](#modify) slot 5; or restructure the host loop to drain completions per batch |
| Host submitted N launches; received fewer than N completions; host log is silent | The DPA-side kernel is stuck mid-execution (most often a missing termination condition in the kernel body) | Walk the kernel-exit condition per [`## modify`](#modify) slot 6; reach for the DPA-side debugger / process-state inspector named in [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability) |
| `doca_ctx_stop` blocks on teardown | A `doca_dpa_thread` still has an in-flight kernel that never exited | Walk the host-side flag (or other termination signal) the DPA kernel polls per [`## modify`](#modify) slot 6; the host cannot force-kill the DPA kernel from the `doca-dpa` API |

Loop termination: stop iterating once two consecutive
iterations of the same kind don't change anything — that means
the cause is below DPA (BlueField mode, DPA-side kernel logic
bug, DPACC bug, NIC firmware). Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
with the captured layer-1-through-5 evidence including BOTH
the host-side DOCA log and the DPA-side tooling output.

## debug

Goal: when a `doca_dpa_*` call (or a DPA kernel launch) returns
a `DOCA_ERROR_*` or does not make forward progress, narrow the
cause to a specific layer and act on it.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver — *before* recommending DPA-specific
fixes. This skill's overlay names the DPA-specific
manifestation at layers 5 (runtime), 6 (program), and 7
(driver):

**Layer 5 (runtime) — DPA overlay.**

- A submitted DPA kernel launch that never produces a host-side
  completion is *almost always* one of three things: the host
  is not progressing the PE; the DPA kernel is in an infinite
  loop with no termination condition; or the completion queue
  attached to the launch's `doca_dpa_thread` is full and being
  silently dropped at the host's failure to drain. Confirm the
  env-side preconditions per [`## configure`](#configure) step
  1 and the host-side progress per [`## run`](#run) step 3
  before assuming the DPA itself is broken.
- `DOCA_ERROR_AGAIN` from `doca_dpa_kernel_launch_update_*` is
  *always* a queue-full / drain-rate problem. Do not recommend
  a tight retry loop; recommend a drain-then-retry pattern via
  the host PE.
- A *"DPA kernel hung; cannot stop the program"* pattern is
  almost always a missing termination signal in the DPA-side
  kernel per [`## modify`](#modify) slot 6.

**Layer 6 (program) — DPA overlay.**

- Lifecycle ordering: the `doca_dpa_thread` objects must be
  destroyed BEFORE the loaded `doca_dpa_app` is released; the
  loaded `doca_dpa_app` must be released BEFORE the
  `doca_dpa` itself is destroyed. Out-of-order returns
  `DOCA_ERROR_BAD_STATE` per
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
- Two-side-program signature mismatch: a host launch call whose
  argument shape does not match the DPA-side kernel function
  signature returns `DOCA_ERROR_INVALID_VALUE`. The fix is to
  rebuild the DPA-side image via `dpacc` after re-aligning the
  signatures, then rebuild the host executable; do not patch
  only one side. See the *do not partial-rebuild one side* rule
  in [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
- Fresh-thread-per-launch anti-pattern: if the user reports
  *"throughput is much lower than expected"* and the host code
  creates a new `doca_dpa_thread` per launch, that is the bug.
  Refactor to persistent-thread + stream-of-launches per the
  anti-pattern note in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).

**Layer 7 (driver) — DPA overlay.**

- `DOCA_ERROR_DRIVER` from `doca_dpa_*` is most often the DPA
  driver layer reporting failure to DOCA, and most often
  because DOCA + DPACC are skewed. Capture `pkg-config
  --modversion doca-dpa`, the installed `dpacc` version, and
  `doca_caps --version`; cross-check against the DOCA
  Compatibility Policy at
  <https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html>.
- A BlueField in the wrong mode that does NOT expose the DPA
  surfaces as `DOCA_ERROR_NOT_SUPPORTED` at `doca_dpa` create
  time. Route to
  [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug)
  layer 5 (driver) for the env-side BlueField mode fix; this
  is NOT a code change in the program.

Once the layer is identified, route to the matching debug verb
on the matching skill: install / build / link / driver to
[`doca-setup ## debug`](../../doca-setup/TASKS.md#debug); version
to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug);
cross-cutting runtime to
[`doca-debug ## debug`](../../doca-debug/TASKS.md#debug);
program-layer Core-context patterns to
[`doca-programming-guide TASKS.md ## debug`](../../doca-programming-guide/TASKS.md#debug).

## comms

The DPA-Comms-specific overlay on the parent verbs. Use AFTER the
parent's [`## configure`](#configure) → [`## debug`](#debug)
sequence has been walked for the host-side `doca-dpa` flow; this
section adds only what the DPA-side comms layer changes. For the
capability surface, routing rule, error overlay, and safety
matrix, see [`CAPABILITIES.md ## comms`](CAPABILITIES.md#comms).

**configure overlay.**

1. **Confirm DPA-side scope, not host-side.** The user is writing
   the DPA-side translation unit `dpacc` compiles into the
   `doca_dpa_app`. If the user wanted host ↔ DPU messaging, that
   is [`doca-comch`](../doca-comch/SKILL.md); if they wanted
   host-to-remote RDMA, that is
   [`doca-rdma`](../doca-rdma/SKILL.md). Route there before any
   DPA-Comms code per the routing rule in
   [`CAPABILITIES.md ## comms`](CAPABILITIES.md#comms).
2. **Parent host-side flow must be green first.** A trivial DPA
   kernel (no DPA-Comms calls) must already launch and complete
   on this host + image per [`## test`](#test) step 1 — the
   parent smoke. A broken parent flow surfaces as a broken
   DPA-Comms launch later. Fix via [`## configure`](#configure)
   first; do NOT start DPA-side comms code on a broken parent.
3. **Confirm the DPA is exposed and the install is matched.** Per
   the capability rule in
   [`CAPABILITIES.md ## comms`](CAPABILITIES.md#comms), there is
   NO `doca_dpa_comms_cap_*` host cap family — host-side DPA
   discovery is only `doca_dpa_cap_is_supported` /
   `doca_dpa_cap_get_max_kernel_time_alive_supported`. Run
   `doca_dpa_cap_is_supported` against the active `doca_devinfo`
   to confirm a DPA is exposed, then confirm the comm primitives
   the kernel will use exist on this BlueField generation +
   matched install by reading `doca_dpa_dev_comch_msgq.h` and the
   shipped sample. Cross-check the single
   `pkg-config --modversion doca-dpa` agrees with the installed
   `dpacc` (there is no separate `doca-dpa-comms.pc`); quote both
   back to the user.
4. **Read the matching shipped sample first.** The verified
   two-side-program baseline lives under
   `/opt/mellanox/doca/samples/doca_dpa/`. Read it on disk
   BEFORE writing any DPA-side comms code; do NOT re-derive the
   kernel-side initialization order from memory.

**build overlay.**

| Slot | Value |
| --- | --- |
| DPA-side archive | `libdoca_dpa_dev_comm.a` (header `doca_dpa_dev_comch_msgq.h`) — part of `doca-dpa`, linked into the DPA image by `dpacc`; NOT a separate pkg-config module |
| Host link line | **Do NOT add `libdoca_dpa_dev_comm.a`**. The host link line stays as the parent [`## build`](#build) prescribes (`-ldoca-dpa -ldoca-common`). There is no host-side DPA-Comms cap call to include a header for; the substantive send / receive / signal calls live DPA-side |
| Version anchors | The single `pkg-config --modversion doca-dpa` MUST agree with the installed `dpacc` per the DOCA Compatibility Policy (there is no separate `doca-dpa-comms.pc`) |

**modify overlay.** Take the closest-fitting sample under
`/opt/mellanox/doca/samples/doca_dpa/` and apply a minimum
diff. Slot fill on top of the parent's [`## modify`](#modify):

- Pick the DPA-Comms primitive family the kernel uses
  (send / receive vs signal / event — separate setups; do NOT
  mix without re-reading the sample's setup).
- The host-side launch argument shape carries the DPA-Comms
  endpoint handles; the host launch call and the DPA-side
  kernel signature MUST agree per the parent's two-side-program
  coupling rule.
- If the modify introduces a new comm primitive, re-check
  `doca_dpa_dev_comch_msgq.h` and the shipped sample that it
  exists on this BlueField generation + matched install (there
  is no host `doca_dpa_comms_cap_*` query to call); otherwise the
  first kernel launch surfaces `DOCA_ERROR_NOT_SUPPORTED` on the
  host completion.
- Kernel-side cooperative back-off on `DOCA_ERROR_AGAIN`: the
  kernel must yield (return from the launch, let the host drain
  via `doca_pe_progress`, re-submit on the next launch) OR
  cooperatively back off if the kernel is persistent. A tight
  in-kernel retry loop pins the DPA processor and starves the
  host drain.
- Rebuild BOTH sides: DPA-side image via `dpacc` AND the host
  executable that embeds it.

**run / test overlay.** Per the safety matrix in
[`CAPABILITIES.md ## comms`](CAPABILITIES.md#comms):

1. **One-send-one-receive smoke** between two DPA threads in the
   same loaded `doca_dpa_app`. Thread A issues ONE DPA-Comms
   send on an endpoint handle, thread B issues a matching
   receive on the same handle, the host observes the launch's
   completion through `doca_dpa_completion` with no error. The
   parent smoke (a trivial kernel with NO DPA-Comms calls) MUST
   already pass first — if it does not, fix the parent skill,
   not the DPA-Comms code.
2. **Multi-message loop** once the smoke is green: 100 sends,
   confirm the host sees all matching completions without
   losing any. Catches `_AGAIN`-handling bugs — if the kernel
   does not yield correctly on a full DPA-side comms queue,
   throughput collapses or the host completion stream stalls.
3. **Unsupported-primitive negative test**: write a kernel that
   calls a comm primitive the device header / sample shows is
   NOT on this BlueField generation + matched install, confirm
   `DOCA_ERROR_NOT_SUPPORTED` cleanly on the launch's completion
   — validates the agent read the supported-primitive surface
   correctly (there is no host cap-query to commit a budget).

**debug overlay.** Layered on the parent's [`## debug`](#debug)
ladder:

- `DOCA_ERROR_AGAIN` on a `doca_dpa_completion` for a
  DPA-Comms-calling kernel is *always* the DPA-side comms queue
  full. Recommend the cooperative back-off per the modify
  overlay above; confirm the host is draining the parent's
  progress engine. Do NOT recommend a tight in-kernel retry.
- `DOCA_ERROR_NOT_SUPPORTED` is *always* a
  hardware-generation / matched-install mismatch. Confirm the
  DPA is exposed via `doca_dpa_cap_is_supported` and re-check the
  primitive in `doca_dpa_dev_comch_msgq.h` / the shipped sample;
  there is no `doca_dpa_comms_cap_*` query to re-run.
- `DOCA_ERROR_BAD_STATE` from a DPA-Comms call must be
  disambiguated from the parent's `_BAD_STATE`. The parent
  meaning is *host-side `doca_dpa` lifecycle violated*; the
  DPA-Comms meaning is *the DPA kernel called a comms
  primitive before its in-kernel comms surface was usable, or
  after kernel-side teardown*. Walk the kernel-side
  initialization order in the shipped sample first.
- A DPA-Comms receive with no matching sender — or a
  signal-wait with no signaler — pins the kernel and the host
  sees no completion. Same shape as the parent's missing-
  kernel-exit case in [`## modify`](#modify) slot 6, layered
  with DPA-Comms semantics.

## verbs

The DPA-Verbs-specific overlay on the parent verbs. EVERY workflow
below begins by re-confirming two things from
[`CAPABILITIES.md ## verbs`](CAPABILITIES.md#verbs): (a) the
4-way-matrix decision places the user in this corner (DPA-side,
raw verbs, latency-bound) AND (b) the parent host-side `doca-dpa`
flow is in scope. If either fails, stop and route back — do not
walk these overlays.

**configure overlay.**

1. **Re-confirm the 4-way-matrix decision.** Ask the user to name
   the specific RDMA op (RDMA read of a remote buffer, send to a
   peer, atomic CmpSwap, …) the DPA kernel will issue. This is
   the cheapest place to catch the *"recommended DPA-side verbs
   unnecessarily"* failure mode. If the user has not walked the
   matrix in
   [`CAPABILITIES.md ## verbs`](CAPABILITIES.md#verbs), route
   back there before continuing.
2. **Confirm the host round-trip is the measured bottleneck.**
   Ask for the latency data (profile, histogram, or per-op
   cost). If the user cannot point at a measurement, ask them
   to take one BEFORE writing any DPA-side code. A guess does
   not count. If the bottleneck is NOT the host round-trip,
   climb back up to [`doca-rdma`](../doca-rdma/SKILL.md) (or
   [`doca-verbs`](../doca-verbs/SKILL.md) for the host-side
   raw-verbs escape hatch when `doca-rdma` does not expose the
   verb the user needs).
3. **Confirm the SPECIFIC verb is exposed on this device.** There
   is NO `doca_dpa_verbs_cap_*` host cap family. From host code,
   confirm a DPA is exposed via `doca_dpa_cap_is_supported`
   against the active `doca_devinfo`, then confirm the specific
   verb / opcode the kernel will post exists on this BlueField
   generation + matched DOCA/DPACC install by reading
   `doca_dpa_dev_verbs.h` and the shipped sample. The DPA-side
   translation unit cannot cap-query from inside the kernel, and
   there is no host-side per-verb query to substitute. If the
   verb is not exposed, that is the answer — the hardware does
   not expose it. Climb back up if a host-side alternative covers
   the case.
4. **Bring up the parent flow AND configure the RDMA QP(s) on
   the host.** Per the host-configures-QP / DPA-uses-QP rule in
   [`CAPABILITIES.md ## verbs`](CAPABILITIES.md#verbs), the QPs
   the DPA kernel posts against are created and configured by
   host code through [`## configure`](#configure). Layer the QP
   create + configure on top per the transport the user picked
   (IB / RoCE). The DPA kernel receives QP handles as launch
   arguments or via DPA-visible memory; the agent must name the
   mechanism explicitly per the parent's two-side-program rule.
5. **Read the DPA-side verbs symbols from the user's install.**
   `doca_dpa_dev_verbs_*` symbols are install-bound on the DPA
   side; the agent must NOT quote them from memory. Direct the
   user to the DPA-side header `doca_dpa_dev_verbs.h` (per the
   DPACC guide via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md))
   and to `/opt/mellanox/doca/samples/doca_dpa/`. Headers
   win over docs.
6. **Pick the completion topology.** Host-side CQE inspection
   (default — the parent's `doca_dpa_completion` surfaces it)
   OR in-kernel completion polling on the DPA side (when the
   kernel must react to its own completions before exiting).
   Pick one per use case explicitly; do NOT mix them on the
   same CQ.

**build overlay.**

| Slot | Value |
| --- | --- |
| DPA-side archive | `libdoca_dpa_dev_verbs.a` (header `doca_dpa_dev_verbs.h`) — part of `doca-dpa`, linked into the DPA image by `dpacc`; NOT a separate pkg-config module |
| Host link line | **Do NOT add `libdoca_dpa_dev_verbs.a`**. The host link line stays as the parent [`## build`](#build) prescribes (`-ldoca-dpa -ldoca-common`). Adding the DPA-side verbs archive to the host link line is the *"link line built, but my host program does not call any DPA-Verbs function"* dead weight |
| DPACC step | The DPA-side translation unit calling the `doca_dpa_dev_verbs_*` primitives is compiled by `dpacc` into the binary embedded as the `doca_dpa_app`. The host system compiler is NOT a substitute |
| Version anchors | The single `pkg-config --modversion doca-dpa` MUST agree with `doca_caps --version` and the installed `dpacc` per the DOCA Compatibility Policy (there is no separate `doca-dpa-verbs.pc`) |

**modify overlay.** Pick a sample under
`/opt/mellanox/doca/samples/doca_dpa/` whose shape matches
the user's intent (same DPA-side WR opcode pattern; same
completion topology; same host-configured QP transport). Slot
fill on top of the parent's [`## modify`](#modify):

- Host-side QP configuration changes live in the host-side
  translation unit. For each added opcode / WR flag, confirm it
  is exposed on this device by reading `doca_dpa_dev_verbs.h` /
  the shipped sample BEFORE the kernel is rebuilt (there is no
  `doca_dpa_verbs_cap_*` host query to call).
- DPA-side WR construction changes (opcode, target QP handle,
  memory region, flags, payload size) live in the DPA-side
  translation unit. Do NOT propose moving the WR construction
  to the host to "simplify" — that defeats the entire reason
  to use DPA-side verbs. Keep the post in the kernel.
- The host launch call and the DPA-side kernel signature MUST
  agree on count, size, and type of QP handles / buffer
  addresses / peer info. Track any change as a single edit
  across both sides, not two.
- Rebuild BOTH sides via `dpacc` + host build per the parent's
  *do not partial-rebuild one side* rule. Rebuilding only one
  side is the canonical way to introduce `DOCA_ERROR_DRIVER`
  at launch or `DOCA_ERROR_INVALID_VALUE` at the first
  DPA-side WR post.

**run / test overlay.** Per
[`CAPABILITIES.md ## verbs`](CAPABILITIES.md#verbs):

1. **One DPA kernel launch, one DPA-side WR post, one
   completion.** Host configures ONE QP, launches the kernel
   ONCE, the kernel posts exactly ONE WR (e.g. an RDMA read of
   a small buffer; matched on the peer when two-sided), the
   host drains exactly ONE completion through the chosen
   completion surface. If this fails, do NOT scale — narrow.
2. **Re-confirm the specific verb is exposed on THIS device.**
   Confirm via `doca_dpa_cap_is_supported` that the DPA is
   exposed and re-check `doca_dpa_dev_verbs.h` / the shipped
   sample for the specific verb / opcode the kernel posts (there
   is no `doca_dpa_verbs_cap_*` host query). If it is not
   exposed, climb back up.
3. **Verify host-configures-QP / DPA-uses-QP coupling.** If the
   kernel constructs a WR whose opcode / flag / payload size
   the host-configured QP cannot honor,
   `DOCA_ERROR_INVALID_VALUE` is the result; fix the side that
   does not match and rebuild BOTH.
4. **On `DOCA_ERROR_IO_FAILED`, inspect the CQE.** The DPA-side
   post return value is NOT the answer; the CQE error field on
   the chosen completion surface IS.
5. **Streaming-launch loop ONLY after smoke is green**, layering
   the parent's streaming-launch pattern from [`## test`](#test)
   step 2 with the DPA-side WR repetition.

**debug overlay.** Layered on the parent's [`## debug`](#debug)
ladder:

- A DPA kernel launched, the kernel posts a `doca_dpa_verbs_*`
  WR, and no completion ever surfaces: (a) the host is not
  progressing the PE / draining the completion (parent's
  [`## run`](#run) step 3); (b) the kernel is stuck mid-post
  or mid-poll with no termination condition; (c) the
  host-configured QP silently dropped the WR. Confirm the
  preconditions BEFORE assuming the DPA-side verbs surface
  itself is broken.
- On `DOCA_ERROR_IO_FAILED` surfaced through the completion,
  the DPA-side post return value is not the answer. Direct
  the user to the CQE error field — same shape as the host-side
  raw-verbs [`doca-verbs`](../doca-verbs/SKILL.md) IO_FAILED.
- **Two-side-program signature mismatch** is the single
  highest-frequency DPA-Verbs program bug. The host configured
  a QP that cannot honor the WR the kernel posts (opcode not
  in the QP feature set, payload past the QP's max message
  size, flag not supported by the transport). Walk both sides
  together; rebuild BOTH.
- **DPA-side post against an unready QP** returns `_BAD_STATE`.
  The host-side `doca-dpa` setup created the QP but has not
  transitioned it through the state machine required to accept
  WRs. The DPA side cannot manufacture readiness — fix on the
  host.
- **Climb-back check.** Before exhausting a layer-6 debug
  session, ask: did the latency-bottleneck premise actually
  hold? Sometimes the cheapest fix is to retire the DPA-side
  RDMA path entirely for the affected subset and route the
  work back to host-side [`doca-rdma`](../doca-rdma/SKILL.md).

## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as
follows so the agent does not invent guidance:

- **install.** Installing DOCA, installing the DPACC compiler,
  choosing matched versions, post-install verification,
  `pkg-config` wiring — defer to
  [`doca-setup`](../../doca-setup/SKILL.md) and to the
  install-tree layout in
  [`doca-public-knowledge-map ## Layout of an installed DOCA package`](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
  This skill assumes DOCA + DPACC are already installed and
  matched.
- **deploy.** Deploying DPA-using applications at scale
  (multi-BlueField clusters, Kubernetes operator workflows
  for DPU workloads, multi-tenant DPA sharing) — out of scope
  for Phase 1 and reserved for a future platform skill. For
  single-host first-run testing, the right verb in this skill
  is `## run`; do not invent a "deploy" workflow.
- **DPA-side kernel programming and DPACC usage.** Writing the
  kernel function body itself, DPA-side memory layout,
  DPACC compile flags, DPA-side debugging from inside the
  kernel — out of scope. Route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  to the public *DOCA DPA*, *DOCA DPACC Compiler*, *DOCA DPA
  Comms*, and *DOCA DPA Verbs* guides plus the *DPA Tools*
  umbrella. This skill prescribes how to *use* the
  DPACC-produced image from the host; it does not redefine
  how to produce it.
- **DPA device-side comm / verbs component surfaces
  (`libdoca_dpa_dev_comm.a` / `libdoca_dpa_dev_verbs.a`).**
  These are DPA-side archives shipped within `doca-dpa` (NOT
  separate pkg-config modules): their symbols are called from
  inside the DPA kernel and linked by `dpacc`, not from the
  host. Route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  to the public *DOCA DPA Comms* and *DOCA DPA Verbs* guides.

## Command appendix

Every command below is **cross-cutting on DOCA DPA** — it
answers a recurring class of question that comes up in the
verbs above. The agent should treat the *class* as
load-bearing; the worked example is a single instance. Run-as
user is the unprivileged user unless noted. Rows that need elevated
privileges call that out explicitly.

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
| `pkg-config --modversion doca-dpa` | `## configure` step 1; `## build` minimum-version slot | What is the build-time DOCA DPA host-side library version? | A semver string matching `doca_caps --version`. Disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2) |
| `pkg-config --cflags --libs doca-dpa` | `## build` | What include + link flags does the host-side linker need? | Trust whatever `pkg-config --cflags --libs` produces on this install. Do not hardcode either the `-I` include path or the `-l<name>` flag form — both can drift between DOCA install profiles and DOCA majors; the on-disk `.so` basenames use underscores on every release where we have ground truth, while the `.pc` package names use hyphens, and `pkg-config` is the only thing that resolves both correctly. Hand-crafted `-l` lines silently break when DOCA upgrades. |
| `which dpacc && dpacc --version` (or the install-tree path) | `## configure` step 1; `## build` minimum-DPACC slot | Is the DPACC compiler installed and at what version? | A version string the agent compares against the DOCA Compatibility Policy linked from [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility). Missing `dpacc` = the DPA-side translation unit cannot be built; route to [`doca-setup`](../../doca-setup/SKILL.md) |
| `doca_caps --list-devs` | `## configure` step 1; `## configure` step 3 | Which DOCA devices does the host see, and which expose a DPA processor? | One entry per `doca_dev` with the BlueField identity and the per-library capability flags including the DPA support axis. No DPA-capable entry = the BlueField is not present, not in the right mode, or not on a generation that exposes the DPA; route to [`doca-setup`](../../doca-setup/SKILL.md) |
| `ls /opt/mellanox/doca/samples/doca_dpa/` | `## modify` slot 1 | Which DPA samples ship in this install (both host-side AND DPA-side translation units), and which is the closest starting point? | A list of sample directories that each contain BOTH host-side and DPA-side source plus a `meson.build` that wires `dpacc` and `pkg-config doca-dpa` together |
| `dmesg \| tail -n 40` (sudo) | `## debug` layer 7 | What did the kernel / driver log around the last DPA call? | Empty or recent benign messages. Repeated mlx5 / DPA-driver errors → driver-layer bug; route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) |
| `DOCA_LOG_LEVEL=trace ./<binary>` | `## run` step 4 | What did the structured DOCA logger emit for the first failing host-side DPA call? | A trace-level line on every host-side lifecycle transition and every launch submit. Silence after a launch submit = either host PE not progressed OR DPA kernel running but stuck — reach for the DPA-side tooling next |
| (route via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)) DPA-side developer tools — DPA debugger, DPA process-state inspector, DPA statistics tool | `## debug` layer 5; `## debug` layer 6 | What is the DPA processor itself doing right now, from the DPA side? | The public *DPA Tools* umbrella documents the per-tool output; the agent's job is to NAME the existence of these tools and route the user there, not to redefine their surface |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the DPA-specific rows on top.
