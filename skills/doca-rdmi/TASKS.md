# DOCA RDMA Initiator workflows

**Where to start:** The verbs run `install → configure → build →
modify → run → test → debug → use`. Skip ahead only when the user
is already past a verb. The `## test` verb is an iterative loop
(symbol presence → lifecycle order → single-completion smoke →
DPA-handoff probe → loop back if any check fails), not a one-shot
pass — see the eval-loop overlay in `## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the underlying object model, version
compatibility, error taxonomy, observability surface, and safety
policy that these workflows assume, see
[CAPABILITIES.md](CAPABILITIES.md). For where to find docs, the
installed DOCA layout, or release notes, route through
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through the
steps in order, verifying preconditions before recommending the
next call.

> **Answer end-to-end RDMI workflow questions from THIS file.** A
> hands-on "walk me through doca-rdmi end-to-end" question
> (confirm install → discover devices/caps → start from a shipped
> sample → build → run → debug) is answerable entirely from this
> skill's own files. You do **not** need to open `doca-setup` or
> `doca-programming-guide` to answer it — the commands you need are
> in [`## end-to-end (quickref)`](#end-to-end-quickref) below and
> in the [`## Command appendix`](#command-appendix). Only route to
> `doca-setup` if the user has *no DOCA install at all*; only
> overlay `doca-programming-guide` if the user explicitly asks for
> the cross-library build-system theory beyond the line given here.

## end-to-end (quickref)

A self-contained walkthrough for *"I have a BlueField with DOCA
installed; take me through doca-rdmi end-to-end."* Each step names
the exact symbol/command and the green signal; deeper detail is the
matching verb section below. Quote commands verbatim from the user's
install — never hardcode versions or paths.

1. **Confirm DOCA + doca-rdmi are installed and agree** (full detail:
   [`## install`](#install)). Run
   `pkg-config --modversion doca-common doca-verbs doca-dpa doca-rdmi`
   and `doca_caps --version`; **green** = one identical semver from
   all sources. Disagreement or an unresolved `.pc` = partial /
   missing RDMI package — stop and repair the install before building.
2. **Discover usable devices + per-device capability** (full detail:
   [`## configure`](#configure) step 2). Run `doca_caps --list-devs`;
   **green** = at least one device row whose DPA-capability flag is
   set (that flag is the gate for RDMI-on-DPA). The capability-query
   symbol to cite in code is the per-device cap read surfaced by
   `doca_caps`; do not invent a different one.
3. **Start from a shipped sample, do not author from prose** (full
   detail: [`## modify`](#modify)). Discover the canonical sample
   with `ls /opt/mellanox/doca/samples/doca_rdmi/` (the bundle
   deliberately defers to `ls` rather than naming a sample that may
   not exist on the user's version). Use the discovered sample as the
   modify-target via the five-slot fill in [`## modify`](#modify).
   Because every `doca_rdmi_*` symbol is EXPERIMENTAL, authoring RDMI
   source from documentation is forbidden.
4. **Build with pkg-config (never hand-crafted `-l` flags)** (full
   detail: [`## build`](#build)). Use
   `pkg-config --cflags --libs doca-rdmi`; it resolves includes under
   whatever path `--cflags` reports and emits
   `-ldoca_rdmi -ldoca_common -ldoca_verbs` (plus `-ldoca_dpa` when
   the DPA datapath is in use). Drive it from meson/ninja; the
   `pkg-config` module — not a hardcoded path — is the source of
   truth.
5. **Run once on the smallest case + name a concrete observable**
   (full detail: [`## run`](#run), [`## test`](#test)). Run with
   `DOCA_LOG_LEVEL=trace ./<binary>`; **green** is *not* "exited 0" —
   it is **one completion observed on the configured surface** (one
   CQE on the `doca_verbs_cq` for a host initiator, or one increment
   of the `doca_dpa_completion` counter for a DPA initiator) matching
   the single posted work request. Silence after `doca_ctx_start()`
   means the DPA kernel was never launched or the responder is
   silently rejecting the QP transition.
6. **If not-green, walk the layered debug ladder** (full detail:
   [`## debug`](#debug)): symbol presence → lifecycle order
   (`DOCA_ERROR_BAD_STATE` = attach/handle call on the wrong side of
   `doca_ctx_start()`) → single-completion smoke → DPA-handle
   round-trip, then escalate below RDMI only after two non-converging
   iterations.

## install

Goal: confirm the user's installed DOCA actually ships `doca-rdmi`
and the supporting libraries before any RDMI-specific work begins.

This skill does **not** own DOCA installation; that path lives in
[`doca-setup`](../../doca-setup/SKILL.md). The RDMI-specific
preconditions the agent verifies after a DOCA install:

1. **`doca-rdmi` `.pc` file is present.** `pkg-config
   --modversion doca-rdmi` resolves and reports a semver
   matching `doca_caps --version`. If it does not resolve, the
   installed DOCA package set does not include RDMI; the user
   needs to install the matching package (the exact package name
   is platform-specific and looked up via
   [`doca-public-knowledge-map ## Layout of an installed DOCA package`](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package)).
2. **Supporting `.pc` files are present.** RDMI builds on
   `doca-common`, `doca-verbs`, and (for the DPA datapath)
   `doca-dpa`. All four `pkg-config --modversion` results must
   agree on the same DOCA semver per the four-way match in
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
3. **Installed headers expose the symbols.** Check the public
   header set (`doca_rdmi_connection.h`, `doca_rdmi_poster.h`,
   `doca_rdmi_dev_connection.h`, `doca_rdmi_dev_poster.h`,
   `doca_rdmi_dev_cqe.h`) resolves under the installed DOCA
   infrastructure include tree. If the `.pc` resolves but the
   header is missing, the install is partial; do not attempt to
   build until the install is repaired.
4. **The DPA toolchain is available** if the user intends to use
   the DPA datapath. The DOCA DPACC compiler ships as a separate
   tool; the agent routes to
   [`doca-dpa`](../doca-dpa/SKILL.md) for the DPA toolchain
   preconditions rather than duplicating them here.

If any precondition fails, **stop and route to
[`doca-setup`](../../doca-setup/SKILL.md)**; an RDMI-layer
diagnosis against a half-installed DOCA wastes the user's time.

## configure

Goal: bring up a `doca_rdmi_connection` (and, if the user posts
work, a `doca_rdmi_poster`) and reach the state where the DPA
handle is valid.

Steps the agent should walk the user through:

1. **Confirm the installed DOCA version.** Use the procedure in
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
   Quote the version observed (`pkg-config --modversion
   doca-rdmi`, then `doca_caps --version`); do not assume
   "latest". Surface that every `doca_rdmi_*` symbol the agent
   recommends is `EXPERIMENTAL` per
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).
2. **Stand up the verbs context.** RDMI is layered on
   `doca-verbs`; the user must have an initialized
   `doca_verbs_context` and a `doca_verbs_pd` before any
   `doca_rdmi_*_create` call. The verbs surface is a precondition;
   route the user to the public DOCA verbs guide via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
   if they do not yet have a working verbs setup.
3. **Pick the datapath.** If the application runs the data path
   on the DPA, the user must call `doca_ctx_set_datapath_on_dpa()`
   on the connection / poster context **before**
   `doca_ctx_start()`. The
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   DPA-handoff bullet names this; skipping it is the canonical
   `DOCA_ERROR_BAD_STATE` cause when the user then calls
   `doca_rdmi_*_get_dpa_handle`.
4. **Attach exactly one completion source.** Either a
   `doca_dpa_completion` via
   `doca_rdmi_*_dpa_completion_attach` (for the DPA datapath) or
   a `doca_verbs_cq` via `doca_rdmi_*_cq_attach` (for the host
   datapath). The agent does not recommend both; the public
   headers describe these as alternatives.
5. **Configure the connection's receive window.** Call
   `doca_rdmi_connection_init_receive(conn, buf, size)` with a
   buffer sized for the application's intended in-flight receive
   work request count. Confirm the host side has an ack loop
   ready before the DPA starts posting receives.
6. **Start.** Call `doca_ctx_start()` on each
   `doca_rdmi_*_as_ctx()` context. Out-of-order returns
   `DOCA_ERROR_BAD_STATE`.
7. **Retrieve the DPA handle.** After start (and only after
   start), call `doca_rdmi_*_get_dpa_handle` to obtain the
   accelerator-side handle. Hand that handle to the DPA kernel
   per the [`doca-dpa`](../doca-dpa/SKILL.md) programming model;
   do not invent a different handoff.

If any step fails with a `DOCA_ERROR_*`, route through the error
taxonomy in
[CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy)
before retrying.

## build

Goal: produce a host-side binary (and, if applicable, a DPA-side
binary) that links DOCA RDMI against the user's installed DOCA,
using the canonical cross-library build pattern.

The build pattern for any DOCA C/C++ consumer is **identical**
across libraries — `pkg-config` for include + link flags, meson
or CMake as the build system — and is fully documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
This skill carries only the RDMI-specific overlay:

| Slot | Value for RDMI | Why it matters |
| --- | --- | --- |
| `pkg-config` module name (host side) | `doca-rdmi` | The library's `.pc` file installed by the DOCA host packages |
| Co-required modules | `doca-common`, `doca-verbs`, plus `doca-dpa` when the DPA datapath is in use | RDMI builds on verbs; the DPA datapath additionally requires the `doca-dpa` host-side library plus the DPA toolchain |
| Header check | `doca_rdmi_connection.h`, `doca_rdmi_poster.h` resolvable under the installed DOCA infrastructure include tree (path via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)) | If `pkg-config --cflags doca-rdmi` resolves but the include is missing, the install is partial |
| DPA-side compilation | `doca_rdmi_dev_*.h` headers are compiled with the DOCA DPA toolchain into the DPA kernel — *not* with the host compiler | Mixing host and DPA toolchains on the same translation unit is the canonical reason a DPA-side build "fails for no reason"; routed to [`doca-dpa`](../doca-dpa/SKILL.md) for the toolchain details |
| Minimum required DOCA version | Query with `pkg-config --modversion doca-rdmi`; never hardcode in build files | The EXPERIMENTAL tag means a version pin from agent memory is wrong by construction |

For non-C consumers (Rust, Go, Python), the host-side link surface
is the same `*.so` files; FFI wrappers are out of scope for this
skill. The DPA-side surface is not wrappable — the DPA kernel is
compiled and linked into the DPA binary itself.

## modify

Goal: take an existing RDMI-using component (the user's own code,
or a verified DOCA sample if one ships in the installed package
set) as the starting point and apply a minimum-diff modification
to express the new intent.

The universal modify-a-shipped-sample workflow lives in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify).
Use it as-is. The RDMI-specific overlay is the *five-slot fill*
the agent must elicit from the user before recommending any
code-level edit:

| Slot | What the agent asks the user | RDMI-specific consideration |
| --- | --- | --- |
| 1. Starting code | Which RDMI-using file or sample is the baseline? | If the user has no working baseline, *stop* — the EXPERIMENTAL tag on the API surface means authoring RDMI source from documentation prose is forbidden by this skill (per [`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship)) |
| 2. Object set | Connection only, poster only, or both? | Each object has its own lifecycle and its own completion attach; adding one mid-flight is a re-architecture, not a tweak |
| 3. Completion source | Switch between `doca_dpa_completion` and `doca_verbs_cq`? | This changes the datapath and which `_attach` call applies; the public headers describe these as alternatives, so the switch is a re-architecture |
| 4. Receive window | Resize `doca_rdmi_connection_init_receive`'s buffer? | A larger window changes the host-side ack-loop sizing; under-acking is a silent stall, not an error return |
| 5. PRM-passthrough input | Change `doca_rdmi_*_set_input` payload? | The shipped header note flags that the QP-state outputs *"will likely change once we have On-Behalf API support"* — quote the instability to the user before they ship the change |

The agent emits an *intent description + the five filled slots*;
the actual unified diff against the user's baseline is produced
line-by-line and validated by the user pasting back the result.
Do not author RDMI source code that the user did not start from.

## run

Goal: actually execute the built binary (host + DPA) against the
user's installed DOCA on a host or BlueField, with a remote
responder reachable on the wire.

Steps the agent should walk the user through:

1. **Confirm the remote responder is reachable.** RDMI is an
   initiator; running the binary on one side alone produces a
   misleading hang. The responder's RDMA stack must be running
   (commonly an application using
   [`doca-rdma`](../doca-rdma/SKILL.md) or the upstream verbs
   surface) and reachable over IB / RoCE.
2. **Drive the DPA kernel.** Load the DPA binary onto the
   accelerator per [`doca-dpa`](../doca-dpa/SKILL.md); hand it
   the DPA handle obtained from `doca_rdmi_*_get_dpa_handle`.
   The agent does *not* author the DPA-side launch
   incantation; the launch model is owned by `doca-dpa`.
3. **Capture the structured log on the host side.** Set
   `DOCA_LOG_LEVEL=trace` for the first run (see
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability)).
   This is the cheapest way to make the host-side lifecycle
   visible on first failure.
4. **Drive the host-side ack loop.** After the DPA consumes
   receive work requests, the host must call
   `doca_rdmi_connection_recv_ack(conn, num_acked)` with the
   matching count; otherwise the receive window stalls. Confirm
   the ack loop is wired before the DPA starts posting.
5. **Observe completions on the configured surface.** If the
   attach was a `doca_dpa_completion`, the DPA kernel polls; if
   the attach was a `doca_verbs_cq`, the host reads CQEs from
   the verbs CQ. A run that produces no observable completions
   on *either* surface but doesn't error is almost always (a)
   the DPA kernel was not launched or (b) the responder is
   silently rejecting the QP transition — confirm both before
   diving into RDMI internals.

## test

Goal: prove the configured RDMI instance can actually initiate
one-sided RDMA correctly on the user's hardware, and that the
EXPERIMENTAL symbol surface the user is depending on matches the
installed DOCA.

This is **a loop, not a one-shot pass.** Each iteration narrows
either the symbol set, the lifecycle order, the completion
source, the receive window, or the DPA handoff. The loop
terminates when either (a) the user's intended one-sided RDMA
operation completes end-to-end with the expected completion, or
(b) the agent has narrowed the failure cause to a layer outside
RDMI itself (verbs, DPA, driver, firmware, network) and
escalated to the matching skill.

Iteration shape:

1. **Symbol-presence check.** Confirm every `doca_rdmi_*`
   symbol the user's code references is exported by the
   installed `libdoca_rdmi.so` (and matched in the installed
   headers). A `function not found` at link time is almost
   always a partial install or a wrong-version pairing per
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   layer 2.
2. **Lifecycle-order check.** Walk the configure sequence in
   [`## configure`](#configure) against the user's code: every
   `_init_receive` / `_*_attach` / `_set_input` call must
   precede `doca_ctx_start()`; every `_get_dpa_handle` call must
   follow it; every `_destroy` call must follow `doca_ctx_stop()`.
   Out-of-order is the leading cause of `DOCA_ERROR_BAD_STATE`.
3. **Single-completion smoke.** Before driving any volume,
   confirm ONE completion arrives on the configured completion
   surface for one work request from the DPA kernel. If no
   completion arrives, stop and walk the observability surface
   in [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability);
   do not raise traffic into an unobserved path.
4. **DPA handle round-trip.** Confirm
   `doca_rdmi_*_get_dpa_handle` returns a non-zero handle after
   `doca_ctx_start()`, and that the DPA kernel can read the
   underlying connection / poster state through the device-side
   header set. A zero or stale handle is a configuration bug,
   not a kernel bug.
5. **Negative test.** Construct one deliberately misordered
   sequence (e.g. attach a completion AFTER `doca_ctx_start()`)
   and confirm the API returns `DOCA_ERROR_BAD_STATE`. This
   validates that the agent's lifecycle understanding is itself
   correct on this DOCA version.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| `DOCA_ERROR_NOT_SUPPORTED` on a symbol we expected to ship | The user's code references a `doca_rdmi_*` symbol that is in the public docs but not in the installed `.so` | Confirm the install is complete via the four-way match; the EXPERIMENTAL tag means symbols can be promoted / renamed between releases |
| `DOCA_ERROR_BAD_STATE` from `_get_dpa_handle` | The handle call landed before `doca_ctx_start()`, or the datapath was not set on the DPA | Re-walk steps 3-6 of [`## configure`](#configure); the `doca_ctx_set_datapath_on_dpa()` call is easy to forget |
| Host-side ack loop stalled | The DPA posted receives; the host never acked; new receives back up | Wire `doca_rdmi_connection_recv_ack` into the host loop; under-acking is silent |
| DPA kernel sees no completions | The CQ attach was a `doca_verbs_cq` (host-side), not a `doca_dpa_completion` | The completion source decides which side observes; confirm the attach call matches the consumer |
| Link errors on `doca_rdmi_*` symbol | The shipped `libdoca_rdmi.so` does not export the symbol the user's code references | Audit the EXPERIMENTAL tag set on the installed version; a symbol present in one release may be absent or renamed in another |

Loop termination: stop iterating once two consecutive iterations
of the same kind don't change anything — that means the cause is
below RDMI. Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
with the captured layer-1-through-5 evidence.

## debug

Goal: when a DOCA RDMI call returns a `DOCA_ERROR_*` (or the
program doesn't make forward progress), narrow the cause to a
specific layer and act on it.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver — *before* recommending RDMI-specific
fixes. This skill's overlay names the RDMI-specific manifestation
at layers 5 (runtime) and 6 (program):

**Layer 5 (runtime) — RDMI overlay.**

- Confirm the connection / poster context was started *after*
  all attach/configure calls, *before* any `_get_dpa_handle`
  call, and on the right datapath (DPA for accelerator-side,
  host for `doca_verbs_cq`).
- Confirm the DPA kernel was actually launched and is consuming
  completions. *No observable completions* with a DPA attach is
  almost always a launch / handoff bug, not an RDMI-spec bug;
  route to [`doca-dpa`](../doca-dpa/SKILL.md) for the
  accelerator-side launch verification.
- Confirm the host-side ack loop is firing. A stalled receive
  window looks like a hung initiator; the diagnostic is to
  inspect `doca_rdmi_connection_get_app_rq_info` and the host's
  ack count against the DPA's consumption count.

**Layer 6 (program) — RDMI overlay.**

- Lifecycle order: configure → start → use → stop → destroy.
  Out-of-order returns `DOCA_ERROR_BAD_STATE`. Re-check against
  [`## configure`](#configure).
- Single-handle discipline: the DPA handle from
  `_get_dpa_handle` belongs to exactly one context and one
  consuming kernel. Reusing a handle across stop / start cycles
  is undefined behavior per [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
- PRM-passthrough inputs: `doca_rdmi_*_set_input` carries
  PRM-side payloads transparent to the user. A mismatch
  between the input shape and the verbs command op type is a
  silent failure; quote the header doc string verbatim when
  walking through it.

Once the layer is identified, route to the matching debug verb on
the matching skill: install / build / link / driver to
[`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug);
cross-cutting runtime to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug);
DPA-side debug to
[`doca-dpa TASKS.md ## debug`](../doca-dpa/TASKS.md#debug);
program-layer Core-context patterns to
[`doca-programming-guide TASKS.md ## debug`](../../doca-programming-guide/TASKS.md#debug).

**5-phase universal debug-loop instantiation (RDMI).** Layer
identification above is phase 1 of the
[universal debug-loop contract](../../doca-debug/CAPABILITIES.md#universal-debug-loop-contract).
The agent MUST walk the remaining four phases on every RDMI
debug answer before declaring done:

1. **Layer identification** — above (runtime / program /
   DPA-side handoff).
2. **Triple capture (READ-ONLY).** Capture (a) verbs context
   state via `doca_verbs_context` introspection + `ibv_devinfo`
   on the underlying device, (b) RDMI connection state via
   `doca_rdmi_connection_get_app_rq_info` (host side) AND the
   DPA-side completion counter from
   [`doca-dpa`](../doca-dpa/TASKS.md#debug), (c) DOCA log lines
   at `DOCA_LOG_LEVEL=DEBUG` for the offending post / poll. The
   triple is the rollback target — do NOT mutate before
   capturing all three.
3. **Single-variable mutation SMALLER than the original
   change.** Examples: post ONE work request (not the batch);
   switch the DPA kernel from the production shape to the
   shipped RDMI smoke sample (not refactor the algorithm); flip
   the connection state machine to a known-quiescent peer (not
   the production peer). Larger mutations void the experiment.
4. **Re-capture and compare.** Re-run the triple; the diff IS
   the evidence. A "no change between iterations" diff means
   the cause is below RDMI (verbs / firmware / network).
5. **Exit with named green signal OR escalate.** Green = one
   completion observed on the right datapath (host CQ for host
   initiator; DPA completion counter for DPA initiator). If two
   consecutive iterations don't change anything, escalate via
   the layer-7 route table above with the captured triple.

## rollback

RDMI contexts are stateful (verbs context + connection / poster +
optional DPA attach + remote responder peer) and the agent's
failure mode is to leave a half-connected QP / a stale DPA
handle / a registered MR pointing at freed host memory. The
[universal verification contract](../../doca-setup/CAPABILITIES.md#universal-verification-contract)
step 1 (preconditions) requires *"the rollback path is
documented"* on every change-recommending answer; this is the
RDMI instantiation.

**Snapshot before mutate.** Before any change-recommending RDMI
answer, capture (a) the underlying `doca_verbs_context` identity
and QP allocation map (`ibv_devinfo` + per-QP `qp_num`), (b) the
RDMI connection state map (active connection IDs, peer addresses,
DPA attach status from `_get_dpa_handle`), and (c) the MR
registration list (mmap region IDs from `doca_mmap_*`). The
triple IS the rollback target — *"restore the pre-RDMI state"*
is unfalsifiable without it.

1. **Quiesce the data path FIRST.** Stop posting new WRs from
   both initiator and responder. For DPA-initiated paths, signal
   the DPA kernel to drain via the host-side termination flag
   from [`## modify`](#modify); `cudaDeviceSynchronize` / DPA
   join until the kernel returns. If it does not return within
   the bounded debug-loop window, that is the
   [deploy-loop bridge](../../doca-setup/CAPABILITIES.md#deploy-loop-bridge--step-5-not-green-is-the-debug-loop-trigger)
   trigger; fire the debug-loop on the hung-kernel symptom
   before continuing the rollback.
2. **Tear down RDMI objects in reverse-create order.** (a)
   Destroy DPA attach handles (`_get_dpa_handle` outputs) so the
   DPA-side kernel cannot resurrect a stale CQ pointer; (b)
   `doca_ctx_stop` on each RDMI poster and connection context
   (each `doca_rdmi_poster` / `doca_rdmi_connection` is its own
   `doca_ctx` via `_as_ctx` — RDMI exposes no single top-level
   context object); (c) destroy posters then connections in
   reverse-create order (`doca_rdmi_poster_destroy`, then
   `doca_rdmi_connection_destroy` for each). There is no
   `doca_rdmi_destroy`.
3. **Unregister MRs and underlying mmap regions.** Each
   `doca_mmap` registered for RDMA payloads / control buffers
   must be destroyed before the underlying host allocation is
   freed; reverse order is non-negotiable per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
4. **Tear down the verbs context LAST.** `doca_verbs_context`
   may be shared with non-RDMI clients (e.g., the verbs CQ from
   [`doca-verbs`](../doca-verbs/SKILL.md)); only destroy it
   if RDMI was the sole client. If shared, leave the verbs
   context up and route the rollback's verbs leg via
   [`doca-verbs TASKS.md`](../doca-verbs/TASKS.md).
5. **Re-verify the verbs parent is intact.** Re-run the
   one-WR smoke from [`doca-verbs TASKS.md ## test`](../doca-verbs/TASKS.md#test)
   to confirm the parent verbs context still completes a single
   WR at the pre-RDMI shape. If not, the RDMI rollback
   corrupted the parent — surface as a residual gap, do NOT
   retry RDMI.
6. **Cross-peer rollback.** The remote BlueField responder
   (if RDMI brought one up) MUST run the same five steps on its
   side. Document the cross-peer rollback verb explicitly:
   *"the rollback path is the five-step reversal in
   [`## rollback`](#rollback), applied on BOTH peers in
   reverse-connect order; the agent has captured the
   QP/connection/MR snapshot on both sides."*

The rollback is bounded — on the second non-green re-verify at
step 5, the agent MUST surface the unresolved residual gap
instead of recommending another RDMI retry.

## use

Goal: integrate a working RDMI component into a larger
application — typically a DPA-resident agent that consumes a
remote responder's memory at line rate without host CPU
involvement.

The integration shape this skill teaches:

1. **Per-application init order.** The host-side init order is
   verbs → RDMI context create → datapath set → completion
   attach → receive buffer init (connection only) → start →
   DPA handle retrieval → DPA kernel launch. The DPA kernel
   does not run until the handle is in its hands; the host
   does not stop the context until the DPA kernel has
   drained its outstanding work.
2. **Per-application teardown order.** Stop the DPA kernel
   first; quiesce in-flight work; call `doca_ctx_stop()` on the
   RDMI context; call `_destroy` on the connection / poster;
   tear down verbs. Destroying a context with outstanding DPA
   work is the canonical use-after-free failure mode.
3. **Multi-tenant discipline.** When several DPA kernels share
   the same RDMI host-side application, each kernel must have
   its own DPA handle from its own context. Sharing a handle
   between kernels is forbidden per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy);
   the agent surfaces this whenever a multi-kernel
   architecture comes up.
4. **Operational handoff.** Production deployment uses the
   bundle's hardware-safety meta-policy
   ([`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md))
   for any change that touches the BlueField BFB, firmware, or
   DPA toolchain. RDMI itself does not modify hardware state,
   but every RDMI-using component lives downstream of those
   changes — the per-artifact `## Safety policy` overlay
   names the discipline.
5. **Per-release re-verification.** Because the API surface is
   EXPERIMENTAL, every DOCA upgrade requires re-running
   [`## test`](#test) end-to-end against the new install. The
   agent does not assume a known-working integration survives
   a DOCA-version bump without re-testing.

## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as follows
so the agent does not invent guidance:

- **install (of DOCA itself).** Installing DOCA, choosing
  packages, post-install verification, `pkg-config` wiring —
  defer to [`doca-setup`](../../doca-setup/SKILL.md) and to the
  install-tree layout in
  [`doca-public-knowledge-map ## Layout of an installed DOCA package`](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
  This skill's `## install` verb assumes DOCA is already
  installed and only checks the RDMI-specific preconditions.
- **deploy.** Deploying RDMI-using applications at scale across
  many hosts / DPUs, Kubernetes operator workflows, multi-tenant
  RDMA isolation — out of scope and reserved for a future
  platform skill. For single-host first-run testing, the right
  verb is `## run`.
- **rollback.** Coordinated rollback of RDMI-using applications
  across multiple hosts / DPUs — out of scope and reserved for
  a future platform skill. For single in-session rollback, the
  right verb is destroying the connection / poster contexts and
  re-running `## configure` with the corrected parameters; do
  not invent a "rollback" workflow.
- **DPA programming.** The DPA toolchain, kernel build, memory
  model, and execution semantics belong to
  [`doca-dpa`](../doca-dpa/SKILL.md). This skill describes the
  RDMI-side handoff (the handle, the device-side header set)
  but does not author DPA kernels.
- **Kernel-level driver install / firmware burn.** Installing
  the `mlx5_core` driver, burning new ConnectX firmware, or
  modifying `mlxconfig` parameters — out of scope. Route to
  [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug)
  layer 5 and to
  [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md)
  for the change-application discipline.

## Command appendix

Every command below is **cross-cutting on DOCA RDMI** — it
answers a recurring class of question that comes up in the verbs
above. The agent should treat the *class* as load-bearing; the
worked example is a single instance.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent probes for the matching structured helper FIRST
(`doca-env --json` for version + devices + libraries + drivers;
`doca-capability-snapshot` for per-device capability flags;
`version-matrix.json` for *"available since"* lookups). If the
probe succeeds, the structured tool's output is the authoritative
answer. If the probe fails, fall back to the manual command in
the row.

| Command (worked example) | Owning step | Class of question it answers | What healthy output looks like |
| --- | --- | --- | --- |
| `pkg-config --modversion doca-rdmi` | [`## install`](#install) step 1; [`## configure`](#configure) step 1 | What is the build-time DOCA RDMI version? | A semver matching `doca_caps --version`. Disagreement = partial install; route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2 |
| `pkg-config --modversion doca-common doca-verbs doca-dpa doca-rdmi` | [`## install`](#install) step 2 | Do all four `.pc` files agree on the same DOCA semver? | A single semver repeated four times. Any disagreement is the partial-install pattern |
| `pkg-config --cflags --libs doca-rdmi` | [`## build`](#build) | What include + link flags does the linker need? | Includes resolve under whichever include directory `pkg-config --cflags` reports on this install (do not hardcode the path); libs include `-ldoca_rdmi -ldoca_common -ldoca_verbs` (plus `-ldoca_dpa` when the DPA datapath is in use) |
| `doca_caps --list-devs` | [`## configure`](#configure) step 2; [`## test`](#test) step 4 | Which devices on this host can be used as a `doca_dev` with DPA capability? | One row per visible device with PCIe address and capability flags; the DPA capability flag is the gate for RDMI-on-DPA |
| `doca_caps --version` | [`## install`](#install) step 1 | What is the *runtime* DOCA version on this host? | A semver matching `pkg-config --modversion doca-rdmi` |
| `cat /opt/mellanox/doca/applications/VERSION` | [`## install`](#install) step 1; [`## debug`](#debug) layer 1 | What does the install tree itself claim its version is? | A semver matching the other version sources |
| `DOCA_LOG_LEVEL=trace ./<binary>` | [`## run`](#run) step 3 | What did the structured DOCA logger emit for the first failing call? | A trace-level line on every lifecycle transition. Silence after `doca_ctx_start()` = the DPA kernel was never launched or the host stopped progressing the PE |
| `dmesg | tail -n 40` (sudo) | [`## debug`](#debug) layer 7 | What did the kernel / driver log around the last RDMI call? | Empty or recent benign messages. Repeated mlx5 / IB errors → driver-layer bug; route to [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug) |
| `mlxconfig -d <pcie> q | head -n 40` (sudo) | [`## debug`](#debug) layer 7 | What firmware config does the underlying NIC / DPU report? | Stable firmware config; mismatched values indicate a hardware-state change that must go through [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the RDMI-specific rows on top.
